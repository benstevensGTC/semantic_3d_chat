"""No-step gradient-cancellation probe for an additional frozen-checkpoint LoRA bank.

This is evaluation/training diagnostics code, never part of chat inference.  It
loads a complete adapter checkpoint, freezes every persisted parameter, adds a
separate exactly-zero-output LoRA bank, and differentiates the two sides of
selected mirrored units independently.  No optimizer is constructed or stepped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from semantic_3d_chat.chat.runtime import construct_scene_tokenizer, validate_checkpoint_contract
from semantic_3d_chat.config import PROJECT_ROOT, artifact_root, load_config, project_path
from semantic_3d_chat.data.dataset import SceneQADataset
from semantic_3d_chat.evaluation.baseline_io import sha256_file
from semantic_3d_chat.language.local_lm import load_local_language_model, prompt_token_ids
from semantic_3d_chat.language.lora import (
    LoRAInstallation,
    LoRASettings,
    install_lora_adapters,
    lora_optimizer_settings,
    lora_settings,
    tensor_state_sha256,
    validate_lora_checkpoint_state,
)
from semantic_3d_chat.language.prefix_injection import (
    ContinuousPrefixComposer,
    native_gemma4_image_contract_setting,
    scene_boundary_mode_setting,
    scene_prefix_after_bos_setting,
)
from semantic_3d_chat.scene_encoder.map_io import load_map_tensors
from semantic_3d_chat.training.checkpointing import load_adapter_checkpoint
from semantic_3d_chat.training.losses import QuestionGroundingHead
from semantic_3d_chat.training.pair_curriculum import (
    CounterfactualPairUnit,
    build_exact_question_pair_units,
    candidate_logit_margins,
    cap_pair_units_per_pair,
    first_answer_token_full_vocab_margins,
    pair_curriculum_settings,
    select_pair_only_records,
    single_differing_answer_token,
    token_normalized_nll,
)
from semantic_3d_chat.training.source_provenance import (
    SOURCE_SCOPE,
    capture_git_source_provenance,
    require_clean_committed_source,
)
from semantic_3d_chat.training.train_adapter import (
    forward_prefix_batch,
    map_forward,
    select_training_records,
    tokenize_answer,
    training_counterfactual_scene_pairs,
)

_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_OBJECT = re.compile(r"[0-9a-f]{40,64}")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
DEFAULT_CONFIG = "configs/experiments/gemma4_color_mirror_spatial_relation_v12.yaml"
DEFAULT_CHECKPOINT = "data_gemma4/checkpoints/gemma4_color_mirror_spatial_relation_v12/epoch_008"
DEFAULT_CHECKPOINT_SHA256 = "a4c85c14a214e4e594992e489a784cb4bacb64d3dfda519ad3da18b1595d9f22"
DEFAULT_SELECTION_SHA256 = "73571e68965916e6cf9f9d37bac0b569727acd5eea7d6addd167c5625d5bad92"
DEFAULT_SELECTION_CONTENT_SHA256 = (
    "3237522a2845f408cfe0b2333171c2a7dc5fc373f07cf2fdf3d8d40054d53150"
)
DEFAULT_ALL_SELECTED_RECORDS_SHA256 = (
    "7f0714e3151c9ddb57c1da95a457820a833e490c070881a88a9fee4a9168f933"
)


@dataclass(frozen=True)
class ProbeBankSpec:
    """Strict architecture contract for the temporary, unpersisted LoRA bank."""

    layers: tuple[int, ...] = (30, 31, 32, 33)
    rank: int = 8
    alpha: float = 16.0
    seed: int = 13_008

    def __post_init__(self) -> None:
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError("Probe layers must be a non-empty unique sequence")
        if any(
            isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
            for layer in self.layers
        ):
            raise ValueError("Probe layer indices must be non-negative integers")
        if tuple(sorted(self.layers)) != self.layers:
            raise ValueError("Probe layer indices must be strictly increasing")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("Probe LoRA rank must be a positive integer")
        if not math.isfinite(float(self.alpha)) or float(self.alpha) <= 0:
            raise ValueError("Probe LoRA alpha must be finite and positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Probe seed must be a non-negative integer")

    @property
    def target_modules(self) -> tuple[str, ...]:
        return tuple(
            f"model.language_model.layers.{layer}.self_attn.{projection}"
            for layer in self.layers
            for projection in ("q_proj", "o_proj")
        )

    @property
    def settings(self) -> LoRASettings:
        return LoRASettings(
            enabled=True,
            rank=self.rank,
            alpha=self.alpha,
            dropout=0.0,
            target_modules=self.target_modules,
        )

    def contract(self) -> dict[str, Any]:
        return {
            **self.settings.contract(),
            "seed": self.seed,
            "initialization": "cpu_kaiming_uniform_a_exact_zero_b",
            "merge": False,
        }


def initialize_probe_bank(installation: LoRAInstallation, *, seed: int) -> None:
    """Reset A deterministically on CPU and B exactly to zero on its target device."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for adapter in installation.adapters:
            source = torch.empty(adapter.lora_a.shape, dtype=torch.float32, device="cpu")
            nn.init.kaiming_uniform_(source, a=math.sqrt(5), generator=generator)
            adapter.lora_a.copy_(source.to(adapter.lora_a.device))
            adapter.lora_b.zero_()
    installation.validate_state()
    if any(torch.count_nonzero(adapter.lora_b).item() for adapter in installation.adapters):
        raise RuntimeError("Probe LoRA bank did not initialize to exact zero output")


def probe_parameter_items(installation: LoRAInstallation) -> dict[str, nn.Parameter]:
    return {
        f"{target}.{name}": parameter
        for target, adapter in zip(installation.target_names, installation.adapters, strict=True)
        for name, parameter in (("lora_a", adapter.lora_a), ("lora_b", adapter.lora_b))
    }


def cancellation_metrics(
    gradients_a: Mapping[str, torch.Tensor],
    gradients_b: Mapping[str, torch.Tensor],
    *,
    names: Sequence[str] | None = None,
) -> dict[str, float | bool | None]:
    """Measure how strongly two same-coordinate gradient vectors cancel."""

    selected = tuple(sorted(gradients_a) if names is None else names)
    if not selected or set(selected) - set(gradients_a) or set(selected) - set(gradients_b):
        raise ValueError("Cancellation metric requires a non-empty shared parameter selection")
    flat_a: list[torch.Tensor] = []
    flat_b: list[torch.Tensor] = []
    for name in selected:
        first = gradients_a[name].detach().float().cpu().reshape(-1)
        second = gradients_b[name].detach().float().cpu().reshape(-1)
        if first.shape != second.shape:
            raise ValueError(f"Gradient shape mismatch for {name}: {first.shape} != {second.shape}")
        if not torch.isfinite(first).all() or not torch.isfinite(second).all():
            raise ValueError(f"Gradient contains NaN or infinity: {name}")
        flat_a.append(first)
        flat_b.append(second)
    vector_a = torch.cat(flat_a).double()
    vector_b = torch.cat(flat_b).double()
    norm_a = float(vector_a.norm())
    norm_b = float(vector_b.norm())
    summed_norm = float((vector_a + vector_b).norm())
    denominator = norm_a + norm_b
    cosine = None
    if norm_a > 0 and norm_b > 0:
        cosine = float(torch.dot(vector_a, vector_b) / (norm_a * norm_b))
    return {
        "gradient_a_l2": norm_a,
        "gradient_b_l2": norm_b,
        "summed_gradient_l2": summed_norm,
        "norm_sum_denominator": denominator,
        "cancellation_ratio": None if denominator == 0 else summed_norm / denominator,
        "cosine_similarity": cosine,
        "both_zero": denominator == 0,
    }


def grouped_cancellation_metrics(
    gradients_a: Mapping[str, torch.Tensor], gradients_b: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    """Return per-module, per-layer, and whole-bank cancellation measurements."""

    names = sorted(gradients_a)
    if names != sorted(gradients_b):
        raise ValueError("Gradient banks have different parameter names")
    by_module: dict[str, list[str]] = {}
    by_layer: dict[str, list[str]] = {}
    for name in names:
        module, _, _parameter = name.rpartition(".")
        by_module.setdefault(module, []).append(name)
        match = re.search(r"\.layers\.(\d+)\.", name)
        if match is None:
            raise ValueError(f"Probe gradient has no decoder layer index: {name}")
        by_layer.setdefault(match.group(1), []).append(name)
    return {
        "per_module": {
            name: cancellation_metrics(gradients_a, gradients_b, names=members)
            for name, members in sorted(by_module.items())
        },
        "per_layer": {
            name: cancellation_metrics(gradients_a, gradients_b, names=members)
            for name, members in sorted(by_layer.items(), key=lambda item: int(item[0]))
        },
        "aggregate": cancellation_metrics(gradients_a, gradients_b, names=names),
    }


def side_objective_terms(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    candidate_spec: tuple[int, int, int],
    candidate_margin: float,
    candidate_weight: float,
    full_vocab_margin: float,
    full_vocab_weight: float,
    pair_side_reduction_scale: float = 0.5,
) -> tuple[dict[str, torch.Tensor], dict[str, float | bool]]:
    """Build one side's exact contribution to the V12 decoder objective.

    V12 averages every component over the two sides of a unit.  Multiplying
    each independently differentiated side by 1/2 therefore makes A+B exactly
    the decoder-directed portion of the original one-unit training objective.
    """

    for name, value in (
        ("candidate_margin", candidate_margin),
        ("candidate_weight", candidate_weight),
        ("full_vocab_margin", full_vocab_margin),
        ("full_vocab_weight", full_vocab_weight),
        ("pair_side_reduction_scale", pair_side_reduction_scale),
    ):
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    language_nll = token_normalized_nll(logits, labels)[0]
    candidate_value = candidate_logit_margins(logits, labels, [candidate_spec])[0][0]
    full_vocab_value = first_answer_token_full_vocab_margins(logits, labels)[0]
    scale = float(pair_side_reduction_scale)
    terms = {
        "language_nll": scale * language_nll,
        "candidate_hinge_weighted": (
            scale
            * float(candidate_weight)
            * torch.relu(candidate_value.new_tensor(float(candidate_margin)) - candidate_value)
        ),
        "full_vocab_hinge_weighted": (
            scale
            * float(full_vocab_weight)
            * torch.relu(full_vocab_value.new_tensor(float(full_vocab_margin)) - full_vocab_value)
        ),
    }
    terms["decoder_total"] = sum(terms.values())
    diagnostics = {
        "candidate_logit_margin": float(candidate_value.detach().cpu()),
        "full_vocab_first_token_margin": float(full_vocab_value.detach().cpu()),
        "candidate_hinge_active": bool(candidate_value.detach() < float(candidate_margin)),
        "full_vocab_hinge_active": bool(full_vocab_value.detach() < float(full_vocab_margin)),
    }
    return terms, diagnostics


def first_answer_distribution(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return the complete-vocabulary logit vector predicting the first answer token."""

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("Expected logits [B,L,V] and aligned labels [B,L]")
    if logits.shape[0] != 1:
        raise ValueError("Step-zero parity requires one side per forward")
    supervised = labels[0].ne(-100).nonzero(as_tuple=False).flatten()
    if supervised.numel() == 0 or int(supervised[0]) == 0:
        raise ValueError("A first answer token must have a preceding causal position")
    return logits[0, int(supervised[0]) - 1]


def require_autograd_compatible_scene_tokens(tokens: torch.Tensor) -> None:
    """Reject inference tensors that a later LoRA backward cannot safely save."""

    if tokens.is_inference():
        raise RuntimeError(
            "Scene tokens are inference tensors; encode under no_grad before decoder probing"
        )
    if tokens.requires_grad:
        raise RuntimeError("Frozen-checkpoint scene tokens must be detached")
    if not torch.isfinite(tokens).all():
        raise ValueError("Scene tokens contain NaN or infinity")


def objective_gradients(
    terms: Mapping[str, torch.Tensor], parameters: Mapping[str, nn.Parameter]
) -> dict[str, dict[str, torch.Tensor]]:
    """Differentiate components without touching ``.grad`` or constructing an optimizer."""

    component_names = (
        "language_nll",
        "candidate_hinge_weighted",
        "full_vocab_hinge_weighted",
    )
    parameter_names = tuple(parameters)
    parameter_values = tuple(parameters.values())
    result: dict[str, dict[str, torch.Tensor]] = {}
    for index, component in enumerate(component_names):
        gradients = torch.autograd.grad(
            terms[component],
            parameter_values,
            retain_graph=index < len(component_names) - 1,
            create_graph=False,
            allow_unused=False,
        )
        result[component] = {
            name: gradient.detach().float().cpu().clone()
            for name, gradient in zip(parameter_names, gradients, strict=True)
        }
    result["decoder_total"] = {
        name: sum(result[component][name] for component in component_names)
        for name in parameter_names
    }
    return result


def selected_mirror_units(
    config: dict[str, Any],
    *,
    pair_id: str,
    expected_count: int,
    expected_sha256: str,
    expected_content_sha256: str,
    expected_all_selected_records_sha256: str,
) -> tuple[list[CounterfactualPairUnit], dict[str, Any]]:
    """Reconstruct the exact deterministic training selection and fail on drift."""

    settings = pair_curriculum_settings(config)
    if not settings.pair_only or settings.ranking_mode != "candidate_logit":
        raise ValueError("Probe requires the V12 pair-only candidate-logit curriculum")
    records = SceneQADataset(artifact_root(config, "qa") / "train.jsonl").records
    selected = select_pair_only_records(records, settings.pair_only_scene_ids)
    selected = cap_pair_units_per_pair(
        selected, settings.max_units_per_pair, seed=int(config["seed"])
    )
    configured_cap = config["training"].get("max_questions_per_scene")
    selected = select_training_records(
        selected,
        max_questions_per_scene=None if configured_cap is None else int(configured_cap),
    )
    all_units = build_exact_question_pair_units(selected)
    all_selected_text = "\n".join(f"{record.scene_id}:{record.question_id}" for record in selected)
    all_selected_hash = hashlib.sha256(all_selected_text.encode("utf-8")).hexdigest()
    if all_selected_hash != expected_all_selected_records_sha256:
        raise ValueError(
            "All-record training selection hash mismatch: "
            f"observed={all_selected_hash} expected={expected_all_selected_records_sha256}"
        )
    units = [unit for unit in all_units if unit.pair_id == pair_id]
    if len(units) != expected_count:
        raise ValueError(f"Selected mirror-unit count mismatch: {len(units)} != {expected_count}")
    if any(record.answer_type != "spatial_relation" for unit in units for record in unit.records):
        raise ValueError("Selected probe pair contains a non-spatial-relation record")
    key_text = "\n".join(f"{unit.pair_id}:{unit.question_key}" for unit in units)
    key_hash = hashlib.sha256(key_text.encode("utf-8")).hexdigest()
    content_text = "\n".join(
        (
            f"{unit.pair_id}:{unit.question_key}:{record.counterfactual_role}:"
            f"{record.scene_id}:{record.question_id}:"
            f"{hashlib.sha256(record.question.encode()).hexdigest()}:{record.answer}"
        )
        for unit in units
        for record in unit.records
    )
    content_hash = hashlib.sha256(content_text.encode("utf-8")).hexdigest()
    if not _HEX_SHA256.fullmatch(expected_sha256) or key_hash != expected_sha256:
        raise ValueError(
            f"Mirror unit-key hash mismatch: observed={key_hash} expected={expected_sha256}"
        )
    if (
        not _HEX_SHA256.fullmatch(expected_content_sha256)
        or content_hash != expected_content_sha256
    ):
        raise ValueError(
            "Mirror unit-content hash mismatch: "
            f"observed={content_hash} expected={expected_content_sha256}"
        )
    pairs = training_counterfactual_scene_pairs(selected)
    membership_text = "\n".join(
        f"{one_pair_id}:{scene_a}:{scene_b}" for one_pair_id, scene_a, scene_b in pairs
    )
    return units, {
        "pair_id": pair_id,
        "unit_count": len(units),
        "side_count": 2 * len(units),
        "unit_keys_sha256": key_hash,
        "unit_content_sha256": content_hash,
        "all_selected_record_ids_sha256": all_selected_hash,
        "all_selected_pair_unit_count": len(all_units),
        "training_counterfactual_pair_count": len(pairs),
        "training_counterfactual_pair_membership_sha256": hashlib.sha256(
            membership_text.encode("utf-8")
        ).hexdigest(),
        "scene_ids": sorted({scene for unit in units for scene in unit.scene_ids}),
    }


def validate_checkpoint_provenance(metadata: Mapping[str, Any], repository: Path) -> dict[str, Any]:
    """Require an internally valid clean historical source commit and a clean current HEAD."""

    checkpoint = metadata.get("source_provenance")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint has no source-provenance mapping")
    valid = (
        checkpoint.get("schema_version") == 1
        and checkpoint.get("scope") == SOURCE_SCOPE
        and checkpoint.get("available") is True
        and checkpoint.get("is_clean") is True
        and _GIT_OBJECT.fullmatch(str(checkpoint.get("head_commit", ""))) is not None
        and _GIT_OBJECT.fullmatch(str(checkpoint.get("head_tree", ""))) is not None
        and checkpoint.get("tracked_diff_sha256") == _EMPTY_SHA256
    )
    if not valid:
        raise ValueError(f"Invalid or dirty checkpoint source provenance: {dict(checkpoint)}")
    commit = str(checkpoint["head_commit"])
    tree = subprocess.run(
        ("git", "-C", str(repository), "rev-parse", f"{commit}^{{tree}}"),
        check=False,
        capture_output=True,
        text=True,
    )
    if tree.returncode != 0 or tree.stdout.strip() != checkpoint["head_tree"]:
        raise ValueError("Checkpoint source commit/tree cannot be verified in the repository")
    ancestor = subprocess.run(
        ("git", "-C", str(repository), "merge-base", "--is-ancestor", commit, "HEAD"),
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise ValueError("Checkpoint source commit is not an ancestor of current HEAD")
    current = capture_git_source_provenance(repository)
    require_clean_committed_source(current)
    return {"checkpoint": dict(checkpoint), "probe_source": current}


def _named_tensor_refs(prefix: str, module: nn.Module) -> dict[str, torch.Tensor]:
    result = {
        f"{prefix}.parameter.{name}": parameter for name, parameter in module.named_parameters()
    }
    result.update({f"{prefix}.buffer.{name}": buffer for name, buffer in module.named_buffers()})
    return result


def _hash_gradients(gradients: Mapping[str, torch.Tensor]) -> str:
    return tensor_state_sha256(gradients)


def _accumulate_gradient_sets(
    values: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not values:
        raise ValueError("Cannot aggregate an empty gradient collection")
    names = tuple(values[0])
    if any(tuple(value) != names for value in values):
        raise ValueError("Gradient collections use inconsistent parameter order")
    return {name: sum(value[name] for value in values) for name in names}


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_layers(value: str) -> tuple[int, ...]:
    try:
        layers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from exc
    try:
        ProbeBankSpec(layers=layers)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return layers


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _checkpoint_preflight(
    config: dict[str, Any],
    checkpoint: Path,
    metadata: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    expected_epoch: int,
    expected_checkpoint_sha256: str | None,
) -> dict[str, Any]:
    if int(metadata.get("epoch", -1)) != expected_epoch:
        raise ValueError(f"Checkpoint epoch mismatch: {metadata.get('epoch')} != {expected_epoch}")
    observed_checkpoint_sha = sha256_file(checkpoint / "adapter.safetensors")
    if expected_checkpoint_sha256 is not None and (
        not _HEX_SHA256.fullmatch(expected_checkpoint_sha256)
        or observed_checkpoint_sha != expected_checkpoint_sha256
    ):
        raise ValueError(
            "Checkpoint SHA-256 mismatch: "
            f"observed={observed_checkpoint_sha} expected={expected_checkpoint_sha256}"
        )
    for key in (
        "counterfactual_pair_unit_count",
        "training_counterfactual_pair_count",
        "training_counterfactual_pair_membership_sha256",
    ):
        observed = selection[
            "all_selected_pair_unit_count" if key == "counterfactual_pair_unit_count" else key
        ]
        if metadata.get(key) != observed:
            raise ValueError(
                f"Checkpoint/selection contract mismatch for {key}: "
                f"{metadata.get(key)} != {observed}"
            )
    if metadata.get("output_namespace") != config["training"].get("output_namespace"):
        raise ValueError("Checkpoint output namespace does not match probe configuration")
    return {
        "adapter_sha256": observed_checkpoint_sha,
        "metadata_sha256": sha256_file(checkpoint / "metadata.json"),
        "epoch": int(metadata["epoch"]),
        "best_epoch": metadata.get("best_epoch"),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    config_path = _resolve_project_path(args.config)
    checkpoint = _resolve_project_path(args.checkpoint)
    output = _resolve_project_path(args.output)
    config = load_config(config_path)
    metadata_path = checkpoint / "metadata.json"
    if not metadata_path.is_file() or not (checkpoint / "adapter.safetensors").is_file():
        raise FileNotFoundError(f"Incomplete adapter checkpoint: {checkpoint}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError("Checkpoint metadata must be a JSON object")
    provenance = validate_checkpoint_provenance(metadata, PROJECT_ROOT)
    units, selection = selected_mirror_units(
        config,
        pair_id=args.pair_id,
        expected_count=args.expected_unit_count,
        expected_sha256=args.expected_selection_sha256,
        expected_content_sha256=args.expected_selection_content_sha256,
        expected_all_selected_records_sha256=args.expected_all_selected_records_sha256,
    )
    checkpoint_identity = _checkpoint_preflight(
        config,
        checkpoint,
        metadata,
        selection,
        expected_epoch=args.expected_epoch,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
    )
    settings = pair_curriculum_settings(config)
    if settings.ranking_weight <= 0 or settings.full_vocab_ranking_weight <= 0:
        raise ValueError("Probe requires positive candidate and full-vocabulary weights")

    language = load_local_language_model(
        str(config["language"]["model_id"]),
        str(config["language"]["revision"]),
        str(config["language"]["dtype"]),
        freeze=True,
        local_files_only=True,
        backend=str(config["language"].get("backend", "auto")),
    )
    persisted_lora_settings = lora_settings(config)
    lora_optimizer_settings(config, persisted_lora_settings)
    persisted_lora = install_lora_adapters(language.model, persisted_lora_settings)
    if persisted_lora is None:
        raise ValueError("V12 checkpoint must contain the existing layer-34 LoRA bank")
    expected_persisted_targets = (
        "model.language_model.layers.34.self_attn.q_proj",
        "model.language_model.layers.34.self_attn.o_proj",
    )
    if persisted_lora.target_names != expected_persisted_targets:
        raise ValueError(
            "Persisted LoRA target contract is not exact layer-34 q/o: "
            f"{persisted_lora.target_names}"
        )
    warnings = validate_checkpoint_contract(
        metadata,
        config,
        semantic_dim=int(metadata["semantic_dim"]),
        language_hidden_dim=language.hidden_size,
        lora_parameter_count=persisted_lora.parameter_count,
    )
    if warnings:
        raise ValueError(f"Checkpoint config contract produced warnings: {warnings}")

    scene_model = construct_scene_tokenizer(
        config, int(metadata["semantic_dim"]), language.hidden_size
    )
    boundary_mode = scene_boundary_mode_setting(config)
    native_contract = language.scene_boundary_contract(boundary_mode)
    if native_contract != native_gemma4_image_contract_setting(config):
        raise ValueError("Loaded Gemma boundary contract does not match V12 configuration")
    composer = ContinuousPrefixComposer(
        language.hidden_size,
        scene_prefix_after_bos=scene_prefix_after_bos_setting(config),
        bos_token_id=language.bos_token_id,
        scene_boundary_mode=boundary_mode,
        native_boundary_embeddings=language.scene_boundary_embeddings(boundary_mode),
    )
    grounding = QuestionGroundingHead(
        int(config["scene_encoder"]["model_dim"]),
        language.hidden_size,
        int(config["scene_encoder"]["global_latents"]),
        int(config["scene_encoder"]["model_dim"]),
    )
    loaded_metadata = load_adapter_checkpoint(
        checkpoint,
        {
            "scene_model": scene_model,
            "composer": composer,
            "grounding": grounding,
            "lora": persisted_lora.state_module,
        },
        device="cpu",
    )
    if loaded_metadata != metadata:
        raise RuntimeError("Checkpoint metadata changed during exact adapter load")
    validate_lora_checkpoint_state(metadata, persisted_lora)
    device = language.device
    scene_model = scene_model.to(device).eval().requires_grad_(False)
    composer = composer.to(device).eval().requires_grad_(False)
    grounding = grounding.to(device).eval().requires_grad_(False)
    language.model.eval().requires_grad_(False)

    protected = {}
    for prefix, module in (
        ("language", language.model),
        ("scene_model", scene_model),
        ("composer", composer),
        ("grounding", grounding),
    ):
        protected.update(_named_tensor_refs(prefix, module))
    protected_before = tensor_state_sha256(protected)

    scene_ids = sorted({scene_id for unit in units for scene_id in unit.scene_ids})
    scene_tokens: dict[str, torch.Tensor] = {}
    prefix_hashes: dict[str, str] = {}
    map_hashes: dict[str, str] = {}
    # ``no_grad`` (not ``inference_mode``) is intentional: the frozen scene
    # tensor is later consumed by decoder operations that must save inputs for
    # gradients into the temporary LoRA parameters.
    with torch.no_grad():
        for scene_id in scene_ids:
            map_path = project_path(config, "maps", scene_id, "voxel_map.npz")
            map_hashes[scene_id] = sha256_file(map_path)
            data = load_map_tensors(
                map_path,
                config["scene"]["room_size_m"],
                device=device,
                input_voxel_size_m=config["scene_encoder"].get("input_voxel_size_m"),
            )
            encoded = map_forward(scene_model, data)
            processed = int(encoded.audit["processed_voxels"].detach().cpu())
            if processed != data.voxel_count:
                raise RuntimeError(
                    f"Incomplete scene encoding for {scene_id}: {processed}/{data.voxel_count}"
                )
            tokens = encoded.scene_tokens.to(next(language.model.parameters()).dtype).detach()
            require_autograd_compatible_scene_tokens(tokens)
            scene_tokens[scene_id] = tokens
            prefix_hashes[scene_id] = hashlib.sha256(
                composer.scene_prefix(tokens).detach().float().cpu().contiguous().numpy().tobytes()
            ).hexdigest()

    # Capture the complete first-answer vocabulary distribution from the exact
    # persisted V12 model before adding the temporary bank.  The same 12
    # distributions must remain bitwise equal with zero B at probe step zero.
    parity_baseline: dict[tuple[str, str], torch.Tensor] = {}
    with torch.inference_mode():
        for unit in units:
            reference_ids = tokenize_answer(
                language.tokenizer, unit.reference.answer, language.device
            )
            counterfactual_ids = tokenize_answer(
                language.tokenizer, unit.counterfactual.answer, language.device
            )
            answer_offset, _, _ = single_differing_answer_token(reference_ids, counterfactual_ids)
            if answer_offset != 0:
                raise ValueError("Mirror candidate token must be the first answer token")
            for side_name, record in (
                ("reference", unit.reference),
                ("counterfactual", unit.counterfactual),
            ):
                prompt_ids = prompt_token_ids(
                    language.tokenizer,
                    config["language"]["system_prompt"],
                    record.question,
                    language.device,
                )
                answer_ids = tokenize_answer(language.tokenizer, record.answer, language.device)
                batch = composer.compose(
                    scene_tokens[record.scene_id],
                    prompt_ids,
                    language.model.get_input_embeddings(),
                    answer_ids,
                    prefix_backend=getattr(language, "prefix_backend", None),
                )
                baseline_output = forward_prefix_batch(language, batch)
                if batch.labels is None:
                    raise RuntimeError("Teacher-forced parity composition returned no labels")
                parity_baseline[(unit.question_key, side_name)] = (
                    first_answer_distribution(baseline_output.logits, batch.labels)
                    .detach()
                    .cpu()
                    .contiguous()
                )
                del baseline_output, batch

    spec = ProbeBankSpec(
        layers=args.layers,
        rank=args.rank,
        alpha=args.alpha,
        seed=args.seed,
    )
    if set(spec.target_modules) & set(persisted_lora.target_names):
        raise ValueError("Probe and persisted LoRA target banks overlap")
    probe_bank = install_lora_adapters(language.model, spec.settings)
    if probe_bank is None:  # pragma: no cover - enabled spec makes this impossible
        raise RuntimeError("Probe LoRA installation unexpectedly returned None")
    initialize_probe_bank(probe_bank, seed=spec.seed)
    probe_parameters = probe_parameter_items(probe_bank)
    probe_before = tensor_state_sha256(probe_bank.state_module.state_dict())
    probe_bank.assert_only_lora_trainable(language.model)

    objective_names = (
        "language_nll",
        "candidate_hinge_weighted",
        "full_vocab_hinge_weighted",
        "decoder_total",
    )
    unit_reports: list[dict[str, Any]] = []
    aggregate_sides: dict[str, dict[str, list[dict[str, torch.Tensor]]]] = {
        objective: {"reference": [], "counterfactual": []} for objective in objective_names
    }
    parity_records: list[dict[str, Any]] = []
    for unit in units:
        reference_answer_ids = tokenize_answer(
            language.tokenizer, unit.reference.answer, language.device
        )
        counterfactual_answer_ids = tokenize_answer(
            language.tokenizer, unit.counterfactual.answer, language.device
        )
        answer_offset, reference_token, counterfactual_token = single_differing_answer_token(
            reference_answer_ids, counterfactual_answer_ids
        )
        side_results: dict[str, Any] = {}
        side_gradients: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
        for side_name, record, candidate_spec in (
            (
                "reference",
                unit.reference,
                (answer_offset, reference_token, counterfactual_token),
            ),
            (
                "counterfactual",
                unit.counterfactual,
                (answer_offset, counterfactual_token, reference_token),
            ),
        ):
            prompt_ids = prompt_token_ids(
                language.tokenizer,
                config["language"]["system_prompt"],
                record.question,
                language.device,
            )
            answer_ids = tokenize_answer(language.tokenizer, record.answer, language.device)
            batch = composer.compose(
                scene_tokens[record.scene_id],
                prompt_ids,
                language.model.get_input_embeddings(),
                answer_ids,
                prefix_backend=getattr(language, "prefix_backend", None),
            )
            output_value = forward_prefix_batch(language, batch)
            if batch.labels is None:
                raise RuntimeError("Teacher-forced probe composition returned no labels")
            baseline_distribution = parity_baseline[(unit.question_key, side_name)]
            probe_distribution = (
                first_answer_distribution(output_value.logits, batch.labels)
                .detach()
                .cpu()
                .contiguous()
            )
            parity_equal = torch.equal(baseline_distribution, probe_distribution)
            baseline_hash = tensor_state_sha256({"full_vocab_logits": baseline_distribution})
            probe_hash = tensor_state_sha256({"full_vocab_logits": probe_distribution})
            if not parity_equal or baseline_hash != probe_hash:
                raise RuntimeError(
                    "Zero-output probe bank changed step-zero first-answer logits: "
                    f"{unit.question_key}/{side_name}"
                )
            parity_records.append(
                {
                    "question_key": unit.question_key,
                    "side": side_name,
                    "vocabulary_size": int(probe_distribution.numel()),
                    "baseline_sha256": baseline_hash,
                    "zero_output_probe_sha256": probe_hash,
                    "bitwise_equal": parity_equal,
                }
            )
            terms, diagnostics = side_objective_terms(
                output_value.logits,
                batch.labels,
                candidate_spec=candidate_spec,
                candidate_margin=settings.ranking_margin,
                candidate_weight=settings.ranking_weight,
                full_vocab_margin=settings.full_vocab_ranking_margin,
                full_vocab_weight=settings.full_vocab_ranking_weight,
            )
            gradients = objective_gradients(terms, probe_parameters)
            side_gradients[side_name] = gradients
            for objective in objective_names:
                aggregate_sides[objective][side_name].append(gradients[objective])
            side_results[side_name] = {
                "scene_id": record.scene_id,
                "question_id": record.question_id,
                "answer_token_id": candidate_spec[1],
                "alternate_token_id": candidate_spec[2],
                "objective_values": {
                    name: float(value.detach().cpu()) for name, value in terms.items()
                },
                "diagnostics": diagnostics,
                "gradient_sha256": {
                    name: _hash_gradients(gradients[name]) for name in objective_names
                },
            }
            del output_value, batch, terms, gradients
        unit_reports.append(
            {
                "pair_id": unit.pair_id,
                "question_key": unit.question_key,
                "question_sha256": hashlib.sha256(unit.reference.question.encode()).hexdigest(),
                "sides": side_results,
                "cancellation": {
                    objective: grouped_cancellation_metrics(
                        side_gradients["reference"][objective],
                        side_gradients["counterfactual"][objective],
                    )
                    for objective in objective_names
                },
            }
        )

    aggregate_report: dict[str, Any] = {}
    for objective in objective_names:
        reference = _accumulate_gradient_sets(aggregate_sides[objective]["reference"])
        counterfactual = _accumulate_gradient_sets(aggregate_sides[objective]["counterfactual"])
        aggregate_report[objective] = {
            **grouped_cancellation_metrics(reference, counterfactual),
            "reference_gradient_sha256": _hash_gradients(reference),
            "counterfactual_gradient_sha256": _hash_gradients(counterfactual),
        }

    if any(parameter.grad is not None for parameter in language.model.parameters()):
        raise RuntimeError("Probe unexpectedly populated parameter .grad fields")
    protected_after = tensor_state_sha256(protected)
    probe_after = tensor_state_sha256(probe_bank.state_module.state_dict())
    no_weight_change = protected_before == protected_after and probe_before == probe_after
    if not no_weight_change:
        raise RuntimeError("A model or probe-bank weight changed during no-step differentiation")

    payload = {
        "schema_version": 1,
        "purpose": "V13 extra-LoRA mirrored-gradient cancellation falsification probe",
        "no_optimizer_constructed": True,
        "optimizer_steps": 0,
        "pair_side_reduction_scale": 0.5,
        "objective_contract": {
            "formula_per_side": (
                "0.5 * (language_nll + candidate_weight * relu(candidate_margin - "
                "candidate_logit_margin) + full_vocab_weight * relu(full_vocab_margin - "
                "target_vs_best_other_margin))"
            ),
            "candidate_weight": settings.ranking_weight,
            "candidate_margin": settings.ranking_margin,
            "full_vocab_weight": settings.full_vocab_ranking_weight,
            "full_vocab_margin": settings.full_vocab_ranking_margin,
            "same_correct_teacher_forced_distribution": True,
        },
        "config": {
            "path": str(config_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(config_path),
        },
        "checkpoint": {
            "path": str(checkpoint.relative_to(PROJECT_ROOT)),
            **checkpoint_identity,
            "persisted_lora": metadata["lora"],
            "persisted_lora_state_sha256": metadata["lora_state_sha256"],
        },
        "source_provenance": provenance,
        "selection": selection,
        "qa_train_sha256": sha256_file(artifact_root(config, "qa") / "train.jsonl"),
        "map_sha256": map_hashes,
        "scene_prefix_sha256": prefix_hashes,
        "step_zero_logit_parity": {
            "scope": "complete first-answer vocabulary distribution for every selected side",
            "side_count": len(parity_records),
            "all_bitwise_equal": all(record["bitwise_equal"] for record in parity_records),
            "records": parity_records,
        },
        "device": str(language.device),
        "probe_bank": {
            **spec.contract(),
            "parameter_count": probe_bank.parameter_count,
            "parameter_counts": probe_bank.parameter_counts,
            "state_sha256_before": probe_before,
            "state_sha256_after": probe_after,
            "all_b_exactly_zero_after": all(
                torch.count_nonzero(adapter.lora_b).item() == 0 for adapter in probe_bank.adapters
            ),
        },
        "protected_persisted_state_sha256_before": protected_before,
        "protected_persisted_state_sha256_after": protected_after,
        "no_weight_change": no_weight_change,
        "unit_results": unit_reports,
        "aggregate_across_six_units": aggregate_report,
    }
    _atomic_write_json(output, payload)
    print(
        json.dumps(
            {
                "phase": "mirrored_gradient_probe_complete",
                "output": str(output),
                "unit_count": len(unit_reports),
                "decoder_total": aggregate_report["decoder_total"]["aggregate"],
                "no_weight_change": no_weight_change,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output",
        default="reports/gemma4/metrics/mirrored_gradient_probe_v13_epoch008.json",
    )
    parser.add_argument("--pair-id", default="pair_000003")
    parser.add_argument("--expected-unit-count", type=int, default=6)
    parser.add_argument("--expected-selection-sha256", default=DEFAULT_SELECTION_SHA256)
    parser.add_argument(
        "--expected-selection-content-sha256", default=DEFAULT_SELECTION_CONTENT_SHA256
    )
    parser.add_argument(
        "--expected-all-selected-records-sha256",
        default=DEFAULT_ALL_SELECTED_RECORDS_SHA256,
    )
    parser.add_argument("--expected-epoch", type=int, default=8)
    parser.add_argument("--expected-checkpoint-sha256", default=DEFAULT_CHECKPOINT_SHA256)
    parser.add_argument("--layers", type=_parse_layers, default=(30, 31, 32, 33))
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=13_008)
    args = parser.parse_args()
    if args.expected_unit_count < 1:
        parser.error("--expected-unit-count must be positive")
    run_probe(args)


if __name__ == "__main__":
    main()
