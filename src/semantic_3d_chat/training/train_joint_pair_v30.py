"""Bounded V30 joint counterfactual repair starting from approved V29.

The environmental path remains question-independent.  Each numeric map is
encoded once into the complete frozen pre-sidecar scene stack; the resulting
base and all-voxel aligned tensors are cached before any question is read.
Only the post-stack sidecar output projection/gain and one fresh, disjoint
decoder LoRA bank are optimized.  Every changed-answer counterfactual unit is
kept atomic and oversampled in every optimizer cycle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from semantic_3d_chat.chat.runtime import (
    construct_scene_tokenizer,
    validate_checkpoint_contract,
)
from semantic_3d_chat.config import PROJECT_ROOT, artifact_root, config_hash, load_config
from semantic_3d_chat.data.dataset import QARecord
from semantic_3d_chat.language.local_lm import load_local_language_model, prompt_token_ids
from semantic_3d_chat.language.lora import (
    LoRABankCollection,
    LoRALinear,
    install_lora_banks,
    lora_banks_checkpoint_contract,
    lora_banks_optimizer_settings,
    lora_banks_settings,
    tensor_state_sha256,
)
from semantic_3d_chat.language.prefix_injection import (
    ContinuousPrefixComposer,
    native_gemma4_image_contract_setting,
    prefix_sha256,
    scene_boundary_mode_setting,
    scene_prefix_after_bos_setting,
    stack_prefix_batches,
)
from semantic_3d_chat.scene_encoder.dense_alignment import (
    DenseAlignmentResidual,
    construct_dense_alignment,
    validate_dense_alignment_state,
)
from semantic_3d_chat.scene_encoder.dense_sidecar_adapter import (
    DenseSidecarAdapter,
    construct_dense_sidecar_adapter,
    validate_dense_sidecar_adapter_state,
)
from semantic_3d_chat.scene_encoder.global_residual import (
    GlobalSceneResidual,
    construct_global_scene_residual,
)
from semantic_3d_chat.scene_encoder.map_io import load_map_tensors
from semantic_3d_chat.scene_encoder.projector import SceneTokenizer
from semantic_3d_chat.scene_encoder.signed_x_dispatch import (
    SignedXSceneResidual,
    construct_signed_x_scene_residual,
)
from semantic_3d_chat.training.checkpointing import (
    RUNTIME_METADATA_FILENAME,
    TRAINING_METADATA_FILENAME,
    load_adapter_checkpoint,
    module_collection_state_sha256,
    save_adapter_checkpoint,
    save_optimizer_checkpoint,
    validate_runtime_checkpoint_metadata,
)
from semantic_3d_chat.training.losses import QuestionGroundingHead
from semantic_3d_chat.training.pair_curriculum import (
    CounterfactualPairUnit,
    build_exact_question_pair_units,
    differing_answer_token_masks,
    pair_ranking_hinge,
    restrict_labels_to_answer_mask,
    token_normalized_nll,
)
from semantic_3d_chat.training.train_adapter import (
    forward_prefix_batch,
    map_forward,
    named_lora_extension_checkpoint_modules,
    named_lora_freeze_and_extend_transition_mismatch,
    tokenize_answer,
    validate_named_lora_extension_transition_state,
)
from semantic_3d_chat.training.train_post_stack_decoder import (
    _file_sha256,
    _finite_number,
    _positive_int,
    _read_metadata,
    _source_validation_nll,
    load_stage_b_qa_records,
    records_by_scene,
)

DEFAULT_CONFIG = Path("configs/experiments/gemma4_diverse20_joint_pair_v30.yaml")
DEFAULT_OUTPUT = Path("data_gemma4/checkpoints/gemma4_v30_diverse20_joint_pair")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SIDECAR_TRAINABLE_NAMES = ("output_projection.weight", "channel_gain")


@dataclass(frozen=True)
class V30Settings:
    enabled: bool
    max_optimizer_steps: int
    evaluation_interval_steps: int
    broad_questions_per_cycle: int
    broad_batch_size: int
    broad_exclude_expected_change: bool
    pair_repeats_per_cycle: int
    pair_units_per_batch: int
    broad_nll_weight: float
    pair_language_nll_weight: float
    pair_margin_weight: float
    pair_margin: float
    sidecar_learning_rate: float
    decoder_learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    minimum_answer_types: int
    trainable_bank: str


@dataclass(frozen=True)
class ApprovedV29Source:
    checkpoint: Path
    selection_report: Path
    selection_sha256: str
    selected_update: int
    selected_arm: dict[str, Any]


@dataclass(frozen=True)
class CachedPreSidecarScene:
    scene_id: str
    base_scene_tokens: torch.Tensor
    aligned_sidecar_tokens: torch.Tensor
    source_prefix_sha256: str
    voxel_count: int
    processed_voxels: int
    minimum_voxel_contribution: float


@dataclass
class V30Bundle:
    config: dict[str, Any]
    source_config: dict[str, Any]
    source_runtime_metadata: dict[str, Any]
    source_training_metadata: dict[str, Any]
    source: ApprovedV29Source
    language: Any
    scene_model: SceneTokenizer
    dense_aligner: DenseAlignmentResidual
    dense_sidecar_adapter: DenseSidecarAdapter
    global_scene_residual: GlobalSceneResidual
    signed_x_scene_residual: SignedXSceneResidual
    composer: ContinuousPrefixComposer
    grounding: QuestionGroundingHead
    lora_installation: LoRABankCollection
    checkpoint_modules: dict[str, torch.nn.Module]
    trainable_bank_name: str


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _boolean(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def v30_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config.get("v30_joint_pair")
    if not isinstance(raw, Mapping):
        raise TypeError("V30 requires a v30_joint_pair mapping")
    required = {
        "schema_version",
        "role",
        "source_config",
        "source_selection_report",
        "source_selection_report_sha256",
        "source_checkpoint_root",
        "source_selected_update",
        "source_adapter_sha256",
        "source_runtime_metadata_sha256",
        "source_sidecar_state_sha256",
        "source_v28_bank_state_sha256",
        "fresh_bank",
        "fresh_bank_parameter_count",
        "fresh_bank_initial_state_sha256",
        "sidecar_trainable_parameter_names",
        "sidecar_trainable_parameter_count",
        "joint_trainable_parameter_count",
        "update_zero_validation_nll_absolute_tolerance",
        "validation_pair_unit_count",
        "selection_requires",
        "promotion_requires",
    }
    missing = sorted(required - set(raw))
    unknown = sorted(set(raw) - required)
    if missing or unknown:
        raise ValueError(f"Invalid v30_joint_pair fields: missing={missing} unknown={unknown}")
    if raw["schema_version"] != 1:
        raise ValueError("v30_joint_pair.schema_version must be 1")
    if raw["role"] != "approved_v29_joint_sidecar_decoder_counterfactual_repair":
        raise ValueError("v30_joint_pair.role does not authorize this transition")
    for field in (
        "source_selection_report_sha256",
        "source_adapter_sha256",
        "source_runtime_metadata_sha256",
        "source_sidecar_state_sha256",
        "source_v28_bank_state_sha256",
        "fresh_bank_initial_state_sha256",
    ):
        if not isinstance(raw[field], str) or _SHA256.fullmatch(raw[field]) is None:
            raise ValueError(f"v30_joint_pair.{field} must be SHA-256")
    _positive_int("v30_joint_pair.source_selected_update", raw["source_selected_update"])
    fresh_count = _positive_int(
        "v30_joint_pair.fresh_bank_parameter_count", raw["fresh_bank_parameter_count"]
    )
    sidecar_count = _positive_int(
        "v30_joint_pair.sidecar_trainable_parameter_count",
        raw["sidecar_trainable_parameter_count"],
    )
    joint_count = _positive_int(
        "v30_joint_pair.joint_trainable_parameter_count",
        raw["joint_trainable_parameter_count"],
    )
    if joint_count != fresh_count + sidecar_count:
        raise ValueError("V30 joint parameter count is not the sum of its two surfaces")
    names = raw["sidecar_trainable_parameter_names"]
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
        raise TypeError("v30_joint_pair.sidecar_trainable_parameter_names must be a sequence")
    if tuple(names) != _SIDECAR_TRAINABLE_NAMES:
        raise ValueError("V30 sidecar surface must be output_projection.weight plus channel_gain")
    _finite_number(
        "v30_joint_pair.update_zero_validation_nll_absolute_tolerance",
        raw["update_zero_validation_nll_absolute_tolerance"],
        positive=False,
    )
    _positive_int("v30_joint_pair.validation_pair_unit_count", raw["validation_pair_unit_count"])
    if not isinstance(raw["fresh_bank"], str) or not raw["fresh_bank"]:
        raise TypeError("v30_joint_pair.fresh_bank must be nonempty")
    if not isinstance(raw["selection_requires"], Mapping):
        raise TypeError("v30_joint_pair.selection_requires must be a mapping")
    promotion = raw["promotion_requires"]
    if not isinstance(promotion, Mapping) or set(promotion) != {
        "validation_changed_complete_pairs_minimum",
        "aggregate_validation_exact_accuracy_no_regression",
        "label",
    }:
        raise ValueError("v30_joint_pair.promotion_requires has invalid fields")
    _positive_int(
        "v30_joint_pair.promotion_requires.validation_changed_complete_pairs_minimum",
        promotion["validation_changed_complete_pairs_minimum"],
    )
    if promotion["aggregate_validation_exact_accuracy_no_regression"] is not True:
        raise ValueError("V30 chat promotion must forbid aggregate validation regression")
    if promotion["label"] != "chat_promotion_not_merely_development_progress":
        raise ValueError("V30 chat-promotion label is invalid")
    return dict(raw)


def v30_settings(config: Mapping[str, Any]) -> V30Settings:
    training = config.get("training")
    if not isinstance(training, Mapping):
        raise TypeError("training config must be a mapping")
    raw = training.get("v30_joint_pair")
    if not isinstance(raw, Mapping):
        raise TypeError("training.v30_joint_pair must be a mapping")
    required = {
        "enabled",
        "max_optimizer_steps",
        "evaluation_interval_steps",
        "broad_questions_per_cycle",
        "broad_batch_size",
        "broad_exclude_expected_change",
        "pair_repeats_per_cycle",
        "pair_units_per_batch",
        "broad_nll_weight",
        "pair_language_nll_weight",
        "pair_margin_weight",
        "pair_margin",
        "sidecar_learning_rate",
        "decoder_learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "minimum_answer_types",
        "trainable_bank",
    }
    missing = sorted(required - set(raw))
    unknown = sorted(set(raw) - required)
    if missing or unknown:
        raise ValueError(
            f"Invalid training.v30_joint_pair fields: missing={missing} unknown={unknown}"
        )
    result = V30Settings(
        enabled=_boolean("training.v30_joint_pair.enabled", raw["enabled"]),
        max_optimizer_steps=_positive_int("max_optimizer_steps", raw["max_optimizer_steps"]),
        evaluation_interval_steps=_positive_int(
            "evaluation_interval_steps", raw["evaluation_interval_steps"]
        ),
        broad_questions_per_cycle=_positive_int(
            "broad_questions_per_cycle", raw["broad_questions_per_cycle"]
        ),
        broad_batch_size=_positive_int("broad_batch_size", raw["broad_batch_size"]),
        broad_exclude_expected_change=_boolean(
            "broad_exclude_expected_change", raw["broad_exclude_expected_change"]
        ),
        pair_repeats_per_cycle=_positive_int(
            "pair_repeats_per_cycle", raw["pair_repeats_per_cycle"]
        ),
        pair_units_per_batch=_positive_int("pair_units_per_batch", raw["pair_units_per_batch"]),
        broad_nll_weight=_finite_number("broad_nll_weight", raw["broad_nll_weight"], positive=True),
        pair_language_nll_weight=_finite_number(
            "pair_language_nll_weight", raw["pair_language_nll_weight"], positive=True
        ),
        pair_margin_weight=_finite_number(
            "pair_margin_weight", raw["pair_margin_weight"], positive=True
        ),
        pair_margin=_finite_number("pair_margin", raw["pair_margin"], positive=True),
        sidecar_learning_rate=_finite_number(
            "sidecar_learning_rate", raw["sidecar_learning_rate"], positive=True
        ),
        decoder_learning_rate=_finite_number(
            "decoder_learning_rate", raw["decoder_learning_rate"], positive=True
        ),
        weight_decay=_finite_number("weight_decay", raw["weight_decay"], positive=False),
        gradient_clip_norm=_finite_number(
            "gradient_clip_norm", raw["gradient_clip_norm"], positive=True
        ),
        minimum_answer_types=_positive_int("minimum_answer_types", raw["minimum_answer_types"]),
        trainable_bank=str(raw["trainable_bank"]),
    )
    contract = v30_contract(config)
    if result.trainable_bank != contract["fresh_bank"]:
        raise ValueError("V30 settings and transition contract name different fresh banks")
    if result.sidecar_learning_rate != float(training["learning_rate"]):
        raise ValueError("V30 sidecar learning rate must equal training.learning_rate")
    if result.decoder_learning_rate != float(training["lora_learning_rate"]):
        raise ValueError("V30 decoder learning rate must equal training.lora_learning_rate")
    if result.weight_decay != float(training["weight_decay"]):
        raise ValueError("V30 weight decay must equal training.weight_decay")
    return result


def require_approved_v29_source(config: Mapping[str, Any]) -> ApprovedV29Source:
    contract = v30_contract(config)
    report_path = _resolve(str(contract["source_selection_report"]))
    if report_path.is_symlink() or not report_path.is_file():
        raise FileNotFoundError(f"V29 selector report is missing: {report_path}")
    report_sha = _file_sha256(report_path)
    if report_sha != contract["source_selection_report_sha256"]:
        raise ValueError("V29 selector report hash differs from the pinned V30 source")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    required_report = {
        "schema_version": 1,
        "artifact": "v28_post_stack_decoder_stage_b_selection",
        "training_evaluation_only": True,
        "question_text_serialized": False,
        "answer_text_serialized": False,
        "oracle_loaded": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "passed": True,
    }
    mismatch = {
        key: {"observed": report.get(key), "required": value}
        for key, value in required_report.items()
        if report.get(key) != value
    }
    if mismatch:
        raise ValueError(f"V29 selector did not approve the V30 source: {mismatch}")
    selected = report.get("selected_checkpoint")
    update = report.get("selected_update")
    if not isinstance(selected, str) or not selected:
        raise ValueError("V29 selector lacks selected_checkpoint")
    if isinstance(update, bool) or not isinstance(update, int):
        raise TypeError("V29 selector lacks integer selected_update")
    if update != int(contract["source_selected_update"]):
        raise ValueError("V29 selected update differs from the V30 pin")
    checkpoint = _resolve(selected)
    root = _resolve(str(contract["source_checkpoint_root"]))
    if not checkpoint.is_relative_to(root) or checkpoint.name != f"update_{update:03d}":
        raise ValueError("V29 selected checkpoint lies outside its pinned root/update")
    for name in ("adapter.safetensors", RUNTIME_METADATA_FILENAME, TRAINING_METADATA_FILENAME):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"V29 selected checkpoint lacks {name}")
    if _file_sha256(checkpoint / "adapter.safetensors") != contract["source_adapter_sha256"]:
        raise ValueError("V29 source adapter SHA-256 mismatch")
    if (
        _file_sha256(checkpoint / RUNTIME_METADATA_FILENAME)
        != contract["source_runtime_metadata_sha256"]
    ):
        raise ValueError("V29 source runtime metadata SHA-256 mismatch")
    arms = report.get("arms")
    if not isinstance(arms, list):
        raise TypeError("V29 selector report lacks arms")
    matching = [
        arm
        for arm in arms
        if isinstance(arm, Mapping)
        and _resolve(str(arm.get("checkpoint", ""))) == checkpoint
        and arm.get("update") == update
        and arm.get("eligible") is True
    ]
    if len(matching) != 1:
        raise ValueError("V29 source is not one unique eligible selector arm")
    return ApprovedV29Source(
        checkpoint=checkpoint,
        selection_report=report_path,
        selection_sha256=report_sha,
        selected_update=update,
        selected_arm=dict(matching[0]),
    )


def _checkpoint_modules(
    *,
    scene_model: SceneTokenizer,
    composer: ContinuousPrefixComposer,
    grounding: QuestionGroundingHead,
    dense_aligner: DenseAlignmentResidual,
    sidecar: DenseSidecarAdapter,
    global_residual: GlobalSceneResidual,
    signed_x_residual: SignedXSceneResidual,
    lora: LoRABankCollection,
) -> dict[str, torch.nn.Module]:
    return {
        "scene_model": scene_model,
        "composer": composer,
        "grounding": grounding,
        "global_scene_residual": global_residual,
        "signed_x_scene_residual": signed_x_residual,
        "dense_aligner": dense_aligner,
        "dense_sidecar_adapter": sidecar,
        **lora.state_modules(),
    }


def _sidecar_trainable_parameters(module: DenseSidecarAdapter) -> list[torch.nn.Parameter]:
    named = dict(module.named_parameters())
    if not set(_SIDECAR_TRAINABLE_NAMES).issubset(named):
        raise RuntimeError("Dense sidecar lacks the V30 output surfaces")
    return [named[name] for name in _SIDECAR_TRAINABLE_NAMES]


def freeze_for_v30(bundle: V30Bundle) -> list[torch.nn.Parameter]:
    bundle.language.model.requires_grad_(False)
    for module in bundle.checkpoint_modules.values():
        module.requires_grad_(False)
        module.eval()
    sidecar_parameters = _sidecar_trainable_parameters(bundle.dense_sidecar_adapter)
    for parameter in sidecar_parameters:
        parameter.requires_grad_(True)
    fresh = bundle.lora_installation.bank(bundle.trainable_bank_name).installation.parameters()
    for parameter in fresh:
        parameter.requires_grad_(True)
    bundle.dense_sidecar_adapter.train()
    bundle.lora_installation.train()
    bundle.lora_installation.assert_trainable_surface(bundle.language.model)
    return [*sidecar_parameters, *fresh]


def assert_v30_trainable_surface(
    bundle: V30Bundle, optimizer: torch.optim.Optimizer | None = None
) -> dict[str, Any]:
    contract = v30_contract(bundle.config)
    sidecar_named = dict(bundle.dense_sidecar_adapter.named_parameters())
    sidecar_ids = {id(sidecar_named[name]) for name in _SIDECAR_TRAINABLE_NAMES}
    fresh_bank = bundle.lora_installation.bank(bundle.trainable_bank_name)
    fresh_ids = {id(parameter) for parameter in fresh_bank.installation.parameters()}
    authorized = sidecar_ids | fresh_ids
    observed = {
        id(parameter)
        for module in bundle.checkpoint_modules.values()
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    if observed != authorized:
        raise RuntimeError(
            f"V30 trainable surface mismatch: expected={len(authorized)} observed={len(observed)}"
        )
    if optimizer is not None:
        optimizer_ids = {
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        }
        if optimizer_ids != authorized:
            raise RuntimeError("V30 optimizer contains unauthorized parameters")
    sidecar_count = sum(sidecar_named[name].numel() for name in _SIDECAR_TRAINABLE_NAMES)
    fresh_count = fresh_bank.installation.parameter_count
    total = sidecar_count + fresh_count
    if sidecar_count != int(contract["sidecar_trainable_parameter_count"]):
        raise RuntimeError("V30 sidecar parameter count mismatch")
    if fresh_count != int(contract["fresh_bank_parameter_count"]):
        raise RuntimeError("V30 fresh-bank parameter count mismatch")
    if total != int(contract["joint_trainable_parameter_count"]):
        raise RuntimeError("V30 joint parameter count mismatch")
    return {
        "sidecar_parameter_names": [
            f"dense_sidecar_adapter.{name}" for name in _SIDECAR_TRAINABLE_NAMES
        ],
        "sidecar_parameter_count": sidecar_count,
        "fresh_bank": bundle.trainable_bank_name,
        "fresh_bank_parameter_names": [
            f"lora_banks.{bundle.trainable_bank_name}.{name}"
            for name in fresh_bank.installation.state_module.state_dict()
        ],
        "fresh_bank_parameter_count": fresh_count,
        "fresh_bank_target_modules": list(fresh_bank.installation.target_names),
        "total_parameter_count": total,
        "every_other_parameter_frozen": True,
    }


def frozen_inherited_state_sha256(bundle: V30Bundle) -> str:
    state: dict[str, torch.Tensor] = {}
    fresh_key = f"lora_banks.{bundle.trainable_bank_name}"
    for module_name, module in bundle.checkpoint_modules.items():
        if module_name == fresh_key:
            continue
        for name, value in module.state_dict().items():
            if module_name == "dense_sidecar_adapter" and name in _SIDECAR_TRAINABLE_NAMES:
                continue
            state[f"{module_name}.{name}"] = value
    return tensor_state_sha256(state)


def assert_frozen_inherited_state(bundle: V30Bundle, expected: str) -> None:
    observed = frozen_inherited_state_sha256(bundle)
    if observed != expected:
        raise RuntimeError(
            f"V30 inherited frozen state changed: expected={expected} observed={observed}"
        )


def verify_fresh_bank_update_zero(bundle: V30Bundle) -> dict[str, Any]:
    contract = v30_contract(bundle.config)
    bank = bundle.lora_installation.bank(bundle.trainable_bank_name).installation
    observed_hash = bank.state_sha256()
    if observed_hash != contract["fresh_bank_initial_state_sha256"]:
        raise ValueError("V30 fresh-bank deterministic hash mismatch")
    if bank.parameter_count != int(contract["fresh_bank_parameter_count"]):
        raise ValueError("V30 fresh-bank parameter count mismatch")
    targets: dict[str, bool] = {}
    for name, adapter in zip(bank.target_names, bank.adapters, strict=True):
        if not isinstance(adapter, LoRALinear):
            raise TypeError("V30 fresh target is not LoRALinear")
        if torch.count_nonzero(adapter.lora_b).item() != 0:
            raise ValueError(f"V30 fresh target is not zero-output: {name}")
        values = (
            torch.linspace(-0.25, 0.25, steps=2 * adapter.in_features, dtype=torch.float32)
            .reshape(2, adapter.in_features)
            .to(adapter.base.weight.device, adapter.base.weight.dtype)
        )
        was_training = adapter.training
        adapter.eval()
        with torch.inference_mode():
            targets[name] = bool(torch.equal(adapter.base(values), adapter(values)))
        adapter.train(was_training)
        if not targets[name]:
            raise RuntimeError(f"V30 fresh bank changed update-zero output: {name}")
    return {
        "fresh_bank_exact_zero_output": True,
        "fresh_bank_initial_state_sha256": observed_hash,
        "fresh_bank_parameter_count": bank.parameter_count,
        "target_outputs_bit_exact": targets,
    }


def load_v30_bundle(config: dict[str, Any], source: ApprovedV29Source) -> V30Bundle:
    contract = v30_contract(config)
    settings = v30_settings(config)
    source_config = load_config(str(contract["source_config"]))
    source_runtime = _read_metadata(source.checkpoint, RUNTIME_METADATA_FILENAME)
    source_training = _read_metadata(source.checkpoint, TRAINING_METADATA_FILENAME)
    validate_runtime_checkpoint_metadata(source_runtime)
    semantic_dim = int(source_runtime["semantic_dim"])

    language = load_local_language_model(
        str(config["language"]["model_id"]),
        str(config["language"]["revision"]),
        str(config["language"]["dtype"]),
        freeze=True,
        local_files_only=True,
        backend=str(config["language"].get("backend", "auto")),
        decoder_gradient_checkpointing=bool(
            config["training"].get("language_decoder_gradient_checkpointing", True)
        ),
    )
    boundary_mode = scene_boundary_mode_setting(config)
    if language.scene_boundary_contract(boundary_mode) != native_gemma4_image_contract_setting(
        config
    ):
        raise ValueError("Loaded Gemma boundary contract does not match V30")
    lora = install_lora_banks(language.model, lora_banks_settings(config))
    if lora is None:
        raise ValueError("V30 requires named LoRA banks")
    fresh = lora.bank(settings.trainable_bank)
    expected_targets = tuple(
        f"model.language_model.layers.{index}.self_attn.q_proj" for index in range(18, 22)
    )
    if (
        fresh.settings.adapter.rank != 8
        or float(fresh.settings.adapter.alpha) != 16.0
        or tuple(fresh.settings.adapter.target_modules) != expected_targets
    ):
        raise ValueError("V30 fresh bank must be rank-8/alpha-16 q_proj layers 18-21")

    scene_model = construct_scene_tokenizer(config, semantic_dim, language.hidden_size)
    dense_aligner = construct_dense_alignment(config, semantic_dim=semantic_dim)
    sidecar = construct_dense_sidecar_adapter(
        config,
        scene_dim=language.hidden_size,
        latent_count=int(config["scene_encoder"]["global_latents"]),
    )
    global_residual = construct_global_scene_residual(
        config,
        scene_dim=language.hidden_size,
        latent_count=int(config["scene_encoder"]["global_latents"]),
    )
    if dense_aligner is None or sidecar is None or global_residual is None:
        raise ValueError("V30 requires the complete V29 scene stack")
    signed_x = construct_signed_x_scene_residual(
        config,
        scene_dim=language.hidden_size,
        latent_count=int(config["scene_encoder"]["global_latents"]),
        content_dim=global_residual.width,
    )
    if signed_x is None:
        raise ValueError("V30 requires the frozen signed-X scene stack")
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
    checkpoint_modules = _checkpoint_modules(
        scene_model=scene_model,
        composer=composer,
        grounding=grounding,
        dense_aligner=dense_aligner,
        sidecar=sidecar,
        global_residual=global_residual,
        signed_x_residual=signed_x,
        lora=lora,
    )
    transition_mismatch = named_lora_freeze_and_extend_transition_mismatch(source_runtime, lora)
    if transition_mismatch is not None:
        raise ValueError(f"V29-to-V30 LoRA transition mismatch: {transition_mismatch}")
    source_modules = named_lora_extension_checkpoint_modules(checkpoint_modules, lora)
    loaded = load_adapter_checkpoint(
        source.checkpoint,
        source_modules,
        device="cpu",
        metadata_filename=RUNTIME_METADATA_FILENAME,
    )
    if loaded != source_runtime:
        raise RuntimeError("V29 source metadata changed during V30 load")
    validate_named_lora_extension_transition_state(source_runtime, lora)

    source_bank_counts = {
        bank.settings.name: bank.installation.parameter_count
        for bank in lora.banks
        if not bank.settings.trainable
    }
    validate_checkpoint_contract(
        source_runtime,
        source_config,
        semantic_dim=semantic_dim,
        language_hidden_dim=language.hidden_size,
        lora_parameter_count=sum(source_bank_counts.values()),
        lora_parameter_counts=source_bank_counts,
        dense_alignment_parameter_count=dense_aligner.parameter_count,
        dense_sidecar_adapter_parameter_count=sidecar.parameter_count,
    )
    validate_dense_alignment_state(
        dense_aligner,
        expected_parameter_count=int(source_runtime["dense_alignment_parameter_count"]),
        context="V30 V29-source load",
    )
    validate_dense_sidecar_adapter_state(
        sidecar,
        expected_parameter_count=int(source_runtime["dense_sidecar_adapter_parameter_count"]),
        expected_state_sha256=str(contract["source_sidecar_state_sha256"]),
        context="V30 V29-source load",
    )
    if sidecar.state_sha256() != source_runtime["dense_sidecar_adapter_state_sha256"]:
        raise ValueError("V30 source sidecar runtime hash mismatch")
    if (
        lora.bank("extension_v28_stage_b_query").installation.state_sha256()
        != contract["source_v28_bank_state_sha256"]
    ):
        raise ValueError("V30 frozen V28 decoder-bank source hash mismatch")
    if dense_aligner.state_sha256() != source_runtime["dense_alignment_state_sha256"]:
        raise ValueError("V30 frozen dense-aligner hash mismatch")
    if (
        module_collection_state_sha256({"global_scene_residual": global_residual})
        != (source_runtime["global_scene_residual_state_sha256"])
    ):
        raise ValueError("V30 frozen global residual hash mismatch")
    if (
        module_collection_state_sha256({"signed_x_scene_residual": signed_x})
        != (source_runtime["signed_x_scene_residual_state_sha256"])
    ):
        raise ValueError("V30 frozen signed-X residual hash mismatch")
    if (
        module_collection_state_sha256(
            {"scene_model": scene_model, "composer": composer, "grounding": grounding}
        )
        != source_runtime["frozen_scene_state_sha256"]
    ):
        raise ValueError("V30 frozen scene/composer/grounding hash mismatch")

    for module in checkpoint_modules.values():
        module.to(language.device)
    bundle = V30Bundle(
        config=config,
        source_config=source_config,
        source_runtime_metadata=source_runtime,
        source_training_metadata=source_training,
        source=source,
        language=language,
        scene_model=scene_model,
        dense_aligner=dense_aligner,
        dense_sidecar_adapter=sidecar,
        global_scene_residual=global_residual,
        signed_x_scene_residual=signed_x,
        composer=composer,
        grounding=grounding,
        lora_installation=lora,
        checkpoint_modules=checkpoint_modules,
        trainable_bank_name=settings.trainable_bank,
    )
    freeze_for_v30(bundle)
    assert_v30_trainable_surface(bundle)
    verify_fresh_bank_update_zero(bundle)
    return bundle


def _audit_scalar(audit: Mapping[str, object], name: str) -> float:
    value = audit.get(name)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Scene audit {name} is not scalar")
        value = value.detach().float().cpu().item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Scene audit {name} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Scene audit {name} is nonfinite")
    return result


def _source_prefix_hashes(metadata: Mapping[str, Any]) -> dict[str, str]:
    stage = metadata.get("v28_stage_b")
    cache = stage.get("scene_cache") if isinstance(stage, Mapping) else None
    hashes = cache.get("prefix_sha256_by_scene") if isinstance(cache, Mapping) else None
    if not isinstance(hashes, Mapping):
        raise TypeError("V29 source lacks cached full-scene prefix hashes")
    parsed = {str(key): str(value) for key, value in hashes.items()}
    if any(_SHA256.fullmatch(value) is None for value in parsed.values()):
        raise ValueError("V29 source contains an invalid scene-prefix hash")
    return parsed


def _source_prefix_provenance(
    *,
    scene_id: str,
    source_prefix: str,
    expected_prefixes: Mapping[str, str],
    allowed_unpinned: set[str],
    repeated_source_prefix: str | None = None,
) -> str:
    """Validate a historical hash or a bit-exact newly derived source hash."""

    expected = expected_prefixes.get(scene_id)
    if expected is not None:
        if source_prefix != expected:
            raise RuntimeError(
                f"V30 update-zero source-prefix mismatch for {scene_id}: "
                f"expected={expected} observed={source_prefix}"
            )
        return "historically_pinned"
    if scene_id not in allowed_unpinned:
        raise RuntimeError(
            f"V30 update-zero source-prefix mismatch for {scene_id}: "
            f"expected=None observed={source_prefix}"
        )
    if repeated_source_prefix != source_prefix:
        raise RuntimeError(
            f"Unpinned source prefix is nondeterministic for {scene_id}: "
            f"first={source_prefix} repeated={repeated_source_prefix}"
        )
    return "deterministically_derived"


def cache_pre_sidecar_scenes(
    bundle: V30Bundle,
    scene_ids: Sequence[str],
    *,
    allow_unpinned_source_scene_ids: Sequence[str] = (),
) -> tuple[dict[str, CachedPreSidecarScene], dict[str, Any]]:
    """Cache the complete frozen stack before the trainable sidecar.

    This is the critical V30 cache boundary: all voxels influence both cached
    tensors, but sidecar gradients remain open for every QA forward.
    """

    expected_prefixes = _source_prefix_hashes(bundle.source_training_metadata)
    requested_scene_ids = set(scene_ids)
    allowed_unpinned = set(allow_unpinned_source_scene_ids)
    if allowed_unpinned - requested_scene_ids:
        raise ValueError("Allowed unpinned source scenes must be present in scene_ids")
    if allowed_unpinned & set(expected_prefixes):
        raise ValueError("A historically pinned source scene cannot be marked unpinned")
    caches: dict[str, CachedPreSidecarScene] = {}
    loaded_files: list[str] = []
    pinned_scene_ids: list[str] = []
    derived_scene_ids: list[str] = []
    started = time.perf_counter()
    model_dtype = next(bundle.language.model.parameters()).dtype
    for scene_id in sorted(set(scene_ids)):
        map_path = (artifact_root(bundle.config, "maps") / scene_id / "voxel_map.npz").resolve()
        if "oracle" in {part.casefold() for part in map_path.parts}:
            raise RuntimeError("V30 refuses oracle environmental input")
        data = load_map_tensors(
            map_path,
            bundle.config["scene"]["room_size_m"],
            bundle.language.device,
            input_voxel_size_m=bundle.config["scene_encoder"].get("input_voxel_size_m"),
        )
        with torch.no_grad():
            output = map_forward(
                bundle.scene_model,
                data,
                bundle.global_scene_residual,
                bundle.signed_x_scene_residual,
                bundle.dense_aligner,
                None,
            )
            if output.aligned_sidecar_tokens is None:
                raise RuntimeError(f"V30 pre-sidecar cache lacks aligned tokens: {scene_id}")
            processed = int(_audit_scalar(output.audit, "processed_voxels"))
            sidecar_processed = int(_audit_scalar(output.audit, "aligned_sidecar_processed_voxels"))
            minimum = _audit_scalar(output.audit, "aligned_sidecar_min_voxel_contribution")
            if processed != data.voxel_count or sidecar_processed != data.voxel_count:
                raise RuntimeError(f"V30 pre-sidecar cache omitted voxels: {scene_id}")
            if minimum <= 0:
                raise RuntimeError(f"V30 sidecar contribution is not positive: {scene_id}")
            base = output.scene_tokens.detach()
            aligned = output.aligned_sidecar_tokens.detach()
            source_tokens = bundle.dense_sidecar_adapter(base, aligned)
            source_prefix = prefix_sha256(
                bundle.composer.scene_prefix(source_tokens.to(model_dtype))
            )
            expected = expected_prefixes.get(scene_id)
            repeated_prefix = None
            if expected is None:
                # The approved V29 report predates an intentionally expanded
                # training scene, so it cannot contain a historical hash for
                # that scene. Re-run the frozen source adapter and require a
                # bit-identical prefix. Validation and all historical scenes
                # remain bound to their previously recorded hashes above.
                repeated_tokens = bundle.dense_sidecar_adapter(base, aligned)
                repeated_prefix = prefix_sha256(
                    bundle.composer.scene_prefix(repeated_tokens.to(model_dtype))
                )
            provenance = _source_prefix_provenance(
                scene_id=scene_id,
                source_prefix=source_prefix,
                expected_prefixes=expected_prefixes,
                allowed_unpinned=allowed_unpinned,
                repeated_source_prefix=repeated_prefix,
            )
            if provenance == "deterministically_derived":
                derived_scene_ids.append(scene_id)
            else:
                pinned_scene_ids.append(scene_id)
            caches[scene_id] = CachedPreSidecarScene(
                scene_id=scene_id,
                base_scene_tokens=base,
                aligned_sidecar_tokens=aligned,
                source_prefix_sha256=source_prefix,
                voxel_count=data.voxel_count,
                processed_voxels=processed,
                minimum_voxel_contribution=minimum,
            )
        loaded_files.append(str(map_path))
        del output, data
        if bundle.language.device.type == "mps":
            torch.mps.empty_cache()
    return caches, {
        "schema_version": 1,
        "cache_boundary": "complete_frozen_pre_sidecar_scene_stack",
        "scene_count": len(caches),
        "cache_build_seconds": time.perf_counter() - started,
        "question_inputs_to_scene_cache": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "all_voxels_covered": True,
        "oracle_environment_files_loaded": False,
        "loaded_environment_files": loaded_files,
        "source_prefix_sha256_by_scene": {
            scene_id: cache.source_prefix_sha256 for scene_id, cache in sorted(caches.items())
        },
        "historically_pinned_source_scene_ids": sorted(pinned_scene_ids),
        "deterministically_derived_source_scene_ids": sorted(derived_scene_ids),
        "derived_source_prefixes_recomputed_bit_exact": True,
        "exact_source_scene_prefixes": True,
    }


def adapted_scene_tokens(cache: CachedPreSidecarScene, bundle: V30Bundle) -> torch.Tensor:
    return bundle.dense_sidecar_adapter(cache.base_scene_tokens, cache.aligned_sidecar_tokens)


def _compose_answer_batch(
    *,
    scene_tokens: torch.Tensor,
    question: str,
    answer: str,
    bundle: V30Bundle,
):
    prompt_ids = prompt_token_ids(
        bundle.language.tokenizer,
        str(bundle.config["language"]["system_prompt"]),
        question,
        bundle.language.device,
    )
    answer_ids = tokenize_answer(bundle.language.tokenizer, answer, bundle.language.device)
    return bundle.composer.compose(
        scene_tokens.to(next(bundle.language.model.parameters()).dtype),
        prompt_ids,
        bundle.language.model.get_input_embeddings(),
        answer_ids,
        prefix_backend=getattr(bundle.language, "prefix_backend", None),
    )


def cached_broad_answer_nll(
    *,
    cache: CachedPreSidecarScene,
    records: Sequence[QARecord],
    bundle: V30Bundle,
) -> torch.Tensor:
    if not records or any(record.scene_id != cache.scene_id for record in records):
        raise ValueError("V30 broad batch must contain one nonempty cached scene")
    tokens = adapted_scene_tokens(cache, bundle)
    batches = [
        _compose_answer_batch(
            scene_tokens=tokens,
            question=record.question,
            answer=record.answer,
            bundle=bundle,
        )
        for record in records
    ]
    batch = stack_prefix_batches(
        batches,
        bundle.language.device,
        prefix_backend=getattr(bundle.language, "prefix_backend", None),
    )
    output = forward_prefix_batch(bundle.language, batch)
    if batch.labels is None:
        raise RuntimeError("V30 broad batch lacks answer labels")
    nll = token_normalized_nll(output.logits, batch.labels).mean()
    if nll.ndim != 0 or not torch.isfinite(nll):
        raise RuntimeError("V30 broad answer NLL is invalid")
    return nll


def paired_canonical_answer_objective(
    *,
    units: Sequence[CounterfactualPairUnit],
    caches: Mapping[str, CachedPreSidecarScene],
    bundle: V30Bundle,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Compare correct and swapped canonical answers on both scene prefixes."""

    if not units:
        raise ValueError("V30 paired objective requires complete units")
    scene_tokens = {
        scene_id: adapted_scene_tokens(caches[scene_id], bundle)
        for scene_id in sorted({scene for unit in units for scene in unit.scene_ids})
    }
    correct_batches = []
    swapped_batches = []
    correct_masks: list[torch.Tensor] = []
    swapped_masks: list[torch.Tensor] = []
    for unit in units:
        first, second = unit.records
        first_ids = tokenize_answer(bundle.language.tokenizer, first.answer, bundle.language.device)
        second_ids = tokenize_answer(
            bundle.language.tokenizer, second.answer, bundle.language.device
        )
        first_mask, second_mask = differing_answer_token_masks(first_ids, second_ids)
        correct_masks.extend((first_mask, second_mask))
        swapped_masks.extend((second_mask, first_mask))
        for record, answer in ((first, first.answer), (second, second.answer)):
            correct_batches.append(
                _compose_answer_batch(
                    scene_tokens=scene_tokens[record.scene_id],
                    question=record.question,
                    answer=answer,
                    bundle=bundle,
                )
            )
        for record, answer in ((first, second.answer), (second, first.answer)):
            swapped_batches.append(
                _compose_answer_batch(
                    scene_tokens=scene_tokens[record.scene_id],
                    question=record.question,
                    answer=answer,
                    bundle=bundle,
                )
            )

    correct = stack_prefix_batches(
        correct_batches,
        bundle.language.device,
        prefix_backend=getattr(bundle.language, "prefix_backend", None),
    )
    correct_output = forward_prefix_batch(bundle.language, correct)
    if correct.labels is None:
        raise RuntimeError("V30 correct pair batch lacks labels")
    correct_answer_nll = token_normalized_nll(correct_output.logits, correct.labels).reshape(
        len(units), 2
    )
    correct_rank_labels = correct.labels.clone()
    for row, mask in enumerate(correct_masks):
        restrict_labels_to_answer_mask(correct_rank_labels, row, mask)
    correct_rank_nll = token_normalized_nll(correct_output.logits, correct_rank_labels).reshape(
        len(units), 2
    )
    del correct_output, correct

    swapped = stack_prefix_batches(
        swapped_batches,
        bundle.language.device,
        prefix_backend=getattr(bundle.language, "prefix_backend", None),
    )
    swapped_output = forward_prefix_batch(bundle.language, swapped)
    if swapped.labels is None:
        raise RuntimeError("V30 swapped pair batch lacks labels")
    swapped_rank_labels = swapped.labels.clone()
    for row, mask in enumerate(swapped_masks):
        restrict_labels_to_answer_mask(swapped_rank_labels, row, mask)
    swapped_rank_nll = token_normalized_nll(swapped_output.logits, swapped_rank_labels).reshape(
        len(units), 2
    )
    ranking_loss, margins = pair_ranking_hinge(correct_rank_nll, swapped_rank_nll, margin=margin)
    language_nll = correct_answer_nll.mean()
    if not torch.isfinite(language_nll) or not torch.isfinite(ranking_loss):
        raise RuntimeError("V30 paired objective is nonfinite")
    return (
        language_nll,
        ranking_loss,
        {
            "margins": margins,
            "correct_answer_nll": correct_answer_nll,
            "correct_ranking_nll": correct_rank_nll,
            "swapped_ranking_nll": swapped_rank_nll,
            "side_accuracy": margins.gt(0).float().mean(),
            "unit_accuracy": margins.gt(0).all(dim=1).float().mean(),
        },
    )


def validation_answer_nll(
    *,
    records: Sequence[QARecord],
    caches: Mapping[str, CachedPreSidecarScene],
    bundle: V30Bundle,
    batch_size: int,
) -> dict[str, Any]:
    bundle.dense_sidecar_adapter.eval()
    bundle.lora_installation.eval()
    weighted = 0.0
    questions = 0
    try:
        with torch.no_grad():
            for scene_id, scene_records in records_by_scene(records).items():
                for offset in range(0, len(scene_records), batch_size):
                    batch = scene_records[offset : offset + batch_size]
                    loss = cached_broad_answer_nll(
                        cache=caches[scene_id], records=batch, bundle=bundle
                    )
                    weighted += float(loss.detach().cpu()) * len(batch)
                    questions += len(batch)
    finally:
        bundle.dense_sidecar_adapter.train()
        bundle.lora_installation.train()
    if questions == 0:
        raise ValueError("V30 validation records are empty")
    return {
        "answer_token_nll": weighted / questions,
        "question_count": questions,
        "cached_complete_pre_sidecar_scenes_reused": True,
        "environmental_oracle_files_loaded": False,
        "question_dependent_scene_processing": False,
    }


def validation_pair_metrics(
    *,
    units: Sequence[CounterfactualPairUnit],
    caches: Mapping[str, CachedPreSidecarScene],
    bundle: V30Bundle,
    margin: float,
) -> dict[str, Any]:
    bundle.dense_sidecar_adapter.eval()
    bundle.lora_installation.eval()
    rows: list[dict[str, Any]] = []
    values: list[list[float]] = []
    language_values: list[float] = []
    hinge_values: list[float] = []
    try:
        with torch.no_grad():
            for unit in units:
                language_nll, ranking_loss, diagnostics = paired_canonical_answer_objective(
                    units=[unit], caches=caches, bundle=bundle, margin=margin
                )
                margins = diagnostics["margins"].detach().float().cpu().reshape(-1)
                pair_values = [float(margins[0]), float(margins[1])]
                values.append(pair_values)
                language_values.append(float(language_nll.detach().cpu()))
                hinge_values.append(float(ranking_loss.detach().cpu()))
                rows.append(
                    {
                        "pair_id": unit.pair_id,
                        "question_key": unit.question_key,
                        "scene_ids": list(unit.scene_ids),
                        "margins": pair_values,
                    }
                )
    finally:
        bundle.dense_sidecar_adapter.train()
        bundle.lora_installation.train()
    tensor = torch.tensor(values, dtype=torch.float32)
    passed = tensor.gt(0).all(dim=1)
    return {
        "unit_count": len(units),
        "side_count": int(tensor.numel()),
        "passed_units": int(passed.sum()),
        "side_accuracy": float(tensor.gt(0).float().mean()),
        "unit_accuracy": float(passed.float().mean()),
        "mean_margin": float(tensor.mean()),
        "minimum_margin": float(tensor.min()),
        "mean_pair_language_nll": sum(language_values) / len(language_values),
        "mean_pair_margin_hinge": sum(hinge_values) / len(hinge_values),
        "margins_by_unit": rows,
        "canonical_correct_vs_swapped_both_scene_prefixes": True,
        "free_generation_evaluated": False,
    }


def select_balanced_broad_records(
    records: Sequence[QARecord],
    *,
    count: int,
    seed: int,
    exclude_expected_change: bool,
) -> list[QARecord]:
    candidates = [
        record
        for record in records
        if not exclude_expected_change or record.counterfactual_expected_change is not True
    ]
    if count > len(candidates):
        raise ValueError("V30 broad cycle requests more unique records than available")
    by_type: defaultdict[str, list[QARecord]] = defaultdict(list)
    for record in candidates:
        by_type[record.answer_type].append(record)
    if not by_type:
        raise ValueError("V30 broad candidate pool is empty")
    for answer_type, values in by_type.items():
        values.sort(
            key=lambda record: (
                hashlib.sha256(
                    f"{seed}:{answer_type}:{record.scene_id}:{record.question_id}".encode()
                ).digest(),
                record.scene_id,
                record.question_id,
            )
        )
    selected: list[QARecord] = []
    offsets = {answer_type: 0 for answer_type in by_type}
    ordered_types = sorted(by_type)
    while len(selected) < count:
        made_progress = False
        for answer_type in ordered_types:
            offset = offsets[answer_type]
            values = by_type[answer_type]
            if offset >= len(values):
                continue
            selected.append(values[offset])
            offsets[answer_type] += 1
            made_progress = True
            if len(selected) == count:
                break
        if not made_progress:
            raise RuntimeError("V30 balanced broad selector exhausted early")
    return selected


def build_v30_cycle(
    records: Sequence[QARecord],
    pair_units: Sequence[CounterfactualPairUnit],
    *,
    settings: V30Settings,
    seed: int,
) -> tuple[list[tuple[str, list[QARecord]]], list[list[CounterfactualPairUnit]], dict[str, Any]]:
    broad = select_balanced_broad_records(
        records,
        count=settings.broad_questions_per_cycle,
        seed=seed,
        exclude_expected_change=settings.broad_exclude_expected_change,
    )
    broad_batches: list[tuple[str, list[QARecord]]] = []
    rng = random.Random(seed)
    for scene_id, scene_records in records_by_scene(broad).items():
        shuffled = list(scene_records)
        rng.shuffle(shuffled)
        broad_batches.extend(
            (scene_id, shuffled[offset : offset + settings.broad_batch_size])
            for offset in range(0, len(shuffled), settings.broad_batch_size)
        )
    rng.shuffle(broad_batches)

    repeated = [unit for _ in range(settings.pair_repeats_per_cycle) for unit in pair_units]
    rng.shuffle(repeated)
    pair_batches = [
        repeated[offset : offset + settings.pair_units_per_batch]
        for offset in range(0, len(repeated), settings.pair_units_per_batch)
    ]
    appearances = Counter((unit.pair_id, unit.question_key) for unit in repeated)
    expected_keys = {(unit.pair_id, unit.question_key) for unit in pair_units}
    if set(appearances) != expected_keys or any(
        value != settings.pair_repeats_per_cycle for value in appearances.values()
    ):
        raise RuntimeError("V30 cycle did not oversample every pair unit exactly")
    if any(not batch for batch in pair_batches):
        raise RuntimeError("V30 cycle contains an empty pair batch")
    audit = {
        "seed": seed,
        "broad_question_count": len(broad),
        "broad_answer_type_counts": dict(sorted(Counter(r.answer_type for r in broad).items())),
        "broad_expected_change_excluded": settings.broad_exclude_expected_change,
        "pair_unit_count": len(pair_units),
        "pair_repeats_per_cycle": settings.pair_repeats_per_cycle,
        "pair_side_presentations": len(repeated) * 2,
        "pair_units_atomic": True,
        "every_pair_unit_present_each_cycle": True,
        "pair_appearance_sha256": hashlib.sha256(
            json.dumps(
                sorted((list(key), value) for key, value in appearances.items()),
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }
    return broad_batches, pair_batches, audit


def _optimizer(bundle: V30Bundle, settings: V30Settings) -> torch.optim.AdamW:
    sidecar = _sidecar_trainable_parameters(bundle.dense_sidecar_adapter)
    decoder = bundle.lora_installation.bank(bundle.trainable_bank_name).installation.parameters()
    optimizer = torch.optim.AdamW(
        [
            {
                "name": "dense_sidecar_adapter.output_surfaces",
                "params": sidecar,
                "lr": settings.sidecar_learning_rate,
                "weight_decay": settings.weight_decay,
            },
            {
                "name": bundle.trainable_bank_name,
                "params": decoder,
                "lr": settings.decoder_learning_rate,
                "weight_decay": settings.weight_decay,
            },
        ]
    )
    assert_v30_trainable_surface(bundle, optimizer)
    return optimizer


def _metadata(
    *,
    bundle: V30Bundle,
    settings: V30Settings,
    cache_audit: Mapping[str, Any],
    qa_audit: Mapping[str, Any],
    frozen_hash: str,
    update_zero: Mapping[str, Any],
    train_records: Sequence[QARecord],
    validation_records: Sequence[QARecord],
    pair_units: Sequence[CounterfactualPairUnit],
    history: Sequence[Mapping[str, Any]],
    optimizer_step: int,
    best_update: int,
    best_validation: float,
    trainable_surface: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(bundle.source_runtime_metadata)
    lora_settings = lora_banks_settings(bundle.config)
    lora_optimizer = lora_banks_optimizer_settings(bundle.config, lora_settings)
    result.update(
        {
            "schema_version": 3,
            "config_hash": config_hash(bundle.config),
            "epoch": optimizer_step,
            "optimizer_step": optimizer_step,
            "history": list(history),
            "best_epoch": best_update,
            "best_monitor_loss": best_validation,
            "monitor_name": "validation_answer_token_nll",
            "lora": lora_banks_checkpoint_contract(
                lora_settings,
                lora_optimizer,
                bundle.lora_installation.parameter_counts,
            ),
            **bundle.lora_installation.checkpoint_metadata(),
            "dense_sidecar_adapter_state_sha256": (bundle.dense_sidecar_adapter.state_sha256()),
            "freeze_scene_adapter": True,
            "question_dependent_scene_processing": False,
            "v30_joint_pair": {
                "schema_version": 1,
                "source_v29_checkpoint": str(bundle.source.checkpoint),
                "source_v29_adapter_sha256": _file_sha256(
                    bundle.source.checkpoint / "adapter.safetensors"
                ),
                "source_v29_runtime_metadata_sha256": _file_sha256(
                    bundle.source.checkpoint / RUNTIME_METADATA_FILENAME
                ),
                "source_v29_selection_report": str(bundle.source.selection_report),
                "source_v29_selection_report_sha256": bundle.source.selection_sha256,
                "source_v29_selected_update": bundle.source.selected_update,
                "source_v29_selected_arm": bundle.source.selected_arm,
                "objective": "broad_answer_nll_plus_atomic_correct_vs_swapped_pair_margin",
                "settings": settings.__dict__,
                "trainable_surface": dict(trainable_surface),
                "fresh_bank": bundle.trainable_bank_name,
                "fresh_bank_initial_state_sha256": v30_contract(bundle.config)[
                    "fresh_bank_initial_state_sha256"
                ],
                "fresh_bank_parameter_count": v30_contract(bundle.config)[
                    "fresh_bank_parameter_count"
                ],
                "update_zero_equivalence": dict(update_zero),
                "frozen_inherited_state_sha256": frozen_hash,
                "scene_cache": dict(cache_audit),
                "qa_dataset": dict(qa_audit),
                "train_scene_ids": sorted({item.scene_id for item in train_records}),
                "validation_scene_ids": sorted({item.scene_id for item in validation_records}),
                "train_question_count": len(train_records),
                "validation_question_count": len(validation_records),
                "train_answer_types": sorted({item.answer_type for item in train_records}),
                "pair_unit_count": len(pair_units),
                "qa_supervision_serialized_to_runtime": False,
                "oracle_environment_files_loaded": False,
                "question_dependent_scene_processing": False,
                "question_dependent_retrieval": False,
                "final_test_scene_ids_loaded": [],
                "all_inherited_lora_banks_frozen": True,
                "all_sidecar_hidden_parameters_frozen": True,
                "composer_grounding_frozen": True,
                "development_validation_model_selection_only": True,
            },
        }
    )
    return result


def _save(
    path: Path,
    *,
    bundle: V30Bundle,
    metadata: dict[str, Any],
    optimizer: torch.optim.Optimizer | None,
) -> None:
    save_adapter_checkpoint(path, bundle.checkpoint_modules, metadata)
    if optimizer is not None:
        save_optimizer_checkpoint(path, optimizer)


def run_v30(
    *,
    config: dict[str, Any],
    output: Path,
    max_optimizer_steps_override: int | None = None,
    allow_unpinned_source_scene_ids: Sequence[str] = (),
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty V30 output: {output}")
    settings = v30_settings(config)
    if not settings.enabled:
        raise ValueError("V30 joint training is disabled")
    if max_optimizer_steps_override is not None:
        settings = V30Settings(
            **{
                **settings.__dict__,
                "max_optimizer_steps": _positive_int(
                    "max_optimizer_steps", max_optimizer_steps_override
                ),
            }
        )
    source = require_approved_v29_source(config)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    train_records, validation_records, qa_audit = load_stage_b_qa_records(
        config, max_train_questions=None, max_validation_questions=None
    )
    if {item.scene_id for item in train_records} & {item.scene_id for item in validation_records}:
        raise ValueError("V30 train/validation scenes overlap")
    if len({item.answer_type for item in train_records}) < settings.minimum_answer_types:
        raise ValueError("V30 broad training lacks required answer-type coverage")
    train_pairs = build_exact_question_pair_units(train_records)
    validation_pairs = build_exact_question_pair_units(validation_records)
    contract = v30_contract(config)
    if len(validation_pairs) != int(contract["validation_pair_unit_count"]):
        raise ValueError("V30 validation pair-unit count differs from its locked contract")
    if not train_pairs:
        raise ValueError("V30 training contains no answer-changing pair units")

    bundle = load_v30_bundle(config, source)
    scene_ids = sorted(
        {item.scene_id for item in train_records} | {item.scene_id for item in validation_records}
    )
    caches, cache_audit = cache_pre_sidecar_scenes(
        bundle,
        scene_ids,
        allow_unpinned_source_scene_ids=allow_unpinned_source_scene_ids,
    )
    frozen_hash = frozen_inherited_state_sha256(bundle)
    trainable_surface = assert_v30_trainable_surface(bundle)
    fresh_zero = verify_fresh_bank_update_zero(bundle)
    baseline = validation_answer_nll(
        records=validation_records,
        caches=caches,
        bundle=bundle,
        batch_size=settings.broad_batch_size,
    )
    observed_nll = float(baseline["answer_token_nll"])
    source_nll = _source_validation_nll(bundle.source_training_metadata)
    tolerance = float(contract["update_zero_validation_nll_absolute_tolerance"])
    if abs(observed_nll - source_nll) > tolerance:
        raise RuntimeError(
            "V30 update zero differs from approved V29 validation NLL: "
            f"source={source_nll} observed={observed_nll} tolerance={tolerance}"
        )
    baseline_pairs = validation_pair_metrics(
        units=validation_pairs,
        caches=caches,
        bundle=bundle,
        margin=settings.pair_margin,
    )
    update_zero = {
        "approved_v29_source": True,
        **fresh_zero,
        "exact_source_scene_prefixes": True,
        "exact_source_validation_nll": True,
        "source_validation_answer_token_nll": source_nll,
        "observed_validation_answer_token_nll": observed_nll,
        "validation_nll_absolute_tolerance": tolerance,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "oracle_environment_files_loaded": False,
    }

    optimizer = _optimizer(bundle, settings)
    history: list[dict[str, Any]] = [
        {
            "optimizer_update": 0,
            "validation_answer_token_nll": observed_nll,
            "validation_pair_metrics": baseline_pairs,
            "update_0_equivalence_verified": True,
        }
    ]
    best_update = 0
    best_validation = observed_nll
    output.mkdir(parents=True, exist_ok=True)
    initial_metadata = _metadata(
        bundle=bundle,
        settings=settings,
        cache_audit=cache_audit,
        qa_audit=qa_audit,
        frozen_hash=frozen_hash,
        update_zero=update_zero,
        train_records=train_records,
        validation_records=validation_records,
        pair_units=train_pairs,
        history=history,
        optimizer_step=0,
        best_update=0,
        best_validation=best_validation,
        trainable_surface=trainable_surface,
    )
    _save(output / "update_000", bundle=bundle, metadata=initial_metadata, optimizer=None)
    _save(output / "best", bundle=bundle, metadata=initial_metadata, optimizer=None)

    all_trainable = freeze_for_v30(bundle)
    for update in range(1, settings.max_optimizer_steps + 1):
        broad_batches, pair_batches, cycle_audit = build_v30_cycle(
            train_records,
            train_pairs,
            settings=settings,
            seed=seed + update,
        )
        bundle.dense_sidecar_adapter.train()
        bundle.lora_installation.train()
        optimizer.zero_grad(set_to_none=True)
        broad_total = 0.0
        for scene_id, records in broad_batches:
            loss = cached_broad_answer_nll(cache=caches[scene_id], records=records, bundle=bundle)
            (settings.broad_nll_weight * loss / len(broad_batches)).backward()
            broad_total += float(loss.detach().cpu())
        pair_language_total = 0.0
        pair_margin_total = 0.0
        for units in pair_batches:
            language_nll, ranking_loss, _diagnostics = paired_canonical_answer_objective(
                units=units,
                caches=caches,
                bundle=bundle,
                margin=settings.pair_margin,
            )
            pair_objective = (
                settings.pair_language_nll_weight * language_nll
                + settings.pair_margin_weight * ranking_loss
            ) / len(pair_batches)
            pair_objective.backward()
            pair_language_total += float(language_nll.detach().cpu())
            pair_margin_total += float(ranking_loss.detach().cpu())
        assert_v30_trainable_surface(bundle, optimizer)
        missing_gradients = [
            index for index, parameter in enumerate(all_trainable) if parameter.grad is None
        ]
        if missing_gradients:
            raise RuntimeError(f"V30 trainable tensors lack gradients: {missing_gradients}")
        if any(not torch.isfinite(parameter.grad).all() for parameter in all_trainable):
            raise RuntimeError("V30 trainable gradient is nonfinite")
        gradient_norm = torch.nn.utils.clip_grad_norm_(all_trainable, settings.gradient_clip_norm)
        optimizer.step()
        assert_frozen_inherited_state(bundle, frozen_hash)
        bundle.lora_installation.validate_state()
        validate_dense_sidecar_adapter_state(
            bundle.dense_sidecar_adapter,
            expected_parameter_count=int(
                bundle.source_runtime_metadata["dense_sidecar_adapter_parameter_count"]
            ),
            context="V30 post-update sidecar",
        )
        validation = (
            validation_answer_nll(
                records=validation_records,
                caches=caches,
                bundle=bundle,
                batch_size=settings.broad_batch_size,
            )
            if update % settings.evaluation_interval_steps == 0
            or update == settings.max_optimizer_steps
            else None
        )
        pair_validation = (
            validation_pair_metrics(
                units=validation_pairs,
                caches=caches,
                bundle=bundle,
                margin=settings.pair_margin,
            )
            if validation is not None
            else None
        )
        validation_value = None if validation is None else float(validation["answer_token_nll"])
        improved = validation_value is not None and validation_value < best_validation
        if improved:
            best_update = update
            best_validation = float(validation_value)
        history.append(
            {
                "optimizer_update": update,
                "cycle": cycle_audit,
                "train_broad_answer_token_nll": broad_total / len(broad_batches),
                "train_pair_answer_token_nll": pair_language_total / len(pair_batches),
                "train_pair_margin_hinge": pair_margin_total / len(pair_batches),
                "validation_answer_token_nll": validation_value,
                "validation_pair_metrics": pair_validation,
                "preclip_gradient_norm": float(gradient_norm.detach().cpu()),
                "fresh_bank_state_sha256": bundle.lora_installation.bank(
                    settings.trainable_bank
                ).installation.state_sha256(),
                "sidecar_state_sha256": bundle.dense_sidecar_adapter.state_sha256(),
                "frozen_inherited_state_sha256": frozen_hash,
            }
        )
        metadata = _metadata(
            bundle=bundle,
            settings=settings,
            cache_audit=cache_audit,
            qa_audit=qa_audit,
            frozen_hash=frozen_hash,
            update_zero=update_zero,
            train_records=train_records,
            validation_records=validation_records,
            pair_units=train_pairs,
            history=history,
            optimizer_step=update,
            best_update=best_update,
            best_validation=best_validation,
            trainable_surface=trainable_surface,
        )
        _save(
            output / f"update_{update:03d}", bundle=bundle, metadata=metadata, optimizer=optimizer
        )
        if improved:
            _save(output / "best", bundle=bundle, metadata=metadata, optimizer=None)
        print(
            json.dumps(
                {
                    "phase": "v30_joint_pair_update",
                    "optimizer_update": update,
                    "validation_answer_token_nll": validation_value,
                    "validation_pair_passed_units": (
                        None if pair_validation is None else pair_validation["passed_units"]
                    ),
                    "best_update": best_update,
                }
            ),
            flush=True,
        )
    assert_frozen_inherited_state(bundle, frozen_hash)
    return {
        "schema_version": 1,
        "artifact": "v30_diverse20_joint_pair_training",
        "output": str(output),
        "best_checkpoint": str(output / "best"),
        "best_update": best_update,
        "baseline_validation_answer_token_nll": observed_nll,
        "best_validation_answer_token_nll": best_validation,
        "optimizer_updates": settings.max_optimizer_steps,
        "trainable_surface": trainable_surface,
        "frozen_inherited_state_sha256": frozen_hash,
        "source_v29_selection_sha256": source.selection_sha256,
        "final_test_scene_ids_loaded": [],
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "oracle_environment_files_loaded": False,
        "causal_selection_required_before_promotion": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-optimizer-steps", type=int)
    args = parser.parse_args()
    report = run_v30(
        config=load_config(args.config),
        output=_resolve(args.output),
        max_optimizer_steps_override=args.max_optimizer_steps,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ApprovedV29Source",
    "CachedPreSidecarScene",
    "V30Bundle",
    "V30Settings",
    "adapted_scene_tokens",
    "assert_v30_trainable_surface",
    "build_v30_cycle",
    "cache_pre_sidecar_scenes",
    "frozen_inherited_state_sha256",
    "load_v30_bundle",
    "paired_canonical_answer_objective",
    "require_approved_v29_source",
    "run_v30",
    "select_balanced_broad_records",
    "v30_contract",
    "v30_settings",
    "validation_pair_metrics",
    "verify_fresh_bank_update_zero",
]
