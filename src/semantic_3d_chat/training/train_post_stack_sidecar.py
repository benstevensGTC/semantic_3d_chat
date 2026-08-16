"""V28 Stage-A training for the zero-output post-stack dense sidecar.

This entry point deliberately has a narrower optimization surface than the
general adapter trainer.  It loads a complete, runtime-valid V28 update-0
candidate, freezes Gemma and every inherited adapter tensor, caches one
question-independent base/sidecar scene representation per scene, and trains
only ``DenseSidecarAdapter.output_projection`` and ``channel_gain`` on broad
answer-token negative log likelihood.

No oracle scene specification is imported or opened.  QA answer text is
training/evaluation supervision only; the environmental input to Gemma is the
cached continuous scene tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
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
    install_lora_banks,
    lora_banks_settings,
    tensor_state_sha256,
    validate_lora_banks_checkpoint_state,
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
    dense_alignment_settings,
    validate_dense_alignment_state,
)
from semantic_3d_chat.scene_encoder.dense_sidecar_adapter import (
    DenseSidecarAdapter,
    construct_dense_sidecar_adapter,
    dense_sidecar_adapter_settings,
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
    load_adapter_checkpoint,
    module_collection_state_sha256,
    save_adapter_checkpoint,
    save_optimizer_checkpoint,
    validate_runtime_checkpoint_metadata,
)
from semantic_3d_chat.training.losses import QuestionGroundingHead
from semantic_3d_chat.training.train_adapter import (
    forward_prefix_batch,
    load_qa_split_dataset,
    map_forward,
    tokenize_answer,
)

DEFAULT_CONFIG = Path("configs/experiments/gemma4_color_mirror_post_stack_sidecar_v28.yaml")
DEFAULT_CANDIDATE = Path(
    "data_gemma4/checkpoints/gemma4_v28_post_stack_sidecar/candidate_zero"
)
DEFAULT_OUTPUT = Path("data_gemma4/checkpoints/gemma4_v28_post_stack_sidecar/stage_a")


@dataclass(frozen=True)
class StageASettings:
    enabled: bool
    max_optimizer_steps: int
    evaluation_interval_steps: int
    batch_size: int
    gradient_accumulation: int
    output_projection_learning_rate: float
    channel_gain_learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    minimum_answer_types: int
    trainable_routes: tuple[str, str]


@dataclass(frozen=True)
class CachedSceneTokens:
    """The only environmental payload consumed by Stage-A question batches."""

    scene_id: str
    base_scene_tokens: torch.Tensor
    aligned_sidecar_tokens: torch.Tensor
    base_prefix_sha256: str
    voxel_count: int
    processed_voxels: int
    minimum_voxel_contribution: float


@dataclass
class StageABundle:
    config: dict[str, Any]
    candidate_metadata: dict[str, Any]
    language: Any
    scene_model: SceneTokenizer
    dense_aligner: DenseAlignmentResidual
    dense_sidecar_adapter: DenseSidecarAdapter
    global_scene_residual: GlobalSceneResidual
    signed_x_scene_residual: SignedXSceneResidual
    composer: ContinuousPrefixComposer
    grounding: QuestionGroundingHead
    lora_installation: LoRABankCollection | None
    checkpoint_modules: dict[str, torch.nn.Module]
    frozen_checkpoint_modules: dict[str, torch.nn.Module]


def _positive_int(name: str, value: object, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _finite_number(name: str, value: object, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or (parsed <= 0.0 if positive else parsed < 0.0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return parsed


def stage_a_settings(config: Mapping[str, Any]) -> StageASettings:
    """Parse the isolated V28 settings with memory-safe Mac defaults."""

    training = config.get("training")
    if not isinstance(training, Mapping):
        raise TypeError("training config must be a mapping")
    raw = training.get("post_stack_sidecar_stage_a", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TypeError("training.post_stack_sidecar_stage_a must be a mapping")
    allowed = {
        "enabled",
        "max_optimizer_steps",
        "evaluation_interval_steps",
        "batch_size",
        "gradient_accumulation",
        "learning_rate",
        "channel_gain_learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "minimum_answer_types",
        "trainable_routes",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown post_stack_sidecar_stage_a settings: {unknown}")

    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise TypeError("post_stack_sidecar_stage_a.enabled must be a boolean")
    shared_lr = raw.get("learning_rate", training.get("learning_rate", 1e-4))
    routes = raw.get("trainable_routes", ["output_projection", "channel_gain"])
    if not isinstance(routes, Sequence) or isinstance(routes, (str, bytes)):
        raise TypeError("post_stack_sidecar_stage_a.trainable_routes must be a sequence")
    parsed_routes = tuple(str(route) for route in routes)
    if len(parsed_routes) != 2 or set(parsed_routes) != {
        "output_projection",
        "channel_gain",
    }:
        raise ValueError(
            "post_stack_sidecar_stage_a.trainable_routes must contain exactly "
            "output_projection and channel_gain"
        )
    return StageASettings(
        enabled=enabled,
        max_optimizer_steps=_positive_int(
            "post_stack_sidecar_stage_a.max_optimizer_steps",
            raw.get("max_optimizer_steps", 4),
        ),
        evaluation_interval_steps=_positive_int(
            "post_stack_sidecar_stage_a.evaluation_interval_steps",
            raw.get("evaluation_interval_steps", 1),
        ),
        batch_size=_positive_int(
            "post_stack_sidecar_stage_a.batch_size",
            raw.get("batch_size", training.get("batch_size", 1)),
        ),
        gradient_accumulation=_positive_int(
            "post_stack_sidecar_stage_a.gradient_accumulation",
            raw.get(
                "gradient_accumulation", training.get("gradient_accumulation", 1)
            ),
        ),
        output_projection_learning_rate=_finite_number(
            "post_stack_sidecar_stage_a.learning_rate",
            shared_lr,
            positive=True,
        ),
        channel_gain_learning_rate=_finite_number(
            "post_stack_sidecar_stage_a.channel_gain_learning_rate",
            raw.get("channel_gain_learning_rate", shared_lr),
            positive=True,
        ),
        weight_decay=_finite_number(
            "post_stack_sidecar_stage_a.weight_decay",
            raw.get("weight_decay", 0.0),
            positive=False,
        ),
        gradient_clip_norm=_finite_number(
            "post_stack_sidecar_stage_a.gradient_clip_norm",
            raw.get("gradient_clip_norm", training.get("gradient_clip_norm", 1.0)),
            positive=True,
        ),
        minimum_answer_types=_positive_int(
            "post_stack_sidecar_stage_a.minimum_answer_types",
            raw.get("minimum_answer_types", 4),
        ),
        trainable_routes=parsed_routes,
    )


def _resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_runtime_metadata(candidate: Path) -> dict[str, Any]:
    path = candidate / RUNTIME_METADATA_FILENAME
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"V28 candidate runtime metadata is missing: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError("V28 candidate runtime metadata must be a JSON object")
    validate_runtime_checkpoint_metadata(metadata)
    return metadata


def _checkpoint_module_inventory(
    scene_model: SceneTokenizer,
    composer: ContinuousPrefixComposer,
    grounding: QuestionGroundingHead,
    dense_aligner: DenseAlignmentResidual,
    sidecar: DenseSidecarAdapter,
    global_residual: GlobalSceneResidual,
    signed_x_residual: SignedXSceneResidual,
    lora_installation: LoRABankCollection | None,
) -> dict[str, torch.nn.Module]:
    modules: dict[str, torch.nn.Module] = {
        "scene_model": scene_model,
        "composer": composer,
        "grounding": grounding,
        "global_scene_residual": global_residual,
        "signed_x_scene_residual": signed_x_residual,
        "dense_aligner": dense_aligner,
        "dense_sidecar_adapter": sidecar,
    }
    if lora_installation is not None:
        modules.update(lora_installation.state_modules())
    return modules


def freeze_for_stage_a(bundle: StageABundle) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    """Freeze everything except the two explicitly authorized output surfaces."""

    bundle.language.model.requires_grad_(False)
    for module in bundle.checkpoint_modules.values():
        module.requires_grad_(False)
        module.eval()
    sidecar = bundle.dense_sidecar_adapter
    sidecar.output_projection.weight.requires_grad_(True)
    sidecar.channel_gain.requires_grad_(True)
    sidecar.train()
    if bundle.lora_installation is not None:
        bundle.lora_installation.eval()
    return sidecar.output_projection.weight, sidecar.channel_gain


def assert_stage_a_trainable_surface(
    bundle: StageABundle,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, int]:
    """Fail if Gemma, inherited LoRA, or any hidden sidecar tensor can update."""

    authorized = {
        id(bundle.dense_sidecar_adapter.output_projection.weight),
        id(bundle.dense_sidecar_adapter.channel_gain),
    }
    observed_checkpoint = {
        id(parameter)
        for module in bundle.checkpoint_modules.values()
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    if observed_checkpoint != authorized:
        raise RuntimeError(
            "Stage-A checkpoint trainable surface mismatch: "
            f"expected=2 observed={len(observed_checkpoint)}"
        )
    language_trainable = [
        name
        for name, parameter in bundle.language.model.named_parameters()
        if parameter.requires_grad and id(parameter) not in authorized
    ]
    if language_trainable:
        raise RuntimeError(f"Gemma or inherited LoRA is trainable: {language_trainable[:8]}")
    if optimizer is not None:
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if optimizer_ids != authorized:
            raise RuntimeError("Stage-A optimizer contains unauthorized parameters")
    return {
        "output_projection_parameters": (
            bundle.dense_sidecar_adapter.output_projection.weight.numel()
        ),
        "channel_gain_parameters": bundle.dense_sidecar_adapter.channel_gain.numel(),
        "total_trainable_parameters": sum(
            parameter.numel()
            for parameter in (
                bundle.dense_sidecar_adapter.output_projection.weight,
                bundle.dense_sidecar_adapter.channel_gain,
            )
        ),
    }


def _frozen_sidecar_state(sidecar: DenseSidecarAdapter) -> dict[str, torch.Tensor]:
    trainable_keys = {"output_projection.weight", "channel_gain"}
    return {
        name: value
        for name, value in sidecar.state_dict().items()
        if name not in trainable_keys
    }


def frozen_stage_a_state_sha256(bundle: StageABundle) -> str:
    """Hash all inherited checkpoint tensors and frozen sidecar internals."""

    state = {
        f"{module_name}.{name}": value
        for module_name, module in bundle.frozen_checkpoint_modules.items()
        for name, value in module.state_dict().items()
    }
    state.update(
        {
            f"dense_sidecar_adapter.{name}": value
            for name, value in _frozen_sidecar_state(
                bundle.dense_sidecar_adapter
            ).items()
        }
    )
    return tensor_state_sha256(state)


def assert_frozen_stage_a_state(bundle: StageABundle, expected_sha256: str) -> None:
    observed = frozen_stage_a_state_sha256(bundle)
    if observed != expected_sha256:
        raise RuntimeError(
            "Frozen V24/V26/composer/grounding/LoRA state changed: "
            f"expected={expected_sha256} observed={observed}"
        )


def _optimizer(bundle: StageABundle, settings: StageASettings) -> torch.optim.AdamW:
    optimizer = torch.optim.AdamW(
        [
            {
                "name": "dense_sidecar_output_projection",
                "params": [bundle.dense_sidecar_adapter.output_projection.weight],
                "lr": settings.output_projection_learning_rate,
                "weight_decay": settings.weight_decay,
            },
            {
                "name": "dense_sidecar_channel_gain",
                "params": [bundle.dense_sidecar_adapter.channel_gain],
                "lr": settings.channel_gain_learning_rate,
                "weight_decay": 0.0,
            },
        ]
    )
    assert_stage_a_trainable_surface(bundle, optimizer)
    return optimizer


def load_stage_a_bundle(config: dict[str, Any], candidate: Path) -> StageABundle:
    """Load and validate the complete update-0 candidate using runtime metadata."""

    candidate_metadata = _read_runtime_metadata(candidate)
    semantic_dim = int(candidate_metadata["semantic_dim"])
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
    loaded_boundary_contract = language.scene_boundary_contract(boundary_mode)
    if loaded_boundary_contract != native_gemma4_image_contract_setting(config):
        raise ValueError("Loaded local Gemma does not match the configured boundary contract")
    lora_installation = install_lora_banks(language.model, lora_banks_settings(config))

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
        raise ValueError("V28 requires dense alignment, post-stack sidecar, and global residual")
    initial_dense_audit = validate_dense_alignment_state(
        dense_aligner, context="V28 deterministic dense construction"
    )
    if initial_dense_audit["state_sha256"] != dense_alignment_settings(
        config
    ).expected_initial_state_sha256:
        raise ValueError("V28 deterministic dense-aligner initial-state hash mismatch")
    initial_sidecar_audit = validate_dense_sidecar_adapter_state(
        sidecar, context="V28 deterministic sidecar construction"
    )
    if initial_sidecar_audit["state_sha256"] != dense_sidecar_adapter_settings(
        config
    ).expected_initial_state_sha256:
        raise ValueError("V28 deterministic sidecar initial-state hash mismatch")
    if not (
        initial_sidecar_audit["output_projection_exact_zero"]
        and initial_sidecar_audit["channel_gain_exact_zero"]
    ):
        raise ValueError("V28 deterministic sidecar is not exact zero-output")
    signed_x_residual = construct_signed_x_scene_residual(
        config,
        scene_dim=language.hidden_size,
        latent_count=int(config["scene_encoder"]["global_latents"]),
        content_dim=global_residual.width,
    )
    if signed_x_residual is None:
        raise ValueError("V28 requires the frozen signed-X residual stack")
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

    checkpoint_modules = _checkpoint_module_inventory(
        scene_model,
        composer,
        grounding,
        dense_aligner,
        sidecar,
        global_residual,
        signed_x_residual,
        lora_installation,
    )
    loaded_metadata = load_adapter_checkpoint(
        candidate,
        checkpoint_modules,
        device="cpu",
        metadata_filename=RUNTIME_METADATA_FILENAME,
    )
    if loaded_metadata != candidate_metadata:
        raise RuntimeError("Candidate runtime metadata changed while loading")
    validate_checkpoint_contract(
        candidate_metadata,
        config,
        semantic_dim=semantic_dim,
        language_hidden_dim=language.hidden_size,
        lora_parameter_count=(
            0 if lora_installation is None else lora_installation.parameter_count
        ),
        lora_parameter_counts=(
            {} if lora_installation is None else lora_installation.parameter_counts
        ),
        dense_alignment_parameter_count=dense_aligner.parameter_count,
        dense_sidecar_adapter_parameter_count=sidecar.parameter_count,
    )
    validate_dense_alignment_state(
        dense_aligner,
        expected_parameter_count=candidate_metadata["dense_alignment_parameter_count"],
        context="V28 candidate load",
    )
    validate_dense_sidecar_adapter_state(
        sidecar,
        expected_parameter_count=candidate_metadata[
            "dense_sidecar_adapter_parameter_count"
        ],
        expected_state_sha256=candidate_metadata["dense_sidecar_adapter_state_sha256"],
        context="V28 candidate load",
    )
    if dense_aligner.state_sha256() != candidate_metadata["dense_alignment_state_sha256"]:
        raise ValueError("Loaded V26 calibrated dense-aligner state hash mismatch")
    if dense_aligner.application_mode != "coverage_sidecar" or dense_aligner.sidecar_scale != 0:
        raise ValueError("V28 requires coverage_sidecar routing with exact zero direct scale")
    if lora_installation is not None:
        validate_lora_banks_checkpoint_state(candidate_metadata, lora_installation)
    global_residual.validate_structural_state()
    if module_collection_state_sha256(
        {"global_scene_residual": global_residual}
    ) != candidate_metadata.get("global_scene_residual_state_sha256"):
        raise ValueError("Loaded global scene residual state hash mismatch")
    signed_x_residual.validate_structural_state()
    if module_collection_state_sha256(
        {"signed_x_scene_residual": signed_x_residual}
    ) != candidate_metadata.get("signed_x_scene_residual_state_sha256"):
        raise ValueError("Loaded signed-X scene residual state hash mismatch")

    scene_state_modules = {
        "scene_model": scene_model,
        "composer": composer,
        "grounding": grounding,
    }
    if module_collection_state_sha256(scene_state_modules) != candidate_metadata.get(
        "frozen_scene_state_sha256"
    ):
        raise ValueError("Loaded V24 scene/composer/grounding state hash mismatch")

    device = language.device
    for module in checkpoint_modules.values():
        module.to(device)
    frozen_modules = {
        name: module
        for name, module in checkpoint_modules.items()
        if name != "dense_sidecar_adapter"
    }
    bundle = StageABundle(
        config=config,
        candidate_metadata=candidate_metadata,
        language=language,
        scene_model=scene_model,
        dense_aligner=dense_aligner,
        dense_sidecar_adapter=sidecar,
        global_scene_residual=global_residual,
        signed_x_scene_residual=signed_x_residual,
        composer=composer,
        grounding=grounding,
        lora_installation=lora_installation,
        checkpoint_modules=checkpoint_modules,
        frozen_checkpoint_modules=frozen_modules,
    )
    freeze_for_stage_a(bundle)
    assert_stage_a_trainable_surface(bundle)
    return bundle


def _scalar_audit(audit: Mapping[str, Any], key: str) -> float:
    value = audit.get(key)
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return float(value.detach().float().cpu())
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise ValueError(f"Scene output audit lacks scalar {key!r}")


def cache_scene_output(
    *,
    scene_id: str,
    output: Any,
    voxel_count: int,
    composer: ContinuousPrefixComposer,
    sidecar: DenseSidecarAdapter,
) -> CachedSceneTokens:
    """Validate and retain only the question-independent post-stack inputs."""

    sidecar_tokens = getattr(output, "aligned_sidecar_tokens", None)
    if not isinstance(sidecar_tokens, torch.Tensor):
        raise TypeError("Cached scene output lacks aligned sidecar tokens")
    base = output.scene_tokens.detach()
    sidecar_tokens = sidecar_tokens.detach()
    processed = int(_scalar_audit(output.audit, "processed_voxels"))
    sidecar_processed = int(
        _scalar_audit(output.audit, "aligned_sidecar_processed_voxels")
    )
    minimum = _scalar_audit(output.audit, "aligned_sidecar_min_voxel_contribution")
    if processed != voxel_count or sidecar_processed != voxel_count:
        raise RuntimeError(
            f"Incomplete scene cache for {scene_id}: base={processed} "
            f"sidecar={sidecar_processed} expected={voxel_count}"
        )
    if not math.isfinite(minimum) or minimum <= 0.0:
        raise RuntimeError(f"At least one voxel has no sidecar contribution in {scene_id}")
    if base.shape != sidecar_tokens.shape:
        raise RuntimeError("Base and sidecar cache shapes differ")
    adapted = sidecar(base, sidecar_tokens)
    if not torch.equal(adapted, base):
        raise RuntimeError(f"V28 update-0 adapter changed base tokens for {scene_id}")
    with torch.no_grad():
        base_prefix = composer.scene_prefix(base)
        adapted_prefix = composer.scene_prefix(adapted)
    if not torch.equal(base_prefix, adapted_prefix):
        raise RuntimeError(f"V28 update-0 adapter changed scene prefix for {scene_id}")
    base_hash = prefix_sha256(base_prefix)
    if prefix_sha256(adapted_prefix) != base_hash:
        raise RuntimeError(f"V28 update-0 prefix hash mismatch for {scene_id}")
    return CachedSceneTokens(
        scene_id=scene_id,
        base_scene_tokens=base,
        aligned_sidecar_tokens=sidecar_tokens,
        base_prefix_sha256=base_hash,
        voxel_count=voxel_count,
        processed_voxels=processed,
        minimum_voxel_contribution=minimum,
    )


def cache_question_independent_scenes(
    bundle: StageABundle, scene_ids: Sequence[str]
) -> tuple[dict[str, CachedSceneTokens], dict[str, Any]]:
    """Encode every selected scene exactly once without accepting questions."""

    caches: dict[str, CachedSceneTokens] = {}
    loaded_environment_files: list[str] = []
    started = time.perf_counter()
    for scene_id in sorted(set(scene_ids)):
        map_path = artifact_root(bundle.config, "maps") / scene_id / "voxel_map.npz"
        resolved = map_path.resolve()
        if "oracle" in {part.casefold() for part in resolved.parts}:
            raise RuntimeError("Refusing to load an oracle path as a Stage-A environment")
        data = load_map_tensors(
            resolved,
            bundle.config["scene"]["room_size_m"],
            bundle.language.device,
            input_voxel_size_m=bundle.config["scene_encoder"].get(
                "input_voxel_size_m"
            ),
        )
        if data.feature_dim != int(bundle.candidate_metadata["semantic_dim"]):
            raise ValueError(f"Semantic dimension mismatch for {scene_id}")
        # no_grad creates ordinary detached tensors that remain legal inputs to
        # a later adapter backward pass. inference_mode tensors cannot be saved
        # for backward by the post-stack adapter.
        with torch.no_grad():
            output = map_forward(
                bundle.scene_model,
                data,
                bundle.global_scene_residual,
                bundle.signed_x_scene_residual,
                bundle.dense_aligner,
                dense_sidecar_adapter=None,
            )
            caches[scene_id] = cache_scene_output(
                scene_id=scene_id,
                output=output,
                voxel_count=data.voxel_count,
                composer=bundle.composer,
                sidecar=bundle.dense_sidecar_adapter,
            )
        loaded_environment_files.append(str(resolved))
        del output, data
        if bundle.language.device.type == "mps":
            torch.mps.empty_cache()
    return caches, {
        "schema_version": 1,
        "scene_count": len(caches),
        "cache_build_seconds": time.perf_counter() - started,
        "question_inputs_to_scene_cache": False,
        "question_dependent_retrieval": False,
        "all_scene_slots_accounted": True,
        "all_voxels_covered": all(
            cache.processed_voxels == cache.voxel_count for cache in caches.values()
        ),
        "oracle_environment_files_loaded": False,
        "loaded_environment_files": loaded_environment_files,
        "scene_prefixes": {
            scene_id: cache.base_prefix_sha256 for scene_id, cache in sorted(caches.items())
        },
    }


def answer_token_nll(
    *,
    cache: CachedSceneTokens,
    records: Sequence[QARecord],
    adapter: DenseSidecarAdapter,
    language: Any,
    composer: ContinuousPrefixComposer,
    config: Mapping[str, Any],
) -> torch.Tensor:
    """Compute broad teacher-forced NLL only on answer tokens."""

    if not records:
        raise ValueError("answer_token_nll requires at least one QA record")
    if any(record.scene_id != cache.scene_id for record in records):
        raise ValueError("Every NLL batch record must match the cached scene")
    scene_tokens = adapter(cache.base_scene_tokens, cache.aligned_sidecar_tokens)
    scene_tokens = scene_tokens.expand(len(records), -1, -1)
    model_dtype = next(language.model.parameters()).dtype
    embedding_layer = language.model.get_input_embeddings()
    batches = []
    for index, record in enumerate(records):
        prompt_ids = prompt_token_ids(
            language.tokenizer,
            str(config["language"]["system_prompt"]),
            record.question,
            language.device,
        )
        answer_ids = tokenize_answer(language.tokenizer, record.answer, language.device)
        batches.append(
            composer.compose(
                scene_tokens[index : index + 1].to(model_dtype),
                prompt_ids,
                embedding_layer,
                answer_ids,
                prefix_backend=getattr(language, "prefix_backend", None),
            )
        )
    prefix_batch = stack_prefix_batches(
        batches,
        language.device,
        prefix_backend=getattr(language, "prefix_backend", None),
    )
    if prefix_batch.labels is None or not torch.any(prefix_batch.labels != -100):
        raise RuntimeError("Stage-A batch contains no supervised answer tokens")
    output = forward_prefix_batch(language, prefix_batch)
    loss = output.loss.float()
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError("Stage-A answer-token NLL is nonfinite or nonscalar")
    return loss


def records_by_scene(records: Sequence[QARecord]) -> dict[str, list[QARecord]]:
    grouped: defaultdict[str, list[QARecord]] = defaultdict(list)
    for record in records:
        grouped[record.scene_id].append(record)
    return {scene_id: grouped[scene_id] for scene_id in sorted(grouped)}


def _bounded_records(
    records: Sequence[QARecord], maximum: int | None, *, seed: int
) -> list[QARecord]:
    selected = list(records)
    if maximum is None or maximum >= len(selected):
        return selected
    if maximum < 1:
        raise ValueError("Question maximum must be positive")
    random.Random(seed).shuffle(selected)
    return sorted(selected[:maximum], key=lambda item: (item.scene_id, item.question_id))


def _epoch_batches(
    records: Sequence[QARecord], *, batch_size: int, seed: int
) -> list[tuple[str, list[QARecord]]]:
    rng = random.Random(seed)
    batches: list[tuple[str, list[QARecord]]] = []
    for scene_id, scene_records in records_by_scene(records).items():
        shuffled = list(scene_records)
        rng.shuffle(shuffled)
        batches.extend(
            (scene_id, shuffled[offset : offset + batch_size])
            for offset in range(0, len(shuffled), batch_size)
        )
    rng.shuffle(batches)
    return batches


def validation_answer_nll(
    *,
    records: Sequence[QARecord],
    caches: Mapping[str, CachedSceneTokens],
    bundle: StageABundle,
    batch_size: int,
) -> dict[str, Any]:
    """Score held-out QA while environmental inference remains oracle-free."""

    was_training = bundle.dense_sidecar_adapter.training
    bundle.dense_sidecar_adapter.eval()
    weighted = 0.0
    questions = 0
    try:
        with torch.inference_mode():
            for scene_id, scene_records in records_by_scene(records).items():
                cache = caches[scene_id]
                for offset in range(0, len(scene_records), batch_size):
                    batch = scene_records[offset : offset + batch_size]
                    loss = answer_token_nll(
                        cache=cache,
                        records=batch,
                        adapter=bundle.dense_sidecar_adapter,
                        language=bundle.language,
                        composer=bundle.composer,
                        config=bundle.config,
                    )
                    weighted += float(loss.detach().cpu()) * len(batch)
                    questions += len(batch)
    finally:
        bundle.dense_sidecar_adapter.train(was_training)
    if not questions:
        raise ValueError("Validation set is empty")
    return {
        "answer_token_nll": weighted / questions,
        "question_count": questions,
        "environmental_oracle_files_loaded": False,
        "question_dependent_scene_processing": False,
        "cached_scene_prefixes_reused": True,
    }


def _metadata(
    *,
    bundle: StageABundle,
    settings: StageASettings,
    candidate: Path,
    cache_audit: Mapping[str, Any],
    frozen_state_hash: str,
    train_records: Sequence[QARecord],
    validation_records: Sequence[QARecord],
    history: Sequence[Mapping[str, Any]],
    epoch: int,
    optimizer_updates: int,
    best_epoch: int,
    best_validation_nll: float,
    trainable_surface: Mapping[str, int],
) -> dict[str, Any]:
    result = dict(bundle.candidate_metadata)
    result.update(
        {
            "schema_version": 3,
            "config_hash": config_hash(bundle.config),
            "epoch": epoch,
            "optimizer_step": optimizer_updates,
            "history": list(history),
            "best_epoch": best_epoch,
            "best_monitor_loss": best_validation_nll,
            "monitor_name": "validation_answer_token_nll",
            "dense_sidecar_adapter": dense_sidecar_adapter_settings(
                bundle.config
            ).contract(),
            "dense_sidecar_adapter_parameter_count": (
                bundle.dense_sidecar_adapter.parameter_count
            ),
            "dense_sidecar_adapter_initial_state_sha256": (
                dense_sidecar_adapter_settings(
                    bundle.config
                ).expected_initial_state_sha256
            ),
            "dense_sidecar_adapter_state_sha256": (
                bundle.dense_sidecar_adapter.state_sha256()
            ),
            "frozen_dense_alignment_state_sha256": (
                bundle.dense_aligner.state_sha256()
            ),
            "freeze_scene_adapter": True,
            "question_dependent_scene_processing": False,
            "all_voxels_transformed": True,
            "v28_stage_a": {
                "schema_version": 1,
                "source_candidate": str(candidate),
                "source_candidate_adapter_sha256": _file_sha256(
                    candidate / "adapter.safetensors"
                ),
                "source_runtime_metadata_sha256": _file_sha256(
                    candidate / RUNTIME_METADATA_FILENAME
                ),
                "objective": "broad_answer_token_nll",
                "trainable_surface": dict(trainable_surface),
                "settings": settings.__dict__,
                "frozen_state_sha256": frozen_state_hash,
                "scene_cache": dict(cache_audit),
                "train_scene_ids": sorted({record.scene_id for record in train_records}),
                "validation_scene_ids": sorted(
                    {record.scene_id for record in validation_records}
                ),
                "train_question_count": len(train_records),
                "validation_question_count": len(validation_records),
                "train_answer_types": sorted(
                    {record.answer_type for record in train_records}
                ),
                "qa_supervision_serialized_to_runtime": False,
                "oracle_environment_files_loaded": False,
                "gemma_frozen": True,
                "inherited_lora_frozen": True,
                "composer_grounding_frozen": True,
                "v24_v26_stack_frozen": True,
            },
        }
    )
    return result


def _save(
    destination: Path,
    *,
    bundle: StageABundle,
    metadata: dict[str, Any],
    optimizer: torch.optim.Optimizer | None,
) -> None:
    save_adapter_checkpoint(destination, bundle.checkpoint_modules, metadata)
    if optimizer is not None:
        save_optimizer_checkpoint(destination, optimizer)


def run_stage_a(
    *,
    config: dict[str, Any],
    candidate: Path,
    output: Path,
    max_optimizer_steps_override: int | None = None,
    max_train_questions: int | None = None,
    max_validation_questions: int | None = None,
) -> dict[str, Any]:
    """Run the complete bounded Stage-A experiment and return its audit."""

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty Stage-A output: {output}")
    settings = stage_a_settings(config)
    if not settings.enabled:
        raise ValueError("V28 Stage-A is disabled by configuration")
    if max_optimizer_steps_override is not None:
        settings = StageASettings(
            **{
                **settings.__dict__,
                "max_optimizer_steps": _positive_int(
                    "max_optimizer_steps", max_optimizer_steps_override
                ),
            }
        )
    seed = int(config["seed"])
    torch.manual_seed(seed)
    random.seed(seed)

    qa_root = artifact_root(config, "qa")
    train_records = _bounded_records(
        load_qa_split_dataset(qa_root, "train").records,
        max_train_questions,
        seed=seed,
    )
    validation_records = _bounded_records(
        load_qa_split_dataset(qa_root, "validation").records,
        max_validation_questions,
        seed=seed + 1,
    )
    train_scenes = {record.scene_id for record in train_records}
    validation_scenes = {record.scene_id for record in validation_records}
    overlap = sorted(train_scenes & validation_scenes)
    if overlap:
        raise ValueError(f"Stage-A train/validation scenes overlap: {overlap}")
    answer_types = {record.answer_type for record in train_records}
    if len(answer_types) < settings.minimum_answer_types:
        raise ValueError(
            "Stage-A broad NLL selection has too few answer types: "
            f"required={settings.minimum_answer_types} observed={sorted(answer_types)}"
        )

    bundle = load_stage_a_bundle(config, candidate)
    caches, cache_audit = cache_question_independent_scenes(
        bundle, sorted(train_scenes | validation_scenes)
    )
    frozen_hash = frozen_stage_a_state_sha256(bundle)
    trainable_surface = assert_stage_a_trainable_surface(bundle)
    optimizer = _optimizer(bundle, settings)
    baseline_validation = validation_answer_nll(
        records=validation_records,
        caches=caches,
        bundle=bundle,
        batch_size=settings.batch_size,
    )
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "optimizer_updates": 0,
            "validation_answer_token_nll": baseline_validation["answer_token_nll"],
            "update_0_identity_verified": True,
        }
    ]
    best_epoch = 0
    best_validation = float(baseline_validation["answer_token_nll"])
    optimizer_updates = 0
    output.mkdir(parents=True, exist_ok=True)
    initial_metadata = _metadata(
        bundle=bundle,
        settings=settings,
        candidate=candidate,
        cache_audit=cache_audit,
        frozen_state_hash=frozen_hash,
        train_records=train_records,
        validation_records=validation_records,
        history=history,
        epoch=0,
        optimizer_updates=0,
        best_epoch=0,
        best_validation_nll=best_validation,
        trainable_surface=trainable_surface,
    )
    _save(output / "update_000", bundle=bundle, metadata=initial_metadata, optimizer=None)
    _save(output / "best", bundle=bundle, metadata=initial_metadata, optimizer=None)

    batch_queue: list[tuple[str, list[QARecord]]] = []
    curriculum_cycle = 0
    for optimizer_update in range(1, settings.max_optimizer_steps + 1):
        bundle.dense_sidecar_adapter.train()
        window: list[tuple[str, list[QARecord]]] = []
        while len(window) < settings.gradient_accumulation:
            if not batch_queue:
                curriculum_cycle += 1
                batch_queue = _epoch_batches(
                    train_records,
                    batch_size=settings.batch_size,
                    seed=seed + curriculum_cycle,
                )
            window.append(batch_queue.pop())
        weighted_loss = 0.0
        question_count = 0
        optimizer.zero_grad(set_to_none=True)
        for scene_id, batch_records in window:
            loss = answer_token_nll(
                cache=caches[scene_id],
                records=batch_records,
                adapter=bundle.dense_sidecar_adapter,
                language=bundle.language,
                composer=bundle.composer,
                config=bundle.config,
            )
            (loss / len(window)).backward()
            weighted_loss += float(loss.detach().cpu()) * len(batch_records)
            question_count += len(batch_records)
        assert_stage_a_trainable_surface(bundle, optimizer)
        parameters = [
            bundle.dense_sidecar_adapter.output_projection.weight,
            bundle.dense_sidecar_adapter.channel_gain,
        ]
        if any(parameter.grad is None for parameter in parameters):
            raise RuntimeError("Stage-A optimizer surface did not receive gradients")
        if any(not torch.isfinite(parameter.grad).all() for parameter in parameters):
            raise RuntimeError("Stage-A gradient contains NaN or infinity")
        norm = torch.nn.utils.clip_grad_norm_(parameters, settings.gradient_clip_norm)
        gradient_norm = float(norm.detach().cpu())
        optimizer.step()
        optimizer_updates += 1
        if optimizer_updates != optimizer_update:
            raise RuntimeError("Stage-A optimizer update accounting drifted")
        assert_frozen_stage_a_state(bundle, frozen_hash)
        validate_dense_sidecar_adapter_state(
            bundle.dense_sidecar_adapter,
            expected_parameter_count=bundle.candidate_metadata[
                "dense_sidecar_adapter_parameter_count"
            ],
            context=f"V28 Stage-A update {optimizer_updates}",
        )
        if not question_count:
            raise RuntimeError("Stage-A update contained no questions")
        validation = (
            validation_answer_nll(
                records=validation_records,
                caches=caches,
                bundle=bundle,
                batch_size=settings.batch_size,
            )
            if optimizer_updates % settings.evaluation_interval_steps == 0
            or optimizer_updates == settings.max_optimizer_steps
            else None
        )
        validation_value = (
            None if validation is None else float(validation["answer_token_nll"])
        )
        if validation_value is not None and validation_value < best_validation:
            best_validation = validation_value
            best_epoch = optimizer_updates
            improved = True
        else:
            improved = False
        history.append(
            {
                "optimizer_update": optimizer_updates,
                "curriculum_cycle": curriculum_cycle,
                "optimizer_updates": optimizer_updates,
                "train_window_answer_token_nll": weighted_loss / question_count,
                "train_question_count": question_count,
                "validation_answer_token_nll": validation_value,
                "preclip_gradient_norm": gradient_norm,
                "direct_gain_abs_max": float(
                    bundle.dense_sidecar_adapter.bounded_channel_gain()
                    .detach()
                    .float()
                    .abs()
                    .max()
                    .cpu()
                ),
                "frozen_state_sha256": frozen_hash,
            }
        )
        metadata = _metadata(
            bundle=bundle,
            settings=settings,
            candidate=candidate,
            cache_audit=cache_audit,
            frozen_state_hash=frozen_hash,
            train_records=train_records,
            validation_records=validation_records,
            history=history,
            epoch=optimizer_updates,
            optimizer_updates=optimizer_updates,
            best_epoch=best_epoch,
            best_validation_nll=best_validation,
            trainable_surface=trainable_surface,
        )
        update_path = output / f"update_{optimizer_updates:03d}"
        _save(update_path, bundle=bundle, metadata=metadata, optimizer=optimizer)
        if improved:
            _save(output / "best", bundle=bundle, metadata=metadata, optimizer=None)
        print(
            json.dumps(
                {
                    "phase": "v28_stage_a_update",
                    "optimizer_updates": optimizer_updates,
                    "train_window_answer_token_nll": history[-1][
                        "train_window_answer_token_nll"
                    ],
                    "validation_answer_token_nll": validation_value,
                    "best_epoch": best_epoch,
                    "best_validation_answer_token_nll": best_validation,
                }
            ),
            flush=True,
        )

    assert_frozen_stage_a_state(bundle, frozen_hash)
    return {
        "schema_version": 1,
        "artifact": "v28_post_stack_sidecar_stage_a",
        "output": str(output),
        "best_checkpoint": str(output / "best"),
        "best_epoch": best_epoch,
        "baseline_validation_answer_token_nll": baseline_validation["answer_token_nll"],
        "best_validation_answer_token_nll": best_validation,
        "optimizer_updates": optimizer_updates,
        "trainable_surface": trainable_surface,
        "frozen_state_sha256": frozen_hash,
        "question_dependent_scene_processing": False,
        "oracle_environment_files_loaded": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--max-train-questions", type=int)
    parser.add_argument("--max-validation-questions", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    report = run_stage_a(
        config=config,
        candidate=_resolve_path(args.candidate),
        output=_resolve_path(args.output),
        max_optimizer_steps_override=args.max_optimizer_steps,
        max_train_questions=args.max_train_questions,
        max_validation_questions=args.max_validation_questions,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
