"""Oracle-isolated static chat runtime for a continuous 3D scene prefix."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT, config_hash, project_path
from semantic_3d_chat.language.generation import generate_from_embeddings
from semantic_3d_chat.language.local_lm import (
    LocalLanguageModel,
    load_local_language_model,
    prompt_token_ids,
    question_token_ids,
)
from semantic_3d_chat.language.lora import (
    install_lora_banks,
    lora_banks_checkpoint_contract,
    lora_banks_optimizer_settings,
    lora_banks_settings,
    lora_checkpoint_contract_mismatch,
    validate_lora_banks_checkpoint_state,
)
from semantic_3d_chat.language.prefix_injection import (
    ContinuousPrefixComposer,
    native_gemma4_image_contract_setting,
    prefix_sha256,
    scene_boundary_contract_mismatch,
    scene_boundary_mode_setting,
    scene_prefix_after_bos_contract_mismatch,
    scene_prefix_after_bos_setting,
)
from semantic_3d_chat.scene_encoder.dense_alignment import (
    DenseAlignmentResidual,
    construct_dense_alignment,
    dense_alignment_settings,
    validate_dense_alignment_state,
)
from semantic_3d_chat.scene_encoder.global_residual import (
    ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
    GlobalSceneResidual,
    apply_global_scene_residual,
    construct_global_scene_residual,
    global_scene_residual_settings,
)
from semantic_3d_chat.scene_encoder.map_io import MapTensorData, load_map_tensors
from semantic_3d_chat.scene_encoder.projector import SceneTokenizer, SceneTokenizerOutput
from semantic_3d_chat.scene_encoder.signed_x_dispatch import (
    SignedXSceneResidual,
    apply_signed_x_scene_residual,
    construct_signed_x_scene_residual,
    frozen_v18_centered_content_values,
    signed_x_scene_residual_settings,
)
from semantic_3d_chat.training.checkpointing import (
    RUNTIME_METADATA_FILENAME,
    load_adapter_checkpoint,
    module_collection_state_sha256,
    validate_runtime_checkpoint_metadata,
)
from semantic_3d_chat.training.losses import QuestionGroundingHead, denormalize_xyz

GenerationFunction = Callable[
    [torch.nn.Module, torch.Tensor, torch.Tensor, int, int | Sequence[int] | None],
    torch.Tensor,
]

_OPAQUE_SCENE_ID = re.compile(r"scene_[0-9]{6}")
_FORBIDDEN_DATA_DIRECTORIES = frozenset({"oracle", "qa", "rendered", "features"})
_SUPPORTED_CHECKPOINT_SCHEMAS = frozenset({1, 2, 3})
_SCENE_TOKENIZER_CONTRACT_DEFAULTS: dict[str, int | float | None] = {
    "language_aligned_tail_dim": 0,
    "native_aligned_coverage_scale": 0.0,
    "learned_scene_token_scale": 1.0,
    "learned_scene_token_rms_target": None,
}
_DENSE_ALIGNMENT_RUNTIME_REQUIRED_KEYS = frozenset(
    {
        "dense_alignment",
        "dense_alignment_parameter_count",
        "dense_alignment_initial_state_sha256",
        "dense_alignment_state_sha256",
        "all_voxels_transformed",
    }
)
_DENSE_ALIGNMENT_TRAINING_ONLY_KEYS = frozenset(
    {
        "dense_alignment_zero_output_equivalence",
        "dense_alignment_calibration",
        "dense_alignment_optimizer",
    }
)
_DENSE_ALIGNMENT_METADATA_KEYS = (
    _DENSE_ALIGNMENT_RUNTIME_REQUIRED_KEYS | _DENSE_ALIGNMENT_TRAINING_ONLY_KEYS
)


@dataclass(frozen=True)
class ChatAnswer:
    question: str
    answer: str
    grounding_xyz_m: tuple[float, float, float]
    grounding_confidence: float
    grounding_support_distance_m: float
    prefix_hash: str
    generated_tokens: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["grounding_xyz_m"] = list(self.grounding_xyz_m)
        return payload


def _guard_runtime_input(path: str | Path, purpose: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    lowered_parts = {part.casefold() for part in resolved.parts}
    forbidden = sorted(lowered_parts & _FORBIDDEN_DATA_DIRECTORIES)
    if forbidden:
        raise ValueError(f"Refusing to load {purpose} from forbidden runtime path: {resolved}")
    return resolved


def _read_checkpoint_metadata(checkpoint: Path, audit: FileAccessAudit | None) -> dict[str, Any]:
    unresolved = checkpoint / RUNTIME_METADATA_FILENAME
    if unresolved.is_symlink():
        raise ValueError("Runtime checkpoint metadata must not be a symbolic link")
    if not unresolved.is_file():
        raise FileNotFoundError(f"Runtime checkpoint metadata is missing: {unresolved}")
    metadata_path = _guard_runtime_input(
        unresolved, "runtime checkpoint metadata"
    )
    if metadata_path.name != RUNTIME_METADATA_FILENAME:
        raise ValueError("Runtime checkpoint metadata resolved to a forbidden filename")
    if audit is not None:
        audit.record(metadata_path)
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise TypeError("Checkpoint metadata must be a JSON object")
    validate_runtime_checkpoint_metadata(metadata)
    return metadata


def _equal_number(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    if left is None or right is None:
        return left is right
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _scene_tokenizer_contract(config: dict[str, Any]) -> dict[str, int | float | None]:
    settings = config["scene_encoder"]
    rms_target = settings.get("learned_scene_token_rms_target")
    return {
        "language_aligned_tail_dim": int(settings.get("language_aligned_tail_dim", 0)),
        "native_aligned_coverage_scale": float(settings.get("native_aligned_coverage_scale", 0.0)),
        "learned_scene_token_scale": float(settings.get("learned_scene_token_scale", 1.0)),
        "learned_scene_token_rms_target": (None if rms_target is None else float(rms_target)),
    }


def _validate_global_scene_residual_state(
    module: GlobalSceneResidual,
    *,
    expected_parameter_count: object,
    context: str,
) -> dict[str, Any]:
    """Fail before inference on nonfinite or structurally inconsistent state."""

    audit = module.validate_structural_state()
    observed = module.parameter_count
    if (
        isinstance(expected_parameter_count, bool)
        or not isinstance(expected_parameter_count, int)
        or expected_parameter_count != observed
    ):
        raise ValueError(
            f"Global scene residual parameter-count mismatch during {context}: "
            f"checkpoint={expected_parameter_count} runtime={observed}"
        )
    if audit.get("parameter_count") != observed:
        raise RuntimeError("Global scene residual structural audit reported a stale count")
    return audit


def _validate_signed_x_scene_residual_state(
    module: SignedXSceneResidual,
    *,
    expected_parameter_count: object,
    context: str,
) -> dict[str, Any]:
    """Fail before inference on signed-X state or parameter-surface drift."""

    audit = module.validate_structural_state()
    observed = module.parameter_count
    if (
        isinstance(expected_parameter_count, bool)
        or not isinstance(expected_parameter_count, int)
        or expected_parameter_count != observed
    ):
        raise ValueError(
            f"Signed-X scene residual parameter-count mismatch during {context}: "
            f"checkpoint={expected_parameter_count} runtime={observed}"
        )
    if audit.get("parameter_count") != observed:
        raise RuntimeError("Signed-X scene residual structural audit reported a stale count")
    return audit


def _signed_x_frozen_base_provenance_mismatch(
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    """Require explicit evidence that V19 loaded and froze its trained V18 base."""

    mismatch: dict[str, Any] = {}
    base_hash = metadata.get("global_scene_residual_state_sha256")
    frozen_hash = metadata.get("frozen_global_scene_residual_state_sha256")
    if (
        not isinstance(base_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", base_hash) is None
        or frozen_hash != base_hash
    ):
        mismatch["frozen_global_scene_residual_state_sha256"] = {
            "checkpoint": frozen_hash,
            "required": base_hash,
        }

    signed_equivalence = metadata.get("signed_x_scene_residual_zero_output_equivalence")
    required_equivalence = {
        "verified": True,
        "base": "loaded_frozen_global_scene_residual",
        "question_dependent_scene_processing": False,
        "all_scene_slots_accounted": True,
    }
    if not isinstance(signed_equivalence, dict):
        mismatch["signed_x_scene_residual_zero_output_equivalence"] = {
            "checkpoint": signed_equivalence,
            "required": required_equivalence,
        }
    else:
        equivalence_mismatch = {
            key: {"checkpoint": signed_equivalence.get(key), "required": value}
            for key, value in required_equivalence.items()
            if signed_equivalence.get(key) != value
        }
        if equivalence_mismatch:
            mismatch["signed_x_scene_residual_zero_output_equivalence"] = equivalence_mismatch

    provenance = metadata.get("initialization_provenance")
    signed_initial_hash = metadata.get("signed_x_scene_residual_initial_state_sha256")
    signed_hash = metadata.get("signed_x_scene_residual_state_sha256")
    frozen_signed_hash = metadata.get("frozen_signed_x_scene_residual_state_sha256")
    if frozen_signed_hash is not None and frozen_signed_hash != signed_hash:
        mismatch["frozen_signed_x_scene_residual_state_sha256"] = {
            "checkpoint": frozen_signed_hash,
            "required": signed_hash,
        }
    required_provenance = {
        "schema_version": 1,
        "source_global_scene_residual_state_sha256": base_hash,
        "source_signed_x_scene_residual_state_sha256": signed_hash,
        "global_scene_residual_frozen": True,
        "signed_x_scene_residual_frozen": (
            True if frozen_signed_hash is not None else None
        ),
        "signed_x_scene_residual_initial_state_sha256": signed_initial_hash,
        "signed_x_zero_output_transition_verified": True,
        "question_dependent_scene_processing": False,
    }
    if not isinstance(provenance, dict):
        mismatch["initialization_provenance"] = {
            "checkpoint": provenance,
            "required": required_provenance,
        }
    else:
        provenance_mismatch = {
            key: {"checkpoint": provenance.get(key), "required": value}
            for key, value in required_provenance.items()
            if provenance.get(key) != value
        }
        if provenance_mismatch:
            mismatch["initialization_provenance"] = provenance_mismatch
    return mismatch or None


def validate_checkpoint_contract(
    metadata: dict[str, Any],
    config: dict[str, Any],
    *,
    semantic_dim: int,
    language_hidden_dim: int,
    lora_parameter_count: int = 0,
    lora_parameter_counts: dict[str, int] | None = None,
    dense_alignment_parameter_count: int = 0,
) -> list[str]:
    """Enforce adapter-shape compatibility while surfacing unrelated config drift."""

    required = {
        "schema_version",
        "semantic_dim",
        "language_hidden_dim",
        "language_model_id",
        "language_revision",
        "scene_latents",
        "scene_model_dim",
        "input_voxel_size_m",
        "config_hash",
    }
    scene_tokenizer_contract = _scene_tokenizer_contract(config)
    residual_settings = global_scene_residual_settings(config)
    residual_contract = residual_settings.contract()
    signed_x_settings = signed_x_scene_residual_settings(config)
    signed_x_contract = signed_x_settings.contract()
    dense_settings = dense_alignment_settings(config)
    dense_contract = dense_settings.contract()
    checkpoint_dense_keys = sorted(_DENSE_ALIGNMENT_METADATA_KEYS & metadata.keys())
    if not dense_settings.enabled and checkpoint_dense_keys:
        raise ValueError(
            "Checkpoint contains dense-alignment metadata while runtime dense alignment "
            f"is disabled: {checkpoint_dense_keys}"
        )
    checkpoint_signed_x_contract = metadata.get("signed_x_scene_residual")
    checkpoint_signed_x_enabled = bool(
        isinstance(checkpoint_signed_x_contract, dict)
        and checkpoint_signed_x_contract.get("enabled") is True
    )
    uses_aligned_bypass = any(
        not _equal_number(value, _SCENE_TOKENIZER_CONTRACT_DEFAULTS[key])
        for key, value in scene_tokenizer_contract.items()
    )
    metadata_has_scene_tokenizer_contract = any(key in metadata for key in scene_tokenizer_contract)
    if uses_aligned_bypass or metadata_has_scene_tokenizer_contract:
        required.update(scene_tokenizer_contract)
    if residual_contract["enabled"] or "global_scene_residual" in metadata:
        required.update(
            {
                "global_scene_residual",
                "global_scene_residual_parameter_count",
                "global_scene_residual_initial_state_sha256",
                "global_scene_residual_state_sha256",
                "global_scene_residual_zero_output_equivalence",
                "question_dependent_scene_processing",
            }
        )
    if signed_x_contract["enabled"] or checkpoint_signed_x_enabled:
        required.update(
            {
                "signed_x_scene_residual",
                "signed_x_scene_residual_parameter_count",
                "signed_x_scene_residual_initial_state_sha256",
                "signed_x_scene_residual_state_sha256",
                "signed_x_scene_residual_zero_output_equivalence",
                "frozen_global_scene_residual_state_sha256",
                "initialization_provenance",
                "question_dependent_scene_processing",
            }
        )
    if dense_settings.enabled:
        required.update(_DENSE_ALIGNMENT_RUNTIME_REQUIRED_KEYS)
        required.add("question_dependent_scene_processing")
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(f"Checkpoint metadata is missing required fields: {missing}")
    schema_version = metadata.get("schema_version")
    if schema_version not in _SUPPORTED_CHECKPOINT_SCHEMAS:
        raise ValueError(
            "Unsupported checkpoint metadata schema: "
            f"{schema_version}; supported={sorted(_SUPPORTED_CHECKPOINT_SCHEMAS)}"
        )
    expected = {
        "semantic_dim": int(semantic_dim),
        "language_hidden_dim": int(language_hidden_dim),
        "language_model_id": str(config["language"]["model_id"]),
        "language_revision": str(config["language"]["revision"]),
        "scene_latents": int(config["scene_encoder"]["global_latents"]),
        "scene_model_dim": int(config["scene_encoder"]["model_dim"]),
    }
    architecture_version = config["scene_encoder"].get("architecture_version")
    if architecture_version is not None:
        expected["scene_encoder_architecture_version"] = str(architecture_version)
    mismatches = {
        key: {"checkpoint": metadata.get(key), "runtime": value}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    configured_voxel_size = config["scene_encoder"].get("input_voxel_size_m")
    if not _equal_number(metadata.get("input_voxel_size_m"), configured_voxel_size):
        mismatches["input_voxel_size_m"] = {
            "checkpoint": metadata.get("input_voxel_size_m"),
            "runtime": configured_voxel_size,
        }
    for key, value in scene_tokenizer_contract.items():
        if key in metadata and not _equal_number(metadata[key], value):
            mismatches[key] = {"checkpoint": metadata[key], "runtime": value}
    if (
        "global_scene_residual" in metadata
        and metadata.get("global_scene_residual") != residual_contract
    ):
        mismatches["global_scene_residual"] = {
            "checkpoint": metadata.get("global_scene_residual"),
            "runtime": residual_contract,
        }
    if (
        "signed_x_scene_residual" in metadata
        and metadata.get("signed_x_scene_residual") != signed_x_contract
    ):
        mismatches["signed_x_scene_residual"] = {
            "checkpoint": metadata.get("signed_x_scene_residual"),
            "runtime": signed_x_contract,
        }
    if dense_settings.enabled:
        if metadata.get("dense_alignment") != dense_contract:
            mismatches["dense_alignment"] = {
                "checkpoint": metadata.get("dense_alignment"),
                "runtime": dense_contract,
            }
        if (
            isinstance(dense_alignment_parameter_count, bool)
            or not isinstance(dense_alignment_parameter_count, int)
            or dense_alignment_parameter_count < 1
        ):
            raise ValueError("Enabled dense alignment requires a positive runtime parameter count")
        if metadata.get("dense_alignment_parameter_count") != (dense_alignment_parameter_count):
            mismatches["dense_alignment_parameter_count"] = {
                "checkpoint": metadata.get("dense_alignment_parameter_count"),
                "runtime": dense_alignment_parameter_count,
            }
        if metadata.get("dense_alignment_initial_state_sha256") != (
            dense_settings.expected_initial_state_sha256
        ):
            mismatches["dense_alignment_initial_state_sha256"] = {
                "checkpoint": metadata.get("dense_alignment_initial_state_sha256"),
                "runtime": dense_settings.expected_initial_state_sha256,
            }
        dense_state_hash = metadata.get("dense_alignment_state_sha256")
        if (
            not isinstance(dense_state_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", dense_state_hash) is None
        ):
            mismatches["dense_alignment_state_sha256"] = {
                "checkpoint": dense_state_hash,
                "runtime": "<required-lowercase-sha256>",
            }
        if metadata.get("all_voxels_transformed") is not True:
            mismatches["all_voxels_transformed"] = {
                "checkpoint": metadata.get("all_voxels_transformed"),
                "runtime": True,
            }
        if metadata.get("question_dependent_scene_processing") is not False:
            mismatches["question_dependent_scene_processing"] = {
                "checkpoint": metadata.get("question_dependent_scene_processing"),
                "runtime": False,
            }
    signed_x_base_provenance_mismatch = None
    if signed_x_contract["enabled"]:
        if not residual_contract["enabled"]:
            mismatches["signed_x_global_scene_residual_base"] = {
                "checkpoint": metadata.get("global_scene_residual"),
                "runtime": "enabled centered V18 global residual required",
            }
        elif residual_settings.architecture_version != ZERO_SPATIAL_MEAN_CONTENT_GATE_V1:
            mismatches["signed_x_global_scene_residual_base"] = {
                "checkpoint": metadata.get("global_scene_residual"),
                "runtime": ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
            }
        if metadata.get("signed_x_scene_residual_initial_state_sha256") != (
            signed_x_settings.expected_initial_state_sha256
        ):
            mismatches["signed_x_scene_residual_initial_state_sha256"] = {
                "checkpoint": metadata.get("signed_x_scene_residual_initial_state_sha256"),
                "runtime": signed_x_settings.expected_initial_state_sha256,
            }
        signed_x_base_provenance_mismatch = _signed_x_frozen_base_provenance_mismatch(metadata)
        if signed_x_base_provenance_mismatch is not None:
            mismatches["signed_x_frozen_v18_base_provenance"] = signed_x_base_provenance_mismatch
    if residual_contract["enabled"]:
        if metadata.get("global_scene_residual_initial_state_sha256") != residual_contract.get(
            "expected_initial_state_sha256"
        ):
            mismatches["global_scene_residual_initial_state_sha256"] = {
                "checkpoint": metadata.get("global_scene_residual_initial_state_sha256"),
                "runtime": residual_contract.get("expected_initial_state_sha256"),
            }
        equivalence = metadata.get("global_scene_residual_zero_output_equivalence")
        allow_loaded_signed_x_base = bool(
            signed_x_contract["enabled"]
            and equivalence is None
            and signed_x_base_provenance_mismatch is None
        )
        if not allow_loaded_signed_x_base and (
            not isinstance(equivalence, dict) or equivalence.get("verified") is not True
        ):
            mismatches["global_scene_residual_zero_output_equivalence"] = {
                "checkpoint": equivalence,
                "runtime": (
                    "verified update-0 equivalence, or explicit frozen loaded V18 base "
                    "provenance for signed-X"
                ),
            }
        if metadata.get("question_dependent_scene_processing") is not False:
            mismatches["question_dependent_scene_processing"] = {
                "checkpoint": metadata.get("question_dependent_scene_processing"),
                "runtime": False,
            }
    prefix_layout_mismatch = scene_prefix_after_bos_contract_mismatch(
        metadata,
        scene_prefix_after_bos_setting(config),
    )
    if prefix_layout_mismatch is not None:
        mismatches["scene_prefix_after_bos"] = prefix_layout_mismatch
    boundary_mismatch = scene_boundary_contract_mismatch(
        metadata,
        scene_boundary_mode_setting(config),
        native_gemma4_image_contract_setting(config),
    )
    if boundary_mismatch is not None:
        mismatches["scene_boundary_mode"] = boundary_mismatch
    configured_lora = lora_banks_settings(config)
    configured_lora_optimizer = lora_banks_optimizer_settings(config, configured_lora)
    if lora_parameter_counts is None:
        if configured_lora.legacy_single_bank:
            lora_parameter_counts = (
                {configured_lora.banks[0].name: lora_parameter_count}
                if configured_lora.banks
                else {}
            )
        elif configured_lora.enabled:
            raise ValueError("Named LoRA runtime validation requires per-bank parameter counts")
        else:
            lora_parameter_counts = {}
    configured_lora_contract = lora_banks_checkpoint_contract(
        configured_lora,
        configured_lora_optimizer,
        lora_parameter_counts,
    )
    lora_mismatch = lora_checkpoint_contract_mismatch(metadata, configured_lora_contract)
    if lora_mismatch is not None:
        mismatches["lora"] = lora_mismatch
    configured_freeze_scene = config.get("training", {}).get("freeze_scene_adapter", False)
    if not isinstance(configured_freeze_scene, bool):
        raise TypeError("training.freeze_scene_adapter must be a boolean")
    checkpoint_freeze_scene = metadata.get("freeze_scene_adapter", False)
    if checkpoint_freeze_scene != configured_freeze_scene:
        mismatches["freeze_scene_adapter"] = {
            "checkpoint": checkpoint_freeze_scene,
            "runtime": configured_freeze_scene,
        }
    if configured_freeze_scene:
        frozen_scene_hash = metadata.get("frozen_scene_state_sha256")
        if (
            not isinstance(frozen_scene_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", frozen_scene_hash) is None
        ):
            mismatches["frozen_scene_state_sha256"] = {
                "checkpoint": frozen_scene_hash,
                "runtime": "<required-lowercase-sha256>",
            }
    if mismatches:
        raise ValueError(f"Checkpoint is incompatible with runtime architecture: {mismatches}")
    warnings: list[str] = []
    runtime_hash = config_hash(config)
    if metadata["config_hash"] != runtime_hash:
        warnings.append(
            "Full config hash differs from training, but all inference architecture fields match: "
            f"checkpoint={metadata['config_hash']} runtime={runtime_hash}"
        )
    return warnings


def construct_scene_tokenizer(
    config: dict[str, Any], semantic_dim: int, language_hidden_dim: int
) -> SceneTokenizer:
    settings = config["scene_encoder"]
    return SceneTokenizer(
        semantic_dim=semantic_dim,
        model_dim=int(settings["model_dim"]),
        language_hidden_dim=language_hidden_dim,
        block_size_m=float(settings["block_size_m"]),
        tokens_per_block=int(settings["tokens_per_block"]),
        global_latents=int(settings["global_latents"]),
        heads=int(settings["heads"]),
        global_layers=int(settings["global_layers"]),
        fourier_bands=int(settings["fourier_bands"]),
        coverage_temperature=float(settings.get("coverage_temperature", 0.35)),
        coverage_scale=float(settings.get("coverage_scale", 1.0)),
        query_identity_scale=float(settings.get("query_identity_scale", 0.5)),
        projection_skip_scale=float(settings.get("projection_skip_scale", 1.0)),
        semantic_skip_scale=float(settings.get("semantic_skip_scale", 1.0)),
        geometry_skip_scale=float(settings.get("geometry_skip_scale", 0.5)),
        block_content_residual_scale=float(settings.get("block_content_residual_scale", 1.0)),
        language_aligned_tail_dim=int(settings.get("language_aligned_tail_dim", 0)),
        native_aligned_coverage_scale=float(settings.get("native_aligned_coverage_scale", 0.0)),
        learned_scene_token_scale=float(settings.get("learned_scene_token_scale", 1.0)),
        learned_scene_token_rms_target=(
            None
            if settings.get("learned_scene_token_rms_target") is None
            else float(settings["learned_scene_token_rms_target"])
        ),
        architecture_version=str(settings["architecture_version"]),
    )


class StaticChatRuntime:
    """Question-independent scene memory with deterministic local generation."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        scene_id: str,
        checkpoint_path: Path,
        checkpoint_metadata: dict[str, Any],
        language: LocalLanguageModel,
        map_data: MapTensorData,
        scene_model: SceneTokenizer,
        dense_aligner: DenseAlignmentResidual | None = None,
        global_scene_residual: GlobalSceneResidual | None = None,
        signed_x_scene_residual: SignedXSceneResidual | None = None,
        composer: ContinuousPrefixComposer,
        grounding: QuestionGroundingHead,
        warnings: list[str] | None = None,
        generation_function: GenerationFunction = generate_from_embeddings,
    ) -> None:
        if not _OPAQUE_SCENE_ID.fullmatch(scene_id):
            raise ValueError("scene_id must be opaque and match scene_ followed by six digits")
        self.config = config
        self.scene_id = scene_id
        self.checkpoint_path = checkpoint_path
        self.checkpoint_metadata = dict(checkpoint_metadata)
        self.language = language
        self.map_data = map_data
        self.scene_model = scene_model.eval()
        dense_settings = dense_alignment_settings(config)
        if dense_settings.enabled != (dense_aligner is not None):
            raise ValueError(
                "Runtime dense-aligner construction does not match the configured enabled state"
            )
        self.dense_aligner = None if dense_aligner is None else dense_aligner.eval()
        if self.dense_aligner is not None:
            required_dense_metadata = _DENSE_ALIGNMENT_RUNTIME_REQUIRED_KEYS | {
                "question_dependent_scene_processing"
            }
            missing_dense_metadata = sorted(
                required_dense_metadata - self.checkpoint_metadata.keys()
            )
            if missing_dense_metadata:
                raise ValueError(
                    f"Runtime dense-alignment metadata is incomplete: {missing_dense_metadata}"
                )
            if self.checkpoint_metadata.get("dense_alignment") != dense_settings.contract():
                raise ValueError("Runtime dense-alignment checkpoint contract mismatch")
            if self.checkpoint_metadata.get("dense_alignment_initial_state_sha256") != (
                dense_settings.expected_initial_state_sha256
            ):
                raise ValueError("Runtime dense-alignment initial-state contract mismatch")
            dense_audit = validate_dense_alignment_state(
                self.dense_aligner,
                expected_parameter_count=self.checkpoint_metadata.get(
                    "dense_alignment_parameter_count"
                ),
                context="runtime device initialization",
            )
            expected_dense_hash = self.checkpoint_metadata.get("dense_alignment_state_sha256")
            if dense_audit["state_sha256"] != expected_dense_hash:
                raise ValueError(
                    "Dense-alignment state mismatch or tamper detected during runtime "
                    f"initialization: checkpoint={expected_dense_hash} "
                    f"runtime={dense_audit['state_sha256']}"
                )
            if self.checkpoint_metadata.get("all_voxels_transformed") is not True:
                raise ValueError("Dense-alignment checkpoint must transform all voxels")
            if self.checkpoint_metadata.get("question_dependent_scene_processing") is not False:
                raise ValueError("Dense alignment must remain question-independent")
        self.global_scene_residual = (
            None if global_scene_residual is None else global_scene_residual.eval()
        )
        if self.global_scene_residual is not None:
            _validate_global_scene_residual_state(
                self.global_scene_residual,
                expected_parameter_count=self.checkpoint_metadata.get(
                    "global_scene_residual_parameter_count"
                ),
                context="runtime device initialization",
            )
        self.signed_x_scene_residual = (
            None if signed_x_scene_residual is None else signed_x_scene_residual.eval()
        )
        if self.signed_x_scene_residual is not None:
            if self.global_scene_residual is None:
                raise ValueError("Signed-X scene residual requires a loaded global residual base")
            if self.global_scene_residual.architecture_version != ZERO_SPATIAL_MEAN_CONTENT_GATE_V1:
                raise ValueError(
                    "Signed-X scene residual requires the centered-content V18 global base"
                )
            _validate_signed_x_scene_residual_state(
                self.signed_x_scene_residual,
                expected_parameter_count=self.checkpoint_metadata.get(
                    "signed_x_scene_residual_parameter_count"
                ),
                context="runtime device initialization",
            )
        self.composer = composer.eval()
        configured_layout = scene_prefix_after_bos_setting(config)
        if self.composer.scene_prefix_after_bos != configured_layout:
            raise ValueError(
                "Composer scene-prefix layout does not match runtime config: "
                f"{self.composer.scene_prefix_after_bos} != {configured_layout}"
            )
        configured_boundary_mode = scene_boundary_mode_setting(config)
        if self.composer.scene_boundary_mode != configured_boundary_mode:
            raise ValueError(
                "Composer scene-boundary mode does not match runtime config: "
                f"{self.composer.scene_boundary_mode} != {configured_boundary_mode}"
            )
        native_embeddings = self.language.scene_boundary_embeddings(configured_boundary_mode)
        if native_embeddings is not None:
            self.composer.validate_native_boundary_embeddings(native_embeddings)
        self.grounding = grounding.eval()
        self.warnings = list(warnings or [])
        self._generation_function = generation_function
        self._questions_answered = 0

        started = time.perf_counter()
        with torch.inference_mode():
            self.core_scene_output = self._encode_complete_scene()
            centered_content = (
                None
                if self.signed_x_scene_residual is None
                else frozen_v18_centered_content_values(
                    self.global_scene_residual,
                    self.core_scene_output.scene_tokens,
                )
            )
            self.global_scene_output = apply_global_scene_residual(
                self.core_scene_output, self.global_scene_residual
            )
            self.scene_output = (
                self.global_scene_output
                if self.signed_x_scene_residual is None
                else apply_signed_x_scene_residual(
                    self.global_scene_output,
                    self.signed_x_scene_residual,
                    centered_content,
                )
            )
            model_dtype = next(self.language.model.parameters()).dtype
            lm_scene_tokens = self.scene_output.scene_tokens.to(dtype=model_dtype)
            self.scene_prefix = self.composer.scene_prefix(lm_scene_tokens).detach()
        if self.scene_prefix.shape[1] != int(config["scene_encoder"]["global_latents"]) + 2:
            raise RuntimeError("Scene prefix length does not match configured global latent count")
        if not torch.isfinite(self.scene_prefix).all():
            raise RuntimeError("Scene prefix contains NaN or infinity")
        self.scene_prefix_hash = prefix_sha256(self.scene_prefix)
        self.prefix_build_seconds = time.perf_counter() - started

    @classmethod
    def load(
        cls,
        config: dict[str, Any],
        scene_id: str,
        checkpoint: str | Path = "data/checkpoints/best",
        *,
        audit: FileAccessAudit | None = None,
        local_files_only: bool = True,
        generation_function: GenerationFunction = generate_from_embeddings,
    ) -> StaticChatRuntime:
        if not _OPAQUE_SCENE_ID.fullmatch(scene_id):
            raise ValueError("scene_id must be opaque and match scene_ followed by six digits")
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        checkpoint_path = _guard_runtime_input(checkpoint_path, "adapter checkpoint")
        metadata = _read_checkpoint_metadata(checkpoint_path, audit)
        adapter_path = _guard_runtime_input(
            checkpoint_path / "adapter.safetensors", "adapter parameters"
        )
        if audit is not None:
            audit.record(adapter_path)

        map_path = _guard_runtime_input(
            project_path(config, "maps", scene_id, "voxel_map.npz"), "numeric voxel map"
        )
        if audit is not None:
            audit.record(map_path)
        map_data = load_map_tensors(
            map_path,
            config["scene"]["room_size_m"],
            device="cpu",
            input_voxel_size_m=config["scene_encoder"].get("input_voxel_size_m"),
        )
        if int(metadata.get("semantic_dim", -1)) != map_data.feature_dim:
            raise ValueError(
                "Checkpoint semantic dimension does not match the numeric voxel map: "
                f"{metadata.get('semantic_dim')} != {map_data.feature_dim}"
            )
        dense_aligner = construct_dense_alignment(config, semantic_dim=map_data.feature_dim)
        if dense_aligner is not None:
            dense_initial_audit = validate_dense_alignment_state(
                dense_aligner,
                context="runtime deterministic construction",
            )
            expected_initial_dense_hash = dense_alignment_settings(
                config
            ).expected_initial_state_sha256
            if dense_initial_audit["state_sha256"] != expected_initial_dense_hash:
                raise ValueError(
                    "Dense-alignment deterministic initial-state mismatch: "
                    f"configured={expected_initial_dense_hash} "
                    f"runtime={dense_initial_audit['state_sha256']}"
                )

        language = load_local_language_model(
            str(config["language"]["model_id"]),
            str(config["language"]["revision"]),
            str(config["language"]["dtype"]),
            freeze=True,
            local_files_only=local_files_only,
            backend=str(config["language"].get("backend", "auto")),
        )
        configured_boundary_mode = scene_boundary_mode_setting(config)
        configured_native_contract = native_gemma4_image_contract_setting(config)
        loaded_native_contract = language.scene_boundary_contract(configured_boundary_mode)
        if loaded_native_contract != configured_native_contract:
            raise ValueError(
                "Loaded language model does not satisfy configured scene-boundary contract: "
                f"loaded={loaded_native_contract} configured={configured_native_contract}"
            )
        configured_lora = lora_banks_settings(config)
        lora_installation = install_lora_banks(language.model, configured_lora)
        checkpoint_backend = metadata.get("language_backend")
        if checkpoint_backend is not None and checkpoint_backend != language.backend_name:
            raise ValueError(
                "Checkpoint language backend does not match the loaded runtime backend: "
                f"{checkpoint_backend} != {language.backend_name}"
            )
        warnings = validate_checkpoint_contract(
            metadata,
            config,
            semantic_dim=map_data.feature_dim,
            language_hidden_dim=language.hidden_size,
            lora_parameter_count=(
                0 if lora_installation is None else lora_installation.parameter_count
            ),
            lora_parameter_counts=(
                {} if lora_installation is None else lora_installation.parameter_counts
            ),
            dense_alignment_parameter_count=(
                0 if dense_aligner is None else dense_aligner.parameter_count
            ),
        )
        scene_model = construct_scene_tokenizer(config, map_data.feature_dim, language.hidden_size)
        global_scene_residual = construct_global_scene_residual(
            config,
            scene_dim=language.hidden_size,
            latent_count=int(config["scene_encoder"]["global_latents"]),
        )
        if global_scene_residual is not None:
            _validate_global_scene_residual_state(
                global_scene_residual,
                expected_parameter_count=metadata.get("global_scene_residual_parameter_count"),
                context="runtime deterministic construction",
            )
        signed_x_scene_residual = construct_signed_x_scene_residual(
            config,
            scene_dim=language.hidden_size,
            latent_count=int(config["scene_encoder"]["global_latents"]),
            content_dim=(0 if global_scene_residual is None else global_scene_residual.width),
        )
        if signed_x_scene_residual is not None:
            if global_scene_residual is None:
                raise ValueError("Signed-X scene residual requires a global residual base")
            if global_scene_residual.architecture_version != ZERO_SPATIAL_MEAN_CONTENT_GATE_V1:
                raise ValueError(
                    "Signed-X scene residual requires the centered-content V18 global base"
                )
            _validate_signed_x_scene_residual_state(
                signed_x_scene_residual,
                expected_parameter_count=metadata.get("signed_x_scene_residual_parameter_count"),
                context="runtime deterministic construction",
            )
            observed_initial_signed_x_hash = module_collection_state_sha256(
                {"signed_x_scene_residual": signed_x_scene_residual}
            )
            expected_initial_signed_x_hash = signed_x_scene_residual_settings(
                config
            ).expected_initial_state_sha256
            if observed_initial_signed_x_hash != expected_initial_signed_x_hash:
                raise ValueError(
                    "Signed-X scene residual deterministic initial-state mismatch: "
                    f"configured={expected_initial_signed_x_hash} "
                    f"runtime={observed_initial_signed_x_hash}"
                )
            if torch.count_nonzero(signed_x_scene_residual.output_projection.weight).item() != 0:
                raise RuntimeError(
                    "Fresh signed-X scene residual does not preserve its loaded V18 base"
                )
        composer = ContinuousPrefixComposer(
            language.hidden_size,
            scene_prefix_after_bos=scene_prefix_after_bos_setting(config),
            bos_token_id=language.bos_token_id,
            scene_boundary_mode=configured_boundary_mode,
            native_boundary_embeddings=language.scene_boundary_embeddings(configured_boundary_mode),
        )
        grounding = QuestionGroundingHead(
            int(config["scene_encoder"]["model_dim"]),
            language.hidden_size,
            int(config["scene_encoder"]["global_latents"]),
            int(config["scene_encoder"]["model_dim"]),
        )
        scene_checkpoint_modules = {
            "scene_model": scene_model,
            "composer": composer,
            "grounding": grounding,
        }
        checkpoint_modules = dict(scene_checkpoint_modules)
        if dense_aligner is not None:
            checkpoint_modules["dense_aligner"] = dense_aligner
        if global_scene_residual is not None:
            checkpoint_modules["global_scene_residual"] = global_scene_residual
        if signed_x_scene_residual is not None:
            checkpoint_modules["signed_x_scene_residual"] = signed_x_scene_residual
        if lora_installation is not None:
            checkpoint_modules.update(lora_installation.state_modules())
        loaded_metadata = load_adapter_checkpoint(
            checkpoint_path,
            checkpoint_modules,
            device="cpu",
            metadata_filename=RUNTIME_METADATA_FILENAME,
        )
        if loaded_metadata != metadata:
            raise RuntimeError("Checkpoint metadata changed while the runtime was loading")
        expected_frozen_scene_hash = metadata.get("frozen_scene_state_sha256")
        if expected_frozen_scene_hash is not None:
            observed_frozen_scene_hash = module_collection_state_sha256(scene_checkpoint_modules)
            if observed_frozen_scene_hash != expected_frozen_scene_hash:
                raise ValueError(
                    "Frozen scene checkpoint state mismatch or tamper detected: "
                    f"checkpoint={expected_frozen_scene_hash} "
                    f"runtime={observed_frozen_scene_hash}"
                )
        if lora_installation is not None:
            validate_lora_banks_checkpoint_state(metadata, lora_installation)
            language.model.requires_grad_(False)
        if dense_aligner is not None:
            dense_loaded_audit = validate_dense_alignment_state(
                dense_aligner,
                expected_parameter_count=metadata.get("dense_alignment_parameter_count"),
                context="runtime checkpoint load",
            )
            expected_dense_state_hash = metadata.get("dense_alignment_state_sha256")
            if dense_loaded_audit["state_sha256"] != expected_dense_state_hash:
                raise ValueError(
                    "Dense-alignment state mismatch or tamper detected: "
                    f"checkpoint={expected_dense_state_hash} "
                    f"runtime={dense_loaded_audit['state_sha256']}"
                )
        observed_residual_hash = None
        if global_scene_residual is not None:
            _validate_global_scene_residual_state(
                global_scene_residual,
                expected_parameter_count=metadata.get("global_scene_residual_parameter_count"),
                context="runtime checkpoint load",
            )
            observed_residual_hash = module_collection_state_sha256(
                {"global_scene_residual": global_scene_residual}
            )
            if observed_residual_hash != metadata.get("global_scene_residual_state_sha256"):
                raise ValueError(
                    "Global scene residual state mismatch or tamper detected: "
                    f"checkpoint={metadata.get('global_scene_residual_state_sha256')} "
                    f"runtime={observed_residual_hash}"
                )
        if signed_x_scene_residual is not None:
            if observed_residual_hash != metadata.get("frozen_global_scene_residual_state_sha256"):
                raise ValueError(
                    "Signed-X frozen global residual base mismatch or tamper detected: "
                    f"checkpoint={metadata.get('frozen_global_scene_residual_state_sha256')} "
                    f"runtime={observed_residual_hash}"
                )
            _validate_signed_x_scene_residual_state(
                signed_x_scene_residual,
                expected_parameter_count=metadata.get("signed_x_scene_residual_parameter_count"),
                context="runtime checkpoint load",
            )
            observed_signed_x_hash = module_collection_state_sha256(
                {"signed_x_scene_residual": signed_x_scene_residual}
            )
            if observed_signed_x_hash != metadata.get("signed_x_scene_residual_state_sha256"):
                raise ValueError(
                    "Signed-X scene residual state mismatch or tamper detected: "
                    f"checkpoint={metadata.get('signed_x_scene_residual_state_sha256')} "
                    f"runtime={observed_signed_x_hash}"
                )
        device = language.device
        scene_model = scene_model.to(device)
        if dense_aligner is not None:
            dense_aligner = dense_aligner.to(device)
        if global_scene_residual is not None:
            global_scene_residual = global_scene_residual.to(device)
        if signed_x_scene_residual is not None:
            signed_x_scene_residual = signed_x_scene_residual.to(device)
        composer = composer.to(device)
        grounding = grounding.to(device)
        map_data = map_data.to(device)
        return cls(
            config=config,
            scene_id=scene_id,
            checkpoint_path=checkpoint_path,
            checkpoint_metadata=metadata,
            language=language,
            map_data=map_data,
            scene_model=scene_model,
            dense_aligner=dense_aligner,
            global_scene_residual=global_scene_residual,
            signed_x_scene_residual=signed_x_scene_residual,
            composer=composer,
            grounding=grounding,
            warnings=warnings,
            generation_function=generation_function,
        )

    def _encode_complete_scene(self) -> SceneTokenizerOutput:
        data = self.map_data
        semantic = data.semantic
        if self.dense_aligner is not None:
            semantic = self.dense_aligner(semantic)
            if semantic.shape != data.semantic.shape:
                raise RuntimeError(
                    "Dense alignment changed the complete semantic-map shape: "
                    f"input={tuple(data.semantic.shape)} output={tuple(semantic.shape)}"
                )
            if not torch.isfinite(semantic).all():
                raise RuntimeError("Dense alignment produced NaN or infinity")
            self.dense_alignment_transformed_voxels = int(semantic.shape[0])
        else:
            self.dense_alignment_transformed_voxels = 0
        output = self.scene_model(
            semantic,
            data.xyz,
            data.rgb,
            data.normal,
            data.confidence,
            data.observation_count,
            data.room_min,
            data.room_max,
        )
        processed = int(output.audit["processed_voxels"].detach().cpu().item())
        if processed != data.voxel_count:
            raise RuntimeError(
                f"Incomplete full-scene encoding: processed {processed}/{data.voxel_count} voxels"
            )
        return output

    @property
    def questions_answered(self) -> int:
        return self._questions_answered

    def current_prefix_hash(self) -> str:
        return prefix_sha256(self.scene_prefix)

    def assert_prefix_unchanged(self) -> None:
        current = self.current_prefix_hash()
        if current != self.scene_prefix_hash:
            raise RuntimeError(
                "Question-independent scene prefix changed unexpectedly: "
                f"{self.scene_prefix_hash} != {current}"
            )

    def startup_summary(self) -> dict[str, Any]:
        summary = {
            "phase": "scene_ready",
            "scene_id": self.scene_id,
            "prefix_hash": self.scene_prefix_hash,
            "prefix_shape": list(self.scene_prefix.shape),
            "scene_latents": int(self.scene_output.scene_tokens.shape[1]),
            "language_hidden_dim": int(self.scene_prefix.shape[-1]),
            "language_backend": self.language.backend_name,
            "scene_prefix_after_bos": scene_prefix_after_bos_setting(self.config),
            "scene_boundary_mode": scene_boundary_mode_setting(self.config),
            "gemma4_native_image_contract": native_gemma4_image_contract_setting(self.config),
            "source_voxels": self.map_data.source_voxel_count,
            "processed_voxels": self.map_data.voxel_count,
            "occupied_blocks": int(self.scene_output.audit["voxel_counts"].numel()),
            "device": str(self.language.device),
            "prefix_build_seconds": self.prefix_build_seconds,
            "question_dependent_scene_processing": False,
            "global_scene_residual": self.checkpoint_metadata.get(
                "global_scene_residual", {"schema_version": 1, "enabled": False}
            ),
            "global_scene_residual_state_sha256": self.checkpoint_metadata.get(
                "global_scene_residual_state_sha256"
            ),
            "signed_x_scene_residual": self.checkpoint_metadata.get(
                "signed_x_scene_residual", {"schema_version": 1, "enabled": False}
            ),
            "signed_x_scene_residual_initial_state_sha256": self.checkpoint_metadata.get(
                "signed_x_scene_residual_initial_state_sha256"
            ),
            "signed_x_scene_residual_state_sha256": self.checkpoint_metadata.get(
                "signed_x_scene_residual_state_sha256"
            ),
            "frozen_global_scene_residual_state_sha256": self.checkpoint_metadata.get(
                "frozen_global_scene_residual_state_sha256"
            ),
            "checkpoint": str(self.checkpoint_path),
            "warnings": self.warnings,
        }
        if "lora" in self.checkpoint_metadata:
            summary["lora"] = self.checkpoint_metadata["lora"]
        if self.dense_aligner is not None:
            summary.update(
                {
                    "dense_alignment": self.checkpoint_metadata["dense_alignment"],
                    "dense_alignment_parameter_count": self.dense_aligner.parameter_count,
                    "dense_alignment_initial_state_sha256": self.checkpoint_metadata[
                        "dense_alignment_initial_state_sha256"
                    ],
                    "dense_alignment_state_sha256": self.dense_aligner.state_sha256(),
                    "dense_alignment_transformed_voxels": (self.dense_alignment_transformed_voxels),
                    "all_voxels_transformed": (
                        self.dense_alignment_transformed_voxels == self.map_data.voxel_count
                    ),
                }
            )
        return summary

    def _question_token_count(self, question: str) -> int:
        encoded = self.language.tokenizer(
            question,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        return int(input_ids.shape[-1])

    def _eos_token_ids(self) -> int | list[int] | None:
        values: list[int] = []
        tokenizer_eos = getattr(self.language.tokenizer, "eos_token_id", None)
        model_eos = getattr(
            getattr(self.language.model, "generation_config", None), "eos_token_id", None
        )
        for candidate in (tokenizer_eos, model_eos):
            if candidate is None:
                continue
            if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
                values.extend(int(value) for value in candidate)
            else:
                values.append(int(candidate))
        unique = sorted(set(values))
        if not unique:
            return None
        return unique[0] if len(unique) == 1 else unique

    @torch.inference_mode()
    def _predict_grounding(
        self, prompt_embeddings: torch.Tensor
    ) -> tuple[tuple[float, float, float], float, float]:
        normalized = self.grounding(
            self.scene_output.native_latents.float(), prompt_embeddings.float().mean(dim=1)
        )
        xyz = denormalize_xyz(normalized, self.map_data.room_min, self.map_data.room_max)[0]
        distances = torch.linalg.vector_norm(self.map_data.xyz - xyz, dim=-1)
        nearest_index = int(torch.argmin(distances).item())
        support_distance = float(distances[nearest_index].detach().cpu())
        map_confidence = float(
            self.map_data.confidence[nearest_index].detach().float().cpu().clamp(0.0, 1.0)
        )
        distance_scale = max(float(self.config["scene_encoder"]["block_size_m"]), 1e-6)
        confidence = max(
            0.0,
            min(1.0, map_confidence * math.exp(-support_distance / distance_scale)),
        )
        coordinates = tuple(float(value) for value in xyz.detach().float().cpu().tolist())
        return coordinates, confidence, support_distance

    def answer(self, question: str) -> ChatAnswer:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")
        max_question_tokens = int(self.config["language"]["max_question_tokens"])
        question_tokens = self._question_token_count(question)
        if question_tokens > max_question_tokens:
            raise ValueError(
                f"Question has {question_tokens} tokens; maximum is {max_question_tokens}"
            )
        self.assert_prefix_unchanged()
        started = time.perf_counter()
        prompt_ids = prompt_token_ids(
            self.language.tokenizer,
            str(self.config["language"]["system_prompt"]),
            question,
            self.language.device,
        )
        embedding_layer = self.language.model.get_input_embeddings()
        with torch.inference_mode():
            grounding_question_ids = question_token_ids(
                self.language.tokenizer, question, self.language.device
            )
            grounding_question_embeddings = embedding_layer(grounding_question_ids)
            generated = self.language.generate_from_scene_prefix(
                self.scene_prefix,
                prompt_ids,
                max_new_tokens=int(self.config["language"]["max_answer_tokens"]),
                eos_token_ids=self._eos_token_ids(),
                scene_prefix_after_bos=scene_prefix_after_bos_setting(self.config),
                scene_boundary_mode=scene_boundary_mode_setting(self.config),
                fallback=self._generation_function,
            )
            grounding_xyz, grounding_confidence, support_distance = self._predict_grounding(
                grounding_question_embeddings
            )
        decoded = self.language.tokenizer.decode(
            generated[0].detach().cpu().tolist(), skip_special_tokens=True
        ).strip()
        self.assert_prefix_unchanged()
        self._questions_answered += 1
        return ChatAnswer(
            question=question,
            answer=decoded or "unknown",
            grounding_xyz_m=grounding_xyz,
            grounding_confidence=grounding_confidence,
            grounding_support_distance_m=support_distance,
            prefix_hash=self.scene_prefix_hash,
            generated_tokens=int(generated.shape[-1]),
            elapsed_seconds=time.perf_counter() - started,
        )


__all__ = [
    "ChatAnswer",
    "StaticChatRuntime",
    "construct_scene_tokenizer",
    "validate_checkpoint_contract",
]
