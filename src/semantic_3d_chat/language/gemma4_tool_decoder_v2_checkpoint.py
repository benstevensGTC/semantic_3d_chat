"""Create-once, inference-safe checkpoints for Gemma-4 tool-decoder V2."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch
from safetensors.torch import load_file, save_file

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.gemma4_tool_decoder_v2_evaluation import (
    promotion_gate_results_v2,
)
from semantic_3d_chat.language.gemma4_tool_decoder_v2 import (
    HIDDEN_SIZE,
    LORA_PARAMETER_COUNT,
    MODEL_ID,
    MODEL_REVISION,
    PROJECTOR_PARAMETER_COUNT,
    TARGET_PROJECTION,
    TOTAL_TRAINABLE_PARAMETER_COUNT,
    NumericToolContextProjectorV2,
)
from semantic_3d_chat.language.lora import LoRAInstallation, tensor_state_sha256
from semantic_3d_chat.robot.llm_tool_policy import validate_tool_call_text
from semantic_3d_chat.robot.navigation_policy import ACTION_NAMES

ARCHITECTURE: Final[str] = "gemma4_continuous_embodied_tool_decoder_v2"
TRAINING_STATUS: Final[str] = "supervised_continuous_gemma4_tool_decoder_v2"
WEIGHTS_FILENAME: Final[str] = "tool_decoder.safetensors"
METADATA_FILENAME: Final[str] = "runtime_metadata.json"
_FILES: Final[frozenset[str]] = frozenset({WEIGHTS_FILENAME, METADATA_FILENAME})
_BLOCKED: Final[frozenset[str]] = frozenset({"oracle", "qa", "training", "scorer_only"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_METADATA_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "architecture",
        "training_status",
        "status",
        "model_id",
        "model_revision",
        "base_checkpoint_sha256",
        "preregistration_sha256",
        "cpu_preflight_sha256",
        "training_authorization_sha256",
        "clearance_cache_sha256",
        "prefix_inventory_sha256",
        "target_module",
        "lora_rank",
        "lora_alpha",
        "lora_parameter_count",
        "numeric_projector_parameter_count",
        "total_trainable_parameter_count",
        "lora_state_sha256",
        "numeric_projector_state_sha256",
        "weights_sha256",
        "scene_prefix_tokens",
        "robot_tokens",
        "target_tokens",
        "clearance_tokens",
        "hidden_size",
        "max_new_tokens",
        "tool_vocabulary",
        "task_trained",
        "promotion_gates_passed",
        "saved_runtime_execution_gate_passed",
        "complete_scene_prefix_required",
        "question_independent_static_scene_prefix_required",
        "numeric_robot_tokens_required",
        "continuous_target_tokens_required",
        "numeric_clearance_tokens_required",
        "all_map_voxels_scored_for_target_grounding",
        "environmental_text_inputs",
        "oracle_inputs_at_runtime",
        "collision_interlock_required",
        "runtime_required_files",
    }
)


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return Path(os.path.abspath(rooted))


def _reject_runtime_path(path: Path) -> None:
    if _BLOCKED & {part.casefold() for part in path.parts}:
        raise ValueError("V2 runtime checkpoint paths cannot enter blocked data trees")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError("V2 runtime checkpoint paths cannot contain symbolic links")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate V2 runtime metadata field: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    if not isinstance(value, dict) or set(value) != set(_METADATA_FIELDS):
        raise ValueError("V2 runtime metadata fields changed")
    return value


def _state_tensors(
    installation: LoRAInstallation,
    projector: NumericToolContextProjectorV2,
) -> dict[str, torch.Tensor]:
    if installation.target_names != (TARGET_PROJECTION,) or len(installation.adapters) != 1:
        raise ValueError("V2 checkpoint requires the exact one-module LoRA installation")
    if installation.parameter_count != LORA_PARAMETER_COUNT:
        raise ValueError("V2 checkpoint LoRA parameter count changed")
    if projector.trainable_parameter_count != PROJECTOR_PARAMETER_COUNT:
        raise ValueError("V2 checkpoint projector parameter count changed")
    installation.validate_state()
    tensors = {
        f"lora.{name}": value.detach().cpu().contiguous()
        for name, value in installation.state_module.state_dict().items()
    }
    tensors.update(
        {
            f"numeric_projector.{name}": value.detach().cpu().contiguous()
            for name, value in projector.state_dict().items()
        }
    )
    if any(not torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError("V2 checkpoint contains NaN or infinity")
    return tensors


def build_runtime_metadata_v2(
    installation: LoRAInstallation,
    projector: NumericToolContextProjectorV2,
    *,
    weights_sha256: str,
    provenance: Mapping[str, str],
    promoted: bool,
) -> dict[str, Any]:
    required = {
        "base_checkpoint_sha256",
        "preregistration_sha256",
        "cpu_preflight_sha256",
        "training_authorization_sha256",
        "clearance_cache_sha256",
        "prefix_inventory_sha256",
    }
    if set(provenance) != required or any(
        not isinstance(value, str) or _SHA256.fullmatch(value) is None
        for value in provenance.values()
    ):
        raise ValueError("V2 checkpoint provenance must contain exact SHA-256 digests")
    if not isinstance(promoted, bool):
        raise TypeError("V2 checkpoint promoted must be a boolean")
    if not isinstance(weights_sha256, str) or _SHA256.fullmatch(weights_sha256) is None:
        raise ValueError("V2 checkpoint weights SHA-256 is invalid")
    return {
        "schema_version": 2,
        "architecture": ARCHITECTURE,
        "training_status": TRAINING_STATUS,
        "status": "promoted_runtime" if promoted else "staged_runtime_probe_only",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        **dict(provenance),
        "target_module": TARGET_PROJECTION,
        "lora_rank": 4,
        "lora_alpha": 8.0,
        "lora_parameter_count": LORA_PARAMETER_COUNT,
        "numeric_projector_parameter_count": PROJECTOR_PARAMETER_COUNT,
        "total_trainable_parameter_count": TOTAL_TRAINABLE_PARAMETER_COUNT,
        "lora_state_sha256": installation.state_sha256(),
        "numeric_projector_state_sha256": tensor_state_sha256(
            projector.state_dict()
        ),
        "weights_sha256": weights_sha256,
        "scene_prefix_tokens": 258,
        "robot_tokens": 4,
        "target_tokens": 2,
        "clearance_tokens": 2,
        "hidden_size": HIDDEN_SIZE,
        "max_new_tokens": 24,
        "tool_vocabulary": list(ACTION_NAMES),
        "task_trained": True,
        "promotion_gates_passed": promoted,
        "saved_runtime_execution_gate_passed": promoted,
        "complete_scene_prefix_required": True,
        "question_independent_static_scene_prefix_required": True,
        "numeric_robot_tokens_required": True,
        "continuous_target_tokens_required": True,
        "numeric_clearance_tokens_required": True,
        "all_map_voxels_scored_for_target_grounding": True,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
        "collision_interlock_required": True,
        "runtime_required_files": [WEIGHTS_FILENAME, METADATA_FILENAME],
    }


def _write_checkpoint(
    directory: Path,
    installation: LoRAInstallation,
    projector: NumericToolContextProjectorV2,
    *,
    provenance: Mapping[str, str],
    promoted: bool,
) -> dict[str, Any]:
    if any(directory.iterdir()):
        raise FileExistsError("V2 staging checkpoint directory is not empty")
    weights = directory / WEIGHTS_FILENAME
    save_file(_state_tensors(installation, projector), str(weights))
    metadata = build_runtime_metadata_v2(
        installation,
        projector,
        weights_sha256=_sha256(weights),
        provenance=provenance,
        promoted=promoted,
    )
    (directory / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def load_runtime_checkpoint_v2(
    checkpoint: str | Path,
    installation: LoRAInstallation,
    projector: NumericToolContextProjectorV2,
    *,
    expected_provenance: Mapping[str, str],
    require_promoted: bool = True,
) -> dict[str, Any]:
    """Strictly load only V2 LoRA/projector weights into an installed surface."""

    root = _rooted(checkpoint)
    _reject_runtime_path(root)
    if not root.is_dir() or {item.name for item in root.iterdir()} != set(_FILES):
        raise ValueError("V2 runtime checkpoint must contain exactly two files")
    if any(item.is_symlink() or not item.is_file() for item in root.iterdir()):
        raise ValueError("V2 runtime checkpoint entries must be regular files")
    metadata = _strict_json(root / METADATA_FILENAME)
    if (
        metadata.get("schema_version") != 2
        or metadata.get("architecture") != ARCHITECTURE
        or metadata.get("training_status") != TRAINING_STATUS
        or metadata.get("model_id") != MODEL_ID
        or metadata.get("model_revision") != MODEL_REVISION
        or metadata.get("target_module") != TARGET_PROJECTION
        or metadata.get("lora_parameter_count") != LORA_PARAMETER_COUNT
        or metadata.get("numeric_projector_parameter_count") != PROJECTOR_PARAMETER_COUNT
        or metadata.get("total_trainable_parameter_count")
        != TOTAL_TRAINABLE_PARAMETER_COUNT
        or metadata.get("max_new_tokens") != 24
        or metadata.get("tool_vocabulary") != list(ACTION_NAMES)
        or metadata.get("task_trained") is not True
        or metadata.get("environmental_text_inputs") != []
        or metadata.get("oracle_inputs_at_runtime") is not False
        or metadata.get("collision_interlock_required") is not True
        or metadata.get("runtime_required_files") != [WEIGHTS_FILENAME, METADATA_FILENAME]
    ):
        raise ValueError("V2 runtime checkpoint contract is not inference safe")
    if require_promoted and (
        metadata.get("status") != "promoted_runtime"
        or metadata.get("promotion_gates_passed") is not True
        or metadata.get("saved_runtime_execution_gate_passed") is not True
    ):
        raise ValueError("V2 runtime checkpoint has not passed promotion and execution gates")
    if not require_promoted and metadata.get("status") not in {
        "staged_runtime_probe_only",
        "promoted_runtime",
    }:
        raise ValueError("V2 staged runtime checkpoint status is invalid")
    if {name: metadata.get(name) for name in expected_provenance} != dict(
        expected_provenance
    ):
        raise ValueError("V2 runtime checkpoint provenance changed")
    weights_path = root / WEIGHTS_FILENAME
    if _sha256(weights_path) != metadata.get("weights_sha256"):
        raise ValueError("V2 runtime checkpoint weights SHA-256 changed")
    tensors = load_file(str(weights_path), device="cpu")
    expected_keys = {
        *(f"lora.{name}" for name in installation.state_module.state_dict()),
        *(f"numeric_projector.{name}" for name in projector.state_dict()),
    }
    if set(tensors) != expected_keys:
        raise ValueError("V2 runtime checkpoint tensor inventory changed")
    lora_state = {
        name: tensors[f"lora.{name}"] for name in installation.state_module.state_dict()
    }
    projector_state = {
        name: tensors[f"numeric_projector.{name}"] for name in projector.state_dict()
    }
    installation.state_module.load_state_dict(lora_state, strict=True)
    projector.load_state_dict(projector_state, strict=True)
    installation.validate_state()
    if installation.state_sha256() != metadata.get("lora_state_sha256"):
        raise ValueError("V2 runtime LoRA state digest changed")
    observed_projector = tensor_state_sha256(projector.state_dict())
    if observed_projector != metadata.get("numeric_projector_state_sha256"):
        raise ValueError("V2 runtime projector state digest changed")
    return metadata


def _finite_numeric_tree(value: object) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite_numeric_tree(nested) for nested in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_finite_numeric_tree(nested) for nested in value)
    return False


def _finite_numeric_only_tree(value: object) -> bool:
    """Accept nested numeric state, rejecting textual or opaque values."""

    if value is None or isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite_numeric_only_tree(nested) for nested in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_finite_numeric_only_tree(nested) for nested in value)
    return False


def validate_saved_runtime_probe_v2(
    probe: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Require a strict-load, finite generation, validation, and tool execution."""

    required = {
        "saved_checkpoint_loaded",
        "generated_text",
        "numeric_robot_state",
        "tool_execution_attempted",
        "tool_execution_result",
        "collision_interlock_checked",
        "oracle_inputs_loaded",
        "environmental_text_inputs",
    }
    if set(probe) != required:
        raise ValueError("V2 saved-runtime probe fields changed")
    validation = validate_tool_call_text(probe.get("generated_text"), config)
    result = probe.get("tool_execution_result")
    state = probe.get("numeric_robot_state")
    canonical_generation = bool(
        validation.call is not None
        and probe.get("generated_text") == validation.call.canonical_json
    )
    finite_numeric_state = bool(
        isinstance(state, Mapping) and state and _finite_numeric_only_tree(state)
    )
    passed = bool(
        probe.get("saved_checkpoint_loaded") is True
        and validation.call is not None
        and validation.error_code is None
        and canonical_generation
        and validation.call.name in ACTION_NAMES
        and finite_numeric_state
        and probe.get("tool_execution_attempted") is True
        and isinstance(result, Mapping)
        and _finite_numeric_tree(result)
        and result.get("success") is True
        and result.get("collision") is False
        and probe.get("collision_interlock_checked") is True
        and probe.get("oracle_inputs_loaded") is False
        and probe.get("environmental_text_inputs") == []
    )
    return {
        "schema": "semantic_3d_chat.gemma4_tool_decoder_saved_runtime_gate.v2",
        "passed": passed,
        "valid_schema": validation.error_code is None,
        "canonical_generation": canonical_generation,
        "validated_tool": None if validation.call is None else validation.call.name,
        "finite_numeric_state": finite_numeric_state,
        "finite_tool_result": _finite_numeric_tree(result),
        "execution_success": isinstance(result, Mapping)
        and result.get("success") is True,
        "collision_free": isinstance(result, Mapping)
        and result.get("collision") is False,
    }


def publish_runtime_checkpoint_v2(
    destination: str | Path,
    installation: LoRAInstallation,
    projector: NumericToolContextProjectorV2,
    *,
    provenance: Mapping[str, str],
    evaluation: Mapping[str, Any],
    runtime_probe: Callable[[Path], Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish only after behavioral and saved-runtime gates pass."""

    output = _rooted(destination)
    _reject_runtime_path(output)
    if output.exists():
        raise FileExistsError("V2 runtime checkpoint publication is create-once")
    gates = promotion_gate_results_v2(evaluation)
    if gates.get("passed") is not True:
        raise RuntimeError(f"V2 behavioral promotion gates failed: {gates.get('failed')}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _write_checkpoint(
            staging,
            installation,
            projector,
            provenance=provenance,
            promoted=False,
        )
        probe = runtime_probe(staging)
        if not isinstance(probe, Mapping):
            raise TypeError("V2 saved-runtime probe must return a mapping")
        execution_gate = validate_saved_runtime_probe_v2(probe, config)
        if execution_gate["passed"] is not True:
            raise RuntimeError(f"V2 saved-runtime execution gate failed: {execution_gate}")
        # Re-seal only the sanitized metadata after the staged bytes themselves
        # have been strict-loaded and exercised. The tensor file is unchanged.
        metadata = build_runtime_metadata_v2(
            installation,
            projector,
            weights_sha256=_sha256(staging / WEIGHTS_FILENAME),
            provenance=provenance,
            promoted=True,
        )
        (staging / METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        load_runtime_checkpoint_v2(
            staging,
            installation,
            projector,
            expected_provenance=provenance,
            require_promoted=True,
        )
        os.replace(staging, output)
        return {
            "schema": "semantic_3d_chat.gemma4_tool_decoder_publication.v2",
            "published": True,
            "checkpoint": str(output),
            "weights_sha256": _sha256(output / WEIGHTS_FILENAME),
            "runtime_metadata_sha256": _sha256(output / METADATA_FILENAME),
            "behavioral_gates": gates,
            "saved_runtime_execution_gate": execution_gate,
        }
    finally:
        if staging.exists():
            for child in staging.iterdir():
                child.unlink(missing_ok=True)
            staging.rmdir()


__all__ = [
    "ARCHITECTURE",
    "METADATA_FILENAME",
    "TRAINING_STATUS",
    "WEIGHTS_FILENAME",
    "build_runtime_metadata_v2",
    "load_runtime_checkpoint_v2",
    "publish_runtime_checkpoint_v2",
    "validate_saved_runtime_probe_v2",
]
