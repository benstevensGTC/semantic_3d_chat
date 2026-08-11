from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from semantic_3d_chat.language.lora import tensor_state_sha256

RUNTIME_METADATA_FILENAME = "runtime_metadata.json"
TRAINING_METADATA_FILENAME = "metadata.json"

# This is intentionally an allowlist.  In particular, training history, QA
# selection/gate details, scene splits, calibration reports, optimizer
# settings, and question IDs must never enter the file opened by chat.
RUNTIME_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "semantic_dim",
        "language_hidden_dim",
        "language_model_id",
        "language_revision",
        "language_backend",
        "scene_latents",
        "scene_model_dim",
        "scene_encoder_architecture_version",
        "input_voxel_size_m",
        "config_hash",
        "scene_prefix_after_bos",
        "scene_boundary_mode",
        "gemma4_native_image_contract",
        "language_aligned_tail_dim",
        "native_aligned_coverage_scale",
        "learned_scene_token_scale",
        "learned_scene_token_rms_target",
        "global_scene_residual",
        "global_scene_residual_parameter_count",
        "global_scene_residual_initial_state_sha256",
        "global_scene_residual_state_sha256",
        "global_scene_residual_zero_output_equivalence",
        "signed_x_scene_residual",
        "signed_x_scene_residual_parameter_count",
        "signed_x_scene_residual_initial_state_sha256",
        "signed_x_scene_residual_state_sha256",
        "signed_x_scene_residual_zero_output_equivalence",
        "frozen_global_scene_residual_state_sha256",
        "frozen_signed_x_scene_residual_state_sha256",
        "initialization_provenance",
        "question_dependent_scene_processing",
        "dense_alignment",
        "dense_alignment_parameter_count",
        "dense_alignment_initial_state_sha256",
        "dense_alignment_state_sha256",
        "all_voxels_transformed",
        "freeze_scene_adapter",
        "frozen_scene_state_sha256",
        "lora",
        "lora_wrapped_modules",
        "lora_trainable_parameter_counts",
        "lora_trainable_parameter_count",
        "lora_state_sha256",
        "lora_bank_wrapped_modules",
        "lora_bank_parameter_counts",
        "lora_bank_state_sha256",
        "lora_parameter_count",
    }
)


def _minimal_equivalence(
    value: object,
    *,
    fields: Sequence[str],
) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return {field: deepcopy(value[field]) for field in fields if field in value}


def _runtime_signed_x_provenance(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Build a stage-independent signed-X attestation from current hashes.

    Training provenance changes shape whenever a new frozen stage is added.
    Chat needs none of that history: it needs only proof that the current
    global base and signed-X state are the exact hash-bound states, and that
    the signed-X branch originated from the verified zero-output transition.
    """

    global_hash = metadata.get("global_scene_residual_state_sha256")
    signed_hash = metadata.get("signed_x_scene_residual_state_sha256")
    frozen_global_hash = metadata.get("frozen_global_scene_residual_state_sha256")
    frozen_signed_hash = metadata.get("frozen_signed_x_scene_residual_state_sha256")
    signed_equivalence = metadata.get("signed_x_scene_residual_zero_output_equivalence")
    equivalence_verified = bool(
        isinstance(signed_equivalence, Mapping)
        and signed_equivalence.get("verified") is True
        and signed_equivalence.get("base") == "loaded_frozen_global_scene_residual"
        and signed_equivalence.get("question_dependent_scene_processing") is False
        and signed_equivalence.get("all_scene_slots_accounted") is True
    )
    return {
        "schema_version": 1,
        "source_global_scene_residual_state_sha256": global_hash,
        "source_signed_x_scene_residual_state_sha256": signed_hash,
        "global_scene_residual_frozen": frozen_global_hash == global_hash,
        # A later dense/LoRA stage records the frozen signed-X hash.  The
        # current state hash remains independently checked for earlier stages.
        "signed_x_scene_residual_frozen": (
            frozen_signed_hash == signed_hash if frozen_signed_hash is not None else None
        ),
        "signed_x_scene_residual_initial_state_sha256": metadata.get(
            "signed_x_scene_residual_initial_state_sha256"
        ),
        "signed_x_zero_output_transition_verified": equivalence_verified,
        "question_dependent_scene_processing": metadata.get(
            "question_dependent_scene_processing"
        ),
    }


def runtime_checkpoint_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return the allowlisted metadata payload that chat is permitted to open."""

    result = {
        key: deepcopy(metadata[key])
        for key in RUNTIME_METADATA_FIELDS
        if key in metadata and key != "initialization_provenance"
    }
    if "global_scene_residual_zero_output_equivalence" in result:
        result["global_scene_residual_zero_output_equivalence"] = _minimal_equivalence(
            result["global_scene_residual_zero_output_equivalence"],
            fields=("verified", "question_dependent_scene_processing"),
        )
    signed_contract = result.get("signed_x_scene_residual")
    signed_enabled = bool(
        isinstance(signed_contract, Mapping) and signed_contract.get("enabled") is True
    )
    if signed_enabled:
        result["signed_x_scene_residual_zero_output_equivalence"] = _minimal_equivalence(
            metadata.get("signed_x_scene_residual_zero_output_equivalence"),
            fields=(
                "verified",
                "base",
                "question_dependent_scene_processing",
                "all_scene_slots_accounted",
            ),
        )
        result["initialization_provenance"] = _runtime_signed_x_provenance(metadata)
    else:
        result.pop("signed_x_scene_residual_zero_output_equivalence", None)
    dense_contract = result.get("dense_alignment")
    dense_enabled = bool(
        isinstance(dense_contract, Mapping) and dense_contract.get("enabled") is True
    )
    if not dense_enabled:
        for field in (
            "dense_alignment",
            "dense_alignment_parameter_count",
            "dense_alignment_initial_state_sha256",
            "dense_alignment_state_sha256",
            "all_voxels_transformed",
        ):
            result.pop(field, None)
    unknown = set(result) - RUNTIME_METADATA_FIELDS
    if unknown:
        raise RuntimeError(f"Runtime metadata sanitizer emitted unknown fields: {sorted(unknown)}")
    return result


def validate_runtime_checkpoint_metadata(metadata: Mapping[str, Any]) -> None:
    """Reject a runtime sidecar containing any training-only top-level field."""

    unknown = sorted(set(metadata) - RUNTIME_METADATA_FIELDS)
    if unknown:
        raise ValueError(f"Runtime checkpoint metadata contains forbidden fields: {unknown}")


def _atomic_json(destination: Path, filename: str, payload: Mapping[str, Any]) -> None:
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f"{filename}.", suffix=".tmp", dir=destination
    )
    try:
        with os.fdopen(temporary_fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination / filename)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def module_collection_state_sha256(modules: dict[str, nn.Module]) -> str:
    """Hash a named module collection without including optimizer state."""

    state = {
        f"{module_name}.{name}": value
        for module_name, module in modules.items()
        for name, value in module.state_dict().items()
    }
    return tensor_state_sha256(state)


def save_adapter_checkpoint(
    directory: str | Path,
    modules: dict[str, nn.Module],
    metadata: dict[str, Any],
) -> Path:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for module_name, module in modules.items():
        for name, value in module.state_dict().items():
            tensors[f"{module_name}.{name}"] = value.detach().cpu().contiguous()
    save_file(tensors, destination / "adapter.safetensors")
    _atomic_json(destination, TRAINING_METADATA_FILENAME, metadata)
    runtime_metadata = runtime_checkpoint_metadata(metadata)
    validate_runtime_checkpoint_metadata(runtime_metadata)
    _atomic_json(destination, RUNTIME_METADATA_FILENAME, runtime_metadata)
    return destination


def load_adapter_checkpoint(
    directory: str | Path,
    modules: dict[str, nn.Module],
    device: str = "cpu",
    *,
    allowed_missing_key_prefixes: Mapping[str, Sequence[str]] | None = None,
    metadata_filename: str = TRAINING_METADATA_FILENAME,
) -> dict[str, Any]:
    """Load a checkpoint without silently dropping or inventing module state.

    ``allowed_missing_key_prefixes`` exists only for explicit architecture
    migrations such as adding an exact-zero residual to an older checkpoint.
    Normal resume and runtime loads remain fully strict.
    """

    if metadata_filename not in {TRAINING_METADATA_FILENAME, RUNTIME_METADATA_FILENAME}:
        raise ValueError(f"Unsupported checkpoint metadata filename: {metadata_filename!r}")
    source = Path(directory)
    tensors = load_file(source / "adapter.safetensors", device=device)
    consumed: set[str] = set()
    allowed_by_module = dict(allowed_missing_key_prefixes or {})
    unknown_modules = sorted(set(allowed_by_module) - set(modules))
    if unknown_modules:
        raise ValueError(f"Missing-key allowances reference unknown modules: {unknown_modules}")
    for module_name, module in modules.items():
        prefix = f"{module_name}."
        state = {
            key[len(prefix) :]: value for key, value in tensors.items() if key.startswith(prefix)
        }
        consumed.update(key for key in tensors if key.startswith(prefix))
        allowed_prefixes = tuple(allowed_by_module.get(module_name, ()))
        if not allowed_prefixes:
            module.load_state_dict(state, strict=True)
            continue
        result = module.load_state_dict(state, strict=False)
        unexpected = sorted(result.unexpected_keys)
        forbidden_missing = sorted(
            key
            for key in result.missing_keys
            if not any(key.startswith(allowed) for allowed in allowed_prefixes)
        )
        if unexpected or forbidden_missing:
            raise RuntimeError(
                f"Checkpoint migration mismatch for {module_name!r}: "
                f"unexpected={unexpected} forbidden_missing={forbidden_missing}"
            )
    unconsumed = sorted(set(tensors) - consumed)
    if unconsumed:
        raise RuntimeError(f"Checkpoint contains unconsumed tensor keys: {unconsumed}")
    metadata = json.loads((source / metadata_filename).read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError("Checkpoint metadata must be a JSON object")
    if metadata_filename == RUNTIME_METADATA_FILENAME:
        validate_runtime_checkpoint_metadata(metadata)
    return metadata


def save_optimizer_checkpoint(directory: str | Path, optimizer: torch.optim.Optimizer) -> Path:
    """Atomically save resumable optimizer state beside adapter weights."""

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix="optimizer.", suffix=".tmp", dir=destination
    )
    os.close(temporary_fd)
    try:
        torch.save(optimizer.state_dict(), temporary_name)
        os.replace(temporary_name, destination / "optimizer.pt")
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination / "optimizer.pt"


def load_optimizer_checkpoint(
    directory: str | Path,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device = "cpu",
) -> None:
    """Restore a trusted local optimizer checkpoint for an exact epoch resume."""

    source = Path(directory) / "optimizer.pt"
    if not source.is_file():
        raise FileNotFoundError(f"Resume checkpoint has no optimizer state: {source}")
    state = torch.load(source, map_location=device, weights_only=True)
    optimizer.load_state_dict(state)
