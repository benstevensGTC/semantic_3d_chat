"""Strict two-file checkpoint for magnitude-gated V6 control."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from semantic_3d_chat.scene_encoder.question_control_v6 import (
    MagnitudeGatedTeacherBasisFullSceneQuestionControlV6,
)
from semantic_3d_chat.training.train_question_control_v56 import (
    _finite_float_state,
    _safe_output_path,
    _sha256_file,
    _validate_hash,
    _write_json,
)


def load_unsealed_v6_checkpoint_for_training_gate(
    checkpoint_path: str | Path,
    *,
    hidden_size: int,
) -> MagnitudeGatedTeacherBasisFullSceneQuestionControlV6:
    """Private training-only loader for a staged, explicitly ungated V6."""

    root = Path(checkpoint_path).resolve()
    metadata_path = root / "runtime_metadata.json"
    weights = root / "control.safetensors"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("architecture")
        != "magnitude_gated_teacher_basis_full_scene_control_v6"
        or metadata.get("saved_runtime_training_gate_required") is not True
        or metadata.get("saved_runtime_training_gate_passed") is not False
        or metadata.get("saved_runtime_training_gate_attestation_sha256") is not None
        or metadata.get("weights_sha256") != _sha256_file(weights)
        or metadata.get("hidden_size") != hidden_size
    ):
        raise ValueError("Staged V6 is not an authenticated ungated checkpoint")
    state = load_file(str(weights), device="cpu")
    module = MagnitudeGatedTeacherBasisFullSceneQuestionControlV6(
        hidden_size,
        state["output_basis"],
        control_tokens=int(metadata["control_tokens"]),
        expected_environment_latents=int(metadata["expected_environment_latents"]),
        moment_count=int(metadata["moment_count"]),
        interaction_dim=int(metadata["interaction_dim"]),
        trunk_dim=int(metadata["trunk_dim"]),
        maximum_control_rms=float(metadata["maximum_control_rms"]),
        initial_control_rms=float(metadata["initial_control_rms"]),
        activation_rms_threshold=float(metadata["activation_rms_threshold"]),
    )
    module.load_state_dict(state, strict=True)
    if v6_value_state_sha256(module) != metadata[
        "source_v65_training_fit_state_sha256"
    ]:
        raise ValueError("Staged V6 training-fit state changed")
    return module.eval()


def v6_value_state_sha256(
    control: MagnitudeGatedTeacherBasisFullSceneQuestionControlV6,
) -> str:
    """Hash the complete numeric V3/V65 value state."""

    digest = hashlib.sha256()
    for name, raw in control.state_dict().items():
        value = raw.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def build_v6_runtime_metadata(
    control: MagnitudeGatedTeacherBasisFullSceneQuestionControlV6,
    *,
    weights_sha256: str,
    base_checkpoint_sha256: str,
    base_runtime_config_sha256: str,
    source_v65_training_fit_state_sha256: str,
    source_v65_value_state_sha256: str,
    saved_runtime_training_gate_passed: bool = False,
    saved_runtime_training_gate_attestation_sha256: str | None = None,
) -> dict[str, Any]:
    if type(saved_runtime_training_gate_passed) is not bool:
        raise TypeError("V6 saved-runtime gate status must be boolean")
    attestation = (
        _validate_hash(
            saved_runtime_training_gate_attestation_sha256,
            "V6 saved-runtime training gate attestation",
        )
        if saved_runtime_training_gate_passed
        else None
    )
    if not saved_runtime_training_gate_passed and saved_runtime_training_gate_attestation_sha256 is not None:
        raise ValueError("Ungated V6 metadata cannot carry a gate attestation")
    return {
        "schema_version": 6,
        "architecture": "magnitude_gated_teacher_basis_full_scene_control_v6",
        "hidden_size": control.hidden_size,
        "control_tokens": control.control_token_count,
        "expected_environment_latents": control.expected_environment_latents,
        "moment_count": control.moment_count,
        "interaction_dim": control.interaction_dim,
        "trunk_dim": control.trunk_dim,
        "output_basis_rank": control.output_basis_rank,
        "maximum_control_rms": control.maximum_control_rms,
        "initial_control_rms": control.initial_control_rms,
        "activation_rms_threshold": control.activation_rms_threshold,
        "activation_rms_aggregation": "maximum_over_control_tokens",
        "weights_sha256": _validate_hash(weights_sha256, "V6 control weights"),
        "base_checkpoint_sha256": _validate_hash(
            base_checkpoint_sha256, "V6 base checkpoint"
        ),
        "base_runtime_config_sha256": _validate_hash(
            base_runtime_config_sha256, "V6 runtime config"
        ),
        "source_v65_training_fit_state_sha256": _validate_hash(
            source_v65_training_fit_state_sha256,
            "V6 source V65 training-fit state",
        ),
        "source_v65_value_state_sha256": _validate_hash(
            source_v65_value_state_sha256, "V6 source V65 value state"
        ),
        "magnitude_gated_continuous_control": True,
        "exact_no_control_below_threshold": True,
        "question_dependent_scene_retrieval": False,
        "complete_scene_prefix_required": True,
        "environmental_text_inputs": [],
        "fixed_global_scene_moments": True,
        "boundary_tokens_excluded_from_scene_signature": True,
        "softmax_scene_attention_used": False,
        "control_values_scene_question_bilinear": True,
        "gate_scene_question_conditioned": True,
        "route_is_environmental_retrieval": False,
        "saved_runtime_training_gate_required": True,
        "saved_runtime_training_gate_passed": saved_runtime_training_gate_passed,
        "saved_runtime_training_gate_attestation_sha256": attestation,
    }


def _state_exact(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> bool:
    return set(left) == set(right) and all(
        torch.equal(left[name], right[name]) for name in left
    )


def save_v6_control_checkpoint(
    checkpoint_path: str | Path,
    *,
    control: MagnitudeGatedTeacherBasisFullSceneQuestionControlV6,
    base_checkpoint_sha256: str,
    base_runtime_config_sha256: str,
    expected_training_fit_state_sha256: str,
    saved_runtime_training_gate_passed: bool = False,
    saved_runtime_training_gate_attestation_sha256: str | None = None,
) -> dict[str, str]:
    if type(control) is not MagnitudeGatedTeacherBasisFullSceneQuestionControlV6:
        raise TypeError("V6 checkpoint requires the exact magnitude-gated architecture")
    observed = v6_value_state_sha256(control)
    if observed != _validate_hash(
        expected_training_fit_state_sha256,
        "expected V65 training-fit state",
    ):
        raise ValueError("V6 numeric V65 training-fit state changed before save")
    destination = _safe_output_path(checkpoint_path, "V6 control checkpoint")
    destination.mkdir(exist_ok=False)
    try:
        weights = destination / "control.safetensors"
        expected = _finite_float_state(control)
        save_file(expected, weights)
        reloaded = load_file(str(weights), device="cpu")
        if not _state_exact(reloaded, expected):
            raise RuntimeError("Saved V6 state failed exact reload")
        metadata = build_v6_runtime_metadata(
            control,
            weights_sha256=_sha256_file(weights),
            base_checkpoint_sha256=base_checkpoint_sha256,
            base_runtime_config_sha256=base_runtime_config_sha256,
            source_v65_training_fit_state_sha256=observed,
            source_v65_value_state_sha256=observed,
            saved_runtime_training_gate_passed=saved_runtime_training_gate_passed,
            saved_runtime_training_gate_attestation_sha256=(
                saved_runtime_training_gate_attestation_sha256
            ),
        )
        _write_json(destination / "runtime_metadata.json", metadata)
        if {item.name for item in destination.iterdir()} != {
            "control.safetensors",
            "runtime_metadata.json",
        }:
            raise RuntimeError("V6 runtime checkpoint inventory is not minimal")
        return {
            "weights_sha256": _sha256_file(weights),
            "runtime_metadata_sha256": _sha256_file(
                destination / "runtime_metadata.json"
            ),
            "source_v65_training_fit_state_sha256": observed,
            "source_v65_value_state_sha256": observed,
        }
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


__all__ = [
    "build_v6_runtime_metadata",
    "load_unsealed_v6_checkpoint_for_training_gate",
    "save_v6_control_checkpoint",
    "v6_value_state_sha256",
]
