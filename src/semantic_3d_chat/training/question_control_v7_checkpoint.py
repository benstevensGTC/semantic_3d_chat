"""Strict two-file checkpoint for V7 always-on continuous control."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from semantic_3d_chat.scene_encoder.question_control_v7 import (
    AlwaysOnTeacherBasisFullSceneQuestionControlV7,
)
from semantic_3d_chat.training.train_question_control_v56 import (
    _finite_float_state,
    _safe_output_path,
    _sha256_file,
    _validate_hash,
    _write_json,
)


def v7_value_state_sha256(
    control: AlwaysOnTeacherBasisFullSceneQuestionControlV7,
) -> str:
    digest = hashlib.sha256()
    for name in sorted(control.state_dict()):
        value = control.state_dict()[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def build_v7_runtime_metadata(
    control: AlwaysOnTeacherBasisFullSceneQuestionControlV7,
    *,
    weights_sha256: str,
    base_checkpoint_sha256: str,
    base_runtime_config_sha256: str,
    source_v66_training_fit_state_sha256: str,
    saved_runtime_training_gate_passed: bool = False,
    saved_runtime_training_gate_attestation_sha256: str | None = None,
) -> dict[str, Any]:
    if type(saved_runtime_training_gate_passed) is not bool:
        raise TypeError("V7 saved-runtime gate status must be boolean")
    if saved_runtime_training_gate_passed:
        attestation = _validate_hash(
            saved_runtime_training_gate_attestation_sha256,
            "V7 saved-runtime training gate attestation",
        )
    else:
        if saved_runtime_training_gate_attestation_sha256 is not None:
            raise ValueError("Ungated V7 metadata cannot carry a gate attestation")
        attestation = None
    return {
        "schema_version": 7,
        "architecture": "always_on_teacher_basis_full_scene_control_v7",
        "hidden_size": control.hidden_size,
        "control_tokens": control.control_token_count,
        "expected_environment_latents": control.expected_environment_latents,
        "moment_count": control.moment_count,
        "interaction_dim": control.interaction_dim,
        "trunk_dim": control.trunk_dim,
        "output_basis_rank": control.output_basis_rank,
        "maximum_control_rms": control.maximum_control_rms,
        "initial_control_rms": control.initial_control_rms,
        "weights_sha256": _validate_hash(weights_sha256, "V7 control weights"),
        "base_checkpoint_sha256": _validate_hash(
            base_checkpoint_sha256, "V7 base checkpoint"
        ),
        "base_runtime_config_sha256": _validate_hash(
            base_runtime_config_sha256, "V7 runtime config"
        ),
        "source_v66_training_fit_state_sha256": _validate_hash(
            source_v66_training_fit_state_sha256,
            "V7 source V66 training-fit state",
        ),
        "always_on_continuous_control": True,
        "legacy_route_parameters_ignored": True,
        "question_dependent_scene_retrieval": False,
        "complete_scene_prefix_required": True,
        "environmental_text_inputs": [],
        "training_answers_runtime_loaded": False,
        "answer_class_codebook_runtime_loaded": False,
        "fixed_global_scene_moments": True,
        "boundary_tokens_excluded_from_scene_signature": True,
        "softmax_scene_attention_used": False,
        "control_values_scene_question_bilinear": True,
        "gate_scene_question_conditioned": False,
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


def save_v7_control_checkpoint(
    checkpoint_path: str | Path,
    *,
    control: AlwaysOnTeacherBasisFullSceneQuestionControlV7,
    base_checkpoint_sha256: str,
    base_runtime_config_sha256: str,
    expected_training_fit_state_sha256: str,
    saved_runtime_training_gate_passed: bool = False,
    saved_runtime_training_gate_attestation_sha256: str | None = None,
) -> dict[str, str]:
    if type(control) is not AlwaysOnTeacherBasisFullSceneQuestionControlV7:
        raise TypeError("V7 checkpoint requires the exact always-on architecture")
    observed = v7_value_state_sha256(control)
    if observed != _validate_hash(
        expected_training_fit_state_sha256,
        "expected V66 training-fit state",
    ):
        raise ValueError("V7 numeric V66 training-fit state changed before save")
    destination = _safe_output_path(checkpoint_path, "V7 control checkpoint")
    destination.mkdir(exist_ok=False)
    try:
        weights = destination / "control.safetensors"
        expected = _finite_float_state(control)
        save_file(expected, weights)
        reloaded = load_file(str(weights), device="cpu")
        if not _state_exact(reloaded, expected):
            raise RuntimeError("Saved V7 state failed exact reload")
        metadata = build_v7_runtime_metadata(
            control,
            weights_sha256=_sha256_file(weights),
            base_checkpoint_sha256=base_checkpoint_sha256,
            base_runtime_config_sha256=base_runtime_config_sha256,
            source_v66_training_fit_state_sha256=observed,
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
            raise RuntimeError("V7 runtime checkpoint inventory is not minimal")
        return {
            "weights_sha256": _sha256_file(weights),
            "runtime_metadata_sha256": _sha256_file(
                destination / "runtime_metadata.json"
            ),
            "source_v66_training_fit_state_sha256": observed,
        }
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def load_unsealed_v7_checkpoint_for_training_gate(
    checkpoint_path: str | Path,
    *,
    hidden_size: int,
) -> AlwaysOnTeacherBasisFullSceneQuestionControlV7:
    root = Path(checkpoint_path).resolve()
    metadata_path = root / "runtime_metadata.json"
    weights_path = root / "control.safetensors"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("architecture")
        != "always_on_teacher_basis_full_scene_control_v7"
        or metadata.get("saved_runtime_training_gate_required") is not True
        or metadata.get("saved_runtime_training_gate_passed") is not False
        or metadata.get("saved_runtime_training_gate_attestation_sha256") is not None
        or metadata.get("weights_sha256") != _sha256_file(weights_path)
        or metadata.get("hidden_size") != hidden_size
    ):
        raise ValueError("Staged V7 is not an authenticated ungated checkpoint")
    state = load_file(str(weights_path), device="cpu")
    control = AlwaysOnTeacherBasisFullSceneQuestionControlV7(
        hidden_size,
        state["output_basis"],
        control_tokens=int(metadata["control_tokens"]),
        expected_environment_latents=int(metadata["expected_environment_latents"]),
        moment_count=int(metadata["moment_count"]),
        interaction_dim=int(metadata["interaction_dim"]),
        trunk_dim=int(metadata["trunk_dim"]),
        maximum_control_rms=float(metadata["maximum_control_rms"]),
        initial_control_rms=float(metadata["initial_control_rms"]),
    )
    control.load_state_dict(state, strict=True)
    if v7_value_state_sha256(control) != metadata[
        "source_v66_training_fit_state_sha256"
    ]:
        raise ValueError("Staged V7 training-fit state changed")
    return control.eval()


__all__ = [
    "build_v7_runtime_metadata",
    "load_unsealed_v7_checkpoint_for_training_gate",
    "save_v7_control_checkpoint",
    "v7_value_state_sha256",
]
