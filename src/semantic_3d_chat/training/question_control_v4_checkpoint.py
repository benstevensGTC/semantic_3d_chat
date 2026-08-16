"""Strict V4 checkpoint contract for a frozen V60 value branch plus new gate."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from semantic_3d_chat.scene_encoder.question_control_v4 import (
    SceneConditionedGateTeacherBasisControlV4,
)
from semantic_3d_chat.training.train_question_control_v56 import (
    _finite_float_state,
    _safe_output_path,
    _sha256_file,
    _validate_hash,
    _write_json,
)


def inherited_value_state_sha256(
    control: SceneConditionedGateTeacherBasisControlV4,
) -> str:
    """Hash all inherited V60 tensors in stable name/shape/dtype/byte order."""

    digest = hashlib.sha256()
    state = control.state_dict()
    for name in control.inherited_state_names:
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def build_v4_runtime_metadata(
    control: SceneConditionedGateTeacherBasisControlV4,
    *,
    weights_sha256: str,
    base_checkpoint_sha256: str,
    base_runtime_config_sha256: str,
    source_v60_checkpoint_sha256: str,
    inherited_state_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 4,
        "architecture": "scene_conditioned_gate_teacher_basis_control_v4",
        "hidden_size": control.hidden_size,
        "control_tokens": control.control_token_count,
        "expected_environment_latents": control.expected_environment_latents,
        "moment_count": control.moment_count,
        "interaction_dim": control.interaction_dim,
        "trunk_dim": control.trunk_dim,
        "output_basis_rank": control.output_basis_rank,
        "gate_hidden_dim": control.gate_hidden_dim,
        "maximum_control_rms": control.maximum_control_rms,
        "initial_control_rms": control.initial_control_rms,
        "gate_threshold": control.gate_threshold,
        "weights_sha256": _validate_hash(weights_sha256, "V4 control weights"),
        "base_checkpoint_sha256": _validate_hash(base_checkpoint_sha256, "V4 base checkpoint"),
        "base_runtime_config_sha256": _validate_hash(
            base_runtime_config_sha256, "V4 runtime config"
        ),
        "source_v60_checkpoint_sha256": _validate_hash(
            source_v60_checkpoint_sha256, "V4 source V60 checkpoint"
        ),
        "inherited_value_state_sha256": _validate_hash(
            inherited_state_sha256, "V4 inherited value state"
        ),
        "only_gate_trainable": True,
        "question_dependent_scene_retrieval": False,
        "complete_scene_prefix_required": True,
        "environmental_text_inputs": [],
        "exact_no_control_route": True,
        "fixed_global_scene_moments": True,
        "boundary_tokens_excluded_from_scene_signature": True,
        "softmax_scene_attention_used": False,
        "control_values_scene_question_bilinear": True,
        "gate_scene_question_conditioned": True,
        "route_is_environmental_retrieval": False,
    }


def save_v4_control_checkpoint(
    checkpoint_path: str | Path,
    *,
    control: SceneConditionedGateTeacherBasisControlV4,
    base_checkpoint_sha256: str,
    base_runtime_config_sha256: str,
    source_v60_checkpoint_sha256: str,
    expected_inherited_state_sha256: str,
) -> dict[str, str]:
    if not control.inherited_v60_state_frozen:
        raise ValueError("V4 cannot save unless every inherited V60 parameter is frozen")
    observed_inherited = inherited_value_state_sha256(control)
    if observed_inherited != _validate_hash(
        expected_inherited_state_sha256, "expected inherited V60 state"
    ):
        raise ValueError("V4 inherited V60 state changed before checkpoint save")
    destination = _safe_output_path(checkpoint_path, "V4 control checkpoint")
    destination.mkdir(exist_ok=False)
    try:
        weights = destination / "control.safetensors"
        expected = _finite_float_state(control)
        save_file(expected, weights)
        reloaded = load_file(str(weights), device="cpu")
        if set(reloaded) != set(expected) or any(
            not torch.equal(reloaded[name], expected[name]) for name in expected
        ):
            raise RuntimeError("Saved V4 state failed exact reload")
        metadata = build_v4_runtime_metadata(
            control,
            weights_sha256=_sha256_file(weights),
            base_checkpoint_sha256=base_checkpoint_sha256,
            base_runtime_config_sha256=base_runtime_config_sha256,
            source_v60_checkpoint_sha256=source_v60_checkpoint_sha256,
            inherited_state_sha256=observed_inherited,
        )
        _write_json(destination / "runtime_metadata.json", metadata)
        if {item.name for item in destination.iterdir()} != {
            "control.safetensors",
            "runtime_metadata.json",
        }:
            raise RuntimeError("V4 runtime checkpoint inventory is not minimal")
        return {
            "weights_sha256": _sha256_file(weights),
            "runtime_metadata_sha256": _sha256_file(destination / "runtime_metadata.json"),
            "inherited_value_state_sha256": observed_inherited,
        }
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


__all__ = [
    "build_v4_runtime_metadata",
    "inherited_value_state_sha256",
    "save_v4_control_checkpoint",
]
