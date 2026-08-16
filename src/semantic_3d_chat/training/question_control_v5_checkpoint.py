"""Strict schema-5 checkpoint for normalized factorized V5 routing."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from semantic_3d_chat.scene_encoder.question_control_v5 import (
    NormalizedFactorizedSceneQuestionControlV5,
)
from semantic_3d_chat.training.train_question_control_v56 import (
    _finite_float_state,
    _safe_output_path,
    _sha256_file,
    _validate_hash,
    _write_json,
)


def inherited_v60_state_sha256(
    control: NormalizedFactorizedSceneQuestionControlV5,
) -> str:
    """Hash every inherited V60 tensor with a deterministic binary contract."""

    digest = hashlib.sha256()
    state = control.state_dict()
    for name in control.inherited_state_names:
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def build_v5_runtime_metadata(
    control: NormalizedFactorizedSceneQuestionControlV5,
    *,
    weights_sha256: str,
    base_checkpoint_sha256: str,
    base_runtime_config_sha256: str,
    source_v60_checkpoint_sha256: str,
    inherited_state_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 5,
        "architecture": "normalized_factorized_scene_question_route_v5",
        "hidden_size": control.hidden_size,
        "control_tokens": control.control_token_count,
        "expected_environment_latents": control.expected_environment_latents,
        "moment_count": control.moment_count,
        "interaction_dim": control.interaction_dim,
        "trunk_dim": control.trunk_dim,
        "output_basis_rank": control.output_basis_rank,
        "route_factor_rank": control.route_factor_rank,
        "maximum_control_rms": control.maximum_control_rms,
        "initial_control_rms": control.initial_control_rms,
        "gate_threshold": control.gate_threshold,
        "weights_sha256": _validate_hash(weights_sha256, "V5 control weights"),
        "base_checkpoint_sha256": _validate_hash(base_checkpoint_sha256, "V5 base checkpoint"),
        "base_runtime_config_sha256": _validate_hash(
            base_runtime_config_sha256, "V5 runtime config"
        ),
        "source_v60_checkpoint_sha256": _validate_hash(
            source_v60_checkpoint_sha256, "V5 source V60 checkpoint"
        ),
        "inherited_value_state_sha256": _validate_hash(
            inherited_state_sha256, "V5 inherited value state"
        ),
        "only_factorized_gate_trainable": True,
        "question_dependent_scene_retrieval": False,
        "complete_scene_prefix_required": True,
        "environmental_text_inputs": [],
        "exact_no_control_route": True,
        "fixed_global_scene_moments": True,
        "boundary_tokens_excluded_from_scene_signature": True,
        "softmax_scene_attention_used": False,
        "control_values_scene_question_bilinear": True,
        "gate_scene_question_conditioned": True,
        "separate_question_scene_route_projections": True,
        "all_scene_moments_consumed_by_route": True,
        "normalized_route_factors": True,
        "low_rank_bilinear_route": True,
        "route_uses_inherited_value_trunk": False,
        "route_is_environmental_retrieval": False,
    }


def save_v5_control_checkpoint(
    checkpoint_path: str | Path,
    *,
    control: NormalizedFactorizedSceneQuestionControlV5,
    base_checkpoint_sha256: str,
    base_runtime_config_sha256: str,
    source_v60_checkpoint_sha256: str,
    expected_inherited_state_sha256: str,
) -> dict[str, str]:
    if not control.inherited_v60_state_frozen:
        raise ValueError("V5 cannot save unless every inherited V60 parameter is frozen")
    observed_inherited = inherited_v60_state_sha256(control)
    if observed_inherited != _validate_hash(
        expected_inherited_state_sha256, "expected inherited V60 state"
    ):
        raise ValueError("V5 inherited V60 state changed before checkpoint save")
    destination = _safe_output_path(checkpoint_path, "V5 control checkpoint")
    destination.mkdir(exist_ok=False)
    try:
        weights = destination / "control.safetensors"
        expected = _finite_float_state(control)
        save_file(expected, weights)
        reloaded = load_file(str(weights), device="cpu")
        if set(reloaded) != set(expected) or any(
            not torch.equal(reloaded[name], expected[name]) for name in expected
        ):
            raise RuntimeError("Saved V5 state failed exact reload")
        metadata = build_v5_runtime_metadata(
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
            raise RuntimeError("V5 runtime checkpoint inventory is not minimal")
        return {
            "weights_sha256": _sha256_file(weights),
            "runtime_metadata_sha256": _sha256_file(destination / "runtime_metadata.json"),
            "inherited_value_state_sha256": observed_inherited,
        }
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


__all__ = [
    "build_v5_runtime_metadata",
    "inherited_v60_state_sha256",
    "save_v5_control_checkpoint",
]
