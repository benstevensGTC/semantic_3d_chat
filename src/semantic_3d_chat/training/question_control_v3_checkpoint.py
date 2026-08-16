"""Strict two-file checkpoint writer for teacher-basis question control V3."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from semantic_3d_chat.scene_encoder.question_control_v3 import (
    TeacherBasisFullSceneQuestionControlV3,
)
from semantic_3d_chat.training.train_question_control_v56 import (
    _finite_float_state,
    _safe_output_path,
    _sha256_file,
    _validate_hash,
    _write_json,
)


def build_v3_runtime_metadata(
    control: TeacherBasisFullSceneQuestionControlV3,
    *,
    weights_sha256: str,
    base_checkpoint_sha256: str,
    base_runtime_config_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "architecture": "teacher_basis_full_scene_question_control_v3",
        "hidden_size": control.hidden_size,
        "control_tokens": control.control_token_count,
        "expected_environment_latents": control.expected_environment_latents,
        "moment_count": control.moment_count,
        "interaction_dim": control.interaction_dim,
        "trunk_dim": control.trunk_dim,
        "output_basis_rank": control.output_basis_rank,
        "maximum_control_rms": control.maximum_control_rms,
        "initial_control_rms": control.initial_control_rms,
        "gate_threshold": control.gate_threshold,
        "weights_sha256": _validate_hash(weights_sha256, "V3 control weights"),
        "base_checkpoint_sha256": _validate_hash(
            base_checkpoint_sha256, "V3 base checkpoint"
        ),
        "base_runtime_config_sha256": _validate_hash(
            base_runtime_config_sha256, "V3 runtime config"
        ),
        "question_dependent_scene_retrieval": False,
        "complete_scene_prefix_required": True,
        "environmental_text_inputs": [],
        "exact_no_control_route": True,
        "fixed_global_scene_moments": True,
        "boundary_tokens_excluded_from_scene_signature": True,
        "softmax_scene_attention_used": False,
        "control_values_scene_question_bilinear": True,
        "route_is_environmental_retrieval": False,
    }


def save_v3_control_checkpoint(
    checkpoint_path: str | Path,
    *,
    control: TeacherBasisFullSceneQuestionControlV3,
    base_checkpoint_sha256: str,
    base_runtime_config_sha256: str,
) -> dict[str, str]:
    destination = _safe_output_path(checkpoint_path, "V3 control checkpoint")
    destination.mkdir(exist_ok=False)
    try:
        weights = destination / "control.safetensors"
        expected = _finite_float_state(control)
        save_file(expected, weights)
        reloaded = load_file(str(weights), device="cpu")
        if set(reloaded) != set(expected) or any(
            not torch.equal(reloaded[name], expected[name]) for name in expected
        ):
            raise RuntimeError("Saved V3 state failed exact reload")
        metadata = build_v3_runtime_metadata(
            control,
            weights_sha256=_sha256_file(weights),
            base_checkpoint_sha256=base_checkpoint_sha256,
            base_runtime_config_sha256=base_runtime_config_sha256,
        )
        _write_json(destination / "runtime_metadata.json", metadata)
        if {item.name for item in destination.iterdir()} != {
            "control.safetensors",
            "runtime_metadata.json",
        }:
            raise RuntimeError("V3 runtime checkpoint inventory is not minimal")
        return {
            "weights_sha256": _sha256_file(weights),
            "runtime_metadata_sha256": _sha256_file(
                destination / "runtime_metadata.json"
            ),
        }
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


__all__ = ["build_v3_runtime_metadata", "save_v3_control_checkpoint"]
