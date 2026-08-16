"""Strict two-file checkpoint writer for residual question control V2."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from semantic_3d_chat.scene_encoder.question_control_v2 import (
    BoundedFullSceneQuestionControlV2,
)
from semantic_3d_chat.training.train_question_control_v56 import (
    _finite_float_state,
    _safe_output_path,
    _sha256_file,
    _validate_hash,
    _write_json,
)


def build_v2_runtime_metadata(
    control: BoundedFullSceneQuestionControlV2,
    *,
    weights_sha256: str,
    base_checkpoint_sha256: str,
    base_runtime_config_sha256: str,
    source_control_checkpoint_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "architecture": "bounded_global_scene_question_control_v2",
        "hidden_size": control.hidden_size,
        "control_tokens": control.control_token_count,
        "expected_environment_latents": control.expected_environment_latents,
        "moment_count": control.moment_count,
        "interaction_dim": control.interaction_dim,
        "output_rank": control.output_rank,
        "maximum_control_rms": control.maximum_control_rms,
        "initial_control_rms": control.initial_control_rms,
        "gate_threshold": control.gate_threshold,
        "weights_sha256": _validate_hash(weights_sha256, "V2 control weights"),
        "base_checkpoint_sha256": _validate_hash(
            base_checkpoint_sha256, "V2 base checkpoint"
        ),
        "base_runtime_config_sha256": _validate_hash(
            base_runtime_config_sha256, "V2 runtime config"
        ),
        "source_control_checkpoint_sha256": _validate_hash(
            source_control_checkpoint_sha256, "V2 source controller"
        ),
        "question_dependent_scene_retrieval": False,
        "complete_scene_prefix_required": True,
        "environmental_text_inputs": [],
        "exact_no_control_route": True,
        "fixed_global_scene_moments": True,
        "boundary_tokens_excluded_from_scene_signature": True,
        "softmax_scene_attention_used": False,
    }


def save_v2_control_checkpoint(
    checkpoint_path: str | Path,
    *,
    control: BoundedFullSceneQuestionControlV2,
    base_checkpoint_sha256: str,
    base_runtime_config_sha256: str,
    source_control_checkpoint_sha256: str,
) -> dict[str, str]:
    """Write only numeric controls and sanitized runtime metadata."""

    destination = _safe_output_path(checkpoint_path, "V2 control checkpoint")
    destination.mkdir(exist_ok=False)
    try:
        weights = destination / "control.safetensors"
        expected_state = _finite_float_state(control)
        save_file(expected_state, weights)
        reloaded = load_file(str(weights), device="cpu")
        if set(reloaded) != set(expected_state) or any(
            not torch.equal(reloaded[name], expected_state[name])
            for name in expected_state
        ):
            raise RuntimeError("Saved V2 control state failed exact reload")
        metadata = build_v2_runtime_metadata(
            control,
            weights_sha256=_sha256_file(weights),
            base_checkpoint_sha256=base_checkpoint_sha256,
            base_runtime_config_sha256=base_runtime_config_sha256,
            source_control_checkpoint_sha256=source_control_checkpoint_sha256,
        )
        _write_json(destination / "runtime_metadata.json", metadata)
        if {item.name for item in destination.iterdir()} != {
            "control.safetensors",
            "runtime_metadata.json",
        }:
            raise RuntimeError("V2 runtime checkpoint inventory is not minimal")
        return {
            "weights_sha256": _sha256_file(weights),
            "runtime_metadata_sha256": _sha256_file(
                destination / "runtime_metadata.json"
            ),
        }
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


__all__ = ["build_v2_runtime_metadata", "save_v2_control_checkpoint"]
