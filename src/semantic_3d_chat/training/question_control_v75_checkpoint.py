"""Strict, sanitized two-file checkpoint contract for the V75 dense reader."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import torch
from safetensors.torch import load_file, save_file

from semantic_3d_chat.scene_encoder.question_control_v75 import (
    DenseFullSceneContinuousControlV75,
)
from semantic_3d_chat.training.train_question_control_v56 import (
    _finite_float_state,
    _safe_output_path,
    _sha256_file,
    _validate_hash,
    _write_json,
)

V75_RUNTIME_ARCHITECTURE: Final[str] = "dense_full_scene_continuous_control_v75"
V75_RUNTIME_STATE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "output_basis",
        "key.weight",
        "value.weight",
        "query.weight",
        "coefficient_hidden.weight",
        "coefficient_output.weight",
    }
)
V75_RUNTIME_METADATA_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "architecture",
        "hidden_size",
        "control_tokens",
        "environment_latents",
        "query_count",
        "model_dimension",
        "coefficient_decoder_hidden_dimension",
        "output_basis_rank",
        "uniform_floor_mass",
        "maximum_control_rms",
        "weights_sha256",
        "base_checkpoint_sha256",
        "base_runtime_config_sha256",
        "source_v75_candidate_sha256",
        "source_v75_training_fit_state_sha256",
        "saved_runtime_training_gate_required",
        "saved_runtime_training_gate_passed",
        "saved_runtime_training_gate_attestation_sha256",
        "complete_scene_prefix_required",
        "full_unchanged_prefix_retained_separately",
        "prequestion_scene_key_value_cache",
        "all_environment_latents_attended",
        "positive_attention_floor",
        "bilinear_question_scene_value_interaction",
        "bias_free_nonlinear_coefficient_decoder",
        "zero_preserving_coefficient_activation",
        "question_only_output_path_exists",
        "question_dependent_scene_retrieval",
        "latent_selection_or_top_k_used",
        "environmental_text_inputs",
        "training_answers_runtime_loaded",
        "answer_text_runtime_loaded",
        "answer_class_codebook_runtime_loaded",
        "teacher_cache_runtime_loaded",
        "oracle_runtime_loaded",
        "question_or_answer_text_serialized",
    }
)


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def v75_state_sha256(control: DenseFullSceneContinuousControlV75) -> str:
    """Hash the exact numeric V75 state without serializing supervision."""

    if type(control) is not DenseFullSceneContinuousControlV75:
        raise TypeError("V75 state hashing requires the exact nonlinear architecture")
    return _state_sha256(control.state_dict())


def build_v75_runtime_metadata(
    control: DenseFullSceneContinuousControlV75,
    *,
    weights_sha256: str,
    base_checkpoint_sha256: str,
    base_runtime_config_sha256: str,
    source_v75_candidate_sha256: str,
    source_v75_training_fit_state_sha256: str,
    saved_runtime_training_gate_attestation_sha256: str,
) -> dict[str, Any]:
    """Build the exact inference-only V75 metadata allowlist."""

    if type(control) is not DenseFullSceneContinuousControlV75:
        raise TypeError("V75 metadata requires the exact nonlinear architecture")
    metadata = {
        "schema_version": 75,
        "architecture": V75_RUNTIME_ARCHITECTURE,
        "hidden_size": control.hidden_size,
        "control_tokens": control.control_token_count,
        "environment_latents": control.environment_latents,
        "query_count": control.query_count,
        "model_dimension": control.model_dimension,
        "coefficient_decoder_hidden_dimension": (
            control.coefficient_decoder_hidden_dimension
        ),
        "output_basis_rank": control.output_basis_rank,
        "uniform_floor_mass": control.uniform_floor_mass,
        "maximum_control_rms": control.maximum_control_rms,
        "weights_sha256": _validate_hash(weights_sha256, "V75 control weights"),
        "base_checkpoint_sha256": _validate_hash(
            base_checkpoint_sha256, "V75 base checkpoint"
        ),
        "base_runtime_config_sha256": _validate_hash(
            base_runtime_config_sha256, "V75 runtime config"
        ),
        "source_v75_candidate_sha256": _validate_hash(
            source_v75_candidate_sha256, "V75 source candidate"
        ),
        "source_v75_training_fit_state_sha256": _validate_hash(
            source_v75_training_fit_state_sha256,
            "V75 source training-fit state",
        ),
        "saved_runtime_training_gate_required": True,
        "saved_runtime_training_gate_passed": True,
        "saved_runtime_training_gate_attestation_sha256": _validate_hash(
            saved_runtime_training_gate_attestation_sha256,
            "V75 saved-runtime training gate attestation",
        ),
        "complete_scene_prefix_required": True,
        "full_unchanged_prefix_retained_separately": True,
        "prequestion_scene_key_value_cache": True,
        "all_environment_latents_attended": True,
        "positive_attention_floor": True,
        "bilinear_question_scene_value_interaction": True,
        "bias_free_nonlinear_coefficient_decoder": True,
        "zero_preserving_coefficient_activation": True,
        "question_only_output_path_exists": False,
        "question_dependent_scene_retrieval": False,
        "latent_selection_or_top_k_used": False,
        "environmental_text_inputs": [],
        "training_answers_runtime_loaded": False,
        "answer_text_runtime_loaded": False,
        "answer_class_codebook_runtime_loaded": False,
        "teacher_cache_runtime_loaded": False,
        "oracle_runtime_loaded": False,
        "question_or_answer_text_serialized": False,
    }
    if set(metadata) != V75_RUNTIME_METADATA_FIELDS:
        raise AssertionError("V75 runtime metadata field contract changed")
    return metadata


def _state_exact(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> bool:
    return set(left) == set(right) and all(
        torch.equal(left[name], right[name]) for name in left
    )


def save_v75_control_checkpoint(
    checkpoint_path: str | Path,
    *,
    control: DenseFullSceneContinuousControlV75,
    base_checkpoint_sha256: str,
    base_runtime_config_sha256: str,
    source_v75_candidate_sha256: str,
    expected_training_fit_state_sha256: str,
    saved_runtime_training_gate_attestation_sha256: str,
) -> dict[str, str]:
    """Create one sealed V75 runtime checkpoint with exactly two files.

    This serializer has no unsealed mode.  Promotion evidence must already have
    been validated and reduced to its deterministic attestation digest.
    """

    if type(control) is not DenseFullSceneContinuousControlV75:
        raise TypeError("V75 checkpoint requires the exact nonlinear architecture")
    observed = v75_state_sha256(control)
    if observed != _validate_hash(
        expected_training_fit_state_sha256,
        "expected V75 training-fit state",
    ):
        raise ValueError("V75 numeric training-fit state changed before save")
    destination = _safe_output_path(checkpoint_path, "V75 control checkpoint")
    destination.mkdir(exist_ok=False)
    try:
        weights = destination / "control.safetensors"
        expected = _finite_float_state(control)
        if set(expected) != V75_RUNTIME_STATE_FIELDS:
            raise ValueError("V75 runtime state fields changed before save")
        save_file(expected, weights)
        reloaded = load_file(str(weights), device="cpu")
        if not _state_exact(reloaded, expected):
            raise RuntimeError("Saved V75 state failed exact reload")
        metadata = build_v75_runtime_metadata(
            control,
            weights_sha256=_sha256_file(weights),
            base_checkpoint_sha256=base_checkpoint_sha256,
            base_runtime_config_sha256=base_runtime_config_sha256,
            source_v75_candidate_sha256=source_v75_candidate_sha256,
            source_v75_training_fit_state_sha256=observed,
            saved_runtime_training_gate_attestation_sha256=(
                saved_runtime_training_gate_attestation_sha256
            ),
        )
        _write_json(destination / "runtime_metadata.json", metadata)
        if {item.name for item in destination.iterdir()} != {
            "control.safetensors",
            "runtime_metadata.json",
        }:
            raise RuntimeError("V75 runtime checkpoint inventory is not minimal")
        return {
            "weights_sha256": _sha256_file(weights),
            "runtime_metadata_sha256": _sha256_file(
                destination / "runtime_metadata.json"
            ),
            "source_v75_candidate_sha256": _validate_hash(
                source_v75_candidate_sha256, "V75 source candidate"
            ),
            "source_v75_training_fit_state_sha256": observed,
        }
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


__all__ = [
    "V75_RUNTIME_ARCHITECTURE",
    "V75_RUNTIME_METADATA_FIELDS",
    "V75_RUNTIME_STATE_FIELDS",
    "build_v75_runtime_metadata",
    "save_v75_control_checkpoint",
    "v75_state_sha256",
]
