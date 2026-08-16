"""Strict, sanitized two-file checkpoint contract for the V74 dense reader."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from semantic_3d_chat.scene_encoder.question_control_v74 import (
    DenseFullSceneContinuousControlV74,
)
from semantic_3d_chat.training.train_question_control_v56 import (
    _finite_float_state,
    _safe_output_path,
    _sha256_file,
    _validate_hash,
    _write_json,
)


V74_RUNTIME_ARCHITECTURE = "dense_full_scene_continuous_control_v74"
V74_RUNTIME_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "architecture",
        "hidden_size",
        "control_tokens",
        "environment_latents",
        "query_count",
        "model_dimension",
        "output_basis_rank",
        "uniform_floor_mass",
        "maximum_control_rms",
        "weights_sha256",
        "base_checkpoint_sha256",
        "base_runtime_config_sha256",
        "source_v74_training_fit_state_sha256",
        "saved_runtime_training_gate_required",
        "saved_runtime_training_gate_passed",
        "saved_runtime_training_gate_attestation_sha256",
        "complete_scene_prefix_required",
        "full_unchanged_prefix_retained_separately",
        "prequestion_scene_key_value_cache",
        "all_environment_latents_attended",
        "positive_attention_floor",
        "bilinear_question_scene_value_interaction",
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


def v74_state_sha256(control: DenseFullSceneContinuousControlV74) -> str:
    """Hash the exact numeric V74 state without serializing any supervision."""

    digest = hashlib.sha256()
    for name in sorted(control.state_dict()):
        value = control.state_dict()[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def build_v74_runtime_metadata(
    control: DenseFullSceneContinuousControlV74,
    *,
    weights_sha256: str,
    base_checkpoint_sha256: str,
    base_runtime_config_sha256: str,
    source_v74_training_fit_state_sha256: str,
    saved_runtime_training_gate_passed: bool = False,
    saved_runtime_training_gate_attestation_sha256: str | None = None,
) -> dict[str, Any]:
    """Build the exact allow-listed metadata consumed by chat inference."""

    if type(control) is not DenseFullSceneContinuousControlV74:
        raise TypeError("V74 metadata requires the exact dense-reader architecture")
    if type(saved_runtime_training_gate_passed) is not bool:
        raise TypeError("V74 saved-runtime gate status must be boolean")
    if saved_runtime_training_gate_passed:
        attestation = _validate_hash(
            saved_runtime_training_gate_attestation_sha256,
            "V74 saved-runtime training gate attestation",
        )
    else:
        if saved_runtime_training_gate_attestation_sha256 is not None:
            raise ValueError("Ungated V74 metadata cannot carry a gate attestation")
        attestation = None
    metadata = {
        "schema_version": 74,
        "architecture": V74_RUNTIME_ARCHITECTURE,
        "hidden_size": control.hidden_size,
        "control_tokens": control.control_token_count,
        "environment_latents": control.environment_latents,
        "query_count": control.query_count,
        "model_dimension": control.model_dimension,
        "output_basis_rank": control.output_basis_rank,
        "uniform_floor_mass": control.uniform_floor_mass,
        "maximum_control_rms": control.maximum_control_rms,
        "weights_sha256": _validate_hash(weights_sha256, "V74 control weights"),
        "base_checkpoint_sha256": _validate_hash(
            base_checkpoint_sha256, "V74 base checkpoint"
        ),
        "base_runtime_config_sha256": _validate_hash(
            base_runtime_config_sha256, "V74 runtime config"
        ),
        "source_v74_training_fit_state_sha256": _validate_hash(
            source_v74_training_fit_state_sha256,
            "V74 source training-fit state",
        ),
        "saved_runtime_training_gate_required": True,
        "saved_runtime_training_gate_passed": saved_runtime_training_gate_passed,
        "saved_runtime_training_gate_attestation_sha256": attestation,
        "complete_scene_prefix_required": True,
        "full_unchanged_prefix_retained_separately": True,
        "prequestion_scene_key_value_cache": True,
        "all_environment_latents_attended": True,
        "positive_attention_floor": True,
        "bilinear_question_scene_value_interaction": True,
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
    if set(metadata) != V74_RUNTIME_METADATA_FIELDS:
        raise AssertionError("V74 runtime metadata field contract changed")
    return metadata


def _state_exact(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> bool:
    return set(left) == set(right) and all(
        torch.equal(left[name], right[name]) for name in left
    )


def save_v74_control_checkpoint(
    checkpoint_path: str | Path,
    *,
    control: DenseFullSceneContinuousControlV74,
    base_checkpoint_sha256: str,
    base_runtime_config_sha256: str,
    expected_training_fit_state_sha256: str,
    saved_runtime_training_gate_passed: bool = False,
    saved_runtime_training_gate_attestation_sha256: str | None = None,
) -> dict[str, str]:
    """Save exactly weights plus sanitized metadata into a new directory."""

    if type(control) is not DenseFullSceneContinuousControlV74:
        raise TypeError("V74 checkpoint requires the exact dense-reader architecture")
    observed = v74_state_sha256(control)
    if observed != _validate_hash(
        expected_training_fit_state_sha256,
        "expected V74 training-fit state",
    ):
        raise ValueError("V74 numeric training-fit state changed before save")
    destination = _safe_output_path(checkpoint_path, "V74 control checkpoint")
    destination.mkdir(exist_ok=False)
    try:
        weights = destination / "control.safetensors"
        expected = _finite_float_state(control)
        save_file(expected, weights)
        reloaded = load_file(str(weights), device="cpu")
        if not _state_exact(reloaded, expected):
            raise RuntimeError("Saved V74 state failed exact reload")
        metadata = build_v74_runtime_metadata(
            control,
            weights_sha256=_sha256_file(weights),
            base_checkpoint_sha256=base_checkpoint_sha256,
            base_runtime_config_sha256=base_runtime_config_sha256,
            source_v74_training_fit_state_sha256=observed,
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
            raise RuntimeError("V74 runtime checkpoint inventory is not minimal")
        return {
            "weights_sha256": _sha256_file(weights),
            "runtime_metadata_sha256": _sha256_file(
                destination / "runtime_metadata.json"
            ),
            "source_v74_training_fit_state_sha256": observed,
        }
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def load_unsealed_v74_checkpoint_for_training_gate(
    checkpoint_path: str | Path,
    *,
    hidden_size: int,
) -> DenseFullSceneContinuousControlV74:
    """Load an authenticated staged candidate solely for its behavioral gate."""

    root = Path(checkpoint_path).resolve()
    if {path.name for path in root.iterdir()} != {
        "control.safetensors",
        "runtime_metadata.json",
    }:
        raise ValueError("Staged V74 checkpoint inventory is not minimal")
    metadata_path = root / "runtime_metadata.json"
    weights_path = root / "control.safetensors"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("architecture") != V74_RUNTIME_ARCHITECTURE
        or metadata.get("saved_runtime_training_gate_required") is not True
        or metadata.get("saved_runtime_training_gate_passed") is not False
        or metadata.get("saved_runtime_training_gate_attestation_sha256") is not None
        or metadata.get("weights_sha256") != _sha256_file(weights_path)
        or metadata.get("hidden_size") != hidden_size
    ):
        raise ValueError("Staged V74 is not an authenticated ungated checkpoint")
    state = load_file(str(weights_path), device="cpu")
    basis = state.get("output_basis")
    if basis is None:
        raise ValueError("Staged V74 checkpoint lacks its numeric output basis")
    control = DenseFullSceneContinuousControlV74(
        hidden_size,
        basis,
        environment_latents=int(metadata["environment_latents"]),
        query_count=int(metadata["query_count"]),
        model_dimension=int(metadata["model_dimension"]),
        uniform_floor_mass=float(metadata["uniform_floor_mass"]),
        maximum_control_rms=float(metadata["maximum_control_rms"]),
    )
    control.load_state_dict(state, strict=True)
    if v74_state_sha256(control) != metadata[
        "source_v74_training_fit_state_sha256"
    ]:
        raise ValueError("Staged V74 training-fit state changed")
    return control.eval()


__all__ = [
    "V74_RUNTIME_ARCHITECTURE",
    "V74_RUNTIME_METADATA_FIELDS",
    "build_v74_runtime_metadata",
    "load_unsealed_v74_checkpoint_for_training_gate",
    "save_v74_control_checkpoint",
    "v74_state_sha256",
]
