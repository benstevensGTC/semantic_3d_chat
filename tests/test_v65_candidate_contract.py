from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.v65_candidate_contract import (
    V65_ARCHITECTURE,
    validate_sealed_v65_checkpoint,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint(path: Path, *, gated: bool = True) -> Path:
    path.mkdir()
    weights = path / "control.safetensors"
    weights.write_bytes(b"synthetic-finite-v65-state")
    metadata = {
        "schema_version": 6,
        "architecture": V65_ARCHITECTURE,
        "hidden_size": 1536,
        "control_tokens": 4,
        "expected_environment_latents": 256,
        "moment_count": 8,
        "interaction_dim": 24,
        "trunk_dim": 128,
        "output_basis_rank": 32,
        "maximum_control_rms": 0.2,
        "initial_control_rms": 0.075,
        "activation_rms_threshold": 0.01,
        "activation_rms_aggregation": "maximum_over_control_tokens",
        "weights_sha256": _sha256(weights),
        "base_checkpoint_sha256": "1" * 64,
        "base_runtime_config_sha256": "2" * 64,
        "source_v65_training_fit_state_sha256": "3" * 64,
        "source_v65_value_state_sha256": "3" * 64,
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
        "saved_runtime_training_gate_passed": gated,
        "saved_runtime_training_gate_attestation_sha256": "4" * 64 if gated else None,
    }
    (path / "runtime_metadata.json").write_text(
        json.dumps(metadata, sort_keys=True), encoding="utf-8"
    )
    return path


def test_sealed_v65_checkpoint_contract_accepts_only_public_gated_artifact(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path / "v65")

    sealed = validate_sealed_v65_checkpoint(
        checkpoint,
        base_checkpoint_sha256="1" * 64,
        runtime_config_sha256="2" * 64,
    )

    assert len(sealed.fingerprint_sha256) == 64
    assert sealed.metadata["saved_runtime_training_gate_passed"] is True
    assert sealed.metadata["saved_runtime_training_gate_attestation_sha256"] == "4" * 64


def test_sealed_v65_checkpoint_rejects_ungated_staging_artifact(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "staged", gated=False)

    with pytest.raises(ValueError, match="V65"):
        validate_sealed_v65_checkpoint(checkpoint)


def test_sealed_v65_checkpoint_rejects_weight_or_metadata_tampering(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path / "weights")
    (checkpoint / "control.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="sealed magnitude-gated"):
        validate_sealed_v65_checkpoint(checkpoint)

    metadata_checkpoint = _checkpoint(tmp_path / "metadata")
    metadata_path = metadata_checkpoint / "runtime_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["saved_runtime_training_gate_attestation_sha256"] = "not-a-hash"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid digest"):
        validate_sealed_v65_checkpoint(metadata_checkpoint)


def test_sealed_v65_checkpoint_rejects_wrong_base_or_runtime(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "v65")

    with pytest.raises(ValueError, match="different base checkpoint"):
        validate_sealed_v65_checkpoint(
            checkpoint,
            base_checkpoint_sha256="9" * 64,
        )
    with pytest.raises(ValueError, match="different runtime configuration"):
        validate_sealed_v65_checkpoint(
            checkpoint,
            runtime_config_sha256="9" * 64,
        )
