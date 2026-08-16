from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.robot.runtime_refresh import robot_state_encoder_sha256
from semantic_3d_chat.robot.state_checkpoint import (
    build_deterministic_robot_state_encoder,
    create_robot_state_checkpoint,
    load_robot_state_checkpoint,
)
from semantic_3d_chat.robot.state_encoder import ROBOT_STATE_FEATURE_DIM

EXPECTED_FILES = {"runtime_metadata.json", "state.safetensors"}
EXPECTED_METADATA_FIELDS = {
    "architecture",
    "environmental_text_inputs",
    "hidden_dim",
    "initialization_seed",
    "numeric_inputs_only",
    "output_dim",
    "output_scale",
    "schema_version",
    "task_trained",
    "token_count",
    "weights_sha256",
}


def _create_checkpoint(path: Path) -> Path:
    create_robot_state_checkpoint(
        path,
        output_dim=32,
        hidden_dim=24,
        token_count=3,
        seed=1701,
        output_scale=0.015,
    )
    return path


def _metadata(path: Path) -> dict[str, object]:
    return json.loads((path / "runtime_metadata.json").read_text(encoding="utf-8"))


def _write_metadata(path: Path, metadata: dict[str, object]) -> None:
    (path / "runtime_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_deterministic_builder_and_checkpoint_bytes(tmp_path: Path) -> None:
    first = build_deterministic_robot_state_encoder(
        32, hidden_dim=24, token_count=3, seed=1701, output_scale=0.015
    )
    second = build_deterministic_robot_state_encoder(
        32, hidden_dim=24, token_count=3, seed=1701, output_scale=0.015
    )
    assert first.training is False
    assert second.training is False
    assert first.state_dict().keys() == second.state_dict().keys()
    assert all(
        torch.equal(first.state_dict()[name], second.state_dict()[name])
        for name in first.state_dict()
    )

    different_seed = build_deterministic_robot_state_encoder(
        32, hidden_dim=24, token_count=3, seed=1702, output_scale=0.015
    )
    assert any(
        not torch.equal(first.state_dict()[name], different_seed.state_dict()[name])
        for name in first.state_dict()
    )

    checkpoint_a = _create_checkpoint(tmp_path / "checkpoint_a")
    checkpoint_b = _create_checkpoint(tmp_path / "checkpoint_b")
    assert (checkpoint_a / "state.safetensors").read_bytes() == (
        checkpoint_b / "state.safetensors"
    ).read_bytes()
    assert (checkpoint_a / "runtime_metadata.json").read_bytes() == (
        checkpoint_b / "runtime_metadata.json"
    ).read_bytes()


def test_strict_load_audits_two_files_and_emits_expected_shape(tmp_path: Path) -> None:
    checkpoint = _create_checkpoint(tmp_path / "checkpoint")
    audit = FileAccessAudit()
    with audit:
        encoder, digest, metadata = load_robot_state_checkpoint(
            checkpoint,
            expected_output_dim=32,
            audit=audit,
        )

    assert {Path(path).name for path in audit.unique_paths} == EXPECTED_FILES
    assert digest == robot_state_encoder_sha256(encoder)
    assert metadata["weights_sha256"] == _sha256(checkpoint / "state.safetensors")
    assert len(digest) == 64
    assert encoder.training is False
    state_batch = torch.linspace(
        -1.0,
        1.0,
        steps=2 * ROBOT_STATE_FEATURE_DIM,
        dtype=torch.float32,
    ).reshape(2, ROBOT_STATE_FEATURE_DIM)
    tokens = encoder(state_batch)
    assert tokens.shape == (2, 3, 32)
    assert tokens.dtype == torch.float32
    assert torch.isfinite(tokens).all()


def test_checkpoint_contains_no_environmental_text_or_runtime_extras(tmp_path: Path) -> None:
    checkpoint = _create_checkpoint(tmp_path / "checkpoint")
    metadata = _metadata(checkpoint)

    assert {item.name for item in checkpoint.iterdir()} == EXPECTED_FILES
    assert set(metadata) == EXPECTED_METADATA_FIELDS
    assert metadata["numeric_inputs_only"] is True
    assert metadata["environmental_text_inputs"] == []
    assert metadata["task_trained"] is False
    assert not any(
        key in metadata
        for key in (
            "caption",
            "category",
            "object_id",
            "object_labels",
            "oracle",
            "relationships",
            "scene_description",
            "scene_id",
        )
    )

    metadata["scene_description"] = "a chair is beside a table"
    _write_metadata(checkpoint, metadata)
    with pytest.raises(ValueError, match="metadata fields changed"):
        load_robot_state_checkpoint(checkpoint, expected_output_dim=32)


def test_environmental_text_field_cannot_be_populated(tmp_path: Path) -> None:
    checkpoint = _create_checkpoint(tmp_path / "checkpoint")
    metadata = _metadata(checkpoint)
    metadata["environmental_text_inputs"] = ["chair"]
    _write_metadata(checkpoint, metadata)

    with pytest.raises(ValueError, match="contract mismatch"):
        load_robot_state_checkpoint(checkpoint, expected_output_dim=32)


@pytest.mark.parametrize("entry_to_remove", sorted(EXPECTED_FILES))
def test_loader_rejects_missing_checkpoint_entry(
    tmp_path: Path,
    entry_to_remove: str,
) -> None:
    checkpoint = _create_checkpoint(tmp_path / f"missing_{entry_to_remove}")
    (checkpoint / entry_to_remove).unlink()

    with pytest.raises(ValueError, match="exactly two sanitized files"):
        load_robot_state_checkpoint(checkpoint, expected_output_dim=32)


def test_loader_rejects_extra_checkpoint_entry(tmp_path: Path) -> None:
    checkpoint = _create_checkpoint(tmp_path / "checkpoint")
    (checkpoint / "notes.txt").write_text("not permitted", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly two sanitized files"):
        load_robot_state_checkpoint(checkpoint, expected_output_dim=32)


def test_loader_rejects_tampered_weights_and_digest(tmp_path: Path) -> None:
    tampered_weights = _create_checkpoint(tmp_path / "tampered_weights")
    weights = tampered_weights / "state.safetensors"
    weights.write_bytes(weights.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="contract mismatch"):
        load_robot_state_checkpoint(tampered_weights, expected_output_dim=32)

    tampered_digest = _create_checkpoint(tmp_path / "tampered_digest")
    metadata = _metadata(tampered_digest)
    metadata["weights_sha256"] = "0" * 64
    _write_metadata(tampered_digest, metadata)
    with pytest.raises(ValueError, match="contract mismatch"):
        load_robot_state_checkpoint(tampered_digest, expected_output_dim=32)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("initialization_seed", 1702), ("output_scale", 0.016)],
)
def test_loader_rejects_plausible_provenance_tampering(
    tmp_path: Path,
    field: str,
    replacement: float,
) -> None:
    checkpoint = _create_checkpoint(tmp_path / f"tampered_{field}")
    metadata = _metadata(checkpoint)
    metadata[field] = replacement
    _write_metadata(checkpoint, metadata)

    with pytest.raises(ValueError, match="deterministic metadata provenance"):
        load_robot_state_checkpoint(checkpoint, expected_output_dim=32)


def test_loader_rejects_rehashed_replacement_weights(tmp_path: Path) -> None:
    checkpoint = _create_checkpoint(tmp_path / "checkpoint")
    replacement = tmp_path / "replacement"
    create_robot_state_checkpoint(
        replacement,
        output_dim=32,
        hidden_dim=24,
        token_count=3,
        seed=1702,
        output_scale=0.015,
    )
    weights = checkpoint / "state.safetensors"
    weights.write_bytes((replacement / "state.safetensors").read_bytes())
    metadata = _metadata(checkpoint)
    metadata["weights_sha256"] = _sha256(weights)
    _write_metadata(checkpoint, metadata)

    with pytest.raises(ValueError, match="deterministic metadata provenance"):
        load_robot_state_checkpoint(checkpoint, expected_output_dim=32)


def test_loader_rejects_wrong_output_dimension_and_architecture(tmp_path: Path) -> None:
    checkpoint = _create_checkpoint(tmp_path / "checkpoint")
    with pytest.raises(ValueError, match="contract mismatch"):
        load_robot_state_checkpoint(checkpoint, expected_output_dim=31)

    metadata = _metadata(checkpoint)
    metadata["architecture"] = "untrusted_robot_state_encoder"
    _write_metadata(checkpoint, metadata)
    with pytest.raises(ValueError, match="contract mismatch"):
        load_robot_state_checkpoint(checkpoint, expected_output_dim=32)


@pytest.mark.parametrize("forbidden_component", ["oracle", "ORACLE", "qa", "QA"])
def test_creation_rejects_oracle_and_qa_paths(
    tmp_path: Path,
    forbidden_component: str,
) -> None:
    with pytest.raises(ValueError, match="oracle or QA paths"):
        _create_checkpoint(tmp_path / forbidden_component / "checkpoint")


def test_creation_and_load_reject_symlinked_paths(tmp_path: Path) -> None:
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    parent_alias = tmp_path / "parent_alias"
    parent_alias.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="cannot contain symlinks"):
        _create_checkpoint(parent_alias / "checkpoint")

    checkpoint = _create_checkpoint(tmp_path / "checkpoint")
    checkpoint_alias = tmp_path / "checkpoint_alias"
    checkpoint_alias.symlink_to(checkpoint, target_is_directory=True)
    with pytest.raises(ValueError, match="cannot contain symlinks"):
        load_robot_state_checkpoint(checkpoint_alias, expected_output_dim=32)


def test_loader_rejects_symlinked_checkpoint_entry(tmp_path: Path) -> None:
    source = _create_checkpoint(tmp_path / "source")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "runtime_metadata.json").write_bytes(
        (source / "runtime_metadata.json").read_bytes()
    )
    (checkpoint / "state.safetensors").symlink_to(source / "state.safetensors")

    with pytest.raises(ValueError, match="regular files"):
        load_robot_state_checkpoint(checkpoint, expected_output_dim=32)


def test_checkpoint_creation_refuses_overwrite(tmp_path: Path) -> None:
    checkpoint = _create_checkpoint(tmp_path / "checkpoint")
    original_weights = (checkpoint / "state.safetensors").read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        _create_checkpoint(checkpoint)
    assert (checkpoint / "state.safetensors").read_bytes() == original_weights
