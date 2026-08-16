from __future__ import annotations

from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.predict_question_control import (
    _cached_prefix_hashes,
    _control_checkpoint_sha256,
    _prediction_condition,
)


def _checkpoint(path: Path) -> Path:
    path.mkdir()
    (path / "control.safetensors").write_bytes(b"continuous-weights")
    (path / "runtime_metadata.json").write_text("{}", encoding="utf-8")
    return path


def test_control_checkpoint_fingerprint_is_inventory_bound(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path / "control")
    first = _control_checkpoint_sha256(source)

    assert len(first) == 64
    assert first == _control_checkpoint_sha256(source)

    (source / "optimizer.pt").write_bytes(b"training-only")
    with pytest.raises(ValueError, match="runtime-minimal"):
        _control_checkpoint_sha256(source)


def test_control_checkpoint_fingerprint_rejects_symlinks(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path / "control")
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)

    with pytest.raises(ValueError, match="must not contain symlinks"):
        _control_checkpoint_sha256(alias)


def test_control_checkpoint_fingerprint_rejects_oracle_directory(tmp_path: Path) -> None:
    parent = tmp_path / "oracle"
    parent.mkdir()
    source = _checkpoint(parent / "control")

    with pytest.raises(ValueError, match="separate from QA/oracle"):
        _control_checkpoint_sha256(source)


def test_prediction_condition_binds_question_limit_and_control_checkpoint() -> None:
    digest = "a" * 64
    assert _prediction_condition(None, digest) == (
        f"all_questions;control_checkpoint_sha256={digest}"
    )
    assert _prediction_condition(7, digest) == (
        f"max_questions_per_scene=7;control_checkpoint_sha256={digest}"
    )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _prediction_condition(None, "invalid")


def test_cached_prefix_hashes_require_one_valid_immutable_identity() -> None:
    scene_id = "scene_000001"
    digest = "b" * 64
    records = [
        {"scene_id": scene_id, "question_id": "q_000001", "prefix_hash": digest},
        {"scene_id": scene_id, "question_id": "q_000002", "prefix_hash": digest},
    ]
    assert _cached_prefix_hashes(records, scene_id) == {digest}

    records[1]["prefix_hash"] = None
    with pytest.raises(RuntimeError, match="invalid prefix hash"):
        _cached_prefix_hashes(records, scene_id)

    records[1]["prefix_hash"] = "c" * 64
    assert _cached_prefix_hashes(records, scene_id) == {digest, "c" * 64}
