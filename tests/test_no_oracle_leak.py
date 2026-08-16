from pathlib import Path

import pytest

from semantic_3d_chat.chat.file_audit import FileAccessAudit


def test_audit_allows_runtime_artifact_and_detects_oracle(tmp_path: Path) -> None:
    oracle = tmp_path / "data" / "oracle"
    runtime = tmp_path / "data" / "maps"
    oracle.mkdir(parents=True)
    runtime.mkdir(parents=True)
    oracle_file = oracle / "truth.json"
    runtime_file = runtime / "voxel_map.npz"
    oracle_file.write_text("secret", encoding="utf-8")
    runtime_file.write_text("continuous", encoding="utf-8")
    audit = FileAccessAudit([oracle])
    with audit:
        runtime_file.read_bytes()
    audit.assert_clean()
    with audit:
        oracle_file.read_bytes()
    with pytest.raises(RuntimeError, match="Forbidden runtime file access"):
        audit.assert_clean()


def test_runtime_read_survives_oracle_directory_rename(tmp_path: Path) -> None:
    oracle = tmp_path / "oracle"
    hidden = tmp_path / "oracle.unavailable"
    runtime = tmp_path / "runtime"
    oracle.mkdir()
    runtime.mkdir()
    (oracle / "truth.json").write_text("not for runtime", encoding="utf-8")
    artifact = runtime / "scene.bin"
    artifact.write_bytes(b"continuous-scene-memory")
    oracle.rename(hidden)
    try:
        assert artifact.read_bytes() == b"continuous-scene-memory"
    finally:
        hidden.rename(oracle)


def test_strict_audit_blocks_forbidden_read_before_bytes_are_returned(tmp_path: Path) -> None:
    forbidden = tmp_path / "outside" / "qa" / "answers.jsonl"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_text("environmental supervision", encoding="utf-8")
    audit = FileAccessAudit(
        forbidden_component_names={"qa", "oracle", "rendered", "features"},
        block_forbidden=True,
    )

    with audit, pytest.raises(PermissionError, match="before open"):
        forbidden.read_text(encoding="utf-8")

    assert str(forbidden.resolve()) in audit.forbidden_accesses()
