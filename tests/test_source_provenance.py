from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from semantic_3d_chat.training.source_provenance import (
    SOURCE_SCOPE,
    capture_git_source_provenance,
    require_clean_committed_source,
    source_provenance_resume_contract_mismatch,
)


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
    )


def _committed_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.email", "tests@example.invalid")
    _git(repository, "config", "user.name", "Provenance Test")
    for relative, contents in (
        ("src/module.py", "SOURCE_SENTINEL = 1\n"),
        ("reports/result.json", '{"generated": 1}\n'),
        ("data/cache.txt", "generated data\n"),
        ("data_gemma4/cache.txt", "generated model data\n"),
    ):
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(contents, encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "--quiet", "-m", "initial")
    return repository


def test_source_provenance_is_deterministic_and_excludes_generated_artifacts(
    tmp_path: Path,
) -> None:
    repository = _committed_repository(tmp_path)
    empty_diff_hash = hashlib.sha256(b"").hexdigest()

    clean = capture_git_source_provenance(repository)

    assert clean == capture_git_source_provenance(repository)
    assert clean["schema_version"] == 1
    assert clean["scope"] == SOURCE_SCOPE
    assert clean["available"] is True
    assert clean["is_clean"] is True
    assert len(clean["head_commit"]) == 40
    assert len(clean["head_tree"]) == 40
    assert clean["tracked_diff_sha256"] == empty_diff_hash
    require_clean_committed_source(clean)

    (repository / "reports/result.json").write_text('{"generated": 2}\n', encoding="utf-8")
    (repository / "data/cache.txt").write_text("changed generated data\n", encoding="utf-8")
    generated_change = capture_git_source_provenance(repository)
    assert generated_change == clean

    (repository / "src/module.py").write_text("SOURCE_SENTINEL = 2\n", encoding="utf-8")
    dirty = capture_git_source_provenance(repository)
    assert dirty["available"] is True
    assert dirty["head_commit"] == clean["head_commit"]
    assert dirty["head_tree"] == clean["head_tree"]
    assert dirty["is_clean"] is False
    assert dirty["tracked_diff_sha256"] != empty_diff_hash
    assert dirty == capture_git_source_provenance(repository)
    with pytest.raises(RuntimeError, match="clean committed Git HEAD"):
        require_clean_committed_source(dirty)
    serialized = json.dumps(dirty, sort_keys=True)
    assert "SOURCE_SENTINEL" not in serialized
    assert "module.py" not in serialized


def test_unavailable_repository_is_recorded_without_file_or_error_details(
    tmp_path: Path,
) -> None:
    provenance = capture_git_source_provenance(tmp_path)

    assert provenance == {
        "schema_version": 1,
        "scope": SOURCE_SCOPE,
        "available": False,
        "head_commit": None,
        "head_tree": None,
        "is_clean": None,
        "tracked_diff_sha256": None,
    }
    with pytest.raises(RuntimeError, match="clean committed Git HEAD"):
        require_clean_committed_source(provenance)


def test_strict_resume_rejects_commit_tree_or_diff_drift(tmp_path: Path) -> None:
    repository = _committed_repository(tmp_path)
    current = capture_git_source_provenance(repository)
    metadata = {"source_provenance": current}
    assert source_provenance_resume_contract_mismatch(metadata, current) is None

    for field, value in (
        ("head_commit", "0" * 40),
        ("head_tree", "1" * 40),
        ("tracked_diff_sha256", "2" * 64),
        ("is_clean", False),
    ):
        changed = {**current, field: value}
        mismatch = source_provenance_resume_contract_mismatch(metadata, changed)
        assert mismatch == {"checkpoint": current, "runtime": changed}

    missing = source_provenance_resume_contract_mismatch({}, current)
    assert missing == {"checkpoint": None, "runtime": current}
