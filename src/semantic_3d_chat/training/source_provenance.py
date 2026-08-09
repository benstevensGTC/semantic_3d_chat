"""Content-free Git provenance for reproducible adapter training."""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_OBJECT_ID = re.compile(r"[0-9a-f]{40,64}")
_SOURCE_PATHSPECS = (
    ".",
    ":(exclude)reports/**",
    ":(exclude)data/**",
    ":(exclude)data_gemma4/**",
)
SOURCE_SCOPE = "repository_excluding_generated_artifacts_v1"


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=False,
        capture_output=True,
    )


def _unavailable() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scope": SOURCE_SCOPE,
        "available": False,
        "head_commit": None,
        "head_tree": None,
        "is_clean": None,
        "tracked_diff_sha256": None,
    }


def capture_git_source_provenance(repository: str | Path) -> dict[str, Any]:
    """Capture commit/tree and a scoped diff hash without persisting file data."""

    root = Path(repository).resolve()
    try:
        head_result = _git(root, "rev-parse", "--verify", "HEAD^{commit}")
        tree_result = _git(root, "rev-parse", "--verify", "HEAD^{tree}")
    except OSError:
        return _unavailable()
    if head_result.returncode != 0 or tree_result.returncode != 0:
        return _unavailable()
    head_commit = head_result.stdout.decode("ascii", errors="strict").strip().casefold()
    head_tree = tree_result.stdout.decode("ascii", errors="strict").strip().casefold()
    if not _OBJECT_ID.fullmatch(head_commit) or not _OBJECT_ID.fullmatch(head_tree):
        return _unavailable()
    diff_result = _git(
        root,
        "diff",
        "HEAD",
        "--binary",
        "--no-ext-diff",
        "--",
        *_SOURCE_PATHSPECS,
    )
    status_result = _git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--",
        *_SOURCE_PATHSPECS,
    )
    if diff_result.returncode != 0 or status_result.returncode != 0:
        return _unavailable()
    return {
        "schema_version": 1,
        "scope": SOURCE_SCOPE,
        "available": True,
        "head_commit": head_commit,
        "head_tree": head_tree,
        "is_clean": status_result.stdout == b"",
        "tracked_diff_sha256": hashlib.sha256(diff_result.stdout).hexdigest(),
    }


def require_clean_committed_source(provenance: Mapping[str, Any]) -> None:
    """Require a valid committed HEAD and no source-scoped working changes."""

    valid = (
        provenance.get("schema_version") == 1
        and provenance.get("scope") == SOURCE_SCOPE
        and provenance.get("available") is True
        and isinstance(provenance.get("head_commit"), str)
        and _OBJECT_ID.fullmatch(str(provenance["head_commit"])) is not None
        and isinstance(provenance.get("head_tree"), str)
        and _OBJECT_ID.fullmatch(str(provenance["head_tree"])) is not None
        and provenance.get("is_clean") is True
        and isinstance(provenance.get("tracked_diff_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", str(provenance["tracked_diff_sha256"])) is not None
    )
    if not valid:
        raise RuntimeError(
            "Enabled LoRA training requires a valid clean committed Git HEAD in the "
            "source provenance scope"
        )


def source_provenance_resume_contract_mismatch(
    metadata: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Return a content-free mismatch for strict LoRA resume preflight."""

    checkpoint = metadata.get("source_provenance")
    runtime = dict(current)
    if checkpoint == runtime:
        return None
    return {"checkpoint": checkpoint, "runtime": runtime}


__all__ = [
    "SOURCE_SCOPE",
    "capture_git_source_provenance",
    "require_clean_committed_source",
    "source_provenance_resume_contract_mismatch",
]
