#!/usr/bin/env python3
"""Audit a sealed navigation run without opening oracle or QA metadata."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from semantic_3d_chat.evaluation.llm_navigation_benchmark import (
    continuous_context_metrics,
    file_sha256,
    validate_navigation_journal,
)

_BLOCKED_COMPONENTS = frozenset({"oracle", "qa", "training", "scorer_only"})


def _safe_path(value: str | Path, *, must_exist: bool) -> Path:
    path = Path(os.path.abspath(Path(value).expanduser()))
    if _BLOCKED_COMPONENTS & {part.casefold() for part in path.parts}:
        raise ValueError("Continuous-context audit cannot enter supervision trees")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError("Continuous-context audit paths cannot contain symlinks")
    if must_exist and (not path.is_file() or path.is_symlink()):
        raise FileNotFoundError(path)
    return path


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--journal",
        default="reports/gemma4/predictions/llm_navigation_scene_000001_learned_v3.json",
    )
    parser.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    journal_path = _safe_path(args.journal, must_exist=True)
    journal = validate_navigation_journal(
        json.loads(journal_path.read_text(encoding="utf-8")),
        require_complete=True,
    )
    metrics = continuous_context_metrics(journal)
    try:
        journal_display = journal_path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        journal_display = str(journal_path)
    report = {
        "schema": "semantic_3d_chat.navigation_continuous_context_audit.v1",
        "passed": metrics["passed"],
        "journal_path": journal_display,
        "journal_file_sha256": file_sha256(journal_path),
        "journal_root_sha256": journal["journal_sha256"],
        "metrics": metrics,
        "oracle_files_opened": 0,
        "qa_files_opened": 0,
    }
    if args.output is not None:
        _atomic_json(_safe_path(args.output, must_exist=False), report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if metrics["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
