"""Export QA supervision into a strict questions-only inference manifest.

This preparation process is evaluation-side and may read the QA split.  The
prediction processes never import the QA reader or open the QA source; they are
given only the sanitized artifact written here.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from semantic_3d_chat.config import artifact_root, load_config, reports_root
from semantic_3d_chat.evaluation.question_manifest import (
    QuestionManifest,
    build_question_manifest,
    load_question_manifest,
    sha256_file,
)
from semantic_3d_chat.evaluation.run import load_jsonl


def prepare_question_manifest(
    qa_path: str | Path,
    output_path: str | Path,
    *,
    force: bool = False,
) -> QuestionManifest:
    """Read QA once and atomically write its three-field inference projection."""

    source = Path(qa_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"QA source does not exist: {source}")
    if source == destination:
        raise ValueError("QA source and question-manifest destination must differ")
    if destination.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite question manifest: {destination}")
    records = load_jsonl(source)
    manifest = build_question_manifest(records, source_qa_sha256=sha256_file(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(manifest.as_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return load_question_manifest(destination)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--qa", type=Path, help="Evaluation-only source QA JSONL")
    parser.add_argument("--output", type=Path, help="Sanitized inference manifest")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_config(args.config)
    source = args.qa or artifact_root(config, "qa") / f"{args.split}.jsonl"
    destination = args.output or reports_root(config) / "questions" / f"{args.split}.json"
    manifest = prepare_question_manifest(source, destination, force=args.force)
    summary: dict[str, Any] = {
        "phase": "prepare_questions_complete",
        "manifest_path": str(manifest.manifest_path),
        "manifest_sha256": manifest.manifest_sha256,
        "questions_sha256": manifest.questions_sha256,
        "source_qa_sha256": manifest.source_qa_sha256,
        "question_count": manifest.question_count,
        "scene_count": manifest.scene_count,
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
