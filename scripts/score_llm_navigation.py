#!/usr/bin/env python3
"""Score a completed navigation journal after inference has exited."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.llm_navigation_benchmark import (
    score_navigation_journal,
)


def _rooted(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return Path(os.path.abspath(value if value.is_absolute() else PROJECT_ROOT / value))


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
        default="reports/gemma4/predictions/llm_navigation_scene_000001.json",
    )
    parser.add_argument(
        "--scoring-spec",
        default="configs/benchmarks/oracle/llm_navigation_scene_000001.json",
    )
    parser.add_argument(
        "--scene-oracle",
        default="data/oracle/scene_000001/oracle.json",
    )
    parser.add_argument(
        "--output",
        default="reports/gemma4/metrics/llm_navigation_scene_000001.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = _rooted(args.output)
    result = score_navigation_journal(
        _rooted(args.journal),
        _rooted(args.scoring_spec),
        _rooted(args.scene_oracle),
    )
    _atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
