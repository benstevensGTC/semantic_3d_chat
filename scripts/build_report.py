"""Build the artifact-backed research report without running inference."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.evaluation.reporting import build_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble reports/final_report.md and figures from measurements already "
            "present on disk. Missing experiments are marked as not measured."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--scene", help="Opaque scene ID; defaults to the configured scene")
    parser.add_argument("--output", type=Path, help="Optional Markdown output path")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    result = build_report(
        PROJECT_ROOT,
        config,
        scene_id=args.scene,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
