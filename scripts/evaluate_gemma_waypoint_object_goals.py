#!/usr/bin/env python3
"""Capture and score fresh live Gemma object-directed waypoint goals.

Use ``capture`` while a freshly started rover backend is running.  That
subcommand deliberately has no oracle-related option.  Restart the backend
before capturing the second goal.  Only after the live captures finish, use
``score`` to open evaluation-only oracle geometry and compute metrics.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from semantic_3d_chat.evaluation.gemma_waypoint_object_eval import (
    GOALS,
    _atomic_json,
    capture_live_goal,
    score_runtime_files,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    capture = commands.add_parser(
        "capture",
        help="Call the loopback public API; this stage cannot receive an oracle path.",
    )
    capture.add_argument("--goal", required=True, choices=sorted(GOALS))
    capture.add_argument("--base-url", default="http://127.0.0.1:8770")
    capture.add_argument("--timeout-seconds", type=float, default=1_800.0)
    capture.add_argument("--output", type=Path, required=True)

    score = commands.add_parser(
        "score",
        help="Validate completed captures, then read oracle geometry and score them.",
    )
    score.add_argument("--runtime", type=Path, action="append", required=True)
    score.add_argument("--oracle-root", type=Path, default=Path("data/oracle"))
    score.add_argument("--maximum-face-yaw-error-degrees", type=float, default=20.0)
    score.add_argument("--minimum-chair-progress-m", type=float, default=0.25)
    score.add_argument("--maximum-chair-bbox-standoff-m", type=float, default=0.60)
    score.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "capture":
        report = capture_live_goal(
            args.base_url,
            args.goal,
            timeout_seconds=args.timeout_seconds,
        )
    else:
        report = score_runtime_files(
            args.runtime,
            oracle_root=args.oracle_root,
            maximum_face_yaw_error_degrees=args.maximum_face_yaw_error_degrees,
            minimum_chair_progress_m=args.minimum_chair_progress_m,
            maximum_chair_bbox_standoff_m=args.maximum_chair_bbox_standoff_m,
        )
    output = args.output.expanduser().resolve()
    _atomic_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
