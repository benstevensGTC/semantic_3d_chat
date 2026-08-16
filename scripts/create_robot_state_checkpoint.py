#!/usr/bin/env python3
"""Create the sanitized deterministic numeric robot-state checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic_3d_chat.robot.state_checkpoint import (
    create_robot_state_checkpoint,
    load_robot_state_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-dim", type=int, default=1536)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--token-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--output-scale", type=float, default=0.02)
    parser.add_argument("--report")
    args = parser.parse_args()
    if Path(args.output).exists():
        _, state_sha256, metadata = load_robot_state_checkpoint(
            args.output,
            expected_output_dim=args.output_dim,
        )
        requested = {
            "hidden_dim": args.hidden_dim,
            "token_count": args.token_count,
            "initialization_seed": args.seed,
            "output_scale": args.output_scale,
        }
        if any(metadata[key] != value for key, value in requested.items()):
            raise ValueError("Existing robot-state checkpoint differs from requested settings")
        result = {
            **metadata,
            "checkpoint": str(Path(args.output)),
            "robot_state_encoder_sha256": state_sha256,
            "cached": True,
        }
    else:
        result = create_robot_state_checkpoint(
            args.output,
            output_dim=args.output_dim,
            hidden_dim=args.hidden_dim,
            token_count=args.token_count,
            seed=args.seed,
            output_scale=args.output_scale,
        )
        _, state_sha256, _ = load_robot_state_checkpoint(
            args.output,
            expected_output_dim=args.output_dim,
        )
        result["robot_state_encoder_sha256"] = state_sha256
        result["cached"] = False
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.report is not None:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
