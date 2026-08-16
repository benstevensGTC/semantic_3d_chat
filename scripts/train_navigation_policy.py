#!/usr/bin/env python3
"""Train and gate the compact continuous-input navigation controller."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from semantic_3d_chat.config import load_config
from semantic_3d_chat.training.train_navigation_policy import train_navigation_policy


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/experiments/navigation_policy_v1.yaml"
    )
    parser.add_argument("--dataset")
    parser.add_argument("--checkpoint")
    parser.add_argument("--metrics")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = train_navigation_policy(
        load_config(args.config),
        dataset=args.dataset,
        checkpoint=args.checkpoint,
        metrics_path=args.metrics,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
