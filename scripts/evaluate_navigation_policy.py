#!/usr/bin/env python3
"""Offline score of a learned navigation checkpoint on held-out trace scenes."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from semantic_3d_chat.config import load_config
from semantic_3d_chat.training.train_navigation_policy import (
    evaluate_navigation_policy,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/experiments/navigation_policy_v1.yaml"
    )
    parser.add_argument("--dataset")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_navigation_policy(
        load_config(args.config),
        dataset=args.dataset,
        checkpoint=args.checkpoint,
    )
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
