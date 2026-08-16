#!/usr/bin/env python3
"""Evaluate the accepted Navigation V4 checkpoint on its held-out scenes."""

from __future__ import annotations

import argparse
import json

from semantic_3d_chat.config import load_config
from semantic_3d_chat.training.train_navigation_policy_v4 import (
    evaluate_navigation_policy_v4,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/navigation_policy_v4.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--output",
        default="reports/gemma4/metrics/navigation_policy_v4_offline_evaluation.json",
    )
    args = parser.parse_args()
    result = evaluate_navigation_policy_v4(
        load_config(args.config),
        checkpoint=args.checkpoint,
        metrics_path=args.output,
    )
    print(json.dumps(result["validation"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
