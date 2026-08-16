#!/usr/bin/env python3
"""Run the single sealed navigation-policy V4 training arm."""

from __future__ import annotations

import argparse

from semantic_3d_chat.config import load_config
from semantic_3d_chat.training.train_navigation_policy_v4 import (
    train_navigation_policy_v4,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/navigation_policy_v4.yaml")
    parser.add_argument(
        "--metrics",
        default="reports/gemma4/metrics/navigation_policy_v4_training.json",
    )
    args = parser.parse_args()
    result = train_navigation_policy_v4(load_config(args.config), metrics_path=args.metrics)
    print(result["status"])
    return 0 if result["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
