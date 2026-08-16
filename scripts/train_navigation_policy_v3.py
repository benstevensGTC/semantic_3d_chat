#!/usr/bin/env python3
"""Train continuous-semantic navigation-policy V3."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from semantic_3d_chat.config import load_config
from semantic_3d_chat.training.train_navigation_policy_v3 import (
    train_navigation_policy_v3,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/navigation_policy_v3.yaml")
    parser.add_argument(
        "--metrics",
        default="reports/gemma4/metrics/navigation_policy_v3_training.json",
    )
    args = parser.parse_args(argv)
    result = train_navigation_policy_v3(load_config(args.config), metrics_path=args.metrics)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
