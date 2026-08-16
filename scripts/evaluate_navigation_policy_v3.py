#!/usr/bin/env python3
"""Evaluate V3 on scene-disjoint offline expert traces."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from semantic_3d_chat.config import load_config
from semantic_3d_chat.training.train_navigation_policy_v3 import (
    evaluate_navigation_policy_v3,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/navigation_policy_v3.yaml")
    parser.add_argument("--checkpoint", default="data_gemma4/checkpoints/navigation_policy_v3")
    parser.add_argument(
        "--output",
        default="reports/gemma4/metrics/navigation_policy_v3_offline_evaluation.json",
    )
    args = parser.parse_args(argv)
    result = evaluate_navigation_policy_v3(load_config(args.config), checkpoint=args.checkpoint)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
