#!/usr/bin/env python3
"""Train the MOVE_TO/FACE/STOP heads from cached actual-Gemma states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.training.gemma_waypoint_policy import (
    load_actual_waypoint_stack,
    load_waypoint_data_from_config,
    train_waypoint_controller,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/experiments/gemma_waypoint_policy_v1.yaml"
    )
    parser.add_argument("--dataset")
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--metrics",
        default="reports/gemma4/metrics/gemma_waypoint_policy_training.json",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    dataset, cache = load_waypoint_data_from_config(config, dataset_path=args.dataset)
    language, controller, state_encoder, _state_hash = load_actual_waypoint_stack(config)
    result = train_waypoint_controller(
        config,
        language,
        controller,
        state_encoder,
        dataset,
        cache,
        checkpoint=args.checkpoint,
    )
    output = Path(args.metrics)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
