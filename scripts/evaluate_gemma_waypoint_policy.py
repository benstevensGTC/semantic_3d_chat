#!/usr/bin/env python3
"""Evaluate primary and causal-control movement decisions through local Gemma."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.training.gemma_waypoint_policy import (
    CONTROL_CONDITIONS,
    DEFAULT_CONTROL_CONDITIONS,
    ActualGemmaWaypointForward,
    evaluate_waypoint_controls,
    load_actual_waypoint_stack,
    load_waypoint_data_from_config,
    validate_waypoint_settings,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/experiments/gemma_waypoint_policy_v1.yaml"
    )
    parser.add_argument("--dataset")
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument(
        "--condition",
        action="append",
        choices=CONTROL_CONDITIONS,
        default=None,
        help=(
            "control conditions to evaluate; repeatable. Defaults to the "
            "historical four. 'primary' is always included first."
        ),
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="override gemma_waypoint_policy.control_sample_limit for this run",
    )
    parser.add_argument(
        "--output",
        default="reports/gemma4/metrics/gemma_waypoint_policy_controls.json",
    )
    args = parser.parse_args()
    if args.condition is None:
        conditions = DEFAULT_CONTROL_CONDITIONS
    else:
        requested = [name for name in args.condition if name != "primary"]
        conditions = ("primary", *dict.fromkeys(requested))
    config = load_config(args.config)
    settings = validate_waypoint_settings(config)
    dataset, cache = load_waypoint_data_from_config(config, dataset_path=args.dataset)
    checkpoint = args.checkpoint or str(settings["checkpoint_output"])
    language, controller, state_encoder, state_hash = load_actual_waypoint_stack(
        config, checkpoint=checkpoint
    )
    runner = ActualGemmaWaypointForward(
        language,
        controller,
        state_encoder,
        scene_token_count=int(settings["scene_token_count"]),
        robot_token_count=int(settings["robot_token_count"]),
        hidden_size=int(settings["hidden_size"]),
        state_dim=int(settings["state_dim"]),
        history_dim=int(settings["history_dim"]),
    )
    result = evaluate_waypoint_controls(
        runner,
        cache,
        dataset.split(args.split),
        sample_limit=(
            args.sample_limit
            if args.sample_limit is not None
            else settings.get("control_sample_limit")
        ),
        conditions=conditions,
    )
    result["robot_state_checkpoint_sha256"] = state_hash
    result["evaluated_split"] = args.split
    result["evaluated_dataset"] = args.dataset or str(settings["trace_dataset"])
    result["evaluated_checkpoint"] = checkpoint
    output = Path(args.output)
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
