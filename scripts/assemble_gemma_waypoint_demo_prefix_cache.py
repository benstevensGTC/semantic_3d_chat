#!/usr/bin/env python3
"""Build the sanitized scene_000001 plus held-out-validation prefix cache."""

from __future__ import annotations

import argparse
import json

from semantic_3d_chat.config import load_config
from semantic_3d_chat.training.gemma_waypoint_policy import (
    assemble_demo_scene_prefix_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/experiments/gemma_waypoint_policy_v1.yaml"
    )
    parser.add_argument(
        "--live-prefix",
        default="data_gemma4/training/gemma_waypoint_policy/scene_000001_prefix.safetensors",
    )
    parser.add_argument(
        "--reference-cache",
        default="data_gemma4/scene_tokens/v56_question_control_full_prefixes",
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    policy = config["gemma_waypoint_policy"]
    validation = config["gemma_waypoint_traces"]["profiles"]["demo"][
        "validation_scene_ids"
    ]
    result = assemble_demo_scene_prefix_cache(
        args.output or policy["prefix_cache_root"],
        live_scene_id="scene_000001",
        live_prefix_path=args.live_prefix,
        reference_cache_root=args.reference_cache,
        validation_scene_ids=validation,
        expected_token_count=int(policy["scene_token_count"]),
        expected_hidden_size=int(policy["hidden_size"]),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
