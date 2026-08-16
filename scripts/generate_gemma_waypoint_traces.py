#!/usr/bin/env python3
"""Generate authenticated offline teachers for Gemma waypoint decisions."""

from __future__ import annotations

import argparse
import json

from semantic_3d_chat.config import load_config
from semantic_3d_chat.training.gemma_waypoint_trace_generator import (
    generate_gemma_waypoint_trace_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/gemma_waypoint_policy_v1.yaml",
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "live", "demo", "full", "production", "operator"),
        default="smoke",
        help=(
            "smoke is the fast conversion proof; live is the compact scene_000001 "
            "fit; demo is a larger scene_000001 fit; full uses every V3 scene; "
            "production combines the live room with disjoint training rooms; "
            "operator retains every exact-start live-room step"
        ),
    )
    parser.add_argument(
        "--destination",
        default=None,
        help="must be a new directory below a path component named training",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["gemma_waypoint_traces"]
    destination = args.destination or f"{settings['output_root']}_{args.profile}"
    manifest = generate_gemma_waypoint_trace_dataset(
        config,
        destination,
        profile=args.profile,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
