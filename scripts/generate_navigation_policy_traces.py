#!/usr/bin/env python3
"""Generate physically isolated oracle-side navigation action traces."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from semantic_3d_chat.config import load_config
from semantic_3d_chat.training.navigation_trace_generator import (
    generate_navigation_trace_dataset,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/experiments/navigation_policy_v1.yaml"
    )
    parser.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    settings = config["navigation_policy"]
    output = args.output or settings["trace_output"]
    result = generate_navigation_trace_dataset(config, output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
