#!/usr/bin/env python3
"""Generate V3 training-only numeric grounded-target traces."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from semantic_3d_chat.config import load_config
from semantic_3d_chat.training.navigation_target_trace_v3 import (
    generate_navigation_target_trace_v3,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/navigation_policy_v3.yaml")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    settings = config["navigation_policy_v3"]
    result = generate_navigation_target_trace_v3(config, settings["trace_output"])
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
