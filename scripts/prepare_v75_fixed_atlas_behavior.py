#!/usr/bin/env python3
"""Create numeric probes and separated historical smoke manifests."""

from __future__ import annotations

import argparse
import json

from semantic_3d_chat.evaluation.v75_fixed_atlas_artifacts import prepare_artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/gemma4_v75_fixed_prefix_atlas_prepare.yaml",
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            prepare_artifacts(args.config),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
