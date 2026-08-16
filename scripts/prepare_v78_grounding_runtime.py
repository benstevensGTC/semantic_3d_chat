#!/usr/bin/env python3
"""Prepare the optional two-file V78 numeric-grounding runtime release."""

from __future__ import annotations

import argparse
import json

from semantic_3d_chat.training.grounding_sidecar_v78_release import (
    materialize_v78_runtime_release,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="reports/gemma4/artifacts/v78_grounding_sidecar_diagnostic",
    )
    parser.add_argument(
        "--destination",
        default=(
            "data_gemma4/runtime/checkpoints/"
            "gemma4_v78_grounding_diagnostic_release_v1"
        ),
    )
    parser.add_argument(
        "--base-checkpoint",
        default="data_gemma4/runtime/checkpoints/gemma4_v54_release_v1",
    )
    parser.add_argument(
        "--runtime-config",
        default="configs/runtime/gemma4_v56_question_control.yaml",
    )
    args = parser.parse_args()
    report = materialize_v78_runtime_release(
        args.source,
        args.destination,
        base_checkpoint=args.base_checkpoint,
        runtime_config=args.runtime_config,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
