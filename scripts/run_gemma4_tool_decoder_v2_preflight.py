#!/usr/bin/env python3
"""Print the read-only Gemma-4 tool-decoder V2 preflight as JSON."""

from __future__ import annotations

import argparse
import json

from semantic_3d_chat.evaluation.gemma4_tool_decoder_preregistration_v2 import (
    run_tiny_cpu_backward_smoke_v2,
    run_tool_decoder_preflight_v2,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-weight-hash",
        action="store_true",
        help="Recompute the 10.25-GB model hash instead of trusting its content-addressed path.",
    )
    parser.add_argument(
        "--tiny-cpu-smoke",
        action="store_true",
        help="Also run one tiny CPU-only forward/backward microbatch (zero optimizer steps).",
    )
    args = parser.parse_args()
    report: dict[str, object] = {
        "structural_preflight": run_tool_decoder_preflight_v2(
            full_weight_hash=args.full_weight_hash
        )
    }
    if args.tiny_cpu_smoke:
        report["tiny_cpu_smoke"] = run_tiny_cpu_backward_smoke_v2()
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
