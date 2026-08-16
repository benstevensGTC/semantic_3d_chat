#!/usr/bin/env python3
"""Run the read-only CPU/tokenizer preflight for the unsealed V3 draft."""

from __future__ import annotations

import argparse
import json

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.gemma4_tool_decoder_v3_preflight import (
    load_local_tokenizer_for_v3,
    run_tool_decoder_v3_preflight,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-json",
        action="store_true",
        help="Print the complete mutable draft instead of a short status summary.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    report = run_tool_decoder_v3_preflight(
        PROJECT_ROOT,
        tokenizer=load_local_tokenizer_for_v3(),
    )
    if arguments.full_json:
        value = report
    else:
        value = {
            "status": report["status"],
            "passed": report["passed"],
            "v2_2_terminal_sha256": report["v2_2_terminal_negative"]["sha256"],
            "v2_runtime_checkpoint_absent": report["v2_2_terminal_negative"][
                "runtime_checkpoint_absent"
            ],
            "training_rows_read": report["training_rows"]["count"],
            "heldout_rows_read": report["training_rows"]["heldout_rows_read"],
            "optimizer_updates_preregistered": report["schedule"][
                "optimizer_updates"
            ],
            "optimizer_steps_executed": report["execution"]["optimizer_steps"],
            "semantic_weight_fraction_increase": report["token_role_audit"][
                "semantic_weight_fraction_increase"
            ],
            "training_authorized": report["execution"]["training_authorized"],
        }
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
