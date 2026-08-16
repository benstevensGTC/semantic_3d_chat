#!/usr/bin/env python3
"""Materialize the create-once anonymous V2 clearance cache."""

from __future__ import annotations

import json

from semantic_3d_chat.config import load_config
from semantic_3d_chat.training.gemma4_tool_decoder_v2_clearance import (
    materialize_clearance_cache_v2,
)


def main() -> None:
    config = load_config("configs/experiments/gemma4_embodied_tool_decoder_v2.yaml")
    report = materialize_clearance_cache_v2(
        config,
        trace_root="data_gemma4/training/navigation_policy_v3",
        map_root="data_gemma4/maps",
        output_directory="data_gemma4/training/gemma4_embodied_tool_decoder_v2",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
