#!/usr/bin/env python3
"""Execute the sealed V2.2 training, evaluation, probe, and publication arm."""

from __future__ import annotations

import argparse
import json

from semantic_3d_chat.config import load_config
from semantic_3d_chat.evaluation.gemma4_tool_decoder_training_authorization_v2_2 import (
    TRAINING_RELEASE_PATH,
)
from semantic_3d_chat.training.train_gemma4_tool_decoder_v2 import (
    train_gemma4_tool_decoder_v2,
)

DEFAULT_CONFIG = "configs/experiments/gemma4_embodied_tool_decoder_v2.yaml"
DEFAULT_REPORT = (
    "reports/gemma4/metrics/gemma4_embodied_tool_decoder_training_v2_2.json"
)
DEFAULT_CHECKPOINT = "data_gemma4/checkpoints/gemma4_embodied_tool_decoder_v2/final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--authorization", default=TRAINING_RELEASE_PATH)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    report = train_gemma4_tool_decoder_v2(
        load_config(arguments.config),
        authorization=arguments.authorization,
        report_path=arguments.report,
        runtime_checkpoint=arguments.checkpoint,
    )
    print(
        json.dumps(
            {
                "status": report.get("status"),
                "report": arguments.report,
                "checkpoint": arguments.checkpoint
                if report.get("runtime_checkpoint_published") is True
                else None,
                "optimizer_updates": report.get("optimizer_updates"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
