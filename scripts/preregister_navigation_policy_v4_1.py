#!/usr/bin/env python3
"""Seal the one-change Navigation V4.1 recovery before its exact rerun."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic_3d_chat.config import load_config
from semantic_3d_chat.evaluation.navigation_policy_v41_preregistration import (
    write_navigation_policy_v41_preregistration,
)
from semantic_3d_chat.training.train_navigation_policy_v4 import (
    prepare_navigation_policy_v4_data,
    validate_navigation_policy_v4_settings,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/experiments/navigation_policy_v4_1.yaml"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    settings = validate_navigation_policy_v4_settings(config)
    manifest, prepared = prepare_navigation_policy_v4_data(
        config, Path(str(settings["source_trace_dataset"])).resolve()
    )
    path, digest = write_navigation_policy_v41_preregistration(
        str(settings["preregistration"]),
        config,
        source_v3_dataset_sha256=str(manifest["dataset_sha256"]),
        v4_dataset_sha256=prepared.dataset_sha256,
        map_sha256=prepared.map_sha256,
    )
    print(
        json.dumps(
            {
                "status": "sealed_before_v4_1_training",
                "path": str(path),
                "sha256": digest,
                "single_arm": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
