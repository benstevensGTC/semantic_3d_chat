#!/usr/bin/env python3
"""Create the immutable Navigation V4 failed-attempt incident artifact."""

from __future__ import annotations

import argparse
import json

from semantic_3d_chat.evaluation.navigation_policy_v4_incident import (
    write_navigation_policy_v4_incident,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="reports/gemma4/metrics/navigation_policy_v4_training_incident.json",
    )
    args = parser.parse_args()
    path, digest = write_navigation_policy_v4_incident(args.output)
    print(json.dumps({"path": str(path), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
