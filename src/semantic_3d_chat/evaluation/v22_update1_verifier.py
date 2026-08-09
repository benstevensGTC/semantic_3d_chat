"""Pinned V22 exact-update-one wrapper around the audited V21 verifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.evaluation.phase_aware_local_field_profile import (
    V22_LOCAL_FIELD_PROFILE,
)
from semantic_3d_chat.evaluation.v21_update1_verifier import (
    V21Update1Violation,
    _atomic_json,
    verify_update1,
)

V22Update1Violation = V21Update1Violation


def verify_v22_update1(config, preflight_path, checkpoint_path):
    return verify_update1(
        config,
        preflight_path,
        checkpoint_path,
        profile=V22_LOCAL_FIELD_PROFILE,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=V22_LOCAL_FIELD_PROFILE.config_path)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify_v22_update1(load_config(args.config), args.preflight, args.checkpoint)
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    _atomic_json(output, report)
    print(json.dumps({"phase": "v22_update1_verified", "report": str(output)}, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - local checkpoint command
    main()


__all__ = ["V22Update1Violation", "verify_v22_update1"]
