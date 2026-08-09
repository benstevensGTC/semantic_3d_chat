"""Pinned V22 isolated update-eight extension and final selector wrapper."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from semantic_3d_chat.evaluation.phase_aware_local_field_profile import (
    V22_LOCAL_FIELD_PROFILE,
)
from semantic_3d_chat.evaluation.v21_extension_controller import (
    V21ExtensionViolation,
    prepare_extension_launch,
    select_final_extension,
    write_report,
)

V22ExtensionViolation = V21ExtensionViolation


def prepare_v22_extension_launch(config_path, screen_path, *, current_provenance=None):
    return prepare_extension_launch(
        config_path,
        screen_path,
        current_provenance=current_provenance,
        profile=V22_LOCAL_FIELD_PROFILE,
    )


def select_v22_final_extension(manifest_path, *, current_provenance=None):
    return select_final_extension(
        manifest_path,
        current_provenance=current_provenance,
        profile=V22_LOCAL_FIELD_PROFILE,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", type=Path, default=V22_LOCAL_FIELD_PROFILE.config_path)
    prepare.add_argument("--screen", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    final = subparsers.add_parser("select-final")
    final.add_argument("--manifest", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        report = prepare_v22_extension_launch(args.config, args.screen)
    else:
        report = select_v22_final_extension(args.manifest)
    destination = write_report(report, args.output)
    print(
        json.dumps(
            {
                "command": args.command,
                "output": str(destination),
                "authorized": report.get("authorized"),
                "decision": report.get("decision"),
                "extension_output_namespace": V22_LOCAL_FIELD_PROFILE.extension_namespace,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - local checkpoint command
    raise SystemExit(main())


__all__ = [
    "V22ExtensionViolation",
    "prepare_v22_extension_launch",
    "select_v22_final_extension",
]
