"""Pinned V22 four-update selector using the audited V21-family machinery."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from semantic_3d_chat.config import load_config
from semantic_3d_chat.evaluation.phase_aware_local_field_profile import (
    V22_LOCAL_FIELD_PROFILE,
)
from semantic_3d_chat.evaluation.v19_epoch_selector import (
    _load_json_strict,
    _reject_forbidden_input_path,
)
from semantic_3d_chat.evaluation.v21_epoch_selector import (
    V21EpochSelectorViolation,
    _parse_epoch_path,
    summarize_v21_epochs,
    write_selector_report,
)

V22EpochSelectorViolation = V21EpochSelectorViolation


def summarize_v22_epochs(
    config: Mapping[str, Any],
    selection: Mapping[str, Any],
    epoch_artifacts: Mapping[int, Mapping[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    return summarize_v21_epochs(
        config,
        selection,
        epoch_artifacts,
        profile=V22_LOCAL_FIELD_PROFILE,
        **kwargs,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=V22_LOCAL_FIELD_PROFILE.config_path)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--update1-report", type=Path, required=True)
    parser.add_argument("--epoch", action="append", type=_parse_epoch_path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    epoch_paths = dict(args.epoch)
    if len(epoch_paths) != len(args.epoch):
        parser.error("duplicate V22 epoch binding")
    _reject_forbidden_input_path(args.config)
    config = load_config(args.config)
    selection, selection_digest = _load_json_strict(args.selection)
    loaded = {epoch: _load_json_strict(path) for epoch, path in epoch_paths.items()}
    summary = summarize_v22_epochs(
        config,
        selection,
        {epoch: value for epoch, (value, _digest) in loaded.items()},
        update1_report_path=args.update1_report,
        selection_path=str(args.selection),
        selection_sha256=selection_digest,
        epoch_paths={epoch: str(path) for epoch, path in epoch_paths.items()},
        epoch_sha256={epoch: digest for epoch, (_value, digest) in loaded.items()},
    )
    destination = write_selector_report(summary, args.output)
    print(
        json.dumps(
            {
                "output": str(destination),
                "selected_epoch": summary["selected_epoch"],
                "continuation_authorized": summary["continuation_authorized"],
                "full_teacher_gate_passed": summary["full_teacher_gate_passed"],
                "greedy_audit_authorized": summary["greedy_audit_authorized"],
                "decision": summary["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - local checkpoint command
    raise SystemExit(main())


__all__ = ["V22EpochSelectorViolation", "summarize_v22_epochs"]
