#!/usr/bin/env python3
"""Audit V3 checkpoint loading with the oracle directory unavailable."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.robot.navigation_policy_v3 import (
    load_navigation_policy_v3_checkpoint,
)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/navigation_policy_v3.yaml")
    parser.add_argument("--checkpoint", default="data_gemma4/checkpoints/navigation_policy_v3")
    parser.add_argument(
        "--output",
        default="reports/gemma4/metrics/navigation_policy_v3_runtime_audit.json",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    settings = config["navigation_policy_v3"]
    oracle = PROJECT_ROOT / "data" / "oracle"
    detached = oracle.with_name(".oracle_navigation_policy_v3_audit_detached")
    if detached.exists():
        raise FileExistsError(f"Stale detached oracle path exists: {detached}")
    audit = FileAccessAudit(
        [
            oracle,
            PROJECT_ROOT / "data_gemma4" / "training",
            PROJECT_ROOT / "reports" / "gemma4" / "scorer_only",
        ],
        forbidden_component_names={"oracle", "qa", "training", "scorer_only"},
        block_forbidden=True,
    )
    oracle_renamed = False
    try:
        os.replace(oracle, detached)
        oracle_renamed = True
        with audit:
            _controller, metadata = load_navigation_policy_v3_checkpoint(
                args.checkpoint,
                expected_hidden_size=int(settings["hidden_size"]),
                expected_model_id=str(config["language"]["model_id"]),
                expected_model_revision=str(config["language"]["revision"]),
                audit=audit,
            )
        audit.assert_clean()
    finally:
        if oracle_renamed:
            os.replace(detached, oracle)
    payload = {
        "schema": "semantic_3d_chat.navigation_policy_v3_runtime_audit.v3",
        "passed": True,
        "oracle_directory_unavailable_during_load": True,
        "oracle_directory_restored": oracle.is_dir() and not detached.exists(),
        "runtime_required_files": metadata["runtime_required_files"],
        "loaded_files": audit.unique_paths,
        "loaded_file_names": sorted(Path(path).name for path in audit.unique_paths),
        "forbidden_accesses": audit.forbidden_accesses(),
        "oracle_inputs_at_runtime": metadata["oracle_inputs_at_runtime"],
        "environmental_text_inputs_at_runtime": metadata["environmental_text_inputs"],
        "continuous_semantic_grounding_required": metadata[
            "continuous_semantic_grounding_required"
        ],
        "query_dependent_grounding_navigation_only": metadata[
            "query_dependent_grounding_navigation_only"
        ],
        "weights_sha256": metadata["weights_sha256"],
    }
    _atomic_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
