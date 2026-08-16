#!/usr/bin/env python3
"""Prove the learned-controller checkpoint loads with oracle data unavailable."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.robot.navigation_policy import (
    load_navigation_policy_checkpoint,
)


def _rooted(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return Path(os.path.abspath(value if value.is_absolute() else PROJECT_ROOT / value))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/experiments/navigation_policy_v2.yaml"
    )
    parser.add_argument(
        "--checkpoint", default="data_gemma4/checkpoints/navigation_policy_v2"
    )
    parser.add_argument(
        "--output",
        default="reports/gemma4/metrics/navigation_policy_v2_runtime_audit_local.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    settings = config["navigation_policy"]
    checkpoint = _rooted(args.checkpoint)
    oracle = PROJECT_ROOT / "data" / "oracle"
    qa = PROJECT_ROOT / "data" / "qa"
    training = PROJECT_ROOT / "data_gemma4" / "training"
    audit = FileAccessAudit(
        [oracle, qa, training, PROJECT_ROOT / "reports/gemma4/scorer_only"],
        forbidden_component_names={"oracle", "qa", "training", "scorer_only"},
        block_forbidden=True,
    )
    oracle_was_available = oracle.exists()
    detached = oracle.with_name(".oracle_navigation_policy_audit_detached")
    if detached.exists():
        raise FileExistsError(detached)
    try:
        if oracle_was_available:
            os.replace(oracle, detached)
        with audit:
            _controller, metadata = load_navigation_policy_checkpoint(
                checkpoint,
                expected_hidden_size=int(settings["hidden_size"]),
                expected_model_id=str(config["language"]["model_id"]),
                expected_model_revision=str(config["language"]["revision"]),
                device="cpu",
                audit=audit,
            )
        audit.assert_clean()
    finally:
        if oracle_was_available and detached.exists():
            os.replace(detached, oracle)
    payload = {
        "schema": "semantic_3d_chat.navigation_policy_runtime_audit.v1",
        "passed": True,
        "blocking_enabled": True,
        "oracle_temporarily_unavailable": oracle_was_available,
        "oracle_restored": oracle.exists() if oracle_was_available else True,
        "loaded_file_count": len(audit.unique_paths),
        "loaded_files": audit.unique_paths,
        "loaded_file_inventory_sha256": hashlib.sha256(
            "\n".join(audit.unique_paths).encode("utf-8")
        ).hexdigest(),
        "forbidden_accesses": audit.forbidden_accesses(),
        "checkpoint_weights_sha256": metadata["weights_sha256"],
        "oracle_inputs_at_runtime": metadata["oracle_inputs_at_runtime"],
        "environmental_text_inputs": metadata["environmental_text_inputs"],
    }
    destination = _rooted(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
