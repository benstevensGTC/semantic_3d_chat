"""Seal the post-evaluation V4 serialization incident before any recovery edit."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT

ORIGINAL_PREREGISTRATION: Final[str] = (
    "reports/gemma4/metrics/navigation_policy_v4_preregistration.json"
)
ORIGINAL_PREREGISTRATION_SHA256: Final[str] = (
    "b855ee22bfbca6b5f709199e5b88937c6643c9ddbea39a102ebebc23f0a28c61"
)
ORIGINAL_CHECKPOINT: Final[str] = "data_gemma4/checkpoints/navigation_policy_v4"
ORIGINAL_TRAINING_REPORT: Final[str] = (
    "reports/gemma4/metrics/navigation_policy_v4_training.json"
)
INCIDENT_SCHEMA: Final[str] = (
    "semantic_3d_chat.navigation_policy_v4_training_incident.v1"
)


def _rooted(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_navigation_policy_v4_incident() -> dict[str, Any]:
    preregistration_path = _rooted(ORIGINAL_PREREGISTRATION)
    if (
        not preregistration_path.is_file()
        or preregistration_path.is_symlink()
        or _sha256(preregistration_path) != ORIGINAL_PREREGISTRATION_SHA256
    ):
        raise ValueError("Original V4 preregistration bytes differ")
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    if not isinstance(preregistration, dict):
        raise TypeError("Original V4 preregistration must be an object")
    source_hashes = preregistration.get("implementation_source_hashes")
    input_hashes = preregistration.get("input_artifact_hashes")
    if not isinstance(source_hashes, dict) or not isinstance(input_hashes, dict):
        raise TypeError("Original V4 preregistration hash inventories are unavailable")
    observed_source_hashes: dict[str, str] = {}
    for relative, expected in source_hashes.items():
        path = _rooted(relative)
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(f"Original V4 source changed before incident seal: {relative}")
        observed_source_hashes[str(relative)] = observed
    observed_input_hashes: dict[str, str] = {}
    for relative, expected in input_hashes.items():
        path = _rooted(relative)
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(f"Original V4 input changed before incident seal: {relative}")
        observed_input_hashes[str(relative)] = observed
    checkpoint = _rooted(ORIGINAL_CHECKPOINT)
    report = _rooted(ORIGINAL_TRAINING_REPORT)
    if checkpoint.exists() or report.exists():
        raise FileExistsError("Original V4 incident requires absent checkpoint and report")
    return {
        "schema": INCIDENT_SCHEMA,
        "status": "sealed_failed_attempt_no_checkpoint_or_training_report",
        "attempt": {
            "command": (
                "env PYTHONPATH=src /usr/bin/time -p .venv-gemma4/bin/python "
                "scripts/train_navigation_policy_v4.py --config "
                "configs/experiments/navigation_policy_v4.yaml"
            ),
            "python_pid": 99031,
            "time_wrapper_pid": 99029,
            "exit_code": 1,
            "elapsed_seconds": 48.89,
            "user_cpu_seconds": 60.12,
            "system_cpu_seconds": 27.28,
            "optimizer_constructed": True,
            "training_loop_completed": True,
            "held_out_evaluation_completed": True,
            "failure_stage": "strict_training_report_json_serialization",
        },
        "failure": {
            "exception_type": "ValueError",
            "exception_message": (
                "Out of range float values are not JSON compliant: nan"
            ),
            "terminal_traceback": [
                "scripts/train_navigation_policy_v4.py:22 main",
                (
                    "src/semantic_3d_chat/training/train_navigation_policy_v4.py:990 "
                    "train_navigation_policy_v4 -> _atomic_json(report_path, result)"
                ),
                (
                    "src/semantic_3d_chat/training/train_navigation_policy_v4.py:95 "
                    "_atomic_json -> json.dump(..., allow_nan=False)"
                ),
                "json/encoder.py:240 -> ValueError",
            ],
            "diagnosis": (
                "evaluate_prepared_v4 computed an unguarded mean over an empty "
                "targeted or targetless subgroup in a causal control condition"
            ),
            "model_acceptance_result_available": False,
            "model_acceptance_result_interpreted": False,
            "live_navigation_benchmark_opened": False,
            "live_navigation_oracle_opened": False,
        },
        "publication_state": {
            "checkpoint_path": ORIGINAL_CHECKPOINT,
            "checkpoint_absent": not checkpoint.exists(),
            "training_report_path": ORIGINAL_TRAINING_REPORT,
            "training_report_absent": not report.exists(),
            "partial_artifact_published": False,
        },
        "original_preregistration": {
            "path": ORIGINAL_PREREGISTRATION,
            "sha256": ORIGINAL_PREREGISTRATION_SHA256,
            "bytes_modified": False,
            "implementation_source_hashes": observed_source_hashes,
            "input_artifact_hashes": observed_input_hashes,
            "prepared_v4_dataset_sha256": preregistration["data"][
                "prepared_v4_dataset_sha256"
            ],
            "seed": preregistration["single_arm"]["seed"],
            "hyperparameters": preregistration["single_arm"]["hyperparameters"],
            "acceptance_gates": preregistration["acceptance_gates"],
        },
        "authorized_recovery_scope": {
            "protocol_version": "v4.1",
            "only_change": (
                "empty_targeted_or_targetless_metric_subgroup_accuracy_is_finite_zero"
            ),
            "matches_v3_evaluator_precedent": True,
            "training_hyperparameters_change": False,
            "training_seed_change": False,
            "training_data_change": False,
            "acceptance_gate_change": False,
            "new_preregistration_required_before_rerun": True,
        },
    }


def write_navigation_policy_v4_incident(path: str | Path) -> tuple[Path, str]:
    destination = _rooted(path)
    if destination.exists():
        raise FileExistsError(f"V4 incident already exists: {destination}")
    payload = build_navigation_policy_v4_incident()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination, _sha256(destination)


__all__ = [
    "INCIDENT_SCHEMA",
    "ORIGINAL_PREREGISTRATION_SHA256",
    "build_navigation_policy_v4_incident",
    "write_navigation_policy_v4_incident",
]
