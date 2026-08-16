"""Fail-closed authentication of the rejected Navigation V4.1 single arm."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT

TRAINING_REPORT: Final[str] = (
    "reports/gemma4/metrics/navigation_policy_v4_1_training.json"
)
TRAINING_REPORT_SHA256: Final[str] = (
    "ab5abd41e00a95ec5a35e93f5871607ff6c961373040531c7323544fd3d29f0f"
)
PREREGISTRATION: Final[str] = (
    "reports/gemma4/metrics/navigation_policy_v4_1_preregistration.json"
)
PREREGISTRATION_SHA256: Final[str] = (
    "3d8df77748017d4ee5fe337f39341252431b2df32068ab81ad7e3f60711f5dc0"
)
INCIDENT: Final[str] = (
    "reports/gemma4/metrics/navigation_policy_v4_training_incident.json"
)
INCIDENT_SHA256: Final[str] = (
    "0bc794c3c41ae13e43f9d78bfd81e560c0e8096a7cdbd9b5d2aadcfe24da8ddb"
)
ORIGINAL_PREREGISTRATION: Final[str] = (
    "reports/gemma4/metrics/navigation_policy_v4_preregistration.json"
)
ORIGINAL_PREREGISTRATION_SHA256: Final[str] = (
    "b855ee22bfbca6b5f709199e5b88937c6643c9ddbea39a102ebebc23f0a28c61"
)
CHECKPOINT: Final[str] = "data_gemma4/checkpoints/navigation_policy_v4_1"


class V41ResultAuthenticationError(RuntimeError):
    """Raised when any V4.1 measurement or source invariant differs."""


def _rooted(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pinned(
    path: str | Path, expected_sha256: str, *, name: str
) -> dict[str, Any]:
    source = _rooted(path)
    if not source.is_file() or source.is_symlink():
        raise V41ResultAuthenticationError(f"V4.1 {name} is unavailable")
    observed = _sha256(source)
    if observed != expected_sha256:
        raise V41ResultAuthenticationError(
            f"V4.1 {name} digest differs: expected {expected_sha256}, observed {observed}"
        )
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise V41ResultAuthenticationError(f"V4.1 {name} must be an object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V41ResultAuthenticationError(message)


def _finite_tree(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and _finite_tree(item) for key, item in value.items())
    return False


def authenticate_navigation_policy_v41_result(
    *,
    training_report: str | Path = TRAINING_REPORT,
    checkpoint: str | Path = CHECKPOINT,
) -> dict[str, Any]:
    """Authenticate the one sealed result without opening oracle/scorer trees."""

    original = _load_pinned(
        ORIGINAL_PREREGISTRATION,
        ORIGINAL_PREREGISTRATION_SHA256,
        name="original preregistration",
    )
    incident = _load_pinned(INCIDENT, INCIDENT_SHA256, name="incident")
    preregistration = _load_pinned(
        PREREGISTRATION, PREREGISTRATION_SHA256, name="preregistration"
    )
    report = _load_pinned(
        training_report, TRAINING_REPORT_SHA256, name="training report"
    )
    checkpoint_path = _rooted(checkpoint)
    _require(not checkpoint_path.exists(), "Rejected V4.1 checkpoint unexpectedly exists")
    _require(_finite_tree(report), "V4.1 training report contains a nonfinite value")

    source_hashes = preregistration.get("source_audit", {}).get(
        "current_v4_1_source_hashes"
    )
    _require(isinstance(source_hashes, Mapping), "V4.1 source inventory is unavailable")
    current_source_drift_paths: list[str] = []
    for relative, expected in source_hashes.items():
        _require(
            isinstance(relative, str)
            and bool(relative)
            and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts
            and isinstance(expected, str)
            and len(expected) == 64
            and all(character in "0123456789abcdef" for character in expected),
            f"V4.1 sealed implementation source inventory is invalid: {relative}",
        )
        current = _rooted(relative)
        if not current.is_file() or current.is_symlink() or _sha256(current) != expected:
            current_source_drift_paths.append(relative)

    gates = report.get("gates")
    validation = report.get("validation")
    controls = report.get("controls")
    _require(
        report.get("schema")
        == "semantic_3d_chat.navigation_policy_v4_1_training_result.v1"
        and report.get("protocol_version") == "v4.1"
        and report.get("status") == "rejected"
        and report.get("checkpoint_written") is False
        and report.get("checkpoint") is None
        and report.get("single_preregistered_arm") is True
        and report.get("v3_base_frozen") is True
        and report.get("train_scene_gradients_only") is True
        and report.get("oracle_inputs_at_runtime") is False
        and report.get("environmental_text_inputs_at_runtime") == []
        and report.get("dataset_sha256")
        == original["data"]["prepared_v4_dataset_sha256"]
        and report.get("source_v3_dataset_sha256")
        == original["data"]["source_v3_dataset_sha256"]
        and report.get("preregistration", {}).get("sha256")
        == PREREGISTRATION_SHA256
        and isinstance(gates, Mapping)
        and isinstance(validation, Mapping)
        and isinstance(controls, Mapping),
        "V4.1 rejected-arm identity differs",
    )
    failed_gates = sorted(name for name, passed in gates.items() if passed is not True)
    _require(
        failed_gates == ["shuffled_clearance_family_drop"],
        "V4.1 failed-gate inventory differs",
    )
    expected_validation = {
        "action_accuracy": 0.9347442388534546,
        "argument_mae": 0.14731742441654205,
        "collision_risk_accuracy": 0.9822530746459961,
        "stop_recall": 0.931174099445343,
        "turn_sign_accuracy": 0.9317129850387573,
        "unsafe_motion_rejection": 1.0,
    }
    for name, expected in expected_validation.items():
        observed = validation.get(name)
        _require(
            isinstance(observed, (int, float))
            and not isinstance(observed, bool)
            and abs(float(observed) - expected) <= 1e-12,
            f"V4.1 validation metric differs: {name}",
        )
    observed_drop = controls.get("shuffled_clearance_obstacle_update_accuracy_drop")
    threshold = report.get("thresholds", {}).get(
        "minimum_shuffled_clearance_family_drop"
    )
    _require(
        observed_drop == 0.049565017223358154
        and threshold == 0.1
        and observed_drop < threshold,
        "V4.1 causal-clearance rejection differs",
    )
    _require(
        incident.get("failure", {}).get("model_acceptance_result_interpreted") is False
        and incident.get("failure", {}).get("live_navigation_benchmark_opened") is False
        and preregistration.get("preserved_single_arm", {}).get("one_arm_only") is True
        and preregistration.get("preserved_single_arm", {}).get(
            "live_benchmark_used_for_training_or_selection"
        )
        is False,
        "V4.1 evidence separation differs",
    )
    return {
        "schema": "semantic_3d_chat.navigation_policy_v4_1_result_authentication.v1",
        "measurement_authenticated": True,
        "status": (
            "historical_evidence_authenticated_current_runtime_compatibility_not_claimed"
        ),
        "training_report_sha256": TRAINING_REPORT_SHA256,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "incident_sha256": INCIDENT_SHA256,
        "original_preregistration_sha256": ORIGINAL_PREREGISTRATION_SHA256,
        "checkpoint_absent": True,
        "live_benchmark_executed": False,
        "oracle_or_scorer_opened": False,
        "failed_gates": failed_gates,
        "passed_gate_count": len(gates) - len(failed_gates),
        "gate_count": len(gates),
        "validation": {name: validation[name] for name in expected_validation},
        "shuffled_clearance_family_drop": observed_drop,
        "required_shuffled_clearance_family_drop": threshold,
        "promotion_eligible": False,
        "historical_source_inventory_authenticated": True,
        "current_source_snapshot_matches_sealed": not current_source_drift_paths,
        "current_source_drift_paths": sorted(current_source_drift_paths),
        "current_runtime_compatibility_claimed": False,
    }


def inspect_navigation_policy_v41_result(**kwargs: Any) -> dict[str, Any]:
    try:
        return authenticate_navigation_policy_v41_result(**kwargs)
    except (V41ResultAuthenticationError, OSError, ValueError, TypeError, KeyError) as exc:
        return {
            "schema": "semantic_3d_chat.navigation_policy_v4_1_result_authentication.v1",
            "measurement_authenticated": False,
            "status": "authentication_failed",
            "error": str(exc),
            "promotion_eligible": False,
        }


def main() -> int:
    result = inspect_navigation_policy_v41_result()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["measurement_authenticated"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "TRAINING_REPORT_SHA256",
    "V41ResultAuthenticationError",
    "authenticate_navigation_policy_v41_result",
    "inspect_navigation_policy_v41_result",
]
