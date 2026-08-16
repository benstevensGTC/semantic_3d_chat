"""Seal the one-change V4.1 recovery protocol before its deterministic rerun."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT

SCHEMA: Final[str] = "semantic_3d_chat.navigation_policy_v4_1_preregistration.v1"
ORIGINAL_PREREGISTRATION: Final[str] = (
    "reports/gemma4/metrics/navigation_policy_v4_preregistration.json"
)
ORIGINAL_PREREGISTRATION_SHA256: Final[str] = (
    "b855ee22bfbca6b5f709199e5b88937c6643c9ddbea39a102ebebc23f0a28c61"
)
INCIDENT: Final[str] = (
    "reports/gemma4/metrics/navigation_policy_v4_training_incident.json"
)
INCIDENT_SHA256: Final[str] = (
    "0bc794c3c41ae13e43f9d78bfd81e560c0e8096a7cdbd9b5d2aadcfe24da8ddb"
)
V4_DATASET_SHA256: Final[str] = (
    "c1a383b27bbfb114354c083fc90a7f92eaefc445d2bf8f71b818bd66826ea8ec"
)
V3_DATASET_SHA256: Final[str] = (
    "d8d97ac248a5821eb971301efb742c25c996627bae22d6417c02755e61d50f9d"
)

_CURRENT_SOURCE_PATHS: Final[tuple[str, ...]] = (
    "src/semantic_3d_chat/evaluation/navigation_policy_v41_preregistration.py",
    "src/semantic_3d_chat/robot/navigation_policy_v4.py",
    "src/semantic_3d_chat/training/train_navigation_policy_v4.py",
    "src/semantic_3d_chat/robot/conversation_cli.py",
    "src/semantic_3d_chat/robot/llm_tool_policy.py",
    "src/semantic_3d_chat/evaluation/llm_navigation_benchmark.py",
    "src/semantic_3d_chat/scene_encoder/map_io.py",
    "src/semantic_3d_chat/robot/collision.py",
    "scripts/preregister_navigation_policy_v4_1.py",
    "scripts/train_navigation_policy_v4_1.py",
    "scripts/evaluate_navigation_policy_v4.py",
    "scripts/audit_navigation_policy_v4_runtime.py",
    "scripts/run_llm_navigation_inference.py",
    "tests/test_navigation_policy_v4.py",
    "tests/test_navigation_policy_v4_preregistration.py",
)


class PreregistrationV41Error(RuntimeError):
    """Raised when the V4.1 recovery differs from its mechanical scope."""


def _rooted(path: str | Path) -> Path:
    value = Path(path).expanduser()
    rooted = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(rooted))


def _sha256(path: str | Path) -> str:
    source = _rooted(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hashes(paths: tuple[str, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in paths:
        path = _rooted(relative)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"V4.1 source is unavailable: {relative}")
        result[relative] = _sha256(path)
    return result


def _load_sealed(path: str, digest: str) -> dict[str, Any]:
    source = _rooted(path)
    if not source.is_file() or source.is_symlink() or _sha256(source) != digest:
        raise PreregistrationV41Error(f"Sealed predecessor evidence changed: {path}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Sealed predecessor evidence must be an object: {path}")
    return value


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def build_navigation_policy_v41_preregistration(
    config: Mapping[str, Any],
    *,
    source_v3_dataset_sha256: str,
    v4_dataset_sha256: str,
    map_sha256: Mapping[str, str],
) -> dict[str, Any]:
    original = _load_sealed(
        ORIGINAL_PREREGISTRATION, ORIGINAL_PREREGISTRATION_SHA256
    )
    incident = _load_sealed(INCIDENT, INCIDENT_SHA256)
    settings = config.get("navigation_policy_v4")
    if not isinstance(settings, Mapping):
        raise TypeError("V4.1 config has no navigation_policy_v4 settings")
    original_hyperparameters = original["single_arm"]["hyperparameters"]
    observed_hyperparameters = {
        name: settings.get(name) for name in original_hyperparameters
    }
    original_gates = original["acceptance_gates"]
    observed_gates = {name: settings.get(name) for name in original_gates}
    expected_maps = original["data"]["map_sha256"]
    original_sources = original["implementation_source_hashes"]
    current_original_sources = {
        relative: _sha256(relative) for relative in original_sources
    }
    changed_original_sources = sorted(
        relative
        for relative, digest in current_original_sources.items()
        if digest != original_sources[relative]
    )
    if (
        incident.get("status")
        != "sealed_failed_attempt_no_checkpoint_or_training_report"
        or incident["publication_state"]["checkpoint_absent"] is not True
        or incident["publication_state"]["training_report_absent"] is not True
        or observed_hyperparameters != original_hyperparameters
        or observed_gates != original_gates
        or settings.get("train_scene_ids") != original["data"]["train_scene_ids"]
        or settings.get("validation_scene_ids")
        != original["data"]["validation_scene_ids"]
        or source_v3_dataset_sha256 != V3_DATASET_SHA256
        or v4_dataset_sha256 != V4_DATASET_SHA256
        or dict(sorted(map_sha256.items())) != expected_maps
        or changed_original_sources
        != ["src/semantic_3d_chat/training/train_navigation_policy_v4.py"]
        or settings.get("protocol_version") != "v4.1"
        or settings.get("incident") != INCIDENT
    ):
        raise PreregistrationV41Error(
            "V4.1 differs from the authorized one-change recovery scope"
        )
    stable_config = {
        key: value for key, value in config.items() if not key.startswith("_")
    }
    return {
        "schema": SCHEMA,
        "status": "sealed_before_v4_1_deterministic_rerun",
        "original_failed_attempt": {
            "preregistration_path": ORIGINAL_PREREGISTRATION,
            "preregistration_sha256": ORIGINAL_PREREGISTRATION_SHA256,
            "incident_path": INCIDENT,
            "incident_sha256": INCIDENT_SHA256,
            "checkpoint_published": False,
            "training_report_published": False,
            "model_result_interpreted": False,
        },
        "mechanical_amendment": {
            "protocol_version": "v4.1",
            "only_behavior_change": (
                "targeted_action_accuracy_and_targetless_action_accuracy_return_"
                "finite_0_when_their_boolean_subgroup_is_empty"
            ),
            "v3_precedent": (
                "evaluate_prepared_v3_uses_boolean_any_guard_and_finite_0_0"
            ),
            "optimizer_math_changed": False,
            "training_loss_changed": False,
            "model_architecture_changed": False,
            "selector_score_changed_for_nonempty_primary_validation_groups": False,
            "controls_and_gates_changed": False,
            "strict_json_allow_nan_remains_false": True,
        },
        "preserved_single_arm": {
            "seed": settings["seed"],
            "hyperparameters": observed_hyperparameters,
            "acceptance_gates": observed_gates,
            "train_scene_ids": list(settings["train_scene_ids"]),
            "validation_scene_ids": list(settings["validation_scene_ids"]),
            "source_v3_dataset_sha256": source_v3_dataset_sha256,
            "prepared_v4_dataset_sha256": v4_dataset_sha256,
            "map_sha256": dict(sorted(map_sha256.items())),
            "one_arm_only": True,
            "hyperparameter_search": False,
            "exact_deterministic_rerun": True,
            "live_benchmark_used_for_training_or_selection": False,
            "benchmark_oracle_used_for_training_or_selection": False,
        },
        "source_audit": {
            "original_implementation_source_hashes": original_sources,
            "current_original_source_hashes": current_original_sources,
            "changed_original_source_paths": changed_original_sources,
            "changed_original_source_count": 1,
            "current_v4_1_source_hashes": _hashes(_CURRENT_SOURCE_PATHS),
        },
        "merged_config_sha256": _canonical_sha256(stable_config),
        "runtime_separation": original["runtime_separation"],
        "publication": {
            "checkpoint_output": settings["checkpoint_output"],
            "checkpoint_written_only_if_every_original_gate_passes": True,
            "training_report_create_once": True,
            "rejected_arm_publishes_no_checkpoint": True,
            "live_six_task_run_only_after_every_original_gate_passes": True,
        },
    }


def write_navigation_policy_v41_preregistration(
    destination: str | Path,
    config: Mapping[str, Any],
    *,
    source_v3_dataset_sha256: str,
    v4_dataset_sha256: str,
    map_sha256: Mapping[str, str],
    training_report: str | Path = (
        "reports/gemma4/metrics/navigation_policy_v4_1_training.json"
    ),
) -> tuple[Path, str]:
    path = _rooted(destination)
    settings = config["navigation_policy_v4"]
    checkpoint = _rooted(str(settings["checkpoint_output"]))
    report = _rooted(training_report)
    if path.exists() or checkpoint.exists() or report.exists():
        raise FileExistsError(
            "V4.1 preregistration requires absent preregistration, checkpoint, and report"
        )
    payload = build_navigation_policy_v41_preregistration(
        config,
        source_v3_dataset_sha256=source_v3_dataset_sha256,
        v4_dataset_sha256=v4_dataset_sha256,
        map_sha256=map_sha256,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path, _sha256(path)


def authenticate_navigation_policy_v41_preregistration(
    path: str | Path,
    config: Mapping[str, Any],
    *,
    source_v3_dataset_sha256: str,
    v4_dataset_sha256: str,
    map_sha256: Mapping[str, str],
) -> dict[str, Any]:
    source = _rooted(path)
    if not source.is_file() or source.is_symlink():
        raise PreregistrationV41Error("Sealed V4.1 preregistration is unavailable")
    value = json.loads(source.read_text(encoding="utf-8"))
    expected = build_navigation_policy_v41_preregistration(
        config,
        source_v3_dataset_sha256=source_v3_dataset_sha256,
        v4_dataset_sha256=v4_dataset_sha256,
        map_sha256=map_sha256,
    )
    if value != expected:
        raise PreregistrationV41Error(
            "Sealed V4.1 preregistration differs from current sources or inputs"
        )
    return {
        "authenticated": True,
        "path": str(
            source.relative_to(PROJECT_ROOT)
            if source.is_relative_to(PROJECT_ROOT)
            else source
        ),
        "sha256": _sha256(source),
        "single_arm": True,
        "sealed_before_training": True,
        "protocol_version": "v4.1",
    }


__all__ = [
    "INCIDENT_SHA256",
    "ORIGINAL_PREREGISTRATION_SHA256",
    "SCHEMA",
    "PreregistrationV41Error",
    "authenticate_navigation_policy_v41_preregistration",
    "build_navigation_policy_v41_preregistration",
    "write_navigation_policy_v41_preregistration",
]
