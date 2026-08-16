"""Immutable V3.1 runtime-interlock preregistration and result authentication.

V3.1 reuses the accepted V3 learned checkpoint.  Its sole intended behavior
change is to apply the existing numeric approach-completion interlock when an
instruction has a protocol-level scan/look preamble.  The live benchmark is a
new successor run; historical V3 artifacts are inputs and are never rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT

SCHEMA: Final[str] = "semantic_3d_chat.navigation_policy_v3_1_preregistration.v1"
RESULT_SCHEMA: Final[str] = "semantic_3d_chat.navigation_policy_v3_1_result.v1"
RUNTIME_VERSION: Final[str] = "v3.1"
DEFAULT_OUTPUT: Final[str] = (
    "reports/gemma4/metrics/navigation_policy_v3_1_runtime_preregistration.json"
)
DEFAULT_RESULT_OUTPUT: Final[str] = (
    "reports/gemma4/metrics/navigation_policy_v3_1_runtime_acceptance.json"
)

SOURCE_PATHS: Final[tuple[str, ...]] = (
    "Makefile",
    "scripts/audit_navigation_continuous_context.py",
    "scripts/run_learned_navigation_benchmark.sh",
    "scripts/run_llm_navigation_inference.py",
    "scripts/score_llm_navigation.py",
    "src/semantic_3d_chat/evaluation/llm_navigation_benchmark.py",
    "src/semantic_3d_chat/evaluation/navigation_policy_v3_1_preregistration.py",
    "src/semantic_3d_chat/robot/action_context.py",
    "src/semantic_3d_chat/robot/conversation_cli.py",
    "src/semantic_3d_chat/robot/navigation_policy_v3.py",
    "src/semantic_3d_chat/robot/runtime_refresh.py",
    "src/semantic_3d_chat/robot/state_encoder.py",
)
INPUT_PATHS: Final[tuple[str, ...]] = (
    "configs/benchmarks/llm_navigation_v2_scene_000001.json",
    "configs/benchmarks/oracle/llm_navigation_v2_scene_000001.json",
    "configs/runtime/embodied_navigation_v2.yaml",
    "data_gemma4/checkpoints/navigation_policy_v3/policy.safetensors",
    "data_gemma4/checkpoints/navigation_policy_v3/runtime_metadata.json",
    "reports/gemma4/metrics/llm_navigation_inference_access_learned_v3.json",
    "reports/gemma4/metrics/llm_navigation_scene_000001_learned_v3.json",
    "reports/gemma4/predictions/llm_navigation_scene_000001_learned_v3.json",
)
SUCCESSOR_OUTPUTS: Final[dict[str, str]] = {
    "journal": (
        "reports/gemma4/predictions/llm_navigation_scene_000001_learned_v3_1.json"
    ),
    "inference_audit": (
        "reports/gemma4/metrics/llm_navigation_inference_access_learned_v3_1.json"
    ),
    "score": "reports/gemma4/metrics/llm_navigation_scene_000001_learned_v3_1.json",
    "continuous_context_audit": (
        "reports/gemma4/metrics/navigation_continuous_context_v3_1.json"
    ),
}
ACCEPTANCE_GATES: Final[dict[str, Any]] = {
    "task_count": 6,
    "minimum_success_count": 6,
    "maximum_collision_count": 0,
    "maximum_action_failure_count": 0,
    "maximum_policy_rejection_count": 0,
    "previously_passing_tasks_must_remain_passing": [
        "nav_000",
        "nav_001",
        "nav_002",
        "nav_003",
        "nav_004",
    ],
    "update_after_scan_task": "nav_005",
    "update_after_scan_required_checks": [
        "all_executed_actions_succeeded",
        "no_collision",
        "post_scan_motion",
        "required_stop",
        "successful_scan",
        "target_progress",
        "target_standoff",
        "updated_prefix_consumed",
    ],
    "update_after_scan_maximum_target_standoff_m": 0.85,
    "continuous_context_audit_required": True,
    "oracle_and_qa_runtime_file_count": 0,
}


def _rooted(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return Path(os.path.abspath(value if value.is_absolute() else PROJECT_ROOT / value))


def _sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _hash_inventory(paths: Sequence[str]) -> dict[str, str]:
    return {path: _sha256(_rooted(path)) for path in paths}


def _validate_parent_failure(score: Mapping[str, Any]) -> None:
    metrics = score.get("metrics")
    tasks = score.get("tasks")
    if not isinstance(metrics, Mapping) or not isinstance(tasks, list):
        raise TypeError("V3 parent score is malformed")
    if (
        metrics.get("task_count") != 6
        or metrics.get("success_count") != 5
        or metrics.get("collision_count") != 0
        or metrics.get("action_failure_count") != 0
        or metrics.get("policy_rejection_count") != 0
    ):
        raise ValueError("V3 parent result differs from the diagnosed 5/6 run")
    by_id = {
        str(row.get("task_id")): row for row in tasks if isinstance(row, Mapping)
    }
    if set(by_id) != {f"nav_{index:03d}" for index in range(6)}:
        raise ValueError("V3 parent task inventory differs")
    if not all(by_id[f"nav_{index:03d}"].get("passed") is True for index in range(5)):
        raise ValueError("A previously passing V3 parent task changed")
    failed = by_id["nav_005"]
    checks = failed.get("checks")
    if (
        failed.get("passed") is not False
        or not isinstance(checks, Mapping)
        or checks.get("target_standoff") is not False
        or any(value is not True for name, value in checks.items() if name != "target_standoff")
    ):
        raise ValueError("V3 parent failure is not the isolated target-standoff miss")


def build_preregistration() -> dict[str, Any]:
    for output in SUCCESSOR_OUTPUTS.values():
        if _rooted(output).exists():
            raise FileExistsError(f"Successor evidence already exists: {output}")
    parent_score = _read_object(
        _rooted("reports/gemma4/metrics/llm_navigation_scene_000001_learned_v3.json")
    )
    _validate_parent_failure(parent_score)
    return {
        "schema": SCHEMA,
        "status": "sealed_before_v3_1_live_successor_run",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "runtime_version": RUNTIME_VERSION,
        "parent_result": {
            "status": "historical_v3_5_of_6_target_standoff_failure",
            "success_count": 5,
            "task_count": 6,
            "collision_count": 0,
            "failed_task_id": "nav_005",
            "failed_check": "target_standoff",
            "final_target_standoff_m": 1.6304395335850022,
        },
        "authorized_change": {
            "kind": "bounded_runtime_interlock_grammar_correction",
            "only_behavior_change": (
                "recognize_action_only_scan_or_look_preamble_before_terminal_approach"
            ),
            "learned_checkpoint_changed": False,
            "policy_architecture_changed": False,
            "training_data_changed": False,
            "scoring_spec_changed": False,
            "collision_interlock_changed": False,
            "object_vocabulary_added": False,
            "environmental_text_inputs": [],
            "oracle_inputs_at_runtime": False,
        },
        "acceptance_gates": ACCEPTANCE_GATES,
        "implementation_source_hashes": _hash_inventory(SOURCE_PATHS),
        "input_artifact_hashes": _hash_inventory(INPUT_PATHS),
        "successor_outputs": SUCCESSOR_OUTPUTS,
        "successor_outputs_absent_at_preregistration": True,
        "benchmark_rerun_completed": False,
        "full_gemma_model_loaded_by_preregistration": False,
        "runtime_promotion_authorized": False,
    }


def _atomic_create(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
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


def authenticate_preregistration(path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    preregistration = _read_object(_rooted(path))
    if (
        preregistration.get("schema") != SCHEMA
        or preregistration.get("status") != "sealed_before_v3_1_live_successor_run"
        or preregistration.get("runtime_version") != RUNTIME_VERSION
        or preregistration.get("acceptance_gates") != ACCEPTANCE_GATES
        or preregistration.get("successor_outputs") != SUCCESSOR_OUTPUTS
        or preregistration.get("successor_outputs_absent_at_preregistration") is not True
        or preregistration.get("benchmark_rerun_completed") is not False
        or preregistration.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V3.1 preregistration contract differs")
    for key, paths in (
        ("implementation_source_hashes", SOURCE_PATHS),
        ("input_artifact_hashes", INPUT_PATHS),
    ):
        recorded = preregistration.get(key)
        if not isinstance(recorded, Mapping) or set(recorded) != set(paths):
            raise ValueError(f"V3.1 {key} inventory differs")
        for relative, expected in recorded.items():
            if _sha256(_rooted(relative)) != expected:
                raise ValueError(f"V3.1 sealed bytes differ: {relative}")
    _validate_parent_failure(
        _read_object(
            _rooted("reports/gemma4/metrics/llm_navigation_scene_000001_learned_v3.json")
        )
    )
    return preregistration


def evaluate_successor(
    preregistration_path: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    preregistration = authenticate_preregistration(preregistration_path)
    score = _read_object(_rooted(SUCCESSOR_OUTPUTS["score"]))
    context = _read_object(_rooted(SUCCESSOR_OUTPUTS["continuous_context_audit"]))
    inference_audit = _read_object(_rooted(SUCCESSOR_OUTPUTS["inference_audit"]))
    journal = _read_object(_rooted(SUCCESSOR_OUTPUTS["journal"]))
    metrics = score.get("metrics")
    rows = score.get("tasks")
    if not isinstance(metrics, Mapping) or not isinstance(rows, list):
        raise TypeError("V3.1 successor score is malformed")
    by_id = {str(row.get("task_id")): row for row in rows if isinstance(row, Mapping)}
    update = by_id.get("nav_005", {})
    update_checks = update.get("checks") if isinstance(update, Mapping) else None
    context_metrics = context.get("metrics")
    run_contract = journal.get("header", {}).get("run_contract", {})
    source_hashes = preregistration["implementation_source_hashes"]
    gates = {
        "six_of_six": metrics.get("task_count") == 6 and metrics.get("success_count") == 6,
        "zero_collisions": metrics.get("collision_count") == 0,
        "zero_action_failures": metrics.get("action_failure_count") == 0,
        "zero_policy_rejections": metrics.get("policy_rejection_count") == 0,
        "previous_five_preserved": all(
            by_id.get(task_id, {}).get("passed") is True
            for task_id in ACCEPTANCE_GATES["previously_passing_tasks_must_remain_passing"]
        ),
        "update_after_scan_passed": update.get("passed") is True,
        "update_after_scan_checks": isinstance(update_checks, Mapping)
        and all(
            update_checks.get(name) is True
            for name in ACCEPTANCE_GATES["update_after_scan_required_checks"]
        ),
        "update_after_scan_standoff": isinstance(update.get("metrics"), Mapping)
        and float(update["metrics"].get("final_target_standoff_m", float("inf")))
        <= ACCEPTANCE_GATES["update_after_scan_maximum_target_standoff_m"],
        "continuous_context": context.get("passed") is True
        and isinstance(context_metrics, Mapping)
        and context_metrics.get("passed") is True
        and context_metrics.get("decision_context_match_count")
        == context_metrics.get("step_count")
        and context_metrics.get("prefix_chain_match_count")
        == context_metrics.get("step_count")
        and context_metrics.get("robot_token_refresh_count")
        == context_metrics.get("numeric_state_change_count")
        and context_metrics.get("scene_prefix_refresh_count")
        == context_metrics.get("map_update_count"),
        "runtime_file_isolation": inference_audit.get("forbidden_accesses") == []
        and context.get("oracle_files_opened") == 0
        and context.get("qa_files_opened") == 0,
        "sealed_runtime_source": run_contract.get("navigation_policy_source_sha256")
        == source_hashes["src/semantic_3d_chat/robot/navigation_policy_v3.py"],
        "v3_1_runtime_declared": run_contract.get(
            "navigation_runtime_interlock_version"
        )
        == RUNTIME_VERSION,
    }
    passed = all(gates.values())
    return {
        "schema": RESULT_SCHEMA,
        "status": "accepted" if passed else "rejected",
        "passed": passed,
        "runtime_version": RUNTIME_VERSION,
        "preregistration_path": str(preregistration_path),
        "preregistration_sha256": _sha256(_rooted(preregistration_path)),
        "evidence_hashes": _hash_inventory(tuple(SUCCESSOR_OUTPUTS.values())),
        "gates": gates,
        "metrics": dict(metrics),
        "runtime_promotion_authorized": passed,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preregister = subparsers.add_parser("preregister")
    preregister.add_argument("--output", default=DEFAULT_OUTPUT)
    authenticate = subparsers.add_parser("authenticate")
    authenticate.add_argument("--preregistration", default=DEFAULT_OUTPUT)
    result = subparsers.add_parser("result")
    result.add_argument("--preregistration", default=DEFAULT_OUTPUT)
    result.add_argument("--output", default=DEFAULT_RESULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preregister":
        destination = _rooted(args.output)
        if destination.exists():
            payload = authenticate_preregistration(destination)
        else:
            payload = build_preregistration()
            _atomic_create(destination, payload)
    elif args.command == "authenticate":
        payload = authenticate_preregistration(args.preregistration)
    else:
        payload = evaluate_successor(args.preregistration)
        _atomic_create(_rooted(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if payload.get("passed", True) is True else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACCEPTANCE_GATES",
    "DEFAULT_OUTPUT",
    "DEFAULT_RESULT_OUTPUT",
    "RUNTIME_VERSION",
    "authenticate_preregistration",
    "build_preregistration",
    "evaluate_successor",
]
