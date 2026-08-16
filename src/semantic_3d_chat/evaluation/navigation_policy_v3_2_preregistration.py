"""Preregister and authenticate the V3.2 development calibration.

This is deliberately labeled development calibration: the same six-task
benchmark that diagnosed V3/V3.1 is used for the single V3.2 run.  It is not a
held-out generalization claim.  Gates and all executable/input hashes are
sealed before Gemma or the simulator is loaded.
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
from semantic_3d_chat.evaluation.llm_navigation_benchmark import tree_sha256

SCHEMA: Final[str] = "semantic_3d_chat.navigation_policy_v3_2_preregistration.v1"
RESULT_SCHEMA: Final[str] = "semantic_3d_chat.navigation_policy_v3_2_result.v1"
RUNTIME_VERSION: Final[str] = "v3.2"
DEFAULT_OUTPUT: Final[str] = (
    "reports/gemma4/metrics/navigation_policy_v3_2_runtime_preregistration.json"
)
DEFAULT_RESULT_OUTPUT: Final[str] = (
    "reports/gemma4/metrics/navigation_policy_v3_2_runtime_acceptance.json"
)
PARENT_PREREGISTRATION: Final[str] = (
    "reports/gemma4/metrics/navigation_policy_v3_1_runtime_preregistration.json"
)
PARENT_PREREGISTRATION_SHA256: Final[str] = (
    "0980784390932b7cde8736b0dde983366909005475a56d7c19ef78e7c5fb60c3"
)
SEALED_PREREGISTRATION_SHA256: Final[str] = (
    "d626db9291d0250e5c50206596d3d66b4bd4105861d459ed360632f98b89db34"
)
SEALED_RESULT_SHA256: Final[str] = (
    "4db9633fbc112abb561f870d7de3fa44ff4d6afc404ef2030f113479725e1f33"
)

SOURCE_PATHS: Final[tuple[str, ...]] = (
    "scripts/audit_navigation_continuous_context.py",
    "scripts/run_learned_navigation_benchmark_v3_2.sh",
    "scripts/run_llm_navigation_inference.py",
    "scripts/run_llm_navigation_inference_v3_2.py",
    "scripts/score_llm_navigation.py",
    "src/semantic_3d_chat/evaluation/llm_navigation_benchmark.py",
    "src/semantic_3d_chat/evaluation/navigation_policy_v3_2_preregistration.py",
    "src/semantic_3d_chat/robot/action_context.py",
    "src/semantic_3d_chat/robot/collision.py",
    "src/semantic_3d_chat/robot/conversation_cli.py",
    "src/semantic_3d_chat/robot/llm_tool_policy.py",
    "src/semantic_3d_chat/robot/navigation_policy_v3.py",
    "src/semantic_3d_chat/robot/navigation_policy_v3_2.py",
    "src/semantic_3d_chat/robot/planner.py",
    "src/semantic_3d_chat/robot/runtime_refresh.py",
    "src/semantic_3d_chat/robot/simulator.py",
    "src/semantic_3d_chat/robot/state_encoder.py",
    "src/semantic_3d_chat/robot/tools.py",
    "tests/test_navigation_policy_v3_2.py",
    "tests/test_numeric_waypoint_planner.py",
)
INPUT_FILE_PATHS: Final[tuple[str, ...]] = (
    "configs/benchmarks/llm_navigation_v2_scene_000001.json",
    "configs/benchmarks/oracle/llm_navigation_v2_scene_000001.json",
    "configs/runtime/embodied_navigation_v2.yaml",
    "configs/runtime/gemma4_v54.yaml",
    "data/runtime_assets/scene_000001/s_000001.blend",
    "reports/gemma4/metrics/llm_navigation_inference_access_learned_v3_1.json",
    "reports/gemma4/metrics/llm_navigation_scene_000001_learned_v3_1.json",
    "reports/gemma4/metrics/navigation_continuous_context_v3_1.json",
    "reports/gemma4/metrics/navigation_policy_v3_1_runtime_acceptance.json",
    PARENT_PREREGISTRATION,
    "reports/gemma4/predictions/llm_navigation_scene_000001_learned_v3_1.json",
)
INPUT_TREE_PATHS: Final[tuple[str, ...]] = (
    "data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000",
    "data_gemma4/checkpoints/navigation_policy_v3",
    "data_gemma4/checkpoints/robot_state_numeric_v1",
)
SUCCESSOR_OUTPUTS: Final[dict[str, str]] = {
    "journal": (
        "reports/gemma4/predictions/llm_navigation_scene_000001_learned_v3_2.json"
    ),
    "inference_audit": (
        "reports/gemma4/metrics/llm_navigation_inference_access_learned_v3_2.json"
    ),
    "score": "reports/gemma4/metrics/llm_navigation_scene_000001_learned_v3_2.json",
    "continuous_context_audit": (
        "reports/gemma4/metrics/navigation_continuous_context_v3_2.json"
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
    "calibration_task": "nav_005",
    "calibration_required_checks": [
        "all_executed_actions_succeeded",
        "no_collision",
        "post_scan_motion",
        "required_stop",
        "successful_scan",
        "target_progress",
        "target_standoff",
        "updated_prefix_consumed",
    ],
    "calibration_maximum_target_standoff_m": 0.85,
    "calibration_maximum_steps": 10,
    "continuous_context_audit_required": True,
    "oracle_and_qa_runtime_file_count": 0,
}
CALIBRATION: Final[dict[str, Any]] = {
    "compound_semantic_standoff_m": 0.35,
    "numeric_planner_grid_resolution_m": 0.15,
    "numeric_planner_standoff_tolerance_m": 0.20,
    "maximum_waypoint_step_m": 0.50,
    "same_diagnostic_benchmark": True,
    "development_calibration": True,
    "held_out_claim": False,
    "single_live_run": True,
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


def _hash_files(paths: Sequence[str]) -> dict[str, str]:
    return {path: _sha256(_rooted(path)) for path in paths}


def _hash_trees(paths: Sequence[str]) -> dict[str, str]:
    return {path: tree_sha256(_rooted(path)) for path in paths}


def _parent_diagnosis() -> dict[str, Any]:
    if _sha256(_rooted(PARENT_PREREGISTRATION)) != PARENT_PREREGISTRATION_SHA256:
        raise ValueError("V3.1 preregistration bytes differ")
    result = _read_object(
        _rooted("reports/gemma4/metrics/navigation_policy_v3_1_runtime_acceptance.json")
    )
    score = _read_object(
        _rooted("reports/gemma4/metrics/llm_navigation_scene_000001_learned_v3_1.json")
    )
    metrics = score.get("metrics")
    tasks = score.get("tasks")
    if not isinstance(metrics, Mapping) or not isinstance(tasks, list):
        raise TypeError("V3.1 score is malformed")
    by_id = {str(row.get("task_id")): row for row in tasks if isinstance(row, Mapping)}
    failed = by_id.get("nav_005", {})
    failed_metrics = failed.get("metrics") if isinstance(failed, Mapping) else None
    if (
        result.get("status") != "rejected"
        or result.get("passed") is not False
        or result.get("preregistration_sha256") != PARENT_PREREGISTRATION_SHA256
        or metrics.get("task_count") != 6
        or metrics.get("success_count") != 5
        or metrics.get("collision_count") != 0
        or not isinstance(failed_metrics, Mapping)
        or failed.get("passed") is not False
        or float(failed_metrics.get("final_target_standoff_m", -1.0))
        != 1.431078162886474
    ):
        raise ValueError("V3.1 parent diagnosis differs")
    return {
        "status": "v3_1_rejected_5_of_6",
        "success_count": 5,
        "task_count": 6,
        "collision_count": 0,
        "failed_task": "nav_005",
        "failed_check": "target_standoff",
        "final_target_standoff_m": 1.431078162886474,
        "continuous_context_passed": True,
        "oracle_runtime_files_opened": 0,
    }


def build_preregistration() -> dict[str, Any]:
    for output in (*SUCCESSOR_OUTPUTS.values(), DEFAULT_RESULT_OUTPUT):
        if _rooted(output).exists():
            raise FileExistsError(f"V3.2 output already exists: {output}")
    return {
        "schema": SCHEMA,
        "status": "sealed_before_single_v3_2_development_run",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "runtime_version": RUNTIME_VERSION,
        "claim_scope": {
            "development_calibration": True,
            "same_benchmark_used_for_diagnosis": True,
            "held_out_claim": False,
            "generalization_claim": False,
        },
        "parent_diagnosis": _parent_diagnosis(),
        "authorized_change": {
            "kind": "generic_numeric_compound_approach_calibration",
            "compound_scan_or_look_action_grammar_only": True,
            "semantic_standoff_lowered": True,
            "numeric_collision_planner_may_cap_effective_standoff": True,
            "learned_checkpoint_changed": False,
            "policy_architecture_changed": False,
            "training_data_changed": False,
            "scoring_spec_changed": False,
            "task_id_special_case": False,
            "object_vocabulary_added": False,
            "oracle_coordinate_added": False,
            "environmental_text_inputs": [],
            "oracle_inputs_at_runtime": False,
        },
        "calibration": CALIBRATION,
        "acceptance_gates": ACCEPTANCE_GATES,
        "implementation_source_hashes": _hash_files(SOURCE_PATHS),
        "input_file_hashes": _hash_files(INPUT_FILE_PATHS),
        "input_tree_hashes": _hash_trees(INPUT_TREE_PATHS),
        "successor_outputs": SUCCESSOR_OUTPUTS,
        "successor_outputs_absent_at_preregistration": True,
        "full_gemma_model_loaded_by_preregistration": False,
        "benchmark_rerun_completed": False,
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
    payload = _read_object(_rooted(path))
    if (
        payload.get("schema") != SCHEMA
        or payload.get("status") != "sealed_before_single_v3_2_development_run"
        or payload.get("runtime_version") != RUNTIME_VERSION
        or payload.get("calibration") != CALIBRATION
        or payload.get("acceptance_gates") != ACCEPTANCE_GATES
        or payload.get("successor_outputs") != SUCCESSOR_OUTPUTS
        or payload.get("successor_outputs_absent_at_preregistration") is not True
        or payload.get("benchmark_rerun_completed") is not False
        or payload.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V3.2 preregistration contract differs")
    for key, paths, observed in (
        ("implementation_source_hashes", SOURCE_PATHS, _hash_files(SOURCE_PATHS)),
        ("input_file_hashes", INPUT_FILE_PATHS, _hash_files(INPUT_FILE_PATHS)),
        ("input_tree_hashes", INPUT_TREE_PATHS, _hash_trees(INPUT_TREE_PATHS)),
    ):
        if payload.get(key) != observed or set(observed) != set(paths):
            raise ValueError(f"V3.2 sealed {key} bytes differ")
    if payload.get("parent_diagnosis") != _parent_diagnosis():
        raise ValueError("V3.2 parent diagnosis changed")
    return payload


def authenticate_historical_result(
    preregistration_path: str | Path = DEFAULT_OUTPUT,
    result_path: str | Path = DEFAULT_RESULT_OUTPUT,
) -> dict[str, Any]:
    """Authenticate sealed V3.2 rejection without current-source equivalence.

    Diagnostic work is allowed after a result is sealed.  This verifier pins
    the original preregistration and result bytes, then recomputes every cited
    evidence hash.  It intentionally does not claim that later source bytes
    reproduce the historical attempt.
    """

    prereg_path = _rooted(preregistration_path)
    sealed_result_path = _rooted(result_path)
    if _sha256(prereg_path) != SEALED_PREREGISTRATION_SHA256:
        raise ValueError("Historical V3.2 preregistration bytes differ")
    if _sha256(sealed_result_path) != SEALED_RESULT_SHA256:
        raise ValueError("Historical V3.2 result bytes differ")
    preregistration = _read_object(prereg_path)
    result = _read_object(sealed_result_path)
    evidence_hashes = _hash_files(tuple(SUCCESSOR_OUTPUTS.values()))
    if (
        preregistration.get("schema") != SCHEMA
        or preregistration.get("runtime_version") != RUNTIME_VERSION
        or preregistration.get("successor_outputs") != SUCCESSOR_OUTPUTS
        or result.get("schema") != RESULT_SCHEMA
        or result.get("status") != "rejected"
        or result.get("passed") is not False
        or result.get("runtime_promotion_authorized") is not False
        or result.get("preregistration_sha256") != SEALED_PREREGISTRATION_SHA256
        or result.get("evidence_hashes") != evidence_hashes
        or result.get("claim_scope", {}).get("held_out_claim") is not False
    ):
        raise ValueError("Historical V3.2 rejection contract differs")
    journal = _read_object(_rooted(SUCCESSOR_OUTPUTS["journal"]))
    audit = _read_object(_rooted(SUCCESSOR_OUTPUTS["inference_audit"]))
    context = _read_object(_rooted(SUCCESSOR_OUTPUTS["continuous_context_audit"]))
    run_contract = journal.get("header", {}).get("run_contract", {})
    recorded_sources = preregistration.get("implementation_source_hashes", {})
    if (
        audit.get("forbidden_accesses") != []
        or context.get("passed") is not True
        or context.get("oracle_files_opened") != 0
        or context.get("qa_files_opened") != 0
        or run_contract.get("navigation_runtime_interlock_version") != RUNTIME_VERSION
        or run_contract.get("navigation_policy_source_sha256")
        != recorded_sources.get("src/semantic_3d_chat/robot/navigation_policy_v3_2.py")
        or run_contract.get("inference_source_sha256")
        != recorded_sources.get("scripts/run_llm_navigation_inference_v3_2.py")
    ):
        raise ValueError("Historical V3.2 runtime evidence differs")
    return result


def evaluate_successor(preregistration_path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    preregistration = authenticate_preregistration(preregistration_path)
    journal = _read_object(_rooted(SUCCESSOR_OUTPUTS["journal"]))
    audit = _read_object(_rooted(SUCCESSOR_OUTPUTS["inference_audit"]))
    score = _read_object(_rooted(SUCCESSOR_OUTPUTS["score"]))
    context = _read_object(_rooted(SUCCESSOR_OUTPUTS["continuous_context_audit"]))
    metrics = score.get("metrics")
    tasks = score.get("tasks")
    context_metrics = context.get("metrics")
    if not isinstance(metrics, Mapping) or not isinstance(tasks, list):
        raise TypeError("V3.2 score is malformed")
    if not isinstance(context_metrics, Mapping):
        raise TypeError("V3.2 continuous-context audit is malformed")
    by_id = {str(row.get("task_id")): row for row in tasks if isinstance(row, Mapping)}
    calibrated = by_id.get("nav_005", {})
    checks = calibrated.get("checks") if isinstance(calibrated, Mapping) else None
    calibrated_metrics = (
        calibrated.get("metrics") if isinstance(calibrated, Mapping) else None
    )
    run_contract = journal.get("header", {}).get("run_contract", {})
    sources = preregistration["implementation_source_hashes"]
    gates = {
        "six_of_six": metrics.get("task_count") == 6 and metrics.get("success_count") == 6,
        "zero_collisions": metrics.get("collision_count") == 0,
        "zero_action_failures": metrics.get("action_failure_count") == 0,
        "zero_policy_rejections": metrics.get("policy_rejection_count") == 0,
        "previous_five_preserved": all(
            by_id.get(task_id, {}).get("passed") is True
            for task_id in ACCEPTANCE_GATES["previously_passing_tasks_must_remain_passing"]
        ),
        "calibration_task_passed": calibrated.get("passed") is True,
        "calibration_checks": isinstance(checks, Mapping)
        and all(
            checks.get(name) is True
            for name in ACCEPTANCE_GATES["calibration_required_checks"]
        ),
        "calibration_standoff": isinstance(calibrated_metrics, Mapping)
        and float(calibrated_metrics.get("final_target_standoff_m", float("inf")))
        <= ACCEPTANCE_GATES["calibration_maximum_target_standoff_m"],
        "calibration_step_bound": isinstance(calibrated_metrics, Mapping)
        and int(calibrated_metrics.get("executed_action_count", 10**9))
        <= ACCEPTANCE_GATES["calibration_maximum_steps"],
        "continuous_context": context.get("passed") is True
        and context_metrics.get("passed") is True
        and context_metrics.get("decision_context_match_count")
        == context_metrics.get("step_count")
        and context_metrics.get("prefix_chain_match_count")
        == context_metrics.get("step_count")
        and context_metrics.get("robot_token_refresh_count")
        == context_metrics.get("numeric_state_change_count")
        and context_metrics.get("scene_prefix_refresh_count")
        == context_metrics.get("map_update_count"),
        "runtime_file_isolation": audit.get("forbidden_accesses") == []
        and context.get("oracle_files_opened") == 0
        and context.get("qa_files_opened") == 0,
        "sealed_inference_source": run_contract.get("inference_source_sha256")
        == sources["scripts/run_llm_navigation_inference_v3_2.py"],
        "sealed_runtime_source": run_contract.get("navigation_policy_source_sha256")
        == sources["src/semantic_3d_chat/robot/navigation_policy_v3_2.py"],
        "v3_2_runtime_declared": run_contract.get(
            "navigation_runtime_interlock_version"
        )
        == RUNTIME_VERSION,
        "numeric_planner_declared": run_contract.get(
            "compound_scan_approach_numeric_planner"
        )
        is True
        and run_contract.get("planner_environmental_text_inputs") == []
        and run_contract.get("planner_oracle_inputs_at_runtime") is False,
    }
    passed = all(gates.values())
    return {
        "schema": RESULT_SCHEMA,
        "status": "accepted_development_calibration" if passed else "rejected",
        "passed": passed,
        "runtime_version": RUNTIME_VERSION,
        "claim_scope": preregistration["claim_scope"],
        "preregistration_path": str(preregistration_path),
        "preregistration_sha256": _sha256(_rooted(preregistration_path)),
        "evidence_hashes": _hash_files(tuple(SUCCESSOR_OUTPUTS.values())),
        "gates": gates,
        "metrics": dict(metrics),
        "calibration_task_metrics": (
            dict(calibrated_metrics) if isinstance(calibrated_metrics, Mapping) else None
        ),
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
    historical = subparsers.add_parser("historical-result")
    historical.add_argument("--preregistration", default=DEFAULT_OUTPUT)
    historical.add_argument("--result", default=DEFAULT_RESULT_OUTPUT)
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
    elif args.command == "historical-result":
        payload = authenticate_historical_result(
            args.preregistration,
            args.result,
        )
    else:
        payload = evaluate_successor(args.preregistration)
        _atomic_create(_rooted(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    if args.command == "historical-result":
        return 0
    return 0 if payload.get("passed", True) is True else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACCEPTANCE_GATES",
    "CALIBRATION",
    "DEFAULT_OUTPUT",
    "DEFAULT_RESULT_OUTPUT",
    "RUNTIME_VERSION",
    "authenticate_historical_result",
    "authenticate_preregistration",
    "build_preregistration",
    "evaluate_successor",
]
