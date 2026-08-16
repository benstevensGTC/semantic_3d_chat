"""Seal, score, and authenticate the one-run V3.3 development calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.llm_navigation_benchmark import tree_sha256
from semantic_3d_chat.evaluation.navigation_policy_v3_2_preregistration import (
    authenticate_historical_result as authenticate_v3_2_rejection,
)

SCHEMA: Final[str] = "semantic_3d_chat.navigation_policy_v3_3_preregistration.v1"
RESULT_SCHEMA: Final[str] = "semantic_3d_chat.navigation_policy_v3_3_result.v1"
RUNTIME_VERSION: Final[str] = "v3.3"
DEFAULT_OUTPUT: Final[str] = (
    "reports/gemma4/metrics/navigation_policy_v3_3_runtime_preregistration.json"
)
DEFAULT_RESULT_OUTPUT: Final[str] = (
    "reports/gemma4/metrics/navigation_policy_v3_3_runtime_acceptance.json"
)
V3_2_PREREGISTRATION_SHA256: Final[str] = (
    "d626db9291d0250e5c50206596d3d66b4bd4105861d459ed360632f98b89db34"
)
V3_2_RESULT_SHA256: Final[str] = "4db9633fbc112abb561f870d7de3fa44ff4d6afc404ef2030f113479725e1f33"
ROUTING_TEST_NODE: Final[str] = (
    "tests/test_navigation_policy_v3_3.py::"
    "test_exact_live_runner_routes_enveloped_sequence_through_v3_3_planner"
)

SOURCE_PATHS: Final[tuple[str, ...]] = (
    "scripts/audit_navigation_continuous_context.py",
    "scripts/run_learned_navigation_benchmark_v3_3.sh",
    "scripts/run_llm_navigation_inference.py",
    "scripts/run_llm_navigation_inference_v3_3.py",
    "scripts/score_llm_navigation.py",
    "src/semantic_3d_chat/evaluation/llm_navigation_benchmark.py",
    "src/semantic_3d_chat/evaluation/navigation_policy_v3_2_preregistration.py",
    "src/semantic_3d_chat/evaluation/navigation_policy_v3_3_preregistration.py",
    "src/semantic_3d_chat/robot/action_context.py",
    "src/semantic_3d_chat/robot/collision.py",
    "src/semantic_3d_chat/robot/conversation_cli.py",
    "src/semantic_3d_chat/robot/llm_tool_policy.py",
    "src/semantic_3d_chat/robot/navigation_policy_v3.py",
    "src/semantic_3d_chat/robot/navigation_policy_v3_2.py",
    "src/semantic_3d_chat/robot/navigation_policy_v3_3.py",
    "src/semantic_3d_chat/robot/planner.py",
    "src/semantic_3d_chat/robot/runtime_refresh.py",
    "src/semantic_3d_chat/robot/simulator.py",
    "src/semantic_3d_chat/robot/state_encoder.py",
    "src/semantic_3d_chat/robot/tools.py",
    "tests/test_navigation_policy_v3_2.py",
    "tests/test_navigation_policy_v3_3.py",
    "tests/test_navigation_policy_v3_3_preregistration.py",
    "tests/test_numeric_waypoint_planner.py",
)
INPUT_FILE_PATHS: Final[tuple[str, ...]] = (
    "configs/benchmarks/llm_navigation_v2_scene_000001.json",
    "configs/benchmarks/oracle/llm_navigation_v2_scene_000001.json",
    "configs/runtime/embodied_navigation_v2.yaml",
    "configs/runtime/gemma4_v54.yaml",
    "data/runtime_assets/scene_000001/s_000001.blend",
    "data_gemma4/maps/scene_000001/voxel_map.npz",
    "reports/gemma4/metrics/llm_navigation_inference_access_learned_v3_2.json",
    "reports/gemma4/metrics/llm_navigation_scene_000001_learned_v3_2.json",
    "reports/gemma4/metrics/navigation_continuous_context_v3_2.json",
    "reports/gemma4/metrics/navigation_policy_v3_2_runtime_acceptance.json",
    "reports/gemma4/metrics/navigation_policy_v3_2_runtime_preregistration.json",
    "reports/gemma4/predictions/llm_navigation_scene_000001_learned_v3_2.json",
)
INPUT_TREE_PATHS: Final[tuple[str, ...]] = (
    "data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000",
    "data_gemma4/checkpoints/navigation_policy_v3",
    "data_gemma4/checkpoints/robot_state_numeric_v1",
)
SUCCESSOR_OUTPUTS: Final[dict[str, str]] = {
    "journal": ("reports/gemma4/predictions/llm_navigation_scene_000001_learned_v3_3.json"),
    "inference_audit": ("reports/gemma4/metrics/llm_navigation_inference_access_learned_v3_3.json"),
    "score": "reports/gemma4/metrics/llm_navigation_scene_000001_learned_v3_3.json",
    "continuous_context_audit": ("reports/gemma4/metrics/navigation_continuous_context_v3_3.json"),
}
SUCCESSOR_RUNTIME_OUTPUTS: Final[dict[str, str]] = {
    "persistent_map": ("data_gemma4/robot_benchmark_learned_v3_3/scene_000001/semantic_map.npz"),
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
    "numeric_planner_action_required": "move_to",
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
    "no_post_result_tuning": True,
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


def _run_routing_preflight() -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", ROUTING_TEST_NODE]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src"
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    stdout = completed.stdout.encode("utf-8")
    stderr = completed.stderr.encode("utf-8")
    passed = completed.returncode == 0 and b"1 passed" in stdout
    if not passed:
        raise RuntimeError(
            "V3.3 exact live-routing integration preflight failed:\n"
            + completed.stdout
            + completed.stderr
        )
    return {
        "required_before_full_gemma_load": True,
        "test_node": ROUTING_TEST_NODE,
        "command": command,
        "exit_code": completed.returncode,
        "passed": True,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "real_v3_checkpoint_controller_constructed_on_cpu": True,
        "exact_live_wrapper_install_function_exercised": True,
        "enveloped_scan_then_move_to_then_stop_proved": True,
        "planner_metadata_proved_before_stop": True,
        "full_gemma_model_loaded": False,
        "optimizer_constructed": False,
        "oracle_loaded": False,
    }


def _v3_2_provenance() -> dict[str, Any]:
    result = authenticate_v3_2_rejection()
    score = _read_object(
        _rooted("reports/gemma4/metrics/llm_navigation_scene_000001_learned_v3_2.json")
    )
    metrics = score.get("metrics")
    if (
        _sha256(
            _rooted("reports/gemma4/metrics/navigation_policy_v3_2_runtime_preregistration.json")
        )
        != V3_2_PREREGISTRATION_SHA256
        or _sha256(_rooted("reports/gemma4/metrics/navigation_policy_v3_2_runtime_acceptance.json"))
        != V3_2_RESULT_SHA256
        or result.get("status") != "rejected"
        or not isinstance(metrics, Mapping)
        or metrics.get("task_count") != 6
        or metrics.get("success_count") != 5
        or metrics.get("collision_count") != 0
    ):
        raise ValueError("V3.2 rejection provenance differs")
    return {
        "preregistration_sha256": V3_2_PREREGISTRATION_SHA256,
        "result_sha256": V3_2_RESULT_SHA256,
        "status": "rejected",
        "success_count": 5,
        "task_count": 6,
        "collision_count": 0,
        "runtime_promotion_authorized": False,
        "held_out_claim": False,
        "failure_diagnosis": "live_protocol_envelope_not_unwrapped_by_v3_2_branch",
        "v3_2_episode_identical_to_v3_1": True,
    }


def build_preregistration(
    routing_preflight: Mapping[str, Any],
    *,
    successor_outputs: Mapping[str, str] = SUCCESSOR_OUTPUTS,
    successor_runtime_outputs: Mapping[str, str] = SUCCESSOR_RUNTIME_OUTPUTS,
    result_output: str = DEFAULT_RESULT_OUTPUT,
) -> dict[str, Any]:
    for output in (
        *successor_outputs.values(),
        *successor_runtime_outputs.values(),
        result_output,
    ):
        if _rooted(output).exists():
            raise FileExistsError(f"V3.3 output already exists: {output}")
    if (
        routing_preflight.get("test_node") != ROUTING_TEST_NODE
        or routing_preflight.get("passed") is not True
        or routing_preflight.get("exit_code") != 0
        or routing_preflight.get("required_before_full_gemma_load") is not True
        or routing_preflight.get("full_gemma_model_loaded") is not False
        or routing_preflight.get("enveloped_scan_then_move_to_then_stop_proved") is not True
        or routing_preflight.get("planner_metadata_proved_before_stop") is not True
    ):
        raise ValueError("V3.3 exact routing preflight is not an accepted pass")
    return {
        "schema": SCHEMA,
        "status": "sealed_before_single_v3_3_development_run",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "runtime_version": RUNTIME_VERSION,
        "claim_scope": {
            "development_calibration": True,
            "same_benchmark_used_for_diagnosis": True,
            "held_out_claim": False,
            "generalization_claim": False,
            "stop_calibration_family_if_rejected": True,
        },
        "v3_2_rejection_provenance": _v3_2_provenance(),
        "routing_integration_preflight": dict(routing_preflight),
        "authorized_change": {
            "kind": "correct_live_protocol_envelope_routing_only",
            "numeric_planner_calibration_unchanged_from_v3_2": True,
            "learned_checkpoint_changed": False,
            "policy_architecture_changed": False,
            "training_data_changed": False,
            "benchmark_changed": False,
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
        "successor_outputs": dict(successor_outputs),
        "successor_runtime_outputs": dict(successor_runtime_outputs),
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
    preflight = payload.get("routing_integration_preflight")
    if (
        payload.get("schema") != SCHEMA
        or payload.get("status") != "sealed_before_single_v3_3_development_run"
        or payload.get("runtime_version") != RUNTIME_VERSION
        or payload.get("calibration") != CALIBRATION
        or payload.get("acceptance_gates") != ACCEPTANCE_GATES
        or payload.get("successor_outputs") != SUCCESSOR_OUTPUTS
        or payload.get("successor_runtime_outputs") != SUCCESSOR_RUNTIME_OUTPUTS
        or payload.get("successor_outputs_absent_at_preregistration") is not True
        or payload.get("benchmark_rerun_completed") is not False
        or payload.get("runtime_promotion_authorized") is not False
        or not isinstance(preflight, Mapping)
        or preflight.get("test_node") != ROUTING_TEST_NODE
        or preflight.get("passed") is not True
        or preflight.get("full_gemma_model_loaded") is not False
    ):
        raise ValueError("V3.3 preregistration contract differs")
    for key, expected in (
        ("implementation_source_hashes", _hash_files(SOURCE_PATHS)),
        ("input_file_hashes", _hash_files(INPUT_FILE_PATHS)),
        ("input_tree_hashes", _hash_trees(INPUT_TREE_PATHS)),
    ):
        if payload.get(key) != expected:
            raise ValueError(f"V3.3 sealed {key} bytes differ")
    if payload.get("v3_2_rejection_provenance") != _v3_2_provenance():
        raise ValueError("V3.3 parent rejection provenance changed")
    return payload


def _calls(episode: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    for step in episode.get("steps", []):
        if not isinstance(step, Mapping):
            continue
        decision = step.get("decision")
        call = decision.get("call") if isinstance(decision, Mapping) else None
        name = call.get("tool") if isinstance(call, Mapping) else None
        if isinstance(name, str):
            names.append(name)
    return names


def evaluate_successor(preregistration_path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    preregistration = authenticate_preregistration(preregistration_path)
    journal = _read_object(_rooted(SUCCESSOR_OUTPUTS["journal"]))
    audit = _read_object(_rooted(SUCCESSOR_OUTPUTS["inference_audit"]))
    score = _read_object(_rooted(SUCCESSOR_OUTPUTS["score"]))
    context = _read_object(_rooted(SUCCESSOR_OUTPUTS["continuous_context_audit"]))
    metrics = score.get("metrics")
    tasks = score.get("tasks")
    context_metrics = context.get("metrics")
    episodes = journal.get("episodes")
    if not isinstance(metrics, Mapping) or not isinstance(tasks, list):
        raise TypeError("V3.3 score is malformed")
    if not isinstance(context_metrics, Mapping) or not isinstance(episodes, list):
        raise TypeError("V3.3 context or journal is malformed")
    by_id = {str(row.get("task_id")): row for row in tasks if isinstance(row, Mapping)}
    episode_by_id = {str(row.get("task_id")): row for row in episodes if isinstance(row, Mapping)}
    calibrated = by_id.get("nav_005", {})
    checks = calibrated.get("checks") if isinstance(calibrated, Mapping) else None
    calibrated_metrics = calibrated.get("metrics") if isinstance(calibrated, Mapping) else None
    calibrated_calls = _calls(episode_by_id.get("nav_005", {}))
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
            checks.get(name) is True for name in ACCEPTANCE_GATES["calibration_required_checks"]
        ),
        "calibration_standoff": isinstance(calibrated_metrics, Mapping)
        and float(calibrated_metrics.get("final_target_standoff_m", float("inf")))
        <= ACCEPTANCE_GATES["calibration_maximum_target_standoff_m"],
        "calibration_step_bound": isinstance(calibrated_metrics, Mapping)
        and int(calibrated_metrics.get("executed_action_count", 10**9))
        <= ACCEPTANCE_GATES["calibration_maximum_steps"],
        "numeric_planner_exercised": bool(calibrated_calls)
        and "move_to" in calibrated_calls
        and calibrated_calls[0] == "scan"
        and calibrated_calls[-1] == "stop",
        "continuous_context": context.get("passed") is True
        and context_metrics.get("passed") is True
        and context_metrics.get("decision_context_match_count") == context_metrics.get("step_count")
        and context_metrics.get("prefix_chain_match_count") == context_metrics.get("step_count")
        and context_metrics.get("robot_token_refresh_count")
        == context_metrics.get("numeric_state_change_count")
        and context_metrics.get("scene_prefix_refresh_count")
        == context_metrics.get("map_update_count"),
        "runtime_file_isolation": audit.get("forbidden_accesses") == []
        and context.get("oracle_files_opened") == 0
        and context.get("qa_files_opened") == 0,
        "sealed_inference_source": run_contract.get("inference_source_sha256")
        == sources["scripts/run_llm_navigation_inference_v3_3.py"],
        "sealed_runtime_source": run_contract.get("navigation_policy_source_sha256")
        == sources["src/semantic_3d_chat/robot/navigation_policy_v3_3.py"]
        and run_contract.get("navigation_policy_parent_source_sha256")
        == sources["src/semantic_3d_chat/robot/navigation_policy_v3_2.py"],
        "v3_3_runtime_declared": run_contract.get("navigation_runtime_interlock_version")
        == RUNTIME_VERSION,
        "corrected_routing_declared": run_contract.get(
            "live_protocol_envelope_unwrapped_before_action_grammar"
        )
        is True,
        "numeric_planner_declared": run_contract.get("compound_scan_approach_numeric_planner")
        is True
        and run_contract.get("planner_environmental_text_inputs") == []
        and run_contract.get("planner_oracle_inputs_at_runtime") is False,
    }
    passed = all(gates.values())
    return {
        "schema": RESULT_SCHEMA,
        "status": "accepted_development_calibration" if passed else "rejected_terminal",
        "passed": passed,
        "runtime_version": RUNTIME_VERSION,
        "claim_scope": preregistration["claim_scope"],
        "preregistration_path": str(preregistration_path),
        "preregistration_sha256": _sha256(_rooted(preregistration_path)),
        "evidence_hashes": _hash_files(tuple(SUCCESSOR_OUTPUTS.values())),
        "runtime_output_hashes": _hash_files(tuple(SUCCESSOR_RUNTIME_OUTPUTS.values())),
        "gates": gates,
        "metrics": dict(metrics),
        "calibration_task_metrics": (
            dict(calibrated_metrics) if isinstance(calibrated_metrics, Mapping) else None
        ),
        "calibration_task_calls": calibrated_calls,
        "runtime_promotion_authorized": passed,
        "calibration_family_must_stop": not passed,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
    }


def authenticate_result(
    preregistration_path: str | Path = DEFAULT_OUTPUT,
    result_path: str | Path = DEFAULT_RESULT_OUTPUT,
) -> dict[str, Any]:
    observed = _read_object(_rooted(result_path))
    recomputed = evaluate_successor(preregistration_path)
    if observed != recomputed:
        raise ValueError("V3.3 stored result differs from independent recomputation")
    return observed


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
    result_auth = subparsers.add_parser("authenticate-result")
    result_auth.add_argument("--preregistration", default=DEFAULT_OUTPUT)
    result_auth.add_argument("--result", default=DEFAULT_RESULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preregister":
        destination = _rooted(args.output)
        if destination.exists():
            payload = authenticate_preregistration(destination)
        else:
            preflight = _run_routing_preflight()
            payload = build_preregistration(preflight)
            _atomic_create(destination, payload)
    elif args.command == "authenticate":
        payload = authenticate_preregistration(args.preregistration)
    elif args.command == "result":
        payload = evaluate_successor(args.preregistration)
        _atomic_create(_rooted(args.output), payload)
    else:
        payload = authenticate_result(args.preregistration, args.result)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if payload.get("passed", True) is True else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACCEPTANCE_GATES",
    "CALIBRATION",
    "DEFAULT_OUTPUT",
    "DEFAULT_RESULT_OUTPUT",
    "ROUTING_TEST_NODE",
    "RUNTIME_VERSION",
    "SUCCESSOR_OUTPUTS",
    "SUCCESSOR_RUNTIME_OUTPUTS",
    "authenticate_preregistration",
    "authenticate_result",
    "build_preregistration",
    "evaluate_successor",
]
