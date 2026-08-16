#!/usr/bin/env python3
"""Run the live, evaluator-separated acceptance test for the semantic rover.

The rover server is treated as an inference-only black box.  This process may
optionally read oracle geometry *after* receiving each numeric rover result in
order to score it; those files are never sent to, or opened by, the server.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "reports/gemma4/metrics/semantic_goal_live_acceptance.json"
DEFAULT_ORACLE = PROJECT_ROOT / "data/oracle/scene_000001/oracle.json"
REQUIRED_CONTROL = {
    "local_inference": True,
    "cloud_model_used": False,
    "high_level_natural_language_only": True,
    "task_trained_navigation": True,
    "untrained_json_backend_enabled": False,
    "static_precomputed_scene_memory": True,
    "camera_control_input": False,
}


def _request_json(
    base_url: str,
    path: str,
    payload: Mapping[str, Any] | None = None,
    *,
    timeout: float = 180.0,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}", data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} failed: HTTP {exc.code}: {detail}") from exc
    if not isinstance(result, dict):
        raise TypeError(f"{method} {path} returned a non-object JSON value")
    return result


def _state(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = payload.get("state")
    if not isinstance(value, Mapping):
        raise TypeError("Rover response has no numeric state object")
    return dict(value)


def _memory(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = payload.get("scene_memory")
    if not isinstance(value, Mapping):
        raise TypeError("Rover response has no scene-memory diagnostic object")
    return dict(value)


def _control(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = payload.get("control")
    if not isinstance(value, Mapping):
        raise TypeError("Rover response has no control diagnostic object")
    return dict(value)


def _xy(state: Mapping[str, Any]) -> tuple[float, float]:
    value = state.get("position_xy_m")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError("Rover state has no two-dimensional position")
    result = float(value[0]), float(value[1])
    if not all(math.isfinite(item) for item in result):
        raise ValueError("Rover position is not finite")
    return result


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.hypot(float(first[0]) - float(second[0]), float(first[1]) - float(second[1]))


def _yaw_error(actual_degrees: float, expected_degrees: float) -> float:
    return abs((actual_degrees - expected_degrees + 180.0) % 360.0 - 180.0)


def _actions(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = payload.get("actions", [])
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise TypeError("Rover response contains invalid action receipts")
    return [dict(item) for item in value]


def _static_signature(payload: Mapping[str, Any]) -> tuple[Any, ...]:
    state = _state(payload)
    memory = _memory(payload)
    return (
        state.get("scene_id"),
        state.get("scene_prefix_hash"),
        state.get("map_version"),
        state.get("scan_count"),
        memory.get("sha256"),
        memory.get("tensor_shape"),
        memory.get("source_voxels"),
        memory.get("processed_voxels"),
        memory.get("semantic_feature_dim"),
    )


def _assert_static(
    payload: Mapping[str, Any],
    expected: tuple[Any, ...],
    *,
    label: str,
) -> None:
    if _static_signature(payload) != expected:
        raise AssertionError(f"{label} changed the fixed scene map or continuous prefix")
    state = _state(payload)
    memory = _memory(payload)
    if state.get("scan_count") != 0 or state.get("map_version") != 0:
        raise AssertionError(f"{label} captured a rover-camera scan or mutated the map")
    if memory.get("question_dependent_scene_retrieval") is not False:
        raise AssertionError(f"{label} enabled question-dependent scene retrieval")


def _oracle_centers(path: Path) -> dict[str, tuple[float, float, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    instances = payload.get("instances") if isinstance(payload, Mapping) else None
    if not isinstance(instances, list):
        raise TypeError("Oracle evaluator file has no instances list")
    result: dict[str, tuple[float, float, float]] = {}
    for item in instances:
        if not isinstance(item, Mapping):
            continue
        category = item.get("category")
        xyz = item.get("expected_center_xyz_m")
        if (
            isinstance(category, str)
            and isinstance(xyz, list)
            and len(xyz) == 3
            and all(isinstance(value, (int, float)) for value in xyz)
        ):
            result.setdefault(category, (float(xyz[0]), float(xyz[1]), float(xyz[2])))
    required = {"floor lamp", "bowl", "chair", "table"}
    if not required.issubset(result):
        raise ValueError("Oracle evaluator file lacks the required scoring centers")
    return result


def _timed_instruction(base_url: str, text: str) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    response = _request_json(base_url, "/api/instruction", {"instruction": text}, timeout=300.0)
    return response, time.perf_counter() - started


def _summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    state = _state(payload)
    memory = _memory(payload)
    control = _control(payload)
    return {
        "reply": str(payload.get("reply", "")),
        "position_xy_m": list(_xy(state)),
        "body_yaw_degrees": float(state.get("body_yaw_degrees", 0.0)),
        "action_count_total": int(state.get("action_count", 0)),
        "action_receipt_count": len(_actions(payload)),
        "scene_prefix_sha256": str(state.get("scene_prefix_hash", "")),
        "active_prefix_sha256": str(memory.get("active_sha256", "")),
        "map_version": int(state.get("map_version", -1)),
        "scan_count": int(state.get("scan_count", -1)),
        "control_mode": str(control.get("control_mode", "")),
        "gemma_attempted": bool(control.get("gemma_attempted", False)),
        "gemma_accepted": bool(control.get("gemma_accepted", False)),
        "fallback_used": bool(control.get("fallback_used", False)),
    }


def run_acceptance(base_url: str, oracle_path: Path) -> dict[str, Any]:
    parsed = urlsplit(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Acceptance verifier connects only to a loopback HTTP backend")
    oracle = _oracle_centers(oracle_path)

    startup = _request_json(base_url, "/api/state")
    startup_state = _state(startup)
    startup_memory = _memory(startup)
    startup_control = _control(startup)
    if startup_state.get("action_count") != 0:
        raise AssertionError("Acceptance requires a fresh rover backend with action_count=0")
    if any(startup_control.get(key) is not expected for key, expected in REQUIRED_CONTROL.items()):
        raise AssertionError("Live rover control contract does not match the static-map design")
    if (
        startup_memory.get("tensor_shape") != [1, 258, 1536]
        or startup_memory.get("active_tensor_shape") != [1, 262, 1536]
        or startup_memory.get("source_voxels") != 74_699
        or startup_memory.get("processed_voxels") != 8_422
        or startup_memory.get("semantic_feature_dim") != 3_072
        or startup_memory.get("all_runtime_voxels_encoded") is not True
        or startup_memory.get("environmental_text_inputs") != []
    ):
        raise AssertionError("Live continuous-memory tensor contract failed")
    static_signature = _static_signature(startup)
    start_xy = _xy(startup_state)
    start_yaw = float(startup_state["body_yaw_degrees"])

    low_level, low_level_seconds = _timed_instruction(base_url, "turn right 90 degrees")
    _assert_static(low_level, static_signature, label="low-level rejection")
    low_state = _state(low_level)
    low_control = _control(low_level)
    low_level_passed = bool(
        not _actions(low_level)
        and _xy(low_state) == start_xy
        and float(low_state["body_yaw_degrees"]) == start_yaw
        and low_state.get("action_count") == 0
        and low_control.get("control_mode") == "low_level_chat_command_disabled"
    )
    if not low_level_passed:
        raise AssertionError("Low-level user motor command was not rejected without side effects")

    lap, lap_seconds = _timed_instruction(base_url, "do a lap around the room")
    _assert_static(lap, static_signature, label="room patrol")
    lap_actions = _actions(lap)
    lap_distance = sum(float(item.get("distance_moved", 0.0)) for item in lap_actions)
    lap_return_error = _distance(_xy(_state(lap)), start_xy)
    lap_passed = bool(len(lap_actions) >= 4 and lap_distance >= 5.0 and lap_return_error <= 0.10)
    if not lap_passed:
        raise AssertionError("Global-map room patrol failed its closed-loop movement checks")

    before_face_xy = _xy(_state(lap))
    lamp = oracle["floor lamp"]
    expected_lamp_yaw = math.degrees(
        math.atan2(-(lamp[0] - before_face_xy[0]), lamp[1] - before_face_xy[1])
    )
    face, face_seconds = _timed_instruction(base_url, "face the floor lamp")
    _assert_static(face, static_signature, label="face goal")
    actual_lamp_yaw = float(_state(face)["body_yaw_degrees"])
    lamp_yaw_error = _yaw_error(actual_lamp_yaw, expected_lamp_yaw)
    face_passed = bool(_actions(face) and lamp_yaw_error <= 10.0)
    if not face_passed:
        raise AssertionError("Semantic face goal did not align with the evaluator-only target")

    bowl = oracle["bowl"]
    bowl_before_xy = _xy(_state(face))
    bowl_before_distance = _distance(bowl_before_xy, bowl)
    approach, approach_seconds = _timed_instruction(base_url, "move close to the bowl")
    _assert_static(approach, static_signature, label="approach goal")
    bowl_after_xy = _xy(_state(approach))
    bowl_after_distance = _distance(bowl_after_xy, bowl)
    bowl_progress = bowl_before_distance - bowl_after_distance
    approach_passed = bool(
        _actions(approach) and bowl_progress >= 0.25 and bowl_after_distance <= 1.0
    )
    if not approach_passed:
        raise AssertionError("Semantic approach goal did not make useful target progress")

    chair = oracle["chair"]
    table = oracle["table"]
    midpoint = ((chair[0] + table[0]) / 2.0, (chair[1] + table[1]) / 2.0)
    between, between_seconds = _timed_instruction(base_url, "stop between the chair and the table")
    _assert_static(between, static_signature, label="between goal")
    midpoint_error = _distance(_xy(_state(between)), midpoint)
    between_passed = bool(_actions(between) and midpoint_error <= 0.75)
    if not between_passed:
        raise AssertionError("Two-region semantic goal missed the evaluator-only midpoint")

    final_memory = _memory(between)
    file_audit = final_memory.get("loaded_file_audit")
    audit_passed = bool(
        isinstance(file_audit, Mapping)
        and file_audit.get("passed") is True
        and file_audit.get("forbidden_access_count") == 0
    )
    if not audit_passed:
        raise AssertionError("Runtime loaded-file audit reported forbidden access")

    return {
        "schema": "semantic_3d_chat.semantic_goal_live_acceptance.v1",
        "passed": True,
        "scope": "single-scene live local proof of concept",
        "backend_url": base_url,
        "evaluator_separation": {
            "runtime_received_oracle_data": False,
            "oracle_used_only_by_this_scoring_process": True,
            "oracle_path": str(oracle_path),
        },
        "continuous_scene_memory": {
            "scene_id": startup_state["scene_id"],
            "tensor_shape": startup_memory["tensor_shape"],
            "active_tensor_shape": startup_memory["active_tensor_shape"],
            "scene_prefix_sha256": startup_memory["sha256"],
            "l2_norm": startup_memory["l2_norm"],
            "rms": startup_memory["rms"],
            "source_voxels": startup_memory["source_voxels"],
            "processed_spatial_representations": startup_memory["processed_voxels"],
            "semantic_feature_dim": startup_memory["semantic_feature_dim"],
            "all_runtime_voxels_encoded": startup_memory["all_runtime_voxels_encoded"],
            "question_dependent_scene_retrieval": startup_memory[
                "question_dependent_scene_retrieval"
            ],
            "unchanged_across_every_goal": True,
            "map_version": _state(between)["map_version"],
            "scan_count": _state(between)["scan_count"],
        },
        "control_contract": dict(REQUIRED_CONTROL),
        "checks": {
            "low_level_user_command_rejected": {
                "passed": low_level_passed,
                "elapsed_seconds": low_level_seconds,
                **_summary(low_level),
            },
            "global_map_room_patrol": {
                "passed": lap_passed,
                "elapsed_seconds": lap_seconds,
                "path_length_m": lap_distance,
                "return_to_start_error_m": lap_return_error,
                **_summary(lap),
            },
            "face_floor_lamp": {
                "passed": face_passed,
                "elapsed_seconds": face_seconds,
                "evaluator_expected_yaw_degrees": expected_lamp_yaw,
                "evaluator_yaw_error_degrees": lamp_yaw_error,
                **_summary(face),
            },
            "approach_bowl": {
                "passed": approach_passed,
                "elapsed_seconds": approach_seconds,
                "evaluator_distance_before_m": bowl_before_distance,
                "evaluator_distance_after_m": bowl_after_distance,
                "evaluator_progress_m": bowl_progress,
                **_summary(approach),
            },
            "between_chair_and_table": {
                "passed": between_passed,
                "elapsed_seconds": between_seconds,
                "evaluator_midpoint_xy_m": list(midpoint),
                "evaluator_midpoint_error_m": midpoint_error,
                **_summary(between),
            },
            "runtime_loaded_file_audit": dict(file_audit),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8770")
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_acceptance(args.base_url, args.oracle.expanduser().resolve())
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    print(f"WROTE {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
