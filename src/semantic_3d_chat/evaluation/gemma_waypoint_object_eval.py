"""Oracle-isolated evaluation of live Gemma waypoint object goals.

The module intentionally exposes two disjoint stages:

``capture_live_goal``
    Talks only to the loopback rover public API.  It has no oracle argument and
    returns a sanitized, self-contained runtime evidence record.

``score_runtime_files``
    First validates every completed runtime record, then opens the physically
    separate evaluation oracle and computes geometric metrics.

This split makes it impossible for a live capture invocation to accidentally
load simulator labels or object geometry.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

RUNTIME_SCHEMA = "semantic_3d_chat.gemma_waypoint_object_runtime.v1"
SCORE_SCHEMA = "semantic_3d_chat.gemma_waypoint_object_oracle_score.v1"
CONTROL_MODE = "actual_local_gemma_model_only_waypoint_policy"


@dataclass(frozen=True)
class GoalDefinition:
    goal_id: str
    instruction: str
    target_category: str
    metric: str


GOALS: dict[str, GoalDefinition] = {
    "face_cube": GoalDefinition(
        goal_id="face_cube",
        instruction="Face the cube, then stop.",
        target_category="cube",
        metric="face_yaw",
    ),
    "approach_chair": GoalDefinition(
        goal_id="approach_chair",
        instruction="Move close to the chair, then stop.",
        target_category="chair",
        metric="approach_standoff",
    ),
}


HttpRequest = Callable[[str, str, Mapping[str, Any] | None, float], dict[str, Any]]


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    result = _finite(value, name)
    if not result.is_integer() or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(result)


def _xy(value: object, name: str) -> tuple[float, float]:
    sequence = _sequence(value, name)
    if len(sequence) != 2:
        raise ValueError(f"{name} must contain exactly two numbers")
    return _finite(sequence[0], f"{name}[0]"), _finite(sequence[1], f"{name}[1]")


def _validate_loopback_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("live capture is restricted to a plain loopback HTTP origin")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("loopback URL has an invalid port") from exc
    if port is not None and not 1 <= port <= 65_535:
        raise ValueError("loopback URL has an invalid port")
    return base_url.rstrip("/")


def _http_request(
    base_url: str,
    path: str,
    payload: Mapping[str, Any] | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    body = None
    method = "GET"
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib.request.Request(f"{base_url}{path}", data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} failed: {error.code}: {detail}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{method} {path} returned a non-object")
    return value


def _assert_public_payload_has_no_oracle_fields(value: object, path: str = "$") -> None:
    """Reject evaluator-only metadata from the public runtime payload.

    User-authored goal text and model replies may naturally mention an object.
    The guard is therefore about metadata field names, not free text.
    """

    prohibited = {
        "oracle",
        "category",
        "instance_id",
        "expected_center",
        "bounding_box",
        "bbox",
        "relationship",
        "scene_description",
        "object_labels",
        "target_xyz",
    }
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).casefold()
            if any(fragment in key for fragment in prohibited):
                raise ValueError(
                    f"public runtime payload exposes evaluation metadata at {path}.{key}"
                )
            _assert_public_payload_has_no_oracle_fields(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _assert_public_payload_has_no_oracle_fields(child, f"{path}[{index}]")


def _state(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = _mapping(payload.get("state"), f"{name}.state")
    scene_id = value.get("scene_id")
    if scene_id != "scene_000001":
        raise ValueError(f"{name} must be the registered evaluation scene scene_000001")
    collision = value.get("collision")
    stopped = value.get("stopped")
    if type(collision) is not bool or type(stopped) is not bool:
        raise TypeError(f"{name} collision and stopped fields must be boolean")
    prefix_hash = str(value.get("scene_prefix_hash", "")).casefold()
    if len(prefix_hash) != 64 or any(
        character not in "0123456789abcdef" for character in prefix_hash
    ):
        raise ValueError(f"{name} scene-prefix hash must be SHA-256")
    return {
        "scene_id": scene_id,
        "position_xy_m": list(_xy(value.get("position_xy_m"), f"{name}.position_xy_m")),
        "body_yaw_degrees": _finite(value.get("body_yaw_degrees"), f"{name}.body_yaw_degrees"),
        "collision": collision,
        "stopped": stopped,
        "action_count": _integer(value.get("action_count"), f"{name}.action_count"),
        "map_version": _integer(value.get("map_version"), f"{name}.map_version"),
        "scan_count": _integer(value.get("scan_count"), f"{name}.scan_count"),
        "scene_prefix_hash": prefix_hash,
    }


def _memory_signature(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    memory = _mapping(payload.get("scene_memory"), f"{name}.scene_memory")
    audit = _mapping(memory.get("loaded_file_audit"), f"{name}.loaded_file_audit")
    tensor_shape = list(_sequence(memory.get("tensor_shape"), f"{name}.tensor_shape"))
    active_shape = list(_sequence(memory.get("active_tensor_shape"), f"{name}.active_tensor_shape"))
    if (
        len(tensor_shape) != 3
        or len(active_shape) != 3
        or any(_integer(item, f"{name}.shape", minimum=1) <= 0 for item in tensor_shape)
        or any(_integer(item, f"{name}.active_shape", minimum=1) <= 0 for item in active_shape)
    ):
        raise ValueError(f"{name} scene-memory shapes are invalid")
    required_true = (
        "all_runtime_voxels_encoded",
        "base_adapter_weights_loaded",
        "control_weights_loaded",
        "control_training_gate_passed",
    )
    if any(memory.get(field) is not True for field in required_true):
        raise ValueError(f"{name} scene-memory checkpoint attestation failed")
    if (
        memory.get("question_dependent_scene_retrieval") is not False
        or memory.get("environmental_text_inputs") != []
        or audit.get("passed") is not True
        or audit.get("forbidden_access_count") != 0
    ):
        raise ValueError(f"{name} scene-memory isolation attestation failed")
    return {
        "scene_prefix_sha256": str(memory.get("sha256", "")),
        "tensor_shape": tensor_shape,
        "token_count": _integer(memory.get("token_count"), f"{name}.token_count", minimum=1),
        "model_dim": _integer(memory.get("model_dim"), f"{name}.model_dim", minimum=1),
        "robot_state_token_count": _integer(
            memory.get("robot_state_token_count"),
            f"{name}.robot_state_token_count",
            minimum=1,
        ),
        "source_voxels": _integer(memory.get("source_voxels"), f"{name}.source_voxels", minimum=1),
        "processed_voxels": _integer(
            memory.get("processed_voxels"), f"{name}.processed_voxels", minimum=1
        ),
        "semantic_feature_dim": _integer(
            memory.get("semantic_feature_dim"),
            f"{name}.semantic_feature_dim",
            minimum=1,
        ),
        "map_version": _integer(memory.get("map_version"), f"{name}.map_version"),
        "forbidden_access_count": 0,
        "question_dependent_scene_retrieval": False,
    }


def _verify_control(payload: Mapping[str, Any], *, attempted: bool) -> None:
    control = _mapping(payload.get("control"), "control")
    expected = {
        "control_mode": CONTROL_MODE,
        "gemma_attempted": attempted,
        "fallback_used": False,
        "local_inference": True,
        "cloud_model_used": False,
        "high_level_natural_language_only": True,
        "task_trained_navigation": True,
        "untrained_json_backend_enabled": False,
        "static_precomputed_scene_memory": True,
        "camera_control_input": False,
    }
    for field, expected_value in expected.items():
        if control.get(field) != expected_value:
            raise ValueError(f"live controller contract failed: control.{field}")
    if attempted and control.get("gemma_accepted") is not True:
        raise ValueError("Gemma did not accept and complete the high-level goal")


def _validated_decisions(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = _sequence(payload.get("model_decisions"), "model_decisions")
    if not raw:
        raise ValueError("live episode contains no Gemma decisions")
    decisions: list[dict[str, Any]] = []
    for expected_step, value in enumerate(raw, start=1):
        item = dict(_mapping(value, f"model_decisions[{expected_step - 1}]"))
        expected = {
            "step": expected_step,
            "actual_gemma_causal_forward": True,
            "model_selected_every_waypoint_and_heading": True,
            "deterministic_route_planner_used": False,
            "substitution_applied": False,
            "synthetic_stop_applied": False,
        }
        if any(item.get(field) != wanted for field, wanted in expected.items()):
            raise ValueError(f"Gemma decision provenance failed at step {expected_step}")
        if item.get("model_action") not in {"move_to", "face", "stop"}:
            raise ValueError(f"unknown model action at step {expected_step}")
        if type(item.get("accepted")) is not bool or type(item.get("executed")) is not bool:
            raise TypeError(f"Gemma decision status is not boolean at step {expected_step}")
        decisions.append(item)
    terminal = decisions[-1]
    if (
        terminal.get("model_action") != "stop"
        or terminal.get("accepted") is not True
        or terminal.get("executed") is not True
    ):
        raise ValueError("terminal action was not an accepted Gemma-selected STOP")
    return decisions


def _sanitized_actions(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = _sequence(payload.get("actions"), "actions")
    actions: list[dict[str, Any]] = []
    for index, value in enumerate(raw):
        item = _mapping(value, f"actions[{index}]")
        sanitized: dict[str, Any] = {}
        if "position_xy_m" in item:
            sanitized["position_xy_m"] = list(
                _xy(item["position_xy_m"], f"actions[{index}].position_xy_m")
            )
        for field in ("distance_moved", "turn_degrees"):
            if field in item:
                sanitized[field] = _finite(item[field], f"actions[{index}].{field}")
        for field in ("success", "collision", "stopped"):
            if field in item:
                if type(item[field]) is not bool:
                    raise TypeError(f"actions[{index}].{field} must be boolean")
                sanitized[field] = item[field]
        if "error_code" in item:
            error = item["error_code"]
            if error is not None and not isinstance(error, str):
                raise TypeError(f"actions[{index}].error_code must be text or null")
            sanitized["error_code"] = error
        actions.append(sanitized)
    return actions


def capture_live_goal(
    base_url: str,
    goal_id: str,
    *,
    timeout_seconds: float = 1_800.0,
    request_fn: HttpRequest = _http_request,
) -> dict[str, Any]:
    """Capture one fresh live episode without accepting or reading oracle data."""

    origin = _validate_loopback_base_url(base_url)
    if goal_id not in GOALS:
        raise ValueError(f"unknown goal {goal_id!r}; expected one of {sorted(GOALS)}")
    timeout = _finite(timeout_seconds, "timeout_seconds")
    if timeout <= 0.0:
        raise ValueError("timeout_seconds must be positive")
    goal = GOALS[goal_id]

    startup = request_fn(origin, "/api/state", None, timeout)
    _assert_public_payload_has_no_oracle_fields(startup)
    _verify_control(startup, attempted=False)
    startup_state = _state(startup, "startup")
    startup_memory = _memory_signature(startup, "startup")
    if startup_state["action_count"] != 0:
        raise ValueError("evaluation requires a freshly started rover with action_count == 0")

    started = time.perf_counter()
    result = request_fn(
        origin,
        "/api/instruction",
        {"instruction": goal.instruction},
        timeout,
    )
    elapsed_seconds = time.perf_counter() - started
    _assert_public_payload_has_no_oracle_fields(result)
    _verify_control(result, attempted=True)
    final_state = _state(result, "result")
    final_memory = _memory_signature(result, "result")
    decisions = _validated_decisions(result)
    actions = _sanitized_actions(result)
    if final_state["scene_id"] != startup_state["scene_id"]:
        raise ValueError("scene changed during the live goal")
    if final_memory != startup_memory:
        raise ValueError("static continuous scene memory changed during the live goal")
    if final_state["scene_prefix_hash"] != startup_state["scene_prefix_hash"]:
        raise ValueError("question-independent scene-prefix hash changed during the goal")
    if startup_memory["scene_prefix_sha256"] != startup_state["scene_prefix_hash"]:
        raise ValueError("continuous scene memory is not bound to the rover state")

    return {
        "schema": RUNTIME_SCHEMA,
        "goal_id": goal.goal_id,
        "instruction": goal.instruction,
        "scene_id": startup_state["scene_id"],
        "capture_transport": "loopback_public_http_api",
        "elapsed_seconds": elapsed_seconds,
        "startup_state": startup_state,
        "final_state": final_state,
        "scene_memory": startup_memory,
        "model_decisions": decisions,
        "actions": actions,
        "model_decision_count": len(decisions),
        "accepted_decision_count": sum(item.get("accepted") is True for item in decisions),
        "rejected_decision_count": sum(item.get("accepted") is not True for item in decisions),
        "path_length_m": sum(float(item.get("distance_moved", 0.0)) for item in actions),
        "model_selected_terminal_stop": True,
        "actual_gemma_causal_forward_every_step": True,
        "model_selected_every_waypoint_and_heading": True,
        "deterministic_route_planner_used": False,
        "substitution_applied": False,
        "synthetic_stop_applied": False,
        "fallback_used": False,
        "local_inference": True,
        "cloud_model_used": False,
        "runtime_forbidden_access_count": 0,
        "runtime_environmental_metadata_fields": [],
    }


def validate_runtime_record(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a persisted capture completely before oracle access is allowed."""

    if value.get("schema") != RUNTIME_SCHEMA:
        raise ValueError("runtime record has the wrong schema")
    _assert_public_payload_has_no_oracle_fields(value)
    goal_id = value.get("goal_id")
    if not isinstance(goal_id, str) or goal_id not in GOALS:
        raise ValueError("runtime record has an unknown goal")
    goal = GOALS[goal_id]
    expected = {
        "instruction": goal.instruction,
        "scene_id": "scene_000001",
        "capture_transport": "loopback_public_http_api",
        "model_selected_terminal_stop": True,
        "actual_gemma_causal_forward_every_step": True,
        "model_selected_every_waypoint_and_heading": True,
        "deterministic_route_planner_used": False,
        "substitution_applied": False,
        "synthetic_stop_applied": False,
        "fallback_used": False,
        "local_inference": True,
        "cloud_model_used": False,
        "runtime_forbidden_access_count": 0,
        "runtime_environmental_metadata_fields": [],
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise ValueError(f"runtime record contract failed: {field}")
    startup = _mapping(value.get("startup_state"), "startup_state")
    final = _mapping(value.get("final_state"), "final_state")
    startup_xy = _xy(startup.get("position_xy_m"), "startup_state.position_xy_m")
    final_xy = _xy(final.get("position_xy_m"), "final_state.position_xy_m")
    if startup.get("action_count") != 0:
        raise ValueError("runtime record did not start fresh")
    if startup.get("scene_prefix_hash") != final.get("scene_prefix_hash"):
        raise ValueError("runtime record scene prefix changed")
    decisions = _validated_decisions(value)
    actions = _sanitized_actions(value)
    accepted_count = sum(item.get("accepted") is True for item in decisions)
    rejected_count = len(decisions) - accepted_count
    recorded_path_length = _finite(value.get("path_length_m"), "path_length_m")
    calculated_path_length = sum(float(item.get("distance_moved", 0.0)) for item in actions)
    if (
        value.get("model_decision_count") != len(decisions)
        or value.get("accepted_decision_count") != accepted_count
        or value.get("rejected_decision_count") != rejected_count
        or not math.isclose(recorded_path_length, calculated_path_length, abs_tol=1e-9)
    ):
        raise ValueError("runtime record decision or path aggregates disagree")
    return {
        "goal": goal,
        "scene_id": "scene_000001",
        "startup_xy_m": startup_xy,
        "final_xy_m": final_xy,
        "final_yaw_degrees": _finite(final.get("body_yaw_degrees"), "final_state.body_yaw_degrees"),
        "elapsed_seconds": _finite(value.get("elapsed_seconds"), "elapsed_seconds"),
        "path_length_m": recorded_path_length,
        "decision_count": _integer(
            value.get("model_decision_count"), "model_decision_count", minimum=1
        ),
        "accepted_decision_count": _integer(
            value.get("accepted_decision_count"), "accepted_decision_count", minimum=1
        ),
        "rejected_decision_count": _integer(
            value.get("rejected_decision_count"), "rejected_decision_count"
        ),
        "scene_prefix_sha256": str(startup.get("scene_prefix_hash", "")),
    }


def _planar_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def _point_to_xy_box_distance(
    point: Sequence[float], minimum: Sequence[float], maximum: Sequence[float]
) -> float:
    dx = max(float(minimum[0]) - float(point[0]), 0.0, float(point[0]) - float(maximum[0]))
    dy = max(float(minimum[1]) - float(point[1]), 0.0, float(point[1]) - float(maximum[1]))
    return math.hypot(dx, dy)


def _shortest_angle_error_degrees(observed: float, expected: float) -> float:
    return abs((float(observed) - float(expected) + 180.0) % 360.0 - 180.0)


def _target_from_oracle(oracle: Mapping[str, Any], category: str) -> Mapping[str, Any]:
    instances = oracle.get("instances")
    if not isinstance(instances, list):
        raise TypeError("oracle instances must be a list")
    matches = [
        item for item in instances if isinstance(item, Mapping) and item.get("category") == category
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one oracle instance for {category!r}")
    return matches[0]


def score_runtime_files(
    runtime_paths: Sequence[str | Path],
    *,
    oracle_root: str | Path,
    maximum_face_yaw_error_degrees: float = 20.0,
    minimum_chair_progress_m: float = 0.25,
    maximum_chair_bbox_standoff_m: float = 0.60,
) -> dict[str, Any]:
    """Score completed runtime captures after validating all of them first."""

    if not runtime_paths:
        raise ValueError("at least one runtime capture is required")
    yaw_threshold = _finite(maximum_face_yaw_error_degrees, "maximum face yaw error")
    progress_threshold = _finite(minimum_chair_progress_m, "minimum chair progress")
    standoff_threshold = _finite(maximum_chair_bbox_standoff_m, "maximum chair standoff")
    if min(yaw_threshold, progress_threshold, standoff_threshold) < 0.0:
        raise ValueError("score thresholds must be non-negative")

    # Security-critical ordering: parse and authenticate every runtime capture
    # before resolving, opening, or parsing an oracle path.
    validated: list[tuple[Path, dict[str, Any]]] = []
    for raw_path in runtime_paths:
        path = Path(os.path.abspath(Path(raw_path).expanduser()))
        record = _read_object(path)
        validated.append((path, validate_runtime_record(record)))
    goal_ids = [row[1]["goal"].goal_id for row in validated]
    if len(goal_ids) != len(set(goal_ids)):
        raise ValueError("runtime captures must contain distinct goals")

    root = Path(os.path.abspath(Path(oracle_root).expanduser()))
    oracle_path = root / "scene_000001" / "oracle.json"
    oracle = _read_object(oracle_path)
    if oracle.get("scene_id") != "scene_000001":
        raise ValueError("oracle scene differs from the completed runtime captures")

    rows: list[dict[str, Any]] = []
    for path, runtime in validated:
        goal: GoalDefinition = runtime["goal"]
        target = _target_from_oracle(oracle, goal.target_category)
        center_raw = _sequence(target.get("expected_center_xyz_m"), "target center")
        if len(center_raw) < 2:
            raise ValueError("oracle target center must contain at least XY")
        center_xy = _xy(center_raw[:2], "target center XY")
        initial_distance = _planar_distance(runtime["startup_xy_m"], center_xy)
        final_distance = _planar_distance(runtime["final_xy_m"], center_xy)
        common = {
            "goal_id": goal.goal_id,
            "instruction": goal.instruction,
            "runtime_file": path.as_posix(),
            "runtime_file_sha256": _sha256(path),
            "target_category": goal.target_category,
            "target_instance_id": target.get("instance_id"),
            "target_center_xyz_m": list(center_raw),
            "initial_position_xy_m": list(runtime["startup_xy_m"]),
            "final_position_xy_m": list(runtime["final_xy_m"]),
            "initial_target_center_distance_m": initial_distance,
            "final_target_center_distance_m": final_distance,
            "target_center_progress_m": initial_distance - final_distance,
            "path_length_m": runtime["path_length_m"],
            "elapsed_seconds": runtime["elapsed_seconds"],
            "model_decision_count": runtime["decision_count"],
            "accepted_decision_count": runtime["accepted_decision_count"],
            "rejected_decision_count": runtime["rejected_decision_count"],
            "scene_prefix_sha256": runtime["scene_prefix_sha256"],
            "model_selected_terminal_stop": True,
        }
        if goal.metric == "face_yaw":
            dx = center_xy[0] - runtime["final_xy_m"][0]
            dy = center_xy[1] - runtime["final_xy_m"][1]
            # Project convention: yaw 0 faces +Y, -90 faces +X.
            oracle_heading = math.degrees(math.atan2(-dx, dy))
            yaw_error = _shortest_angle_error_degrees(runtime["final_yaw_degrees"], oracle_heading)
            checks = {
                "accepted_gemma_stop": True,
                "maximum_oracle_yaw_error": yaw_error <= yaw_threshold,
            }
            row = {
                **common,
                "metric": "face_yaw",
                "final_body_yaw_degrees": runtime["final_yaw_degrees"],
                "oracle_target_heading_degrees": oracle_heading,
                "oracle_yaw_error_degrees": yaw_error,
                "maximum_yaw_error_degrees": yaw_threshold,
                "checks": checks,
                "passed": all(checks.values()),
            }
        elif goal.metric == "approach_standoff":
            bbox = _mapping(target.get("bbox"), "target bbox")
            minimum_raw = _sequence(bbox.get("min_xyz_m"), "bbox minimum")
            maximum_raw = _sequence(bbox.get("max_xyz_m"), "bbox maximum")
            standoff = _point_to_xy_box_distance(runtime["final_xy_m"], minimum_raw, maximum_raw)
            progress = initial_distance - final_distance
            checks = {
                "accepted_gemma_stop": True,
                "minimum_oracle_center_progress": progress >= progress_threshold,
                "maximum_oracle_bbox_standoff": standoff <= standoff_threshold,
            }
            row = {
                **common,
                "metric": "approach_standoff",
                "final_oracle_bbox_standoff_m": standoff,
                "minimum_center_progress_m": progress_threshold,
                "maximum_bbox_standoff_m": standoff_threshold,
                "checks": checks,
                "passed": all(checks.values()),
            }
        else:
            raise AssertionError(f"unsupported registered metric: {goal.metric}")
        rows.append(row)

    return {
        "schema": SCORE_SCHEMA,
        "status": "evaluation_only_oracle_score",
        "runtime_capture_process_accepts_oracle_arguments": False,
        "all_runtime_evidence_validated_before_oracle_open": True,
        "runtime_process_read_oracle": False,
        "scorer_reads_oracle": True,
        "oracle_file": oracle_path.as_posix(),
        "oracle_file_sha256": _sha256(oracle_path),
        "scene_id": "scene_000001",
        "goal_count": len(rows),
        "passed_count": sum(row["passed"] for row in rows),
        "all_passed": all(row["passed"] for row in rows),
        "goals": rows,
    }


__all__ = [
    "GOALS",
    "RUNTIME_SCHEMA",
    "SCORE_SCHEMA",
    "_atomic_json",
    "capture_live_goal",
    "score_runtime_files",
    "validate_runtime_record",
]
