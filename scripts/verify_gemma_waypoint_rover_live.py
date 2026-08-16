#!/usr/bin/env python3
"""Verify the live model-only Gemma waypoint loop without reading oracle data."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "reports/gemma4/metrics/gemma_waypoint_live_acceptance.json"
MIN_LAP_PATH_LENGTH_M = 5.0
MAX_LAP_RETURN_ERROR_M = 0.35
MIN_LAP_ABS_WINDING_AREA_M2 = 0.5
_SHA256_HEX = frozenset("0123456789abcdef")


def _request(
    base_url: str,
    path: str,
    payload: Mapping[str, Any] | None = None,
    *,
    timeout: float = 360.0,
) -> dict[str, Any]:
    body = None
    headers = {"Accept": "application/json"}
    method = "GET"
    if payload is not None:
        body = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} failed: {error.code}: {detail}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{method} {path} returned a non-object")
    return value


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _xy(payload: Mapping[str, Any]) -> tuple[float, float]:
    state = _mapping(payload.get("state"), "state")
    value = state.get("position_xy_m")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError("state.position_xy_m must contain two numbers")
    result = float(value[0]), float(value[1])
    if not all(math.isfinite(item) for item in result):
        raise ValueError("state.position_xy_m is non-finite")
    return result


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.hypot(float(first[0]) - float(second[0]), float(first[1]) - float(second[1]))


def _finite_xy(value: object, *, name: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must contain two finite numbers")
    if any(type(item) is bool for item in value):
        raise TypeError(f"{name} must contain numbers, not booleans")
    try:
        result = float(value[0]), float(value[1])
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain two finite numbers") from error
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} is non-finite")
    return result


def _action_xy(value: Mapping[str, Any], *, index: int) -> tuple[float, float]:
    """Read one public numeric action pose, rejecting missing or conflicting forms."""

    xy_value = value.get("position_xy_m")
    xyz_value = value.get("position_m")
    if xy_value is None and xyz_value is None:
        raise ValueError(f"actions[{index}] has no numeric position")

    xy = None if xy_value is None else _finite_xy(xy_value, name=f"actions[{index}].position_xy_m")
    xyz_xy: tuple[float, float] | None = None
    if xyz_value is not None:
        if (
            not isinstance(xyz_value, Sequence)
            or isinstance(xyz_value, (str, bytes))
            or len(xyz_value) != 3
        ):
            raise ValueError(f"actions[{index}].position_m must contain three finite numbers")
        if any(type(item) is bool for item in xyz_value):
            raise TypeError(f"actions[{index}].position_m must contain numbers, not booleans")
        try:
            xyz = tuple(float(item) for item in xyz_value)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"actions[{index}].position_m must contain three finite numbers"
            ) from error
        if not all(math.isfinite(item) for item in xyz):
            raise ValueError(f"actions[{index}].position_m is non-finite")
        xyz_xy = xyz[0], xyz[1]

    if xy is not None and xyz_xy is not None and _distance(xy, xyz_xy) > 1e-6:
        raise ValueError(f"actions[{index}] contains conflicting numeric positions")
    if xy is not None:
        return xy
    if xyz_xy is None:  # Defensive proof for type checkers; both-None failed above.
        raise AssertionError("Numeric action position unexpectedly disappeared")
    return xyz_xy


def _trajectory_metrics(
    start_xy: Sequence[float],
    actions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reconstruct a public-pose trajectory and compute closed signed area.

    Closing the final point back to the initial point is intentional: it computes
    the signed winding/polygon area of the traversed path.  Long out-and-back
    trajectories therefore have zero area and cannot masquerade as a room lap.
    """

    points = [_finite_xy(start_xy, name="initial_position_xy_m")]
    for index, action in enumerate(actions):
        if not isinstance(action, Mapping):
            raise TypeError(f"actions[{index}] must be an object")
        points.append(_action_xy(action, index=index))

    path_length = sum(_distance(first, second) for first, second in pairwise(points))
    signed_area_twice = sum(
        first[0] * second[1] - second[0] * first[1]
        for first, second in pairwise(points + points[:1])
    )
    signed_area = 0.5 * signed_area_twice
    if not math.isfinite(path_length) or not math.isfinite(signed_area):
        raise ValueError("Reconstructed trajectory metrics are non-finite")
    return {
        "trajectory_point_count": len(points),
        "path_length_m": path_length,
        "signed_winding_area_m2": signed_area,
        "abs_winding_area_m2": abs(signed_area),
    }


def _decision_trajectory_metrics(
    start_xy: Sequence[float],
    decisions: Sequence[Mapping[str, Any]],
    *,
    final_xy: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Reconstruct the complete trajectory from every public model decision.

    The simulator action history is deliberately bounded and can contain only a
    suffix of a long goal.  The model-decision stream is untruncated and binds
    every MOVE_TO to its exact world-space waypoint.  Rejected actions and
    FACE/STOP decisions retain the current position; an executed action still
    contributes a completion point so the result remains receipt-aligned.
    """

    current = _finite_xy(start_xy, name="initial_position_xy_m")
    points = [current]
    executed_positions: list[tuple[float, float]] = []
    receipt_positions: list[tuple[float, float]] = []
    for index, raw in enumerate(decisions):
        decision = _mapping(raw, f"model_decisions[{index}]")
        action = decision.get("model_action")
        if action not in {"move_to", "face", "stop"}:
            raise ValueError(f"model_decisions[{index}].model_action is invalid")
        accepted = decision.get("accepted")
        executed = decision.get("executed")
        if type(accepted) is not bool or type(executed) is not bool:
            raise TypeError(f"model_decisions[{index}] acceptance status must be boolean")
        if accepted and not executed:
            raise ValueError(f"model_decisions[{index}] was accepted without execution")
        error_code = decision.get("error_code")
        if accepted:
            if error_code is not None:
                raise ValueError(f"model_decisions[{index}] accepted with an error code")
        elif not isinstance(error_code, str) or not error_code.startswith("E_"):
            raise ValueError(f"model_decisions[{index}] rejection lacks a protocol error")

        world_target = decision.get("derived_world_waypoint_xy_m")
        if action == "move_to":
            target = _finite_xy(
                world_target,
                name=f"model_decisions[{index}].derived_world_waypoint_xy_m",
            )
            # Only a successful primitive may change pose.  Both a preflight
            # rejection (executed=false) and a failed executed primitive retain
            # the exact prior position.
            if accepted:
                current = target
        elif world_target is not None:
            raise ValueError(f"model_decisions[{index}] non-MOVE action carries a waypoint")

        if executed:
            points.append(current)
            executed_positions.append(current)
            # The simulator records bounded motion receipts for FACE and
            # MOVE_TO primitives. Terminal STOP completes the model protocol
            # but does not append a simulator motion receipt.
            if action != "stop":
                receipt_positions.append(current)

    if final_xy is not None:
        expected_final = _finite_xy(final_xy, name="final_position_xy_m")
        if _distance(current, expected_final) > 1e-6:
            raise ValueError("Decision-reconstructed final position differs from runtime state")

    path_length = sum(_distance(first, second) for first, second in pairwise(points))
    signed_area_twice = sum(
        first[0] * second[1] - second[0] * first[1]
        for first, second in pairwise(points + points[:1])
    )
    signed_area = 0.5 * signed_area_twice
    if not math.isfinite(path_length) or not math.isfinite(signed_area):
        raise ValueError("Decision-reconstructed trajectory metrics are non-finite")
    return {
        "trajectory_source": "complete_model_decisions",
        "trajectory_point_count": len(points),
        "path_length_m": path_length,
        "signed_winding_area_m2": signed_area,
        "abs_winding_area_m2": abs(signed_area),
        "reconstructed_final_position_xy_m": list(current),
        "executed_completion_positions": executed_positions,
        "receipt_completion_positions": receipt_positions,
    }


def _verify_action_receipt_suffix(
    actions: Sequence[Mapping[str, Any]],
    receipt_completion_positions: Sequence[Sequence[float]],
) -> None:
    """Validate the bounded motion log against receipt-bearing decisions.

    STOP is an authenticated, executed model decision but is not a simulator
    motion receipt. The public action log may also be a bounded suffix, so its
    positions align with the final FACE/MOVE_TO completions rather than with
    every executed decision indiscriminately.
    """

    receipt_positions: list[tuple[float, float]] = []
    for index, action in enumerate(actions):
        if not isinstance(action, Mapping):
            raise TypeError(f"actions[{index}] must be an object")
        receipt_positions.append(_action_xy(action, index=index))
    if len(receipt_positions) > len(receipt_completion_positions):
        raise ValueError(
            "Action receipt log is longer than the receipt-bearing decision stream"
        )
    expected = (
        list(receipt_completion_positions[-len(receipt_positions) :])
        if receipt_positions
        else []
    )
    for index, (receipt, reconstructed) in enumerate(zip(receipt_positions, expected, strict=True)):
        if _distance(receipt, reconstructed) > 1e-6:
            raise ValueError(f"actions[{index}] differs from its reconstructed decision pose")


def _static_signature(payload: Mapping[str, Any]) -> tuple[Any, ...]:
    state = _mapping(payload.get("state"), "state")
    memory = _mapping(payload.get("scene_memory"), "scene_memory")
    return (
        state.get("scene_id"),
        state.get("scene_prefix_hash"),
        state.get("map_version"),
        state.get("scan_count"),
        memory.get("sha256"),
        memory.get("tensor_shape"),
        memory.get("active_tensor_shape"),
        memory.get("source_voxels"),
        memory.get("processed_voxels"),
        memory.get("semantic_feature_dim"),
    )


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or not set(value).issubset(_SHA256_HEX):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _verify_control(payload: Mapping[str, Any], *, attempted: bool) -> tuple[str, str]:
    control = _mapping(payload.get("control"), "control")
    expected = {
        "control_mode": "actual_local_gemma_model_only_waypoint_policy",
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
    if any(control.get(name) != value for name, value in expected.items()):
        raise AssertionError("Live controller differs from the model-only local contract")
    return (
        _sha256(
            control.get("navigation_checkpoint_sha256"),
            name="control.navigation_checkpoint_sha256",
        ),
        _sha256(
            control.get("gemma_runtime_binding_sha256"),
            name="control.gemma_runtime_binding_sha256",
        ),
    )


def _verify_decisions(
    payload: Mapping[str, Any],
    *,
    expected_scene_prefix_sha256: str,
    expected_navigation_checkpoint_sha256: str | None = None,
) -> list[dict[str, Any]]:
    raw = payload.get("model_decisions")
    if not isinstance(raw, list) or not raw:
        raise AssertionError("Navigation returned no Gemma decisions")
    decisions: list[dict[str, Any]] = []
    for expected_step, value in enumerate(raw, start=1):
        item = dict(_mapping(value, f"model_decisions[{expected_step - 1}]"))
        if (
            item.get("step") != expected_step
            or item.get("actual_gemma_causal_forward") is not True
            or item.get("scene_prefix_sha256") != expected_scene_prefix_sha256
            or (
                expected_navigation_checkpoint_sha256 is not None
                and item.get("checkpoint_sha256")
                != expected_navigation_checkpoint_sha256
            )
            or item.get("model_selected_every_waypoint_and_heading") is not True
            or item.get("deterministic_route_planner_used") is not False
            or item.get("substitution_applied") is not False
            or item.get("synthetic_stop_applied") is not False
        ):
            raise AssertionError(f"Gemma decision {expected_step} failed provenance")
        if item.get("model_action") == "stop" and expected_step != len(raw):
            raise AssertionError("Gemma selected STOP before the terminal decision")
        decisions.append(item)
    terminal = decisions[-1]
    if (
        terminal.get("model_action") != "stop"
        or terminal.get("accepted") is not True
        or terminal.get("executed") is not True
    ):
        raise AssertionError("Gemma did not select and execute the terminal STOP")
    return decisions


def _authenticate_snapshot_decisions(
    startup: Mapping[str, Any],
    result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Authenticate a failure snapshot before deriving any trajectory from it."""

    startup_identity = _verify_control(startup, attempted=False)
    result_identity = _verify_control(result, attempted=True)
    if result_identity != startup_identity:
        raise AssertionError("Navigation checkpoint identity changed during the live turn")
    startup_signature = _static_signature(startup)
    if _static_signature(result) != startup_signature:
        raise AssertionError("Navigation changed the static map or scene prefix")
    scene_prefix_sha256 = startup_signature[1]
    if not isinstance(scene_prefix_sha256, str):
        raise TypeError("Startup scene prefix hash is not text")
    return _verify_decisions(
        result,
        expected_scene_prefix_sha256=scene_prefix_sha256,
        expected_navigation_checkpoint_sha256=startup_identity[0],
    )


def run_acceptance(
    base_url: str,
    instruction: str,
    *,
    expected_navigation_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    parsed = urlsplit(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Live verification is restricted to a loopback backend")
    startup = _request(base_url, "/api/state")
    navigation_identity = _verify_control(startup, attempted=False)
    if expected_navigation_checkpoint_sha256 is not None:
        expected_digest = _sha256(
            expected_navigation_checkpoint_sha256,
            name="expected_navigation_checkpoint_sha256",
        )
        if navigation_identity[0] != expected_digest:
            raise AssertionError("Live navigation checkpoint differs from the requested candidate")
    startup_state = _mapping(startup.get("state"), "state")
    memory = _mapping(startup.get("scene_memory"), "scene_memory")
    audit = _mapping(memory.get("loaded_file_audit"), "loaded_file_audit")
    if (
        startup_state.get("action_count") != 0
        or memory.get("tensor_shape") != [1, 258, 1536]
        or memory.get("active_tensor_shape") != [1, 262, 1536]
        or memory.get("source_voxels") != 74_699
        or memory.get("semantic_feature_dim") != 3_072
        or memory.get("question_dependent_scene_retrieval") is not False
        or audit.get("forbidden_access_count") != 0
        or audit.get("passed") is not True
    ):
        raise AssertionError("Startup scene-memory or isolation contract failed")
    signature = _static_signature(startup)
    start_xy = _xy(startup)

    started = time.perf_counter()
    result = _request(
        base_url,
        "/api/instruction",
        {"instruction": instruction},
    )
    elapsed = time.perf_counter() - started
    if _verify_control(result, attempted=True) != navigation_identity:
        raise AssertionError("Navigation checkpoint identity changed during the live turn")
    if _mapping(result.get("control"), "control").get("gemma_accepted") is not True:
        raise AssertionError("Gemma goal did not complete successfully")
    if _static_signature(result) != signature:
        raise AssertionError("Navigation changed the static map or scene prefix")
    scene_prefix_sha256 = signature[1]
    if not isinstance(scene_prefix_sha256, str):
        raise TypeError("Startup scene prefix hash is not text")
    decisions = _verify_decisions(
        result,
        expected_scene_prefix_sha256=scene_prefix_sha256,
        expected_navigation_checkpoint_sha256=navigation_identity[0],
    )
    actions_raw = result.get("actions")
    if not isinstance(actions_raw, list):
        raise TypeError("actions must be a list")
    actions = [dict(_mapping(value, "actions[]")) for value in actions_raw]
    final_xy = _xy(result)
    trajectory = _decision_trajectory_metrics(
        start_xy,
        decisions,
        final_xy=final_xy,
    )
    _verify_action_receipt_suffix(
        actions,
        trajectory["receipt_completion_positions"],
    )
    path_length = float(trajectory["path_length_m"])
    signed_winding_area = float(trajectory["signed_winding_area_m2"])
    abs_winding_area = float(trajectory["abs_winding_area_m2"])
    return_error = _distance(start_xy, final_xy)
    is_lap = "lap" in instruction.casefold()
    geometry_passed = not is_lap or (
        path_length >= MIN_LAP_PATH_LENGTH_M
        and return_error <= MAX_LAP_RETURN_ERROR_M
        and abs_winding_area >= MIN_LAP_ABS_WINDING_AREA_M2
    )
    if not geometry_passed:
        raise AssertionError("Gemma selected STOP, but the requested lap geometry failed")
    return {
        "schema": "semantic_3d_chat.gemma_waypoint_live_acceptance.v2",
        "passed": True,
        "instruction": instruction,
        "elapsed_seconds": elapsed,
        "model_decision_count": len(decisions),
        "executed_decision_count": sum(item.get("executed") is True for item in decisions),
        "rejected_decision_count": sum(item.get("accepted") is not True for item in decisions),
        "model_action_counts": dict(Counter(str(item["model_action"]) for item in decisions)),
        "numeric_action_receipt_count": len(actions),
        "action_receipt_alignment_passed": True,
        "trajectory_source": trajectory["trajectory_source"],
        "trajectory_point_count": trajectory["trajectory_point_count"],
        "path_length_m": path_length,
        "return_error_m": return_error,
        "signed_winding_area_m2": signed_winding_area,
        "abs_winding_area_m2": abs_winding_area,
        "lap_geometry_thresholds": {
            "minimum_path_length_m": MIN_LAP_PATH_LENGTH_M,
            "maximum_return_error_m": MAX_LAP_RETURN_ERROR_M,
            "minimum_abs_winding_area_m2": MIN_LAP_ABS_WINDING_AREA_M2,
        },
        "lap_geometry_passed": True,
        "model_selected_terminal_stop": True,
        "actual_gemma_causal_forward_every_step": True,
        "deterministic_route_planner_used": False,
        "fallback_used": False,
        "substitution_applied": False,
        "synthetic_stop_applied": False,
        "local_inference": True,
        "cloud_model_used": False,
        "navigation_checkpoint_sha256": navigation_identity[0],
        "gemma_runtime_binding_sha256": navigation_identity[1],
        "scene_prefix_sha256": signature[1],
        "scene_prefix_unchanged": True,
        "every_decision_scene_prefix_unchanged": True,
        "map_version": signature[2],
        "scan_count": signature[3],
        "runtime_oracle_access_count": 0,
        "final_position_xy_m": list(final_xy),
        "reply": str(result.get("reply", "")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8770")
    parser.add_argument("--instruction", default="Do a lap around the room.")
    parser.add_argument(
        "--expected-navigation-checkpoint-sha256",
        help="Fail unless the live backend exposes this exact candidate weights digest.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    startup: Mapping[str, Any] = {}
    try:
        startup = _request(args.base_url, "/api/state")
        report = run_acceptance(
            args.base_url,
            args.instruction,
            expected_navigation_checkpoint_sha256=(args.expected_navigation_checkpoint_sha256),
        )
        exit_code = 0
    except (AssertionError, RuntimeError, TypeError, ValueError, OSError) as error:
        try:
            result = _request(args.base_url, "/api/state")
        except (RuntimeError, TypeError, ValueError, OSError) as snapshot_error:
            result = {}
            snapshot_failure = f"{type(snapshot_error).__name__}: {snapshot_error}"
        else:
            snapshot_failure = None
        decisions_raw = result.get("model_decisions", [])
        raw_decisions = decisions_raw if isinstance(decisions_raw, list) else []
        actions_raw = result.get("actions", [])
        actions = actions_raw if isinstance(actions_raw, list) else []
        try:
            start_xy = _xy(startup)
            final_xy = _xy(result)
            return_error = _distance(start_xy, final_xy)
        except (TypeError, ValueError, KeyError, UnboundLocalError):
            start_xy = None
            final_xy = None
            return_error = None
        try:
            decisions = _authenticate_snapshot_decisions(startup, result)
        except (AssertionError, TypeError, ValueError) as authentication_error:
            decisions = []
            decision_authentication_failure = (
                f"{type(authentication_error).__name__}: {authentication_error}"
            )
        else:
            decision_authentication_failure = None
        try:
            if start_xy is None:
                raise ValueError("Initial pose is unavailable")
            if decision_authentication_failure is not None:
                raise ValueError("Model decisions are not authenticated")
            trajectory = _decision_trajectory_metrics(
                start_xy,
                decisions,
                final_xy=final_xy,
            )
        except (AssertionError, TypeError, ValueError) as trajectory_error:
            trajectory = {
                "trajectory_source": "complete_model_decisions",
                "trajectory_point_count": None,
                "path_length_m": None,
                "signed_winding_area_m2": None,
                "abs_winding_area_m2": None,
            }
            trajectory_metric_failure = f"{type(trajectory_error).__name__}: {trajectory_error}"
            action_receipt_alignment_passed = None
            action_receipt_alignment_failure = None
        else:
            trajectory_metric_failure = None
            try:
                _verify_action_receipt_suffix(
                    actions,
                    trajectory["receipt_completion_positions"],
                )
            except (AssertionError, TypeError, ValueError) as receipt_error:
                action_receipt_alignment_passed = False
                action_receipt_alignment_failure = (
                    f"{type(receipt_error).__name__}: {receipt_error}"
                )
            else:
                action_receipt_alignment_passed = True
                action_receipt_alignment_failure = None
        control = result.get("control")
        if decision_authentication_failure is None:
            fallback_used = (
                control.get("fallback_used") if isinstance(control, Mapping) else None
            )
            planner_used = any(
                item.get("deterministic_route_planner_used") is True
                for item in decisions
            )
            substitution_applied = any(
                item.get("substitution_applied") is True for item in decisions
            )
            synthetic_stop_applied = any(
                item.get("synthetic_stop_applied") is True for item in decisions
            )
        else:
            fallback_used = None
            planner_used = None
            substitution_applied = None
            synthetic_stop_applied = None
        report = {
            "schema": "semantic_3d_chat.gemma_waypoint_live_acceptance.v2",
            "passed": False,
            "instruction": args.instruction,
            "failure_type": type(error).__name__,
            "failure_reason": str(error),
            "snapshot_failure": snapshot_failure,
            "model_decision_count": len(raw_decisions),
            "executed_decision_count": sum(
                isinstance(item, Mapping) and item.get("executed") is True
                for item in raw_decisions
            ),
            "rejected_decision_count": sum(
                isinstance(item, Mapping) and item.get("accepted") is not True
                for item in raw_decisions
            ),
            "model_action_counts": dict(
                Counter(
                    str(item.get("model_action"))
                    for item in raw_decisions
                    if isinstance(item, Mapping)
                )
            ),
            "model_decisions_authenticated": decision_authentication_failure is None,
            "decision_authentication_failure": decision_authentication_failure,
            "numeric_action_receipt_count": len(actions),
            "action_receipt_alignment_passed": action_receipt_alignment_passed,
            "action_receipt_alignment_failure": action_receipt_alignment_failure,
            "trajectory_source": trajectory["trajectory_source"],
            "trajectory_point_count": trajectory["trajectory_point_count"],
            "path_length_m": trajectory["path_length_m"],
            "return_error_m": return_error,
            "signed_winding_area_m2": trajectory["signed_winding_area_m2"],
            "abs_winding_area_m2": trajectory["abs_winding_area_m2"],
            "trajectory_metric_failure": trajectory_metric_failure,
            "lap_geometry_thresholds": {
                "minimum_path_length_m": MIN_LAP_PATH_LENGTH_M,
                "maximum_return_error_m": MAX_LAP_RETURN_ERROR_M,
                "minimum_abs_winding_area_m2": MIN_LAP_ABS_WINDING_AREA_M2,
            },
            "lap_geometry_passed": False,
            "initial_position_xy_m": None if start_xy is None else list(start_xy),
            "final_position_xy_m": None if final_xy is None else list(final_xy),
            "deterministic_route_planner_used": planner_used,
            "fallback_used": fallback_used,
            "substitution_applied": substitution_applied,
            "synthetic_stop_applied": synthetic_stop_applied,
            "local_inference": True,
            "cloud_model_used": False,
            "expected_navigation_checkpoint_sha256": (args.expected_navigation_checkpoint_sha256),
            "navigation_checkpoint_sha256": (
                control.get("navigation_checkpoint_sha256")
                if isinstance(control, Mapping)
                else None
            ),
            "gemma_runtime_binding_sha256": (
                control.get("gemma_runtime_binding_sha256")
                if isinstance(control, Mapping)
                else None
            ),
            "runtime_snapshot": result,
        }
        exit_code = 1
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if exit_code:
        print(f"live acceptance failed; evidence written to {output}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
