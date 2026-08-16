"""Camera-free continuous-semantic fallback for face and approach goals.

This fallback is intentionally narrower than a learned goal policy.  It uses a
local continuous text embedding only to ground the user's phrase against every
voxel of the already-bound semantic map.  Bounded turning and path planning
then use numeric geometry and robot state.  No current camera frame, scan,
label inventory, caption, oracle record, or simulator object name is accepted.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np

from semantic_3d_chat.robot.collision import NumericCollisionMap
from semantic_3d_chat.robot.llm_tool_policy import (
    execute_validated_tool_call,
    validate_tool_call_text,
)
from semantic_3d_chat.robot.planner import NumericPathPlan, NumericWaypointPlanner
from semantic_3d_chat.robot.semantic_agent import (
    ContinuousSemanticTargetGrounder,
    ContinuousTextEncoder,
)
from semantic_3d_chat.robot.semantic_between import (
    _binding,
    _bound_map_path,
    _static_signature,
)
from semantic_3d_chat.robot.semantic_mapping import semantic_map_content_hash

GoalKind = Literal["face", "approach"]


def _validate_settings(
    runtime: Any,
    config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], list[float] | tuple[float, ...]]:
    robot = config.get("robot")
    scene = config.get("scene")
    room = scene.get("room_size_m") if isinstance(scene, Mapping) else None
    if not isinstance(robot, Mapping) or not isinstance(room, (list, tuple)):
        raise TypeError("Semantic goal fallback lacks robot or room settings")
    if robot.get("auto_scan_after_motion") is not False:
        raise ValueError("Semantic goal fallback requires auto_scan_after_motion=false")
    simulator = getattr(runtime, "simulator", None)
    runtime_settings = getattr(simulator, "settings", None)
    if (
        isinstance(runtime_settings, Mapping)
        and runtime_settings.get("auto_scan_after_motion") is not False
    ):
        raise ValueError("Semantic goal runtime is not in static-map motion mode")
    if getattr(runtime, "auto_scan_after_motion", False) is not False:
        raise ValueError("Semantic goal runtime enables camera scans after motion")
    return robot, room


def _state(runtime: Any) -> dict[str, Any]:
    value = runtime.get_robot_state()
    if not isinstance(value, Mapping):
        raise TypeError("Semantic goal runtime returned an invalid robot state")
    result = dict(value)
    position = np.asarray(result.get("position_m"), dtype=np.float64)
    yaw = result.get("body_yaw_degrees")
    scan_count = result.get("scan_count")
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("Semantic goal position_m must contain three finite values")
    if isinstance(yaw, bool) or not isinstance(yaw, (int, float)) or not math.isfinite(yaw):
        raise ValueError("Semantic goal body_yaw_degrees must be finite")
    if isinstance(scan_count, bool) or not isinstance(scan_count, int) or scan_count < 0:
        raise ValueError("Semantic goal scan_count must be a nonnegative integer")
    return result


def _turn_arguments(
    state: Mapping[str, Any],
    target_xy: np.ndarray,
    *,
    max_turn_degrees: float,
    deadband_degrees: float,
) -> list[dict[str, float]]:
    position = np.asarray(state["position_m"][:2], dtype=np.float64)
    delta = target_xy - position
    if float(np.linalg.norm(delta)) <= 1e-8:
        return []
    desired = math.degrees(math.atan2(-float(delta[0]), float(delta[1])))
    current = float(state["body_yaw_degrees"])
    remaining = (desired - current + 180.0) % 360.0 - 180.0
    if abs(remaining) <= deadband_degrees:
        return []
    arguments: list[dict[str, float]] = []
    while abs(remaining) > deadband_degrees:
        step = max(-max_turn_degrees, min(max_turn_degrees, remaining))
        arguments.append({"angle_degrees": step})
        remaining -= step
    return arguments


def _execute_action(
    runtime: Any,
    config: Mapping[str, Any],
    *,
    name: Literal["move_to", "turn"],
    arguments: Mapping[str, float],
    initial_signature: tuple[Any, ...],
    scan_count: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    before = _binding(runtime)
    before_state = _state(runtime)
    if (
        _static_signature(before) != initial_signature
        or before_state["scan_count"] != scan_count
    ):
        return None, None, "E_STATIC_SCENE_CHANGED"
    proposal = json.dumps(
        {"tool": name, "arguments": dict(arguments)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    validation = validate_tool_call_text(proposal, config, robot_state=before_state)
    if validation.call is None:
        return None, None, validation.error_code or "E_SCHEMA"
    receipt = execute_validated_tool_call(runtime, validation.call, config=config)
    after = _binding(runtime)
    after_state = _state(runtime)
    step = {
        "tool": name,
        "arguments": dict(arguments),
        "numeric_tool_receipt": receipt,
        "scene_prefix_sha256": after["scene_prefix_sha256"],
        "active_prefix_sha256": after["active_prefix_sha256"],
    }
    if receipt.get("success") is not True:
        return receipt, step, str(receipt.get("error_code") or "E_ACTION")
    if (
        _static_signature(after) != initial_signature
        or after_state["scan_count"] != scan_count
    ):
        return receipt, step, "E_STATIC_SCENE_CHANGED"
    if (
        after["robot_tokens_sha256"] == before["robot_tokens_sha256"]
        or after["active_prefix_sha256"] == before["active_prefix_sha256"]
    ):
        return receipt, step, "E_ROBOT_PREFIX_STALE"
    return receipt, step, None


def execute_grounded_goal_fallback(
    runtime: Any,
    config: Mapping[str, Any],
    *,
    kind: GoalKind,
    target_text: str,
    text_encoder: ContinuousTextEncoder,
) -> dict[str, Any]:
    """Execute a label-free face or approach goal from the fixed semantic map."""

    if kind not in {"face", "approach"}:
        raise ValueError("Semantic goal fallback supports only face or approach")
    if not isinstance(target_text, str) or not target_text.strip():
        raise ValueError("Semantic goal fallback requires a non-empty user target phrase")
    normalized_target = " ".join(target_text.strip().split())
    if not isinstance(config, Mapping):
        raise TypeError("Semantic goal fallback configuration must be a mapping")
    robot, room_size = _validate_settings(runtime, config)

    initial_binding = _binding(runtime)
    initial_signature = _static_signature(initial_binding)
    active_map_path = _bound_map_path(runtime, initial_binding)
    initial_map_hash = semantic_map_content_hash(active_map_path)
    grounder = ContinuousSemanticTargetGrounder(
        active_map_path,
        text_encoder,
        room_size_m=room_size,
    )
    grounding = grounder.ground(normalized_target)
    if (
        grounder.scene_id != initial_binding["scene_id"]
        or grounding.map_sha256 != initial_binding["map_sha256"]
        or grounding.map_sha256 != initial_map_hash
        or grounding.scored_voxels != initial_binding["source_voxels"]
        or grounding.scored_voxels != len(grounder.xyz)
    ):
        raise RuntimeError("Semantic goal did not ground against the exact bound map")
    if _static_signature(_binding(runtime)) != initial_signature:
        raise RuntimeError("Static scene changed during all-voxel semantic grounding")

    initial_state = _state(runtime)
    scan_count = int(initial_state["scan_count"])
    target_xy = np.asarray(grounding.target_xyz_m[:2], dtype=np.float64)
    max_turn = float(robot.get("max_turn_degrees", 45.0))
    deadband = float(robot.get("face_alignment_deadband_degrees", 3.0))
    max_step = min(0.50, float(robot.get("max_move_to_m", 1.0)))
    max_actions = int(robot.get("grounded_fallback_max_actions", 64))
    if (
        not math.isfinite(max_turn)
        or max_turn <= 0.0
        or not math.isfinite(deadband)
        or not 0.0 <= deadband < max_turn
        or not math.isfinite(max_step)
        or max_step <= 0.0
        or isinstance(max_actions, bool)
        or not 1 <= max_actions <= 256
    ):
        raise ValueError("Semantic goal fallback limits are invalid")

    plan: NumericPathPlan | None = None
    action_specs: list[tuple[Literal["move_to", "turn"], dict[str, float]]] = []
    collision_map: NumericCollisionMap | None = None
    if kind == "approach":
        collision_map = NumericCollisionMap.from_voxel_map(
            active_map_path,
            room_size_m=room_size,
            robot_radius_m=float(robot.get("radius_m", 0.25)),
            collision_z_min_m=float(robot.get("collision_z_min_m", 0.12)),
            collision_z_max_m=float(robot.get("collision_z_max_m", 1.80)),
            surface_padding_m=float(robot.get("surface_padding_m", 0.035)),
        )
        planner = NumericWaypointPlanner(
            collision_map,
            grid_resolution_m=float(robot.get("grounded_fallback_grid_resolution_m", 0.15)),
            standoff_m=float(robot.get("approach_target_standoff_m", 0.60)),
            standoff_tolerance_m=float(
                robot.get("grounded_fallback_standoff_tolerance_m", 0.20)
            ),
            max_waypoint_step_m=max_step,
        )
        plan = planner.plan(initial_state["position_m"][:2], target_xy)
        action_specs.extend(
            ("move_to", {"x": x, "y": y}) for x, y in plan.waypoints_xy_m
        )
        predicted_face_state = dict(initial_state)
        predicted_face_state["position_m"] = [*plan.goal_xy_m, 0.0]
        action_specs.extend(
            ("turn", arguments)
            for arguments in _turn_arguments(
                predicted_face_state,
                target_xy,
                max_turn_degrees=max_turn,
                deadband_degrees=deadband,
            )
        )
    else:
        action_specs.extend(
            ("turn", arguments)
            for arguments in _turn_arguments(
                initial_state,
                target_xy,
                max_turn_degrees=max_turn,
                deadband_degrees=deadband,
            )
        )
    if len(action_specs) > max_actions:
        raise RuntimeError("Semantic goal fallback exceeds its bounded action budget")

    receipts: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    error_code: str | None = None
    for index, (name, arguments) in enumerate(action_specs, start=1):
        receipt, step, error_code = _execute_action(
            runtime,
            config,
            name=name,
            arguments=arguments,
            initial_signature=initial_signature,
            scan_count=scan_count,
        )
        if receipt is not None:
            receipts.append(receipt)
        if step is not None:
            steps.append({"step": index, **step})
        if error_code is not None:
            break

    final_binding = _binding(runtime)
    final_state = _state(runtime)
    final_map_hash = semantic_map_content_hash(active_map_path)
    if error_code is None and (
        _static_signature(final_binding) != initial_signature
        or final_state["scan_count"] != scan_count
        or final_map_hash != initial_map_hash
    ):
        error_code = "E_STATIC_SCENE_CHANGED"
    final_position = np.asarray(final_state["position_m"], dtype=np.float64)
    success = error_code is None and len(receipts) == len(action_specs)
    return {
        "schema": "semantic_3d_chat.grounded_goal_fallback.v1",
        "kind": "navigation",
        "command": f"semantic_grounded_{kind}",
        "success": success,
        "error_code": error_code,
        "grounding": grounding.as_dict(),
        "grounding_scope": "every_active_map_voxel",
        "all_target_groundings_scored_complete_bound_map": (
            grounding.scored_voxels == initial_binding["source_voxels"]
            and grounding.map_sha256 == initial_binding["map_sha256"]
        ),
        "plan": None if plan is None else plan.as_dict(),
        "planned_action_count": len(action_specs),
        "completed_action_count": sum(item.get("success") is True for item in receipts),
        "action_receipts": receipts,
        "steps": steps,
        "initial_prefix_binding": initial_binding,
        "final_prefix_binding": final_binding,
        "initial_scan_count": scan_count,
        "final_scan_count": final_state["scan_count"],
        "final_position_m": final_position.tolist(),
        "final_target_distance_m": float(np.linalg.norm(final_position[:2] - target_xy)),
        "static_scene_prefix_unchanged": (
            _static_signature(final_binding) == initial_signature
            and final_map_hash == initial_map_hash
        ),
        "robot_tokens_refreshed_after_every_action": (
            error_code != "E_ROBOT_PREFIX_STALE"
            and len(receipts) == len(steps)
        ),
        "camera_observations_during_goal": 0,
        "scene_prefix_computed_before_goal": True,
        "geometry_source": "complete_precomputed_numeric_voxel_map",
        "fallback_controller": "continuous_all_voxel_grounding_plus_numeric_geometry",
        "user_motor_commands_exposed": False,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
    }


__all__ = ["GoalKind", "execute_grounded_goal_fallback"]
