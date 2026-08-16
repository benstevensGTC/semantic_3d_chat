"""Camera-free navigation to a point between two continuous semantic targets.

The caller supplies only two phrases from the user's goal.  Each phrase is
embedded and scored against every voxel in the already-bound numeric semantic
map.  The resulting two coordinates are reduced to a numeric midpoint; from
there onward planning and action validation are geometry-only.  This module
does not accept labels, inventories, captions, oracle records, segmentation,
simulator object names, or camera observations.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from semantic_3d_chat.robot.collision import NumericCollisionMap
from semantic_3d_chat.robot.llm_tool_policy import (
    execute_validated_tool_call,
    validate_tool_call_text,
)
from semantic_3d_chat.robot.planner import NumericPathPlan, NumericWaypointPlanner
from semantic_3d_chat.robot.semantic_agent import (
    ContinuousSemanticGrounding,
    ContinuousSemanticTargetGrounder,
    ContinuousTextEncoder,
)
from semantic_3d_chat.robot.semantic_mapping import semantic_map_content_hash

_STATIC_BINDING_FIELDS = (
    "scene_id",
    "map_version",
    "map_sha256",
    "scene_prefix_sha256",
    "source_voxels",
    "processed_voxels",
)


def _binding(runtime: Any) -> dict[str, Any]:
    value = runtime.prefix_binding()
    if not isinstance(value, Mapping):
        raise TypeError("Between-goal runtime returned an invalid prefix binding")
    result = dict(value)
    for field in (
        "scene_id",
        "map_sha256",
        "scene_prefix_sha256",
        "active_prefix_sha256",
        "robot_tokens_sha256",
    ):
        if not isinstance(result.get(field), str) or not result[field]:
            raise ValueError(f"Between-goal prefix binding lacks {field}")
    for field in ("map_version", "source_voxels", "processed_voxels"):
        item = result.get(field)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"Between-goal prefix binding has invalid {field}")
    if result["source_voxels"] < 1:
        raise ValueError("Between-goal prefix binding has no source voxels")
    return result


def _static_signature(binding: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(binding.get(field) for field in _STATIC_BINDING_FIELDS)


def _bound_map_path(runtime: Any, binding: Mapping[str, Any]) -> Path:
    """Resolve the updater-owned map whose numeric hash is bound to the prefix."""

    updater = getattr(runtime, "map_updater", None)
    if updater is None:
        raise TypeError("Between-goal runtime has no semantic map updater")
    version = binding["map_version"]
    current_version = getattr(updater, "current_version", version)
    if current_version != version:
        raise RuntimeError("Semantic map updater version differs from the bound prefix")
    attribute = "base_map_path" if version == 0 else "persistent_map_path"
    value = getattr(updater, attribute, None)
    if not isinstance(value, (str, Path)):
        raise TypeError("Semantic map updater does not expose its active numeric map")
    candidate = Path(value).expanduser()
    if not candidate.is_file():
        raise FileNotFoundError("The bound active numeric map is unavailable")
    expected = str(binding["map_sha256"])
    if semantic_map_content_hash(candidate) != expected:
        raise RuntimeError("The updater-owned numeric map does not match the bound prefix")
    return candidate


def _robot_settings(config: Mapping[str, Any], runtime: Any) -> Mapping[str, Any]:
    robot = config.get("robot")
    scene = config.get("scene")
    room = scene.get("room_size_m") if isinstance(scene, Mapping) else None
    if not isinstance(robot, Mapping) or not isinstance(room, (list, tuple)):
        raise TypeError("Between-goal configuration lacks robot or room settings")
    if robot.get("auto_scan_after_motion") is not False:
        raise ValueError("Between-goal execution requires auto_scan_after_motion=false")
    simulator = getattr(runtime, "simulator", None)
    runtime_settings = getattr(simulator, "settings", None)
    if (
        isinstance(runtime_settings, Mapping)
        and runtime_settings.get("auto_scan_after_motion") is not False
    ):
        raise ValueError("Between-goal runtime is not in static-map motion mode")
    auto_scan = getattr(runtime, "auto_scan_after_motion", False)
    if auto_scan is not False:
        raise ValueError("Between-goal runtime enables camera scans after motion")
    return robot


def _ground_complete_map(
    grounder: ContinuousSemanticTargetGrounder,
    target_text: str,
    *,
    binding: Mapping[str, Any],
) -> ContinuousSemanticGrounding:
    result = grounder.ground(target_text)
    if (
        result.map_sha256 != binding["map_sha256"]
        or result.scored_voxels != binding["source_voxels"]
        or result.scored_voxels != len(grounder.xyz)
    ):
        raise RuntimeError("Semantic target grounding did not score the exact bound map")
    return result


def _zero_length_plan(point: np.ndarray) -> NumericPathPlan:
    xy = (float(point[0]), float(point[1]))
    return NumericPathPlan(
        target_xy_m=xy,
        goal_xy_m=xy,
        waypoints_xy_m=(),
        path_length_m=0.0,
        target_distance_m=0.0,
        direct_path=True,
        expanded_nodes=0,
    )


def _closest_reachable_midpoint(
    midpoint: np.ndarray,
    start: np.ndarray,
    collision_map: NumericCollisionMap,
    planner: NumericWaypointPlanner,
    *,
    resolution_m: float,
    angular_samples: int,
    max_actions: int,
) -> tuple[np.ndarray, NumericPathPlan]:
    """Find the closest reachable free shell around the requested midpoint.

    The exact midpoint is tried first.  If it is occupied, complete concentric
    shells are tested in increasing ``resolution_m`` increments.  Therefore
    the selected point is no farther than one search increment beyond the
    closest reachable point represented by the deterministic angular grid.
    """

    room_diagonal = float(
        np.linalg.norm(collision_map.room_max_xy_m - collision_map.room_min_xy_m)
    )
    shell_count = max(1, math.ceil(room_diagonal / resolution_m))
    for shell in range(shell_count + 1):
        radius = shell * resolution_m
        if shell == 0:
            candidates = [midpoint]
        else:
            candidates = [
                midpoint
                + radius
                * np.asarray(
                    [math.cos(2.0 * math.pi * index / angular_samples),
                     math.sin(2.0 * math.pi * index / angular_samples)],
                    dtype=np.float64,
                )
                for index in range(angular_samples)
            ]
        reachable: list[tuple[NumericPathPlan, np.ndarray]] = []
        for candidate in candidates:
            if collision_map.point_check(candidate).collision:
                continue
            try:
                plan = (
                    _zero_length_plan(candidate)
                    if float(np.linalg.norm(candidate - start)) <= 1e-8
                    else planner.plan_to_free_point(start, candidate)
                )
            except RuntimeError:
                continue
            if len(plan.waypoints_xy_m) <= max_actions:
                reachable.append((plan, candidate))
        if reachable:
            plan, candidate = min(
                reachable,
                key=lambda value: (
                    value[0].path_length_m,
                    float(value[1][0]),
                    float(value[1][1]),
                ),
            )
            return candidate, plan
    raise RuntimeError("No reachable collision-free point exists near the semantic midpoint")


def execute_semantic_between_goal(
    runtime: Any,
    config: Mapping[str, Any],
    *,
    first_target_text: str,
    second_target_text: str,
    text_encoder: ContinuousTextEncoder,
) -> dict[str, Any]:
    """Ground two user phrases globally and move to their numeric midpoint.

    Only validated ``move_to`` calls can leave this function.  The precomputed
    scene map and scene prefix must remain byte-bound and scan count must remain
    unchanged; successful motion must refresh only the continuous robot-state
    tokens and active-prefix hash.
    """

    if not isinstance(config, Mapping):
        raise TypeError("Between-goal configuration must be a mapping")
    normalized_targets = []
    for value in (first_target_text, second_target_text):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Between-goal target phrases must be non-empty user text")
        normalized_targets.append(" ".join(value.strip().split()))
    if normalized_targets[0].casefold() == normalized_targets[1].casefold():
        raise ValueError("Between-goal target phrases must identify two distinct targets")

    robot = _robot_settings(config, runtime)
    scene = config.get("scene")
    assert isinstance(scene, Mapping)
    room_size = scene.get("room_size_m")
    if not isinstance(room_size, (list, tuple)):
        raise TypeError("Between-goal room_size_m is unavailable")

    initial_binding = _binding(runtime)
    initial_signature = _static_signature(initial_binding)
    active_map_path = _bound_map_path(runtime, initial_binding)
    active_map_hash = semantic_map_content_hash(active_map_path)
    grounder = ContinuousSemanticTargetGrounder(
        active_map_path,
        text_encoder,
        room_size_m=room_size,
    )
    if (
        grounder.scene_id != initial_binding["scene_id"]
        or grounder.map_sha256 != active_map_hash
    ):
        raise RuntimeError("Semantic grounder is not bound to the active opaque scene")
    first = _ground_complete_map(
        grounder,
        normalized_targets[0],
        binding=initial_binding,
    )
    second = _ground_complete_map(
        grounder,
        normalized_targets[1],
        binding=initial_binding,
    )

    post_grounding_binding = _binding(runtime)
    if _static_signature(post_grounding_binding) != initial_signature:
        raise RuntimeError("Static scene changed during all-voxel semantic grounding")
    initial_state = runtime.get_robot_state()
    if not isinstance(initial_state, Mapping):
        raise TypeError("Between-goal runtime returned an invalid robot state")
    position = np.asarray(initial_state.get("position_m"), dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("Between-goal position_m must contain three finite values")
    scan_count = initial_state.get("scan_count")
    if isinstance(scan_count, bool) or not isinstance(scan_count, int) or scan_count < 0:
        raise ValueError("Between-goal runtime returned an invalid scan_count")

    collision_map = NumericCollisionMap.from_voxel_map(
        active_map_path,
        room_size_m=room_size,
        robot_radius_m=float(robot.get("radius_m", 0.25)),
        collision_z_min_m=float(robot.get("collision_z_min_m", 0.12)),
        collision_z_max_m=float(robot.get("collision_z_max_m", 1.80)),
        surface_padding_m=float(robot.get("surface_padding_m", 0.035)),
    )
    max_step = min(0.50, float(robot.get("max_move_to_m", 1.0)))
    if not math.isfinite(max_step) or max_step <= 0.0:
        raise ValueError("robot.max_move_to_m must be finite and positive")
    resolution = float(robot.get("between_goal_search_resolution_m", 0.05))
    angular_samples = int(robot.get("between_goal_angular_samples", 48))
    max_actions = int(robot.get("between_goal_max_actions", 64))
    if (
        not math.isfinite(resolution)
        or resolution <= 0.0
        or isinstance(angular_samples, bool)
        or angular_samples < 16
        or isinstance(max_actions, bool)
        or not 1 <= max_actions <= 256
    ):
        raise ValueError("Between-goal search settings are invalid")
    planner = NumericWaypointPlanner(
        collision_map,
        grid_resolution_m=max(0.05, resolution),
        max_waypoint_step_m=max_step,
    )
    first_xyz = np.asarray(first.target_xyz_m, dtype=np.float64)
    second_xyz = np.asarray(second.target_xyz_m, dtype=np.float64)
    midpoint_xyz = (first_xyz + second_xyz) / 2.0
    selected_goal, plan = _closest_reachable_midpoint(
        midpoint_xyz[:2],
        position[:2],
        collision_map,
        planner,
        resolution_m=resolution,
        angular_samples=angular_samples,
        max_actions=max_actions,
    )

    receipts: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    error_code: str | None = None
    for index, (x, y) in enumerate(plan.waypoints_xy_m, start=1):
        before = _binding(runtime)
        before_state = runtime.get_robot_state()
        if (
            _static_signature(before) != initial_signature
            or not isinstance(before_state, Mapping)
            or before_state.get("scan_count") != scan_count
        ):
            error_code = "E_STATIC_SCENE_CHANGED"
            break
        proposal = json.dumps(
            {"tool": "move_to", "arguments": {"x": x, "y": y}},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        validation = validate_tool_call_text(proposal, config, robot_state=before_state)
        if validation.call is None:
            error_code = validation.error_code or "E_SCHEMA"
            break
        receipt = execute_validated_tool_call(runtime, validation.call, config=config)
        receipts.append(receipt)
        after = _binding(runtime)
        after_state = runtime.get_robot_state()
        steps.append(
            {
                "step": index,
                "waypoint_xy_m": [x, y],
                "numeric_tool_receipt": receipt,
                "scene_prefix_sha256": after["scene_prefix_sha256"],
                "active_prefix_sha256": after["active_prefix_sha256"],
            }
        )
        if receipt.get("success") is not True:
            error_code = str(receipt.get("error_code") or "E_ACTION")
            break
        if (
            _static_signature(after) != initial_signature
            or not isinstance(after_state, Mapping)
            or after_state.get("scan_count") != scan_count
        ):
            error_code = "E_STATIC_SCENE_CHANGED"
            break
        if (
            after["robot_tokens_sha256"] == before["robot_tokens_sha256"]
            or after["active_prefix_sha256"] == before["active_prefix_sha256"]
        ):
            error_code = "E_ROBOT_PREFIX_STALE"
            break

    final_binding = _binding(runtime)
    final_state = runtime.get_robot_state()
    final_map_hash = semantic_map_content_hash(active_map_path)
    if error_code is None and (
        _static_signature(final_binding) != initial_signature
        or final_map_hash != active_map_hash
        or not isinstance(final_state, Mapping)
        or final_state.get("scan_count") != scan_count
    ):
        error_code = "E_STATIC_SCENE_CHANGED"
    final_position = (
        np.asarray(final_state.get("position_m"), dtype=np.float64)
        if isinstance(final_state, Mapping)
        else np.full(3, np.nan)
    )
    success = error_code is None and len(receipts) == len(plan.waypoints_xy_m)
    return {
        "schema": "semantic_3d_chat.semantic_between_execution.v1",
        "kind": "navigation",
        "command": "semantic_between",
        "success": success,
        "error_code": error_code,
        "groundings": [first.as_dict(), second.as_dict()],
        "midpoint_xyz_m": midpoint_xyz.tolist(),
        "selected_goal_xy_m": selected_goal.tolist(),
        "selected_goal_offset_from_midpoint_m": float(
            np.linalg.norm(selected_goal - midpoint_xyz[:2])
        ),
        "midpoint_search_resolution_m": resolution,
        "midpoint_optimality_tolerance_m": resolution,
        "plan": plan.as_dict(),
        "planned_action_count": len(plan.waypoints_xy_m),
        "completed_action_count": sum(item.get("success") is True for item in receipts),
        "action_receipts": receipts,
        "steps": steps,
        "initial_prefix_binding": initial_binding,
        "final_prefix_binding": final_binding,
        "initial_scan_count": scan_count,
        "final_scan_count": (
            final_state.get("scan_count") if isinstance(final_state, Mapping) else None
        ),
        "final_position_m": final_position.tolist(),
        "final_goal_distance_m": (
            float(np.linalg.norm(final_position[:2] - selected_goal))
            if np.isfinite(final_position).all()
            else None
        ),
        "all_target_groundings_scored_complete_bound_map": all(
            grounding.scored_voxels == initial_binding["source_voxels"]
            and grounding.map_sha256 == initial_binding["map_sha256"]
            for grounding in (first, second)
        ),
        "static_scene_prefix_unchanged": (
            _static_signature(final_binding) == initial_signature
            and final_map_hash == active_map_hash
        ),
        "robot_tokens_refreshed_after_every_motion": all(
            step["numeric_tool_receipt"].get("success") is True for step in steps
        )
        and error_code != "E_ROBOT_PREFIX_STALE",
        "camera_observations_during_goal": 0,
        "scene_prefix_computed_before_goal": True,
        "geometry_source": "complete_precomputed_numeric_voxel_map",
        "user_motor_commands_exposed": False,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
    }


__all__ = ["execute_semantic_between_goal"]
