"""Geometry-only patrol planning over an anonymous semantic voxel map.

The language layer may select a high-level patrol/lap objective, but it never
chooses motor increments.  This module derives a closed route from the numeric
collision field, checks every continuous segment, and releases only bounded
``move_to`` waypoints.  It accepts no labels, object IDs, captions, scene
graphs, or simulator metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from semantic_3d_chat.robot.collision import NumericCollisionMap
from semantic_3d_chat.robot.llm_tool_policy import (
    execute_validated_tool_call,
    validate_tool_call_text,
)
from semantic_3d_chat.robot.planner import NumericPathPlan, NumericWaypointPlanner


@dataclass(frozen=True)
class NumericPatrolPlan:
    """A closed, collision-checked route through anonymous free space."""

    start_xy_m: tuple[float, float]
    anchors_xy_m: tuple[tuple[float, float], ...]
    waypoints_xy_m: tuple[tuple[float, float], ...]
    path_length_m: float
    expanded_nodes: int
    winding_area_m2: float
    plan_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_xy_m": list(self.start_xy_m),
            "anchors_xy_m": [list(value) for value in self.anchors_xy_m],
            "waypoints_xy_m": [list(value) for value in self.waypoints_xy_m],
            "path_length_m": self.path_length_m,
            "expanded_nodes": self.expanded_nodes,
            "winding_area_m2": self.winding_area_m2,
            "plan_sha256": self.plan_sha256,
            "closed_route": True,
            "numeric_map_only": True,
            "environmental_text_inputs": [],
            "oracle_inputs_at_runtime": False,
        }


def _finite_xy(value: tuple[float, float] | np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain two finite values")
    return result


def _polygon_area(points: list[np.ndarray]) -> float:
    if len(points) < 3:
        return 0.0
    x = np.asarray([point[0] for point in points], dtype=np.float64)
    y = np.asarray([point[1] for point in points], dtype=np.float64)
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


class NumericPatrolPlanner:
    """Create a visible room-scale lap without semantic or oracle inputs."""

    def __init__(
        self,
        collision_map: NumericCollisionMap,
        *,
        anchor_count: int = 8,
        wall_margin_m: float = 0.45,
        grid_resolution_m: float = 0.15,
        max_waypoint_step_m: float = 0.50,
        max_waypoints: int = 64,
    ) -> None:
        if isinstance(anchor_count, bool) or not 4 <= anchor_count <= 16:
            raise ValueError("anchor_count must be in [4, 16]")
        if isinstance(max_waypoints, bool) or not 8 <= max_waypoints <= 256:
            raise ValueError("max_waypoints must be in [8, 256]")
        values = (wall_margin_m, grid_resolution_m, max_waypoint_step_m)
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("Patrol planner distances must be finite and positive")
        self.collision_map = collision_map
        self.anchor_count = anchor_count
        self.wall_margin_m = float(wall_margin_m)
        self.max_waypoints = max_waypoints
        self.path_planner = NumericWaypointPlanner(
            collision_map,
            grid_resolution_m=grid_resolution_m,
            standoff_m=max(0.20, collision_map.inflated_radius_m),
            standoff_tolerance_m=0.05,
            max_waypoint_step_m=max_waypoint_step_m,
        )

    def _anchor_for_angle(
        self,
        angle: float,
        center: np.ndarray,
        radii: np.ndarray,
        used: list[np.ndarray],
    ) -> np.ndarray:
        angular_step = 2.0 * math.pi / self.anchor_count
        offsets = (0.0, 0.10, -0.10, 0.20, -0.20, 0.32, -0.32)
        scales = (1.0, 0.92, 0.84, 0.76, 0.68, 0.58, 0.48, 0.38)
        candidates: list[tuple[float, np.ndarray]] = []
        for scale in scales:
            for offset in offsets:
                candidate_angle = angle + offset * angular_step
                point = center + scale * radii * np.asarray(
                    [math.cos(candidate_angle), math.sin(candidate_angle)],
                    dtype=np.float64,
                )
                check = self.collision_map.point_check(point)
                if check.collision:
                    continue
                if any(float(np.linalg.norm(point - prior)) < 0.35 for prior in used):
                    continue
                # Prefer the outer room-scale ring, then locally generous
                # clearance, then the requested angular sector.
                score = 4.0 * scale + min(check.clearance_m, 0.75) - abs(offset)
                candidates.append((score, point))
        if not candidates:
            raise RuntimeError("No anonymous free-space patrol anchor is available")
        candidates.sort(
            key=lambda value: (
                -value[0],
                float(value[1][0]),
                float(value[1][1]),
            )
        )
        return candidates[0][1]

    def _anchors(self, start: np.ndarray) -> list[np.ndarray]:
        lower = self.collision_map.room_min_xy_m
        upper = self.collision_map.room_max_xy_m
        center = (lower + upper) / 2.0
        half = (upper - lower) / 2.0
        radii = half - self.collision_map.robot_radius_m - self.wall_margin_m
        if np.any(radii <= self.collision_map.inflated_radius_m):
            raise RuntimeError("Room is too small for a room-scale patrol")
        anchors: list[np.ndarray] = []
        for index in range(self.anchor_count):
            angle = 2.0 * math.pi * index / self.anchor_count
            anchors.append(self._anchor_for_angle(angle, center, radii, anchors))

        # Enter the ring at its nearest anchor so a lap does not begin with a
        # needless traversal across the whole room.  Preserve cyclic order.
        first = min(
            range(len(anchors)),
            key=lambda index: (
                float(np.linalg.norm(anchors[index] - start)),
                index,
            ),
        )
        return anchors[first:] + anchors[:first]

    def plan(
        self,
        start_xy_m: tuple[float, float] | np.ndarray,
    ) -> NumericPatrolPlan:
        start = _finite_xy(start_xy_m, name="start_xy_m")
        if self.collision_map.point_check(start).collision:
            raise ValueError("Patrol start position is in collision")
        anchors = self._anchors(start)
        if _polygon_area(anchors) <= 0.5:
            raise RuntimeError("Patrol anchors do not span a room-scale loop")

        legs: list[NumericPathPlan] = []
        cursor = start
        for goal in [*anchors, start]:
            leg = self.path_planner.plan_to_free_point(cursor, goal)
            legs.append(leg)
            cursor = np.asarray(leg.goal_xy_m, dtype=np.float64)

        waypoints: list[tuple[float, float]] = []
        for leg in legs:
            for point in leg.waypoints_xy_m:
                if waypoints and math.dist(waypoints[-1], point) <= 1e-8:
                    continue
                waypoints.append(point)
        if len(waypoints) > self.max_waypoints:
            raise RuntimeError("Patrol route exceeds the bounded action budget")

        route = [start, *(np.asarray(point, dtype=np.float64) for point in waypoints)]
        for first, second in pairwise(route):
            if self.collision_map.segment_check(first, second).collision:
                raise RuntimeError("Patrol route contains a colliding segment")
            if (
                float(np.linalg.norm(second - first))
                > self.path_planner.max_waypoint_step_m + 1e-9
            ):
                raise RuntimeError("Patrol route contains an over-limit segment")
        if float(np.linalg.norm(route[-1] - start)) > 1e-6:
            raise RuntimeError("Patrol route is not closed")

        path_length = sum(
            float(np.linalg.norm(second - first))
            for first, second in pairwise(route)
        )
        payload = {
            "start_xy_m": start.tolist(),
            "anchors_xy_m": [point.tolist() for point in anchors],
            "waypoints_xy_m": [list(point) for point in waypoints],
        }
        digest = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        return NumericPatrolPlan(
            start_xy_m=(float(start[0]), float(start[1])),
            anchors_xy_m=tuple(
                (float(point[0]), float(point[1])) for point in anchors
            ),
            waypoints_xy_m=tuple(waypoints),
            path_length_m=path_length,
            expanded_nodes=sum(leg.expanded_nodes for leg in legs),
            winding_area_m2=_polygon_area(anchors),
            plan_sha256=digest,
        )


def _binding(runtime: Any) -> dict[str, Any]:
    value = runtime.prefix_binding()
    if not isinstance(value, Mapping):
        raise TypeError("Patrol runtime returned an invalid prefix binding")
    required = (
        "active_prefix_sha256",
        "scene_prefix_sha256",
        "robot_tokens_sha256",
        "map_sha256",
    )
    if any(not isinstance(value.get(field), str) for field in required):
        raise ValueError("Patrol runtime prefix binding is incomplete")
    return dict(value)


def _static_signature(binding: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(
        binding.get(field)
        for field in (
            "scene_id",
            "map_version",
            "map_sha256",
            "scene_prefix_sha256",
            "source_voxels",
            "processed_voxels",
        )
    )


def execute_numeric_patrol(
    runtime: Any,
    config: Mapping[str, Any],
    *,
    anchor_count: int = 8,
    max_waypoints: int = 64,
) -> dict[str, Any]:
    """Execute one closed room-scale patrol against the bound static 3D map.

    This is the deterministic geometry half of an outcome-level agent.  It
    never consumes a camera frame, question-dependent retrieval result, label,
    scene caption, or object inventory.  Each generated waypoint is validated
    through the same bounded tool schema used by the learned semantic policy.
    """

    if not isinstance(config, Mapping):
        raise TypeError("Patrol configuration must be a mapping")
    simulator = getattr(runtime, "simulator", None)
    settings = getattr(simulator, "settings", None)
    collision_map = getattr(simulator, "collision_map", None)
    if not isinstance(settings, Mapping) or settings.get("auto_scan_after_motion") is not False:
        raise ValueError("Patrol requires static-map auto_scan_after_motion=false")
    if not isinstance(collision_map, NumericCollisionMap):
        raise TypeError("Patrol requires the runtime's anonymous numeric collision map")

    initial_binding = _binding(runtime)
    initial_signature = _static_signature(initial_binding)
    initial_state = runtime.get_robot_state()
    if not isinstance(initial_state, Mapping):
        raise TypeError("Patrol runtime returned an invalid numeric robot state")
    position = np.asarray(initial_state.get("position_m"), dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("Patrol runtime position_m must contain three finite values")
    scan_count = initial_state.get("scan_count")
    if isinstance(scan_count, bool) or not isinstance(scan_count, int) or scan_count < 0:
        raise ValueError("Patrol runtime returned an invalid scan_count")

    robot = config.get("robot")
    if not isinstance(robot, Mapping):
        raise TypeError("Patrol configuration has no robot settings")
    max_step = float(robot.get("max_move_to_m", 1.0))
    if not math.isfinite(max_step) or max_step <= 0.0:
        raise ValueError("robot.max_move_to_m must be finite and positive")
    plan = NumericPatrolPlanner(
        collision_map,
        anchor_count=anchor_count,
        max_waypoint_step_m=min(0.5, max_step),
        max_waypoints=max_waypoints,
    ).plan(position[:2])

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
        candidate = json.dumps(
            {"tool": "move_to", "arguments": {"x": x, "y": y}},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        validation = validate_tool_call_text(
            candidate,
            config,
            robot_state=before_state,
        )
        if validation.call is None:
            error_code = validation.error_code or "E_SCHEMA"
            break
        receipt = execute_validated_tool_call(runtime, validation.call, config=config)
        receipts.append(dict(receipt))
        after = _binding(runtime)
        after_state = runtime.get_robot_state()
        steps.append(
            {
                "step": index,
                "waypoint_xy_m": [x, y],
                "numeric_tool_receipt": dict(receipt),
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
            after.get("active_prefix_sha256") == before.get("active_prefix_sha256")
            or after.get("robot_tokens_sha256") == before.get("robot_tokens_sha256")
        ):
            error_code = "E_ROBOT_PREFIX_STALE"
            break

    final_binding = _binding(runtime)
    final_state = runtime.get_robot_state()
    success = error_code is None and len(receipts) == len(plan.waypoints_xy_m)
    return {
        "schema": "semantic_3d_chat.numeric_patrol_execution.v1",
        "kind": "navigation",
        "command": "semantic_map_patrol",
        "success": success,
        "error_code": error_code,
        "planned_action_count": len(plan.waypoints_xy_m),
        "completed_action_count": sum(item.get("success") is True for item in receipts),
        "action_receipts": receipts,
        "steps": steps,
        "plan": plan.as_dict(),
        "initial_prefix_binding": initial_binding,
        "final_prefix_binding": final_binding,
        "initial_scan_count": scan_count,
        "final_scan_count": final_state.get("scan_count") if isinstance(final_state, Mapping) else None,
        "camera_observations_during_goal": 0,
        "static_scene_prefix_unchanged": _static_signature(final_binding)
        == initial_signature,
        "scene_prefix_computed_before_goal": True,
        "geometry_source": "complete_precomputed_numeric_voxel_map",
        "user_motor_commands_exposed": False,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
    }


__all__ = ["NumericPatrolPlan", "NumericPatrolPlanner", "execute_numeric_patrol"]
