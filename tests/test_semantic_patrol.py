from __future__ import annotations

import hashlib
import json
from collections import deque
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

from semantic_3d_chat.robot.collision import NumericCollisionMap
from semantic_3d_chat.robot.semantic_patrol import (
    NumericPatrolPlanner,
    execute_numeric_patrol,
)


def _room_map() -> NumericCollisionMap:
    # Anonymous vertical obstacle samples leave connected free space around the
    # center and force at least one patrol leg through the A* path.
    vertical = np.stack(
        (
            np.zeros(25, dtype=np.float32),
            np.linspace(-0.9, 0.9, 25, dtype=np.float32),
        ),
        axis=1,
    )
    side = np.asarray([[1.8, 1.1], [-1.7, -1.2]], dtype=np.float32)
    return NumericCollisionMap(
        np.concatenate((vertical, side), axis=0),
        room_min_xy_m=(-3.0, -2.5),
        room_max_xy_m=(3.0, 2.5),
        robot_radius_m=0.18,
        surface_padding_m=0.03,
    )


def test_patrol_is_closed_room_scale_collision_free_and_bounded() -> None:
    collision = _room_map()
    start = np.asarray([-0.8, -0.4], dtype=np.float64)
    plan = NumericPatrolPlanner(
        collision,
        anchor_count=8,
        max_waypoint_step_m=0.35,
    ).plan(start)

    assert len(plan.anchors_xy_m) == 8
    assert 8 <= len(plan.waypoints_xy_m) <= 64
    assert plan.winding_area_m2 > 4.0
    assert plan.path_length_m > 8.0
    assert np.asarray(plan.waypoints_xy_m[-1]) == pytest.approx(start)
    route = [start, *(np.asarray(point) for point in plan.waypoints_xy_m)]
    for first, second in pairwise(route):
        assert collision.segment_check(first, second).collision is False
        assert np.linalg.norm(second - first) <= 0.35 + 1e-9


def test_patrol_payload_is_numeric_and_has_no_environmental_text() -> None:
    plan = NumericPatrolPlanner(_room_map(), anchor_count=6).plan((-0.8, -0.4))

    encoded = json.dumps(plan.as_dict(), sort_keys=True, allow_nan=False).casefold()
    assert len(plan.plan_sha256) == 64
    assert '"environmental_text_inputs": []' in encoded
    for prohibited in (
        "object_name",
        "object_label",
        "category",
        "caption",
        "scene_graph",
        "relationship",
    ):
        assert prohibited not in encoded


class _StaticPatrolRuntime:
    def __init__(self) -> None:
        self.position = np.asarray([-0.8, -0.4], dtype=np.float64)
        self.action_count = 0
        self.simulator = SimpleNamespace(
            settings={"auto_scan_after_motion": False},
            collision_map=_room_map(),
            history=deque(maxlen=128),
        )

    def _robot_hash(self) -> str:
        return hashlib.sha256(self.position.tobytes()).hexdigest()

    def prefix_binding(self) -> dict[str, object]:
        robot_hash = self._robot_hash()
        return {
            "scene_id": "scene_000001",
            "map_version": 0,
            "map_sha256": "a" * 64,
            "scene_prefix_sha256": "b" * 64,
            "active_prefix_sha256": hashlib.sha256(
                ("b" * 64 + robot_hash).encode("ascii")
            ).hexdigest(),
            "robot_tokens_sha256": robot_hash,
            "source_voxels": 1234,
            "processed_voxels": 1234,
        }

    def get_robot_state(self) -> dict[str, object]:
        return {
            "success": True,
            "scene_id": "scene_000001",
            "scene_version": 0,
            "position_m": [float(self.position[0]), float(self.position[1]), 0.0],
            "body_yaw_degrees": 0.0,
            "camera_yaw_degrees": 0.0,
            "pitch_degrees": 0.0,
            "collision": False,
            "scan_count": 0,
            "action_count": self.action_count,
            "stopped": False,
            **self.prefix_binding(),
        }

    def move_to(self, x: float, y: float) -> dict[str, object]:
        target = np.asarray([x, y], dtype=np.float64)
        check = self.simulator.collision_map.segment_check(self.position, target)
        self.action_count += 1
        if check.collision:
            return {**self.get_robot_state(), "success": False, "error_code": "E_COLLISION"}
        distance = float(np.linalg.norm(target - self.position))
        self.position = target
        receipt = {
            **self.get_robot_state(),
            "success": True,
            "error_code": None,
            "distance_moved": distance,
        }
        self.simulator.history.append(receipt)
        return receipt


def test_execute_patrol_uses_static_prefix_and_never_scans() -> None:
    runtime = _StaticPatrolRuntime()
    config = {
        "scene": {"room_size_m": [6.0, 5.0, 3.0]},
        "robot": {
            "max_move_to_m": 1.0,
            "max_move_m": 0.5,
            "max_turn_degrees": 45.0,
            "max_look_delta_degrees": 45.0,
            "max_camera_yaw_offset_degrees": 90.0,
            "max_pitch_degrees": 45.0,
        },
    }

    result = execute_numeric_patrol(runtime, config)

    assert result["success"] is True
    assert result["planned_action_count"] == result["completed_action_count"]
    assert result["planned_action_count"] >= 8
    assert result["camera_observations_during_goal"] == 0
    assert result["initial_scan_count"] == result["final_scan_count"] == 0
    assert result["static_scene_prefix_unchanged"] is True
    assert runtime.position == pytest.approx([-0.8, -0.4])
    assert all(step["scene_prefix_sha256"] == "b" * 64 for step in result["steps"])
