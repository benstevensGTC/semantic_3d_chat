from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from semantic_3d_chat.mapping.voxel_map import SparseVoxelMap
from semantic_3d_chat.robot.collision import NumericCollisionMap
from semantic_3d_chat.robot.semantic_between import execute_semantic_between_goal
from semantic_3d_chat.robot.semantic_goal_fallback import execute_grounded_goal_fallback
from semantic_3d_chat.robot.semantic_mapping import semantic_map_content_hash

_ALIGNED_START = 1536
_ALIGNED_DIM = 1536


class _TwoTargetTextEncoder:
    output_dim = _ALIGNED_DIM

    def __init__(self) -> None:
        self.queries: list[str] = []

    def encode_queries(self, queries: list[str] | tuple[str, ...]) -> np.ndarray:
        self.queries.extend(queries)
        output = np.zeros((len(queries), self.output_dim), dtype=np.float32)
        for index, query in enumerate(queries):
            normalized = query.casefold()
            if "first beacon" in normalized:
                output[index, 0] = 1.0
            elif "second beacon" in normalized:
                output[index, 1] = 1.0
            else:
                raise AssertionError(f"Unexpected test query: {query}")
        return output


def _write_map(path: Path, *, occupied_midpoint: bool = False) -> int:
    first = np.asarray(
        [
            [-1.05, -0.05, 0.45],
            [-1.00, -0.05, 0.50],
            [-0.95, -0.05, 0.55],
            [-1.05, 0.05, 0.50],
            [-0.95, 0.05, 0.45],
        ],
        dtype=np.float32,
    )
    second = first.copy()
    second[:, 0] *= -1.0
    distractors = np.asarray([[2.0, 1.5, 0.50], [-2.0, 1.5, 0.50]], dtype=np.float32)
    point_groups = [first, second, distractors]
    if occupied_midpoint:
        point_groups.append(np.asarray([[0.025, 0.025, 0.50]], dtype=np.float32))
    points = np.concatenate(point_groups)
    features = np.zeros((len(points), _ALIGNED_START + _ALIGNED_DIM), dtype=np.float32)
    features[: len(first), _ALIGNED_START] = 1.0
    features[len(first) : len(first) + len(second), _ALIGNED_START + 1] = 1.0
    features[len(first) + len(second) :, _ALIGNED_START + 2] = 1.0
    voxel_map = SparseVoxelMap(0.05, feature_dim=features.shape[1])
    voxel_map.add_observations(
        points,
        features,
        rgb=np.full((len(points), 3), 128.0, dtype=np.float32),
        frame_id="f_000001",
    )
    voxel_map.save(path, metadata={"scene_id": "scene_000001"})
    with np.load(path, allow_pickle=False) as archive:
        return len(archive["centers_world"])


def _config() -> dict[str, Any]:
    return {
        "scene": {"room_size_m": [6.0, 5.0, 3.0]},
        "robot": {
            "radius_m": 0.20,
            "max_move_m": 0.50,
            "max_move_to_m": 0.50,
            "max_turn_degrees": 45.0,
            "max_look_delta_degrees": 30.0,
            "max_camera_yaw_offset_degrees": 60.0,
            "max_pitch_degrees": 45.0,
            "collision_z_min_m": 0.12,
            "collision_z_max_m": 1.80,
            "surface_padding_m": 0.02,
            "auto_scan_after_motion": False,
            "between_goal_search_resolution_m": 0.05,
            "between_goal_angular_samples": 48,
            "between_goal_max_actions": 64,
        },
    }


class _StaticBetweenRuntime:
    def __init__(
        self,
        map_path: Path,
        source_voxels: int,
        *,
        refresh_robot_tokens: bool = True,
        mutate_scene_after_motion: bool = False,
    ) -> None:
        self.map_path = map_path
        self.source_voxels = source_voxels
        self.map_hash = semantic_map_content_hash(map_path)
        self.position = np.asarray([0.0, -1.20], dtype=np.float64)
        self.yaw = 0.0
        self.robot_version = 0
        self.map_version = 0
        self.scan_count = 0
        self.refresh_robot_tokens = refresh_robot_tokens
        self.mutate_scene_after_motion = mutate_scene_after_motion
        self.actions: list[tuple[float, float] | str] = []
        self.map_updater = SimpleNamespace(
            base_map_path=map_path,
            persistent_map_path=map_path.parent / "unused_semantic_map.npz",
        )
        collision = NumericCollisionMap.from_voxel_map(
            map_path,
            room_size_m=[6.0, 5.0, 3.0],
            robot_radius_m=0.20,
            collision_z_min_m=0.12,
            collision_z_max_m=1.80,
            surface_padding_m=0.02,
        )
        self.simulator = SimpleNamespace(
            settings={"auto_scan_after_motion": False},
            collision_map=collision,
        )

    def _robot_hash(self) -> str:
        return hashlib.sha256(str(self.robot_version).encode("ascii")).hexdigest()

    def prefix_binding(self) -> dict[str, Any]:
        robot_hash = self._robot_hash()
        scene_hash = hashlib.sha256(f"scene:{self.map_version}".encode()).hexdigest()
        return {
            "scene_id": "scene_000001",
            "map_version": self.map_version,
            "map_sha256": self.map_hash,
            "scene_prefix_sha256": scene_hash,
            "active_prefix_sha256": hashlib.sha256(
                f"{scene_hash}:{robot_hash}".encode()
            ).hexdigest(),
            "robot_tokens_sha256": robot_hash,
            "source_voxels": self.source_voxels,
            "processed_voxels": self.source_voxels,
        }

    def get_robot_state(self) -> dict[str, Any]:
        return {
            "success": True,
            "scene_id": "scene_000001",
            "scene_version": self.map_version,
            "position_m": [float(self.position[0]), float(self.position[1]), 0.0],
            "body_yaw_degrees": self.yaw,
            "camera_yaw_degrees": self.yaw,
            "pitch_degrees": 0.0,
            "collision": False,
            "scan_count": self.scan_count,
            "stopped": False,
            **self.prefix_binding(),
        }

    def move_to(self, x: float, y: float) -> dict[str, Any]:
        target = np.asarray([x, y], dtype=np.float64)
        check = self.simulator.collision_map.segment_check(self.position, target)
        if check.collision:
            return {
                **self.get_robot_state(),
                "command": "move_to",
                "success": False,
                "error_code": "E_COLLISION",
                "distance_moved": 0.0,
            }
        distance = float(np.linalg.norm(target - self.position))
        self.position = target
        self.actions.append((x, y))
        if self.refresh_robot_tokens:
            self.robot_version += 1
        if self.mutate_scene_after_motion:
            self.map_version += 1
        return {
            **self.get_robot_state(),
            "command": "move_to",
            "success": True,
            "error_code": None,
            "distance_moved": distance,
        }

    def turn(self, angle_degrees: float) -> dict[str, Any]:
        self.yaw = (self.yaw + angle_degrees + 180.0) % 360.0 - 180.0
        self.actions.append("turn")
        if self.refresh_robot_tokens:
            self.robot_version += 1
        if self.mutate_scene_after_motion:
            self.map_version += 1
        return {
            **self.get_robot_state(),
            "command": "turn",
            "success": True,
            "error_code": None,
            "distance_moved": 0.0,
        }


def _execute(runtime: _StaticBetweenRuntime, encoder: _TwoTargetTextEncoder) -> dict[str, Any]:
    return execute_semantic_between_goal(
        runtime,
        _config(),
        first_target_text="the first beacon",
        second_target_text="the second beacon",
        text_encoder=encoder,
    )


def test_between_goal_scores_complete_bound_map_and_moves_without_camera(
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "maps" / "scene_000001" / "voxel_map.npz"
    source_voxels = _write_map(map_path)
    runtime = _StaticBetweenRuntime(map_path, source_voxels)
    encoder = _TwoTargetTextEncoder()

    result = _execute(runtime, encoder)

    assert result["success"] is True
    assert result["all_target_groundings_scored_complete_bound_map"] is True
    assert [item["scored_voxels"] for item in result["groundings"]] == [
        source_voxels,
        source_voxels,
    ]
    assert result["selected_goal_offset_from_midpoint_m"] == pytest.approx(0.0)
    assert result["planned_action_count"] == result["completed_action_count"]
    assert result["planned_action_count"] >= 2
    assert result["camera_observations_during_goal"] == 0
    assert result["initial_scan_count"] == result["final_scan_count"] == 0
    assert result["static_scene_prefix_unchanged"] is True
    assert result["robot_tokens_refreshed_after_every_motion"] is True
    assert runtime.position == pytest.approx(result["selected_goal_xy_m"])
    assert encoder.queries == [
        "the first beacon",
        "the second beacon",
    ]
    encoded = json.dumps(result, sort_keys=True, allow_nan=False).casefold()
    assert "the first beacon" not in encoded
    assert "the second beacon" not in encoded
    assert result["environmental_text_inputs"] == []
    assert result["oracle_inputs_at_runtime"] is False


def test_between_goal_selects_nearest_free_shell_when_midpoint_is_occupied(
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "maps" / "scene_000001" / "voxel_map.npz"
    source_voxels = _write_map(map_path, occupied_midpoint=True)
    runtime = _StaticBetweenRuntime(map_path, source_voxels)

    result = _execute(runtime, _TwoTargetTextEncoder())

    assert result["success"] is True
    assert 0.0 < result["selected_goal_offset_from_midpoint_m"] <= 0.25 + 1e-6
    assert runtime.simulator.collision_map.point_check(
        tuple(result["selected_goal_xy_m"])
    ).collision is False


def test_between_goal_rejects_source_count_mismatch_before_motion(tmp_path: Path) -> None:
    map_path = tmp_path / "maps" / "scene_000001" / "voxel_map.npz"
    source_voxels = _write_map(map_path)
    runtime = _StaticBetweenRuntime(map_path, source_voxels + 1)

    with pytest.raises(RuntimeError, match="exact bound map"):
        _execute(runtime, _TwoTargetTextEncoder())

    assert runtime.actions == []


@pytest.mark.parametrize(
    ("runtime_options", "expected_error"),
    [
        ({"refresh_robot_tokens": False}, "E_ROBOT_PREFIX_STALE"),
        ({"mutate_scene_after_motion": True}, "E_STATIC_SCENE_CHANGED"),
    ],
)
def test_between_goal_fails_closed_on_runtime_invariance_violation(
    tmp_path: Path,
    runtime_options: dict[str, bool],
    expected_error: str,
) -> None:
    map_path = tmp_path / "maps" / "scene_000001" / "voxel_map.npz"
    source_voxels = _write_map(map_path)
    runtime = _StaticBetweenRuntime(map_path, source_voxels, **runtime_options)

    result = _execute(runtime, _TwoTargetTextEncoder())

    assert result["success"] is False
    assert result["error_code"] == expected_error
    assert result["camera_observations_during_goal"] == 0
    assert result["initial_scan_count"] == result["final_scan_count"] == 0


def _fallback(
    runtime: _StaticBetweenRuntime,
    *,
    kind: str,
) -> dict[str, Any]:
    return execute_grounded_goal_fallback(
        runtime,
        _config(),
        kind=kind,  # type: ignore[arg-type]
        target_text="the first beacon",
        text_encoder=_TwoTargetTextEncoder(),
    )


def test_face_fallback_globally_grounds_then_turns_without_camera(tmp_path: Path) -> None:
    map_path = tmp_path / "maps" / "scene_000001" / "voxel_map.npz"
    source_voxels = _write_map(map_path)
    runtime = _StaticBetweenRuntime(map_path, source_voxels)

    result = _fallback(runtime, kind="face")

    assert result["success"] is True
    assert result["grounding"]["scored_voxels"] == source_voxels
    assert result["grounding_scope"] == "every_active_map_voxel"
    assert result["all_target_groundings_scored_complete_bound_map"] is True
    assert result["plan"] is None
    assert result["planned_action_count"] == result["completed_action_count"] >= 1
    assert {step["tool"] for step in result["steps"]} == {"turn"}
    assert runtime.position == pytest.approx([0.0, -1.20])
    target = np.asarray(result["grounding"]["target_xyz_m"][:2])
    delta = target - runtime.position
    desired = np.degrees(np.arctan2(-delta[0], delta[1]))
    error = (desired - runtime.yaw + 180.0) % 360.0 - 180.0
    assert abs(error) <= 3.0
    assert result["camera_observations_during_goal"] == 0
    assert result["initial_scan_count"] == result["final_scan_count"] == 0
    assert result["static_scene_prefix_unchanged"] is True
    assert result["robot_tokens_refreshed_after_every_action"] is True
    assert "the first beacon" not in json.dumps(result).casefold()


def test_approach_fallback_plans_standoff_moves_then_faces_target(tmp_path: Path) -> None:
    map_path = tmp_path / "maps" / "scene_000001" / "voxel_map.npz"
    source_voxels = _write_map(map_path)
    runtime = _StaticBetweenRuntime(map_path, source_voxels)

    result = _fallback(runtime, kind="approach")

    assert result["success"] is True
    assert result["plan"] is not None
    assert result["plan"]["target_distance_m"] == pytest.approx(0.50, abs=0.11)
    assert result["final_target_distance_m"] == pytest.approx(0.50, abs=0.11)
    assert result["planned_action_count"] == result["completed_action_count"]
    assert result["planned_action_count"] >= 1
    assert result["steps"][0]["tool"] == "move_to"
    assert result["camera_observations_during_goal"] == 0
    assert result["static_scene_prefix_unchanged"] is True
    assert result["robot_tokens_refreshed_after_every_action"] is True


def test_face_fallback_fails_closed_when_robot_tokens_do_not_refresh(
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "maps" / "scene_000001" / "voxel_map.npz"
    source_voxels = _write_map(map_path)
    runtime = _StaticBetweenRuntime(
        map_path,
        source_voxels,
        refresh_robot_tokens=False,
    )

    result = _fallback(runtime, kind="face")

    assert result["success"] is False
    assert result["error_code"] == "E_ROBOT_PREFIX_STALE"
    assert result["camera_observations_during_goal"] == 0
