from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from semantic_3d_chat.mapping.voxel_map import SparseVoxelMap
from semantic_3d_chat.robot.semantic_agent import (
    ContinuousSemanticTargetGrounder,
    LabelFreeSemanticNavigator,
)
from semantic_3d_chat.robot.simulator import EmbodiedCameraSimulator


class FakeTextEncoder:
    output_dim = 4

    def encode_queries(self, queries: list[str] | tuple[str, ...]) -> np.ndarray:
        values = {
            "fixture": np.asarray([0.65, 0.76, 0.0, 0.0], dtype=np.float32),
            "a fixture": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }
        return np.stack([values[query] for query in queries])


def _write_map(path: Path) -> None:
    target = np.asarray(
        [
            [1.45, -0.05, 0.45],
            [1.50, -0.05, 0.50],
            [1.55, -0.05, 0.55],
            [1.45, 0.05, 0.50],
            [1.50, 0.05, 0.55],
            [1.55, 0.05, 0.45],
        ],
        dtype=np.float32,
    )
    distractor = target.copy()
    distractor[:, 0] *= -1.0
    points = np.concatenate((target, distractor))
    features = np.zeros((len(points), 4), dtype=np.float32)
    features[: len(target), 0] = 1.0
    features[len(target) :, 1] = 1.0
    voxel_map = SparseVoxelMap(0.05, feature_dim=4)
    voxel_map.add_observations(
        points,
        features,
        rgb=np.tile(np.asarray([[80.0, 120.0, 160.0]], dtype=np.float32), (len(points), 1)),
        frame_id="f_000001",
    )
    voxel_map.save(path, metadata={"scene_id": "scene_000001"})


def _config(tmp_path: Path, map_path: Path) -> dict:
    return {
        "seed": 7,
        "paths": {
            "data_root": str(tmp_path / "runtime"),
            "maps_root": str(map_path.parents[1]),
        },
        "scene": {"room_size_m": [6.0, 5.0, 3.0]},
        "render": {"resolution": [16, 16], "horizontal_fov_degrees": 72.0},
        "robot": {
            "radius_m": 0.20,
            "camera_height_m": 1.20,
            "max_move_m": 0.50,
            "max_move_to_m": 0.50,
            "max_turn_degrees": 45.0,
            "max_look_delta_degrees": 30.0,
            "max_camera_yaw_offset_degrees": 60.0,
            "max_pitch_degrees": 45.0,
            "collision_z_min_m": 0.12,
            "collision_z_max_m": 1.80,
            "surface_padding_m": 0.02,
            "scan_depth_min_m": 0.10,
            "scan_depth_max_m": 6.0,
            "initial_position_xy_m": [0.0, 0.0],
            "history_length": 16,
        },
    }


def test_continuous_grounding_uses_all_numeric_voxels_and_adaptive_article(
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "maps" / "scene_000001" / "voxel_map.npz"
    _write_map(map_path)
    grounder = ContinuousSemanticTargetGrounder(
        map_path,
        FakeTextEncoder(),
        room_size_m=(6.0, 5.0, 3.0),
        feature_start=0,
        feature_dim=4,
    )

    result = grounder.ground("fixture")

    assert result.prompt_variant_index == 1
    assert result.scored_voxels == 12
    assert result.eligible_voxels == 12
    assert 4 <= result.local_support_voxels <= 6
    assert result.target_xyz_m[0] == pytest.approx(1.5, abs=0.05)
    assert result.target_xyz_m[1] == pytest.approx(0.0, abs=0.05)
    assert result.cosine_similarity == pytest.approx(1.0)
    assert len(result.query_embedding_sha256) == 64
    assert len(result.map_sha256) == 64


def test_semantic_navigator_executes_bounded_collision_free_numeric_actions(
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "maps" / "scene_000001" / "voxel_map.npz"
    _write_map(map_path)
    config = _config(tmp_path, map_path)
    simulator = EmbodiedCameraSimulator(config, "scene_000001")
    grounder = ContinuousSemanticTargetGrounder(
        map_path,
        FakeTextEncoder(),
        room_size_m=config["scene"]["room_size_m"],
        feature_start=0,
        feature_dim=4,
    )

    result = LabelFreeSemanticNavigator(simulator, grounder).navigate(
        "fixture",
        scan_on_arrival=False,
    )

    assert result.success is True
    assert result.collision_count == 0
    assert result.movement_actions >= 1
    assert result.final_target_distance_m == pytest.approx(0.60, abs=0.21)
    assert all(
        np.linalg.norm(np.asarray(current) - np.asarray(previous)) <= 0.50 + 1e-9
        for previous, current in zip(
            ((0.0, 0.0), *result.plan.waypoints_xy_m[:-1]),
            result.plan.waypoints_xy_m,
            strict=True,
        )
    )
    encoded = json.dumps(result.as_dict(), sort_keys=True, allow_nan=False).casefold()
    for prohibited in ("object", "category", "caption", "relationship", "oracle"):
        assert prohibited not in encoded
    assert "fixture" not in encoded


def test_semantic_navigator_can_execute_through_refreshing_action_surface(
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "maps" / "scene_000001" / "voxel_map.npz"
    _write_map(map_path)
    config = _config(tmp_path, map_path)
    simulator = EmbodiedCameraSimulator(config, "scene_000001")
    grounder = ContinuousSemanticTargetGrounder(
        map_path,
        FakeTextEncoder(),
        room_size_m=config["scene"]["room_size_m"],
        feature_start=0,
        feature_dim=4,
    )

    class RecordingSurface:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def turn(self, angle_degrees: float) -> dict:
            self.calls.append("turn")
            return simulator.turn(angle_degrees)

        def move_to(self, x: float, y: float) -> dict:
            self.calls.append("move_to")
            return simulator.move_to(x, y)

        def scan(self) -> dict:
            self.calls.append("scan")
            return {"success": True}

    surface = RecordingSurface()
    result = LabelFreeSemanticNavigator(
        simulator,
        grounder,
        action_surface=surface,
    ).navigate("fixture", scan_on_arrival=True)

    assert result.success is True
    assert "move_to" in surface.calls
    assert "turn" in surface.calls
    assert surface.calls[-1] == "scan"


def test_semantic_grounder_rejects_oracle_and_qa_paths(tmp_path: Path) -> None:
    for prohibited in ("oracle", "qa"):
        path = tmp_path / prohibited / "scene_000001" / "voxel_map.npz"
        with pytest.raises(ValueError, match="oracle or QA"):
            ContinuousSemanticTargetGrounder(
                path,
                FakeTextEncoder(),
                room_size_m=(6.0, 5.0, 3.0),
                feature_start=0,
                feature_dim=4,
            )
