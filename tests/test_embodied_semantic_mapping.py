from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from semantic_3d_chat.mapping.voxel_map import SparseVoxelMap
from semantic_3d_chat.robot.semantic_mapping import (
    build_semantic_embodied_runtime,
    semantic_map_content_hash,
)
from semantic_3d_chat.robot.simulator import NumericMapScanner, NumericObservation
from semantic_3d_chat.vision.patch_features import DensePatchFeatures


class CountingDenseEncoder:
    def __init__(self, *, component_dim: int = 2) -> None:
        self.calls = 0
        self.component_dim = component_dim
        self.complete_shapes: list[tuple[int, ...]] = []

    def encode_image(self, image: np.ndarray) -> DensePatchFeatures:
        self.calls += 1
        self.complete_shapes.append(tuple(image.shape))
        value = float(image.astype(np.float32).mean() / 255.0) + 1.0
        streams = [
            torch.full((2, 2, self.component_dim), value + offset)
            for offset in (0.0, 1.0, 2.0)
        ]
        return DensePatchFeatures(*streams)


class FreshObservationScanner:
    """Small renderer-shaped fixture that does not reproject the base map."""

    def __init__(self) -> None:
        self.capture_calls = 0
        self.directional_coverage = 0.0

    def capture(
        self,
        *,
        observation_index: int,
        camera_position_m: tuple[float, float, float],
        yaw_degrees: float,
        pitch_degrees: float,
    ) -> NumericObservation:
        del yaw_degrees, pitch_degrees
        self.capture_calls += 1
        camera_to_world = np.eye(4, dtype=np.float64)
        camera_to_world[:3, 3] = np.asarray(camera_position_m, dtype=np.float64)
        return NumericObservation(
            observation_id=f"o_{observation_index:06d}",
            rgb=np.full((8, 8, 3), 96, dtype=np.uint8),
            depth_m=np.full((8, 8), 1.25, dtype=np.float32),
            intrinsics=np.asarray(
                [[5.0, 0.0, 3.5], [0.0, 5.0, 3.5], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            camera_to_world=camera_to_world,
            visible_voxel_indices=np.empty(0, dtype=np.int64),
        )

    def integrate(self, observation: NumericObservation, observed_mask: np.ndarray) -> int:
        del observed_mask
        self.directional_coverage = 0.125
        return int(np.count_nonzero(observation.depth_m > 0))


def _base_map(path: Path) -> None:
    points = np.asarray(
        [
            [-0.30, 1.00, 1.00],
            [-0.10, 1.00, 1.10],
            [0.10, 1.00, 1.20],
            [0.30, 1.00, 1.30],
            [-0.25, 1.25, 1.35],
            [0.00, 1.25, 1.20],
            [0.25, 1.25, 1.05],
        ],
        dtype=np.float32,
    )
    features = np.arange(1, 43, dtype=np.float32).reshape(7, 6)
    rgb = np.tile(np.asarray([[40.0, 130.0, 210.0]], dtype=np.float32), (7, 1))
    voxel_map = SparseVoxelMap(0.05, feature_dim=6)
    voxel_map.add_observations(
        points,
        features,
        rgb=rgb,
        frame_id="f_000000",
    )
    voxel_map.save(path, metadata={"scene_id": "scene_000001"})


def _config(tmp_path: Path) -> dict:
    maps = tmp_path / "maps"
    _base_map(maps / "scene_000001" / "voxel_map.npz")
    return {
        "seed": 7,
        "paths": {
            "data_root": str(tmp_path / "runtime"),
            "maps_root": str(maps),
        },
        "scene": {"room_size_m": [6.0, 5.0, 3.0]},
        "render": {"resolution": [8, 8], "horizontal_fov_degrees": 72.0},
        "mapping": {
            "depth_min_m": 0.1,
            "depth_max_m": 6.0,
            "pixel_stride": 1,
            "confidence_distance_scale_m": 6.0,
        },
        "robot": {
            "radius_m": 0.20,
            "camera_height_m": 1.20,
            "max_move_m": 0.50,
            "max_move_to_m": 1.0,
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


def test_numeric_pose_observation_is_deterministic_and_sanitized(tmp_path: Path) -> None:
    config = _config(tmp_path)
    map_path = Path(config["paths"]["maps_root"]) / "scene_000001" / "voxel_map.npz"
    scanner = NumericMapScanner(
        map_path,
        resolution=(8, 8),
        horizontal_fov_degrees=72.0,
        depth_min_m=0.1,
        depth_max_m=6.0,
        output_directory=tmp_path / "observations",
    )
    pose = {
        "camera_position_m": (0.0, 0.0, 1.2),
        "yaw_degrees": 0.0,
        "pitch_degrees": 0.0,
    }
    first = scanner.capture(observation_index=1, **pose)
    second = scanner.capture(observation_index=2, **pose)
    assert np.array_equal(first.rgb, second.rgb)
    assert np.array_equal(first.depth_m, second.depth_m)
    assert np.array_equal(first.intrinsics, second.intrinsics)
    assert np.array_equal(first.camera_to_world, second.camera_to_world)
    assert first.depth_m.max() > 0
    with np.load(tmp_path / "observations" / "o_000001.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {
            "rgb",
            "depth_m",
            "intrinsics",
            "camera_to_world",
            "visible_voxel_indices",
        }


def test_scan_runs_one_complete_image_pass_and_commits_map(tmp_path: Path) -> None:
    config = _config(tmp_path)
    encoder = CountingDenseEncoder()
    persistent = tmp_path / "persistent" / "semantic_map.npz"
    runtime = build_semantic_embodied_runtime(
        config,
        "scene_000001",
        encoder=encoder,
        persistent_map_path=persistent,
    )

    result = runtime.simulator.scan()

    assert result["success"] is True
    assert result["scene_version"] == 1
    assert encoder.calls == 1
    assert encoder.complete_shapes == [(8, 8, 3)]
    assert isinstance(result["map_sha256"], str) and len(result["map_sha256"]) == 64
    assert persistent.is_file()
    committed = SparseVoxelMap.load(persistent)
    assert committed.feature_dim == 6
    assert semantic_map_content_hash(persistent) == result["map_sha256"]
    with np.load(persistent, allow_pickle=False) as archive:
        header = json.loads(str(archive["metadata_json"].item()))
    receipt = header["metadata"]
    assert receipt["map_version"] == 1
    assert receipt["vision_encoder_calls"] == 1
    assert receipt["map_sha256"] == result["map_sha256"]
    encoded = json.dumps(result)
    for prohibited in ("category", "caption", "relationship", "object_name", "oracle"):
        assert prohibited not in encoded


def test_semantic_runtime_accepts_fresh_observation_scanner(tmp_path: Path) -> None:
    config = _config(tmp_path)
    encoder = CountingDenseEncoder()
    scanner = FreshObservationScanner()
    runtime = build_semantic_embodied_runtime(
        config,
        "scene_000001",
        encoder=encoder,
        observation_scanner=scanner,
        persistent_map_path=tmp_path / "persistent" / "semantic_map.npz",
    )

    result = runtime.simulator.scan()

    assert result["success"] is True
    assert result["valid_depth_pixels"] == 64
    assert result["visible_voxels"] == 0
    assert result["scan_coverage"] == 0.125
    assert scanner.capture_calls == 1
    assert runtime.simulator.scanner is scanner
    assert encoder.complete_shapes == [(8, 8, 3)]


def test_failed_feature_fusion_leaves_map_and_version_uncommitted(tmp_path: Path) -> None:
    config = _config(tmp_path)
    encoder = CountingDenseEncoder(component_dim=3)
    persistent = tmp_path / "persistent" / "semantic_map.npz"
    runtime = build_semantic_embodied_runtime(
        config,
        "scene_000001",
        encoder=encoder,
        persistent_map_path=persistent,
    )

    result = runtime.simulator.scan()

    assert result["success"] is False
    assert result["error_code"] == "E_MAP_UPDATE"
    assert result["scene_version"] == 0
    assert result["scan_count"] == 0
    assert result["scan_coverage"] == 0.0
    assert encoder.calls == 1
    assert not persistent.exists()


def test_second_scan_resumes_exact_committed_map_version(tmp_path: Path) -> None:
    config = _config(tmp_path)
    persistent = tmp_path / "persistent" / "semantic_map.npz"
    first_encoder = CountingDenseEncoder()
    first = build_semantic_embodied_runtime(
        config,
        "scene_000001",
        encoder=first_encoder,
        persistent_map_path=persistent,
    )
    assert first.simulator.scan()["scene_version"] == 1

    second_encoder = CountingDenseEncoder()
    resumed = build_semantic_embodied_runtime(
        config,
        "scene_000001",
        encoder=second_encoder,
        persistent_map_path=persistent,
    )
    assert resumed.simulator.get_robot_state()["scene_version"] == 1
    result = resumed.simulator.scan()
    assert result["success"] is True
    assert result["scene_version"] == 2
    assert second_encoder.calls == 1
