from pathlib import Path

import numpy as np
import pytest

from semantic_3d_chat.mapping.fusion import fuse_frame, sample_spatial_field
from semantic_3d_chat.mapping.semantic_codec import IdentitySemanticCodec
from semantic_3d_chat.mapping.voxel_map import (
    SparseVoxelMap,
    persisted_voxel_map_content_hash,
    voxel_coordinates,
)


def test_negative_world_coordinates_use_floor_voxel_assignment() -> None:
    points = np.array([[-0.001, 0.0, 0.049], [0.0, -0.051, 0.05]])
    assert np.array_equal(voxel_coordinates(points, 0.05), [[-1, 0, 0], [0, -2, 1]])


def test_weighted_fusion_across_calls_preserves_counts_means_and_variance() -> None:
    voxel_map = SparseVoxelMap(0.05, feature_dim=2)
    new_count = voxel_map.add_observations(
        np.array([[0.001, 0.002, 0.003]]),
        np.array([[1.0, 0.0]]),
        rgb=np.array([[100, 0, 0]]),
        confidence=np.array([0.2]),
        weights=np.array([1.0]),
        view_directions_world=np.array([[0, 1, 0]]),
        frame_id="frame_000000",
    )
    assert new_count == 1
    new_count = voxel_map.add_observations(
        np.array([[0.049, 0.048, 0.047]]),
        np.array([[0.0, 1.0]]),
        rgb=np.array([[0, 100, 0]]),
        confidence=np.array([0.8]),
        weights=np.array([3.0]),
        view_directions_world=np.array([[0, 1, 0]]),
        frame_id="frame_000001",
    )
    assert new_count == 0

    arrays = voxel_map.to_arrays()
    assert len(voxel_map) == 1
    assert np.array_equal(arrays["voxel_coordinates"], [[0, 0, 0]])
    assert np.allclose(arrays["centers_world"], [[0.025, 0.025, 0.025]])
    assert arrays["observation_count"].item() == 2
    assert arrays["weight_sum"].item() == pytest.approx(4.0)
    assert np.allclose(arrays["mean_rgb"], [[25.0, 75.0, 0.0]])
    assert arrays["semantic_features"].dtype == np.float16
    assert np.allclose(arrays["semantic_features"], [[0.25, 0.75]])
    # Weighted per-component variance: 1.5 total M2 / 4 weight / 2 dimensions.
    assert arrays["semantic_variance"].item() == pytest.approx(0.1875)
    assert arrays["confidence"].item() == pytest.approx(0.65)
    assert np.allclose(arrays["view_direction"], [[0, 1, 0]])
    assert arrays["last_frame"].item() == "frame_000001"


def test_map_save_load_and_previews_are_safe_and_reproducible(tmp_path: Path) -> None:
    voxel_map = SparseVoxelMap(0.1, codec=IdentitySemanticCodec(np.float16))
    voxel_map.add_observations(
        np.array([[0.01, 0.01, 0.01], [0.11, 0.01, 0.01]]),
        np.array([[1.0, 0.25, -0.5], [0.5, 1.0, 0.25]]),
        rgb=np.array([[255, 0, 0], [0, 0, 255]]),
        normals_world=np.array([[0, 0, 1], [0, 0, 1]]),
        frame_id="frame_000000",
    )
    expected_hash = voxel_map.content_hash()
    map_path = voxel_map.save(tmp_path / "map.npz", metadata={"scene_id": "scene_000001"})
    assert persisted_voxel_map_content_hash(map_path) == expected_hash
    loaded = SparseVoxelMap.load(map_path)
    assert loaded.content_hash() == expected_hash
    assert np.allclose(
        loaded.to_arrays()["semantic_features"], voxel_map.to_arrays()["semantic_features"]
    )

    ply_path = loaded.write_ply(tmp_path / "map_rgb.ply")
    png_path = loaded.write_png(tmp_path / "map_uncertainty.png", color_by="uncertainty")
    assert ply_path.read_bytes().startswith(b"ply\nformat binary_little_endian 1.0")
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_max_voxels_fails_before_mutating_map() -> None:
    voxel_map = SparseVoxelMap(0.1, max_voxels=1)
    with pytest.raises(MemoryError):
        voxel_map.add_observations(
            np.array([[0.01, 0, 0], [0.11, 0, 0]]),
            np.array([[1.0, 0.1], [0.1, 1.0]]),
        )
    assert len(voxel_map) == 0


def test_patch_grid_sampling_and_frame_fusion() -> None:
    patch_grid = np.array(
        [
            [[1.0, 0.1], [2.0, 0.1]],
            [[3.0, 0.1], [4.0, 0.1]],
        ],
        dtype=np.float32,
    )
    pixels = np.array([[0, 0], [3, 3], [1, 1]], dtype=np.int32)
    sampled = sample_spatial_field(patch_grid, pixels, image_shape=(4, 4))
    assert np.allclose(sampled[:2], [[1.0, 0.1], [4.0, 0.1]])
    assert sampled[2, 0] == pytest.approx(1.75)

    depth = np.ones((4, 4), dtype=np.float32)
    rgb = np.full((4, 4, 3), 127, dtype=np.uint8)
    intrinsics = np.array([[2.0, 0, 1.5], [0, 2.0, 1.5], [0, 0, 1.0]])
    voxel_map = SparseVoxelMap(0.25)
    stats = fuse_frame(
        voxel_map,
        depth_m=depth,
        rgb=rgb,
        spatial_features=patch_grid,
        intrinsics=intrinsics,
        camera_to_world=np.eye(4),
        frame_id="frame_000000",
        pixel_stride=2,
    )
    assert stats.valid_depth_points == 4
    assert stats.feature_dim == 2
    assert stats.total_occupied_voxels == 4
