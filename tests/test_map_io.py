from pathlib import Path

import numpy as np

from semantic_3d_chat.scene_encoder.map_io import load_map_tensors


def test_map_io_loads_only_numeric_fields(tmp_path: Path) -> None:
    path = tmp_path / "map.npz"
    np.savez(
        path,
        semantic_features=np.ones((3, 8), dtype=np.float16),
        centers_world=np.zeros((3, 3), dtype=np.float32),
        mean_rgb=np.full((3, 3), 127, dtype=np.float32),
        normal=np.zeros((3, 3), dtype=np.float32),
        confidence=np.ones(3, dtype=np.float32),
        observation_count=np.ones(3, dtype=np.int32),
    )
    data = load_map_tensors(path, [6, 5, 3])
    assert data.feature_dim == 8
    assert data.voxel_count == 3
    assert data.room_min.tolist() == [-3.0, -2.5, 0.0]


def test_map_io_coarsening_preserves_every_source_observation(tmp_path: Path) -> None:
    path = tmp_path / "map.npz"
    np.savez(
        path,
        semantic_features=np.array([[1, 0], [3, 0], [0, 2]], dtype=np.float16),
        centers_world=np.array([[0.01, 0, 0], [0.04, 0, 0], [0.21, 0, 0]], dtype=np.float32),
        mean_rgb=np.zeros((3, 3), dtype=np.float32),
        normal=np.zeros((3, 3), dtype=np.float32),
        confidence=np.ones(3, dtype=np.float32),
        observation_count=np.array([1, 3, 2], dtype=np.int32),
    )
    data = load_map_tensors(path, [6, 5, 3], input_voxel_size_m=0.1)
    assert data.source_voxel_count == 3
    assert data.voxel_count == 2
    assert data.observation_count.sum().item() == 6
    assert np.allclose(data.semantic[0].numpy(), [2.5, 0.0])
