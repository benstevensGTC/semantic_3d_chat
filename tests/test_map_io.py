import json
from pathlib import Path

import numpy as np
import pytest

from semantic_3d_chat.scene_encoder.map_io import (
    load_map_tensors,
    validate_runtime_map_sidecars,
)


def _sanitized_map_arrays(
    *,
    metadata: dict[str, object] | None = None,
    header_extra: dict[str, object] | None = None,
    last_frame: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    count = 3
    header: dict[str, object] = {
        "schema_version": 1,
        "voxel_size_m": 0.05,
        "occupied_voxels": count,
        "feature_dim": 8,
        "semantic_dtype_on_disk": "float16",
        "codec": "identity-float16",
        "total_observations": count,
        "max_voxels": 100,
        "metadata": metadata if metadata is not None else {"scene_id": "scene_000001"},
    }
    if header_extra:
        header.update(header_extra)
    return {
        "semantic_features": np.ones((count, 8), dtype=np.float16),
        "centers_world": np.zeros((count, 3), dtype=np.float32),
        "mean_rgb": np.full((count, 3), 127, dtype=np.float32),
        "normal": np.zeros((count, 3), dtype=np.float32),
        "confidence": np.ones(count, dtype=np.float32),
        "observation_count": np.ones(count, dtype=np.int32),
        "last_frame": (
            last_frame
            if last_frame is not None
            else np.asarray([f"f_{index:06d}" for index in range(count)])
        ),
        "metadata_json": np.asarray(json.dumps(header, sort_keys=True)),
    }


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


def test_map_io_rejects_unexpected_label_or_segmentation_payload(tmp_path: Path) -> None:
    path = tmp_path / "map.npz"
    np.savez(
        path,
        semantic_features=np.ones((1, 8), dtype=np.float16),
        centers_world=np.zeros((1, 3), dtype=np.float32),
        mean_rgb=np.zeros((1, 3), dtype=np.float32),
        normal=np.zeros((1, 3), dtype=np.float32),
        confidence=np.ones(1, dtype=np.float32),
        observation_count=np.ones(1, dtype=np.int32),
        object_labels=np.asarray(["chair"]),
    )

    with pytest.raises(ValueError, match="non-allowlisted fields"):
        load_map_tensors(path, [6, 5, 3])


def test_map_io_accepts_exact_sanitized_sidecar_schema(tmp_path: Path) -> None:
    path = tmp_path / "map.npz"
    np.savez(path, **_sanitized_map_arrays())

    data = load_map_tensors(path, [6, 5, 3])

    assert data.voxel_count == 3
    assert data.feature_dim == 8


@pytest.mark.parametrize(
    ("metadata", "header_extra"),
    [
        ({"scene_id": "scene_000001", "category": "chair"}, None),
        ({"scene_id": "scene_000001", "object_name": "table"}, None),
        ({"scene_id": "scene_000001"}, {"category": "chair"}),
    ],
)
def test_map_io_rejects_semantic_strings_hidden_in_metadata_json(
    tmp_path: Path,
    metadata: dict[str, object],
    header_extra: dict[str, object] | None,
) -> None:
    path = tmp_path / "poisoned_map.npz"
    np.savez(
        path,
        **_sanitized_map_arrays(metadata=metadata, header_extra=header_extra),
    )

    with pytest.raises(ValueError, match="fields|semantic"):
        load_map_tensors(path, [6, 5, 3])


def test_map_io_rejects_semantic_last_frame_payload(tmp_path: Path) -> None:
    path = tmp_path / "poisoned_map.npz"
    np.savez(
        path,
        **_sanitized_map_arrays(last_frame=np.asarray(["f_000000", "chair", "f_000002"])),
    )

    with pytest.raises(ValueError, match="non-opaque identifier"):
        load_map_tensors(path, [6, 5, 3])


def test_current_gemma4_runtime_map_sidecars_remain_compatible() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "data_gemma4"
        / "maps"
        / "scene_000001"
        / "voxel_map.npz"
    )
    if not path.is_file():
        pytest.skip("Prepared Gemma 4 runtime map is not available")

    # Sidecar-only validation deliberately avoids materializing the 3,072-D feature matrix.
    validate_runtime_map_sidecars(path)
