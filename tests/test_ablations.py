import json
from pathlib import Path

import numpy as np
import pytest

from semantic_3d_chat.evaluation.ablations import (
    ABLATION_MODES,
    apply_ablation,
    create_ablation_map,
    deterministic_permutation,
    load_safe_map_arrays,
)
from semantic_3d_chat.scene_encoder.map_io import load_map_tensors


def _map_arrays() -> dict[str, np.ndarray]:
    count = 4
    return {
        "voxel_coordinates": np.arange(count * 3, dtype=np.int32).reshape(count, 3),
        "centers_world": np.arange(count * 3, dtype=np.float32).reshape(count, 3) / 10,
        "observation_count": np.arange(1, count + 1, dtype=np.int32),
        "weight_sum": np.arange(1, count + 1, dtype=np.float32),
        "mean_rgb": np.arange(count * 3, dtype=np.float32).reshape(count, 3),
        "semantic_features": np.arange(1, 1 + count * 5, dtype=np.float16).reshape(count, 5),
        "semantic_feature_m2": np.arange(count, dtype=np.float64),
        "semantic_variance": np.arange(count, dtype=np.float32) / 10,
        "normal": np.eye(4, 3, dtype=np.float32),
        "normal_valid": np.ones(count, dtype=bool),
        "view_direction": np.flip(np.eye(4, 3, dtype=np.float32), axis=1),
        "view_direction_valid": np.ones(count, dtype=bool),
        "confidence": np.ones(count, dtype=np.float32),
        "last_frame": np.asarray([f"f_{index:06d}" for index in range(count)]),
        "metadata_json": np.asarray(json.dumps({"schema_version": 1})),
    }


def test_deterministic_shuffles_change_only_declared_bundles() -> None:
    source = _map_arrays()
    permutation = deterministic_permutation(4, 17)
    assert np.array_equal(permutation, deterministic_permutation(4, 17))
    assert not np.array_equal(permutation, np.arange(4))

    semantic, semantic_metadata = apply_ablation(source, "semantic_shuffle", seed=17)
    assert np.array_equal(semantic["semantic_features"], source["semantic_features"][permutation])
    assert np.array_equal(semantic["semantic_variance"], source["semantic_variance"][permutation])
    assert np.array_equal(semantic["centers_world"], source["centers_world"])
    assert semantic_metadata["permutation_sha256"] is not None

    geometry, _ = apply_ablation(source, "geometry_shuffle", seed=17)
    assert np.array_equal(geometry["centers_world"], source["centers_world"][permutation])
    assert np.array_equal(geometry["normal"], source["normal"][permutation])
    assert np.array_equal(geometry["semantic_features"], source["semantic_features"])

    xyz, _ = apply_ablation(source, "xyz_shuffle", seed=17)
    assert np.array_equal(xyz["centers_world"], source["centers_world"][permutation])
    assert np.array_equal(xyz["normal"], source["normal"])


@pytest.mark.parametrize(
    ("mode", "zero_fields"),
    [
        ("zero_semantics", ("semantic_features", "semantic_variance")),
        ("zero_rgb", ("mean_rgb",)),
        ("zero_normals", ("normal", "normal_valid")),
        ("zero_xyz", ("centers_world", "voxel_coordinates")),
    ],
)
def test_zero_ablations_preserve_keys_shapes_and_dtypes(
    mode: str, zero_fields: tuple[str, ...]
) -> None:
    source = _map_arrays()
    ablated, metadata = apply_ablation(source, mode, seed=5)
    assert set(ablated) == set(source)
    for name, original in source.items():
        assert ablated[name].shape == original.shape
        assert ablated[name].dtype == original.dtype
    assert all(not np.any(ablated[name]) for name in zero_fields)
    assert set(zero_fields) <= set(metadata["affected_fields"])


def test_ablation_npz_is_pickle_free_and_loadable_by_scene_encoder(tmp_path: Path) -> None:
    source = tmp_path / "map.npz"
    np.savez_compressed(source, **_map_arrays())
    output = tmp_path / "ablations" / "map_zero_semantics.npz"
    result = create_ablation_map(source, output, "zero_semantics", seed=9)
    arrays = load_safe_map_arrays(output)
    with np.load(output, allow_pickle=False) as archive:
        assert all(not archive[name].dtype.hasobject for name in archive.files)
        metadata = json.loads(str(archive["metadata_json"].item()))
    assert metadata["ablation"]["mode"] == "zero_semantics"
    assert metadata["ablation"]["source_sha256"] == result["source_sha256"]
    assert not np.any(arrays["semantic_features"])
    loaded = load_map_tensors(output, [6.0, 5.0, 3.0])
    assert loaded.voxel_count == 4
    assert not bool(loaded.semantic.any())
    with pytest.raises(FileExistsError):
        create_ablation_map(source, output, "zero_semantics", seed=9)


def test_all_documented_modes_are_implemented() -> None:
    source = _map_arrays()
    for mode in ABLATION_MODES:
        ablated, metadata = apply_ablation(source, mode, seed=20260808)
        assert metadata["mode"] == mode
        assert set(ablated) == set(source)
