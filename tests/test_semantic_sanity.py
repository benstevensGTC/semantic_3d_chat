import json
from pathlib import Path

import numpy as np
import pytest

from semantic_3d_chat.evaluation.semantic_sanity import (
    ALIGNED_DIM,
    ALIGNED_START,
    TOTAL_SEMANTIC_DIM,
    OracleTarget,
    SemanticQuery,
    _reject_oracle_runtime_input,
    compute_multiview_consistency,
    extract_aligned_features,
    score_semantic_queries,
    write_query_heatmaps,
)


def _feature(vector: np.ndarray) -> np.ndarray:
    feature = np.zeros(TOTAL_SEMANTIC_DIM, dtype=np.float32)
    feature[ALIGNED_START:] = vector
    return feature


def _unit(index: int) -> np.ndarray:
    value = np.zeros(ALIGNED_DIM, dtype=np.float32)
    value[index] = 1.0
    return value


def _synthetic_problem() -> tuple[
    np.ndarray, np.ndarray, list[SemanticQuery], np.ndarray, list[OracleTarget]
]:
    centers = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
    features = np.stack(
        [_feature(_unit(0)), _feature(_unit(0)), _feature(_unit(1)), _feature(_unit(1))]
    )
    queries = [
        SemanticQuery("query_000", "alpha", "alpha", "a photo of alpha"),
        SemanticQuery("query_001", "beta", "beta", "a photo of beta"),
    ]
    text = np.stack([_unit(0), _unit(1)])
    targets = [
        OracleTarget("i_000", "alpha", (-0.05, -0.05, -0.05), (0.15, 0.05, 0.05)),
        OracleTarget("i_001", "beta", (0.95, -0.05, -0.05), (1.15, 0.05, 0.05)),
    ]
    return centers, features, queries, text, targets


def test_aligned_slice_and_bbox_localization_are_exact() -> None:
    centers, features, queries, text, targets = _synthetic_problem()
    aligned = extract_aligned_features(features)
    assert aligned.shape == (4, ALIGNED_DIM)
    assert np.allclose(aligned[:2], _unit(0))
    assert np.allclose(aligned[2:], _unit(1))

    metrics, similarities = score_semantic_queries(
        centers, features, queries, text, targets, top_k=1
    )
    assert similarities.shape == (4, 2)
    assert metrics["aggregate"]["top1_localization_accuracy"] == 1.0
    assert metrics["aggregate"]["top_k_localization_accuracy"] == 1.0
    assert metrics["aggregate"]["mean_precision_at_k"] == 1.0
    assert metrics["aggregate"]["top1_accuracy_minus_random"] == 0.5
    assert metrics["aggregate"]["top_k_accuracy_minus_random"] == 0.5
    assert metrics["aggregate"]["positive_region_margin_rate"] == 1.0
    assert metrics["aggregate"]["positive_correct_vs_distractor_margin_rate"] == 1.0
    assert all(result["correct_vs_distractor_margin"] == 1.0 for result in metrics["queries"])


def test_wrong_feature_layout_and_oracle_map_path_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="middle768"):
        extract_aligned_features(np.ones((2, 1536), dtype=np.float32))
    oracle_map = tmp_path / "oracle" / "scene_000001" / "map.npz"
    with pytest.raises(ValueError, match="opaque runtime artifact"):
        _reject_oracle_runtime_input(oracle_map, "Fused map")


def test_heatmaps_use_opaque_query_filenames(tmp_path: Path) -> None:
    centers, features, queries, text, targets = _synthetic_problem()
    _, similarities = score_semantic_queries(centers, features, queries, text, targets, top_k=1)
    artifacts = write_query_heatmaps(centers, similarities, queries, targets, tmp_path)
    assert [Path(item["path"]).name for item in artifacts] == ["query_000.png", "query_001.png"]
    assert all(Path(item["path"]).read_bytes().startswith(b"\x89PNG") for item in artifacts)


def test_multiview_consistency_exceeds_different_voxel_control(tmp_path: Path) -> None:
    runtime = tmp_path / "data" / "rendered" / "scene_000001"
    depth_root = runtime / "depth"
    feature_root = tmp_path / "data" / "features" / "scene_000001"
    depth_root.mkdir(parents=True)
    feature_root.mkdir(parents=True)
    frame_entries = []
    feature_entries = []
    spatial = np.stack([_feature(_unit(0)), _feature(_unit(1))], axis=0)[None, ...]
    for index in range(2):
        frame_id = f"f_{index:06d}"
        depth_path = depth_root / f"{frame_id}.npy"
        np.save(depth_path, np.ones((1, 2), dtype=np.float32), allow_pickle=False)
        feature_path = feature_root / f"{frame_id}.npz"
        np.savez(feature_path, spatial_features=spatial.astype(np.float16))
        frame_entries.append(
            {
                "frame_id": frame_id,
                "rgb_path": f"rgb/{frame_id}.png",
                "depth_path": f"depth/{frame_id}.npy",
                "intrinsics": np.eye(3).tolist(),
                "camera_to_world": np.eye(4).tolist(),
            }
        )
        feature_entries.append({"frame_id": frame_id, "feature_path": feature_path.name})
    render_manifest = runtime / "manifest.json"
    render_manifest.write_text(json.dumps({"frames": frame_entries}), encoding="utf-8")
    (feature_root / "manifest.json").write_text(
        json.dumps({"frames": feature_entries}), encoding="utf-8"
    )

    metrics, same, different = compute_multiview_consistency(
        render_manifest, feature_root, voxel_size_m=0.25, pixel_stride=1
    )
    assert metrics["available"] is True
    assert metrics["frames"] == 2
    assert metrics["multiview_voxels"] == 2
    assert metrics["same_voxel_pair_count"] == 2
    assert np.allclose(same, 1.0)
    assert np.allclose(different, 0.0)
    assert metrics["same_minus_different_mean"] == pytest.approx(1.0)
