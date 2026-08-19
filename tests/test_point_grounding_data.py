"""What the point-level supervision must guarantee about itself."""

from __future__ import annotations

import numpy as np
import pytest

from semantic_3d_chat.spatial_lens.perceive import SemanticCloud
from semantic_3d_chat.spatial_lens.point_grounding_data import downsample


def _cloud(points: np.ndarray) -> SemanticCloud:
    count = points.shape[0]
    return SemanticCloud(
        centers_m=points.astype(np.float32),
        rgb=np.zeros((count, 3), dtype=np.float32),
        features=np.zeros((count, 4), dtype=np.float16),
        counts=np.ones(count, dtype=np.int32),
        voxel_size_m=0.05,
        room_size_m=(4.0, 4.0, 2.4),
    )


def test_downsample_respects_the_budget() -> None:
    rng = np.random.default_rng(0)
    cloud = _cloud(rng.uniform(-2.0, 2.0, size=(9000, 3)))
    chosen = downsample(cloud, token_budget=512, cell_m=0.1, seed=0)
    assert chosen.size <= 512
    assert np.array_equal(chosen, np.unique(chosen))


def test_downsample_is_deterministic() -> None:
    rng = np.random.default_rng(1)
    cloud = _cloud(rng.uniform(-2.0, 2.0, size=(4000, 3)))
    first = downsample(cloud, token_budget=256, cell_m=0.2, seed=7)
    second = downsample(cloud, token_budget=256, cell_m=0.2, seed=7)
    assert np.array_equal(first, second)


def test_downsample_keeps_a_thin_object_off_a_dense_floor() -> None:
    """Uniform sampling would bury furniture under the floor it stands on."""

    rng = np.random.default_rng(2)
    floor = np.column_stack([
        rng.uniform(-2.0, 2.0, 8000), rng.uniform(-2.0, 2.0, 8000),
        np.zeros(8000),
    ])
    # A small object: few points, but its own patch of space.
    obj = rng.uniform([0.6, 0.6, 0.5], [1.0, 1.0, 1.0], size=(60, 3))
    cloud = _cloud(np.vstack([floor, obj]))
    chosen = downsample(cloud, token_budget=600, cell_m=0.14, seed=0)
    kept = int((chosen >= 8000).sum())
    uniform_share = 600 * 60 / 8060
    assert kept > 0
    assert kept > uniform_share


def test_downsample_returns_everything_when_it_fits() -> None:
    cloud = _cloud(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    chosen = downsample(cloud, token_budget=64, cell_m=0.14, seed=0)
    assert chosen.size == 2


def test_relational_phrases_need_two_distinct_candidates() -> None:
    """A near-tie has no right answer, so it must not become a label."""

    from semantic_3d_chat.spatial_lens.point_grounding import PointExample

    example = PointExample(
        room="r", phrase="the chair nearest the shelf",
        points=np.zeros((4, 3), dtype=np.float32),
        features=np.zeros((4, 2), dtype=np.float32),
        target=np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32),
        room_size_m=(4.0, 4.0, 2.4),
    )
    assert example.target.sum() == pytest.approx(1.0)
    assert example.footprint is None
