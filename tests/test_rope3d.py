"""The properties that make 3D rotary position worth having."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from semantic_3d_chat.language.rope3d_patch import (
    ScenePositions,
    index_units,
    scene_span_from_mask,
)
from semantic_3d_chat.spatial_lens.point_grounding import PointGroundingModel
from semantic_3d_chat.spatial_lens.rope3d import Rope3D, Rope3DBlock


def test_bands_cover_the_head_exactly() -> None:
    for head_dim in (12, 24, 48, 64, 128):
        encoder = Rope3D(head_dim)
        assert sum(encoder.widths) == head_dim
        assert all(width % 2 == 0 for width in encoder.widths)


def test_odd_head_dimension_is_rejected() -> None:
    with pytest.raises(ValueError, match="even"):
        Rope3D(15)


def test_rotation_preserves_length() -> None:
    """A rotary encoding may not change how much of a feature is present."""

    encoder = Rope3D(48)
    values = torch.randn(2, 3, 5, 48)
    positions = torch.randn(2, 5, 3) * 3.0
    rotated = encoder(values, positions)
    assert torch.allclose(values.norm(dim=-1), rotated.norm(dim=-1), atol=1e-4)


def test_score_depends_on_displacement_not_place() -> None:
    """The whole point: the same offset scores the same anywhere in the room."""

    encoder = Rope3D(96)
    query, key = torch.randn(1, 1, 1, 96), torch.randn(1, 1, 1, 96)
    offset = torch.tensor([0.4, -0.7, 0.2])

    def score(origin: torch.Tensor) -> float:
        rotated_q = encoder(query, origin.reshape(1, 1, 3))
        rotated_k = encoder(key, (origin + offset).reshape(1, 1, 3))
        return float((rotated_q * rotated_k).sum())

    here = score(torch.zeros(3))
    there = score(torch.tensor([2.5, -1.5, 0.8]))
    assert here == pytest.approx(there, abs=1e-3)


def test_different_displacements_score_differently() -> None:
    """Translation invariance must not have collapsed into ignoring position."""

    encoder = Rope3D(96)
    query, key = torch.randn(1, 1, 1, 96), torch.randn(1, 1, 1, 96)

    def score(offset: list[float]) -> float:
        rotated_q = encoder(query, torch.zeros(1, 1, 3))
        rotated_k = encoder(key, torch.tensor(offset).reshape(1, 1, 3))
        return float((rotated_q * rotated_k).sum())

    assert abs(score([0.0, 0.0, 0.0]) - score([1.5, 0.0, 0.0])) > 1e-3


def test_each_axis_moves_the_encoding() -> None:
    encoder = Rope3D(96)
    values = torch.randn(1, 1, 1, 96)
    base = encoder(values, torch.zeros(1, 1, 3))
    for axis in range(3):
        moved = torch.zeros(1, 1, 3)
        moved[0, 0, axis] = 1.0
        assert not torch.allclose(base, encoder(values, moved), atol=1e-4)


def test_block_is_permutation_equivariant() -> None:
    """Points are a set: reordering them must reorder the outputs, nothing more."""

    torch.manual_seed(0)
    block = Rope3DBlock(64, 4).eval()
    tokens = torch.randn(1, 12, 64)
    positions = torch.randn(1, 12, 3)
    order = torch.randperm(12)
    with torch.no_grad():
        straight = block(tokens, positions)[:, order]
        permuted = block(tokens[:, order], positions[:, order])
    assert torch.allclose(straight, permuted, atol=1e-4)


def test_model_prediction_is_a_real_position() -> None:
    torch.manual_seed(0)
    model = PointGroundingModel(feature_dim=32, model_dim=64, heads=4, layers=2).eval()
    features = torch.randn(2, 40, 32)
    positions = torch.rand(2, 40, 3) * 4.0
    query = torch.randn(2, 32)
    with torch.no_grad():
        predicted = model.predict_position(features, positions, query)
    assert predicted.shape == (2, 3)
    # A convex combination of the points cannot leave their bounding box.
    assert (predicted >= positions.amin(dim=1) - 1e-4).all()
    assert (predicted <= positions.amax(dim=1) + 1e-4).all()


@pytest.mark.parametrize("mode", ["rope3d", "learned_absolute", "none"])
def test_every_position_mode_runs(mode: str) -> None:
    model = PointGroundingModel(
        feature_dim=16, model_dim=32, heads=4, layers=1, position_mode=mode
    ).eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 9, 16), torch.rand(1, 9, 3), torch.randn(1, 16))
    assert logits.shape == (1, 9)


def test_unknown_position_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="position_mode"):
        PointGroundingModel(position_mode="magic")


def test_position_mode_none_ignores_geometry() -> None:
    """The control has to actually be a control."""

    torch.manual_seed(0)
    model = PointGroundingModel(
        feature_dim=16, model_dim=32, heads=4, layers=1, position_mode="none"
    ).eval()
    features, query = torch.randn(1, 9, 16), torch.randn(1, 16)
    with torch.no_grad():
        first = model(features, torch.rand(1, 9, 3), query)
        second = model(features, torch.rand(1, 9, 3) * 10.0, query)
    assert torch.allclose(first, second, atol=1e-5)


def test_index_units_spans_the_expected_range() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [6.0, 3.0, 2.0]])
    scaled = index_units(positions, span_units=256.0)
    assert float(scaled.min()) == pytest.approx(0.0)
    assert float(scaled.max()) == pytest.approx(256.0, abs=1e-3)


def test_index_units_keeps_axes_to_one_scale() -> None:
    """Anisotropic scaling would distort the room; one scale keeps it rigid."""

    positions = torch.tensor([[0.0, 0.0, 0.0], [8.0, 4.0, 2.0]])
    scaled = index_units(positions, span_units=256.0)
    extent = scaled.max(dim=0).values - scaled.min(dim=0).values
    assert float(extent[0] / extent[1]) == pytest.approx(2.0, abs=1e-3)


def test_scene_span_is_found_from_the_multimodal_mask() -> None:
    mask = torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0]])
    assert scene_span_from_mask(mask) == (2, 6)


def test_scene_span_needs_marked_tokens() -> None:
    with pytest.raises(ValueError, match="no multimodal"):
        scene_span_from_mask(torch.zeros(1, 5, dtype=torch.long))


def test_scene_positions_reject_the_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"\[n, 3\]"):
        ScenePositions(0, torch.zeros(4, 2))


def test_scene_tokens_carry_true_centroids() -> None:
    """A token's place is where its voxels are, not the middle of its cell."""

    from semantic_3d_chat.spatial_lens.perceive import SemanticCloud
    from semantic_3d_chat.spatial_lens.scene_tokens_3d import build_scene_tokens_3d

    rng = np.random.default_rng(0)
    points = rng.uniform([-2.0, -2.0, 0.2], [2.0, 2.0, 2.0], size=(400, 3))
    cloud = SemanticCloud(
        centers_m=points.astype(np.float32),
        rgb=np.zeros((400, 3), dtype=np.float32),
        features=rng.normal(size=(400, 8)).astype(np.float16),
        counts=np.ones(400, dtype=np.int32),
        voxel_size_m=0.05,
        room_size_m=(4.0, 4.0, 2.4),
    )
    tokens = build_scene_tokens_3d(cloud, grid=4)
    assert tokens.centroids_m is not None
    assert tokens.centroids_m.shape == (16, 3)
    occupied = tokens.occupancy.reshape(-1)
    assert (tokens.centroids_m[occupied, 2] > 0.0).all()
