"""Contracts for the learned spatial grounding head.

This is the piece that lets language address the 3D field by position, so the
tests protect the two properties that make it meaningful: that its supervision
never touches an oracle, and that it is measured on rooms it did not train on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from semantic_3d_chat.spatial_lens.grounding import (
    SpatialGroundingHead,
    dihedral,
    load_head,
    locate_error_m,
    save_head,
    soft_cross_entropy,
)

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "gemma4" / "metrics" / "spatial_lens_grounding.json"


def test_head_scores_every_cell_and_stays_small() -> None:
    head = SpatialGroundingHead(feature_dim=32, model_dim=64, heads=4, layers=1, grid=8)
    scene = torch.randn(3, 64, 32)
    query = torch.randn(3, 32)
    logits = head(scene, query)
    assert logits.shape == (3, 64)
    assert torch.isfinite(logits).all()
    # The decoder stays frozen; only this reads from it, so it must stay cheap.
    assert sum(p.numel() for p in head.parameters()) < 5_000_000


def test_prediction_depends_on_the_phrase() -> None:
    """A head that ignores the query is not grounding anything."""

    torch.manual_seed(0)
    head = SpatialGroundingHead(feature_dim=32, model_dim=64, heads=4, layers=1, grid=8)
    scene = torch.randn(1, 64, 32)
    first = head(scene, torch.randn(1, 32))
    second = head(scene, torch.randn(1, 32))
    assert not torch.allclose(first, second)


def test_prediction_depends_on_the_scene() -> None:
    torch.manual_seed(0)
    head = SpatialGroundingHead(feature_dim=32, model_dim=64, heads=4, layers=1, grid=8)
    query = torch.randn(1, 32)
    first = head(torch.randn(1, 64, 32), query)
    second = head(torch.randn(1, 64, 32), query)
    assert not torch.allclose(first, second)


def test_dihedral_moves_scene_and_target_together() -> None:
    """Augmentation is only valid if the label follows the room."""

    grid = 4
    scene = np.arange(grid * grid * 2, dtype=np.float32).reshape(grid * grid, 2)
    target = np.zeros(grid * grid, dtype=np.float32)
    target[grid * 1 + 2] = 1.0  # a single marked cell

    for variant in range(8):
        moved_scene, moved_target = dihedral(scene, target, grid, variant)
        assert moved_scene.shape == scene.shape
        assert moved_target.shape == target.shape
        assert moved_target.sum() == pytest.approx(1.0)
        # The marked cell must still sit on the same scene row it started on.
        source = int(target.argmax())
        destination = int(moved_target.argmax())
        assert np.allclose(moved_scene[destination], scene[source])
    # Identity leaves everything alone; some variant must actually move it.
    assert np.array_equal(dihedral(scene, target, grid, 0)[1], target)
    assert any(
        not np.array_equal(dihedral(scene, target, grid, v)[1], target)
        for v in range(1, 8)
    )


def test_soft_cross_entropy_prefers_the_footprint() -> None:
    target = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    good = torch.tensor([[5.0, 5.0, -5.0, -5.0]])
    bad = torch.tensor([[-5.0, -5.0, 5.0, 5.0]])
    assert float(soft_cross_entropy(good, target)) < float(soft_cross_entropy(bad, target))


def test_locate_error_is_zero_when_the_argmax_is_the_target() -> None:
    target = torch.zeros(1, 16)
    target[0, 5] = 1.0
    logits = torch.full((1, 16), -10.0)
    logits[0, 5] = 10.0
    assert locate_error_m(logits, target, 4, (4.0, 4.0, 2.8))[0] == pytest.approx(0.0)


def test_head_round_trips(tmp_path: Path) -> None:
    head = SpatialGroundingHead(feature_dim=16, model_dim=32, heads=4, layers=1, grid=4)
    save_head(tmp_path / "head", head, {
        "feature_dim": 16, "model_dim": 32, "heads": 4, "layers": 1, "grid": 4,
    })
    restored, metadata = load_head(tmp_path / "head")
    assert metadata["grid"] == 4
    scene, query = torch.randn(1, 16, 16), torch.randn(1, 16)
    with torch.no_grad():
        assert torch.allclose(head(scene, query), restored(scene, query))


@pytest.mark.skipif(not METRICS.is_file(), reason="grounding head has not been trained here")
def test_grounding_generalizes_to_unseen_rooms() -> None:
    report = json.loads(METRICS.read_text(encoding="utf-8"))
    # The split must be by room, and the two sets must not overlap.
    train, held = set(report["train_rooms"]), set(report["heldout_rooms"])
    assert train and held and not (train & held)
    # No oracle anywhere in the supervision.
    assert report["oracle_used"] is False
    assert report["supervision"] == "self_supervised_from_perception"
    held_out = report["heldout"]["hits_object_cell"]
    chance = report["random_baseline_hit_rate"]
    assert held_out > 0.45, "grounding must clearly beat chance in unseen rooms"
    assert held_out > 8 * chance


@pytest.mark.skipif(
    not (ROOT / "reports/gemma4/metrics/spatial_lens_method_comparison.json").is_file(),
    reason="method comparison has not been produced here",
)
def test_both_methods_are_compared_on_equal_terms() -> None:
    """The zero-training route must be reported, not quietly dropped."""

    report = json.loads(
        (ROOT / "reports/gemma4/metrics/spatial_lens_method_comparison.json")
        .read_text(encoding="utf-8")
    )
    methods = report["methods"]
    assert set(methods) == {"zero_training_topdown_render", "trained_grounding_head"}
    # Same rooms, same metric, and the untrained method genuinely trains nothing.
    assert methods["zero_training_topdown_render"]["training_parameters"] == 0
    assert methods["trained_grounding_head"]["training_parameters"] > 0
    assert len(report["rooms"]) >= 4
    for value in methods.values():
        assert value["answered"] > 0
        assert 0.0 <= value["lands_on_object"] <= 1.0
    # Both must beat blind guessing, and the honest ordering must be recorded.
    chance = report["random_baseline_lands_on_object"]
    assert methods["zero_training_topdown_render"]["lands_on_object"] > chance
    assert methods["trained_grounding_head"]["lands_on_object"] > chance
    assert report["trained_head_advantage"] > 1.0


@pytest.mark.skipif(not METRICS.is_file(), reason="grounding head has not been trained here")
def test_grounding_lands_on_the_object_in_unseen_rooms() -> None:
    """The navigation-relevant metric, which is what the rover consumes."""

    report = json.loads(METRICS.read_text(encoding="utf-8"))
    held = report["heldout"]
    assert held["median_footprint_gap_m"] <= 0.5
    assert held["footprint_gap_under_0p5m"] > 0.6
    assert report["skipped_nonfinite_steps"] == 0, "training diverged"
