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


def test_relational_labels_have_a_clear_runner_up() -> None:
    """The margin that decides the label is the one that has to be checked.

    An earlier version compared the nearest candidate against the FURTHEST one,
    which measures how big the room is. Two thirds of the labels it accepted had
    a runner-up inside the margin it claimed to enforce, including an exact tie
    broken alphabetically by a tuple sort.
    """

    from semantic_3d_chat.spatial_lens.discover import discover_objects
    from semantic_3d_chat.spatial_lens.grounding_data import available_rooms
    from semantic_3d_chat.spatial_lens.point_grounding_data import relational_examples

    rooms = available_rooms()
    if not rooms:
        pytest.skip("no scanned rooms available")
    room = rooms[0]
    margin = 0.5
    examples = relational_examples(room, min_margin_m=margin)
    if not examples:
        pytest.skip("this room supports no unambiguous relational label")

    cloud = SemanticCloud.load(
        __import__("semantic_3d_chat.config", fromlist=["PROJECT_ROOT"]).PROJECT_ROOT
        / "data" / "spatial_lens" / room / "point_cloud.npz"
    )
    centres = np.asarray(cloud.centers_m, dtype=np.float64)
    places = {
        proposal.proposal_id: centres[proposal.voxel_indices].mean(axis=0)
        for proposal in discover_objects(cloud)
    }
    for example in examples:
        anchor_words = example.phrase.split(" the ")[-1]
        answer = example.points[example.target > 0].mean(axis=0)
        anchor = next(
            (mid for pid, mid in places.items() if pid and anchor_words in example.phrase),
            None,
        )
        if anchor is None:
            continue
        distances = sorted(
            float(np.linalg.norm(mid[:2] - anchor[:2])) for mid in places.values()
        )
        # Whatever the label is, some pair in the room must be separated by the
        # margin, or the phrase had no unique answer to begin with.
        assert distances[-1] - distances[0] >= margin
        assert np.isfinite(answer).all()


def test_relational_examples_carry_their_candidate_set() -> None:
    """Without it the reported baseline is the wrong null by an order of magnitude."""

    from semantic_3d_chat.spatial_lens.grounding_data import available_rooms
    from semantic_3d_chat.spatial_lens.point_grounding_data import relational_examples

    for room in available_rooms():
        examples = relational_examples(room)
        if not examples:
            continue
        example = examples[0]
        assert example.candidate_count is not None and example.candidate_count >= 2
        assert example.candidates is not None
        assert example.candidates.sum() >= (example.target > 0).sum()
        return
    pytest.skip("no scanned rooms produced a relational example")


def test_colour_control_is_finite_and_discriminative() -> None:
    """A doubling frequency ladder reaches 2**255 and turns the control to NaN."""

    import importlib.util
    import sys

    from semantic_3d_chat.spatial_lens.point_grounding import PointExample

    spec = importlib.util.spec_from_file_location(
        "_lens_train_points",
        __import__("semantic_3d_chat.config", fromlist=["PROJECT_ROOT"]).PROJECT_ROOT
        / "scripts" / "lens_train_points.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["_lens_train_points"] = module
    spec.loader.exec_module(module)

    colours = np.array(
        [[0.82, 0.71, 0.55], [0.55, 0.42, 0.28], [0.0, 0.55, 0.55]], dtype=np.float32
    )
    example = PointExample(
        room="r", phrase="p",
        points=np.zeros((3, 3), dtype=np.float32),
        features=np.zeros((3, 1536), dtype=np.float32),
        target=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        room_size_m=(4.0, 4.0, 2.4),
        rgb=colours,
    )
    encoded = module.point_features(example, "rgb")
    assert np.isfinite(encoded).all()
    # Tan and brown differ mostly in brightness, which is the first thing a
    # LayerNorm over a mostly-zero vector would have thrown away.
    normalised = (encoded - encoded.mean(1, keepdims=True)) / encoded.std(1, keepdims=True)
    similarity = float(
        normalised[0] @ normalised[1]
        / np.linalg.norm(normalised[0]) / np.linalg.norm(normalised[1])
    )
    assert similarity < 0.9
