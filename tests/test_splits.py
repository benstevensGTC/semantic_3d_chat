import pytest

from semantic_3d_chat.data.splits import (
    assert_group_disjoint,
    assert_scene_disjoint,
    scene_level_splits,
)


def test_splits_are_deterministic_and_scene_disjoint() -> None:
    scenes = [f"scene_{index:06d}" for index in range(10)]
    first = scene_level_splits(scenes, 11)
    second = scene_level_splits(scenes, 11)
    assert first == second
    assert_scene_disjoint(first)
    assert sorted(scene for members in first.values() for scene in members) == scenes


def test_smoke_scene_stays_in_training() -> None:
    assert scene_level_splits(["scene_000001"], 1) == {
        "train": ["scene_000001"], "validation": [], "test": []
    }


def test_counterfactual_groups_are_deterministic_and_never_cross_splits() -> None:
    scenes = [f"scene_{index:06d}" for index in range(1, 11)]
    groups = {
        "scene_000003": "pair_000001",
        "scene_000004": "pair_000001",
        "scene_000005": "pair_000002",
        "scene_000006": "pair_000002",
        "scene_000007": "pair_000003",
        "scene_000008": "pair_000003",
        "scene_000009": "pair_000004",
        "scene_000010": "pair_000004",
    }

    first = scene_level_splits(scenes, 20260808, groups)
    second = scene_level_splits(scenes, 20260808, groups)

    assert first == second
    assert sorted(scene for members in first.values() for scene in members) == scenes
    assert_scene_disjoint(first)
    assert_group_disjoint(first, groups)
    scene_to_split = {
        scene_id: split for split, members in first.items() for scene_id in members
    }
    for left, right in ((3, 4), (5, 6), (7, 8), (9, 10)):
        assert scene_to_split[f"scene_{left:06d}"] == scene_to_split[f"scene_{right:06d}"]


def test_group_split_rejects_unknown_scene_and_cross_split_assertion() -> None:
    scenes = ["scene_000001", "scene_000002", "scene_000003"]
    with pytest.raises(ValueError, match="unknown scenes"):
        scene_level_splits(scenes, 1, {"scene_999999": "pair_000001"})
    with pytest.raises(ValueError, match="crosses"):
        assert_group_disjoint(
            {"train": ["scene_000001"], "validation": ["scene_000002"]},
            {"scene_000001": "pair_000001", "scene_000002": "pair_000001"},
        )
