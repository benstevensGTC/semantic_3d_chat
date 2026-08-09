from __future__ import annotations

import copy

import pytest

from semantic_3d_chat.config import load_config
from semantic_3d_chat.data.scene_variants import (
    batch_scene_plans,
    derive_scene_seed,
    oracle_control_facts,
    project_oracle_counterfactual,
    validate_oracle_geometry,
)


def _instance(
    instance_id: str,
    category: str,
    color: str,
    center: tuple[float, float, float],
    bbox_min: tuple[float, float, float],
    bbox_max: tuple[float, float, float],
    support: str,
) -> dict:
    return {
        "instance_id": instance_id,
        "kind": "object",
        "category": category,
        "color": {"name": color, "rgba": [0.1, 0.2, 0.3, 1.0]},
        "pose": {"center_xyz_m": list(center), "rotation_euler_degrees": [0.0, 0.0, 0.0]},
        "bbox": {"min_xyz_m": list(bbox_min), "max_xyz_m": list(bbox_max)},
        "expected_center_xyz_m": list(center),
        "support_surface": support,
        "visible_from_center_scan": True,
    }


def _reference_oracle() -> dict:
    return {
        "schema_version": 1,
        "scene_id": "scene_000003",
        "seed": 20262826,
        "room": {
            "bounds_min_m": [-3.0, -2.5, 0.0],
            "bounds_max_m": [3.0, 2.5, 3.0],
            "size_m": [6.0, 5.0, 3.0],
        },
        "instances": [
            _instance(
                "i_000100",
                "table",
                "wood",
                (0.8, 0.7, 0.405),
                (0.1, 0.2, 0.0),
                (1.5, 1.1, 0.81),
                "i_000001",
            ),
            _instance(
                "i_000101",
                "chair",
                "blue",
                (-1.2, 0.5, 0.63),
                (-1.5, 0.2, 0.0),
                (-0.9, 0.8, 1.26),
                "i_000001",
            ),
            _instance(
                "i_000103",
                "bowl",
                "red",
                (-1.5, -1.1, 0.09),
                (-1.75, -1.35, 0.01),
                (-1.25, -0.85, 0.17),
                "i_000001",
            ),
            _instance(
                "i_000105",
                "cube",
                "red",
                (0.5, 0.6, 0.96),
                (0.36, 0.46, 0.81),
                (0.64, 0.74, 1.11),
                "i_000100",
            ),
            _instance(
                "i_000106",
                "book",
                "green",
                (1.2, 0.5, 0.845),
                (1.0, 0.35, 0.81),
                (1.4, 0.65, 0.88),
                "i_000100",
            ),
        ],
        "relationships": [
            {
                "subject_instance_id": "i_000105",
                "predicate": "on",
                "object_instance_id": "i_000100",
            },
            {
                "subject_instance_id": "i_000100",
                "predicate": "under",
                "object_instance_id": "i_000105",
            },
            {
                "subject_instance_id": "i_000106",
                "predicate": "on",
                "object_instance_id": "i_000100",
            },
            {
                "subject_instance_id": "i_000101",
                "predicate": "left_of",
                "object_instance_id": "i_000100",
            },
            {
                "subject_instance_id": "i_000100",
                "predicate": "right_of",
                "object_instance_id": "i_000101",
            },
        ],
    }


def _plans() -> dict[str, object]:
    config = load_config("configs/experiments/multiscene.yaml")
    return {plan.scene_id: plan for plan in batch_scene_plans(config)}


def test_multiscene_plan_has_ten_scenes_and_four_valid_pairs() -> None:
    config = load_config("configs/experiments/multiscene.yaml")
    plans = batch_scene_plans(config)

    assert len(plans) == 10
    assert [plan.scene_id for plan in plans] == [f"scene_{index:06d}" for index in range(1, 11)]
    assert plans[0].seed == config["seed"]
    assert derive_scene_seed(config["seed"], "scene_000001", seed_stride=1009) == config["seed"]
    pairs: dict[str, list] = {}
    for plan in plans:
        if plan.pair_id:
            pairs.setdefault(plan.pair_id, []).append(plan)
    assert set(pairs) == {"pair_000001", "pair_000002", "pair_000003", "pair_000004"}
    assert all(len(members) == 2 for members in pairs.values())
    assert all(members[0].seed == members[1].seed for members in pairs.values())
    assert plans[1].seed != plans[0].seed


def test_color_swap_pair_changes_colors_but_not_geometry_or_presence() -> None:
    plan = _plans()["scene_000004"]
    before = oracle_control_facts(_reference_oracle())
    after = oracle_control_facts(project_oracle_counterfactual(_reference_oracle(), plan))

    assert after["colors"]["i_000101"] == "red"
    assert after["colors"]["i_000103"] == "blue"
    assert after["colors"]["i_000105"] == "blue"
    assert after["center_x_m"] == before["center_x_m"]
    assert after["present_instance_ids"] == before["present_instance_ids"]
    assert after["cube_support_surface"] == before["cube_support_surface"]
    assert after["relationships"] == before["relationships"]


def test_cube_support_pair_moves_only_cube_from_on_to_under_table() -> None:
    plan = _plans()["scene_000006"]
    reference = _reference_oracle()
    projected = project_oracle_counterfactual(reference, plan)
    before = oracle_control_facts(reference)
    after = oracle_control_facts(projected)
    cube = next(entry for entry in projected["instances"] if entry["instance_id"] == "i_000105")

    assert after["colors"] == before["colors"]
    assert after["center_x_m"] == before["center_x_m"]
    assert after["present_instance_ids"] == before["present_instance_ids"]
    assert before["cube_support_surface"] == "i_000100"
    assert after["cube_support_surface"] == "i_000001"
    assert cube["expected_center_xyz_m"][2] == 0.15
    assert ("i_000105", "under", "i_000100") in after["relationships"]
    assert ("i_000105", "on", "i_000100") not in after["relationships"]


def test_mirror_pair_negates_x_and_reverses_left_right_only() -> None:
    plan = _plans()["scene_000008"]
    before = oracle_control_facts(_reference_oracle())
    after = oracle_control_facts(project_oracle_counterfactual(_reference_oracle(), plan))

    assert after["colors"] == before["colors"]
    assert after["present_instance_ids"] == before["present_instance_ids"]
    assert after["cube_support_surface"] == before["cube_support_surface"]
    for instance_id, x_coordinate in before["center_x_m"].items():
        assert after["center_x_m"][instance_id] == -x_coordinate
    assert ("i_000101", "right_of", "i_000100") in after["relationships"]
    assert ("i_000101", "left_of", "i_000100") not in after["relationships"]


def test_object_removal_pair_removes_book_and_only_its_relationships() -> None:
    plan = _plans()["scene_000010"]
    before = oracle_control_facts(_reference_oracle())
    after = oracle_control_facts(project_oracle_counterfactual(_reference_oracle(), plan))

    assert set(before["present_instance_ids"]) - set(after["present_instance_ids"]) == {
        "i_000106"
    }
    assert after["colors"] == {
        key: value for key, value in before["colors"].items() if key != "i_000106"
    }
    assert after["center_x_m"] == {
        key: value for key, value in before["center_x_m"].items() if key != "i_000106"
    }
    assert all("i_000106" not in relationship for relationship in after["relationships"])


def test_pragmatic_geometry_validation_accepts_room_and_rejects_failures() -> None:
    oracle = _reference_oracle()
    result = validate_oracle_geometry(
        oracle,
        camera_position_m=(0.0, 0.0, 1.4),
        pitch_degrees=(-25.0, 0.0, 25.0),
        horizontal_fov_degrees=72.0,
        image_size=(224, 224),
    )
    assert result["inside_room"] is True
    assert result["nonintersection"] is True
    assert result["center_scan_angular_coverage"] is True

    outside = copy.deepcopy(oracle)
    outside["instances"][1]["bbox"]["min_xyz_m"][0] = -3.5
    with pytest.raises(ValueError, match="room bound"):
        validate_oracle_geometry(
            outside,
            camera_position_m=(0.0, 0.0, 1.4),
            pitch_degrees=(-25.0, 0.0, 25.0),
            horizontal_fov_degrees=72.0,
            image_size=(224, 224),
        )

    intersecting = copy.deepcopy(oracle)
    intersecting["instances"][2]["bbox"] = copy.deepcopy(
        intersecting["instances"][1]["bbox"]
    )
    with pytest.raises(ValueError, match="intersect"):
        validate_oracle_geometry(
            intersecting,
            camera_position_m=(0.0, 0.0, 1.4),
            pitch_degrees=(-25.0, 0.0, 25.0),
            horizontal_fov_degrees=72.0,
            image_size=(224, 224),
        )
