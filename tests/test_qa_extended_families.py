from __future__ import annotations

import copy

import pytest

from semantic_3d_chat.data.qa_generator import generate_scene_questions


def _object(
    instance_id: str,
    category: str,
    color: str,
    center: tuple[float, float, float],
) -> dict:
    return {
        "instance_id": instance_id,
        "kind": "object",
        "category": category,
        "color": {"name": color},
        "expected_center_xyz_m": list(center),
        "support_surface": None,
    }


def _oracle() -> dict:
    return {
        "scene_id": "scene_000901",
        "room": {
            "bounds_min_m": [-3.0, -2.5, 0.0],
            "bounds_max_m": [3.0, 2.5, 3.0],
        },
        "scan_origin_xyz_m": [0.0, 0.0, 1.4],
        "instances": [
            _object("i_000100", "table", "blue", (1.8, 2.0, 0.4)),
            _object("i_000101", "chair", "green", (-1.8, -1.5, 0.6)),
            _object("i_000102", "bowl", "red", (0.0, 1.0, 0.2)),
            _object("i_000103", "book", "yellow", (0.1, 1.0, 0.3)),
            _object("i_000104", "cube", "red", (2.0, -2.0, 0.2)),
            _object("i_000105", "floor lamp", "yellow", (0.0, 0.25, 1.2)),
        ],
        "relationships": [
            {
                "subject_instance_id": "i_000103",
                "predicate": "inside",
                "object_instance_id": "i_000102",
            }
        ],
    }


def _visibility(oracle: dict) -> dict:
    object_ids = [item["instance_id"] for item in oracle["instances"]]
    counts = {instance_id: 20 for instance_id in object_ids}
    counts["i_000104"] = 0
    return {
        "schema_version": 1,
        "scene_id": oracle["scene_id"],
        "method": "exact_depth_raycast",
        "minimum_visible_pixels": 2,
        "expected_instance_ids": object_ids,
        "visible_pixel_counts": counts,
        "all_required_visible": False,
    }


def _by_question(records: list[dict]) -> dict[str, dict]:
    return {record["question"]: record for record in records}


def test_extended_qa_families_have_canonical_targets_and_references() -> None:
    oracle = _oracle()
    evidence = _visibility(oracle)
    records = generate_scene_questions(oracle, seed=41, visibility_evidence=evidence)
    questions = _by_question(records)

    assert records == generate_scene_questions(oracle, seed=41, visibility_evidence=evidence)
    assert len({record["question_id"] for record in records}) == len(records)
    assert {
        "object_location",
        "containment",
        "viewpoint_relative",
        "metric",
        "uncertainty",
    } <= {record["answer_type"] for record in records}

    table_location = questions["Where is the table relative to the room center?"]
    assert table_location["answer"] == "right"
    assert table_location["target_instance"] == "i_000100"
    assert table_location["target_xyz"] == [1.8, 2.0, 0.4]
    assert table_location["reference_xyz"] == [0.0, 0.0, 1.5]
    assert questions["Where is the chair relative to the room center?"]["answer"] == "left"
    assert questions["Where is the bowl relative to the room center?"]["answer"] == "center"

    contained = questions["What is inside the bowl?"]
    assert contained["answer"] == "book"
    assert contained["answer_items"] == ["book"]
    assert contained["target_instance"] == "i_000103"
    assert contained["reference_instance"] == "i_000102"
    assert contained["reference_xyz"] == [0.0, 1.0, 0.2]
    assert questions["Is the book inside the bowl?"]["answer"] == "yes"

    table_horizontal = questions[
        "Is the table to the left or right of the current viewpoint?"
    ]
    assert table_horizontal["answer"] == "right"
    assert table_horizontal["viewpoint_yaw_degrees"] == 0.0
    assert table_horizontal["viewpoint_convention"] == "x_right_y_forward_z_up_yaw_0"
    assert table_horizontal["reference_xyz"] == [0.0, 0.0, 1.4]
    assert questions["Is the chair in front of or behind the camera?"]["answer"] == "behind"
    assert questions["Which way should the camera turn to face the table?"]["answer"] == "right"
    assert (
        questions["Which way should the camera turn to face the floor lamp?"]["answer"]
        == "straight ahead"
    )

    distance = questions["Approximately how far is the table from the camera?"]
    assert distance["answer"] == "3.0 meters"
    assert distance["approximate_distance_m"] == 3.0
    assert distance["tolerance_m"] == 0.25
    assert distance["target_instance"] == "i_000100"
    comparison = questions[
        "Which is farther from the camera, the chair or the table?"
    ]
    assert comparison["answer"] == "table"
    assert comparison["target_instance"] == "i_000100"
    assert comparison["reference_instance"] == "i_000101"
    assert questions["Which object is closest to the camera?"]["answer"] == "floor lamp"
    assert "Which is farther from the camera, the book or the bowl?" not in questions

    hidden = questions["Is there enough visual evidence to locate the cube?"]
    assert hidden["answer"] == "no"
    assert hidden["target_instance"] == "i_000104"
    assert hidden["target_xyz"] is None
    assert hidden["visibility_pixels"] == 0
    assert hidden["evidence_sufficient"] is False
    visible = questions["Is there enough visual evidence to locate the table?"]
    assert visible["answer"] == "yes"
    assert visible["target_xyz"] == [1.8, 2.0, 0.4]

    extended_types = {
        "object_location",
        "containment",
        "viewpoint_relative",
        "metric",
    }
    assert not any(
        "cube" in record["question"] and record["answer_type"] in extended_types
        for record in records
    )
    assert [
        record["question"]
        for record in records
        if "cube" in record["question"] and record["answer_type"] != "uncertainty"
    ] == []


def test_extended_questions_skip_duplicate_categories_and_ambiguous_boundaries() -> None:
    oracle = _oracle()
    oracle["instances"].append(_object("i_000106", "table", "red", (-2.2, 1.8, 0.4)))
    evidence = _visibility(oracle)
    evidence["visible_pixel_counts"]["i_000106"] = 15
    records = generate_scene_questions(oracle, seed=2, visibility_evidence=evidence)

    extended_types = {
        "object_location",
        "containment",
        "viewpoint_relative",
        "metric",
        "uncertainty",
    }
    extended = [record for record in records if record["answer_type"] in extended_types]
    assert not any("table" in record["question"] for record in extended)

    boundary_oracle = _oracle()
    boundary_oracle["instances"][0]["expected_center_xyz_m"][0] = 1.0
    questions = _by_question(generate_scene_questions(boundary_oracle, seed=3))
    assert "Where is the table relative to the room center?" not in questions
    assert not any(record["answer_type"] == "uncertainty" for record in questions.values())


def test_visibility_evidence_must_truthfully_cover_the_oracle() -> None:
    oracle = _oracle()
    evidence = _visibility(oracle)
    evidence["all_required_visible"] = True
    with pytest.raises(ValueError, match="disagrees with measured counts"):
        generate_scene_questions(oracle, seed=1, visibility_evidence=evidence)

    incomplete = _visibility(oracle)
    incomplete["expected_instance_ids"].pop()
    incomplete["visible_pixel_counts"].pop("i_000105")
    with pytest.raises(ValueError, match="cover every oracle object exactly"):
        generate_scene_questions(oracle, seed=1, visibility_evidence=incomplete)

    wrong_scene = copy.deepcopy(_visibility(oracle))
    wrong_scene["scene_id"] = "scene_000902"
    with pytest.raises(ValueError, match="scene_id does not match"):
        generate_scene_questions(oracle, seed=1, visibility_evidence=wrong_scene)
