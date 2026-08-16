from __future__ import annotations

import pytest

from semantic_3d_chat.data.qa_generator import (
    annotate_counterfactual_questions,
    counterfactual_scene_groups,
    generate_scene_questions,
    select_balanced_split_records,
)


def _oracle(
    scene_id: str,
    paired_scene_id: str,
    role: str,
    *,
    include_book: bool,
) -> dict:
    instances = []
    if include_book:
        instances.append(
            {
                "instance_id": "i_000106",
                "kind": "object",
                "category": "book",
                "color": {"name": "green"},
                "expected_center_xyz_m": [1.0, 0.5, 0.85],
                "support_surface": None,
            }
        )
    return {
        "scene_id": scene_id,
        "instances": instances,
        "relationships": [],
        "scan_origin_xyz_m": [0.0, 0.0, 1.4],
        "generation": {
            "counterfactual_pair": {
                "pair_id": "pair_000004",
                "paired_scene_id": paired_scene_id,
                "change_type": "object_removal",
                "role": role,
            }
        },
    }


def test_paired_questions_have_stable_keys_and_expected_change_flags() -> None:
    oracles = {
        "scene_000009": _oracle(
            "scene_000009", "scene_000010", "reference", include_book=True
        ),
        "scene_000010": _oracle(
            "scene_000010", "scene_000009", "counterfactual", include_book=False
        ),
    }
    assert counterfactual_scene_groups(oracles) == {
        "scene_000009": "pair_000004",
        "scene_000010": "pair_000004",
    }
    records = {
        scene_id: generate_scene_questions(
            oracle,
            10 + index,
            category_universe={"book"},
        )
        for index, (scene_id, oracle) in enumerate(oracles.items())
    }

    annotated = annotate_counterfactual_questions(records, oracles)

    assert annotated > 0 and annotated % 2 == 0
    by_scene_and_question = {
        (scene_id, record["question"]): record
        for scene_id, scene_records in records.items()
        for record in scene_records
    }
    for question in (
        "Is there a book in the room?",
        "Can you find a book?",
        "How many books are present?",
    ):
        reference = by_scene_and_question[("scene_000009", question)]
        counterfactual = by_scene_and_question[("scene_000010", question)]
        assert reference["counterfactual_question_key"] == counterfactual[
            "counterfactual_question_key"
        ]
        assert reference["counterfactual_expected_change"] is True
        assert counterfactual["counterfactual_expected_change"] is True
        assert reference["counterfactual_role"] == "reference"
        assert counterfactual["counterfactual_role"] == "counterfactual"

    reference_sofa = by_scene_and_question[
        ("scene_000009", "Is there a sofa in the room?")
    ]
    counterfactual_sofa = by_scene_and_question[
        ("scene_000010", "Is there a sofa in the room?")
    ]
    assert reference_sofa["counterfactual_question_key"] == counterfactual_sofa[
        "counterfactual_question_key"
    ]
    assert reference_sofa["counterfactual_expected_change"] is False
    assert counterfactual_sofa["counterfactual_expected_change"] is False


def test_only_questions_present_in_both_scenes_receive_pair_metadata() -> None:
    oracles = {
        "scene_000009": _oracle(
            "scene_000009", "scene_000010", "reference", include_book=True
        ),
        "scene_000010": _oracle(
            "scene_000010", "scene_000009", "counterfactual", include_book=False
        ),
    }
    records = {
        scene_id: generate_scene_questions(oracle, 1)
        for scene_id, oracle in oracles.items()
    }
    annotate_counterfactual_questions(records, oracles)

    color_question = next(
        record
        for record in records["scene_000009"]
        if record["question"] == "What color is the book?"
    )
    assert "counterfactual_pair_id" not in color_question


def test_balanced_selection_keeps_changed_pair_units_atomic() -> None:
    scene_ids = ["scene_000011", "scene_000012"]
    records: dict[str, list[dict]] = {scene_id: [] for scene_id in scene_ids}
    for side, scene_id in enumerate(scene_ids):
        for unit in range(6):
            records[scene_id].append(
                {
                    "scene_id": scene_id,
                    "question_id": f"q_{unit:06d}",
                    "question": f"Shared changed question {unit}?",
                    "answer": "left" if side == 0 else "right",
                    "answer_type": "spatial_relation",
                    "counterfactual_pair_id": "pair_000005",
                    "counterfactual_question_key": f"cfq_{unit:06d}",
                    "counterfactual_expected_change": True,
                }
            )
        for unit, (answer_type, answer) in enumerate(
            [
                ("presence", "yes"),
                ("presence", "no"),
                ("attribute", "red"),
                ("attribute", "blue"),
                ("count", "1"),
                ("support", "on"),
            ],
            start=6,
        ):
            records[scene_id].append(
                {
                    "scene_id": scene_id,
                    "question_id": f"q_{unit:06d}",
                    "question": f"Filler question {unit}?",
                    "answer": answer,
                    "answer_type": answer_type,
                }
            )

    selected = select_balanced_split_records(
        records,
        scene_ids,
        per_scene_limit=6,
        seed=42,
        max_changed_units_per_pair=2,
    )

    assert all(len(scene_records) == 6 for scene_records in selected.values())
    changed_keys = {
        scene_id: {
            record["counterfactual_question_key"]
            for record in scene_records
            if record.get("counterfactual_expected_change") is True
        }
        for scene_id, scene_records in selected.items()
    }
    assert changed_keys[scene_ids[0]] == changed_keys[scene_ids[1]]
    assert len(changed_keys[scene_ids[0]]) == 2
    assert all(
        len({record["answer_type"] for record in scene_records}) >= 4
        for scene_records in selected.values()
    )


def test_balanced_selection_keeps_every_labeled_pair_unit_atomic() -> None:
    scene_ids = ["scene_000011", "scene_000012"]
    records: dict[str, list[dict]] = {scene_id: [] for scene_id in scene_ids}
    for side, scene_id in enumerate(scene_ids):
        records[scene_id].append(
            {
                "scene_id": scene_id,
                "question_id": "q_000000",
                "question": "Changed question?",
                "answer": "left" if side == 0 else "right",
                "answer_type": "spatial_relation",
                "counterfactual_pair_id": "pair_000005",
                "counterfactual_question_key": "cfq_changed",
                "counterfactual_expected_change": True,
            }
        )
        for unit in range(1, 3):
            records[scene_id].append(
                {
                    "scene_id": scene_id,
                    "question_id": f"q_{unit:06d}",
                    "question": f"Stable question {unit}?",
                    "answer": "yes",
                    "answer_type": "presence",
                    "counterfactual_pair_id": "pair_000005",
                    "counterfactual_question_key": f"cfq_stable_{unit}",
                    "counterfactual_expected_change": False,
                }
            )
        for unit in range(3, 6):
            records[scene_id].append(
                {
                    "scene_id": scene_id,
                    "question_id": f"q_{unit:06d}",
                    "question": f"Unpaired question {unit}?",
                    "answer": str(unit),
                    "answer_type": "count",
                }
            )

    selected = select_balanced_split_records(
        records,
        scene_ids,
        per_scene_limit=5,
        seed=42,
        max_changed_units_per_pair=1,
    )

    assert all(len(scene_records) == 5 for scene_records in selected.values())
    paired_keys = {
        scene_id: {
            record["counterfactual_question_key"]
            for record in scene_records
            if "counterfactual_question_key" in record
        }
        for scene_id, scene_records in selected.items()
    }
    assert paired_keys[scene_ids[0]] == paired_keys[scene_ids[1]] == {
        "cfq_changed",
        "cfq_stable_1",
        "cfq_stable_2",
    }
    assert all(
        sum("counterfactual_question_key" not in record for record in scene_records) == 2
        for scene_records in selected.values()
    )


def test_balanced_selection_fails_closed_for_incomplete_labeled_pair() -> None:
    records = {
        "scene_000011": [
            {
                "scene_id": "scene_000011",
                "question_id": "q_000000",
                "question": "Stable question?",
                "answer": "yes",
                "answer_type": "presence",
                "counterfactual_pair_id": "pair_000005",
                "counterfactual_question_key": "cfq_incomplete",
                "counterfactual_expected_change": False,
            }
        ],
        "scene_000012": [],
    }

    with pytest.raises(ValueError, match="must have two sides"):
        select_balanced_split_records(
            records,
            ["scene_000011", "scene_000012"],
            per_scene_limit=1,
            seed=42,
            max_changed_units_per_pair=1,
        )


def test_generated_questions_cover_orientation_and_targeted_support() -> None:
    oracle = {
        "scene_id": "scene_000011",
        "instances": [
            {
                "instance_id": "i_000001",
                "kind": "surface",
                "category": "floor",
                "color": {"name": "neutral"},
                "expected_center_xyz_m": [0.0, 0.0, 0.0],
                "support_surface": None,
            },
            {
                "instance_id": "i_000004",
                "kind": "surface",
                "category": "wall",
                "color": {"name": "neutral"},
                "expected_center_xyz_m": [0.0, 2.5, 1.5],
                "support_surface": None,
            },
            {
                "instance_id": "i_000100",
                "kind": "object",
                "category": "table",
                "color": {"name": "wood"},
                "expected_center_xyz_m": [0.8, 0.7, 0.4],
                "support_surface": "i_000001",
            },
            {
                "instance_id": "i_000101",
                "kind": "object",
                "category": "chair",
                "color": {"name": "blue"},
                "pose": {"rotation_euler_degrees": [180.0, 0.0, 0.0]},
                "expected_center_xyz_m": [-1.2, 0.5, 0.6],
                "support_surface": "i_000001",
            },
            {
                "instance_id": "i_000102",
                "kind": "object",
                "category": "picture frame",
                "color": {"name": "yellow"},
                "expected_center_xyz_m": [-0.5, 2.4, 1.7],
                "support_surface": "i_000004",
            },
            {
                "instance_id": "i_000106",
                "kind": "object",
                "category": "book",
                "color": {"name": "green"},
                "expected_center_xyz_m": [0.8, 0.7, 0.04],
                "support_surface": "i_000001",
            },
        ],
        "relationships": [
            {
                "subject_instance_id": "i_000106",
                "predicate": "under",
                "object_instance_id": "i_000100",
            }
        ],
        "scan_origin_xyz_m": [0.0, 0.0, 1.4],
    }

    by_question = {
        record["question"]: record for record in generate_scene_questions(oracle, seed=7)
    }

    assert by_question["Is the chair upright or upside down?"]["answer"] == "upside down"
    assert by_question["Is the picture frame on the wall or on the floor?"]["answer"] == "wall"
    assert by_question["Is the book on the table or under the table?"]["answer"] == "under"
    assert "What is on the floor?" not in by_question
