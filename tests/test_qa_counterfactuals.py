from __future__ import annotations

from semantic_3d_chat.data.qa_generator import (
    annotate_counterfactual_questions,
    counterfactual_scene_groups,
    generate_scene_questions,
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
