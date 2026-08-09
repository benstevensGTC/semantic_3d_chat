from __future__ import annotations

import json

from semantic_3d_chat.data.dataset import SceneQADataset
from semantic_3d_chat.data.qa_generator import generate_scene_questions


def _ordered_relation_oracle() -> dict:
    return {
        "scene_id": "scene_000321",
        "instances": [
            {
                "instance_id": "i_subject",
                "kind": "object",
                "category": "cube",
                "color": {"name": "red"},
                "expected_center_xyz_m": [-1.25, 0.75, 0.4],
                "support_surface": None,
            },
            {
                "instance_id": "i_reference",
                "kind": "object",
                "category": "chair",
                "color": {"name": "blue"},
                "expected_center_xyz_m": [1.5, -0.25, 0.9],
                "support_surface": None,
            },
        ],
        "relationships": [
            {
                "subject_instance_id": "i_subject",
                "predicate": "left_of",
                "object_instance_id": "i_reference",
            }
        ],
        "scan_origin_xyz_m": [0.0, 0.0, 1.4],
    }


def test_spatial_relation_records_include_exact_ordered_reference_center() -> None:
    records = generate_scene_questions(_ordered_relation_oracle(), seed=17)

    relation = next(record for record in records if record["answer_type"] == "spatial_relation")

    assert relation["target_instance"] == "i_subject"
    assert relation["target_xyz"] == [-1.25, 0.75, 0.4]
    assert relation["reference_instance"] == "i_reference"
    assert relation["reference_xyz"] == [1.5, -0.25, 0.9]
    assert all(
        "reference_xyz" not in record
        for record in records
        if record["answer_type"] != "spatial_relation"
    )


def test_dataset_parses_reference_xyz_and_accepts_legacy_records(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    records = [
        {
            "scene_id": "scene_000321",
            "question_id": "q_000001",
            "question": "Is the cube left or right of the chair?",
            "answer": "left",
            "answer_type": "spatial_relation",
            "target_xyz": [-1.25, 0.75, 0.4],
            "reference_xyz": [1.5, -0.25, 0.9],
        },
        {
            "scene_id": "scene_000321",
            "question_id": "q_000002",
            "question": "Is there a cube in the room?",
            "answer": "yes",
            "answer_type": "presence",
            "target_xyz": None,
        },
    ]
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    dataset = SceneQADataset(path)

    assert dataset[0].reference_xyz == [1.5, -0.25, 0.9]
    assert dataset[1].reference_xyz is None
