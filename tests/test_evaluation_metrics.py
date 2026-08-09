import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.metrics import (
    canonical_relation,
    counterfactual_consistency_metrics,
    exact_normalized_match,
    list_order_insensitive_match,
    score_predictions,
)
from semantic_3d_chat.evaluation.run import main as evaluation_main


def test_normalized_exact_and_order_insensitive_lists() -> None:
    assert exact_normalized_match("The RED-cube!", "red cube")
    assert list_order_insensitive_match("cube and book", ["book", "cube"])
    assert not list_order_insensitive_match("cube, cube", ["cube"])
    assert canonical_relation("It is to the left.") == "left"
    assert canonical_relation("left or right") is None


def test_structured_metrics_cover_missing_predictions_and_grounding() -> None:
    references = [
        {
            "scene_id": "scene_000001",
            "question_id": "q_presence_yes",
            "question": "Is it present?",
            "answer": "yes",
            "answer_type": "presence",
            "target_xyz": None,
        },
        {
            "scene_id": "scene_000001",
            "question_id": "q_presence_no",
            "question": "Is the distractor present?",
            "answer": "no",
            "answer_type": "presence",
            "target_xyz": None,
        },
        {
            "scene_id": "scene_000001",
            "question_id": "q_presence_missing",
            "question": "Is the other object present?",
            "answer": "yes",
            "answer_type": "presence",
            "target_xyz": None,
        },
        {
            "scene_id": "scene_000001",
            "question_id": "q_count",
            "question": "How many?",
            "answer": "2",
            "answer_type": "count",
            "count": 2,
            "target_xyz": None,
        },
        {
            "scene_id": "scene_000001",
            "question_id": "q_relation",
            "question": "Which side?",
            "answer": "left",
            "answer_type": "spatial_relation",
            "target_xyz": [1.0, 2.0, 3.0],
        },
        {
            "scene_id": "scene_000001",
            "question_id": "q_list",
            "question": "What is supported?",
            "answer": "book, cube",
            "answer_items": ["book", "cube"],
            "answer_type": "support",
            "target_xyz": None,
        },
    ]
    predictions = [
        {
            "scene_id": "scene_000001",
            "question_id": "q_presence_yes",
            "answer": "Yes.",
        },
        {
            "scene_id": "scene_000001",
            "question_id": "q_presence_no",
            "answer": "yes",
        },
        {
            "scene_id": "scene_000001",
            "question_id": "q_count",
            "answer": "There are two.",
        },
        {
            "scene_id": "scene_000001",
            "question_id": "q_relation",
            "answer": "It is to the left.",
            "grounding_xyz": [1.1, 2.0, 3.0],
        },
        {
            "scene_id": "scene_000001",
            "question_id": "q_list",
            "answer": "cube and book",
        },
        {"scene_id": "scene_999999", "question_id": "extra", "answer": "unused"},
    ]
    metrics = score_predictions(references, predictions)
    assert metrics["matched_prediction_count"] == 5
    assert metrics["missing_prediction_count"] == 1
    assert metrics["extra_prediction_count"] == 1
    assert metrics["presence"]["precision"] == pytest.approx(0.5)
    assert metrics["presence"]["recall"] == pytest.approx(0.5)
    assert metrics["presence"]["f1"] == pytest.approx(0.5)
    assert metrics["count"]["accuracy"] == 1.0
    assert metrics["count"]["mean_absolute_error"] == 0.0
    assert metrics["spatial_relation_accuracy"] == 1.0
    assert metrics["list_order_insensitive_accuracy"] == 1.0
    assert metrics["grounding"]["coverage"] == 1.0
    assert metrics["grounding"]["mean_coordinate_error_m"] == pytest.approx(0.1)
    assert metrics["per_type"]["presence"]["total"] == 3


def test_prediction_target_xyz_is_not_accepted_as_grounding() -> None:
    references = [
        {
            "scene_id": "scene_000001",
            "question_id": "q_000001",
            "question": "Where?",
            "answer": "left",
            "answer_type": "spatial_relation",
            "target_xyz": [1, 2, 3],
        }
    ]
    predictions = [
        {
            "scene_id": "scene_000001",
            "question_id": "q_000001",
            "answer": "left",
            # This is an oracle target field, not a model prediction field.
            "target_xyz": [1, 2, 3],
        }
    ]
    metrics = score_predictions(references, predictions)
    assert metrics["grounding"]["coverage"] == 0.0


def test_counterfactual_pairs_must_follow_changed_and_invariant_truth() -> None:
    references = [
        {
            "scene_id": "scene_000010",
            "question_id": "q_side",
            "question": "Which side?",
            "answer": "left",
            "answer_type": "spatial_relation",
            "pair_id": "pair_000001",
        },
        {
            "scene_id": "scene_000011",
            "question_id": "q_side",
            "question": "Which side?",
            "answer": "right",
            "answer_type": "spatial_relation",
            "pair_id": "pair_000001",
        },
        {
            "scene_id": "scene_000020",
            "question_id": "q_present",
            "question": "Is there a table?",
            "answer": "yes",
            "answer_type": "presence",
            "pair_id": "pair_000002",
        },
        {
            "scene_id": "scene_000021",
            "question_id": "q_present",
            "question": "Is there a table?",
            "answer": "yes",
            "answer_type": "presence",
            "pair_id": "pair_000002",
        },
    ]
    predictions = [
        {"scene_id": item["scene_id"], "question_id": item["question_id"], "answer": item["answer"]}
        for item in references
    ]
    by_key = {(item["scene_id"], item["question_id"]): item for item in predictions}
    metrics = counterfactual_consistency_metrics(references, by_key)
    assert metrics == {
        "eligible_pairs": 2,
        "malformed_pair_groups": 0,
        "expected_change_pairs": 1,
        "invariant_pairs": 1,
        "pair_accuracy": 1.0,
        "changed_when_expected_rate": 1.0,
        "invariant_when_expected_rate": 1.0,
    }


def test_jsonl_scoring_cli_writes_machine_readable_metrics(tmp_path: Path) -> None:
    reference = tmp_path / "references.jsonl"
    prediction = tmp_path / "predictions.jsonl"
    output = tmp_path / "metrics.json"
    reference.write_text(
        json.dumps(
            {
                "scene_id": "scene_000001",
                "question_id": "q_000001",
                "question": "Is it present?",
                "answer": "yes",
                "answer_type": "presence",
                "target_xyz": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prediction.write_text(
        json.dumps({"scene_id": "scene_000001", "question_id": "q_000001", "answer": "yes"}) + "\n",
        encoding="utf-8",
    )
    assert (
        evaluation_main(
            [
                "--references",
                str(reference),
                "--predictions",
                str(prediction),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["normalized_exact_accuracy"] == 1.0
    assert payload["presence"]["precision"] == 1.0
