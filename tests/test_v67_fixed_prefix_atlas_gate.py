from __future__ import annotations

from typing import Any

from semantic_3d_chat.evaluation import fixed_prefix_atlas_gate as gate
from semantic_3d_chat.evaluation.question_manifest import (
    QuestionManifest,
    QuestionRecord,
    questions_sha256,
)


def _synthetic_v67_population() -> tuple[
    QuestionManifest,
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    dict[str, Any],
]:
    questions: list[QuestionRecord] = []
    references: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    changed_by_pair = (4, 4, 4, 4, 3, 3, 2, 2)
    ordinal = 1
    for pair_index, changed_count in enumerate(changed_by_pair, start=1):
        scenes = (f"scene_{200 + pair_index * 2 - 1:06d}", f"scene_{200 + pair_index * 2:06d}")
        for unit_index in range(24):
            changed = unit_index < changed_count
            question = f"Synthetic paired question {pair_index}-{unit_index}?"
            unit_key = f"unit_{pair_index:02d}_{unit_index:02d}"
            for side_index, scene_id in enumerate(scenes):
                question_id = f"q_{ordinal:06d}"
                ordinal += 1
                answer = "left" if side_index == 0 or not changed else "right"
                questions.append(QuestionRecord(scene_id, question_id, question))
                references.append(
                    {
                        "scene_id": scene_id,
                        "question_id": question_id,
                        "answer": answer,
                        "answer_items": None,
                        "answer_type": "spatial_relation",
                        "route_label": changed,
                        "counterfactual_pair_id": f"pair_{pair_index:02d}",
                        "counterfactual_paired_scene_id": scenes[1 - side_index],
                        "counterfactual_question_key": unit_key,
                        "counterfactual_change_type": "synthetic_change",
                        "counterfactual_role": (
                            "reference" if side_index == 0 else "counterfactual"
                        ),
                    }
                )
                predictions.append(
                    {
                        "scene_id": scene_id,
                        "question_id": question_id,
                        "predicted_answer": answer,
                    }
                )
    question_records = tuple(questions)
    manifest = QuestionManifest(
        questions=question_records,
        questions_sha256=questions_sha256(question_records),
        source_qa_sha256="a" * 64,
    )
    reference_rows = tuple(references)
    contract = {
        "population": {
            "changed_paired_units": 26,
            "changed_sides": 52,
            "natural_question_count": 384,
            "paired_units": 192,
            "scene_count": 16,
        },
        "source_boundary": {
            "scorer_records_sha256": gate._canonical_jsonl_sha256(reference_rows),
        },
        "thresholds": {
            "natural_canonical_exact": {"minimum": 192, "total": 384},
            "changed_side_exact": {"minimum": 32, "total": 52},
            "changed_paired_unit_complete": {"minimum": 10, "total": 26},
            "changed_paired_unit_correct_direction": {"minimum": 15, "total": 26},
            "normalized_exact_accuracy": {"minimum": 0.5, "total": 384},
        },
    }
    return manifest, reference_rows, tuple(predictions), contract


def test_v67_terminal_metrics_pass_exact_synthetic_population() -> None:
    questions, references, predictions, contract = _synthetic_v67_population()

    result = gate._score_v67_terminal_metrics(
        questions=questions,
        references=references,
        predictions=predictions,
        preregistration=contract,
        normalized_exact_accuracy=1.0,
    )

    assert result["passed"] is True
    assert result["metrics"] == {
        "natural_canonical_exact": 384,
        "natural_question_total": 384,
        "changed_side_exact": 52,
        "changed_side_total": 52,
        "changed_paired_unit_complete": 26,
        "changed_paired_unit_correct_direction": 26,
        "changed_paired_unit_total": 26,
        "normalized_exact_accuracy": 1.0,
    }
    assert all(result["checks"].values())


def test_v67_terminal_metrics_reject_scene_insensitive_changed_answers() -> None:
    questions, references, predictions, contract = _synthetic_v67_population()
    insensitive = []
    references_by_key = {(str(row["scene_id"]), str(row["question_id"])): row for row in references}
    for prediction in predictions:
        key = str(prediction["scene_id"]), str(prediction["question_id"])
        reference = references_by_key[key]
        insensitive.append(
            {
                **prediction,
                "predicted_answer": (
                    "left" if reference["route_label"] is True else prediction["predicted_answer"]
                ),
            }
        )

    result = gate._score_v67_terminal_metrics(
        questions=questions,
        references=references,
        predictions=insensitive,
        preregistration=contract,
        normalized_exact_accuracy=358 / 384,
    )

    assert result["metrics"]["natural_canonical_exact"] == 358
    assert result["checks"]["natural_canonical_exact"] is True
    assert result["checks"]["normalized_exact_accuracy"] is True
    assert result["metrics"]["changed_side_exact"] == 26
    assert result["checks"]["changed_side_exact"] is False
    assert result["metrics"]["changed_paired_unit_complete"] == 0
    assert result["checks"]["changed_paired_unit_complete"] is False
    assert result["metrics"]["changed_paired_unit_correct_direction"] == 0
    assert result["checks"]["changed_paired_unit_correct_direction"] is False
    assert result["passed"] is False
