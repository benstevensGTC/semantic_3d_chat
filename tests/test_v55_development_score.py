from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import v55_development_score as score


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_canonical_type_specific_interpretation() -> None:
    assert score.canonical_type_specific_match("presence", "Yes, there is one.", "yes")
    assert score.canonical_type_specific_match("count", "There are two.", "2")
    assert score.canonical_type_specific_match(
        "spatial_relation", "It is on the left-hand side.", "left"
    )
    assert score.canonical_type_specific_match("support", ["book", "cube"], "cube, book")
    assert not score.canonical_type_specific_match("presence", "unknown", "yes")


def _synthetic_development_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    answers: dict[str, object] = {
        "attribute": "red",
        "count": "1",
        "metric": "1 meter",
        "orientation": "upright",
        "presence": "yes",
        "spatial_relation": "left",
        "support": ["book"],
    }
    references: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    spatial_indices: list[int] = []
    question_index = 0
    for answer_type, count in score.EXPECTED_TYPE_COUNTS.items():
        for _ in range(count):
            scene_id = score.EXPECTED_SCENE_IDS[question_index % 6]
            question_index += 1
            row: dict[str, object] = {
                "scene_id": scene_id,
                "question_id": f"q_{question_index:06d}",
                "question": f"Synthetic question {question_index}?",
                "answer_type": answer_type,
                "answer": answers[answer_type],
            }
            references.append(row)
            predictions.append(
                {
                    "scene_id": scene_id,
                    "question_id": row["question_id"],
                    "predicted_answer": answers[answer_type],
                    "prefix_hash": hashlib.sha256(scene_id.encode()).hexdigest(),
                }
            )
            if answer_type == "spatial_relation":
                spatial_indices.append(len(references) - 1)

    for unit, first_index in enumerate(spatial_indices[:24:2]):
        second_index = spatial_indices[2 * unit + 1]
        family_index = unit // 4
        pair_id = tuple(score.FAMILY_PAIR_IDS.values())[family_index]
        question_key = f"cfq_{unit:016x}"
        for side, row_index in enumerate((first_index, second_index)):
            expected = "left" if side == 0 else "right"
            references[row_index].update(
                {
                    "counterfactual_expected_change": True,
                    "counterfactual_pair_id": pair_id,
                    "counterfactual_question_key": question_key,
                    "answer": expected,
                }
            )
            predictions[row_index]["predicted_answer"] = expected
    return references, predictions


def test_full_synthetic_score_recomputes_comparator_and_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    references, predictions = _synthetic_development_rows()
    baseline_predictions = []
    for index, (reference, prediction) in enumerate(zip(references, predictions, strict=True)):
        baseline_predictions.append(
            {
                "scene_id": reference["scene_id"],
                "question_id": reference["question_id"],
                "predicted_answer": (
                    prediction["predicted_answer"] if index < 91 else "definitely wrong"
                ),
            }
        )

    references_path = tmp_path / "references.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    baseline_predictions_path = tmp_path / "baseline_predictions.jsonl"
    baseline_path = tmp_path / "baseline.json"
    _write_jsonl(references_path, references)
    _write_jsonl(predictions_path, predictions)
    _write_jsonl(baseline_predictions_path, baseline_predictions)
    reference_sha = _sha256(references_path)
    baseline_path.write_text(
        json.dumps(
            {
                "reference_count": 216,
                "prediction_count": 216,
                "missing_prediction_count": 0,
                "extra_prediction_count": 0,
                "normalized_exact_accuracy": 0.375,
                "references_sha256": reference_sha,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(score, "REFERENCE_SHA256", reference_sha)
    monkeypatch.setattr(score, "BASELINE_SHA256", _sha256(baseline_path))
    monkeypatch.setattr(
        score,
        "BASELINE_PREDICTIONS_SHA256",
        _sha256(baseline_predictions_path),
    )

    report = score.score_development(
        references_path,
        predictions_path,
        baseline_path,
        baseline_predictions_path,
    )

    assert report["passed"] is True
    assert report["canonical_type_specific"] == {
        "correct": 216,
        "total": 216,
        "accuracy": 1.0,
        "scorer": "presence/count/spatial/list canonicalization; normalized exact otherwise",
    }
    changed = report["changed_counterfactual"]
    assert changed["canonical_correct_sides"] == 24
    assert changed["canonical_complete_units"] == 12
    assert changed["canonical_prediction_changed_units"] == 12
    assert changed["physical_change_families_with_complete_unit"] == 3
    assert report["inputs"]["baseline_predictions"][
        "canonical_type_specific_correct"
    ] == 91
    serialized = json.dumps(report, sort_keys=True)
    assert '"question":' not in serialized
    assert '"answer":' not in serialized
    assert '"predicted_answer":' not in serialized


def test_prefix_inventory_rejects_question_dependent_prefix() -> None:
    rows = []
    for scene_id in score.EXPECTED_SCENE_IDS:
        rows.append(
            {
                "scene_id": scene_id,
                "question_id": f"q_{len(rows) + 1:06d}",
                "prefix_hash": "0" * 64,
            }
        )
    rows.append(
        {
            "scene_id": score.EXPECTED_SCENE_IDS[0],
            "question_id": "q_999999",
            "prefix_hash": "1" * 64,
        }
    )
    with pytest.raises(ValueError, match="prefix hashes are not invariant"):
        score._prefix_inventory(rows)
