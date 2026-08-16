from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import v56_fresh_development_score as score


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _synthetic_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
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
            reference: dict[str, object] = {
                "scene_id": scene_id,
                "question_id": f"q_{question_index:06d}",
                "question": f"Synthetic question {question_index}?",
                "answer_type": answer_type,
                "answer": answers[answer_type],
            }
            if question_index <= score.EXPECTED_GROUNDING_TARGET_COUNT:
                reference["target_xyz"] = [0.0, 0.0, 0.0]
            references.append(reference)
            predictions.append(
                {
                    "scene_id": scene_id,
                    "question_id": reference["question_id"],
                    "predicted_answer": answers[answer_type],
                    "prefix_hash": hashlib.sha256(scene_id.encode()).hexdigest(),
                    "grounding_xyz": [0.0, 0.0, 0.0],
                }
            )
            if answer_type == "spatial_relation":
                spatial_indices.append(len(references) - 1)

    offsets_by_family = {
        "book_support": ((0, 1), (6, 7), (12, 13), (18, 19)),
        "mirror_lr": ((2, 3), (8, 9), (14, 15), (20, 21)),
        "picture_support": ((4, 5), (10, 11), (16, 17), (22, 23)),
    }
    unit = 0
    for family, offsets in offsets_by_family.items():
        for first_offset, second_offset in offsets:
            pair_id = score.FAMILY_PAIR_IDS[family]
            question_key = f"cfq_{unit:016x}"
            for side, offset in enumerate((first_offset, second_offset)):
                row_index = spatial_indices[offset]
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
            unit += 1
    return references, predictions


def _terminal(
    path: Path,
    references: Path,
    predictions: Path,
) -> str:
    reference_sha = hashlib.sha256(references.read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "artifact": score.TERMINAL_ARTIFACT,
        "passed": True,
        "authorization": {
            "authorization_id": score.AUTHORIZATION_ID,
            "only_exact_action": "one_control_one_shot_fresh_development",
            "explicit_terminal_sha256_required": True,
            "thresholds": score.threshold_contract(),
            "development": {
                "reference_path": str(references),
                "reference_sha256": reference_sha,
            },
            "outputs": {"predictions": str(predictions)},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_thresholds_are_predeclared_above_v55_failure_points() -> None:
    thresholds = score.threshold_contract()

    assert thresholds["normalized_exact_accuracy_minimum"] == 0.42
    assert thresholds["canonical_correct_minimum"] == 93
    assert thresholds["spatial_relation_accuracy_minimum"] == 0.60
    assert thresholds["count_accuracy_minimum"] == 0.80
    assert thresholds["presence_f1_minimum"] == 0.30
    assert thresholds["canonical_complete_units_minimum"] == 2
    assert thresholds["successful_physical_change_families_minimum"] == 2


def test_full_synthetic_fresh_score_passes_without_serializing_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    references, predictions = _synthetic_rows()
    references_path = tmp_path / "references.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    terminal_path = tmp_path / "terminal.json"
    _write_jsonl(references_path, references)
    _write_jsonl(predictions_path, predictions)
    terminal_sha = _terminal(terminal_path, references_path, predictions_path)
    monkeypatch.setattr(score, "DEFAULT_TERMINAL", terminal_path)
    monkeypatch.setattr(score, "DEFAULT_REFERENCES", references_path)
    monkeypatch.setattr(score, "DEFAULT_PREDICTIONS", predictions_path)

    report = score.score_development(
        references_path,
        predictions_path,
        terminal_path=terminal_path,
        expected_terminal_sha256=terminal_sha,
    )

    assert report["passed"] is True
    assert report["canonical_type_specific"]["correct"] == 216
    assert report["changed_counterfactual"]["canonical_complete_units"] == 12
    assert report["changed_counterfactual"][
        "physical_change_families_with_complete_unit"
    ] == 3
    assert report["grounding"]["target_count"] == 132
    assert report["grounding"]["mean_coordinate_error_m"] == 0.0
    serialized = json.dumps(report, sort_keys=True)
    assert '"question":' not in serialized
    assert '"answer":' not in serialized
    assert '"predicted_answer":' not in serialized


def test_count_regression_fails_predeclared_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    references, predictions = _synthetic_rows()
    count_indices = [
        index
        for index, reference in enumerate(references)
        if reference["answer_type"] == "count"
    ]
    for index in count_indices[:9]:
        predictions[index]["predicted_answer"] = "unknown"
    references_path = tmp_path / "references.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    terminal_path = tmp_path / "terminal.json"
    _write_jsonl(references_path, references)
    _write_jsonl(predictions_path, predictions)
    terminal_sha = _terminal(terminal_path, references_path, predictions_path)
    monkeypatch.setattr(score, "DEFAULT_TERMINAL", terminal_path)
    monkeypatch.setattr(score, "DEFAULT_REFERENCES", references_path)
    monkeypatch.setattr(score, "DEFAULT_PREDICTIONS", predictions_path)

    report = score.score_development(
        references_path,
        predictions_path,
        terminal_path=terminal_path,
        expected_terminal_sha256=terminal_sha,
    )

    assert report["passed"] is False
    assert report["gates"]["count_accuracy_at_least_0_80"] is False


def test_changed_unit_rejects_the_wrong_atomic_scene_pair() -> None:
    references, predictions = _synthetic_rows()
    changed = next(
        reference
        for reference in references
        if reference.get("counterfactual_expected_change") is True
    )
    changed["scene_id"] = "scene_000062"
    indexed = score._prediction_index(predictions)

    with pytest.raises(ValueError, match="wrong atomic scene pair"):
        score._changed_metrics(references, indexed)


def test_changed_unit_requires_reference_answers_to_encode_a_change() -> None:
    references, predictions = _synthetic_rows()
    first = next(
        reference
        for reference in references
        if reference.get("counterfactual_expected_change") is True
    )
    pair_id = first["counterfactual_pair_id"]
    question_key = first["counterfactual_question_key"]
    members = [
        reference
        for reference in references
        if reference.get("counterfactual_pair_id") == pair_id
        and reference.get("counterfactual_question_key") == question_key
    ]
    members[1]["answer"] = members[0]["answer"]
    indexed = score._prediction_index(predictions)

    with pytest.raises(ValueError, match="do not encode a changed fact"):
        score._changed_metrics(references, indexed)
