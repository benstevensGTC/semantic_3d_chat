"""Deterministic structured QA metrics for evaluation-only reference records."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

ARTICLES = frozenset({"a", "an", "the"})
PRESENCE_POSITIVE = frozenset({"yes", "present", "true"})
PRESENCE_NEGATIVE = frozenset({"no", "absent", "false"})
NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
RELATION_PHRASES = (
    ("in front of", "in front"),
    ("in front", "in front"),
    ("to the left of", "left"),
    ("on the left", "left"),
    ("left of", "left"),
    ("to the right of", "right"),
    ("on the right", "right"),
    ("right of", "right"),
    ("underneath", "below"),
    ("under", "below"),
    ("below", "below"),
    ("above", "above"),
    ("behind", "behind"),
    ("inside", "inside"),
    ("near", "near"),
    ("far", "far"),
    ("left", "left"),
    ("right", "right"),
)
LIST_ANSWER_TYPES = frozenset({"list", "support", "containment"})


def normalize_answer(value: Any) -> str:
    """Apply a SQuAD-style exact-match normalization to an answer."""

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    characters = [
        character if (character.isalnum() or character.isspace()) else " " for character in text
    ]
    tokens = "".join(characters).split()
    return " ".join(token for token in tokens if token not in ARTICLES)


def exact_normalized_match(prediction: Any, reference: Any) -> bool:
    return normalize_answer(prediction) == normalize_answer(reference)


def normalize_answer_items(value: Any) -> tuple[str, ...]:
    """Normalize a list answer without imposing an output order."""

    if value is None:
        return ()
    if isinstance(value, str):
        raw_items = re.split(r"\s*(?:,|;|\band\b)\s*", value, flags=re.IGNORECASE)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_items = [str(item) for item in value]
    else:
        raw_items = [str(value)]
    return tuple(item for item in (normalize_answer(item) for item in raw_items) if item)


def list_order_insensitive_match(prediction: Any, reference: Any) -> bool:
    return Counter(normalize_answer_items(prediction)) == Counter(normalize_answer_items(reference))


def canonical_presence(value: Any) -> bool | None:
    normalized = normalize_answer(value)
    tokens = normalized.split()
    positive = any(token in PRESENCE_POSITIVE for token in tokens)
    negative = any(token in PRESENCE_NEGATIVE for token in tokens)
    if positive == negative:
        return None
    return positive


def extract_count(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(value) if np.isfinite(value) and float(value).is_integer() else None
    normalized = normalize_answer(value)
    candidates: list[int] = []
    for token in normalized.split():
        if re.fullmatch(r"[+-]?[0-9]+", token):
            candidates.append(int(token))
        elif token in NUMBER_WORDS:
            candidates.append(NUMBER_WORDS[token])
    return candidates[0] if len(set(candidates)) == 1 else None


def canonical_relation(value: Any) -> str | None:
    normalized = normalize_answer(value)
    matches = {
        canonical
        for phrase, canonical in RELATION_PHRASES
        if re.search(rf"(?:^|\s){re.escape(phrase)}(?:$|\s)", normalized)
    }
    return next(iter(matches)) if len(matches) == 1 else None


def coordinate_from_record(record: Mapping[str, Any], *, prediction: bool) -> np.ndarray | None:
    keys = (
        ("predicted_xyz", "grounding_xyz", "coordinate")
        if prediction
        else ("target_xyz", "ground_truth_xyz", "coordinate")
    )
    value = next((record[key] for key in keys if record.get(key) is not None), None)
    if value is None:
        return None
    coordinate = np.asarray(value, dtype=np.float64)
    if coordinate.shape != (3,) or not np.isfinite(coordinate).all():
        return None
    return coordinate


def _safe_divide(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _prediction_key(record: Mapping[str, Any]) -> tuple[str, str]:
    scene_id = record.get("scene_id")
    question_id = record.get("question_id")
    if not isinstance(scene_id, str) or not scene_id:
        raise ValueError("Every record requires a non-empty scene_id")
    if not isinstance(question_id, str) or not question_id:
        raise ValueError("Every record requires a non-empty question_id")
    return scene_id, question_id


def _prediction_answer(record: Mapping[str, Any] | None) -> Any:
    if record is None:
        return None
    if "predicted_answer" in record:
        return record["predicted_answer"]
    return record.get("answer")


def _reference_items(record: Mapping[str, Any]) -> Any:
    return record.get("answer_items", record.get("answer"))


def _counterfactual_pair_id(record: Mapping[str, Any]) -> str | None:
    direct = record.get("counterfactual_pair_id", record.get("pair_id"))
    if isinstance(direct, str) and direct:
        return direct
    nested = record.get("counterfactual_pair")
    if isinstance(nested, Mapping):
        pair_id = nested.get("pair_id")
        if isinstance(pair_id, str) and pair_id:
            return pair_id
    return None


def _counterfactual_key(record: Mapping[str, Any]) -> str:
    explicit = record.get("counterfactual_question_key")
    if isinstance(explicit, str) and explicit:
        return explicit
    return normalize_answer(record.get("question", record.get("question_id", "")))


def counterfactual_consistency_metrics(
    references: Sequence[Mapping[str, Any]],
    predictions_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Score paired answers against the direction of the oracle-backed change."""

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for reference in references:
        pair_id = _counterfactual_pair_id(reference)
        if pair_id is not None:
            groups[(pair_id, _counterfactual_key(reference))].append(reference)

    eligible = {key: members for key, members in groups.items() if len(members) == 2}
    malformed = len(groups) - len(eligible)
    pair_correct: list[bool] = []
    changed_when_expected: list[bool] = []
    invariant_when_expected: list[bool] = []
    expected_change_pairs = 0
    invariant_pairs = 0
    for members in eligible.values():
        first, second = members
        first_prediction = predictions_by_key.get(_prediction_key(first))
        second_prediction = predictions_by_key.get(_prediction_key(second))
        first_predicted_answer = normalize_answer(_prediction_answer(first_prediction))
        second_predicted_answer = normalize_answer(_prediction_answer(second_prediction))
        first_answer = normalize_answer(first.get("answer"))
        second_answer = normalize_answer(second.get("answer"))
        both_correct = (
            bool(first_predicted_answer)
            and bool(second_predicted_answer)
            and first_predicted_answer == first_answer
            and second_predicted_answer == second_answer
        )
        pair_correct.append(both_correct)
        if first_answer != second_answer:
            expected_change_pairs += 1
            changed_when_expected.append(
                bool(first_predicted_answer)
                and bool(second_predicted_answer)
                and first_predicted_answer != second_predicted_answer
            )
        else:
            invariant_pairs += 1
            invariant_when_expected.append(
                bool(first_predicted_answer) and first_predicted_answer == second_predicted_answer
            )

    return {
        "eligible_pairs": len(eligible),
        "malformed_pair_groups": malformed,
        "expected_change_pairs": expected_change_pairs,
        "invariant_pairs": invariant_pairs,
        "pair_accuracy": _mean([float(value) for value in pair_correct]),
        "changed_when_expected_rate": _mean([float(value) for value in changed_when_expected]),
        "invariant_when_expected_rate": _mean([float(value) for value in invariant_when_expected]),
    }


def score_predictions(
    references: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score prediction records without importing or invoking a language model."""

    if not references:
        raise ValueError("At least one reference record is required")
    reference_keys: set[tuple[str, str]] = set()
    for reference in references:
        key = _prediction_key(reference)
        if key in reference_keys:
            raise ValueError(f"Duplicate reference key: {key}")
        reference_keys.add(key)
        if "answer" not in reference or "answer_type" not in reference:
            raise ValueError(f"Reference {key} requires answer and answer_type")

    predictions_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for prediction in predictions:
        key = _prediction_key(prediction)
        if key in predictions_by_key:
            raise ValueError(f"Duplicate prediction key: {key}")
        predictions_by_key[key] = prediction

    per_type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    exact_values: list[float] = []
    list_values: list[float] = []
    count_values: list[float] = []
    count_errors: list[float] = []
    relation_values: list[float] = []
    coordinate_errors: list[float] = []
    coordinate_targets = 0
    presence_counts: Counter[str] = Counter()
    example_results: list[dict[str, Any]] = []

    for reference in references:
        key = _prediction_key(reference)
        prediction = predictions_by_key.get(key)
        answer_type = str(reference["answer_type"])
        predicted_answer = _prediction_answer(prediction)
        expected_answer = reference["answer"]
        exact = bool(prediction is not None) and exact_normalized_match(
            predicted_answer, expected_answer
        )
        exact_values.append(float(exact))
        per_type_counts[answer_type]["total"] += 1
        per_type_counts[answer_type]["exact_correct"] += int(exact)
        result: dict[str, Any] = {
            "scene_id": key[0],
            "question_id": key[1],
            "answer_type": answer_type,
            "prediction_present": prediction is not None,
            "normalized_exact": exact,
        }

        is_list = "answer_items" in reference or answer_type in LIST_ANSWER_TYPES
        if is_list:
            list_correct = bool(prediction is not None) and list_order_insensitive_match(
                predicted_answer, _reference_items(reference)
            )
            list_values.append(float(list_correct))
            per_type_counts[answer_type]["list_correct"] += int(list_correct)
            per_type_counts[answer_type]["list_total"] += 1
            result["list_order_insensitive"] = list_correct

        if answer_type == "presence":
            expected_presence = canonical_presence(expected_answer)
            predicted_presence = canonical_presence(predicted_answer)
            if expected_presence is None:
                raise ValueError(f"Presence reference {key} is not a yes/no value")
            if expected_presence and predicted_presence is True:
                presence_counts["tp"] += 1
            elif expected_presence:
                presence_counts["fn"] += 1
            elif predicted_presence is True:
                presence_counts["fp"] += 1
            elif predicted_presence is False:
                presence_counts["tn"] += 1
            else:
                presence_counts["invalid_negative"] += 1
            result["presence_correct"] = predicted_presence == expected_presence

        if answer_type == "count":
            expected_count = (
                extract_count(reference.get("count"))
                if reference.get("count") is not None
                else extract_count(expected_answer)
            )
            if expected_count is None:
                raise ValueError(f"Count reference {key} has no canonical integer")
            predicted_count = extract_count(predicted_answer)
            count_correct = predicted_count == expected_count
            count_values.append(float(count_correct))
            if predicted_count is not None:
                count_errors.append(float(abs(predicted_count - expected_count)))
            result.update(
                {
                    "count_correct": count_correct,
                    "predicted_count": predicted_count,
                    "expected_count": expected_count,
                }
            )

        if answer_type == "spatial_relation":
            expected_relation = canonical_relation(expected_answer)
            predicted_relation = canonical_relation(predicted_answer)
            if expected_relation is None:
                raise ValueError(f"Spatial-relation reference {key} is not canonical")
            relation_correct = predicted_relation == expected_relation
            relation_values.append(float(relation_correct))
            result.update(
                {
                    "relation_correct": relation_correct,
                    "predicted_relation": predicted_relation,
                    "expected_relation": expected_relation,
                }
            )

        expected_coordinate = coordinate_from_record(reference, prediction=False)
        if expected_coordinate is not None:
            coordinate_targets += 1
            predicted_coordinate = (
                coordinate_from_record(prediction, prediction=True)
                if prediction is not None
                else None
            )
            coordinate_error = (
                float(np.linalg.norm(predicted_coordinate - expected_coordinate))
                if predicted_coordinate is not None
                else None
            )
            if coordinate_error is not None:
                coordinate_errors.append(coordinate_error)
            result["coordinate_error_m"] = coordinate_error

        example_results.append(result)

    per_type = {
        answer_type: {
            "total": counts["total"],
            "normalized_exact_accuracy": _safe_divide(counts["exact_correct"], counts["total"]),
            **(
                {
                    "list_order_insensitive_accuracy": _safe_divide(
                        counts["list_correct"], counts["list_total"]
                    )
                }
                if counts["list_total"]
                else {}
            ),
        }
        for answer_type, counts in sorted(per_type_counts.items())
    }
    tp, fp, fn, tn = (
        presence_counts["tp"],
        presence_counts["fp"],
        presence_counts["fn"],
        presence_counts["tn"],
    )
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    coordinate_array = np.asarray(coordinate_errors, dtype=np.float64)
    matched_keys = reference_keys & predictions_by_key.keys()
    extra_keys = predictions_by_key.keys() - reference_keys
    return {
        "schema_version": 1,
        "reference_count": len(references),
        "prediction_count": len(predictions),
        "matched_prediction_count": len(matched_keys),
        "missing_prediction_count": len(reference_keys - predictions_by_key.keys()),
        "extra_prediction_count": len(extra_keys),
        "normalized_exact_accuracy": _mean(exact_values),
        "list_order_insensitive_accuracy": _mean(list_values),
        "count": {
            "examples": len(count_values),
            "accuracy": _mean(count_values),
            "prediction_coverage": _safe_divide(len(count_errors), len(count_values)),
            "mean_absolute_error": _mean(count_errors),
        },
        "spatial_relation_accuracy": _mean(relation_values),
        "presence": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "invalid_negative_predictions": presence_counts["invalid_negative"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "grounding": {
            "target_count": coordinate_targets,
            "prediction_count": len(coordinate_errors),
            "coverage": _safe_divide(len(coordinate_errors), coordinate_targets),
            "mean_coordinate_error_m": (
                float(coordinate_array.mean()) if coordinate_array.size else None
            ),
            "median_coordinate_error_m": (
                float(np.median(coordinate_array)) if coordinate_array.size else None
            ),
            "rmse_coordinate_error_m": (
                float(math.sqrt(np.mean(coordinate_array**2))) if coordinate_array.size else None
            ),
            "within_0_25m_accuracy": (
                float(np.mean(coordinate_array <= 0.25)) if coordinate_array.size else None
            ),
            "within_0_50m_accuracy": (
                float(np.mean(coordinate_array <= 0.50)) if coordinate_array.size else None
            ),
            "within_1_00m_accuracy": (
                float(np.mean(coordinate_array <= 1.00)) if coordinate_array.size else None
            ),
        },
        "per_type": per_type,
        "counterfactual": counterfactual_consistency_metrics(references, predictions_by_key),
        "examples": example_results,
    }
