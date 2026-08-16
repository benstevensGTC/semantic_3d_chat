"""Score the one-shot V55 development predictions without loading a model.

This module is deliberately separate from inference.  It is the only V55
component that opens answer-bearing development references.  The resulting
report contains aggregate numbers, opaque hashes, and prefix hashes, but no
question, reference-answer, or generated-answer strings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.metrics import (
    LIST_ANSWER_TYPES,
    canonical_presence,
    canonical_relation,
    exact_normalized_match,
    extract_count,
    list_order_insensitive_match,
    normalize_answer,
    normalize_answer_items,
    score_predictions,
)
from semantic_3d_chat.evaluation.run import load_jsonl

AUTHORIZATION_ID: Final[str] = "v55_one_shot_development_score"
EXPECTED_SCENE_IDS: Final[tuple[str, ...]] = tuple(
    f"scene_{index:06d}" for index in range(19, 25)
)
EXPECTED_REFERENCE_COUNT: Final[int] = 216
EXPECTED_CHANGED_UNIT_COUNT: Final[int] = 12
EXPECTED_CHANGED_SIDE_COUNT: Final[int] = 24
FAMILY_PAIR_IDS: Final[dict[str, str]] = {
    "book_support": "pair_000009",
    "mirror_lr": "pair_000010",
    "picture_support": "pair_000011",
}
EXPECTED_TYPE_COUNTS: Final[dict[str, int]] = {
    "attribute": 48,
    "count": 42,
    "metric": 6,
    "orientation": 6,
    "presence": 42,
    "spatial_relation": 48,
    "support": 24,
}

DEFAULT_REFERENCES: Final[Path] = Path("data_diverse28/qa/validation.jsonl")
DEFAULT_PREDICTIONS: Final[Path] = Path(
    "reports/gemma4/predictions/v55_development_validation.jsonl"
)
DEFAULT_BASELINE: Final[Path] = Path(
    "reports/gemma4/metrics/v29_diverse_validation.json"
)
DEFAULT_BASELINE_PREDICTIONS: Final[Path] = Path(
    "reports/gemma4/predictions/v29_diverse_validation.jsonl"
)
DEFAULT_OUTPUT: Final[Path] = Path(
    "reports/gemma4/metrics/v55_development_score.json"
)
BASELINE_SHA256: Final[str] = (
    "21bf97ce8c5afd68d512fa04e5a526701e3dc5e5bed2c9fa745a0dbb0c775e09"
)
BASELINE_PREDICTIONS_SHA256: Final[str] = (
    "25d0cf742c9a0aec409853aa75ddd994e53b1266dd9dddfc4e5b97310f8c8a72"
)
REFERENCE_SHA256: Final[str] = (
    "67fb14685b3f4cb43f2409db7eb84220ec89d6390205b7bb86eb148b4d4e68b2"
)
BASELINE_EXACT_CORRECT: Final[int] = 81
BASELINE_EXACT_ACCURACY: Final[float] = 0.375
BASELINE_CANONICAL_CORRECT: Final[int] = 91
BASELINE_CANONICAL_ACCURACY: Final[float] = (
    BASELINE_CANONICAL_CORRECT / EXPECTED_REFERENCE_COUNT
)

MIN_EXACT_ACCURACY: Final[float] = 0.375
MIN_SPATIAL_RELATION_ACCURACY: Final[float] = 0.55
MIN_COUNT_ACCURACY: Final[float] = 0.80
MIN_PRESENCE_F1: Final[float] = 0.15
MIN_CANONICAL_COMPLETE_UNITS: Final[int] = 2
MIN_CANONICAL_CORRECT_SIDES: Final[int] = 12
MIN_CANONICAL_CHANGED_UNITS: Final[int] = 2
MIN_PHYSICAL_CHANGE_FAMILIES: Final[int] = 2
MIN_CANONICAL_AGGREGATE_CORRECT: Final[int] = BASELINE_CANONICAL_CORRECT

_HEX64 = re.compile(r"[0-9a-f]{64}")


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    combined = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(combined))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _prediction_answer(record: Mapping[str, Any] | None) -> Any:
    if record is None:
        return None
    if "predicted_answer" in record:
        return record["predicted_answer"]
    return record.get("answer")


def canonical_answer_key(answer_type: str, value: Any) -> object | None:
    """Return a comparable, type-aware answer value or ``None`` if invalid."""

    if answer_type in LIST_ANSWER_TYPES:
        items = normalize_answer_items(value)
        return tuple(sorted(items)) if items else None
    if answer_type == "presence":
        return canonical_presence(value)
    if answer_type == "count":
        return extract_count(value)
    if answer_type == "spatial_relation":
        return canonical_relation(value)
    normalized = normalize_answer(value)
    return normalized if normalized else None


def canonical_type_specific_match(
    answer_type: str,
    prediction: Any,
    reference: Any,
) -> bool:
    """Use V54's precommitted type-aware interpretation for development QA."""

    if answer_type in LIST_ANSWER_TYPES:
        return list_order_insensitive_match(prediction, reference)
    expected = canonical_answer_key(answer_type, reference)
    observed = canonical_answer_key(answer_type, prediction)
    if answer_type in {"presence", "count", "spatial_relation"}:
        return expected is not None and observed == expected
    return exact_normalized_match(prediction, reference)


def _key(record: Mapping[str, Any]) -> tuple[str, str]:
    scene_id = record.get("scene_id")
    question_id = record.get("question_id")
    if not isinstance(scene_id, str) or not isinstance(question_id, str):
        raise TypeError("V55 records require opaque scene_id and question_id strings")
    return scene_id, question_id


def _prediction_index(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in predictions:
        key = _key(row)
        if key in result:
            raise ValueError(f"Duplicate V55 prediction key: {key}")
        result[key] = row
    return result


def _authenticate_baseline(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"V55 baseline metrics are unavailable or unsafe: {path}")
    observed_sha = _sha256(path)
    if observed_sha != BASELINE_SHA256:
        raise ValueError("V55 immutable V29 baseline metrics changed")
    report = _mapping(json.loads(path.read_text(encoding="utf-8")), "V29 baseline")
    count = report.get("reference_count")
    exact = report.get("normalized_exact_accuracy")
    if (
        count != EXPECTED_REFERENCE_COUNT
        or report.get("prediction_count") != EXPECTED_REFERENCE_COUNT
        or report.get("missing_prediction_count") != 0
        or report.get("extra_prediction_count") != 0
        or report.get("references_sha256") != REFERENCE_SHA256
        or exact != BASELINE_EXACT_ACCURACY
        or round(float(exact) * int(count)) != BASELINE_EXACT_CORRECT
    ):
        raise ValueError("V55 immutable V29 baseline numeric contract changed")
    return {
        "path": str(DEFAULT_BASELINE),
        "sha256": observed_sha,
        "reference_count": EXPECTED_REFERENCE_COUNT,
        "normalized_exact_correct": BASELINE_EXACT_CORRECT,
        "normalized_exact_accuracy": BASELINE_EXACT_ACCURACY,
        "canonical_type_specific_correct": BASELINE_CANONICAL_CORRECT,
        "canonical_type_specific_accuracy": BASELINE_CANONICAL_ACCURACY,
    }


def _authenticate_baseline_predictions(
    path: Path,
    references: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute the canonical V29 comparator on the exact V55 references."""

    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(
            f"V55 baseline predictions are unavailable or unsafe: {path}"
        )
    observed_sha = _sha256(path)
    if observed_sha != BASELINE_PREDICTIONS_SHA256:
        raise ValueError("V55 immutable V29 baseline predictions changed")
    predictions = load_jsonl(path)
    if len(predictions) != EXPECTED_REFERENCE_COUNT:
        raise ValueError("V55 V29 baseline must contain exactly 216 predictions")
    prediction_index = _prediction_index(predictions)
    reference_keys = {_key(row) for row in references}
    if set(prediction_index) != reference_keys:
        raise ValueError("V55 V29 baseline does not exactly cover current references")
    canonical_correct = sum(
        canonical_type_specific_match(
            str(reference["answer_type"]),
            _prediction_answer(prediction_index[_key(reference)]),
            reference["answer"],
        )
        for reference in references
    )
    if canonical_correct != BASELINE_CANONICAL_CORRECT:
        raise ValueError(
            "V55 recomputed V29 canonical comparator changed: "
            f"expected={BASELINE_CANONICAL_CORRECT} observed={canonical_correct}"
        )
    return {
        "path": str(DEFAULT_BASELINE_PREDICTIONS),
        "sha256": observed_sha,
        "canonical_type_specific_correct": canonical_correct,
        "canonical_type_specific_accuracy": (
            canonical_correct / EXPECTED_REFERENCE_COUNT
        ),
    }


def _changed_metrics(
    references: Sequence[Mapping[str, Any]],
    predictions: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for reference in references:
        if reference.get("counterfactual_expected_change") is not True:
            continue
        pair_id = reference.get("counterfactual_pair_id")
        question_key = reference.get("counterfactual_question_key")
        if not isinstance(pair_id, str) or not isinstance(question_key, str):
            raise TypeError("V55 changed reference lacks opaque pair metadata")
        grouped[(pair_id, question_key)].append(reference)
    if len(grouped) != EXPECTED_CHANGED_UNIT_COUNT:
        raise ValueError(
            "V55 expected exactly 12 changed units; "
            f"observed={len(grouped)}"
        )
    reverse = {pair_id: family for family, pair_id in FAMILY_PAIR_IDS.items()}
    by_family: dict[str, Counter[str]] = {
        family: Counter() for family in FAMILY_PAIR_IDS
    }
    correct_sides = 0
    complete_units = 0
    prediction_changed_units = 0
    for (pair_id, _question_key), rows in sorted(grouped.items()):
        if pair_id not in reverse or len(rows) != 2:
            raise ValueError("V55 changed units must be two-sided locked physical families")
        if len({_key(row)[0] for row in rows}) != 2:
            raise ValueError("V55 changed unit repeats one scene")
        answer_types = {str(row.get("answer_type")) for row in rows}
        if len(answer_types) != 1:
            raise ValueError("V55 changed unit answer types disagree")
        answer_type = next(iter(answer_types))
        observed_values: list[object | None] = []
        correctness: list[bool] = []
        for reference in rows:
            prediction = predictions.get(_key(reference))
            predicted_answer = _prediction_answer(prediction)
            correctness.append(
                prediction is not None
                and canonical_type_specific_match(
                    answer_type,
                    predicted_answer,
                    reference.get("answer"),
                )
            )
            observed_values.append(canonical_answer_key(answer_type, predicted_answer))
        family = reverse[pair_id]
        side_count = sum(correctness)
        complete = int(all(correctness))
        changed = int(
            all(value is not None for value in observed_values)
            and observed_values[0] != observed_values[1]
        )
        correct_sides += side_count
        complete_units += complete
        prediction_changed_units += changed
        by_family[family]["unit_count"] += 1
        by_family[family]["correct_sides"] += side_count
        by_family[family]["complete_units"] += complete
        by_family[family]["prediction_changed_units"] += changed
    if any(values["unit_count"] != 4 for values in by_family.values()):
        raise ValueError("V55 physical families must each contain exactly four units")
    successful_families = sum(
        values["complete_units"] >= 1 for values in by_family.values()
    )
    return {
        "unit_count": EXPECTED_CHANGED_UNIT_COUNT,
        "side_count": EXPECTED_CHANGED_SIDE_COUNT,
        "canonical_correct_sides": correct_sides,
        "canonical_complete_units": complete_units,
        "canonical_prediction_changed_units": prediction_changed_units,
        "physical_change_family_count": len(by_family),
        "physical_change_families_with_complete_unit": successful_families,
        "by_family": {
            family: dict(values) for family, values in sorted(by_family.items())
        },
    }


def _finite_metric(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _prefix_inventory(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    by_scene: defaultdict[str, set[str]] = defaultdict(set)
    for row in predictions:
        scene_id, _question_id = _key(row)
        prefix_hash = row.get("prefix_hash")
        if not isinstance(prefix_hash, str) or _HEX64.fullmatch(prefix_hash) is None:
            raise ValueError("V55 every prediction requires a valid prefix hash")
        by_scene[scene_id].add(prefix_hash)
    if set(by_scene) != set(EXPECTED_SCENE_IDS) or any(
        len(values) != 1 for values in by_scene.values()
    ):
        raise ValueError("V55 prefix hashes are not invariant within all six scenes")
    return {
        scene_id: next(iter(by_scene[scene_id])) for scene_id in EXPECTED_SCENE_IDS
    }


def score_development(
    references_path: str | Path,
    predictions_path: str | Path,
    baseline_path: str | Path = DEFAULT_BASELINE,
    baseline_predictions_path: str | Path = DEFAULT_BASELINE_PREDICTIONS,
) -> dict[str, Any]:
    references_source = _resolve(references_path)
    predictions_source = _resolve(predictions_path)
    baseline_source = _resolve(baseline_path)
    baseline_predictions_source = _resolve(baseline_predictions_path)
    for label, source in (
        ("references", references_source),
        ("predictions", predictions_source),
    ):
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(f"V55 {label} are unavailable or unsafe: {source}")
    baseline = _authenticate_baseline(baseline_source)
    references_sha256 = _sha256(references_source)
    if references_sha256 != REFERENCE_SHA256:
        raise ValueError("V55 current validation references differ from sealed V29")
    references = load_jsonl(references_source)
    predictions = load_jsonl(predictions_source)
    if len(references) != EXPECTED_REFERENCE_COUNT or len(predictions) != EXPECTED_REFERENCE_COUNT:
        raise ValueError("V55 requires exactly 216 references and 216 predictions")
    if {str(row.get("scene_id")) for row in references} != set(EXPECTED_SCENE_IDS):
        raise ValueError("V55 references are not exactly development scenes 19--24")
    type_counts = Counter(str(row.get("answer_type")) for row in references)
    if dict(sorted(type_counts.items())) != EXPECTED_TYPE_COUNTS:
        raise ValueError("V55 development answer-type inventory changed")
    baseline_predictions = _authenticate_baseline_predictions(
        baseline_predictions_source,
        references,
    )
    prediction_index = _prediction_index(predictions)
    reference_keys = {_key(row) for row in references}
    if set(prediction_index) != reference_keys:
        raise ValueError("V55 prediction keys do not exactly cover development references")
    prefix_hashes = _prefix_inventory(predictions)

    standard = score_predictions(references, predictions)
    canonical_correct = sum(
        canonical_type_specific_match(
            str(reference["answer_type"]),
            _prediction_answer(prediction_index[_key(reference)]),
            reference["answer"],
        )
        for reference in references
    )
    canonical_accuracy = canonical_correct / EXPECTED_REFERENCE_COUNT
    changed = _changed_metrics(references, prediction_index)
    count_metrics = _mapping(standard.get("count"), "count metrics")
    presence_metrics = _mapping(standard.get("presence"), "presence metrics")
    exact_accuracy = _finite_metric(
        standard.get("normalized_exact_accuracy"), "normalized exact accuracy"
    )
    spatial_accuracy = _finite_metric(
        standard.get("spatial_relation_accuracy"), "spatial relation accuracy"
    )
    count_accuracy = _finite_metric(count_metrics.get("accuracy"), "count accuracy")
    presence_f1 = _finite_metric(presence_metrics.get("f1"), "presence F1")
    gates = {
        "full_validation_coverage_216": (
            standard.get("reference_count") == EXPECTED_REFERENCE_COUNT
            and standard.get("prediction_count") == EXPECTED_REFERENCE_COUNT
            and standard.get("missing_prediction_count") == 0
            and standard.get("extra_prediction_count") == 0
        ),
        "normalized_exact_accuracy_at_least_0_375": (
            exact_accuracy >= MIN_EXACT_ACCURACY
        ),
        "spatial_relation_accuracy_at_least_0_55": (
            spatial_accuracy >= MIN_SPATIAL_RELATION_ACCURACY
        ),
        "count_accuracy_at_least_0_80": count_accuracy >= MIN_COUNT_ACCURACY,
        "presence_f1_at_least_0_15": presence_f1 >= MIN_PRESENCE_F1,
        "canonical_changed_complete_units_at_least_2_of_12": (
            changed["canonical_complete_units"] >= MIN_CANONICAL_COMPLETE_UNITS
        ),
        "canonical_changed_correct_sides_at_least_12_of_24": (
            changed["canonical_correct_sides"] >= MIN_CANONICAL_CORRECT_SIDES
        ),
        "canonical_prediction_changed_units_at_least_2_of_12": (
            changed["canonical_prediction_changed_units"]
            >= MIN_CANONICAL_CHANGED_UNITS
        ),
        "physical_change_families_at_least_2_of_3": (
            changed["physical_change_families_with_complete_unit"]
            >= MIN_PHYSICAL_CHANGE_FAMILIES
        ),
        "canonical_aggregate_correct_at_least_v29_91_of_216": (
            canonical_correct >= MIN_CANONICAL_AGGREGATE_CORRECT
        ),
    }
    compact_standard = {
        key: standard[key]
        for key in (
            "reference_count",
            "prediction_count",
            "matched_prediction_count",
            "missing_prediction_count",
            "extra_prediction_count",
            "normalized_exact_accuracy",
            "list_order_insensitive_accuracy",
            "count",
            "spatial_relation_accuracy",
            "presence",
            "grounding",
            "per_type",
            "counterfactual",
        )
    }
    return {
        "schema_version": 1,
        "artifact": AUTHORIZATION_ID,
        "passed": all(gates.values()),
        "scope": {
            "split": "validation",
            "scene_ids": list(EXPECTED_SCENE_IDS),
            "final_test_scenes_touched": False,
            "oracle_loaded": False,
            "model_loaded": False,
            "map_loaded": False,
            "question_or_answer_text_serialized": False,
        },
        "inputs": {
            "references_path": str(DEFAULT_REFERENCES),
            "references_sha256": references_sha256,
            "predictions_path": str(DEFAULT_PREDICTIONS),
            "predictions_sha256": _sha256(predictions_source),
            "baseline": baseline,
            "baseline_predictions": baseline_predictions,
        },
        "prefix_sha256_by_scene": prefix_hashes,
        "standard_metrics": compact_standard,
        "canonical_type_specific": {
            "correct": canonical_correct,
            "total": EXPECTED_REFERENCE_COUNT,
            "accuracy": canonical_accuracy,
            "scorer": (
                "presence/count/spatial/list canonicalization; normalized exact otherwise"
            ),
        },
        "changed_counterfactual": changed,
        "thresholds": {
            "normalized_exact_accuracy": MIN_EXACT_ACCURACY,
            "spatial_relation_accuracy": MIN_SPATIAL_RELATION_ACCURACY,
            "count_accuracy": MIN_COUNT_ACCURACY,
            "presence_f1": MIN_PRESENCE_F1,
            "canonical_complete_units": MIN_CANONICAL_COMPLETE_UNITS,
            "canonical_correct_sides": MIN_CANONICAL_CORRECT_SIDES,
            "canonical_prediction_changed_units": MIN_CANONICAL_CHANGED_UNITS,
            "physical_change_families": MIN_PHYSICAL_CHANGE_FAMILIES,
            "canonical_aggregate_correct": MIN_CANONICAL_AGGREGATE_CORRECT,
            "canonical_aggregate_accuracy": BASELINE_CANONICAL_ACCURACY,
        },
        "gates": gates,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        default=DEFAULT_BASELINE_PREDICTIONS,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = _resolve(args.output)
    if output.is_symlink() or output.exists():
        raise FileExistsError(f"V55 score report is one-shot: {output}")
    report = score_development(
        args.references,
        args.predictions,
        args.baseline,
        args.baseline_predictions,
    )
    _atomic_json(output, report)
    print(
        json.dumps(
            {
                "artifact": report["artifact"],
                "passed": report["passed"],
                "normalized_exact_accuracy": report["standard_metrics"][
                    "normalized_exact_accuracy"
                ],
                "canonical_accuracy": report["canonical_type_specific"]["accuracy"],
                "canonical_complete_units": report["changed_counterfactual"][
                    "canonical_complete_units"
                ],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    # A scientifically valid below-threshold result is not a process failure.
    # The selector, rather than this answer-bearing scorer subprocess, owns the
    # one-candidate promotion decision.
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AUTHORIZATION_ID",
    "canonical_answer_key",
    "canonical_type_specific_match",
    "main",
    "score_development",
]
