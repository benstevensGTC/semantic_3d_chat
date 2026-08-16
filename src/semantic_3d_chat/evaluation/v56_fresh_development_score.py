"""Score the sealed V56 fresh-development run without loading a model.

This is the only V56 evaluation process allowed to open answer-bearing fresh
development references.  Its output contains aggregate measurements and
content hashes only; it never serializes questions, answers, predictions, or
environmental descriptions.
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

ARTIFACT: Final[str] = "v56_sealed_fresh_development_score"
TERMINAL_ARTIFACT: Final[str] = "v56_sealed_fresh_development_terminal"
AUTHORIZATION_ID: Final[str] = "v56_one_shot_fresh_development"
EXPECTED_SCENE_IDS: Final[tuple[str, ...]] = tuple(
    f"scene_{index:06d}" for index in range(57, 63)
)
EXPECTED_REFERENCE_COUNT: Final[int] = 216
EXPECTED_SCENE_COUNT: Final[int] = 6
EXPECTED_QUESTIONS_PER_SCENE: Final[int] = 36
EXPECTED_CHANGED_UNIT_COUNT: Final[int] = 12
EXPECTED_CHANGED_SIDE_COUNT: Final[int] = 24
EXPECTED_GROUNDING_TARGET_COUNT: Final[int] = 132
FAMILY_PAIR_IDS: Final[dict[str, str]] = {
    "book_support": "pair_000028",
    "mirror_lr": "pair_000029",
    "picture_support": "pair_000030",
}
FAMILY_SCENE_PAIRS: Final[dict[str, tuple[str, str]]] = {
    "book_support": ("scene_000057", "scene_000058"),
    "mirror_lr": ("scene_000059", "scene_000060"),
    "picture_support": ("scene_000061", "scene_000062"),
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

MIN_EXACT_ACCURACY: Final[float] = 0.42
MIN_CANONICAL_CORRECT: Final[int] = 93
MIN_SPATIAL_RELATION_ACCURACY: Final[float] = 0.60
MIN_COUNT_ACCURACY: Final[float] = 0.80
MIN_PRESENCE_F1: Final[float] = 0.30
MIN_CANONICAL_COMPLETE_UNITS: Final[int] = 2
MIN_CANONICAL_CORRECT_SIDES: Final[int] = 12
MIN_CANONICAL_CHANGED_UNITS: Final[int] = 2
MIN_SUCCESSFUL_FAMILIES: Final[int] = 2

DEFAULT_TERMINAL: Final[Path] = Path(
    "reports/gemma4/metrics/v56_fresh_development_terminal.json"
)
DEFAULT_REFERENCES: Final[Path] = Path("data_diverse52/qa/validation.jsonl")
DEFAULT_PREDICTIONS: Final[Path] = Path(
    "reports/gemma4/predictions/v56_fresh_development_validation.jsonl"
)
DEFAULT_OUTPUT: Final[Path] = Path(
    "reports/gemma4/metrics/v56_fresh_development_score.json"
)
DEFAULT_CLAIM: Final[Path] = Path(
    "reports/gemma4/metrics/v56_fresh_development_launch_claim.json"
)
_HEX64 = re.compile(r"[0-9a-f]{64}")
GATE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "full_validation_coverage_216",
        "three_atomic_fresh_pairs_present",
        "normalized_exact_accuracy_at_least_0_42",
        "canonical_correct_at_least_93_of_216",
        "spatial_relation_accuracy_at_least_0_60",
        "count_accuracy_at_least_0_80",
        "presence_f1_at_least_0_30",
        "canonical_changed_complete_units_at_least_2_of_12",
        "canonical_changed_correct_sides_at_least_12_of_24",
        "canonical_prediction_changed_units_at_least_2_of_12",
        "physical_change_families_at_least_2_of_3",
    }
)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    rooted = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(rooted))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_symlink_components(path: Path, field: str) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"V56 {field} path contains a symbolic link: {current}")


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"V56 {field} must be a mapping")
    return value


def _key(record: Mapping[str, Any]) -> tuple[str, str]:
    scene_id = record.get("scene_id")
    question_id = record.get("question_id")
    if not isinstance(scene_id, str) or not isinstance(question_id, str):
        raise TypeError("V56 records require opaque scene and question identifiers")
    return scene_id, question_id


def _prediction_answer(record: Mapping[str, Any] | None) -> Any:
    return None if record is None else record.get("predicted_answer")


def canonical_answer_key(answer_type: str, value: Any) -> object | None:
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
    if answer_type in LIST_ANSWER_TYPES:
        return list_order_insensitive_match(prediction, reference)
    expected = canonical_answer_key(answer_type, reference)
    observed = canonical_answer_key(answer_type, prediction)
    if answer_type in {"presence", "count", "spatial_relation"}:
        return expected is not None and observed == expected
    return exact_normalized_match(prediction, reference)


def _prediction_index(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in predictions:
        key = _key(row)
        if key in indexed:
            raise ValueError(f"Duplicate V56 prediction key: {key}")
        indexed[key] = row
    return indexed


def _load_terminal(path: str | Path, expected_sha256: str) -> Mapping[str, Any]:
    if not isinstance(expected_sha256, str) or _HEX64.fullmatch(expected_sha256) is None:
        raise ValueError("V56 scorer requires an explicit terminal SHA-256")
    source = _resolve(path)
    if source != _resolve(DEFAULT_TERMINAL):
        raise ValueError("V56 scorer terminal path is pinned")
    _reject_symlink_components(source, "terminal")
    if not source.is_file():
        raise FileNotFoundError(f"V56 terminal is unavailable or unsafe: {source}")
    if _sha256(source) != expected_sha256:
        raise ValueError("V56 scorer terminal differs from the explicit digest")
    terminal = _mapping(json.loads(source.read_text(encoding="utf-8")), "terminal")
    authorization = _mapping(terminal.get("authorization"), "terminal authorization")
    thresholds = _mapping(authorization.get("thresholds"), "terminal thresholds")
    if (
        terminal.get("schema_version") != 1
        or terminal.get("artifact") != TERMINAL_ARTIFACT
        or terminal.get("passed") is not True
        or authorization.get("authorization_id") != AUTHORIZATION_ID
        or authorization.get("only_exact_action")
        != "one_control_one_shot_fresh_development"
        or authorization.get("explicit_terminal_sha256_required") is not True
        or dict(thresholds) != threshold_contract()
    ):
        raise ValueError("V56 scorer terminal authorization changed")
    return terminal


def threshold_contract() -> dict[str, int | float]:
    """Return the preregistered gate, independent of any prediction bytes."""

    return {
        "normalized_exact_accuracy_minimum": MIN_EXACT_ACCURACY,
        "canonical_correct_minimum": MIN_CANONICAL_CORRECT,
        "spatial_relation_accuracy_minimum": MIN_SPATIAL_RELATION_ACCURACY,
        "count_accuracy_minimum": MIN_COUNT_ACCURACY,
        "presence_f1_minimum": MIN_PRESENCE_F1,
        "canonical_complete_units_minimum": MIN_CANONICAL_COMPLETE_UNITS,
        "canonical_correct_sides_minimum": MIN_CANONICAL_CORRECT_SIDES,
        "canonical_prediction_changed_units_minimum": MIN_CANONICAL_CHANGED_UNITS,
        "successful_physical_change_families_minimum": MIN_SUCCESSFUL_FAMILIES,
    }


def _prefix_inventory(predictions: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    by_scene: defaultdict[str, set[str]] = defaultdict(set)
    for row in predictions:
        scene_id, _ = _key(row)
        prefix = row.get("prefix_hash")
        if not isinstance(prefix, str) or _HEX64.fullmatch(prefix) is None:
            raise ValueError("V56 every prediction requires a valid prefix hash")
        by_scene[scene_id].add(prefix)
    if set(by_scene) != set(EXPECTED_SCENE_IDS) or any(
        len(values) != 1 for values in by_scene.values()
    ):
        raise ValueError("V56 prefix hashes are not invariant within all fresh scenes")
    return {
        scene_id: next(iter(by_scene[scene_id])) for scene_id in EXPECTED_SCENE_IDS
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
            raise TypeError("V56 changed reference lacks opaque pair metadata")
        grouped[(pair_id, question_key)].append(reference)
    if len(grouped) != EXPECTED_CHANGED_UNIT_COUNT:
        raise ValueError("V56 fresh split must contain exactly twelve changed units")

    pair_to_family = {pair_id: family for family, pair_id in FAMILY_PAIR_IDS.items()}
    by_family: dict[str, Counter[str]] = {
        family: Counter() for family in FAMILY_PAIR_IDS
    }
    correct_sides = 0
    complete_units = 0
    changed_units = 0
    for (pair_id, _), rows in sorted(grouped.items()):
        family = pair_to_family.get(pair_id)
        if family is None or len(rows) != 2:
            raise ValueError("V56 changed unit is outside the three sealed atomic pairs")
        observed_scenes = tuple(sorted(_key(row)[0] for row in rows))
        if observed_scenes != tuple(sorted(FAMILY_SCENE_PAIRS[family])):
            raise ValueError("V56 changed unit uses the wrong atomic scene pair")
        answer_types = {str(row.get("answer_type")) for row in rows}
        if len(answer_types) != 1:
            raise ValueError("V56 changed unit answer types disagree")
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
        expected_values = [
            canonical_answer_key(answer_type, reference.get("answer"))
            for reference in rows
        ]
        if (
            any(value is None for value in expected_values)
            or expected_values[0] == expected_values[1]
        ):
            raise ValueError("V56 changed unit references do not encode a changed fact")
        side_count = sum(correctness)
        complete = int(all(correctness))
        changed = int(
            all(value is not None for value in observed_values)
            and observed_values[0] != observed_values[1]
        )
        correct_sides += side_count
        complete_units += complete
        changed_units += changed
        by_family[family]["unit_count"] += 1
        by_family[family]["correct_sides"] += side_count
        by_family[family]["complete_units"] += complete
        by_family[family]["prediction_changed_units"] += changed
    if any(values["unit_count"] != 4 for values in by_family.values()):
        raise ValueError("V56 each physical-change family must contain four changed units")
    successful_families = sum(
        values["complete_units"] >= 1 for values in by_family.values()
    )
    return {
        "atomic_scene_pair_count": len(FAMILY_PAIR_IDS),
        "unit_count": EXPECTED_CHANGED_UNIT_COUNT,
        "side_count": EXPECTED_CHANGED_SIDE_COUNT,
        "canonical_correct_sides": correct_sides,
        "canonical_complete_units": complete_units,
        "canonical_prediction_changed_units": changed_units,
        "physical_change_family_count": len(by_family),
        "physical_change_families_with_complete_unit": successful_families,
        "by_family": {
            family: dict(values) for family, values in sorted(by_family.items())
        },
    }


def _finite_metric(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"V56 {field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"V56 {field} must be finite")
    return result


def score_development(
    references_path: str | Path,
    predictions_path: str | Path,
    *,
    terminal_path: str | Path,
    expected_terminal_sha256: str,
) -> dict[str, Any]:
    terminal = _load_terminal(terminal_path, expected_terminal_sha256)
    authorization = _mapping(terminal["authorization"], "terminal authorization")
    development = _mapping(authorization.get("development"), "development contract")
    outputs = _mapping(authorization.get("outputs"), "output contract")
    references_source = _resolve(references_path)
    predictions_source = _resolve(predictions_path)
    if references_source != _resolve(DEFAULT_REFERENCES):
        raise ValueError("V56 scorer reference path is pinned")
    if predictions_source != _resolve(DEFAULT_PREDICTIONS):
        raise ValueError("V56 scorer prediction path is pinned")
    if development.get("reference_path") != str(DEFAULT_REFERENCES):
        raise ValueError("V56 terminal reference path is not the sealed path")
    if outputs.get("predictions") != str(DEFAULT_PREDICTIONS):
        raise ValueError("V56 terminal prediction path is not the sealed path")
    if references_source != _resolve(str(development.get("reference_path"))):
        raise ValueError("V56 scorer reference path differs from the terminal")
    if predictions_source != _resolve(str(outputs.get("predictions"))):
        raise ValueError("V56 scorer prediction path differs from the terminal")
    for label, source in (
        ("references", references_source),
        ("predictions", predictions_source),
    ):
        _reject_symlink_components(source, label)
        if not source.is_file():
            raise FileNotFoundError(f"V56 {label} are unavailable or unsafe: {source}")
    reference_sha256 = _sha256(references_source)
    if reference_sha256 != development.get("reference_sha256"):
        raise ValueError("V56 fresh-development references changed after sealing")

    references = load_jsonl(references_source)
    predictions = load_jsonl(predictions_source)
    if len(references) != EXPECTED_REFERENCE_COUNT or len(predictions) != (
        EXPECTED_REFERENCE_COUNT
    ):
        raise ValueError("V56 requires exactly 216 references and predictions")
    scene_counts = Counter(str(row.get("scene_id")) for row in references)
    if scene_counts != Counter(
        {scene_id: EXPECTED_QUESTIONS_PER_SCENE for scene_id in EXPECTED_SCENE_IDS}
    ):
        raise ValueError("V56 references are not exactly 36 questions on each fresh scene")
    type_counts = Counter(str(row.get("answer_type")) for row in references)
    if dict(sorted(type_counts.items())) != EXPECTED_TYPE_COUNTS:
        raise ValueError("V56 fresh-development answer-type inventory changed")
    grounding_targets = 0
    for row in references:
        target = row.get("target_xyz")
        if target is None:
            continue
        if (
            not isinstance(target, list)
            or len(target) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in target
            )
        ):
            raise ValueError("V56 fresh-development grounding target is invalid")
        grounding_targets += 1
    if grounding_targets != EXPECTED_GROUNDING_TARGET_COUNT:
        raise ValueError("V56 fresh-development grounding-target inventory changed")

    prediction_index = _prediction_index(predictions)
    reference_keys = {_key(row) for row in references}
    if set(prediction_index) != reference_keys:
        raise ValueError("V56 predictions do not exactly cover fresh references")
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
    changed = _changed_metrics(references, prediction_index)
    count_metrics = _mapping(standard.get("count"), "count metrics")
    presence_metrics = _mapping(standard.get("presence"), "presence metrics")
    grounding_metrics = _mapping(standard.get("grounding"), "grounding metrics")
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
        "three_atomic_fresh_pairs_present": changed["atomic_scene_pair_count"] == 3,
        "normalized_exact_accuracy_at_least_0_42": (
            exact_accuracy >= MIN_EXACT_ACCURACY
        ),
        "canonical_correct_at_least_93_of_216": (
            canonical_correct >= MIN_CANONICAL_CORRECT
        ),
        "spatial_relation_accuracy_at_least_0_60": (
            spatial_accuracy >= MIN_SPATIAL_RELATION_ACCURACY
        ),
        "count_accuracy_at_least_0_80": count_accuracy >= MIN_COUNT_ACCURACY,
        "presence_f1_at_least_0_30": presence_f1 >= MIN_PRESENCE_F1,
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
            >= MIN_SUCCESSFUL_FAMILIES
        ),
    }
    if set(gates) != GATE_KEYS or not all(isinstance(value, bool) for value in gates.values()):
        raise AssertionError("V56 preregistered gate implementation changed")
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
    if grounding_metrics.get("target_count") != EXPECTED_GROUNDING_TARGET_COUNT:
        raise ValueError("V56 grounding metrics lost sealed targets")
    return {
        "schema_version": 1,
        "artifact": ARTIFACT,
        "passed": all(gates.values()),
        "scope": {
            "split": "validation",
            "scene_ids": list(EXPECTED_SCENE_IDS),
            "scene_count": EXPECTED_SCENE_COUNT,
            "atomic_scene_pair_count": 3,
            "deferred_final_scenes_touched": False,
            "simulator_oracle_loaded": False,
            "model_loaded": False,
            "map_loaded": False,
            "answer_references_loaded_by_isolated_scorer": True,
            "question_or_answer_text_serialized": False,
        },
        "inputs": {
            "terminal_path": str(DEFAULT_TERMINAL),
            "terminal_sha256": expected_terminal_sha256,
            "references_path": str(DEFAULT_REFERENCES),
            "references_sha256": reference_sha256,
            "predictions_path": str(DEFAULT_PREDICTIONS),
            "predictions_sha256": _sha256(predictions_source),
        },
        "prefix_sha256_by_scene": prefix_hashes,
        "standard_metrics": compact_standard,
        "canonical_type_specific": {
            "correct": canonical_correct,
            "total": EXPECTED_REFERENCE_COUNT,
            "accuracy": canonical_correct / EXPECTED_REFERENCE_COUNT,
            "scorer": (
                "presence/count/spatial/list canonicalization; normalized exact otherwise"
            ),
        },
        "changed_counterfactual": changed,
        "grounding": dict(grounding_metrics),
        "thresholds": threshold_contract(),
        "gates": gates,
    }


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = _resolve(path)
    _reject_symlink_components(destination, "score output")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"V56 score report is immutable: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _require_launch_claim(expected_terminal_sha256: str) -> None:
    """Prove the selector's permanent claim exists before opening references."""

    source = _resolve(DEFAULT_CLAIM)
    _reject_symlink_components(source, "launch claim")
    if not source.is_file():
        raise FileNotFoundError("V56 scorer requires the permanent launch claim")
    claim = _mapping(json.loads(source.read_text(encoding="utf-8")), "launch claim")
    if (
        claim.get("schema_version") != 1
        or claim.get("artifact")
        != "v56_permanent_fresh_development_launch_claim"
        or claim.get("authorization_id") != AUTHORIZATION_ID
        or claim.get("terminal_sha256") != expected_terminal_sha256
        or claim.get("reference_path") != str(DEFAULT_REFERENCES)
        or claim.get("prediction_path") != str(DEFAULT_PREDICTIONS)
        or claim.get("one_candidate_only") is not True
        or claim.get("crash_resume_same_artifacts_only") is not True
        or claim.get("outputs_may_not_be_cleared_or_overwritten") is not True
    ):
        raise ValueError("V56 scorer launch claim contract changed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal", type=Path, default=DEFAULT_TERMINAL)
    parser.add_argument("--expected-terminal-sha256", required=True)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if _resolve(args.terminal) != _resolve(DEFAULT_TERMINAL):
        raise ValueError("V56 scorer terminal path is pinned")
    if _resolve(args.references) != _resolve(DEFAULT_REFERENCES):
        raise ValueError("V56 scorer reference path is pinned")
    if _resolve(args.predictions) != _resolve(DEFAULT_PREDICTIONS):
        raise ValueError("V56 scorer prediction path is pinned")
    output = _resolve(args.output)
    if output != _resolve(DEFAULT_OUTPUT):
        raise ValueError("V56 scorer output path is pinned")
    _reject_symlink_components(output, "score output")
    if output.is_symlink() or output.exists():
        raise FileExistsError(f"V56 score report is immutable and already exists: {output}")
    _require_launch_claim(args.expected_terminal_sha256)
    report = score_development(
        args.references,
        args.predictions,
        terminal_path=args.terminal,
        expected_terminal_sha256=args.expected_terminal_sha256,
    )
    _atomic_create_json(output, report)
    print(
        json.dumps(
            {
                "artifact": ARTIFACT,
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
    # A valid below-gate result is evidence, not a scorer process failure.
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ARTIFACT",
    "GATE_KEYS",
    "canonical_answer_key",
    "canonical_type_specific_match",
    "main",
    "score_development",
    "threshold_contract",
]
