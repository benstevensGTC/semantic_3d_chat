"""Preregister and score the V59 authorized multi-scene training gate.

This is deliberately a *training-only* intermediate gate.  It locks six
authorized scenes before candidate training: the proven V58 anchor pair plus a
mirrored-room pair and an object-removal pair.  No development or final scene
is accepted by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.evaluation.metrics import (
    canonical_presence,
    canonical_relation,
    exact_normalized_match,
    extract_count,
    list_order_insensitive_match,
    normalize_answer,
    normalize_answer_items,
)
from semantic_3d_chat.evaluation.question_manifest import (
    build_question_manifest,
    load_question_manifest,
    sha256_file,
)

LOCKED_SCENE_IDS: Final[tuple[str, ...]] = (
    "scene_000031",
    "scene_000032",
    "scene_000033",
    "scene_000034",
    "scene_000037",
    "scene_000038",
)
ANCHOR_PAIR_ID: Final[str] = "pair_000015"
EXPANSION_PAIR_IDS: Final[tuple[str, ...]] = ("pair_000016", "pair_000018")
_AUTHORIZED_TRAIN_NUMBERS: Final[frozenset[int]] = frozenset(
    (*range(11, 25), *range(31, 57))
)


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_train_qa(path: str | Path) -> tuple[list[dict[str, Any]], str]:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.suffix.casefold() != ".jsonl":
        raise FileNotFoundError(f"V59 training QA is unavailable: {source}")
    forbidden = {"oracle", "validation", "development", "test", "final", "v55"}
    try:
        scoped = source.relative_to(Path(__file__).resolve().parents[3])
    except ValueError:
        scoped = Path(source.name)
    tokens = {
        token
        for part in scoped.parts
        for token in part.casefold().replace("-", "_").split("_")
    }
    if forbidden & tokens:
        raise ValueError("V59 training QA path contains a forbidden split token")
    rows: list[dict[str, Any]] = []
    keys: set[tuple[str, str]] = set()
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"V59 QA row {line_number} must be an object")
        scene_id = value.get("scene_id")
        question_id = value.get("question_id")
        if not isinstance(scene_id, str) or not scene_id.startswith("scene_"):
            raise ValueError(f"V59 QA row {line_number} has an invalid scene ID")
        try:
            scene_number = int(scene_id.removeprefix("scene_"))
        except ValueError as exc:
            raise ValueError(f"V59 QA row {line_number} has an invalid scene ID") from exc
        if scene_number not in _AUTHORIZED_TRAIN_NUMBERS:
            raise ValueError(f"V59 QA opened a non-training scene: {scene_id}")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"V59 QA row {line_number} has an invalid question ID")
        key = scene_id, question_id
        if key in keys:
            raise ValueError(f"V59 QA contains duplicate opaque key: {key}")
        keys.add(key)
        rows.append(value)
    if not rows:
        raise ValueError("V59 training QA is empty")
    return rows, sha256_file(source)


def locked_gate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Select changed and retention rows without inspecting model predictions."""

    selected = [dict(row) for row in rows if row.get("scene_id") in LOCKED_SCENE_IDS]
    if {row["scene_id"] for row in selected} != set(LOCKED_SCENE_IDS):
        raise ValueError("V59 QA does not cover every locked training scene")
    changed = [row for row in selected if row.get("counterfactual_expected_change") is True]
    anchor = [row for row in changed if row.get("counterfactual_pair_id") == ANCHOR_PAIR_ID]
    expansion = [
        row
        for row in changed
        if row.get("counterfactual_pair_id") in EXPANSION_PAIR_IDS
    ]
    unexpected = [row for row in changed if row not in anchor and row not in expansion]
    if unexpected or len(anchor) != 8 or len(expansion) != 14:
        raise ValueError(
            "V59 locked changed inventory mismatch: "
            f"anchor={len(anchor)} expansion={len(expansion)} unexpected={len(unexpected)}"
        )
    # Preserve the complete ordinary-QA inventory from the accepted anchor
    # pair.  V54's exact no-control path scores 15/40 here; a learned route
    # gate must retain that behavior rather than merely imitate V58's degraded
    # 9/40 control-token replay.
    retention = [
        row
        for row in selected
        if row["scene_id"] in {"scene_000031", "scene_000032"}
        and row.get("counterfactual_expected_change") is not True
    ]
    if len(retention) != 40:
        raise AssertionError("V59 retention lock must contain the complete anchor 40")
    return {
        "anchor_changed": sorted(anchor, key=lambda row: (row["scene_id"], row["question_id"])),
        "expansion_changed": sorted(
            expansion, key=lambda row: (row["scene_id"], row["question_id"])
        ),
        "retention": sorted(retention, key=lambda row: (row["scene_id"], row["question_id"])),
    }


def _opaque_inventory(groups: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    return {
        name: [
            {
                "scene_id": row["scene_id"],
                "question_id": row["question_id"],
                "answer_type": row["answer_type"],
                **(
                    {
                        "pair_id": row["counterfactual_pair_id"],
                        "pair_question_key": row["counterfactual_question_key"],
                    }
                    if name != "retention"
                    else {}
                ),
            }
            for row in values
        ]
        for name, values in groups.items()
    }


def prepare_gate(
    *,
    train_qa: str | Path,
    questions_output: str | Path,
    preregistration_output: str | Path,
) -> dict[str, Any]:
    rows, qa_sha256 = _read_train_qa(train_qa)
    groups = locked_gate_rows(rows)
    ordered = [
        *groups["anchor_changed"],
        *groups["expansion_changed"],
        *groups["retention"],
    ]
    manifest = build_question_manifest(ordered, source_qa_sha256=qa_sha256)
    questions_path = Path(questions_output).expanduser().resolve()
    prereg_path = Path(preregistration_output).expanduser().resolve()
    for destination in (questions_path, prereg_path):
        if destination.exists():
            raise FileExistsError(f"V59 preregistration output exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
    question_payload = manifest.as_dict()
    questions_path.write_text(
        json.dumps(question_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    verified = load_question_manifest(questions_path)
    inventory = _opaque_inventory(groups)
    preregistration = {
        "schema_version": 1,
        "artifact": "v59_authorized_multiscene_train_preregistration",
        "status": "awaiting_baseline",
        "locked_scene_ids": list(LOCKED_SCENE_IDS),
        "anchor_pair_id": ANCHOR_PAIR_ID,
        "expansion_pair_ids": list(EXPANSION_PAIR_IDS),
        "training_qa_sha256": qa_sha256,
        "questions_manifest_sha256": verified.manifest_sha256,
        "questions_sha256": verified.questions_sha256,
        "counts": {name: len(values) for name, values in groups.items()},
        "opaque_inventory": inventory,
        "opaque_inventory_sha256": _sha256_json(inventory),
        "threshold_policy": {
            "anchor_exact_required": 8,
            "anchor_complete_units_required": 4,
            "anchor_changed_units_required": 4,
            "expansion_minimum_exact_fraction": 0.6,
            "expansion_minimum_exact_improvement": 4,
            "expansion_minimum_complete_unit_improvement": 2,
            "expansion_minimum_changed_unit_improvement": 2,
            "retention_allowed_exact_loss": 0,
            "retention_allowed_canonical_loss": 0,
            "maximum_control_rms": 0.2,
        },
        "prohibited_scene_numbers": [*range(25, 31), *range(57, 63)],
        "fresh_development_loaded": False,
        "deferred_final_loaded": False,
    }
    prereg_path.write_text(
        json.dumps(preregistration, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return preregistration


def _prediction_rows(path: str | Path) -> tuple[dict[tuple[str, str], dict[str, Any]], str]:
    source = Path(path).expanduser().resolve()
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("scene_id"), row.get("question_id")
        if not all(isinstance(value, str) and value for value in key):
            raise ValueError("V59 prediction lacks an opaque key")
        if key in rows:
            raise ValueError(f"V59 prediction contains duplicate key: {key}")
        rows[key] = row
    return rows, sha256_file(source)


def _canonical_value(answer: Any, answer_type: str) -> object:
    if answer_type == "presence":
        return canonical_presence(answer)
    if answer_type == "count":
        return extract_count(answer)
    if answer_type == "spatial_relation":
        return canonical_relation(answer)
    if answer_type in {"list", "containment"}:
        return tuple(sorted(normalize_answer_items(answer)))
    if answer_type == "support":
        relation = canonical_relation(answer)
        items = tuple(sorted(normalize_answer_items(answer)))
        return relation if relation is not None else items
    return normalize_answer(answer)


def _canonical_correct(prediction: Any, reference: Any, answer_type: str) -> bool:
    if answer_type in {"list", "containment"}:
        return list_order_insensitive_match(prediction, reference)
    predicted = _canonical_value(prediction, answer_type)
    expected = _canonical_value(reference, answer_type)
    return expected not in {None, "", ()} and predicted == expected


def _group_metrics(
    references: Sequence[Mapping[str, Any]],
    predictions: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    paired: bool,
) -> dict[str, int]:
    rows = []
    for reference in references:
        key = str(reference["scene_id"]), str(reference["question_id"])
        if key not in predictions:
            raise ValueError(f"V59 prediction is missing locked key: {key}")
        prediction = predictions[key].get("predicted_answer")
        rows.append(
            {
                "reference": reference,
                "prediction": prediction,
                "exact": exact_normalized_match(prediction, reference["answer"]),
                "canonical": _canonical_correct(
                    prediction, reference["answer"], str(reference["answer_type"])
                ),
            }
        )
    result = {
        "total": len(rows),
        "exact": sum(row["exact"] for row in rows),
        "canonical": sum(row["canonical"] for row in rows),
    }
    if paired:
        units: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            reference = row["reference"]
            units[
                (
                    str(reference["counterfactual_pair_id"]),
                    str(reference["counterfactual_question_key"]),
                )
            ].append(row)
        if any(len(values) != 2 for values in units.values()):
            raise ValueError("V59 changed gate has an incomplete pair unit")
        result.update(
            {
                "unit_total": len(units),
                "complete_units": sum(
                    all(row["canonical"] for row in values) for values in units.values()
                ),
                "changed_units": sum(
                    len(
                        {
                            json.dumps(
                                _canonical_value(
                                    row["prediction"],
                                    str(row["reference"]["answer_type"]),
                                ),
                                sort_keys=True,
                            )
                            for row in values
                        }
                    )
                    == 2
                    for values in units.values()
                ),
            }
        )
    return result


def _correct_keys(
    references: Sequence[Mapping[str, Any]],
    predictions: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[list[str]]:
    result = []
    for reference in references:
        key = str(reference["scene_id"]), str(reference["question_id"])
        prediction = predictions[key].get("predicted_answer")
        if exact_normalized_match(prediction, reference["answer"]):
            result.append([*key])
    return sorted(result)


def _opaque_output_hashes(
    references: Sequence[Mapping[str, Any]],
    predictions: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, str]]:
    result = []
    for reference in references:
        key = str(reference["scene_id"]), str(reference["question_id"])
        prediction = predictions[key].get("predicted_answer")
        if not isinstance(prediction, str):
            raise TypeError("V59 locked baseline prediction must be text")
        result.append(
            {
                "scene_id": key[0],
                "question_id": key[1],
                "raw_output_sha256": hashlib.sha256(prediction.encode()).hexdigest(),
            }
        )
    return sorted(result, key=lambda row: (row["scene_id"], row["question_id"]))


def score_predictions(
    *,
    train_qa: str | Path,
    predictions: str | Path,
    preregistration: str | Path,
) -> dict[str, Any]:
    rows, qa_sha256 = _read_train_qa(train_qa)
    groups = locked_gate_rows(rows)
    prereg_path = Path(preregistration).expanduser().resolve()
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if (
        prereg.get("artifact") != "v59_authorized_multiscene_train_preregistration"
        or prereg.get("training_qa_sha256") != qa_sha256
        or prereg.get("locked_scene_ids") != list(LOCKED_SCENE_IDS)
        or prereg.get("opaque_inventory_sha256")
        != _sha256_json(_opaque_inventory(groups))
    ):
        raise ValueError("V59 preregistration does not match the locked QA inventory")
    prediction_rows, prediction_sha256 = _prediction_rows(predictions)
    locked_keys = {
        (str(row["scene_id"]), str(row["question_id"]))
        for values in groups.values()
        for row in values
    }
    if set(prediction_rows) != locked_keys:
        raise ValueError("V59 prediction inventory differs from preregistered questions")
    return {
        "anchor_changed": _group_metrics(
            groups["anchor_changed"], prediction_rows, paired=True
        ),
        "expansion_changed": _group_metrics(
            groups["expansion_changed"], prediction_rows, paired=True
        ),
        "retention": _group_metrics(groups["retention"], prediction_rows, paired=False),
        "predictions_sha256": prediction_sha256,
    }


def seal_baseline(
    *,
    train_qa: str | Path,
    source_predictions: str | Path,
    no_control_predictions: str | Path,
    preregistration: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    prereg_path = Path(preregistration).expanduser().resolve()
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    source_metrics = score_predictions(
        train_qa=train_qa,
        predictions=source_predictions,
        preregistration=prereg_path,
    )
    rows, _qa_sha256 = _read_train_qa(train_qa)
    groups = locked_gate_rows(rows)
    no_control_rows, no_control_sha256 = _prediction_rows(no_control_predictions)
    retention_keys = {
        (str(row["scene_id"]), str(row["question_id"]))
        for row in groups["retention"]
    }
    if not retention_keys.issubset(no_control_rows):
        raise ValueError("V59 no-control baseline is missing locked retention keys")
    no_control_retention = _group_metrics(
        groups["retention"], no_control_rows, paired=False
    )
    required_retention_keys = _correct_keys(groups["retention"], no_control_rows)
    retention_output_hashes = _opaque_output_hashes(
        groups["retention"], no_control_rows
    )
    policy = prereg["threshold_policy"]
    expansion = source_metrics["expansion_changed"]
    retention = no_control_retention
    thresholds = {
        "anchor_exact": int(policy["anchor_exact_required"]),
        "anchor_canonical": int(policy["anchor_exact_required"]),
        "anchor_complete_units": int(policy["anchor_complete_units_required"]),
        "anchor_changed_units": int(policy["anchor_changed_units_required"]),
        "expansion_exact": min(
            expansion["total"],
            max(
                expansion["exact"] + int(policy["expansion_minimum_exact_improvement"]),
                math.ceil(
                    expansion["total"]
                    * float(policy["expansion_minimum_exact_fraction"])
                ),
            ),
        ),
        "expansion_canonical": min(
            expansion["total"],
            max(
                expansion["canonical"]
                + int(policy["expansion_minimum_exact_improvement"]),
                math.ceil(
                    expansion["total"]
                    * float(policy["expansion_minimum_exact_fraction"])
                ),
            ),
        ),
        "expansion_complete_units": min(
            expansion["unit_total"],
            expansion["complete_units"]
            + int(policy["expansion_minimum_complete_unit_improvement"]),
        ),
        "expansion_changed_units": min(
            expansion["unit_total"],
            expansion["changed_units"]
            + int(policy["expansion_minimum_changed_unit_improvement"]),
        ),
        "retention_exact": max(
            0, retention["exact"] - int(policy["retention_allowed_exact_loss"])
        ),
        "retention_canonical": max(
            0,
            retention["canonical"] - int(policy["retention_allowed_canonical_loss"]),
        ),
        "maximum_control_rms": float(policy["maximum_control_rms"]),
    }
    result = {
        "schema_version": 1,
        "artifact": "v59_authorized_multiscene_train_baseline",
        "preregistration_sha256": sha256_file(prereg_path),
        "source_control_baseline": source_metrics,
        "no_control_retention_baseline": {
            **no_control_retention,
            "predictions_sha256": no_control_sha256,
            "required_correct_keys": required_retention_keys,
            "required_correct_keys_sha256": _sha256_json(required_retention_keys),
            "required_output_hashes": retention_output_hashes,
            "required_output_hashes_sha256": _sha256_json(retention_output_hashes),
        },
        "thresholds": thresholds,
        "fresh_development_loaded": False,
        "deferred_final_loaded": False,
    }
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"V59 baseline output exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def evaluate_candidate(
    *,
    train_qa: str | Path,
    predictions: str | Path,
    preregistration: str | Path,
    baseline: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    baseline_path = Path(baseline).expanduser().resolve()
    baseline_value = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline_value.get("preregistration_sha256") != sha256_file(preregistration):
        raise ValueError("V59 baseline belongs to another preregistration")
    metrics = score_predictions(
        train_qa=train_qa,
        predictions=predictions,
        preregistration=preregistration,
    )
    thresholds = baseline_value["thresholds"]
    anchor = metrics["anchor_changed"]
    expansion = metrics["expansion_changed"]
    retention = metrics["retention"]
    rows, _qa_sha256 = _read_train_qa(train_qa)
    groups = locked_gate_rows(rows)
    candidate_rows, _candidate_sha256 = _prediction_rows(predictions)
    candidate_retention_keys = _correct_keys(groups["retention"], candidate_rows)
    required_retention_keys = baseline_value["no_control_retention_baseline"][
        "required_correct_keys"
    ]
    candidate_output_hashes = _opaque_output_hashes(
        groups["retention"], candidate_rows
    )
    required_output_hashes = baseline_value["no_control_retention_baseline"][
        "required_output_hashes"
    ]
    changed_keys = {
        (str(row["scene_id"]), str(row["question_id"]))
        for group in (groups["anchor_changed"], groups["expansion_changed"])
        for row in group
    }
    retention_keys = {
        (str(row["scene_id"]), str(row["question_id"]))
        for row in groups["retention"]
    }
    audits = {key: candidate_rows[key].get("control_audit") for key in candidate_rows}

    def valid_common_audit(value: object) -> bool:
        return bool(
            isinstance(value, Mapping)
            and value.get("architecture") == "bounded_global_scene_question_control_v2"
            and value.get("environment_latent_count") == 256
            and value.get("every_scene_token_influenced_output") is True
            and value.get("question_dependent_scene_retrieval") is False
            and value.get("softmax_scene_attention_used") is False
            and isinstance(value.get("maximum_control_rms"), (int, float))
            and float(value["maximum_control_rms"])
            <= float(baseline_value["thresholds"].get("maximum_control_rms", 0.2))
            + 1e-6
        )
    checks = {
        "anchor_exact": anchor["exact"] >= thresholds["anchor_exact"],
        "anchor_canonical": anchor["canonical"] >= thresholds["anchor_canonical"],
        "anchor_complete_units": (
            anchor["complete_units"] >= thresholds["anchor_complete_units"]
        ),
        "anchor_changed_units": (
            anchor["changed_units"] >= thresholds["anchor_changed_units"]
        ),
        "expansion_exact": expansion["exact"] >= thresholds["expansion_exact"],
        "expansion_canonical": (
            expansion["canonical"] >= thresholds["expansion_canonical"]
        ),
        "expansion_complete_units": (
            expansion["complete_units"] >= thresholds["expansion_complete_units"]
        ),
        "expansion_changed_units": (
            expansion["changed_units"] >= thresholds["expansion_changed_units"]
        ),
        "retention_exact": retention["exact"] >= thresholds["retention_exact"],
        "retention_canonical": (
            retention["canonical"] >= thresholds["retention_canonical"]
        ),
        "retention_preserves_every_no_control_correct_key": set(
            map(tuple, required_retention_keys)
        ).issubset(map(tuple, candidate_retention_keys)),
        "retention_exact_no_control_output_identity": (
            candidate_output_hashes == required_output_hashes
        ),
        "control_audit_global_bounded_no_softmax": all(
            valid_common_audit(value) for value in audits.values()
        ),
        "changed_rows_use_continuous_control": all(
            isinstance(audits[key], Mapping)
            and audits[key].get("control_used") is True
            for key in changed_keys
        ),
        "retention_rows_use_exact_no_control_route": all(
            isinstance(audits[key], Mapping)
            and audits[key].get("control_used") is False
            and audits[key].get("exact_no_control_route") is True
            for key in retention_keys
        ),
    }
    result = {
        "schema_version": 1,
        "artifact": "v59_authorized_multiscene_train_gate",
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "thresholds": thresholds,
        "baseline_sha256": sha256_file(baseline_path),
        "fresh_development_loaded": False,
        "deferred_final_loaded": False,
    }
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"V59 candidate gate output exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--train-qa", required=True)
    prepare.add_argument("--questions-output", required=True)
    prepare.add_argument("--preregistration-output", required=True)
    baseline = subparsers.add_parser("seal-baseline")
    baseline.add_argument("--train-qa", required=True)
    baseline.add_argument("--source-predictions", required=True)
    baseline.add_argument("--no-control-predictions", required=True)
    baseline.add_argument("--preregistration", required=True)
    baseline.add_argument("--output", required=True)
    candidate = subparsers.add_parser("score-candidate")
    candidate.add_argument("--train-qa", required=True)
    candidate.add_argument("--predictions", required=True)
    candidate.add_argument("--preregistration", required=True)
    candidate.add_argument("--baseline", required=True)
    candidate.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_gate(
            train_qa=args.train_qa,
            questions_output=args.questions_output,
            preregistration_output=args.preregistration_output,
        )
    elif args.command == "seal-baseline":
        result = seal_baseline(
            train_qa=args.train_qa,
            source_predictions=args.source_predictions,
            no_control_predictions=args.no_control_predictions,
            preregistration=args.preregistration,
            output=args.output,
        )
    else:
        result = evaluate_candidate(
            train_qa=args.train_qa,
            predictions=args.predictions,
            preregistration=args.preregistration,
            baseline=args.baseline,
            output=args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.get("passed", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANCHOR_PAIR_ID",
    "EXPANSION_PAIR_IDS",
    "LOCKED_SCENE_IDS",
    "evaluate_candidate",
    "locked_gate_rows",
    "main",
    "prepare_gate",
    "score_predictions",
    "seal_baseline",
]
