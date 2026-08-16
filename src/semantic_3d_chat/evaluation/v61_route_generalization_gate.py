"""Preregister and score V61 scene-conditioned routing generalization.

The inference manifest emitted here is deliberately questions-only.  Route
labels and answer references remain in this evaluation module and are never
loaded by the V61 trainer or chat runtime.  Each novel paraphrase is asked in
all six locked training scenes: it should activate control only for the one
counterfactual pair whose cached teacher covers that semantic fact.  The other
four occurrences are exact no-control retention checks with identical wording.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.evaluation.metrics import (
    exact_normalized_match,
    list_order_insensitive_match,
)
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.evaluation.question_manifest import (
    build_question_manifest,
    load_question_manifest,
    sha256_file,
)

_SCENES: Final[tuple[str, ...]] = tuple(
    f"scene_{number:06d}" for number in (31, 32, 33, 34, 37, 38)
)
_PAIR_SCENES: Final[dict[str, tuple[str, str]]] = {
    "anchor": ("scene_000031", "scene_000032"),
    "mirror": ("scene_000033", "scene_000034"),
    "removal": ("scene_000037", "scene_000038"),
}
_SHA256 = re.compile(r"[0-9a-f]{64}")

# These strings are intentionally distinct from the consumed V59 paraphrase
# set.  The two answers correspond to the two positive scenes in _PAIR_SCENES.
_FAMILIES: Final[tuple[dict[str, Any], ...]] = (
    {
        "pair": "anchor",
        "question": "Relative to the cabinet, is the book higher or lower?",
        "answers": ("higher", "lower"),
        "answer_type": "spatial_relation",
    },
    {
        "pair": "anchor",
        "question": "Does the book sit atop the table, or is it underneath?",
        "answers": ("atop", "underneath"),
        "answer_type": "support",
    },
    {
        "pair": "anchor",
        "question": "Compared with the book, is the table higher or lower?",
        "answers": ("lower", "higher"),
        "answer_type": "spatial_relation",
    },
    {
        "pair": "anchor",
        "question": "Which things are supported by the table?",
        "answers": ("book, cube", "cube"),
        "answer_type": "support_list",
    },
    {
        "pair": "mirror",
        "question": "Relative to the cabinet, does the table lie to its left or its right?",
        "answers": ("left", "right"),
        "answer_type": "spatial_relation",
    },
    {
        "pair": "mirror",
        "question": "Which side of the book is the picture frame on: left or right?",
        "answers": ("left", "right"),
        "answer_type": "spatial_relation",
    },
    {
        "pair": "mirror",
        "question": "Is the floor lamp positioned to the left or to the right of the plant pot?",
        "answers": ("left", "right"),
        "answer_type": "spatial_relation",
    },
    {
        "pair": "mirror",
        "question": "Which side of the plant pot holds the picture frame: left or right?",
        "answers": ("left", "right"),
        "answer_type": "spatial_relation",
    },
    {
        "pair": "removal",
        "question": "What number of floor lamps does the room contain?",
        "answers": ("1", "0"),
        "answer_type": "count",
    },
    {
        "pair": "removal",
        "question": "Do you observe any floor lamp here?",
        "answers": ("yes", "no"),
        "answer_type": "presence",
    },
    {
        "pair": "removal",
        "question": "Does this room contain a floor lamp?",
        "answers": ("yes", "no"),
        "answer_type": "presence",
    },
)


def _rows() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    ordinal = 910_001
    for family_index, family in enumerate(_FAMILIES, start=1):
        positive_scenes = _PAIR_SCENES[family["pair"]]
        expected_by_scene = dict(zip(positive_scenes, family["answers"], strict=True))
        for scene_id in _SCENES:
            route_expected = scene_id in positive_scenes
            rows.append(
                {
                    "scene_id": scene_id,
                    "question_id": f"q_{ordinal:06d}",
                    "family_id": f"v61ru_{family_index:06d}",
                    "pair_group": family["pair"],
                    "question": family["question"],
                    "route_expected": route_expected,
                    "answer": expected_by_scene.get(scene_id),
                    "answer_type": family["answer_type"] if route_expected else None,
                }
            )
            ordinal += 1
    return tuple(rows)


_ROWS: Final[tuple[dict[str, Any], ...]] = _rows()


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _prediction_rows(path: str | Path) -> tuple[dict[tuple[str, str], dict[str, Any]], str]:
    source = Path(path).expanduser().resolve()
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError("V61 prediction rows must be JSON objects")
        key = value.get("scene_id"), value.get("question_id")
        if key in rows:
            raise ValueError("V61 predictions contain a duplicate key")
        if not all(isinstance(part, str) for part in key):
            raise TypeError("V61 prediction keys must be strings")
        rows[key] = value
    expected = {(row["scene_id"], row["question_id"]) for row in _ROWS}
    if set(rows) != expected:
        raise ValueError("V61 prediction inventory differs from locked questions")
    return rows, sha256_file(source)


def prepare(*, questions_output: str | Path, preregistration_output: str | Path) -> dict[str, Any]:
    """Write immutable questions-only and threshold artifacts before V4 exists."""

    questions_path = Path(questions_output).expanduser().resolve()
    prereg_path = Path(preregistration_output).expanduser().resolve()
    for destination in (questions_path, prereg_path):
        if destination.exists():
            raise FileExistsError(f"V61 preregistration output exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
    reference_sha256 = _digest(_ROWS)
    manifest = build_question_manifest(_ROWS, source_qa_sha256=reference_sha256)
    questions_path.write_text(
        json.dumps(manifest.as_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    verified = load_question_manifest(questions_path)
    positive = [row for row in _ROWS if row["route_expected"]]
    negative = [row for row in _ROWS if not row["route_expected"]]
    preregistration = {
        "schema_version": 1,
        "artifact": "v61_scene_conditioned_route_preregistration",
        "status": "locked_before_v4_implementation",
        "reference_sha256": reference_sha256,
        "questions_manifest_sha256": verified.manifest_sha256,
        "questions_sha256": verified.questions_sha256,
        "locked_scene_ids": list(_SCENES),
        "question_count": len(_ROWS),
        "family_count": len(_FAMILIES),
        "positive_count": len(positive),
        "negative_same_wording_count": len(negative),
        "opaque_route_inventory": [
            {
                "scene_id": row["scene_id"],
                "question_id": row["question_id"],
                "family_id": row["family_id"],
                "pair_group": row["pair_group"],
                "route_expected": row["route_expected"],
            }
            for row in _ROWS
        ],
        "thresholds": {
            "route_positive": 18,
            "route_negative": 40,
            "route_anchor_positive": 6,
            "route_mirror_positive": 6,
            "route_removal_positive": 4,
            "contradictory_complete_families": 8,
            "positive_answer_exact": 15,
            "positive_complete_families": 6,
            "positive_changed_families": 8,
            "negative_exact_v54_output_identity": 40,
        },
        "training_inputs_permitted": False,
        "trainer_must_not_load_questions_or_preregistration": True,
        "fresh_development_loaded": False,
        "deferred_final_loaded": False,
        "prohibited_scene_numbers": list(range(25, 31)) + list(range(57, 63)),
    }
    prereg_path.write_text(
        json.dumps(preregistration, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return preregistration


def lock_baseline(
    *,
    predictions: str | Path,
    preregistration: str | Path,
    base_checkpoint: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Freeze V54 no-control outputs without copying answer text into the lock."""

    prereg_path = Path(preregistration).expanduser().resolve()
    prereg = _load_json(prereg_path)
    if (
        prereg.get("artifact") != "v61_scene_conditioned_route_preregistration"
        or prereg.get("reference_sha256") != _digest(_ROWS)
        or prereg.get("training_inputs_permitted") is not False
    ):
        raise ValueError("V61 preregistration changed before baseline lock")
    rows, predictions_sha256 = _prediction_rows(predictions)
    checkpoint_sha256, checkpoint_files = checkpoint_fingerprint(base_checkpoint)
    required_hashes = []
    prefix_hashes: defaultdict[str, set[str]] = defaultdict(set)
    for row in _ROWS:
        key = row["scene_id"], row["question_id"]
        prediction = rows[key]
        raw = prediction.get("predicted_answer")
        prefix = prediction.get("prefix_hash")
        if not isinstance(raw, str):
            raise TypeError("V61 baseline prediction must contain answer text")
        if not isinstance(prefix, str) or _SHA256.fullmatch(prefix) is None:
            raise ValueError("V61 baseline prediction has an invalid prefix hash")
        prefix_hashes[row["scene_id"]].add(prefix)
        required_hashes.append(
            {
                "scene_id": row["scene_id"],
                "question_id": row["question_id"],
                "raw_output_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            }
        )
    if set(prefix_hashes) != set(_SCENES) or any(
        len(values) != 1 for values in prefix_hashes.values()
    ):
        raise ValueError("V61 baseline does not prove one fixed prefix per scene")
    prediction_path = Path(predictions).expanduser().resolve()
    provenance_path = prediction_path.with_suffix(prediction_path.suffix + ".provenance.json")
    if not provenance_path.is_file():
        raise FileNotFoundError("V61 baseline prediction provenance is unavailable")
    result = {
        "schema_version": 1,
        "artifact": "v61_v54_no_control_baseline_lock",
        "preregistration_sha256": sha256_file(prereg_path),
        "questions_manifest_sha256": prereg["questions_manifest_sha256"],
        "questions_sha256": prereg["questions_sha256"],
        "predictions_sha256": predictions_sha256,
        "prediction_provenance_sha256": sha256_file(provenance_path),
        "base_checkpoint_sha256": checkpoint_sha256,
        "base_checkpoint_files": checkpoint_files,
        "prefix_hashes": {scene_id: next(iter(prefix_hashes[scene_id])) for scene_id in _SCENES},
        "required_output_hashes": required_hashes,
        "required_output_hashes_sha256": _digest(required_hashes),
        "question_count": len(required_hashes),
        "environmental_answer_text_stored": False,
        "fresh_development_loaded": False,
        "deferred_final_loaded": False,
    }
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"V61 baseline lock exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def score(
    *,
    predictions: str | Path,
    preregistration: str | Path,
    baseline: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Score V4 routing, positive QA, and exact no-control preservation."""

    prereg_path = Path(preregistration).expanduser().resolve()
    baseline_path = Path(baseline).expanduser().resolve()
    prereg = _load_json(prereg_path)
    baseline_value = _load_json(baseline_path)
    if (
        prereg.get("reference_sha256") != _digest(_ROWS)
        or prereg.get("training_inputs_permitted") is not False
        or baseline_value.get("preregistration_sha256") != sha256_file(prereg_path)
        or baseline_value.get("required_output_hashes_sha256")
        != _digest(baseline_value.get("required_output_hashes"))
    ):
        raise ValueError("V61 gate prerequisites changed")
    rows, predictions_sha256 = _prediction_rows(predictions)
    baseline_hashes = {
        (row["scene_id"], row["question_id"]): row["raw_output_sha256"]
        for row in baseline_value["required_output_hashes"]
    }
    route_positive = 0
    route_negative = 0
    positive_answer_exact = 0
    negative_identity = 0
    positive_by_pair: defaultdict[str, int] = defaultdict(int)
    family_routes: defaultdict[str, list[bool]] = defaultdict(list)
    family_positive: defaultdict[str, list[tuple[bool, str]]] = defaultdict(list)
    audits_valid = True
    for reference in _ROWS:
        key = reference["scene_id"], reference["question_id"]
        prediction = rows[key]
        answer = prediction.get("predicted_answer")
        audit = prediction.get("control_audit")
        expected_route = bool(reference["route_expected"])
        valid_audit = (
            isinstance(audit, Mapping)
            and audit.get("architecture") == "scene_conditioned_gate_teacher_basis_control_v4"
            and audit.get("environment_latent_count") == 256
            and audit.get("every_scene_token_influenced_output") is True
            and audit.get("question_dependent_scene_retrieval") is False
            and audit.get("softmax_scene_attention_used") is False
            and audit.get("control_used") is expected_route
            and audit.get("exact_no_control_route") is (not expected_route)
        )
        audits_valid = audits_valid and valid_audit
        route_correct = bool(
            isinstance(audit, Mapping) and audit.get("control_used") is expected_route
        )
        family_routes[reference["family_id"]].append(route_correct)
        if expected_route:
            route_positive += int(route_correct)
            positive_by_pair[reference["pair_group"]] += int(route_correct)
            exact = (
                list_order_insensitive_match(answer, reference["answer"])
                if reference["answer_type"] == "support_list"
                else exact_normalized_match(answer, reference["answer"])
            )
            positive_answer_exact += int(exact)
            family_positive[reference["family_id"]].append((exact, str(answer)))
        else:
            route_negative += int(route_correct)
            raw_hash = hashlib.sha256(str(answer).encode()).hexdigest()
            negative_identity += int(raw_hash == baseline_hashes[key])
    positive_complete = sum(
        len(values) == 2 and all(item[0] for item in values) for values in family_positive.values()
    )
    positive_changed = sum(
        len(values) == 2 and len({item[1].strip().casefold() for item in values}) == 2
        for values in family_positive.values()
    )
    contradictory_complete = sum(all(values) for values in family_routes.values())
    thresholds = prereg["thresholds"]
    checks = {
        "route_positive": route_positive >= thresholds["route_positive"],
        "route_negative": route_negative >= thresholds["route_negative"],
        "route_anchor_positive": positive_by_pair["anchor"] >= thresholds["route_anchor_positive"],
        "route_mirror_positive": positive_by_pair["mirror"] >= thresholds["route_mirror_positive"],
        "route_removal_positive": positive_by_pair["removal"]
        >= thresholds["route_removal_positive"],
        "contradictory_complete_families": contradictory_complete
        >= thresholds["contradictory_complete_families"],
        "positive_answer_exact": positive_answer_exact >= thresholds["positive_answer_exact"],
        "positive_complete_families": positive_complete >= thresholds["positive_complete_families"],
        "positive_changed_families": positive_changed >= thresholds["positive_changed_families"],
        "negative_exact_v54_output_identity": negative_identity
        >= thresholds["negative_exact_v54_output_identity"],
        "continuous_global_v4_audits_valid": audits_valid,
    }
    result = {
        "schema_version": 1,
        "artifact": "v61_scene_conditioned_route_generalization_gate",
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "route_positive": route_positive,
            "route_positive_total": 22,
            "route_negative": route_negative,
            "route_negative_total": 44,
            "route_positive_by_pair": dict(sorted(positive_by_pair.items())),
            "contradictory_complete_families": contradictory_complete,
            "family_total": 11,
            "positive_answer_exact": positive_answer_exact,
            "positive_answer_total": 22,
            "positive_complete_families": positive_complete,
            "positive_changed_families": positive_changed,
            "negative_exact_v54_output_identity": negative_identity,
            "negative_total": 44,
        },
        "thresholds": thresholds,
        "predictions_sha256": predictions_sha256,
        "preregistration_sha256": sha256_file(prereg_path),
        "baseline_sha256": sha256_file(baseline_path),
        "fresh_development_loaded": False,
        "deferred_final_loaded": False,
    }
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"V61 score exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--questions-output", required=True)
    prepare_parser.add_argument("--preregistration-output", required=True)
    baseline_parser = commands.add_parser("lock-baseline")
    baseline_parser.add_argument("--predictions", required=True)
    baseline_parser.add_argument("--preregistration", required=True)
    baseline_parser.add_argument("--base-checkpoint", required=True)
    baseline_parser.add_argument("--output", required=True)
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--predictions", required=True)
    score_parser.add_argument("--preregistration", required=True)
    score_parser.add_argument("--baseline", required=True)
    score_parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(
            questions_output=args.questions_output,
            preregistration_output=args.preregistration_output,
        )
    elif args.command == "lock-baseline":
        result = lock_baseline(
            predictions=args.predictions,
            preregistration=args.preregistration,
            base_checkpoint=args.base_checkpoint,
            output=args.output,
        )
    else:
        result = score(
            predictions=args.predictions,
            preregistration=args.preregistration,
            baseline=args.baseline,
            output=args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2 if result.get("passed") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
