"""Preregister and score the deterministic V59 anchor paraphrase gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.evaluation.metrics import (
    exact_normalized_match,
    list_order_insensitive_match,
)
from semantic_3d_chat.evaluation.question_manifest import (
    build_question_manifest,
    load_question_manifest,
    sha256_file,
)

_ROWS: Final[tuple[dict[str, str], ...]] = (
    {
        "scene_id": "scene_000031",
        "question_id": "q_900001",
        "unit_id": "v59pu_000001",
        "question": "Is the book positioned above the cabinet, or below it?",
        "answer": "above",
        "answer_type": "spatial_relation",
    },
    {
        "scene_id": "scene_000032",
        "question_id": "q_900002",
        "unit_id": "v59pu_000001",
        "question": "Is the book positioned above the cabinet, or below it?",
        "answer": "below",
        "answer_type": "spatial_relation",
    },
    {
        "scene_id": "scene_000031",
        "question_id": "q_900003",
        "unit_id": "v59pu_000002",
        "question": "Is the book resting on top of the table, or beneath it?",
        "answer": "top",
        "answer_type": "support",
    },
    {
        "scene_id": "scene_000032",
        "question_id": "q_900004",
        "unit_id": "v59pu_000002",
        "question": "Is the book resting on top of the table, or beneath it?",
        "answer": "beneath",
        "answer_type": "support",
    },
    {
        "scene_id": "scene_000031",
        "question_id": "q_900005",
        "unit_id": "v59pu_000003",
        "question": "Is the table higher or lower than the book?",
        "answer": "lower",
        "answer_type": "spatial_relation",
    },
    {
        "scene_id": "scene_000032",
        "question_id": "q_900006",
        "unit_id": "v59pu_000003",
        "question": "Is the table higher or lower than the book?",
        "answer": "higher",
        "answer_type": "spatial_relation",
    },
    {
        "scene_id": "scene_000031",
        "question_id": "q_900007",
        "unit_id": "v59pu_000004",
        "question": "Name the items resting on the table.",
        "answer": "book, cube",
        "answer_type": "support_list",
    },
    {
        "scene_id": "scene_000032",
        "question_id": "q_900008",
        "unit_id": "v59pu_000004",
        "question": "Name the items resting on the table.",
        "answer": "cube",
        "answer_type": "support_list",
    },
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def prepare(*, questions_output: str | Path, preregistration_output: str | Path) -> dict[str, Any]:
    questions_path = Path(questions_output).expanduser().resolve()
    prereg_path = Path(preregistration_output).expanduser().resolve()
    for destination in (questions_path, prereg_path):
        if destination.exists():
            raise FileExistsError(f"V59 paraphrase preregistration exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
    source_digest = _digest(_ROWS)
    manifest = build_question_manifest(_ROWS, source_qa_sha256=source_digest)
    questions_path.write_text(
        json.dumps(manifest.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verified = load_question_manifest(questions_path)
    preregistration = {
        "schema_version": 1,
        "artifact": "v59_withheld_anchor_paraphrase_preregistration",
        "status": "locked_before_v2_training",
        "reference_sha256": source_digest,
        "questions_manifest_sha256": verified.manifest_sha256,
        "questions_sha256": verified.questions_sha256,
        "question_count": 8,
        "unit_count": 4,
        "thresholds": {
            "exact": 6,
            "complete_units": 3,
            "changed_units": 3,
        },
        "v58_locked_baseline": {
            "exact": 4,
            "complete_units": 2,
            "changed_units": 4,
        },
        "training_inputs_permitted": False,
        "fresh_development_loaded": False,
        "deferred_final_loaded": False,
    }
    prereg_path.write_text(
        json.dumps(preregistration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return preregistration


def score(
    *, predictions: str | Path, preregistration: str | Path, output: str | Path
) -> dict[str, Any]:
    prereg_path = Path(preregistration).expanduser().resolve()
    locked = json.loads(prereg_path.read_text(encoding="utf-8"))
    if (
        locked.get("artifact") != "v59_withheld_anchor_paraphrase_preregistration"
        or locked.get("reference_sha256") != _digest(_ROWS)
        or locked.get("training_inputs_permitted") is not False
    ):
        raise ValueError("V59 paraphrase preregistration changed")
    prediction_path = Path(predictions).expanduser().resolve()
    prediction_rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    for line in prediction_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("scene_id"), row.get("question_id")
        if key in prediction_rows:
            raise ValueError("V59 paraphrase prediction contains a duplicate key")
        prediction_rows[key] = row
    expected_keys = {(row["scene_id"], row["question_id"]) for row in _ROWS}
    if set(prediction_rows) != expected_keys:
        raise ValueError("V59 paraphrase prediction inventory changed")
    scored = []
    for row in _ROWS:
        prediction = prediction_rows[(row["scene_id"], row["question_id"])].get(
            "predicted_answer"
        )
        exact = (
            list_order_insensitive_match(prediction, row["answer"])
            if row["answer_type"] == "support_list"
            else exact_normalized_match(prediction, row["answer"])
        )
        scored.append({"row": row, "prediction": prediction, "exact": exact})
    units: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in scored:
        units[item["row"]["unit_id"]].append(item)
    if any(len(items) != 2 for items in units.values()):
        raise ValueError("V59 paraphrase unit inventory changed")
    exact = sum(item["exact"] for item in scored)
    complete_units = sum(all(item["exact"] for item in items) for items in units.values())
    changed_units = sum(
        len({str(item["prediction"]).strip().casefold() for item in items}) == 2
        for items in units.values()
    )
    thresholds = locked["thresholds"]
    audits = [
        prediction_rows[(row["scene_id"], row["question_id"])].get("control_audit")
        for row in _ROWS
    ]
    audits_valid = all(
        isinstance(audit, Mapping)
        and audit.get("architecture") == "bounded_global_scene_question_control_v2"
        and audit.get("environment_latent_count") == 256
        and audit.get("every_scene_token_influenced_output") is True
        and audit.get("question_dependent_scene_retrieval") is False
        and audit.get("softmax_scene_attention_used") is False
        and audit.get("control_used") is True
        for audit in audits
    )
    checks = {
        "exact": exact >= thresholds["exact"],
        "complete_units": complete_units >= thresholds["complete_units"],
        "changed_units": changed_units >= thresholds["changed_units"],
        "bounded_global_control_used": audits_valid,
    }
    result = {
        "schema_version": 1,
        "artifact": "v59_withheld_anchor_paraphrase_gate",
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "exact": exact,
            "total": len(scored),
            "complete_units": complete_units,
            "changed_units": changed_units,
            "unit_total": len(units),
        },
        "thresholds": thresholds,
        "predictions_sha256": sha256_file(prediction_path),
        "preregistration_sha256": sha256_file(prereg_path),
        "fresh_development_loaded": False,
        "deferred_final_loaded": False,
    }
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"V59 paraphrase score exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--questions-output", required=True)
    prepare_parser.add_argument("--preregistration-output", required=True)
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--predictions", required=True)
    score_parser.add_argument("--preregistration", required=True)
    score_parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = (
        prepare(
            questions_output=args.questions_output,
            preregistration_output=args.preregistration_output,
        )
        if args.command == "prepare"
        else score(
            predictions=args.predictions,
            preregistration=args.preregistration,
            output=args.output,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passed", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "prepare", "score"]
