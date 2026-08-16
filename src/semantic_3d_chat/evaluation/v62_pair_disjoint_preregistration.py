"""Create the immutable, pair-disjoint V62 training/evaluation boundary.

This module is deliberately a *preparation* boundary, not a trainer or scorer.
It authenticates and reads the complete diverse-52 training QA file once, then
declassifies four create-once artifacts:

* a 12-pair training-only QA JSONL;
* an 8-pair questions-only internal-validation manifest;
* a physically separate scorer-only answer/route-label sidecar; and
* a preregistration binding the split, artifact hashes, natural population,
  paired-unit metrics, controls, and thresholds.

No V62 trainer needs (or is allowed) to accept validation questions or scorer
references.  ``add_filtered_training_data_argument`` exposes the sole V62
dataset argument future trainers may add to their parser.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.evaluation.question_manifest import build_question_manifest

_PINNED_SOURCE_QA_SHA256: Final[str] = (
    "01721bf904b1ab0b65ce8acac6e366287040873cda1356da6c70c4981abe7619"
)
_PINNED_SOURCE_QA_SIZE_BYTES: Final[int] = 526_153
_PINNED_FILTERED_TRAIN_SHA256: Final[str] = (
    "84b99385fadc5d06e44465ada5902f56131192298ca1539373dc3b334608cbf1"
)
# A trainer validates its hash-only authorization against these constants; it
# never needs a preregistration, question-manifest, validation, or scorer path.
PINNED_V62_PREREGISTRATION_SHA256: Final[str] = (
    "a8bd6db776f772f579db2eaeb6dd817f46c23581807559c34870f6be22d6e1e7"
)
PINNED_V62_QUESTIONS_MANIFEST_SHA256: Final[str] = (
    "078f65e1402e6e382a7bfdb2ad4b8a65d58e3164705a8a46cd222503aa201052"
)
PINNED_V62_QUESTIONS_SHA256: Final[str] = (
    "05bd92897b1888b92cfe7be651cc83f9b94cc4a36950c17cb58859ec73325167"
)
_PINNED_V61_TERMINAL_SHA256: Final[str] = (
    "cb7ce887c03dc156693cca489e7638d32adc1ad11cbd0f33464bcfcc4ae5db38"
)
_PINNED_V61_ARTIFACT: Final[str] = "v61_scene_conditioned_route_generalization_gate"
_PINNED_V54_CHECKPOINT_SHA256: Final[str] = (
    "3e128b40c1b73bb32750285679cda6b1bea364e67465e986a94a81dfc95e81e8"
)
_PINNED_VALIDATION_QUESTION_KEYS_SHA256: Final[str] = (
    "f36885e43100a5b7a3682ca38f7a06187c1f9b204095f5dc89b2e597e227ba27"
)
_PREREGISTRATION_SCHEMA: Final[str] = (
    "semantic_3d_chat.v62.pair_disjoint_preregistration.v1"
)
_PREREGISTRATION_ARTIFACT: Final[str] = "v62_pair_disjoint_preregistration"
_BASELINE_LOCK_SCHEMA: Final[str] = (
    "semantic_3d_chat.v62.v54_no_control_baseline_lock.v1"
)
_BASELINE_LOCK_ARTIFACT: Final[str] = "v62_v54_no_control_baseline_lock"

_SCENE_ID = re.compile(r"scene_[0-9]{6}")
_QUESTION_ID = re.compile(r"q_[0-9]{6}")
_PAIR_ID = re.compile(r"pair_[0-9]{6}")
_QUESTION_KEY = re.compile(r"cfq_[0-9a-f]{16}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class PairSpec:
    """Pinned composition of one complete 24-question counterfactual pair."""

    pair_id: str
    reference_scene_id: str
    counterfactual_scene_id: str
    change_type: str
    changed_unit_count: int

    @property
    def scene_ids(self) -> tuple[str, str]:
        return self.reference_scene_id, self.counterfactual_scene_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "scene_ids": list(self.scene_ids),
            "change_type": self.change_type,
            "paired_unit_count": 24,
            "changed_unit_count": self.changed_unit_count,
            "retention_unit_count": 24 - self.changed_unit_count,
        }


PAIR_INVENTORY: Final[tuple[PairSpec, ...]] = (
    PairSpec("pair_000005", "scene_000011", "scene_000012", "chair_orientation", 1),
    PairSpec("pair_000006", "scene_000013", "scene_000014", "object_relocation", 4),
    PairSpec("pair_000007", "scene_000015", "scene_000016", "color_swap", 4),
    PairSpec("pair_000008", "scene_000017", "scene_000018", "object_count", 1),
    PairSpec("pair_000009", "scene_000019", "scene_000020", "book_support", 4),
    PairSpec("pair_000010", "scene_000021", "scene_000022", "mirror_lr", 4),
    PairSpec("pair_000011", "scene_000023", "scene_000024", "picture_support", 4),
    PairSpec("pair_000015", "scene_000031", "scene_000032", "book_support", 4),
    PairSpec("pair_000016", "scene_000033", "scene_000034", "mirror_lr", 4),
    PairSpec("pair_000017", "scene_000035", "scene_000036", "picture_support", 4),
    PairSpec("pair_000018", "scene_000037", "scene_000038", "object_removal", 3),
    PairSpec("pair_000019", "scene_000039", "scene_000040", "chair_orientation", 1),
    PairSpec("pair_000020", "scene_000041", "scene_000042", "object_relocation", 4),
    PairSpec("pair_000021", "scene_000043", "scene_000044", "color_swap", 4),
    PairSpec("pair_000022", "scene_000045", "scene_000046", "object_count", 1),
    PairSpec("pair_000023", "scene_000047", "scene_000048", "book_support", 4),
    PairSpec("pair_000024", "scene_000049", "scene_000050", "mirror_lr", 4),
    PairSpec("pair_000025", "scene_000051", "scene_000052", "picture_support", 4),
    PairSpec("pair_000026", "scene_000053", "scene_000054", "cube_support", 3),
    PairSpec("pair_000027", "scene_000055", "scene_000056", "object_removal", 4),
)

TRAIN_PAIR_IDS: Final[tuple[str, ...]] = (
    "pair_000005",
    "pair_000006",
    "pair_000007",
    "pair_000008",
    "pair_000009",
    "pair_000010",
    "pair_000011",
    "pair_000015",
    "pair_000016",
    "pair_000017",
    "pair_000018",
    "pair_000026",
)
INTERNAL_VALIDATION_PAIR_IDS: Final[tuple[str, ...]] = (
    "pair_000019",
    "pair_000020",
    "pair_000021",
    "pair_000022",
    "pair_000023",
    "pair_000024",
    "pair_000025",
    "pair_000027",
)

_REQUIRED_ROW_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "scene_id",
        "question_id",
        "question",
        "answer",
        "answer_type",
        "counterfactual_pair_id",
        "counterfactual_paired_scene_id",
        "counterfactual_question_key",
        "counterfactual_change_type",
        "counterfactual_role",
        "counterfactual_expected_change",
    }
)
_PROTECTED_SCENE_NUMBERS: Final[tuple[int, ...]] = tuple(range(25, 31)) + tuple(
    range(57, 63)
)

_EXPECTED_SELECTION_DISTRIBUTIONS: Final[dict[str, dict[str, dict[str, int]]]] = {
    "training": {
        "answer_type": {
            "changed": {
                "attribute": 8,
                "count": 4,
                "metric": 2,
                "orientation": 2,
                "presence": 4,
                "spatial_relation": 48,
                "support": 12,
            },
            "retention": {
                "attribute": 112,
                "count": 92,
                "metric": 22,
                "orientation": 20,
                "presence": 96,
                "spatial_relation": 72,
                "support": 82,
            },
        },
        "question_family": {
            "changed": {
                "attribute_tell": 4,
                "attribute_what": 4,
                "count": 4,
                "metric": 2,
                "orientation": 2,
                "presence_find": 2,
                "presence_is_there": 2,
                "spatial_horizontal": 24,
                "spatial_vertical": 24,
                "support_book_on_under": 4,
                "support_list": 6,
                "support_picture_wall_floor": 2,
            },
            "retention": {
                "attribute_tell": 62,
                "attribute_what": 50,
                "count": 92,
                "metric": 22,
                "orientation": 20,
                "presence_find": 18,
                "presence_is_there": 78,
                "spatial_depth": 34,
                "spatial_horizontal": 24,
                "spatial_vertical": 14,
                "support_book_on_under": 20,
                "support_bowl_floor_table": 24,
                "support_list": 18,
                "support_picture_wall_floor": 20,
            },
        },
    },
    "internal_validation": {
        "answer_type": {
            "changed": {
                "attribute": 8,
                "count": 4,
                "metric": 2,
                "orientation": 2,
                "presence": 4,
                "spatial_relation": 28,
                "support": 4,
            },
            "retention": {
                "attribute": 72,
                "count": 60,
                "metric": 14,
                "orientation": 12,
                "presence": 66,
                "spatial_relation": 52,
                "support": 56,
            },
        },
        "question_family": {
            "changed": {
                "attribute_tell": 4,
                "attribute_what": 4,
                "count": 4,
                "metric": 2,
                "orientation": 2,
                "presence_find": 2,
                "presence_is_there": 2,
                "spatial_horizontal": 18,
                "spatial_vertical": 10,
                "support_book_on_under": 2,
                "support_list": 2,
            },
            "retention": {
                "attribute_tell": 26,
                "attribute_what": 46,
                "count": 60,
                "metric": 14,
                "orientation": 12,
                "presence_find": 20,
                "presence_is_there": 46,
                "spatial_depth": 16,
                "spatial_horizontal": 18,
                "spatial_vertical": 18,
                "support_book_on_under": 14,
                "support_bowl_floor_table": 16,
                "support_list": 12,
                "support_picture_wall_floor": 14,
            },
        },
    },
}

# A future trainer may add normal model/optimizer arguments, but these are the
# complete V62 dataset boundary: there is one training data path and no scorer
# or internal-validation data argument.
V62_TRAINER_DATA_ARGUMENTS: Final[tuple[str, ...]] = ("filtered_train_qa",)
V62_TRAINER_AUTHORIZATION_ARGUMENTS: Final[tuple[str, ...]] = ("baseline_lock",)
V62_PROHIBITED_TRAINER_DATA_ARGUMENTS: Final[frozenset[str]] = frozenset(
    {
        "source_train_qa",
        "validation_questions",
        "internal_validation_questions",
        "scorer_references",
        "scorer_sidecar",
    }
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _question_key_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    keys = [
        {"scene_id": row["scene_id"], "question_id": row["question_id"]} for row in rows
    ]
    return _sha256_bytes(_canonical_jsonl_bytes(keys))


def _canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {
        "sort_keys": True,
        "ensure_ascii": False,
        "allow_nan": False,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(value, **options) + "\n").encode("utf-8")


def _canonical_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(dict(row)) for row in rows)


def _decode_jsonl(raw: bytes, *, label: str) -> tuple[dict[str, Any], ...]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    if not text.endswith("\n"):
        raise ValueError(f"{label} must end with a newline")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{label} contains a blank row at line {index}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} line {index} is invalid JSON") from exc
        if not isinstance(value, dict):
            raise TypeError(f"{label} line {index} must be a JSON object")
        rows.append(value)
    return tuple(rows)


def _answer_signature(row: Mapping[str, Any]) -> object:
    answer_items = row.get("answer_items")
    if answer_items is not None:
        if not isinstance(answer_items, list) or not all(
            isinstance(item, str) for item in answer_items
        ):
            raise TypeError("answer_items must be a list of strings when present")
        return tuple(sorted(" ".join(item.casefold().split()) for item in answer_items))
    answer = row["answer"]
    if not isinstance(answer, str) or not answer.strip():
        raise TypeError("answer must be a non-empty string")
    return " ".join(answer.casefold().split())


def _question_family(question: str) -> str:
    if question.startswith("Tell me ") and question.endswith("'s color."):
        return "attribute_tell"
    if question.startswith("What color is "):
        return "attribute_what"
    if question.startswith("How many "):
        return "count"
    if question == "Which object is closest to the camera?":
        return "metric"
    if question == "Is the chair upright or upside down?":
        return "orientation"
    if question.startswith("Can you find "):
        return "presence_find"
    if question.startswith("Is there "):
        return "presence_is_there"
    if question == "Is the book on the table or under the table?":
        return "support_book_on_under"
    if question == "Is the bowl on the floor or on the table?":
        return "support_bowl_floor_table"
    if question == "What is on the table?":
        return "support_list"
    if question == "Is the picture frame on the wall or on the floor?":
        return "support_picture_wall_floor"
    if " above or below " in question:
        return "spatial_vertical"
    if " in front of or behind " in question:
        return "spatial_depth"
    if " left or right of " in question:
        return "spatial_horizontal"
    raise ValueError(f"Unregistered V62 question family: {question!r}")


def _validate_row_shape(row: Mapping[str, Any], *, index: int) -> None:
    missing = _REQUIRED_ROW_FIELDS - set(row)
    if missing:
        raise ValueError(f"QA row {index} is missing fields: {sorted(missing)}")
    scene_id = row["scene_id"]
    paired_scene_id = row["counterfactual_paired_scene_id"]
    question_id = row["question_id"]
    pair_id = row["counterfactual_pair_id"]
    question_key = row["counterfactual_question_key"]
    if not isinstance(scene_id, str) or _SCENE_ID.fullmatch(scene_id) is None:
        raise ValueError(f"QA row {index} has invalid scene_id")
    if not isinstance(paired_scene_id, str) or _SCENE_ID.fullmatch(paired_scene_id) is None:
        raise ValueError(f"QA row {index} has invalid paired scene_id")
    if not isinstance(question_id, str) or _QUESTION_ID.fullmatch(question_id) is None:
        raise ValueError(f"QA row {index} has invalid question_id")
    if not isinstance(pair_id, str) or _PAIR_ID.fullmatch(pair_id) is None:
        raise ValueError(f"QA row {index} has invalid pair_id")
    if not isinstance(question_key, str) or _QUESTION_KEY.fullmatch(question_key) is None:
        raise ValueError(f"QA row {index} has invalid counterfactual question key")
    if not isinstance(row["question"], str) or not row["question"].strip():
        raise TypeError(f"QA row {index} has invalid question text")
    if not isinstance(row["answer_type"], str) or not row["answer_type"].strip():
        raise TypeError(f"QA row {index} has invalid answer_type")
    if row["counterfactual_role"] not in {"reference", "counterfactual"}:
        raise ValueError(f"QA row {index} has invalid counterfactual role")
    if type(row["counterfactual_expected_change"]) is not bool:
        raise TypeError(f"QA row {index} has a non-boolean route label")
    _answer_signature(row)
    _question_family(row["question"])


def _validate_inventory(
    rows: Sequence[Mapping[str, Any]], *, specs: Sequence[PairSpec], label: str
) -> dict[str, Any]:
    expected_specs = {spec.pair_id: spec for spec in specs}
    expected_scene_ids = {scene_id for spec in specs for scene_id in spec.scene_ids}
    expected_row_count = 48 * len(specs)
    if len(rows) != expected_row_count:
        raise ValueError(f"{label} needs exactly {expected_row_count} rows")

    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen_question_ids: set[tuple[str, str]] = set()
    actual_scene_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        _validate_row_shape(row, index=index)
        pair_id = str(row["counterfactual_pair_id"])
        if pair_id not in expected_specs:
            raise ValueError(f"{label} contains an unregistered pair: {pair_id}")
        key = str(row["scene_id"]), str(row["question_id"])
        if key in seen_question_ids:
            raise ValueError(f"{label} contains duplicate question key {key}")
        seen_question_ids.add(key)
        grouped[pair_id].append(row)
        actual_scene_ids.add(str(row["scene_id"]))

    if set(grouped) != set(expected_specs):
        raise ValueError(f"{label} pair inventory differs from the pin")
    if actual_scene_ids != expected_scene_ids:
        raise ValueError(f"{label} scene inventory differs from the pin")
    protected = {
        scene_id
        for scene_id in actual_scene_ids
        if int(scene_id.removeprefix("scene_")) in _PROTECTED_SCENE_NUMBERS
    }
    if protected:
        raise ValueError(f"{label} contains protected scene IDs: {sorted(protected)}")

    changed_unit_total = 0
    pair_summaries: list[dict[str, Any]] = []
    for spec in specs:
        pair_rows = grouped[spec.pair_id]
        if len(pair_rows) != 48:
            raise ValueError(f"{spec.pair_id} must contain exactly 48 rows")
        if {str(row["scene_id"]) for row in pair_rows} != set(spec.scene_ids):
            raise ValueError(f"{spec.pair_id} has incorrect scene membership")
        role_by_scene = {
            scene_id: {row["counterfactual_role"] for row in pair_rows if row["scene_id"] == scene_id}
            for scene_id in spec.scene_ids
        }
        if role_by_scene != {
            spec.reference_scene_id: {"reference"},
            spec.counterfactual_scene_id: {"counterfactual"},
        }:
            raise ValueError(f"{spec.pair_id} role assignment differs from the pin")
        if {row["counterfactual_change_type"] for row in pair_rows} != {spec.change_type}:
            raise ValueError(f"{spec.pair_id} has incorrect change type")

        units: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in pair_rows:
            units[str(row["counterfactual_question_key"])].append(row)
        if len(units) != 24:
            raise ValueError(f"{spec.pair_id} must contain 24 paired question units")
        changed_units = 0
        for question_key, sides in units.items():
            if len(sides) != 2:
                raise ValueError(f"{spec.pair_id}/{question_key} is not a two-sided unit")
            by_scene = {str(row["scene_id"]): row for row in sides}
            if set(by_scene) != set(spec.scene_ids):
                raise ValueError(f"{spec.pair_id}/{question_key} misses a paired scene")
            left = by_scene[spec.reference_scene_id]
            right = by_scene[spec.counterfactual_scene_id]
            if (
                left["counterfactual_paired_scene_id"] != spec.counterfactual_scene_id
                or right["counterfactual_paired_scene_id"] != spec.reference_scene_id
            ):
                raise ValueError(f"{spec.pair_id}/{question_key} pairing is not reciprocal")
            if left["question"] != right["question"]:
                raise ValueError(f"{spec.pair_id}/{question_key} question text differs by side")
            if left["answer_type"] != right["answer_type"]:
                raise ValueError(f"{spec.pair_id}/{question_key} answer type differs by side")
            expected_change = left["counterfactual_expected_change"]
            if right["counterfactual_expected_change"] is not expected_change:
                raise ValueError(f"{spec.pair_id}/{question_key} route labels disagree")
            answer_changed = _answer_signature(left) != _answer_signature(right)
            if answer_changed is not expected_change:
                raise ValueError(f"{spec.pair_id}/{question_key} route label contradicts answers")
            changed_units += int(expected_change)
        if changed_units != spec.changed_unit_count:
            raise ValueError(f"{spec.pair_id} changed-unit inventory differs from the pin")
        changed_unit_total += changed_units
        pair_summaries.append(spec.as_dict())

    route_answer_types: dict[str, Counter[str]] = {
        "changed": Counter(),
        "retention": Counter(),
    }
    route_question_families: dict[str, Counter[str]] = {
        "changed": Counter(),
        "retention": Counter(),
    }
    for row in rows:
        route = "changed" if row["counterfactual_expected_change"] else "retention"
        route_answer_types[route][str(row["answer_type"])] += 1
        route_question_families[route][_question_family(str(row["question"]))] += 1
    changed_sides = 2 * changed_unit_total
    paired_units = 24 * len(specs)
    return {
        "pair_count": len(specs),
        "scene_count": len(expected_scene_ids),
        "row_count": len(rows),
        "paired_unit_count": paired_units,
        "changed_side_count": changed_sides,
        "retention_side_count": len(rows) - changed_sides,
        "changed_unit_count": changed_unit_total,
        "retention_unit_count": paired_units - changed_unit_total,
        "natural_changed_side_fraction": changed_sides / len(rows),
        "answer_type_by_route": {
            route: dict(sorted(counts.items())) for route, counts in route_answer_types.items()
        },
        "question_family_by_route": {
            route: dict(sorted(counts.items()))
            for route, counts in route_question_families.items()
        },
        "pair_inventory": pair_summaries,
    }


def _assert_selection_distribution(name: str, inventory: Mapping[str, Any]) -> None:
    expected = _EXPECTED_SELECTION_DISTRIBUTIONS[name]
    if inventory["answer_type_by_route"] != expected["answer_type"]:
        raise ValueError(f"{name} answer-type distribution differs from the pin")
    if inventory["question_family_by_route"] != expected["question_family"]:
        raise ValueError(f"{name} question-family distribution differs from the pin")


def _load_v61_terminal_once(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    if _sha256_bytes(raw) != _PINNED_V61_TERMINAL_SHA256:
        raise ValueError("V61 terminal SHA-256 differs from the preserved terminal")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("V61 terminal is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise TypeError("V61 terminal must be a JSON object")
    if (
        payload.get("artifact") != _PINNED_V61_ARTIFACT
        or payload.get("passed") is not False
        or payload.get("fresh_development_loaded") is not False
        or payload.get("deferred_final_loaded") is not False
    ):
        raise ValueError("V61 terminal semantics changed; its failed terminal must be preserved")
    return payload, raw


def _load_json_object_once(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object")
    return payload, raw


def _validate_preregistration_payload(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("schema") != _PREREGISTRATION_SCHEMA
        or payload.get("artifact") != _PREREGISTRATION_ARTIFACT
        or payload.get("status")
        != "locked_before_v62_training_or_internal_validation_inference"
    ):
        raise ValueError("V62 preregistration identity changed")
    source = payload.get("source")
    artifacts = payload.get("artifacts")
    boundary = payload.get("data_boundaries")
    if (
        not isinstance(source, Mapping)
        or source.get("training_qa_sha256") != _PINNED_SOURCE_QA_SHA256
        or not isinstance(artifacts, Mapping)
        or not isinstance(artifacts.get("internal_validation_questions"), Mapping)
        or not isinstance(boundary, Mapping)
        or boundary.get("baseline_lock_required_before_training") is not True
        or boundary.get("held_out_qa_or_oracle_loaded") is not False
        or boundary.get("trainer_may_load_scorer_references") is not False
    ):
        raise ValueError("V62 preregistration data boundary changed")


def _validate_baseline_prediction_rows(
    raw: bytes, *, expected_keys: set[tuple[str, str]]
) -> tuple[dict[str, Any], ...]:
    rows = _decode_jsonl(raw, label="V62 V54 baseline predictions")
    seen: set[tuple[str, str]] = set()
    required_fields = {"scene_id", "question_id", "predicted_answer", "prefix_hash"}
    for index, row in enumerate(rows, start=1):
        missing = required_fields - set(row)
        if missing:
            raise ValueError(f"Baseline prediction {index} misses fields: {sorted(missing)}")
        scene_id = row["scene_id"]
        question_id = row["question_id"]
        answer = row["predicted_answer"]
        prefix_hash = row["prefix_hash"]
        if not isinstance(scene_id, str) or _SCENE_ID.fullmatch(scene_id) is None:
            raise ValueError(f"Baseline prediction {index} has invalid scene_id")
        if not isinstance(question_id, str) or _QUESTION_ID.fullmatch(question_id) is None:
            raise ValueError(f"Baseline prediction {index} has invalid question_id")
        if not isinstance(answer, str):
            raise TypeError(f"Baseline prediction {index} answer must be raw UTF-8 text")
        if not isinstance(prefix_hash, str) or _SHA256.fullmatch(prefix_hash) is None:
            raise ValueError(f"Baseline prediction {index} has invalid prefix hash")
        key = scene_id, question_id
        if key in seen:
            raise ValueError(f"Baseline predictions contain duplicate key: {key}")
        seen.add(key)
    if seen != expected_keys:
        raise ValueError("Baseline prediction inventory differs from the questions manifest")
    return rows


def _validate_destinations(
    *,
    filtered_train_output: str | Path,
    validation_questions_output: str | Path,
    scorer_references_output: str | Path,
    preregistration_output: str | Path,
) -> tuple[Path, Path, Path, Path]:
    paths = tuple(
        Path(value).expanduser().resolve()
        for value in (
            filtered_train_output,
            validation_questions_output,
            scorer_references_output,
            preregistration_output,
        )
    )
    if len(set(paths)) != len(paths):
        raise ValueError("V62 outputs must be four distinct files")
    for path in paths:
        if path.exists():
            raise FileExistsError(f"V62 create-once output already exists: {path}")
    train_path, questions_path, scorer_path, _prereg_path = paths
    if len({train_path.parent, questions_path.parent, scorer_path.parent}) != 3:
        raise ValueError(
            "Filtered training, inference questions, and scorer references need separate directories"
        )
    if {part.casefold() for part in scorer_path.parts} & {"runtime", "chat"}:
        raise ValueError("Scorer references cannot be written under a runtime/chat directory")
    return paths


def _publish_create_once(artifacts: Sequence[tuple[Path, bytes]]) -> None:
    """Create a small artifact group without overwriting and roll back partial writes."""

    published: list[Path] = []
    try:
        for path, payload in artifacts:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            published.append(path)
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise


def _scorer_record(row: Mapping[str, Any]) -> dict[str, Any]:
    answer_items = row.get("answer_items")
    return {
        "scene_id": row["scene_id"],
        "question_id": row["question_id"],
        "answer": row["answer"],
        "answer_items": list(answer_items) if isinstance(answer_items, list) else None,
        "answer_type": row["answer_type"],
        "route_label": bool(row["counterfactual_expected_change"]),
        "counterfactual_pair_id": row["counterfactual_pair_id"],
        "counterfactual_paired_scene_id": row["counterfactual_paired_scene_id"],
        "counterfactual_question_key": row["counterfactual_question_key"],
        "counterfactual_change_type": row["counterfactual_change_type"],
        "counterfactual_role": row["counterfactual_role"],
    }


def add_filtered_training_data_argument(parser: argparse.ArgumentParser) -> None:
    """Add the sole V62 data path that a future trainer is permitted to accept."""

    parser.add_argument(
        "--filtered-train-qa",
        required=True,
        help="Create-once 12-pair V62 training JSONL; never the 40-scene source QA.",
    )


def add_baseline_lock_authorization_argument(parser: argparse.ArgumentParser) -> None:
    """Add the hash-only, pre-training V62 authorization path."""

    parser.add_argument(
        "--baseline-lock",
        required=True,
        help="Create-once V54 hash-only baseline lock; validated before training data.",
    )


def load_filtered_training_qa(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Authenticate and load only the pinned 12-pair training artifact."""

    source = Path(path).expanduser().resolve()
    raw = source.read_bytes()
    if _sha256_bytes(raw) != _PINNED_FILTERED_TRAIN_SHA256:
        raise ValueError("Filtered V62 training QA SHA-256 differs from the pin")
    rows = _decode_jsonl(raw, label="filtered V62 training QA")
    specs_by_id = {spec.pair_id: spec for spec in PAIR_INVENTORY}
    train_specs = tuple(specs_by_id[pair_id] for pair_id in TRAIN_PAIR_IDS)
    inventory = _validate_inventory(rows, specs=train_specs, label="filtered V62 training QA")
    _assert_selection_distribution("training", inventory)
    return rows


def prepare(
    *,
    source_train_qa: str | Path,
    v61_terminal: str | Path,
    filtered_train_output: str | Path,
    validation_questions_output: str | Path,
    scorer_references_output: str | Path,
    preregistration_output: str | Path,
) -> dict[str, Any]:
    """Authenticate once and emit the immutable V62 pair-disjoint boundary."""

    train_path, questions_path, scorer_path, prereg_path = _validate_destinations(
        filtered_train_output=filtered_train_output,
        validation_questions_output=validation_questions_output,
        scorer_references_output=scorer_references_output,
        preregistration_output=preregistration_output,
    )

    # The overwrite refusal above intentionally precedes both input opens.
    terminal_path = Path(v61_terminal).expanduser().resolve()
    _terminal, terminal_raw = _load_v61_terminal_once(terminal_path)
    source_path = Path(source_train_qa).expanduser().resolve()
    source_raw = source_path.read_bytes()  # The complete source QA's sole open/read.
    source_sha256 = _sha256_bytes(source_raw)
    if source_sha256 != _PINNED_SOURCE_QA_SHA256:
        raise ValueError("Diverse-52 source training QA SHA-256 differs from the pin")
    if len(source_raw) != _PINNED_SOURCE_QA_SIZE_BYTES:
        raise ValueError("Diverse-52 source training QA byte count differs from the pin")
    source_rows = _decode_jsonl(source_raw, label="diverse-52 source training QA")
    full_inventory = _validate_inventory(
        source_rows,
        specs=PAIR_INVENTORY,
        label="diverse-52 source training QA",
    )

    train_pair_ids = set(TRAIN_PAIR_IDS)
    validation_pair_ids = set(INTERNAL_VALIDATION_PAIR_IDS)
    if train_pair_ids & validation_pair_ids or train_pair_ids | validation_pair_ids != {
        spec.pair_id for spec in PAIR_INVENTORY
    }:
        raise AssertionError("V62 pair partitions are not disjoint and exhaustive")
    train_rows = tuple(
        row for row in source_rows if row["counterfactual_pair_id"] in train_pair_ids
    )
    validation_rows = tuple(
        row for row in source_rows if row["counterfactual_pair_id"] in validation_pair_ids
    )
    specs_by_id = {spec.pair_id: spec for spec in PAIR_INVENTORY}
    train_specs = tuple(specs_by_id[pair_id] for pair_id in TRAIN_PAIR_IDS)
    validation_specs = tuple(specs_by_id[pair_id] for pair_id in INTERNAL_VALIDATION_PAIR_IDS)
    train_inventory = _validate_inventory(
        train_rows,
        specs=train_specs,
        label="V62 filtered training QA",
    )
    validation_inventory = _validate_inventory(
        validation_rows,
        specs=validation_specs,
        label="V62 internal validation",
    )
    _assert_selection_distribution("training", train_inventory)
    _assert_selection_distribution("internal_validation", validation_inventory)

    filtered_train_bytes = _canonical_jsonl_bytes(train_rows)
    filtered_train_sha256 = _sha256_bytes(filtered_train_bytes)
    if filtered_train_sha256 != _PINNED_FILTERED_TRAIN_SHA256:
        raise AssertionError("Canonical filtered V62 training bytes changed unexpectedly")

    question_manifest = build_question_manifest(
        validation_rows,
        source_qa_sha256=source_sha256,
    )
    question_key_inventory_sha256 = _question_key_sha256(validation_rows)
    if question_key_inventory_sha256 != _PINNED_VALIDATION_QUESTION_KEYS_SHA256:
        raise AssertionError("V62 internal-validation opaque question inventory changed")
    questions_bytes = _canonical_json_bytes(question_manifest.as_dict(), pretty=True)
    scorer_records = [_scorer_record(row) for row in validation_rows]
    scorer_records_sha256 = _sha256_bytes(_canonical_jsonl_bytes(scorer_records))
    scorer_payload = {
        "schema": "semantic_3d_chat.v62.scorer_references.v1",
        "schema_version": 1,
        "source_qa_sha256": source_sha256,
        "question_count": len(scorer_records),
        "pair_count": len(validation_specs),
        "paired_unit_count": validation_inventory["paired_unit_count"],
        "records_sha256": scorer_records_sha256,
        "contains_question_text": False,
        "runtime_access_permitted": False,
        "records": scorer_records,
    }
    scorer_bytes = _canonical_json_bytes(scorer_payload, pretty=True)

    training_scene_ids = [scene_id for spec in train_specs for scene_id in spec.scene_ids]
    validation_scene_ids = [scene_id for spec in validation_specs for scene_id in spec.scene_ids]
    preregistration = {
        "schema": _PREREGISTRATION_SCHEMA,
        "schema_version": 1,
        "artifact": _PREREGISTRATION_ARTIFACT,
        "status": "locked_before_v62_training_or_internal_validation_inference",
        "source": {
            "training_qa_sha256": source_sha256,
            "training_qa_size_bytes": len(source_raw),
            "read_count_during_prepare": 1,
            "full_inventory": full_inventory,
        },
        "preserved_v61_terminal": {
            "artifact": _PINNED_V61_ARTIFACT,
            "sha256": _sha256_bytes(terminal_raw),
            "passed": False,
            "may_be_replaced_or_reinterpreted": False,
        },
        "split": {
            "unit": "counterfactual_pair_id",
            "seed": 20260808,
            "pair_disjoint": True,
            "scene_disjoint": True,
            "exhaustive_over_pinned_training_source": True,
            "training_pair_ids": list(TRAIN_PAIR_IDS),
            "training_scene_ids": training_scene_ids,
            "internal_validation_pair_ids": list(INTERNAL_VALIDATION_PAIR_IDS),
            "internal_validation_scene_ids": validation_scene_ids,
            "protected_scene_numbers_never_loaded": list(_PROTECTED_SCENE_NUMBERS),
        },
        "artifacts": {
            "filtered_training": {
                "sha256": filtered_train_sha256,
                "row_count": len(train_rows),
                "contains_answers_and_route_labels": True,
            },
            "internal_validation_questions": {
                "sha256": _sha256_bytes(questions_bytes),
                "questions_sha256": question_manifest.questions_sha256,
                "question_key_inventory_sha256": question_key_inventory_sha256,
                "row_count": question_manifest.question_count,
                "record_fields": ["scene_id", "question_id", "question"],
                "contains_answers_or_route_labels": False,
            },
            "scorer_references": {
                "sha256": _sha256_bytes(scorer_bytes),
                "records_sha256": scorer_records_sha256,
                "row_count": len(scorer_records),
                "contains_question_text": False,
                "runtime_access_permitted": False,
                "separate_directory_enforced": True,
            },
        },
        "natural_population": {
            "training": train_inventory,
            "internal_validation": validation_inventory,
            "primary_reporting_uses_all_384_sides_without_rebalancing": True,
            "paired_unit_is_two_identically_worded_sides": True,
            "changed_unit_completeness_requires_both_sides_correct": True,
        },
        "required_metric_reporting": {
            "primary": "natural_all_side_exact_normalized_accuracy",
            "breakdowns": [
                "route_label",
                "answer_type",
                "question_family",
                "counterfactual_change_type",
            ],
            "paired": [
                "changed_side_exact",
                "changed_paired_unit_complete",
                "changed_paired_unit_correct_direction",
                "retention_side_exact",
            ],
            "balanced_or_subsampled_metrics_may_replace_natural_metrics": False,
        },
        "thresholds": {
            "internal_validation": {
                "changed_side_exact": {"minimum": 42, "total": 52},
                "changed_paired_unit_complete": {"minimum": 19, "total": 26},
                "changed_paired_unit_correct_direction": {"minimum": 23, "total": 26},
                "retention_exact_no_control_output_identity": {
                    "minimum": 332,
                    "total": 332,
                    "comparison": "exact_utf8_output_bytes_sha256",
                },
                "minimum_complete_changed_units_by_change_type": {
                    "book_support": 2,
                    "chair_orientation": 1,
                    "color_swap": 2,
                    "mirror_lr": 2,
                    "object_count": 1,
                    "object_relocation": 2,
                    "object_removal": 2,
                    "picture_support": 2,
                },
            },
            "same_question_different_prefix_control": {
                "complete_unit_coverage": {"minimum": 26, "total": 26},
                "distinct_scene_prefix_hashes": {"minimum": 26, "total": 26},
                "question_text_identity": {"minimum": 26, "total": 26},
                "changed_side_exact": {"minimum": 42, "total": 52},
                "changed_paired_unit_complete": {"minimum": 19, "total": 26},
                "correct_changed_direction": {"minimum": 23, "total": 26},
            },
            "scene_swap_control": {
                "swapped_side_coverage": {"minimum": 52, "total": 52},
                "question_bytes_unchanged": {"minimum": 52, "total": 52},
                "opposite_prefix_hash_exact": {"minimum": 52, "total": 52},
                "answer_follows_injected_scene": {"minimum": 42, "total": 52},
                "bidirectional_unit_complete": {"minimum": 19, "total": 26},
            },
        },
        "controls": {
            "exact_no_control_identity": {
                "population": "all 332 natural retention sides",
                "candidate_route": "exact base/no-control path",
                "required": "all raw output byte hashes equal a preregistered frozen baseline",
            },
            "same_question_different_prefix": {
                "population": "all 26 changed paired units",
                "question_must_be_byte_identical_within_unit": True,
                "scene_prefix_must_be_computed_before_question": True,
                "scene_prefix_hashes_must_differ_between_sides": True,
            },
            "scene_swap": {
                "population": "both directions for all 26 changed paired units",
                "question_must_remain_unchanged": True,
                "only_scene_prefix_is_swapped": True,
                "expected_answer_follows_injected_prefix_scene": True,
            },
        },
        "data_boundaries": {
            "trainer_data_arguments": list(V62_TRAINER_DATA_ARGUMENTS),
            "trainer_must_not_accept": sorted(V62_PROHIBITED_TRAINER_DATA_ARGUMENTS),
            "trainer_may_load_internal_validation_questions": False,
            "trainer_may_load_scorer_references": False,
            "inference_may_load_scorer_references": False,
            "baseline_lock_required_before_training": True,
            "expected_baseline_lock_schema": _BASELINE_LOCK_SCHEMA,
            "expected_baseline_lock_artifact": _BASELINE_LOCK_ARTIFACT,
            "baseline_lock_must_bind": [
                "this exact preregistration SHA-256",
                "the exact internal-validation question manifest and questions SHA-256",
                "all 384 opaque question keys",
                "all raw UTF-8 V54 output SHA-256 values",
                "one invariant scene-prefix SHA-256 for each of 16 scenes",
                "the exact V54 checkpoint fingerprint",
            ],
            "trainer_must_validate_baseline_lock_before_any_training_input": True,
            "held_out_qa_or_oracle_loaded": False,
            "fresh_development_loaded": False,
            "deferred_final_loaded": False,
        },
    }
    prereg_bytes = _canonical_json_bytes(preregistration, pretty=True)
    _publish_create_once(
        (
            (train_path, filtered_train_bytes),
            (questions_path, questions_bytes),
            (scorer_path, scorer_bytes),
            (prereg_path, prereg_bytes),
        )
    )
    return preregistration


def lock_baseline(
    *,
    predictions: str | Path,
    preregistration: str | Path,
    v54_checkpoint: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Freeze all 384 V54 outputs as hashes before V62 training is authorized.

    The prediction input is a questions-only inference product.  This function
    never opens the filtered training QA or scorer sidecar, and the lock stores
    no answer text.  It binds output hashes to the exact preregistration,
    question inventory, fixed scene prefixes, and V54 checkpoint fingerprint.
    """

    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"V62 baseline lock already exists: {destination}")

    prereg_path = Path(preregistration).expanduser().resolve()
    prereg, prereg_raw = _load_json_object_once(
        prereg_path, label="V62 preregistration"
    )
    _validate_preregistration_payload(prereg)
    if _sha256_bytes(prereg_raw) != PINNED_V62_PREREGISTRATION_SHA256:
        raise ValueError("V62 preregistration bytes differ from the public pin")
    artifacts = prereg["artifacts"]
    questions_artifact = artifacts["internal_validation_questions"]
    prediction_path = Path(predictions).expanduser().resolve()
    prediction_raw = prediction_path.read_bytes()
    prediction_rows = _decode_jsonl(
        prediction_raw, label="V62 V54 baseline predictions"
    )
    raw_keys: list[dict[str, str]] = []
    for index, row in enumerate(prediction_rows, start=1):
        scene_id = row.get("scene_id")
        question_id = row.get("question_id")
        if not isinstance(scene_id, str) or _SCENE_ID.fullmatch(scene_id) is None:
            raise ValueError(f"Baseline prediction {index} has invalid scene_id")
        if not isinstance(question_id, str) or _QUESTION_ID.fullmatch(question_id) is None:
            raise ValueError(f"Baseline prediction {index} has invalid question_id")
        raw_keys.append({"scene_id": scene_id, "question_id": question_id})
    expected_keys_hash = questions_artifact["question_key_inventory_sha256"]
    if _sha256_bytes(_canonical_jsonl_bytes(raw_keys)) != expected_keys_hash:
        raise ValueError("Baseline prediction inventory differs from the questions manifest")
    prediction_rows = _validate_baseline_prediction_rows(
        prediction_raw,
        expected_keys={(row["scene_id"], row["question_id"]) for row in raw_keys},
    )
    checkpoint_sha256, checkpoint_files = checkpoint_fingerprint(v54_checkpoint)
    if checkpoint_sha256 != _PINNED_V54_CHECKPOINT_SHA256:
        raise ValueError("V54 checkpoint fingerprint differs from the pin")
    provenance_path = prediction_path.with_suffix(prediction_path.suffix + ".provenance.json")
    if not provenance_path.is_file():
        raise FileNotFoundError("V62 V54 baseline prediction provenance is unavailable")
    provenance, provenance_raw = _load_json_object_once(
        provenance_path, label="V62 V54 baseline prediction provenance"
    )
    scene_map_manifest = provenance.get("scene_map_manifest")
    if (
        provenance.get("references_sha256") != questions_artifact["sha256"]
        or provenance.get("checkpoint_sha256") != checkpoint_sha256
        or not isinstance(scene_map_manifest, Mapping)
        or set(scene_map_manifest) != set(prereg["split"]["internal_validation_scene_ids"])
    ):
        raise ValueError(
            "V62 baseline provenance does not bind the exact questions, V54 checkpoint, and scenes"
        )

    prefix_by_scene: defaultdict[str, set[str]] = defaultdict(set)
    output_hashes: list[dict[str, str]] = []
    for row in prediction_rows:
        scene_id = str(row["scene_id"])
        prefix_by_scene[scene_id].add(str(row["prefix_hash"]))
        output_hashes.append(
            {
                "scene_id": scene_id,
                "question_id": str(row["question_id"]),
                "raw_output_sha256": hashlib.sha256(
                    str(row["predicted_answer"]).encode("utf-8")
                ).hexdigest(),
            }
        )
    expected_scenes = set(prereg["split"]["internal_validation_scene_ids"])
    if set(prefix_by_scene) != expected_scenes:
        raise ValueError("Baseline predictions do not cover all 16 internal-validation scenes")
    if any(len(prefixes) != 1 for prefixes in prefix_by_scene.values()):
        raise ValueError("V54 baseline must prove one invariant prefix per scene")
    fixed_prefixes = {
        scene_id: next(iter(prefix_by_scene[scene_id])) for scene_id in sorted(prefix_by_scene)
    }
    if len(set(fixed_prefixes.values())) != len(fixed_prefixes):
        raise ValueError("V54 baseline prefix hashes must be distinct across all 16 scenes")

    question_keys_sha256 = _question_key_sha256(prediction_rows)
    expected_question_keys_sha256 = questions_artifact["question_key_inventory_sha256"]
    if question_keys_sha256 != expected_question_keys_sha256:
        raise ValueError("Baseline opaque question-key ordering differs from the pin")
    output_hashes_sha256 = _sha256_bytes(_canonical_jsonl_bytes(output_hashes))
    result = {
        "schema": _BASELINE_LOCK_SCHEMA,
        "schema_version": 1,
        "artifact": _BASELINE_LOCK_ARTIFACT,
        "status": "locked_before_v62_training",
        "preregistration_sha256": _sha256_bytes(prereg_raw),
        "questions_manifest_sha256": questions_artifact["sha256"],
        "questions_sha256": questions_artifact["questions_sha256"],
        "question_key_inventory_sha256": question_keys_sha256,
        "predictions_sha256": _sha256_bytes(prediction_raw),
        "prediction_provenance_sha256": _sha256_bytes(provenance_raw),
        "v54_checkpoint_sha256": checkpoint_sha256,
        "v54_checkpoint_files": checkpoint_files,
        "question_count": len(output_hashes),
        "scene_count": len(fixed_prefixes),
        "scene_prefix_hashes": fixed_prefixes,
        "one_invariant_prefix_per_scene": True,
        "distinct_prefix_per_scene": True,
        "required_output_hashes": output_hashes,
        "required_output_hashes_sha256": output_hashes_sha256,
        "environmental_answer_text_stored": False,
        "question_text_stored": False,
        "filtered_training_qa_loaded": False,
        "scorer_references_loaded": False,
        "held_out_qa_or_oracle_loaded": False,
        "fresh_development_loaded": False,
        "deferred_final_loaded": False,
    }
    _publish_create_once(((destination, _canonical_json_bytes(result, pretty=True)),))
    return result


def validate_baseline_lock(
    path: str | Path,
    *,
    preregistration: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the hash-only authorization a V62 trainer must require first.

    A trainer calls this with only ``path`` and the public pins in this module.
    Supplying ``preregistration`` is an optional preparation/audit cross-check,
    not a trainer input.
    """

    prereg: dict[str, Any] | None = None
    if preregistration is not None:
        prereg_path = Path(preregistration).expanduser().resolve()
        prereg, prereg_raw = _load_json_object_once(
            prereg_path, label="V62 preregistration"
        )
        _validate_preregistration_payload(prereg)
        if _sha256_bytes(prereg_raw) != PINNED_V62_PREREGISTRATION_SHA256:
            raise ValueError("V62 preregistration bytes differ from the public pin")
    lock_path = Path(path).expanduser().resolve()
    lock, _lock_raw = _load_json_object_once(lock_path, label="V62 baseline lock")
    expected_fields = {
        "schema",
        "schema_version",
        "artifact",
        "status",
        "preregistration_sha256",
        "questions_manifest_sha256",
        "questions_sha256",
        "question_key_inventory_sha256",
        "predictions_sha256",
        "prediction_provenance_sha256",
        "v54_checkpoint_sha256",
        "v54_checkpoint_files",
        "question_count",
        "scene_count",
        "scene_prefix_hashes",
        "one_invariant_prefix_per_scene",
        "distinct_prefix_per_scene",
        "required_output_hashes",
        "required_output_hashes_sha256",
        "environmental_answer_text_stored",
        "question_text_stored",
        "filtered_training_qa_loaded",
        "scorer_references_loaded",
        "held_out_qa_or_oracle_loaded",
        "fresh_development_loaded",
        "deferred_final_loaded",
    }
    if set(lock) != expected_fields:
        raise ValueError("V62 baseline lock fields differ from the strict schema")
    if (
        lock["schema"] != _BASELINE_LOCK_SCHEMA
        or lock["schema_version"] != 1
        or lock["artifact"] != _BASELINE_LOCK_ARTIFACT
        or lock["status"] != "locked_before_v62_training"
        or lock["preregistration_sha256"] != PINNED_V62_PREREGISTRATION_SHA256
        or lock["questions_manifest_sha256"] != PINNED_V62_QUESTIONS_MANIFEST_SHA256
        or lock["questions_sha256"] != PINNED_V62_QUESTIONS_SHA256
        or lock["question_key_inventory_sha256"]
        != _PINNED_VALIDATION_QUESTION_KEYS_SHA256
        or lock["v54_checkpoint_sha256"] != _PINNED_V54_CHECKPOINT_SHA256
        or lock["question_count"] != 384
        or lock["scene_count"] != 16
        or lock["one_invariant_prefix_per_scene"] is not True
        or lock["distinct_prefix_per_scene"] is not True
        or lock["environmental_answer_text_stored"] is not False
        or lock["question_text_stored"] is not False
        or lock["filtered_training_qa_loaded"] is not False
        or lock["scorer_references_loaded"] is not False
        or lock["held_out_qa_or_oracle_loaded"] is not False
        or lock["fresh_development_loaded"] is not False
        or lock["deferred_final_loaded"] is not False
    ):
        raise ValueError("V62 baseline lock prerequisite binding is invalid")

    prefixes = lock["scene_prefix_hashes"]
    specs_by_id = {spec.pair_id: spec for spec in PAIR_INVENTORY}
    expected_scenes = {
        scene_id
        for pair_id in INTERNAL_VALIDATION_PAIR_IDS
        for scene_id in specs_by_id[pair_id].scene_ids
    }
    if (
        not isinstance(prefixes, dict)
        or set(prefixes) != expected_scenes
        or any(not isinstance(value, str) or _SHA256.fullmatch(value) is None for value in prefixes.values())
        or len(set(prefixes.values())) != 16
    ):
        raise ValueError("V62 baseline lock prefix inventory is invalid")
    output_hashes = lock["required_output_hashes"]
    if not isinstance(output_hashes, list) or len(output_hashes) != 384:
        raise ValueError("V62 baseline lock output hash inventory is invalid")
    for record in output_hashes:
        if (
            not isinstance(record, dict)
            or set(record) != {"scene_id", "question_id", "raw_output_sha256"}
            or not isinstance(record["raw_output_sha256"], str)
            or _SHA256.fullmatch(record["raw_output_sha256"]) is None
        ):
            raise ValueError("V62 baseline lock contains an invalid output-hash record")
    if (
        _question_key_sha256(output_hashes) != _PINNED_VALIDATION_QUESTION_KEYS_SHA256
        or _sha256_bytes(_canonical_jsonl_bytes(output_hashes))
        != lock["required_output_hashes_sha256"]
    ):
        raise ValueError("V62 baseline lock output hash inventory changed")
    if prereg is not None and (
        prereg["artifacts"]["internal_validation_questions"]["sha256"]
        != PINNED_V62_QUESTIONS_MANIFEST_SHA256
        or prereg["artifacts"]["internal_validation_questions"]["questions_sha256"]
        != PINNED_V62_QUESTIONS_SHA256
    ):
        raise ValueError("V62 baseline lock/preregistration question binding changed")
    return lock


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--source-train-qa", required=True)
    prepare_parser.add_argument("--v61-terminal", required=True)
    prepare_parser.add_argument("--filtered-train-output", required=True)
    prepare_parser.add_argument("--validation-questions-output", required=True)
    prepare_parser.add_argument("--scorer-references-output", required=True)
    prepare_parser.add_argument("--preregistration-output", required=True)
    baseline_parser = commands.add_parser("lock-baseline")
    baseline_parser.add_argument("--predictions", required=True)
    baseline_parser.add_argument("--preregistration", required=True)
    baseline_parser.add_argument("--v54-checkpoint", required=True)
    baseline_parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(
            source_train_qa=args.source_train_qa,
            v61_terminal=args.v61_terminal,
            filtered_train_output=args.filtered_train_output,
            validation_questions_output=args.validation_questions_output,
            scorer_references_output=args.scorer_references_output,
            preregistration_output=args.preregistration_output,
        )
    elif args.command == "lock-baseline":
        result = lock_baseline(
            predictions=args.predictions,
            preregistration=args.preregistration,
            v54_checkpoint=args.v54_checkpoint,
            output=args.output,
        )
    else:
        raise AssertionError(f"Unhandled V62 command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
