"""Strict schemas for the evaluation-only oracle-text upper bound.

This module deliberately contains no simulator/oracle reader, QA reader, or
model loader.  It is the narrow data boundary shared by the physically
separate preparation, inference, and scoring commands.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.evaluation.baseline_io import read_jsonl, sha256_file

SCENE_TEXT_SCHEMA: Final[str] = "semantic_3d_chat.oracle_text_scenes.v1"
PREDICTION_PROVENANCE_SCHEMA: Final[str] = "semantic_3d_chat.oracle_text_prediction_provenance.v1"
PREDICTION_REPORT_SCHEMA: Final[str] = "semantic_3d_chat.oracle_text_prediction_report.v1"
PREDICTION_BASELINE: Final[str] = "oracle_text_upper_bound"

V55_DEVELOPMENT_SCENE_IDS: Final[tuple[str, ...]] = tuple(
    f"scene_{index:06d}" for index in range(19, 25)
)
V55_DEVELOPMENT_QUESTION_COUNT: Final[int] = 216

_SCENE_ID = re.compile(r"scene_[0-9]{6}")
_QUESTION_ID = re.compile(r"q_[0-9]{6}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PROHIBITED_INFERENCE_PATH_PARTS: Final[frozenset[str]] = frozenset({"oracle", "qa"})

_SCENE_TEXT_BUNDLE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "description_kind",
        "evaluation_only",
        "primary_path_eligible",
        "prohibited_primary_input",
        "question_independent",
        "question_manifest_sha256",
        "questions_sha256",
        "scene_count",
        "scene_descriptions_sha256",
        "scenes",
        "schema",
        "schema_version",
        "source_qa_sha256",
    }
)
_SCENE_TEXT_RECORD_KEYS: Final[frozenset[str]] = frozenset(
    {"scene_id", "scene_text", "scene_text_sha256", "source_oracle_sha256"}
)
_PREDICTION_RECORD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "answer",
        "baseline",
        "evaluation_only",
        "inference_provenance_sha256",
        "primary_path_eligible",
        "prohibited_primary_input",
        "question_id",
        "question_sha256",
        "scene_id",
        "scene_text_sha256",
    }
)


def canonical_json_sha256(value: Any) -> str:
    """Hash deterministic JSON without permitting non-finite values."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], field: str) -> None:
    keys = frozenset(value)
    if keys != expected:
        raise ValueError(
            f"{field} has invalid fields; missing={sorted(expected - keys)} "
            f"unexpected={sorted(keys - expected)}"
        )


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_opaque_id(value: object, pattern: re.Pattern[str], field: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{field} must be an opaque identifier")
    return value


def reject_oracle_or_qa_path(path: str | Path, *, label: str) -> Path:
    """Reject inference inputs located in directories reserved for hidden truth."""

    source = Path(path).expanduser().resolve()
    overlap = _PROHIBITED_INFERENCE_PATH_PARTS & {
        component.casefold() for component in source.parts
    }
    if overlap:
        raise ValueError(
            f"{label} cannot be loaded from QA/oracle directories during inference: {source}"
        )
    return source


@dataclass(frozen=True)
class SceneTextRecord:
    scene_id: str
    scene_text: str
    scene_text_sha256: str
    source_oracle_sha256: str

    def as_dict(self) -> dict[str, str]:
        return {
            "scene_id": self.scene_id,
            "scene_text": self.scene_text,
            "scene_text_sha256": self.scene_text_sha256,
            "source_oracle_sha256": self.source_oracle_sha256,
        }


@dataclass(frozen=True)
class SceneTextBundle:
    scenes: tuple[SceneTextRecord, ...]
    question_manifest_sha256: str
    questions_sha256: str
    source_qa_sha256: str
    scene_descriptions_sha256: str
    path: Path | None = None
    file_sha256: str | None = None

    def by_scene(self) -> dict[str, SceneTextRecord]:
        return {record.scene_id: record for record in self.scenes}

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCENE_TEXT_SCHEMA,
            "schema_version": 1,
            "evaluation_only": True,
            "primary_path_eligible": False,
            "prohibited_primary_input": True,
            "description_kind": "question_independent_exact_scene_facts",
            "question_independent": True,
            "question_manifest_sha256": self.question_manifest_sha256,
            "questions_sha256": self.questions_sha256,
            "source_qa_sha256": self.source_qa_sha256,
            "scene_count": len(self.scenes),
            "scene_descriptions_sha256": self.scene_descriptions_sha256,
            "scenes": [record.as_dict() for record in self.scenes],
        }


def scene_descriptions_sha256(records: Sequence[SceneTextRecord]) -> str:
    return canonical_json_sha256([record.as_dict() for record in records])


def build_scene_text_bundle(
    records: Sequence[SceneTextRecord],
    *,
    question_manifest_sha256: str,
    questions_sha256: str,
    source_qa_sha256: str,
) -> SceneTextBundle:
    if not records:
        raise ValueError("At least one scene description is required")
    ordered = tuple(sorted(records, key=lambda record: record.scene_id))
    scene_ids = [record.scene_id for record in ordered]
    if len(set(scene_ids)) != len(scene_ids):
        raise ValueError("Scene description IDs must be unique")
    for record in ordered:
        _require_opaque_id(record.scene_id, _SCENE_ID, "scene_id")
        if not record.scene_text or record.scene_text != record.scene_text.strip():
            raise ValueError(f"{record.scene_id} scene text must be non-empty and stripped")
        if text_sha256(record.scene_text) != record.scene_text_sha256:
            raise ValueError(f"{record.scene_id} scene text hash does not match")
        _require_sha256(record.source_oracle_sha256, "source_oracle_sha256")
    return SceneTextBundle(
        scenes=ordered,
        question_manifest_sha256=_require_sha256(
            question_manifest_sha256, "question_manifest_sha256"
        ),
        questions_sha256=_require_sha256(questions_sha256, "questions_sha256"),
        source_qa_sha256=_require_sha256(source_qa_sha256, "source_qa_sha256"),
        scene_descriptions_sha256=scene_descriptions_sha256(ordered),
    )


def load_scene_text_bundle(path: str | Path) -> SceneTextBundle:
    """Load the sole environmental-text artifact allowed by this control."""

    source = reject_oracle_or_qa_path(path, label="Scene-description bundle")
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"Scene-description bundle is unavailable or unsafe: {source}")
    raw_bytes = source.read_bytes()
    try:
        payload = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid scene-description JSON: {source}") from error
    if not isinstance(payload, Mapping):
        raise TypeError("Scene-description bundle must be a JSON object")
    _require_exact_keys(payload, _SCENE_TEXT_BUNDLE_KEYS, "Scene-description bundle")
    if payload["schema"] != SCENE_TEXT_SCHEMA or payload["schema_version"] != 1:
        raise ValueError("Unsupported scene-description schema")
    required_flags = {
        "evaluation_only": True,
        "primary_path_eligible": False,
        "prohibited_primary_input": True,
        "question_independent": True,
        "description_kind": "question_independent_exact_scene_facts",
    }
    if any(payload[key] != expected for key, expected in required_flags.items()):
        raise ValueError("Scene-description control flags were weakened")
    raw_scenes = payload["scenes"]
    if not isinstance(raw_scenes, list):
        raise TypeError("Scene-description scenes must be a list")
    records: list[SceneTextRecord] = []
    for index, raw in enumerate(raw_scenes):
        if not isinstance(raw, Mapping):
            raise TypeError(f"Scene-description record {index} must be an object")
        _require_exact_keys(raw, _SCENE_TEXT_RECORD_KEYS, f"Scene-description record {index}")
        scene_id = _require_opaque_id(raw["scene_id"], _SCENE_ID, "scene_id")
        scene_text = raw["scene_text"]
        if not isinstance(scene_text, str) or not scene_text or scene_text != scene_text.strip():
            raise ValueError(f"{scene_id} scene text must be non-empty and stripped")
        recorded_text_hash = _require_sha256(raw["scene_text_sha256"], "scene_text_sha256")
        if text_sha256(scene_text) != recorded_text_hash:
            raise ValueError(f"{scene_id} scene text hash does not match")
        records.append(
            SceneTextRecord(
                scene_id=scene_id,
                scene_text=scene_text,
                scene_text_sha256=recorded_text_hash,
                source_oracle_sha256=_require_sha256(
                    raw["source_oracle_sha256"], "source_oracle_sha256"
                ),
            )
        )
    bundle = build_scene_text_bundle(
        records,
        question_manifest_sha256=_require_sha256(
            payload["question_manifest_sha256"], "question_manifest_sha256"
        ),
        questions_sha256=_require_sha256(payload["questions_sha256"], "questions_sha256"),
        source_qa_sha256=_require_sha256(payload["source_qa_sha256"], "source_qa_sha256"),
    )
    if payload["scene_count"] != len(bundle.scenes):
        raise ValueError("Scene-description scene_count does not match")
    if payload["scene_descriptions_sha256"] != bundle.scene_descriptions_sha256:
        raise ValueError("Scene-description content hash does not match")
    return SceneTextBundle(
        scenes=bundle.scenes,
        question_manifest_sha256=bundle.question_manifest_sha256,
        questions_sha256=bundle.questions_sha256,
        source_qa_sha256=bundle.source_qa_sha256,
        scene_descriptions_sha256=bundle.scene_descriptions_sha256,
        path=source,
        file_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def prediction_provenance_sha256(payload: Mapping[str, Any]) -> str:
    """Validate and hash the immutable identity portion of a provenance sidecar."""

    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        raise TypeError("Prediction provenance identity must be an object")
    return canonical_json_sha256(identity)


def load_prediction_provenance(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"Prediction provenance is unavailable or unsafe: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Prediction provenance must be a JSON object")
    if payload.get("schema") != PREDICTION_PROVENANCE_SCHEMA or payload.get("schema_version") != 1:
        raise ValueError("Unsupported oracle-text prediction provenance schema")
    expected = prediction_provenance_sha256(payload)
    if payload.get("inference_provenance_sha256") != expected:
        raise ValueError("Prediction provenance hash does not match its identity")
    identity = payload["identity"]
    if (
        identity.get("baseline") != PREDICTION_BASELINE
        or identity.get("evaluation_only") is not True
        or identity.get("primary_path_eligible") is not False
        or identity.get("prohibited_primary_input") is not True
    ):
        raise ValueError("Prediction provenance control flags were weakened")
    return payload


def load_prediction_records(path: str | Path, *, provenance_sha256: str) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"Prediction artifact is unavailable or unsafe: {source}")
    records = read_jsonl(source)
    seen: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        _require_exact_keys(record, _PREDICTION_RECORD_KEYS, f"Prediction record {index}")
        scene_id = _require_opaque_id(record["scene_id"], _SCENE_ID, "scene_id")
        question_id = _require_opaque_id(record["question_id"], _QUESTION_ID, "question_id")
        key = (scene_id, question_id)
        if key in seen:
            raise ValueError(f"Duplicate oracle-text prediction key: {key}")
        seen.add(key)
        if not isinstance(record["answer"], str):
            raise TypeError(f"Prediction {key} answer must be text")
        _require_sha256(record["question_sha256"], "question_sha256")
        _require_sha256(record["scene_text_sha256"], "scene_text_sha256")
        if record["inference_provenance_sha256"] != provenance_sha256:
            raise ValueError(f"Prediction {key} has incorrect inference provenance")
        if (
            record["baseline"] != PREDICTION_BASELINE
            or record["evaluation_only"] is not True
            or record["primary_path_eligible"] is not False
            or record["prohibited_primary_input"] is not True
        ):
            raise ValueError(f"Prediction {key} control flags were weakened")
    return records


def validate_v55_development_scope(
    scene_ids: Sequence[str], question_count: int, *, required: bool
) -> None:
    if not required:
        return
    if tuple(sorted(set(scene_ids))) != V55_DEVELOPMENT_SCENE_IDS:
        raise ValueError("Oracle-text V55 mode requires exactly development scenes 19--24")
    if question_count != V55_DEVELOPMENT_QUESTION_COUNT:
        raise ValueError("Oracle-text V55 mode requires exactly 216 development questions")


def default_provenance_path(predictions_path: str | Path) -> Path:
    destination = Path(predictions_path).expanduser().resolve()
    return destination.with_name(f"{destination.name}.provenance.json")


def default_prediction_report_path(predictions_path: str | Path) -> Path:
    destination = Path(predictions_path).expanduser().resolve()
    return destination.with_name(f"{destination.name}.report.json")


def authenticate_completed_prediction_report(
    report_path: str | Path,
    *,
    predictions_path: str | Path,
    provenance_path: str | Path,
) -> dict[str, Any]:
    source = Path(report_path).expanduser().resolve()
    predictions_source = Path(predictions_path).expanduser().resolve()
    provenance_source = Path(provenance_path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"Completed prediction report is unavailable or unsafe: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Completed prediction report must be a JSON object")
    if (
        payload.get("schema") != PREDICTION_REPORT_SCHEMA
        or payload.get("schema_version") != 1
        or payload.get("status") != "complete"
    ):
        raise ValueError("Oracle-text prediction report is not a completed v1 artifact")
    if payload.get("predictions_sha256") != sha256_file(predictions_source):
        raise ValueError("Completed prediction report does not authenticate predictions")
    if payload.get("provenance_file_sha256") != sha256_file(provenance_source):
        raise ValueError("Completed prediction report does not authenticate provenance")
    return payload
