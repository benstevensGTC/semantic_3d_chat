"""Strict, inference-safe question manifests.

The primary and control prediction processes import this module instead of the
QA dataset reader.  A manifest contains only opaque identifiers, user question
text, counts, and integrity hashes.  It cannot carry reference answers,
grounding targets, oracle identifiers, or arbitrary metadata.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

QUESTION_MANIFEST_SCHEMA: Final[str] = "semantic_3d_chat.questions.v1"
QUESTION_MANIFEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "schema_version",
        "question_count",
        "scene_count",
        "questions_sha256",
        "source_qa_sha256",
        "questions",
    }
)
QUESTION_RECORD_KEYS: Final[frozenset[str]] = frozenset(
    {"scene_id", "question_id", "question"}
)
_SCENE_ID = re.compile(r"scene_[0-9]{6}")
_QUESTION_ID = re.compile(r"q_[0-9]{6}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PROHIBITED_RUNTIME_PATH_PARTS: Final[frozenset[str]] = frozenset({"oracle", "qa"})


@dataclass(frozen=True)
class QuestionRecord:
    """The complete environmental-question input allowed at inference."""

    scene_id: str
    question_id: str
    question: str

    def as_dict(self) -> dict[str, str]:
        return {
            "scene_id": self.scene_id,
            "question_id": self.question_id,
            "question": self.question,
        }


@dataclass(frozen=True)
class QuestionManifest:
    """Validated question records and their deterministic provenance."""

    questions: tuple[QuestionRecord, ...]
    questions_sha256: str
    source_qa_sha256: str
    manifest_path: Path | None = None
    manifest_sha256: str | None = None

    @property
    def question_count(self) -> int:
        return len(self.questions)

    @property
    def scene_count(self) -> int:
        return len({record.scene_id for record in self.questions})

    def by_scene(self) -> dict[str, list[QuestionRecord]]:
        grouped: defaultdict[str, list[QuestionRecord]] = defaultdict(list)
        for record in self.questions:
            grouped[record.scene_id].append(record)
        return dict(grouped)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": QUESTION_MANIFEST_SCHEMA,
            "schema_version": 1,
            "question_count": self.question_count,
            "scene_count": self.scene_count,
            "questions_sha256": self.questions_sha256,
            "source_qa_sha256": self.source_qa_sha256,
            "questions": [record.as_dict() for record in self.questions],
        }


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of a file without interpreting its contents."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_question_bytes(records: Sequence[QuestionRecord]) -> bytes:
    return b"".join(
        (
            json.dumps(
                record.as_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for record in records
    )


def questions_sha256(records: Sequence[QuestionRecord]) -> str:
    """Hash the ordered, canonical three-field question records."""

    return hashlib.sha256(_canonical_question_bytes(records)).hexdigest()


def _validate_record(raw: Mapping[str, Any], index: int) -> QuestionRecord:
    keys = set(raw)
    if keys != QUESTION_RECORD_KEYS:
        missing = sorted(QUESTION_RECORD_KEYS - keys)
        unexpected = sorted(keys - QUESTION_RECORD_KEYS)
        raise ValueError(
            f"Question record {index} has invalid fields; missing={missing}, "
            f"unexpected={unexpected}"
        )
    scene_id = raw["scene_id"]
    question_id = raw["question_id"]
    question = raw["question"]
    if not isinstance(scene_id, str) or not _SCENE_ID.fullmatch(scene_id):
        raise ValueError(f"Question record {index} has a non-opaque scene_id")
    if not isinstance(question_id, str) or not _QUESTION_ID.fullmatch(question_id):
        raise ValueError(f"Question record {index} has a non-opaque question_id")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Question record {index} needs non-empty question text")
    if question != question.strip():
        raise ValueError(f"Question record {index} question has surrounding whitespace")
    if "\x00" in question:
        raise ValueError(f"Question record {index} question contains a NUL character")
    return QuestionRecord(scene_id=scene_id, question_id=question_id, question=question)


def build_question_manifest(
    records: Sequence[Mapping[str, Any]], *, source_qa_sha256: str
) -> QuestionManifest:
    """Build a sanitized manifest from QA-side mappings.

    This is the one deliberate declassification step: only the three named
    fields are copied.  The returned object is validated exactly like a manifest
    loaded by inference.
    """

    if not _SHA256.fullmatch(source_qa_sha256):
        raise ValueError("source_qa_sha256 must be a lowercase SHA-256 digest")
    sanitized = tuple(
        _validate_record(
            {
                "scene_id": raw.get("scene_id"),
                "question_id": raw.get("question_id"),
                "question": raw.get("question"),
            },
            index,
        )
        for index, raw in enumerate(records)
    )
    _validate_unique_nonempty(sanitized)
    return QuestionManifest(
        questions=sanitized,
        questions_sha256=questions_sha256(sanitized),
        source_qa_sha256=source_qa_sha256,
    )


def _validate_unique_nonempty(records: Sequence[QuestionRecord]) -> None:
    if not records:
        raise ValueError("Question manifest contains no questions")
    seen: set[tuple[str, str]] = set()
    for record in records:
        key = (record.scene_id, record.question_id)
        if key in seen:
            raise ValueError(f"Duplicate question key: {key}")
        seen.add(key)


def _validate_manifest_payload(payload: Any) -> QuestionManifest:
    if not isinstance(payload, dict):
        raise TypeError("Question manifest must be a JSON object")
    keys = set(payload)
    if keys != QUESTION_MANIFEST_KEYS:
        missing = sorted(QUESTION_MANIFEST_KEYS - keys)
        unexpected = sorted(keys - QUESTION_MANIFEST_KEYS)
        raise ValueError(
            f"Question manifest has invalid fields; missing={missing}, unexpected={unexpected}"
        )
    if payload["schema"] != QUESTION_MANIFEST_SCHEMA or payload["schema_version"] != 1:
        raise ValueError("Unsupported question manifest schema")
    raw_questions = payload["questions"]
    if not isinstance(raw_questions, list):
        raise TypeError("Question manifest questions must be a list")
    questions: list[QuestionRecord] = []
    for index, raw in enumerate(raw_questions):
        if not isinstance(raw, dict):
            raise TypeError(f"Question record {index} must be a JSON object")
        questions.append(_validate_record(raw, index))
    records = tuple(questions)
    _validate_unique_nonempty(records)
    expected_question_count = payload["question_count"]
    expected_scene_count = payload["scene_count"]
    if (
        not isinstance(expected_question_count, int)
        or isinstance(expected_question_count, bool)
        or expected_question_count != len(records)
    ):
        raise ValueError("Question manifest question_count does not match its records")
    actual_scene_count = len({record.scene_id for record in records})
    if (
        not isinstance(expected_scene_count, int)
        or isinstance(expected_scene_count, bool)
        or expected_scene_count != actual_scene_count
    ):
        raise ValueError("Question manifest scene_count does not match its records")
    recorded_questions_hash = payload["questions_sha256"]
    source_hash = payload["source_qa_sha256"]
    if not isinstance(recorded_questions_hash, str) or not _SHA256.fullmatch(
        recorded_questions_hash
    ):
        raise ValueError("Question manifest questions_sha256 is invalid")
    if not isinstance(source_hash, str) or not _SHA256.fullmatch(source_hash):
        raise ValueError("Question manifest source_qa_sha256 is invalid")
    actual_questions_hash = questions_sha256(records)
    if recorded_questions_hash != actual_questions_hash:
        raise ValueError("Question manifest content hash does not match its questions")
    return QuestionManifest(
        questions=records,
        questions_sha256=actual_questions_hash,
        source_qa_sha256=source_hash,
    )


def _reject_qa_or_oracle_path(path: Path) -> None:
    overlap = _PROHIBITED_RUNTIME_PATH_PARTS & {part.casefold() for part in path.parts}
    if overlap:
        raise ValueError(
            "Inference question manifests cannot be loaded from QA/oracle directories: "
            f"{path}"
        )


def load_question_manifest(path: str | Path) -> QuestionManifest:
    """Load and fully validate an inference manifest before model construction.

    The path check intentionally happens before the first file open so an
    inference process cannot be pointed at ``data/qa`` or ``data/oracle``.
    """

    source = Path(path).expanduser().resolve()
    _reject_qa_or_oracle_path(source)
    if not source.is_file():
        raise FileNotFoundError(f"Question manifest does not exist: {source}")
    raw_bytes = source.read_bytes()
    try:
        payload = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid question manifest JSON: {source}") from error
    manifest = _validate_manifest_payload(payload)
    return QuestionManifest(
        questions=manifest.questions,
        questions_sha256=manifest.questions_sha256,
        source_qa_sha256=manifest.source_qa_sha256,
        manifest_path=source,
        manifest_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def validate_question_manifest(manifest: QuestionManifest) -> QuestionManifest:
    """Revalidate an in-memory manifest passed to an inference runner."""

    if not isinstance(manifest, QuestionManifest):
        raise TypeError("Inference requires a validated QuestionManifest")
    validated = _validate_manifest_payload(manifest.as_dict())
    return QuestionManifest(
        questions=validated.questions,
        questions_sha256=validated.questions_sha256,
        source_qa_sha256=validated.source_qa_sha256,
        manifest_path=manifest.manifest_path,
        manifest_sha256=manifest.manifest_sha256,
    )
