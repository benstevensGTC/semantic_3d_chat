from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.control_predict import run_control_suite
from semantic_3d_chat.evaluation.prepare_questions import prepare_question_manifest
from semantic_3d_chat.evaluation.question_manifest import load_question_manifest


def _qa_records() -> list[dict]:
    return [
        {
            "scene_id": "scene_000005",
            "question_id": "q_000001",
            "question": "What is on the table?",
            "answer": "book, cube",
            "answer_type": "support",
            "target_xyz": [1.0, 2.0, 0.8],
            "target_instance": "i_000100",
            "oracle_relationship": "on",
        },
        {
            "scene_id": "scene_000006",
            "question_id": "q_000002",
            "question": "Is the cube under the table?",
            "answer": "yes",
            "answer_type": "presence",
            "target_xyz": [0.5, 1.5, 0.2],
        },
    ]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_prep_exports_only_questions_and_verifiable_content_hashes(tmp_path: Path) -> None:
    qa_path = tmp_path / "data" / "qa" / "test.jsonl"
    manifest_path = tmp_path / "reports" / "questions" / "test.json"
    _write_jsonl(qa_path, _qa_records())

    manifest = prepare_question_manifest(qa_path, manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert all(
        set(record) == {"scene_id", "question_id", "question"}
        for record in payload["questions"]
    )
    serialized = json.dumps(payload, sort_keys=True).casefold()
    for forbidden in (
        "answer_type",
        "target_xyz",
        "target_instance",
        "oracle_relationship",
        "book, cube",
    ):
        assert forbidden not in serialized
    assert manifest.source_qa_sha256 == hashlib.sha256(qa_path.read_bytes()).hexdigest()
    assert manifest.manifest_sha256 == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert manifest.question_count == 2
    assert manifest.scene_count == 2


@pytest.mark.parametrize(
    "forbidden_field, forbidden_value",
    [
        ("answer", "yes"),
        ("target_xyz", [0.0, 0.0, 0.0]),
        ("target_instance", "i_000100"),
        ("oracle", {"category": "cube"}),
    ],
)
def test_inference_manifest_rejects_extra_question_fields(
    tmp_path: Path, forbidden_field: str, forbidden_value: object
) -> None:
    qa_path = tmp_path / "source" / "test.jsonl"
    manifest_path = tmp_path / "runtime" / "questions.json"
    _write_jsonl(qa_path, _qa_records())
    prepare_question_manifest(qa_path, manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["questions"][0][forbidden_field] = forbidden_value
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected"):
        load_question_manifest(manifest_path)


def test_inference_manifest_rejects_extra_top_level_oracle_field(tmp_path: Path) -> None:
    qa_path = tmp_path / "source" / "test.jsonl"
    manifest_path = tmp_path / "runtime" / "questions.json"
    _write_jsonl(qa_path, _qa_records())
    prepare_question_manifest(qa_path, manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["oracle_path"] = "data/oracle/scene.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected"):
        load_question_manifest(manifest_path)


def test_inference_manifest_detects_question_tampering(tmp_path: Path) -> None:
    qa_path = tmp_path / "source" / "test.jsonl"
    manifest_path = tmp_path / "runtime" / "questions.json"
    _write_jsonl(qa_path, _qa_records())
    prepare_question_manifest(qa_path, manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["questions"][0]["question"] = "A different question?"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="content hash"):
        load_question_manifest(manifest_path)


def test_runtime_rejects_qa_path_before_opening_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qa_path = tmp_path / "data" / "qa" / "questions.json"
    qa_path.parent.mkdir(parents=True)
    qa_path.write_text("{}", encoding="utf-8")
    opened = False
    original_read_bytes = Path.read_bytes

    def tracked_read_bytes(path: Path) -> bytes:
        nonlocal opened
        if path.resolve() == qa_path.resolve():
            opened = True
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", tracked_read_bytes)
    with pytest.raises(ValueError, match="QA/oracle"):
        load_question_manifest(qa_path)
    assert opened is False


def test_control_inference_requires_validated_manifest(tmp_path: Path) -> None:
    leaked_records = [
        {
            "scene_id": "scene_000001",
            "question_id": "q_000001",
            "question": "Is anything present?",
            "answer": "oracle answer",
        }
    ]
    with pytest.raises(TypeError, match="validated QuestionManifest"):
        run_control_suite(  # type: ignore[arg-type]
            leaked_records,
            runtime_builder=lambda *_: pytest.fail("runtime must not be built"),
            output_directory=tmp_path,
            conditions=("primary",),
        )
