from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from semantic_3d_chat.evaluation import v96_deferred_final_qa as qa


def _record(scene: str, question_id: str) -> dict[str, Any]:
    return {
        "scene_id": scene,
        "question_id": question_id,
        "question": "Is the object present?",
        "answer": "yes",
    }


def test_v96_authorization_precedes_first_label_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    raw = tmp_path / "raw.jsonl"
    final = tmp_path / "test.jsonl"
    manifest = tmp_path / "selection.json"
    raw.write_text(json.dumps(_record("scene_000025", "q_000001")) + "\n")

    def authenticate(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        events.append("unlock")
        return {
            "unlock_file_sha256": "a" * 64,
            "unlock_identity_sha256": "b" * 64,
            "candidate_fingerprint_sha256": "c" * 64,
        }

    def selector(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        events.append("selector")
        return records, {
            "selection_uses_question_or_answer_text": False,
            "selection_uses_answer_values": False,
        }

    monkeypatch.setattr(qa, "RAW_QA", raw)
    monkeypatch.setattr(qa, "FINAL_QA", final)
    monkeypatch.setattr(qa, "SELECTION_MANIFEST", manifest)
    monkeypatch.setattr(qa, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(qa, "_authenticate_deferred_final_unlock_under_guard_v96", authenticate)
    monkeypatch.setattr(qa, "select_exact_final_records_v95", selector)
    original_read = qa._read_jsonl

    def read(path: Path) -> list[dict[str, Any]]:
        events.append("labels")
        return original_read(path)

    monkeypatch.setattr(qa, "_read_jsonl", read)
    result = qa.select_final_qa_v96("synthetic.yaml")

    assert events == ["unlock", "labels", "selector"]
    assert result["authorization_checked_before_label_read"] is True
    assert result["pure_v95_selector_reused_unchanged"] is True
    assert result["v95_unlock_required"] is False
    assert final.is_file() and manifest.is_file()


def test_v96_qa_wrapper_never_calls_v95_authorized_wrapper() -> None:
    source = inspect.getsource(qa)
    assert "select_final_qa_v95" not in source
    assert "select_exact_final_records_v95" in source
    assert source.index("_authenticate_deferred_final_unlock_under_guard_v96") < source.index(
        "_read_jsonl(source)"
    )


def test_v96_qa_output_is_create_once(tmp_path: Path) -> None:
    output = tmp_path / "output.jsonl"
    assert qa._create_or_authenticate(output, b"one\n") is True
    assert qa._create_or_authenticate(output, b"one\n") is False
    with pytest.raises(FileExistsError):
        qa._create_or_authenticate(output, b"two\n")
