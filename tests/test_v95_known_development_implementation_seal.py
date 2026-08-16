from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import (
    authenticate_v95_known_development as authenticate_stage,
)
from semantic_3d_chat.evaluation import nll_v95_known_development as nll_stage
from semantic_3d_chat.evaluation import predict_v95_known_development as predict_stage
from semantic_3d_chat.evaluation import score_v95_known_development as score_stage
from semantic_3d_chat.evaluation import seal_v95_known_development as seal_stage
from semantic_3d_chat.evaluation import v95_known_development_implementation as guard


def _fake_sources(root: Path) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for index, name in enumerate(
        (
            "common",
            "predict",
            "authenticate",
            "score",
            "nll",
            "seal",
            "implementation_guard",
        )
    ):
        path = root / "src" / f"{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# synthetic source {index}\n", encoding="utf-8")
        sources[name] = path
    return sources


def test_v95_implementation_inventory_covers_all_evaluation_stages() -> None:
    payload = guard.build_evaluation_implementation_seal_v95()

    assert payload["artifact"] == guard.ARTIFACT
    assert payload["source_count"] == 7
    assert set(payload["sources"]) == {
        "common",
        "predict",
        "authenticate",
        "score",
        "nll",
        "seal",
        "implementation_guard",
    }
    assert all(len(row["sha256"]) == 64 for row in payload["sources"].values())
    assert payload["known_development_outputs_present_before_seal"] == []
    assert payload["questions_opened"] is False
    assert payload["labels_opened"] is False
    assert payload["model_loaded"] is False
    assert not guard.IMPLEMENTATION_SEAL.exists()


def test_v95_implementation_seal_is_create_once_and_detects_source_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(guard, "PROJECT_ROOT", tmp_path)
    sources = _fake_sources(tmp_path)
    output = tmp_path / "seal.json"
    sealed = guard.seal_evaluation_implementation_v95(
        seal_path=output,
        sources=sources,
        outputs=(),
    )
    authenticated = guard.authenticate_evaluation_implementation_v95(
        seal_path=output,
        sources=sources,
    )

    assert sealed["seal_sha256"] == authenticated["seal_sha256"]
    assert authenticated["authenticated"] is True
    with pytest.raises(FileExistsError, match="create-once"):
        guard.seal_evaluation_implementation_v95(
            seal_path=output,
            sources=sources,
            outputs=(),
        )

    sources["score"].write_text("# tampered scorer\n", encoding="utf-8")
    with pytest.raises(ValueError, match="implementation seal changed"):
        guard.authenticate_evaluation_implementation_v95(
            seal_path=output,
            sources=sources,
        )


def test_v95_implementation_seal_rejects_existing_evaluation_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(guard, "PROJECT_ROOT", tmp_path)
    sources = _fake_sources(tmp_path)
    output = tmp_path / "prediction.jsonl"
    output.write_text("existing\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="before evaluation output creation"):
        guard.build_evaluation_implementation_seal_v95(
            sources=sources,
            outputs=(output,),
        )


def test_v95_exclusive_lock_is_reentrant_and_rejects_parallel_thread(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "evaluation.lock"
    error: list[BaseException] = []
    with guard.exclusive_evaluation_lock_v95(lock):
        with guard.exclusive_evaluation_lock_v95(lock):
            pass

        def contend() -> None:
            try:
                with guard.exclusive_evaluation_lock_v95(lock):
                    pass
            except RuntimeError as caught:
                error.append(caught)

        thread = threading.Thread(target=contend)
        thread.start()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert len(error) == 1
    assert isinstance(error[0], RuntimeError)
    assert "already active" in str(error[0])


def test_hardened_stage_authenticates_inside_lock_before_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    @contextmanager
    def fake_lock() -> object:
        events.append("lock-enter")
        try:
            yield
        finally:
            events.append("lock-exit")

    monkeypatch.setattr(guard, "exclusive_evaluation_lock_v95", fake_lock)
    monkeypatch.setattr(
        guard,
        "authenticate_evaluation_implementation_v95",
        lambda: events.append("authenticate"),
    )

    @guard.hardened_evaluation_stage_v95
    def stage(value: int) -> int:
        events.append("body")
        return value + 1

    assert stage(4) == 5
    assert events == ["lock-enter", "authenticate", "body", "lock-exit"]


def test_every_public_evaluation_stage_is_guarded() -> None:
    functions = (
        predict_stage.predict_known_development_v95,
        authenticate_stage.authenticate_v95,
        score_stage.score_known_development_v95,
        nll_stage.measure_known_development_nll_v95,
        seal_stage.seal_known_development_v95,
        seal_stage.authenticate_final_evidence_v95,
    )
    assert all(hasattr(function, "__wrapped__") for function in functions)


def test_seal_artifact_schema_contains_no_questions_labels_or_outputs() -> None:
    payload = guard.build_evaluation_implementation_seal_v95()
    encoded = json.dumps(payload, sort_keys=True)

    assert '"questions_opened": false' in encoded
    assert '"labels_opened": false' in encoded
    assert '"model_loaded": false' in encoded
    assert "question_text" not in encoded
    assert "reference_answer" not in encoded
    assert "prediction_rows" not in encoded
