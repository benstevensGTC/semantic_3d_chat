from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import (
    authenticate_v96_known_development as authenticate_stage,
)
from semantic_3d_chat.evaluation import nll_v96_known_development as nll_stage
from semantic_3d_chat.evaluation import predict_v96_known_development as predict_stage
from semantic_3d_chat.evaluation import score_v96_known_development as score_stage
from semantic_3d_chat.evaluation import seal_v96_known_development as seal_stage
from semantic_3d_chat.evaluation import v96_known_development_implementation as guard


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


def _contract_inventory() -> dict[str, dict[str, str]]:
    return {
        "config": {"path": "config.yaml", "sha256": "1" * 64},
        "preflight": {"path": "preflight.py", "sha256": "2" * 64},
        "trainer": {"path": "trainer.py", "sha256": "3" * 64},
    }


def test_v96_contract_inventory_propagates_unsealed_config_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_draft(*_args: object, **_kwargs: object) -> object:
        raise ValueError("V96 config status is not authorized")

    monkeypatch.setattr(guard, "load_config_v96", reject_draft)
    with pytest.raises(ValueError, match="status is not authorized"):
        guard.contract_source_inventory_v96()


def test_v96_implementation_seal_binds_all_evaluator_and_contract_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(guard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        guard, "contract_source_inventory_v96", lambda _path=guard.CONFIG: _contract_inventory()
    )
    payload = guard.build_evaluation_implementation_seal_v96(
        sources=_fake_sources(tmp_path), outputs=()
    )

    assert payload["artifact"] == guard.ARTIFACT
    assert payload["source_count"] == 7
    assert payload["contract_source_count"] == 3
    assert set(payload["sources"]) == {
        "common",
        "predict",
        "authenticate",
        "score",
        "nll",
        "seal",
        "implementation_guard",
    }
    assert set(payload["contract_sources"]) == {"config", "preflight", "trainer"}
    assert payload["known_development_outputs_present_before_seal"] == []
    assert payload["questions_opened"] is False
    assert payload["labels_opened"] is False
    assert payload["model_loaded"] is False


def test_v96_implementation_seal_is_create_once_and_detects_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(guard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        guard, "contract_source_inventory_v96", lambda _path=guard.CONFIG: _contract_inventory()
    )
    sources = _fake_sources(tmp_path)
    output = tmp_path / "seal.json"
    sealed = guard.seal_evaluation_implementation_v96(
        seal_path=output, sources=sources, outputs=()
    )
    authenticated = guard.authenticate_evaluation_implementation_v96(
        seal_path=output, sources=sources
    )
    assert sealed["seal_sha256"] == authenticated["seal_sha256"]
    assert authenticated["authenticated"] is True

    with pytest.raises(FileExistsError, match="create-once"):
        guard.seal_evaluation_implementation_v96(
            seal_path=output, sources=sources, outputs=()
        )
    sources["score"].write_text("# tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="implementation seal changed"):
        guard.authenticate_evaluation_implementation_v96(
            seal_path=output, sources=sources
        )


def test_v96_implementation_seal_rejects_any_existing_evaluation_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(guard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        guard, "contract_source_inventory_v96", lambda _path=guard.CONFIG: _contract_inventory()
    )
    output = tmp_path / "prediction.jsonl"
    output.write_text("existing\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="before evaluation output creation"):
        guard.build_evaluation_implementation_seal_v96(
            sources=_fake_sources(tmp_path), outputs=(output,)
        )


def test_v96_exclusive_lock_is_reentrant_and_rejects_parallel_thread(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "evaluation.lock"
    errors: list[BaseException] = []
    with guard.exclusive_evaluation_lock_v96(lock):
        with guard.exclusive_evaluation_lock_v96(lock):
            pass

        def contend() -> None:
            try:
                with guard.exclusive_evaluation_lock_v96(lock):
                    pass
            except RuntimeError as caught:
                errors.append(caught)

        thread = threading.Thread(target=contend)
        thread.start()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert len(errors) == 1
    assert "already active" in str(errors[0])


def test_hardened_v96_stage_authenticates_inside_lock_before_body(
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

    monkeypatch.setattr(guard, "exclusive_evaluation_lock_v96", fake_lock)
    monkeypatch.setattr(
        guard,
        "authenticate_evaluation_implementation_v96",
        lambda: events.append("authenticate"),
    )

    @guard.hardened_evaluation_stage_v96
    def stage(value: int) -> int:
        events.append("body")
        return value + 1

    assert stage(4) == 5
    assert events == ["lock-enter", "authenticate", "body", "lock-exit"]


def test_every_public_v96_evaluation_stage_is_guarded() -> None:
    functions = (
        predict_stage.predict_known_development_v96,
        authenticate_stage.authenticate_v96,
        score_stage.score_known_development_v96,
        nll_stage.measure_known_development_nll_v96,
        seal_stage.seal_known_development_v96,
        seal_stage.authenticate_final_evidence_v96,
    )
    assert all(hasattr(function, "__wrapped__") for function in functions)


def test_v96_seal_schema_contains_no_questions_labels_or_behavior_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(guard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        guard, "contract_source_inventory_v96", lambda _path=guard.CONFIG: _contract_inventory()
    )
    payload = guard.build_evaluation_implementation_seal_v96(
        sources=_fake_sources(tmp_path), outputs=()
    )
    encoded = json.dumps(payload, sort_keys=True)
    assert '"questions_opened": false' in encoded
    assert '"labels_opened": false' in encoded
    assert '"model_loaded": false' in encoded
    assert "question_text" not in encoded
    assert "reference_answer" not in encoded
    assert "prediction_rows" not in encoded
