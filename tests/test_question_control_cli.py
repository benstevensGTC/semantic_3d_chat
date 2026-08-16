from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from semantic_3d_chat.chat.question_control_cli import main


def test_control_cli_requires_explicit_checkpoint_paths() -> None:
    with pytest.raises(SystemExit) as error:
        main(["--config", "missing", "--scene", "scene_000001"])
    assert error.value.code == 2


def test_control_cli_builds_prefix_before_questions_and_audits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text("runtime: true\n", encoding="utf-8")
    base = tmp_path / "base"
    control = tmp_path / "control"
    base.mkdir()
    control.mkdir()
    audit_log = tmp_path / "audit.json"
    chat_log = tmp_path / "chat.jsonl"

    config = {
        "paths": {
            "data_root": str(tmp_path / "data"),
            "reports_root": str(tmp_path / "reports"),
        }
    }

    class Answer:
        def to_dict(self) -> dict[str, object]:
            return {
                "question": "Is it left?",
                "answer": "left",
                "prefix_hash": "a" * 64,
            }

    class Runtime:
        scene_prefix_hash = "a" * 64
        scene_control_signature_hash = "d" * 64
        questions_answered = 0
        control_metadata: ClassVar[dict[str, object]] = {
            "architecture": "always_on_teacher_basis_full_scene_control_v7",
            "schema_version": 7,
        }
        base = SimpleNamespace(startup_summary=lambda: {"device": "cpu"})

        @classmethod
        def load(cls, *_: object, **__: object) -> Runtime:
            return cls()

        def answer(self, _: str) -> Answer:
            return Answer()

        def assert_prefix_unchanged(self) -> None:
            return None

        def current_prefix_hash(self) -> str:
            return self.scene_prefix_hash

        def startup_summary(self) -> dict[str, object]:
            return {"device": "cpu", "prefix_shape": [1, 258, 1536]}

    monkeypatch.setattr(
        "semantic_3d_chat.chat.runtime_config.load_runtime_config",
        lambda *_args, **_kwargs: config,
    )
    monkeypatch.setattr(
        "semantic_3d_chat.chat.question_control_runtime.QuestionControlledChatRuntime",
        Runtime,
    )
    monkeypatch.setattr(
        "semantic_3d_chat.evaluation.prediction_artifacts.checkpoint_fingerprint",
        lambda _path: ("b" * 64, []),
    )
    monkeypatch.setattr(
        "semantic_3d_chat.evaluation.predict_question_control._control_checkpoint_sha256",
        lambda _path: "c" * 64,
    )

    result = main(
        [
            "--config",
            str(config_path),
            "--scene",
            "scene_000001",
            "--base-checkpoint",
            str(base),
            "--control-checkpoint",
            str(control),
            "--question",
            "Is it left?",
            "--audit-log",
            str(audit_log),
            "--chat-log",
            str(chat_log),
        ]
    )

    assert result == 0
    output = capsys.readouterr().out.splitlines()
    startup = json.loads(output[0])
    assert startup["scene_prefix_computed_before_question"] is True
    assert startup["scene_control_signature_sha256"] == "d" * 64
    assert startup["control_schema_version"] == 7
    assert startup["questions_answered"] == 0
    assert startup["environmental_text_inputs"] == []
    assert startup["question_dependent_scene_retrieval"] is False
    answer = json.loads(chat_log.read_text(encoding="utf-8"))
    assert answer["answer"] == "left"
    audit = json.loads(audit_log.read_text(encoding="utf-8"))
    assert audit["forbidden_accesses"] == []
