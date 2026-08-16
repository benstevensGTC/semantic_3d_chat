from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_3d_chat.chat.strict_prefix_cli import (
    _human_output_requested,
    _parser,
    main,
)


def test_strict_prefix_cli_requires_explicit_inputs() -> None:
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2


def test_output_mode_is_human_for_interactive_and_json_for_finite_automation() -> None:
    required = [
        "--config",
        "runtime.yaml",
        "--scene",
        "scene_000001",
        "--checkpoint",
        "checkpoint",
    ]

    assert _human_output_requested(_parser().parse_args(required)) is True
    assert (
        _human_output_requested(
            _parser().parse_args([*required, "--question", "Is there a chair?"])
        )
        is False
    )
    assert (
        _human_output_requested(
            _parser().parse_args(
                [*required, "--human", "--question", "Is there a chair?"]
            )
        )
        is True
    )
    assert _human_output_requested(_parser().parse_args([*required, "--json"])) is False


def test_human_mode_is_concise_but_keeps_full_json_transcript(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text("runtime: true\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    chat_log = tmp_path / "chat.jsonl"
    audit_log = tmp_path / "audit.json"
    prefix_hash = "c" * 64
    config = {
        "paths": {
            "data_root": str(tmp_path / "data"),
            "reports_root": str(tmp_path / "reports"),
        }
    }

    class Answer:
        prefix_hash = "c" * 64

        def to_dict(self) -> dict[str, object]:
            return {
                "question": "Where is it?",
                "answer": "near the table",
                "grounding_xyz_m": [1.25, -0.5, 0.75],
                "grounding_confidence": 0.625,
                "grounding_support_distance_m": 0.1,
                "prefix_hash": self.prefix_hash,
                "generated_tokens": 4,
                "elapsed_seconds": 0.125,
            }

    class Runtime:
        scene_prefix_hash = prefix_hash
        questions_answered = 0

        @classmethod
        def load(cls, *_args: object, **_kwargs: object) -> Runtime:
            return cls()

        def current_prefix_hash(self) -> str:
            return self.scene_prefix_hash

        def startup_summary(self) -> dict[str, object]:
            return {
                "device": "cpu",
                "language_backend": "local-test",
                "prefix_shape": [1, 258, 1536],
            }

        def answer(self, _question: str) -> Answer:
            self.questions_answered += 1
            return Answer()

        def assert_prefix_unchanged(self) -> None:
            assert self.scene_prefix_hash == prefix_hash

    monkeypatch.setattr(
        "semantic_3d_chat.chat.runtime_config.load_runtime_config",
        lambda *_args, **_kwargs: config,
    )
    monkeypatch.setattr("semantic_3d_chat.chat.runtime.StaticChatRuntime", Runtime)

    result = main(
        [
            "--config",
            str(config_path),
            "--scene",
            "scene_000001",
            "--checkpoint",
            str(checkpoint),
            "--question",
            "Where is it?",
            "--human",
            "--audit-log",
            str(audit_log),
            "--chat-log",
            str(chat_log),
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "Semantic 3D Chat ready" in output
    assert "Continuous memory: 258 tokens x 1536 dimensions" in output
    assert f"Fixed prefix: {prefix_hash} (built before questions)" in output
    assert "You> Where is it?" in output
    assert "Assistant> near the table" in output
    assert "Grounding: (+1.25, -0.50, +0.75) m, confidence 0.62" in output
    assert "Generation: 4 tokens in 0.12 s" in output
    assert "Verification: PASS - fixed prefix unchanged; 0 forbidden file reads" in output
    assert not any(line.startswith("{") for line in output.splitlines())

    rows = [json.loads(line) for line in chat_log.read_text().splitlines()]
    assert rows == [
        {
            "answer": "near the table",
            "elapsed_seconds": 0.125,
            "environment_conditioned_input_sha256": prefix_hash,
            "generated_tokens": 4,
            "grounding_confidence": 0.625,
            "grounding_support_distance_m": 0.1,
            "grounding_xyz_m": [1.25, -0.5, 0.75],
            "phase": "answer",
            "prefix_hash": prefix_hash,
            "question": "Where is it?",
            "scene_id": "scene_000001",
            "strict_fixed_environment_embedding_input": True,
        }
    ]
    audit = json.loads(audit_log.read_text())
    assert audit["forbidden_accesses"] == []


def test_strict_demo_launcher_defaults_to_human_with_json_override() -> None:
    launcher = Path("scripts/run_strict_fixed_prefix_demo.sh").read_text(
        encoding="utf-8"
    )

    assert 'STRICT_OUTPUT="human"' in launcher
    assert 'STRICT_ARGS+=(--human)' in launcher
    assert '--json) STRICT_OUTPUT="json"' in launcher


def test_strict_prefix_cli_reuses_one_environment_hash_and_audits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text("runtime: true\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    training_metadata = checkpoint / "metadata.json"
    training_metadata.write_text(
        '{"counterfactual_family":"book_support","answer_text":"left"}\n',
        encoding="utf-8",
    )
    audit_log = tmp_path / "audit.json"
    chat_log = tmp_path / "chat.jsonl"
    prefix_hash = "a" * 64
    config = {
        "paths": {
            "data_root": str(tmp_path / "data"),
            "reports_root": str(tmp_path / "reports"),
        }
    }

    class Answer:
        def __init__(self, question: str) -> None:
            self.prefix_hash = prefix_hash
            self.question = question

        def to_dict(self) -> dict[str, object]:
            return {
                "question": self.question,
                "answer": "yes",
                "prefix_hash": self.prefix_hash,
            }

    class Runtime:
        scene_id = "scene_000001"
        scene_prefix_hash = prefix_hash
        questions_answered = 0

        @classmethod
        def load(cls, *_args: object, **_kwargs: object) -> Runtime:
            return cls()

        def current_prefix_hash(self) -> str:
            return self.scene_prefix_hash

        def startup_summary(self) -> dict[str, object]:
            return {"device": "cpu", "prefix_shape": [1, 258, 1536]}

        def answer(self, question: str) -> Answer:
            self.questions_answered += 1
            return Answer(question)

        def assert_prefix_unchanged(self) -> None:
            assert self.scene_prefix_hash == prefix_hash

    monkeypatch.setattr(
        "semantic_3d_chat.chat.runtime_config.load_runtime_config",
        lambda *_args, **_kwargs: config,
    )
    monkeypatch.setattr(
        "semantic_3d_chat.chat.runtime.StaticChatRuntime",
        Runtime,
    )

    result = main(
        [
            "--config",
            str(config_path),
            "--scene",
            "scene_000001",
            "--checkpoint",
            str(checkpoint),
            "--question",
            "First?",
            "--question",
            "Second?",
            "--audit-log",
            str(audit_log),
            "--chat-log",
            str(chat_log),
        ]
    )

    assert result == 0
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert output[0]["scene_prefix_computed_before_question"] is True
    assert output[0]["strict_fixed_environment_embedding_input"] is True
    assert output[0]["question_conditioned_scene_readout_tokens"] is False
    assert output[-1]["prefix_invariant"] is True
    rows = [json.loads(line) for line in chat_log.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert {row["environment_conditioned_input_sha256"] for row in rows} == {
        prefix_hash
    }
    assert all(row["strict_fixed_environment_embedding_input"] is True for row in rows)
    audit = json.loads(audit_log.read_text(encoding="utf-8"))
    assert audit["forbidden_accesses"] == []
    assert str(training_metadata.resolve()) in audit["forbidden_roots"]
    assert str(training_metadata.resolve()) not in audit["loaded_files"]


def test_strict_prefix_cli_can_replace_finite_transcript(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text("runtime: true\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    chat_log = tmp_path / "chat.jsonl"
    chat_log.write_text('{"stale":true}\n', encoding="utf-8")
    prefix_hash = "b" * 64
    config = {
        "paths": {
            "data_root": str(tmp_path / "data"),
            "reports_root": str(tmp_path / "reports"),
        }
    }

    class Answer:
        def __init__(self) -> None:
            self.prefix_hash = "b" * 64

        def to_dict(self) -> dict[str, object]:
            return {"answer": "yes", "prefix_hash": self.prefix_hash}

    class Runtime:
        scene_prefix_hash = prefix_hash
        questions_answered = 0

        @classmethod
        def load(cls, *_args: object, **_kwargs: object) -> Runtime:
            return cls()

        def current_prefix_hash(self) -> str:
            return self.scene_prefix_hash

        def startup_summary(self) -> dict[str, object]:
            return {"device": "cpu", "prefix_shape": [1, 258, 1536]}

        def answer(self, _question: str) -> Answer:
            self.questions_answered += 1
            return Answer()

        def assert_prefix_unchanged(self) -> None:
            assert self.scene_prefix_hash == prefix_hash

    monkeypatch.setattr(
        "semantic_3d_chat.chat.runtime_config.load_runtime_config",
        lambda *_args, **_kwargs: config,
    )
    monkeypatch.setattr("semantic_3d_chat.chat.runtime.StaticChatRuntime", Runtime)

    result = main(
        [
            "--config",
            str(config_path),
            "--scene",
            "scene_000001",
            "--checkpoint",
            str(checkpoint),
            "--question",
            "Fresh?",
            "--audit-log",
            str(tmp_path / "audit.json"),
            "--chat-log",
            str(chat_log),
            "--replace-chat-log",
        ]
    )

    assert result == 0
    capsys.readouterr()
    rows = [json.loads(line) for line in chat_log.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["environment_conditioned_input_sha256"] == prefix_hash
    assert "stale" not in rows[0]


def test_replace_chat_log_requires_finite_questions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(
        [
            "--config",
            str(tmp_path / "runtime.yaml"),
            "--scene",
            "scene_000001",
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--replace-chat-log",
        ]
    )
    assert result == 2
    assert "requires at least one finite" in capsys.readouterr().err
