from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

import semantic_3d_chat.chat.fixed_prefix_ple_reader_cli as reader_cli
import semantic_3d_chat.chat.fixed_prefix_ple_reader_runtime as reader_runtime
from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.runtime import ChatAnswer
from semantic_3d_chat.language.lora import LoRALinear, tensor_state_sha256
from semantic_3d_chat.language.prefix_injection import prefix_sha256


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reader_state() -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(54)
    return {
        "adapters.0.lora_a": torch.randn(
            reader_runtime.PLE_READER_RANK,
            reader_runtime.PLE_READER_PROJECTION_IN_FEATURES,
            generator=generator,
        ),
        "adapters.0.lora_b": torch.randn(
            reader_runtime.PLE_READER_PROJECTION_OUT_FEATURES,
            reader_runtime.PLE_READER_RANK,
            generator=generator,
        ),
    }


def _reader_metadata(weights: Path, state: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact": reader_runtime.PLE_READER_ARTIFACT,
        "base_checkpoint_sha256": reader_runtime._V54_BASE_CHECKPOINT_SHA256,
        "base_runtime_config_effective_sha256": (
            reader_runtime._V54_RUNTIME_CONFIG_EFFECTIVE_SHA256
        ),
        "model_id": reader_runtime.PLE_READER_MODEL_ID,
        "model_revision": reader_runtime.PLE_READER_MODEL_REVISION,
        "fixed_prefix_tokens": reader_runtime.PLE_READER_PREFIX_TOKENS,
        "scene_latents": reader_runtime.PLE_READER_SCENE_LATENTS,
        "scene_hidden_dimension": reader_runtime.PLE_READER_HIDDEN_DIMENSION,
        "prefix_computed_before_question": True,
        "question_dependent_scene_retrieval": False,
        "environmental_text_inputs": [],
        "oracle_runtime_access": False,
        "target_module": reader_runtime.PLE_READER_TARGET_MODULE,
        "rank": reader_runtime.PLE_READER_RANK,
        "alpha": reader_runtime.PLE_READER_ALPHA,
        "dropout": reader_runtime.PLE_READER_DROPOUT,
        "trainable_parameter_count": reader_runtime.PLE_READER_PARAMETER_COUNT,
        "adapter_state_sha256": tensor_state_sha256(state),
        "adapter_file_sha256": _sha256(weights),
        "selection_summary_sha256": "6" * 64,
        "preregistration_sha256": reader_runtime._V4_PREREGISTRATION_SHA256,
    }


def _reader_checkpoint(path: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    path.mkdir(parents=True)
    state = _reader_state()
    weights = path / "adapter.safetensors"
    save_file(state, str(weights))
    metadata = _reader_metadata(weights, state)
    (path / "runtime_metadata.json").write_text(
        json.dumps(metadata, sort_keys=True), encoding="utf-8"
    )
    return path, state


def _base_checkpoint(
    path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    path.mkdir(parents=True)
    adapter = path / "adapter.safetensors"
    runtime_metadata = path / "runtime_metadata.json"
    training_metadata = path / "metadata.json"
    adapter.write_bytes(b"sanitized-v54-adapter")
    runtime_metadata.write_text('{"sanitized":true}\n', encoding="utf-8")
    training_metadata.write_text('{"offline":true}\n', encoding="utf-8")
    monkeypatch.setattr(reader_runtime, "_V54_ADAPTER_SHA256", _sha256(adapter))
    monkeypatch.setattr(
        reader_runtime,
        "_V54_RUNTIME_METADATA_SHA256",
        _sha256(runtime_metadata),
    )
    return path, training_metadata


class _ProjectionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.per_layer_model_projection = nn.Linear(
            reader_runtime.PLE_READER_PROJECTION_IN_FEATURES,
            reader_runtime.PLE_READER_PROJECTION_OUT_FEATURES,
            bias=False,
        )


class _FakeInstallation:
    parameter_count = reader_runtime.PLE_READER_PARAMETER_COUNT
    training = False

    def __init__(self, digest: str) -> None:
        self.digest = digest

    def state_sha256(self) -> str:
        return self.digest


class _FakeBase:
    def __init__(self) -> None:
        self.config = {"language": {"backend": "gemma4"}}
        self.scene_id = "scene_000001"
        self.scene_prefix = torch.zeros(
            1,
            reader_runtime.PLE_READER_PREFIX_TOKENS,
            reader_runtime.PLE_READER_HIDDEN_DIMENSION,
            dtype=torch.bfloat16,
        )
        self.scene_prefix_hash = prefix_sha256(self.scene_prefix)
        self.checkpoint_path = Path("/tmp/base-v54")
        self.language = SimpleNamespace(model=nn.Identity())
        self._questions: list[str] = []

    @property
    def questions_answered(self) -> int:
        return len(self._questions)

    def current_prefix_hash(self) -> str:
        return prefix_sha256(self.scene_prefix)

    def assert_prefix_unchanged(self) -> None:
        assert self.current_prefix_hash() == self.scene_prefix_hash

    def startup_summary(self) -> dict[str, Any]:
        return {
            "phase": "scene_ready",
            "scene_id": self.scene_id,
            "prefix_hash": self.scene_prefix_hash,
            "prefix_shape": list(self.scene_prefix.shape),
            "environmental_text_inputs": [],
        }

    def answer(self, question: str) -> ChatAnswer:
        self._questions.append(question)
        return ChatAnswer(
            question=question,
            answer="embedded answer",
            grounding_xyz_m=(0.0, 0.0, 0.0),
            grounding_confidence=0.5,
            grounding_support_distance_m=0.1,
            prefix_hash=self.scene_prefix_hash,
            generated_tokens=2,
            elapsed_seconds=0.01,
        )


def test_reader_checkpoint_loads_exact_numeric_adapter_and_freezes_it(
    tmp_path: Path,
) -> None:
    checkpoint, state = _reader_checkpoint(tmp_path / "reader")
    model = _ProjectionModel().requires_grad_(False)
    audit = FileAccessAudit()

    with audit:
        installation, metadata, root = reader_runtime.load_ple_reader_adapter(
            model, checkpoint, audit=audit
        )

    assert root == checkpoint.resolve()
    assert metadata["environmental_text_inputs"] == []
    assert metadata["question_dependent_scene_retrieval"] is False
    assert installation.parameter_count == reader_runtime.PLE_READER_PARAMETER_COUNT
    assert installation.state_sha256() == tensor_state_sha256(state)
    assert installation.training is False
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert isinstance(
        model.get_submodule(reader_runtime.PLE_READER_TARGET_MODULE), LoRALinear
    )
    assert set(audit.unique_paths) >= {
        str((checkpoint / "adapter.safetensors").resolve()),
        str((checkpoint / "runtime_metadata.json").resolve()),
    }


def test_reader_checkpoint_rejects_extra_files_and_tampering(tmp_path: Path) -> None:
    checkpoint, _ = _reader_checkpoint(tmp_path / "reader")
    (checkpoint / "training_answers.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="only sanitized runtime files"):
        reader_runtime.validate_ple_reader_checkpoint(checkpoint)

    (checkpoint / "training_answers.json").unlink()
    weights = checkpoint / "adapter.safetensors"
    weights.write_bytes(weights.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="file digest changed"):
        reader_runtime.validate_ple_reader_checkpoint(checkpoint)


def test_reader_checkpoint_rejects_environmental_metadata_or_location(
    tmp_path: Path,
) -> None:
    checkpoint, _ = _reader_checkpoint(tmp_path / "reader")
    metadata_path = checkpoint / "runtime_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["object_labels"] = ["forbidden"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata fields changed"):
        reader_runtime.validate_ple_reader_checkpoint(checkpoint)

    oracle_checkpoint, _ = _reader_checkpoint(tmp_path / "oracle" / "reader")
    with pytest.raises(ValueError, match="physically separate"):
        reader_runtime.validate_ple_reader_checkpoint(oracle_checkpoint)


def test_v54_authentication_never_opens_adjacent_training_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, training_metadata = _base_checkpoint(tmp_path / "base", monkeypatch)
    audit = FileAccessAudit(block_forbidden=True)

    with audit:
        result = reader_runtime.validate_v54_checkpoint(checkpoint, audit=audit)
        with pytest.raises(PermissionError, match="Blocked forbidden runtime file"):
            training_metadata.read_text(encoding="utf-8")

    assert result.root == checkpoint.resolve()
    assert str(training_metadata.resolve()) in audit.forbidden_accesses()
    assert str(training_metadata.resolve()) not in {
        str((checkpoint / "adapter.safetensors").resolve()),
        str((checkpoint / "runtime_metadata.json").resolve()),
    }


def test_wrapper_reuses_one_prefix_for_every_question() -> None:
    base = _FakeBase()
    digest = "7" * 64
    metadata = {
        "adapter_state_sha256": digest,
        "artifact": "reader",
        "target_module": "ple",
        "rank": 4,
    }
    runtime = reader_runtime.FixedPrefixPLEReaderChatRuntime(
        base,  # type: ignore[arg-type]
        reader_installation=_FakeInstallation(digest),  # type: ignore[arg-type]
        reader_metadata=metadata,
        reader_checkpoint_path=Path("/tmp/reader"),
    )
    initial_hash = runtime.scene_prefix_hash

    first = runtime.answer("Question one?")
    second = runtime.answer("A different question?")
    summary = runtime.startup_summary()

    assert {first.prefix_hash, second.prefix_hash, runtime.current_prefix_hash()} == {
        initial_hash
    }
    assert runtime.questions_answered == 2
    assert summary["strict_fixed_environment_embedding_input"] is True
    assert summary["question_dependent_scene_retrieval"] is False
    assert summary["environmental_text_inputs"] == []
    assert summary["reader_parameter_count"] == reader_runtime.PLE_READER_PARAMETER_COUNT


def test_runtime_load_installs_reader_after_base_prefix_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _FakeBase()
    base.checkpoint_path = Path("/tmp/base").resolve()
    reader_root = Path("/tmp/reader").resolve()
    digest = "8" * 64
    events: list[str] = []
    reader = reader_runtime.ValidatedPLEReaderCheckpoint(
        reader_root,
        reader_root / "adapter.safetensors",
        reader_root / "runtime_metadata.json",
        {
            "base_checkpoint_sha256": reader_runtime._V54_BASE_CHECKPOINT_SHA256,
            "base_runtime_config_effective_sha256": "c" * 64,
            "adapter_state_sha256": digest,
            "artifact": reader_runtime.PLE_READER_ARTIFACT,
            "target_module": reader_runtime.PLE_READER_TARGET_MODULE,
        },
    )
    base_contract = reader_runtime.ValidatedV54Checkpoint(
        base.checkpoint_path, "a" * 64, "b" * 64
    )
    monkeypatch.setattr(
        reader_runtime, "validate_v54_runtime_config", lambda _: "c" * 64
    )
    monkeypatch.setattr(
        reader_runtime,
        "validate_ple_reader_checkpoint",
        lambda *_, **__: reader,
    )
    monkeypatch.setattr(
        reader_runtime, "validate_v54_checkpoint", lambda *_, **__: base_contract
    )

    def load_base(*_: object, **__: object) -> _FakeBase:
        events.append("prefix")
        assert base.questions_answered == 0
        return base

    def install(*_: object, **__: object) -> _FakeInstallation:
        events.append("reader")
        assert base.questions_answered == 0
        assert base.current_prefix_hash() == base.scene_prefix_hash
        return _FakeInstallation(digest)

    monkeypatch.setattr(reader_runtime.StaticChatRuntime, "load", load_base)
    monkeypatch.setattr(reader_runtime, "_install_validated_reader", install)

    loaded = reader_runtime.FixedPrefixPLEReaderChatRuntime.load(
        {},
        "scene_000001",
        base_checkpoint=base.checkpoint_path,
        reader_checkpoint=reader_root,
    )

    assert events == ["prefix", "reader"]
    assert loaded.questions_answered == 0
    assert loaded.scene_prefix_hash == base.scene_prefix_hash


def test_cli_check_authenticates_without_loading_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audit_path = tmp_path / "audit.json"
    monkeypatch.setattr(
        reader_cli,
        "_load_config",
        lambda *_: {"paths": {"reports_root": str(tmp_path / "reports")}},
    )
    monkeypatch.setattr(reader_cli, "_extend_forbidden_roots", lambda *_: None)
    monkeypatch.setattr(
        reader_cli,
        "_authenticate_inputs",
        lambda *_: {
            "phase": "fixed_prefix_ple_reader_preflight",
            "passed": True,
            "environmental_text_inputs": [],
        },
    )
    monkeypatch.setattr(
        reader_cli,
        "_load_runtime",
        lambda *_: pytest.fail("check mode must not load Gemma runtime"),
    )

    result = reader_cli.main(
        [
            "--check",
            "--config",
            str(tmp_path / "config.yaml"),
            "--base-checkpoint",
            str(tmp_path / "base"),
            "--reader-checkpoint",
            str(tmp_path / "reader"),
            "--audit-log",
            str(audit_path),
        ]
    )

    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert result == 0
    assert [row["phase"] for row in output] == [
        "fixed_prefix_ple_reader_preflight",
        "fixed_prefix_ple_reader_audit_complete",
    ]
    assert output[-1]["check_only"] is True
    assert json.loads(audit_path.read_text(encoding="utf-8"))["passed"] is True


def test_cli_finite_questions_emit_one_environment_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base = _FakeBase()
    digest = "9" * 64
    runtime = reader_runtime.FixedPrefixPLEReaderChatRuntime(
        base,  # type: ignore[arg-type]
        reader_installation=_FakeInstallation(digest),  # type: ignore[arg-type]
        reader_metadata={
            "adapter_state_sha256": digest,
            "artifact": "reader",
            "target_module": "ple",
            "rank": 4,
        },
        reader_checkpoint_path=tmp_path / "reader",
    )
    audit_path = tmp_path / "audit.json"
    chat_path = tmp_path / "chat.jsonl"
    monkeypatch.setattr(reader_cli, "_load_config", lambda *_: {})
    monkeypatch.setattr(reader_cli, "_extend_forbidden_roots", lambda *_: None)
    monkeypatch.setattr(
        reader_cli,
        "_authenticate_inputs",
        lambda *_: {"phase": "fixed_prefix_ple_reader_preflight", "passed": True},
    )
    monkeypatch.setattr(reader_cli, "_load_runtime", lambda *_: runtime)

    result = reader_cli.main(
        [
            "--config",
            str(tmp_path / "config.yaml"),
            "--base-checkpoint",
            str(tmp_path / "base"),
            "--reader-checkpoint",
            str(tmp_path / "reader"),
            "--question",
            "First?",
            "--question",
            "Second?",
            "--audit-log",
            str(audit_path),
            "--chat-log",
            str(chat_path),
            "--replace-chat-log",
        ]
    )

    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    answers = [row for row in output if row["phase"] == "answer"]
    transcripts = [json.loads(line) for line in chat_path.read_text().splitlines()]
    assert result == 0
    assert len(answers) == len(transcripts) == 2
    assert {row["environment_conditioned_input_sha256"] for row in answers} == {
        runtime.scene_prefix_hash
    }
    assert output[-1]["prefix_invariant"] is True
    assert output[-1]["questions_answered"] == 2
