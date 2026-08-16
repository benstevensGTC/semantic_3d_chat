from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.question_control_runtime import (
    QuestionControlledChatRuntime,
    _load_control_head,
    block_question_control_training_artifacts,
    sanitize_generated_answer,
)
from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.scene_encoder.question_control import FullSceneQuestionControl


def _checkpoint(path: Path, *, extra: dict[str, object] | None = None) -> Path:
    path.mkdir()
    torch.manual_seed(19)
    module = FullSceneQuestionControl(
        16,
        attention_dim=8,
        control_tokens=2,
        uniform_floor=0.05,
        output_scale=0.25,
    )
    weights = path / "control.safetensors"
    save_file(module.state_dict(), str(weights))
    metadata: dict[str, object] = {
        "schema_version": 1,
        "architecture": "full_scene_question_control_v1",
        "hidden_size": 16,
        "attention_dim": 8,
        "control_tokens": 2,
        "uniform_floor": 0.05,
        "output_scale": 0.25,
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "base_checkpoint_sha256": "1" * 64,
        "base_runtime_config_sha256": "2" * 64,
        "question_dependent_scene_retrieval": False,
        "complete_scene_prefix_required": True,
        "environmental_text_inputs": [],
    }
    metadata.update(extra or {})
    (path / "runtime_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return path


def test_control_checkpoint_loads_only_sanitized_runtime_contract(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path / "control")

    module, metadata = _load_control_head(
        source,
        hidden_size=16,
        device=torch.device("cpu"),
    )

    assert module.parameter_count > 0
    assert metadata["environmental_text_inputs"] == []
    assert metadata["question_dependent_scene_retrieval"] is False
    result = module(torch.randn(1, 258, 16), torch.randn(1, 5, 16))
    assert result.shape == (1, 2, 16)


def test_control_checkpoint_rejects_training_or_label_metadata(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path / "control", extra={"answer_labels": ["left"]})

    with pytest.raises(ValueError, match="metadata fields changed"):
        _load_control_head(source, hidden_size=16, device=torch.device("cpu"))


def test_control_checkpoint_detects_weight_tampering(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path / "control")
    weights = source / "control.safetensors"
    weights.write_bytes(weights.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="weights changed"):
        _load_control_head(source, hidden_size=16, device=torch.device("cpu"))


def test_control_checkpoint_is_runtime_minimal_and_audited(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path / "control")
    audit = FileAccessAudit()

    with audit:
        _load_control_head(
            source,
            hidden_size=16,
            device=torch.device("cpu"),
            audit=audit,
        )

    assert str((source / "control.safetensors").resolve()) in audit.unique_paths
    assert str((source / "runtime_metadata.json").resolve()) in audit.unique_paths

    (source / "training_answers.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="only sanitized runtime files"):
        _load_control_head(source, hidden_size=16, device=torch.device("cpu"))


def test_control_checkpoint_rejects_symlinked_roots(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path / "control")
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)

    with pytest.raises(ValueError, match="must not contain symlinks"):
        _load_control_head(alias, hidden_size=16, device=torch.device("cpu"))


def test_training_artifact_block_is_exact_and_keeps_source_modules_readable(
    tmp_path: Path,
) -> None:
    derived_root = tmp_path / "data_gemma4"
    checkpoints = derived_root / "checkpoints"
    teacher_file = derived_root / "training" / "numeric_teacher" / "teachers.safetensors"
    source_module = tmp_path / "src" / "semantic_3d_chat" / "training" / "checkpointing.py"
    checkpoints.mkdir(parents=True)
    teacher_file.parent.mkdir(parents=True)
    source_module.parent.mkdir(parents=True)
    teacher_file.write_bytes(b"numeric teacher")
    source_module.write_text("SOURCE_MARKER = True\n", encoding="utf-8")
    config = {
        "paths": {
            "data_root": str(tmp_path / "data"),
            "checkpoints_root": str(checkpoints),
        }
    }
    audit = FileAccessAudit(block_forbidden=True)

    blocked_root = block_question_control_training_artifacts(audit, config)

    assert blocked_root == teacher_file.parents[1]
    assert "training" not in audit.forbidden_component_names
    with audit:
        assert "SOURCE_MARKER" in source_module.read_text(encoding="utf-8")
        with pytest.raises(PermissionError, match="Blocked forbidden runtime file"):
            teacher_file.read_bytes()
    assert str(source_module.resolve()) not in audit.forbidden_accesses()
    assert str(teacher_file.resolve()) in audit.forbidden_accesses()


@pytest.mark.parametrize("forbidden", ["oracle", "qa"])
def test_control_checkpoint_rejects_oracle_or_qa_directories(
    tmp_path: Path, forbidden: str
) -> None:
    parent = tmp_path / forbidden
    parent.mkdir()
    source = _checkpoint(parent / "control")

    with pytest.raises(ValueError, match="physically separate"):
        _load_control_head(source, hidden_size=16, device=torch.device("cpu"))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", True, "contract mismatch"),
        ("hidden_size", 16.0, "positive integer"),
        ("attention_dim", True, "positive integer"),
        ("uniform_floor", float("inf"), "finite positive"),
        ("base_runtime_config_sha256", "not-a-hash", "digest is invalid"),
    ],
)
def test_control_checkpoint_rejects_ambiguous_metadata_values(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    source = _checkpoint(tmp_path / "control", extra={field: value})

    with pytest.raises(ValueError, match=message):
        _load_control_head(source, hidden_size=16, device=torch.device("cpu"))


def test_question_control_runtime_load_binds_base_config_and_forwards_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = FileAccessAudit()
    base = SimpleNamespace(
        language=SimpleNamespace(hidden_size=16, device=torch.device("cpu")),
        scene_prefix_hash="a" * 64,
    )
    observed: dict[str, object] = {}

    def load_base(*args: object, **kwargs: object) -> object:
        observed["base_audit"] = kwargs.get("audit")
        return base

    def load_control(*args: object, **kwargs: object) -> tuple[object, dict[str, object]]:
        observed["control_audit"] = kwargs.get("audit")
        return object(), {
            "base_checkpoint_sha256": "b" * 64,
            "base_runtime_config_sha256": "c" * 64,
        }

    monkeypatch.setattr(
        "semantic_3d_chat.chat.question_control_runtime.StaticChatRuntime.load",
        load_base,
    )
    monkeypatch.setattr(
        "semantic_3d_chat.chat.question_control_runtime._load_control_head",
        load_control,
    )
    monkeypatch.setattr(
        "semantic_3d_chat.chat.question_control_runtime.checkpoint_fingerprint",
        lambda _: ("b" * 64, []),
    )
    monkeypatch.setattr(
        "semantic_3d_chat.chat.question_control_runtime.effective_runtime_config_sha256",
        lambda _: "c" * 64,
    )

    config = load_runtime_config("configs/runtime/gemma4_v56_question_control.yaml")
    with audit:
        runtime = QuestionControlledChatRuntime.load(
            config,
            "scene_000001",
            base_checkpoint="base",
            control_checkpoint="control",
            audit=audit,
        )

    assert runtime.scene_prefix_hash == "a" * 64
    assert observed == {"base_audit": audit, "control_audit": audit}
    assert str(Path(config["_config_path"]).resolve()) in audit.unique_paths


def test_question_control_runtime_rejects_non_runtime_config_before_base_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def load_base(*_: object, **__: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("base runtime must not load")

    monkeypatch.setattr(
        "semantic_3d_chat.chat.question_control_runtime.StaticChatRuntime.load",
        load_base,
    )
    with pytest.raises(ValueError, match="standalone validated runtime config"):
        QuestionControlledChatRuntime.load(
            {"batch": {"scenes": {"scene_000001": {"category": "chair"}}}},
            "scene_000001",
            base_checkpoint="base",
            control_checkpoint="control",
        )
    assert called is False


def test_question_control_runtime_answer_injects_continuous_tokens_only() -> None:
    hidden_size = 8

    class Tokenizer:
        def apply_chat_template(self, *_: object, **__: object) -> torch.Tensor:
            return torch.tensor([[2, 3, 4]])

        def __call__(self, *_: object, **__: object) -> dict[str, torch.Tensor]:
            return {"input_ids": torch.tensor([[4, 5]])}

        def decode(self, *_: object, **__: object) -> str:
            return "left"

    class Backend:
        def __init__(self) -> None:
            self.control_tokens: torch.Tensor | None = None

        def prepare(
            self,
            scene_prefix: torch.Tensor,
            prompt_ids: torch.Tensor,
            **kwargs: object,
        ) -> object:
            assert scene_prefix.shape == (1, 6, hidden_size)
            assert prompt_ids.shape == (1, 3)
            control = kwargs["control_tokens"]
            assert isinstance(control, torch.Tensor)
            self.control_tokens = control
            return object()

        def generate(self, *_: object, **__: object) -> torch.Tensor:
            return torch.tensor([[7]])

    embedding = torch.nn.Embedding(16, hidden_size)
    backend = Backend()
    base = SimpleNamespace(
        language=SimpleNamespace(
            tokenizer=Tokenizer(),
            model=SimpleNamespace(get_input_embeddings=lambda: embedding),
            device=torch.device("cpu"),
            prefix_backend=backend,
        ),
        scene_prefix=torch.randn(1, 6, hidden_size),
        scene_prefix_hash="d" * 64,
        config={
            "language": {
                "system_prompt": "Use only continuous memory.",
                "max_question_tokens": 8,
                "max_answer_tokens": 4,
                "scene_prefix_after_bos": False,
                "scene_boundary_mode": "learned",
            }
        },
        assert_prefix_unchanged=lambda: None,
        _eos_token_ids=lambda: 1,
        _predict_grounding=lambda _: ((1.0, 2.0, 3.0), 0.75, 0.25),
    )
    control = FullSceneQuestionControl(
        hidden_size,
        attention_dim=4,
        control_tokens=2,
        uniform_floor=0.1,
    ).eval()
    runtime = QuestionControlledChatRuntime(base, control, {})

    result = runtime.answer("Where is it?")

    assert result.answer == "left"
    assert result.prefix_hash == "d" * 64
    assert result.grounding_xyz_m == (1.0, 2.0, 3.0)
    assert backend.control_tokens is not None
    assert backend.control_tokens.shape == (1, 2, hidden_size)
    assert torch.isfinite(backend.control_tokens).all()

    base.config["language"]["max_question_tokens"] = 1
    with pytest.raises(ValueError, match="1-token control limit"):
        runtime.answer("Where is it?")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" red ", "red"),
        ("2.5 m", "2.5 m"),
        ("red, blue, and green", "red, blue, and green"),
        ("}|=", "unknown"),
        ('\">= \"', "unknown"),
        ("زي", "unknown"),
        ("한 3D 장면", "unknown"),
        ("", "unknown"),
    ],
)
def test_generated_answer_sanitizer_is_vocabulary_free_and_fails_closed(
    raw: str,
    expected: str,
) -> None:
    assert sanitize_generated_answer(raw) == expected


def test_generated_answer_sanitizer_rejects_non_text() -> None:
    with pytest.raises(TypeError, match="must be text"):
        sanitize_generated_answer(3)  # type: ignore[arg-type]
