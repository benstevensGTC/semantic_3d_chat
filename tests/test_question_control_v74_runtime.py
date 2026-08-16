from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.question_control_runtime import (
    QuestionControlledChatRuntime,
    _load_control_head,
    block_question_control_training_artifacts,
)
from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.scene_encoder.question_control_v74 import (
    DenseFullSceneContinuousControlV74,
)
from semantic_3d_chat.training.question_control_v74_checkpoint import (
    V74_RUNTIME_ARCHITECTURE,
    load_unsealed_v74_checkpoint_for_training_gate,
    save_v74_control_checkpoint,
    v74_state_sha256,
)


def _control() -> DenseFullSceneContinuousControlV74:
    torch.manual_seed(740074)
    return DenseFullSceneContinuousControlV74(
        8,
        torch.eye(4, 8),
        environment_latents=4,
        query_count=2,
        model_dimension=4,
        uniform_floor_mass=0.05,
        maximum_control_rms=0.25,
    ).eval()


def _save(
    path: Path,
    *,
    sealed: bool,
) -> tuple[DenseFullSceneContinuousControlV74, Path]:
    control = _control()
    save_v74_control_checkpoint(
        path,
        control=control,
        base_checkpoint_sha256="1" * 64,
        base_runtime_config_sha256="2" * 64,
        expected_training_fit_state_sha256=v74_state_sha256(control),
        saved_runtime_training_gate_passed=sealed,
        saved_runtime_training_gate_attestation_sha256=("3" * 64 if sealed else None),
    )
    return control, path


def test_v74_checkpoint_is_exactly_two_sanitized_files_and_roundtrips(
    tmp_path: Path,
) -> None:
    expected, checkpoint = _save(tmp_path / "staged", sealed=False)

    assert {path.name for path in checkpoint.iterdir()} == {
        "control.safetensors",
        "runtime_metadata.json",
    }
    metadata = json.loads((checkpoint / "runtime_metadata.json").read_text())
    assert metadata["architecture"] == V74_RUNTIME_ARCHITECTURE
    assert metadata["environmental_text_inputs"] == []
    assert metadata["training_answers_runtime_loaded"] is False
    assert metadata["answer_text_runtime_loaded"] is False
    assert metadata["answer_class_codebook_runtime_loaded"] is False
    assert metadata["teacher_cache_runtime_loaded"] is False
    assert metadata["oracle_runtime_loaded"] is False
    assert metadata["question_or_answer_text_serialized"] is False

    loaded = load_unsealed_v74_checkpoint_for_training_gate(
        checkpoint, hidden_size=8
    )
    assert v74_state_sha256(loaded) == v74_state_sha256(expected)


def test_v74_public_runtime_requires_seal_and_loads_only_allowlisted_files(
    tmp_path: Path,
) -> None:
    expected, sealed = _save(tmp_path / "sealed", sealed=True)
    _unused, staged = _save(tmp_path / "staged", sealed=False)
    audit = FileAccessAudit()

    with audit:
        loaded, metadata = _load_control_head(
            sealed,
            hidden_size=8,
            device=torch.device("cpu"),
            audit=audit,
        )

    assert type(loaded) is DenseFullSceneContinuousControlV74
    assert v74_state_sha256(loaded) == v74_state_sha256(expected)
    assert metadata["saved_runtime_training_gate_passed"] is True
    assert set(audit.unique_paths) == {
        str((sealed / "control.safetensors").resolve()),
        str((sealed / "runtime_metadata.json").resolve()),
    }
    with pytest.raises(ValueError, match="runtime contract mismatch"):
        _load_control_head(
            staged,
            hidden_size=8,
            device=torch.device("cpu"),
        )


def test_v74_high_level_runtime_rejects_checkpoint_inside_training_root_without_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_runtime_config(
        "configs/runtime/gemma4_v56_question_control.yaml"
    )
    training_root = block_question_control_training_artifacts(None, config)
    base_called = False

    def refuse_base_load(*_args: object, **_kwargs: object) -> object:
        nonlocal base_called
        base_called = True
        raise AssertionError("unsafe checkpoint path must fail before base load")

    monkeypatch.setattr(
        "semantic_3d_chat.chat.question_control_runtime.StaticChatRuntime.load",
        refuse_base_load,
    )
    unsafe = training_root / "v74_runtime_candidate"

    with pytest.raises(ValueError, match="physically separate"):
        QuestionControlledChatRuntime.load(
            config,
            "scene_000001",
            base_checkpoint="data_gemma4/checkpoints/base",
            control_checkpoint=unsafe,
            audit=None,
        )
    assert base_called is False


def test_v74_runtime_rejects_any_answer_label_or_codebook_metadata(
    tmp_path: Path,
) -> None:
    _control_value, checkpoint = _save(tmp_path / "sealed", sealed=True)
    metadata_path = checkpoint / "runtime_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["answer_labels"] = ["forbidden"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata fields changed"):
        _load_control_head(
            checkpoint,
            hidden_size=8,
            device=torch.device("cpu"),
        )


def test_v74_runtime_caches_scene_kv_before_questions_and_injects_only_continuous_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    raw_questions: list[str] = []
    control = _control()
    original_encode_scene = control.encode_scene

    def observed_encode_scene(
        scene_prefix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        events.append("encode_scene")
        assert not torch.is_grad_enabled()
        return original_encode_scene(scene_prefix)

    monkeypatch.setattr(control, "encode_scene", observed_encode_scene)

    class Tokenizer:
        def __call__(
            self,
            text: str,
            *,
            add_special_tokens: bool,
            return_tensors: str,
        ) -> dict[str, torch.Tensor]:
            events.append("tokenize_question")
            raw_questions.append(text)
            token = 4 if "first" in text else 5
            return {"input_ids": torch.tensor([[token, 6]])}

        def apply_chat_template(
            self, _messages: object, **_kwargs: object
        ) -> torch.Tensor:
            return torch.tensor([[2, 3, 4]])

        def decode(self, *_args: object, **_kwargs: object) -> str:
            return "left"

    class Backend:
        def __init__(self) -> None:
            self.controls: list[torch.Tensor] = []
            self.prefixes: list[torch.Tensor] = []

        def prepare(
            self,
            scene_prefix: torch.Tensor,
            _prompt_ids: torch.Tensor,
            **kwargs: object,
        ) -> object:
            continuous = kwargs.get("control_tokens")
            assert isinstance(continuous, torch.Tensor)
            self.controls.append(continuous.detach().clone())
            self.prefixes.append(scene_prefix.detach().clone())
            return object()

        def generate(self, *_args: object, **_kwargs: object) -> torch.Tensor:
            return torch.tensor([[7]])

    scene_prefix = torch.randn(1, 6, 8)
    scene_hash = prefix_sha256(scene_prefix)
    embedding = torch.nn.Embedding(16, 8)
    backend = Backend()
    base = SimpleNamespace(
        language=SimpleNamespace(
            tokenizer=Tokenizer(),
            model=SimpleNamespace(get_input_embeddings=lambda: embedding),
            device=torch.device("cpu"),
            prefix_backend=backend,
        ),
        scene_prefix=scene_prefix,
        scene_prefix_hash=scene_hash,
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
        current_prefix_hash=lambda: scene_hash,
        startup_summary=lambda: {"prefix_shape": [1, 6, 8], "device": "cpu"},
        _eos_token_ids=lambda: 1,
        _predict_grounding=lambda _value: ((1.0, 2.0, 3.0), 0.75, 0.25),
    )
    runtime = QuestionControlledChatRuntime(
        base,
        control,
        {"architecture": V74_RUNTIME_ARCHITECTURE, "schema_version": 74},
    )

    assert events == ["encode_scene"]
    assert runtime._scene_control_key is not None
    assert runtime._scene_control_value is not None
    assert runtime.scene_control_signature_hash is not None
    key_before = runtime._scene_control_key.clone()
    value_before = runtime._scene_control_value.clone()

    def refuse_scene_reencoding(
        _scene_prefix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("V74 scene K/V was recomputed after question arrival")

    monkeypatch.setattr(control, "encode_scene", refuse_scene_reencoding)
    first = runtime.answer("first question")
    second = runtime.answer("second question")

    assert first.prefix_hash == second.prefix_hash == scene_hash
    assert events.count("encode_scene") == 1
    assert raw_questions == [
        "first question",
        "first question",
        "second question",
        "second question",
    ]
    assert torch.equal(runtime._scene_control_key, key_before)
    assert torch.equal(runtime._scene_control_value, value_before)
    assert len(backend.controls) == 2
    assert all(value.shape == (1, 2, 8) for value in backend.controls)
    assert not torch.equal(backend.controls[0], backend.controls[1])
    assert all(torch.equal(value, scene_prefix) for value in backend.prefixes)
    assert runtime.last_control_audit is not None
    assert runtime.last_control_audit["prequestion_scene_key_value_cache"] is True
    assert runtime.last_control_audit["answer_class_codebook_runtime_loaded"] is False
    assert runtime.last_control_audit["answer_text_runtime_loaded"] is False
    assert runtime.last_control_audit["question_dependent_scene_retrieval"] is False
    assert runtime.startup_summary()["prequestion_scene_key_value_cache"] is True


def test_v74_runtime_detects_cached_kv_mutation() -> None:
    scene_prefix = torch.randn(1, 6, 8)
    base = SimpleNamespace(
        scene_prefix=scene_prefix,
        scene_prefix_hash=prefix_sha256(scene_prefix),
        assert_prefix_unchanged=lambda: None,
    )
    runtime = QuestionControlledChatRuntime(
        base,
        _control(),
        {"architecture": V74_RUNTIME_ARCHITECTURE, "schema_version": 74},
    )
    assert runtime._scene_control_key is not None
    runtime._scene_control_key[..., 0].add_(1.0)

    with pytest.raises(RuntimeError, match="cached scene K/V changed"):
        runtime.assert_prefix_unchanged()
