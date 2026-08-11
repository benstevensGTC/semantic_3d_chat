from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import semantic_3d_chat.chat.runtime as runtime_module
from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.runtime import StaticChatRuntime, validate_checkpoint_contract
from semantic_3d_chat.config import config_hash
from semantic_3d_chat.language.local_lm import LocalLanguageModel
from semantic_3d_chat.language.prefix_injection import ContinuousPrefixComposer
from semantic_3d_chat.scene_encoder.dense_alignment import (
    DenseAlignmentResidual,
    construct_dense_alignment,
    dense_alignment_settings,
)
from semantic_3d_chat.scene_encoder.map_io import MapTensorData
from semantic_3d_chat.scene_encoder.projector import SceneTokenizerOutput
from semantic_3d_chat.training.checkpointing import (
    runtime_checkpoint_metadata,
    save_adapter_checkpoint,
)
from semantic_3d_chat.training.losses import QuestionGroundingHead


class _Tokenizer:
    eos_token_id = 2

    def __call__(self, text: str, **_kwargs):
        count = max(1, len(text.split()))
        return {"input_ids": torch.arange(3, 3 + count).reshape(1, -1)}

    def apply_chat_template(self, messages, **_kwargs):
        count = 4 + sum(len(message["content"].split()) for message in messages)
        return torch.arange(3, 3 + count).reshape(1, -1)

    def decode(self, _ids, **_kwargs):
        return "numeric scene answer"


class _LanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(128, 8)
        self.config = SimpleNamespace(hidden_size=8)
        self.generation_config = SimpleNamespace(eos_token_id=2)

    def get_input_embeddings(self):
        return self.embedding


class _RecordingSceneModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(5, 8, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.linspace(-0.2, 0.2, 40).reshape(8, 5))
        self.calls = 0
        self.seen_semantic: torch.Tensor | None = None

    def forward(self, semantic: torch.Tensor, *_args) -> SceneTokenizerOutput:
        self.calls += 1
        self.seen_semantic = semantic.detach().clone()
        pooled = self.projection(semantic.float()).mean(dim=0)
        scene_tokens = pooled.reshape(1, 1, 8).repeat(1, 4, 1)
        return SceneTokenizerOutput(
            scene_tokens=scene_tokens,
            native_latents=torch.zeros(1, 4, 6, device=semantic.device),
            block_tokens=torch.zeros(1, 6, device=semantic.device),
            audit={
                "processed_voxels": torch.tensor(semantic.shape[0], device=semantic.device),
                "voxel_counts": torch.tensor([semantic.shape[0]], device=semantic.device),
            },
        )


def _config(*, enabled: bool) -> dict:
    config = {
        "paths": {"data_root": "data", "reports_root": "reports"},
        "scene": {"room_size_m": [6.0, 5.0, 3.0]},
        "scene_encoder": {
            "input_voxel_size_m": 0.15,
            "block_size_m": 0.25,
            "tokens_per_block": 2,
            "model_dim": 6,
            "global_latents": 4,
            "heads": 2,
            "global_layers": 1,
            "fourier_bands": 2,
        },
        "language": {
            "model_id": "local/tiny",
            "revision": "revision-1",
            "dtype": "float32",
            "backend": "causal_lm",
            "max_question_tokens": 32,
            "max_answer_tokens": 4,
            "system_prompt": "Use only the continuous scene memory.",
        },
    }
    if enabled:
        provisional = {
            "enabled": True,
            "dense_dim": 3,
            "aligned_dim": 2,
            "rank": 2,
            "alpha": 4.0,
            "initialization_seed": 25025,
            "expected_initial_state_sha256": "0" * 64,
        }
        config["scene_encoder"]["dense_alignment"] = provisional
        initial = construct_dense_alignment(config, semantic_dim=5)
        assert initial is not None
        provisional["expected_initial_state_sha256"] = initial.state_sha256()
    return config


def _map() -> MapTensorData:
    semantic = torch.tensor(
        [
            [-1.0, -0.5, 0.25, 0.2, -0.1],
            [0.0, 0.5, 1.0, -0.2, 0.4],
            [1.0, 0.25, -0.75, 0.3, 0.1],
        ],
        dtype=torch.float32,
    )
    return MapTensorData(
        semantic=semantic,
        xyz=torch.tensor([[-1.0, 0.0, 0.5], [0.0, 0.0, 0.5], [1.0, 0.0, 0.5]]),
        rgb=torch.zeros(3, 3),
        normal=torch.zeros(3, 3),
        confidence=torch.ones(3),
        observation_count=torch.ones(3),
        room_min=torch.tensor([-3.0, -2.5, 0.0]),
        room_max=torch.tensor([3.0, 2.5, 3.0]),
        source_voxel_count=3,
        input_voxel_size_m=0.15,
    )


def _language() -> LocalLanguageModel:
    return LocalLanguageModel(
        model=_LanguageModel().eval().requires_grad_(False),
        tokenizer=_Tokenizer(),
        device=torch.device("cpu"),
        backend_name="causal_lm",
    )


def _metadata(config: dict, dense_aligner: DenseAlignmentResidual | None) -> dict:
    metadata = {
        "schema_version": 3,
        "semantic_dim": 5,
        "language_hidden_dim": 8,
        "language_model_id": "local/tiny",
        "language_revision": "revision-1",
        "language_backend": "causal_lm",
        "scene_latents": 4,
        "scene_model_dim": 6,
        "input_voxel_size_m": 0.15,
        "config_hash": config_hash(config),
    }
    if dense_aligner is not None:
        settings = dense_alignment_settings(config)
        metadata.update(
            {
                "dense_alignment": settings.contract(),
                "dense_alignment_parameter_count": dense_aligner.parameter_count,
                "dense_alignment_initial_state_sha256": (settings.expected_initial_state_sha256),
                "dense_alignment_state_sha256": dense_aligner.state_sha256(),
                "all_voxels_transformed": True,
                "question_dependent_scene_processing": False,
            }
        )
    return metadata


def _checkpoint(
    tmp_path: Path,
    config: dict,
    *,
    include_dense_tensors: bool,
    include_dense_metadata: bool,
) -> tuple[Path, DenseAlignmentResidual | None]:
    dense_aligner = construct_dense_alignment(config, semantic_dim=5)
    if dense_aligner is None and include_dense_tensors:
        dense_aligner = DenseAlignmentResidual(
            semantic_dim=5,
            dense_dim=3,
            aligned_dim=2,
            rank=2,
            alpha=4.0,
            initialization_seed=25025,
        )
    if dense_aligner is not None:
        with torch.no_grad():
            dense_aligner.alignment_b.copy_(
                torch.tensor([[0.2, -0.1], [-0.3, 0.4]], dtype=torch.float32)
            )

    scene_model = _RecordingSceneModel()
    composer = ContinuousPrefixComposer(8)
    grounding = QuestionGroundingHead(6, 8, 4, 6)
    modules: dict[str, nn.Module] = {
        "scene_model": scene_model,
        "composer": composer,
        "grounding": grounding,
    }
    if include_dense_tensors:
        assert dense_aligner is not None
        modules["dense_aligner"] = dense_aligner
    metadata = _metadata(
        config,
        dense_aligner if include_dense_metadata else None,
    )
    return save_adapter_checkpoint(tmp_path / "checkpoint", modules, metadata), dense_aligner


def _install_runtime_mocks(monkeypatch: pytest.MonkeyPatch, map_data: MapTensorData) -> None:
    monkeypatch.setattr(runtime_module, "load_map_tensors", lambda *_args, **_kwargs: map_data)
    monkeypatch.setattr(
        runtime_module,
        "load_local_language_model",
        lambda *_args, **_kwargs: _language(),
    )
    monkeypatch.setattr(
        runtime_module,
        "construct_scene_tokenizer",
        lambda *_args, **_kwargs: _RecordingSceneModel(),
    )


def test_dense_runtime_loads_strict_state_and_transforms_every_voxel_before_tokenizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(enabled=True)
    checkpoint, trained_dense = _checkpoint(
        tmp_path,
        config,
        include_dense_tensors=True,
        include_dense_metadata=True,
    )
    assert trained_dense is not None
    map_data = _map()
    raw_semantic = map_data.semantic.clone()
    expected_semantic = trained_dense(raw_semantic).detach()
    _install_runtime_mocks(monkeypatch, map_data)

    def fake_generation(_model, _embeddings, _mask, _maximum, _eos):
        return torch.tensor([[7, 2]])

    runtime = StaticChatRuntime.load(
        config,
        "scene_000001",
        checkpoint,
        generation_function=fake_generation,
    )

    assert runtime.dense_aligner is not None
    assert runtime.dense_aligner.state_sha256() == trained_dense.state_sha256()
    assert runtime.scene_model.calls == 1
    assert torch.equal(runtime.map_data.semantic, raw_semantic)
    assert torch.allclose(runtime.scene_model.seen_semantic, expected_semantic)
    assert not torch.equal(runtime.scene_model.seen_semantic, raw_semantic)
    summary = runtime.startup_summary()
    assert summary["dense_alignment_parameter_count"] == trained_dense.parameter_count
    assert summary["dense_alignment_state_sha256"] == trained_dense.state_sha256()
    assert summary["dense_alignment_transformed_voxels"] == map_data.voxel_count
    assert summary["all_voxels_transformed"] is True
    initial_prefix_hash = runtime.scene_prefix_hash
    first = runtime.answer("first numeric query")
    second = runtime.answer("second numeric query")
    assert first.prefix_hash == second.prefix_hash == initial_prefix_hash
    assert runtime.scene_model.calls == 1
    runtime.assert_prefix_unchanged()


def test_chat_never_opens_training_metadata_with_qa_or_category_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(enabled=True)
    checkpoint, _ = _checkpoint(
        tmp_path,
        config,
        include_dense_tensors=True,
        include_dense_metadata=True,
    )
    training_path = checkpoint / "metadata.json"
    training_metadata = json.loads(training_path.read_text(encoding="utf-8"))
    training_metadata["history"] = [
        {
            "question_id": "q_000031",
            "question_text": "SENTINEL: where is the bowl?",
            "category_prototype": "cabinet",
        }
    ]
    training_path.write_text(json.dumps(training_metadata), encoding="utf-8")
    runtime_path = checkpoint / "runtime_metadata.json"
    assert "SENTINEL" not in runtime_path.read_text(encoding="utf-8")
    _install_runtime_mocks(monkeypatch, _map())
    audit = FileAccessAudit()

    with audit:
        runtime = StaticChatRuntime.load(
            config,
            "scene_000001",
            checkpoint,
            audit=audit,
            generation_function=lambda *_args: torch.tensor([[7, 2]]),
        )

    assert str(runtime_path.resolve()) in audit.unique_paths
    assert str(training_path.resolve()) not in audit.unique_paths
    assert "history" not in runtime.checkpoint_metadata
    assert runtime.answer("numeric query").prefix_hash == runtime.scene_prefix_hash


def test_chat_rejects_runtime_metadata_symlink_to_training_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(enabled=True)
    checkpoint, _ = _checkpoint(
        tmp_path,
        config,
        include_dense_tensors=True,
        include_dense_metadata=True,
    )
    runtime_path = checkpoint / "runtime_metadata.json"
    training_path = checkpoint / "metadata.json"
    runtime_path.unlink()
    runtime_path.symlink_to(training_path.name)
    _install_runtime_mocks(monkeypatch, _map())
    audit = FileAccessAudit()

    with audit, pytest.raises(ValueError, match="must not be a symbolic link"):
        StaticChatRuntime.load(config, "scene_000001", checkpoint, audit=audit)

    assert str(training_path.resolve()) not in audit.unique_paths


def test_dense_runtime_rejects_trained_state_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(enabled=True)
    checkpoint, _ = _checkpoint(
        tmp_path,
        config,
        include_dense_tensors=True,
        include_dense_metadata=True,
    )
    metadata_path = checkpoint / "runtime_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["dense_alignment_state_sha256"] = "f" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _install_runtime_mocks(monkeypatch, _map())

    with pytest.raises(ValueError, match="Dense-alignment state mismatch or tamper"):
        StaticChatRuntime.load(config, "scene_000001", checkpoint)


def test_enabled_runtime_rejects_missing_dense_aligner_tensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(enabled=True)
    checkpoint, _ = _checkpoint(
        tmp_path,
        config,
        include_dense_tensors=False,
        include_dense_metadata=True,
    )
    _install_runtime_mocks(monkeypatch, _map())

    with pytest.raises(RuntimeError, match=r"Missing key\(s\) in state_dict"):
        StaticChatRuntime.load(config, "scene_000001", checkpoint)


def test_disabled_contract_is_backward_compatible_and_rejects_stray_metadata() -> None:
    config = _config(enabled=False)
    explicit_disabled = deepcopy(config)
    explicit_disabled["scene_encoder"]["dense_alignment"] = {"enabled": False}
    metadata = _metadata(config, None)

    baseline = validate_checkpoint_contract(
        metadata,
        config,
        semantic_dim=5,
        language_hidden_dim=8,
    )
    explicit = validate_checkpoint_contract(
        metadata,
        explicit_disabled,
        semantic_dim=5,
        language_hidden_dim=8,
    )
    assert baseline == []
    assert len(explicit) == 1
    assert "config hash differs" in explicit[0].lower()
    training_metadata = {
        **metadata,
        "dense_alignment": {"schema_version": 1, "enabled": False},
        "dense_alignment_parameter_count": 0,
        "dense_alignment_state_sha256": None,
        "all_voxels_transformed": False,
    }
    sanitized = runtime_checkpoint_metadata(training_metadata)
    assert not (set(sanitized) & runtime_module._DENSE_ALIGNMENT_METADATA_KEYS)

    metadata["dense_alignment"] = {"schema_version": 1, "enabled": False}
    with pytest.raises(ValueError, match="dense-alignment metadata"):
        validate_checkpoint_contract(
            metadata,
            config,
            semantic_dim=5,
            language_hidden_dim=8,
        )

    for training_only_key in (
        "dense_alignment_calibration",
        "dense_alignment_optimizer",
        "dense_alignment_zero_output_equivalence",
    ):
        stray = _metadata(config, None)
        stray[training_only_key] = {}
        with pytest.raises(ValueError, match="dense-alignment metadata"):
            validate_checkpoint_contract(
                stray,
                config,
                semantic_dim=5,
                language_hidden_dim=8,
            )


def test_disabled_runtime_rejects_unconsumed_dense_aligner_tensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(enabled=False)
    checkpoint, _ = _checkpoint(
        tmp_path,
        config,
        include_dense_tensors=True,
        include_dense_metadata=False,
    )
    _install_runtime_mocks(monkeypatch, _map())

    with pytest.raises(RuntimeError, match="unconsumed tensor keys"):
        StaticChatRuntime.load(config, "scene_000001", checkpoint)
