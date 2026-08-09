from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

import semantic_3d_chat.chat.runtime as runtime_module
from semantic_3d_chat.chat.runtime import (
    StaticChatRuntime,
    construct_scene_tokenizer,
    validate_checkpoint_contract,
)
from semantic_3d_chat.config import config_hash
from semantic_3d_chat.language.local_lm import LocalLanguageModel
from semantic_3d_chat.language.lora import (
    LoRALinear,
    install_lora_adapters,
    install_lora_banks,
    lora_banks_checkpoint_contract,
    lora_banks_optimizer_settings,
    lora_banks_settings,
    lora_checkpoint_contract,
    lora_optimizer_settings,
    lora_settings,
)
from semantic_3d_chat.language.prefix_injection import ContinuousPrefixComposer
from semantic_3d_chat.scene_encoder.map_io import MapTensorData
from semantic_3d_chat.scene_encoder.projector import SceneTokenizerOutput
from semantic_3d_chat.training.checkpointing import (
    module_collection_state_sha256,
    save_adapter_checkpoint,
)
from semantic_3d_chat.training.losses import QuestionGroundingHead


class TinyTokenizer:
    eos_token_id = 2

    def __call__(self, text, **_kwargs):
        count = max(1, len(str(text).split()))
        return {"input_ids": torch.arange(3, 3 + count).reshape(1, -1)}

    def apply_chat_template(self, messages, **_kwargs):
        count = 4 + sum(len(message["content"].split()) for message in messages)
        return torch.arange(3, 3 + count).reshape(1, -1)

    def decode(self, _token_ids, **_kwargs):
        return "continuous answer"


class TinyLanguageModel(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(128, hidden_size)
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.generation_config = SimpleNamespace(eos_token_id=2)

    def get_input_embeddings(self):
        return self.embedding


class CountingSceneModel(nn.Module):
    def __init__(self, semantic_dim: int, hidden_size: int, scene_dim: int, latents: int) -> None:
        super().__init__()
        self.calls = 0
        self.semantic_dim = semantic_dim
        self.hidden_size = hidden_size
        self.scene_dim = scene_dim
        self.latents = latents

    def forward(self, semantic, *_args):
        self.calls += 1
        device = semantic.device
        return SceneTokenizerOutput(
            scene_tokens=torch.linspace(
                -1.0, 1.0, self.latents * self.hidden_size, device=device
            ).reshape(1, self.latents, self.hidden_size),
            native_latents=torch.zeros(1, self.latents, self.scene_dim, device=device),
            block_tokens=torch.zeros(2, self.scene_dim, device=device),
            audit={
                "processed_voxels": torch.tensor(semantic.shape[0], device=device),
                "voxel_counts": torch.tensor([semantic.shape[0]], device=device),
            },
        )


class ZeroGrounding(nn.Module):
    def forward(self, scene_latents, question_embeddings):
        return torch.zeros(scene_latents.shape[0], 3, device=scene_latents.device)


def tiny_config() -> dict:
    return {
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
            "max_question_tokens": 96,
            "max_answer_tokens": 4,
            "system_prompt": "Use the continuous memory only.",
        },
    }


def tiny_checkpoint_metadata() -> dict:
    return {
        "schema_version": 3,
        "semantic_dim": 7,
        "language_hidden_dim": 8,
        "language_model_id": "local/tiny",
        "language_revision": "revision-1",
        "scene_latents": 4,
        "scene_model_dim": 6,
        "input_voxel_size_m": 0.15,
        "config_hash": "training-hash",
    }


def tiny_map(count: int = 5, semantic_dim: int = 7) -> MapTensorData:
    return MapTensorData(
        semantic=torch.randn(count, semantic_dim),
        xyz=torch.linspace(-1.0, 1.0, count * 3).reshape(count, 3),
        rgb=torch.zeros(count, 3),
        normal=torch.zeros(count, 3),
        confidence=torch.full((count,), 0.8),
        observation_count=torch.ones(count),
        room_min=torch.tensor([-3.0, -2.5, 0.0]),
        room_max=torch.tensor([3.0, 2.5, 3.0]),
        source_voxel_count=count,
        input_voxel_size_m=0.15,
    )


def test_runtime_builds_complete_prefix_once_before_questions() -> None:
    torch.manual_seed(8)
    config = tiny_config()
    language = LocalLanguageModel(
        model=TinyLanguageModel(hidden_size=8),
        tokenizer=TinyTokenizer(),
        device=torch.device("cpu"),
    )
    scene_model = CountingSceneModel(semantic_dim=7, hidden_size=8, scene_dim=6, latents=4)

    def fake_generation(_model, _embeddings, _mask, _maximum, _eos):
        return torch.tensor([[7, 2]])

    runtime = StaticChatRuntime(
        config=config,
        scene_id="scene_000001",
        checkpoint_path=Path("checkpoint"),
        checkpoint_metadata={},
        language=language,
        map_data=tiny_map(),
        scene_model=scene_model,
        composer=ContinuousPrefixComposer(8),
        grounding=ZeroGrounding(),
        generation_function=fake_generation,
    )
    initial_hash = runtime.scene_prefix_hash
    assert runtime.questions_answered == 0
    assert scene_model.calls == 1
    first = runtime.answer("Is there a chair?")
    second = runtime.answer("Where is the bowl?")
    assert scene_model.calls == 1
    assert runtime.questions_answered == 2
    assert first.prefix_hash == second.prefix_hash == initial_hash
    assert runtime.current_prefix_hash() == initial_hash
    assert first.answer == "continuous answer"
    assert first.grounding_xyz_m == (0.0, 0.0, 1.5)


def test_checkpoint_contract_allows_only_nonarchitectural_config_hash_drift() -> None:
    config = tiny_config()
    metadata = {
        "schema_version": 1,
        "semantic_dim": 7,
        "language_hidden_dim": 8,
        "language_model_id": "local/tiny",
        "language_revision": "revision-1",
        "scene_latents": 4,
        "scene_model_dim": 6,
        "input_voxel_size_m": 0.15,
        "config_hash": "training-hash",
    }
    warnings = validate_checkpoint_contract(metadata, config, semantic_dim=7, language_hidden_dim=8)
    assert warnings and "config hash differs" in warnings[0].lower()
    metadata["scene_latents"] = 8
    try:
        validate_checkpoint_contract(metadata, config, semantic_dim=7, language_hidden_dim=8)
    except ValueError as exc:
        assert "scene_latents" in str(exc)
    else:
        raise AssertionError("Architecture mismatch was accepted")


def test_checkpoint_contract_accepts_resumable_schema_v2() -> None:
    config = tiny_config()
    metadata = {
        "schema_version": 2,
        "semantic_dim": 7,
        "language_hidden_dim": 8,
        "language_model_id": "local/tiny",
        "language_revision": "revision-1",
        "scene_latents": 4,
        "scene_model_dim": 6,
        "input_voxel_size_m": 0.15,
        "config_hash": "training-hash",
    }
    warnings = validate_checkpoint_contract(metadata, config, semantic_dim=7, language_hidden_dim=8)
    assert warnings and "config hash differs" in warnings[0].lower()


def test_checkpoint_contract_enforces_full_lora_runtime_contract() -> None:
    config = tiny_config()
    config["language"].update(
        {
            "backend": "gemma4",
            "lora": {
                "enabled": True,
                "rank": 2,
                "alpha": 4.0,
                "dropout": 0.0,
                "target_modules": ["model.language_model.layers.1.self_attn.q_proj"],
            },
        }
    )
    config["training"] = {
        "lora_learning_rate": 1e-4,
        "lora_weight_decay": 0.0,
    }
    settings = lora_settings(config)
    optimizer_settings = lora_optimizer_settings(config, settings)
    assert optimizer_settings is not None
    metadata = {
        **tiny_checkpoint_metadata(),
        "lora": lora_checkpoint_contract(settings, optimizer_settings, 32),
    }

    validate_checkpoint_contract(
        metadata,
        config,
        semantic_dim=7,
        language_hidden_dim=8,
        lora_parameter_count=32,
    )
    with pytest.raises(ValueError, match="lora"):
        validate_checkpoint_contract(
            metadata,
            config,
            semantic_dim=7,
            language_hidden_dim=8,
            lora_parameter_count=31,
        )
    metadata["lora"] = {**metadata["lora"], "learning_rate": 2e-4}
    with pytest.raises(ValueError, match="lora"):
        validate_checkpoint_contract(
            metadata,
            config,
            semantic_dim=7,
            language_hidden_dim=8,
            lora_parameter_count=32,
        )


def test_checkpoint_contract_enforces_opt_in_bos_layout_but_accepts_legacy_false() -> None:
    config = tiny_config()
    metadata = tiny_checkpoint_metadata()

    # Missing metadata is the historical scene-before-BOS layout.
    validate_checkpoint_contract(metadata, config, semantic_dim=7, language_hidden_dim=8)

    config["language"]["scene_prefix_after_bos"] = True
    with pytest.raises(ValueError, match="scene_prefix_after_bos"):
        validate_checkpoint_contract(metadata, config, semantic_dim=7, language_hidden_dim=8)

    metadata["scene_prefix_after_bos"] = True
    validate_checkpoint_contract(metadata, config, semantic_dim=7, language_hidden_dim=8)

    config["language"]["scene_prefix_after_bos"] = False
    with pytest.raises(ValueError, match="scene_prefix_after_bos"):
        validate_checkpoint_contract(metadata, config, semantic_dim=7, language_hidden_dim=8)


def test_checkpoint_contract_requires_complete_nondefault_aligned_bypass_metadata() -> None:
    config = tiny_config()
    aligned_contract = {
        "language_aligned_tail_dim": 4,
        "native_aligned_coverage_scale": 0.75,
        "learned_scene_token_scale": 0.25,
        "learned_scene_token_rms_target": 0.9,
    }
    config["scene_encoder"].update(aligned_contract)
    metadata = tiny_checkpoint_metadata()

    with pytest.raises(ValueError, match="language_aligned_tail_dim") as exc_info:
        validate_checkpoint_contract(metadata, config, semantic_dim=7, language_hidden_dim=8)
    assert all(key in str(exc_info.value) for key in aligned_contract)

    metadata.update(aligned_contract)
    warnings = validate_checkpoint_contract(metadata, config, semantic_dim=7, language_hidden_dim=8)
    assert warnings and "config hash differs" in warnings[0].lower()


def test_checkpoint_contract_rejects_partial_default_aligned_bypass_metadata() -> None:
    config = tiny_config()
    metadata = {
        **tiny_checkpoint_metadata(),
        "language_aligned_tail_dim": 0,
    }

    with pytest.raises(ValueError, match="native_aligned_coverage_scale"):
        validate_checkpoint_contract(metadata, config, semantic_dim=7, language_hidden_dim=8)

    metadata.update(
        {
            "native_aligned_coverage_scale": 0.0,
            "learned_scene_token_scale": 1.0,
            "learned_scene_token_rms_target": None,
        }
    )
    warnings = validate_checkpoint_contract(metadata, config, semantic_dim=7, language_hidden_dim=8)
    assert warnings and "config hash differs" in warnings[0].lower()


@pytest.mark.parametrize(
    ("field", "checkpoint_value"),
    [
        ("language_aligned_tail_dim", 2),
        ("native_aligned_coverage_scale", 0.5),
        ("learned_scene_token_scale", 0.5),
        ("learned_scene_token_rms_target", None),
    ],
)
def test_checkpoint_contract_rejects_aligned_bypass_mismatch(
    field: str, checkpoint_value: float | None
) -> None:
    config = tiny_config()
    aligned_contract = {
        "language_aligned_tail_dim": 4,
        "native_aligned_coverage_scale": 0.75,
        "learned_scene_token_scale": 0.25,
        "learned_scene_token_rms_target": 0.9,
    }
    config["scene_encoder"].update(aligned_contract)
    metadata = {**tiny_checkpoint_metadata(), **aligned_contract, field: checkpoint_value}

    with pytest.raises(ValueError, match=field):
        validate_checkpoint_contract(metadata, config, semantic_dim=7, language_hidden_dim=8)


def test_construct_scene_tokenizer_wires_aligned_bypass_defaults_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict] = []
    monkeypatch.setattr(
        runtime_module,
        "SceneTokenizer",
        lambda **kwargs: captured.append(kwargs) or object(),
    )
    config = tiny_config()
    config["scene_encoder"]["architecture_version"] = "test-architecture"

    construct_scene_tokenizer(config, semantic_dim=7, language_hidden_dim=8)
    config["scene_encoder"].update(
        {
            "language_aligned_tail_dim": 4,
            "native_aligned_coverage_scale": 0.75,
            "learned_scene_token_scale": 0.25,
            "learned_scene_token_rms_target": 0.9,
        }
    )
    construct_scene_tokenizer(config, semantic_dim=7, language_hidden_dim=8)

    contract_keys = {
        "language_aligned_tail_dim",
        "native_aligned_coverage_scale",
        "learned_scene_token_scale",
        "learned_scene_token_rms_target",
    }
    assert {key: captured[0][key] for key in contract_keys} == {
        "language_aligned_tail_dim": 0,
        "native_aligned_coverage_scale": 0.0,
        "learned_scene_token_scale": 1.0,
        "learned_scene_token_rms_target": None,
    }
    assert {key: captured[1][key] for key in contract_keys} == {
        "language_aligned_tail_dim": 4,
        "native_aligned_coverage_scale": 0.75,
        "learned_scene_token_scale": 0.25,
        "learned_scene_token_rms_target": 0.9,
    }


def test_checkpoint_contract_rejects_unknown_schema() -> None:
    config = tiny_config()
    metadata = {
        "schema_version": 99,
        "semantic_dim": 7,
        "language_hidden_dim": 8,
        "language_model_id": "local/tiny",
        "language_revision": "revision-1",
        "scene_latents": 4,
        "scene_model_dim": 6,
        "input_voxel_size_m": 0.15,
        "config_hash": "training-hash",
    }
    try:
        validate_checkpoint_contract(metadata, config, semantic_dim=7, language_hidden_dim=8)
    except ValueError as exc:
        assert "Unsupported checkpoint metadata schema" in str(exc)
    else:
        raise AssertionError("Unknown checkpoint schema was accepted")


def test_checkpoint_contract_rejects_legacy_resampler_for_v2_config() -> None:
    config = tiny_config()
    config["scene_encoder"]["architecture_version"] = "spatial_coverage_resampler_v2"
    metadata = {
        "schema_version": 2,
        "semantic_dim": 7,
        "language_hidden_dim": 8,
        "language_model_id": "local/tiny",
        "language_revision": "revision-1",
        "scene_latents": 4,
        "scene_model_dim": 6,
        "input_voxel_size_m": 0.15,
        "config_hash": "legacy-hash",
    }
    try:
        validate_checkpoint_contract(metadata, config, semantic_dim=7, language_hidden_dim=8)
    except ValueError as exc:
        assert "scene_encoder_architecture_version" in str(exc)
    else:
        raise AssertionError("Legacy collapsed-resampler checkpoint was accepted by v2 runtime")


class _TinyRuntimeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)


class _TinyRuntimeDecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _TinyRuntimeAttention()


class _TinyRuntimeGemma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(128, 8)
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList(
            [_TinyRuntimeDecoderLayer(), _TinyRuntimeDecoderLayer()]
        )
        self.config = SimpleNamespace(hidden_size=8)
        self.generation_config = SimpleNamespace(eos_token_id=2)

    def get_input_embeddings(self):
        return self.embedding


def _tiny_lora_runtime_config() -> dict:
    config = tiny_config()
    config["scene_encoder"]["architecture_version"] = "signal_preserving_resampler_v3"
    config["language"].update(
        {
            "backend": "gemma4",
            "dtype": "float32",
            "lora": {
                "enabled": True,
                "rank": 2,
                "alpha": 4.0,
                "dropout": 0.0,
                "target_modules": [
                    "model.language_model.layers.1.self_attn.q_proj",
                    "model.language_model.layers.1.self_attn.o_proj",
                ],
            },
        }
    )
    config["training"] = {
        "lora_learning_rate": 1e-4,
        "lora_weight_decay": 0.0,
    }
    return config


def _tiny_lora_language() -> LocalLanguageModel:
    model = _TinyRuntimeGemma().eval().requires_grad_(False)
    return LocalLanguageModel(
        model=model,
        tokenizer=TinyTokenizer(),
        device=torch.device("cpu"),
        backend_name="gemma4",
    )


def _tiny_lora_banks_runtime_config() -> dict:
    config = tiny_config()
    config["scene_encoder"]["architecture_version"] = "signal_preserving_resampler_v3"
    config["language"].update(
        {
            "backend": "gemma4",
            "dtype": "float32",
            "lora_banks": {
                "inherited": {
                    "trainable": False,
                    "rank": 2,
                    "alpha": 4.0,
                    "dropout": 0.0,
                    "initialization_algorithm": "checkpoint_overwrite",
                    "initialization_seed": None,
                    "expected_initial_state_sha256": None,
                    "target_modules": [
                        "model.language_model.layers.0.self_attn.q_proj",
                        "model.language_model.layers.0.self_attn.o_proj",
                    ],
                },
                "extension": {
                    "trainable": True,
                    "rank": 3,
                    "alpha": 6.0,
                    "dropout": 0.0,
                    "initialization_algorithm": "cpu_kaiming_uniform_a_exact_zero_b",
                    "initialization_seed": 13008,
                    "expected_initial_state_sha256": None,
                    "target_modules": [
                        "model.language_model.layers.1.self_attn.q_proj",
                        "model.language_model.layers.1.self_attn.o_proj",
                    ],
                },
            },
        }
    )
    config["training"] = {
        "lora_learning_rate": 5e-5,
        "lora_weight_decay": 0.0,
        "freeze_scene_adapter": True,
    }
    return config


def test_static_chat_loads_exact_lora_state_and_rejects_tensor_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(8188)
    config = _tiny_lora_runtime_config()
    settings = lora_settings(config)
    optimizer_settings = lora_optimizer_settings(config, settings)
    source_language = _tiny_lora_language()
    source_lora = install_lora_adapters(source_language.model, settings)
    assert source_lora is not None and optimizer_settings is not None
    with torch.no_grad():
        source_lora.adapters[0].lora_b.fill_(0.125)
        source_lora.adapters[1].lora_b.fill_(-0.25)
    scene_model = construct_scene_tokenizer(config, semantic_dim=7, language_hidden_dim=8)
    composer = ContinuousPrefixComposer(8)
    grounding = QuestionGroundingHead(6, 8, 4, 6)
    metadata = {
        **tiny_checkpoint_metadata(),
        "config_hash": config_hash(config),
        "language_backend": "gemma4",
        "scene_encoder_architecture_version": "signal_preserving_resampler_v3",
        "lora": lora_checkpoint_contract(settings, optimizer_settings, source_lora.parameter_count),
        "lora_wrapped_modules": list(source_lora.target_names),
        "lora_trainable_parameter_counts": source_lora.parameter_counts,
        "lora_trainable_parameter_count": source_lora.parameter_count,
        "lora_state_sha256": source_lora.state_sha256(),
    }
    checkpoint = save_adapter_checkpoint(
        tmp_path / "runtime_checkpoint",
        {
            "scene_model": scene_model,
            "composer": composer,
            "grounding": grounding,
            "lora": source_lora.state_module,
        },
        metadata,
    )
    monkeypatch.setattr(runtime_module, "load_map_tensors", lambda *_args, **_kwargs: tiny_map())
    monkeypatch.setattr(
        runtime_module, "load_local_language_model", lambda *_args, **_kwargs: _tiny_lora_language()
    )

    runtime = StaticChatRuntime.load(config, "scene_000001", checkpoint)

    restored_q = runtime.language.model.model.language_model.layers[1].self_attn.q_proj
    restored_o = runtime.language.model.model.language_model.layers[1].self_attn.o_proj
    assert isinstance(restored_q, LoRALinear) and isinstance(restored_o, LoRALinear)
    assert torch.equal(restored_q.lora_b, source_lora.adapters[0].lora_b)
    assert torch.equal(restored_o.lora_b, source_lora.adapters[1].lora_b)
    assert all(not parameter.requires_grad for parameter in runtime.language.model.parameters())

    tensors = load_file(checkpoint / "adapter.safetensors")
    tensors["lora.adapters.0.lora_b"][0, 0].add_(1.0)
    save_file(tensors, checkpoint / "adapter.safetensors")
    with pytest.raises(ValueError, match="tamper"):
        StaticChatRuntime.load(config, "scene_000001", checkpoint)


def test_static_chat_roundtrips_all_named_lora_banks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _tiny_lora_banks_runtime_config()
    settings = lora_banks_settings(config)
    optimizer_settings = lora_banks_optimizer_settings(config, settings)
    source_language = _tiny_lora_language()
    source = install_lora_banks(source_language.model, settings)
    assert source is not None and optimizer_settings is not None
    with torch.no_grad():
        source.bank("inherited").installation.adapters[0].lora_b.fill_(0.125)
        source.bank("extension").installation.adapters[1].lora_b.fill_(-0.25)
    scene_model = construct_scene_tokenizer(config, semantic_dim=7, language_hidden_dim=8)
    composer = ContinuousPrefixComposer(8)
    grounding = QuestionGroundingHead(6, 8, 4, 6)
    scene_modules = {
        "scene_model": scene_model,
        "composer": composer,
        "grounding": grounding,
    }
    metadata = {
        **tiny_checkpoint_metadata(),
        "config_hash": config_hash(config),
        "language_backend": "gemma4",
        "scene_encoder_architecture_version": "signal_preserving_resampler_v3",
        "lora": lora_banks_checkpoint_contract(
            settings, optimizer_settings, source.parameter_counts
        ),
        "freeze_scene_adapter": True,
        "frozen_scene_state_sha256": module_collection_state_sha256(scene_modules),
        **source.checkpoint_metadata(),
    }
    for frozen_hash in (None, "INVALID"):
        invalid_metadata = dict(metadata)
        if frozen_hash is None:
            invalid_metadata.pop("frozen_scene_state_sha256")
        else:
            invalid_metadata["frozen_scene_state_sha256"] = frozen_hash
        with pytest.raises(ValueError, match="frozen_scene_state_sha256"):
            validate_checkpoint_contract(
                invalid_metadata,
                config,
                semantic_dim=7,
                language_hidden_dim=8,
                lora_parameter_count=source.parameter_count,
                lora_parameter_counts=source.parameter_counts,
            )
    checkpoint_modules = {
        **scene_modules,
        **source.state_modules(),
    }
    checkpoint = save_adapter_checkpoint(tmp_path / "runtime_banks", checkpoint_modules, metadata)
    monkeypatch.setattr(runtime_module, "load_map_tensors", lambda *_args, **_kwargs: tiny_map())
    monkeypatch.setattr(
        runtime_module, "load_local_language_model", lambda *_args, **_kwargs: _tiny_lora_language()
    )

    runtime = StaticChatRuntime.load(config, "scene_000001", checkpoint)
    layers = runtime.language.model.model.language_model.layers
    assert torch.equal(
        layers[0].self_attn.q_proj.lora_b,
        source.bank("inherited").installation.adapters[0].lora_b,
    )
    assert torch.equal(
        layers[1].self_attn.o_proj.lora_b,
        source.bank("extension").installation.adapters[1].lora_b,
    )
    assert all(not parameter.requires_grad for parameter in runtime.language.model.parameters())

    tensors = load_file(checkpoint / "adapter.safetensors")
    scene_key = next(
        key for key in tensors if key.startswith("scene_model.") and tensors[key].numel()
    )
    original_scene_tensor = tensors[scene_key].clone()
    tensors[scene_key].view(-1)[0].add_(1.0)
    save_file(tensors, checkpoint / "adapter.safetensors")
    with pytest.raises(ValueError, match="Frozen scene checkpoint state mismatch"):
        StaticChatRuntime.load(config, "scene_000001", checkpoint)

    tensors[scene_key] = original_scene_tensor
    tensors["lora_banks.extension.adapters.0.lora_b"][0, 0].add_(1.0)
    save_file(tensors, checkpoint / "adapter.safetensors")
    with pytest.raises(ValueError, match="tamper"):
        StaticChatRuntime.load(config, "scene_000001", checkpoint)


def test_chat_runtime_source_does_not_import_qa_or_scene_generation() -> None:
    source = Path("src/semantic_3d_chat/chat/runtime.py").read_text(encoding="utf-8")
    assert "semantic_3d_chat.data" not in source
    assert "qa_generator" not in source
    assert "generate_scene" not in source
