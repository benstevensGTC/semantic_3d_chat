from __future__ import annotations

import json
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
    tensor_state_sha256,
)
from semantic_3d_chat.language.prefix_injection import ContinuousPrefixComposer
from semantic_3d_chat.scene_encoder.global_residual import (
    ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
    GlobalSceneResidual,
    apply_global_scene_residual,
    construct_global_scene_residual,
    global_scene_residual_settings,
)
from semantic_3d_chat.scene_encoder.map_io import MapTensorData
from semantic_3d_chat.scene_encoder.projector import SceneTokenizerOutput
from semantic_3d_chat.scene_encoder.signed_x_dispatch import (
    apply_signed_x_scene_residual,
    construct_signed_x_scene_residual,
    frozen_v18_centered_content_values,
    signed_x_scene_residual_settings,
)
from semantic_3d_chat.scene_encoder.signed_x_local_field import (
    SIGNED_X_LOCAL_FIELD_ARCHITECTURE_MARKER,
    SIGNED_X_LOCAL_FIELD_V2,
    SignedXLocalFieldSceneResidual,
)
from semantic_3d_chat.scene_encoder.signed_x_residual import (
    SIGNED_X_MOMENT_V1,
    SignedXSceneResidual,
)
from semantic_3d_chat.training.checkpointing import (
    module_collection_state_sha256,
    runtime_checkpoint_metadata,
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


def _tiny_global_residual_runtime_checkpoint(
    tmp_path: Path,
    *,
    content_gated: bool = False,
) -> tuple[dict, Path, GlobalSceneResidual, str]:
    """Build a fully strict synthetic checkpoint with a trained residual."""

    torch.manual_seed(16161)
    config = tiny_config()
    config["scene_encoder"]["architecture_version"] = "signal_preserving_resampler_v3"
    config["language"].update({"backend": "causal_lm", "dtype": "float32"})
    architecture = (
        {"architecture_version": ZERO_SPATIAL_MEAN_CONTENT_GATE_V1, "gate_temperature": 0.75}
        if content_gated
        else {}
    )
    initial_residual = GlobalSceneResidual(
        scene_dim=8,
        latent_count=4,
        width=4,
        fourier_bands=2,
        initialization_seed=16162,
        **architecture,
    )
    initial_hash = module_collection_state_sha256({"global_scene_residual": initial_residual})
    config["scene_encoder"]["global_scene_residual"] = {
        "enabled": True,
        "width": 4,
        "fourier_bands": 2,
        "initialization_seed": 16162,
        "expected_initial_state_sha256": initial_hash,
        **architecture,
    }
    source_residual = construct_global_scene_residual(
        config,
        scene_dim=8,
        latent_count=4,
    )
    assert source_residual is not None
    with torch.no_grad():
        source_residual.output_projection.weight.fill_(0.03125)
    trained_hash = module_collection_state_sha256({"global_scene_residual": source_residual})

    scene_model = construct_scene_tokenizer(config, semantic_dim=7, language_hidden_dim=8)
    composer = ContinuousPrefixComposer(8)
    grounding = QuestionGroundingHead(6, 8, 4, 6)
    metadata = {
        **tiny_checkpoint_metadata(),
        "config_hash": config_hash(config),
        "language_backend": "causal_lm",
        "scene_encoder_architecture_version": "signal_preserving_resampler_v3",
        "global_scene_residual": global_scene_residual_settings(config).contract(),
        "global_scene_residual_parameter_count": sum(
            parameter.numel() for parameter in source_residual.parameters()
        ),
        "global_scene_residual_initial_state_sha256": initial_hash,
        "global_scene_residual_state_sha256": trained_hash,
        "global_scene_residual_zero_output_equivalence": {
            "verified": True,
            "question_dependent_scene_processing": False,
        },
        "question_dependent_scene_processing": False,
    }
    checkpoint = save_adapter_checkpoint(
        tmp_path
        / ("runtime_content_gated_residual" if content_gated else "runtime_global_residual"),
        {
            "scene_model": scene_model,
            "composer": composer,
            "grounding": grounding,
            "global_scene_residual": source_residual,
        },
        metadata,
    )
    return config, checkpoint, source_residual, trained_hash


def _tiny_signed_x_runtime_checkpoint(
    tmp_path: Path,
    architecture_version: str = SIGNED_X_MOMENT_V1,
) -> tuple[
    dict,
    Path,
    GlobalSceneResidual,
    SignedXSceneResidual,
    str,
    str,
]:
    """Build a strict signed-X checkpoint over a frozen trained V18 base."""

    torch.manual_seed(19191)
    config = tiny_config()
    config["scene_encoder"]["architecture_version"] = "signal_preserving_resampler_v3"
    config["language"].update({"backend": "causal_lm", "dtype": "float32"})

    initial_global = GlobalSceneResidual(
        scene_dim=8,
        latent_count=4,
        width=4,
        fourier_bands=2,
        initialization_seed=19192,
        architecture_version=ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
        gate_temperature=0.75,
    )
    initial_global_hash = module_collection_state_sha256({"global_scene_residual": initial_global})
    config["scene_encoder"]["global_scene_residual"] = {
        "enabled": True,
        "width": 4,
        "fourier_bands": 2,
        "initialization_seed": 19192,
        "expected_initial_state_sha256": initial_global_hash,
        "architecture_version": ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
        "gate_temperature": 0.75,
    }
    source_global = construct_global_scene_residual(
        config,
        scene_dim=8,
        latent_count=4,
    )
    assert source_global is not None
    with torch.no_grad():
        source_global.output_projection.weight.fill_(0.03125)
    global_state_hash = module_collection_state_sha256({"global_scene_residual": source_global})

    signed_x_type = (
        SignedXSceneResidual
        if architecture_version == SIGNED_X_MOMENT_V1
        else SignedXLocalFieldSceneResidual
    )
    initial_signed_x = signed_x_type(scene_dim=8, latent_count=4, content_dim=4)
    initial_signed_x_hash = module_collection_state_sha256(
        {"signed_x_scene_residual": initial_signed_x}
    )
    config["scene_encoder"]["signed_x_scene_residual"] = {
        "enabled": True,
        "architecture_version": architecture_version,
        "expected_initial_state_sha256": initial_signed_x_hash,
    }
    config["training"] = {
        "freeze_scene_adapter": True,
        "train_signed_x_scene_residual_only": True,
    }
    source_signed_x = construct_signed_x_scene_residual(
        config,
        scene_dim=8,
        latent_count=4,
        content_dim=source_global.width,
    )
    assert source_signed_x is not None
    with torch.no_grad():
        source_signed_x.output_projection.weight.fill_(0.0625)
    signed_x_state_hash = module_collection_state_sha256(
        {"signed_x_scene_residual": source_signed_x}
    )

    scene_model = construct_scene_tokenizer(config, semantic_dim=7, language_hidden_dim=8)
    composer = ContinuousPrefixComposer(8)
    grounding = QuestionGroundingHead(6, 8, 4, 6)
    scene_modules = {
        "scene_model": scene_model,
        "composer": composer,
        "grounding": grounding,
    }
    update_zero_prefix_hash = "a" * 64
    metadata = {
        **tiny_checkpoint_metadata(),
        "config_hash": config_hash(config),
        "language_backend": "causal_lm",
        "scene_encoder_architecture_version": "signal_preserving_resampler_v3",
        "global_scene_residual": global_scene_residual_settings(config).contract(),
        "global_scene_residual_parameter_count": source_global.parameter_count,
        "global_scene_residual_initial_state_sha256": initial_global_hash,
        "global_scene_residual_state_sha256": global_state_hash,
        "global_scene_residual_zero_output_equivalence": None,
        "signed_x_scene_residual": signed_x_scene_residual_settings(config).contract(),
        "signed_x_scene_residual_parameter_count": source_signed_x.parameter_count,
        "signed_x_scene_residual_initial_state_sha256": initial_signed_x_hash,
        "signed_x_scene_residual_state_sha256": signed_x_state_hash,
        "signed_x_scene_residual_zero_output_equivalence": {
            "verified": True,
            "base": "loaded_frozen_global_scene_residual",
            "question_dependent_scene_processing": False,
            "all_scene_slots_accounted": True,
            "scene_count": 1,
            "scene_prefixes": {
                "scene_000001": {
                    "v18_base_prefix_sha256": update_zero_prefix_hash,
                    "signed_x_adapted_prefix_sha256": update_zero_prefix_hash,
                }
            },
        },
        "question_dependent_scene_processing": False,
        "freeze_scene_adapter": True,
        "frozen_scene_state_sha256": module_collection_state_sha256(scene_modules),
        "frozen_global_scene_residual_state_sha256": global_state_hash,
        "train_signed_x_scene_residual_only": True,
        "initialization_provenance": {
            "schema_version": 4,
            "mode": "frozen_v18_residual_base_plus_zero_output_signed_x_residual",
            "source_global_scene_residual_state_sha256": global_state_hash,
            "expected_source_global_scene_residual_state_sha256": global_state_hash,
            "global_scene_residual_frozen": True,
            "signed_x_scene_residual_initial_state_sha256": initial_signed_x_hash,
            "signed_x_scene_residual_zero_output": True,
            "optimizer_state_loaded": False,
            "history_loaded": False,
        },
    }
    checkpoint = save_adapter_checkpoint(
        tmp_path / f"runtime_signed_x_residual_{architecture_version}",
        {
            **scene_modules,
            "global_scene_residual": source_global,
            "signed_x_scene_residual": source_signed_x,
        },
        metadata,
    )
    return (
        config,
        checkpoint,
        source_global,
        source_signed_x,
        global_state_hash,
        signed_x_state_hash,
    )


def _mock_tiny_global_residual_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_module, "load_map_tensors", lambda *_args, **_kwargs: tiny_map())
    monkeypatch.setattr(
        runtime_module,
        "load_local_language_model",
        lambda *_args, **_kwargs: LocalLanguageModel(
            model=TinyLanguageModel(hidden_size=8).eval().requires_grad_(False),
            tokenizer=TinyTokenizer(),
            device=torch.device("cpu"),
            backend_name="causal_lm",
        ),
    )


def test_static_chat_roundtrips_global_residual_and_applies_it_before_questions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, checkpoint, source_residual, trained_hash = _tiny_global_residual_runtime_checkpoint(
        tmp_path
    )
    _mock_tiny_global_residual_runtime(monkeypatch)

    runtime = StaticChatRuntime.load(config, "scene_000001", checkpoint)

    assert runtime.questions_answered == 0
    assert runtime.global_scene_residual is not None
    assert (
        module_collection_state_sha256({"global_scene_residual": runtime.global_scene_residual})
        == trained_hash
    )
    for name, expected in source_residual.state_dict().items():
        assert torch.equal(runtime.global_scene_residual.state_dict()[name], expected)
    model_dtype = next(runtime.language.model.parameters()).dtype
    core_prefix = runtime.composer.scene_prefix(
        runtime.core_scene_output.scene_tokens.to(dtype=model_dtype)
    )
    assert not torch.equal(runtime.scene_prefix, core_prefix)
    assert runtime.scene_output.audit["global_scene_residual_delta_rms"].item() > 0.0
    assert runtime.current_prefix_hash() == runtime.scene_prefix_hash
    assert runtime.startup_summary()["global_scene_residual_state_sha256"] == trained_hash


def test_static_chat_roundtrips_content_gated_residual_contract_and_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, checkpoint, source_residual, trained_hash = _tiny_global_residual_runtime_checkpoint(
        tmp_path, content_gated=True
    )
    _mock_tiny_global_residual_runtime(monkeypatch)

    runtime = StaticChatRuntime.load(config, "scene_000001", checkpoint)

    assert runtime.questions_answered == 0
    assert runtime.global_scene_residual is not None
    assert runtime.global_scene_residual.parameter_count == source_residual.parameter_count
    assert (
        runtime.global_scene_residual.validate_structural_state()["architecture_version"]
        == ZERO_SPATIAL_MEAN_CONTENT_GATE_V1
    )
    assert torch.equal(
        runtime.global_scene_residual.content_gate_projection.weight,
        source_residual.content_gate_projection.weight,
    )
    assert torch.equal(
        runtime.global_scene_residual.gate_temperature,
        source_residual.gate_temperature,
    )
    assert runtime.checkpoint_metadata["global_scene_residual"]["schema_version"] == 2
    assert runtime.startup_summary()["global_scene_residual_state_sha256"] == trained_hash
    assert runtime.current_prefix_hash() == runtime.scene_prefix_hash


def test_static_chat_roundtrips_signed_x_and_applies_core_then_global_then_signed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        config,
        checkpoint,
        source_global,
        source_signed_x,
        global_hash,
        signed_x_hash,
    ) = _tiny_signed_x_runtime_checkpoint(tmp_path)
    _mock_tiny_global_residual_runtime(monkeypatch)

    def fake_generation(_model, _embeddings, _mask, _maximum, _eos):
        return torch.tensor([[7, 2]])

    runtime = StaticChatRuntime.load(
        config,
        "scene_000001",
        checkpoint,
        generation_function=fake_generation,
    )

    assert runtime.questions_answered == 0
    assert runtime.global_scene_residual is not None
    assert runtime.signed_x_scene_residual is not None
    assert (
        module_collection_state_sha256({"global_scene_residual": runtime.global_scene_residual})
        == global_hash
    )
    assert (
        module_collection_state_sha256({"signed_x_scene_residual": runtime.signed_x_scene_residual})
        == signed_x_hash
    )
    for name, expected in source_global.state_dict().items():
        assert torch.equal(runtime.global_scene_residual.state_dict()[name], expected)
    for name, expected in source_signed_x.state_dict().items():
        assert torch.equal(runtime.signed_x_scene_residual.state_dict()[name], expected)

    with torch.inference_mode():
        centered_content = frozen_v18_centered_content_values(
            runtime.global_scene_residual,
            runtime.core_scene_output.scene_tokens,
        )
        expected_global = apply_global_scene_residual(
            runtime.core_scene_output,
            runtime.global_scene_residual,
        )
        expected_signed = apply_signed_x_scene_residual(
            expected_global,
            runtime.signed_x_scene_residual,
            centered_content,
        )
    assert torch.equal(runtime.global_scene_output.scene_tokens, expected_global.scene_tokens)
    assert torch.equal(runtime.scene_output.scene_tokens, expected_signed.scene_tokens)
    assert runtime.scene_output.audit["signed_x_scene_residual_delta_rms"].item() > 0.0
    assert (
        runtime.scene_output.audit["signed_x_scene_residual_accounted_slots"].item()
        == config["scene_encoder"]["global_latents"]
    )

    initial_prefix_hash = runtime.scene_prefix_hash
    answer = runtime.answer("Which side is occupied?")
    assert answer.prefix_hash == initial_prefix_hash
    assert runtime.current_prefix_hash() == initial_prefix_hash
    summary = runtime.startup_summary()
    assert summary["signed_x_scene_residual"] == signed_x_scene_residual_settings(config).contract()
    assert summary["signed_x_scene_residual_state_sha256"] == signed_x_hash
    assert summary["frozen_global_scene_residual_state_sha256"] == global_hash


def test_static_chat_roundtrips_v20_local_field_and_reuses_question_independent_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        config,
        checkpoint,
        _source_global,
        source_local_field,
        _global_hash,
        local_field_hash,
    ) = _tiny_signed_x_runtime_checkpoint(tmp_path, SIGNED_X_LOCAL_FIELD_V2)
    _mock_tiny_global_residual_runtime(monkeypatch)

    runtime = StaticChatRuntime.load(
        config,
        "scene_000001",
        checkpoint,
        generation_function=lambda *_args: torch.tensor([[7, 2]]),
    )

    assert isinstance(runtime.signed_x_scene_residual, SignedXLocalFieldSceneResidual)
    assert runtime.signed_x_scene_residual.architecture_marker.item() == (
        SIGNED_X_LOCAL_FIELD_ARCHITECTURE_MARKER
    )
    assert (
        module_collection_state_sha256({"signed_x_scene_residual": runtime.signed_x_scene_residual})
        == local_field_hash
    )
    for name, expected in source_local_field.state_dict().items():
        assert torch.equal(runtime.signed_x_scene_residual.state_dict()[name], expected)
    assert runtime.scene_output.audit["signed_x_scene_residual_local_field_rms"].item() > 0.0
    assert "signed_x_scene_residual_moment_rms" not in runtime.scene_output.audit

    prefix_hash = runtime.current_prefix_hash()
    assert runtime.answer("Which side is occupied?").prefix_hash == prefix_hash
    assert runtime.answer("What changed?").prefix_hash == prefix_hash
    assert runtime.current_prefix_hash() == prefix_hash
    assert runtime.startup_summary()["signed_x_scene_residual"] == (
        signed_x_scene_residual_settings(config).contract()
    )


@pytest.mark.parametrize(
    ("metadata_path", "bad_value"),
    [
        (("initialization_provenance", "global_scene_residual_frozen"), False),
        (
            ("signed_x_scene_residual_zero_output_equivalence", "all_scene_slots_accounted"),
            False,
        ),
        (
            ("initialization_provenance", "signed_x_zero_output_transition_verified"),
            False,
        ),
    ],
)
def test_checkpoint_contract_rejects_signed_x_without_explicit_frozen_base_evidence(
    tmp_path: Path,
    metadata_path: tuple[str, str],
    bad_value: object,
) -> None:
    config, checkpoint, *_rest = _tiny_signed_x_runtime_checkpoint(tmp_path)
    metadata = json.loads((checkpoint / "runtime_metadata.json").read_text(encoding="utf-8"))
    section, field = metadata_path
    metadata[section][field] = bad_value

    with pytest.raises(ValueError, match="signed_x_frozen_v18_base_provenance"):
        validate_checkpoint_contract(
            metadata,
            config,
            semantic_dim=7,
            language_hidden_dim=8,
        )


def test_runtime_provenance_sanitizes_later_stage_training_history(tmp_path: Path) -> None:
    config, checkpoint, _global, _signed, _global_hash, signed_hash = (
        _tiny_signed_x_runtime_checkpoint(tmp_path)
    )
    training_metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    training_metadata["frozen_signed_x_scene_residual_state_sha256"] = signed_hash
    training_metadata["initialization_provenance"] = {
        "schema_version": 7,
        "mode": "frozen_named_lora_scene_stack_plus_zero_output_dense_alignment",
        "checkpoint": "training-only/source",
        "history_loaded": False,
        "question_ids": ["q_000031"],
    }

    runtime_metadata = runtime_checkpoint_metadata(training_metadata)

    assert runtime_metadata["initialization_provenance"] == {
        "schema_version": 1,
        "source_global_scene_residual_state_sha256": training_metadata[
            "global_scene_residual_state_sha256"
        ],
        "source_signed_x_scene_residual_state_sha256": signed_hash,
        "global_scene_residual_frozen": True,
        "signed_x_scene_residual_frozen": True,
        "signed_x_scene_residual_initial_state_sha256": training_metadata[
            "signed_x_scene_residual_initial_state_sha256"
        ],
        "signed_x_zero_output_transition_verified": True,
        "question_dependent_scene_processing": False,
    }
    assert "history" not in runtime_metadata
    assert "question_ids" not in json.dumps(runtime_metadata)
    validate_checkpoint_contract(
        runtime_metadata,
        config,
        semantic_dim=7,
        language_hidden_dim=8,
    )
    runtime_metadata["frozen_signed_x_scene_residual_state_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="frozen_signed_x_scene_residual_state_sha256"):
        validate_checkpoint_contract(
            runtime_metadata,
            config,
            semantic_dim=7,
            language_hidden_dim=8,
        )


def test_checkpoint_contract_does_not_allow_null_global_equivalence_without_signed_x(
    tmp_path: Path,
) -> None:
    config, checkpoint, *_rest = _tiny_global_residual_runtime_checkpoint(
        tmp_path,
        content_gated=True,
    )
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    metadata["global_scene_residual_zero_output_equivalence"] = None

    with pytest.raises(ValueError, match="global_scene_residual_zero_output_equivalence"):
        validate_checkpoint_contract(
            metadata,
            config,
            semantic_dim=7,
            language_hidden_dim=8,
        )


def test_checkpoint_contract_accepts_explicitly_disabled_signed_x_metadata() -> None:
    config = tiny_config()
    metadata = {
        **tiny_checkpoint_metadata(),
        "signed_x_scene_residual": {"schema_version": 1, "enabled": False},
    }

    warnings = validate_checkpoint_contract(
        metadata,
        config,
        semantic_dim=7,
        language_hidden_dim=8,
    )

    assert warnings and "config hash differs" in warnings[0].lower()


def test_static_chat_rejects_signed_x_deterministic_initial_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, checkpoint, *_rest = _tiny_signed_x_runtime_checkpoint(tmp_path)
    _mock_tiny_global_residual_runtime(monkeypatch)
    wrong_initial_hash = "b" * 64
    config["scene_encoder"]["signed_x_scene_residual"]["expected_initial_state_sha256"] = (
        wrong_initial_hash
    )
    metadata_path = checkpoint / "runtime_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["signed_x_scene_residual"]["expected_initial_state_sha256"] = wrong_initial_hash
    metadata["signed_x_scene_residual_initial_state_sha256"] = wrong_initial_hash
    metadata["initialization_provenance"]["signed_x_scene_residual_initial_state_sha256"] = (
        wrong_initial_hash
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="deterministic initial-state mismatch"):
        StaticChatRuntime.load(config, "scene_000001", checkpoint)


def test_static_chat_rejects_signed_x_parameter_count_metadata_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, checkpoint, _global, signed_x, *_hashes = _tiny_signed_x_runtime_checkpoint(tmp_path)
    _mock_tiny_global_residual_runtime(monkeypatch)
    metadata_path = checkpoint / "runtime_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["signed_x_scene_residual_parameter_count"] = signed_x.parameter_count - 1
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Signed-X.*parameter-count mismatch"):
        StaticChatRuntime.load(config, "scene_000001", checkpoint)


def test_static_chat_rejects_signed_x_state_hash_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, checkpoint, *_rest = _tiny_signed_x_runtime_checkpoint(tmp_path)
    _mock_tiny_global_residual_runtime(monkeypatch)
    tensors = load_file(checkpoint / "adapter.safetensors")
    tensors["signed_x_scene_residual.output_projection.weight"].view(-1)[0].add_(0.125)
    save_file(tensors, checkpoint / "adapter.safetensors")

    with pytest.raises(ValueError, match="Signed-X scene residual state mismatch"):
        StaticChatRuntime.load(config, "scene_000001", checkpoint)


def test_static_chat_rejects_rehashed_signed_x_anchor_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, checkpoint, *_rest = _tiny_signed_x_runtime_checkpoint(tmp_path)
    _mock_tiny_global_residual_runtime(monkeypatch)
    tensors = load_file(checkpoint / "adapter.safetensors")
    tensors["signed_x_scene_residual.signed_x_anchors"].view(-1)[0].add_(0.125)
    save_file(tensors, checkpoint / "adapter.safetensors")
    metadata_path = checkpoint / "runtime_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    signed_x_state = {
        name: value
        for name, value in tensors.items()
        if name.startswith("signed_x_scene_residual.")
    }
    rehashed_state = tensor_state_sha256(signed_x_state)
    metadata["signed_x_scene_residual_state_sha256"] = rehashed_state
    metadata["initialization_provenance"][
        "source_signed_x_scene_residual_state_sha256"
    ] = rehashed_state
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="signed-X anchors"):
        StaticChatRuntime.load(config, "scene_000001", checkpoint)


@pytest.mark.parametrize(
    "tampered_key",
    [
        "global_scene_residual.output_projection.weight",
        "global_scene_residual.content_gate_projection.weight",
        "global_scene_residual.gate_temperature",
        "global_scene_residual.position_features",
    ],
)
def test_static_chat_rejects_global_residual_parameter_or_buffer_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tampered_key: str,
) -> None:
    config, checkpoint, _source_residual, _trained_hash = _tiny_global_residual_runtime_checkpoint(
        tmp_path, content_gated=True
    )
    _mock_tiny_global_residual_runtime(monkeypatch)
    tensors = load_file(checkpoint / "adapter.safetensors")
    tensors[tampered_key].view(-1)[0].add_(0.125)
    save_file(tensors, checkpoint / "adapter.safetensors")

    with pytest.raises(ValueError, match="Global scene residual|Persistent"):
        StaticChatRuntime.load(config, "scene_000001", checkpoint)


def _rewrite_residual_tensors_and_attest_hash(
    checkpoint: Path,
    tensors: dict[str, torch.Tensor],
) -> None:
    """Model an attacker changing both tensor state and its adjacent metadata hash."""

    save_file(tensors, checkpoint / "adapter.safetensors")
    metadata_path = checkpoint / "runtime_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    residual_state = {
        name: value for name, value in tensors.items() if name.startswith("global_scene_residual.")
    }
    metadata["global_scene_residual_state_sha256"] = tensor_state_sha256(residual_state)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("tampered_key", "message"),
    [
        ("global_scene_residual.gate_temperature", "temperature"),
        ("global_scene_residual.position_features", "position features"),
    ],
)
def test_static_chat_rejects_rehashed_immutable_residual_buffer_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tampered_key: str,
    message: str,
) -> None:
    config, checkpoint, _source_residual, _trained_hash = _tiny_global_residual_runtime_checkpoint(
        tmp_path, content_gated=True
    )
    _mock_tiny_global_residual_runtime(monkeypatch)
    tensors = load_file(checkpoint / "adapter.safetensors")
    tensors[tampered_key].view(-1)[0].add_(0.125)
    _rewrite_residual_tensors_and_attest_hash(checkpoint, tensors)

    with pytest.raises(ValueError, match=message):
        StaticChatRuntime.load(config, "scene_000001", checkpoint)


def test_static_chat_rejects_rehashed_nonfinite_residual_parameter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, checkpoint, _source_residual, _trained_hash = _tiny_global_residual_runtime_checkpoint(
        tmp_path, content_gated=True
    )
    _mock_tiny_global_residual_runtime(monkeypatch)
    tensors = load_file(checkpoint / "adapter.safetensors")
    tensors["global_scene_residual.content_gate_projection.weight"].view(-1)[0] = float("nan")
    _rewrite_residual_tensors_and_attest_hash(checkpoint, tensors)

    with pytest.raises(ValueError, match="nonfinite.*content_gate_projection.weight"):
        StaticChatRuntime.load(config, "scene_000001", checkpoint)


def test_static_chat_rejects_residual_parameter_count_metadata_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, checkpoint, source_residual, _trained_hash = _tiny_global_residual_runtime_checkpoint(
        tmp_path, content_gated=True
    )
    _mock_tiny_global_residual_runtime(monkeypatch)
    metadata_path = checkpoint / "runtime_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["global_scene_residual_parameter_count"] = source_residual.parameter_count - 1
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="parameter-count mismatch"):
        StaticChatRuntime.load(config, "scene_000001", checkpoint)


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
