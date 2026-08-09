from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import semantic_3d_chat.evaluation.scene_signal_audit as audit_module
from semantic_3d_chat.evaluation.scene_signal_audit import (
    _configured_runtime_dtype,
    _encode_scene,
    _install_checkpoint_lora,
    _unvalidated_runtime_prefix_status,
    _validate_runtime_prefix_against_loaded_model,
)
from semantic_3d_chat.language.lora import (
    install_lora_adapters,
    lora_checkpoint_contract,
    lora_optimizer_settings,
    lora_settings,
)
from semantic_3d_chat.language.prefix_injection import (
    SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
    ContinuousPrefixComposer,
)
from semantic_3d_chat.scene_encoder.map_io import MapTensorData
from semantic_3d_chat.training.checkpointing import save_adapter_checkpoint


def _native_contract() -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_revision": "pinned-revision",
        "bos_token_id": 2,
        "pad_token_id": 0,
        "boi_token_id": 10,
        "image_token_id": 11,
        "eoi_token_id": 12,
        "use_bidirectional_attention": None,
    }


def _native_config(dtype: str = "bfloat16") -> dict:
    return {
        "paths": {"data_root": "data", "maps_root": "data/maps"},
        "scene": {"room_size_m": [6.0, 5.0, 3.0]},
        "scene_encoder": {"input_voxel_size_m": 0.15},
        "language": {
            "backend": "gemma4",
            "revision": "pinned-revision",
            "dtype": dtype,
            "scene_prefix_after_bos": True,
            "scene_boundary_mode": SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
            "gemma4_native_image_contract": _native_contract(),
        },
    }


def _tiny_map() -> MapTensorData:
    return MapTensorData(
        semantic=torch.zeros(2, 3),
        xyz=torch.zeros(2, 3),
        rgb=torch.zeros(2, 3),
        normal=torch.zeros(2, 3),
        confidence=torch.ones(2),
        observation_count=torch.ones(2),
        room_min=torch.tensor([-3.0, -2.5, 0.0]),
        room_max=torch.tensor([3.0, 2.5, 3.0]),
        source_voxel_count=2,
        input_voxel_size_m=0.15,
    )


class _SceneModel(nn.Module):
    def forward(self, semantic, *_args):
        return SimpleNamespace(
            scene_tokens=torch.tensor(
                [[[0.33333334, -0.7777778], [1.125, -1.875]]],
                device=semantic.device,
            ),
            native_latents=torch.zeros(1, 2, 2, device=semantic.device),
            block_tokens=torch.zeros(1, 2, device=semantic.device),
            audit={"block_indices": torch.zeros(1, 3, dtype=torch.long)},
        )


class _LoadedLanguage:
    def __init__(self, contract, boundaries, dtype: torch.dtype) -> None:
        self._contract = contract
        self._boundaries = boundaries
        self.model = nn.Linear(2, 2, bias=False).to(dtype=dtype)

    def scene_boundary_contract(self, _mode):
        return self._contract

    def scene_boundary_embeddings(self, _mode):
        return self._boundaries


def test_audit_projection_uses_supplied_effective_runtime_dtype_and_neutral_names(
    monkeypatch,
) -> None:
    monkeypatch.setattr(audit_module, "load_map_tensors", lambda *_args, **_kwargs: _tiny_map())
    composer = ContinuousPrefixComposer(2)
    _, representation = _encode_scene(
        _native_config(),
        "scene_000001",
        _SceneModel(),
        composer,
        torch.device("cpu"),
        torch.bfloat16,
    )

    assert representation["projected_scene_tokens_runtime_dtype"].dtype is torch.bfloat16
    assert representation["final_prefix_runtime_dtype"].dtype is torch.bfloat16
    assert "projected_scene_tokens_runtime_float16" not in representation
    assert "final_prefix_runtime_float16" not in representation


def test_skip_generation_explicitly_disclaims_model_and_boundary_validation(
    monkeypatch,
) -> None:
    captured = {}

    def fake_safe_dtype(device, requested):
        captured.update(device=device, requested=requested)
        return torch.bfloat16

    monkeypatch.setattr(audit_module, "safe_dtype", fake_safe_dtype)
    config = _native_config()
    runtime_dtype = _configured_runtime_dtype(config, torch.device("mps"))
    status = _unvalidated_runtime_prefix_status(config, runtime_dtype)

    assert runtime_dtype is torch.bfloat16
    assert captured == {"device": torch.device("mps"), "requested": "bfloat16"}
    assert status["status"] == "checkpoint_projected_not_model_validated"
    assert status["configured_runtime_dtype"] == "bfloat16"
    assert status["base_model_loaded"] is False
    assert status["native_boundary_validation_required"] is True
    assert status["native_boundary_embeddings_validated"] is False
    assert status["runtime_prefix_parity_validated"] is False


def test_loaded_model_validation_checks_dtype_contract_and_checkpoint_boundaries() -> None:
    torch.manual_seed(101)
    native = (
        torch.randn(1, 1, 2, dtype=torch.bfloat16),
        torch.randn(1, 1, 2, dtype=torch.bfloat16),
    )
    composer = ContinuousPrefixComposer(
        2,
        scene_prefix_after_bos=True,
        bos_token_id=2,
        scene_boundary_mode=SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
        native_boundary_embeddings=native,
    )
    language = _LoadedLanguage(_native_contract(), native, torch.bfloat16)
    status = _validate_runtime_prefix_against_loaded_model(
        _native_config(), language, composer, torch.bfloat16
    )

    assert status["status"] == "model_validated_runtime_prefix"
    assert status["runtime_dtype_validated_against_loaded_model"] is True
    assert status["native_boundary_embeddings_validated"] is True
    assert status["runtime_prefix_parity_validated"] is True

    with torch.no_grad():
        composer.scene_start.add_(1)
    with pytest.raises(ValueError, match="BOI boundary embedding"):
        _validate_runtime_prefix_against_loaded_model(
            _native_config(), language, composer, torch.bfloat16
        )

    fresh = ContinuousPrefixComposer(
        2,
        scene_prefix_after_bos=True,
        bos_token_id=2,
        scene_boundary_mode=SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
        native_boundary_embeddings=native,
    )
    with pytest.raises(ValueError, match="dtype"):
        _validate_runtime_prefix_against_loaded_model(
            _native_config(), language, fresh, torch.float16
        )


def test_audit_installs_and_hash_validates_checkpoint_lora_before_forward(tmp_path) -> None:
    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = nn.Module()
            self.decoder.proj = nn.Linear(3, 5, bias=False)

    config = _native_config()
    config["language"].update(
        {
            "lora": {
                "enabled": True,
                "rank": 2,
                "alpha": 4.0,
                "dropout": 0.0,
                "target_modules": ["decoder.proj"],
            }
        }
    )
    config["training"] = {
        "lora_learning_rate": 1e-4,
        "lora_weight_decay": 0.0,
    }
    settings = lora_settings(config)
    optimizer_settings = lora_optimizer_settings(config, settings)
    source_model = TinyModel().requires_grad_(False)
    source = install_lora_adapters(source_model, settings)
    assert source is not None and optimizer_settings is not None
    with torch.no_grad():
        source.adapters[0].lora_b.fill_(0.375)
    metadata = {
        "lora": lora_checkpoint_contract(settings, optimizer_settings, source.parameter_count),
        "lora_wrapped_modules": list(source.target_names),
        "lora_trainable_parameter_counts": source.parameter_counts,
        "lora_trainable_parameter_count": source.parameter_count,
        "lora_state_sha256": source.state_sha256(),
    }
    checkpoint = save_adapter_checkpoint(
        tmp_path / "checkpoint", {"lora": source.state_module}, metadata
    )
    runtime_model = TinyModel().requires_grad_(False)
    language = SimpleNamespace(model=runtime_model)

    restored = _install_checkpoint_lora(config, language, checkpoint, metadata)

    assert restored is not None
    assert torch.equal(restored.adapters[0].lora_b, source.adapters[0].lora_b)
    assert restored.training is False
    assert all(not parameter.requires_grad for parameter in runtime_model.parameters())
