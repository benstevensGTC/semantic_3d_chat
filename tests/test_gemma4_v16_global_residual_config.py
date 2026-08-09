from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.language.lora import (
    install_lora_banks,
    lora_banks_optimizer_settings,
    lora_banks_settings,
)
from semantic_3d_chat.scene_encoder.global_residual import (
    construct_global_scene_residual,
    global_scene_residual_settings,
)
from semantic_3d_chat.training.checkpointing import module_collection_state_sha256
from semantic_3d_chat.training.train_adapter import (
    build_adapter_optimizer,
    file_sha256,
    named_lora_freeze_transition_mismatch,
)

V16_CONFIG = "configs/experiments/gemma4_color_mirror_global_scene_residual_v16.yaml"
V14_ADAPTER_SHA256 = "9e15e8c93da083bd23c009bf67cdf4d532d6beb01b12f17f8bf664e2374294c7"
V14_METADATA_SHA256 = "e4cf9134f5ef931df821820c80f96f1839fd2ae9a89b4c06ce4998db330e930e"
SCENE_SHA256 = "690bd890bfda024dbb5c7d3c68087b8113bc3b8ee81dd6143c7eb2a884e7245b"
INHERITED_SHA256 = "dec768bed654c8c4e16da0318857543ad54d8f5f68f4d24a9a87cd19ec706594"
EXTENSION_SHA256 = "4eb90fb9b0bea579d14cfcb0f61ebd5b6d566fd600bd3d5e1bfe5177a39e1b34"
RESIDUAL_SHA256 = "fb4ebaac06dccbc04a461b10546d00f48cdf8cfbb372cbe5f6fe925f71461bd3"
V17_SWEEP_CONFIG = (
    "configs/experiments/gemma4_color_mirror_global_scene_residual_v17_lr_sweep.yaml"
)


class _ShapeOnlyLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int) -> None:
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.zeros(1).expand(out_features, in_features),
            requires_grad=False,
        )
        self.bias = None


class _Attention(nn.Module):
    def __init__(self, layer: int) -> None:
        super().__init__()
        if layer in (30, 31, 32, 33):
            self.q_proj = _ShapeOnlyLinear(1536, 2048)
            self.o_proj = _ShapeOnlyLinear(2048, 1536)
        if layer == 34:
            self.q_proj = _ShapeOnlyLinear(1536, 4096)
            self.o_proj = _ShapeOnlyLinear(4096, 1536)


class _Layer(nn.Module):
    def __init__(self, layer: int) -> None:
        super().__init__()
        self.self_attn = _Attention(layer)


class _ShapeOnlyGemma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList(
            _Layer(layer) for layer in range(35)
        )


def _source_checkpoint(config: dict) -> Path:
    path = Path(config["training"]["initialize_from"])
    return path if path.is_absolute() else PROJECT_ROOT / path


def test_v16_source_metadata_and_exact_v14_artifact_contract() -> None:
    config = load_config(V16_CONFIG)
    checkpoint = _source_checkpoint(config)
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    training = config["training"]
    experiment = config["experiment"]

    assert training["initialize_from"].endswith(
        "/gemma4_color_mirror_decoder_banks_v14_lr2e3/epoch_007"
    )
    assert training["initialize_expected_adapter_sha256"] == V14_ADAPTER_SHA256
    assert training["initialize_expected_metadata_sha256"] == V14_METADATA_SHA256
    assert file_sha256(checkpoint / "adapter.safetensors") == V14_ADAPTER_SHA256
    assert file_sha256(checkpoint / "metadata.json") == V14_METADATA_SHA256
    assert metadata["epoch"] == experiment["source_checkpoint_epoch"] == 7
    assert metadata["language_hidden_dim"] == 1536
    assert metadata["language_model_id"] == config["language"]["model_id"]
    assert metadata["language_revision"] == config["language"]["revision"]
    assert metadata["scene_latents"] == config["scene_encoder"]["global_latents"] == 256
    assert metadata["scene_model_dim"] == config["scene_encoder"]["model_dim"] == 384
    assert metadata["scene_encoder_architecture_version"] == (
        config["scene_encoder"]["architecture_version"]
    )
    assert experiment["source_scene_state_sha256"] == SCENE_SHA256
    assert metadata["frozen_scene_state_sha256"] == SCENE_SHA256


def test_v16_residual_contract_hash_parameter_count_optimizer_and_screen() -> None:
    config = load_config(V16_CONFIG)
    settings = global_scene_residual_settings(config)
    module = construct_global_scene_residual(config, scene_dim=1536, latent_count=256)
    assert module is not None

    assert settings.contract() == {
        "schema_version": 1,
        "enabled": True,
        "width": 128,
        "fourier_bands": 4,
        "initialization_seed": 16015,
        "expected_initial_state_sha256": RESIDUAL_SHA256,
    }
    assert sum(parameter.numel() for parameter in module.parameters()) == 400_000
    assert config["experiment"]["residual_parameter_count"] == 400_000
    assert module_collection_state_sha256({"global_scene_residual": module}) == (
        RESIDUAL_SHA256
    )
    assert torch.count_nonzero(module.output_projection.weight).item() == 0

    training = config["training"]
    assert training["freeze_scene_adapter"] is True
    assert training["train_global_scene_residual_only"] is True
    assert training["initialize_named_lora_freeze_transition"] is True
    assert training["initialize_legacy_lora_into_bank"] is None
    assert training["learning_rate"] == 0.001
    assert training["weight_decay"] == 0.0
    assert training["epochs"] == config["experiment"]["screen_optimizer_updates"] == 4
    assert training["pair_steps_per_epoch"] == training["gradient_accumulation"] == 12
    assert training["pair_gate_every_epochs"] == 1
    assert training["pair_gate_stop_when_passed"] is False
    assert training["early_stopping_patience"] == 0
    assert config["experiment"]["question_dependent_scene_processing"] is False
    assert config["experiment"]["screen_extension_requires"] == {
        "color_full_vocab_sides": 12,
        "color_full_vocab_units": 6,
        "color_positive_minimum_candidate_margin": True,
        "color_positive_minimum_full_vocab_margin": True,
        "mirror_minimum_full_vocab_sides": 8,
        "mirror_minimum_full_vocab_units": 2,
    }


def test_v16_banks_are_frozen_persisted_and_optimizer_is_residual_only() -> None:
    config = load_config(V16_CONFIG)
    settings = lora_banks_settings(config)
    assert settings.trainable is False
    assert lora_banks_optimizer_settings(config, settings) is None
    assert settings.bank("inherited_v12").trainable is False
    assert settings.bank("extension_v13").trainable is False
    assert settings.bank("inherited_v12").expected_initial_state_sha256 == INHERITED_SHA256
    assert settings.bank("extension_v13").expected_initial_state_sha256 == EXTENSION_SHA256
    assert config["experiment"]["source_inherited_bank_sha256"] == INHERITED_SHA256
    assert config["experiment"]["source_extension_bank_sha256"] == EXTENSION_SHA256

    source_metadata = json.loads(
        (_source_checkpoint(config) / "metadata.json").read_text(encoding="utf-8")
    )
    assert source_metadata["lora_bank_state_sha256"] == {
        "extension_v13": EXTENSION_SHA256,
        "inherited_v12": INHERITED_SHA256,
    }
    collection = install_lora_banks(_ShapeOnlyGemma().requires_grad_(False), settings)
    assert collection is not None
    assert named_lora_freeze_transition_mismatch(source_metadata, collection) is None
    assert all(not parameter.requires_grad for parameter in collection.all_parameters())
    assert collection.parameters() == []

    residual = construct_global_scene_residual(config, scene_dim=16, latent_count=9)
    assert residual is not None
    optimizer, parameters = build_adapter_optimizer(
        config,
        list(residual.parameters()),
        collection,
        configured_lora_optimizer=None,
    )
    assert {id(parameter) for parameter in parameters} == {
        id(parameter) for parameter in residual.parameters()
    }
    assert [group["name"] for group in optimizer.param_groups] == [
        "global_scene_residual"
    ]
    assert optimizer.param_groups[0]["lr"] == 0.001
    assert optimizer.param_groups[0]["weight_decay"] == 0.0


def test_v17_lr_arms_are_exact_v16_restarts_with_only_optimizer_response_changed() -> None:
    template = load_config(V17_SWEEP_CONFIG)
    v16 = load_config(V16_CONFIG)
    arms = {"lr1e4": 1e-4, "lr3e4": 3e-4}

    assert template["sweep"] is None
    assert template["lr_response"]["arms"] == [1e-4, 3e-4]
    assert template["lr_response"]["updates_per_arm"] == 4
    assert template["lr_response"]["conditional_max_updates"] == 12
    assert template["lr_response"]["expected_source_adapter_sha256"] == V14_ADAPTER_SHA256
    assert template["lr_response"]["expected_source_metadata_sha256"] == V14_METADATA_SHA256
    assert template["lr_response"]["expected_frozen_scene_state_sha256"] == SCENE_SHA256
    assert template["lr_response"]["expected_frozen_inherited_bank_sha256"] == INHERITED_SHA256
    assert template["lr_response"]["expected_frozen_extension_bank_sha256"] == EXTENSION_SHA256
    assert template["lr_response"]["expected_initial_residual_state_sha256"] == RESIDUAL_SHA256
    assert template["lr_response"]["continuation_requires"] == (
        v16["experiment"]["screen_extension_requires"]
    )
    assert template["lr_response"]["full_teacher_gate_requires"] == (
        v16["experiment"]["full_teacher_gate_requires"]
    )
    assert template["lr_response"]["greedy_audit_only_after_full_teacher_gate"] is True

    for suffix, learning_rate in arms.items():
        config = load_config(
            "configs/experiments/"
            f"gemma4_color_mirror_global_scene_residual_v17_{suffix}.yaml"
        )
        normalized_template = deepcopy(template)
        normalized_arm = deepcopy(config)
        normalized_template.pop("_config_path", None)
        normalized_arm.pop("_config_path", None)
        normalized_arm["training"]["output_namespace"] = normalized_template["training"][
            "output_namespace"
        ]
        normalized_arm["training"]["learning_rate"] = normalized_template["training"][
            "learning_rate"
        ]
        arm_learning_rate = normalized_arm["lr_response"].pop("arm_learning_rate")
        assert normalized_arm == normalized_template

        assert arm_learning_rate == learning_rate
        assert config["training"]["learning_rate"] == learning_rate
        assert config["training"]["output_namespace"] == (
            f"gemma4_color_mirror_global_scene_residual_v17_{suffix}"
        )
        assert config["training"]["initialize_from"] == v16["training"]["initialize_from"]
        assert config["training"]["initialize_expected_adapter_sha256"] == V14_ADAPTER_SHA256
        assert config["training"]["initialize_expected_metadata_sha256"] == (
            V14_METADATA_SHA256
        )
        assert config["training"]["train_global_scene_residual_only"] is True
        assert config["training"]["freeze_scene_adapter"] is True
        assert config["experiment"]["question_dependent_scene_processing"] is False
