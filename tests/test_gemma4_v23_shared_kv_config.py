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
from semantic_3d_chat.training.train_adapter import (
    build_adapter_optimizer,
    named_lora_extension_checkpoint_modules,
    named_lora_extension_transition_mismatch,
)

V23_CONFIG = "configs/experiments/gemma4_color_mirror_shared_kv_v23.yaml"
SOURCE_ADAPTER_SHA256 = "ce9e97061389a7eae5703593d0a8869f87bd12544f56f5976570965056b65f44"
SOURCE_METADATA_SHA256 = "bbc8309d25db86e40fa01ec744e19b3c0fc1c61953ebfc5072f11c84bbd2e997"
SOURCE_SCENE_SHA256 = "690bd890bfda024dbb5c7d3c68087b8113bc3b8ee81dd6143c7eb2a884e7245b"
SOURCE_GLOBAL_SHA256 = "ce3bf864eed6dd4a50f1b67296981e144d6a79e9cf192ad9a9230f2ae18208dc"
SOURCE_SIGNED_X_SHA256 = "e8dabc69627f60723b89520b02dfee985e49b7b7e35fdd1213cc79f7b8164f58"
V23_BANK_SHA256 = "707defddb599baf670ab3fec6594d8f8ccccd6b31689393c1c7ca30abaeed59d"
V23_TARGETS = (
    "model.language_model.layers.13.self_attn.k_proj",
    "model.language_model.layers.13.self_attn.v_proj",
    "model.language_model.layers.14.self_attn.k_proj",
    "model.language_model.layers.14.self_attn.v_proj",
)


class _ShapeOnlyLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int) -> None:
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.zeros(1, dtype=torch.float32).expand(out_features, in_features),
            requires_grad=False,
        )
        self.bias = None


class _Attention(nn.Module):
    def __init__(self, layer: int) -> None:
        super().__init__()
        if layer == 13:
            self.k_proj = _ShapeOnlyLinear(1536, 256)
            self.v_proj = _ShapeOnlyLinear(1536, 256)
        if layer == 14:
            self.k_proj = _ShapeOnlyLinear(1536, 512)
            self.v_proj = _ShapeOnlyLinear(1536, 512)
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
        self.model.language_model.layers = nn.ModuleList(_Layer(layer) for layer in range(35))


def _source_metadata(config: dict) -> dict:
    source = Path(config["training"]["initialize_from"])
    source = source if source.is_absolute() else PROJECT_ROOT / source
    return json.loads((source / "metadata.json").read_text(encoding="utf-8"))


def test_v23_is_exact_v21_epoch8_frozen_stack_kv_only_transition() -> None:
    config = load_config(V23_CONFIG)
    training = config["training"]
    experiment = config["experiment"]

    assert training["initialize_from"].endswith(
        "/gemma4_v21_phase_aware_local_field_extension_u8/epoch_008"
    )
    assert training["initialize_expected_adapter_sha256"] == SOURCE_ADAPTER_SHA256
    assert training["initialize_expected_metadata_sha256"] == SOURCE_METADATA_SHA256
    assert training["initialize_expected_scene_state_sha256"] == SOURCE_SCENE_SHA256
    assert training["initialize_expected_global_scene_residual_state_sha256"] == (
        SOURCE_GLOBAL_SHA256
    )
    assert training["initialize_expected_signed_x_scene_residual_state_sha256"] == (
        SOURCE_SIGNED_X_SHA256
    )
    assert training["train_lora_with_frozen_scene_residual_stack"] is True
    assert training["train_global_scene_residual_only"] is False
    assert training["train_signed_x_scene_residual_only"] is False
    assert training["initialize_source_residual_into_frozen_base"] is False
    assert training["freeze_scene_adapter"] is True
    assert training["learning_rate"] == 3e-4
    assert training["lora_learning_rate"] == 3e-4
    assert training["optimizer"]["learning_rate"] == 3e-4
    assert training["epochs"] == training["gradient_accumulation"] // 3
    assert training["pair_steps_per_epoch"] == training["gradient_accumulation"] == 12
    assert experiment["decoder_trainable_parameter_count"] == 30_720
    assert experiment["source_scene_state_sha256"] == SOURCE_SCENE_SHA256
    assert experiment["source_global_scene_residual_state_sha256"] == SOURCE_GLOBAL_SHA256
    assert experiment["source_signed_x_scene_residual_state_sha256"] == (SOURCE_SIGNED_X_SHA256)
    assert experiment["question_dependent_scene_processing"] is False
    assert experiment["question_dependent_retrieval"] is False
    assert experiment["runtime_oracle_access"] is False


def test_v23_bank_is_deterministic_zero_output_and_exactly_30720_parameters() -> None:
    config = load_config(V23_CONFIG)
    settings = lora_banks_settings(config)
    first = install_lora_banks(_ShapeOnlyGemma().requires_grad_(False), settings)
    second = install_lora_banks(_ShapeOnlyGemma().requires_grad_(False), settings)
    assert first is not None and second is not None

    bank = first.bank("extension_v23_shared_kv")
    assert bank.settings.trainable is True
    assert bank.settings.adapter.rank == 4
    assert bank.settings.adapter.alpha == 8.0
    assert bank.settings.adapter.target_modules == V23_TARGETS
    assert bank.installation.parameter_count == 30_720
    assert bank.installation.state_sha256() == V23_BANK_SHA256
    assert second.bank("extension_v23_shared_kv").installation.state_sha256() == (V23_BANK_SHA256)
    assert all(
        torch.count_nonzero(adapter.lora_b).item() == 0 for adapter in bank.installation.adapters
    )
    assert first.trainable_parameter_count == 30_720


def test_v23_source_contract_excludes_only_new_bank_and_optimizer_is_lora_only() -> None:
    config = load_config(V23_CONFIG)
    settings = lora_banks_settings(config)
    collection = install_lora_banks(_ShapeOnlyGemma().requires_grad_(False), settings)
    assert collection is not None
    source_metadata = _source_metadata(config)
    assert named_lora_extension_transition_mismatch(source_metadata, collection) is None

    checkpoint_modules = {
        "scene_model": nn.Identity(),
        "composer": nn.Identity(),
        "grounding": nn.Identity(),
        "global_scene_residual": nn.Identity(),
        "signed_x_scene_residual": nn.Identity(),
        **collection.state_modules(),
    }
    source_modules = named_lora_extension_checkpoint_modules(
        checkpoint_modules,
        collection,
    )
    assert "lora_banks.extension_v23_shared_kv" not in source_modules
    assert set(source_modules) == set(checkpoint_modules) - {"lora_banks.extension_v23_shared_kv"}

    optimizer_settings = lora_banks_optimizer_settings(config, settings)
    assert optimizer_settings is not None
    optimizer, parameters = build_adapter_optimizer(
        config,
        [],
        collection,
        optimizer_settings,
    )
    assert parameters == collection.parameters()
    assert [group["name"] for group in optimizer.param_groups] == ["language_lora"]
    assert optimizer.param_groups[0]["lr"] == 3e-4
    assert optimizer.param_groups[0]["weight_decay"] == 0.0


def test_v23_transition_rejects_a_source_that_claims_the_new_bank() -> None:
    config = load_config(V23_CONFIG)
    settings = lora_banks_settings(config)
    collection = install_lora_banks(_ShapeOnlyGemma().requires_grad_(False), settings)
    assert collection is not None
    source_metadata = _source_metadata(config)
    source_metadata["lora"]["banks"].append(
        {
            "name": "extension_v23_shared_kv",
            "trainable": True,
        }
    )
    mismatch = named_lora_extension_transition_mismatch(source_metadata, collection)
    assert mismatch is not None
    assert "bank_names" in mismatch


def test_v23_transition_requires_exact_source_hash_wrapping_and_count_key_sets() -> None:
    config = load_config(V23_CONFIG)
    settings = lora_banks_settings(config)
    collection = install_lora_banks(_ShapeOnlyGemma().requires_grad_(False), settings)
    assert collection is not None
    source_metadata = _source_metadata(config)

    for field in (
        "lora_bank_state_sha256",
        "lora_bank_wrapped_modules",
        "lora_bank_parameter_counts",
    ):
        missing = deepcopy(source_metadata)
        missing.pop(field)
        assert named_lora_extension_transition_mismatch(missing, collection) is not None

        stale = deepcopy(source_metadata)
        stale[field]["stale_bank"] = stale[field][next(iter(stale[field]))]
        assert named_lora_extension_transition_mismatch(stale, collection) is not None


def test_v23_transition_rejects_duplicate_and_malformed_source_bank_records() -> None:
    config = load_config(V23_CONFIG)
    settings = lora_banks_settings(config)
    collection = install_lora_banks(_ShapeOnlyGemma().requires_grad_(False), settings)
    assert collection is not None
    source_metadata = _source_metadata(config)

    duplicate = deepcopy(source_metadata)
    duplicate["lora"]["banks"].append(deepcopy(duplicate["lora"]["banks"][0]))
    duplicate_mismatch = named_lora_extension_transition_mismatch(duplicate, collection)
    assert duplicate_mismatch is not None
    assert duplicate_mismatch["bank_record_count"] == {
        "checkpoint": 3,
        "runtime_frozen_source": 2,
    }
    assert duplicate_mismatch["duplicate_bank_names"] == ["inherited_v12"]

    malformed = deepcopy(source_metadata)
    malformed["lora"]["banks"][0] = {"trainable": False}
    malformed_mismatch = named_lora_extension_transition_mismatch(malformed, collection)
    assert malformed_mismatch is not None
    assert malformed_mismatch["malformed_bank_records"] == [{"index": 0, "reason": "missing_name"}]
