from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from semantic_3d_chat.config import PROJECT_ROOT, config_hash, load_config
from semantic_3d_chat.language.lora import (
    install_lora_banks,
    lora_banks_optimizer_settings,
    lora_banks_settings,
)
from semantic_3d_chat.training.train_adapter import (
    build_adapter_optimizer,
    named_lora_extension_checkpoint_modules,
    named_lora_freeze_and_extend_transition_mismatch,
)

V23_CONFIG = "configs/experiments/gemma4_color_mirror_shared_kv_v23.yaml"
V24_CONFIG = "configs/experiments/gemma4_color_mirror_shared_query_v24.yaml"
V23_CHECKPOINT = Path("data_gemma4/checkpoints/gemma4_v23_shared_kv/epoch_002")
V23_ADAPTER_SHA256 = "dba2511db49fa46af905b293fc999642286f8533fa1d4cca2c872ffda2980ea8"
V23_METADATA_SHA256 = "1c0436549e832c2ac9723e2556ad8bf09862020c6cda47db8358b2232b391ba0"
V23_ARCHIVE_SHA256 = "cdb7cabc2b6a9a8420e682a20067dd501cc58a69addb2fb828d2b99fc94df208"
SCENE_SHA256 = "690bd890bfda024dbb5c7d3c68087b8113bc3b8ee81dd6143c7eb2a884e7245b"
GLOBAL_SHA256 = "ce3bf864eed6dd4a50f1b67296981e144d6a79e9cf192ad9a9230f2ae18208dc"
SIGNED_X_SHA256 = "e8dabc69627f60723b89520b02dfee985e49b7b7e35fdd1213cc79f7b8164f58"
V23_BANK_SHA256 = "91a9eea577cab5a37e840cdf4007722a398415846af91280713bcb2cda0f045c"
V24_INITIAL_SHA256 = "e8734db171db6bd47a9a4f8c9d4a540903cc214a88abaab74820d566ee245f6b"
EXPECTED_RESOLVED_CONFIG_SHA256 = "82d5fee205842fb86133498eb4ac7765e61c22e7e7bc2745cfa6a2e36b9447f1"
V24_TARGETS = (
    "model.language_model.layers.28.self_attn.q_proj",
    "model.language_model.layers.29.self_attn.q_proj",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        if layer == 28:
            self.q_proj = _ShapeOnlyLinear(1536, 2048)
        if layer == 29:
            self.q_proj = _ShapeOnlyLinear(1536, 4096)
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


def _source_metadata() -> dict[str, object]:
    path = PROJECT_ROOT / V23_CHECKPOINT / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_v24_pins_sealed_v23_epoch2_and_freezes_complete_prior_stack() -> None:
    config = load_config(V24_CONFIG)
    training = config["training"]
    settings = lora_banks_settings(config)

    assert config_hash(config, length=64) == EXPECTED_RESOLVED_CONFIG_SHA256
    assert training["initialize_from"] == V23_CHECKPOINT.as_posix()
    assert training["initialize_expected_adapter_sha256"] == V23_ADAPTER_SHA256
    assert training["initialize_expected_metadata_sha256"] == V23_METADATA_SHA256
    assert training["initialize_expected_scene_state_sha256"] == SCENE_SHA256
    assert training["initialize_expected_global_scene_residual_state_sha256"] == GLOBAL_SHA256
    assert training["initialize_expected_signed_x_scene_residual_state_sha256"] == (SIGNED_X_SHA256)
    assert training["freeze_scene_adapter"] is True
    assert training["initialize_named_lora_freeze_transition"] is False
    assert training["initialize_named_lora_freeze_and_extend_transition"] is True
    assert training["train_global_scene_residual_only"] is False
    assert training["train_signed_x_scene_residual_only"] is False
    assert training["train_lora_with_frozen_scene_residual_stack"] is True

    by_name = {bank.name: bank for bank in settings.banks}
    assert tuple(by_name) == (
        "inherited_v12",
        "extension_v13",
        "extension_v23_shared_kv",
        "extension_v24_shared_query",
    )
    assert all(not by_name[name].trainable for name in tuple(by_name)[:-1])
    assert by_name["extension_v24_shared_query"].trainable is True
    assert by_name["extension_v23_shared_kv"].initialization_algorithm == "checkpoint_overwrite"
    assert by_name["extension_v23_shared_kv"].expected_initial_state_sha256 == V23_BANK_SHA256


def test_v24_source_artifacts_and_every_frozen_bank_state_are_exactly_pinned() -> None:
    checkpoint = PROJECT_ROOT / V23_CHECKPOINT
    assert _sha256(checkpoint / "adapter.safetensors") == V23_ADAPTER_SHA256
    assert _sha256(checkpoint / "metadata.json") == V23_METADATA_SHA256

    metadata = _source_metadata()
    assert metadata["epoch"] == 2
    assert metadata["optimizer_step"] == 2
    assert metadata["frozen_scene_state_sha256"] == SCENE_SHA256
    assert metadata["global_scene_residual_state_sha256"] == GLOBAL_SHA256
    assert metadata["signed_x_scene_residual_state_sha256"] == SIGNED_X_SHA256
    assert metadata["lora_bank_state_sha256"] == {
        "inherited_v12": "dec768bed654c8c4e16da0318857543ad54d8f5f68f4d24a9a87cd19ec706594",
        "extension_v13": "4eb90fb9b0bea579d14cfcb0f61ebd5b6d566fd600bd3d5e1bfe5177a39e1b34",
        "extension_v23_shared_kv": V23_BANK_SHA256,
    }

    config = load_config(V24_CONFIG)
    frozen = {
        bank.name: bank.expected_initial_state_sha256
        for bank in lora_banks_settings(config).banks
        if not bank.trainable
    }
    assert frozen == metadata["lora_bank_state_sha256"]
    assert config["v24_screen"]["source_archive_summary_sha256"] == V23_ARCHIVE_SHA256


def test_v24_new_bank_is_query_only_deterministic_zero_output_and_36864_parameters() -> None:
    config = load_config(V24_CONFIG)
    settings = lora_banks_settings(config)
    first = install_lora_banks(_ShapeOnlyGemma().requires_grad_(False), settings)
    second = install_lora_banks(_ShapeOnlyGemma().requires_grad_(False), settings)
    assert first is not None and second is not None

    bank = first.bank("extension_v24_shared_query")
    assert bank.settings.adapter.rank == 4
    assert bank.settings.adapter.alpha == 8.0
    assert bank.settings.adapter.dropout == 0.0
    assert bank.settings.adapter.target_modules == V24_TARGETS
    assert all(target.endswith(".q_proj") for target in V24_TARGETS)
    assert bank.installation.parameter_counts == {
        V24_TARGETS[0]: 14_336,
        V24_TARGETS[1]: 22_528,
    }
    layer28, layer29 = bank.installation.adapters
    assert layer28.lora_a.shape == (4, 1536)
    assert layer28.lora_b.shape == (2048, 4)
    assert layer29.lora_a.shape == (4, 1536)
    assert layer29.lora_b.shape == (4096, 4)
    assert bank.installation.parameter_count == 36_864
    assert first.trainable_parameter_count == 36_864
    assert bank.installation.state_sha256() == V24_INITIAL_SHA256
    assert second.bank("extension_v24_shared_query").installation.state_sha256() == (
        V24_INITIAL_SHA256
    )
    assert all(
        torch.count_nonzero(adapter.lora_b).item() == 0 for adapter in bank.installation.adapters
    )
    assert all(
        not parameter.requires_grad
        for frozen_bank in first.banks
        if frozen_bank.settings.name != "extension_v24_shared_query"
        for parameter in frozen_bank.installation.parameters()
    )
    assert {id(parameter) for parameter in first.parameters()} == {
        id(parameter) for parameter in bank.installation.parameters()
    }


def test_v24_uses_fresh_adamw_at_3e4_on_only_the_new_bank() -> None:
    config = load_config(V24_CONFIG)
    training = config["training"]
    settings = lora_banks_settings(config)
    collection = install_lora_banks(_ShapeOnlyGemma().requires_grad_(False), settings)
    assert collection is not None

    optimizer_settings = lora_banks_optimizer_settings(config, settings)
    assert optimizer_settings is not None
    optimizer, parameters = build_adapter_optimizer(
        config,
        [],
        collection,
        optimizer_settings,
    )
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.state == {}
    assert parameters == collection.parameters()
    assert sum(parameter.numel() for parameter in parameters) == 36_864
    assert [group["name"] for group in optimizer.param_groups] == ["language_lora"]
    assert optimizer.param_groups[0]["lr"] == 3e-4
    assert optimizer.param_groups[0]["weight_decay"] == 0.0
    assert training.get("resume_from") is None
    assert training["optimizer"] == {
        "name": "AdamW",
        "learning_rate": 3e-4,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 0.0,
        "foreach": False,
        "fused": False,
        "capturable": False,
        "maximize": False,
        "amsgrad": False,
        "gradient_clip_norm": 1.0,
        "accumulation_divisor": 12,
        "step_index": 1,
    }


def test_v24_explicitly_freezes_v23_bank_while_excluding_only_new_query_bank() -> None:
    config = load_config(V24_CONFIG)
    settings = lora_banks_settings(config)
    collection = install_lora_banks(_ShapeOnlyGemma().requires_grad_(False), settings)
    assert collection is not None
    source_metadata = _source_metadata()

    assert (
        named_lora_freeze_and_extend_transition_mismatch(source_metadata, collection) is None
    )
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
    assert set(checkpoint_modules) - set(source_modules) == {
        "lora_banks.extension_v24_shared_query"
    }
    assert "lora_banks.extension_v23_shared_kv" in source_modules


def test_v24_freeze_and_extend_transition_fails_on_nonprovenance_drift() -> None:
    config = load_config(V24_CONFIG)
    collection = install_lora_banks(
        _ShapeOnlyGemma().requires_grad_(False), lora_banks_settings(config)
    )
    assert collection is not None
    source_metadata = _source_metadata()

    wrong_hash = deepcopy(source_metadata)
    wrong_hash["lora_bank_state_sha256"]["extension_v23_shared_kv"] = "0" * 64
    mismatch = named_lora_freeze_and_extend_transition_mismatch(wrong_hash, collection)
    assert mismatch is not None
    assert "extension_v23_shared_kv.source_state" in mismatch

    wrong_target = deepcopy(source_metadata)
    wrong_target["lora"]["banks"][2]["target_modules"][0] = (
        "model.language_model.layers.12.self_attn.k_proj"
    )
    mismatch = named_lora_freeze_and_extend_transition_mismatch(wrong_target, collection)
    assert mismatch is not None
    assert "extension_v23_shared_kv.architecture" in mismatch

    no_freeze = deepcopy(source_metadata)
    no_freeze["lora"]["banks"][2]["trainable"] = False
    mismatch = named_lora_freeze_and_extend_transition_mismatch(no_freeze, collection)
    assert mismatch is not None
    assert "source_trainable_bank_transition" in mismatch


def test_v24_preserves_v23_objectives_and_has_bounded_question_independent_screen() -> None:
    v23 = load_config(V23_CONFIG)
    config = load_config(V24_CONFIG)
    training = config["training"]
    screen = config["v24_screen"]
    experiment = config["experiment"]

    assert training["pair_objectives"] == v23["training"]["pair_objectives"]
    assert training["pair_only_scene_ids"] == v23["training"]["pair_only_scene_ids"]
    assert training["pair_max_units_per_pair"] == v23["training"]["pair_max_units_per_pair"]
    assert training["max_questions_per_scene"] == v23["training"]["max_questions_per_scene"]
    assert training["epochs"] == screen["screen_optimizer_updates"] == 4
    assert training["pair_steps_per_epoch"] == training["gradient_accumulation"] == 12
    assert screen["conditional_max_optimizer_updates"] == 8
    assert screen["stage_1_optimizer_updates"] == 1
    assert screen["stage_1_stop_required"] is True
    assert config["v23_screen"] is None
    assert experiment["decoder_trainable_parameter_count"] == 36_864
    assert experiment["question_dependent_scene_processing"] is False
    assert experiment["question_dependent_retrieval"] is False
    assert experiment["runtime_oracle_access"] is False
    assert screen["full_teacher_gate_requires"] == v23["v23_screen"]["full_teacher_gate_requires"]
    assert screen["greedy_audit_only_after_full_teacher_gate"] is True
