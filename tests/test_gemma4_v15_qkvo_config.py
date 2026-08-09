from __future__ import annotations

import torch
from torch import nn

from semantic_3d_chat.config import load_config
from semantic_3d_chat.language.lora import install_lora_banks, lora_banks_settings


V15_CONFIG = "configs/experiments/gemma4_color_mirror_decoder_qkvo_v15.yaml"
V12_ADAPTER_SHA256 = "a4c85c14a214e4e594992e489a784cb4bacb64d3dfda519ad3da18b1595d9f22"
V12_METADATA_SHA256 = "f097c6477546460440e77a3d225afb55818cb13abf9cbb4a90500f75a879b0f5"
V12_SCENE_SHA256 = "690bd890bfda024dbb5c7d3c68087b8113bc3b8ee81dd6143c7eb2a884e7245b"
V12_LORA_SHA256 = "dec768bed654c8c4e16da0318857543ad54d8f5f68f4d24a9a87cd19ec706594"
V15_EXTENSION_SHA256 = "7ad3cd4296e78c6cb3daeae6bbce762b0f5399eeb85f01b6ffcb47233dc3d814"

INHERITED_TARGETS = (
    "model.language_model.layers.34.self_attn.q_proj",
    "model.language_model.layers.34.self_attn.o_proj",
)
EXTENSION_TARGETS = (
    "model.language_model.layers.13.self_attn.k_proj",
    "model.language_model.layers.13.self_attn.v_proj",
    "model.language_model.layers.14.self_attn.k_proj",
    "model.language_model.layers.14.self_attn.v_proj",
    *(
        f"model.language_model.layers.{layer}.self_attn.{projection}"
        for layer in range(30, 34)
        for projection in ("q_proj", "o_proj")
    ),
)


class _ShapeOnlyLinear(nn.Linear):
    """An ``nn.Linear`` with exact Gemma shapes and one-element base storage."""

    def __init__(self, in_features: int, out_features: int) -> None:
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.zeros(1, dtype=torch.float32).expand(out_features, in_features),
            requires_grad=False,
        )
        self.bias = None


class _TargetAttention(nn.Module):
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


class _TargetLayer(nn.Module):
    def __init__(self, layer: int) -> None:
        super().__init__()
        self.self_attn = _TargetAttention(layer)


class _ExactShapeToyGemma(nn.Module):
    """Only V15 target projections exist; all base matrices share scalar storage."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList(
            _TargetLayer(layer) for layer in range(35)
        )


def test_v15_qkvo_config_pins_v12_source_fixed_prefix_and_screen_gate() -> None:
    config = load_config(V15_CONFIG)
    training = config["training"]
    experiment = config["experiment"]

    assert training["initialize_from"] == (
        "data_gemma4/checkpoints/gemma4_color_mirror_spatial_relation_v12/epoch_008"
    )
    assert training["initialize_expected_adapter_sha256"] == V12_ADAPTER_SHA256
    assert training["initialize_expected_metadata_sha256"] == V12_METADATA_SHA256
    assert training["initialize_legacy_lora_into_bank"] == "inherited_v12"
    assert training["freeze_scene_adapter"] is True
    assert experiment["source_scene_state_sha256"] == V12_SCENE_SHA256
    assert experiment["source_inherited_bank_sha256"] == V12_LORA_SHA256
    assert experiment["question_dependent_scene_processing"] is False
    assert "dense_relation_token" not in training
    assert "control_token" not in training
    assert "retrieval" not in training

    assert training["pair_only_scene_ids"] == [
        "scene_000003",
        "scene_000004",
        "scene_000007",
        "scene_000008",
    ]
    assert training["pair_max_units_per_pair"] == 6
    assert training["max_questions_per_scene"] == 6
    assert training["pair_steps_per_epoch"] == training["gradient_accumulation"] == 12
    assert training["epochs"] == experiment["screen_optimizer_updates"] == 4
    assert training["pair_gate_every_epochs"] == 1
    assert training["pair_gate_stop_when_passed"] is False
    assert training["early_stopping_patience"] == 0
    assert training["pair_ranking_mode"] == "candidate_logit"
    assert training["pair_ranking_weight"] == 8.0
    assert training["pair_full_vocab_ranking_weight"] == 2.0
    assert training["pair_full_vocab_ranking_margin"] == 1.0
    assert training["pair_gate_first_answer_token_top1_accuracy"] == 1.0
    assert training["lora_learning_rate"] == 0.002
    assert training["lora_weight_decay"] == 0.0
    for frozen_objective in (
        "latent_diversity_weight",
        "paired_scene_separation_weight",
        "grounding_weight",
        "grounding_anchor_weight",
        "spatial_answer_contrastive_weight",
        "spatial_answer_warmup_steps",
        "spatial_relation_contrastive_weight",
        "spatial_relation_warmup_steps",
    ):
        assert training[frozen_objective] == 0
    assert experiment["screen_extension_requires"] == {
        "color_full_vocab_sides": 12,
        "color_full_vocab_units": 6,
        "color_positive_minimum_candidate_margin": True,
        "color_positive_minimum_full_vocab_margin": True,
        "mirror_minimum_full_vocab_sides": 8,
        "mirror_minimum_full_vocab_units": 2,
    }
    assert experiment["full_teacher_gate_requires"] == {
        "color_full_vocab_sides": 12,
        "color_full_vocab_units": 6,
        "mirror_full_vocab_sides": 12,
        "mirror_full_vocab_units": 6,
        "all_candidate_and_full_vocab_minimum_margins_positive": True,
    }
    assert experiment["greedy_audit_only_after_full_teacher_gate"] is True


def test_v15_qkvo_targets_only_real_shared_kv_and_predeclared_qo_layers() -> None:
    settings = lora_banks_settings(load_config(V15_CONFIG))
    inherited = settings.bank("inherited_v12")
    extension = settings.bank("extension_v15_qkvo")

    assert inherited.adapter.target_modules == INHERITED_TARGETS
    assert inherited.expected_initial_state_sha256 == V12_LORA_SHA256
    assert extension.adapter.target_modules == EXTENSION_TARGETS
    assert extension.initialization_seed == 15015
    assert extension.expected_initial_state_sha256 == V15_EXTENSION_SHA256
    assert len(extension.adapter.target_modules) == 12
    assert not any(
        int(target.split(".")[3]) >= 15 and target.endswith(("k_proj", "v_proj"))
        for target in extension.adapter.target_modules
    )

    toy = _ExactShapeToyGemma()
    for layer in range(15, 35):
        attention = toy.model.language_model.layers[layer].self_attn
        assert not hasattr(attention, "k_proj")
        assert not hasattr(attention, "v_proj")


def test_v15_qkvo_exact_shape_installation_is_zero_deterministic_and_isolated() -> None:
    settings = lora_banks_settings(load_config(V15_CONFIG))
    first = install_lora_banks(_ExactShapeToyGemma().requires_grad_(False), settings)
    second = install_lora_banks(_ExactShapeToyGemma().requires_grad_(False), settings)
    assert first is not None and second is not None

    inherited = first.bank("inherited_v12")
    extension = first.bank("extension_v15_qkvo")
    assert inherited.settings.trainable is False
    assert extension.settings.trainable is True
    assert inherited.installation.parameter_count == 45_056
    assert extension.installation.parameter_count == 290_816
    assert extension.installation.state_sha256() == V15_EXTENSION_SHA256
    assert second.bank("extension_v15_qkvo").installation.state_sha256() == (
        V15_EXTENSION_SHA256
    )

    assert all(
        torch.count_nonzero(adapter.lora_b).item() == 0
        for bank in first.banks
        for adapter in bank.installation.adapters
    )
    assert all(
        not parameter.requires_grad for parameter in inherited.installation.parameters()
    )
    assert all(parameter.requires_grad for parameter in extension.installation.parameters())
    assert {id(parameter) for parameter in first.parameters()} == {
        id(parameter) for parameter in extension.installation.parameters()
    }
    assert first.trainable_parameter_count == 290_816
    assert first.parameter_count == 335_872
