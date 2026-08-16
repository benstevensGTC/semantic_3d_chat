from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.evaluation.evaluate_v85_strict_multiscene import (
    score_records_v85,
)
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import (
    FRESH_BANK_NAME,
    authenticate_cpu_preflight_v85,
    canonical_sha256_v85,
    load_config_v85,
    ordered_training_rows_v85,
    split_preflight_v85,
)
from semantic_3d_chat.language.lora import (
    LoRABankCollection,
    LoRABankSettings,
    LoRABanksSettings,
    LoRASettings,
    install_lora_banks,
)
from semantic_3d_chat.training.train_v85_strict_multiscene import (
    combined_lora_settings_v85,
    discover_resume_checkpoint_v85,
    pair_margin_objective_v85,
    restore_resume_checkpoint_v85,
    save_resume_checkpoint_v85,
)


def test_v85_preregisters_exact_full_train_and_disjoint_development() -> None:
    config = load_config_v85()
    split, train, development = split_preflight_v85(config)
    schedule = ordered_training_rows_v85(train, seed=850085)

    assert len(train) == 576
    assert len({row.scene_id for row in train}) == 24
    assert len(development) == 384
    assert len({row.scene_id for row in development}) == 16
    assert split["train_changed_units"] == 40
    assert split["train_changed_sides"] == 80
    assert split["development_changed_units"] == 26
    assert split["pair_and_scene_disjoint"] is True
    assert canonical_sha256_v85(
        [[row.scene_id, row.question_id] for row in schedule]
    ) == config["training"]["row_order_sha256"]
    assert config["training"]["optimizer_updates"] == 72
    assert config["training"]["checkpoint_selection"] == "fixed_final_update_72"
    assert config["training"]["development_driven_checkpoint_selection"] is False


def test_v85_retains_v84_memory_and_adds_only_one_fresh_bank() -> None:
    config = load_config_v85()
    runtime = load_runtime_config(config["sources"]["runtime_config"])
    settings = combined_lora_settings_v85(runtime, config)

    assert len(settings.banks) == 7
    assert sum(bank.trainable for bank in settings.banks) == 1
    fresh = settings.bank(FRESH_BANK_NAME)
    assert fresh.adapter.target_modules == (
        "model.language_model.layers.34.mlp.down_proj",
    )
    assert fresh.adapter.rank == 4
    assert fresh.initialization_seed == 840084
    assert config["strict_input_contract"]["shape_per_scene"] == [1, 738, 1536]
    assert config["strict_input_contract"]["question_derived_environmental_tokens"] == 0
    assert config["bridge"]["starts_from_v84_candidate"] is False


def test_v85_pair_margin_rewards_correct_scene_separation() -> None:
    correct = torch.tensor(2.0, requires_grad=True)
    wrong = torch.tensor(1.8, requires_grad=True)

    objective, observed, penalty = pair_margin_objective_v85(
        correct,
        wrong,
        target_margin=0.5,
        ce_weight=1.0,
        margin_weight=1.0,
    )
    objective.backward()

    assert observed.item() == pytest.approx(-0.2)
    assert penalty.item() == pytest.approx(0.7)
    assert correct.grad is not None and correct.grad.item() == pytest.approx(2.0)
    assert wrong.grad is not None and wrong.grad.item() == pytest.approx(-1.0)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Module()
        self.block.linear = nn.Linear(5, 3, bias=False)


def _tiny_collection(seed: int = 5) -> LoRABankCollection:
    settings = LoRABanksSettings(
        (
            LoRABankSettings(
                name=FRESH_BANK_NAME,
                trainable=True,
                adapter=LoRASettings(
                    enabled=True,
                    rank=2,
                    alpha=4.0,
                    dropout=0.0,
                    target_modules=("block.linear",),
                ),
                initialization_algorithm="cpu_kaiming_uniform_a_exact_zero_b",
                initialization_seed=seed,
            ),
        )
    )
    installed = install_lora_banks(_TinyModel(), settings)
    assert isinstance(installed, LoRABankCollection)
    return installed


def test_v85_safetensors_resume_restores_bridge_and_adamw(tmp_path: Path) -> None:
    first = _tiny_collection()
    optimizer = torch.optim.AdamW(first.parameters(), lr=0.001)
    loss = sum(parameter.square().mean() for parameter in first.parameters())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    expected_hash = first.bank(FRESH_BANK_NAME).installation.state_sha256()
    bindings = {"config_sha256": "a" * 64}

    checkpoint = save_resume_checkpoint_v85(
        tmp_path,
        first,
        optimizer,
        update=1,
        row_cursor=8,
        history=[{"update": 1}],
        bindings=bindings,
        row_order_sha256="b" * 64,
    )
    discovered = discover_resume_checkpoint_v85(
        tmp_path,
        bindings=bindings,
        row_order_sha256="b" * 64,
        gradient_accumulation_rows=8,
    )
    assert discovered is not None and discovered[0] == checkpoint

    second = _tiny_collection()
    second_optimizer = torch.optim.AdamW(second.parameters(), lr=0.001)
    restore_resume_checkpoint_v85(
        discovered[0], discovered[1], second, second_optimizer
    )
    assert second.bank(FRESH_BANK_NAME).installation.state_sha256() == expected_hash
    assert second_optimizer.state_dict()["state"]


def test_v85_preregistered_runtime_candidate_gates_are_not_weak() -> None:
    gates = load_config_v85()["runtime_candidate_gates"]

    assert gates["canonical_accuracy_minimum"] == 0.40
    assert gates["canonical_accuracy_margin_over_answer_frequency_majority"] == 0.05
    assert gates["spatial_relation_accuracy_minimum"] == 0.45
    assert gates["complete_changed_units_minimum"] == 4
    assert gates["canonical_prediction_changing_units_minimum"] == 8
    assert gates["protected_read_count_maximum"] == 0
    assert gates["automatic_runtime_promotion"] is False


def test_v85_model_free_scoring_passes_perfect_fixed_predictions() -> None:
    config = load_config_v85()
    _split, _train, development = split_preflight_v85(config)
    opposite = {}
    for row in development:
        opposite[(row.pair_id, row.question_key, row.scene_id)] = row
    records = []
    for row in development:
        paired = opposite[(row.pair_id, row.question_key, row.paired_scene_id)]
        records.append(
            {
                "scene_id": row.scene_id,
                "question_id": row.question_id,
                "correct_scene_prediction": row.answer,
                "normalized_prediction": row.answer,
                "paired_wrong_scene_prediction": paired.answer
                if row.expected_change
                else None,
                "correct_scene_mean_nll": 1.0,
                "wrong_minus_correct_nll": 0.5 if row.expected_change else 0.0,
            }
        )

    score = score_records_v85(
        development,
        records,
        gates=config["runtime_candidate_gates"],
        prefix_hash_invariant=True,
        every_memory_hash_retained=True,
        protected_read_count=0,
    )

    assert score["canonical_type_specific"]["accuracy"] == 1.0
    assert score["changed_complete_units"] == 26
    assert score["canonical_prediction_changing_units"] == 26
    assert score["runtime_candidate_gate_passed"] is True
    assert score["separate_leakage_runtime_packaging_authorized"] is True
    assert score["automatic_runtime_promotion"] is False


def test_v85_create_once_cpu_preflight_is_sealed() -> None:
    config = load_config_v85()
    path = Path(config["outputs"]["cpu_preflight"])
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    if not path.is_file():
        pytest.skip("V85 create-once preflight has not been sealed yet")
    bindings = authenticate_cpu_preflight_v85(config)
    assert len(bindings["config_sha256"]) == 64
    assert len(bindings["preregistration_sha256"]) == 64
    assert len(bindings["cpu_preflight_sha256"]) == 64
