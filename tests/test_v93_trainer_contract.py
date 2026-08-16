from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.evaluation.v93_scene1_termination_paraphrase_preflight import (
    EXPECTED_INITIAL_STATE_SHA256,
    FRESH_BANK_NAME,
    TARGET_MODULE,
    derive_training_items_v93,
    load_canonical_rows_v93,
    load_config_v93,
)
from semantic_3d_chat.training import (
    train_v93_scene1_termination_paraphrase_repair as trainer,
)


def test_v93_authenticates_exact_failed_v92_parent_without_model() -> None:
    evidence = trainer._authenticate_failed_v92(load_config_v93(allow_draft=True))

    assert evidence["failed_but_authenticated"] is True
    assert evidence["runtime_promotion_authorized"] is False
    assert evidence["canonical_correct"] == 123
    assert evidence["canonical_errors"] == 15
    assert evidence["primary_correct"] == 12
    assert evidence["semantic_primary_failed_intents"] == ["table_contents"]
    assert evidence["primary_strict_failures"] == ["inventory", "table_contents"]
    assert evidence["held_wording_correct"] == 17
    assert evidence["state_sha256"] == trainer.V92_STATE_SHA256
    assert evidence["checkpoint_sha256"] == trainer.V92_CHECKPOINT_SHA256


def test_v93_combined_settings_are_exact_fifteen_bank_surface() -> None:
    config = load_config_v93(allow_draft=True)
    runtime = load_runtime_config(config["sources"]["runtime_config"])
    settings = trainer.combined_lora_settings_v93(runtime, config)

    assert len(settings.banks) == trainer.TOTAL_BANK_COUNT == 15
    assert tuple(bank.name for bank in settings.banks[:-1]) == trainer.EXPECTED_PARENT_BANKS
    assert sum(bank.trainable for bank in settings.banks) == 1
    fresh = settings.banks[-1]
    assert fresh.name == FRESH_BANK_NAME
    assert fresh.trainable is True
    assert fresh.adapter.target_modules == (TARGET_MODULE,)
    assert fresh.adapter.rank == 8
    assert fresh.adapter.alpha == 16.0
    assert fresh.initialization_seed == 930093
    assert fresh.expected_initial_state_sha256 == EXPECTED_INITIAL_STATE_SHA256
    assert TARGET_MODULE not in {
        target for bank in settings.banks[:-1] for target in bank.adapter.target_modules
    }


def test_v93_trainer_schedule_contract_is_exact_1770_rows() -> None:
    config = load_config_v93(allow_draft=True)
    items = derive_training_items_v93(config, load_canonical_rows_v93(config))

    assert Counter(item.kind for item in items) == {
        "canonical": 138,
        "parent_error_replay": 60,
        "parent_correct_anchor": 123,
        "conversational_known": 130,
        "training_paraphrase": 78,
        "conversational_error_replay": 50,
        "support_error_replay": 10,
        "primary_inventory_anchor": 1,
    }
    assert sum(item.causal_margin for item in items) == 13
    assert len(items) == trainer.EXPECTED_ROWS_PER_EPOCH
    assert len(items) * 3 == trainer.EXPECTED_MICRO_ROWS
    assert len(items) * 3 // 6 == trainer.EXPECTED_OPTIMIZER_UPDATES


def test_v93_eos_augmented_objective_requires_terminal_token() -> None:
    mean = torch.tensor(2.0, requires_grad=True)
    tail = SimpleNamespace(
        targets=torch.tensor([11, 12, 1]),
        per_token_nll=torch.tensor([2.0, 1.0, 0.25], requires_grad=True),
        mean_nll=mean,
    )

    objective, eos_nll = trainer.eos_augmented_answer_objective_v93(
        tail,
        eos_token_id=1,
        ce_weight=1.0,
        eos_extra_weight=4.0,
    )
    assert float(objective.detach()) == 3.0
    assert float(eos_nll.detach()) == 0.25
    combined = trainer.add_causal_margin_v93(
        objective,
        torch.tensor(0.5),
        margin_weight=1.0,
    )
    assert float(combined.detach()) == 3.5
    combined.backward()
    assert mean.grad is not None

    tail.targets[-1] = 2
    with pytest.raises(ValueError, match="EOS-terminated"):
        trainer.eos_augmented_answer_objective_v93(
            tail,
            eos_token_id=1,
            ce_weight=1.0,
            eos_extra_weight=4.0,
        )
