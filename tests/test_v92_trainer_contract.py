from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import torch
from torch import nn

from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.evaluation.v92_scene1_retention_conversation_preflight import (
    FRESH_BANK_NAME,
    TARGET_MODULE,
    derive_training_items_v92,
    load_canonical_rows_v92,
    load_config_v92,
)
from semantic_3d_chat.language.lora import (
    LoRABankSettings,
    LoRABanksSettings,
    LoRASettings,
    install_lora_banks,
)
from semantic_3d_chat.training import (
    train_v92_scene1_retention_conversation_repair as trainer,
)


def _synthetic_fresh_collection() -> tuple[nn.Module, object]:
    root = nn.Module()
    root.model = nn.Module()
    root.model.language_model = nn.Module()
    root.model.language_model.layers = nn.ModuleList(nn.Module() for _ in range(30))
    layer = root.model.language_model.layers[29]
    layer.self_attn = nn.Module()
    layer.self_attn.o_proj = nn.Linear(
        4_096,
        1_536,
        bias=False,
        dtype=torch.bfloat16,
    )
    settings = LoRABanksSettings(
        (
            LoRABankSettings(
                name=FRESH_BANK_NAME,
                trainable=True,
                adapter=LoRASettings(
                    enabled=True,
                    rank=8,
                    alpha=16.0,
                    dropout=0.0,
                    target_modules=(TARGET_MODULE,),
                ),
                initialization_algorithm="cpu_kaiming_uniform_a_exact_zero_b",
                initialization_seed=920092,
                expected_initial_state_sha256=(
                    "c10d38c727df1520418a5bb9be7bac262a2b6acdef07203a669ff52f8cd08cc1"
                ),
            ),
        )
    )
    collection = install_lora_banks(root, settings)
    assert collection is not None
    return root, collection


def test_v92_trainer_authenticates_exact_v91_failure_without_model() -> None:
    config = load_config_v92(allow_draft=True)
    evidence = trainer._authenticate_failed_v91(config)

    assert evidence["failed_but_authenticated"] is True
    assert evidence["runtime_promotion_authorized"] is False
    assert evidence["canonical_correct"] == 115
    assert evidence["canonical_errors"] == 23
    assert evidence["primary_correct"] == 10
    assert evidence["primary_failures"] == [
        "inventory",
        "bowl_left_chair",
        "table_contents",
    ]
    assert evidence["held_wording_correct"] == 23
    assert evidence["state_sha256"] == trainer.V91_STATE_SHA256
    assert evidence["checkpoint_sha256"] == trainer.V91_CHECKPOINT_SHA256


def test_v92_combined_settings_have_one_trainable_disjoint_bank() -> None:
    config = load_config_v92(allow_draft=True)
    runtime = load_runtime_config(config["sources"]["runtime_config"])
    settings = trainer.combined_lora_settings_v92(runtime, config)

    assert len(settings.banks) == 14
    assert sum(bank.trainable for bank in settings.banks) == 1
    assert tuple(bank.name for bank in settings.banks[:-1]) == trainer.EXPECTED_PARENT_BANKS
    fresh = settings.banks[-1]
    assert fresh.name == FRESH_BANK_NAME
    assert fresh.trainable is True
    assert fresh.adapter.target_modules == (TARGET_MODULE,)
    assert fresh.adapter.rank == 8
    assert fresh.adapter.alpha == 16.0
    assert TARGET_MODULE not in {
        target
        for bank in settings.banks[:-1]
        for target in bank.adapter.target_modules
    }


def test_v92_trainer_schedule_kind_contract_matches_1770_rows() -> None:
    config = load_config_v92(allow_draft=True)
    canonical = load_canonical_rows_v92(config)
    items = derive_training_items_v92(config, canonical)

    assert Counter(item.kind for item in items) == {
        "canonical": 138,
        "error_replay": 115,
        "correct_anchor_replay": 115,
        "conversational_known": 104,
        "conversational_error_replay": 60,
        "primary_failed_intent_replay": 48,
        "primary_success_anchor": 10,
    }
    assert sum(item.causal_margin for item in items) == 13
    assert len(items) == trainer.EXPECTED_ROWS_PER_EPOCH
    assert len(items) * 3 == trainer.EXPECTED_MICRO_ROWS
    assert len(items) * 3 // 6 == trainer.EXPECTED_OPTIMIZER_UPDATES


def test_v92_candidate_is_create_once_two_tensor_sanitized_artifact(
    tmp_path: Path,
) -> None:
    config = load_config_v92(allow_draft=True)
    _root, collection = _synthetic_fresh_collection()
    candidate = tmp_path / "candidate"
    metadata = trainer.publish_fixed_final_candidate_v92(
        candidate,
        collection,
        bindings={"synthetic_test": True},
        experiment=config,
    )

    assert {path.name for path in candidate.iterdir()} == {
        trainer.WEIGHTS_FILENAME,
        trainer.METADATA_FILENAME,
    }
    assert metadata["parameter_count"] == 45_056
    assert metadata["frozen_parent_parameter_count"] == 1_122_304
    assert metadata["total_adapter_parameter_count"] == 1_167_360
    assert metadata["environmental_memory_serialized"] is False
    assert metadata["environmental_text_serialized"] is False
    assert metadata["questions_or_answers_serialized"] is False
    assert metadata["oracle_serialized"] is False
    assert metadata["runtime_promotion_authorized"] is False

    authenticated, _fingerprint, files = (
        trainer._authenticate_fixed_final_candidate_v92(candidate)
    )
    assert authenticated == metadata
    assert [row["path"] for row in files] == [
        trainer.WEIGHTS_FILENAME,
        trainer.METADATA_FILENAME,
    ]

    _other_root, other = _synthetic_fresh_collection()
    loaded = trainer.load_fixed_final_bridge_v92(other, candidate)
    assert loaded["state_sha256"] == metadata["state_sha256"]
    assert (
        other.bank(FRESH_BANK_NAME).installation.state_sha256()
        == metadata["state_sha256"]
    )
    with pytest.raises(FileExistsError):
        trainer.publish_fixed_final_candidate_v92(
            candidate,
            other,
            bindings={"synthetic_test": True},
            experiment=config,
        )
