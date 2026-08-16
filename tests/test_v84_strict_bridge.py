from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.evaluation.v84_strict_bridge_preflight import load_config_v84
from semantic_3d_chat.training.train_v84_pair_margin_followup import (
    authenticate_pair_margin_sources_v84,
    canonical_support_relation_v84,
    load_pair_margin_config_v84,
    pair_margin_objective_v84,
    select_pair_margin_rows_v84,
)
from semantic_3d_chat.training.train_v84_strict_bridge import (
    FRESH_BANK_NAME,
    audit_training_layout_v84,
    authenticate_preflight_v84,
    combined_lora_settings_v84,
    select_wiring_rows_v84,
)

ROOT = Path(__file__).resolve().parents[1]


def test_v84_preflight_is_bound_to_zero_control_fixed_memory() -> None:
    config = load_config_v84()
    bindings = authenticate_preflight_v84(config)

    assert config["strict_input_contract"]["shape_per_scene"] == [1, 738, 1536]
    assert config["strict_input_contract"]["question_derived_environmental_tokens"] == 0
    assert config["strict_input_contract"]["question_conditioned_environmental_readout"] is False
    assert config["strict_input_contract"]["question_dependent_retrieval"] is False
    assert len(bindings["config_sha256"]) == 64


def test_v84_adds_one_disjoint_trainable_bank_to_six_frozen_banks() -> None:
    config = load_config_v84()
    runtime = load_runtime_config(config["sources"]["runtime_config"])
    settings = combined_lora_settings_v84(runtime, config)

    assert len(settings.banks) == 7
    assert sum(bank.trainable for bank in settings.banks) == 1
    fresh = settings.bank(FRESH_BANK_NAME)
    assert fresh.adapter.target_modules == (
        "model.language_model.layers.34.mlp.down_proj",
    )
    assert fresh.initialization_seed == 840084


def test_v84_fixed_wiring_unit_is_same_question_different_answer() -> None:
    left, right = select_wiring_rows_v84(load_config_v84())

    assert (left.scene_id, left.question_id, left.answer) == (
        "scene_000019",
        "q_000130",
        "on",
    )
    assert (right.scene_id, right.question_id, right.answer) == (
        "scene_000020",
        "q_000001",
        "under",
    )
    assert left.question == right.question
    assert left.answer != right.answer


def test_v84_training_layout_retains_all_memory_and_answer_only_labels() -> None:
    memory = torch.zeros((1, 738, 1536), dtype=torch.bfloat16)
    prompt = torch.tensor([[2, 10, 11]], dtype=torch.long)
    answer = torch.tensor([[20, 1]], dtype=torch.long)
    total = 738 + prompt.shape[1] + answer.shape[1]
    embeddings = torch.zeros((1, total, 1536), dtype=torch.bfloat16)
    embeddings[:, 1:739] = memory
    labels = torch.full((1, total), -100, dtype=torch.long)
    labels[:, -2:] = answer
    modality = torch.zeros((1, total), dtype=torch.long)
    modality[:, 2:738] = 1
    prepared = SimpleNamespace(
        inputs_embeds=embeddings,
        scene_prefix_length=738,
        labels=labels,
        mm_token_type_ids=modality,
        attention_mask=torch.ones((1, total), dtype=torch.long),
    )

    audit = audit_training_layout_v84(
        memory=memory,
        prompt_ids=prompt,
        answer_ids=answer,
        prepared=prepared,
    )

    assert audit["memory_tokens"] == 738
    assert audit["memory_supplied_directly"] is True
    assert audit["answer_only_supervision"] is True
    assert audit["control_tokens"] == 0


def test_v84_pair_margin_followup_is_fixed_train_only_protocol() -> None:
    config = load_pair_margin_config_v84()
    sources = authenticate_pair_margin_sources_v84(config)
    left, right = select_pair_margin_rows_v84(config)

    assert config["training"]["optimizer_updates"] == 32
    assert config["training"]["wrong_scene_target_margin_nll"] == 0.5
    assert config["scope"]["historical_pair_scene_disjoint_development_scored"] is False
    assert config["scope"]["oracle_loaded"] is False
    assert [left.answer, right.answer] == ["on", "under"]
    assert len(sources) == 9


def test_v84_pair_margin_objective_rewards_correct_scene_separation() -> None:
    correct = torch.tensor(2.0, requires_grad=True)
    wrong = torch.tensor(1.8, requires_grad=True)

    objective, observed, penalty = pair_margin_objective_v84(
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


def test_v84_support_relation_canonicalizer_scores_verbose_greedy_answers() -> None:
    assert canonical_support_relation_v84("under the table") == "under"
    assert canonical_support_relation_v84("It is on the table.") == "on"
    assert canonical_support_relation_v84("unknown") == "unknown"


def test_v84_pair_margin_result_is_honest_and_not_promoted() -> None:
    path = (
        ROOT
        / "reports/gemma4/metrics/gemma4_v84_strict_bridge_pair_margin_wiring.json"
    )
    if not path.is_file():
        pytest.skip("Local create-once V84 pair-margin result is unavailable")
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["status"] == "passed_behavioral_wiring_gate"
    assert report["optimizer_updates"] == 32
    assert report["passed"] is True
    assert all(report["gates"].values())
    assert [row["canonical_greedy_prediction"] for row in report["final_rows"]] == [
        "on",
        "under",
    ]
    assert all(row["wrong_minus_correct_nll"] > 0 for row in report["final_rows"])
    assert report["scene_memories"]["hash_invariant"] is True
    assert report["development_behavior_scored"] is False
    assert report["official_validation_loaded"] is False
    assert report["oracle_loaded"] is False
    assert report["runtime_promotion_authorized"] is False
