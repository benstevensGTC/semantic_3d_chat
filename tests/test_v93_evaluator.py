from __future__ import annotations

from copy import deepcopy

import pytest

from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.evaluation import (
    evaluate_v93_scene1_termination_paraphrase_repair as evaluator,
)
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import (
    canonical_sha256_v85,
)
from semantic_3d_chat.evaluation.v93_scene1_termination_paraphrase_preflight import (
    held_wording_rows_v93,
    load_canonical_rows_v93,
    load_config_v93,
    primary_rows_v93,
)


def _perfect_records() -> tuple[dict, tuple, list, list, list, list]:
    config = load_config_v93()
    canonical = load_canonical_rows_v93(config)
    primary = primary_rows_v93(config)
    held = held_wording_rows_v93(config)
    canonical_records = [
        {
            "question_id": row.question_id,
            "prediction": row.answer,
            "correct_mean_nll": 0.1,
        }
        for row in canonical
    ]
    primary_records = [
        {
            "question_id": row.question_id,
            "answer_type": row.answer_type,
            "reference_answer": row.answer,
            "prediction": row.answer,
        }
        for row in primary
    ]
    held_records = [
        {
            "question_id": row.question_id,
            "answer_type": row.answer_type,
            "reference_answer": row.answer,
            "prediction": row.answer,
        }
        for row in held
    ]
    causal_records = [
        {
            "question_id": row.question_id,
            "answer_type": row.answer_type,
            "correct_prediction": row.answer,
            "zero_prediction": "unknown",
            "correct_mean_nll": 0.1,
            "zero_mean_nll": 1.0,
            "zero_minus_correct_nll": 0.9,
        }
        for row in primary
    ]
    return (
        config,
        canonical,
        canonical_records,
        primary_records,
        held_records,
        causal_records,
    )


def _score(
    *,
    parent_state_invariant: bool = True,
    candidate_state_invariant: bool = True,
    protected_read_count: int = 0,
    primary_failure: bool = False,
    held_failure: bool = False,
) -> dict:
    config, canonical, canonical_records, primary, held, causal = _perfect_records()
    if primary_failure:
        primary[0]["prediction"] = "wrong"
    if held_failure:
        # Both rows belong to inventory.  The per-intent floor must fail even
        # though every other held intent remains correct.
        held[0]["prediction"] = "wrong"
        held[1]["prediction"] = "wrong"
    return evaluator.score_records_v93(
        canonical,
        canonical_records,
        primary,
        held,
        causal,
        gates=config["gates"],
        prefix_hash_invariant=True,
        environment_input_invariant=True,
        parent_state_invariant=parent_state_invariant,
        candidate_state_invariant=candidate_state_invariant,
        protected_read_count=protected_read_count,
    )


def test_v93_evaluator_reuses_exact_sealed_question_inventories() -> None:
    config = load_config_v93()

    assert evaluator.primary_rows_v93(config) == primary_rows_v93(config)
    assert evaluator.held_wording_rows_v93(config) == held_wording_rows_v93(config)
    assert len(evaluator.primary_rows_v93(config)) == 13
    assert len(evaluator.held_wording_rows_v93(config)) == 26
    assert all(
        row.question_id.startswith("v93_") for row in evaluator.held_wording_rows_v93(config)
    )


def test_v93_perfect_records_pass_every_preregistered_gate() -> None:
    score = _score()

    assert score["model_acceptance_gate_passed"] is True
    assert all(score["model_acceptance_gates"].values())
    assert score["canonical_type_specific"]["correct"] == 138
    assert score["primary_conversational"]["correct"] == 13
    assert score["new_held_wording"]["correct"] == 26
    assert score["causal_control"]["canonical_prediction_changes"] == 13
    assert score["runtime_promotion_authorized"] is False


def test_v93_evaluator_fails_closed_on_candidate_mutation() -> None:
    score = _score(candidate_state_invariant=False)

    assert score["model_acceptance_gate_passed"] is False
    assert score["model_acceptance_gates"]["fixed_final_candidate_state_invariance"] is False
    assert score["separate_runtime_packaging_authorized"] is False


def test_v93_evaluator_enforces_each_held_intent_floor() -> None:
    score = _score(held_failure=True)

    assert score["model_acceptance_gate_passed"] is False
    assert score["model_acceptance_gates"]["new_held_wording_each_intent_at_least_minimum"] is False


def test_v93_evaluator_fails_closed_on_parent_mutation() -> None:
    score = _score(parent_state_invariant=False)

    assert score["model_acceptance_gate_passed"] is False
    assert score["model_acceptance_gates"]["frozen_parent_state_invariance"] is False


def test_v93_evaluator_requires_all_thirteen_primary_answers() -> None:
    score = _score(primary_failure=True)

    assert score["model_acceptance_gate_passed"] is False
    assert (
        score["model_acceptance_gates"][
            "primary_conversational_correct_at_least_required"
        ]
        is False
    )


def test_v93_evaluator_fails_on_protected_runtime_read() -> None:
    score = _score(protected_read_count=1)

    assert score["model_acceptance_gate_passed"] is False
    assert (
        score["model_acceptance_gates"][
            "protected_read_count_at_most_preregistered_maximum"
        ]
        is False
    )


def test_v93_evaluator_seals_exact_environment_contract() -> None:
    assert evaluator.EXPECTED_PREFIX_SHAPE == (1, 738, 1536)
    assert evaluator.EXPECTED_EVALUATED_QUESTION_COUNT == 177
    assert evaluator.EXPECTED_CAUSAL_CONTROL_COUNT == 13
    assert evaluator.EXPECTED_FROZEN_PARENT_BANK_COUNT == 14


def test_v93_evaluator_binds_the_termination_prompt_not_v89_baseline() -> None:
    config = load_config_v93()
    runtime = load_runtime_config(config["sources"]["runtime_config"])

    contract = evaluator.prompt_contract_v93(config, runtime["language"])

    assert contract["system_prompt_sha256"] == canonical_sha256_v85(
        config["system_prompt"]
    )
    assert contract["max_answer_tokens"] == 32
    assert contract["differs_from_v89_runtime_baseline"] is True
    assert config["system_prompt"] != runtime["language"]["system_prompt"]


@pytest.mark.parametrize("mutation", ("prompt", "cap", "model"))
def test_v93_evaluator_prompt_contract_fails_closed(mutation: str) -> None:
    config = load_config_v93()
    runtime = load_runtime_config(config["sources"]["runtime_config"])
    changed_config = deepcopy(config)
    changed_runtime = deepcopy(runtime["language"])
    if mutation == "prompt":
        changed_config["system_prompt"] = changed_runtime["system_prompt"]
    elif mutation == "cap":
        changed_config["max_answer_tokens"] = 31
    else:
        changed_runtime["revision"] = "wrong-revision"

    with pytest.raises(ValueError, match="prompt, generation cap, or model identity"):
        evaluator.prompt_contract_v93(changed_config, changed_runtime)
