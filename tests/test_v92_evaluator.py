from __future__ import annotations

from semantic_3d_chat.evaluation import (
    evaluate_v92_scene1_retention_conversation_repair as evaluator,
)
from semantic_3d_chat.evaluation.v92_scene1_retention_conversation_preflight import (
    held_wording_rows_v92,
    load_canonical_rows_v92,
    load_config_v92,
    primary_rows_v92,
)


def _perfect_records() -> tuple[dict, tuple, list, list, list, list]:
    config = load_config_v92()
    canonical = load_canonical_rows_v92(config)
    primary = primary_rows_v92(config)
    held = held_wording_rows_v92(config)
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


def _score(*, candidate_state_invariant: bool = True, held_failure: bool = False) -> dict:
    config, canonical, canonical_records, primary, held, causal = _perfect_records()
    if held_failure:
        # Both rows belong to inventory.  The per-intent floor must fail even
        # though every other held intent remains correct.
        held[0]["prediction"] = "wrong"
        held[1]["prediction"] = "wrong"
    return evaluator.score_records_v92(
        canonical,
        canonical_records,
        primary,
        held,
        causal,
        gates=config["gates"],
        prefix_hash_invariant=True,
        environment_input_invariant=True,
        parent_state_invariant=True,
        candidate_state_invariant=candidate_state_invariant,
        protected_read_count=0,
    )


def test_v92_evaluator_reuses_exact_sealed_question_inventories() -> None:
    config = load_config_v92()

    assert evaluator.primary_rows_v92(config) == primary_rows_v92(config)
    assert evaluator.held_wording_rows_v92(config) == held_wording_rows_v92(config)
    assert len(evaluator.primary_rows_v92(config)) == 13
    assert len(evaluator.held_wording_rows_v92(config)) == 26
    assert all(
        row.question_id.startswith("v92_") for row in evaluator.held_wording_rows_v92(config)
    )


def test_v92_perfect_records_pass_every_preregistered_gate() -> None:
    score = _score()

    assert score["model_acceptance_gate_passed"] is True
    assert all(score["model_acceptance_gates"].values())
    assert score["canonical_type_specific"]["correct"] == 138
    assert score["primary_conversational"]["correct"] == 13
    assert score["new_held_wording"]["correct"] == 26
    assert score["causal_control"]["canonical_prediction_changes"] == 13
    assert score["runtime_promotion_authorized"] is False


def test_v92_evaluator_fails_closed_on_candidate_mutation() -> None:
    score = _score(candidate_state_invariant=False)

    assert score["model_acceptance_gate_passed"] is False
    assert score["model_acceptance_gates"]["fixed_final_candidate_state_invariance"] is False
    assert score["separate_runtime_packaging_authorized"] is False


def test_v92_evaluator_enforces_each_held_intent_floor() -> None:
    score = _score(held_failure=True)

    assert score["model_acceptance_gate_passed"] is False
    assert score["model_acceptance_gates"]["new_held_wording_each_intent_at_least_minimum"] is False
