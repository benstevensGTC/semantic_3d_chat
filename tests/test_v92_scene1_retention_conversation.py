from __future__ import annotations

from collections import Counter

from semantic_3d_chat.evaluation import (
    evaluate_v92_scene1_retention_conversation_repair as v92_evaluator,
)
from semantic_3d_chat.evaluation.v92_scene1_retention_conversation_preflight import (
    CONFIG,
    CONVERSATIONAL_ERROR_IDS,
    EXPECTED_INITIAL_STATE_SHA256,
    PRIMARY_FAILED_INTENTS,
    TARGET_MODULE,
    authenticate_parent_v92,
    authenticate_pinned_model_tensor_v92,
    authenticate_sources_v92,
    conversational_errors_v92,
    derive_contract_v92,
    derive_training_items_v92,
    held_wording_rows_v92,
    known_wording_rows_v92,
    load_canonical_rows_v92,
    load_config_v92,
    lora_preflight_v92,
    memory_preflight_v92,
    primary_rows_v92,
    protocol_v92,
    schedule_v92,
)


def _protocol() -> tuple[dict, tuple, tuple]:
    config = load_config_v92()
    canonical = load_canonical_rows_v92(config)
    return config, canonical, derive_training_items_v92(config, canonical)


def test_v92_uses_all_104_known_wordings_and_holds_26_new_rows() -> None:
    config = load_config_v92()
    known = known_wording_rows_v92(config)
    primary = primary_rows_v92(config)
    held = held_wording_rows_v92(config)

    assert len(known) == 104
    assert len(primary) == 13
    assert len(held) == 26
    assert set(Counter(row.pair_id for row in known).values()) == {8}
    assert set(Counter(row.pair_id for row in held).values()) == {2}
    assert {row.question_id for row in held}.isdisjoint(
        row.question_id for row in known
    )
    assert all(row.question_id.startswith("v91_") for row in known)
    assert all(row.question_id.startswith("v92_") for row in held)
    assert v92_evaluator.primary_rows_v92(config) == primary
    assert v92_evaluator.held_wording_rows_v92(config) == held


def test_v92_authenticates_exact_failed_v91_parent_measurements() -> None:
    config = load_config_v92()
    parent = authenticate_parent_v92(config)

    assert parent["total_frozen_bank_count"] == 13
    assert parent["total_frozen_parameter_count"] == 1_122_304
    assert parent["v91_model_acceptance_gate_passed"] is False
    assert parent["v91_runtime_promotion_authorized"] is False
    assert parent["canonical_correct"] == 115
    assert parent["canonical_errors"] == 23
    assert parent["primary_correct"] == 10
    assert parent["held_wording_correct"] == 23
    assert tuple(parent["conversational_error_question_ids"]) == (
        CONVERSATIONAL_ERROR_IDS
    )


def test_v92_exact_six_conversation_errors_and_three_failed_primary_intents() -> None:
    config = load_config_v92()
    errors = conversational_errors_v92(config)

    assert tuple(row.question_id for row in errors) == CONVERSATIONAL_ERROR_IDS
    primary_failures = {
        row.pair_id.removeprefix("v91_conversation_")
        for row in errors
        if row.question_id.endswith("existing_00")
    }
    assert primary_failures == set(PRIMARY_FAILED_INTENTS)


def test_v92_exact_590_row_retention_repair_inventory() -> None:
    _config, _canonical, items = _protocol()

    assert len(items) == 590
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
    assert {
        item.source_question_id
        for item in items
        if item.kind == "conversational_error_replay"
    } == set(CONVERSATIONAL_ERROR_IDS)


def test_v92_schedule_and_derived_hashes_are_deterministic() -> None:
    config, _canonical, items = _protocol()
    schedule = schedule_v92(items)
    contract = derive_contract_v92()

    assert len(schedule) == 1770
    assert Counter(epoch for epoch, _item in schedule) == {0: 590, 1: 590, 2: 590}
    assert set(Counter(item.schedule_id for _epoch, item in schedule).values()) == {3}
    assert sum(item.causal_margin for _epoch, item in schedule) == 39
    assert len(schedule) // config["training"]["gradient_accumulation_rows"] == 295
    assert contract["training_inventory_sha256"] == (
        "86f4b744c7b7fcc15b81a42952ca0a579da7a8d65d6458b2e7a5a5ede9bef726"
    )
    assert contract["training_schedule_sha256"] == (
        "e54e6fe528afbee7fcf9548a3b90d247292135173e26a1becf144ba90d7f34b7"
    )
    assert all(
        protocol_v92(config, config_path=CONFIG, require_sealed_hashes=False)[
            "checks"
        ].values()
    )


def test_v92_fresh_bank_is_exact_zero_rank8_layer29_projection() -> None:
    config = load_config_v92()
    result = lora_preflight_v92(config)

    assert result["target_modules"] == [TARGET_MODULE]
    assert result["base_projection_weight_shape"] == [1536, 4096]
    assert result["parameter_count"] == 45056
    assert result["lora_a_shape"] == [8, 4096]
    assert result["lora_b_shape"] == [1536, 8]
    assert result["lora_b_nonzero_count"] == 0
    assert result["initial_state_sha256"] == EXPECTED_INITIAL_STATE_SHA256
    assert result["full_gemma_model_loaded"] is False


def test_v92_source_tensor_and_memory_authentication_are_model_free() -> None:
    config = load_config_v92()
    sources = authenticate_sources_v92(config, require_implementation_sources=False)
    tensor = authenticate_pinned_model_tensor_v92(config)
    memory = memory_preflight_v92(config)

    assert sources["gemma_model_blob_sha256_identity"] == config["sources"][
        "model_blob_sha256_identity"
    ]
    assert tensor["tensor_name"] == TARGET_MODULE + ".weight"
    assert tensor["shape"] == [1536, 4096]
    assert tensor["dtype"] == "BF16"
    assert tensor["tensor_materialized"] is False
    assert tensor["full_gemma_model_loaded"] is False
    assert memory["shape"] == [1, 738, 1536]
    assert memory["canonical_prefix_sha256"] == (
        "a428f5147c815839ae7315a0adab952ab210814fb21dcdc5bf13b167f28a6e37"
    )
    assert memory["questions_or_answers_serialized"] is False
    assert memory["oracle_loaded"] is False
    assert memory["model_loaded"] is False


def test_v92_preregistered_gates_preserve_retention_and_generalization() -> None:
    gates = load_config_v92()["gates"]

    assert gates["canonical_correct_minimum"] == 122
    assert gates["canonical_presence_correct_minimum"] == 21
    assert gates["canonical_count_correct_minimum"] == 9
    assert gates["canonical_metric_correct_minimum"] == 1
    assert gates["canonical_attribute_correct_minimum"] == 15
    assert gates["canonical_spatial_correct_minimum"] == 73
    assert gates["canonical_support_correct_minimum"] == 1
    assert gates["primary_conversational_required_correct"] == 12
    assert gates["core_actionable_required_correct"] == 6
    assert gates["new_held_wording_required_correct"] == 22
    assert gates["new_held_wording_each_intent_minimum"] == 1
    assert gates["causal_mean_zero_minus_correct_nll_minimum"] == 0.5
    assert gates["causal_prediction_change_minimum"] == 8


def test_v92_evaluator_scores_perfect_records_against_sealed_rows() -> None:
    config = load_config_v92()
    canonical = load_canonical_rows_v92(config)
    primary = primary_rows_v92(config)
    held = held_wording_rows_v92(config)
    canonical_records = [
        {"question_id": row.question_id, "prediction": row.answer, "correct_mean_nll": 0.1}
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
    score = v92_evaluator.score_records_v92(
        canonical,
        canonical_records,
        primary_records,
        held_records,
        causal_records,
        gates=config["gates"],
        prefix_hash_invariant=True,
        environment_input_invariant=True,
        parent_state_invariant=True,
        candidate_state_invariant=True,
        protected_read_count=0,
    )

    assert score["model_acceptance_gate_passed"] is True
    assert all(score["model_acceptance_gates"].values())
    assert score["canonical_type_specific"]["correct"] == 138
    assert score["primary_conversational"]["correct"] == 13
    assert score["new_held_wording"]["correct"] == 26
    assert score["causal_control"]["canonical_prediction_changes"] == 13
