from __future__ import annotations

from collections import Counter

from semantic_3d_chat.evaluation.v93_scene1_termination_paraphrase_preflight import (
    CONFIG,
    CONVERSATIONAL_ERROR_IDS,
    EXPECTED_INITIAL_STATE_SHA256,
    SUPPORT_ERROR_IDS,
    TARGET_MODULE,
    authenticate_parent_v93,
    authenticate_pinned_model_tensor_v93,
    authenticate_sources_v93,
    conversational_errors_v93,
    derive_contract_v93,
    derive_training_items_v93,
    held_wording_rows_v93,
    known_wording_rows_v93,
    load_canonical_rows_v93,
    load_config_v93,
    lora_preflight_v93,
    memory_preflight_v93,
    primary_rows_v93,
    protocol_v93,
    schedule_v93,
    training_paraphrase_rows_v93,
)
from semantic_3d_chat.training.train_question_control_v73 import RowV73


def _inventory() -> tuple[dict, tuple, tuple]:
    config = load_config_v93(allow_draft=True)
    canonical = load_canonical_rows_v93(config)
    return config, canonical, derive_training_items_v93(config, canonical)


def _intent_id(row: RowV73) -> str:
    pair_id = row.pair_id
    return pair_id.split("_conversation_", 1)[1]


def test_v93_wording_splits_are_complete_and_disjoint() -> None:
    config = load_config_v93(allow_draft=True)
    known = known_wording_rows_v93(config)
    training = training_paraphrase_rows_v93(config)
    held = held_wording_rows_v93(config)
    primary = primary_rows_v93(config)

    assert len(known) == 130
    assert len(training) == 78
    assert len(held) == 26
    assert len(primary) == 13
    assert set(Counter(_intent_id(row) for row in known).values()) == {10}
    assert set(Counter(_intent_id(row) for row in training).values()) == {6}
    assert set(Counter(_intent_id(row) for row in held).values()) == {2}
    question_text = [{row.question.casefold() for row in rows} for rows in (known, training, held)]
    assert question_text[0].isdisjoint(question_text[1])
    assert question_text[0].isdisjoint(question_text[2])
    assert question_text[1].isdisjoint(question_text[2])


def test_v93_authenticates_exact_failed_v92_measurements() -> None:
    config = load_config_v93(allow_draft=True)
    parent = authenticate_parent_v93(config)

    assert parent["total_frozen_bank_count"] == 14
    assert parent["total_frozen_parameter_count"] == 1_167_360
    assert parent["v92_model_acceptance_gate_passed"] is False
    assert parent["v92_runtime_promotion_authorized"] is False
    assert parent["canonical_correct"] == 123
    assert parent["canonical_errors"] == 15
    assert parent["primary_correct"] == 12
    assert parent["semantic_primary_failed_intents"] == ["table_contents"]
    assert parent["held_wording_correct"] == 17
    assert tuple(parent["conversational_error_question_ids"]) == CONVERSATIONAL_ERROR_IDS
    assert tuple(parent["support_error_question_ids"]) == SUPPORT_ERROR_IDS


def test_v93_exact_schedule_and_hashes_are_deterministic() -> None:
    config, _canonical, items = _inventory()
    schedule = schedule_v93(items)
    contract = derive_contract_v93()

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
    assert len(schedule) == 1_770
    assert Counter(epoch for epoch, _item in schedule) == {0: 590, 1: 590, 2: 590}
    assert set(Counter(item.schedule_id for _epoch, item in schedule).values()) == {3}
    assert sum(item.causal_margin for _epoch, item in schedule) == 39
    assert len(schedule) // config["training"]["gradient_accumulation_rows"] == 295
    assert contract["training_inventory_sha256"] == (
        "3fcdbb63c3c4a580ecadfacbabf1bf79bc4ea5f3bf540f368f7ce169882b4e40"
    )
    assert contract["training_schedule_sha256"] == (
        "2bb11cccb6572c6a5248987a5693dd45134f5f662e5ba1ab30f2223d354077ed"
    )
    assert contract["eos_supervised_rows"] == 1_770
    assert contract["eos_extra_weight"] == 4.0
    assert all(
        protocol_v93(config, config_path=CONFIG, require_sealed_hashes=False)[
            "checks"
        ].values()
    )


def test_v93_exact_parent_conversation_errors_are_replayed() -> None:
    config = load_config_v93(allow_draft=True)
    errors = conversational_errors_v93(config)

    assert tuple(row.question_id for row in errors) == CONVERSATIONAL_ERROR_IDS


def test_v93_fresh_bank_is_zero_rank8_layer24_projection() -> None:
    config = load_config_v93(allow_draft=True)
    result = lora_preflight_v93(config)

    assert result["target_modules"] == [TARGET_MODULE]
    assert result["base_projection_weight_shape"] == [1536, 4096]
    assert result["parameter_count"] == 45_056
    assert result["lora_a_shape"] == [8, 4096]
    assert result["lora_b_shape"] == [1536, 8]
    assert result["lora_b_nonzero_count"] == 0
    assert result["initial_state_sha256"] == EXPECTED_INITIAL_STATE_SHA256
    assert result["full_gemma_model_loaded"] is False


def test_v93_sources_tensor_and_memory_are_model_free() -> None:
    config = load_config_v93(allow_draft=True)
    sources = authenticate_sources_v93(config, require_implementation_sources=False)
    tensor = authenticate_pinned_model_tensor_v93(config)
    memory = memory_preflight_v93(config)

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


def test_v93_preregistered_gates_require_complete_conversation() -> None:
    gates = load_config_v93(allow_draft=True)["gates"]

    assert gates["canonical_correct_minimum"] == 122
    assert gates["canonical_support_correct_minimum"] == 1
    assert gates["primary_conversational_required_correct"] == 13
    assert gates["core_actionable_required_correct"] == 6
    assert gates["new_held_wording_required_correct"] == 22
    assert gates["new_held_wording_each_intent_minimum"] == 1
    assert gates["causal_mean_zero_minus_correct_nll_minimum"] == 0.5
    assert gates["causal_prediction_change_minimum"] == 8
