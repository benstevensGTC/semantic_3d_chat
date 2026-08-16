from __future__ import annotations

from collections import Counter

from semantic_3d_chat.evaluation import (
    evaluate_v91_scene1_conversational_repair as v91_evaluator,
)
from semantic_3d_chat.evaluation.v91_scene1_conversational_preflight import (
    CONFIG,
    EXPECTED_INITIAL_STATE_SHA256,
    FAILED_INTENTS,
    SUCCESSFUL_INTENTS,
    TARGET_MODULE,
    authenticate_parent_v91,
    authenticate_pinned_model_tensor_v91,
    authenticate_sources_v91,
    derive_contract_v91,
    derive_training_items_v91,
    held_wording_rows_v91,
    load_canonical_rows_v91,
    load_config_v91,
    lora_preflight_v91,
    memory_preflight_v91,
    primary_rows_v91,
    protocol_v91,
    schedule_v91,
    training_wording_rows_v91,
)


def _training_protocol() -> tuple[dict, tuple, tuple]:
    config = load_config_v91()
    canonical = load_canonical_rows_v91(config)
    items = derive_training_items_v91(config, canonical)
    return config, canonical, items


def test_v91_trains_all_six_existing_wordings_and_holds_two_new_per_intent() -> None:
    config = load_config_v91()
    training = training_wording_rows_v91(config)
    primary = primary_rows_v91(config)
    held = held_wording_rows_v91(config)

    assert len(training) == 78
    assert len(primary) == 13
    assert len(held) == 26
    assert set(Counter(row.pair_id for row in training).values()) == {6}
    assert set(Counter(row.pair_id for row in held).values()) == {2}
    assert {row.question_id for row in primary} <= {row.question_id for row in training}
    assert {row.question_id for row in held}.isdisjoint(
        row.question_id for row in training
    )
    assert [row.question for row in primary] == [
        raw["existing_wordings"][0] for raw in config["conversational_intents"]
    ]


def test_v91_authenticates_exact_failed_v90_parent_and_124_14_split() -> None:
    config, canonical, items = _training_protocol()
    parent = authenticate_parent_v91(config)

    assert len(canonical) == 138
    assert parent == {
        "v89_frozen_bank_count": 11,
        "v89_frozen_parameter_count": 872448,
        "v90_bank_name": "v90_scene1_conversational_bridge",
        "v90_state_sha256": (
            "70e236711d8ac1fe7cf808f6f4e939b29db476016c8ef49db143707df0f3bde7"
        ),
        "v90_parameter_count": 28672,
        "total_frozen_bank_count": 12,
        "total_frozen_parameter_count": 901120,
        "v90_model_acceptance_gate_passed": False,
        "v90_runtime_promotion_authorized": False,
        "canonical_correct": 124,
        "canonical_errors": 14,
        "primary_correct": 7,
        "new_repair_is_post_failure_development": True,
    }
    canonical_exposures = Counter(
        item.source_question_id
        for item in items
        if item.kind in {"canonical", "error_replay", "correct_anchor_replay"}
    )
    assert Counter(canonical_exposures.values()) == {2: 124, 3: 14}


def test_v91_exact_590_row_evidence_weighted_repair_inventory() -> None:
    _config, _canonical, items = _training_protocol()

    assert len(items) == 590
    assert Counter(item.kind for item in items) == {
        "canonical": 138,
        "error_replay": 28,
        "correct_anchor_replay": 124,
        "conversational_success": 84,
        "conversational_repair": 216,
    }
    conversation = [item for item in items if item.intent_id is not None]
    exposures = Counter(item.intent_id for item in conversation)
    assert {exposures[intent_id] for intent_id in SUCCESSFUL_INTENTS} == {12}
    assert {exposures[intent_id] for intent_id in FAILED_INTENTS} == {36}
    causal = [item for item in items if item.causal_margin]
    assert len(causal) == 13
    assert {item.intent_id for item in causal} == set(FAILED_INTENTS) | set(
        SUCCESSFUL_INTENTS
    )
    assert {item.wording_ordinal for item in causal} == {0}
    assert {item.copy_ordinal for item in causal} == {0}


def test_v91_schedule_and_hashes_are_deterministic() -> None:
    config, _canonical, items = _training_protocol()
    schedule = schedule_v91(items)
    contract = derive_contract_v91()

    assert len(schedule) == 1770
    assert Counter(epoch for epoch, _item in schedule) == {0: 590, 1: 590, 2: 590}
    assert set(Counter(item.schedule_id for _epoch, item in schedule).values()) == {3}
    assert sum(item.causal_margin for _epoch, item in schedule) == 39
    assert len(schedule) // config["training"]["gradient_accumulation_rows"] == 295
    assert contract["training_inventory_sha256"] == (
        "27d478cf90b4e5f9b1c7047b1be79786d28926b984d9ddd8dc559627c307d5fe"
    )
    assert contract["training_schedule_sha256"] == (
        "f3d66ef1f5ee6922f06b4d976097379eb0104c7bcfc7bbde7509c00b1a23bc11"
    )
    checks = protocol_v91(
        config, config_path=CONFIG, require_sealed_hashes=False
    )["checks"]
    assert all(checks.values())


def test_v91_fresh_bank_is_exact_zero_rank16_layer33_mlp_projection() -> None:
    config = load_config_v91()
    result = lora_preflight_v91(config)

    assert result["target_modules"] == [TARGET_MODULE]
    assert result["base_projection_type"] == "torch.nn.Linear"
    assert result["base_projection_weight_shape"] == [1536, 12288]
    assert result["parameter_count"] == 221184
    assert result["lora_a_shape"] == [16, 12288]
    assert result["lora_b_shape"] == [1536, 16]
    assert result["lora_b_nonzero_count"] == 0
    assert result["initial_state_sha256"] == EXPECTED_INITIAL_STATE_SHA256
    assert result["device"] == "cpu"
    assert result["full_gemma_model_loaded"] is False


def test_v91_source_and_memory_authentication_are_model_free() -> None:
    config = load_config_v91()
    sources = authenticate_sources_v91(config, require_implementation_sources=False)
    model_tensor = authenticate_pinned_model_tensor_v91(config)
    memory = memory_preflight_v91(config)

    assert sources["gemma_model_blob_sha256_identity"] == config["sources"][
        "model_blob_sha256_identity"
    ]
    assert sources["pinned_model_tensor"] == model_tensor
    assert model_tensor["tensor_name"] == (
        "model.language_model.layers.33.mlp.down_proj.weight"
    )
    assert model_tensor["shape"] == [1536, 12288]
    assert model_tensor["dtype"] == "BF16"
    assert model_tensor["header_read_via_safe_open"] is True
    assert model_tensor["tensor_materialized"] is False
    assert model_tensor["full_gemma_model_loaded"] is False
    assert memory["shape"] == [1, 738, 1536]
    assert memory["canonical_prefix_sha256"] == (
        "a428f5147c815839ae7315a0adab952ab210814fb21dcdc5bf13b167f28a6e37"
    )
    assert memory["zero_payload_prefix_sha256"] == (
        "c455ed77bf3fe841a4e360e89c036ffb4e8cdb8935c97d767f556d25a9f790d8"
    )
    assert memory["questions_or_answers_serialized"] is False
    assert memory["oracle_loaded"] is False
    assert memory["model_loaded"] is False


def test_v91_gates_are_preregistered_stronger_than_v90_failures() -> None:
    config = load_config_v91()
    gates = config["gates"]

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


def test_v91_evaluator_reuses_sealed_rows_and_scores_perfect_records() -> None:
    config = load_config_v91()
    canonical = load_canonical_rows_v91(config)
    primary = primary_rows_v91(config)
    held = held_wording_rows_v91(config)

    assert v91_evaluator.primary_rows_v91(config) == primary
    assert v91_evaluator.held_wording_rows_v91(config) == held
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
    score = v91_evaluator.score_records_v91(
        canonical,
        canonical_records,
        primary_records,
        held_records,
        causal_records,
        gates=config["gates"],
        prefix_hash_invariant=True,
        environment_input_invariant=True,
        parent_state_invariant=True,
        protected_read_count=0,
    )

    assert score["model_acceptance_gate_passed"] is True
    assert all(score["model_acceptance_gates"].values())
    assert score["canonical_type_specific"]["correct"] == 138
    assert score["primary_conversational"]["correct"] == 13
    assert score["new_held_wording"]["correct"] == 26
    assert score["causal_control"]["canonical_prediction_changes"] == 13


def test_v91_config_is_sealed_before_any_full_model_load() -> None:
    config = load_config_v91(allow_draft=False)

    assert config["status"] == "sealed_before_full_model_load"
    assert all(
        config["sources"][key] != "TO_FILL"
        for key in (
            "preflight_source_sha256",
            "training_source_sha256",
            "evaluation_source_sha256",
        )
    )
    assert config["dataset"]["training_inventory_sha256"] != "TO_FILL"
    assert config["dataset"]["training_schedule_sha256"] != "TO_FILL"


def test_v91_topology_amendment_preserves_v1_evidence_and_uses_v2_outputs() -> None:
    config = load_config_v91(allow_draft=False)
    amendment = config["topology_amendment"]

    assert amendment["amendment_version"] == 2
    assert amendment["superseded_config_sha256"] == (
        "b5c95ec12fd0040731417700936be94b865abcbfbf16f157be0aedf7d4e76e09"
    )
    assert amendment["superseded_synthetic_weight_shape"] == [1536, 6144]
    assert amendment["pinned_weight_shape"] == [1536, 12288]
    assert amendment["corrected_trainable_parameter_count"] == 221184
    assert amendment["corrected_initial_state_sha256"] == (
        "0f255efb26255dcac0815511e44aabad5e21820f78f9a7662dc1bf59f627db2b"
    )
    assert amendment["superseded_artifacts_preserved"] is True
    assert all("v2" in value for value in config["outputs"].values())
