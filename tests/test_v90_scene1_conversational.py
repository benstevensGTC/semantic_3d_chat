from __future__ import annotations

from collections import Counter

import pytest

from semantic_3d_chat.evaluation import (
    evaluate_v90_scene1_conversational as v90_evaluator,
)
from semantic_3d_chat.evaluation.v90_scene1_conversational_preflight import (
    CONFIG,
    EXPECTED_INITIAL_STATE_SHA256,
    TARGET_MODULE,
    authenticate_parent_v90,
    authenticate_sources_v90,
    conversational_rows_v90,
    derive_contract_v90,
    derive_training_items_v90,
    held_wording_rows_v90,
    load_canonical_rows_v90,
    load_config_v90,
    lora_preflight_v90,
    memory_preflight_v90,
    primary_rows_v90,
    protocol_v90,
    schedule_v90,
)


def _training_protocol() -> tuple[dict, tuple, tuple]:
    config = load_config_v90()
    canonical = load_canonical_rows_v90(config)
    items = derive_training_items_v90(config, canonical)
    return config, canonical, items


def test_v90_expands_13_intents_without_training_on_26_held_wordings() -> None:
    config = load_config_v90()
    primary = primary_rows_v90(config)
    conversational = conversational_rows_v90(config)
    held = held_wording_rows_v90(config)

    assert len(primary) == 13
    assert len(conversational) == 52
    assert len(held) == 26
    assert set(Counter(row.pair_id for row in conversational).values()) == {4}
    assert set(Counter(row.pair_id for row in held).values()) == {2}
    assert {row.question_id for row in primary} <= {
        row.question_id for row in conversational
    }
    assert {row.question_id for row in held}.isdisjoint(
        row.question_id for row in conversational
    )
    assert [row.question for row in primary] == [
        raw["primary"] for raw in config["conversational_intents"]
    ]


def test_v90_exact_344_row_inventory_and_v89_retention_exposures() -> None:
    _config, canonical, items = _training_protocol()

    assert len(canonical) == 138
    assert len(items) == 344
    assert Counter(item.kind for item in items) == {
        "canonical": 138,
        "error_replay": 32,
        "correct_anchor_replay": 122,
        "conversational": 52,
    }
    canonical_exposures = Counter(
        item.source_question_id
        for item in items
        if item.kind
        in {"canonical", "error_replay", "correct_anchor_replay"}
    )
    assert Counter(canonical_exposures.values()) == {2: 122, 3: 16}
    causal = [item for item in items if item.causal_margin]
    assert len(causal) == 13
    assert {item.wording for item in causal} == {"primary"}
    assert {item.intent_id for item in causal} == {
        raw["id"] for raw in _config["conversational_intents"]
    }


def test_v90_schedule_is_three_deterministic_344_row_epochs() -> None:
    config, _canonical, items = _training_protocol()
    schedule = schedule_v90(items)

    assert len(schedule) == 1_032
    assert Counter(epoch for epoch, _item in schedule) == {0: 344, 1: 344, 2: 344}
    assert set(Counter(item.schedule_id for _epoch, item in schedule).values()) == {3}
    assert sum(item.causal_margin for _epoch, item in schedule) == 39
    assert config["training"]["gradient_accumulation_rows"] == 6
    assert len(schedule) // config["training"]["gradient_accumulation_rows"] == 172

    contract = derive_contract_v90()
    assert contract["training_inventory_sha256"] == (
        "8963085d8276f2000480e11651cbe97fcc5b7eb7711c9c3c2cb3e20090fc9cc7"
    )
    assert contract["training_schedule_sha256"] == (
        "3c3b199b723d9bf6be6686a421b997fbac2eb209c5886bcd4544a67fd7954643"
    )
    assert protocol_v90(
        config, config_path=CONFIG, require_sealed_hashes=False
    )["checks"] == {
        "canonical_138_exact": True,
        "v89_errors_32_exact": True,
        "v89_anchors_122_exact": True,
        "conversational_52_exact": True,
        "held_wording_26_excluded_from_training": True,
        "rows_per_epoch_344_exact": True,
        "three_epochs_exact": True,
        "micro_rows_1032_exact": True,
        "accumulation_6_exact": True,
        "optimizer_updates_172_exact": True,
        "primary_causal_13_per_epoch_exact": True,
        "primary_causal_39_total_exact": True,
        "oracle_not_loaded_by_trainer": True,
        "runtime_serializes_no_supervision": True,
        "fresh_target_disjoint": True,
    }


def test_v90_fresh_bank_is_exact_zero_rank8_layer28_output_projection() -> None:
    config = load_config_v90()
    result = lora_preflight_v90(config)

    assert result["target_modules"] == [TARGET_MODULE]
    assert result["base_projection_type"] == "torch.nn.Linear"
    assert result["base_projection_weight_shape"] == [1536, 2048]
    assert result["parameter_count"] == 28_672
    assert result["lora_a_shape"] == [8, 2048]
    assert result["lora_b_shape"] == [1536, 8]
    assert result["lora_b_nonzero_count"] == 0
    assert result["initial_state_sha256"] == EXPECTED_INITIAL_STATE_SHA256
    assert result["device"] == "cpu"
    assert result["full_gemma_model_loaded"] is False


def test_v90_authenticates_v89_parent_and_immutable_memory_without_model_load() -> None:
    config = load_config_v90()
    sources = authenticate_sources_v90(
        config, require_implementation_sources=False
    )
    parent = authenticate_parent_v90(config)
    memory = memory_preflight_v90(config)

    assert sources["gemma_model_blob_sha256_identity"] == config["sources"][
        "model_blob_sha256_identity"
    ]
    assert parent == {
        "release_artifact": "gemma4_v89_strict_runtime_release_v1",
        "promotion_decision": "strict_scene1_experimental_primary",
        "all_release_gates_passed": True,
        "canonical_correct": 122,
        "canonical_errors": 16,
        "frozen_bank_count": 11,
        "frozen_adapter_parameter_count": 872448,
        "fresh_target_disjoint": True,
        "runtime_promotion_authorized": True,
        "held_out_generalization_claim": False,
    }
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


def test_v90_config_is_sealed_for_full_model_preflight() -> None:
    config = load_config_v90()
    assert config["status"] == "sealed_before_full_model_load"
    assert load_config_v90(allow_draft=False)["status"] == config["status"]


def test_v90_evaluator_reuses_the_preflight_sealed_question_inventories() -> None:
    config = load_config_v90()

    evaluator_primary = v90_evaluator.primary_rows_v90(config)
    evaluator_held = v90_evaluator.held_wording_rows_v90(config)

    assert evaluator_primary == primary_rows_v90(config)
    assert evaluator_held == held_wording_rows_v90(config)
    assert {
        v90_evaluator._intent_id(row) for row in evaluator_primary
    } == v90_evaluator.INTENT_IDS
    assert Counter(v90_evaluator._intent_id(row) for row in evaluator_held) == {
        identifier: 2 for identifier in v90_evaluator.INTENT_IDS
    }


def test_v90_evaluator_list_scoring_consumes_long_names_and_rejects_extras() -> None:
    inventory = (
        "table, chair, picture frame, bowl, floor lamp, cube, book, cabinet, "
        "plant pot"
    )

    assert v90_evaluator._extract_object_items(inventory) == frozenset(
        {
            "table",
            "chair",
            "picture frame",
            "bowl",
            "floor lamp",
            "cube",
            "book",
            "cabinet",
            "plant pot",
        }
    )
    assert "floor" not in v90_evaluator._extract_object_items(inventory)
    assert "floor" in v90_evaluator._extract_object_items(f"{inventory}, floor")
    assert v90_evaluator.conversational_match_v90(
        "inventory", "inventory", inventory, inventory
    )
    assert not v90_evaluator.conversational_match_v90(
        "inventory", "inventory", f"{inventory}, floor", inventory
    )
    assert v90_evaluator.conversational_match_v90(
        "table_contents", "support_list", "cube and book", "book, cube"
    )


@pytest.mark.parametrize(
    ("intent_id", "family", "prediction", "reference"),
    (
        ("cube_location", "object_location", "table", "on the table"),
        ("cube_location", "object_location", "on top of the table", "on the table"),
        ("frame_support", "frame_support", "wall-mounted", "wall"),
        ("wall_object", "wall_object", "frame", "picture frame"),
    ),
)
def test_v90_evaluator_accepts_bounded_conversational_aliases(
    intent_id: str,
    family: str,
    prediction: str,
    reference: str,
) -> None:
    assert v90_evaluator.conversational_match_v90(
        intent_id, family, prediction, reference
    )


def _perfect_v90_evaluation_records() -> tuple[dict, tuple, list, list, list, list]:
    config = load_config_v90()
    canonical = load_canonical_rows_v90(config)
    primary = primary_rows_v90(config)
    held = held_wording_rows_v90(config)
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


def test_v90_evaluator_perfect_synthetic_records_pass_all_preregistered_gates() -> None:
    config, canonical, canonical_records, primary_records, held_records, causal = (
        _perfect_v90_evaluation_records()
    )

    score = v90_evaluator.score_records_v90(
        canonical,
        canonical_records,
        primary_records,
        held_records,
        causal,
        gates=config["gates"],
        prefix_hash_invariant=True,
        environment_input_invariant=True,
        parent_state_invariant=True,
        protected_read_count=0,
    )

    assert score["model_acceptance_gate_passed"] is True
    assert len(score["model_acceptance_gates"]) == 18
    assert all(score["model_acceptance_gates"].values())
    assert score["primary_conversational"]["correct"] == 13
    assert score["held_wording"]["correct"] == 26
    assert score["causal_control"]["canonical_prediction_changes"] == 13
