from __future__ import annotations

from collections import Counter

from semantic_3d_chat.evaluation.evaluate_v88_scene1_augmented import _smoke_rows_v88
from semantic_3d_chat.evaluation.v88_scene1_augmented_preflight import (
    AUGMENTED_CAUSAL_IDS,
    CANONICAL_CAUSAL_IDS,
    derive_lora_preflight_v88,
    derive_training_items_v88,
    derive_v87_error_inventory_v88,
    load_canonical_rows_v88,
    load_config_v88,
    training_schedule_v88,
)


def test_v88_parent_error_and_training_only_augmentation_inventory() -> None:
    config = load_config_v88()
    rows = load_canonical_rows_v88(config)
    errors, hard_rows = derive_v87_error_inventory_v88(config, rows)
    items = derive_training_items_v88(config, rows, hard_rows)

    assert len(rows) == 138
    assert len(errors) == len(hard_rows) == 35
    assert Counter(record["answer_type"] for record in errors) == {
        "attribute": 11,
        "spatial_relation": 23,
        "support": 1,
    }
    assert len(items) == 282
    assert Counter(item.kind for item in items) == {
        "canonical": 138,
        "hard_error_replay": 35,
        "inverse_spatial": 86,
        "alternate_attribute": 9,
        "alternate_presence": 13,
        "development_known_smoke": 1,
    }
    by_id = {item.schedule_id: item for item in items}
    assert by_id["q_000108"].row.question == "What color is the bowl?"
    assert by_id["q_000108"].row.answer == "red"
    assert (
        by_id["v88_inverse_q_000014"].row.question
        == "Is the bowl left or right of the chair?"
    )
    assert by_id["v88_inverse_q_000014"].row.answer == "left"
    assert by_id["v88_smoke_chair"].row.question == "Is there a chair?"
    assert by_id["v88_smoke_chair"].row.answer == "yes"


def test_v88_fixed_schedule_and_causal_controls() -> None:
    config = load_config_v88()
    rows = load_canonical_rows_v88(config)
    _errors, hard_rows = derive_v87_error_inventory_v88(config, rows)
    items = derive_training_items_v88(config, rows, hard_rows)
    schedule = training_schedule_v88(items)

    assert len(schedule) == 1128
    assert Counter(epoch for epoch, _item in schedule) == {0: 282, 1: 282, 2: 282, 3: 282}
    assert set(Counter(item.schedule_id for _epoch, item in schedule).values()) == {4}
    causal = Counter(item.schedule_id for _epoch, item in schedule if item.causal_margin)
    assert causal == {question_id: 4 for question_id in CANONICAL_CAUSAL_IDS + AUGMENTED_CAUSAL_IDS}


def test_v88_fresh_bank_is_rank16_zero_output_layer27_q_projection() -> None:
    config = load_config_v88()
    result = derive_lora_preflight_v88(config)

    assert result["target_modules"] == [
        "model.language_model.layers.27.self_attn.q_proj"
    ]
    assert result["parameter_count"] == 57_344
    assert result["lora_a_shape"] == [16, 1536]
    assert result["lora_b_shape"] == [2048, 16]
    assert result["lora_b_nonzero_count"] == 0
    assert result["initial_state_sha256"] == config["bridge"]["expected_initial_state_sha256"]


def test_v88_smoke_is_explicitly_training_known_not_held_out() -> None:
    config = load_config_v88()
    rows = _smoke_rows_v88(config)

    assert config["gates"]["live_smoke_is_development_known_and_trained"] is True
    assert config["gates"]["live_smoke_is_held_out"] is False
    assert [row.question for row in rows] == [
        "Is there a chair?",
        "What color is the bowl?",
        "Is the bowl left or right of the chair?",
    ]
    assert [row.answer for row in rows] == ["yes", "red", "left"]
    assert {row.pair_id for row in rows} == {"v88_development_known_smoke"}
