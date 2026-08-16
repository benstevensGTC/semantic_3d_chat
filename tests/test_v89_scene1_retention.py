from __future__ import annotations

from collections import Counter

from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.evaluation.v89_scene1_retention_preflight import (
    CANONICAL_CAUSAL_IDS,
    SMOKE_CAUSAL_IDS,
    derive_lora_preflight_v89,
    derive_parent_behavior_v89,
    derive_training_items_v89,
    development_smoke_rows_v89,
    load_canonical_rows_v89,
    load_config_v89,
    training_schedule_v89,
)
from semantic_3d_chat.training.train_v89_scene1_retention import (
    combined_lora_settings_v89,
)


def _protocol() -> tuple[dict, tuple, tuple, tuple, tuple]:
    config = load_config_v89()
    rows = load_canonical_rows_v89(config)
    transitions, errors, anchors, error_rows, anchor_rows = derive_parent_behavior_v89(
        config, rows
    )
    items = derive_training_items_v89(config, rows, error_rows, anchor_rows)
    return config, transitions, errors, anchors, items


def test_v89_parent_transitions_and_retention_inventory() -> None:
    _config, transitions, errors, anchors, items = _protocol()

    assert len(transitions) == 138
    assert Counter(record["transition"] for record in transitions) == {
        "retained_correct": 83,
        "recovered": 24,
        "regressed": 20,
        "retained_wrong": 11,
    }
    assert len(errors) == 31
    assert Counter(record["answer_type"] for record in errors) == {
        "attribute": 7,
        "presence": 1,
        "spatial_relation": 22,
        "support": 1,
    }
    assert len(anchors) == 107
    assert len(items) == 310
    assert Counter(item.kind for item in items) == {
        "canonical": 138,
        "error_replay": 62,
        "correct_anchor_replay": 107,
        "development_known_smoke": 3,
    }


def test_v89_fixed_schedule_exposures_and_causal_controls() -> None:
    config, _transitions, errors, anchors, items = _protocol()
    schedule = training_schedule_v89(items)

    assert len(schedule) == 930
    assert Counter(epoch for epoch, _item in schedule) == {0: 310, 1: 310, 2: 310}
    assert set(Counter(item.schedule_id for _epoch, item in schedule).values()) == {3}
    causal = Counter(item.schedule_id for _epoch, item in schedule if item.causal_margin)
    assert causal == {
        schedule_id: 3 for schedule_id in CANONICAL_CAUSAL_IDS + SMOKE_CAUSAL_IDS
    }
    exposure = Counter(
        item.source_question_id
        for item in items
        if item.kind != "development_known_smoke"
    )
    assert {exposure[record["question_id"]] for record in errors} == {3}
    assert {exposure[record["question_id"]] for record in anchors} == {2}
    assert config["training"]["optimizer_updates"] == 155


def test_v89_fresh_bank_is_disjoint_rank8_zero_output_layer27_o_projection() -> None:
    config = load_config_v89()
    result = derive_lora_preflight_v89(config)

    assert result["target_modules"] == [
        "model.language_model.layers.27.self_attn.o_proj"
    ]
    assert result["parameter_count"] == 28_672
    assert result["lora_a_shape"] == [8, 2048]
    assert result["lora_b_shape"] == [1536, 8]
    assert result["lora_b_nonzero_count"] == 0
    assert result["initial_state_sha256"] == config["bridge"][
        "expected_initial_state_sha256"
    ]


def test_v89_combined_stack_freezes_ten_banks_and_only_trains_v89() -> None:
    config = load_config_v89()
    runtime = load_runtime_config(config["sources"]["runtime_config"])
    settings = combined_lora_settings_v89(runtime, config)

    assert len(settings.banks) == 11
    assert sum(not bank.trainable for bank in settings.banks) == 10
    fresh = [bank for bank in settings.banks if bank.trainable]
    assert len(fresh) == 1
    assert fresh[0].name == "v89_scene1_retention_bridge"
    assert fresh[0].adapter.target_modules == (
        "model.language_model.layers.27.self_attn.o_proj",
    )


def test_v89_smoke_is_explicitly_trained_not_held_out() -> None:
    config = load_config_v89()
    rows = development_smoke_rows_v89(config, load_canonical_rows_v89(config))

    assert config["gates"]["live_smoke_is_development_known_and_trained"] is True
    assert config["gates"]["live_smoke_is_held_out"] is False
    assert [row.question for row in rows] == [
        "Is there a chair?",
        "What color is the bowl?",
        "Is the bowl left or right of the chair?",
    ]
    assert [row.answer for row in rows] == ["yes", "red", "left"]
