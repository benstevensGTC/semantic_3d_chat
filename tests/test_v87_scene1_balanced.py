from __future__ import annotations

from collections import Counter

from semantic_3d_chat.evaluation.evaluate_v87_scene1_balanced import (
    score_records_v87,
)
from semantic_3d_chat.evaluation.v86_scene1_demo_preflight import (
    causal_rows_v86,
    load_scene1_rows_v86,
)
from semantic_3d_chat.evaluation.v87_scene1_balanced_preflight import (
    answer_class_balance_v87,
    balanced_schedule_v87,
    load_config_v87,
    lora_preflight_v87,
    protocol_preflight_v87,
    validate_parent_v86,
)


def test_v87_parent_failure_is_bound_without_mutation() -> None:
    config = load_config_v87()
    parent = validate_parent_v86(config)

    assert parent["parent_v86_canonical_correct"] == 86
    assert parent["parent_v86_canonical_total"] == 138
    assert parent["parent_v86_generic_smoke_correct"] == 3
    assert parent["parent_v86_only_failed_gate"] == (
        "all_scene1_canonical_accuracy_at_least_0_80"
    )
    assert parent["parent_v86_runtime_promoted"] is False
    assert parent["parent_v86_mutated"] is False


def test_v87_class_balance_and_schedule_are_exact() -> None:
    config = load_config_v87()
    rows = load_scene1_rows_v86(config)
    counts, weights = answer_class_balance_v87(config, rows)
    schedule = balanced_schedule_v87(rows)

    assert len(counts) == 19
    assert len(schedule) == 1104
    assert set(Counter(epoch for epoch, _row in schedule).values()) == {138}
    assert set(Counter(row.question_id for _epoch, row in schedule).values()) == {8}
    assert abs(sum(weights[row.answer_class] for row in rows) - 138.0) < 1e-10
    aggregate = {
        answer_class: weights[answer_class] * frequency
        for answer_class, frequency in counts.items()
    }
    assert max(aggregate.values()) - min(aggregate.values()) < 1e-10


def test_v87_protocol_and_fresh_gate_projection() -> None:
    config = load_config_v87()
    protocol = protocol_preflight_v87(config)
    lora = lora_preflight_v87(config)

    assert protocol["row_count"] == 138
    assert protocol["opaque_answer_class_count"] == 19
    assert protocol["schedule_rows"] == 1104
    assert protocol["causal_rows_total"] == 24
    assert protocol["zero_payload_nonzero_scalar_count"] == 0
    assert lora["target_modules"] == [
        "model.language_model.layers.34.mlp.gate_proj"
    ]
    assert lora["parameter_count"] == 110_592
    assert lora["exact_zero_output_at_initialization"] is True


def test_v87_synthetic_perfect_score_passes_all_model_gates() -> None:
    config = load_config_v87()
    rows = load_scene1_rows_v86(config)
    records = [
        {
            "question_id": row.question_id,
            "prediction": row.answer,
            "correct_mean_nll": 0.1,
        }
        for row in rows
    ]
    causal = [
        {
            "answer_type": row.answer_type,
            "correct_prediction": row.answer,
            "zero_prediction": "unknown",
            "correct_mean_nll": 0.1,
            "zero_mean_nll": 1.1,
            "zero_minus_correct_nll": 1.0,
        }
        for row in causal_rows_v86(config, rows)
    ]
    smoke = [{"exact_correct": True} for _ in range(3)]

    score = score_records_v87(
        rows,
        records,
        causal,
        smoke,
        gates=config["gates"],
        prefix_hash_invariant=True,
        environment_input_invariant=True,
        protected_read_count=0,
    )

    assert score["canonical_type_specific"]["accuracy"] == 1.0
    assert all(score["model_acceptance_gates"].values())
    assert score["model_acceptance_gate_passed"] is True
    assert score["separate_runtime_packaging_authorized"] is True
    assert score["runtime_promotion_authorized"] is False


def test_v87_synthetic_gate_fails_attribute_floor() -> None:
    config = load_config_v87()
    rows = load_scene1_rows_v86(config)
    records = [
        {
            "question_id": row.question_id,
            "prediction": "wrong" if row.answer_type == "attribute" else row.answer,
            "correct_mean_nll": 0.1,
        }
        for row in rows
    ]
    causal = [
        {
            "answer_type": row.answer_type,
            "correct_prediction": row.answer,
            "zero_prediction": "unknown",
            "correct_mean_nll": 0.1,
            "zero_mean_nll": 1.1,
            "zero_minus_correct_nll": 1.0,
        }
        for row in causal_rows_v86(config, rows)
    ]
    score = score_records_v87(
        rows,
        records,
        causal,
        [{"exact_correct": True} for _ in range(3)],
        gates=config["gates"],
        prefix_hash_invariant=True,
        environment_input_invariant=True,
        protected_read_count=0,
    )

    assert score["model_acceptance_gates"][
        "attribute_accuracy_at_least_0_50"
    ] is False
    assert score["model_acceptance_gate_passed"] is False
    assert score["separate_runtime_packaging_authorized"] is False
