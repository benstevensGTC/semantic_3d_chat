from __future__ import annotations

import torch

from semantic_3d_chat.evaluation.evaluate_v86_scene1_demo import (
    score_records_v86,
)
from semantic_3d_chat.evaluation.v86_scene1_demo_preflight import (
    EXPECTED_PREFIX_SHA256,
    _protocol_preflight,
    authenticate_sources_v86,
    causal_rows_v86,
    load_config_v86,
    load_scene1_memory_v86,
    load_scene1_rows_v86,
    lora_preflight_v86,
    training_schedule_v86,
    zero_payload_memory_v86,
)
from semantic_3d_chat.training.train_v86_scene1_demo import (
    zero_payload_margin_objective_v86,
)


def test_v86_sealed_sources_rows_and_schedule() -> None:
    config = load_config_v86()
    sources = authenticate_sources_v86(config)
    rows = load_scene1_rows_v86(config)
    schedule = training_schedule_v86(rows)

    assert len(sources) == 11
    assert len(rows) == 138
    assert len(schedule) == 552
    assert [row.question_id for row in causal_rows_v86(config, rows)] == [
        "q_000080",
        "q_000108",
        "q_000014",
    ]


def test_v86_zero_payload_keeps_native_boundaries() -> None:
    config = load_config_v86()
    memory, digest, _metadata = load_scene1_memory_v86(config)
    zero = zero_payload_memory_v86(memory)

    assert digest == EXPECTED_PREFIX_SHA256
    assert memory.shape == zero.shape == (1, 738, 1536)
    assert memory.dtype == zero.dtype == torch.bfloat16
    assert torch.equal(memory[:, :1], zero[:, :1])
    assert torch.equal(memory[:, -1:], zero[:, -1:])
    assert torch.count_nonzero(zero[:, 1:-1]).item() == 0
    assert not torch.equal(memory, zero)


def test_v86_protocol_and_lora_cpu_preflight() -> None:
    config = load_config_v86()
    protocol = _protocol_preflight(config)
    lora = lora_preflight_v86(config)

    assert protocol["row_count"] == 138
    assert protocol["schedule_rows"] == 552
    assert protocol["zero_payload_token_count"] == 736
    assert protocol["zero_payload_nonzero_scalar_count"] == 0
    assert lora["parameter_count"] == 110_592
    assert lora["lora_a_shape"] == [8, 1536]
    assert lora["lora_b_shape"] == [12_288, 8]
    assert lora["exact_zero_output_at_initialization"] is True


def test_v86_zero_payload_margin_objective() -> None:
    objective, observed, penalty = zero_payload_margin_objective_v86(
        torch.tensor(1.0),
        torch.tensor(1.2),
        target_margin=0.5,
        ce_weight=1.0,
        margin_weight=1.0,
    )

    assert torch.isclose(observed, torch.tensor(0.2))
    assert torch.isclose(penalty, torch.tensor(0.3))
    assert torch.isclose(objective, torch.tensor(1.3))


def test_v86_synthetic_perfect_score_passes_model_gates() -> None:
    config = load_config_v86()
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

    score = score_records_v86(
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
    assert score["causal_control"]["canonical_prediction_changes"] == 3
    assert score["model_acceptance_gate_passed"] is True
    assert score["runtime_promotion_authorized"] is False
