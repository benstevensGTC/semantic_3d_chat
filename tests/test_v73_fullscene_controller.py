from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from semantic_3d_chat.scene_encoder.question_control_v73 import (
    DCT40QuestionControlBaselineV73,
    FullSceneSetAttentionQuestionControlV73,
    PositiveFloorMultiheadAttentionV73,
)
from semantic_3d_chat.training.train_question_control_v73 import (
    EXPECTED_HELD_CHANGED_UNITS,
    HELD_PAIR_IDS,
    LOCKED_ABSOLUTE_GATES,
    LOCKED_GATES,
    TRAIN_PAIR_IDS,
    absolute_reader_gate_v73,
    build_prototype_bank_v73,
    changed_units_v73,
    load_config_v73,
    load_prefixes_v73,
    load_training_rows_v73,
    split_rows_v73,
)


def _basis(hidden: int = 16, rank: int = 8) -> torch.Tensor:
    return torch.eye(rank, hidden)


def _controller(
    cls: type[FullSceneSetAttentionQuestionControlV73] = (
        FullSceneSetAttentionQuestionControlV73
    ),
) -> FullSceneSetAttentionQuestionControlV73:
    torch.manual_seed(730073)
    return cls(
        16,
        _basis(),
        model_dimension=16,
        head_count=4,
        feedforward_dimension=32,
        scene_encoder_layers=0,
        scene_cross_attention_layers=2,
        internal_reader_slots=8,
        uniform_floor_mass=0.10,
    )


def test_positive_floor_attention_weights_every_memory_token() -> None:
    torch.manual_seed(73)
    attention = PositiveFloorMultiheadAttentionV73(
        16, 4, uniform_floor_mass=0.10
    )
    output, trace = attention(
        torch.randn(2, 8, 16), torch.randn(2, 256, 16), return_trace=True
    )

    assert output.shape == (2, 8, 16)
    assert trace is not None
    assert trace.memory_tokens == 256
    assert trace.head_count == 4
    assert trace.required_minimum_weight == pytest.approx(0.10 / 256)
    assert trace.observed_minimum_weight >= trace.required_minimum_weight - 1e-8
    assert trace.all_memory_tokens_receive_positive_weight is True


def test_v73_has_exact_zero_initialization_and_no_question_only_bypass() -> None:
    control = _controller()
    prefix = torch.randn(2, 258, 16)
    question_a = torch.randn(2, 5, 16)
    question_b = torch.randn(2, 7, 16)

    assert torch.count_nonzero(control(prefix, question_a).control_tokens) == 0
    assert torch.count_nonzero(control(prefix, question_b).control_tokens) == 0
    assert control.coefficient_output.bias is None

    # Once trained, the reader must still be unable to emit anything from a
    # zero scene regardless of the question.
    target = torch.randn(2, 4, 16)
    optimizer = torch.optim.AdamW(control.parameters(), lr=0.003)
    optimizer.zero_grad(set_to_none=True)
    output = control(prefix, question_a).control_tokens
    (output - target).square().mean().backward()
    # The exact-zero last linear head intentionally blocks upstream gradient on
    # step one, while the head itself must move.  The next step then opens the
    # scene/question path.
    assert control.coefficient_output.weight.grad is not None
    assert float(control.coefficient_output.weight.grad.abs().sum()) > 0.0
    assert control.scene_projection.weight.grad is not None
    assert float(control.scene_projection.weight.grad.abs().sum()) == 0.0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    output = control(prefix, question_a).control_tokens
    (output - target).square().mean().backward()
    assert control.scene_projection.weight.grad is not None
    assert float(control.scene_projection.weight.grad.abs().sum()) > 0.0
    assert control.question_projection.weight.grad is not None
    assert float(control.question_projection.weight.grad.abs().sum()) > 0.0
    assert control.scene_cross_blocks[0].attention.value_projection.weight.grad is not None
    assert (
        float(
            control.scene_cross_blocks[
                0
            ].attention.value_projection.weight.grad.abs().sum()
        )
        > 0.0
    )
    optimizer.step()
    assert (
        float(control(prefix, question_a).control_tokens.detach().abs().max()) > 0.0
    )
    assert torch.count_nonzero(
        control(torch.zeros_like(prefix), question_a).control_tokens
    ) == 0
    assert torch.count_nonzero(
        control(torch.zeros_like(prefix), question_b).control_tokens
    ) == 0


def test_v73_traced_forward_processes_all_256_latents_and_keeps_full_prefix() -> None:
    control = _controller()
    with torch.no_grad():
        control.coefficient_output.weight.normal_(std=0.02)
    prefix = torch.randn(1, 258, 16)
    question = torch.randn(1, 6, 16)
    baseline = control(prefix, question, return_traces=True).control_tokens
    audit = control.audit()

    changed = prefix.clone()
    changed[:, -2, :] += 5.0
    moved = control(changed, question).control_tokens
    assert not torch.equal(baseline, moved)
    assert audit.environment_latent_count == 256
    assert audit.scene_memory_tokens == 256
    assert audit.internal_reader_slots == 8
    assert audit.control_token_count == 4
    assert audit.scene_cross_attention_layers == 2
    assert audit.every_environment_latent_processed is True
    assert audit.full_prefix_retained_separately_for_language_model is True
    assert audit.question_dependent_retrieval is False
    assert audit.latent_selection_or_top_k_used is False
    assert audit.environmental_text_inputs == 0
    assert audit.question_only_output_path_exists is False
    assert audit.output_computed_only_from_scene_value_contexts is True


def test_dct40_is_explicit_same_stack_bottleneck_ablation() -> None:
    full = _controller()
    dct = _controller(DCT40QuestionControlBaselineV73)
    prefix = torch.randn(1, 258, 16)

    full_memory, _ = full.encode_scene(prefix)
    dct_memory, _ = dct.encode_scene(prefix)
    assert full_memory.shape == (1, 256, 16)
    assert dct_memory.shape == (1, 40, 16)
    assert full.dct_scene_bottleneck_used is False
    assert dct.dct_scene_bottleneck_used is True
    # The duplicated DC term reproduces V71's independent 8 + 32 branches.
    assert torch.equal(dct_memory[:, 0], dct_memory[:, 8])


def test_real_training_pool_split_is_pair_and_scene_disjoint() -> None:
    rows = load_training_rows_v73("data_diverse52/qa/train.jsonl")
    train, held = split_rows_v73(rows)

    assert len(rows) == 960
    assert len(train) == 576
    assert len(held) == 384
    assert {row.pair_id for row in train} == set(TRAIN_PAIR_IDS)
    assert {row.pair_id for row in held} == set(HELD_PAIR_IDS)
    assert {row.scene_id for row in train}.isdisjoint(
        {row.scene_id for row in held}
    )
    assert sum(row.expected_change for row in held) == 52
    assert len(changed_units_v73(held)) == EXPECTED_HELD_CHANGED_UNITS == 26
    train_classes = {row.answer_class for row in train}
    unsupported = [row for row in held if row.answer_class not in train_classes]
    assert len(unsupported) == 1


def test_real_prefix_loader_opens_only_opaque_continuous_scene_files() -> None:
    prefixes, manifest = load_prefixes_v73(
        "data_gemma4/scene_tokens/v56_question_control_full_prefixes",
        ["scene_000011", "scene_000056"],
    )

    assert set(prefixes) == {"scene_000011", "scene_000056"}
    assert all(value.shape == (1, 258, 1536) for value in prefixes.values())
    assert all(torch.isfinite(value).all() for value in prefixes.values())
    assert manifest["question_inputs_used"] is False
    assert manifest["question_dependent_scene_retrieval"] is False
    assert manifest["environmental_text_inputs"] == []


def test_v73_config_locks_split_scope_and_both_gate_families(tmp_path: Path) -> None:
    config = load_config_v73(
        "configs/experiments/gemma4_v73_fullscene_controller.yaml"
    )
    assert config["gates"] == vars(LOCKED_GATES)
    assert config["absolute_reader_gates"] == vars(LOCKED_ABSOLUTE_GATES)
    assert config["scope"]["oracle_loaded"] is False
    assert config["scope"]["official_validation_loaded"] is False
    assert config["scope"]["official_test_loaded"] is False
    assert config["architecture"]["question_only_residual_to_output"] is False

    payload = Path(
        "configs/experiments/gemma4_v73_fullscene_controller.yaml"
    ).read_text()
    tampered = tmp_path / "v73.yaml"
    tampered.write_text(payload.replace("prediction_change_units: 13", "prediction_change_units: 12", 1))
    with pytest.raises(ValueError, match="gates changed"):
        load_config_v73(tampered)


def test_native_prototype_bank_is_training_fold_only_and_orthonormal() -> None:
    rows = load_training_rows_v73("data_diverse52/qa/train.jsonl")
    train, _held = split_rows_v73(rows)
    class_ids = sorted({row.answer_class for row in train})
    torch.manual_seed(74)
    embeddings = {
        class_id: torch.randn(1 + index % 5, 1536)
        for index, class_id in enumerate(class_ids)
    }
    bank = build_prototype_bank_v73(
        train, embeddings, target_rms=0.10, basis_rank=32
    )

    assert bank.prototypes.shape == (28, 4, 1536)
    assert bank.output_basis.shape == (32, 1536)
    assert torch.allclose(
        bank.output_basis @ bank.output_basis.T,
        torch.eye(32),
        atol=2e-4,
        rtol=2e-4,
    )
    assert torch.allclose(
        bank.prototypes.square().mean(dim=-1).sqrt(),
        torch.full((28, 4), 0.10),
        atol=1e-6,
    )


def test_absolute_reader_gate_is_independent_of_dct_advantage() -> None:
    metrics = {
        "supported_accuracy": 0.85,
        "changed_supported_accuracy": 0.70,
        "complete_class_units": 9,
        "prediction_change_units": 13,
        "positive_own_over_opposite_sides": 37,
        "mean_own_over_opposite_margin": 0.5,
        "mean_correct_over_wrong_scene_margin": 0.4,
        "zero_scene_maximum_absolute_control": 0.0,
    }
    assert absolute_reader_gate_v73(metrics)["passed"] is True
    failed = copy.deepcopy(metrics)
    failed["prediction_change_units"] = 12
    result = absolute_reader_gate_v73(failed)
    assert result["passed"] is False
    assert result["checks"]["prediction_change_units"] is False
