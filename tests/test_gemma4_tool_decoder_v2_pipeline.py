from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from semantic_3d_chat.config import load_config
from semantic_3d_chat.evaluation.gemma4_tool_decoder_evaluation_preregistration_v2_1 import (
    PARENT_EVALUATION_SHA256,
    build_evaluation_preregistration_v2_1,
)
from semantic_3d_chat.evaluation.gemma4_tool_decoder_v2_evaluation import (
    CONTROL_MODES,
    evaluate_all_heldout_teacher_forced_v2,
    evaluate_teacher_forced_causal_controls_v2,
    teacher_forced_causal_gate_results_v2,
    teacher_forced_gate_results_v2,
)
from semantic_3d_chat.language.gemma4_answer_tail import (
    answer_tail_forward,
    answer_tail_model_kwargs,
    answer_tail_positions,
    reference_answer_tail_from_full_logits,
)
from semantic_3d_chat.training.gemma4_tool_decoder_v2_data import (
    CAUSAL_VALIDATION_SAMPLE_IDS_SHA256,
    GREEDY_CONTROL_SAMPLE_IDS_SHA256_V2_1,
    action_balanced_schedule_v2,
    causal_validation_indices_v2,
    controlled_sample_inputs_v2,
    greedy_control_validation_indices_v2_1,
    load_tool_decoder_dataset_v2,
)


@pytest.fixture(scope="module")
def dataset():
    return load_tool_decoder_dataset_v2(
        load_config("configs/experiments/gemma4_embodied_tool_decoder_v2.yaml")
    )


def test_v2_dataset_and_finite_evaluation_rows_are_exact(dataset) -> None:
    assert len(dataset.samples) == 6468
    assert len(dataset.train_indices) == 4200
    assert len(dataset.validation_indices) == 2268
    assert {dataset.samples[index].scene_id for index in dataset.train_indices}.isdisjoint(
        dataset.samples[index].scene_id for index in dataset.validation_indices
    )
    causal = causal_validation_indices_v2(dataset)
    controls = greedy_control_validation_indices_v2_1(dataset)
    assert len(causal) == 448
    assert len(controls) == 56
    assert CAUSAL_VALIDATION_SAMPLE_IDS_SHA256 == (
        "a411ffacbbcf0ba348a528e605884f1309ddc837b1bcd0c32ac1af2446e4a622"
    )
    assert GREEDY_CONTROL_SAMPLE_IDS_SHA256_V2_1 == (
        "1f83bd0479016fb12cca0cc835af01f395f124ce6ade1e0afffcd3a83c867a5b"
    )


def test_v2_sampler_and_every_wrong_zero_control_are_real(dataset) -> None:
    schedule = action_balanced_schedule_v2(dataset, microbatch_count=512, seed=2026081218)
    counts = {
        action: sum(dataset.samples[index].action_name == action for index in schedule)
        for action in ("stop", "scan", "turn", "move_forward", "move_backward")
    }
    assert max(counts.values()) - min(counts.values()) <= 1
    index = dataset.validation_indices[0]
    base = controlled_sample_inputs_v2(dataset, index, control="primary")
    expected_changed = {
        "wrong_scene": 0,
        "zero_scene": 0,
        "wrong_robot": 0,
        "zero_robot": 0,
        "wrong_target": 1,
        "zero_target": 1,
        "wrong_clearance": 2,
        "zero_clearance": 2,
    }
    for control, changed in expected_changed.items():
        values = controlled_sample_inputs_v2(dataset, index, control=control)
        assert not torch.equal(values[changed], base[changed])
        assert all(
            torch.equal(values[position], base[position])
            for position in range(3)
            if position != changed
        )


def test_v2_1_preregistration_caps_greedy_and_forbids_heavy_execution() -> None:
    contract = build_evaluation_preregistration_v2_1()
    assert contract["supersedes_resource_plan_only"]["sha256"] == PARENT_EVALUATION_SHA256
    assert contract["bounded_greedy_generation"]["total_unique_sequences"] == 896
    assert contract["bounded_greedy_generation"]["hard_maximum_total_unique_sequences"] == 1024
    assert contract["answer_tail_memory_contract"] == {
        "training_and_teacher_forcing_labels_passed_to_model": False,
        "labels_used_only_to_locate_contiguous_answer_suffix": True,
        "model_logits_to_keep": "answer_label_positions_minus_one",
        "selected_logits_shape": "[1,answer_token_count,vocabulary_size]",
        "full_sequence_vocabulary_logits_materialized_during_training": False,
        "cross_entropy_dtype": "float32",
        "token_normalized_objective_unchanged": True,
        "real_one_row_full_vs_tail_nll_equivalence_tolerance": 1e-6,
        "real_one_row_equivalence_required_before_optimizer_construction": True,
        "tail_gradient_required_before_optimizer_construction": True,
    }
    assert contract["execution"] == {
        "checkpoint_published": False,
        "full_model_loaded": False,
        "greedy_generations": 0,
        "mps_used": False,
        "optimizer_steps": 0,
        "teacher_forced_forwards": 0,
    }


def test_answer_tail_selection_and_full_reference_are_exact() -> None:
    generator = torch.Generator().manual_seed(2026081218)
    logits = torch.randn(1, 13, 29, generator=generator)
    labels = torch.full((1, 13), -100, dtype=torch.long)
    labels[0, -5:] = torch.tensor([2, 7, 5, 11, 3])
    label_positions, causal_positions = answer_tail_positions(labels)
    reference = reference_answer_tail_from_full_logits(logits, labels)
    manual = torch.nn.functional.cross_entropy(
        logits[0, causal_positions].float(), labels[0, label_positions], reduction="none"
    )
    assert torch.equal(reference.per_token_nll, manual)
    assert torch.equal(reference.mean_nll, manual.sum() / 5)
    labels[0, -4] = -100
    with pytest.raises(ValueError, match="contiguous suffix"):
        answer_tail_positions(labels)


def test_answer_tail_model_call_explicitly_uses_labels_none_and_selected_positions() -> None:
    prepared = SimpleNamespace(
        inputs_embeds=torch.randn(1, 8, 4),
        per_layer_inputs=torch.randn(1, 8, 2, 3),
        attention_mask=torch.ones(1, 8, dtype=torch.long),
        mm_token_type_ids=torch.zeros(1, 8, dtype=torch.long),
        labels=torch.tensor([[-100, -100, -100, -100, -100, 4, 5, 6]]),
    )
    kwargs, positions = answer_tail_model_kwargs(prepared)
    assert kwargs["labels"] is None
    assert torch.equal(kwargs["logits_to_keep"], torch.tensor([4, 5, 6]))
    assert torch.equal(positions, torch.tensor([5, 6, 7]))

    class FakeModel:
        def __init__(self) -> None:
            self.kwargs = None

        def __call__(self, **values):
            self.kwargs = values
            return SimpleNamespace(logits=torch.randn(1, 3, 17))

    model = FakeModel()
    language = SimpleNamespace(backend_name="gemma4", model=model)
    tail = answer_tail_forward(language, prepared)
    assert model.kwargs["labels"] is None
    assert tail.logits.shape == (1, 3, 17)
    assert tail.targets.tolist() == [4, 5, 6]


def _perfect_teacher_row(dataset, index: int, control: str):
    sample = dataset.samples[index]
    degraded = control != "primary"
    return {
        "sample_id": sample.sample_id,
        "control": control,
        "token_nll_sum": 5.0 if degraded else 0.1,
        "answer_token_count": 10,
        "answer_token_correct": 5 if degraded else 10,
        "exact_sequence": not degraded,
        "teacher_forced_argmax_valid_schema": not degraded,
        "teacher_forced_argmax_canonical": not degraded,
        "teacher_forced_argmax_tool": None if degraded else sample.action_name,
        "teacher_forced_argmax_tool_correct": not degraded,
    }


def test_teacher_forced_all_row_and_causal_gates_are_operational(dataset) -> None:
    all_rows = evaluate_all_heldout_teacher_forced_v2(dataset, _perfect_teacher_row)
    assert all_rows["sample_count"] == 2268
    assert teacher_forced_gate_results_v2(all_rows)["passed"] is True
    causal = evaluate_teacher_forced_causal_controls_v2(dataset, _perfect_teacher_row)
    assert causal["condition_count"] == len(CONTROL_MODES) == 9
    assert causal["teacher_forced_forward_count"] == 4032
    assert teacher_forced_causal_gate_results_v2(causal)["passed"] is True


def test_trainer_never_uses_full_sequence_loss_outside_one_row_equivalence() -> None:
    source = Path(
        "src/semantic_3d_chat/training/train_gemma4_tool_decoder_v2.py"
    ).read_text(encoding="utf-8")
    evaluation = Path(
        "src/semantic_3d_chat/evaluation/gemma4_tool_decoder_v2_evaluation.py"
    ).read_text(encoding="utf-8")
    assert source.count("prefix_backend.prefill(prepared, use_cache=False)") == 1
    assert ".loss.float()" not in source
    assert "answer_tail_forward(language, prepared).mean_nll.float()" in source
    assert "answer_tail_forward(language, prepared)" in evaluation
