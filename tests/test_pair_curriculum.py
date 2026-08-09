from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from semantic_3d_chat.config import load_config
from semantic_3d_chat.data.dataset import QARecord
from semantic_3d_chat.language.prefix_injection import ContinuousPrefixComposer
from semantic_3d_chat.training.pair_curriculum import (
    build_epoch_curriculum,
    build_exact_question_pair_units,
    candidate_logit_margins,
    cap_pair_units_per_pair,
    differing_answer_token_masks,
    first_answer_token_full_vocab_margins,
    pair_curriculum_settings,
    pair_gate_metrics,
    pair_ranking_hinge,
    restrict_labels_to_answer_mask,
    select_pair_only_records,
    single_differing_answer_token,
    token_normalized_nll,
)
from semantic_3d_chat.training.train_adapter import (
    best_pair_gate_passed_from_history,
    combine_pair_training_losses,
    pair_batch_objective,
    pair_gate_checkpoint_improved,
    pair_gate_monitor_value,
    should_stop_after_pair_gate,
)


def _paired_record(
    scene_id: str,
    role: str,
    *,
    pair_id: str = "pair_1",
    key: str = "question_1",
    question: str = "Which side?",
    answer: str,
) -> QARecord:
    return QARecord(
        scene_id=scene_id,
        question_id=f"q_{scene_id}_{key}",
        question=question,
        answer=answer,
        answer_type="spatial_relation",
        target_xyz=None,
        counterfactual_pair_id=pair_id,
        counterfactual_question_key=key,
        counterfactual_expected_change=True,
        counterfactual_role=role,
        counterfactual_change_type="synthetic",
    )


def _unit_records(key: str = "question_1", pair_id: str = "pair_1") -> tuple[QARecord, QARecord]:
    return (
        _paired_record("scene_a", "reference", pair_id=pair_id, key=key, answer="left"),
        _paired_record("scene_b", "counterfactual", pair_id=pair_id, key=key, answer="right"),
    )


def test_exact_question_units_require_two_scenes_roles_and_changed_answers() -> None:
    first, second = _unit_records()
    unit = build_exact_question_pair_units([second, first])[0]

    assert unit.pair_id == "pair_1"
    assert unit.scene_ids == ("scene_a", "scene_b")
    assert unit.reference.answer == "left"
    assert unit.counterfactual.answer == "right"

    with pytest.raises(ValueError, match="incomplete"):
        build_exact_question_pair_units([first])
    with pytest.raises(ValueError, match="exact same question"):
        build_exact_question_pair_units(
            [
                first,
                _paired_record(
                    "scene_b",
                    "counterfactual",
                    question="Different wording?",
                    answer="right",
                ),
            ]
        )
    with pytest.raises(ValueError, match="identical answers"):
        build_exact_question_pair_units(
            [first, _paired_record("scene_b", "counterfactual", answer="left")]
        )


def test_pair_only_selection_excludes_unpaired_and_unconfigured_scenes() -> None:
    first, second = _unit_records()
    unrelated = QARecord(
        scene_id="scene_a",
        question_id="q_unrelated",
        question="Is anything present?",
        answer="yes",
        answer_type="presence",
        target_xyz=None,
    )
    other_first, other_second = _unit_records("other", "pair_2")
    other_first = QARecord(**{**other_first.__dict__, "scene_id": "scene_c"})
    other_second = QARecord(**{**other_second.__dict__, "scene_id": "scene_d"})

    selected = select_pair_only_records(
        [first, unrelated, other_first, second, other_second], ["scene_a", "scene_b"]
    )

    assert selected == [first, second]


def test_pair_unit_cap_is_deterministic_complete_and_balanced_by_pair() -> None:
    records: list[QARecord] = []
    for pair_id in ("pair_1", "pair_2"):
        for index in range(5):
            first, second = _unit_records(f"question_{index}", pair_id)
            if pair_id == "pair_2":
                first = QARecord(**{**first.__dict__, "scene_id": "scene_c"})
                second = QARecord(**{**second.__dict__, "scene_id": "scene_d"})
            records.extend((first, second))

    selected = cap_pair_units_per_pair(records, 2, seed=17)
    units = build_exact_question_pair_units(selected)

    assert selected == cap_pair_units_per_pair(records, 2, seed=17)
    assert len(selected) == 8
    assert len(units) == 4
    assert {unit.pair_id for unit in units} == {"pair_1", "pair_2"}
    assert all(
        sum(unit.pair_id == pair_id for unit in units) == 2 for pair_id in ("pair_1", "pair_2")
    )
    assert cap_pair_units_per_pair(records, None, seed=17) == records
    with pytest.raises(ValueError, match="positive"):
        cap_pair_units_per_pair(records, 0, seed=17)


def test_epoch_curriculum_interleaves_two_scene_pair_batches_at_requested_fraction() -> None:
    records = []
    for index in range(4):
        records.extend(_unit_records(f"question_{index}"))
    units = build_exact_question_pair_units(records)
    records_by_scene = {
        scene_id: [record for record in records if record.scene_id == scene_id]
        for scene_id in ("scene_a", "scene_b")
    }

    batches = build_epoch_curriculum(
        records_by_scene,
        units,
        standard_batch_size=2,
        pair_units_per_batch=2,
        pair_batch_fraction=0.5,
        pair_only=False,
        seed=17,
        steps_per_epoch=8,
    )

    assert len(batches) == 8
    assert sum(batch.kind == "pair" for batch in batches) == 4
    assert batches == build_epoch_curriculum(
        records_by_scene,
        units,
        standard_batch_size=2,
        pair_units_per_batch=2,
        pair_batch_fraction=0.5,
        pair_only=False,
        seed=17,
        steps_per_epoch=8,
    )
    for batch in (item for item in batches if item.kind == "pair"):
        assert batch.pair_id == "pair_1"
        assert all(unit.scene_ids == ("scene_a", "scene_b") for unit in batch.pair_units)

    with pytest.raises(ValueError, match=">= 0.5"):
        build_epoch_curriculum(
            records_by_scene,
            units,
            standard_batch_size=2,
            pair_units_per_batch=2,
            pair_batch_fraction=0.25,
            pair_only=False,
            seed=17,
        )


def test_token_normalized_nll_does_not_overweight_longer_answers() -> None:
    logits = torch.zeros(2, 4, 2, requires_grad=True)
    labels = torch.tensor(
        [
            [-100, 0, -100, -100],
            [-100, 0, 1, -100],
        ]
    )

    nll = token_normalized_nll(logits, labels)

    assert torch.allclose(nll, torch.full((2,), math.log(2.0)))
    nll.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_pair_ranking_masks_shared_eos_but_keeps_full_language_labels() -> None:
    first = torch.tensor([[11, 99]])
    second = torch.tensor([[22, 99]])
    first_mask, second_mask = differing_answer_token_masks(first, second)

    assert first_mask.tolist() == [True, False]
    assert second_mask.tolist() == [True, False]
    labels = torch.tensor(
        [
            [-100, -100, 11, 99],
            [-100, -100, 22, 99],
        ]
    )
    full_labels = labels.clone()
    restrict_labels_to_answer_mask(labels, 0, first_mask)
    restrict_labels_to_answer_mask(labels, 1, second_mask)

    assert labels.tolist() == [[-100, -100, 11, -100], [-100, -100, 22, -100]]
    assert full_labels.tolist() == [[-100, -100, 11, 99], [-100, -100, 22, 99]]


def test_candidate_logit_contract_requires_one_aligned_token_change() -> None:
    assert single_differing_answer_token(torch.tensor([[11, 99]]), torch.tensor([[22, 99]])) == (
        0,
        11,
        22,
    )

    with pytest.raises(ValueError, match="equal-length"):
        single_differing_answer_token(torch.tensor([[11, 99]]), torch.tensor([[22]]))
    with pytest.raises(ValueError, match="exactly one differing"):
        single_differing_answer_token(torch.tensor([[11, 98, 99]]), torch.tensor([[22, 97, 99]]))


def test_candidate_logit_margin_equals_single_token_nll_difference_and_has_gradients() -> None:
    logits = torch.tensor(
        [
            [[0.0, 0.0, 0.0, 0.0], [0.0, 1.3, -0.2, 0.0], [0.0, 0.0, 0.0, 2.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.0, -0.4, 0.9, 0.0], [0.0, 0.0, 0.0, 2.0]],
        ],
        requires_grad=True,
    )
    correct_labels = torch.tensor([[-100, -100, 1], [-100, -100, 2]])
    # The answer label is at index 2 and therefore reads logits at index 1.
    margins, own, alternate = candidate_logit_margins(
        logits,
        correct_labels,
        [(0, 1, 2), (0, 2, 1)],
    )
    alternate_labels = torch.tensor([[-100, -100, 2], [-100, -100, 1]])
    correct_nll = token_normalized_nll(logits, correct_labels)
    alternate_nll = token_normalized_nll(logits, alternate_labels)

    assert torch.allclose(margins, alternate_nll - correct_nll)
    assert torch.allclose(margins, own - alternate)
    assert margins.tolist() == pytest.approx([1.5, 1.3])
    (-margins.mean()).backward()
    assert logits.grad is not None
    assert float(logits.grad[:, 1, :].abs().sum()) > 0.0
    assert float(logits.grad[:, 0, :].abs().sum()) == 0.0


def test_full_vocab_first_answer_token_margin_detects_non_candidate_winner() -> None:
    logits = torch.zeros(2, 3, 5)
    labels = torch.tensor([[-100, -100, 1], [-100, -100, 2]])
    # Both target tokens beat their paired alternative, but token 4 beats the
    # first target over the complete vocabulary.
    logits[0, 1] = torch.tensor([0.0, 3.0, 2.0, -1.0, 4.0])
    logits[1, 1] = torch.tensor([0.0, 1.0, 5.0, -1.0, 0.5])

    margins = first_answer_token_full_vocab_margins(logits, labels)

    assert margins.tolist() == pytest.approx([-1.0, 4.0])


def test_full_vocab_margin_hinge_targets_only_strongest_non_target() -> None:
    logits = torch.zeros(1, 3, 5, requires_grad=True)
    labels = torch.tensor([[-100, -100, 1]])
    with torch.no_grad():
        # Target 1 beats paired alternative 2, but unrelated token 4 wins.
        logits[0, 1] = torch.tensor([0.0, 3.0, 2.0, -1.0, 4.0])
    margins = first_answer_token_full_vocab_margins(logits, labels)
    full_vocab_loss = torch.relu(torch.tensor(1.0) - margins).mean()

    full_vocab_loss.backward()

    assert margins.tolist() == pytest.approx([-1.0])
    assert float(logits.grad[0, 1, 1]) == pytest.approx(-1.0)
    assert float(logits.grad[0, 1, 4]) == pytest.approx(1.0)
    assert float(logits.grad[0, 1, 2]) == 0.0
    assert float(logits.grad[0, 0].abs().sum()) == 0.0


def test_pair_training_loss_applies_full_vocab_weight_and_zero_preserves_v10() -> None:
    terms = [torch.tensor(value) for value in (1.0, 2.0, 3.0, 4.0, 5.0)]
    v10 = combine_pair_training_losses(
        *terms,
        pair_ranking_weight=8.0,
        full_vocab_ranking_weight=0.0,
        diversity_weight=0.05,
        scene_separation_weight=20.0,
    )
    v11 = combine_pair_training_losses(
        *terms,
        pair_ranking_weight=8.0,
        full_vocab_ranking_weight=2.0,
        diversity_weight=0.05,
        scene_separation_weight=20.0,
    )

    assert float(v10) == pytest.approx(117.2)
    assert float(v11) == pytest.approx(123.2)
    assert float(v11 - v10) == pytest.approx(6.0)


def test_full_vocab_training_settings_are_opt_in_and_validated() -> None:
    settings = pair_curriculum_settings(
        {
            "training": {
                "pair_ranking_weight": 8.0,
                "pair_full_vocab_ranking_weight": 2.0,
                "pair_full_vocab_ranking_margin": 1.0,
            }
        }
    )

    assert settings.full_vocab_ranking_weight == 2.0
    assert settings.full_vocab_ranking_margin == 1.0
    with pytest.raises(ValueError, match="requires pair_ranking_weight"):
        pair_curriculum_settings({"training": {"pair_full_vocab_ranking_weight": 1.0}})
    with pytest.raises(ValueError, match="margin cannot be negative"):
        pair_curriculum_settings(
            {
                "training": {
                    "pair_ranking_weight": 1.0,
                    "pair_full_vocab_ranking_margin": -0.1,
                }
            }
        )


def test_full_vocab_top1_gate_is_opt_in_and_composes_with_pairwise_gate() -> None:
    pairwise_margins = [[1.0, 1.0]]
    full_vocab_margins = [[-1.0, 4.0]]

    diagnostic_only = pair_gate_metrics(
        pairwise_margins,
        ranking_mode="candidate_logit",
        first_answer_token_full_vocab_margins=full_vocab_margins,
    )
    strict = pair_gate_metrics(
        pairwise_margins,
        ranking_mode="candidate_logit",
        first_answer_token_full_vocab_margins=full_vocab_margins,
        first_answer_token_top1_accuracy_threshold=1.0,
    )
    passed = pair_gate_metrics(
        pairwise_margins,
        ranking_mode="candidate_logit",
        first_answer_token_full_vocab_margins=[[0.25, 0.5]],
        first_answer_token_top1_accuracy_threshold=1.0,
    )

    assert diagnostic_only["pairwise_passed"] is True
    assert diagnostic_only["passed"] is True
    assert strict["pairwise_passed"] is True
    assert strict["first_answer_token_top1_accuracy"] == 0.5
    assert strict["first_answer_token_top1_unit_accuracy"] == 0.0
    assert strict["mean_first_answer_token_target_vs_best_other_logit_margin"] == 1.5
    assert strict["minimum_first_answer_token_target_vs_best_other_logit_margin"] == -1.0
    assert strict["first_answer_token_target_vs_best_other_hinge"] == 0.5
    assert strict["first_answer_token_top1_gate_passed"] is False
    assert strict["passed"] is False
    assert passed["first_answer_token_top1_gate_passed"] is True
    assert passed["passed"] is True


def test_full_vocab_gate_rejects_missing_or_misaligned_diagnostics() -> None:
    with pytest.raises(ValueError, match="requires full-vocabulary margins"):
        pair_gate_metrics(
            [[1.0, 1.0]],
            first_answer_token_top1_accuracy_threshold=1.0,
        )
    with pytest.raises(ValueError, match="must match pair gate shape"):
        pair_gate_metrics(
            [[1.0, 1.0]],
            first_answer_token_full_vocab_margins=[[1.0]],
        )
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        pair_gate_metrics(
            [[1.0, 1.0]],
            first_answer_token_full_vocab_margins=[[1.0, 1.0]],
            first_answer_token_top1_accuracy_threshold=1.1,
        )


def test_pair_hinge_and_candidate_gate_report_flips_without_calling_generation() -> None:
    correct = torch.tensor([[1.0, 2.0]], requires_grad=True)
    swapped = torch.tensor([[1.6, 2.2]], requires_grad=True)
    loss, margins = pair_ranking_hinge(correct, swapped, margin=0.5)

    assert torch.allclose(margins, torch.tensor([[0.6, 0.2]]))
    assert float(loss.detach()) == pytest.approx(0.15)
    loss.backward()
    # Once one side clears the margin it contributes no canceling gradient;
    # the still-wrong side receives the full directional correction.
    assert float(correct.grad[0, 0]) == 0.0
    assert float(swapped.grad[0, 0]) == 0.0
    assert float(correct.grad[0, 1]) > 0.0
    assert float(swapped.grad[0, 1]) < 0.0

    metrics = pair_gate_metrics(
        [[0.8, 0.7], [0.6, 0.9]],
        changed_unit_accuracy_threshold=0.95,
        prediction_flip_threshold=1.0,
        wrong_prefix_flip_threshold=1.0,
        ranking_margin=0.5,
    )
    assert metrics["changed_unit_accuracy"] == 1.0
    assert metrics["prediction_flip_rate"] == 1.0
    assert metrics["wrong_prefix_flip_rate"] == 1.0
    assert metrics["free_generation_evaluated"] is False
    assert metrics["passed"] is True

    failed = pair_gate_metrics([[0.8, -0.1], [-0.2, -0.3]])
    assert failed["changed_unit_accuracy"] == 0.0
    assert failed["prediction_flip_rate"] == 0.5
    assert failed["wrong_prefix_flip_rate"] == 0.0
    assert failed["passed"] is False

    candidate = pair_gate_metrics([[0.8, 0.7]], ranking_mode="candidate_logit")
    assert candidate["evaluation_type"] == (
        "teacher_forced_same_distribution_candidate_logit_ranking"
    )
    assert candidate["same_next_token_distribution"] is True
    assert candidate["mean_own_vs_alternate_candidate_logit_margin"] == pytest.approx(0.75)
    assert "mean_correct_vs_swapped_nll_margin" not in candidate


def test_pair_gate_config_is_explicit_and_defaults_remain_disabled() -> None:
    default = pair_curriculum_settings(load_config("configs/default.yaml"))
    gate = pair_curriculum_settings(load_config("configs/experiments/pair_gate.yaml"))

    assert default.enabled is False
    assert default.ranking_margin == 0.5
    assert default.ranking_mode == "nll"
    assert default.batch_fraction == 0.0
    assert gate.enabled is True
    assert gate.pair_only is True
    assert gate.batch_fraction == 1.0
    assert gate.ranking_margin == 0.5
    assert gate.ranking_mode == "nll"
    assert gate.changed_unit_accuracy_threshold == 0.95
    assert gate.prediction_flip_threshold == 1.0
    assert gate.wrong_prefix_flip_threshold == 1.0
    assert gate.stop_when_gate_passes is True
    assert gate.first_answer_token_top1_accuracy_threshold is None
    discriminative_config = load_config("configs/experiments/pair_gate_discriminative.yaml")
    discriminative = pair_curriculum_settings(discriminative_config)
    assert discriminative.ranking_weight == 8.0
    assert discriminative_config["training"]["gradient_accumulation"] == 1
    assert discriminative_config["training"]["grounding_weight"] == 0.0
    assert discriminative_config["training"]["paired_scene_separation_weight"] == 20.0
    gemma_v1 = load_config("configs/experiments/gemma4_color_wiring.yaml")
    gemma_v2 = load_config("configs/experiments/gemma4_color_wiring_v2.yaml")
    gemma_v8 = load_config("configs/experiments/gemma4_color_wiring_v8.yaml")
    gemma_v9 = load_config("configs/experiments/gemma4_color_wiring_v9.yaml")
    gemma_v10 = load_config("configs/experiments/gemma4_color_mirror_wiring_v10.yaml")
    gemma_v11 = load_config("configs/experiments/gemma4_color_mirror_full_vocab_v11.yaml")
    assert pair_curriculum_settings(gemma_v1).ranking_mode == "nll"
    assert pair_curriculum_settings(gemma_v2).ranking_mode == "candidate_logit"
    assert gemma_v1["training"]["output_namespace"] == "gemma4_color_wiring"
    assert gemma_v2["training"]["output_namespace"] == "gemma4_color_wiring_v2"
    assert gemma_v2["scene_encoder"]["learned_scene_token_scale"] == 0.25
    assert pair_curriculum_settings(gemma_v8).first_answer_token_top1_accuracy_threshold is None
    assert pair_curriculum_settings(gemma_v8).stop_when_gate_passes is True
    assert pair_curriculum_settings(gemma_v9).first_answer_token_top1_accuracy_threshold == 1.0
    assert pair_curriculum_settings(gemma_v9).stop_when_gate_passes is False
    assert gemma_v9["training"]["output_namespace"] == "gemma4_color_wiring_v9"
    assert gemma_v9["training"]["epochs"] == 36
    v10_curriculum = pair_curriculum_settings(gemma_v10)
    assert v10_curriculum.max_units_per_pair == 6
    assert v10_curriculum.pair_only_scene_ids == (
        "scene_000003",
        "scene_000004",
        "scene_000007",
        "scene_000008",
    )
    assert gemma_v10["training"]["gradient_accumulation"] == 12
    assert gemma_v10["training"]["initialize_from"].endswith("epoch_036")
    v11_curriculum = pair_curriculum_settings(gemma_v11)
    assert gemma_v11["training"]["output_namespace"] == ("gemma4_color_mirror_full_vocab_v11")
    assert v11_curriculum.full_vocab_ranking_weight == 2.0
    assert v11_curriculum.full_vocab_ranking_margin == 1.0
    assert gemma_v11["training"]["initialize_from"].endswith("epoch_036")


def test_pair_gate_best_checkpoint_selection_is_gate_pass_first() -> None:
    assert pair_gate_checkpoint_improved(
        monitor_value=0.8,
        best_monitor_value=0.1,
        min_delta=0.0,
        gate_passed=True,
        best_gate_passed=False,
    )
    assert not pair_gate_checkpoint_improved(
        monitor_value=0.01,
        best_monitor_value=0.8,
        min_delta=0.0,
        gate_passed=False,
        best_gate_passed=True,
    )
    assert best_pair_gate_passed_from_history(
        [{"epoch": 3, "pair_candidate_gate": {"passed": True}}], 3
    )
    assert not best_pair_gate_passed_from_history(
        [{"epoch": 3, "pair_candidate_gate": {"passed": True}}], 2
    )
    assert should_stop_after_pair_gate(True, {"pairwise_passed": True, "passed": True})
    assert not should_stop_after_pair_gate(True, {"pairwise_passed": True, "passed": False})
    assert not should_stop_after_pair_gate(False, {"pairwise_passed": True, "passed": True})


def test_full_vocab_gate_monitor_prefers_larger_minimum_passing_margin() -> None:
    failed = {
        "passed": False,
        "first_answer_token_target_vs_best_other_hinge": 0.4,
        "ranking_hinge_at_configured_margin": 0.2,
        "minimum_first_answer_token_target_vs_best_other_logit_margin": -1.0,
    }
    barely_passed = {
        "passed": True,
        "first_answer_token_target_vs_best_other_hinge": 0.0,
        "ranking_hinge_at_configured_margin": 0.0,
        "minimum_first_answer_token_target_vs_best_other_logit_margin": 0.03,
    }
    stronger_passed = {
        **barely_passed,
        "minimum_first_answer_token_target_vs_best_other_logit_margin": 1.0,
    }

    assert pair_gate_monitor_value(failed, full_vocab_gate=True) == pytest.approx(0.6)
    assert pair_gate_monitor_value(barely_passed, full_vocab_gate=True) == pytest.approx(-0.03)
    assert pair_gate_monitor_value(stronger_passed, full_vocab_gate=True) == pytest.approx(-1.0)


class _TinyTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, *args, **kwargs) -> torch.Tensor:
        del args, kwargs
        return torch.tensor([[0]])

    def __call__(self, text: str, **kwargs) -> SimpleNamespace:
        del kwargs
        if text.startswith("left"):
            ids = [1, 3]
        elif text.startswith("right"):
            ids = [2, 3]
        else:
            ids = [0]
        return SimpleNamespace(input_ids=torch.tensor([ids]))


class _TinyCausalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(4, 2)
        self.projection = torch.nn.Linear(2, 4, bias=False)
        self.forward_calls = 0
        torch.nn.init.zeros_(self.embedding.weight)
        torch.nn.init.zeros_(self.projection.weight)
        with torch.no_grad():
            self.projection.weight[1, 0] = 1.0
            self.projection.weight[2, 0] = -1.0

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.embedding

    def forward(self, *, inputs_embeds, attention_mask, labels=None, use_cache):
        del attention_mask, labels, use_cache
        self.forward_calls += 1
        hidden = inputs_embeds.cumsum(dim=1)
        return SimpleNamespace(logits=self.projection(hidden))


def test_pair_batch_objective_places_both_prefix_sides_against_swapped_answers() -> None:
    records = _unit_records()
    unit = build_exact_question_pair_units(records)[0]
    model = _TinyCausalModel()
    composer = ContinuousPrefixComposer(2)
    with torch.no_grad():
        composer.scene_start.zero_()
        composer.scene_end.zero_()
    reference_tokens = torch.tensor([[[0.2, 0.0]]], requires_grad=True)
    counterfactual_tokens = torch.tensor([[[-0.2, 0.0]]], requires_grad=True)
    outputs = {
        "scene_a": SimpleNamespace(scene_tokens=reference_tokens, native_latents=reference_tokens),
        "scene_b": SimpleNamespace(
            scene_tokens=counterfactual_tokens, native_latents=counterfactual_tokens
        ),
    }
    language = SimpleNamespace(
        model=model,
        tokenizer=_TinyTokenizer(),
        device=torch.device("cpu"),
    )

    base, language_loss, grounding_loss, ranking_loss, diagnostics = pair_batch_objective(
        outputs,
        [unit],
        {},
        language,
        composer,
        torch.nn.Identity(),
        {"language": {"system_prompt": "stable"}, "training": {"grounding_weight": 0.0}},
        ranking_margin=0.5,
    )

    assert float(grounding_loss) == 0.0
    assert diagnostics["margins"].shape == (1, 2)
    assert torch.all(diagnostics["margins"] > 0.0)
    assert torch.all(diagnostics["margins"] < 0.5)
    assert diagnostics["ranking_tokens_per_side"].tolist() == [[1, 1]]
    assert float(diagnostics["unit_accuracy"]) == 1.0
    assert float(ranking_loss.detach()) > 0.0
    assert float(language_loss.detach()) == pytest.approx(float(base.detach()))
    ranking_loss.backward()
    assert reference_tokens.grad is not None and reference_tokens.grad.abs().sum() > 0
    assert counterfactual_tokens.grad is not None and counterfactual_tokens.grad.abs().sum() > 0
    assert model.forward_calls == 2


def test_candidate_logit_pair_objective_uses_only_correct_forward_and_backpropagates() -> None:
    records = _unit_records()
    unit = build_exact_question_pair_units(records)[0]
    model = _TinyCausalModel()
    composer = ContinuousPrefixComposer(2)
    with torch.no_grad():
        composer.scene_start.zero_()
        composer.scene_end.zero_()
    reference_tokens = torch.tensor([[[0.2, 0.0]]], requires_grad=True)
    counterfactual_tokens = torch.tensor([[[-0.2, 0.0]]], requires_grad=True)
    outputs = {
        "scene_a": SimpleNamespace(scene_tokens=reference_tokens, native_latents=reference_tokens),
        "scene_b": SimpleNamespace(
            scene_tokens=counterfactual_tokens, native_latents=counterfactual_tokens
        ),
    }
    language = SimpleNamespace(
        model=model,
        tokenizer=_TinyTokenizer(),
        device=torch.device("cpu"),
    )

    base, language_loss, grounding_loss, ranking_loss, diagnostics = pair_batch_objective(
        outputs,
        [unit],
        {},
        language,
        composer,
        torch.nn.Identity(),
        {"language": {"system_prompt": "stable"}, "training": {"grounding_weight": 0.0}},
        ranking_margin=0.5,
        ranking_mode="candidate_logit",
        collect_full_vocab_first_answer_token=True,
        full_vocab_ranking_margin=10.0,
    )

    assert model.forward_calls == 1
    assert float(grounding_loss) == 0.0
    assert float(language_loss.detach()) == pytest.approx(float(base.detach()))
    assert diagnostics["ranking_mode"] == "candidate_logit"
    assert diagnostics["evaluation_type"] == (
        "teacher_forced_same_distribution_candidate_logit_ranking"
    )
    assert diagnostics["margin_name"] == "own_vs_alternate_candidate_logit_margin"
    assert diagnostics["correct_nll"] is None
    assert diagnostics["swapped_nll"] is None
    assert diagnostics["own_candidate_logits"] is not None
    assert diagnostics["alternate_candidate_logits"] is not None
    assert torch.all(diagnostics["first_answer_token_full_vocab_margins"] > 0.0)
    full_vocab_loss = diagnostics["first_answer_token_full_vocab_ranking_loss"]
    assert float(full_vocab_loss.detach()) > 0.0
    assert diagnostics["ranking_tokens_per_side"].tolist() == [[1, 1]]
    assert torch.all(diagnostics["margins"] > 0.0)
    assert float(ranking_loss.detach()) > 0.0
    (ranking_loss + full_vocab_loss).backward()
    assert reference_tokens.grad is not None and reference_tokens.grad.abs().sum() > 0
    assert counterfactual_tokens.grad is not None and counterfactual_tokens.grad.abs().sum() > 0
