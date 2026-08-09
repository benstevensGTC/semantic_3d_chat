from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from semantic_3d_chat.config import load_config
from semantic_3d_chat.data.dataset import QARecord
from semantic_3d_chat.scene_encoder.perceiver import spatial_anchors
from semantic_3d_chat.training.losses import (
    nearest_spatial_anchor_indices,
    spatial_scene_answer_contrastive_loss,
)
from semantic_3d_chat.training.pair_curriculum import CounterfactualPairUnit
from semantic_3d_chat.training.train_adapter import (
    pair_spatial_answer_contrastive_objective,
    spatial_answer_contrastive_settings,
    spatial_answer_resume_contract_mismatch,
    spatial_answer_target_audit,
)


def test_metric_target_selects_exact_matching_global_spatial_anchor() -> None:
    anchors = spatial_anchors(16)
    expected_index = 11
    room_min = torch.tensor([-3.0, -2.0, 0.0])
    room_max = torch.tensor([3.0, 4.0, 3.0])
    metric_target = (
        anchors[expected_index].add(1.0).div(2.0).mul(room_max - room_min).add(room_min)
    ).unsqueeze(0)

    selected = nearest_spatial_anchor_indices(metric_target, room_min, room_max, latent_count=16)

    assert selected.tolist() == [expected_index]


def test_spatial_answer_hinge_penalizes_swapped_preference_and_reaches_selected_tokens() -> None:
    anchor_indices = torch.tensor([1, 3])
    own = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], requires_grad=True)
    alternate = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], requires_grad=True)
    preferred_tokens = torch.zeros(2, 4, 3)
    preferred_tokens[0, 1] = own.detach()[0]
    preferred_tokens[1, 3] = own.detach()[1]
    preferred_tokens.requires_grad_()
    swapped_tokens = torch.zeros(2, 4, 3)
    swapped_tokens[0, 1] = alternate.detach()[0]
    swapped_tokens[1, 3] = alternate.detach()[1]
    swapped_tokens.requires_grad_()

    preferred_loss, preferred_diagnostics = spatial_scene_answer_contrastive_loss(
        preferred_tokens, anchor_indices, own, alternate, margin=0.2
    )
    swapped_loss, swapped_diagnostics = spatial_scene_answer_contrastive_loss(
        swapped_tokens, anchor_indices, own, alternate, margin=0.2
    )

    assert float(preferred_loss.detach()) == pytest.approx(0.0)
    assert float(swapped_loss.detach()) == pytest.approx(1.2)
    assert preferred_diagnostics["achieved_margin"].tolist() == pytest.approx([1.0, 1.0])
    assert swapped_diagnostics["achieved_margin"].tolist() == pytest.approx([-1.0, -1.0])
    swapped_loss.backward()
    assert swapped_tokens.grad is not None
    assert float(swapped_tokens.grad[0, 1].abs().sum()) > 0.0
    assert float(swapped_tokens.grad[1, 3].abs().sum()) > 0.0
    assert float(swapped_tokens.grad[:, [0, 2]].abs().sum()) == 0.0
    # Frozen-answer targets do not receive gradients; only continuous scene
    # tokens (and therefore their scene encoder) are optimized by this loss.
    assert own.grad is None
    assert alternate.grad is None


def test_spatial_answer_objective_is_default_off_and_v3_is_isolated() -> None:
    default = spatial_answer_contrastive_settings(load_config("configs/default.yaml"))
    v2_config = load_config("configs/experiments/gemma4_color_wiring_v2.yaml")
    v3_config = load_config("configs/experiments/gemma4_color_wiring_v3.yaml")

    assert default == {"weight": 0.0, "margin": 0.2}
    assert spatial_answer_contrastive_settings(v2_config) == default
    assert spatial_answer_contrastive_settings(v3_config) == {
        "weight": 16.0,
        "margin": 0.1,
    }
    assert v2_config["training"]["output_namespace"] == "gemma4_color_wiring_v2"
    assert v3_config["training"]["output_namespace"] == "gemma4_color_wiring_v3"

    invalid_weight = load_config("configs/default.yaml")
    invalid_weight["training"]["spatial_answer_contrastive_weight"] = -1.0
    with pytest.raises(ValueError, match="weight cannot be negative"):
        spatial_answer_contrastive_settings(invalid_weight)
    invalid_margin = load_config("configs/default.yaml")
    invalid_margin["training"]["spatial_answer_contrastive_margin"] = 2.1
    with pytest.raises(ValueError, match="margin must be in"):
        spatial_answer_contrastive_settings(invalid_margin)


def test_resume_contract_accepts_legacy_off_but_rejects_missing_or_changed_enabled_loss() -> None:
    disabled = {"weight": 0.0, "margin": 0.2}
    enabled = {"weight": 5.0, "margin": 0.2}

    assert spatial_answer_resume_contract_mismatch({}, disabled) is None
    assert (
        spatial_answer_resume_contract_mismatch({"spatial_answer_contrastive": enabled}, enabled)
        is None
    )
    assert spatial_answer_resume_contract_mismatch({}, enabled) == {
        "checkpoint": None,
        "runtime": enabled,
    }
    assert spatial_answer_resume_contract_mismatch(
        {"spatial_answer_contrastive": {"weight": 3.0, "margin": 0.2}}, enabled
    ) == {
        "checkpoint": {"weight": 3.0, "margin": 0.2},
        "runtime": enabled,
    }


def test_target_audit_exposes_repeated_questions_over_one_physical_target() -> None:
    def record(scene_id: str, question_id: str, role: str, answer: str) -> QARecord:
        return QARecord(
            scene_id=scene_id,
            question_id=question_id,
            question="What color is it?",
            answer=answer,
            answer_type="attribute",
            target_xyz=[1.0, 2.0, 0.5],
            counterfactual_pair_id="pair_1",
            counterfactual_question_key=question_id,
            counterfactual_expected_change=True,
            counterfactual_role=role,
            counterfactual_change_type="color_swap",
        )

    units = []
    for index in range(2):
        reference = record("scene_a", f"q_{index}", "reference", "red")
        counterfactual = record("scene_b", f"q_{index}", "counterfactual", "blue")
        units.append(
            CounterfactualPairUnit(
                pair_id="pair_1",
                question_key=f"q_{index}",
                reference=reference,
                counterfactual=counterfactual,
            )
        )

    assert spatial_answer_target_audit(units) == {
        "eligible_unit_count": 2,
        "eligible_side_count": 4,
        "unique_target_count": 1,
        "unit_to_unique_target_ratio": 2.0,
    }


def test_pair_training_bridge_uses_final_scene_token_and_one_token_answers() -> None:
    class Tokenizer:
        eos_token_id = 2

        def __call__(self, text: str, **kwargs) -> SimpleNamespace:
            assert kwargs["add_special_tokens"] is False
            token_id = {"red": 0, "blue": 1}[text]
            return SimpleNamespace(input_ids=torch.tensor([[token_id]]))

    embedding = torch.nn.Embedding(3, 3)
    with torch.no_grad():
        embedding.weight.copy_(torch.eye(3))
    reference = QARecord(
        scene_id="scene_a",
        question_id="q_a",
        question="What color is the target?",
        answer="red",
        answer_type="attribute",
        target_xyz=[0.0, 0.0, 0.0],
        counterfactual_pair_id="pair_1",
        counterfactual_question_key="color",
        counterfactual_expected_change=True,
        counterfactual_role="reference",
        counterfactual_change_type="color_swap",
    )
    counterfactual = QARecord(
        **{
            **reference.__dict__,
            "scene_id": "scene_b",
            "question_id": "q_b",
            "answer": "blue",
            "counterfactual_role": "counterfactual",
        }
    )
    unit = CounterfactualPairUnit("pair_1", "color", reference, counterfactual)
    room_min = torch.tensor([-1.0, -1.0, -1.0])
    room_max = torch.tensor([1.0, 1.0, 1.0])
    target_index = int(
        nearest_spatial_anchor_indices(torch.zeros(1, 3), room_min, room_max, latent_count=8)[0]
    )
    reference_tokens = torch.zeros(1, 8, 3, requires_grad=True)
    counterfactual_tokens = torch.zeros(1, 8, 3, requires_grad=True)
    with torch.no_grad():
        reference_tokens[0, target_index, 0] = 1.0
        counterfactual_tokens[0, target_index, 1] = 1.0
    outputs = {
        "scene_a": SimpleNamespace(scene_tokens=reference_tokens),
        "scene_b": SimpleNamespace(scene_tokens=counterfactual_tokens),
    }
    maps = {scene_id: SimpleNamespace(room_min=room_min, room_max=room_max) for scene_id in outputs}
    language = SimpleNamespace(
        model=SimpleNamespace(get_input_embeddings=lambda: embedding),
        tokenizer=Tokenizer(),
        device=torch.device("cpu"),
    )

    loss, diagnostics = pair_spatial_answer_contrastive_objective(
        outputs,
        [unit],
        maps,
        language,
        latent_count=8,
        margin=0.1,
    )

    assert float(loss.detach()) == pytest.approx(0.0)
    assert diagnostics["anchor_indices"].tolist() == [target_index, target_index]
    assert diagnostics["unique_target_count"] == 1
