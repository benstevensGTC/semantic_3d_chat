from __future__ import annotations

import pytest
import torch

import semantic_3d_chat.training.train_adapter as training_module
from semantic_3d_chat.config import load_config
from semantic_3d_chat.data.dataset import QARecord
from semantic_3d_chat.training.losses import (
    QuestionGroundingHead,
    latent_diversity_loss,
    paired_scene_separation_loss,
)
from semantic_3d_chat.training.train_adapter import (
    anti_collapse_settings,
    construct_scene_tokenizer,
    training_counterfactual_scene_pairs,
)


def _record(scene_id: str, pair_id: str | None) -> QARecord:
    return QARecord(
        scene_id=scene_id,
        question_id=f"q_{scene_id}",
        question="Synthetic training question?",
        answer="yes",
        answer_type="presence",
        target_xyz=None,
        counterfactual_pair_id=pair_id,
    )


def test_diversity_loss_penalizes_collapsed_latents_and_has_finite_gradients() -> None:
    collapsed = torch.ones(1, 8, 8, requires_grad=True)
    diverse = torch.eye(8).unsqueeze(0).requires_grad_()

    collapsed_loss, diagnostics = latent_diversity_loss(collapsed, cosine_margin=0.2)
    diverse_loss, _ = latent_diversity_loss(diverse, cosine_margin=0.2)

    assert collapsed_loss > diverse_loss
    assert torch.isfinite(collapsed_loss)
    assert diagnostics["sampled_latent_count"] == 8
    assert torch.isclose(diagnostics["mean_off_diagonal_cosine"], torch.tensor(1.0))
    collapsed_loss.backward()
    assert collapsed.grad is not None
    assert torch.isfinite(collapsed.grad).all()


def test_diversity_subsampling_is_deterministic_and_gradients_reach_latents() -> None:
    generator = torch.Generator().manual_seed(17)
    latents = torch.randn(2, 32, 16, generator=generator, requires_grad=True)

    first, first_diagnostics = latent_diversity_loss(latents, cosine_margin=0.0, max_latents=7)
    second, second_diagnostics = latent_diversity_loss(latents, cosine_margin=0.0, max_latents=7)

    assert torch.equal(first, second)
    assert first_diagnostics["sampled_latent_count"] == 7
    assert second_diagnostics["sampled_latent_count"] == 7
    first.backward()
    assert latents.grad is not None
    assert torch.isfinite(latents.grad).all()
    assert latents.grad.abs().sum() > 0


def test_paired_scene_separation_loss_rewards_sensitivity_with_finite_gradients() -> None:
    generator = torch.Generator().manual_seed(23)
    first = torch.randn(1, 16, 12, generator=generator, requires_grad=True)
    nearly_same = (
        first.detach() + 0.001 * torch.randn(1, 16, 12, generator=generator)
    ).requires_grad_()
    opposite = -first.detach()

    close_loss, close_diagnostics = paired_scene_separation_loss(
        first, nearly_same, cosine_distance_margin=0.1
    )
    separated_loss, separated_diagnostics = paired_scene_separation_loss(
        first.detach(), opposite, cosine_distance_margin=0.1
    )

    assert close_loss > separated_loss
    assert close_diagnostics["cosine_distance"] < 0.1
    assert separated_diagnostics["cosine_distance"] > 0.1
    close_loss.backward()
    assert first.grad is not None and nearly_same.grad is not None
    assert torch.isfinite(first.grad).all()
    assert torch.isfinite(nearly_same.grad).all()


def test_pair_discovery_never_reaches_outside_supplied_training_records() -> None:
    train_records = [
        _record("scene_train_a", "pair_train"),
        _record("scene_train_b", "pair_train"),
        _record("scene_incomplete", "pair_not_in_train"),
    ]

    assert training_counterfactual_scene_pairs(train_records) == [
        ("pair_train", "scene_train_a", "scene_train_b")
    ]


def test_anti_collapse_config_is_opt_in_and_inherits_multiscene_plan() -> None:
    default = load_config("configs/default.yaml")
    enabled = load_config("configs/experiments/multiscene_anticollapse.yaml")

    default_settings = anti_collapse_settings(default)
    enabled_settings = anti_collapse_settings(enabled)
    assert default_settings["latent_diversity_weight"] == 0.0
    assert default_settings["paired_scene_separation_weight"] == 0.0
    assert enabled_settings["latent_diversity_weight"] == 0.01
    assert enabled_settings["paired_scene_separation_weight"] == 0.02
    assert enabled["training"]["max_questions_per_scene"] == 48
    assert enabled["batch"]["expected_scene_count"] == 10
    assert enabled["scene_encoder"]["global_latents"] == 256


def test_training_constructor_wires_native_aligned_scene_token_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict] = []
    monkeypatch.setattr(
        training_module,
        "SceneTokenizer",
        lambda **kwargs: captured.append(kwargs) or object(),
    )

    construct_scene_tokenizer(load_config("configs/default.yaml"), 2048, 896)
    construct_scene_tokenizer(load_config("configs/gemma4_e2b.yaml"), 3072, 1536)

    assert captured[0]["language_aligned_tail_dim"] == 0
    assert captured[0]["native_aligned_coverage_scale"] == 0.0
    assert captured[0]["learned_scene_token_scale"] == 1.0
    assert captured[0]["learned_scene_token_rms_target"] is None
    assert captured[1]["language_aligned_tail_dim"] == 1536
    assert captured[1]["native_aligned_coverage_scale"] == 1.0
    assert captured[1]["learned_scene_token_scale"] == 0.1
    assert captured[1]["learned_scene_token_rms_target"] == 0.65


def test_question_grounding_attends_spatial_latents_and_backpropagates() -> None:
    torch.manual_seed(37)
    head = QuestionGroundingHead(
        scene_dim=16,
        language_dim=12,
        latent_count=8,
        hidden_dim=20,
    )
    latents = torch.randn(2, 8, 16, requires_grad=True)
    questions = torch.randn(2, 5, 12, requires_grad=True)

    predicted, logits, weights = head.forward_with_attention(latents, questions)

    assert predicted.shape == (2, 3)
    assert logits.shape == weights.shape == (2, 8)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-6)
    targets = head.nearest_anchor_targets(torch.zeros(2, 3))
    loss = predicted.square().mean() + torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()
    assert latents.grad is not None and latents.grad.abs().sum() > 0
    assert questions.grad is not None and questions.grad.abs().sum() > 0
