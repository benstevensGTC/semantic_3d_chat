from __future__ import annotations

import torch

from semantic_3d_chat.scene_encoder.question_control_v74 import (
    DenseFullSceneContinuousControlV74,
)


def _control() -> DenseFullSceneContinuousControlV74:
    torch.manual_seed(74)
    return DenseFullSceneContinuousControlV74(
        16, torch.eye(8, 16), environment_latents=256, model_dimension=16
    )


def test_v74_all_latents_receive_positive_attention_and_change_output() -> None:
    control = _control()
    scene = torch.randn(1, 258, 16)
    question = torch.randn(1, 5, 16)
    before = control(scene, question).control_tokens
    audit = control.audit()
    changed = scene.clone()
    changed[:, -2] += 4
    after = control(changed, question).control_tokens
    assert not torch.equal(before, after)
    assert audit.all_latents_receive_positive_weight is True
    assert audit.question_dependent_retrieval is False
    assert audit.question_only_output_path_exists is False


def test_v74_zero_scene_cannot_emit_question_only_controls_after_updates() -> None:
    control = _control()
    optimizer = torch.optim.AdamW(control.parameters(), lr=0.003)
    scene = torch.randn(2, 258, 16)
    question = torch.randn(2, 6, 16)
    target = torch.randn(2, 4, 16)
    for _ in range(2):
        loss = (control(scene, question).control_tokens - target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    zero = torch.zeros_like(scene)
    assert torch.count_nonzero(control(zero, question).control_tokens) == 0
    assert torch.count_nonzero(control(zero, question * 3).control_tokens) == 0


def test_v74_question_modulates_nonzero_scene_values() -> None:
    control = _control()
    scene = torch.randn(1, 258, 16)
    first = control(scene, torch.randn(1, 4, 16)).control_tokens
    second = control(scene, torch.randn(1, 7, 16)).control_tokens
    assert not torch.equal(first, second)
