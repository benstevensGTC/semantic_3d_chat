from __future__ import annotations

import pytest
import torch

from semantic_3d_chat.scene_encoder.question_control_v75 import (
    DenseFullSceneContinuousControlV75,
)


def _control() -> DenseFullSceneContinuousControlV75:
    torch.manual_seed(75)
    return DenseFullSceneContinuousControlV75(
        16,
        torch.eye(8, 16),
        environment_latents=256,
        model_dimension=16,
        coefficient_decoder_hidden_dimension=24,
    )


def test_v75_uses_bias_free_nonlinear_decoder_and_all_scene_latents() -> None:
    control = _control()
    assert control.coefficient_hidden.bias is None
    assert control.coefficient_output.bias is None
    assert isinstance(control.coefficient_activation, torch.nn.GELU)

    scene = torch.randn(1, 258, 16)
    question = torch.randn(1, 5, 16)
    before = control(scene, question).control_tokens
    changed = scene.clone()
    changed[:, -2] += 4.0
    after = control(changed, question).control_tokens
    audit = control.audit()

    assert not torch.equal(before, after)
    assert audit.all_latents_receive_positive_weight is True
    assert audit.question_dependent_retrieval is False
    assert audit.question_only_output_path_exists is False
    assert audit.bias_free_nonlinear_coefficient_decoder is True
    assert audit.zero_preserving_coefficient_activation is True
    assert audit.coefficient_decoder_hidden_dimension == 24


def test_v75_zero_scene_cannot_emit_controls_after_parameter_updates() -> None:
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
    assert torch.count_nonzero(control(zero, question * 7.0).control_tokens) == 0


def test_v75_rejects_nonpositive_decoder_dimension() -> None:
    with pytest.raises(ValueError, match="decoder dimension"):
        DenseFullSceneContinuousControlV75(
            16,
            torch.eye(8, 16),
            coefficient_decoder_hidden_dimension=0,
        )
