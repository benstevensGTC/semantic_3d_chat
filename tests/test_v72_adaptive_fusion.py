from __future__ import annotations

import torch

from semantic_3d_chat.scene_encoder.question_control_v7 import (
    AlwaysOnTeacherBasisFullSceneQuestionControlV7,
)
from semantic_3d_chat.scene_encoder.question_control_v72 import (
    AdaptiveMultiscaleTeacherBasisControlV72,
)


def _controller() -> AdaptiveMultiscaleTeacherBasisControlV72:
    basis = torch.eye(4, 16)
    torch.manual_seed(7208)
    branch_8 = AlwaysOnTeacherBasisFullSceneQuestionControlV7(
        16,
        basis,
        expected_environment_latents=256,
        moment_count=8,
        interaction_dim=4,
        trunk_dim=8,
    )
    torch.manual_seed(7232)
    branch_32 = AlwaysOnTeacherBasisFullSceneQuestionControlV7(
        16,
        basis,
        expected_environment_latents=256,
        moment_count=32,
        interaction_dim=4,
        trunk_dim=8,
    )
    return AdaptiveMultiscaleTeacherBasisControlV72(
        branch_8, branch_32, gate_hidden_size=5
    )


def test_v72_starts_as_exact_half_mix_and_processes_all_latents() -> None:
    control = _controller().eval()
    prefix = torch.randn(1, 258, 16)
    question = torch.randn(1, 3, 16)
    signature = control.encode_scene(prefix)
    branch_8, branch_32 = control.branch_outputs_from_signature(
        signature, question
    )
    output = control.forward_from_signature(signature, question)

    assert signature.shape == (1, 40, 16)
    assert torch.equal(signature[:, :8], signature[:, 8:16])
    assert torch.equal(control.fusion_weights(question), torch.full((1, 4, 4), 0.5))
    # With the toy basis, only its first four native dimensions are retained.
    expected = 0.5 * branch_8 + 0.5 * branch_32
    expected[..., 4:] = 0.0
    assert torch.allclose(output.control_tokens, expected, atol=1e-6, rtol=1e-6)

    changed = prefix.clone()
    changed[:, -2, 0] += 2.0
    assert not torch.equal(signature[:, 0], control.encode_scene(changed)[:, 0])
    assert not torch.equal(signature[:, 8], control.encode_scene(changed)[:, 8])
    audit = control.audit()
    assert audit.every_environment_latent_influences_both_branches is True
    assert audit.both_complete_scene_branches_executed is True
    assert audit.question_dependent_scene_retrieval is False
    assert audit.latent_selection_or_top_k_used is False


def test_v72_gate_varies_by_question_not_by_scene_latent_selection() -> None:
    control = _controller().eval()
    with torch.no_grad():
        control.coefficient_output.fusion_output.weight.normal_(std=0.2)
        control.coefficient_output.fusion_output.bias.normal_(std=0.1)
    question_a = torch.randn(1, 3, 16)
    question_b = torch.randn(1, 3, 16)
    weights_a = control.fusion_weights(question_a)
    weights_b = control.fusion_weights(question_b)

    assert weights_a.shape == (1, 4, 4)
    assert not torch.equal(weights_a, weights_b)
    assert float(weights_a.detach().min()) >= 0.05
    assert float(weights_a.detach().max()) <= 0.95
    # Scene tensors are not an argument to fusion_weights.  Changing every
    # latent therefore cannot alter which continuous branch coefficients mix.
    prefix_a = torch.randn(1, 258, 16)
    prefix_b = torch.randn(1, 258, 16)
    control.forward(prefix_a, question_a)
    first = control.fusion_weights(question_a)
    control.forward(prefix_b, question_a)
    second = control.fusion_weights(question_a)
    assert torch.equal(first, second)


def test_v72_gate_receives_gradients_and_controller_remains_bounded() -> None:
    control = _controller().train()
    prefix = torch.randn(2, 258, 16)
    questions = torch.randn(2, 4, 16)
    target = torch.randn(2, 4, 16)
    output = control(prefix, questions)
    loss = torch.nn.functional.mse_loss(output.control_tokens, target)
    loss.backward()

    assert control.coefficient_output.fusion_output.weight.grad is not None
    assert torch.isfinite(control.coefficient_output.fusion_output.weight.grad).all()
    assert (
        float(output.control_rms.detach().max())
        <= control.maximum_control_rms + 1e-6
    )
