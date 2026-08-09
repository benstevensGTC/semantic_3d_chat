from __future__ import annotations

import copy
import math

import pytest
import torch

from semantic_3d_chat.evaluation.v16_gradient_audit import (
    _functional_residual_delta,
    gradient_comparison,
    pair_delta_metrics,
    scene_delta_metrics,
    simulate_first_adamw_update,
)
from semantic_3d_chat.scene_encoder.global_residual import GlobalSceneResidual


def test_gradient_comparison_reports_orthogonal_vectors() -> None:
    metrics = gradient_comparison(torch.tensor([3.0, 0.0]), torch.tensor([0.0, 4.0]))

    assert metrics["parameter_count"] == 2
    assert metrics["first_l2_norm"] == pytest.approx(3.0)
    assert metrics["second_l2_norm"] == pytest.approx(4.0)
    assert metrics["dot_product"] == pytest.approx(0.0)
    assert metrics["cosine_similarity"] == pytest.approx(0.0)


def test_simulated_first_adamw_update_is_dense_sign_step_without_clipping() -> None:
    gradient = torch.tensor([[2.0, -4.0]], dtype=torch.float32)
    update, metrics = simulate_first_adamw_update(
        gradient,
        learning_rate=1.0e-3,
        gradient_clip_norm=10.0,
        epsilon=1.0e-12,
    )

    assert torch.allclose(update, torch.tensor([[-1.0e-3, 1.0e-3]]), atol=1.0e-10)
    assert metrics["clip_scale"] == pytest.approx(1.0)
    assert metrics["nonzero_update_count"] == 2
    assert metrics["update_rms"] == pytest.approx(1.0e-3)


def test_simulated_first_adamw_update_applies_global_norm_clip() -> None:
    gradient = torch.tensor([3.0, 4.0])
    update, metrics = simulate_first_adamw_update(
        gradient,
        learning_rate=1.0e-3,
        gradient_clip_norm=1.0,
    )

    assert metrics["pre_clip_gradient_l2_norm"] == pytest.approx(5.0)
    assert metrics["clip_scale"] == pytest.approx(1.0 / (5.0 + 1.0e-6))
    assert metrics["post_clip_gradient_l2_norm"] == pytest.approx(5.0 / (5.0 + 1.0e-6))
    assert torch.isfinite(update).all()


def test_scene_delta_metrics_separates_common_and_slot_varying_energy() -> None:
    core = torch.ones(1, 2, 2)
    common_delta = torch.full_like(core, 0.5)
    common = scene_delta_metrics(core, common_delta)
    assert common["delta_to_core_rms_ratio"] == pytest.approx(0.5)
    assert common["across_slot_mean_energy_fraction"] == pytest.approx(1.0)
    assert common["slot_varying_energy_fraction"] == pytest.approx(0.0)

    varying_delta = torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]])
    varying = scene_delta_metrics(core, varying_delta)
    assert varying["across_slot_mean_energy_fraction"] == pytest.approx(0.0)
    assert varying["slot_varying_energy_fraction"] == pytest.approx(1.0)


def test_pair_delta_metrics_reports_relative_change() -> None:
    first_core = torch.tensor([[[2.0], [4.0]]])
    second_core = torch.tensor([[[1.0], [2.0]]])
    first_delta = torch.tensor([[[0.2], [0.4]]])
    second_delta = torch.tensor([[[0.1], [0.2]]])

    metrics = pair_delta_metrics(first_core, second_core, first_delta, second_delta)

    assert metrics["residual_to_core_pair_difference_ratio"] == pytest.approx(0.1)
    assert metrics["residual_core_difference_cosine"] == pytest.approx(1.0)
    assert math.isfinite(metrics["core_pair_difference_rms"])


def test_functional_residual_simulation_uses_forward_without_mutating_module() -> None:
    module = GlobalSceneResidual(
        scene_dim=8,
        latent_count=4,
        width=3,
        fourier_bands=2,
        initialization_seed=91,
    )
    scene_tokens = torch.randn(2, 4, 8)
    simulated_weight = torch.randn_like(module.output_projection.weight) * 1.0e-3
    expected_module = copy.deepcopy(module)

    state_before = {name: value.detach().clone() for name, value in module.state_dict().items()}
    observed = _functional_residual_delta(module, scene_tokens, simulated_weight)
    with torch.no_grad():
        expected_module.output_projection.weight.copy_(simulated_weight)
        expected = expected_module(scene_tokens) - scene_tokens

    assert torch.equal(observed, expected)
    for name, value in state_before.items():
        assert torch.equal(module.state_dict()[name], value)
