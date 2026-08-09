from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from semantic_3d_chat.scene_encoder import (
    SIGNED_X_MOMENT_V1,
    SignedXSceneResidual,
    apply_signed_x_scene_residual,
    construct_signed_x_scene_residual,
    frozen_v18_centered_content_values,
    signed_x_scene_residual_settings,
)
from semantic_3d_chat.scene_encoder.global_residual import (
    ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
    GlobalSceneResidual,
)
from semantic_3d_chat.training.checkpointing import module_collection_state_sha256

FULL_SIZE_INITIAL_STATE_SHA256 = "55b7cb21d0ecbe945cabccfacd5b6aa94693743ceee78443f37a5ca0d1ac68b1"


def _module() -> SignedXSceneResidual:
    return SignedXSceneResidual(scene_dim=16, latent_count=8, content_dim=6)


def _centered_content(batch: int = 2) -> torch.Tensor:
    values = torch.randn(batch, 8, 6)
    return values - values.mean(dim=1, keepdim=True)


def _enable_nonzero_output(module: SignedXSceneResidual) -> None:
    with torch.no_grad():
        values = torch.linspace(-0.03, 0.03, module.output_projection.weight.numel())
        module.output_projection.weight.copy_(values.reshape_as(module.output_projection.weight))


@dataclass
class _Output:
    scene_tokens: torch.Tensor
    native_latents: torch.Tensor
    block_tokens: torch.Tensor
    audit: dict[str, torch.Tensor]


def test_signed_x_zero_output_is_exact_and_owns_only_projection() -> None:
    module = _module()
    base = torch.randn(2, 8, 16)
    content = _centered_content()

    output = module(base, content)

    assert torch.equal(output, base)
    assert list(dict(module.named_parameters())) == ["output_projection.weight"]
    assert module.output_projection.bias is None
    assert module.parameter_count == 16 * 6
    assert torch.count_nonzero(module.output_projection.weight).item() == 0
    assert set(module.state_dict()) == {"signed_x_anchors", "output_projection.weight"}


def test_signed_x_anchors_are_persistent_centered_unit_rms_and_all_slot() -> None:
    module = _module()
    anchors = module.signed_x_anchors
    audit = module.validate_structural_state()

    assert "signed_x_anchors" in dict(module.named_buffers())
    assert anchors.shape == (8,)
    assert anchors.dtype == torch.float32
    assert anchors.mean().abs().item() <= 1e-6
    assert anchors.square().mean().sqrt().item() == pytest.approx(1.0, abs=1e-6)
    assert torch.all(anchors != 0)
    assert audit["accounted_slot_count"] == 8
    assert audit["all_slots_accounted"] is True
    assert audit["trainable_surface"] == "bias_free_output_projection_only"


def test_signed_x_moment_and_hidden_match_explicit_formula() -> None:
    module = _module()
    content = _centered_content()
    signed = module.signed_x_anchors.view(1, 8, 1)

    expected_moment = (signed * content).mean(dim=1, keepdim=True)
    moment = module.moment_values(content)
    hidden = module.hidden_values(content)

    assert moment.dtype == torch.float32
    assert moment.shape == (2, 1, 6)
    assert torch.allclose(moment, expected_moment)
    assert torch.allclose(hidden, signed * torch.tanh(expected_moment))


def test_frozen_v18_content_extraction_matches_pinned_architecture_formula() -> None:
    source = GlobalSceneResidual(
        scene_dim=16,
        latent_count=8,
        width=6,
        fourier_bands=2,
        architecture_version=ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
    ).requires_grad_(False)
    tokens = torch.randn(2, 8, 16)

    extracted = frozen_v18_centered_content_values(source, tokens)
    local = source.scene_projection(source.scene_norm(tokens))
    expected = local - local.float().mean(dim=1, keepdim=True).to(local.dtype)

    assert torch.equal(extracted, expected)
    assert extracted.shape == (2, 8, 6)
    assert extracted.mean(dim=1).abs().max().item() <= 1e-6


def test_nonzero_signed_x_delta_is_fp32_centered_and_mirror_odd() -> None:
    module = _module()
    _enable_nonzero_output(module)
    content = _centered_content()

    delta = module.centered_delta_values(content)
    opposite = module.centered_delta_values(-content)

    assert delta.dtype == torch.float32
    assert delta.shape == (2, 8, 16)
    assert delta.square().mean().item() > 0
    assert delta.mean(dim=1).abs().max().item() <= 1e-7
    assert torch.allclose(opposite, -delta, atol=1e-7, rtol=1e-6)


def test_every_centered_content_slot_influences_the_signed_moment_field() -> None:
    module = _module()
    _enable_nonzero_output(module)
    raw = torch.randn(1, 8, 6, requires_grad=True)
    content = raw - raw.mean(dim=1, keepdim=True)

    first_slot_delta = module.centered_delta_values(content)[:, 0]
    gradient = torch.autograd.grad(first_slot_delta.square().sum(), raw)[0]

    assert torch.all(gradient.abs().sum(dim=-1) > 0)
    assert torch.isfinite(gradient).all()


def test_only_output_projection_receives_branch_gradient_at_zero_output() -> None:
    module = _module()
    base = torch.randn(2, 8, 16)
    content = _centered_content().requires_grad_()
    target = torch.randn_like(base)

    loss = (module(base, content) * target).sum()
    loss.backward()

    gradient = module.output_projection.weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient).item() > 0
    assert content.grad is not None
    assert torch.count_nonzero(content.grad).item() == 0


def test_frozen_eval_module_validates_and_runs_exact_identity() -> None:
    module = _module().requires_grad_(False).eval()
    base = torch.randn(2, 8, 16, dtype=torch.bfloat16)
    content = _centered_content().to(torch.bfloat16)

    audit = module.validate_structural_state()
    output = module(base, content)

    assert audit["parameter_count"] == 96
    assert torch.equal(output, base)
    assert output.dtype == torch.bfloat16


def test_dtype_move_preserves_fp32_audited_branch_state() -> None:
    module = _module().to(dtype=torch.bfloat16)

    assert module.signed_x_anchors.dtype == torch.float32
    assert module.output_projection.weight.dtype == torch.float32
    module.validate_structural_state()


def test_signed_x_input_validation_rejects_bad_shapes_nonfinite_and_uncentered() -> None:
    module = _module()
    base = torch.randn(2, 8, 16)
    content = _centered_content()

    with pytest.raises(ValueError, match="base_tokens"):
        module(base[:, :-1], content)
    with pytest.raises(ValueError, match="centered_content"):
        module(base, content[:, :-1])
    with pytest.raises(ValueError, match="batch sizes"):
        module(base, content[:1])
    with pytest.raises(ValueError, match="centered across"):
        module(base, torch.ones_like(content))
    bad_base = base.clone()
    bad_base[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        module(bad_base, content)
    bad_content = content.clone()
    bad_content[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        module(base, bad_content)


def test_structural_validation_rejects_state_drift_and_nonfinite_projection() -> None:
    drifted = _module()
    with torch.no_grad():
        drifted.signed_x_anchors[0].add_(0.25)
    with pytest.raises(ValueError, match="deterministic anchors"):
        drifted.validate_structural_state()

    nonfinite = _module()
    with torch.no_grad():
        nonfinite.output_projection.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite.*output_projection.weight"):
        nonfinite.validate_structural_state()


def test_settings_and_constructor_are_strict_and_backward_compatible() -> None:
    assert signed_x_scene_residual_settings({"scene_encoder": {}}).contract() == {
        "schema_version": 1,
        "enabled": False,
    }
    assert (
        construct_signed_x_scene_residual(
            {"scene_encoder": {}}, scene_dim=16, latent_count=8, content_dim=6
        )
        is None
    )
    config = {
        "scene_encoder": {
            "signed_x_scene_residual": {
                "enabled": True,
                "architecture_version": SIGNED_X_MOMENT_V1,
                "expected_initial_state_sha256": "a" * 64,
            }
        }
    }
    settings = signed_x_scene_residual_settings(config)
    module = construct_signed_x_scene_residual(config, scene_dim=16, latent_count=8, content_dim=6)

    assert settings.contract()["spatial_statistic"] == "centered_unit_rms_signed_x_moment"
    assert isinstance(module, SignedXSceneResidual)
    with pytest.raises(ValueError, match="requires expected"):
        signed_x_scene_residual_settings(
            {"scene_encoder": {"signed_x_scene_residual": {"enabled": True}}}
        )
    with pytest.raises(ValueError, match="Unknown"):
        signed_x_scene_residual_settings(
            {"scene_encoder": {"signed_x_scene_residual": {"mystery": 1}}}
        )


def test_apply_signed_x_residual_replaces_only_scene_tokens_and_adds_audit() -> None:
    module = _module()
    _enable_nonzero_output(module)
    base = torch.randn(2, 8, 16)
    content = _centered_content()
    source = _Output(
        scene_tokens=base,
        native_latents=torch.randn(2, 8, 4),
        block_tokens=torch.randn(2, 3, 4),
        audit={"existing": torch.tensor(1)},
    )

    adapted = apply_signed_x_scene_residual(source, module, content)

    assert torch.equal(adapted.scene_tokens, module(base, content))
    assert adapted.native_latents is source.native_latents
    assert adapted.block_tokens is source.block_tokens
    assert adapted.audit["existing"].item() == 1
    assert adapted.audit["signed_x_scene_residual_delta_rms"].item() > 0
    assert adapted.audit["signed_x_scene_residual_accounted_slots"].item() == 8
    assert apply_signed_x_scene_residual(source, None, content) is source


def test_full_size_initial_state_hash_is_reproducible_and_state_sensitive() -> None:
    first = SignedXSceneResidual(scene_dim=1536, latent_count=256, content_dim=128)
    repeated = SignedXSceneResidual(scene_dim=1536, latent_count=256, content_dim=128)
    first_hash = module_collection_state_sha256({"signed_x_scene_residual": first})

    assert first.parameter_count == 196_608
    assert first_hash == FULL_SIZE_INITIAL_STATE_SHA256
    assert first_hash == module_collection_state_sha256({"signed_x_scene_residual": repeated})
    assert len(first_hash) == 64
    with torch.no_grad():
        first.output_projection.weight[0, 0] = 1e-4
    assert first_hash != module_collection_state_sha256({"signed_x_scene_residual": first})
