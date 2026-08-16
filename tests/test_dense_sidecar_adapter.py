from __future__ import annotations

import inspect
from dataclasses import dataclass

import pytest
import torch

from semantic_3d_chat.scene_encoder.dense_sidecar_adapter import (
    DENSE_SIDECAR_ADAPTER_ARCHITECTURE_MARKER,
    DENSE_SIDECAR_ADAPTER_ARCHITECTURE_VERSION,
    DenseSidecarAdapter,
    apply_dense_sidecar_adapter,
    construct_dense_sidecar_adapter,
    dense_sidecar_adapter_settings,
    validate_dense_sidecar_adapter_state,
)


def _adapter(*, seed: int = 28028, max_direct_scale: float = 0.2) -> DenseSidecarAdapter:
    return DenseSidecarAdapter(
        scene_dim=16,
        latent_count=9,
        width=8,
        fourier_bands=2,
        max_direct_scale=max_direct_scale,
        initialization_seed=seed,
    )


def _tokens() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(2, 9, 16), torch.randn(2, 9, 16)


def test_zero_output_identity_shapes_and_scene_only_api() -> None:
    module = _adapter()
    base, sidecar = _tokens()

    output = module(base, sidecar)
    audit = module.validate_structural_state()

    assert torch.equal(output, base)
    assert torch.count_nonzero(module.output_projection.weight).item() == 0
    assert torch.count_nonzero(module.channel_gain).item() == 0
    assert module.bounded_channel_gain().shape == (16,)
    assert torch.count_nonzero(module.bounded_channel_gain()).item() == 0
    assert audit["architecture_version"] == DENSE_SIDECAR_ADAPTER_ARCHITECTURE_VERSION
    assert audit["architecture_marker"] == DENSE_SIDECAR_ADAPTER_ARCHITECTURE_MARKER
    assert audit["output_projection_exact_zero"] is True
    assert audit["channel_gain_exact_zero"] is True
    assert audit["application_point"] == "post_frozen_scene_stack"
    assert audit["question_dependent_inputs"] is False
    assert audit["environmental_metadata_inputs"] is False
    assert list(inspect.signature(type(module).forward).parameters) == [
        "self",
        "base_scene_tokens",
        "aligned_sidecar_tokens",
    ]


def test_initialization_is_seed_reproducible_without_consuming_global_rng() -> None:
    torch.manual_seed(991)
    expected_next = torch.rand(5)
    torch.manual_seed(991)
    first = _adapter(seed=77)
    observed_next = torch.rand(5)
    repeated = _adapter(seed=77)
    different = _adapter(seed=78)

    assert torch.equal(observed_next, expected_next)
    assert first.state_sha256() == repeated.state_sha256()
    assert first.state_sha256() != different.state_sha256()
    assert len(first.state_sha256()) == 64
    assert torch.equal(first.output_projection.weight, repeated.output_projection.weight)
    assert torch.count_nonzero(first.output_projection.weight).item() == 0
    assert torch.count_nonzero(different.channel_gain).item() == 0


def test_direct_route_is_full_dimensional_channelwise_and_strictly_bounded() -> None:
    module = _adapter(max_direct_scale=0.2)
    base, sidecar = _tokens()
    with torch.no_grad():
        module.channel_gain.zero_()
        module.channel_gain[3] = 100.0

    delta = module.residual_delta(base, sidecar)
    gain = module.bounded_channel_gain()

    assert gain[3].item() == pytest.approx(0.2)
    assert torch.count_nonzero(gain).item() == 1
    assert torch.count_nonzero(delta[..., :3]).item() == 0
    assert torch.count_nonzero(delta[..., 4:]).item() == 0
    assert torch.count_nonzero(delta[..., 3]).item() > 0
    assert delta[..., 3].abs().max().item() <= 0.2


def test_hidden_route_jointly_uses_base_sidecar_and_fixed_position() -> None:
    module = _adapter()
    with torch.no_grad():
        module.output_projection.weight.normal_(mean=0.0, std=0.05)
    base, sidecar = _tokens()
    original = module.residual_delta(base, sidecar)

    changed_base = module.residual_delta(base + torch.randn_like(base), sidecar)
    changed_sidecar = module.residual_delta(base, sidecar + torch.randn_like(sidecar))
    repeated_slot_content = torch.ones_like(base)
    position_residual = module.residual_delta(repeated_slot_content, repeated_slot_content)

    assert not torch.equal(original, changed_base)
    assert not torch.equal(original, changed_sidecar)
    assert not torch.allclose(
        position_residual,
        position_residual[:, :1].expand_as(position_residual),
    )


def test_first_backward_opens_both_zero_output_surfaces_only() -> None:
    module = _adapter()
    base, sidecar = _tokens()
    target = torch.randn_like(base)

    (module(base, sidecar) * target).sum().backward()

    assert module.output_projection.weight.grad is not None
    assert torch.count_nonzero(module.output_projection.weight.grad).item() > 0
    assert module.channel_gain.grad is not None
    assert torch.count_nonzero(module.channel_gain.grad).item() > 0
    for parameter in (
        module.base_projection.weight,
        module.sidecar_projection.weight,
        module.position_projection.weight,
        module.base_norm.weight,
        module.sidecar_norm.weight,
    ):
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad).item() == 0


def test_settings_contract_construction_and_fail_closed_parser() -> None:
    initial = _adapter(seed=17, max_direct_scale=0.125)
    config = {
        "scene_encoder": {
            "dense_sidecar_adapter": {
                "enabled": True,
                "width": 8,
                "fourier_bands": 2,
                "max_direct_scale": 0.125,
                "initialization_seed": 17,
                "expected_initial_state_sha256": initial.state_sha256(),
            }
        }
    }

    settings = dense_sidecar_adapter_settings(config)
    module = construct_dense_sidecar_adapter(config, scene_dim=16, latent_count=9)

    assert settings.contract() == {
        "schema_version": 1,
        "enabled": True,
        "architecture_version": DENSE_SIDECAR_ADAPTER_ARCHITECTURE_VERSION,
        "width": 8,
        "fourier_bands": 2,
        "max_direct_scale": 0.125,
        "initialization_seed": 17,
        "expected_initial_state_sha256": initial.state_sha256(),
        "application_point": "post_frozen_scene_stack",
        "output_projection_initialization": "exact_zero",
        "channel_gain_initialization": "exact_zero",
        "normalization": "separate_affine_layer_norm",
        "direct_route": "full_dimensional_tanh_bounded",
        "base_identity_path": True,
        "question_dependent_inputs": False,
        "environmental_metadata_inputs": False,
    }
    assert module is not None
    assert module.state_sha256() == initial.state_sha256()
    assert validate_dense_sidecar_adapter_state(
        module,
        expected_parameter_count=608,
        expected_state_sha256=initial.state_sha256(),
        context="V28 preflight",
    )["parameter_count"] == 608

    assert dense_sidecar_adapter_settings({"scene_encoder": {}}).contract() == {
        "schema_version": 1,
        "enabled": False,
    }
    assert construct_dense_sidecar_adapter(
        {"scene_encoder": {}}, scene_dim=16, latent_count=9
    ) is None
    with pytest.raises(ValueError, match="requires expected_initial_state_sha256"):
        dense_sidecar_adapter_settings(
            {"scene_encoder": {"dense_sidecar_adapter": {"enabled": True}}}
        )
    with pytest.raises(ValueError, match="Unknown dense_sidecar_adapter settings"):
        dense_sidecar_adapter_settings(
            {"scene_encoder": {"dense_sidecar_adapter": {"surprise": True}}}
        )
    with pytest.raises(ValueError, match="state SHA-256 mismatch"):
        validate_dense_sidecar_adapter_state(
            module,
            expected_state_sha256="0" * 64,
            context="V28 preflight",
        )


@dataclass
class _Output:
    scene_tokens: torch.Tensor
    native_latents: torch.Tensor
    block_tokens: torch.Tensor
    audit: dict[str, torch.Tensor]
    aligned_sidecar_tokens: torch.Tensor | None


def test_apply_helper_requires_sidecar_replaces_only_post_stack_tokens_and_audits() -> None:
    module = _adapter()
    base, sidecar = _tokens()
    native = torch.randn(2, 9, 8)
    blocks = torch.randn(2, 4, 8)
    original = _Output(base, native, blocks, {"existing": torch.tensor(1)}, sidecar)

    adapted = apply_dense_sidecar_adapter(original, module)

    assert adapted is not original
    assert torch.equal(adapted.scene_tokens, base)
    assert adapted.native_latents is native
    assert adapted.block_tokens is blocks
    assert adapted.aligned_sidecar_tokens is sidecar
    assert adapted.audit is not original.audit
    assert adapted.audit["existing"].item() == 1
    assert adapted.audit["dense_sidecar_adapter_delta_rms"].item() == 0.0
    assert adapted.audit["dense_sidecar_adapter_direct_gain_abs_max"].item() == 0.0
    assert apply_dense_sidecar_adapter(original, None) is original

    unavailable = _Output(base, native, blocks, {}, None)
    with pytest.raises(ValueError, match="requires nonempty"):
        apply_dense_sidecar_adapter(unavailable, module)


@pytest.mark.parametrize(
    ("base_shape", "sidecar_shape"),
    [
        ((2, 8, 16), (2, 9, 16)),
        ((2, 9, 15), (2, 9, 16)),
        ((0, 9, 16), (0, 9, 16)),
        ((2, 9, 16), (1, 9, 16)),
    ],
)
def test_input_shape_validation_is_strict(
    base_shape: tuple[int, ...], sidecar_shape: tuple[int, ...]
) -> None:
    module = _adapter()
    base = torch.zeros(base_shape)
    sidecar = torch.zeros(sidecar_shape)

    with pytest.raises(ValueError, match="shape|at least one|match exactly"):
        module(base, sidecar)


def test_input_and_persistent_state_finiteness_are_strict() -> None:
    module = _adapter()
    base, sidecar = _tokens()
    nonfinite = sidecar.clone()
    nonfinite[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or infinity"):
        module(base, nonfinite)
    with pytest.raises(ValueError, match="same dtype"):
        module(base, sidecar.double())

    with torch.no_grad():
        module.sidecar_projection.weight[0, 0] = float("inf")
    with pytest.raises(ValueError, match="sidecar_projection.weight contains"):
        module.validate_structural_state()


def test_structural_buffers_and_shapes_fail_closed() -> None:
    module = _adapter()
    with torch.no_grad():
        module.position_features[0, 0].add_(0.25)
    with pytest.raises(ValueError, match="position features"):
        module.validate_structural_state()

    module = _adapter()
    with torch.no_grad():
        module.max_direct_scale.add_(0.01)
    with pytest.raises(ValueError, match="max direct scale"):
        module.validate_structural_state()

    module = _adapter()
    module.output_projection.weight = torch.nn.Parameter(torch.zeros(15, 8))
    with pytest.raises(ValueError, match="output_projection.weight shape mismatch"):
        module.validate_structural_state()
