from __future__ import annotations

import inspect

import pytest
import torch

from semantic_3d_chat.scene_encoder.global_residual import (
    GLOBAL_MEAN_V1,
    ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
    GlobalSceneResidual,
    construct_global_scene_residual,
    global_scene_residual_settings,
)
from semantic_3d_chat.training.checkpointing import module_collection_state_sha256

LEGACY_V16_STATE_SHA256 = "fb4ebaac06dccbc04a461b10546d00f48cdf8cfbb372cbe5f6fe925f71461bd3"


def _residual(*, seed: int = 16015) -> GlobalSceneResidual:
    return GlobalSceneResidual(
        scene_dim=16,
        latent_count=9,
        width=8,
        fourier_bands=2,
        initialization_seed=seed,
    )


def _content_gated_residual(*, seed: int = 18018) -> GlobalSceneResidual:
    return GlobalSceneResidual(
        scene_dim=16,
        latent_count=9,
        width=8,
        fourier_bands=2,
        initialization_seed=seed,
        architecture_version=ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
        gate_temperature=1.0,
    )


def _enable_nonzero_residual(module: GlobalSceneResidual) -> None:
    """Move only the public zero-output projection off its identity start."""

    with torch.no_grad():
        module.output_projection.weight.fill_(0.03125)
        if module.output_projection.bias is not None:
            module.output_projection.bias.zero_()


def test_global_scene_residual_shape_identity_and_scene_only_api() -> None:
    module = _residual()
    scene_tokens = torch.randn(2, 9, 16)

    output = module(scene_tokens)

    assert output.shape == scene_tokens.shape
    assert torch.equal(output, scene_tokens)
    assert output.device == scene_tokens.device
    assert module.output_projection.bias is None
    assert torch.count_nonzero(module.output_projection.weight).item() == 0
    assert list(inspect.signature(type(module).forward).parameters) == [
        "self",
        "scene_tokens",
    ]
    with pytest.raises(TypeError):
        module(scene_tokens, torch.randn(2, 3, 16))


def test_zero_output_is_exact_for_bfloat16_and_inputs_are_validated() -> None:
    module = _residual().to(dtype=torch.bfloat16)
    scene_tokens = torch.randn(2, 9, 16, dtype=torch.bfloat16)

    assert torch.equal(module(scene_tokens), scene_tokens)
    with pytest.raises(ValueError):
        module(torch.zeros(2, 8, 16, dtype=torch.bfloat16))
    with pytest.raises(ValueError):
        module(torch.zeros(2, 9, 15, dtype=torch.bfloat16))
    nonfinite = scene_tokens.clone()
    nonfinite[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN|finite"):
        module(nonfinite)


def test_nonzero_global_residual_connects_each_output_to_every_scene_token() -> None:
    module = _residual()
    _enable_nonzero_residual(module)
    scene_tokens = torch.randn(2, 9, 16, requires_grad=True)

    residual_at_first_slot = (module(scene_tokens) - scene_tokens)[:, 0]
    gradient = torch.autograd.grad(
        residual_at_first_slot.square().sum(),
        scene_tokens,
    )[0]

    assert torch.all(gradient.abs().sum(dim=-1) > 0)
    assert torch.isfinite(gradient).all()


def test_global_scene_residual_is_position_dependent() -> None:
    module = _residual()
    _enable_nonzero_residual(module)
    identical_content = torch.ones(1, 9, 16)

    residual = module(identical_content) - identical_content

    assert not torch.allclose(
        residual,
        residual[:, :1].expand_as(residual),
    )


def test_global_scene_residual_state_hash_is_seed_reproducible() -> None:
    torch.manual_seed(8123)
    rng_before = torch.random.get_rng_state().clone()
    first = module_collection_state_sha256({"global_scene_residual": _residual(seed=91)})
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    repeated = module_collection_state_sha256({"global_scene_residual": _residual(seed=91)})
    different = module_collection_state_sha256({"global_scene_residual": _residual(seed=92)})

    assert first == repeated
    assert first != different
    assert len(first) == 64


def test_position_features_are_persistent_and_part_of_state_identity() -> None:
    module = _residual()
    buffers = dict(module.named_buffers())

    assert "position_features" in buffers
    assert "position_features" in module.state_dict()
    before = module_collection_state_sha256({"global_scene_residual": module})
    with torch.no_grad():
        module.position_features[0, 0].add_(0.125)
    after = module_collection_state_sha256({"global_scene_residual": module})

    assert before != after
    with pytest.raises(ValueError, match="position features"):
        module.validate_structural_state()


def test_absent_architecture_preserves_exact_legacy_contract_state_and_forward() -> None:
    config = {
        "scene_encoder": {
            "global_scene_residual": {
                "enabled": True,
                "width": 128,
                "fourier_bands": 4,
                "initialization_seed": 16015,
                "expected_initial_state_sha256": LEGACY_V16_STATE_SHA256,
            }
        }
    }
    settings = global_scene_residual_settings(config)
    module = construct_global_scene_residual(config, scene_dim=1536, latent_count=256)
    assert module is not None

    assert settings.architecture_version == GLOBAL_MEAN_V1
    assert settings.contract() == {
        "schema_version": 1,
        "enabled": True,
        "width": 128,
        "fourier_bands": 4,
        "initialization_seed": 16015,
        "expected_initial_state_sha256": LEGACY_V16_STATE_SHA256,
    }
    assert sum(parameter.numel() for parameter in module.parameters()) == 400_000
    assert module_collection_state_sha256({"global_scene_residual": module}) == (
        LEGACY_V16_STATE_SHA256
    )
    assert "content_gate_projection.weight" not in module.state_dict()
    assert "gate_temperature" not in module.state_dict()

    small = _residual()
    _enable_nonzero_residual(small)
    scene_tokens = torch.randn(2, 9, 16)
    normalized = small.scene_norm(scene_tokens)
    local_content = small.scene_projection(normalized)
    global_content = local_content.float().mean(dim=1, keepdim=True).to(local_content.dtype)
    positions = small.position_projection(
        small.position_features.to(dtype=scene_tokens.dtype)
    ).unsqueeze(0)
    expected = scene_tokens + small.output_projection(
        torch.tanh(local_content + global_content + positions)
    )

    assert torch.equal(small(scene_tokens), expected)
    with pytest.raises(RuntimeError, match="unavailable"):
        small.content_gate_values(scene_tokens)
    with pytest.raises(RuntimeError, match="unavailable"):
        small.centered_delta_values(scene_tokens)


def test_content_gated_contract_validation_and_construction() -> None:
    config = {
        "scene_encoder": {
            "global_scene_residual": {
                "enabled": True,
                "architecture_version": ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
                "width": 8,
                "fourier_bands": 2,
                "initialization_seed": 18018,
                "gate_temperature": 0.75,
                "expected_initial_state_sha256": "a" * 64,
            }
        }
    }
    settings = global_scene_residual_settings(config)
    module = construct_global_scene_residual(config, scene_dim=16, latent_count=9)
    assert module is not None

    assert settings.contract() == {
        "schema_version": 2,
        "enabled": True,
        "architecture_version": ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
        "width": 8,
        "fourier_bands": 2,
        "initialization_seed": 18018,
        "gate_temperature": 0.75,
        "spatial_centering": "all_slots_fp32",
        "content_gate": "bias_free_scalar_sigmoid_centered_content",
        "expected_initial_state_sha256": "a" * 64,
    }
    assert module.architecture_version == ZERO_SPATIAL_MEAN_CONTENT_GATE_V1
    assert module.content_gate_projection.bias is None
    assert module.gate_temperature.item() == pytest.approx(0.75)
    assert module.validate_structural_state() == {
        "architecture_version": ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
        "parameter_count": 432,
        "latent_count": 9,
        "scene_dim": 16,
        "gate_temperature": 0.75,
        "spatial_centering": "all_slots_fp32",
        "content_gate": "bias_free_scalar_sigmoid_centered_content",
    }

    for invalid in (0.0, -1.0, float("inf"), float("nan")):
        invalid_config = {
            "scene_encoder": {
                "global_scene_residual": {
                    **config["scene_encoder"]["global_scene_residual"],
                    "gate_temperature": invalid,
                }
            }
        }
        with pytest.raises(ValueError, match="gate_temperature"):
            global_scene_residual_settings(invalid_config)
    for invalid in (True, "1.0"):
        with pytest.raises(TypeError, match="gate_temperature"):
            GlobalSceneResidual(
                16,
                9,
                8,
                2,
                architecture_version=ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
                gate_temperature=invalid,
            )
    with pytest.raises(ValueError, match="Unsupported"):
        GlobalSceneResidual(16, 9, 8, 2, architecture_version="unreviewed_v2")
    with pytest.raises(ValueError, match="only valid"):
        global_scene_residual_settings(
            {
                "scene_encoder": {
                    "global_scene_residual": {
                        "gate_temperature": 1.0,
                    }
                }
            }
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_content_gated_update_zero_is_exact_identity(dtype: torch.dtype) -> None:
    module = _content_gated_residual().to(dtype=dtype)
    scene_tokens = torch.randn(2, 9, 16, dtype=dtype)

    output = module(scene_tokens)

    assert torch.equal(output, scene_tokens)
    assert torch.count_nonzero(module.output_projection.weight).item() == 0
    assert torch.isfinite(module.content_gate_values(scene_tokens)).all()


def test_full_size_content_gated_parameter_count_and_persistent_state() -> None:
    module = GlobalSceneResidual(
        scene_dim=1536,
        latent_count=256,
        width=128,
        fourier_bands=4,
        initialization_seed=18018,
        architecture_version=ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
        gate_temperature=1.0,
    )

    assert module.parameter_count == 400_128
    assert module.content_gate_projection.weight.shape == (1, 128)
    assert module.content_gate_projection.bias is None
    assert module.position_features.shape == (256, 27)
    assert set(dict(module.named_buffers())) >= {"position_features", "gate_temperature"}
    assert set(module.state_dict()) >= {
        "position_features",
        "gate_temperature",
        "content_gate_projection.weight",
    }


def test_content_gate_is_bounded_content_dependent_and_globally_coupled() -> None:
    module = _content_gated_residual()
    scene_tokens = torch.randn(1, 9, 16)
    changed = scene_tokens.clone()
    changed[:, 0] = torch.randn(1, 16) * 3.0

    original_gate = module.content_gate_values(scene_tokens)
    changed_gate = module.content_gate_values(changed)

    assert original_gate.shape == (1, 9, 1)
    assert torch.all((original_gate > 0.0) & (original_gate < 2.0))
    assert not torch.equal(original_gate, changed_gate)
    # The centered-content mean makes even untouched slots depend on the
    # changed slot, proving that the gate consumes the complete token set.
    assert torch.all(original_gate[:, 1:] != changed_gate[:, 1:])


def test_content_gated_delta_has_zero_spatial_mean_and_no_common_energy() -> None:
    module = _content_gated_residual()
    _enable_nonzero_residual(module)
    scene_tokens = torch.randn(2, 9, 16)

    delta = module.centered_delta_values(scene_tokens)
    spatial_mean = delta.mean(dim=1, keepdim=True)
    total_energy = delta.square().mean()
    common_energy = spatial_mean.square().mean()

    assert delta.dtype == torch.float32
    assert torch.equal(module(scene_tokens), scene_tokens + delta)
    assert total_energy.item() > 0.0
    assert spatial_mean.abs().max().item() <= 1e-7
    assert (common_energy / total_energy).item() <= 1e-10


def test_bfloat16_audit_delta_is_centered_in_fp32_before_quantized_addition() -> None:
    module = _content_gated_residual().to(dtype=torch.bfloat16)
    _enable_nonzero_residual(module)
    scene_tokens = torch.randn(2, 9, 16, dtype=torch.bfloat16)

    pre_cast_delta = module.centered_delta_values(scene_tokens)
    effective_delta = module(scene_tokens).float() - scene_tokens.float()

    assert pre_cast_delta.dtype == torch.float32
    assert pre_cast_delta.mean(dim=1).abs().max().item() <= 1e-7
    assert torch.isfinite(effective_delta).all()


def test_content_gated_nonzero_residual_connects_every_input_slot() -> None:
    module = _content_gated_residual()
    _enable_nonzero_residual(module)
    scene_tokens = torch.randn(2, 9, 16, requires_grad=True)

    residual_at_first_slot = (module(scene_tokens) - scene_tokens)[:, 0]
    gradient = torch.autograd.grad(residual_at_first_slot.square().sum(), scene_tokens)[0]

    assert torch.all(gradient.abs().sum(dim=-1) > 0)
    assert torch.isfinite(gradient).all()


def test_content_gated_zero_output_opens_only_output_projection_gradient() -> None:
    module = _content_gated_residual()
    scene_tokens = torch.randn(2, 9, 16)
    target = torch.randn_like(scene_tokens)

    loss = (module(scene_tokens) * target).sum()
    loss.backward()

    assert module.output_projection.weight.grad is not None
    assert torch.count_nonzero(module.output_projection.weight.grad).item() > 0
    for name, parameter in module.named_parameters():
        if name == "output_projection.weight":
            continue
        assert parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0


def test_content_gated_state_hash_is_deterministic_without_consuming_rng() -> None:
    torch.manual_seed(18019)
    rng_before = torch.random.get_rng_state().clone()
    first_module = _content_gated_residual(seed=18018)
    first = module_collection_state_sha256({"global_scene_residual": first_module})

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    repeated = module_collection_state_sha256(
        {"global_scene_residual": _content_gated_residual(seed=18018)}
    )
    different = module_collection_state_sha256(
        {"global_scene_residual": _content_gated_residual(seed=18020)}
    )
    assert first == repeated
    assert first != different

    before = first
    with torch.no_grad():
        first_module.gate_temperature.add_(0.125)
    after = module_collection_state_sha256({"global_scene_residual": first_module})
    assert before != after
    first_module.gate_temperature.zero_()
    with pytest.raises(ValueError, match="temperature"):
        first_module.validate_structural_state()


def test_content_gated_structural_validation_rejects_loaded_drift_and_nonfinite_state() -> None:
    configured = _content_gated_residual()
    different_temperature = GlobalSceneResidual(
        scene_dim=16,
        latent_count=9,
        width=8,
        fourier_bands=2,
        initialization_seed=18018,
        architecture_version=ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
        gate_temperature=0.5,
    )

    configured.load_state_dict(different_temperature.state_dict(), strict=True)
    with pytest.raises(ValueError, match="active configuration"):
        configured.validate_structural_state()

    nonfinite = _content_gated_residual()
    with torch.no_grad():
        nonfinite.content_gate_projection.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite.*content_gate_projection.weight"):
        nonfinite.validate_structural_state()


def test_legacy_and_content_gated_checkpoint_states_are_strictly_incompatible() -> None:
    legacy = _residual()
    content_gated = _content_gated_residual()

    with pytest.raises(RuntimeError, match="Unexpected key|unexpected key"):
        legacy.load_state_dict(content_gated.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match="Missing key|missing key"):
        content_gated.load_state_dict(legacy.state_dict(), strict=True)
