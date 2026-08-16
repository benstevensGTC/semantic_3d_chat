from __future__ import annotations

import inspect
from dataclasses import dataclass

import pytest
import torch

from semantic_3d_chat.scene_encoder.block_cross_residual import (
    BLOCK_CROSS_RESIDUAL_ARCHITECTURE_VERSION,
    BLOCK_CROSS_RESIDUAL_PARAMETER_COUNT,
    BlockCrossResidual,
    apply_block_cross_residual,
    block_cross_residual_settings,
    construct_block_cross_residual,
    validate_block_cross_residual_state,
)

INITIAL_STATE_SHA256 = "72ae7f492f5953e58d809b6782d559dc64669637d5d6a79ae98f3a31296a12cd"


def _inputs(
    *, token_count: int = 7, seed: int = 901, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    scene = torch.randn(1, 256, 1536, generator=generator, dtype=dtype)
    blocks = torch.randn(token_count, 384, generator=generator, dtype=dtype)
    positions = torch.rand(token_count, 3, generator=generator, dtype=dtype).mul(2).sub(1)
    return scene, blocks, positions


def _enabled_config(expected_hash: str = INITIAL_STATE_SHA256) -> dict[str, object]:
    return {
        "scene_encoder": {
            "block_cross_residual": {
                "enabled": True,
                "attention_dim": 256,
                "heads": 4,
                "spatial_temperature": 0.20,
                "uniform_floor": 0.01,
                "residual_scale": 0.25,
                "initialization_seed": 35035,
                "expected_initial_state_sha256": expected_hash,
            }
        }
    }


def test_exact_surface_shapes_deterministic_hash_and_rng_isolation() -> None:
    torch.manual_seed(77)
    expected_next = torch.rand(5)
    torch.manual_seed(77)
    first = BlockCrossResidual()
    observed_next = torch.rand(5)
    repeated = BlockCrossResidual()

    assert torch.equal(observed_next, expected_next)
    assert first.parameter_count == BLOCK_CROSS_RESIDUAL_PARAMETER_COUNT == 983_040
    assert {name: tuple(parameter.shape) for name, parameter in first.named_parameters()} == {
        "w_q": (1536, 256),
        "w_k": (384, 256),
        "w_v": (384, 256),
        "w_o": (256, 1536),
    }
    assert all(parameter.dtype == torch.float32 for parameter in first.parameters())
    assert torch.count_nonzero(first.w_o).item() == 0
    assert first.state_sha256() == repeated.state_sha256() == INITIAL_STATE_SHA256
    audit = first.validate_structural_state()
    assert audit["architecture_version"] == BLOCK_CROSS_RESIDUAL_ARCHITECTURE_VERSION
    assert audit["parameter_count"] == 983_040
    assert audit["output_projection_exact_zero"] is True
    assert audit["all_blocks_processed"] is True
    assert audit["question_dependent_inputs"] is False
    assert audit["environmental_metadata_inputs"] is False


def test_zero_output_is_bit_exact_identity_with_full_coverage_audit() -> None:
    module = BlockCrossResidual()
    scene, blocks, positions = _inputs(token_count=11, dtype=torch.float16)

    output, audit = module.forward_with_audit(scene, blocks, positions)

    assert torch.equal(output, scene)
    assert output.dtype == scene.dtype
    assert audit["block_cross_residual_processed_block_tokens"].item() == 11
    assert audit["block_cross_residual_delta_rms"].item() == 0.0
    assert audit["block_cross_residual_attention_row_sum_max_error"].item() <= 2e-6
    assert audit["block_cross_residual_attention_min_weight"].item() + 1e-9 >= 0.01 / 11
    assert audit["block_cross_residual_min_block_contribution"].item() > 0.0
    assert list(inspect.signature(type(module).forward).parameters) == [
        "self",
        "base_scene_tokens",
        "block_tokens",
        "block_positions_normalized",
    ]


def test_attention_is_fp32_normalized_and_gives_every_block_a_floor() -> None:
    module = BlockCrossResidual()
    scene, blocks, positions = _inputs(token_count=13, dtype=torch.float16)

    weights = module.attention_weights(scene, blocks, positions)

    assert weights.dtype == torch.float32
    assert weights.shape == (1, 4, 256, 13)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 4, 256), atol=2e-6, rtol=0)
    assert weights.min().item() + 1e-9 >= 0.01 / 13
    assert torch.all(weights.sum(dim=(0, 1, 2)) > 0)


def test_nonaffine_normalization_and_block_positions_affect_attention() -> None:
    module = BlockCrossResidual()
    scene, blocks, positions = _inputs(token_count=6)
    common_block_offset = torch.linspace(-2.0, 2.0, 384)
    per_slot_offset = torch.linspace(-1.5, 1.5, 256).view(1, 256, 1)

    baseline = module.attention_weights(scene, blocks, positions)
    normalized_equivalent = module.attention_weights(
        scene.mul(3.0).add(per_slot_offset),
        blocks + common_block_offset,
        positions,
    )
    moved_positions = positions.roll(1, dims=0)
    moved = module.attention_weights(scene, blocks, moved_positions)

    assert torch.allclose(normalized_equivalent, baseline, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(moved, baseline)

    with torch.no_grad():
        generator = torch.Generator(device="cpu").manual_seed(17)
        module.w_o.normal_(std=0.005, generator=generator)
    assert not torch.allclose(
        module(scene, blocks, moved_positions), module(scene, blocks, positions)
    )


def test_nonzero_output_has_a_differentiable_route_from_every_block() -> None:
    module = BlockCrossResidual()
    scene, blocks, positions = _inputs(token_count=8)
    blocks.requires_grad_(True)
    with torch.no_grad():
        generator = torch.Generator(device="cpu").manual_seed(19)
        module.w_o.normal_(std=0.003, generator=generator)
    target = torch.randn(scene.shape, generator=torch.Generator().manual_seed(20))

    (module(scene, blocks, positions) * target).sum().backward()

    assert blocks.grad is not None
    assert torch.all(blocks.grad.abs().sum(dim=-1) > 0)


def test_first_backward_opens_wo_only_then_qkv_receive_gradients() -> None:
    module = BlockCrossResidual()
    scene, blocks, positions = _inputs(token_count=7)
    target = torch.randn(scene.shape, generator=torch.Generator().manual_seed(801))

    (module(scene, blocks, positions) * target).mean().backward()

    assert module.w_o.grad is not None
    assert torch.count_nonzero(module.w_o.grad).item() > 0
    for parameter in (module.w_q, module.w_k, module.w_v):
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad).item() == 0

    with torch.no_grad():
        module.w_o.add_(0.05 * module.w_o.grad.sign())
    module.zero_grad(set_to_none=True)
    (module(scene, blocks, positions) * target).mean().backward()

    for parameter in (module.w_q, module.w_k, module.w_v, module.w_o):
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad).item() > 0


def test_strict_settings_construct_validation_and_nested_config_rejection() -> None:
    config = _enabled_config()
    settings = block_cross_residual_settings(config)
    module = construct_block_cross_residual(
        config, scene_dim=1536, block_dim=384, latent_count=256
    )

    assert settings.contract()["parameter_count"] == 983_040
    assert settings.contract()["attention"] == "custom_fp32_all_block_cross_attention"
    assert module is not None
    assert validate_block_cross_residual_state(
        module,
        expected_parameter_count=983_040,
        expected_state_sha256=INITIAL_STATE_SHA256,
        context="V35 preflight",
    )["state_sha256"] == INITIAL_STATE_SHA256
    assert block_cross_residual_settings({"scene_encoder": {}}).contract() == {
        "schema_version": 1,
        "enabled": False,
    }
    assert (
        construct_block_cross_residual(
            {"scene_encoder": {}}, scene_dim=1536, block_dim=384, latent_count=256
        )
        is None
    )

    with pytest.raises(ValueError, match="requires expected_initial_state_sha256"):
        block_cross_residual_settings(
            {"scene_encoder": {"block_cross_residual": {"enabled": True}}}
        )
    with pytest.raises(ValueError, match="Unknown block_cross_residual settings"):
        block_cross_residual_settings(
            {"scene_encoder": {"block_cross_residual": {"surprise": True}}}
        )
    for key, bad in (
        ("attention_dim", 128),
        ("heads", 8),
        ("spatial_temperature", 0.3),
        ("uniform_floor", 0.0),
        ("residual_scale", 1.0),
    ):
        raw = dict(config["scene_encoder"]["block_cross_residual"])  # type: ignore[index]
        raw[key] = bad
        with pytest.raises(ValueError, match=key):
            block_cross_residual_settings(
                {"scene_encoder": {"block_cross_residual": raw}}
            )
    with pytest.raises(ValueError, match="scene_dim"):
        construct_block_cross_residual(config, scene_dim=768, block_dim=384, latent_count=256)
    with pytest.raises(ValueError, match="state SHA-256 mismatch"):
        validate_block_cross_residual_state(module, expected_state_sha256="0" * 64)


@dataclass
class _Output:
    scene_tokens: torch.Tensor
    native_latents: torch.Tensor
    block_tokens: torch.Tensor
    audit: dict[str, torch.Tensor]


def test_apply_replaces_only_scene_tokens_and_audit() -> None:
    module = BlockCrossResidual()
    scene, blocks, positions = _inputs(token_count=5)
    native = torch.randn(1, 256, 384)
    source = _Output(
        scene,
        native,
        blocks,
        {"block_token_positions_normalized": positions, "existing": torch.tensor(7)},
    )

    adapted = apply_block_cross_residual(source, module)

    assert adapted is not source
    assert torch.equal(adapted.scene_tokens, source.scene_tokens)
    assert adapted.native_latents is native
    assert adapted.block_tokens is blocks
    assert adapted.audit is not source.audit
    assert adapted.audit["existing"].item() == 7
    assert adapted.audit["block_cross_residual_processed_block_tokens"].item() == 5
    assert apply_block_cross_residual(source, None) is source


def test_structural_tamper_nonfinite_and_input_shapes_fail_closed() -> None:
    module = BlockCrossResidual()
    original_hash = module.state_sha256()
    with torch.no_grad():
        module.latent_anchors[0, 0].add_(0.25)
    assert module.state_sha256() != original_hash
    with pytest.raises(ValueError, match="latent anchors mismatch"):
        module.validate_structural_state()

    module = BlockCrossResidual()
    with torch.no_grad():
        module.spatial_temperature.add_(0.01)
    with pytest.raises(ValueError, match="spatial_temperature mismatch"):
        module.validate_structural_state()

    module = BlockCrossResidual()
    with torch.no_grad():
        module.w_k[0, 0] = float("inf")
    with pytest.raises(ValueError, match="w_k contains NaN or infinity"):
        module.validate_structural_state()

    module = BlockCrossResidual()
    scene, blocks, positions = _inputs(token_count=4)
    with pytest.raises(ValueError, match="base_scene_tokens must have shape"):
        module(scene[:, :-1], blocks, positions)
    with pytest.raises(ValueError, match="block_tokens must be nonempty"):
        module(scene, blocks[:, :-1], positions)
    with pytest.raises(ValueError, match="match block token count"):
        module(scene, blocks, positions[:-1])
    bad_scene = scene.clone()
    bad_scene[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="base_scene_tokens contains NaN or infinity"):
        module(bad_scene, blocks, positions)
