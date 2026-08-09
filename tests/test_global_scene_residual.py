from __future__ import annotations

import inspect

import pytest
import torch

from semantic_3d_chat.scene_encoder.global_residual import GlobalSceneResidual
from semantic_3d_chat.training.checkpointing import module_collection_state_sha256


def _residual(*, seed: int = 16015) -> GlobalSceneResidual:
    return GlobalSceneResidual(
        scene_dim=16,
        latent_count=9,
        width=8,
        fourier_bands=2,
        initialization_seed=seed,
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
