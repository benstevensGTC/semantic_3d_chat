from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from semantic_3d_chat.scene_encoder.dense_alignment import (
    DENSE_ALIGNMENT_ARCHITECTURE_MARKER,
    DENSE_ALIGNMENT_ARCHITECTURE_VERSION,
    DenseAlignmentResidual,
    construct_dense_alignment,
    dense_alignment_settings,
    validate_dense_alignment_state,
)

FULL_SIZE_INITIAL_STATE_SHA256 = "bdff604538cfc82ba16ce5a701a573a4a64de486791c6e4ae7c4d5faba094874"


def _small(*, seed: int = 25025) -> DenseAlignmentResidual:
    return DenseAlignmentResidual(
        semantic_dim=12,
        dense_dim=7,
        aligned_dim=5,
        rank=3,
        alpha=6.0,
        initialization_seed=seed,
    )


def test_default_contract_is_exact_zero_all_voxel_identity() -> None:
    module = DenseAlignmentResidual()
    semantic = torch.randn(11, 3072)

    output = module(semantic)
    audit = module.validate_structural_state()

    assert torch.equal(output, semantic)
    assert output.shape == semantic.shape
    assert module.alignment_a.shape == (8, 1536)
    assert module.alignment_b.shape == (1536, 8)
    assert torch.count_nonzero(module.alignment_b).item() == 0
    assert module.parameter_count == 24_576
    assert module.scale == 2.0
    assert module.state_sha256() == FULL_SIZE_INITIAL_STATE_SHA256
    assert audit == {
        "architecture_version": DENSE_ALIGNMENT_ARCHITECTURE_VERSION,
        "semantic_dim": 3072,
        "dense_dim": 1536,
        "aligned_dim": 1536,
        "rank": 8,
        "alpha": 16.0,
        "scale": 2.0,
        "parameter_count": 24_576,
        "dense_slice": [0, 1536],
        "aligned_tail_slice": [1536, 3072],
        "layer_norm_elementwise_affine": False,
        "layer_norm_eps": 1e-5,
        "question_dependent_inputs": False,
        "environmental_metadata_inputs": False,
        "spatial_reduction": "none",
        "every_voxel_retained": True,
        "b_exact_zero": True,
        "state_sha256": module.state_sha256(),
    }


def test_cpu_seed_is_reproducible_and_does_not_consume_global_rng() -> None:
    torch.manual_seed(91)
    expected_next = torch.rand(4)
    torch.manual_seed(91)
    first = _small(seed=17)
    observed_next = torch.rand(4)
    repeated = _small(seed=17)
    different = _small(seed=18)

    assert torch.equal(observed_next, expected_next)
    assert torch.equal(first.alignment_a, repeated.alignment_a)
    assert first.state_sha256() == repeated.state_sha256()
    assert not torch.equal(first.alignment_a, different.alignment_a)
    assert first.state_sha256() != different.state_sha256()
    assert torch.count_nonzero(first.alignment_b).item() == 0
    assert torch.count_nonzero(different.alignment_b).item() == 0


def test_nonzero_residual_matches_formula_and_changes_only_aligned_tail() -> None:
    module = _small()
    semantic = torch.randn(9, 12)
    with torch.no_grad():
        values = torch.linspace(-0.03, 0.04, module.alignment_b.numel())
        module.alignment_b.copy_(values.reshape_as(module.alignment_b))

    output = module(semantic)
    normalized = F.layer_norm(semantic[:, :7].float(), (7,), eps=1e-5)
    expected_delta = F.linear(F.linear(normalized, module.alignment_a), module.alignment_b) * (
        module.alpha / module.rank
    )

    assert torch.equal(output[:, :7], semantic[:, :7])
    assert torch.allclose(output[:, 7:], semantic[:, 7:] + expected_delta)
    assert torch.equal(module.residual_delta(semantic), expected_delta)
    assert module.validate_structural_state()["b_exact_zero"] is False


def test_residual_is_voxel_local_and_retains_every_row_in_order() -> None:
    module = _small()
    with torch.no_grad():
        module.alignment_b.normal_(mean=0.0, std=0.02)
    semantic = torch.randn(6, 12)
    changed_other_voxels = semantic.clone()
    changed_other_voxels[1:] += torch.randn_like(changed_other_voxels[1:]) * 50.0

    original = module(semantic)
    changed = module(changed_other_voxels)

    assert original.shape[0] == semantic.shape[0]
    assert torch.equal(original[0], changed[0])
    for index in range(semantic.shape[0]):
        single = module(semantic[index : index + 1])
        assert torch.allclose(single[0], original[index], atol=1e-7, rtol=1e-6)


def test_settings_constructor_and_state_validation_contract() -> None:
    config = {
        "scene_encoder": {
            "dense_alignment": {
                "enabled": True,
                "dense_dim": 7,
                "aligned_dim": 5,
                "rank": 3,
                "alpha": 6.0,
                "initialization_seed": 17,
                "expected_initial_state_sha256": "a" * 64,
            }
        }
    }
    settings = dense_alignment_settings(config)
    module = construct_dense_alignment(config, semantic_dim=12)

    assert settings.contract() == {
        "schema_version": 1,
        "enabled": True,
        "architecture_version": DENSE_ALIGNMENT_ARCHITECTURE_VERSION,
        "dense_dim": 7,
        "aligned_dim": 5,
        "rank": 3,
        "alpha": 6.0,
        "scale": 2.0,
        "initialization_seed": 17,
        "expected_initial_state_sha256": "a" * 64,
        "initialization_algorithm": "cpu_kaiming_uniform_a_exact_zero_b",
        "layer_norm_elementwise_affine": False,
        "question_dependent_inputs": False,
        "environmental_metadata_inputs": False,
        "spatial_reduction": "none",
        "every_voxel_retained": True,
    }
    assert isinstance(module, DenseAlignmentResidual)
    audit = validate_dense_alignment_state(
        module,
        expected_parameter_count=36,
        context="V25 preflight",
    )
    assert audit["state_sha256"] == module.state_sha256()
    with pytest.raises(ValueError, match="V25 preflight: parameter count mismatch"):
        validate_dense_alignment_state(
            module,
            expected_parameter_count=37,
            context="V25 preflight",
        )


def test_disabled_settings_and_parser_fail_closed() -> None:
    config = {"scene_encoder": {}}
    assert dense_alignment_settings(config).contract() == {
        "schema_version": 1,
        "enabled": False,
    }
    assert construct_dense_alignment(config, semantic_dim=3072) is None

    with pytest.raises(ValueError, match="requires expected_initial_state_sha256"):
        dense_alignment_settings({"scene_encoder": {"dense_alignment": {"enabled": True}}})
    with pytest.raises(ValueError, match="Unknown dense_alignment settings"):
        dense_alignment_settings({"scene_encoder": {"dense_alignment": {"surprise": True}}})


def test_zero_b_first_backward_updates_only_b_then_a_receives_gradient() -> None:
    module = _small()
    semantic = torch.randn(4, 12)
    target = torch.randn(4, 5)

    (module(semantic)[:, 7:] * target).sum().backward()

    assert module.alignment_a.grad is not None
    assert torch.count_nonzero(module.alignment_a.grad).item() == 0
    assert module.alignment_b.grad is not None
    assert torch.count_nonzero(module.alignment_b.grad).item() > 0
    with torch.no_grad():
        module.alignment_b.copy_(module.alignment_b.grad * 1e-3)
    module.zero_grad(set_to_none=True)
    module(semantic).square().sum().backward()
    assert module.alignment_a.grad is not None
    assert torch.count_nonzero(module.alignment_a.grad).item() > 0


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"rank": 0}, ValueError, "rank"),
        ({"rank": True}, ValueError, "rank"),
        ({"alpha": 0.0}, ValueError, "alpha"),
        ({"alpha": math.inf}, ValueError, "alpha"),
        ({"alpha": True}, TypeError, "alpha"),
        ({"initialization_seed": True}, TypeError, "initialization_seed"),
        ({"semantic_dim": 10, "dense_dim": 6, "aligned_dim": 5}, ValueError, "must equal"),
    ],
)
def test_constructor_rejects_invalid_contract(
    kwargs: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        DenseAlignmentResidual(**kwargs)


@pytest.mark.parametrize(
    ("semantic", "error", "message"),
    [
        (torch.randn(12), ValueError, "shape"),
        (torch.randn(2, 11), ValueError, "shape"),
        (torch.empty(0, 12), ValueError, "at least one voxel"),
        (torch.ones(2, 12, dtype=torch.int64), TypeError, "floating-point"),
        (torch.full((2, 12), math.nan), ValueError, "NaN or infinity"),
        (torch.full((2, 12), math.inf), ValueError, "NaN or infinity"),
    ],
)
def test_forward_rejects_invalid_semantic_inputs(
    semantic: torch.Tensor, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        _small()(semantic)


def test_structural_audit_rejects_marker_scale_dtype_and_nonfinite_tamper() -> None:
    marker = _small()
    with torch.no_grad():
        marker.architecture_marker.add_(1)
    with pytest.raises(ValueError, match="architecture marker"):
        marker.validate_structural_state()

    scale = _small()
    with torch.no_grad():
        scale.scaling.add_(1.0)
    with pytest.raises(ValueError, match="persistent scale"):
        scale.validate_structural_state()

    nonfinite = _small()
    with torch.no_grad():
        nonfinite.alignment_a[0, 0] = math.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        nonfinite.validate_structural_state()

    converted = _small().to(dtype=torch.bfloat16)
    assert converted.alignment_a.dtype == torch.float32
    assert converted.alignment_b.dtype == torch.float32
    assert converted.scaling.dtype == torch.float32
    assert converted.architecture_marker.item() == DENSE_ALIGNMENT_ARCHITECTURE_MARKER
    converted.validate_structural_state()
