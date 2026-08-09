from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from semantic_3d_chat.scene_encoder import (
    SIGNED_X_LOCAL_FIELD_ARCHITECTURE_MARKER,
    SIGNED_X_LOCAL_FIELD_V2,
    SIGNED_X_MOMENT_V1,
    SignedXLocalFieldSceneResidual,
    SignedXSceneResidual,
    apply_signed_x_scene_residual,
    construct_signed_x_scene_residual,
    signed_x_scene_residual_settings,
)
from semantic_3d_chat.scene_encoder.global_residual import (
    ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
    GlobalSceneResidual,
)
from semantic_3d_chat.scene_encoder.projector import SceneTokenizerOutput
from semantic_3d_chat.training.checkpointing import module_collection_state_sha256
from semantic_3d_chat.training.train_adapter import training_map_forward

FULL_SIZE_INITIAL_STATE_SHA256 = "3f249307901df75ba07a758a7dc5b02c7c6ff9bbb969987741a106b8d8977ce1"


def _module() -> SignedXLocalFieldSceneResidual:
    return SignedXLocalFieldSceneResidual(scene_dim=16, latent_count=8, content_dim=6)


def _centered_content(batch: int = 2) -> torch.Tensor:
    slots = torch.arange(1, 9, dtype=torch.float32).view(1, 8, 1)
    channels = torch.arange(1, 7, dtype=torch.float32).view(1, 1, 6)
    batches = torch.arange(batch, dtype=torch.float32).view(batch, 1, 1)
    values = torch.sin(0.37 * slots * channels + 0.19 * batches)
    values = values + 0.2 * torch.cos(0.11 * slots * channels.square())
    return values - values.mean(dim=1, keepdim=True)


def _enable_nonzero_output(module: SignedXLocalFieldSceneResidual) -> None:
    with torch.no_grad():
        values = torch.linspace(-0.03, 0.03, module.output_projection.weight.numel())
        module.output_projection.weight.copy_(values.reshape_as(module.output_projection.weight))


def test_v2_local_field_formula_has_no_spatial_reduction_and_rank_exceeds_one() -> None:
    module = _module()
    content = _centered_content()
    signed = module.signed_x_anchors.view(1, 8, 1)

    hidden = module.hidden_values(content)
    expected = signed * torch.tanh(content.float())

    assert hidden.shape == content.shape
    assert hidden.dtype == torch.float32
    assert torch.equal(hidden, expected)
    assert torch.linalg.matrix_rank(hidden[0]).item() > 1
    assert not torch.allclose(hidden[:, :1].expand_as(hidden), hidden)


def test_v2_zero_output_state_is_distinct_and_owns_only_the_shared_projection() -> None:
    module = _module()
    base = torch.randn(2, 8, 16)
    content = _centered_content()

    assert torch.equal(module(base, content), base)
    assert list(dict(module.named_parameters())) == ["output_projection.weight"]
    assert module.parameter_count == 16 * 6
    assert set(module.state_dict()) == {
        "signed_x_anchors",
        "architecture_marker",
        "output_projection.weight",
    }
    assert module.architecture_marker.dtype == torch.int64
    assert module.architecture_marker.item() == SIGNED_X_LOCAL_FIELD_ARCHITECTURE_MARKER


def test_v2_delta_is_fp32_centered_and_odd_in_local_content() -> None:
    module = _module()
    _enable_nonzero_output(module)
    content = _centered_content()

    delta = module.centered_delta_values(content)
    opposite = module.centered_delta_values(-content)

    assert delta.shape == (2, 8, 16)
    assert delta.dtype == torch.float32
    assert delta.square().mean().item() > 0.0
    assert delta.mean(dim=1).abs().max().item() <= 1e-7
    assert torch.allclose(opposite, -delta, atol=1e-7, rtol=1e-6)


def test_v2_apply_reports_local_field_without_claiming_a_moment() -> None:
    module = _module()
    _enable_nonzero_output(module)
    content = _centered_content()
    output = SceneTokenizerOutput(
        scene_tokens=torch.randn(2, 8, 16),
        native_latents=torch.randn(2, 8, 6),
        block_tokens=torch.randn(2, 4, 6),
        audit={"existing": torch.tensor(1)},
    )

    adapted = apply_signed_x_scene_residual(output, module, content)

    assert adapted.audit["existing"].item() == 1
    assert adapted.audit["signed_x_scene_residual_delta_rms"].item() > 0.0
    assert adapted.audit["signed_x_scene_residual_local_field_rms"].item() > 0.0
    assert adapted.audit["signed_x_scene_residual_accounted_slots"].item() == 8
    assert (
        adapted.audit["signed_x_scene_residual_architecture_marker"].item()
        == SIGNED_X_LOCAL_FIELD_ARCHITECTURE_MARKER
    )
    assert "signed_x_scene_residual_moment_rms" not in adapted.audit


def test_v2_dispatch_contract_and_v1_backward_compatibility() -> None:
    config = {
        "scene_encoder": {
            "signed_x_scene_residual": {
                "enabled": True,
                "architecture_version": SIGNED_X_LOCAL_FIELD_V2,
                "expected_initial_state_sha256": "a" * 64,
            }
        }
    }
    settings = signed_x_scene_residual_settings(config)
    module = construct_signed_x_scene_residual(
        config,
        scene_dim=16,
        latent_count=8,
        content_dim=6,
    )

    assert settings.contract() == {
        "schema_version": 1,
        "enabled": True,
        "architecture_version": SIGNED_X_LOCAL_FIELD_V2,
        "expected_initial_state_sha256": "a" * 64,
        "spatial_statistic": "centered_local_content_times_unit_rms_signed_x",
        "spatial_reduction": "none",
        "spatial_centering": "all_slots_fp32",
        "trainable_surface": "bias_free_output_projection_only",
    }
    assert isinstance(module, SignedXLocalFieldSceneResidual)

    config["scene_encoder"]["signed_x_scene_residual"]["architecture_version"] = SIGNED_X_MOMENT_V1
    legacy = construct_signed_x_scene_residual(
        config,
        scene_dim=16,
        latent_count=8,
        content_dim=6,
    )
    assert type(legacy) is SignedXSceneResidual
    assert "spatial_reduction" not in signed_x_scene_residual_settings(config).contract()


def test_v2_structural_validation_rejects_architecture_marker_tamper() -> None:
    module = _module()
    audit = module.validate_structural_state()

    assert audit["architecture_version"] == SIGNED_X_LOCAL_FIELD_V2
    assert audit["spatial_reduction"] == "none"
    with torch.no_grad():
        module.architecture_marker.add_(1)
    with pytest.raises(ValueError, match="architecture marker"):
        module.validate_structural_state()


def test_dispatch_rejects_module_type_and_architecture_version_disagreement() -> None:
    content = _centered_content()
    output = SceneTokenizerOutput(
        scene_tokens=torch.randn(2, 8, 16),
        native_latents=torch.randn(2, 8, 6),
        block_tokens=torch.randn(2, 4, 6),
        audit={},
    )
    local_field = _module()
    local_field.architecture_version = SIGNED_X_MOMENT_V1
    with pytest.raises(ValueError, match="architecture version does not match"):
        local_field.validate_structural_state()
    with pytest.raises(ValueError, match="architecture version does not match"):
        apply_signed_x_scene_residual(output, local_field, content)

    legacy = SignedXSceneResidual(scene_dim=16, latent_count=8, content_dim=6)
    legacy.architecture_version = SIGNED_X_LOCAL_FIELD_V2
    with pytest.raises(ValueError, match="architecture version does not match"):
        apply_signed_x_scene_residual(output, legacy, content)


def test_v2_full_size_initial_state_hash_is_reproducible_and_distinct_from_v1() -> None:
    first = SignedXLocalFieldSceneResidual(scene_dim=1536, latent_count=256, content_dim=128)
    repeated = SignedXLocalFieldSceneResidual(scene_dim=1536, latent_count=256, content_dim=128)
    legacy = SignedXSceneResidual(scene_dim=1536, latent_count=256, content_dim=128)
    observed = module_collection_state_sha256({"signed_x_scene_residual": first})

    assert first.parameter_count == 196_608
    assert observed == FULL_SIZE_INITIAL_STATE_SHA256
    assert observed == module_collection_state_sha256({"signed_x_scene_residual": repeated})
    assert observed != module_collection_state_sha256({"signed_x_scene_residual": legacy})


def test_training_map_forward_freezes_v18_and_trains_only_v2_projection() -> None:
    class FrozenScene(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(4, 16, bias=False)

        def forward(self, semantic: torch.Tensor, *_args: torch.Tensor) -> SceneTokenizerOutput:
            tokens = self.projection(semantic)
            return SceneTokenizerOutput(
                scene_tokens=tokens,
                native_latents=tokens[..., :6],
                block_tokens=tokens[..., :6],
                audit={},
            )

    scene = FrozenScene().requires_grad_(False).eval()
    global_residual = (
        GlobalSceneResidual(
            scene_dim=16,
            latent_count=8,
            width=6,
            fourier_bands=2,
            architecture_version=ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
        )
        .requires_grad_(False)
        .eval()
    )
    with torch.no_grad():
        global_residual.output_projection.weight.fill_(0.01)
    local_field = _module()
    data = SimpleNamespace(
        semantic=torch.randn(1, 8, 4),
        xyz=torch.zeros(1, 8, 3),
        rgb=torch.zeros(1, 8, 3),
        normal=torch.zeros(1, 8, 3),
        confidence=torch.ones(1, 8),
        observation_count=torch.ones(1, 8),
        room_min=torch.zeros(3),
        room_max=torch.ones(3),
    )

    output = training_map_forward(
        scene,
        data,
        freeze_scene_adapter=True,
        global_scene_residual=global_residual,
        signed_x_scene_residual=local_field,
    )
    target = torch.randn_like(output.scene_tokens)
    (output.scene_tokens * target).sum().backward()

    assert not output.scene_tokens.is_inference()
    assert all(parameter.grad is None for parameter in scene.parameters())
    assert all(parameter.grad is None for parameter in global_residual.parameters())
    gradient = local_field.output_projection.weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient).item() > 0
    assert torch.isfinite(gradient).all()
