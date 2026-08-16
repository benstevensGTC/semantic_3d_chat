from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from semantic_3d_chat.language.prefix_injection import ContinuousPrefixComposer
from semantic_3d_chat.scene_encoder.dense_alignment import DenseAlignmentResidual
from semantic_3d_chat.scene_encoder.dense_sidecar_adapter import (
    DenseSidecarAdapter,
    apply_dense_sidecar_adapter,
)
from semantic_3d_chat.scene_encoder.global_residual import (
    ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
    GlobalSceneResidual,
)
from semantic_3d_chat.scene_encoder.map_io import MapTensorData
from semantic_3d_chat.scene_encoder.projector import SceneTokenizerOutput
from semantic_3d_chat.scene_encoder.signed_x_residual import SignedXSceneResidual
from semantic_3d_chat.training.train_adapter import (
    build_adapter_optimizer,
    declared_dense_alignment_parameter_count,
    dense_alignment_resume_metadata_mismatch,
    dense_alignment_source_checkpoint_modules,
    map_forward,
    training_map_forward,
    validate_dense_alignment_calibration_audit,
    verify_zero_output_dense_alignment_equivalence,
)


class _TinySceneTokenizer(nn.Module):
    def __init__(self, semantic_dim: int = 6, scene_dim: int = 4, latents: int = 2) -> None:
        super().__init__()
        self.projection = nn.Linear(semantic_dim, scene_dim, bias=False)
        self.latents = latents
        self.last_semantic: torch.Tensor | None = None

    def forward(
        self,
        semantic: torch.Tensor,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        normal: torch.Tensor,
        confidence: torch.Tensor,
        observation_count: torch.Tensor,
        room_min: torch.Tensor,
        room_max: torch.Tensor,
    ) -> SceneTokenizerOutput:
        del xyz, rgb, normal, confidence, observation_count, room_min, room_max
        self.last_semantic = semantic
        pooled = self.projection(semantic).mean(dim=0).reshape(1, 1, -1)
        tokens = pooled.expand(1, self.latents, -1)
        return SceneTokenizerOutput(
            scene_tokens=tokens,
            native_latents=tokens,
            block_tokens=tokens,
            audit={"voxel_counts": torch.ones(1)},
        )


class _TinySidecarTokenizer(_TinySceneTokenizer):
    def forward(self, *args, aligned_sidecar=None, aligned_sidecar_scale=0.0, **kwargs):
        output = super().forward(*args, **kwargs)
        if aligned_sidecar is None:
            return output
        sidecar_tokens = aligned_sidecar.mean(dim=0).reshape(1, 1, -1)
        sidecar_tokens = sidecar_tokens.expand(1, self.latents, -1)
        scene_tokens = output.scene_tokens
        if aligned_sidecar_scale > 0:
            scene_tokens = scene_tokens + aligned_sidecar_scale * sidecar_tokens
        return SceneTokenizerOutput(
            scene_tokens=scene_tokens,
            native_latents=output.native_latents,
            block_tokens=output.block_tokens,
            audit=output.audit,
            aligned_sidecar_tokens=sidecar_tokens,
        )


def _map() -> MapTensorData:
    generator = torch.Generator().manual_seed(9)
    semantic = torch.randn(7, 6, generator=generator)
    return MapTensorData(
        semantic=semantic,
        xyz=torch.randn(7, 3, generator=generator),
        rgb=torch.rand(7, 3, generator=generator),
        normal=torch.randn(7, 3, generator=generator),
        confidence=torch.ones(7),
        observation_count=torch.ones(7),
        room_min=torch.tensor([-3.0, -2.5, 0.0]),
        room_max=torch.tensor([3.0, 2.5, 3.0]),
        source_voxel_count=7,
        input_voxel_size_m=0.1,
    )


def _dense() -> DenseAlignmentResidual:
    return DenseAlignmentResidual(
        semantic_dim=6,
        dense_dim=3,
        aligned_dim=3,
        rank=2,
        alpha=4.0,
        initialization_seed=25,
    )


def _sidecar_dense() -> DenseAlignmentResidual:
    module = DenseAlignmentResidual(
        semantic_dim=8,
        dense_dim=4,
        aligned_dim=4,
        rank=2,
        alpha=4.0,
        initialization_seed=25,
        application_mode="coverage_sidecar",
        sidecar_scale=0.0,
    )
    with torch.no_grad():
        module.alignment_b.fill_(0.05)
    return module


def _sidecar_map() -> MapTensorData:
    data = _map()
    generator = torch.Generator().manual_seed(91)
    data.semantic = torch.randn(7, 8, generator=generator)
    return data


def test_map_forward_applies_dense_alignment_without_mutating_map() -> None:
    data = _map()
    source = data.semantic.clone()
    dense = _dense()
    with torch.no_grad():
        dense.alignment_b.fill_(0.125)
    tokenizer = _TinySceneTokenizer()

    output = map_forward(tokenizer, data, dense_aligner=dense)

    assert output.scene_tokens.shape == (1, 2, 4)
    assert tokenizer.last_semantic is not None
    assert torch.equal(data.semantic, source)
    assert tokenizer.last_semantic.shape == source.shape
    assert not torch.equal(tokenizer.last_semantic, source)
    assert torch.equal(tokenizer.last_semantic[:, :3], source[:, :3])


def test_training_map_forward_routes_gradient_only_to_dense_surface() -> None:
    data = _map()
    tokenizer = _TinySceneTokenizer()
    tokenizer.requires_grad_(False)
    dense = _dense()
    with torch.no_grad():
        dense.alignment_b.fill_(0.05)

    output = training_map_forward(
        tokenizer,
        data,
        freeze_scene_adapter=True,
        dense_aligner=dense,
    )
    output.scene_tokens.square().mean().backward()

    assert all(parameter.grad is None for parameter in tokenizer.parameters())
    assert dense.alignment_a.grad is not None
    assert dense.alignment_b.grad is not None
    assert float(dense.alignment_b.grad.norm()) > 0.0
    assert torch.equal(data.semantic, _map().semantic)


def test_post_stack_sidecar_application_order_and_frozen_gradient_boundary() -> None:
    data = _sidecar_map()
    tokenizer = _TinySidecarTokenizer(semantic_dim=8)
    tokenizer.requires_grad_(False)
    dense = _sidecar_dense().requires_grad_(False)
    adapter = DenseSidecarAdapter(
        scene_dim=4,
        latent_count=2,
        width=3,
        fourier_bands=1,
        initialization_seed=28,
    )
    with torch.no_grad():
        adapter.output_projection.weight.fill_(0.02)

    base = map_forward(tokenizer, data, dense_aligner=dense)
    adapted = map_forward(
        tokenizer,
        data,
        dense_aligner=dense,
        dense_sidecar_adapter=adapter,
    )
    expected = apply_dense_sidecar_adapter(base, adapter)

    assert torch.equal(adapted.scene_tokens, expected.scene_tokens)
    assert not torch.equal(adapted.scene_tokens, base.scene_tokens)

    adapter.zero_grad(set_to_none=True)
    trained = training_map_forward(
        tokenizer,
        data,
        freeze_scene_adapter=True,
        dense_aligner=dense,
        dense_sidecar_adapter=adapter,
    )
    trained.scene_tokens.square().mean().backward()
    assert adapter.output_projection.weight.grad is not None
    assert float(adapter.output_projection.weight.grad.norm()) > 0.0
    assert all(parameter.grad is None for parameter in tokenizer.parameters())
    assert all(parameter.grad is None for parameter in dense.parameters())


def test_dense_optimizer_is_the_only_authorized_surface() -> None:
    dense = _dense()
    config = {
        "training": {
            "dense_alignment_learning_rate": 3.0e-4,
            "dense_alignment_weight_decay": 0.0,
        }
    }

    optimizer, selected = build_adapter_optimizer(
        config,
        [],
        None,
        None,
        dense_alignment_parameters=tuple(dense.parameters()),
    )

    assert selected == list(dense.parameters())
    assert [group["name"] for group in optimizer.param_groups] == ["dense_alignment"]
    assert optimizer.param_groups[0]["lr"] == 3.0e-4
    assert optimizer.param_groups[0]["weight_decay"] == 0.0
    foreign = nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match="cannot include scene"):
        build_adapter_optimizer(
            config,
            [foreign],
            None,
            None,
            dense_alignment_parameters=tuple(dense.parameters()),
        )


def test_dense_source_loader_excludes_only_fresh_dense_module() -> None:
    modules = {
        "scene_model": nn.Linear(1, 1),
        "composer": nn.Linear(1, 1),
        "grounding": nn.Linear(1, 1),
        "global_scene_residual": nn.Linear(1, 1),
        "signed_x_scene_residual": nn.Linear(1, 1),
        "lora_bank_alpha": nn.Linear(1, 1),
        "dense_aligner": _dense(),
    }

    selected = dense_alignment_source_checkpoint_modules(modules)

    assert set(selected) == set(modules) - {"dense_aligner"}
    assert selected["scene_model"] is modules["scene_model"]
    with pytest.raises(ValueError, match="fresh dense module"):
        dense_alignment_source_checkpoint_modules(
            {name: module for name, module in modules.items() if name != "dense_aligner"}
        )


def test_dense_resume_metadata_and_declared_count_fail_closed() -> None:
    dense = _dense()
    initial_hash = dense.state_sha256()
    metadata = {
        "dense_alignment_initial_state_sha256": initial_hash,
        "dense_alignment_parameter_count": dense.parameter_count,
    }
    assert (
        dense_alignment_resume_metadata_mismatch(
            metadata,
            dense,
            expected_initial_state_sha256=initial_hash,
        )
        is None
    )
    changed = deepcopy(metadata)
    changed["dense_alignment_parameter_count"] = dense.parameter_count - 1
    assert dense_alignment_resume_metadata_mismatch(
        changed,
        dense,
        expected_initial_state_sha256=initial_hash,
    ) == {
        "dense_alignment_parameter_count": {
            "checkpoint": dense.parameter_count - 1,
            "runtime": dense.parameter_count,
        }
    }
    assert declared_dense_alignment_parameter_count({}) is None
    assert (
        declared_dense_alignment_parameter_count(
            {"experiment": {"dense_alignment_trainable_parameter_count": 12}}
        )
        == 12
    )
    with pytest.raises(ValueError, match="must not declare both"):
        declared_dense_alignment_parameter_count(
            {
                "experiment": {
                    "dense_alignment_trainable_parameter_count": 12,
                    "dense_alignment_parameter_count": 12,
                }
            }
        )


def test_update_zero_equivalence_covers_full_frozen_scene_stack() -> None:
    data = _map()
    source = data.semantic.clone()
    tokenizer = _TinySceneTokenizer()
    global_residual = GlobalSceneResidual(
        scene_dim=4,
        latent_count=2,
        width=2,
        fourier_bands=1,
        initialization_seed=18,
        architecture_version=ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
        gate_temperature=0.75,
    )
    signed_x = SignedXSceneResidual(scene_dim=4, latent_count=2, content_dim=2)
    dense = _dense()
    composer = ContinuousPrefixComposer(4)

    audit = verify_zero_output_dense_alignment_equivalence(
        tokenizer,
        global_residual,
        signed_x,
        dense,
        composer,
        {"scene_000001": data},
        model_dtype=torch.float32,
    )

    assert audit["verified"] is True
    assert audit["all_voxels_transformed"] is True
    assert audit["source_map_mutated"] is False
    assert audit["scene_count"] == 1
    hashes = audit["scene_prefixes"]["scene_000001"]
    assert hashes["frozen_source_prefix_sha256"] == hashes["dense_aligned_prefix_sha256"]
    assert torch.equal(data.semantic, source)


def test_calibration_audit_binds_exact_bridge_and_empty_pair_optimizer() -> None:
    dense = _dense()
    state_hash = dense.state_sha256()
    audit = {
        "qa_update_authorized": True,
        "initial_state_sha256": state_hash,
        "final_state_sha256": state_hash,
        "training": {"final_state_sha256": state_hash},
        "pair_optimizer_state_empty_before_warmup": True,
        "pair_optimizer_rebuilt_after_warmup": True,
        "pair_optimizer_state_empty_after_warmup": True,
        "pair_optimizer_steps_before_qa": 0,
        "held_out_scene_gradient_access": False,
        "category_text_prototypes_serialized": False,
        "oracle_payload_retained": False,
    }

    validate_dense_alignment_calibration_audit(
        audit,
        dense,
        expected_initial_state_sha256=state_hash,
    )
    changed = deepcopy(audit)
    changed["pair_optimizer_steps_before_qa"] = 1
    with pytest.raises(ValueError, match="pair_optimizer_steps_before_qa"):
        validate_dense_alignment_calibration_audit(
            changed,
            dense,
            expected_initial_state_sha256=state_hash,
        )


def test_calibration_audit_resume_binds_pre_qa_hash_not_current_bridge() -> None:
    dense = _dense()
    initial_hash = dense.state_sha256()
    audit = {
        "qa_update_authorized": True,
        "initial_state_sha256": initial_hash,
        "final_state_sha256": initial_hash,
        "training": {"final_state_sha256": initial_hash},
        "pair_optimizer_state_empty_before_warmup": True,
        "pair_optimizer_rebuilt_after_warmup": True,
        "pair_optimizer_state_empty_after_warmup": True,
        "pair_optimizer_steps_before_qa": 0,
        "held_out_scene_gradient_access": False,
        "category_text_prototypes_serialized": False,
        "oracle_payload_retained": False,
    }
    with torch.no_grad():
        dense.alignment_b.add_(0.125)
    assert dense.state_sha256() != initial_hash

    validate_dense_alignment_calibration_audit(
        audit,
        dense,
        expected_initial_state_sha256=initial_hash,
        expected_calibration_final_state_sha256=initial_hash,
    )
    with pytest.raises(ValueError, match="final_state_sha256"):
        validate_dense_alignment_calibration_audit(
            audit,
            dense,
            expected_initial_state_sha256=initial_hash,
        )
