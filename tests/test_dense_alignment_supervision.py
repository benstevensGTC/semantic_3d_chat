from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from semantic_3d_chat.scene_encoder.dense_alignment import DenseAlignmentResidual
from semantic_3d_chat.training.dense_alignment_supervision import (
    DenseAlignmentRegionTargets,
    build_object_region_targets,
    dense_alignment_calibration_objective,
    dense_alignment_region_contrastive_loss,
    dense_alignment_supervision_settings,
    dense_alignment_warmup_settings,
)


def _supervision_config() -> dict[str, object]:
    return {
        "training": {
            "dense_alignment_warmup": {
                "enabled": True,
                "training_only": True,
                "max_optimizer_steps": 20,
                "evaluation_interval_steps": 1,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "delta_rms_cap": 1.0,
                "delta_abs_max_cap": 3.5,
                "delta_rms_regularization_weight": 0.01,
                "early_stop_top1_accuracy": 1.0,
                "early_stop_minimum_margin": 0.10,
                "held_out_scene_gradient_access": False,
                "reset_pair_optimizer_after_warmup": True,
            },
            "dense_alignment_supervision": {
                "enabled": True,
                "training_only": True,
                "oracle_access_process": "training_and_evaluation_only",
                "runtime_oracle_access": False,
                "runtime_serializes_category_strings": False,
                "runtime_serializes_text_embeddings": False,
                "question_dependent_scene_processing": False,
                "all_voxels_transformed": True,
                "calibration_scene_ids": ["scene_000001", "scene_000002"],
                "held_out_scene_ids": ["scene_000007", "scene_000008"],
                "dense_dim": 1536,
                "aligned_start": 1536,
                "aligned_dim": 1536,
                "temperature": 0.07,
                "bbox_padding_m": 0.0375,
                "minimum_voxels_per_region": 8,
                "loss_weight": 1.0,
            }
        }
    }


def _oracle() -> dict[str, object]:
    return {
        "instances": [
            {
                "instance_id": "i_000002",
                "kind": "object",
                "category": "second",
                "visible_from_center_scan": True,
                "bbox": {"min_xyz_m": [1.0, 0.0, 0.0], "max_xyz_m": [1.2, 0.2, 0.2]},
            },
            {
                "instance_id": "i_000001",
                "kind": "object",
                "category": "first",
                "visible_from_center_scan": True,
                "bbox": {"min_xyz_m": [0.0, 0.0, 0.0], "max_xyz_m": [0.2, 0.2, 0.2]},
            },
            {
                "instance_id": "i_000000",
                "kind": "surface",
                "category": "ignored surface",
                "visible_from_center_scan": True,
                "bbox": {"min_xyz_m": [-2.0, -2.0, -0.1], "max_xyz_m": [2.0, 2.0, 0.0]},
            },
        ]
    }


def test_settings_are_fail_closed_and_exclude_held_out_scenes() -> None:
    settings = dense_alignment_supervision_settings(_supervision_config())

    assert settings.calibration_scene_ids == ("scene_000001", "scene_000002")
    assert settings.held_out_scene_ids == ("scene_000007", "scene_000008")
    assert settings.semantic_dim == 3072
    assert settings.contract()["checkpoint_tensor_payload_only"] is True
    assert settings.contract()["runtime_oracle_access"] is False
    contract = settings.contract()
    assert all(
        not isinstance(value, dict)
        for key, value in contract.items()
        if key not in {"runtime_serializes_category_strings"}
    )
    assert contract["runtime_serializes_category_strings"] is False


def test_warmup_settings_pin_bounded_stage_and_optimizer_reset() -> None:
    settings = dense_alignment_warmup_settings(_supervision_config())

    assert settings.max_optimizer_steps == 20
    assert settings.learning_rate == pytest.approx(0.01)
    assert settings.delta_rms_cap == pytest.approx(1.0)
    assert settings.delta_abs_max_cap == pytest.approx(3.5)
    assert settings.delta_rms_regularization_weight == pytest.approx(0.01)
    assert settings.contract()["stop_at_first_passing_evaluation"] is True
    assert settings.held_out_scene_gradient_access is False
    assert settings.reset_pair_optimizer_after_warmup is True


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("training_only", False, "training-only"),
        ("held_out_scene_gradient_access", True, "Held-out"),
        ("reset_pair_optimizer_after_warmup", False, "optimizer"),
        ("max_optimizer_steps", 0, "positive"),
        ("evaluation_interval_steps", 21, "cannot exceed"),
        ("delta_rms_cap", 0.0, "positive"),
        ("early_stop_top1_accuracy", 1.1, "in \\[0,1\\]"),
        ("enabled", 1, "boolean"),
        ("training_only", "yes", "boolean"),
        ("held_out_scene_gradient_access", 0, "boolean"),
        ("reset_pair_optimizer_after_warmup", 1, "boolean"),
    ],
)
def test_warmup_settings_fail_closed(
    key: str, value: object, message: str
) -> None:
    config = _supervision_config()
    config["training"]["dense_alignment_warmup"][key] = value  # type: ignore[index]
    with pytest.raises((TypeError, ValueError), match=message):
        dense_alignment_warmup_settings(config)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("training_only", False, "training-only"),
        ("runtime_oracle_access", True, "Runtime oracle"),
        ("runtime_serializes_category_strings", True, "category strings"),
        ("runtime_serializes_text_embeddings", True, "text embeddings"),
        ("question_dependent_scene_processing", True, "user question"),
        ("all_voxels_transformed", False, "complete voxel"),
    ],
)
def test_settings_reject_runtime_leakage_or_selection(
    key: str, value: object, message: str
) -> None:
    config = _supervision_config()
    config["training"]["dense_alignment_supervision"][key] = value  # type: ignore[index]
    with pytest.raises(ValueError, match=message):
        dense_alignment_supervision_settings(config)


def test_settings_reject_overlapping_calibration_and_held_out_scenes() -> None:
    config = _supervision_config()
    config["training"]["dense_alignment_supervision"]["held_out_scene_ids"] = [  # type: ignore[index]
        "scene_000002"
    ]
    with pytest.raises(ValueError, match="disjoint"):
        dense_alignment_supervision_settings(config)


def test_object_region_builder_is_deterministic_numeric_and_ignores_surfaces() -> None:
    centers = torch.tensor(
        [
            [0.05, 0.05, 0.05],
            [0.15, 0.15, 0.15],
            [1.05, 0.05, 0.05],
            [1.15, 0.15, 0.15],
            [0.5, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    targets = build_object_region_targets(
        centers,
        _oracle(),
        {"first": 0, "second": 1},
        padding_m=0.0,
        minimum_voxels_per_region=2,
    )

    assert targets.region_membership.tolist() == [
        [True, True, False, False, False],
        [False, False, True, True, False],
    ]
    assert targets.category_indices.tolist() == [0, 1]
    assert targets.voxel_counts.tolist() == [2, 2]
    assert targets.region_count == 2
    assert all(field.name not in {"category", "instance_id"} for field in fields(targets))
    targets.validate()


def test_region_builder_rejects_missing_category_mapping_and_empty_region() -> None:
    centers = torch.tensor([[0.05, 0.05, 0.05]], dtype=torch.float32)
    with pytest.raises(ValueError, match="category"):
        build_object_region_targets(
            centers,
            _oracle(),
            {"first": 0, "other": 1},
            minimum_voxels_per_region=1,
        )
    with pytest.raises(ValueError, match="too few"):
        build_object_region_targets(
            centers,
            _oracle(),
            {"first": 0, "second": 1},
            minimum_voxels_per_region=1,
        )


def _small_loss_inputs() -> tuple[
    DenseAlignmentResidual,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    module = DenseAlignmentResidual(
        semantic_dim=6,
        dense_dim=3,
        aligned_dim=3,
        rank=2,
        alpha=4.0,
        initialization_seed=25025,
    )
    semantic = torch.tensor(
        [
            [1.0, 0.0, -1.0, 0.2, 0.1, 0.3],
            [0.8, 0.1, -0.9, 0.2, 0.1, 0.3],
            [-1.0, 0.0, 1.0, 0.2, 0.1, 0.3],
            [-0.8, -0.1, 0.9, 0.2, 0.1, 0.3],
            [0.0, 1.0, -1.0, 0.2, 0.1, 0.3],
        ],
        dtype=torch.float32,
    )
    membership = torch.tensor(
        [
            [True, True, False, False, False],
            [False, False, True, True, False],
        ]
    )
    categories = torch.tensor([0, 1], dtype=torch.long)
    text = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    return module, semantic, membership, categories, text


def test_contrastive_loss_is_differentiable_only_through_alignment_state() -> None:
    module, semantic, membership, categories, text = _small_loss_inputs()
    transformed = module(semantic)
    loss, audit = dense_alignment_region_contrastive_loss(
        transformed,
        membership,
        categories,
        text,
        aligned_start=3,
        aligned_dim=3,
        temperature=0.1,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert module.alignment_a.grad is not None
    assert torch.count_nonzero(module.alignment_a.grad).item() == 0
    assert module.alignment_b.grad is not None
    assert torch.count_nonzero(module.alignment_b.grad).item() > 0
    assert text.grad is None
    assert audit["input_voxel_count"] == 5
    assert audit["supervised_region_count"] == 2
    assert audit["all_voxels_transformed_before_region_pooling"] is True
    assert audit["text_embeddings_detached"] is True
    assert audit["runtime_supervision_required"] is False
    assert all(
        not (isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(text.shape))
        for value in audit.values()
    )


def test_nonzero_alignment_can_reduce_region_text_contrastive_loss() -> None:
    module, semantic, membership, categories, text = _small_loss_inputs()
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.05, weight_decay=0.0)

    initial, _ = dense_alignment_region_contrastive_loss(
        module(semantic), membership, categories, text, aligned_start=3, aligned_dim=3
    )
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = dense_alignment_region_contrastive_loss(
            module(semantic), membership, categories, text, aligned_start=3, aligned_dim=3
        )
        loss.backward()
        optimizer.step()
    final, audit = dense_alignment_region_contrastive_loss(
        module(semantic), membership, categories, text, aligned_start=3, aligned_dim=3
    )

    assert float(final.detach()) < float(initial.detach())
    assert float(audit["top1_accuracy"]) == pytest.approx(1.0)
    assert torch.all(audit["correct_vs_best_alternate_margin"] > 0)


def test_calibration_objective_regularizes_numeric_delta_and_can_pass_gate() -> None:
    module, semantic, membership, categories, text = _small_loss_inputs()
    supervision = dense_alignment_supervision_settings(_supervision_config())
    warmup = dense_alignment_warmup_settings(_supervision_config())
    # The production parser is pinned to 3,072 dimensions; use its immutable
    # values with a tiny equivalent contract for this differentiability test.
    supervision = type(supervision)(
        **{
            **supervision.__dict__,
            "dense_dim": 3,
            "aligned_start": 3,
            "aligned_dim": 3,
            "temperature": 0.07,
        }
    )
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.05, weight_decay=0.0)
    audit: dict[str, object] = {}
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        total, audit = dense_alignment_calibration_objective(
            semantic,
            module(semantic),
            membership,
            categories,
            text,
            supervision=supervision,
            warmup=warmup,
        )
        total.backward()
        optimizer.step()

    assert torch.isfinite(total)
    assert float(audit["delta_regularization"]) > 0.0
    assert float(audit["delta_rms"]) <= warmup.delta_rms_cap
    assert float(audit["delta_abs_max"]) <= warmup.delta_abs_max_cap
    assert audit["held_out_scene_gradient_access"] is False
    assert text.grad is None


@pytest.mark.parametrize(
    ("mutator", "error", "message"),
    [
        (lambda m, c, t: (m[:, :-1], c, t), ValueError, "shape"),
        (lambda m, c, t: (m.float().mul(-1), c, t), ValueError, "non-negative"),
        (lambda m, c, t: (m, c.to(torch.int32), t), TypeError, "torch.long"),
        (lambda m, c, t: (m, torch.tensor([0, 2]), t), ValueError, "missing"),
        (lambda m, c, t: (m, c, t[:1]), ValueError, "C >= 2"),
    ],
)
def test_contrastive_loss_rejects_invalid_numeric_supervision(
    mutator: object, error: type[Exception], message: str
) -> None:
    module, semantic, membership, categories, text = _small_loss_inputs()
    changed_membership, changed_categories, changed_text = mutator(  # type: ignore[operator]
        membership, categories, text
    )
    with pytest.raises(error, match=message):
        dense_alignment_region_contrastive_loss(
            module(semantic),
            changed_membership,
            changed_categories,
            changed_text,
            aligned_start=3,
            aligned_dim=3,
        )


def test_region_target_validation_rejects_string_or_shape_free_numeric_drift() -> None:
    invalid = DenseAlignmentRegionTargets(
        region_membership=torch.tensor([[True, False]]),
        category_indices=torch.tensor([0], dtype=torch.long),
        voxel_counts=torch.tensor([2], dtype=torch.long),
        input_voxel_count=2,
    )
    with pytest.raises(ValueError, match="voxel_counts"):
        invalid.validate()
