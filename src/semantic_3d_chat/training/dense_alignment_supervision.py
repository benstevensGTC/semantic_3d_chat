"""Training-only supervision for dense visual-to-language alignment.

This module is deliberately split from the runtime dense-alignment residual.
Oracle boxes and category names may enter :func:`build_object_region_targets`
only inside an explicit training/evaluation process.  The differentiable loss
itself consumes only numeric tensors, detaches the frozen Gemma text targets,
and returns diagnostics that contain neither strings nor prototype vectors.

The learned runtime artifact is therefore just the state of
``DenseAlignmentResidual``.  No oracle box, category name, category index
mapping, text embedding, question, or selected voxel list is required to apply
that residual to every voxel before the question-independent scene prefix is
built.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

_SCENE_ID_PATTERN = re.compile(r"scene_[0-9]{6}")


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return parsed


def _finite_non_negative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite non-negative number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return parsed


def _scene_ids(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of opaque scene IDs")
    parsed = tuple(value)
    if not parsed:
        raise ValueError(f"{name} cannot be empty")
    if any(not isinstance(item, str) or _SCENE_ID_PATTERN.fullmatch(item) is None for item in parsed):
        raise ValueError(f"{name} must contain only opaque scene IDs")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} cannot contain duplicate scene IDs")
    return parsed


@dataclass(frozen=True)
class DenseAlignmentSupervisionSettings:
    """Fail-closed contract for the training-only oracle supervision path."""

    enabled: bool
    training_only: bool
    oracle_access_process: str
    runtime_oracle_access: bool
    runtime_serializes_category_strings: bool
    runtime_serializes_text_embeddings: bool
    question_dependent_scene_processing: bool
    all_voxels_transformed: bool
    calibration_scene_ids: tuple[str, ...]
    held_out_scene_ids: tuple[str, ...]
    dense_dim: int
    aligned_start: int
    aligned_dim: int
    temperature: float
    bbox_padding_m: float
    minimum_voxels_per_region: int
    loss_weight: float

    def __post_init__(self) -> None:
        if not self.enabled:
            raise ValueError("V25 dense-alignment supervision must be enabled")
        if not self.training_only:
            raise ValueError("Dense-alignment oracle supervision must be training-only")
        if self.oracle_access_process != "training_and_evaluation_only":
            raise ValueError("Oracle access must be restricted to training/evaluation")
        if self.runtime_oracle_access:
            raise ValueError("Runtime oracle access must be disabled")
        if self.runtime_serializes_category_strings:
            raise ValueError("Runtime checkpoint cannot serialize category strings")
        if self.runtime_serializes_text_embeddings:
            raise ValueError("Runtime checkpoint cannot serialize text embeddings")
        if self.question_dependent_scene_processing:
            raise ValueError("Dense alignment cannot depend on the user question")
        if not self.all_voxels_transformed:
            raise ValueError("Dense alignment must transform the complete voxel tensor")
        if set(self.calibration_scene_ids) & set(self.held_out_scene_ids):
            raise ValueError("Calibration and held-out scene splits must be disjoint")
        if self.aligned_start != self.dense_dim:
            raise ValueError("aligned_start must equal dense_dim for the pinned payload layout")

    @property
    def semantic_dim(self) -> int:
        return self.aligned_start + self.aligned_dim

    def contract(self) -> dict[str, Any]:
        """Return a string-safe contract with no environmental categories."""

        return {
            "schema_version": 1,
            "enabled": self.enabled,
            "training_only": self.training_only,
            "oracle_access_process": self.oracle_access_process,
            "runtime_oracle_access": self.runtime_oracle_access,
            "runtime_serializes_category_strings": self.runtime_serializes_category_strings,
            "runtime_serializes_text_embeddings": self.runtime_serializes_text_embeddings,
            "question_dependent_scene_processing": self.question_dependent_scene_processing,
            "all_voxels_transformed": self.all_voxels_transformed,
            "calibration_scene_ids": list(self.calibration_scene_ids),
            "held_out_scene_ids": list(self.held_out_scene_ids),
            "dense_dim": self.dense_dim,
            "aligned_start": self.aligned_start,
            "aligned_dim": self.aligned_dim,
            "semantic_dim": self.semantic_dim,
            "temperature": self.temperature,
            "bbox_padding_m": self.bbox_padding_m,
            "minimum_voxels_per_region": self.minimum_voxels_per_region,
            "loss_weight": self.loss_weight,
            "objective": "pooled_object_region_to_detached_gemma_token_embedding_infonce",
            "checkpoint_tensor_payload_only": True,
        }


@dataclass(frozen=True)
class DenseAlignmentWarmupSettings:
    """Bounded semantic-calibration stage before the paired QA optimizer."""

    enabled: bool
    training_only: bool
    max_optimizer_steps: int
    evaluation_interval_steps: int
    learning_rate: float
    weight_decay: float
    delta_rms_cap: float
    delta_abs_max_cap: float
    delta_rms_regularization_weight: float
    early_stop_top1_accuracy: float
    early_stop_minimum_margin: float
    held_out_scene_gradient_access: bool
    reset_pair_optimizer_after_warmup: bool

    def __post_init__(self) -> None:
        for name, value in {
            "enabled": self.enabled,
            "training_only": self.training_only,
            "held_out_scene_gradient_access": self.held_out_scene_gradient_access,
            "reset_pair_optimizer_after_warmup": self.reset_pair_optimizer_after_warmup,
        }.items():
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean")
        if not self.enabled:
            raise ValueError("V25 dense-alignment warm-up must be enabled")
        if not self.training_only:
            raise ValueError("Dense-alignment warm-up must be training-only")
        if self.held_out_scene_gradient_access:
            raise ValueError("Held-out scenes cannot contribute warm-up gradients")
        if not self.reset_pair_optimizer_after_warmup:
            raise ValueError("The paired QA optimizer must be reset after warm-up")
        if self.evaluation_interval_steps > self.max_optimizer_steps:
            raise ValueError(
                "evaluation_interval_steps cannot exceed max_optimizer_steps"
            )
        if not 0.0 <= self.early_stop_top1_accuracy <= 1.0:
            raise ValueError("early_stop_top1_accuracy must be in [0,1]")

    def contract(self) -> dict[str, Any]:
        """Return the exact bounded-stage contract used by preflight."""

        return {
            "schema_version": 1,
            "enabled": self.enabled,
            "training_only": self.training_only,
            "max_optimizer_steps": self.max_optimizer_steps,
            "evaluation_interval_steps": self.evaluation_interval_steps,
            "optimizer": "AdamW",
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "delta_rms_cap": self.delta_rms_cap,
            "delta_abs_max_cap": self.delta_abs_max_cap,
            "delta_rms_regularization_weight": self.delta_rms_regularization_weight,
            "regularizer": "weight_times_mean_squared_residual_delta",
            "early_stop_top1_accuracy": self.early_stop_top1_accuracy,
            "early_stop_minimum_margin": self.early_stop_minimum_margin,
            "stop_at_first_passing_evaluation": True,
            "held_out_scene_gradient_access": self.held_out_scene_gradient_access,
            "reset_pair_optimizer_after_warmup": self.reset_pair_optimizer_after_warmup,
        }


def dense_alignment_warmup_settings(
    config: Mapping[str, Any],
) -> DenseAlignmentWarmupSettings:
    """Parse the explicit V25 pre-QA semantic warm-up contract."""

    training = config.get("training")
    if not isinstance(training, Mapping):
        raise TypeError("training config must be a mapping")
    raw = training.get("dense_alignment_warmup")
    if not isinstance(raw, Mapping):
        raise TypeError("training.dense_alignment_warmup must be a mapping")
    allowed = {
        "enabled",
        "training_only",
        "max_optimizer_steps",
        "evaluation_interval_steps",
        "learning_rate",
        "weight_decay",
        "delta_rms_cap",
        "delta_abs_max_cap",
        "delta_rms_regularization_weight",
        "early_stop_top1_accuracy",
        "early_stop_minimum_margin",
        "held_out_scene_gradient_access",
        "reset_pair_optimizer_after_warmup",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown dense-alignment warm-up settings: {unknown}")
    return DenseAlignmentWarmupSettings(
        enabled=raw.get("enabled", False),
        training_only=raw.get("training_only", False),
        max_optimizer_steps=_positive_int(
            raw.get("max_optimizer_steps", 20), "max_optimizer_steps"
        ),
        evaluation_interval_steps=_positive_int(
            raw.get("evaluation_interval_steps", 1), "evaluation_interval_steps"
        ),
        learning_rate=_finite_positive_float(raw.get("learning_rate", 0.01), "learning_rate"),
        weight_decay=_finite_non_negative_float(
            raw.get("weight_decay", 0.0), "weight_decay"
        ),
        delta_rms_cap=_finite_positive_float(raw.get("delta_rms_cap", 1.0), "delta_rms_cap"),
        delta_abs_max_cap=_finite_positive_float(
            raw.get("delta_abs_max_cap", 3.5), "delta_abs_max_cap"
        ),
        delta_rms_regularization_weight=_finite_non_negative_float(
            raw.get("delta_rms_regularization_weight", 0.01),
            "delta_rms_regularization_weight",
        ),
        early_stop_top1_accuracy=_finite_non_negative_float(
            raw.get("early_stop_top1_accuracy", 1.0), "early_stop_top1_accuracy"
        ),
        early_stop_minimum_margin=_finite_positive_float(
            raw.get("early_stop_minimum_margin", 0.10),
            "early_stop_minimum_margin",
        ),
        held_out_scene_gradient_access=raw.get("held_out_scene_gradient_access", True),
        reset_pair_optimizer_after_warmup=raw.get(
            "reset_pair_optimizer_after_warmup", False
        ),
    )


def dense_alignment_supervision_settings(
    config: Mapping[str, Any],
) -> DenseAlignmentSupervisionSettings:
    """Parse the explicit V25 training-only supervision contract."""

    training = config.get("training")
    if not isinstance(training, Mapping):
        raise TypeError("training config must be a mapping")
    raw = training.get("dense_alignment_supervision")
    if not isinstance(raw, Mapping):
        raise TypeError("training.dense_alignment_supervision must be a mapping")
    allowed = {
        "enabled",
        "training_only",
        "oracle_access_process",
        "runtime_oracle_access",
        "runtime_serializes_category_strings",
        "runtime_serializes_text_embeddings",
        "question_dependent_scene_processing",
        "all_voxels_transformed",
        "calibration_scene_ids",
        "held_out_scene_ids",
        "dense_dim",
        "aligned_start",
        "aligned_dim",
        "temperature",
        "bbox_padding_m",
        "minimum_voxels_per_region",
        "loss_weight",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown dense-alignment supervision settings: {unknown}")

    bbox_padding_m = raw.get("bbox_padding_m", 0.0375)
    if isinstance(bbox_padding_m, bool) or not isinstance(bbox_padding_m, (int, float)):
        raise TypeError("bbox_padding_m must be a finite non-negative number")
    parsed_padding = float(bbox_padding_m)
    if not math.isfinite(parsed_padding) or parsed_padding < 0.0:
        raise ValueError("bbox_padding_m must be a finite non-negative number")

    return DenseAlignmentSupervisionSettings(
        enabled=raw.get("enabled", False),
        training_only=raw.get("training_only", False),
        oracle_access_process=str(raw.get("oracle_access_process", "")),
        runtime_oracle_access=raw.get("runtime_oracle_access", True),
        runtime_serializes_category_strings=raw.get(
            "runtime_serializes_category_strings", True
        ),
        runtime_serializes_text_embeddings=raw.get(
            "runtime_serializes_text_embeddings", True
        ),
        question_dependent_scene_processing=raw.get(
            "question_dependent_scene_processing", True
        ),
        all_voxels_transformed=raw.get("all_voxels_transformed", False),
        calibration_scene_ids=_scene_ids(
            raw.get("calibration_scene_ids", ()), "calibration_scene_ids"
        ),
        held_out_scene_ids=_scene_ids(
            raw.get("held_out_scene_ids", ()), "held_out_scene_ids"
        ),
        dense_dim=_positive_int(raw.get("dense_dim", 1536), "dense_dim"),
        aligned_start=_positive_int(raw.get("aligned_start", 1536), "aligned_start"),
        aligned_dim=_positive_int(raw.get("aligned_dim", 1536), "aligned_dim"),
        temperature=_finite_positive_float(raw.get("temperature", 0.07), "temperature"),
        bbox_padding_m=parsed_padding,
        minimum_voxels_per_region=_positive_int(
            raw.get("minimum_voxels_per_region", 8), "minimum_voxels_per_region"
        ),
        loss_weight=_finite_positive_float(raw.get("loss_weight", 1.0), "loss_weight"),
    )


@dataclass(frozen=True)
class DenseAlignmentRegionTargets:
    """Numeric training targets derived from oracle boxes in an isolated process."""

    region_membership: torch.Tensor
    category_indices: torch.Tensor
    voxel_counts: torch.Tensor
    input_voxel_count: int

    @property
    def region_count(self) -> int:
        return int(self.category_indices.numel())

    def validate(self) -> None:
        membership = self.region_membership
        if membership.ndim != 2 or membership.shape[1] != self.input_voxel_count:
            raise ValueError("region_membership must have shape [R,input_voxel_count]")
        if membership.dtype != torch.bool:
            raise TypeError("region_membership must be boolean")
        if self.category_indices.shape != (membership.shape[0],):
            raise ValueError("category_indices must have shape [R]")
        if self.category_indices.dtype != torch.long:
            raise TypeError("category_indices must use torch.long")
        if self.voxel_counts.shape != (membership.shape[0],):
            raise ValueError("voxel_counts must have shape [R]")
        if self.voxel_counts.dtype != torch.long:
            raise TypeError("voxel_counts must use torch.long")
        observed_counts = membership.sum(dim=1).to(dtype=torch.long)
        if not torch.equal(observed_counts.cpu(), self.voxel_counts.cpu()):
            raise ValueError("voxel_counts do not match region_membership")
        if membership.shape[0] < 1 or bool((observed_counts < 1).any()):
            raise ValueError("Every supervised object region must contain voxels")
        if bool((self.category_indices < 0).any()):
            raise ValueError("category_indices cannot be negative")


def build_object_region_targets(
    centers_world: torch.Tensor,
    oracle: Mapping[str, Any],
    category_to_index: Mapping[str, int],
    *,
    padding_m: float = 0.0375,
    minimum_voxels_per_region: int = 8,
) -> DenseAlignmentRegionTargets:
    """Build numeric object-box masks in an explicit training process.

    The returned dataclass contains no category or instance strings.  Callers
    must discard ``oracle`` and ``category_to_index`` after constructing the
    batch; neither is part of the learned runtime state.
    """

    if centers_world.ndim != 2 or centers_world.shape[1] != 3 or centers_world.shape[0] < 1:
        raise ValueError("centers_world must be nonempty [N,3]")
    if not torch.is_floating_point(centers_world) or not bool(torch.isfinite(centers_world).all()):
        raise ValueError("centers_world must be a finite floating-point tensor")
    if isinstance(padding_m, bool) or not isinstance(padding_m, (int, float)):
        raise TypeError("padding_m must be a finite non-negative number")
    padding = float(padding_m)
    if not math.isfinite(padding) or padding < 0.0:
        raise ValueError("padding_m must be a finite non-negative number")
    minimum = _positive_int(minimum_voxels_per_region, "minimum_voxels_per_region")

    if not isinstance(category_to_index, Mapping) or len(category_to_index) < 2:
        raise ValueError("category_to_index must contain at least two categories")
    parsed_indices: dict[str, int] = {}
    for category, index in category_to_index.items():
        if not isinstance(category, str) or not category.strip():
            raise ValueError("category_to_index keys must be non-empty category strings")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("category_to_index values must be non-negative integers")
        parsed_indices[category.strip()] = int(index)
    expected_indices = set(range(len(parsed_indices)))
    if set(parsed_indices.values()) != expected_indices:
        raise ValueError("category_to_index values must be contiguous from zero")

    instances = oracle.get("instances")
    if not isinstance(instances, list) or not instances:
        raise ValueError("Training oracle must contain a non-empty instances list")
    object_instances = [
        instance
        for instance in instances
        if isinstance(instance, Mapping)
        and instance.get("kind") == "object"
        and bool(instance.get("visible_from_center_scan", True))
    ]
    if not object_instances:
        raise ValueError("Training oracle contains no visible object instances")
    object_instances.sort(key=lambda item: str(item.get("instance_id", "")))

    masks: list[torch.Tensor] = []
    category_indices: list[int] = []
    voxel_counts: list[int] = []
    centers = centers_world.float()
    for instance in object_instances:
        category = instance.get("category")
        if not isinstance(category, str) or category.strip() not in parsed_indices:
            raise ValueError("Every visible object category must have a training index")
        bbox = instance.get("bbox")
        if not isinstance(bbox, Mapping):
            raise TypeError("Every visible object must contain a bounding box mapping")
        lower = torch.as_tensor(
            bbox.get("min_xyz_m"), device=centers.device, dtype=torch.float32
        )
        upper = torch.as_tensor(
            bbox.get("max_xyz_m"), device=centers.device, dtype=torch.float32
        )
        if lower.shape != (3,) or upper.shape != (3,):
            raise ValueError("Every object bounding box must contain two XYZ triples")
        if not bool(torch.isfinite(lower).all()) or not bool(torch.isfinite(upper).all()):
            raise ValueError("Object bounding boxes must be finite")
        if bool((upper < lower).any()):
            raise ValueError("Object bounding boxes cannot be inverted")
        mask = torch.all(
            (centers >= lower.sub(padding)) & (centers <= upper.add(padding)), dim=1
        )
        count = int(mask.sum().item())
        if count < minimum:
            raise ValueError("A supervised object region has too few mapped voxels")
        masks.append(mask)
        category_indices.append(parsed_indices[category.strip()])
        voxel_counts.append(count)

    targets = DenseAlignmentRegionTargets(
        region_membership=torch.stack(masks),
        category_indices=torch.tensor(
            category_indices, device=centers.device, dtype=torch.long
        ),
        voxel_counts=torch.tensor(voxel_counts, device=centers.device, dtype=torch.long),
        input_voxel_count=int(centers.shape[0]),
    )
    targets.validate()
    return targets


def dense_alignment_region_contrastive_loss(
    transformed_semantic: torch.Tensor,
    region_membership: torch.Tensor,
    category_indices: torch.Tensor,
    frozen_text_embeddings: torch.Tensor,
    *,
    aligned_start: int = 1536,
    aligned_dim: int = 1536,
    temperature: float = 0.07,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int | float | bool]]:
    """Contrast pooled transformed object regions with frozen Gemma text rows.

    ``transformed_semantic`` must be the complete tensor returned by the
    question-independent dense-alignment module.  The numeric region masks are
    training-only supervision: they choose loss regions but never change which
    voxels the runtime module transforms or which voxels enter the scene map.
    """

    if transformed_semantic.ndim != 2 or transformed_semantic.shape[0] < 1:
        raise ValueError("transformed_semantic must be nonempty [N,D]")
    if not torch.is_floating_point(transformed_semantic):
        raise TypeError("transformed_semantic must be floating point")
    if not bool(torch.isfinite(transformed_semantic).all()):
        raise ValueError("transformed_semantic contains NaN or infinity")
    parsed_start = _positive_int(aligned_start, "aligned_start")
    parsed_dim = _positive_int(aligned_dim, "aligned_dim")
    if parsed_start + parsed_dim != transformed_semantic.shape[1]:
        raise ValueError("aligned slice must be the complete trailing semantic payload")
    parsed_temperature = _finite_positive_float(temperature, "temperature")
    parsed_epsilon = _finite_positive_float(epsilon, "epsilon")

    if region_membership.ndim != 2 or region_membership.shape[1] != transformed_semantic.shape[0]:
        raise ValueError("region_membership must have shape [R,N]")
    if region_membership.shape[0] < 1:
        raise ValueError("region_membership must contain at least one object region")
    if region_membership.dtype == torch.bool:
        weights = region_membership.to(device=transformed_semantic.device, dtype=torch.float32)
    elif torch.is_floating_point(region_membership):
        weights = region_membership.to(device=transformed_semantic.device, dtype=torch.float32)
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError("Floating region_membership must be finite and non-negative")
    else:
        raise TypeError("region_membership must be boolean or floating point")
    counts = weights.sum(dim=1)
    if bool((counts <= 0).any()):
        raise ValueError("Every supervised object region must contain positive weight")

    if category_indices.shape != (region_membership.shape[0],):
        raise ValueError("category_indices must have shape [R]")
    if category_indices.dtype != torch.long:
        raise TypeError("category_indices must use torch.long")
    targets = category_indices.to(device=transformed_semantic.device)

    if (
        frozen_text_embeddings.ndim != 2
        or frozen_text_embeddings.shape[0] < 2
        or frozen_text_embeddings.shape[1] != parsed_dim
    ):
        raise ValueError(f"frozen_text_embeddings must have shape [C,{parsed_dim}] with C >= 2")
    if not torch.is_floating_point(frozen_text_embeddings):
        raise TypeError("frozen_text_embeddings must be floating point")
    if not bool(torch.isfinite(frozen_text_embeddings).all()):
        raise ValueError("frozen_text_embeddings contains NaN or infinity")
    if bool((targets < 0).any()) or bool((targets >= frozen_text_embeddings.shape[0]).any()):
        raise ValueError("category_indices reference a missing text embedding")

    aligned = transformed_semantic[:, parsed_start:].float()
    pooled = torch.matmul(weights, aligned) / counts.unsqueeze(1)
    pooled = F.normalize(pooled, dim=-1, eps=parsed_epsilon)
    # This detach is a safety boundary: no training call can update or serialize
    # the Gemma token table through this objective.
    text = F.normalize(
        frozen_text_embeddings.detach().to(
            device=transformed_semantic.device, dtype=torch.float32
        ),
        dim=-1,
        eps=parsed_epsilon,
    )
    if bool((pooled.norm(dim=-1) <= parsed_epsilon).any()):
        raise ValueError("A pooled transformed object region has zero norm")
    if bool((text.norm(dim=-1) <= parsed_epsilon).any()):
        raise ValueError("A frozen text embedding has zero norm")

    logits = torch.matmul(pooled, text.transpose(0, 1)) / parsed_temperature
    loss = F.cross_entropy(logits, targets)
    correct = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    alternate = logits.masked_fill(
        F.one_hot(targets, num_classes=logits.shape[1]).bool(),
        -torch.inf,
    ).max(dim=1).values
    margin = (correct - alternate) * parsed_temperature
    top1 = logits.argmax(dim=1)
    return loss, {
        "input_voxel_count": int(transformed_semantic.shape[0]),
        "all_voxels_transformed_before_region_pooling": True,
        "supervised_region_count": int(region_membership.shape[0]),
        "category_embedding_count": int(frozen_text_embeddings.shape[0]),
        "aligned_start": parsed_start,
        "aligned_dim": parsed_dim,
        "temperature": parsed_temperature,
        "region_voxel_weight": counts.detach(),
        "correct_cosine": (correct * parsed_temperature).detach(),
        "best_alternate_cosine": (alternate * parsed_temperature).detach(),
        "correct_vs_best_alternate_margin": margin.detach(),
        "top1_accuracy": (top1 == targets).float().mean().detach(),
        "text_embeddings_detached": True,
        "question_dependent_selection": False,
        "runtime_supervision_required": False,
    }


def dense_alignment_calibration_objective(
    original_semantic: torch.Tensor,
    transformed_semantic: torch.Tensor,
    region_membership: torch.Tensor,
    category_indices: torch.Tensor,
    frozen_text_embeddings: torch.Tensor,
    *,
    supervision: DenseAlignmentSupervisionSettings,
    warmup: DenseAlignmentWarmupSettings,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int | float | bool]]:
    """Return the preregistered contrastive objective plus delta regularizer.

    This is the pure differentiable callable used by the trainer's bounded
    warm-up runner.  It owns no optimizer, performs no file I/O, and receives
    only numeric tensors after the isolated training process has converted
    oracle boxes and category strings into masks, indices, and detached Gemma
    token embeddings.
    """

    if original_semantic.shape != transformed_semantic.shape:
        raise ValueError("original and transformed semantic tensors must have equal shape")
    if original_semantic.ndim != 2 or original_semantic.shape[1] != supervision.semantic_dim:
        raise ValueError(
            f"original_semantic must have shape [N,{supervision.semantic_dim}]"
        )
    if original_semantic.device != transformed_semantic.device:
        raise ValueError("original and transformed semantic tensors must share a device")
    if not torch.is_floating_point(original_semantic):
        raise TypeError("original_semantic must be floating point")
    if not bool(torch.isfinite(original_semantic).all()):
        raise ValueError("original_semantic contains NaN or infinity")

    contrastive, contrastive_audit = dense_alignment_region_contrastive_loss(
        transformed_semantic,
        region_membership,
        category_indices,
        frozen_text_embeddings,
        aligned_start=supervision.aligned_start,
        aligned_dim=supervision.aligned_dim,
        temperature=supervision.temperature,
    )
    delta = transformed_semantic[:, supervision.aligned_start :].float() - original_semantic[
        :, supervision.aligned_start :
    ].float()
    mean_squared_delta = delta.square().mean()
    delta_rms = mean_squared_delta.sqrt()
    delta_abs_max = delta.abs().max()
    regularization = mean_squared_delta * warmup.delta_rms_regularization_weight
    total = contrastive * supervision.loss_weight + regularization
    if not bool(torch.isfinite(total)):
        raise RuntimeError("Dense-alignment calibration objective became non-finite")
    margins = contrastive_audit["correct_vs_best_alternate_margin"]
    if not isinstance(margins, torch.Tensor):
        raise TypeError("Contrastive audit did not return numeric margins")
    top1_accuracy = contrastive_audit["top1_accuracy"]
    if not isinstance(top1_accuracy, torch.Tensor):
        raise TypeError("Contrastive audit did not return numeric accuracy")
    passes = bool(
        float(top1_accuracy) >= warmup.early_stop_top1_accuracy
        and float(margins.min()) >= warmup.early_stop_minimum_margin
        and float(delta_rms.detach()) <= warmup.delta_rms_cap
        and float(delta_abs_max.detach()) <= warmup.delta_abs_max_cap
    )
    return total, {
        **contrastive_audit,
        "contrastive_loss": contrastive.detach(),
        "loss_weight": supervision.loss_weight,
        "delta_mean_squared": mean_squared_delta.detach(),
        "delta_rms": delta_rms.detach(),
        "delta_abs_max": delta_abs_max.detach(),
        "delta_regularization": regularization.detach(),
        "delta_rms_cap": warmup.delta_rms_cap,
        "delta_abs_max_cap": warmup.delta_abs_max_cap,
        "minimum_margin": margins.min().detach(),
        "early_stop_passed": passes,
        "held_out_scene_gradient_access": False,
    }


__all__ = [
    "DenseAlignmentRegionTargets",
    "DenseAlignmentSupervisionSettings",
    "DenseAlignmentWarmupSettings",
    "build_object_region_targets",
    "dense_alignment_calibration_objective",
    "dense_alignment_region_contrastive_loss",
    "dense_alignment_supervision_settings",
    "dense_alignment_warmup_settings",
]
