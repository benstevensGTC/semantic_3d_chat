"""Deterministic, oracle-isolated warm-up for dense alignment bridges.

This module is training/evaluation-only.  Oracle strings and Gemma token rows
exist only as short-lived local variables while the warm-up is running.  The
callable returned to the adapter trainer mutates an already-constructed
``DenseAlignmentResidual`` in place and returns a numeric/hash-only audit.  It
never returns or serializes categories, text embeddings, region masks, or
prototypes.

The sufficient statistic for a region is exact for the voxel-local bridge:
each dense voxel is LayerNorm'ed first, and those normalized vectors are then
averaged.  Because the two learned projections are linear, applying them to
that mean is exactly the same as pooling the per-voxel residuals.  The existing
aligned tail is likewise represented by its exact region mean.  Maps are read
one scene at a time and are never modified or replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from semantic_3d_chat.config import (
    artifact_root,
    load_config,
    project_path,
    reports_root,
)
from semantic_3d_chat.evaluation.gemma4_semantic_sanity import (
    GEMMA4_TOKEN_EMBEDDING_KEY,
    load_category_embeddings_selective,
    resolve_local_snapshot,
)
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.scene_encoder.dense_alignment import (
    DenseAlignmentResidual,
    construct_dense_alignment,
    dense_alignment_settings,
    validate_dense_alignment_state,
)
from semantic_3d_chat.scene_encoder.map_io import MapTensorData, load_map_tensors
from semantic_3d_chat.training.dense_alignment_supervision import (
    DenseAlignmentRegionTargets,
    DenseAlignmentSupervisionSettings,
    DenseAlignmentWarmupSettings,
    build_object_region_targets,
    dense_alignment_supervision_settings,
    dense_alignment_warmup_settings,
)

_EXPECTED_HELD_OUT_SCENES = ("scene_000007", "scene_000008")
_HELD_OUT_TARGET_CATEGORIES = ("bowl", "cabinet")
_SHA256_PATTERN_LENGTH = 64
_V25_CALIBRATION_SCENES = tuple(
    f"scene_{index:06d}" for index in (*range(1, 7), 9, 10)
)
_V26_CALIBRATION_SCENES = (
    "scene_000001",
    "scene_000002",
    "scene_000009",
    "scene_000010",
)
_V26_FORBIDDEN_SCENES = (
    "scene_000003",
    "scene_000004",
    "scene_000005",
    "scene_000006",
)
_QA_FINAL_TEST_SCENES = ("scene_000005", "scene_000006")


@dataclass(frozen=True)
class DenseAlignmentCalibrationProfile:
    """One exact, preregistered calibration contract; arbitrary arms are rejected."""

    version: int
    screen_key: str
    calibration_scene_ids: tuple[str, ...]
    held_out_scene_ids: tuple[str, ...]
    forbidden_scene_ids: tuple[str, ...]
    max_optimizer_steps: int
    learning_rate: float
    weight_decay: float
    delta_mse_regularization_weight: float
    include_access_audit: bool


_CALIBRATION_PROFILES = {
    25: DenseAlignmentCalibrationProfile(
        version=25,
        screen_key="v25_screen",
        calibration_scene_ids=_V25_CALIBRATION_SCENES,
        held_out_scene_ids=_EXPECTED_HELD_OUT_SCENES,
        forbidden_scene_ids=(),
        max_optimizer_steps=20,
        learning_rate=0.01,
        weight_decay=0.0001,
        delta_mse_regularization_weight=0.01,
        include_access_audit=False,
    ),
    26: DenseAlignmentCalibrationProfile(
        version=26,
        screen_key="v26_screen",
        calibration_scene_ids=_V26_CALIBRATION_SCENES,
        held_out_scene_ids=_EXPECTED_HELD_OUT_SCENES,
        forbidden_scene_ids=_V26_FORBIDDEN_SCENES,
        max_optimizer_steps=40,
        learning_rate=0.005,
        weight_decay=0.0001,
        delta_mse_regularization_weight=0.01,
        include_access_audit=True,
    ),
}


class MapLoader(Protocol):
    def __call__(self, scene_id: str) -> MapTensorData: ...


class OracleLoader(Protocol):
    def __call__(self, scene_id: str) -> Mapping[str, Any]: ...


class EmbeddingLoader(Protocol):
    def __call__(
        self,
        snapshot: Path,
        categories: Sequence[str],
        tokenizer: Any | None,
    ) -> tuple[np.ndarray | torch.Tensor, Mapping[str, Any]]: ...


@dataclass(frozen=True)
class DenseAlignmentRegionSummaries:
    """Numeric sufficient statistics for oracle-selected object regions."""

    mean_layernorm_dense: torch.Tensor
    mean_aligned_tail: torch.Tensor
    category_indices: torch.Tensor
    voxel_counts: torch.Tensor

    @property
    def region_count(self) -> int:
        return int(self.category_indices.numel())

    @property
    def category_count(self) -> int:
        return int(self.category_indices.max().item()) + 1

    def validate(
        self,
        *,
        dense_dim: int,
        aligned_dim: int,
        require_complete_categories: bool = False,
    ) -> None:
        region_count = self.region_count
        expected = {
            "mean_layernorm_dense": (region_count, dense_dim),
            "mean_aligned_tail": (region_count, aligned_dim),
            "category_indices": (region_count,),
            "voxel_counts": (region_count,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if value.device.type != "cpu":
                raise ValueError(f"{name} must remain on CPU")
        for name in ("mean_layernorm_dense", "mean_aligned_tail"):
            value = getattr(self, name)
            if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite CPU float32")
        if self.category_indices.dtype != torch.long:
            raise TypeError("category_indices must use torch.long")
        if self.voxel_counts.dtype != torch.long:
            raise TypeError("voxel_counts must use torch.long")
        if region_count < 1 or bool((self.category_indices < 0).any()):
            raise ValueError("Region summaries require at least one valid region")
        if bool((self.voxel_counts < 1).any()):
            raise ValueError("Every summarized region must contain voxels")
        observed = sorted({int(value) for value in self.category_indices.tolist()})
        if require_complete_categories and observed != list(range(max(observed) + 1)):
            raise ValueError("Every contiguous category index must appear in the summaries")


def summarize_dense_alignment_regions(
    semantic: torch.Tensor,
    targets: DenseAlignmentRegionTargets,
    module: DenseAlignmentResidual,
) -> DenseAlignmentRegionSummaries:
    """Compute exact mean-per-voxel-LayerNorm and aligned-tail statistics."""

    if semantic.device.type != "cpu" or module.alignment_a.device.type != "cpu":
        raise ValueError("Dense alignment calibration is CPU-only")
    if semantic.ndim != 2 or semantic.shape[1] != module.semantic_dim:
        raise ValueError(f"semantic must have shape [N,{module.semantic_dim}]")
    if semantic.dtype != torch.float32 or not bool(torch.isfinite(semantic).all()):
        raise ValueError("semantic must be finite CPU float32")
    targets.validate()
    if targets.input_voxel_count != semantic.shape[0]:
        raise ValueError("Region targets and semantic tensor have different voxel counts")

    dense = semantic[:, : module.dense_dim]
    tail = semantic[:, module.dense_dim :]
    normalized = F.layer_norm(
        dense,
        (module.dense_dim,),
        weight=None,
        bias=None,
        eps=module.layer_norm_eps,
    )
    weights = targets.region_membership.to(dtype=torch.float32)
    counts = targets.voxel_counts.to(dtype=torch.float32).unsqueeze(1)
    summaries = DenseAlignmentRegionSummaries(
        mean_layernorm_dense=(weights @ normalized).div(counts).contiguous(),
        mean_aligned_tail=(weights @ tail).div(counts).contiguous(),
        category_indices=targets.category_indices.detach().cpu().clone(),
        voxel_counts=targets.voxel_counts.detach().cpu().clone(),
    )
    summaries.validate(dense_dim=module.dense_dim, aligned_dim=module.aligned_dim)
    return summaries


def _concatenate_summaries(
    values: Sequence[DenseAlignmentRegionSummaries],
    module: DenseAlignmentResidual,
    *,
    expected_category_count: int,
) -> DenseAlignmentRegionSummaries:
    if not values:
        raise ValueError("At least one scene summary is required")
    result = DenseAlignmentRegionSummaries(
        mean_layernorm_dense=torch.cat([value.mean_layernorm_dense for value in values]),
        mean_aligned_tail=torch.cat([value.mean_aligned_tail for value in values]),
        category_indices=torch.cat([value.category_indices for value in values]),
        voxel_counts=torch.cat([value.voxel_counts for value in values]),
    )
    result.validate(
        dense_dim=module.dense_dim,
        aligned_dim=module.aligned_dim,
    )
    if result.region_count < 2:
        raise ValueError("Combined calibration requires at least two regions")
    observed = {int(value) for value in result.category_indices.tolist()}
    if observed != set(range(expected_category_count)):
        raise ValueError("Every calibration category must remain represented globally")
    return result


def _summary_forward(
    module: DenseAlignmentResidual,
    summaries: DenseAlignmentRegionSummaries,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = F.linear(summaries.mean_layernorm_dense, module.alignment_a)
    delta = F.linear(hidden, module.alignment_b) * module.scaling
    adapted_tail = summaries.mean_aligned_tail + delta
    if not bool(torch.isfinite(adapted_tail).all()):
        raise RuntimeError("Dense-alignment region summary became non-finite")
    return adapted_tail, delta


def _calibration_metrics(
    module: DenseAlignmentResidual,
    summaries: DenseAlignmentRegionSummaries,
    frozen_text_embeddings: torch.Tensor,
    *,
    supervision: DenseAlignmentSupervisionSettings,
    warmup: DenseAlignmentWarmupSettings,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    adapted_tail, delta = _summary_forward(module, summaries)
    pooled = F.normalize(adapted_tail, dim=-1, eps=1e-6)
    text = F.normalize(frozen_text_embeddings.detach(), dim=-1, eps=1e-6)
    logits = pooled @ text.transpose(0, 1)
    scaled_logits = logits / supervision.temperature
    targets = summaries.category_indices
    contrastive = F.cross_entropy(scaled_logits, targets)
    correct = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    alternate = (
        logits.masked_fill(F.one_hot(targets, num_classes=logits.shape[1]).bool(), -torch.inf)
        .max(dim=1)
        .values
    )
    margins = correct - alternate
    delta_mse = delta.square().mean()
    delta_rms = delta_mse.sqrt()
    delta_abs_max = delta.abs().max()
    total = contrastive * supervision.loss_weight + (
        delta_mse * warmup.delta_rms_regularization_weight
    )
    if not bool(torch.isfinite(total)):
        raise RuntimeError("Dense-alignment calibration loss became non-finite")
    top1_accuracy = (logits.argmax(dim=1) == targets).float().mean()
    minimum_margin = margins.min()
    passed = bool(
        float(top1_accuracy.detach()) >= warmup.early_stop_top1_accuracy
        and float(minimum_margin.detach()) >= warmup.early_stop_minimum_margin
        and float(delta_rms.detach()) <= warmup.delta_rms_cap
        and float(delta_abs_max.detach()) <= warmup.delta_abs_max_cap
    )
    return total, {
        "region_count": summaries.region_count,
        "category_count": int(frozen_text_embeddings.shape[0]),
        "contrastive_loss": float(contrastive.detach()),
        "total_loss": float(total.detach()),
        "top1_accuracy": float(top1_accuracy.detach()),
        "minimum_cosine_margin": float(minimum_margin.detach()),
        "mean_cosine_margin": float(margins.mean().detach()),
        "delta_mean_squared": float(delta_mse.detach()),
        "delta_rms": float(delta_rms.detach()),
        "delta_abs_max": float(delta_abs_max.detach()),
        "passed": passed,
    }


def train_dense_alignment_from_summaries(
    module: DenseAlignmentResidual,
    summaries: DenseAlignmentRegionSummaries,
    frozen_text_embeddings: torch.Tensor,
    *,
    supervision: DenseAlignmentSupervisionSettings,
    warmup: DenseAlignmentWarmupSettings,
) -> dict[str, Any]:
    """Train ``module`` in place using only numeric CPU sufficient statistics."""

    if module.alignment_a.device.type != "cpu":
        raise ValueError("Dense alignment warm-up must run on CPU")
    if frozen_text_embeddings.device.type != "cpu":
        raise ValueError("Frozen text embeddings must remain on CPU")
    if frozen_text_embeddings.shape != (summaries.category_count, module.aligned_dim):
        raise ValueError(
            "Frozen text embeddings must have shape "
            f"[{summaries.category_count},{module.aligned_dim}]"
        )
    if frozen_text_embeddings.dtype != torch.float32:
        raise TypeError("Frozen text embeddings must use float32")
    if not bool(torch.isfinite(frozen_text_embeddings).all()):
        raise ValueError("Frozen text embeddings contain NaN or infinity")
    summaries.validate(dense_dim=module.dense_dim, aligned_dim=module.aligned_dim)
    frozen_text_embeddings = frozen_text_embeddings.detach().contiguous()

    optimizer = torch.optim.AdamW(
        [module.alignment_a, module.alignment_b],
        lr=warmup.learning_rate,
        weight_decay=warmup.weight_decay,
    )
    history: list[dict[str, float | int | bool]] = []
    previous_training = module.training
    module.train()
    try:
        for step in range(1, warmup.max_optimizer_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss, _before = _calibration_metrics(
                module,
                summaries,
                frozen_text_embeddings,
                supervision=supervision,
                warmup=warmup,
            )
            loss.backward()
            for parameter in (module.alignment_a, module.alignment_b):
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError("Dense-alignment calibration gradient is missing/non-finite")
            optimizer.step()
            if step % warmup.evaluation_interval_steps != 0:
                continue
            with torch.no_grad():
                _loss, metrics = _calibration_metrics(
                    module,
                    summaries,
                    frozen_text_embeddings,
                    supervision=supervision,
                    warmup=warmup,
                )
            history.append({"optimizer_step": step, **metrics})
            if bool(metrics["passed"]):
                break
    finally:
        module.train(previous_training)

    if not history:
        raise RuntimeError("Dense-alignment warm-up produced no evaluation")
    structural = validate_dense_alignment_state(module, context="calibration warm-up")
    return {
        "optimizer_steps": int(history[-1]["optimizer_step"]),
        "stopped_at_first_pass": bool(history[-1]["passed"]),
        "calibration_passed": bool(history[-1]["passed"]),
        "history": history,
        "final_state_sha256": str(structural["state_sha256"]),
    }


def _visible_object_categories(oracle: Mapping[str, Any]) -> tuple[str, ...]:
    instances = oracle.get("instances")
    if not isinstance(instances, list) or not instances:
        raise ValueError("Training oracle must contain instances")
    categories = {
        str(instance["category"]).strip()
        for instance in instances
        if isinstance(instance, Mapping)
        and instance.get("kind") == "object"
        and bool(instance.get("visible_from_center_scan", True))
        and isinstance(instance.get("category"), str)
        and str(instance["category"]).strip()
    }
    if not categories:
        raise ValueError("Training oracle contains no visible object categories")
    return tuple(sorted(categories))


def _default_loaders(config: Mapping[str, Any]) -> tuple[MapLoader, OracleLoader]:
    def load_map(scene_id: str) -> MapTensorData:
        return load_map_tensors(
            project_path(dict(config), "maps", scene_id, "voxel_map.npz"),
            config["scene"]["room_size_m"],
            "cpu",
            input_voxel_size_m=config["scene_encoder"].get("input_voxel_size_m"),
        )

    def load_oracle(scene_id: str) -> Mapping[str, Any]:
        path = project_path(dict(config), "oracle", scene_id, "oracle.json")
        with path.open("r", encoding="utf-8") as handle:
            oracle = json.load(handle)
        if not isinstance(oracle, Mapping) or oracle.get("scene_id") != scene_id:
            raise ValueError("Oracle scene identity mismatch")
        return oracle

    return load_map, load_oracle


def _filter_underfilled_regions(
    centers_world: torch.Tensor,
    oracle: Mapping[str, Any],
    *,
    padding_m: float,
    minimum_voxels_per_region: int,
) -> tuple[dict[str, Any], int]:
    """Drop only per-scene object boxes below the configured voxel floor.

    Coarsening can leave a small object with fewer than eight voxels in one
    scene even though the same category is well represented in the remaining
    calibration scenes.  Such a region is not allowed to contribute a noisy
    summary.  Global category completeness is checked after concatenation, so
    this never silently removes a category from calibration.
    """

    instances = oracle.get("instances")
    if not isinstance(instances, list) or not instances:
        raise ValueError("Training oracle must contain instances")
    retained: list[Mapping[str, Any]] = []
    skipped = 0
    centers = centers_world.float()
    for instance in instances:
        if not isinstance(instance, Mapping):
            continue
        if instance.get("kind") != "object" or not bool(
            instance.get("visible_from_center_scan", True)
        ):
            continue
        bbox = instance.get("bbox")
        if not isinstance(bbox, Mapping):
            raise TypeError("Every visible object must contain a bounding box mapping")
        lower = torch.as_tensor(bbox.get("min_xyz_m"), dtype=torch.float32)
        upper = torch.as_tensor(bbox.get("max_xyz_m"), dtype=torch.float32)
        if lower.shape != (3,) or upper.shape != (3,):
            raise ValueError("Every object bounding box must contain two XYZ triples")
        mask = torch.all(
            (centers >= lower.sub(padding_m)) & (centers <= upper.add(padding_m)), dim=1
        )
        if int(mask.sum()) < minimum_voxels_per_region:
            skipped += 1
        else:
            retained.append(instance)
    if not retained:
        raise ValueError("A calibration scene has no sufficiently observed object regions")
    return {"instances": retained}, skipped


def _prepare_calibration_summaries(
    config: Mapping[str, Any],
    module: DenseAlignmentResidual,
    *,
    map_loader: MapLoader,
    oracle_loader: OracleLoader,
    supervision: DenseAlignmentSupervisionSettings,
) -> tuple[DenseAlignmentRegionSummaries, tuple[str, ...], int]:
    # First collect a deterministic training-only vocabulary from calibration
    # oracles.  Oracles are small, but each object is still discarded before
    # any map is opened.
    category_set: set[str] = set()
    for scene_id in sorted(supervision.calibration_scene_ids):
        oracle = oracle_loader(scene_id)
        category_set.update(_visible_object_categories(oracle))
        del oracle
    categories = tuple(sorted(category_set))
    if len(categories) < 2:
        raise ValueError("Calibration requires at least two object categories")
    category_to_index = {category: index for index, category in enumerate(categories)}

    per_scene: list[DenseAlignmentRegionSummaries] = []
    skipped_underfilled_regions = 0
    for scene_id in sorted(supervision.calibration_scene_ids):
        data = map_loader(scene_id)
        if data.semantic.device.type != "cpu":
            data = data.to("cpu")
        oracle = oracle_loader(scene_id)
        filtered_oracle, skipped = _filter_underfilled_regions(
            data.xyz,
            oracle,
            padding_m=supervision.bbox_padding_m,
            minimum_voxels_per_region=supervision.minimum_voxels_per_region,
        )
        skipped_underfilled_regions += skipped
        targets = build_object_region_targets(
            data.xyz,
            filtered_oracle,
            category_to_index,
            padding_m=supervision.bbox_padding_m,
            minimum_voxels_per_region=supervision.minimum_voxels_per_region,
        )
        per_scene.append(summarize_dense_alignment_regions(data.semantic, targets, module))
        # Explicitly sever every oracle/mask/map reference before the next
        # scene.  Only the compact numeric region summaries survive.
        del targets, filtered_oracle, oracle, data
    return (
        _concatenate_summaries(
            per_scene,
            module,
            expected_category_count=len(categories),
        ),
        categories,
        skipped_underfilled_regions,
    )


def _top_mean(values: torch.Tensor, count: int) -> torch.Tensor:
    if values.numel() < 1:
        raise ValueError("Cannot compute a top mean over an empty tensor")
    return values.topk(min(count, int(values.numel())), sorted=False).values.mean()


def _adapted_aligned_features(
    module: DenseAlignmentResidual,
    semantic: torch.Tensor,
    *,
    chunk_size: int = 2048,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, semantic.shape[0], chunk_size):
            selected = semantic[start : start + chunk_size]
            delta = module.residual_delta(selected)
            tail = selected[:, module.dense_dim :].float() + delta
            chunks.append(F.normalize(tail, dim=-1, eps=1e-6).cpu())
    return torch.cat(chunks)


def _held_out_localization(
    module: DenseAlignmentResidual,
    category_embeddings: torch.Tensor,
    categories: Sequence[str],
    *,
    map_loader: MapLoader,
    oracle_loader: OracleLoader,
    supervision: DenseAlignmentSupervisionSettings,
    top_k: int,
    minimum_precision: float,
    maximum_mirror_centroid_error_m: float,
) -> dict[str, Any]:
    category_to_index = {category: index for index, category in enumerate(categories)}
    missing = [name for name in _HELD_OUT_TARGET_CATEGORIES if name not in category_to_index]
    if missing:
        raise ValueError("Held-out target categories are absent from calibration")
    target_indices = [category_to_index[name] for name in _HELD_OUT_TARGET_CATEGORIES]
    target_embeddings = F.normalize(category_embeddings[target_indices], dim=-1, eps=1e-6)

    results: list[dict[str, Any]] = []
    centroids: dict[tuple[int, int], torch.Tensor | None] = {}
    held_out = tuple(sorted(supervision.held_out_scene_ids))
    for scene_index, scene_id in enumerate(held_out):
        data = map_loader(scene_id)
        if data.semantic.device.type != "cpu":
            data = data.to("cpu")
        oracle = oracle_loader(scene_id)
        all_targets = build_object_region_targets(
            data.xyz,
            oracle,
            category_to_index,
            padding_m=supervision.bbox_padding_m,
            minimum_voxels_per_region=supervision.minimum_voxels_per_region,
        )
        all_object_mask = all_targets.region_membership.any(dim=0)
        aligned = _adapted_aligned_features(module, data.semantic)
        similarities = aligned @ target_embeddings.transpose(0, 1)
        ranking_k = min(top_k, data.voxel_count)
        for category_slot, category_index in enumerate(target_indices):
            selected_regions = all_targets.category_indices == category_index
            if not bool(selected_regions.any()):
                raise ValueError("Held-out oracle is missing a required target category")
            target_mask = all_targets.region_membership[selected_regions].any(dim=0)
            distractor_mask = all_object_mask & ~target_mask
            if not bool(target_mask.any()) or not bool(distractor_mask.any()):
                raise ValueError("Held-out target/distractor regions must be non-empty")
            scores = similarities[:, category_slot]
            top_indices = torch.argsort(scores, descending=True, stable=True)[:ranking_k]
            hit_mask = target_mask[top_indices]
            hit_indices = top_indices[hit_mask]
            precision = float(hit_mask.float().mean())
            correct_target_top = _top_mean(scores[target_mask], top_k)
            distractor_top = _top_mean(scores[distractor_mask], top_k)
            other_slot = 1 - category_slot
            other_target_top = _top_mean(similarities[target_mask, other_slot], top_k)
            region_margin = float(correct_target_top - distractor_top)
            query_margin = float(correct_target_top - other_target_top)
            if hit_indices.numel():
                centroid = data.xyz[hit_indices].float().mean(dim=0)
            else:
                centroid = None
            centroids[(scene_index, category_slot)] = centroid
            results.append(
                {
                    "scene_index": scene_index,
                    "category_slot": category_slot,
                    "ranking_k": ranking_k,
                    "target_voxel_count": int(target_mask.sum()),
                    "top_k_target_count": int(hit_mask.sum()),
                    "hit_at_k": bool(hit_indices.numel()),
                    "precision_at_k": precision,
                    "region_margin": region_margin,
                    "correct_vs_distractor_margin": query_margin,
                    "predicted_target_centroid_xyz_m": (
                        [float(value) for value in centroid] if centroid is not None else None
                    ),
                }
            )
        del all_targets, oracle, data, aligned, similarities

    mirror_errors: list[float | None] = []
    for category_slot in range(len(_HELD_OUT_TARGET_CATEGORIES)):
        first_raw = centroids[(0, category_slot)]
        second = centroids[(1, category_slot)]
        if first_raw is None or second is None:
            mirror_errors.append(None)
            continue
        first = first_raw.clone()
        first[0] = -first[0]
        mirror_errors.append(float(torch.linalg.vector_norm(first - second)))

    all_hit = all(bool(result["hit_at_k"]) for result in results)
    minimum_observed_precision = min(float(result["precision_at_k"]) for result in results)
    minimum_region_margin = min(float(result["region_margin"]) for result in results)
    minimum_query_margin = min(float(result["correct_vs_distractor_margin"]) for result in results)
    finite_mirror_errors = [value for value in mirror_errors if value is not None]
    mirror_centroids_available = len(finite_mirror_errors) == len(mirror_errors)
    maximum_mirror_error = (
        max(finite_mirror_errors)
        if mirror_centroids_available
        else maximum_mirror_centroid_error_m + 1.0
    )
    passed = bool(
        all_hit
        and minimum_observed_precision >= minimum_precision
        and minimum_region_margin > 0.0
        and minimum_query_margin > 0.0
        and mirror_centroids_available
        and math.isfinite(maximum_mirror_error)
        and maximum_mirror_error <= maximum_mirror_centroid_error_m
    )
    return {
        "target_region_count": len(results),
        "top_k": top_k,
        "minimum_precision_required": minimum_precision,
        "maximum_mirror_centroid_error_required_m": maximum_mirror_centroid_error_m,
        "all_target_hit_at_k": all_hit,
        "minimum_precision_at_k": minimum_observed_precision,
        "minimum_region_margin": minimum_region_margin,
        "minimum_correct_vs_distractor_margin": minimum_query_margin,
        "mirror_centroids_available": mirror_centroids_available,
        "maximum_mirror_centroid_error_m": maximum_mirror_error,
        "mirror_centroid_errors_m": mirror_errors,
        "targets": results,
        "passed": passed,
    }


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_numeric_hash_report(value: Any, *, location: str = "report") -> None:
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} contains NaN or infinity")
        return
    if isinstance(value, str):
        if len(value) != _SHA256_PATTERN_LENGTH or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"{location} contains a non-hash string value")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{location} contains a non-string key")
            _validate_numeric_hash_report(nested, location=f"{location}.{key}")
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _validate_numeric_hash_report(nested, location=f"{location}[{index}]")
        return
    raise TypeError(f"{location} contains a forbidden value type {type(value).__name__}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _validate_numeric_hash_report(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _save_tensor_only_bridge(path: Path, module: DenseAlignmentResidual) -> str:
    if path.suffix != ".safetensors":
        raise ValueError("Dense-alignment bridge output must use .safetensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".safetensors", dir=path.parent
    )
    os.close(descriptor)
    try:
        tensors = {
            f"dense_aligner.{name}": value.detach().cpu().contiguous()
            for name, value in module.state_dict().items()
        }
        save_file(tensors, temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return _sha256_file(path)


def dense_alignment_calibration_profile(
    config: Mapping[str, Any],
) -> DenseAlignmentCalibrationProfile:
    """Select exactly one supported preregistered calibration profile."""

    active: list[DenseAlignmentCalibrationProfile] = []
    for profile in _CALIBRATION_PROFILES.values():
        value = config.get(profile.screen_key)
        if isinstance(value, Mapping):
            active.append(profile)
        elif value is not None:
            raise TypeError(f"{profile.screen_key} must be a mapping or null")
    if len(active) != 1:
        raise ValueError("Exactly one of v25_screen or v26_screen must be active")
    return active[0]


def _require_pinned_contract(
    config: Mapping[str, Any],
    module: DenseAlignmentResidual,
    supervision: DenseAlignmentSupervisionSettings,
    warmup: DenseAlignmentWarmupSettings,
) -> DenseAlignmentCalibrationProfile:
    profile = dense_alignment_calibration_profile(config)
    if set(supervision.calibration_scene_ids) & set(supervision.held_out_scene_ids):
        raise ValueError("Calibration and held-out scenes must remain disjoint")
    if supervision.calibration_scene_ids != profile.calibration_scene_ids:
        raise ValueError(
            f"V{profile.version} calibration scenes must equal the pinned split"
        )
    if supervision.held_out_scene_ids != profile.held_out_scene_ids:
        raise ValueError(
            f"V{profile.version} held-out scenes must equal the pinned mirrored pair"
        )
    if set(supervision.calibration_scene_ids) & set(profile.forbidden_scene_ids):
        raise ValueError(f"V{profile.version} forbidden scenes cannot enter calibration")
    if profile.version == 26 and not set(_QA_FINAL_TEST_SCENES).issubset(
        profile.forbidden_scene_ids
    ):
        raise ValueError("V26 must prohibit every final QA test scene")
    required = {
        "max_optimizer_steps": (
            warmup.max_optimizer_steps,
            profile.max_optimizer_steps,
        ),
        "evaluation_interval_steps": (warmup.evaluation_interval_steps, 1),
        "learning_rate": (warmup.learning_rate, profile.learning_rate),
        "weight_decay": (warmup.weight_decay, profile.weight_decay),
        "delta_rms_regularization_weight": (
            warmup.delta_rms_regularization_weight,
            profile.delta_mse_regularization_weight,
        ),
        "early_stop_top1_accuracy": (warmup.early_stop_top1_accuracy, 1.0),
        "early_stop_minimum_margin": (warmup.early_stop_minimum_margin, 0.10),
        "delta_rms_cap": (warmup.delta_rms_cap, 1.0),
        "delta_abs_max_cap": (warmup.delta_abs_max_cap, 3.5),
    }
    for name, (observed, expected) in required.items():
        if observed != expected:
            raise ValueError(
                f"V{profile.version} warm-up {name} must equal {expected}"
            )
    settings = dense_alignment_settings(config)
    if not settings.enabled:
        raise ValueError(f"V{profile.version} dense alignment must be enabled")
    if (settings.dense_dim, settings.aligned_dim) != (module.dense_dim, module.aligned_dim):
        raise ValueError("Dense-alignment config/module dimensions differ")
    if (supervision.dense_dim, supervision.aligned_dim) != (
        module.dense_dim,
        module.aligned_dim,
    ):
        raise ValueError("Dense-alignment supervision/module dimensions differ")
    if module.alignment_a.device.type != "cpu":
        raise ValueError(f"V{profile.version} dense-alignment warm-up is CPU-only")
    if torch.count_nonzero(module.alignment_b).item() != 0:
        raise ValueError(
            f"V{profile.version} warm-up must start from the exact-zero output bridge"
        )
    return profile


@dataclass
class _SceneAccessRecorder:
    profile: DenseAlignmentCalibrationProfile
    map_scene_ids: list[str]
    oracle_scene_ids: list[str]

    @classmethod
    def create(cls, profile: DenseAlignmentCalibrationProfile) -> _SceneAccessRecorder:
        return cls(profile=profile, map_scene_ids=[], oracle_scene_ids=[])

    @property
    def allowed_scene_ids(self) -> frozenset[str]:
        return frozenset(
            (*self.profile.calibration_scene_ids, *self.profile.held_out_scene_ids)
        )

    def record_map(self, scene_id: str) -> None:
        self._record(scene_id, self.map_scene_ids, "map")

    def record_oracle(self, scene_id: str) -> None:
        self._record(scene_id, self.oracle_scene_ids, "oracle")

    def _record(self, scene_id: str, destination: list[str], kind: str) -> None:
        if scene_id not in self.allowed_scene_ids:
            raise ValueError(
                f"V{self.profile.version} {kind} loader requested a scene outside its split"
            )
        if scene_id in self.profile.forbidden_scene_ids:
            raise ValueError(
                f"V{self.profile.version} {kind} loader requested a forbidden scene"
            )
        destination.append(scene_id)

    def report(self) -> dict[str, Any]:
        calibration = set(self.profile.calibration_scene_ids)
        held_out = set(self.profile.held_out_scene_ids)
        forbidden = set(self.profile.forbidden_scene_ids)
        qa_final = set(_QA_FINAL_TEST_SCENES)
        calibration_map = sum(value in calibration for value in self.map_scene_ids)
        calibration_oracle = sum(value in calibration for value in self.oracle_scene_ids)
        held_out_map = sum(value in held_out for value in self.map_scene_ids)
        held_out_oracle = sum(value in held_out for value in self.oracle_scene_ids)
        forbidden_map = sum(value in forbidden for value in self.map_scene_ids)
        forbidden_oracle = sum(value in forbidden for value in self.oracle_scene_ids)
        qa_final_map = sum(value in qa_final for value in self.map_scene_ids)
        qa_final_oracle = sum(value in qa_final for value in self.oracle_scene_ids)
        expected_calibration_count = len(self.profile.calibration_scene_ids)
        expected_held_out_count = len(self.profile.held_out_scene_ids)
        expected = {
            "map": expected_calibration_count + expected_held_out_count,
            "oracle": expected_calibration_count * 2 + expected_held_out_count,
            "calibration_map": expected_calibration_count,
            "calibration_oracle": expected_calibration_count * 2,
            "held_out_map": expected_held_out_count,
            "held_out_oracle": expected_held_out_count,
        }
        observed = {
            "map": len(self.map_scene_ids),
            "oracle": len(self.oracle_scene_ids),
            "calibration_map": calibration_map,
            "calibration_oracle": calibration_oracle,
            "held_out_map": held_out_map,
            "held_out_oracle": held_out_oracle,
        }
        if observed != expected:
            raise RuntimeError(
                f"V{self.profile.version} scene loader access counts drifted"
            )
        v26_final_test_access = self.profile.version == 26 and (
            qa_final_map or qa_final_oracle
        )
        if forbidden_map or forbidden_oracle or v26_final_test_access:
            raise RuntimeError(
                f"V{self.profile.version} touched a forbidden or final QA test scene"
            )
        return {
            "schema_version": 1,
            "calibration_split_sha256": _sha256_json(
                list(self.profile.calibration_scene_ids)
            ),
            "held_out_split_sha256": _sha256_json(
                list(self.profile.held_out_scene_ids)
            ),
            "forbidden_split_sha256": _sha256_json(
                list(self.profile.forbidden_scene_ids)
            ),
            "qa_final_test_split_sha256": _sha256_json(list(_QA_FINAL_TEST_SCENES)),
            "map_access_sequence_sha256": _sha256_json(self.map_scene_ids),
            "oracle_access_sequence_sha256": _sha256_json(self.oracle_scene_ids),
            "map_access_count": len(self.map_scene_ids),
            "oracle_access_count": len(self.oracle_scene_ids),
            "calibration_map_access_count": calibration_map,
            "calibration_oracle_access_count": calibration_oracle,
            "held_out_map_access_count": held_out_map,
            "held_out_oracle_access_count": held_out_oracle,
            "forbidden_map_access_count": forbidden_map,
            "forbidden_oracle_access_count": forbidden_oracle,
            "qa_final_test_map_access_count": qa_final_map,
            "qa_final_test_oracle_access_count": qa_final_oracle,
            "forbidden_zero_access": True,
            "qa_final_test_zero_access": True,
        }


def run_dense_alignment_calibration_warmup(
    config: Mapping[str, Any],
    module: DenseAlignmentResidual,
    *,
    model_snapshot: str | Path | None = None,
    tokenizer: Any | None = None,
    map_loader: MapLoader | None = None,
    oracle_loader: OracleLoader | None = None,
    embedding_loader: EmbeddingLoader | None = None,
    bridge_output: str | Path | None = None,
    report_output: str | Path | None = None,
) -> dict[str, Any]:
    """Run the pinned warm-up in place and return only a numeric/hash audit.

    The caller must check ``qa_update_authorized`` before constructing the
    paired-QA optimizer.  ``require_dense_alignment_calibration_authorized`` is
    provided as the fail-closed assertion.  Resume paths should load the
    checkpointed bridge state and must not call this warm-up again.
    """

    supervision = dense_alignment_supervision_settings(config)
    warmup = dense_alignment_warmup_settings(config)
    profile = _require_pinned_contract(config, module, supervision, warmup)
    initial_audit = validate_dense_alignment_state(module, context="pre-calibration")
    initial_hash = str(initial_audit["state_sha256"])

    default_map_loader, default_oracle_loader = _default_loaders(config)
    raw_map_loader = map_loader or default_map_loader
    raw_oracle_loader = oracle_loader or default_oracle_loader
    access_recorder = _SceneAccessRecorder.create(profile)

    def selected_map_loader(scene_id: str) -> MapTensorData:
        access_recorder.record_map(scene_id)
        return raw_map_loader(scene_id)

    def selected_oracle_loader(scene_id: str) -> Mapping[str, Any]:
        access_recorder.record_oracle(scene_id)
        return raw_oracle_loader(scene_id)

    summaries, categories, skipped_underfilled_regions = _prepare_calibration_summaries(
        config,
        module,
        map_loader=selected_map_loader,
        oracle_loader=selected_oracle_loader,
        supervision=supervision,
    )

    language = config.get("language")
    if not isinstance(language, Mapping):
        raise TypeError("language config must be a mapping")
    snapshot = resolve_local_snapshot(
        str(language["model_id"]),
        str(language["revision"]),
        model_snapshot,
    )
    selected_embedding_loader = embedding_loader
    if selected_embedding_loader is None:
        if tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as error:  # pragma: no cover - setup failure
                raise RuntimeError(
                    f"Transformers is required for V{profile.version} calibration"
                ) from error
            # The pinned Gemma snapshot predates Transformers' dictionary form
            # for model-specific extra tokens and stores a legacy list.  Bare
            # category tokenization needs none of those aliases, so override it
            # with the current empty mapping rather than mutating the snapshot.
            tokenizer = AutoTokenizer.from_pretrained(
                snapshot,
                local_files_only=True,
                trust_remote_code=False,
                extra_special_tokens={},
            )

        def selected_embedding_loader(
            source: Path,
            names: Sequence[str],
            selected_tokenizer: Any | None,
        ) -> tuple[np.ndarray | torch.Tensor, Mapping[str, Any]]:
            return load_category_embeddings_selective(
                source,
                names,
                tokenizer=selected_tokenizer,
                expected_dim=module.aligned_dim,
            )

    loaded_embeddings, load_audit = selected_embedding_loader(snapshot, categories, tokenizer)
    category_embeddings = torch.as_tensor(loaded_embeddings, dtype=torch.float32).cpu()
    if category_embeddings.shape != (len(categories), module.aligned_dim):
        raise ValueError("Selective Gemma category embedding shape mismatch")
    loaded_keys = load_audit.get("loaded_parameter_keys")
    if loaded_keys != [GEMMA4_TOKEN_EMBEDDING_KEY]:
        raise ValueError("Calibration may load only the Gemma input embedding tensor")
    if load_audit.get("selective_row_read") is not True:
        raise ValueError("Calibration requires row-selective Gemma embedding I/O")
    unique_rows = load_audit.get("unique_token_rows_read")
    if isinstance(unique_rows, bool) or not isinstance(unique_rows, int) or unique_rows < 1:
        raise ValueError("Selective Gemma load audit has an invalid row count")

    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        training = train_dense_alignment_from_summaries(
            module,
            summaries,
            category_embeddings,
            supervision=supervision,
            warmup=warmup,
        )
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.set_num_threads(previous_threads)

    screen = config.get(profile.screen_key)
    if not isinstance(screen, Mapping):
        raise TypeError(f"{profile.screen_key} config must be a mapping")
    localization_requirements = screen.get("held_out_localization_requires")
    if not isinstance(localization_requirements, Mapping):
        raise TypeError(
            f"{profile.screen_key}.held_out_localization_requires must be a mapping"
        )
    if int(localization_requirements.get("scene_count", -1)) != 2:
        raise ValueError(f"V{profile.version} localization scene_count must equal two")
    if int(localization_requirements.get("target_region_count", -1)) != 4:
        raise ValueError(
            f"V{profile.version} localization target_region_count must equal four"
        )
    held_out = _held_out_localization(
        module,
        category_embeddings,
        categories,
        map_loader=selected_map_loader,
        oracle_loader=selected_oracle_loader,
        supervision=supervision,
        top_k=100,
        minimum_precision=float(localization_requirements["minimum_precision_at_k"]),
        maximum_mirror_centroid_error_m=float(
            localization_requirements["maximum_mirror_centroid_error_m"]
        ),
    )
    access_audit = access_recorder.report()

    qa_authorized = bool(training["calibration_passed"] and held_out["passed"])
    report: dict[str, Any] = {
        "schema_version": 1,
        "config_sha256": _sha256_json(
            {key: value for key, value in config.items() if not str(key).startswith("_")}
        ),
        "calibration_split_sha256": _sha256_json(list(supervision.calibration_scene_ids)),
        "held_out_split_sha256": _sha256_json(list(supervision.held_out_scene_ids)),
        "category_vocabulary_sha256": _sha256_json(list(categories)),
        "category_embedding_sha256": tensor_state_sha256(
            {"category_embedding": category_embeddings}
        ),
        "calibration_summary_sha256": tensor_state_sha256(
            {
                "mean_layernorm_dense": summaries.mean_layernorm_dense,
                "mean_aligned_tail": summaries.mean_aligned_tail,
                "category_indices": summaries.category_indices,
                "voxel_counts": summaries.voxel_counts,
            }
        ),
        "initial_state_sha256": initial_hash,
        "final_state_sha256": str(training["final_state_sha256"]),
        "calibration_scene_count": len(supervision.calibration_scene_ids),
        "held_out_scene_count": len(supervision.held_out_scene_ids),
        "category_count": len(categories),
        "region_count": summaries.region_count,
        "skipped_underfilled_region_count": skipped_underfilled_regions,
        "summarized_region_voxel_count": int(summaries.voxel_counts.sum()),
        "selective_token_row_count": int(unique_rows),
        "loaded_parameter_count": 1,
        "cpu_only": True,
        "local_files_only": True,
        "raw_map_write_count": 0,
        "raw_maps_preserved": True,
        "question_dependent_selection": False,
        "training": {
            "learning_rate": warmup.learning_rate,
            "weight_decay": warmup.weight_decay,
            "delta_mse_regularization_weight": warmup.delta_rms_regularization_weight,
            "maximum_optimizer_steps": warmup.max_optimizer_steps,
            **training,
        },
        "held_out_localization": held_out,
        "qa_update_authorized": qa_authorized,
        "bridge_written": False,
        "bridge_sha256": None,
    }
    if profile.include_access_audit:
        report["scene_access_audit"] = access_audit

    # Eliminate every training-only semantic handle before exposing the audit.
    del categories, category_embeddings, summaries, loaded_embeddings, load_audit
    if qa_authorized and bridge_output is not None:
        report["bridge_sha256"] = _save_tensor_only_bridge(Path(bridge_output), module)
        report["bridge_written"] = True
    _validate_numeric_hash_report(report)
    if report_output is not None:
        _atomic_json(Path(report_output), report)
    return report


def require_dense_alignment_calibration_authorized(audit: Mapping[str, Any]) -> None:
    """Fail closed before paired QA if either calibration gate did not pass."""

    _validate_numeric_hash_report(audit)
    if audit.get("qa_update_authorized") is not True:
        raise RuntimeError("Dense-alignment calibration did not authorize paired QA updates")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/gemma4_color_mirror_dense_alignment_v25.yaml",
    )
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--bridge-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_config(args.config)
    profile = dense_alignment_calibration_profile(config)
    module = construct_dense_alignment(config, semantic_dim=3072)
    if module is None:
        raise ValueError(
            f"V{profile.version} config did not construct dense alignment"
        )
    namespace = str(config["training"]["output_namespace"])
    bridge_output = args.bridge_output or (
        artifact_root(config, "checkpoints") / namespace / "calibration_bridge.safetensors"
    )
    report_output = args.report_output or (
        reports_root(config)
        / "metrics"
        / f"v{profile.version}_dense_alignment_calibration.json"
    )
    audit = run_dense_alignment_calibration_warmup(
        config,
        module,
        model_snapshot=args.model_snapshot,
        bridge_output=bridge_output,
        report_output=report_output,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["qa_update_authorized"] else 2


__all__ = [
    "DenseAlignmentCalibrationProfile",
    "DenseAlignmentRegionSummaries",
    "dense_alignment_calibration_profile",
    "require_dense_alignment_calibration_authorized",
    "run_dense_alignment_calibration_warmup",
    "summarize_dense_alignment_regions",
    "train_dense_alignment_from_summaries",
]


if __name__ == "__main__":
    raise SystemExit(main())
