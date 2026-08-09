"""Evaluation-only zero-shot semantic localization for fused 3D maps.

The fused map is an opaque runtime artifact whose semantic layout is exactly
``[middle_768, late_768, CLIP-aligned_512]``.  Only the final 512 dimensions are
compared with the matching local CLIP text encoder.  Oracle labels and bounding
boxes are opened solely by this evaluation command and are never imported by
the chat runtime.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from semantic_3d_chat.config import PROJECT_ROOT, load_config, project_path
from semantic_3d_chat.device import select_device
from semantic_3d_chat.mapping.depth_projection import project_depth_to_world
from semantic_3d_chat.mapping.fusion import load_feature_field, sample_spatial_field
from semantic_3d_chat.mapping.voxel_map import SparseVoxelMap, voxel_coordinates
from semantic_3d_chat.rendering_io import iter_frames
from semantic_3d_chat.vision.encoder import DenseCLIPEncoder
from semantic_3d_chat.vision.model_registry import get_model_spec

LOGGER = logging.getLogger(__name__)

MIDDLE_DIM = 768
LATE_DIM = 768
ALIGNED_DIM = 512
ALIGNED_START = MIDDLE_DIM + LATE_DIM
TOTAL_SEMANTIC_DIM = ALIGNED_START + ALIGNED_DIM
SCENE_ID_PATTERN = re.compile(r"^scene_[0-9]{6}$")


class TextQueryEncoder(Protocol):
    def encode_text_queries(
        self, queries: Sequence[str], *, normalize: bool = True
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class SemanticQuery:
    query_id: str
    category: str
    text: str
    encoder_prompt: str


@dataclass(frozen=True)
class OracleTarget:
    instance_id: str
    category: str
    minimum_xyz_m: tuple[float, float, float]
    maximum_xyz_m: tuple[float, float, float]


@dataclass(frozen=True)
class SemanticFeatureSlice:
    """One named, dimension-checked stream inside a fused semantic feature."""

    total_dim: int
    start: int
    dimension: int
    name: str

    def __post_init__(self) -> None:
        if self.total_dim < 1:
            raise ValueError("Semantic feature total_dim must be positive")
        if self.start < 0 or self.dimension < 1:
            raise ValueError("Semantic feature slice start/dimension is invalid")
        if self.end > self.total_dim:
            raise ValueError("Semantic feature slice extends beyond total_dim")
        if not self.name.strip():
            raise ValueError("Semantic feature slice requires a non-empty name")

    @property
    def end(self) -> int:
        return self.start + self.dimension


CLIP_ALIGNED_SLICE = SemanticFeatureSlice(
    total_dim=TOTAL_SEMANTIC_DIM,
    start=ALIGNED_START,
    dimension=ALIGNED_DIM,
    name="CLIP-aligned voxel slice",
)


def _validate_scene_id(scene_id: str) -> str:
    if not SCENE_ID_PATTERN.fullmatch(scene_id):
        raise ValueError(f"Expected opaque scene ID like scene_000001, got {scene_id!r}")
    return scene_id


def _reject_oracle_runtime_input(path: str | Path, purpose: str) -> Path:
    resolved = Path(path).resolve()
    if "oracle" in {part.lower() for part in resolved.parts}:
        raise ValueError(f"{purpose} must be an opaque runtime artifact, not {resolved}")
    return resolved


def normalize_embedding_matrix(
    embeddings: np.ndarray | torch.Tensor,
    *,
    expected_dim: int,
    label: str,
) -> np.ndarray:
    """Validate and row-normalize an arbitrary-dimensional embedding matrix."""

    values = (
        embeddings.detach().float().cpu().numpy()
        if isinstance(embeddings, torch.Tensor)
        else np.asarray(embeddings, dtype=np.float32)
    )
    if expected_dim < 1:
        raise ValueError("expected_dim must be positive")
    if values.ndim != 2 or values.shape[1] != expected_dim:
        raise ValueError(f"Expected {label} [N, {expected_dim}], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contain NaN or infinite values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError(f"{label} contain a zero-norm embedding")
    return values / norms


def extract_feature_slice(
    semantic_features: np.ndarray,
    feature_slice: SemanticFeatureSlice,
) -> np.ndarray:
    """Extract and normalize one configured stream from a fused voxel map."""

    semantic = np.asarray(semantic_features)
    if semantic.ndim != 2 or semantic.shape[1] != feature_slice.total_dim:
        raise ValueError(
            f"Expected fused semantic_features [N, {feature_slice.total_dim}] for "
            f"{feature_slice.name}; got {semantic.shape}"
        )
    selected = semantic[:, feature_slice.start : feature_slice.end]
    return normalize_embedding_matrix(
        selected,
        expected_dim=feature_slice.dimension,
        label=feature_slice.name,
    )


def extract_aligned_features(semantic_features: np.ndarray) -> np.ndarray:
    """Validate the fused layout and return normalized CLIP-aligned voxels."""

    semantic = np.asarray(semantic_features)
    if semantic.ndim != 2 or semantic.shape[1] != TOTAL_SEMANTIC_DIM:
        raise ValueError(
            "Expected fused semantic_features [N, 2048] laid out as "
            "[middle768, late768, aligned512]; "
            f"got {semantic.shape}"
        )
    return extract_feature_slice(semantic, CLIP_ALIGNED_SLICE)


def normalize_text_embeddings(text_embeddings: np.ndarray | torch.Tensor) -> np.ndarray:
    return normalize_embedding_matrix(
        text_embeddings,
        expected_dim=ALIGNED_DIM,
        label="CLIP text embeddings",
    )


def oracle_targets(oracle: dict[str, Any]) -> list[OracleTarget]:
    """Validate evaluation-only oracle boxes without retaining Blender names."""

    instances = oracle.get("instances")
    if not isinstance(instances, list) or not instances:
        raise ValueError("Oracle must contain a non-empty instances list")
    targets: list[OracleTarget] = []
    for instance in instances:
        if not bool(instance.get("visible_from_center_scan", True)):
            continue
        category = instance.get("category")
        instance_id = instance.get("instance_id")
        bbox = instance.get("bbox", {})
        minimum = np.asarray(bbox.get("min_xyz_m"), dtype=np.float64)
        maximum = np.asarray(bbox.get("max_xyz_m"), dtype=np.float64)
        if not isinstance(category, str) or not category.strip():
            raise ValueError("Oracle instance has an invalid category")
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError("Oracle instance has an invalid opaque ID")
        if minimum.shape != (3,) or maximum.shape != (3,):
            raise ValueError(f"Oracle bbox for {instance_id} must contain two XYZ triples")
        if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
            raise ValueError(f"Oracle bbox for {instance_id} is not finite")
        if np.any(maximum < minimum):
            raise ValueError(f"Oracle bbox for {instance_id} has inverted limits")
        targets.append(
            OracleTarget(
                instance_id=instance_id,
                category=category.strip(),
                minimum_xyz_m=tuple(float(value) for value in minimum),
                maximum_xyz_m=tuple(float(value) for value in maximum),
            )
        )
    if not targets:
        raise ValueError("Oracle has no instances marked visible from the scan")
    return targets


def queries_from_targets(
    targets: Sequence[OracleTarget], prompt_template: str = "a photo of a {}"
) -> list[SemanticQuery]:
    """Create one deterministic CLIP query per distinct oracle category."""

    if prompt_template.count("{}") != 1:
        raise ValueError("prompt_template must contain exactly one '{}' placeholder")
    categories = sorted({target.category for target in targets})
    return [
        SemanticQuery(
            query_id=f"query_{index:03d}",
            category=category,
            text=category,
            encoder_prompt=prompt_template.format(category),
        )
        for index, category in enumerate(categories)
    ]


def bounding_box_mask(
    centers_world: np.ndarray,
    targets: Sequence[OracleTarget],
    *,
    padding_m: float = 0.0,
) -> np.ndarray:
    centers = np.asarray(centers_world, dtype=np.float32)
    if centers.ndim != 2 or centers.shape[1] != 3 or not np.isfinite(centers).all():
        raise ValueError("centers_world must be finite with shape [N, 3]")
    if not np.isfinite(padding_m) or padding_m < 0:
        raise ValueError("padding_m must be finite and non-negative")
    mask = np.zeros(centers.shape[0], dtype=bool)
    for target in targets:
        minimum = np.asarray(target.minimum_xyz_m, dtype=np.float32) - padding_m
        maximum = np.asarray(target.maximum_xyz_m, dtype=np.float32) + padding_m
        mask |= np.all((centers >= minimum) & (centers <= maximum), axis=1)
    return mask


def _top_mean(values: np.ndarray, count: int) -> float:
    if values.size == 0:
        return float("nan")
    selected_count = min(max(int(count), 1), values.size)
    selected = np.argpartition(values, -selected_count)[-selected_count:]
    return float(values[selected].mean())


def _random_hit_probability(total: int, targets: int, draws: int) -> float:
    """Probability that uniform sampling without replacement hits a target."""

    if total < 1 or targets <= 0 or draws <= 0:
        return 0.0
    draws = min(draws, total)
    targets = min(targets, total)
    if draws > total - targets:
        return 1.0
    indices = np.arange(draws, dtype=np.float64)
    log_miss = np.log((total - targets - indices) / (total - indices)).sum()
    return float(-np.expm1(log_miss))


def score_semantic_queries(
    centers_world: np.ndarray,
    semantic_features: np.ndarray,
    queries: Sequence[SemanticQuery],
    text_embeddings: np.ndarray | torch.Tensor,
    targets: Sequence[OracleTarget],
    *,
    top_k: int = 100,
    bbox_padding_m: float = 0.04,
    feature_slice: SemanticFeatureSlice | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Score zero-shot voxel localization against evaluation-only boxes.

    The default remains the historical 512D CLIP stream. Evaluation-only
    variants can supply another checked slice while sharing identical scoring.
    """

    centers = np.asarray(centers_world, dtype=np.float32)
    if centers.ndim != 2 or centers.shape[1] != 3 or not np.isfinite(centers).all():
        raise ValueError("centers_world must be finite with shape [N, 3]")
    if centers.shape[0] == 0:
        raise ValueError("Cannot evaluate an empty semantic map")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not queries:
        raise ValueError("At least one semantic query is required")
    selected_slice = feature_slice or CLIP_ALIGNED_SLICE
    aligned = (
        extract_aligned_features(semantic_features)
        if feature_slice is None
        else extract_feature_slice(semantic_features, selected_slice)
    )
    if aligned.shape[0] != centers.shape[0]:
        raise ValueError("centers_world and semantic_features must have the same voxel count")
    text = (
        normalize_text_embeddings(text_embeddings)
        if feature_slice is None
        else normalize_embedding_matrix(
            text_embeddings,
            expected_dim=selected_slice.dimension,
            label=f"{selected_slice.name} query embeddings",
        )
    )
    if text.shape[0] != len(queries):
        raise ValueError("Text embedding count does not match semantic query count")
    similarities = aligned @ text.T
    if not np.isfinite(similarities).all():
        raise ValueError("Cosine similarity produced NaN or infinite values")

    category_targets = {
        query.category: [target for target in targets if target.category == query.category]
        for query in queries
    }
    all_instance_mask = bounding_box_mask(centers, targets, padding_m=bbox_padding_m)
    ranking_k = min(top_k, centers.shape[0])
    query_results: list[dict[str, Any]] = []
    for query_index, query in enumerate(queries):
        target_mask = bounding_box_mask(
            centers, category_targets[query.category], padding_m=bbox_padding_m
        )
        distractor_region_mask = all_instance_mask & ~target_mask
        scores = similarities[:, query_index]
        ranked = np.argsort(-scores, kind="stable")
        top_indices = ranked[:ranking_k]
        target_count = int(target_mask.sum())
        top_target_count = int(target_mask[top_indices].sum())
        random_top1 = target_count / centers.shape[0]
        random_hit_at_k = _random_hit_probability(centers.shape[0], target_count, ranking_k)
        target_scores = scores[target_mask]
        distractor_region_scores = scores[distractor_region_mask]
        target_top_mean = _top_mean(target_scores, top_k)
        distractor_region_top_mean = _top_mean(distractor_region_scores, top_k)
        region_margin = (
            target_top_mean - distractor_region_top_mean
            if np.isfinite(target_top_mean) and np.isfinite(distractor_region_top_mean)
            else float("nan")
        )

        competing_query_scores: list[tuple[float, str]] = []
        if target_count:
            for other_index, other_query in enumerate(queries):
                if other_index == query_index:
                    continue
                competing_query_scores.append(
                    (
                        _top_mean(similarities[target_mask, other_index], top_k),
                        other_query.category,
                    )
                )
        if competing_query_scores:
            best_other_score, best_other_category = max(competing_query_scores)
            query_margin = target_top_mean - best_other_score
        else:
            best_other_score, best_other_category, query_margin = (
                float("nan"),
                None,
                float("nan"),
            )

        top_voxel_index = int(ranked[0])
        query_results.append(
            {
                **asdict(query),
                "target_instance_count": len(category_targets[query.category]),
                "target_voxel_count": target_count,
                "scorable": bool(target_count),
                "ranking_k": ranking_k,
                "top1_hit": bool(target_mask[top_voxel_index]) if target_count else False,
                "hit_at_k": bool(top_target_count) if target_count else False,
                "random_top1_probability": random_top1 if target_count else None,
                "random_hit_at_k_probability": random_hit_at_k if target_count else None,
                "precision_at_k": top_target_count / ranking_k if target_count else None,
                "random_precision_at_k": random_top1 if target_count else None,
                "target_recall_at_k": (top_target_count / target_count if target_count else None),
                "target_top_similarity": (
                    target_top_mean if np.isfinite(target_top_mean) else None
                ),
                "distractor_region_top_similarity": (
                    distractor_region_top_mean if np.isfinite(distractor_region_top_mean) else None
                ),
                "region_margin": region_margin if np.isfinite(region_margin) else None,
                "best_other_query_similarity_on_target": (
                    best_other_score if np.isfinite(best_other_score) else None
                ),
                "best_distractor_category": best_other_category,
                "correct_vs_distractor_margin": (
                    query_margin if np.isfinite(query_margin) else None
                ),
                "top_voxel_xyz_m": centers[top_voxel_index].astype(float).tolist(),
                "top_voxel_similarity": float(scores[top_voxel_index]),
            }
        )

    scorable = [result for result in query_results if result["scorable"]]

    def mean_available(key: str) -> float | None:
        values = [float(result[key]) for result in scorable if result[key] is not None]
        return float(np.mean(values)) if values else None

    aggregate = {
        "scorable_queries": len(scorable),
        "unscorable_queries": len(query_results) - len(scorable),
        "top1_localization_accuracy": mean_available("top1_hit"),
        "mean_random_top1_probability": mean_available("random_top1_probability"),
        "top_k_localization_accuracy": mean_available("hit_at_k"),
        "mean_random_hit_at_k_probability": mean_available("random_hit_at_k_probability"),
        "mean_precision_at_k": mean_available("precision_at_k"),
        "mean_random_precision_at_k": mean_available("random_precision_at_k"),
        "mean_target_recall_at_k": mean_available("target_recall_at_k"),
        "mean_region_margin": mean_available("region_margin"),
        "mean_correct_vs_distractor_margin": mean_available("correct_vs_distractor_margin"),
        "positive_region_margin_rate": (
            float(
                np.mean(
                    [
                        result["region_margin"] > 0
                        for result in scorable
                        if result["region_margin"] is not None
                    ]
                )
            )
            if any(result["region_margin"] is not None for result in scorable)
            else None
        ),
        "positive_correct_vs_distractor_margin_rate": (
            float(
                np.mean(
                    [
                        result["correct_vs_distractor_margin"] > 0
                        for result in scorable
                        if result["correct_vs_distractor_margin"] is not None
                    ]
                )
            )
            if any(result["correct_vs_distractor_margin"] is not None for result in scorable)
            else None
        ),
    }

    def difference(first: float | None, second: float | None) -> float | None:
        return first - second if first is not None and second is not None else None

    aggregate["top1_accuracy_minus_random"] = difference(
        aggregate["top1_localization_accuracy"],
        aggregate["mean_random_top1_probability"],
    )
    aggregate["top_k_accuracy_minus_random"] = difference(
        aggregate["top_k_localization_accuracy"],
        aggregate["mean_random_hit_at_k_probability"],
    )
    aggregate["precision_at_k_minus_random"] = difference(
        aggregate["mean_precision_at_k"], aggregate["mean_random_precision_at_k"]
    )
    return {"aggregate": aggregate, "queries": query_results}, similarities


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "p05": None,
            "p95": None,
        }
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "median": float(np.median(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p95": float(np.percentile(finite, 95)),
    }


def _feature_index(feature_location: Path) -> tuple[Path, dict[str, Path]]:
    if feature_location.is_file():
        manifest_path = feature_location
    else:
        direct = feature_location / "manifest.json"
        if direct.is_file():
            manifest_path = direct
        else:
            matches = sorted(feature_location.glob("*/manifest.json"))
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected one feature manifest below {feature_location}, found {matches}"
                )
            manifest_path = matches[0]
    manifest_path = _reject_oracle_runtime_input(manifest_path, "Feature manifest")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("frames")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Feature manifest contains no frames: {manifest_path}")
    paths: dict[str, Path] = {}
    for entry in entries:
        frame_id = entry.get("frame_id")
        relative_path = entry.get("feature_path")
        if not isinstance(frame_id, str) or not isinstance(relative_path, str):
            raise TypeError("Feature manifest frame entries require frame_id and feature_path")
        resolved = _reject_oracle_runtime_input(
            manifest_path.parent / relative_path, "Frame feature cache"
        )
        if frame_id in paths:
            raise ValueError(f"Duplicate feature entry for frame {frame_id}")
        paths[frame_id] = resolved
    return manifest_path, paths


def compute_multiview_consistency(
    render_manifest_path: str | Path,
    feature_location: str | Path,
    *,
    voxel_size_m: float,
    depth_min_m: float = 0.1,
    depth_max_m: float = 10.0,
    pixel_stride: int = 2,
    feature_slice: SemanticFeatureSlice | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Compare same-voxel cross-view features with different-voxel controls."""

    render_manifest = _reject_oracle_runtime_input(render_manifest_path, "Render manifest")
    _, feature_paths = _feature_index(Path(feature_location))
    selected_slice = feature_slice or CLIP_ALIGNED_SLICE
    # One unit vector per (voxel, frame), accumulated across frames without
    # retaining per-pixel tensors.  Pairwise means can be recovered from ||sum||.
    accumulator: dict[tuple[int, int, int], tuple[int, np.ndarray]] = {}
    frame_count = 0
    frame_voxel_observations = 0
    for frame in iter_frames(render_manifest):
        feature_path = feature_paths.get(frame.frame_id)
        if feature_path is None:
            raise FileNotFoundError(f"Missing feature cache for frame {frame.frame_id}")
        depth_path = _reject_oracle_runtime_input(frame.depth_path, "Depth frame")
        depth = np.load(depth_path, allow_pickle=False).astype(np.float32, copy=False)
        features = load_feature_field(feature_path)
        projection = project_depth_to_world(
            depth,
            frame.intrinsics,
            frame.camera_to_world,
            min_depth_m=depth_min_m,
            max_depth_m=depth_max_m,
            pixel_stride=pixel_stride,
        )
        sampled = sample_spatial_field(features, projection.pixels_uv, image_shape=depth.shape)
        aligned = (
            extract_aligned_features(sampled)
            if feature_slice is None
            else extract_feature_slice(sampled, selected_slice)
        )
        coordinates = voxel_coordinates(projection.points_world, voxel_size_m)
        order = np.lexsort((coordinates[:, 2], coordinates[:, 1], coordinates[:, 0]))
        ordered = coordinates[order]
        boundaries = np.flatnonzero(np.any(np.diff(ordered, axis=0), axis=1)) + 1
        starts = np.concatenate(([0], boundaries))
        stops = np.concatenate((boundaries, [len(order)]))
        for start, stop in zip(starts, stops, strict=True):
            indices = order[start:stop]
            view_feature = aligned[indices].mean(axis=0)
            norm = np.linalg.norm(view_feature)
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError(f"Invalid frame-level voxel feature in {frame.frame_id}")
            view_feature = (view_feature / norm).astype(np.float32)
            key = tuple(int(value) for value in ordered[start])
            previous_count, previous_sum = accumulator.get(
                key, (0, np.zeros(selected_slice.dimension, dtype=np.float32))
            )
            accumulator[key] = (previous_count + 1, previous_sum + view_feature)
            frame_voxel_observations += 1
        frame_count += 1

    same_values: list[float] = []
    representative_keys: list[tuple[int, int, int]] = []
    representatives: list[np.ndarray] = []
    pair_count = 0
    for key in sorted(accumulator):
        view_count, feature_sum = accumulator[key]
        norm = np.linalg.norm(feature_sum)
        if norm > 0:
            representative_keys.append(key)
            representatives.append(feature_sum / norm)
        if view_count < 2:
            continue
        # Mean over unique unordered pairs for unit vectors.
        mean_pair_similarity = (float(np.dot(feature_sum, feature_sum)) - view_count) / (
            view_count * (view_count - 1)
        )
        same_values.append(float(np.clip(mean_pair_similarity, -1.0, 1.0)))
        pair_count += view_count * (view_count - 1) // 2

    different_values = np.empty((0,), dtype=np.float32)
    if len(representatives) >= 2:
        representative_array = np.stack(representatives).astype(np.float32)
        shift = max(1, len(representatives) // 2)
        different_values = np.einsum(
            "nd,nd->n", representative_array, np.roll(representative_array, shift, axis=0)
        )
    same_array = np.asarray(same_values, dtype=np.float32)
    same_distribution = _distribution(same_array)
    different_distribution = _distribution(different_values)
    mean_margin = (
        float(same_distribution["mean"] - different_distribution["mean"])
        if same_distribution["mean"] is not None and different_distribution["mean"] is not None
        else None
    )
    metrics = {
        "available": bool(same_array.size and different_values.size),
        "frames": frame_count,
        "unique_voxels": len(accumulator),
        "multiview_voxels": int(same_array.size),
        "frame_voxel_observations": frame_voxel_observations,
        "same_voxel_pair_count": pair_count,
        "same_voxel_similarity": same_distribution,
        "different_voxel_similarity": different_distribution,
        "same_minus_different_mean": mean_margin,
    }
    return metrics, same_array, different_values


def write_query_heatmaps(
    centers_world: np.ndarray,
    similarities: np.ndarray,
    queries: Sequence[SemanticQuery],
    targets: Sequence[OracleTarget],
    output_directory: str | Path,
    *,
    max_points: int = 100_000,
    similarity_label: str = "CLIP cosine similarity",
) -> list[dict[str, str]]:
    """Render evaluation-only semantic heatmaps with opaque filenames."""

    centers = np.asarray(centers_world, dtype=np.float32)
    scores = np.asarray(similarities, dtype=np.float32)
    if scores.shape != (centers.shape[0], len(queries)):
        raise ValueError(
            f"similarities must have shape {(centers.shape[0], len(queries))}, got {scores.shape}"
        )
    if max_points < 1:
        raise ValueError("max_points must be positive")
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    if len(centers) > max_points:
        base_indices = np.linspace(0, len(centers) - 1, max_points, dtype=np.int64)
    else:
        base_indices = np.arange(len(centers))

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    artifacts: list[dict[str, str]] = []
    for query_index, query in enumerate(queries):
        query_scores = scores[:, query_index]
        # Retain the strongest locations even when the cloud is downsampled.
        strongest = np.argsort(-query_scores, kind="stable")[: min(512, len(centers))]
        indices = np.unique(np.concatenate((base_indices, strongest)))
        points = centers[indices]
        display_scores = query_scores[indices]
        low, high = np.percentile(query_scores, [2, 98])
        if high <= low:
            low, high = float(query_scores.min()) - 1e-6, float(query_scores.max()) + 1e-6

        figure = Figure(figsize=(10.5, 4.8), dpi=140, constrained_layout=True)
        FigureCanvasAgg(figure)
        top = figure.add_subplot(1, 2, 1)
        front = figure.add_subplot(1, 2, 2)
        marker_size = max(0.2, min(5.0, 10_000.0 / max(len(indices), 1)))
        top_plot = top.scatter(
            points[:, 0],
            points[:, 1],
            c=display_scores,
            cmap="coolwarm",
            vmin=low,
            vmax=high,
            s=marker_size,
            linewidths=0,
        )
        front.scatter(
            points[:, 0],
            points[:, 2],
            c=display_scores,
            cmap="coolwarm",
            vmin=low,
            vmax=high,
            s=marker_size,
            linewidths=0,
        )
        for target in targets:
            if target.category != query.category:
                continue
            minimum = np.asarray(target.minimum_xyz_m)
            maximum = np.asarray(target.maximum_xyz_m)
            top.add_patch(
                Rectangle(
                    (minimum[0], minimum[1]),
                    maximum[0] - minimum[0],
                    maximum[1] - minimum[1],
                    fill=False,
                    edgecolor="lime",
                    linewidth=1.25,
                )
            )
            front.add_patch(
                Rectangle(
                    (minimum[0], minimum[2]),
                    maximum[0] - minimum[0],
                    maximum[2] - minimum[2],
                    fill=False,
                    edgecolor="lime",
                    linewidth=1.25,
                )
            )
        top.set(xlabel="X — right (m)", ylabel="Y — forward (m)", title="Top view")
        front.set(xlabel="X — right (m)", ylabel="Z — up (m)", title="Front view")
        top.set_aspect("equal", adjustable="box")
        front.set_aspect("equal", adjustable="box")
        figure.colorbar(top_plot, ax=[top, front], label=similarity_label, shrink=0.85)
        figure.suptitle(f"Zero-shot 3D localization: {query.text}")
        path = destination / f"{query.query_id}.png"
        figure.savefig(path, format="png")
        figure.clear()
        artifacts.append(
            {
                "query_id": query.query_id,
                "category": query.category,
                "path": str(path),
            }
        )
    return artifacts


def write_consistency_histogram(
    same_voxel_similarities: np.ndarray,
    different_voxel_similarities: np.ndarray,
    path: str | Path,
    *,
    feature_label: str = "CLIP-aligned cosine similarity",
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    same = np.asarray(same_voxel_similarities, dtype=np.float32)
    different = np.asarray(different_voxel_similarities, dtype=np.float32)
    if not same.size or not different.size:
        raise ValueError("Both consistency distributions must be non-empty")

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(7.5, 4.8), dpi=140, constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(1, 1, 1)
    bins = np.linspace(-1.0, 1.0, 61)
    axes.hist(same, bins=bins, density=True, alpha=0.65, label="same voxel / different view")
    axes.hist(different, bins=bins, density=True, alpha=0.55, label="different voxel")
    axes.set(
        xlabel=feature_label,
        ylabel="Density",
        title="Cross-view feature consistency control",
        xlim=(-1.0, 1.0),
    )
    axes.legend()
    figure.savefig(destination, format="png")
    figure.clear()
    return destination


def _load_oracle_for_evaluation(path: str | Path, scene_id: str) -> dict[str, Any]:
    """The sole semantic-metadata file read in this evaluation module."""

    source = Path(path).resolve()
    if "oracle" not in {part.lower() for part in source.parts}:
        raise ValueError(f"Expected isolated oracle path for evaluation, got {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("scene_id") != scene_id:
        raise ValueError(
            f"Oracle scene mismatch: requested {scene_id}, found {payload.get('scene_id')}"
        )
    return payload


def _default_map_path(config: dict[str, Any], scene_id: str) -> Path:
    scene_root = project_path(config, "maps", scene_id)
    preferred = scene_root / "map.npz"
    if preferred.is_file():
        return preferred
    matches = sorted(scene_root.glob("*.npz")) if scene_root.is_dir() else []
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected {preferred} or exactly one NPZ map below {scene_root}; found {matches}"
        )
    return matches[0]


def _default_feature_location(config: dict[str, Any], scene_id: str) -> Path:
    return project_path(config, "features", scene_id)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_semantic_sanity(
    config: dict[str, Any],
    scene_id: str,
    *,
    map_path: str | Path | None = None,
    oracle_path: str | Path | None = None,
    render_manifest_path: str | Path | None = None,
    feature_location: str | Path | None = None,
    output_path: str | Path | None = None,
    figures_directory: str | Path | None = None,
    top_k: int | None = None,
    prompt_template: str | None = None,
    local_files_only: bool = False,
    device: torch.device | None = None,
    skip_consistency: bool = False,
    write_figures: bool = True,
    encoder: TextQueryEncoder | None = None,
) -> dict[str, Any]:
    """Run the complete evaluation command and persist machine-readable metrics."""

    scene_id = _validate_scene_id(scene_id)
    selected_map_path = _reject_oracle_runtime_input(
        map_path or _default_map_path(config, scene_id), "Fused map"
    )
    voxel_map = SparseVoxelMap.load(selected_map_path)
    arrays = voxel_map.to_arrays(encode_semantics=False)
    if arrays["semantic_features"].shape[1] != TOTAL_SEMANTIC_DIM:
        raise ValueError(
            f"Map feature dimension is {arrays['semantic_features'].shape[1]}, expected "
            f"{TOTAL_SEMANTIC_DIM} = middle768 + late768 + aligned512"
        )

    sanity_config = config.get("evaluation", {}).get("semantic_sanity", {})
    selected_top_k = int(top_k if top_k is not None else sanity_config.get("top_k", 100))
    selected_prompt_template = str(
        prompt_template
        if prompt_template is not None
        else sanity_config.get("prompt_template", "a photo of a {}")
    )
    bbox_padding_voxels = float(sanity_config.get("bbox_padding_voxels", 0.75))
    if selected_top_k < 1:
        raise ValueError("semantic_sanity.top_k must be positive")
    if not np.isfinite(bbox_padding_voxels) or bbox_padding_voxels < 0:
        raise ValueError("semantic_sanity.bbox_padding_voxels must be non-negative")

    # Semantic names enter only below this line, inside this evaluation command.
    selected_oracle_path = (
        Path(oracle_path)
        if oracle_path is not None
        else project_path(config, "oracle", scene_id, "oracle.json")
    )
    oracle = _load_oracle_for_evaluation(selected_oracle_path, scene_id)
    targets = oracle_targets(oracle)
    queries = queries_from_targets(targets, selected_prompt_template)

    vision_config = config["vision"]
    model_id = str(vision_config["model_id"])
    spec = get_model_spec(model_id)
    if spec.aligned_dim != ALIGNED_DIM or spec.native_dim != MIDDLE_DIM:
        raise ValueError(f"Unsupported semantic-sanity model layout: {spec}")
    active_encoder = encoder or DenseCLIPEncoder.from_pretrained(
        model_id,
        revision=str(vision_config.get("revision", "main")),
        device=device or select_device(),
        requested_dtype=str(vision_config.get("dtype", "float16")),
        middle_layer=int(vision_config.get("middle_layer", spec.default_middle_layer)),
        late_layer=int(vision_config.get("late_layer", spec.default_late_layer)),
        local_files_only=local_files_only,
    )
    text_embeddings = active_encoder.encode_text_queries(
        [query.encoder_prompt for query in queries], normalize=True
    )
    padding_m = max(voxel_map.voxel_size_m * bbox_padding_voxels, 1e-4)
    localization, similarities = score_semantic_queries(
        arrays["centers_world"],
        arrays["semantic_features"],
        queries,
        text_embeddings,
        targets,
        top_k=selected_top_k,
        bbox_padding_m=padding_m,
    )

    reports_root = PROJECT_ROOT / str(config["paths"].get("reports_root", "reports"))
    selected_output_path = (
        Path(output_path)
        if output_path
        else (reports_root / "metrics" / f"semantic_sanity_{scene_id}.json")
    )
    selected_figures_directory = (
        Path(figures_directory)
        if figures_directory
        else (reports_root / "figures" / "semantic_sanity" / scene_id)
    )
    heatmaps: list[dict[str, str]] = []
    if write_figures:
        heatmaps = write_query_heatmaps(
            arrays["centers_world"],
            similarities,
            queries,
            targets,
            selected_figures_directory,
            max_points=int(sanity_config.get("heatmap_max_points", 100_000)),
        )

    consistency_metrics: dict[str, Any]
    consistency_figure: str | None = None
    if skip_consistency:
        consistency_metrics = {"available": False, "reason": "disabled_by_cli"}
    else:
        selected_render_manifest = (
            Path(render_manifest_path)
            if render_manifest_path
            else (project_path(config, "rendered", scene_id, "manifest.json"))
        )
        selected_feature_location = (
            Path(feature_location)
            if feature_location
            else (_default_feature_location(config, scene_id))
        )
        try:
            consistency_metrics, same_values, different_values = compute_multiview_consistency(
                selected_render_manifest,
                selected_feature_location,
                voxel_size_m=voxel_map.voxel_size_m,
                depth_min_m=float(config["mapping"].get("depth_min_m", 0.1)),
                depth_max_m=float(config["mapping"].get("depth_max_m", 10.0)),
                pixel_stride=int(config["mapping"].get("pixel_stride", 1)),
            )
            if write_figures and consistency_metrics["available"]:
                consistency_path = write_consistency_histogram(
                    same_values,
                    different_values,
                    selected_figures_directory / "view_consistency.png",
                )
                consistency_figure = str(consistency_path)
        except FileNotFoundError as error:
            consistency_metrics = {"available": False, "reason": str(error)}

    metrics = {
        "schema_version": 1,
        "phase": "semantic_sanity",
        "scene_id": scene_id,
        "map_path": str(selected_map_path),
        "map_content_hash": voxel_map.content_hash(),
        "voxel_count": len(voxel_map),
        "voxel_size_m": voxel_map.voxel_size_m,
        "feature_layout": {
            "total_dim": TOTAL_SEMANTIC_DIM,
            "middle": [0, MIDDLE_DIM],
            "late": [MIDDLE_DIM, ALIGNED_START],
            "clip_aligned": [ALIGNED_START, TOTAL_SEMANTIC_DIM],
            "scored_slice": "clip_aligned",
            "aligned_method": str(vision_config.get("aligned_method", "tokenwise_projection")),
        },
        "vision_model": model_id,
        "vision_revision": str(vision_config.get("revision", "main")),
        "query_count": len(queries),
        "top_k": selected_top_k,
        "prompt_template": selected_prompt_template,
        "bbox_padding_voxels": bbox_padding_voxels,
        "bbox_padding_m": padding_m,
        **localization,
        "same_voxel_consistency": consistency_metrics,
        "artifacts": {
            "heatmaps": heatmaps,
            "view_consistency_histogram": consistency_figure,
        },
    }
    _atomic_json(selected_output_path, metrics)
    LOGGER.info(
        "phase=semantic_sanity scene=%s voxels=%d queries=%d top_k_accuracy=%s",
        scene_id,
        len(voxel_map),
        len(queries),
        metrics["aggregate"]["top_k_localization_accuracy"],
    )
    return {**metrics, "metrics_path": str(selected_output_path)}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--map", type=Path)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--render-manifest", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--figures", type=Path)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--prompt-template")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--skip-consistency", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.device == "auto":
        device = select_device()
    else:
        device = torch.device(args.device)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
    metrics = run_semantic_sanity(
        load_config(args.config),
        args.scene,
        map_path=args.map,
        oracle_path=args.oracle,
        render_manifest_path=args.render_manifest,
        feature_location=args.features,
        output_path=args.output,
        figures_directory=args.figures,
        top_k=args.top_k,
        prompt_template=args.prompt_template,
        local_files_only=args.offline,
        device=device,
        skip_consistency=args.skip_consistency,
        write_figures=not args.no_figures,
    )
    print(
        json.dumps(
            {
                "scene_id": metrics["scene_id"],
                "metrics_path": metrics["metrics_path"],
                "voxel_count": metrics["voxel_count"],
                "aggregate": metrics["aggregate"],
                "same_voxel_consistency": metrics["same_voxel_consistency"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
