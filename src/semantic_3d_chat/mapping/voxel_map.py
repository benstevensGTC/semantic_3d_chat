"""Deterministic sparse voxel-hash fusion for continuous semantic features."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np

from semantic_3d_chat.mapping.semantic_codec import (
    IdentitySemanticCodec,
    SemanticCodec,
)

ColorMode = Literal["rgb", "feature_norm", "uncertainty", "observation_count", "confidence"]

PERSISTED_MAP_CONTENT_HASH_DOMAIN = "semantic_3d_chat.voxel_map.persisted_numeric_arrays.v1"
MATERIALIZED_MAP_CONTENT_HASH_DOMAIN = "semantic_3d_chat.voxel_map.materialized_numeric_arrays.v1"


def _numeric_arrays_content_hash(
    voxel_size_m: float,
    arrays: dict[str, np.ndarray],
) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("<d", voxel_size_m))
    for name, values in sorted(arrays.items()):
        digest.update(name.encode("utf-8"))
        contiguous = np.ascontiguousarray(values)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def persisted_voxel_map_content_hash(path: str | Path) -> str:
    """Hash the exact persisted numeric arrays, excluding JSON metadata.

    This domain is stable across loading and is therefore the only hash domain
    suitable for comparing two independently generated report artifacts.
    """

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        if not isinstance(metadata, dict):
            raise TypeError("Voxel-map metadata must be a JSON object")
        voxel_size_m = float(metadata["voxel_size_m"])
        digest = hashlib.sha256()
        digest.update(struct.pack("<d", voxel_size_m))
        for name in sorted(set(archive.files) - {"metadata_json"}):
            values = np.ascontiguousarray(archive[name])
            digest.update(name.encode("utf-8"))
            digest.update(str(values.dtype).encode("ascii"))
            digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
            digest.update(values.tobytes())
        return digest.hexdigest()


@dataclass
class _VoxelAccumulator:
    coordinate: tuple[int, int, int]
    observation_count: int
    weight_sum: float
    mean_rgb: np.ndarray
    mean_feature: np.ndarray
    feature_m2: float
    confidence_mean: float
    normal_sum: np.ndarray
    normal_weight_sum: float
    view_direction_sum: np.ndarray
    view_direction_weight_sum: float
    last_frame: str


def _as_rows(
    values: np.ndarray,
    *,
    name: str,
    count: int,
    width: int,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.shape != (count, width):
        raise ValueError(f"{name} must have shape {(count, width)}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _normalize_rows(values: np.ndarray, *, name: str) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError(f"{name} contains a zero-length or invalid vector")
    return values / norms


def voxel_coordinates(points_world: np.ndarray, voxel_size_m: float) -> np.ndarray:
    """Return signed integer voxel coordinates using floor assignment.

    Floor, rather than truncation, is essential for points on the negative side
    of a world axis: ``-0.01`` belongs to voxel ``-1`` at a 5 cm resolution.
    """

    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_world must have shape [N, 3], got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("points_world contains NaN or infinite values")
    if not np.isfinite(voxel_size_m) or voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be finite and positive")
    scaled = np.floor(points / voxel_size_m)
    limits = np.iinfo(np.int32)
    if scaled.size and (scaled.min() < limits.min or scaled.max() > limits.max):
        raise OverflowError("Voxel coordinate exceeds int32 range")
    return scaled.astype(np.int32)


class SparseVoxelMap:
    """Persistent CPU voxel map with weighted semantic and appearance fusion.

    Accumulation is float32/float64 for stability.  The configured
    :class:`SemanticCodec` is applied only when materializing or saving the map,
    so repeated updates do not repeatedly quantize high-dimensional features.
    Every input point is assigned to exactly one occupied voxel; exceeding
    ``max_voxels`` raises instead of silently dropping part of the room.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        voxel_size_m: float,
        *,
        feature_dim: int | None = None,
        max_voxels: int | None = None,
        codec: SemanticCodec | None = None,
    ) -> None:
        if not np.isfinite(voxel_size_m) or voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be finite and positive")
        if feature_dim is not None and feature_dim < 1:
            raise ValueError("feature_dim must be positive when supplied")
        if max_voxels is not None and max_voxels < 1:
            raise ValueError("max_voxels must be positive when supplied")
        self.voxel_size_m = float(voxel_size_m)
        self.feature_dim = int(feature_dim) if feature_dim is not None else None
        self.max_voxels = int(max_voxels) if max_voxels is not None else None
        self.codec = codec or IdentitySemanticCodec()
        self._voxels: dict[tuple[int, int, int], _VoxelAccumulator] = {}

    def __len__(self) -> int:
        return len(self._voxels)

    @property
    def occupied_voxel_count(self) -> int:
        return len(self)

    def add_observations(
        self,
        points_world: np.ndarray,
        semantic_features: np.ndarray,
        *,
        rgb: np.ndarray | None = None,
        normals_world: np.ndarray | None = None,
        view_directions_world: np.ndarray | None = None,
        confidence: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        frame_id: str | int = "",
    ) -> int:
        """Fuse a batch and return the number of newly occupied voxels."""

        points = np.asarray(points_world, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points_world must have shape [N, 3], got {points.shape}")
        if not np.isfinite(points).all():
            raise ValueError("points_world contains NaN or infinite values")
        count = points.shape[0]

        features = np.asarray(semantic_features, dtype=np.float32)
        if features.ndim != 2 or features.shape[0] != count or features.shape[1] < 1:
            raise ValueError(
                "semantic_features must have shape [N, D] with the same N as points_world"
            )
        if not np.isfinite(features).all():
            raise ValueError("semantic_features contains NaN or infinite values")
        if count and np.any(np.linalg.norm(features, axis=1) == 0):
            raise ValueError("semantic_features contains a zero-norm embedding")
        if self.feature_dim is None:
            self.feature_dim = int(features.shape[1])
        elif features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected semantic feature dimension {self.feature_dim}, got {features.shape[1]}"
            )
        if count == 0:
            return 0

        if rgb is None:
            rgb_values = np.zeros((count, 3), dtype=np.float32)
        else:
            rgb_values = _as_rows(rgb, name="rgb", count=count, width=3)
            if np.any(rgb_values < 0) or np.any(rgb_values > 255):
                raise ValueError("rgb values must be in [0, 255]")

        normal_values: np.ndarray | None = None
        if normals_world is not None:
            normal_values = _normalize_rows(
                _as_rows(normals_world, name="normals_world", count=count, width=3),
                name="normals_world",
            )

        view_values: np.ndarray | None = None
        if view_directions_world is not None:
            view_values = _normalize_rows(
                _as_rows(
                    view_directions_world,
                    name="view_directions_world",
                    count=count,
                    width=3,
                ),
                name="view_directions_world",
            )

        if confidence is None:
            confidence_values = np.ones(count, dtype=np.float32)
        else:
            confidence_values = np.asarray(confidence, dtype=np.float32)
            if confidence_values.shape != (count,) or not np.isfinite(confidence_values).all():
                raise ValueError(f"confidence must be finite with shape {(count,)}")
            if np.any(confidence_values < 0) or np.any(confidence_values > 1):
                raise ValueError("confidence values must be in [0, 1]")

        if weights is None:
            weight_values = confidence_values.copy()
        else:
            weight_values = np.asarray(weights, dtype=np.float32)
            if weight_values.shape != (count,) or not np.isfinite(weight_values).all():
                raise ValueError(f"weights must be finite with shape {(count,)}")
        if np.any(weight_values <= 0):
            raise ValueError("All fusion weights must be strictly positive")

        coordinates = voxel_coordinates(points, self.voxel_size_m)
        unique_coordinates = np.unique(coordinates, axis=0)
        new_keys = {
            tuple(int(component) for component in coordinate) for coordinate in unique_coordinates
        } - self._voxels.keys()
        if self.max_voxels is not None and len(self) + len(new_keys) > self.max_voxels:
            raise MemoryError(
                f"Fusion would create {len(self) + len(new_keys)} voxels, exceeding "
                f"max_voxels={self.max_voxels}; no observations were fused"
            )

        # Stable lexicographic ordering makes results reproducible across runs.
        order = np.lexsort((coordinates[:, 2], coordinates[:, 1], coordinates[:, 0]))
        ordered_coordinates = coordinates[order]
        boundaries = np.flatnonzero(np.any(np.diff(ordered_coordinates, axis=0), axis=1)) + 1
        starts = np.concatenate(([0], boundaries))
        stops = np.concatenate((boundaries, [count]))
        frame = str(frame_id)

        for start, stop in zip(starts, stops, strict=True):
            indices = order[start:stop]
            coordinate = tuple(int(value) for value in ordered_coordinates[start])
            group_weights = weight_values[indices].astype(np.float64)
            batch_weight = float(group_weights.sum())
            batch_count = int(stop - start)
            group_features = features[indices].astype(np.float64)
            batch_feature = np.average(group_features, axis=0, weights=group_weights)
            feature_delta = group_features - batch_feature
            batch_m2 = float(np.einsum("n,nd,nd->", group_weights, feature_delta, feature_delta))
            batch_rgb = np.average(
                rgb_values[indices].astype(np.float64), axis=0, weights=group_weights
            )
            batch_confidence = float(
                np.average(confidence_values[indices].astype(np.float64), weights=group_weights)
            )

            normal_sum = np.zeros(3, dtype=np.float64)
            normal_weight = 0.0
            if normal_values is not None:
                normal_sum = np.einsum(
                    "n,nd->d", group_weights, normal_values[indices].astype(np.float64)
                )
                normal_weight = batch_weight

            view_sum = np.zeros(3, dtype=np.float64)
            view_weight = 0.0
            if view_values is not None:
                view_sum = np.einsum(
                    "n,nd->d", group_weights, view_values[indices].astype(np.float64)
                )
                view_weight = batch_weight

            existing = self._voxels.get(coordinate)
            if existing is None:
                self._voxels[coordinate] = _VoxelAccumulator(
                    coordinate=coordinate,
                    observation_count=batch_count,
                    weight_sum=batch_weight,
                    mean_rgb=batch_rgb.astype(np.float32),
                    mean_feature=batch_feature.astype(np.float32),
                    feature_m2=batch_m2,
                    confidence_mean=batch_confidence,
                    normal_sum=normal_sum,
                    normal_weight_sum=normal_weight,
                    view_direction_sum=view_sum,
                    view_direction_weight_sum=view_weight,
                    last_frame=frame,
                )
                continue

            old_weight = existing.weight_sum
            combined_weight = old_weight + batch_weight
            semantic_delta = batch_feature - existing.mean_feature.astype(np.float64)
            existing.feature_m2 += batch_m2 + float(
                np.dot(semantic_delta, semantic_delta) * old_weight * batch_weight / combined_weight
            )
            existing.mean_feature = (
                existing.mean_feature.astype(np.float64)
                + semantic_delta * (batch_weight / combined_weight)
            ).astype(np.float32)
            existing.mean_rgb = (
                (existing.mean_rgb.astype(np.float64) * old_weight + batch_rgb * batch_weight)
                / combined_weight
            ).astype(np.float32)
            existing.confidence_mean = (
                existing.confidence_mean * old_weight + batch_confidence * batch_weight
            ) / combined_weight
            existing.observation_count += batch_count
            existing.weight_sum = combined_weight
            existing.normal_sum += normal_sum
            existing.normal_weight_sum += normal_weight
            existing.view_direction_sum += view_sum
            existing.view_direction_weight_sum += view_weight
            existing.last_frame = frame

        return len(new_keys)

    def _sorted_accumulators(self) -> list[_VoxelAccumulator]:
        return [self._voxels[key] for key in sorted(self._voxels)]

    def to_arrays(self, *, encode_semantics: bool = True) -> dict[str, np.ndarray]:
        """Materialize the map in deterministic lexicographic voxel order."""

        accumulators = self._sorted_accumulators()
        count = len(accumulators)
        dimension = self.feature_dim or 0
        coordinates = np.asarray([value.coordinate for value in accumulators], dtype=np.int32)
        if count == 0:
            coordinates = np.empty((0, 3), dtype=np.int32)
        centers = (coordinates.astype(np.float32) + 0.5) * self.voxel_size_m
        features = (
            np.stack([value.mean_feature for value in accumulators]).astype(np.float32)
            if count
            else np.empty((0, dimension), dtype=np.float32)
        )
        if encode_semantics and count:
            features = self.codec.encode(features)

        normal = np.zeros((count, 3), dtype=np.float32)
        normal_valid = np.zeros(count, dtype=bool)
        view_direction = np.zeros((count, 3), dtype=np.float32)
        view_direction_valid = np.zeros(count, dtype=bool)
        for index, value in enumerate(accumulators):
            normal_norm = np.linalg.norm(value.normal_sum)
            if value.normal_weight_sum > 0 and normal_norm > 0:
                normal[index] = value.normal_sum / normal_norm
                normal_valid[index] = True
            view_norm = np.linalg.norm(value.view_direction_sum)
            if value.view_direction_weight_sum > 0 and view_norm > 0:
                view_direction[index] = value.view_direction_sum / view_norm
                view_direction_valid[index] = True

        weight_sum = np.asarray([value.weight_sum for value in accumulators], dtype=np.float32)
        feature_m2 = np.asarray([value.feature_m2 for value in accumulators], dtype=np.float64)
        divisor = np.maximum(weight_sum.astype(np.float64) * max(dimension, 1), 1e-12)
        return {
            "voxel_coordinates": coordinates,
            "centers_world": centers.astype(np.float32),
            "observation_count": np.asarray(
                [value.observation_count for value in accumulators], dtype=np.int32
            ),
            "weight_sum": weight_sum,
            "mean_rgb": (
                np.stack([value.mean_rgb for value in accumulators]).astype(np.float32)
                if count
                else np.empty((0, 3), dtype=np.float32)
            ),
            "semantic_features": features,
            "semantic_feature_m2": feature_m2,
            "semantic_variance": (feature_m2 / divisor).astype(np.float32),
            "normal": normal,
            "normal_valid": normal_valid,
            "view_direction": view_direction,
            "view_direction_valid": view_direction_valid,
            "confidence": np.asarray(
                [value.confidence_mean for value in accumulators], dtype=np.float32
            ),
            "last_frame": np.asarray([value.last_frame for value in accumulators], dtype=np.str_),
        }

    def summary(self) -> dict[str, int | float | str | None]:
        arrays = self.to_arrays(encode_semantics=False)
        return {
            "schema_version": self.SCHEMA_VERSION,
            "voxel_size_m": self.voxel_size_m,
            "occupied_voxels": len(self),
            "feature_dim": self.feature_dim,
            "semantic_dtype_on_disk": np.dtype(
                getattr(self.codec, "storage_dtype", np.float32)
            ).name,
            "codec": self.codec.name,
            "total_observations": int(arrays["observation_count"].sum()),
        }

    def content_hash(self) -> str:
        """Hash the numeric arrays materialized from the current map object.

        A loaded map can normalize stored direction vectors while rebuilding
        its accumulators, so this materialized-object hash is not necessarily
        identical to :func:`persisted_voxel_map_content_hash` for the source
        file. Reports must declare which domain they use.
        """

        return _numeric_arrays_content_hash(
            self.voxel_size_m,
            self.to_arrays(encode_semantics=True),
        )

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> Path:
        """Atomically save a non-empty map as an allow-pickle-free NPZ."""

        if not self._voxels:
            raise ValueError("Refusing to save an empty voxel map")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays = self.to_arrays(encode_semantics=True)
        header = {
            **self.summary(),
            "max_voxels": self.max_voxels,
            "metadata": metadata or {},
        }
        temporary_fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        try:
            with os.fdopen(temporary_fd, "wb") as handle:
                np.savez_compressed(
                    handle,
                    **arrays,
                    metadata_json=np.asarray(json.dumps(header, sort_keys=True)),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        return destination

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        codec: SemanticCodec | None = None,
    ) -> Self:
        """Load a map without permitting pickled Python objects."""

        source = Path(path)
        with np.load(source, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if metadata.get("schema_version") != cls.SCHEMA_VERSION:
                raise ValueError(f"Unsupported voxel-map schema: {metadata.get('schema_version')}")
            selected_codec = codec or IdentitySemanticCodec(
                storage_dtype=archive["semantic_features"].dtype
            )
            if selected_codec.name != metadata["codec"]:
                raise ValueError(
                    f"Codec mismatch: map uses {metadata['codec']}, got {selected_codec.name}"
                )
            semantic_features = selected_codec.decode(archive["semantic_features"])
            result = cls(
                float(metadata["voxel_size_m"]),
                feature_dim=int(metadata["feature_dim"]),
                max_voxels=metadata.get("max_voxels"),
                codec=selected_codec,
            )
            coordinates = archive["voxel_coordinates"].astype(np.int32)
            counts = archive["observation_count"].astype(np.int64)
            weights = archive["weight_sum"].astype(np.float64)
            rgbs = archive["mean_rgb"].astype(np.float32)
            feature_m2 = archive["semantic_feature_m2"].astype(np.float64)
            confidence = archive["confidence"].astype(np.float64)
            normals = archive["normal"].astype(np.float64)
            normal_valid = archive["normal_valid"].astype(bool)
            views = archive["view_direction"].astype(np.float64)
            view_valid = archive["view_direction_valid"].astype(bool)
            last_frames = archive["last_frame"].astype(str)

        for index, coordinate_array in enumerate(coordinates):
            coordinate = tuple(int(value) for value in coordinate_array)
            weight = float(weights[index])
            result._voxels[coordinate] = _VoxelAccumulator(
                coordinate=coordinate,
                observation_count=int(counts[index]),
                weight_sum=weight,
                mean_rgb=rgbs[index],
                mean_feature=semantic_features[index].astype(np.float32),
                feature_m2=float(feature_m2[index]),
                confidence_mean=float(confidence[index]),
                normal_sum=normals[index] * weight if normal_valid[index] else np.zeros(3),
                normal_weight_sum=weight if normal_valid[index] else 0.0,
                view_direction_sum=views[index] * weight if view_valid[index] else np.zeros(3),
                view_direction_weight_sum=weight if view_valid[index] else 0.0,
                last_frame=str(last_frames[index]),
            )
        return result

    def _colors(self, color_by: ColorMode) -> np.ndarray:
        arrays = self.to_arrays(encode_semantics=False)
        if color_by == "rgb":
            return np.clip(np.rint(arrays["mean_rgb"]), 0, 255).astype(np.uint8)
        if color_by == "feature_norm":
            values = np.linalg.norm(arrays["semantic_features"], axis=1)
        elif color_by == "uncertainty":
            values = np.sqrt(np.maximum(arrays["semantic_variance"], 0))
        elif color_by == "observation_count":
            values = np.log1p(arrays["observation_count"])
        elif color_by == "confidence":
            values = arrays["confidence"]
        else:
            raise ValueError(f"Unknown color mode: {color_by}")
        if not len(values):
            return np.empty((0, 3), dtype=np.uint8)
        low, high = np.percentile(values, [2, 98])
        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError(f"Invalid scalar values for {color_by} visualization")
        if high <= low:
            normalized = np.full_like(values, 0.5, dtype=np.float64)
        else:
            normalized = np.clip((values - low) / (high - low), 0, 1)
        from matplotlib import colormaps

        return np.rint(colormaps["viridis"](normalized)[:, :3] * 255).astype(np.uint8)

    def write_ply(self, path: str | Path, *, color_by: ColorMode = "rgb") -> Path:
        """Write a binary little-endian point-cloud preview."""

        if not self._voxels:
            raise ValueError("Cannot preview an empty voxel map")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays = self.to_arrays(encode_semantics=False)
        colors = self._colors(color_by)
        dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
                ("confidence", "<f4"),
                ("observation_count", "<u4"),
                ("semantic_variance", "<f4"),
            ]
        )
        vertices = np.empty(len(self), dtype=dtype)
        vertices["x"], vertices["y"], vertices["z"] = arrays["centers_world"].T
        vertices["red"], vertices["green"], vertices["blue"] = colors.T
        vertices["confidence"] = arrays["confidence"]
        vertices["observation_count"] = arrays["observation_count"]
        vertices["semantic_variance"] = arrays["semantic_variance"]
        header = "\n".join(
            [
                "ply",
                "format binary_little_endian 1.0",
                "comment semantic_3d_chat world axes: X right, Y forward, Z up",
                f"element vertex {len(vertices)}",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "property float confidence",
                "property uint observation_count",
                "property float semantic_variance",
                "end_header",
                "",
            ]
        ).encode("ascii")
        with destination.open("wb") as handle:
            handle.write(header)
            vertices.tofile(handle)
        return destination

    def write_png(
        self,
        path: str | Path,
        *,
        color_by: ColorMode = "rgb",
        max_points: int = 100_000,
    ) -> Path:
        """Write deterministic top and side projections for visual validation."""

        if not self._voxels:
            raise ValueError("Cannot preview an empty voxel map")
        if max_points < 1:
            raise ValueError("max_points must be positive")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays = self.to_arrays(encode_semantics=False)
        count = len(self)
        if count > max_points:
            indices = np.linspace(0, count - 1, max_points, dtype=np.int64)
        else:
            indices = np.arange(count)
        points = arrays["centers_world"][indices]
        colors = self._colors(color_by)[indices].astype(np.float32) / 255.0

        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(10, 4.8), dpi=140, constrained_layout=True)
        FigureCanvasAgg(figure)
        top = figure.add_subplot(1, 2, 1)
        side = figure.add_subplot(1, 2, 2)
        marker_size = max(0.2, min(5.0, 10000.0 / max(len(indices), 1)))
        top.scatter(points[:, 0], points[:, 1], c=colors, s=marker_size, linewidths=0)
        top.set(xlabel="X — right (m)", ylabel="Y — forward (m)", title=f"Top: {color_by}")
        top.set_aspect("equal", adjustable="box")
        side.scatter(points[:, 0], points[:, 2], c=colors, s=marker_size, linewidths=0)
        side.set(xlabel="X — right (m)", ylabel="Z — up (m)", title=f"Front: {color_by}")
        side.set_aspect("equal", adjustable="box")
        figure.suptitle(f"Semantic voxel map — {len(self):,} occupied voxels")
        figure.savefig(destination, format="png")
        figure.clear()
        return destination

    def export_previews(self, directory: str | Path, *, stem: str = "map") -> list[Path]:
        """Export the standard human-facing, non-semantic-label visualizations."""

        output_directory = Path(directory)
        outputs = [self.write_ply(output_directory / f"{stem}_rgb.ply")]
        for mode in ("rgb", "feature_norm", "uncertainty", "observation_count"):
            outputs.append(self.write_png(output_directory / f"{stem}_{mode}.png", color_by=mode))
        return outputs
