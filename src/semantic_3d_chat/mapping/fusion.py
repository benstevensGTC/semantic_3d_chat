"""Fuse rendered RGB-D frames and dense full-image features into a voxel map."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from semantic_3d_chat.mapping.depth_projection import project_depth_to_world
from semantic_3d_chat.mapping.semantic_codec import IdentitySemanticCodec
from semantic_3d_chat.mapping.voxel_map import (
    PERSISTED_MAP_CONTENT_HASH_DOMAIN,
    SparseVoxelMap,
    persisted_voxel_map_content_hash,
)
from semantic_3d_chat.rendering_io import iter_frames, load_rgb_depth


@dataclass(frozen=True)
class FrameFusionStats:
    frame_id: str
    valid_depth_points: int
    newly_occupied_voxels: int
    total_occupied_voxels: int
    feature_dim: int


def distance_confidence(depth_m: np.ndarray, scale_m: float) -> np.ndarray:
    """Smooth deterministic confidence that decreases with camera distance."""

    depth = np.asarray(depth_m, dtype=np.float32)
    if not np.isfinite(scale_m) or scale_m <= 0:
        raise ValueError("scale_m must be finite and positive")
    if not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError("depth_m must contain finite positive values")
    confidence = np.exp(-depth / scale_m)
    return np.clip(confidence, np.finfo(np.float32).eps, 1.0).astype(np.float32)


def sample_spatial_field(
    field: np.ndarray,
    pixels_uv: np.ndarray,
    *,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Bilinearly sample an ``[grid_h, grid_w, D]`` field at image pixels.

    The mapping uses pixel-cell centers and ``align_corners=False`` semantics.
    It therefore works both for an already upsampled image-resolution field and
    for a native patch grid such as CLIP's 14x14 full-image token grid.
    """

    values = np.asarray(field)
    pixels = np.asarray(pixels_uv)
    if values.ndim != 3 or values.shape[2] < 1:
        raise ValueError(f"field must have shape [grid_h, grid_w, D], got {values.shape}")
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError(f"pixels_uv must have shape [N, 2], got {pixels.shape}")
    if not np.isfinite(values).all():
        raise ValueError("field contains NaN or infinite values")
    image_height, image_width = image_shape
    if image_height < 1 or image_width < 1:
        raise ValueError("image_shape must contain positive dimensions")
    if pixels.size:
        if np.any(pixels[:, 0] < 0) or np.any(pixels[:, 0] >= image_width):
            raise ValueError("pixels_uv contains a U coordinate outside image_shape")
        if np.any(pixels[:, 1] < 0) or np.any(pixels[:, 1] >= image_height):
            raise ValueError("pixels_uv contains a V coordinate outside image_shape")

    grid_height, grid_width, dimension = values.shape
    if (grid_height, grid_width) == (image_height, image_width):
        return values[pixels[:, 1], pixels[:, 0]].astype(np.float32, copy=False)
    if not len(pixels):
        return np.empty((0, dimension), dtype=np.float32)

    x = (pixels[:, 0].astype(np.float64) + 0.5) * grid_width / image_width - 0.5
    y = (pixels[:, 1].astype(np.float64) + 0.5) * grid_height / image_height - 0.5
    x = np.clip(x, 0.0, grid_width - 1.0)
    y = np.clip(y, 0.0, grid_height - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, grid_width - 1)
    y1 = np.minimum(y0 + 1, grid_height - 1)
    wx = (x - x0).astype(np.float32)[:, None]
    wy = (y - y0).astype(np.float32)[:, None]
    top = values[y0, x0].astype(np.float32) * (1.0 - wx) + values[y0, x1] * wx
    bottom = values[y1, x0].astype(np.float32) * (1.0 - wx) + values[y1, x1] * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.float32, copy=False)


def fuse_frame(
    voxel_map: SparseVoxelMap,
    *,
    depth_m: np.ndarray,
    rgb: np.ndarray,
    spatial_features: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    frame_id: str | int,
    min_depth_m: float = 0.1,
    max_depth_m: float = 10.0,
    pixel_stride: int = 1,
    confidence_distance_scale_m: float = 6.0,
    normals_camera: np.ndarray | None = None,
) -> FrameFusionStats:
    """Project and fuse one complete-image feature field.

    ``spatial_features`` must come from one full-image encoder call.  This
    function only interpolates its spatial token grid; it never crops or invokes
    an image encoder.
    """

    depth = np.asarray(depth_m)
    color = np.asarray(rgb)
    if depth.ndim != 2:
        raise ValueError(f"depth_m must have shape [H, W], got {depth.shape}")
    if color.shape != (*depth.shape, 3):
        raise ValueError(f"rgb must have shape {(*depth.shape, 3)}, got {color.shape}")
    if not np.isfinite(color).all() or np.any(color < 0) or np.any(color > 255):
        raise ValueError("rgb must contain finite values in [0, 255]")

    projection = project_depth_to_world(
        depth,
        intrinsics,
        camera_to_world,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        pixel_stride=pixel_stride,
    )
    if len(projection) == 0:
        raise ValueError(f"Frame {frame_id} has no valid depth samples")
    sampled_features = sample_spatial_field(
        spatial_features, projection.pixels_uv, image_shape=depth.shape
    )
    sampled_rgb = color[projection.pixels_uv[:, 1], projection.pixels_uv[:, 0]]
    confidence = distance_confidence(projection.depth_m, confidence_distance_scale_m)

    normals_world: np.ndarray | None = None
    if normals_camera is not None:
        sampled_normals = sample_spatial_field(
            normals_camera, projection.pixels_uv, image_shape=depth.shape
        )
        if sampled_normals.shape[1] != 3:
            raise ValueError("normals_camera must have three channels")
        rotation = np.asarray(camera_to_world, dtype=np.float64)[:3, :3]
        normals_world = sampled_normals.astype(np.float64) @ rotation.T

    new_voxels = voxel_map.add_observations(
        projection.points_world,
        sampled_features,
        rgb=sampled_rgb,
        normals_world=normals_world,
        view_directions_world=projection.view_directions_world,
        confidence=confidence,
        frame_id=frame_id,
    )
    return FrameFusionStats(
        frame_id=str(frame_id),
        valid_depth_points=len(projection),
        newly_occupied_voxels=new_voxels,
        total_occupied_voxels=len(voxel_map),
        feature_dim=int(sampled_features.shape[1]),
    )


def load_feature_field(path: str | Path) -> np.ndarray:
    """Load a cached spatial feature field from NPY or safe NPZ storage."""

    source = Path(path)
    if source.suffix == ".npy":
        features = np.load(source, allow_pickle=False)
    elif source.suffix == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            for candidate in ("spatial_features", "features", "patch_features"):
                if candidate in archive.files:
                    features = archive[candidate].copy()
                    break
            else:
                raise ValueError(
                    f"{source} contains none of spatial_features/features/patch_features"
                )
    else:
        raise ValueError(f"Unsupported feature cache extension: {source.suffix}")
    features = np.asarray(features)
    if features.ndim != 3:
        raise ValueError(f"Feature cache must contain [grid_h, grid_w, D], got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"Feature cache contains NaN or infinity: {source}")
    if np.any(np.linalg.norm(features.astype(np.float32), axis=2) == 0):
        raise ValueError(f"Feature cache contains zero-norm patch embeddings: {source}")
    return features


def _resolve_feature_path(features_directory: Path, frame_id: str) -> Path:
    candidates = [features_directory / f"{frame_id}{suffix}" for suffix in (".npy", ".npz")]
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"Expected exactly one opaque feature cache for {frame_id}; found {existing}"
        )
    return existing[0]


def fuse_manifest(
    manifest_path: str | Path,
    features_directory: str | Path,
    *,
    voxel_size_m: float,
    depth_min_m: float = 0.1,
    depth_max_m: float = 10.0,
    pixel_stride: int = 1,
    max_voxels: int | None = None,
    confidence_distance_scale_m: float = 6.0,
    feature_loader: Callable[[Path], np.ndarray] = load_feature_field,
) -> tuple[SparseVoxelMap, list[FrameFusionStats]]:
    """Fuse every frame in a sanitized manifest, without oracle metadata."""

    features_root = Path(features_directory)
    voxel_map = SparseVoxelMap(
        voxel_size_m,
        max_voxels=max_voxels,
        codec=IdentitySemanticCodec(storage_dtype=np.float16),
    )
    statistics: list[FrameFusionStats] = []
    for frame in iter_frames(manifest_path):
        rgb, depth = load_rgb_depth(frame)
        feature_path = _resolve_feature_path(features_root, frame.frame_id)
        features = feature_loader(feature_path)
        frame_stats = fuse_frame(
            voxel_map,
            depth_m=depth,
            rgb=rgb,
            spatial_features=features,
            intrinsics=frame.intrinsics,
            camera_to_world=frame.camera_to_world,
            frame_id=frame.frame_id,
            min_depth_m=depth_min_m,
            max_depth_m=depth_max_m,
            pixel_stride=pixel_stride,
            confidence_distance_scale_m=confidence_distance_scale_m,
        )
        statistics.append(frame_stats)
    if not statistics:
        raise ValueError("Sanitized manifest contains no frames")
    return voxel_map, statistics


app = typer.Typer(add_completion=False, help=__doc__)


@app.command("build")
def build_map_cli(
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    features: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    preview_directory: Annotated[Path | None, typer.Option(file_okay=False)] = None,
    voxel_size_m: Annotated[float, typer.Option(min=0.001)] = 0.05,
    depth_min_m: Annotated[float, typer.Option(min=0.0)] = 0.1,
    depth_max_m: Annotated[float, typer.Option(min=0.001)] = 10.0,
    pixel_stride: Annotated[int, typer.Option(min=1)] = 1,
    max_voxels: Annotated[int | None, typer.Option(min=1)] = None,
    confidence_distance_scale_m: Annotated[float, typer.Option(min=0.001)] = 6.0,
) -> None:
    """Build and preview a full-scene map from opaque frame feature caches."""

    voxel_map, statistics = fuse_manifest(
        manifest,
        features,
        voxel_size_m=voxel_size_m,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        pixel_stride=pixel_stride,
        max_voxels=max_voxels,
        confidence_distance_scale_m=confidence_distance_scale_m,
    )
    voxel_map.save(output, metadata={"manifest": manifest.name})
    previews = voxel_map.export_previews(preview_directory or output.parent)
    for stats in statistics:
        typer.echo(json.dumps({"phase": "mapping", **asdict(stats)}, sort_keys=True))
    typer.echo(
        json.dumps(
            {
                "phase": "mapping_complete",
                **voxel_map.summary(),
                "map_path": str(output),
                "preview_paths": [str(path) for path in previews],
                "content_hash": persisted_voxel_map_content_hash(output),
                "content_hash_domain": PERSISTED_MAP_CONTENT_HASH_DOMAIN,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    app()
