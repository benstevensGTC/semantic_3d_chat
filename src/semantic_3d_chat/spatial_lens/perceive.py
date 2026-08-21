"""Turn a scan into a semantic point cloud using Gemma's own vision encoder.

Each scanned frame goes through Gemma's vision tower once, producing a 48x48
grid of patch features.  Every depth pixel is back-projected to a world point
through the exact camera pose, the patch grid is bilinearly sampled at that
pixel, and the result is fused into a voxel grid.  The stored payload is the
1536-D language-aligned stream -- the output of Gemma's own trained vision
projector, which already lives in its decoder embedding space.

Nothing here knows what any object is.  The output is coordinates, colour and
continuous features; discovering objects is a separate, later step.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from semantic_3d_chat.config import PROJECT_ROOT

CLOUD_SCHEMA = "semantic_3d_chat.spatial_lens.point_cloud.v1"


@dataclass(frozen=True)
class SemanticCloud:
    """A fused, voxelized semantic point cloud in world coordinates."""

    centers_m: np.ndarray  # [N, 3] float32
    rgb: np.ndarray  # [N, 3] float32 in [0, 1]
    features: np.ndarray  # [N, D] float16, Gemma language-aligned stream
    counts: np.ndarray  # [N] int32
    voxel_size_m: float
    room_size_m: tuple[float, float, float]

    def __len__(self) -> int:
        return int(self.centers_m.shape[0])

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            centers_m=self.centers_m.astype(np.float32, copy=False),
            rgb=self.rgb.astype(np.float32, copy=False),
            features=self.features.astype(np.float16, copy=False),
            counts=self.counts.astype(np.int32, copy=False),
            voxel_size_m=np.float32(self.voxel_size_m),
            room_size_m=np.asarray(self.room_size_m, dtype=np.float32),
            schema=np.str_(CLOUD_SCHEMA),
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> SemanticCloud:
        with np.load(Path(path), allow_pickle=False) as data:
            if str(data["schema"]) != CLOUD_SCHEMA:
                raise ValueError("Unexpected point-cloud schema")
            return cls(
                centers_m=data["centers_m"],
                rgb=data["rgb"],
                features=data["features"],
                counts=data["counts"],
                voxel_size_m=float(data["voxel_size_m"]),
                room_size_m=tuple(float(v) for v in data["room_size_m"]),
            )


def _load_scan(scan_root: Path) -> Mapping[str, Any]:
    manifest = json.loads((scan_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("contains_instance_labels") is not False:
        raise ValueError("Scan manifest must assert it carries no instance labels")
    return manifest


def build_semantic_cloud(
    room: str,
    *,
    voxel_size_m: float = 0.05,
    pixel_stride: int = 5,
    max_depth_m: float = 12.0,
    device: str | None = None,
    progress: Any | None = None,
    model_id: str | None = None,
    revision: str | None = None,
) -> SemanticCloud:
    """Fuse every scanned frame into one semantic point cloud.

    The cloud is not portable between models. Every point carries the output of
    one decoder's vision projector, so its width is that decoder's hidden size
    -- 1536 for E2B, 2560 for E4B -- and a map built with one cannot be read by
    the other. Changing ``model_id`` means rebuilding the map, not reusing it.
    """

    import torch
    from PIL import Image

    from semantic_3d_chat.mapping.depth_projection import project_depth_to_world
    from semantic_3d_chat.mapping.fusion import sample_spatial_field
    from semantic_3d_chat.vision.gemma4_encoder import DenseGemma4Encoder

    room_root = PROJECT_ROOT / "data" / "spatial_lens" / room
    scan_root = room_root / "scans"
    manifest = _load_scan(scan_root)
    frames = list(manifest["frames"])
    if not frames:
        raise ValueError("Scan has no frames")

    from semantic_3d_chat.language.model_choice import DEFAULT_MODEL, revision_for

    chosen = model_id or DEFAULT_MODEL
    encoder = DenseGemma4Encoder.from_pretrained(
        chosen,
        revision=revision or revision_for(chosen),
        device=None if device is None else torch.device(device),
        requested_dtype="bfloat16",
        storage_dtype=torch.float16,
        local_files_only=True,
    )

    # Accumulate per integer voxel so memory stays bounded by room volume
    # rather than by pixel count.
    sums: dict[tuple[int, int, int], np.ndarray] = {}
    colors: dict[tuple[int, int, int], np.ndarray] = {}
    counts: dict[tuple[int, int, int], int] = {}

    for index, frame in enumerate(frames):
        rgb_image = Image.open(scan_root / frame["rgb_path"]).convert("RGB")
        rgb_array = np.asarray(rgb_image, dtype=np.float32) / 255.0
        depth = np.load(scan_root / frame["depth_path"])
        intrinsics = np.asarray(frame["intrinsics"], dtype=np.float64)
        camera_to_world = np.asarray(frame["camera_to_world"], dtype=np.float64)

        # The dense encoder is pinned to complete 224x224 renders. The 48x48
        # patch grid it returns is resolution-independent, so sampling it at
        # full-resolution depth pixels stays exact while the scan keeps the
        # higher-resolution RGB that object naming needs later.
        encoder_image = (
            rgb_image
            if rgb_image.size == (224, 224)
            else rgb_image.resize((224, 224), Image.BICUBIC)
        )
        dense = encoder.encode_image(encoder_image)
        aligned = dense.clip_aligned.detach().to(torch.float32).cpu().numpy()

        projection = project_depth_to_world(
            depth,
            intrinsics,
            camera_to_world,
            min_depth_m=0.05,
            max_depth_m=max_depth_m,
            pixel_stride=int(pixel_stride),
        )
        points = projection.points_world
        if not len(points):
            continue
        pixels = projection.pixels_uv
        sampled = sample_spatial_field(
            aligned, pixels, image_shape=(depth.shape[0], depth.shape[1])
        )
        columns = np.clip(np.round(pixels[:, 0]).astype(int), 0, rgb_array.shape[1] - 1)
        rows = np.clip(np.round(pixels[:, 1]).astype(int), 0, rgb_array.shape[0] - 1)
        point_rgb = rgb_array[rows, columns]

        keys = np.floor(points / voxel_size_m).astype(np.int64)
        for key, feature, color in zip(map(tuple, keys), sampled, point_rgb, strict=True):
            if key in sums:
                sums[key] += feature
                colors[key] += color
                counts[key] += 1
            else:
                sums[key] = feature.astype(np.float32).copy()
                colors[key] = color.astype(np.float32).copy()
                counts[key] = 1
        if progress is not None:
            progress(index + 1, len(frames), len(sums))

    if not sums:
        raise ValueError("Fusion produced no occupied voxels")

    ordered = sorted(sums)
    count_array = np.array([counts[key] for key in ordered], dtype=np.int32)
    centers = (np.array(ordered, dtype=np.float32) + 0.5) * float(voxel_size_m)
    feature_array = np.stack(
        [sums[key] / counts[key] for key in ordered]
    ).astype(np.float16)
    color_array = np.stack([colors[key] / counts[key] for key in ordered]).astype(
        np.float32
    )
    return SemanticCloud(
        centers_m=centers,
        rgb=color_array,
        features=feature_array,
        counts=count_array,
        voxel_size_m=float(voxel_size_m),
        room_size_m=tuple(float(v) for v in manifest["room_size_m"]),
    )


__all__ = ["CLOUD_SCHEMA", "SemanticCloud", "build_semantic_cloud"]
