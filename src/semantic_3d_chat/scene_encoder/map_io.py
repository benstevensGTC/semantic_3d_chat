from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class MapTensorData:
    semantic: torch.Tensor
    xyz: torch.Tensor
    rgb: torch.Tensor
    normal: torch.Tensor
    confidence: torch.Tensor
    observation_count: torch.Tensor
    room_min: torch.Tensor
    room_max: torch.Tensor
    source_voxel_count: int
    input_voxel_size_m: float | None

    @property
    def feature_dim(self) -> int:
        return int(self.semantic.shape[1])

    @property
    def voxel_count(self) -> int:
        return int(self.semantic.shape[0])

    def to(self, device: torch.device | str) -> MapTensorData:
        return MapTensorData(
            **{
                name: value.to(device) if isinstance(value, torch.Tensor) else value
                for name, value in vars(self).items()
            }
        )


def _aggregate_rows(values: np.ndarray, inverse: np.ndarray, groups: int, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    result = np.zeros((groups, values.shape[1]), dtype=np.float32)
    np.add.at(result, inverse, values * weights[:, None])
    totals = np.bincount(inverse, weights=weights, minlength=groups).astype(np.float32)
    return result / np.maximum(totals[:, None], np.finfo(np.float32).eps)


def _coarsen_arrays(arrays: dict[str, np.ndarray], voxel_size_m: float) -> dict[str, np.ndarray]:
    xyz = arrays["centers_world"].astype(np.float32)
    coordinates = np.floor(xyz / voxel_size_m).astype(np.int32)
    _, inverse = np.unique(coordinates, axis=0, return_inverse=True)
    groups = int(inverse.max()) + 1
    weights = np.maximum(arrays["observation_count"].astype(np.float32), 1.0)
    features = arrays["semantic_features"].astype(np.float32)
    # Chunk the 2048D payload to cap temporary memory without dropping dimensions.
    aggregated_features = np.empty((groups, features.shape[1]), dtype=np.float32)
    for start in range(0, features.shape[1], 128):
        stop = min(start + 128, features.shape[1])
        aggregated_features[:, start:stop] = _aggregate_rows(
            features[:, start:stop], inverse, groups, weights
        )
    normals = _aggregate_rows(arrays["normal"], inverse, groups, weights)
    normal_norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, normal_norms, out=np.zeros_like(normals), where=normal_norms > 0)
    return {
        "semantic_features": aggregated_features,
        "centers_world": _aggregate_rows(xyz, inverse, groups, weights),
        "mean_rgb": _aggregate_rows(arrays["mean_rgb"], inverse, groups, weights),
        "normal": normals,
        "confidence": (
            np.bincount(
                inverse,
                weights=arrays["confidence"].astype(np.float32) * weights,
                minlength=groups,
            )
            / np.maximum(np.bincount(inverse, weights=weights, minlength=groups), 1e-7)
        ).astype(np.float32),
        "observation_count": np.bincount(
            inverse, weights=arrays["observation_count"], minlength=groups
        ).astype(np.float32),
    }


def load_map_tensors(
    path: str | Path,
    room_size_m: list[float] | tuple[float, float, float],
    device: torch.device | str = "cpu",
    input_voxel_size_m: float | None = None,
) -> MapTensorData:
    """Load only numerical runtime fields from a safe semantic-map archive."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "semantic_features",
            "centers_world",
            "mean_rgb",
            "normal",
            "confidence",
            "observation_count",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Map is missing fields: {sorted(missing)}")
        arrays = {name: archive[name].copy() for name in required}
    count = arrays["centers_world"].shape[0]
    if count == 0:
        raise ValueError("Cannot tokenize an empty semantic map")
    source_voxel_count = int(count)
    if input_voxel_size_m is not None:
        if not np.isfinite(input_voxel_size_m) or input_voxel_size_m <= 0:
            raise ValueError("input_voxel_size_m must be finite and positive")
        arrays = _coarsen_arrays(arrays, float(input_voxel_size_m))
        count = arrays["centers_world"].shape[0]
    semantic = arrays["semantic_features"].astype(np.float32)
    if semantic.ndim != 2 or semantic.shape[0] != count or not np.isfinite(semantic).all():
        raise ValueError("Invalid semantic feature matrix")
    size = np.asarray(room_size_m, dtype=np.float32)
    if size.shape != (3,) or np.any(size <= 0):
        raise ValueError("room_size_m must contain three positive values")
    room_min = np.array([-size[0] / 2, -size[1] / 2, 0.0], dtype=np.float32)
    room_max = np.array([size[0] / 2, size[1] / 2, size[2]], dtype=np.float32)
    result = MapTensorData(
        semantic=torch.from_numpy(semantic),
        xyz=torch.from_numpy(arrays["centers_world"].astype(np.float32)),
        rgb=torch.from_numpy(arrays["mean_rgb"].astype(np.float32)),
        normal=torch.from_numpy(arrays["normal"].astype(np.float32)),
        confidence=torch.from_numpy(arrays["confidence"].astype(np.float32)),
        observation_count=torch.from_numpy(arrays["observation_count"].astype(np.float32)),
        room_min=torch.from_numpy(room_min),
        room_max=torch.from_numpy(room_max),
        source_voxel_count=source_voxel_count,
        input_voxel_size_m=input_voxel_size_m,
    )
    for name, value in vars(result).items():
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            raise ValueError(f"Map field {name} contains NaN or infinity")
    return result.to(device)
