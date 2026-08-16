from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

RUNTIME_MAP_FIELDS = frozenset(
    {
        "voxel_coordinates",
        "centers_world",
        "observation_count",
        "weight_sum",
        "mean_rgb",
        "semantic_features",
        "semantic_feature_m2",
        "semantic_variance",
        "normal",
        "normal_valid",
        "view_direction",
        "view_direction_valid",
        "confidence",
        "last_frame",
        "metadata_json",
    }
)

_REQUIRED_NUMERIC_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "semantic_features",
        "centers_world",
        "mean_rgb",
        "normal",
        "confidence",
        "observation_count",
    }
)
_MAP_HEADER_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "voxel_size_m",
        "occupied_voxels",
        "feature_dim",
        "semantic_dtype_on_disk",
        "codec",
        "total_observations",
        "max_voxels",
        "metadata",
    }
)
_PERSISTENT_METADATA_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "scene_id",
        "map_version",
        "map_sha256",
        "prior_map_sha256",
        "observation_id",
        "vision_encoder_calls",
        "feature_grid_height",
        "feature_grid_width",
        "feature_dim",
    }
)
_ABLATION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "mode",
        "seed",
        "voxel_count",
        "affected_fields",
        "permutation_algorithm",
        "permutation_sha256",
        "source_sha256",
    }
)
_ABLATION_MODES: Final[frozenset[str]] = frozenset(
    {
        "geometry_shuffle",
        "semantic_shuffle",
        "zero_semantics",
        "zero_rgb",
        "zero_normals",
        "zero_xyz",
        "xyz_shuffle",
    }
)
_ABLATION_AFFECTED_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "semantic_features",
        "semantic_feature_m2",
        "semantic_variance",
        "mean_rgb",
        "normal",
        "normal_valid",
        "voxel_coordinates",
        "centers_world",
        "view_direction",
        "view_direction_valid",
    }
)
_OPTIONAL_NUMERIC_FIELD_CONTRACTS: Final[dict[str, tuple[frozenset[str], int]]] = {
    "voxel_coordinates": (frozenset({"i", "u"}), 2),
    "weight_sum": (frozenset({"f", "i", "u"}), 1),
    "semantic_feature_m2": (frozenset({"f", "i", "u"}), 1),
    "semantic_variance": (frozenset({"f", "i", "u"}), 1),
    "normal_valid": (frozenset({"b"}), 1),
    "view_direction": (frozenset({"f", "i", "u"}), 2),
    "view_direction_valid": (frozenset({"b"}), 1),
}
_FRAME_ID: Final[re.Pattern[str]] = re.compile(r"(?:f|frame|o)_[0-9a-f]{6,64}")
_SCENE_ID: Final[re.Pattern[str]] = re.compile(r"scene_[0-9]{6}")
_OBSERVATION_ID: Final[re.Pattern[str]] = re.compile(r"o_[0-9]{6}")
_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_PERSISTENT_SCHEMA: Final[str] = "semantic_3d_chat.embodied_map.v1"
_MAX_METADATA_CHARACTERS: Final[int] = 32_768


def _json_object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Semantic-map metadata repeats field {key!r}")
        result[key] = value
    return result


def _strict_positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Semantic-map {field} must be a positive integer")
    return value


def _strict_nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Semantic-map {field} must be a nonnegative integer")
    return value


def _strict_sha256(value: object, *, field: str, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"Semantic-map {field} must be a lowercase SHA-256 digest")


def _validate_opaque_scene_id(value: object) -> None:
    if not isinstance(value, str) or _SCENE_ID.fullmatch(value) is None:
        raise ValueError("Semantic-map scene_id must be opaque")


def _parse_metadata_json(value: np.ndarray) -> dict[str, Any]:
    if value.shape != () or value.dtype.kind not in {"S", "U"}:
        raise ValueError("Semantic-map metadata_json must be a scalar string")
    raw = value.item()
    try:
        serialized = raw.decode("utf-8") if isinstance(raw, (bytes, np.bytes_)) else str(raw)
    except UnicodeDecodeError as exc:
        raise ValueError("Semantic-map metadata_json must be valid UTF-8") from exc
    if len(serialized) > _MAX_METADATA_CHARACTERS:
        raise ValueError("Semantic-map metadata_json exceeds the runtime size limit")
    try:
        decoded = json.loads(serialized, object_pairs_hook=_json_object_without_duplicate_keys)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Semantic-map metadata_json is not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise TypeError("Semantic-map metadata_json must decode to an object")
    return decoded


def _validate_persistent_metadata(metadata: Mapping[str, Any], *, feature_dim: int) -> None:
    if set(metadata) != set(_PERSISTENT_METADATA_FIELDS):
        raise ValueError("Semantic-map persistent metadata fields differ from the runtime schema")
    if metadata.get("schema") != _PERSISTENT_SCHEMA:
        raise ValueError("Semantic-map persistent schema is unsupported")
    _validate_opaque_scene_id(metadata.get("scene_id"))
    _strict_positive_int(metadata.get("map_version"), field="map_version")
    _strict_sha256(metadata.get("map_sha256"), field="map_sha256")
    _strict_sha256(metadata.get("prior_map_sha256"), field="prior_map_sha256")
    observation_id = metadata.get("observation_id")
    if not isinstance(observation_id, str) or _OBSERVATION_ID.fullmatch(observation_id) is None:
        raise ValueError("Semantic-map observation_id must be opaque")
    if (
        _strict_positive_int(metadata.get("vision_encoder_calls"), field="vision_encoder_calls")
        != 1
    ):
        raise ValueError("Semantic-map update must record exactly one full-image encoder call")
    _strict_positive_int(metadata.get("feature_grid_height"), field="feature_grid_height")
    _strict_positive_int(metadata.get("feature_grid_width"), field="feature_grid_width")
    if _strict_positive_int(metadata.get("feature_dim"), field="feature_dim") != feature_dim:
        raise ValueError("Semantic-map persistent feature_dim differs from the numeric map")


def _validate_ablation_metadata(ablation: object, *, voxel_count: int) -> None:
    if not isinstance(ablation, Mapping) or set(ablation) != set(_ABLATION_FIELDS):
        raise ValueError("Semantic-map ablation metadata fields differ from the runtime schema")
    if ablation.get("mode") not in _ABLATION_MODES:
        raise ValueError("Semantic-map ablation mode is not allowlisted")
    _strict_nonnegative_int(ablation.get("seed"), field="ablation seed")
    if (
        _strict_positive_int(ablation.get("voxel_count"), field="ablation voxel_count")
        != voxel_count
    ):
        raise ValueError("Semantic-map ablation voxel_count differs from the numeric map")
    affected = ablation.get("affected_fields")
    if (
        not isinstance(affected, list)
        or not all(isinstance(name, str) for name in affected)
        or len(affected) != len(set(affected))
        or not set(affected) <= set(_ABLATION_AFFECTED_FIELDS)
    ):
        raise ValueError("Semantic-map ablation affected_fields are not allowlisted")
    algorithm = ablation.get("permutation_algorithm")
    if algorithm not in {None, "numpy.PCG64"}:
        raise ValueError("Semantic-map ablation permutation algorithm is not allowlisted")
    _strict_sha256(
        ablation.get("permutation_sha256"),
        field="ablation permutation_sha256",
        optional=True,
    )
    _strict_sha256(ablation.get("source_sha256"), field="ablation source_sha256")


def _validate_map_header(
    header: Mapping[str, Any],
    *,
    voxel_count: int,
    feature_dim: int,
    semantic_dtype: np.dtype[Any],
    total_observations: int,
) -> None:
    header_fields = set(header)
    if header_fields not in (
        set(_MAP_HEADER_FIELDS),
        set(_MAP_HEADER_FIELDS) | {"ablation"},
    ):
        raise ValueError("Semantic-map header fields differ from the runtime schema")
    if _strict_positive_int(header.get("schema_version"), field="schema_version") != 1:
        raise ValueError("Semantic-map schema_version is unsupported")
    voxel_size = header.get("voxel_size_m")
    if (
        isinstance(voxel_size, bool)
        or not isinstance(voxel_size, (int, float))
        or not np.isfinite(voxel_size)
        or voxel_size <= 0
    ):
        raise ValueError("Semantic-map voxel_size_m must be finite and positive")
    if _strict_positive_int(header.get("occupied_voxels"), field="occupied_voxels") != voxel_count:
        raise ValueError("Semantic-map occupied_voxels differs from the numeric map")
    if _strict_positive_int(header.get("feature_dim"), field="feature_dim") != feature_dim:
        raise ValueError("Semantic-map feature_dim differs from the numeric map")
    dtype_name = np.dtype(semantic_dtype).name
    if header.get("semantic_dtype_on_disk") != dtype_name:
        raise ValueError("Semantic-map semantic dtype header differs from the numeric map")
    if header.get("codec") != f"identity-{dtype_name}":
        raise ValueError("Semantic-map codec is not the lossless-dimension identity codec")
    if (
        _strict_positive_int(header.get("total_observations"), field="total_observations")
        != total_observations
    ):
        raise ValueError("Semantic-map total_observations differs from the numeric map")
    max_voxels = header.get("max_voxels")
    if max_voxels is not None and (
        _strict_positive_int(max_voxels, field="max_voxels") < voxel_count
    ):
        raise ValueError("Semantic-map max_voxels is smaller than occupied_voxels")
    metadata = header.get("metadata")
    if not isinstance(metadata, Mapping):
        raise TypeError("Semantic-map metadata must be an object")
    if not metadata:
        pass
    elif set(metadata) == {"scene_id"}:
        _validate_opaque_scene_id(metadata.get("scene_id"))
    elif set(metadata) == set(_PERSISTENT_METADATA_FIELDS):
        _validate_persistent_metadata(metadata, feature_dim=feature_dim)
    else:
        raise ValueError("Semantic-map metadata contains non-runtime or semantic fields")
    if "ablation" in header:
        _validate_ablation_metadata(header["ablation"], voxel_count=voxel_count)


def _validate_last_frame(values: np.ndarray, *, voxel_count: int) -> None:
    if values.dtype.kind not in {"S", "U"} or values.shape != (voxel_count,):
        raise ValueError("Semantic-map last_frame must be one opaque string per voxel")
    try:
        frames = values.astype(str).tolist()
    except UnicodeDecodeError as exc:
        raise ValueError("Semantic-map last_frame must contain valid UTF-8") from exc
    if any(_FRAME_ID.fullmatch(frame) is None for frame in frames):
        raise ValueError("Semantic-map last_frame contains a non-opaque identifier")


def _validate_optional_numeric_fields(archive: np.lib.npyio.NpzFile, *, voxel_count: int) -> None:
    for name, (allowed_kinds, dimensions) in _OPTIONAL_NUMERIC_FIELD_CONTRACTS.items():
        if name not in archive.files:
            continue
        values = archive[name]
        if values.dtype.kind not in allowed_kinds or values.ndim != dimensions:
            raise ValueError(f"Semantic-map field {name} is not a permitted numeric array")
        if values.shape[0] != voxel_count:
            raise ValueError(f"Semantic-map field {name} does not share the voxel count")
        if dimensions == 2 and values.shape[1] != 3:
            raise ValueError(f"Semantic-map field {name} must have three columns")
        if values.dtype.kind != "b" and not np.isfinite(values).all():
            raise ValueError(f"Semantic-map field {name} contains NaN or infinity")


def _validate_archive_field_names(archive: np.lib.npyio.NpzFile) -> None:
    fields = set(archive.files)
    unexpected = fields - set(RUNTIME_MAP_FIELDS)
    if unexpected:
        raise ValueError(
            f"Runtime map contains non-numeric or non-allowlisted fields: {sorted(unexpected)}"
        )
    sidecars = {"metadata_json", "last_frame"} & fields
    if sidecars and sidecars != {"metadata_json", "last_frame"}:
        raise ValueError("Semantic-map metadata_json and last_frame must appear together")


def _validated_runtime_sidecars(
    archive: np.lib.npyio.NpzFile,
) -> dict[str, Any] | None:
    """Validate and return the tiny structural header, never free-form scene text."""

    _validate_archive_field_names(archive)
    if "metadata_json" not in archive.files:
        return None
    header = _parse_metadata_json(archive["metadata_json"])
    declared_count = _strict_positive_int(header.get("occupied_voxels"), field="occupied_voxels")
    declared_feature_dim = _strict_positive_int(header.get("feature_dim"), field="feature_dim")
    dtype_name = header.get("semantic_dtype_on_disk")
    if not isinstance(dtype_name, str):
        raise TypeError("Semantic-map semantic_dtype_on_disk must be a fixed dtype name")
    try:
        semantic_dtype = np.dtype(dtype_name)
    except TypeError as exc:
        raise ValueError("Semantic-map semantic_dtype_on_disk is unsupported") from exc
    if semantic_dtype.kind != "f":
        raise ValueError("Semantic-map semantic_dtype_on_disk must be floating point")
    declared_observations = _strict_positive_int(
        header.get("total_observations"), field="total_observations"
    )
    _validate_last_frame(archive["last_frame"], voxel_count=declared_count)
    _validate_map_header(
        header,
        voxel_count=declared_count,
        feature_dim=declared_feature_dim,
        semantic_dtype=semantic_dtype,
        total_observations=declared_observations,
    )
    return header


def validate_runtime_map_sidecars(path: str | Path) -> None:
    """Check map string sidecars without loading the high-dimensional feature matrix.

    This is useful for cheap runtime preflight and tests. ``load_map_tensors`` repeats
    the same validation and additionally binds the declared values to the numeric arrays.
    """

    with np.load(Path(path), allow_pickle=False) as archive:
        _validated_runtime_sidecars(archive)


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


def _aggregate_rows(
    values: np.ndarray, inverse: np.ndarray, groups: int, weights: np.ndarray
) -> np.ndarray:
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
        header = _validated_runtime_sidecars(archive)
        missing = set(_REQUIRED_NUMERIC_FIELDS) - set(archive.files)
        if missing:
            raise ValueError(f"Map is missing fields: {sorted(missing)}")
        arrays = {name: archive[name].copy() for name in _REQUIRED_NUMERIC_FIELDS}
        semantic_on_disk = arrays["semantic_features"]
        if semantic_on_disk.dtype.kind != "f" or semantic_on_disk.ndim != 2:
            raise ValueError("semantic_features must be a floating-point [N, D] matrix")
        count, feature_dim = semantic_on_disk.shape
        expected_shapes = {
            "centers_world": (count, 3),
            "mean_rgb": (count, 3),
            "normal": (count, 3),
            "confidence": (count,),
            "observation_count": (count,),
        }
        for name, shape in expected_shapes.items():
            values = arrays[name]
            if values.dtype.kind not in {"f", "i", "u"} or values.shape != shape:
                raise ValueError(f"Semantic-map field {name} must be numeric with shape {shape}")
        if arrays["observation_count"].dtype.kind not in {"i", "u"}:
            raise ValueError("Semantic-map observation_count must contain integers")
        if np.any(arrays["observation_count"] < 0):
            raise ValueError("Semantic-map observation_count cannot be negative")
        _validate_optional_numeric_fields(archive, voxel_count=count)
        if header is not None:
            _validate_map_header(
                header,
                voxel_count=count,
                feature_dim=feature_dim,
                semantic_dtype=semantic_on_disk.dtype,
                total_observations=int(
                    arrays["observation_count"].astype(np.int64, copy=False).sum()
                ),
            )
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
