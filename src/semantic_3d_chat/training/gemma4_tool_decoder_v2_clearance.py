"""Anonymous numeric clearance cache for Gemma-4 tool-decoder V2.

The cache is training-only and binds each authenticated V3 trace row to a
24-ray robot-frame free-space vector.  It reads only ``centers_world`` from
sanitized voxel maps plus numeric robot pose features.  Semantic features,
object labels, oracle relationships, and scene-generation metadata are never
inputs to the clearance calculation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.scene_encoder.map_io import validate_runtime_map_sidecars

CLEARANCE_RAY_COUNT: Final[int] = 24
CLEARANCE_MAX_RANGE_M: Final[float] = 1.0
COLLISION_PROBE_DISTANCES_M: Final[tuple[float, ...]] = (0.125, 0.25, 0.375, 0.5)
CACHE_SCHEMA: Final[str] = "semantic_3d_chat.gemma4_tool_decoder_clearance.v2"
MANIFEST_SCHEMA: Final[str] = "semantic_3d_chat.gemma4_tool_decoder_clearance_manifest.v2"
TRACE_SCHEMA: Final[str] = "semantic_3d_chat.navigation_target_trace_sample.v3"


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return Path(os.path.abspath(rooted))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _strict_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    scene = config.get("scene")
    robot = config.get("robot")
    experiment = config.get("gemma4_embodied_tool_decoder_v2")
    if not isinstance(scene, Mapping) or not isinstance(robot, Mapping):
        raise TypeError("V2 clearance requires scene and robot mappings")
    if not isinstance(experiment, Mapping):
        raise TypeError("V2 clearance requires its experiment mapping")
    room = np.asarray(scene.get("room_size_m"), dtype=np.float64)
    if room.shape != (3,) or not np.isfinite(room).all() or np.any(room <= 0.0):
        raise ValueError("V2 clearance room_size_m must contain three positive values")
    values = {
        "room_size_m": room.tolist(),
        "robot_radius_m": robot.get("radius_m"),
        "collision_z_min_m": robot.get("collision_z_min_m"),
        "collision_z_max_m": robot.get("collision_z_max_m"),
        "surface_padding_m": robot.get("surface_padding_m"),
        "ray_count": experiment.get("clearance_state_dim"),
    }
    for name in (
        "robot_radius_m",
        "collision_z_min_m",
        "collision_z_max_m",
        "surface_padding_m",
    ):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"V2 clearance {name} must be numeric")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"V2 clearance {name} must be finite and nonnegative")
        values[name] = float(value)
    if values["robot_radius_m"] <= 0.0:
        raise ValueError("V2 clearance robot radius must be positive")
    if not 0.0 <= values["collision_z_min_m"] < values["collision_z_max_m"]:
        raise ValueError("V2 clearance collision height interval is invalid")
    if values["ray_count"] != CLEARANCE_RAY_COUNT:
        raise ValueError("V2 clearance preregisters exactly 24 rays")
    values["max_range_m"] = CLEARANCE_MAX_RANGE_M
    return values


def world_pose_from_state_features(
    state_features: Sequence[float] | torch.Tensor,
    room_size_m: Sequence[float],
) -> tuple[np.ndarray, float]:
    """Invert the numeric robot-state normalization for XY and body yaw."""

    state = torch.as_tensor(state_features, dtype=torch.float32)
    room = torch.as_tensor(room_size_m, dtype=torch.float32)
    if state.shape != (18,) or not torch.isfinite(state).all():
        raise ValueError("V2 clearance state_features must be finite with shape [18]")
    if room.shape != (3,) or not torch.isfinite(room).all() or torch.any(room <= 0.0):
        raise ValueError("V2 clearance room_size_m must contain three positive values")
    minimum = torch.tensor([-room[0] / 2.0, -room[1] / 2.0, 0.0])
    maximum = torch.tensor([room[0] / 2.0, room[1] / 2.0, room[2]])
    position = (state[:3] + 1.0) * 0.5 * (maximum - minimum) + minimum
    yaw = math.degrees(math.atan2(float(state[3]), float(state[4])))
    return position[:2].numpy().astype(np.float64), yaw


class AnonymousNumericClearanceMap:
    """2D collision geometry built only from anonymous world coordinates."""

    def __init__(
        self,
        obstacle_points_xy_m: np.ndarray,
        *,
        room_size_m: Sequence[float],
        robot_radius_m: float,
        surface_padding_m: float,
    ) -> None:
        points = np.asarray(obstacle_points_xy_m, dtype=np.float32)
        room = np.asarray(room_size_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("Anonymous clearance obstacle points must be finite [N,2]")
        if len(points) < 1:
            raise ValueError("Anonymous clearance geometry is empty")
        if room.shape != (3,) or not np.isfinite(room).all() or np.any(room <= 0.0):
            raise ValueError("Anonymous clearance room size is invalid")
        if not math.isfinite(robot_radius_m) or robot_radius_m <= 0.0:
            raise ValueError("Anonymous clearance robot radius must be positive")
        if not math.isfinite(surface_padding_m) or surface_padding_m < 0.0:
            raise ValueError("Anonymous clearance surface padding must be nonnegative")
        self.obstacle_points_xy_m = points
        self.room_min_xy_m = np.asarray((-room[0] / 2.0, -room[1] / 2.0))
        self.room_max_xy_m = np.asarray((room[0] / 2.0, room[1] / 2.0))
        self.robot_radius_m = float(robot_radius_m)
        self.inflated_radius_m = float(robot_radius_m + surface_padding_m)

    @classmethod
    def from_voxel_map(
        cls,
        path: str | Path,
        *,
        room_size_m: Sequence[float],
        robot_radius_m: float,
        collision_z_min_m: float,
        collision_z_max_m: float,
        surface_padding_m: float,
    ) -> AnonymousNumericClearanceMap:
        source = Path(path)
        validate_runtime_map_sidecars(source)
        with np.load(source, allow_pickle=False) as archive:
            if "centers_world" not in archive.files:
                raise ValueError("Sanitized voxel map has no centers_world")
            centers = archive["centers_world"].astype(np.float32)
        if centers.ndim != 2 or centers.shape[1] != 3 or not np.isfinite(centers).all():
            raise ValueError("Sanitized voxel-map centers_world is invalid")
        mask = (centers[:, 2] >= collision_z_min_m) & (
            centers[:, 2] <= collision_z_max_m
        )
        return cls(
            centers[mask, :2],
            room_size_m=room_size_m,
            robot_radius_m=robot_radius_m,
            surface_padding_m=surface_padding_m,
        )

    def collides(self, position_xy_m: np.ndarray) -> bool:
        position = np.asarray(position_xy_m, dtype=np.float64)
        if position.shape != (2,) or not np.isfinite(position).all():
            raise ValueError("Anonymous clearance position must contain two finite values")
        lower = self.room_min_xy_m + self.robot_radius_m
        upper = self.room_max_xy_m - self.robot_radius_m
        if np.any(position < lower) or np.any(position > upper):
            return True
        delta = self.obstacle_points_xy_m.astype(np.float64) - position
        return bool(np.any(np.linalg.norm(delta, axis=1) <= self.inflated_radius_m))

    def ray_clearance(
        self,
        position_xy_m: np.ndarray,
        yaw_degrees: float,
        *,
        max_range_m: float = CLEARANCE_MAX_RANGE_M,
        ray_count: int = CLEARANCE_RAY_COUNT,
        obstacle_chunk_size: int = 4096,
    ) -> torch.Tensor:
        """Compute analytic robot-frame clearance without ray-marching gaps."""

        start = np.asarray(position_xy_m, dtype=np.float64)
        if start.shape != (2,) or not np.isfinite(start).all():
            raise ValueError("Anonymous clearance position must contain two finite values")
        if not math.isfinite(yaw_degrees):
            raise ValueError("Anonymous clearance yaw must be finite")
        if ray_count != CLEARANCE_RAY_COUNT or ray_count % 2:
            raise ValueError("Anonymous clearance requires exactly 24 even rays")
        if not math.isfinite(max_range_m) or max_range_m <= 0.0:
            raise ValueError("Anonymous clearance range must be positive")
        if self.collides(start):
            raise ValueError("Anonymous clearance cannot encode a colliding pose")
        angles = np.deg2rad(float(yaw_degrees)) + np.arange(ray_count) * (
            2.0 * math.pi / ray_count
        )
        directions = np.stack((-np.sin(angles), np.cos(angles)), axis=1)
        lower = self.room_min_xy_m + self.robot_radius_m
        upper = self.room_max_xy_m - self.robot_radius_m
        boundary = np.full(ray_count, max_range_m, dtype=np.float64)
        for axis in range(2):
            component = directions[:, axis]
            positive = component > 1e-12
            negative = component < -1e-12
            boundary[positive] = np.minimum(
                boundary[positive], (upper[axis] - start[axis]) / component[positive]
            )
            boundary[negative] = np.minimum(
                boundary[negative], (lower[axis] - start[axis]) / component[negative]
            )
        free = np.clip(boundary, 0.0, max_range_m)
        radius_squared = self.inflated_radius_m**2
        points = self.obstacle_points_xy_m.astype(np.float64, copy=False)
        for offset in range(0, len(points), obstacle_chunk_size):
            relative = points[offset : offset + obstacle_chunk_size] - start
            projection = relative @ directions.T
            radial_squared = np.sum(relative * relative, axis=1, keepdims=True)
            perpendicular_squared = np.maximum(
                radial_squared - projection * projection, 0.0
            )
            intersects = (projection >= 0.0) & (
                perpendicular_squared <= radius_squared
            )
            root = np.sqrt(np.maximum(radius_squared - perpendicular_squared, 0.0))
            entry = projection - root
            candidates = np.where(intersects & (entry >= 0.0), entry, np.inf)
            free = np.minimum(free, np.min(candidates, axis=0))
        normalized = np.clip(free / max_range_m, 0.0, 1.0).astype(np.float32)
        result = torch.from_numpy(normalized)
        if result.shape != (CLEARANCE_RAY_COUNT,) or not torch.isfinite(result).all():
            raise RuntimeError("Anonymous clearance calculation produced invalid values")
        return result


def collision_probe_targets(clearance_state: torch.Tensor) -> torch.Tensor:
    clearance = torch.as_tensor(clearance_state, dtype=torch.float32)
    if clearance.shape != (CLEARANCE_RAY_COUNT,) or not torch.isfinite(clearance).all():
        raise ValueError("V2 clearance state must have shape [24]")
    if torch.any((clearance < 0.0) | (clearance > 1.0)):
        raise ValueError("V2 clearance state must be normalized")
    probes = torch.tensor(COLLISION_PROBE_DISTANCES_M, dtype=torch.float32)
    forward = clearance[0] * CLEARANCE_MAX_RANGE_M
    backward = clearance[CLEARANCE_RAY_COUNT // 2] * CLEARANCE_MAX_RANGE_M
    return torch.cat(((probes >= forward).float(), (probes >= backward).float()))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def materialize_clearance_cache_v2(
    config: Mapping[str, Any],
    *,
    trace_root: str | Path,
    map_root: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Create the V2 cache once and return its sealed manifest."""

    settings = _strict_settings(config)
    traces = _rooted(trace_root) / "traces.jsonl"
    trace_manifest = _rooted(trace_root) / "manifest.json"
    maps = _rooted(map_root)
    destination = _rooted(output_directory)
    cache_path = destination / "clearance.safetensors"
    manifest_path = destination / "manifest.json"
    if cache_path.exists() or manifest_path.exists():
        raise FileExistsError("V2 clearance cache is create-once and already exists")
    destination.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads(trace_manifest.read_text(encoding="utf-8"))
    expected_count = source_manifest.get("sample_count")
    if expected_count != 6468 or source_manifest.get("schema") != (
        "semantic_3d_chat.navigation_target_trace_dataset.v3"
    ):
        raise ValueError("V2 clearance source trace manifest changed")

    rows: list[dict[str, Any]] = []
    with traces.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            if (
                row.get("schema") != TRACE_SCHEMA
                or row.get("sample_id") != f"g_{index:08d}"
                or row.get("oracle_available_at_runtime") is not False
            ):
                raise ValueError("V2 clearance source trace row contract changed")
            rows.append(row)
    if len(rows) != expected_count:
        raise ValueError("V2 clearance trace row count differs from its manifest")

    scene_ids = list(dict.fromkeys(str(row["scene_id"]) for row in rows))
    allowed_scenes = [
        *source_manifest.get("train_scene_ids", []),
        *source_manifest.get("validation_scene_ids", []),
    ]
    if scene_ids != allowed_scenes or len(scene_ids) != 22:
        raise ValueError("V2 clearance scene ordering differs from the sealed split")
    map_sha256 = {
        scene_id: _sha256(maps / scene_id / "voxel_map.npz") for scene_id in scene_ids
    }
    geometry: dict[str, AnonymousNumericClearanceMap] = {}
    pose_cache: dict[tuple[str, tuple[float, ...]], torch.Tensor] = {}
    states: list[torch.Tensor] = []
    risks: list[torch.Tensor] = []
    sample_bindings: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        scene_id = str(row["scene_id"])
        numeric_state = row.get("state_features")
        if not isinstance(numeric_state, list) or len(numeric_state) != 18:
            raise ValueError("V2 clearance trace row has no numeric state vector")
        state = torch.tensor(numeric_state, dtype=torch.float32)
        key = (scene_id, tuple(float(value) for value in state[:5]))
        clearance = pose_cache.get(key)
        if clearance is None:
            collision_map = geometry.get(scene_id)
            if collision_map is None:
                collision_map = AnonymousNumericClearanceMap.from_voxel_map(
                    maps / scene_id / "voxel_map.npz",
                    room_size_m=settings["room_size_m"],
                    robot_radius_m=settings["robot_radius_m"],
                    collision_z_min_m=settings["collision_z_min_m"],
                    collision_z_max_m=settings["collision_z_max_m"],
                    surface_padding_m=settings["surface_padding_m"],
                )
                geometry[scene_id] = collision_map
            position, yaw = world_pose_from_state_features(
                state, settings["room_size_m"]
            )
            clearance = collision_map.ray_clearance(position, yaw).contiguous()
            pose_cache[key] = clearance
        states.append(clearance)
        risks.append(collision_probe_targets(clearance))
        sample_bindings.append(
            {
                "sample_id": f"g_{index:08d}",
                "scene_id": scene_id,
                "pose_sha256": _canonical_sha256(
                    [float(value) for value in state[:5].tolist()]
                ),
            }
        )
    clearance_states = torch.stack(states).contiguous()
    collision_targets = torch.stack(risks).contiguous()
    if clearance_states.shape != (6468, CLEARANCE_RAY_COUNT):
        raise RuntimeError("V2 clearance tensor shape changed")
    if collision_targets.shape != (6468, 8):
        raise RuntimeError("V2 collision target tensor shape changed")
    if not torch.isfinite(clearance_states).all() or torch.any(
        (clearance_states < 0.0) | (clearance_states > 1.0)
    ):
        raise RuntimeError("V2 clearance cache contains invalid values")

    tensors = {
        "clearance_states": clearance_states,
        "collision_targets": collision_targets,
    }
    metadata = {
        "schema": CACHE_SCHEMA,
        "sample_count": str(len(rows)),
        "clearance_ray_count": str(CLEARANCE_RAY_COUNT),
        "clearance_max_range_m": str(CLEARANCE_MAX_RANGE_M),
        "source_traces_sha256": _sha256(traces),
        "sample_binding_sha256": _canonical_sha256(sample_bindings),
        "map_inventory_sha256": _canonical_sha256(map_sha256),
        "environmental_text_inputs": "[]",
        "oracle_inputs_for_clearance": "false",
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".clearance.", suffix=".safetensors.tmp", dir=destination
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(tensors, temporary, metadata=metadata)
        os.replace(temporary, cache_path)
    finally:
        temporary.unlink(missing_ok=True)
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "status": "materialized_and_hash_locked",
        "cache_file": "clearance.safetensors",
        "cache_sha256": _sha256(cache_path),
        "trace_manifest_sha256": _sha256(trace_manifest),
        "trace_rows_sha256": _sha256(traces),
        "sample_count": len(rows),
        "unique_numeric_pose_count": len(pose_cache),
        "train_scene_ids": source_manifest["train_scene_ids"],
        "validation_scene_ids": source_manifest["validation_scene_ids"],
        "scene_splits_disjoint": True,
        "map_sha256": map_sha256,
        "map_inventory_sha256": _canonical_sha256(map_sha256),
        "sample_binding_sha256": _canonical_sha256(sample_bindings),
        "tensors": {
            "clearance_states": {
                "shape": [len(rows), CLEARANCE_RAY_COUNT],
                "dtype": "torch.float32",
                "sha256": _tensor_sha256(clearance_states),
                "minimum": float(clearance_states.min()),
                "maximum": float(clearance_states.max()),
            },
            "collision_targets": {
                "shape": [len(rows), 8],
                "dtype": "torch.float32",
                "sha256": _tensor_sha256(collision_targets),
                "positive_count": int(collision_targets.sum()),
            },
        },
        "geometry_contract": {
            **settings,
            "collision_probe_distances_m": list(COLLISION_PROBE_DISTANCES_M),
            "ray_zero": "robot_forward",
            "ray_half": "robot_backward",
            "ray_direction_order": "counterclockwise_in_robot_frame",
            "calculation": "analytic_room_and_inflated_point_circle_intersection",
            "map_fields_loaded": ["centers_world"],
            "semantic_features_loaded": False,
            "rgb_loaded": False,
            "object_labels_loaded": False,
            "oracle_loaded_for_clearance": False,
            "environmental_text_inputs": [],
        },
    }
    _atomic_json(manifest_path, manifest)
    manifest["manifest_sha256"] = _sha256(manifest_path)
    return manifest


def load_clearance_cache_v2(
    directory: str | Path,
    *,
    expected_cache_sha256: str,
    expected_manifest_sha256: str,
    expected_trace_rows_sha256: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Load a sealed cache and fail closed on any identity mismatch."""

    source = _rooted(directory)
    cache_path = source / "clearance.safetensors"
    manifest_path = source / "manifest.json"
    if _sha256(cache_path) != expected_cache_sha256:
        raise ValueError("V2 clearance cache SHA-256 changed")
    if _sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("V2 clearance manifest SHA-256 changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("status") != "materialized_and_hash_locked"
        or manifest.get("cache_sha256") != expected_cache_sha256
        or manifest.get("trace_rows_sha256") != expected_trace_rows_sha256
        or manifest.get("sample_count") != 6468
        or manifest.get("scene_splits_disjoint") is not True
    ):
        raise ValueError("V2 clearance manifest contract changed")
    with safe_open(cache_path, framework="pt", device="cpu") as archive:
        metadata = archive.metadata()
        if list(archive.keys()) != ["clearance_states", "collision_targets"]:
            raise ValueError("V2 clearance tensor inventory changed")
        clearance = archive.get_tensor("clearance_states")
        collision = archive.get_tensor("collision_targets")
    if (
        metadata.get("schema") != CACHE_SCHEMA
        or metadata.get("source_traces_sha256") != expected_trace_rows_sha256
        or metadata.get("environmental_text_inputs") != "[]"
        or metadata.get("oracle_inputs_for_clearance") != "false"
    ):
        raise ValueError("V2 clearance safetensors metadata changed")
    if clearance.shape != (6468, 24) or collision.shape != (6468, 8):
        raise ValueError("V2 clearance tensor shapes changed")
    if not torch.isfinite(clearance).all() or torch.any(
        (clearance < 0.0) | (clearance > 1.0)
    ):
        raise ValueError("V2 clearance cache contains invalid numeric values")
    if _tensor_sha256(clearance) != manifest["tensors"]["clearance_states"]["sha256"]:
        raise ValueError("V2 clearance tensor digest changed")
    if _tensor_sha256(collision) != manifest["tensors"]["collision_targets"]["sha256"]:
        raise ValueError("V2 collision-target tensor digest changed")
    return clearance, collision, manifest


__all__ = [
    "CACHE_SCHEMA",
    "CLEARANCE_MAX_RANGE_M",
    "CLEARANCE_RAY_COUNT",
    "COLLISION_PROBE_DISTANCES_M",
    "MANIFEST_SCHEMA",
    "AnonymousNumericClearanceMap",
    "collision_probe_targets",
    "load_clearance_cache_v2",
    "materialize_clearance_cache_v2",
    "world_pose_from_state_features",
]
