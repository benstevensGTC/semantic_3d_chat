"""Leakage-safe embodied RGB-D to persistent semantic-map updates.

The simulator already turns an arbitrary numeric camera pose into a sanitized
``NumericObservation``.  This module completes the missing production seam:
one complete RGB image is encoded exactly once, its localized Gemma patch grid
is projected through the observation's exact depth and pose, and a copy-on-write
voxel map is atomically promoted only after round-trip verification.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np

from semantic_3d_chat.config import project_path
from semantic_3d_chat.mapping.fusion import fuse_frame
from semantic_3d_chat.mapping.voxel_map import (
    SparseVoxelMap,
    persisted_voxel_map_content_hash,
)
from semantic_3d_chat.robot.simulator import (
    EmbodiedCameraSimulator,
    NumericObservation,
)
from semantic_3d_chat.scene_encoder.map_io import RUNTIME_MAP_FIELDS
from semantic_3d_chat.vision.encoder import (
    DenseImageEncoder,
    load_configured_dense_image_encoder,
)

_SCENE_ID: Final[re.Pattern[str]] = re.compile(r"scene_[0-9]{6}")
_OBSERVATION_ID: Final[re.Pattern[str]] = re.compile(r"o_[0-9]{6}")
_FRAME_ID: Final[re.Pattern[str]] = re.compile(r"(?:f|frame|o)_[0-9a-f]{6,64}")
_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_PERSISTENT_SCHEMA: Final[str] = "semantic_3d_chat.embodied_map.v1"
_PROTECTED_MAP_COMPONENTS: Final[frozenset[str]] = frozenset(
    {
        "experiments",
        "features",
        "oracle",
        "predictions",
        "qa",
        "questions",
        "rendered",
        "scorer",
        "scorer_only",
        "scorer-only",
        "training",
    }
)
_HEADER_KEYS: Final[frozenset[str]] = frozenset(
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
_PERSISTENT_METADATA_KEYS: Final[frozenset[str]] = frozenset(
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


def _safe_numeric_path(path: str | Path, *, purpose: str) -> Path:
    raw = Path(path).expanduser()
    unresolved = Path(os.path.abspath(raw))
    current = Path(unresolved.anchor)
    for component in unresolved.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{purpose} cannot use symbolic-link path components")
    if _PROTECTED_MAP_COMPONENTS.intersection(
        part.casefold() for part in unresolved.parts
    ):
        raise ValueError(
            f"{purpose} cannot use an oracle or QA path or another protected runtime path"
        )
    return unresolved


def _json_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a JSON object")
    return value


def _archive_metadata(
    path: Path,
    *,
    scene_id: str,
    persistent: bool,
) -> Mapping[str, Any]:
    """Validate that a map archive exposes only the numeric runtime schema."""

    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Sanitized semantic map is unavailable: {path}")
    with np.load(path, allow_pickle=False) as archive:
        fields = set(archive.files)
        if fields != set(RUNTIME_MAP_FIELDS):
            raise ValueError("Semantic map fields differ from the numeric runtime allowlist")
        raw_header = archive["metadata_json"].item()
        last_frames = archive["last_frame"].astype(str)
    if any(_FRAME_ID.fullmatch(value) is None for value in last_frames):
        raise ValueError("Semantic map contains a non-opaque frame identifier")
    if not isinstance(raw_header, str):
        raise TypeError("Semantic map metadata must be a JSON string")
    header = _json_mapping(json.loads(raw_header), field="semantic map header")
    if set(header) != set(_HEADER_KEYS):
        raise ValueError("Semantic map header fields differ from the numeric schema")
    metadata = _json_mapping(header.get("metadata"), field="semantic map metadata")
    if persistent:
        if set(metadata) != set(_PERSISTENT_METADATA_KEYS):
            raise ValueError("Persistent semantic-map receipt fields changed")
        if metadata.get("schema") != _PERSISTENT_SCHEMA:
            raise ValueError("Unsupported persistent semantic-map schema")
        version = metadata.get("map_version")
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ValueError("Persistent semantic-map version must be positive")
        for field in ("map_sha256", "prior_map_sha256"):
            value = metadata.get(field)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"Persistent semantic-map {field} is invalid")
        observation_id = metadata.get("observation_id")
        if not isinstance(observation_id, str) or _OBSERVATION_ID.fullmatch(observation_id) is None:
            raise ValueError("Persistent semantic-map observation ID is invalid")
        if metadata.get("vision_encoder_calls") != 1:
            raise ValueError("Persistent semantic-map update was not one full-image call")
    elif set(metadata) != {"scene_id"}:
        raise ValueError("Base semantic map contains non-runtime metadata")
    if metadata.get("scene_id") != scene_id:
        raise ValueError("Semantic map opaque scene ID mismatch")
    return metadata


def semantic_map_content_hash(path: str | Path) -> str:
    """Hash the exact persisted numeric arrays, excluding JSON metadata.

    This matches :meth:`SparseVoxelMap.content_hash` immediately before save.
    Hashing stored arrays directly avoids introducing a false mismatch from
    reconstructing weighted accumulators that are intentionally not serialized.
    """

    return persisted_voxel_map_content_hash(path)


def _validate_observation(observation: NumericObservation) -> None:
    if (
        not isinstance(observation.observation_id, str)
        or _OBSERVATION_ID.fullmatch(observation.observation_id) is None
    ):
        raise ValueError("Observation ID must be opaque")
    rgb = np.asarray(observation.rgb)
    depth = np.asarray(observation.depth_m)
    intrinsics = np.asarray(observation.intrinsics)
    camera_to_world = np.asarray(observation.camera_to_world)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Observation RGB must be uint8 [H, W, 3]")
    if depth.shape != rgb.shape[:2] or not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError("Observation depth must be finite nonnegative [H, W]")
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("Observation intrinsics must be finite [3, 3]")
    if camera_to_world.shape != (4, 4) or not np.isfinite(camera_to_world).all():
        raise ValueError("Observation camera_to_world must be finite [4, 4]")


@dataclass(frozen=True)
class SemanticMapUpdateReceipt:
    """Numeric and opaque result of one committed embodied observation."""

    scene_id: str
    observation_id: str
    map_version: int
    prior_map_sha256: str
    map_sha256: str
    vision_encoder_calls: int
    feature_grid_height: int
    feature_grid_width: int
    feature_dim: int
    valid_depth_points: int
    newly_occupied_voxels: int
    total_occupied_voxels: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "scene_id": self.scene_id,
            "observation_id": self.observation_id,
            "map_version": self.map_version,
            "prior_map_sha256": self.prior_map_sha256,
            "map_sha256": self.map_sha256,
            "vision_encoder_calls": self.vision_encoder_calls,
            "feature_grid_height": self.feature_grid_height,
            "feature_grid_width": self.feature_grid_width,
            "feature_dim": self.feature_dim,
            "valid_depth_points": self.valid_depth_points,
            "newly_occupied_voxels": self.newly_occupied_voxels,
            "total_occupied_voxels": self.total_occupied_voxels,
        }


class SemanticMapCommitParticipant(Protocol):
    """Prepare an in-memory consumer before a numeric map becomes current."""

    def prepare_map_commit(
        self,
        staged_map_path: Path,
        receipt: Mapping[str, Any],
    ) -> object: ...

    def commit_map(self, prepared: object) -> None: ...

    def rollback_map(self, prepared: object) -> None: ...

    def prepare_base_reset(self) -> object: ...

    def commit_base_reset(self, prepared: object) -> None: ...

    def rollback_base_reset(self, prepared: object) -> None: ...


def persistent_semantic_map_identity(
    path: str | Path,
    scene_id: str,
) -> dict[str, int | str]:
    """Authenticate and return an opaque numeric persistent-map identity."""

    source = _safe_numeric_path(path, purpose="persistent semantic map")
    metadata = _archive_metadata(source, scene_id=scene_id, persistent=True)
    observed_hash = semantic_map_content_hash(source)
    if metadata.get("map_sha256") != observed_hash:
        raise ValueError("Persistent semantic-map receipt does not match numeric content")
    return {
        "scene_id": scene_id,
        "map_version": int(metadata["map_version"]),
        "map_sha256": observed_hash,
    }


class PersistentSemanticMapUpdater:
    """Copy-on-write full-image semantic fusion for one opaque scene."""

    def __init__(
        self,
        config: Mapping[str, Any],
        scene_id: str,
        encoder: DenseImageEncoder,
        *,
        base_map_path: str | Path | None = None,
        persistent_map_path: str | Path | None = None,
        commit_participant: SemanticMapCommitParticipant | None = None,
    ) -> None:
        if not isinstance(scene_id, str) or _SCENE_ID.fullmatch(scene_id) is None:
            raise ValueError("scene_id must be opaque")
        self.config = config
        self.scene_id = scene_id
        self.encoder = encoder
        self.commit_participant = commit_participant
        self.base_map_path = _safe_numeric_path(
            base_map_path
            or project_path(dict(config), "maps", scene_id, "voxel_map.npz"),
            purpose="base semantic map",
        )
        self.persistent_map_path = _safe_numeric_path(
            persistent_map_path
            or project_path(dict(config), "robot", scene_id, "semantic_map.npz"),
            purpose="persistent semantic map",
        )
        if self.base_map_path == self.persistent_map_path:
            raise ValueError("Embodied updates cannot overwrite the static reference map")
        _archive_metadata(self.base_map_path, scene_id=scene_id, persistent=False)
        self.persistent_map_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.persistent_map_path.with_suffix(
            f"{self.persistent_map_path.suffix}.lock"
        )
        self._current_version = 0
        self._current_hash: str | None = None
        if self.persistent_map_path.exists():
            metadata = _archive_metadata(
                self.persistent_map_path, scene_id=scene_id, persistent=True
            )
            self._current_version = int(metadata["map_version"])
            self._current_hash = str(metadata["map_sha256"])

    @property
    def current_version(self) -> int:
        return self._current_version

    @property
    def current_map_sha256(self) -> str | None:
        return self._current_hash

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _committed_state(self) -> tuple[Path, int, str | None]:
        if not self.persistent_map_path.exists():
            return self.base_map_path, 0, None
        metadata = _archive_metadata(
            self.persistent_map_path,
            scene_id=self.scene_id,
            persistent=True,
        )
        return (
            self.persistent_map_path,
            int(metadata["map_version"]),
            str(metadata["map_sha256"]),
        )

    def update(
        self,
        observation: NumericObservation,
        map_version: int,
    ) -> Mapping[str, Any]:
        """Encode, fuse, verify, and atomically commit one observation."""

        _validate_observation(observation)
        if isinstance(map_version, bool) or not isinstance(map_version, int):
            raise TypeError("map_version must be an integer")
        with self._exclusive_lock():
            source, current_version, recorded_hash = self._committed_state()
            if map_version != current_version + 1:
                raise ValueError("Map version is not the next committed version")

            # Exactly one encoder call receives the complete RGB array.  Patch
            # interpolation happens only later inside ``fuse_frame``.
            dense = self.encoder.encode_image(observation.rgb)
            spatial = dense.spatial_features.detach().cpu().numpy()
            grid_height, grid_width, feature_dim = (int(value) for value in spatial.shape)

            prior_hash = semantic_map_content_hash(source)
            if recorded_hash is not None and prior_hash != recorded_hash:
                raise ValueError("Committed semantic-map content hash changed")
            candidate = SparseVoxelMap.load(source)
            if candidate.feature_dim != feature_dim:
                raise ValueError(
                    "Full-image feature dimension differs from the persistent map"
                )
            mapping = self.config["mapping"]
            stats = fuse_frame(
                candidate,
                depth_m=observation.depth_m,
                rgb=observation.rgb,
                spatial_features=spatial,
                intrinsics=observation.intrinsics,
                camera_to_world=observation.camera_to_world,
                frame_id=observation.observation_id,
                min_depth_m=float(mapping["depth_min_m"]),
                max_depth_m=float(mapping["depth_max_m"]),
                pixel_stride=int(mapping["pixel_stride"]),
                confidence_distance_scale_m=float(
                    mapping.get("confidence_distance_scale_m", 6.0)
                ),
            )
            map_hash = candidate.content_hash()
            receipt = SemanticMapUpdateReceipt(
                scene_id=self.scene_id,
                observation_id=observation.observation_id,
                map_version=map_version,
                prior_map_sha256=prior_hash,
                map_sha256=map_hash,
                vision_encoder_calls=1,
                feature_grid_height=grid_height,
                feature_grid_width=grid_width,
                feature_dim=feature_dim,
                valid_depth_points=stats.valid_depth_points,
                newly_occupied_voxels=stats.newly_occupied_voxels,
                total_occupied_voxels=stats.total_occupied_voxels,
            )
            staging = self.persistent_map_path.with_name(
                f".{self.persistent_map_path.name}.{os.getpid()}.pending.npz"
            )
            if staging.exists():
                raise FileExistsError("A pending semantic-map transaction already exists")
            metadata = {
                "schema": _PERSISTENT_SCHEMA,
                "scene_id": self.scene_id,
                "map_version": map_version,
                "map_sha256": map_hash,
                "prior_map_sha256": prior_hash,
                "observation_id": observation.observation_id,
                "vision_encoder_calls": 1,
                "feature_grid_height": grid_height,
                "feature_grid_width": grid_width,
                "feature_dim": feature_dim,
            }
            try:
                candidate.save(staging, metadata=metadata)
                staged_metadata = _archive_metadata(
                    staging, scene_id=self.scene_id, persistent=True
                )
                if staged_metadata.get("map_sha256") != map_hash:
                    raise ValueError("Staged semantic-map receipt hash changed")
                if semantic_map_content_hash(staging) != map_hash:
                    raise ValueError("Staged semantic-map numeric content hash changed")
                prepared: object | None = None
                if self.commit_participant is not None:
                    prepared = self.commit_participant.prepare_map_commit(
                        staging,
                        receipt.as_dict(),
                    )
                backup = self.persistent_map_path.with_name(
                    f".{self.persistent_map_path.name}.{os.getpid()}.rollback.npz"
                )
                if backup.exists():
                    raise FileExistsError("A semantic-map rollback artifact already exists")
                had_persistent_map = self.persistent_map_path.exists()
                if had_persistent_map:
                    os.link(self.persistent_map_path, backup)
                os.replace(staging, self.persistent_map_path)
                try:
                    if self.commit_participant is not None:
                        assert prepared is not None
                        self.commit_participant.commit_map(prepared)
                    directory_fd = os.open(self.persistent_map_path.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                except BaseException:
                    if had_persistent_map:
                        os.replace(backup, self.persistent_map_path)
                    else:
                        self.persistent_map_path.unlink(missing_ok=True)
                    if self.commit_participant is not None and prepared is not None:
                        self.commit_participant.rollback_map(prepared)
                    raise
                finally:
                    backup.unlink(missing_ok=True)
            finally:
                staging.unlink(missing_ok=True)
            self._current_version = map_version
            self._current_hash = map_hash
            return receipt.as_dict()

    def reset_to_base(self) -> Mapping[str, Any]:
        """Atomically remove embodied observations and restore static memory."""

        with self._exclusive_lock():
            _source, prior_version, prior_hash = self._committed_state()
            base_hash = semantic_map_content_hash(self.base_map_path)
            prepared: object | None = None
            if self.commit_participant is not None:
                prepared = self.commit_participant.prepare_base_reset()
            backup = self.persistent_map_path.with_name(
                f".{self.persistent_map_path.name}.{os.getpid()}.reset-rollback.npz"
            )
            if backup.exists():
                raise FileExistsError("A semantic-map reset rollback artifact already exists")
            had_persistent_map = self.persistent_map_path.exists()
            try:
                if had_persistent_map:
                    os.link(self.persistent_map_path, backup)
                    self.persistent_map_path.unlink()
                try:
                    if self.commit_participant is not None:
                        assert prepared is not None
                        self.commit_participant.commit_base_reset(prepared)
                    directory_fd = os.open(self.persistent_map_path.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                except BaseException:
                    if had_persistent_map:
                        os.replace(backup, self.persistent_map_path)
                    if self.commit_participant is not None and prepared is not None:
                        self.commit_participant.rollback_base_reset(prepared)
                    raise
            finally:
                backup.unlink(missing_ok=True)
            self._current_version = 0
            self._current_hash = None
            return {
                "scene_id": self.scene_id,
                "prior_map_version": prior_version,
                "prior_map_sha256": prior_hash,
                "map_version": 0,
                "map_sha256": base_hash,
                "persistent_map_removed": had_persistent_map,
            }


@dataclass(frozen=True)
class SemanticEmbodiedRuntime:
    simulator: EmbodiedCameraSimulator
    map_updater: PersistentSemanticMapUpdater


def build_semantic_embodied_runtime(
    config: dict[str, Any],
    scene_id: str,
    *,
    encoder: DenseImageEncoder | None = None,
    observation_scanner: Any | None = None,
    persistent_map_path: str | Path | None = None,
    local_files_only: bool = True,
) -> SemanticEmbodiedRuntime:
    """Build the runnable pose→RGB-D→Gemma→persistent-map vertical slice."""

    active_encoder = encoder or load_configured_dense_image_encoder(
        config,
        local_files_only=local_files_only,
    )
    updater = PersistentSemanticMapUpdater(
        config,
        scene_id,
        active_encoder,
        persistent_map_path=persistent_map_path,
    )
    simulator = EmbodiedCameraSimulator(
        config,
        scene_id,
        map_update_hook=updater.update,
    )
    if observation_scanner is not None:
        simulator.scanner = observation_scanner
    simulator.state.scene_version = updater.current_version
    simulator.state.scan_count = updater.current_version
    simulator.state.map_sha256 = updater.current_map_sha256
    return SemanticEmbodiedRuntime(simulator=simulator, map_updater=updater)


__all__ = [
    "PersistentSemanticMapUpdater",
    "SemanticEmbodiedRuntime",
    "SemanticMapCommitParticipant",
    "SemanticMapUpdateReceipt",
    "build_semantic_embodied_runtime",
    "persistent_semantic_map_identity",
    "semantic_map_content_hash",
]
