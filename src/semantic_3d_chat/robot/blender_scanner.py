"""Arbitrary-pose RGB-D scanner over a sanitized runtime Blender asset."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
from PIL import Image

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.robot.simulator import NumericObservation

_SCENE_ID: Final[re.Pattern[str]] = re.compile(r"scene_[0-9]{6}")
_OBSERVATION_ID: Final[re.Pattern[str]] = re.compile(r"o_[0-9]{6}")
_ASSET_FILE: Final[re.Pattern[str]] = re.compile(r"[a-z]_[0-9]{6}\.blend")
_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_ASSET_MANIFEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "scene_id",
        "asset_file",
        "asset_sha256",
        "object_names_opaque",
        "nested_names_opaque",
        "custom_properties_present",
        "external_assets_present",
        "automation_present",
        "animation_present",
        "unsupported_datablocks_present",
        "strict_nested_datablock_audit_passed",
        "mesh_objects",
        "light_objects",
        "materials",
        "collections",
        "node_trees",
    }
)
_OBSERVATION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "scene_id",
        "observation_id",
        "rgb_path",
        "depth_path",
        "intrinsics",
        "camera_to_world",
        "width",
        "height",
        "valid_depth_pixels",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path(path: str | Path, *, purpose: str, must_exist: bool) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    unresolved = Path(os.path.abspath(rooted))
    current = Path(unresolved.anchor)
    for component in unresolved.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{purpose} cannot use a symbolic-link path")
    if {"oracle", "qa"} & {part.casefold() for part in unresolved.parts}:
        raise ValueError(f"{purpose} cannot use an oracle or QA path")
    if must_exist and not unresolved.is_file():
        raise FileNotFoundError(f"{purpose} is unavailable: {unresolved}")
    return unresolved


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a JSON object")
    return value


def _expected_rotation(yaw_degrees: float, pitch_degrees: float) -> np.ndarray:
    yaw = math.radians(yaw_degrees)
    pitch = math.radians(pitch_degrees)
    right = np.asarray([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
    forward = np.asarray(
        [
            -math.sin(yaw) * math.cos(pitch),
            math.cos(yaw) * math.cos(pitch),
            math.sin(pitch),
        ],
        dtype=np.float64,
    )
    down = np.cross(forward, right)
    return np.stack((right, down, forward), axis=1)


class SanitizedBlenderScanner:
    """Render complete deterministic observations without opening oracle data."""

    def __init__(
        self,
        scene_id: str,
        asset_path: str | Path,
        *,
        resolution: tuple[int, int],
        horizontal_fov_degrees: float,
        engine: str,
        samples: int,
        max_depth_m: float,
        output_directory: str | Path,
        blender_executable: str | Path | None = None,
        command_runner: Any = subprocess.run,
    ) -> None:
        if not isinstance(scene_id, str) or _SCENE_ID.fullmatch(scene_id) is None:
            raise ValueError("scene_id must be opaque")
        self.scene_id = scene_id
        self.asset_path = _safe_path(
            asset_path,
            purpose="sanitized runtime scene asset",
            must_exist=True,
        )
        if _ASSET_FILE.fullmatch(self.asset_path.name) is None:
            raise ValueError("Sanitized runtime scene asset filename must be opaque")
        manifest_path = _safe_path(
            self.asset_path.with_suffix(".json"),
            purpose="sanitized runtime scene manifest",
            must_exist=True,
        )
        manifest = _mapping(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            field="runtime scene manifest",
        )
        if set(manifest) != set(_ASSET_MANIFEST_KEYS):
            raise ValueError("Runtime scene manifest fields changed")
        if (
            manifest.get("schema") != "semantic_3d_chat.runtime_scene.v2"
            or manifest.get("scene_id") != scene_id
            or manifest.get("asset_file") != self.asset_path.name
            or manifest.get("object_names_opaque") is not True
            or manifest.get("nested_names_opaque") is not True
            or manifest.get("strict_nested_datablock_audit_passed") is not True
            or any(
                manifest.get(field) is not False
                for field in (
                    "custom_properties_present",
                    "external_assets_present",
                    "automation_present",
                    "animation_present",
                    "unsupported_datablocks_present",
                )
            )
        ):
            raise ValueError("Runtime scene manifest is not a strict v2 attestation")
        expected_asset_hash = manifest.get("asset_sha256")
        if (
            not isinstance(expected_asset_hash, str)
            or _SHA256.fullmatch(expected_asset_hash) is None
            or _sha256(self.asset_path) != expected_asset_hash
        ):
            raise ValueError("Runtime scene asset hash differs from its manifest")
        for field in ("mesh_objects", "light_objects", "materials", "collections", "node_trees"):
            value = manifest.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Runtime scene manifest contains an invalid numeric count")
        if int(manifest["mesh_objects"]) < 1:
            raise ValueError("Runtime scene manifest contains no mesh geometry")
        self.asset_sha256 = expected_asset_hash
        width, height = (int(value) for value in resolution)
        if width < 2 or height < 2:
            raise ValueError("Runtime render resolution must be at least 2x2")
        if not 1.0 <= horizontal_fov_degrees < 179.0:
            raise ValueError("Runtime render horizontal FOV is invalid")
        if samples < 1 or not math.isfinite(max_depth_m) or max_depth_m <= 0:
            raise ValueError("Runtime render samples or max depth is invalid")
        self.width = width
        self.height = height
        self.horizontal_fov_degrees = float(horizontal_fov_degrees)
        self.engine = str(engine)
        self.samples = int(samples)
        self.max_depth_m = float(max_depth_m)
        self._direction_bins: set[tuple[int, int]] = set()
        self._azimuth_bins = 72
        self._elevation_bins = 36
        self.output_directory = _safe_path(
            output_directory,
            purpose="sanitized runtime observation output",
            must_exist=False,
        )
        executable = (
            str(blender_executable)
            if blender_executable is not None
            else shutil.which("blender")
        )
        if not executable:
            raise FileNotFoundError("Blender executable is unavailable")
        self.blender_executable = _safe_path(
            Path(executable).resolve(),
            purpose="Blender executable",
            must_exist=True,
        )
        self.render_script = _safe_path(
            PROJECT_ROOT / "blender" / "render_runtime_observation.py",
            purpose="runtime observation renderer",
            must_exist=True,
        )
        self.command_runner = command_runner

    def _command(
        self,
        observation_id: str,
        camera_position_m: tuple[float, float, float],
        yaw_degrees: float,
        pitch_degrees: float,
    ) -> list[str]:
        return [
            str(self.blender_executable),
            "--background",
            "--disable-autoexec",
            str(self.asset_path),
            "--python-exit-code",
            "1",
            "--python",
            str(self.render_script),
            "--",
            "--scene",
            self.scene_id,
            "--observation",
            observation_id,
            "--asset-sha256",
            self.asset_sha256,
            "--output",
            str(self.output_directory),
            "--x",
            repr(float(camera_position_m[0])),
            "--y",
            repr(float(camera_position_m[1])),
            "--z",
            repr(float(camera_position_m[2])),
            "--yaw",
            repr(float(yaw_degrees)),
            "--pitch",
            repr(float(pitch_degrees)),
            "--width",
            str(self.width),
            "--height",
            str(self.height),
            "--horizontal-fov",
            repr(self.horizontal_fov_degrees),
            "--engine",
            self.engine,
            "--samples",
            str(self.samples),
            "--max-depth",
            repr(self.max_depth_m),
        ]

    def capture(
        self,
        *,
        observation_index: int,
        camera_position_m: tuple[float, float, float],
        yaw_degrees: float,
        pitch_degrees: float,
    ) -> NumericObservation:
        observation_id = f"o_{observation_index:06d}"
        if _OBSERVATION_ID.fullmatch(observation_id) is None:
            raise ValueError("Observation index is outside the opaque ID range")
        pose = (*camera_position_m, yaw_degrees, pitch_degrees)
        if len(camera_position_m) != 3 or not all(math.isfinite(float(value)) for value in pose):
            raise ValueError("Runtime camera pose must contain finite numeric values")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        if _sha256(self.asset_path) != self.asset_sha256:
            raise ValueError("Runtime scene asset changed after scanner construction")
        command = self._command(
            observation_id,
            camera_position_m,
            yaw_degrees,
            pitch_degrees,
        )
        self.command_runner(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        receipt_path = _safe_path(
            self.output_directory / f"{observation_id}.json",
            purpose="runtime observation receipt",
            must_exist=True,
        )
        receipt = _mapping(
            json.loads(receipt_path.read_text(encoding="utf-8")),
            field="runtime observation receipt",
        )
        if set(receipt) != set(_OBSERVATION_KEYS):
            raise ValueError("Runtime observation receipt fields changed")
        if (
            receipt.get("schema") != "semantic_3d_chat.runtime_observation.v1"
            or receipt.get("scene_id") != self.scene_id
            or receipt.get("observation_id") != observation_id
            or receipt.get("width") != self.width
            or receipt.get("height") != self.height
        ):
            raise ValueError("Runtime observation receipt identity changed")
        rgb_name = receipt.get("rgb_path")
        depth_name = receipt.get("depth_path")
        if rgb_name != f"{observation_id}.png":
            raise ValueError("Runtime RGB receipt path must be the opaque observation filename")
        if depth_name != f"{observation_id}.npy":
            raise ValueError("Runtime depth receipt path must be the opaque observation filename")
        rgb_path = _safe_path(
            self.output_directory / rgb_name,
            purpose="runtime RGB observation",
            must_exist=True,
        )
        depth_path = _safe_path(
            self.output_directory / depth_name,
            purpose="runtime depth observation",
            must_exist=True,
        )
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        depth = np.load(depth_path, allow_pickle=False)
        intrinsics = np.asarray(receipt.get("intrinsics"), dtype=np.float64)
        camera_to_world = np.asarray(receipt.get("camera_to_world"), dtype=np.float64)
        valid = int(np.count_nonzero(depth > 0))
        if (
            rgb.shape != (self.height, self.width, 3)
            or depth.shape != (self.height, self.width)
            or depth.dtype != np.float32
            or not np.isfinite(depth).all()
            or np.any(depth < 0)
            or valid < 1
            or receipt.get("valid_depth_pixels") != valid
            or intrinsics.shape != (3, 3)
            or camera_to_world.shape != (4, 4)
            or not np.isfinite(intrinsics).all()
            or not np.isfinite(camera_to_world).all()
            or intrinsics[0, 0] <= 0
            or intrinsics[1, 1] <= 0
            or not np.allclose(intrinsics[2], [0.0, 0.0, 1.0], rtol=0.0, atol=1e-12)
            or not 0.0 <= intrinsics[0, 2] < self.width
            or not 0.0 <= intrinsics[1, 2] < self.height
            or not np.allclose(
                camera_to_world[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-12
            )
        ):
            raise ValueError("Runtime renderer produced an invalid numeric observation")
        if not np.allclose(
            camera_to_world[:3, 3],
            np.asarray(camera_position_m, dtype=np.float64),
            rtol=0.0,
            # Blender stores object transforms as 32-bit floats internally.
            # Authenticate the requested numeric pose within that storage
            # precision instead of requiring impossible float64 bit identity.
            atol=1e-6,
        ):
            raise ValueError("Runtime renderer camera translation differs from numeric pose")
        rotation = camera_to_world[:3, :3]
        if (
            not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-5)
            or not math.isclose(float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1e-5)
            or not np.allclose(
                rotation,
                _expected_rotation(yaw_degrees, pitch_degrees),
                rtol=0.0,
                atol=1e-5,
            )
        ):
            raise ValueError("Runtime renderer camera rotation differs from numeric pose")
        return NumericObservation(
            observation_id=observation_id,
            rgb=rgb,
            depth_m=depth,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
            visible_voxel_indices=np.empty(0, dtype=np.int64),
        )

    def integrate(self, observation: NumericObservation, observed_mask: np.ndarray) -> int:
        del observed_mask
        valid = observation.depth_m > 0
        yy, xx = np.nonzero(valid)
        if len(xx):
            camera_rays = np.stack(
                (
                    (xx - observation.intrinsics[0, 2]) / observation.intrinsics[0, 0],
                    (yy - observation.intrinsics[1, 2]) / observation.intrinsics[1, 1],
                    np.ones(len(xx), dtype=np.float64),
                ),
                axis=1,
            )
            camera_rays /= np.linalg.norm(camera_rays, axis=1, keepdims=True)
            world_rays = camera_rays @ observation.camera_to_world[:3, :3].T
            azimuth = np.arctan2(world_rays[:, 1], world_rays[:, 0])
            elevation = np.arcsin(np.clip(world_rays[:, 2], -1.0, 1.0))
            azimuth_bins = np.floor(
                (azimuth + np.pi) * self._azimuth_bins / (2.0 * np.pi)
            ).astype(np.int64)
            elevation_bins = np.floor(
                (elevation + np.pi / 2.0) * self._elevation_bins / np.pi
            ).astype(np.int64)
            azimuth_bins %= self._azimuth_bins
            elevation_bins = np.clip(elevation_bins, 0, self._elevation_bins - 1)
            self._direction_bins.update(
                zip(azimuth_bins.tolist(), elevation_bins.tolist(), strict=True)
            )
        return len(xx)

    @property
    def directional_coverage(self) -> float:
        """Fraction of fixed world-direction bins observed by valid RGB-D rays."""

        total = self._azimuth_bins * self._elevation_bins
        return float(len(self._direction_bins) / total)

    def reset_episode(self) -> None:
        """Clear pose-coverage state while retaining the authenticated asset."""

        self._direction_bins.clear()

    def snapshot_episode_state(self) -> frozenset[tuple[int, int]]:
        return frozenset(self._direction_bins)

    def restore_episode_state(self, state: frozenset[tuple[int, int]]) -> None:
        self._direction_bins = set(state)


__all__ = ["SanitizedBlenderScanner"]
