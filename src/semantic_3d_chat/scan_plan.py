"""Deterministic, semantic-free camera scan planning and manifest assembly."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

_OPAQUE_SCENE_ID = re.compile(r"scene_[0-9]{6}")
_CONFIG_HASH = re.compile(r"[0-9a-f]{12,64}")


@dataclass(frozen=True)
class ScanPose:
    """One exact camera pose request in canonical scan order."""

    position_m: tuple[float, float, float]
    yaw_degrees: float
    pitch_degrees: float


def _numeric_sequence(value: object, *, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"render.{field} must be a numeric sequence")
    return value


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"render.{field} values must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"render.{field} values must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"render.{field} values must be finite")
    return result


def _position(value: object, *, field: str) -> tuple[float, float, float]:
    items = _numeric_sequence(value, field=field)
    if len(items) != 3:
        raise ValueError(f"render.{field} entries must contain exactly three coordinates")
    return (
        _finite_float(items[0], field=field),
        _finite_float(items[1], field=field),
        _finite_float(items[2], field=field),
    )


def expand_scan_poses(render_config: Mapping[str, Any]) -> tuple[ScanPose, ...]:
    """Expand the primary position and optional extra positions deterministically.

    The legacy single-position order is preserved exactly: pitch is the outer
    loop and yaw is the inner loop.  When ``additional_camera_positions_m`` is
    present, the complete legacy center scan is emitted first, followed by the
    same angular scan at each numeric position in declared order.
    """

    primary = _position(render_config.get("camera_position_m"), field="camera_position_m")
    raw_additional = render_config.get("additional_camera_positions_m", ())
    if raw_additional is None:
        raw_additional = ()
    additional_items = _numeric_sequence(raw_additional, field="additional_camera_positions_m")
    additional = tuple(
        _position(item, field="additional_camera_positions_m") for item in additional_items
    )
    positions = (primary, *additional)
    if len(set(positions)) != len(positions):
        raise ValueError("render camera positions must be unique")

    raw_yaws = _numeric_sequence(render_config.get("yaw_degrees"), field="yaw_degrees")
    raw_pitches = _numeric_sequence(render_config.get("pitch_degrees"), field="pitch_degrees")
    if not raw_yaws or not raw_pitches:
        raise ValueError("render yaw_degrees and pitch_degrees must be nonempty")
    yaws = tuple(_finite_float(value, field="yaw_degrees") for value in raw_yaws)
    pitches = tuple(_finite_float(value, field="pitch_degrees") for value in raw_pitches)

    return tuple(
        ScanPose(position_m=position, yaw_degrees=yaw, pitch_degrees=pitch)
        for position in positions
        for pitch in pitches
        for yaw in yaws
    )


def _numeric_matrix(
    value: object,
    *,
    field: str,
    rows: int,
    columns: int,
) -> list[list[float]]:
    raw_rows = _numeric_sequence(value, field=field)
    if len(raw_rows) != rows:
        raise ValueError(f"{field} must have shape [{rows}, {columns}]")
    matrix: list[list[float]] = []
    for raw_row in raw_rows:
        row = _numeric_sequence(raw_row, field=field)
        if len(row) != columns:
            raise ValueError(f"{field} must have shape [{rows}, {columns}]")
        matrix.append([_finite_float(item, field=field) for item in row])
    return matrix


def build_runtime_frame(
    frame_number: int,
    *,
    intrinsics: object,
    camera_to_world: object,
) -> dict[str, Any]:
    """Build one complete runtime frame containing no environmental semantics."""

    if isinstance(frame_number, bool) or not isinstance(frame_number, int):
        raise TypeError("frame_number must be an integer")
    if frame_number < 0:
        raise ValueError("frame_number must be nonnegative")
    frame_id = f"f_{frame_number:06d}"
    camera_id = f"c_{frame_number:06d}"
    rgb_path = PurePosixPath("rgb") / f"{frame_id}.png"
    depth_path = PurePosixPath("depth") / f"{frame_id}.npy"
    return {
        "frame_id": frame_id,
        "camera_id": camera_id,
        "frame_number": frame_number,
        "rgb_path": rgb_path.as_posix(),
        "depth_path": depth_path.as_posix(),
        "intrinsics": _numeric_matrix(intrinsics, field="intrinsics", rows=3, columns=3),
        "camera_to_world": _numeric_matrix(
            camera_to_world, field="camera_to_world", rows=4, columns=4
        ),
    }


def build_runtime_manifest(
    *,
    scene_id: str,
    config_digest: str,
    width: int,
    height: int,
    horizontal_fov_degrees: float,
    frames: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the strict sanitized scan manifest shared by all scan layouts."""

    if not _OPAQUE_SCENE_ID.fullmatch(scene_id):
        raise ValueError("scene_id must be opaque")
    if not _CONFIG_HASH.fullmatch(config_digest):
        raise ValueError("config_digest must be a lowercase hexadecimal content hash")
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
    ):
        raise TypeError("image dimensions must be integers")
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive integers")
    horizontal_fov = _finite_float(horizontal_fov_degrees, field="horizontal_fov_degrees")
    if not 0.0 < horizontal_fov < 180.0:
        raise ValueError("horizontal_fov_degrees must lie between zero and 180")

    normalized_frames: list[dict[str, Any]] = []
    for frame_number, raw_frame in enumerate(frames):
        expected = build_runtime_frame(
            frame_number,
            intrinsics=raw_frame.get("intrinsics"),
            camera_to_world=raw_frame.get("camera_to_world"),
        )
        if dict(raw_frame) != expected:
            raise ValueError("runtime frame fields or opaque identifiers changed")
        normalized_frames.append(expected)
    if not normalized_frames:
        raise ValueError("runtime manifest must contain at least one frame")

    return {
        "schema_version": 1,
        "scene_id": scene_id,
        "config_hash": config_digest,
        "coordinate_system": {
            "world": "x_right_y_forward_z_up",
            "camera": "x_right_y_down_z_forward",
            "units": "meters",
            "depth": "axial_camera_z",
        },
        "image_size": {"width": width, "height": height},
        "horizontal_fov_degrees": horizontal_fov,
        "frames": normalized_frames,
    }


__all__ = [
    "ScanPose",
    "build_runtime_frame",
    "build_runtime_manifest",
    "expand_scan_poses",
]
