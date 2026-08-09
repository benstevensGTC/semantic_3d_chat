"""Exact metric-depth projection in the project's documented coordinate frames.

Camera-space points use the conventional computer-vision axes ``+X`` right,
``+Y`` down, and ``+Z`` forward.  World-space points use ``+X`` right, ``+Y``
forward, and ``+Z`` up.  Consequently, the orientation of those camera axes in
the world is carried entirely by the supplied camera-to-world transform.

Depth values are interpreted as *projective Z depth* in metres (the camera-space
Z coordinate), not Euclidean ray length.  This is the convention used by the
sanitized rendering manifests and makes the pinhole back-projection exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semantic_3d_chat.coordinates import validate_transform


@dataclass(frozen=True)
class DepthProjection:
    """A depth image projected into camera and canonical world coordinates.

    ``pixels_uv`` stores integer ``(u, v)`` pixel coordinates, so it can index
    RGB and feature fields without an ambiguous row/column conversion.
    """

    points_camera: np.ndarray
    points_world: np.ndarray
    pixels_uv: np.ndarray
    depth_m: np.ndarray
    view_directions_world: np.ndarray

    def __post_init__(self) -> None:
        count = self.points_camera.shape[0]
        expected = {
            "points_camera": (count, 3),
            "points_world": (count, 3),
            "pixels_uv": (count, 2),
            "depth_m": (count,),
            "view_directions_world": (count, 3),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {getattr(self, name).shape}")

    def __len__(self) -> int:
        return self.points_world.shape[0]


def validate_intrinsics(intrinsics: np.ndarray) -> np.ndarray:
    """Validate and return a float64 3x3 pinhole intrinsic matrix."""

    matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Intrinsics must be 3x3, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("Intrinsics must contain only finite values")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("Intrinsics focal lengths fx and fy must be positive")
    if not np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("Intrinsics must have homogeneous bottom row [0, 0, 1]")
    if abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("Intrinsics matrix is singular")
    return matrix


def depth_to_camera_points(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    *,
    min_depth_m: float = 0.0,
    max_depth_m: float = np.inf,
    pixel_stride: int = 1,
    valid_mask: np.ndarray | None = None,
    output_dtype: np.dtype | type = np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Back-project valid sampled pixels into CV camera coordinates.

    Returns ``(points_camera, pixels_uv, sampled_depth_m)``.  NaN, infinity,
    non-positive depth, and values outside the configured interval are omitted.
    Pixels are sampled from the deterministic lattice beginning at ``(0, 0)``.
    """

    depth = np.asarray(depth_m)
    if depth.ndim != 2:
        raise ValueError(f"Depth must have shape [H, W], got {depth.shape}")
    if not isinstance(pixel_stride, (int, np.integer)) or pixel_stride < 1:
        raise ValueError("pixel_stride must be a positive integer")
    if not np.isfinite(min_depth_m) or min_depth_m < 0:
        raise ValueError("min_depth_m must be finite and non-negative")
    if np.isnan(max_depth_m) or max_depth_m <= min_depth_m:
        raise ValueError("max_depth_m must be greater than min_depth_m")

    matrix = validate_intrinsics(intrinsics)
    height, width = depth.shape
    rows = np.arange(0, height, pixel_stride, dtype=np.int32)
    columns = np.arange(0, width, pixel_stride, dtype=np.int32)
    grid_u, grid_v = np.meshgrid(columns, rows, indexing="xy")
    sampled_depth = depth[grid_v, grid_u].astype(np.float64, copy=False)

    valid = np.isfinite(sampled_depth)
    valid &= sampled_depth > 0.0
    valid &= sampled_depth >= min_depth_m
    valid &= sampled_depth <= max_depth_m
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != depth.shape:
            raise ValueError(f"valid_mask must have shape {depth.shape}, got {mask.shape}")
        valid &= mask[grid_v, grid_u]

    selected_u = grid_u[valid]
    selected_v = grid_v[valid]
    selected_depth = sampled_depth[valid]
    if selected_depth.size == 0:
        return (
            np.empty((0, 3), dtype=output_dtype),
            np.empty((0, 2), dtype=np.int32),
            np.empty((0,), dtype=output_dtype),
        )

    homogeneous_pixels = np.stack(
        [selected_u, selected_v, np.ones_like(selected_u)], axis=1
    ).astype(np.float64)
    rays = np.linalg.solve(matrix, homogeneous_pixels.T).T
    if not np.allclose(rays[:, 2], 1.0, atol=1e-7):
        raise ValueError("Intrinsics do not produce unit-Z pinhole rays")
    points_camera = rays * selected_depth[:, None]
    pixels_uv = np.stack([selected_u, selected_v], axis=1).astype(np.int32, copy=False)
    return (
        points_camera.astype(output_dtype, copy=False),
        pixels_uv,
        selected_depth.astype(output_dtype, copy=False),
    )


def transform_camera_points(
    points_camera: np.ndarray,
    camera_to_world: np.ndarray,
    *,
    output_dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Transform CV camera-space points to canonical world coordinates."""

    points = np.asarray(points_camera)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Camera points must have shape [N, 3], got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("Camera points must be finite")
    transform = np.asarray(camera_to_world, dtype=np.float64)
    validate_transform(transform)
    if points.shape[0] == 0:
        return np.empty((0, 3), dtype=output_dtype)
    points_world = points.astype(np.float64) @ transform[:3, :3].T + transform[:3, 3]
    return points_world.astype(output_dtype, copy=False)


def project_depth_to_world(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    *,
    min_depth_m: float = 0.0,
    max_depth_m: float = np.inf,
    pixel_stride: int = 1,
    valid_mask: np.ndarray | None = None,
    output_dtype: np.dtype | type = np.float32,
) -> DepthProjection:
    """Back-project one metric depth image through an exact camera pose."""

    transform = np.asarray(camera_to_world, dtype=np.float64)
    validate_transform(transform)
    points_camera, pixels_uv, selected_depth = depth_to_camera_points(
        depth_m,
        intrinsics,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        pixel_stride=pixel_stride,
        valid_mask=valid_mask,
        output_dtype=output_dtype,
    )
    points_world = transform_camera_points(points_camera, transform, output_dtype=output_dtype)
    if len(points_world):
        offsets = points_world.astype(np.float64) - transform[:3, 3]
        norms = np.linalg.norm(offsets, axis=1, keepdims=True)
        if np.any(norms <= 0) or not np.isfinite(norms).all():
            raise ValueError("Projected points produced invalid world-space view directions")
        view_directions = (offsets / norms).astype(output_dtype)
    else:
        view_directions = np.empty((0, 3), dtype=output_dtype)
    return DepthProjection(
        points_camera=points_camera,
        points_world=points_world,
        pixels_uv=pixels_uv,
        depth_m=selected_depth,
        view_directions_world=view_directions,
    )


def project_world_points_to_pixels(
    points_world: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points back to fractional ``(u, v)`` and camera Z depth."""

    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("World points must be a finite array with shape [N, 3]")
    matrix = validate_intrinsics(intrinsics)
    transform = np.asarray(camera_to_world, dtype=np.float64)
    validate_transform(transform)
    points_camera = (points - transform[:3, 3]) @ transform[:3, :3]
    depth = points_camera[:, 2]
    pixels_h = points_camera @ matrix.T
    pixels_uv = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    in_front = depth > 0
    pixels_uv[in_front] = pixels_h[in_front, :2] / pixels_h[in_front, 2:3]
    return pixels_uv, depth
