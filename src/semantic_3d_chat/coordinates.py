from __future__ import annotations

import numpy as np

# Blender camera local coordinates are +X right, +Y up, -Z forward.
# The runtime CV convention is +X right, +Y down, +Z forward.
BLENDER_CAMERA_FROM_CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


def camera_intrinsics(width: int, height: int, horizontal_fov_degrees: float) -> np.ndarray:
    fov = np.deg2rad(horizontal_fov_degrees)
    fx = 0.5 * width / np.tan(0.5 * fov)
    fy = fx
    return np.array(
        [[fx, 0.0, (width - 1) / 2], [0.0, fy, (height - 1) / 2], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def blender_camera_to_world_cv(matrix_world: np.ndarray) -> np.ndarray:
    matrix_world = np.asarray(matrix_world, dtype=np.float64)
    if matrix_world.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform, got {matrix_world.shape}")
    return matrix_world @ BLENDER_CAMERA_FROM_CV


def validate_transform(transform: np.ndarray, atol: float = 1e-5) -> None:
    transform = np.asarray(transform)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Camera transform must be a finite 4x4 matrix")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol):
        raise ValueError("Camera rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=atol):
        raise ValueError("Camera rotation determinant must be +1")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=atol):
        raise ValueError("Invalid homogeneous transform bottom row")
