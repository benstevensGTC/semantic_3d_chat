import numpy as np
import pytest

from semantic_3d_chat.mapping.depth_projection import (
    depth_to_camera_points,
    project_depth_to_world,
    project_world_points_to_pixels,
    validate_intrinsics,
)


def _camera_looking_world_positive_y() -> np.ndarray:
    # Columns are world directions of CV camera +X right, +Y down, +Z forward.
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ]
    )
    transform[:3, 3] = [1.0, 2.0, 1.0]
    return transform


def test_metric_depth_back_projects_cv_axes_and_world_axes() -> None:
    depth = np.full((3, 3), 2.0, dtype=np.float32)
    intrinsics = np.array([[2.0, 0, 1.0], [0, 2.0, 1.0], [0, 0, 1.0]])
    projection = project_depth_to_world(depth, intrinsics, _camera_looking_world_positive_y())

    assert len(projection) == 9
    assert np.array_equal(projection.pixels_uv[[0, 4, 8]], [[0, 0], [1, 1], [2, 2]])
    assert np.allclose(
        projection.points_camera[[0, 4, 8]],
        [[-1.0, -1.0, 2.0], [0.0, 0.0, 2.0], [1.0, 1.0, 2.0]],
    )
    # Top image pixels have negative camera Y, hence positive world Z.
    assert np.allclose(
        projection.points_world[[0, 4, 8]],
        [[0.0, 4.0, 2.0], [1.0, 4.0, 1.0], [2.0, 4.0, 0.0]],
    )
    assert np.allclose(np.linalg.norm(projection.view_directions_world, axis=1), 1.0)


def test_projection_round_trip_preserves_pixels_and_metric_z() -> None:
    depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    intrinsics = np.array([[3.0, 0, 0.5], [0, 2.5, 0.5], [0, 0, 1.0]])
    transform = _camera_looking_world_positive_y()
    projection = project_depth_to_world(depth, intrinsics, transform)
    pixels, recovered_depth = project_world_points_to_pixels(
        projection.points_world, intrinsics, transform
    )
    assert np.allclose(pixels, projection.pixels_uv, atol=1e-6)
    assert np.allclose(recovered_depth, projection.depth_m, atol=1e-6)


def test_world_yaw_rotation_moves_cv_forward_as_expected() -> None:
    base = _camera_looking_world_positive_y()
    yaw_positive_90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transform = base.copy()
    transform[:3, :3] = yaw_positive_90 @ base[:3, :3]
    projection = project_depth_to_world(np.array([[2.0]], dtype=np.float32), np.eye(3), transform)
    # A positive right-handed yaw around world +Z turns world +Y toward -X.
    assert np.allclose(projection.points_world[0], [-1.0, 2.0, 1.0])


def test_invalid_depth_is_removed_and_stride_is_deterministic() -> None:
    depth = np.array([[1.0, 1.0, 2.0], [1.0, np.nan, 1.0], [3.0, 1.0, np.inf]], dtype=np.float32)
    intrinsics = np.eye(3)
    points, pixels, selected_depth = depth_to_camera_points(
        depth,
        intrinsics,
        min_depth_m=1.5,
        max_depth_m=3.0,
        pixel_stride=2,
    )
    assert np.array_equal(pixels, [[2, 0], [0, 2]])
    assert np.allclose(selected_depth, [2.0, 3.0])
    assert np.allclose(points, [[4.0, 0.0, 2.0], [0.0, 6.0, 3.0]])


@pytest.mark.parametrize(
    "intrinsics",
    [np.eye(4), np.diag([0.0, 1.0, 1.0]), np.full((3, 3), np.nan)],
)
def test_invalid_intrinsics_are_rejected(intrinsics: np.ndarray) -> None:
    with pytest.raises(ValueError):
        validate_intrinsics(intrinsics)
