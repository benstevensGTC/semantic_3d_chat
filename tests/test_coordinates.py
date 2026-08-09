import numpy as np

from semantic_3d_chat.coordinates import (
    BLENDER_CAMERA_FROM_CV,
    blender_camera_to_world_cv,
    camera_intrinsics,
    validate_transform,
)


def test_intrinsics_center_and_focal_length() -> None:
    intrinsics = camera_intrinsics(224, 224, 90.0)
    assert np.isclose(intrinsics[0, 0], 112.0)
    assert np.isclose(intrinsics[1, 1], 112.0)
    assert np.allclose(intrinsics[:2, 2], [111.5, 111.5])


def test_blender_camera_conversion_maps_cv_forward_to_local_minus_z() -> None:
    converted = blender_camera_to_world_cv(np.eye(4))
    assert np.allclose(converted, BLENDER_CAMERA_FROM_CV)
    assert np.allclose(converted[:3, :3] @ [0, 0, 1], [0, 0, -1])


def test_validate_transform_rejects_scaling() -> None:
    invalid = np.eye(4)
    invalid[0, 0] = 2
    try:
        validate_transform(invalid)
    except ValueError as exc:
        assert "orthonormal" in str(exc)
    else:
        raise AssertionError("Scaled transform should be rejected")
