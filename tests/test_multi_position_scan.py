from __future__ import annotations

import json
import re
import sys
import types
from importlib import util
from pathlib import Path, PurePosixPath

import numpy as np
import pytest

from semantic_3d_chat.config import config_hash, load_config
from semantic_3d_chat.rendering_io import (
    RUNTIME_FRAME_KEYS,
    RUNTIME_MANIFEST_KEYS,
    load_manifest,
)
from semantic_3d_chat.scan_plan import (
    ScanPose,
    build_runtime_frame,
    build_runtime_manifest,
    expand_scan_poses,
)

EXPERIMENT_CONFIG = "configs/experiments/gemma4_multi_position_scan.yaml"


def test_default_center_scan_order_is_unchanged() -> None:
    render = load_config("configs/default.yaml")["render"]

    poses = expand_scan_poses(render)

    assert len(poses) == 24
    assert poses[:9] == (
        ScanPose((0.0, 0.0, 1.4), 0.0, -25.0),
        ScanPose((0.0, 0.0, 1.4), 45.0, -25.0),
        ScanPose((0.0, 0.0, 1.4), 90.0, -25.0),
        ScanPose((0.0, 0.0, 1.4), 135.0, -25.0),
        ScanPose((0.0, 0.0, 1.4), 180.0, -25.0),
        ScanPose((0.0, 0.0, 1.4), 225.0, -25.0),
        ScanPose((0.0, 0.0, 1.4), 270.0, -25.0),
        ScanPose((0.0, 0.0, 1.4), 315.0, -25.0),
        ScanPose((0.0, 0.0, 1.4), 0.0, 0.0),
    )
    assert poses[-1] == ScanPose((0.0, 0.0, 1.4), 315.0, 25.0)


def test_gemma4_multi_position_config_expands_center_then_numeric_positions() -> None:
    experiment = load_config(EXPERIMENT_CONFIG)
    gemma = load_config("configs/gemma4_e2b.yaml")

    assert experiment["vision"] == gemma["vision"]
    assert experiment["language"] == gemma["language"]
    poses = expand_scan_poses(experiment["render"])

    assert len(poses) == 4 * 3 * 8 == 96
    assert poses[0] == ScanPose((0.0, 0.0, 1.4), 0.0, -25.0)
    assert poses[23] == ScanPose((0.0, 0.0, 1.4), 315.0, 25.0)
    assert poses[24] == ScanPose((-1.8, -1.5, 1.4), 0.0, -25.0)
    assert poses[47] == ScanPose((-1.8, -1.5, 1.4), 315.0, 25.0)
    assert poses[48] == ScanPose((1.8, -1.5, 1.4), 0.0, -25.0)
    assert poses[72] == ScanPose((1.8, 1.0, 1.4), 0.0, -25.0)
    assert poses[-1] == ScanPose((1.8, 1.0, 1.4), 315.0, 25.0)


def test_blender_loader_resolves_the_nested_experiment_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise Blender's stdlib-only YAML path without launching Blender."""

    monkeypatch.setitem(sys.modules, "bpy", types.ModuleType("bpy"))
    mathutils = types.ModuleType("mathutils")
    mathutils.Matrix = type("Matrix", (), {})  # type: ignore[attr-defined]
    mathutils.Vector = type("Vector", (), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mathutils", mathutils)
    module_path = Path("blender/scene_utils.py").resolve()
    specification = util.spec_from_file_location("_scan_scene_utils_test", module_path)
    assert specification is not None and specification.loader is not None
    module = util.module_from_spec(specification)
    specification.loader.exec_module(module)

    config, resolved = module.load_config(EXPERIMENT_CONFIG)

    assert resolved == Path(EXPERIMENT_CONFIG).resolve()
    assert config["render"]["resolution"] == [224, 224]
    assert config["render"]["camera_position_m"] == [0.0, 0.0, 1.4]
    assert len(config["render"]["additional_camera_positions_m"]) == 3
    assert config["vision"]["model_id"] == "google/gemma-4-E2B-it"


def test_multi_position_manifest_is_complete_opaque_and_semantic_free(
    tmp_path: Path,
) -> None:
    config = load_config(EXPERIMENT_CONFIG)
    poses = expand_scan_poses(config["render"])
    intrinsics = np.asarray(
        [[154.0, 0.0, 111.5], [0.0, 154.0, 111.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    frames = []
    for frame_number, pose in enumerate(poses):
        camera_to_world = np.eye(4, dtype=np.float64)
        camera_to_world[:3, 3] = pose.position_m
        frames.append(
            build_runtime_frame(
                frame_number,
                intrinsics=intrinsics.tolist(),
                camera_to_world=camera_to_world.tolist(),
            )
        )
    manifest = build_runtime_manifest(
        scene_id="scene_000001",
        config_digest=config_hash(config),
        width=224,
        height=224,
        horizontal_fov_degrees=72.0,
        frames=frames,
    )

    assert set(manifest) == set(RUNTIME_MANIFEST_KEYS)
    assert len(manifest["frames"]) == 96
    opaque_frame = re.compile(r"f_[0-9]{6}")
    opaque_camera = re.compile(r"c_[0-9]{6}")
    semantic_filename_terms = {
        "book",
        "bowl",
        "cabinet",
        "chair",
        "cube",
        "frame",
        "lamp",
        "plant",
        "table",
    }
    for frame_number, frame in enumerate(manifest["frames"]):
        assert set(frame) == set(RUNTIME_FRAME_KEYS) - {"timestamp"}
        assert frame["frame_number"] == frame_number
        assert opaque_frame.fullmatch(frame["frame_id"])
        assert opaque_camera.fullmatch(frame["camera_id"])
        for path_field in ("rgb_path", "depth_path"):
            filename = PurePosixPath(frame[path_field]).name.casefold()
            assert not any(term in filename for term in semantic_filename_terms)
        assert np.asarray(frame["intrinsics"]).shape == (3, 3)
        assert np.asarray(frame["camera_to_world"]).shape == (4, 4)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert load_manifest(manifest_path) == manifest


def test_multi_position_plan_rejects_duplicate_or_nonfinite_positions() -> None:
    render = {
        "camera_position_m": [0.0, 0.0, 1.4],
        "yaw_degrees": [0.0],
        "pitch_degrees": [0.0],
        "additional_camera_positions_m": [[0.0, 0.0, 1.4]],
    }
    with pytest.raises(ValueError, match="unique"):
        expand_scan_poses(render)

    render["additional_camera_positions_m"] = [[1.0, float("nan"), 1.4]]
    with pytest.raises(ValueError, match="finite"):
        expand_scan_poses(render)
