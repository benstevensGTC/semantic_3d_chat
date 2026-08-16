from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from semantic_3d_chat.robot.blender_scanner import SanitizedBlenderScanner


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _asset(root: Path) -> Path:
    asset = root / "runtime_assets" / "scene_000001" / "s_000001.blend"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"opaque sanitized blender fixture")
    manifest = {
        "schema": "semantic_3d_chat.runtime_scene.v2",
        "scene_id": "scene_000001",
        "asset_file": "s_000001.blend",
        "asset_sha256": _sha256(asset),
        "object_names_opaque": True,
        "nested_names_opaque": True,
        "custom_properties_present": False,
        "external_assets_present": False,
        "automation_present": False,
        "animation_present": False,
        "unsupported_datablocks_present": False,
        "strict_nested_datablock_audit_passed": True,
        "mesh_objects": 4,
        "light_objects": 1,
        "materials": 3,
        "collections": 1,
        "node_trees": 4,
    }
    asset.with_suffix(".json").write_text(json.dumps(manifest), encoding="utf-8")
    return asset


class FakeBlenderRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **options: object) -> None:
        assert options["check"] is True
        self.commands.append(command)
        separator = command.index("--")
        arguments = command[separator + 1 :]
        parsed = {arguments[index]: arguments[index + 1] for index in range(0, len(arguments), 2)}
        output = Path(parsed["--output"])
        output.mkdir(parents=True, exist_ok=True)
        observation = parsed["--observation"]
        width = int(parsed["--width"])
        height = int(parsed["--height"])
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[:, :, 0] = 80
        rgb[:, :, 1] = np.arange(width, dtype=np.uint8)[None, :]
        Image.fromarray(rgb).save(output / f"{observation}.png")
        depth = np.full((height, width), 1.25, dtype=np.float32)
        np.save(output / f"{observation}.npy", depth, allow_pickle=False)
        camera_to_world = np.eye(4, dtype=np.float64)
        yaw = math.radians(float(parsed["--yaw"]))
        pitch = math.radians(float(parsed["--pitch"]))
        right = np.asarray([math.cos(yaw), math.sin(yaw), 0.0])
        forward = np.asarray(
            [
                -math.sin(yaw) * math.cos(pitch),
                math.cos(yaw) * math.cos(pitch),
                math.sin(pitch),
            ]
        )
        camera_to_world[:3, :3] = np.stack((right, np.cross(forward, right), forward), axis=1)
        camera_to_world[:3, 3] = [
            float(parsed["--x"]),
            float(parsed["--y"]),
            float(parsed["--z"]),
        ]
        focal = width / 2.0
        receipt = {
            "schema": "semantic_3d_chat.runtime_observation.v1",
            "scene_id": parsed["--scene"],
            "observation_id": observation,
            "rgb_path": f"{observation}.png",
            "depth_path": f"{observation}.npy",
            "intrinsics": [
                [focal, 0.0, (width - 1) / 2.0],
                [0.0, focal, (height - 1) / 2.0],
                [0.0, 0.0, 1.0],
            ],
            "camera_to_world": camera_to_world.tolist(),
            "width": width,
            "height": height,
            "valid_depth_pixels": width * height,
        }
        (output / f"{observation}.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )


def test_same_numeric_pose_produces_identical_sanitized_rgbd(tmp_path: Path) -> None:
    runner = FakeBlenderRunner()
    scanner = SanitizedBlenderScanner(
        "scene_000001",
        _asset(tmp_path),
        resolution=(8, 6),
        horizontal_fov_degrees=72.0,
        engine="BLENDER_EEVEE_NEXT",
        samples=1,
        max_depth_m=8.0,
        output_directory=tmp_path / "observations",
        blender_executable="/bin/sh",
        command_runner=runner,
    )
    pose = {
        "camera_position_m": (0.25, -0.5, 1.2),
        "yaw_degrees": 15.0,
        "pitch_degrees": -5.0,
    }

    first = scanner.capture(observation_index=1, **pose)
    second = scanner.capture(observation_index=2, **pose)

    assert np.array_equal(first.rgb, second.rgb)
    assert np.array_equal(first.depth_m, second.depth_m)
    assert np.array_equal(first.intrinsics, second.intrinsics)
    assert np.array_equal(first.camera_to_world, second.camera_to_world)
    assert scanner.integrate(first, np.zeros(1, dtype=bool)) == 48
    first_coverage = scanner.directional_coverage
    assert 0.0 < first_coverage < 1.0
    assert scanner.integrate(second, np.zeros(1, dtype=bool)) == 48
    assert scanner.directional_coverage == first_coverage
    assert len(runner.commands) == 2
    for command in runner.commands:
        encoded = " ".join(command).casefold()
        assert "oracle" not in encoded and "/qa/" not in encoded
        assert command[2] == "--disable-autoexec"
        assert command[3].endswith("s_000001.blend")
        assert "--asset-sha256" in command


def test_runtime_scanner_rejects_oracle_asset_path_before_open(tmp_path: Path) -> None:
    oracle_asset = tmp_path / "oracle" / "scene.blend"
    oracle_asset.parent.mkdir()
    oracle_asset.write_bytes(b"not allowed")
    with pytest.raises(ValueError, match="oracle or QA"):
        SanitizedBlenderScanner(
            "scene_000001",
            oracle_asset,
            resolution=(8, 8),
            horizontal_fov_degrees=72.0,
            engine="BLENDER_EEVEE_NEXT",
            samples=1,
            max_depth_m=8.0,
            output_directory=tmp_path / "observations",
            blender_executable="/bin/sh",
        )


def test_runtime_scanner_authenticates_asset_hash(tmp_path: Path) -> None:
    asset = _asset(tmp_path)
    asset.write_bytes(b"changed after manifest")
    with pytest.raises(ValueError, match="hash differs"):
        SanitizedBlenderScanner(
            "scene_000001",
            asset,
            resolution=(8, 8),
            horizontal_fov_degrees=72.0,
            engine="BLENDER_EEVEE_NEXT",
            samples=1,
            max_depth_m=8.0,
            output_directory=tmp_path / "observations",
            blender_executable="/bin/sh",
        )


def test_runtime_scanner_rejects_non_strict_or_legacy_attestation(tmp_path: Path) -> None:
    asset = _asset(tmp_path)
    manifest_path = asset.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema"] = "semantic_3d_chat.runtime_scene.v1"
    manifest["strict_nested_datablock_audit_passed"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="strict v2"):
        SanitizedBlenderScanner(
            "scene_000001",
            asset,
            resolution=(8, 8),
            horizontal_fov_degrees=72.0,
            engine="BLENDER_EEVEE_NEXT",
            samples=1,
            max_depth_m=8.0,
            output_directory=tmp_path / "observations",
            blender_executable="/bin/sh",
        )
