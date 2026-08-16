from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.robot.blender_scanner import SanitizedBlenderScanner

BLENDER = shutil.which("blender")


def _run_blender(arguments: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    assert BLENDER is not None
    return subprocess.run(
        [BLENDER, *arguments],
        cwd=PROJECT_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def _source(path: Path, *, modifier: bool) -> None:
    setup = [
        "import bpy",
        "bpy.ops.mesh.primitive_cube_add()",
        "o=bpy.context.object",
        "m=bpy.data.materials.new('fixture_material')",
        "m.use_nodes=True",
        "o.data.materials.append(m)",
        "bpy.ops.object.light_add(type='AREA', location=(0,0,2))",
        "bpy.context.scene.world.use_nodes=True",
    ]
    if modifier:
        setup.append("o.modifiers.new(name='fixture_modifier',type='BEVEL')")
    setup.append(f"bpy.ops.wm.save_as_mainfile(filepath={str(path)!r})")
    _run_blender(
        [
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python-exit-code",
            "11",
            "--python-expr",
            ";".join(setup),
        ]
    )


@pytest.mark.skipif(BLENDER is None, reason="Blender is unavailable")
def test_runtime_scene_v2_clean_export_reopens_and_renders(tmp_path: Path) -> None:
    source = tmp_path / "source.blend"
    asset = tmp_path / "s_000001.blend"
    _source(source, modifier=False)
    command = [
        "--background",
        "--disable-autoexec",
        str(source),
        "--python-exit-code",
        "11",
        "--python",
        str(PROJECT_ROOT / "blender" / "export_runtime_scene.py"),
        "--",
        "--scene",
        "scene_000001",
        "--output",
        str(asset),
    ]
    _run_blender(command)

    manifest = json.loads(asset.with_suffix(".json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "semantic_3d_chat.runtime_scene.v2"
    assert manifest["strict_nested_datablock_audit_passed"] is True
    for field in (
        "automation_present",
        "animation_present",
        "external_assets_present",
        "unsupported_datablocks_present",
        "custom_properties_present",
    ):
        assert manifest[field] is False
    scanner = SanitizedBlenderScanner(
        "scene_000001",
        asset,
        resolution=(16, 16),
        horizontal_fov_degrees=72.0,
        engine="BLENDER_EEVEE_NEXT",
        samples=1,
        max_depth_m=10.0,
        output_directory=tmp_path / "runtime",
        blender_executable=BLENDER,
    )
    observation = scanner.capture(
        observation_index=1,
        camera_position_m=(0.0, -4.0, 1.2),
        yaw_degrees=0.0,
        pitch_degrees=0.0,
    )
    assert int((observation.depth_m > 0).sum()) > 0


@pytest.mark.skipif(BLENDER is None, reason="Blender is unavailable")
def test_runtime_scene_v2_rejects_modifier_with_nonzero_exit(tmp_path: Path) -> None:
    source = tmp_path / "source.blend"
    asset = tmp_path / "s_000001.blend"
    _source(source, modifier=True)
    result = _run_blender(
        [
            "--background",
            "--disable-autoexec",
            str(source),
            "--python-exit-code",
            "11",
            "--python",
            str(PROJECT_ROOT / "blender" / "export_runtime_scene.py"),
            "--",
            "--scene",
            "scene_000001",
            "--output",
            str(asset),
        ],
        check=False,
    )
    assert result.returncode == 11
    assert "geometry modifier" in result.stderr
    assert not asset.exists()
