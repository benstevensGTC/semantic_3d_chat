"""Shared Blender helpers for deterministic metric room generation and scanning.

This module deliberately depends only on Blender's bundled Python packages.  In
particular, it does not assume that the project's uv environment (or PyYAML) is
importable from Blender's Python 3.13 runtime.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import struct
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from mathutils import Matrix, Vector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_ID_PATTERN = re.compile(r"scene_[0-9]{6}")


def blender_cli_args() -> list[str]:
    """Return arguments after Blender's ``--`` script separator."""

    import sys

    try:
        separator = sys.argv.index("--")
    except ValueError:
        return []
    return sys.argv[separator + 1 :]


def validate_scene_id(scene_id: str) -> str:
    if not SCENE_ID_PATTERN.fullmatch(scene_id):
        raise ValueError("scene ID must match scene_ followed by six digits")
    return scene_id


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    """Parse the mapping/list/scalar YAML subset used by project configs.

    Blender does not ship PyYAML.  Keeping this small reader here avoids
    modifying Blender's bundled environment while still supporting inherited
    project profiles and folded block strings.
    """

    lines = path.read_text(encoding="utf-8").splitlines()
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    index = 0
    while index < len(lines):
        raw_line = lines[index]
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            index += 1
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML at {path}:{index + 1}: {stripped!r}")
        key, raw_value = stripped.split(":", 1)
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        raw_value = raw_value.strip()
        if raw_value in {">", ">-", "|", "|-"}:
            block_lines: list[str] = []
            index += 1
            while index < len(lines):
                candidate = lines[index]
                if candidate.strip():
                    candidate_indent = len(candidate) - len(candidate.lstrip(" "))
                    if candidate_indent <= indent:
                        break
                    block_lines.append(candidate.strip())
                index += 1
            separator = " " if raw_value.startswith(">") else "\n"
            parent[key] = separator.join(block_lines)
            continue
        if raw_value:
            parent[key] = _parse_scalar(raw_value)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        index += 1
    return root


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(base))
    for key, value in override.items():
        if key == "_base_":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_argument: str) -> tuple[dict[str, Any], Path]:
    config_path = Path(config_argument).expanduser()
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration does not exist: {config_path}")

    def load_recursive(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
        resolved = path.resolve()
        if resolved in stack:
            chain = " -> ".join(str(item) for item in (*stack, resolved))
            raise ValueError(f"Cyclic _base_ config inheritance: {chain}")
        if not resolved.is_file():
            raise FileNotFoundError(f"Configuration does not exist: {resolved}")
        loaded = _load_simple_yaml(resolved)
        if base_name := loaded.get("_base_"):
            base = load_recursive(
                resolved.parent / str(base_name),
                (*stack, resolved),
            )
            return _deep_merge(base, loaded)
        return loaded

    config = load_recursive(config_path, ())
    return config, config_path


def config_hash(config: dict[str, Any], length: int = 12) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:length]


def scene_paths(config: dict[str, Any], scene_id: str) -> dict[str, Path]:
    validate_scene_id(scene_id)
    data_root = (PROJECT_ROOT / str(config["paths"]["data_root"])).resolve()
    oracle = data_root / "oracle" / scene_id
    rendered = data_root / "rendered" / scene_id
    paths = {
        "data_root": data_root,
        "oracle": oracle,
        "rendered": rendered,
        "rgb": rendered / "rgb",
        "depth": rendered / "depth",
    }
    for path in (oracle, rendered, paths["rgb"], paths["depth"]):
        path.mkdir(parents=True, exist_ok=True)
    return paths


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def reset_scene() -> bpy.types.Scene:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "METERS"
    scene.unit_settings.scale_length = 1.0
    return scene


def _resolve_render_engine(requested: str) -> str:
    # Blender 4 used BLENDER_EEVEE_NEXT.  Blender 5.2 exposes the same renderer
    # under BLENDER_EEVEE again.  Resolve by probing the installed enum.
    available = {
        item.identifier for item in bpy.context.scene.render.bl_rna.properties["engine"].enum_items
    }
    candidates = [requested]
    if requested == "BLENDER_EEVEE_NEXT":
        candidates.append("BLENDER_EEVEE")
    for candidate in candidates:
        if candidate in available:
            return candidate
    if "BLENDER_EEVEE" in available:
        return "BLENDER_EEVEE"
    raise RuntimeError(f"Requested render engine {requested!r} is unavailable: {sorted(available)}")


def configure_render(
    scene: bpy.types.Scene,
    *,
    width: int,
    height: int,
    engine: str,
    samples: int,
) -> str:
    resolved_engine = _resolve_render_engine(engine)
    scene.render.engine = resolved_engine
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    scene.render.use_border = False
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    if hasattr(scene, "eevee"):
        if hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = max(1, int(samples))
        if hasattr(scene.eevee, "taa_samples"):
            scene.eevee.taa_samples = max(1, int(samples))
    try:
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Medium High Contrast"
    except (TypeError, ValueError):
        scene.view_settings.view_transform = "Standard"
    return resolved_engine


def create_material(
    name: str,
    rgba: tuple[float, float, float, float],
    *,
    roughness: float = 0.62,
    metallic: float = 0.0,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name=name)
    material.diffuse_color = rgba
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = rgba
        principled.inputs["Roughness"].default_value = roughness
        principled.inputs["Metallic"].default_value = metallic
    return material


def _finish_primitive(
    name: str,
    material: bpy.types.Material,
    *,
    scale: tuple[float, float, float] | None = None,
    rotation_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    obj = bpy.context.object
    obj.name = name
    if scale is not None:
        obj.scale = scale
    obj.rotation_euler = tuple(math.radians(value) for value in rotation_degrees)
    if material is not None:
        obj.data.materials.append(material)
    return obj


def add_box(
    name: str,
    center: tuple[float, float, float],
    dimensions: tuple[float, float, float],
    material: bpy.types.Material,
    *,
    rotation_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center)
    return _finish_primitive(
        name,
        material,
        scale=tuple(float(value) for value in dimensions),
        rotation_degrees=rotation_degrees,
    )


def add_cylinder(
    name: str,
    center: tuple[float, float, float],
    radius: float,
    depth: float,
    material: bpy.types.Material,
    *,
    vertices: int = 32,
    rotation_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=depth,
        location=center,
    )
    return _finish_primitive(name, material, rotation_degrees=rotation_degrees)


def add_cone(
    name: str,
    center: tuple[float, float, float],
    radius1: float,
    radius2: float,
    depth: float,
    material: bpy.types.Material,
    *,
    vertices: int = 32,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cone_add(
        vertices=vertices,
        radius1=radius1,
        radius2=radius2,
        depth=depth,
        location=center,
    )
    return _finish_primitive(name, material)


def add_uv_sphere(
    name: str,
    center: tuple[float, float, float],
    scale: tuple[float, float, float],
    material: bpy.types.Material,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=1.0, location=center)
    return _finish_primitive(name, material, scale=scale)


def add_torus(
    name: str,
    center: tuple[float, float, float],
    major_radius: float,
    minor_radius: float,
    material: bpy.types.Material,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_torus_add(
        align="WORLD",
        major_segments=36,
        minor_segments=12,
        location=center,
        major_radius=major_radius,
        minor_radius=minor_radius,
    )
    return _finish_primitive(name, material)


def assign_instance(parts: Iterable[bpy.types.Object], instance_id: str) -> list[str]:
    names: list[str] = []
    for part_index, obj in enumerate(parts):
        obj.name = f"{instance_id}_p_{part_index:02d}"
        # Opaque numeric identity is harmless in the oracle-only .blend and
        # makes offline rendering diagnostics easier without embedding labels.
        obj["instance_index"] = int(instance_id.split("_")[-1])
        names.append(obj.name)
    return names


def combined_bbox(part_names: Iterable[str]) -> tuple[list[float], list[float]]:
    bpy.context.view_layer.update()
    corners: list[Vector] = []
    for name in part_names:
        obj = bpy.data.objects[name]
        corners.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    if not corners:
        raise ValueError("Cannot compute a bounding box for an empty instance")
    minimum = [min(corner[axis] for corner in corners) for axis in range(3)]
    maximum = [max(corner[axis] for corner in corners) for axis in range(3)]
    return [float(value) for value in minimum], [float(value) for value in maximum]


def oracle_instance(
    *,
    instance_id: str,
    kind: str,
    category: str,
    color_name: str,
    rgba: tuple[float, float, float, float],
    part_names: list[str],
    support_surface: str | None,
    rotation_euler_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0),
    visible_from_center_scan: bool = True,
) -> dict[str, Any]:
    bbox_min, bbox_max = combined_bbox(part_names)
    center = [(lower + upper) / 2.0 for lower, upper in zip(bbox_min, bbox_max)]
    dimensions = [upper - lower for lower, upper in zip(bbox_min, bbox_max)]
    return {
        "instance_id": instance_id,
        "kind": kind,
        "category": category,
        "color": {"name": color_name, "rgba": [float(value) for value in rgba]},
        "pose": {
            "center_xyz_m": center,
            "rotation_euler_degrees": [float(value) for value in rotation_euler_degrees],
        },
        "dimensions_m": dimensions,
        "bbox": {"min_xyz_m": bbox_min, "max_xyz_m": bbox_max},
        "support_surface": support_surface,
        "visible_from_center_scan": bool(visible_from_center_scan),
        "expected_center_xyz_m": center,
        "part_count": len(part_names),
    }


def create_camera(name: str, horizontal_fov_degrees: float) -> bpy.types.Object:
    camera_data = bpy.data.cameras.new(f"{name}_data")
    camera_data.type = "PERSP"
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.sensor_width = 36.0
    camera_data.lens = camera_data.sensor_width / (
        2.0 * math.tan(math.radians(horizontal_fov_degrees) / 2.0)
    )
    camera_data.clip_start = 0.05
    camera_data.clip_end = 30.0
    camera = bpy.data.objects.new(name, camera_data)
    bpy.context.scene.collection.objects.link(camera)
    return camera


def point_camera_at(camera: bpy.types.Object, target: tuple[float, float, float]) -> None:
    direction = Vector(target) - camera.location
    if direction.length_squared <= 1e-12:
        raise ValueError("Camera target must differ from camera position")
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()


def set_camera_yaw_pitch(
    camera: bpy.types.Object,
    position: tuple[float, float, float] | list[float],
    yaw_degrees: float,
    pitch_degrees: float,
) -> None:
    """Set pose with yaw=0 looking +Y and right-handed positive yaw about +Z."""

    yaw = math.radians(float(yaw_degrees))
    pitch = math.radians(float(pitch_degrees))
    forward = Vector(
        (
            -math.sin(yaw) * math.cos(pitch),
            math.cos(yaw) * math.cos(pitch),
            math.sin(pitch),
        )
    )
    camera.location = tuple(float(value) for value in position)
    camera.rotation_euler = forward.to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()


def camera_intrinsics(width: int, height: int, horizontal_fov_degrees: float) -> np.ndarray:
    focal = 0.5 * float(width) / math.tan(math.radians(horizontal_fov_degrees) / 2.0)
    return np.asarray(
        [
            [focal, 0.0, (float(width) - 1.0) / 2.0],
            [0.0, focal, (float(height) - 1.0) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def camera_to_world_cv(camera: bpy.types.Object) -> np.ndarray:
    blender_camera_from_cv = Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))
    transform = camera.matrix_world @ blender_camera_from_cv
    return np.asarray([[float(transform[row][column]) for column in range(4)] for row in range(4)])


def render_png(scene: bpy.types.Scene, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.png")
    scene.render.filepath = str(temporary)
    try:
        bpy.ops.render.render(write_still=True)
        os.replace(temporary, path)
        _strip_nondeterministic_png_metadata(path)
    finally:
        temporary.unlink(missing_ok=True)


def _strip_nondeterministic_png_metadata(path: Path) -> None:
    """Remove Blender's render-date text chunks without decoding pixels."""

    payload = path.read_bytes()
    signature = b"\x89PNG\r\n\x1a\n"
    if not payload.startswith(signature):
        raise ValueError(f"Rendered output is not a PNG: {path}")
    normalized = bytearray(signature)
    offset = len(signature)
    discarded = {b"tEXt", b"zTXt", b"iTXt", b"tIME"}
    while offset < len(payload):
        if offset + 12 > len(payload):
            raise ValueError(f"Truncated PNG chunk in {path}")
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        end = offset + 12 + length
        if end > len(payload):
            raise ValueError(f"Invalid PNG chunk length in {path}")
        chunk_type = payload[offset + 4 : offset + 8]
        if chunk_type not in discarded:
            normalized.extend(payload[offset:end])
        offset = end
    temporary = path.with_name(f".{path.name}.{os.getpid()}.normalized.tmp")
    try:
        temporary.write_bytes(normalized)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def add_area_light(
    name: str,
    location: tuple[float, float, float],
    *,
    energy: float,
    size: float,
    color: tuple[float, float, float],
) -> bpy.types.Object:
    data = bpy.data.lights.new(name=f"{name}_data", type="AREA")
    data.energy = float(energy)
    data.shape = "DISK"
    data.size = float(size)
    data.color = color
    light = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(light)
    light.location = location
    return light


__all__ = [
    "PROJECT_ROOT",
    "add_area_light",
    "add_box",
    "add_cone",
    "add_cylinder",
    "add_torus",
    "add_uv_sphere",
    "assign_instance",
    "atomic_json",
    "blender_cli_args",
    "camera_intrinsics",
    "camera_to_world_cv",
    "combined_bbox",
    "config_hash",
    "configure_render",
    "create_camera",
    "create_material",
    "load_config",
    "oracle_instance",
    "point_camera_at",
    "render_png",
    "reset_scene",
    "scene_paths",
    "set_camera_yaw_pitch",
    "validate_scene_id",
]
