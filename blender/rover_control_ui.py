"""Blender-native 3D operator console for the local Gemma rover backend.

Launch this script after the sanitized furnished ``.blend`` asset.  It adds a
bright toy rover as a human-only overlay, animates that rover from numeric API
state, and registers a compact ``Gemma Rover`` sidebar in every 3D viewport.
The original room asset is never saved or rewritten.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import textwrap
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import bpy
from bpy.props import BoolProperty, CollectionProperty, IntProperty, StringProperty
from mathutils import Vector

bl_info = {
    "name": "Local Gemma Toy Rover",
    "author": "semantic_3d_chat",
    "version": (1, 0, 0),
    "blender": (4, 3, 0),
    "location": "View3D > Sidebar > Gemma Rover",
    "description": "Animate a toy rover from a loopback-only local Gemma controller",
    "category": "3D View",
}

_ROOT_NAME = "r_900000"
_OVERVIEW_CAMERA_NAME = "r_900020"
_ROVER_CAMERA_NAME = "r_900021"
_COLLISION_NAME = "r_900012"
_TRAIL_NAME = "r_900030"
_MAP_NAME = "r_920000"
_MAP_MATERIAL_NAME = "r_920001"
_MAP_ATTRIBUTE_NAME = "s_920000"
# Blender can display this many Geometry Nodes points comfortably on the target
# Mac, and the denser sample keeps furniture silhouettes recognizable.  This is
# only a human visualization: the controller still receives every occupied
# block through the cached continuous scene prefix.
_MAX_MAP_POINTS = 25_000
_REQUESTS: queue.Queue[tuple[str, Any, bool] | None] = queue.Queue()
_RESULTS: queue.Queue[tuple[str, Any, bool, str | None]] = queue.Queue()
_STOP = threading.Event()
_WORKER: threading.Thread | None = None
_CLIENT: Any = None
_PENDING = 0
_LAST_POLL = 0.0
_POLL_DELAY_SECONDS = 0.8
_ANIMATION: dict[str, Any] | None = None
_ANIMATION_QUEUE: deque[Any] = deque()
_LAST_TARGET: Any = None
_LAST_VISUAL_STATE: tuple[float, float, float, float, float, bool] | None = None
_CAMERA_YAW_OFFSET_DEGREES = 0.0
_CAMERA_PITCH_DEGREES = 0.0
_CUTAWAY_NAMES: set[str] = set()
_ANNOUNCED_CONNECTION = False
_DISPLAYED_MAP_VERSION: int | None = None
# One worst-case 128-decision turn can wrap to well over 1,000 sidebar rows
# once exact model outputs and causal provenance are shown. Keep several such
# turns instead of silently discarding the beginning of the conversation.
_MAX_TRANSCRIPT_LINES = 4_096


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend-url", default="http://127.0.0.1:8770")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--no-network", action="store_true")
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(values)


_ARGS = _arguments()
_PROJECT_ROOT = _ARGS.project_root.expanduser().resolve()
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if not (_SOURCE_ROOT / "semantic_3d_chat/robot/blender_rover_bridge.py").is_file():
    raise RuntimeError("The local Blender rover bridge is unavailable")
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from semantic_3d_chat.robot.blender_rover_bridge import (
    LoopbackRoverClient,
    interpolate_pose,
    shortest_yaw_delta,
)


def _material(name: str, color: tuple[float, float, float, float], metallic: float = 0.0):
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.diffuse_color = color
    material.metallic = metallic
    material.roughness = 0.28 if metallic else 0.42
    material.use_nodes = True
    principled = next(
        (node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"),
        None,
    )
    if principled is not None:
        principled.inputs["Base Color"].default_value = color
        principled.inputs["Metallic"].default_value = metallic
        principled.inputs["Roughness"].default_value = material.roughness
    return material


def _finish_part(obj: Any, name: str, root: Any, material: Any) -> Any:
    obj.name = name
    if obj.data is not None:
        obj.data.name = name.replace("r_", "d_", 1)
    obj.parent = root
    if material is not None and hasattr(obj.data, "materials"):
        obj.data.materials.append(material)
    return obj


def _cube(
    name: str,
    root: Any,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    material: Any,
) -> Any:
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return _finish_part(obj, name, root, material)


def _cylinder(
    name: str,
    root: Any,
    location: tuple[float, float, float],
    radius: float,
    depth: float,
    rotation: tuple[float, float, float],
    material: Any,
) -> Any:
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=24, radius=radius, depth=depth, location=location, rotation=rotation
    )
    return _finish_part(bpy.context.object, name, root, material)


def _build_toy_rover() -> Any:
    existing = bpy.data.objects.get(_ROOT_NAME)
    if existing is not None:
        return existing

    root = bpy.data.objects.new(_ROOT_NAME, None)
    bpy.context.scene.collection.objects.link(root)
    orange = _material("r_910000", (1.0, 0.18, 0.025, 1.0), metallic=0.28)
    dark = _material("r_910001", (0.018, 0.025, 0.032, 1.0), metallic=0.18)
    cyan = _material("r_910002", (0.02, 0.78, 1.0, 1.0), metallic=0.12)
    silver = _material("r_910003", (0.32, 0.38, 0.42, 1.0), metallic=0.65)
    red = _material("r_910004", (1.0, 0.015, 0.01, 1.0), metallic=0.05)

    _cube("r_900001", root, (0.0, 0.0, 0.25), (0.22, 0.30, 0.085), orange)
    _cube("r_900002", root, (0.0, 0.025, 0.37), (0.16, 0.18, 0.045), dark)
    for index, (x, y) in enumerate(
        ((-0.255, -0.19), (0.255, -0.19), (-0.255, 0.19), (0.255, 0.19)),
        start=3,
    ):
        _cylinder(
            f"r_9000{index:02d}",
            root,
            (x, y, 0.13),
            0.115,
            0.085,
            (0.0, math.pi / 2.0, 0.0),
            dark,
        )
    _cylinder("r_900007", root, (0.0, 0.02, 0.51), 0.035, 0.25, (0, 0, 0), silver)
    _cube("r_900008", root, (0.0, 0.08, 0.65), (0.11, 0.09, 0.075), dark)
    for index, x in enumerate((-0.065, 0.065), start=9):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12, radius=0.037, location=(x, 0.18, 0.665))
        _finish_part(bpy.context.object, f"r_9000{index:02d}", root, cyan)
    bpy.ops.mesh.primitive_cone_add(
        vertices=24,
        radius1=0.085,
        radius2=0.0,
        depth=0.25,
        location=(0.0, 0.42, 0.29),
        rotation=(-math.pi / 2.0, 0.0, 0.0),
    )
    _finish_part(bpy.context.object, "r_900011", root, orange)
    bpy.ops.mesh.primitive_torus_add(
        major_radius=0.38, minor_radius=0.025, major_segments=40, minor_segments=10,
        location=(0.0, 0.0, 0.035),
    )
    collision = _finish_part(bpy.context.object, _COLLISION_NAME, root, red)
    collision.hide_viewport = True
    collision.hide_render = True

    camera_data = bpy.data.cameras.new("d_900021")
    camera_data.lens = 32
    camera_data.clip_start = 0.02
    camera_data.clip_end = 20.0
    rover_camera = bpy.data.objects.new(_ROVER_CAMERA_NAME, camera_data)
    bpy.context.scene.collection.objects.link(rover_camera)
    rover_camera.parent = root
    # Sit just ahead of the decorative binocular housing so the human rover
    # view is not clipped by the two cyan eye meshes.
    rover_camera.location = (0.0, 0.24, 0.68)
    rover_camera.rotation_euler = (math.pi / 2.0, 0.0, 0.0)

    trail_data = bpy.data.curves.new("d_900030", type="CURVE")
    trail_data.dimensions = "3D"
    trail_data.bevel_depth = 0.018
    trail_data.bevel_resolution = 2
    trail = bpy.data.objects.new(_TRAIL_NAME, trail_data)
    bpy.context.scene.collection.objects.link(trail)
    trail.data.materials.append(cyan)
    return root


def _build_overview_camera() -> Any:
    existing = bpy.data.objects.get(_OVERVIEW_CAMERA_NAME)
    if existing is not None:
        return existing
    camera_data = bpy.data.cameras.new("d_900020")
    camera_data.lens = 48
    camera_data.clip_start = 0.05
    camera_data.clip_end = 50.0
    camera = bpy.data.objects.new(_OVERVIEW_CAMERA_NAME, camera_data)
    bpy.context.scene.collection.objects.link(camera)
    camera.location = (6.8, -7.3, 8.2)
    direction = Vector((0.0, 0.0, 0.75)) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return camera


def _semantic_map_path(scene_id: str) -> Path:
    if len(scene_id) != 12 or not scene_id.startswith("scene_") or not scene_id[6:].isdigit():
        raise ValueError("The point map requires an opaque scene ID")
    # Fail closed on the immutable pre-goal map.  In particular, do not fall
    # back to any robot-camera refresh or legacy practical-rover artifact.
    map_root = (_PROJECT_ROOT / "data_gemma4/maps").resolve()
    scene_root = map_root / scene_id
    path = (scene_root / "voxel_map.npz").resolve()
    if path.parent != scene_root or path.is_symlink() or not path.is_file():
        raise FileNotFoundError("The immutable sanitized continuous point map is unavailable")
    return path


def _build_semantic_map_overlay(scene_id: str, *, refresh: bool = False) -> int:
    """Create an RGB-readable 3D overlay of the embedded, immutable point map."""

    existing = bpy.data.objects.get(_MAP_NAME)
    if existing is not None and not refresh:
        return len(existing.data.vertices)
    import numpy as np

    with np.load(_semantic_map_path(scene_id), allow_pickle=False) as archive:
        # Merely requiring the semantic payload key verifies that this is the
        # embedded map without decompressing its 74k x 3072 tensor in Blender.
        # The model process, not this human-only overlay, consumes that tensor.
        required = {"centers_world", "mean_rgb", "semantic_features", "confidence"}
        if not required.issubset(archive.files):
            raise ValueError("The continuous point map lacks required numeric arrays")
        points = archive["centers_world"].astype(np.float32)
        rgb = archive["mean_rgb"].astype(np.float32)
        confidence = archive["confidence"].astype(np.float32)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or rgb.shape != (len(points), 3)
        or confidence.shape != (len(points),)
        or not np.isfinite(points).all()
        or not np.isfinite(rgb).all()
        or not np.isfinite(confidence).all()
    ):
        raise ValueError("The continuous point map contains invalid numeric arrays")
    step = max(1, math.ceil(len(points) / _MAX_MAP_POINTS))
    points = points[::step]
    rgb = rgb[::step]
    confidence = confidence[::step]
    rgb = np.clip(rgb / 255.0, 0.0, 1.0)
    confidence = np.clip(confidence, 0.0, 1.0)
    colors = np.concatenate(
        (0.05 + 0.95 * rgb, (0.65 + 0.35 * confidence)[:, None]), axis=1
    ).astype(np.float32)

    if existing is not None:
        prior_mesh = existing.data
        prior_groups = [modifier.node_group for modifier in existing.modifiers]
        bpy.data.objects.remove(existing, do_unlink=True)
        if prior_mesh.users == 0:
            bpy.data.meshes.remove(prior_mesh)
        for group in prior_groups:
            if group is not None and group.users == 0:
                bpy.data.node_groups.remove(group)

    mesh = bpy.data.meshes.new("d_920000")
    mesh.from_pydata(points.tolist(), [], [])
    color_attribute = mesh.color_attributes.new(
        name=_MAP_ATTRIBUTE_NAME, type="FLOAT_COLOR", domain="POINT"
    )
    color_attribute.data.foreach_set("color", colors.reshape(-1))
    mesh.update()
    obj = bpy.data.objects.new(_MAP_NAME, mesh)
    bpy.context.scene.collection.objects.link(obj)

    material = _material(_MAP_MATERIAL_NAME, (0.1, 0.8, 1.0, 1.0))
    principled = next(
        (node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"),
        None,
    )
    old_attribute = material.node_tree.nodes.get("r_920004")
    if old_attribute is not None:
        material.node_tree.nodes.remove(old_attribute)
    attribute = material.node_tree.nodes.new("ShaderNodeAttribute")
    attribute.name = "r_920004"
    attribute.attribute_name = _MAP_ATTRIBUTE_NAME
    if principled is not None:
        material.node_tree.links.new(attribute.outputs["Color"], principled.inputs["Base Color"])
        material.node_tree.links.new(
            attribute.outputs["Color"], principled.inputs["Emission Color"]
        )
        principled.inputs["Emission Strength"].default_value = 0.35

    group = bpy.data.node_groups.new("r_920002", "GeometryNodeTree")
    group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    group_input = group.nodes.new("NodeGroupInput")
    group_output = group.nodes.new("NodeGroupOutput")
    mesh_to_points = group.nodes.new("GeometryNodeMeshToPoints")
    mesh_to_points.mode = "VERTICES"
    mesh_to_points.inputs["Radius"].default_value = 0.012
    set_material = group.nodes.new("GeometryNodeSetMaterial")
    set_material.inputs["Material"].default_value = material
    group.links.new(group_input.outputs["Geometry"], mesh_to_points.inputs["Mesh"])
    group.links.new(mesh_to_points.outputs["Points"], set_material.inputs["Geometry"])
    group.links.new(set_material.outputs["Geometry"], group_output.inputs["Geometry"])
    modifier = obj.modifiers.new(name="r_920003", type="NODES")
    modifier.node_group = group
    obj.show_in_front = True
    return len(points)


def _toggle_map(scene: Any, _context: Any) -> None:
    obj = bpy.data.objects.get(_MAP_NAME)
    if obj is not None:
        obj.hide_viewport = not bool(scene.gemma_rover_show_map)
        obj.hide_render = not bool(scene.gemma_rover_show_map)


def _prepare_numeric_cutaway() -> None:
    """Find the ceiling and two camera-facing shell slabs from geometry alone."""

    _CUTAWAY_NAMES.clear()
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.name.startswith("r_9"):
            continue
        dx, dy, dz = (float(value) for value in obj.dimensions)
        x, y, z = (float(value) for value in obj.location)
        ceiling = dz <= 0.2 and dx >= 4.0 and dy >= 4.0 and z >= 2.5
        positive_x_shell = dx <= 0.2 and dy >= 4.0 and dz >= 2.5 and x > 0.0
        negative_y_shell = dy <= 0.2 and dx >= 4.0 and dz >= 2.5 and y < 0.0
        if ceiling or positive_x_shell or negative_y_shell:
            _CUTAWAY_NAMES.add(obj.name)


def _set_cutaway(enabled: bool) -> None:
    for name in _CUTAWAY_NAMES:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.hide_viewport = enabled
            obj.hide_render = enabled


def _set_camera(camera_name: str) -> None:
    camera = bpy.data.objects.get(camera_name)
    if camera is None:
        return
    _set_cutaway(camera_name == _OVERVIEW_CAMERA_NAME)
    bpy.context.scene.camera = camera
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            space = area.spaces.active
            space.show_region_ui = True
            space.region_3d.view_perspective = "CAMERA"
            space.region_3d.view_camera_zoom = 4.0
            space.shading.type = "MATERIAL"
            space.overlay.show_floor = True
            space.overlay.show_axis_x = True
            space.overlay.show_axis_y = True


def _append_trail(x_m: float, y_m: float) -> None:
    trail = bpy.data.objects.get(_TRAIL_NAME)
    if trail is None or trail.type != "CURVE":
        return
    if not trail.data.splines:
        spline = trail.data.splines.new("POLY")
        spline.points.add(0)
        spline.points[0].co = (x_m, y_m, 0.035, 1.0)
        return
    spline = trail.data.splines[0]
    last = spline.points[-1].co
    if math.hypot(float(last.x) - x_m, float(last.y) - y_m) < 0.01:
        return
    spline.points.add(1)
    spline.points[-1].co = (x_m, y_m, 0.035, 1.0)


def _set_rover_camera_orientation(yaw_offset_degrees: float, pitch_degrees: float) -> None:
    """Aim the parented Blender camera using the simulator's CV convention."""

    camera = bpy.data.objects.get(_ROVER_CAMERA_NAME)
    if camera is None:
        return
    yaw = math.radians(float(yaw_offset_degrees))
    pitch = math.radians(float(pitch_degrees))
    direction = Vector(
        (
            -math.sin(yaw) * math.cos(pitch),
            math.cos(yaw) * math.cos(pitch),
            math.sin(pitch),
        )
    )
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = direction.to_track_quat("-Z", "Y")


def _visual_state(pose: Any) -> tuple[float, float, float, float, float, bool]:
    return (
        float(pose.x_m),
        float(pose.y_m),
        float(pose.body_yaw_degrees),
        float(pose.camera_yaw_degrees),
        float(pose.pitch_degrees),
        bool(pose.collision),
    )


def _begin_animation(pose: Any) -> None:
    global _ANIMATION, _LAST_TARGET, _LAST_VISUAL_STATE
    root = _build_toy_rover()
    start_xy = (float(root.location.x), float(root.location.y))
    start_yaw = math.degrees(float(root.rotation_euler.z))
    target_xy = (float(pose.x_m), float(pose.y_m))
    target_yaw = float(pose.body_yaw_degrees)
    distance = math.dist(start_xy, target_xy)
    yaw_delta = abs((target_yaw - start_yaw + 180.0) % 360.0 - 180.0)
    duration = max(0.28, min(1.4, 0.45 + distance * 0.75 + yaw_delta / 180.0))
    _ANIMATION = {
        "started": time.monotonic(),
        "duration": duration,
        "start_xy": start_xy,
        "target_xy": target_xy,
        "start_yaw": start_yaw,
        "target_yaw": target_yaw,
        "start_camera_yaw_offset": _CAMERA_YAW_OFFSET_DEGREES,
        "target_camera_yaw_offset": shortest_yaw_delta(
            target_yaw, float(pose.camera_yaw_degrees)
        ),
        "start_camera_pitch": _CAMERA_PITCH_DEGREES,
        "target_camera_pitch": float(pose.pitch_degrees),
    }
    if _LAST_TARGET is None or math.dist(_LAST_TARGET, target_xy) >= 0.01:
        _append_trail(*target_xy)
    _LAST_TARGET = target_xy
    collision = bpy.data.objects.get(_COLLISION_NAME)
    if collision is not None:
        collision.hide_viewport = not bool(pose.collision)
        collision.hide_render = not bool(pose.collision)
    _LAST_VISUAL_STATE = _visual_state(pose)


def _queue_animation(poses: Any) -> int:
    """Preserve ordered numeric action receipts as visible 3D motion."""

    previous = _LAST_VISUAL_STATE
    added = 0
    if _ANIMATION_QUEUE:
        previous = _visual_state(_ANIMATION_QUEUE[-1])
    for pose in poses:
        visual = _visual_state(pose)
        if visual == previous:
            continue
        _ANIMATION_QUEUE.append(pose)
        added += 1
        previous = visual
    if _ANIMATION is None and _ANIMATION_QUEUE:
        _begin_animation(_ANIMATION_QUEUE.popleft())
    return added


def _animate() -> None:
    global _ANIMATION, _CAMERA_YAW_OFFSET_DEGREES, _CAMERA_PITCH_DEGREES
    if _ANIMATION is None:
        if _ANIMATION_QUEUE:
            _begin_animation(_ANIMATION_QUEUE.popleft())
        return
    elapsed = time.monotonic() - float(_ANIMATION["started"])
    fraction = elapsed / float(_ANIMATION["duration"])
    x_m, y_m, yaw = interpolate_pose(
        _ANIMATION["start_xy"],
        _ANIMATION["target_xy"],
        _ANIMATION["start_yaw"],
        _ANIMATION["target_yaw"],
        fraction,
    )
    root = _build_toy_rover()
    root.location.x = x_m
    root.location.y = y_m
    root.rotation_euler.z = math.radians(yaw)
    amount = min(1.0, max(0.0, fraction))
    smooth = amount * amount * (3.0 - 2.0 * amount)
    start_offset = float(_ANIMATION["start_camera_yaw_offset"])
    target_offset = float(_ANIMATION["target_camera_yaw_offset"])
    _CAMERA_YAW_OFFSET_DEGREES = start_offset + smooth * shortest_yaw_delta(
        start_offset, target_offset
    )
    start_pitch = float(_ANIMATION["start_camera_pitch"])
    target_pitch = float(_ANIMATION["target_camera_pitch"])
    _CAMERA_PITCH_DEGREES = start_pitch + smooth * (target_pitch - start_pitch)
    _set_rover_camera_orientation(
        _CAMERA_YAW_OFFSET_DEGREES, _CAMERA_PITCH_DEGREES
    )
    if fraction >= 1.0:
        _ANIMATION = None
        if not _ANIMATION_QUEUE and not bpy.context.scene.gemma_rover_busy:
            bpy.context.scene.gemma_rover_status = "Ready · local Gemma"


def _network_worker() -> None:
    while not _STOP.is_set():
        try:
            item = _REQUESTS.get(timeout=0.25)
        except queue.Empty:
            continue
        if item is None:
            return
        kind, payload, silent = item
        try:
            if kind == "state":
                response = _CLIENT.state()
            elif kind == "instruction":
                response = _CLIENT.instruct(str(payload))
            else:
                raise ValueError("Unknown local rover request")
            _RESULTS.put((kind, response, silent, None))
        except (ConnectionError, OSError, RuntimeError, TypeError, ValueError) as exc:
            # Blender must keep running when the backend restarts.
            _RESULTS.put((kind, None, silent, str(exc)[:500]))


def _submit(kind: str, payload: Any = None, *, silent: bool = False) -> bool:
    global _PENDING
    if _PENDING:
        return False
    _PENDING += 1
    scene = bpy.context.scene
    if not silent:
        scene.gemma_rover_busy = True
        scene.gemma_rover_status = (
            "Gemma is thinking locally…" if kind == "instruction" else "Executing bounded move…"
        )
    _REQUESTS.put((kind, payload, silent))
    return True


def _receive_results() -> None:
    global _ANNOUNCED_CONNECTION, _DISPLAYED_MAP_VERSION, _PENDING
    while True:
        try:
            kind, response, silent, error = _RESULTS.get_nowait()
        except queue.Empty:
            return
        _PENDING = max(0, _PENDING - 1)
        scene = bpy.context.scene
        scene.gemma_rover_busy = False
        if error is not None:
            scene.gemma_rover_connected = False
            scene.gemma_rover_status = "Waiting for local rover backend…"
            if not silent:
                scene.gemma_rover_reply = error
                _append_transcript("status", f"Local request failed: {error}")
            continue
        pose = response.pose
        visual_steps = _queue_animation(response.trajectory)
        scene.gemma_rover_connected = True
        scene.gemma_rover_status = (
            f"Replaying already-returned Gemma decisions · {visual_steps} visual steps"
            if kind != "state" and visual_steps
            else "Ready · local Gemma"
        )
        scene.gemma_rover_pose = (
            f"x {pose.x_m:+.2f} m   y {pose.y_m:+.2f} m   yaw {pose.body_yaw_degrees:+.0f}°"
        )
        scene.gemma_rover_scene = pose.scene_id
        _update_memory_diagnostics(scene, response.scene_memory)
        if not _ANNOUNCED_CONNECTION:
            _append_transcript("status", "Local Gemma connected; continuous 3D memory is ready.")
            _ANNOUNCED_CONNECTION = True
        if _DISPLAYED_MAP_VERSION != pose.map_version:
            try:
                point_count = _build_semantic_map_overlay(pose.scene_id, refresh=True)
                scene.gemma_rover_map_points = f"{point_count:,} embedded map points shown"
                _DISPLAYED_MAP_VERSION = pose.map_version
            except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
                scene.gemma_rover_map_points = "embedded map unavailable"
                if not silent:
                    _append_transcript("status", f"3D point-map overlay unavailable: {exc}")
        if kind != "state":
            # Blender may re-commit an active text field after the operator has
            # cleared it. Clear again when the response arrives so the next
            # high-level goal always starts from an empty composer.
            scene.gemma_rover_command = ""
            for event in response.events:
                _append_transcript("agent", event)
            if response.reply.strip():
                scene.gemma_rover_reply = response.reply.strip()
                _append_transcript("assistant", response.reply.strip())


def _timer() -> float | None:
    global _LAST_POLL
    if _STOP.is_set():
        return None
    try:
        _receive_results()
        _animate()
        now = time.monotonic()
        if (
            _PENDING == 0
            and _ANIMATION is None
            and not _ANIMATION_QUEUE
            and now - _LAST_POLL >= _POLL_DELAY_SECONDS
        ):
            _LAST_POLL = now
            _submit("state", silent=True)
    except (AttributeError, ReferenceError, RuntimeError):
        return 0.5
    return 0.05


def _append_transcript(role: str, text: str) -> None:
    scene = bpy.context.scene
    normalized = " ".join(str(text).split())
    if not normalized:
        return
    lines = textwrap.wrap(normalized, width=42) or [normalized]
    for index, line in enumerate(lines):
        item = scene.gemma_rover_messages.add()
        item.role = role if index == 0 else "continuation"
        item.text = line
    while len(scene.gemma_rover_messages) > _MAX_TRANSCRIPT_LINES:
        scene.gemma_rover_messages.remove(0)
    scene.gemma_rover_message_index = max(0, len(scene.gemma_rover_messages) - 1)


def _update_memory_diagnostics(scene: Any, diagnostics: Any) -> None:
    if diagnostics is None:
        scene.gemma_rover_memory_shape = "shape pending"
        scene.gemma_rover_memory_hash = "hash pending"
        scene.gemma_rover_memory_norm = "norm pending"
        return
    scene.gemma_rover_memory_shape = " × ".join(str(value) for value in diagnostics.shape)
    scene.gemma_rover_memory_hash = f"{diagnostics.sha256[:12]}…"
    scene.gemma_rover_memory_norm = f"L2 {diagnostics.l2_norm:.3f}"


class GEMMA_ROVER_PG_message(bpy.types.PropertyGroup):
    role: StringProperty(default="status")
    text: StringProperty(default="")


class GEMMA_ROVER_UL_transcript(bpy.types.UIList):
    """Scrollable, line-wrapped conversation and agent event history."""

    def draw_item(
        self,
        _context: Any,
        layout: Any,
        _data: Any,
        item: Any,
        _icon: int,
        _active_data: Any,
        _active_property: str,
        _index: int,
    ) -> None:
        icons = {
            "user": "USER",
            "assistant": "LIGHT",
            "agent": "TRACKING_FORWARDS",
            "status": "INFO",
            "continuation": "BLANK1",
        }
        prefixes = {
            "user": "You",
            "assistant": "Gemma",
            "agent": "Gemma decision",
            "status": "System",
        }
        prefix = prefixes.get(item.role)
        text = f"{prefix}: {item.text}" if prefix else item.text
        layout.label(text=text, icon=icons.get(item.role, "BLANK1"))


class GEMMA_ROVER_OT_send(bpy.types.Operator):
    bl_idname = "gemma_rover.send"
    bl_label = "Send to Gemma"
    bl_description = "Send this instruction to the local Gemma rover controller"

    def execute(self, context: Any):
        text = context.scene.gemma_rover_command.strip()
        if not text:
            self.report({"WARNING"}, "Enter an instruction first")
            return {"CANCELLED"}
        if not _submit("instruction", text):
            self.report({"WARNING"}, "The local rover is still working")
            return {"CANCELLED"}
        _append_transcript("user", text)
        _append_transcript("status", "Gemma is thinking over continuous 3D memory.")
        context.scene.gemma_rover_command = ""
        return {"FINISHED"}


class GEMMA_ROVER_OT_view(bpy.types.Operator):
    bl_idname = "gemma_rover.view"
    bl_label = "Change rover view"

    camera_name: StringProperty(default=_OVERVIEW_CAMERA_NAME)

    def execute(self, _context: Any):
        if self.camera_name not in {_OVERVIEW_CAMERA_NAME, _ROVER_CAMERA_NAME}:
            return {"CANCELLED"}
        _set_camera(self.camera_name)
        return {"FINISHED"}


class GEMMA_ROVER_PT_panel(bpy.types.Panel):
    bl_label = "Gemma Toy Rover"
    bl_idname = "GEMMA_ROVER_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Gemma Rover"

    def draw(self, context: Any) -> None:
        scene = context.scene
        layout = self.layout
        status_box = layout.box()
        status = status_box.row()
        status.alert = not scene.gemma_rover_connected or scene.gemma_rover_busy
        status.label(
            text=scene.gemma_rover_status,
            icon=(
                "TIME"
                if scene.gemma_rover_busy
                else "CHECKMARK"
                if scene.gemma_rover_connected
                else "LINKED"
            ),
        )
        views = layout.row()
        overview = views.operator("gemma_rover.view", text="Return to Room Overview", icon="HOME")
        overview.camera_name = _OVERVIEW_CAMERA_NAME
        overlay = layout.row()
        overlay.prop(scene, "gemma_rover_show_map", text="Embedded 3D map overlay")

        layout.separator()
        layout.label(text="Conversation", icon="COMMUNITY")
        layout.template_list(
            "GEMMA_ROVER_UL_transcript",
            "conversation",
            scene,
            "gemma_rover_messages",
            scene,
            "gemma_rover_message_index",
            rows=10,
        )

        layout.separator()
        layout.label(text="Give Gemma a goal")
        layout.label(text="Gemma chooses every waypoint, heading, and stop.")
        layout.label(text="No direct driving controls.")
        layout.prop(scene, "gemma_rover_command", text="", icon="CONSOLE")
        send = layout.row()
        send.enabled = not scene.gemma_rover_busy
        send.operator("gemma_rover.send", text="Send goal to local Gemma", icon="PLAY")

        details = layout.row()
        details.prop(
            scene,
            "gemma_rover_show_technical_details",
            text="Technical scene-memory details",
            icon=(
                "DISCLOSURE_TRI_DOWN"
                if scene.gemma_rover_show_technical_details
                else "DISCLOSURE_TRI_RIGHT"
            ),
            emboss=False,
        )
        if scene.gemma_rover_show_technical_details:
            memory = layout.box()
            memory.label(text="Continuous scene tokens", icon="NODETREE")
            memory.label(text=f"shape  {scene.gemma_rover_memory_shape}")
            memory.label(
                text=(
                    f"{scene.gemma_rover_memory_norm}   hash "
                    f"{scene.gemma_rover_memory_hash}"
                )
            )
            memory.label(text=scene.gemma_rover_map_points, icon="POINTCLOUD_DATA")
            memory.label(text="Cyan line = numeric motion receipts", icon="CURVE_DATA")
            memory.label(text=scene.gemma_rover_pose, icon="ORIENTATION_GLOBAL")


_CLASSES = (
    GEMMA_ROVER_PG_message,
    GEMMA_ROVER_UL_transcript,
    GEMMA_ROVER_OT_send,
    GEMMA_ROVER_OT_view,
    GEMMA_ROVER_PT_panel,
)


def register() -> None:
    global _CLIENT, _WORKER
    for cls in _CLASSES:
        try:
            bpy.utils.unregister_class(cls)
        except (RuntimeError, ValueError):
            pass
        bpy.utils.register_class(cls)
    bpy.types.Scene.gemma_rover_command = StringProperty(default="")
    bpy.types.Scene.gemma_rover_status = StringProperty(default="Starting local rover…")
    bpy.types.Scene.gemma_rover_reply = StringProperty(default="")
    bpy.types.Scene.gemma_rover_pose = StringProperty(default="pose unavailable")
    bpy.types.Scene.gemma_rover_scene = StringProperty(default="")
    bpy.types.Scene.gemma_rover_connected = BoolProperty(default=False)
    bpy.types.Scene.gemma_rover_busy = BoolProperty(default=False)
    bpy.types.Scene.gemma_rover_show_map = BoolProperty(default=True, update=_toggle_map)
    bpy.types.Scene.gemma_rover_show_technical_details = BoolProperty(default=False)
    bpy.types.Scene.gemma_rover_messages = CollectionProperty(type=GEMMA_ROVER_PG_message)
    bpy.types.Scene.gemma_rover_message_index = IntProperty(default=0, min=0)
    bpy.types.Scene.gemma_rover_memory_shape = StringProperty(default="shape pending")
    bpy.types.Scene.gemma_rover_memory_hash = StringProperty(default="hash pending")
    bpy.types.Scene.gemma_rover_memory_norm = StringProperty(default="norm pending")
    bpy.types.Scene.gemma_rover_map_points = StringProperty(default="map pending")
    source_object_count = len(bpy.context.scene.objects)
    _prepare_numeric_cutaway()
    _build_toy_rover()
    overview = _build_overview_camera()
    _set_cutaway(True)
    bpy.context.scene.camera = overview
    _append_transcript(
        "status",
        "The room was pre-scanned before any goal; no rover camera is used as an agent input.",
    )
    scene_hint = Path(bpy.data.filepath).parent.name
    try:
        map_points = _build_semantic_map_overlay(scene_hint)
        bpy.context.scene.gemma_rover_map_points = f"{map_points:,} embedded map points shown"
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError):
        pass
    if not bpy.app.background:
        bpy.app.timers.register(lambda: (_set_camera(_OVERVIEW_CAMERA_NAME), None)[1], first_interval=0.5)
    if not _ARGS.no_network:
        _CLIENT = LoopbackRoverClient(_ARGS.backend_url)
        _WORKER = threading.Thread(target=_network_worker, name="gemma-rover-loopback", daemon=True)
        _WORKER.start()
        bpy.app.timers.register(_timer, first_interval=0.1, persistent=True)
    else:
        bpy.context.scene.gemma_rover_status = "3D rover mesh ready · network disabled"
    print(
        "BLENDER_ROVER_READY="
        + json.dumps(
            {
                "source_scene_objects": source_object_count,
                "rover_root": _ROOT_NAME,
                "overview_camera": _OVERVIEW_CAMERA_NAME,
                "rover_camera": _ROVER_CAMERA_NAME,
                "network_enabled": not _ARGS.no_network,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def unregister() -> None:
    _STOP.set()
    _REQUESTS.put(None)
    for name in (
        "gemma_rover_command",
        "gemma_rover_status",
        "gemma_rover_reply",
        "gemma_rover_pose",
        "gemma_rover_scene",
        "gemma_rover_connected",
        "gemma_rover_busy",
        "gemma_rover_show_map",
        "gemma_rover_show_technical_details",
        "gemma_rover_messages",
        "gemma_rover_message_index",
        "gemma_rover_memory_shape",
        "gemma_rover_memory_hash",
        "gemma_rover_memory_norm",
        "gemma_rover_map_points",
    ):
        if hasattr(bpy.types.Scene, name):
            delattr(bpy.types.Scene, name)
    for cls in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except (RuntimeError, ValueError):
            pass


if __name__ == "__main__":
    register()
