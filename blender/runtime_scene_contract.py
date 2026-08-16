"""Strict Blender runtime-asset sanitization and render-time auditing.

The runtime scene is deliberately much smaller than a general ``.blend``
file: one scene containing static mesh and light objects, numeric transforms,
simple numeric shader graphs, collections, and a world.  Anything capable of
automation, animation, external I/O, hidden geometry generation, or carrying
custom textual metadata fails closed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final

import bpy

RUNTIME_SCENE_SCHEMA: Final[str] = "semantic_3d_chat.runtime_scene.v2"
OPAQUE_NAME: Final[re.Pattern[str]] = re.compile(r"[a-z]_[0-9]{6}")
SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
MANIFEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "scene_id",
        "asset_file",
        "asset_sha256",
        "object_names_opaque",
        "nested_names_opaque",
        "custom_properties_present",
        "external_assets_present",
        "automation_present",
        "animation_present",
        "unsupported_datablocks_present",
        "strict_nested_datablock_audit_passed",
        "mesh_objects",
        "light_objects",
        "materials",
        "collections",
        "node_trees",
    }
)

_ALLOWED_OBJECT_TYPES: Final[frozenset[str]] = frozenset({"MESH", "LIGHT"})
_ALLOWED_LIGHT_TYPES: Final[frozenset[str]] = frozenset({"AREA", "POINT", "SPOT", "SUN"})
_ALLOWED_MATERIAL_NODES: Final[frozenset[str]] = frozenset(
    {"ShaderNodeBsdfPrincipled", "ShaderNodeOutputMaterial"}
)
_ALLOWED_WORLD_NODES: Final[frozenset[str]] = frozenset(
    {"ShaderNodeBackground", "ShaderNodeOutputWorld"}
)
_ALLOWED_LIGHT_NODES: Final[frozenset[str]] = frozenset(
    {"ShaderNodeEmission", "ShaderNodeOutputLight"}
)
_SAFE_MESH_ATTRIBUTES: Final[frozenset[str]] = frozenset(
    {"position", "sharp_face", ".edge_verts", ".corner_vert", ".corner_edge"}
)
_FORBIDDEN_DATA_COLLECTIONS: Final[tuple[str, ...]] = (
    "actions",
    "annotations",
    "armatures",
    "cache_files",
    "curves",
    "fonts",
    "grease_pencils",
    "hair_curves",
    "lattices",
    "libraries",
    "lightprobes",
    "masks",
    "metaballs",
    "movieclips",
    "node_groups",
    "paint_curves",
    "particles",
    "pointclouds",
    "sounds",
    "speakers",
    "shape_keys",
    "texts",
    "textures",
    "volumes",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strip_properties(value: object) -> None:
    try:
        keys = list(value.keys()) if hasattr(value, "keys") else []
    except TypeError:
        keys = []
    for key in keys:
        try:
            del value[key]
        except (KeyError, TypeError):
            pass


def _assert_no_properties(value: object, *, kind: str) -> None:
    try:
        keys = list(value.keys()) if hasattr(value, "keys") else []
    except TypeError:
        keys = []
    if keys:
        raise ValueError(f"Runtime scene contains custom {kind} metadata")


def _assert_local(value: object, *, kind: str) -> None:
    if getattr(value, "library", None) is not None:
        raise ValueError(f"Runtime scene contains an external {kind} library link")
    if getattr(value, "override_library", None) is not None:
        raise ValueError(f"Runtime scene contains a {kind} library override")
    if getattr(value, "asset_data", None) is not None:
        raise ValueError(f"Runtime scene contains custom {kind} asset metadata")


def _assert_static(value: object, *, kind: str) -> None:
    animation = getattr(value, "animation_data", None)
    if animation is not None:
        raise ValueError(f"Runtime scene contains {kind} animation or drivers")


def _finite(values: Iterable[float], *, kind: str) -> None:
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"Runtime scene contains non-finite {kind} values")


def _rename(values: Iterable[Any], prefix: str) -> None:
    for index, value in enumerate(sorted(values, key=lambda item: item.name)):
        value.name = f"{prefix}_{index:06d}"
        _strip_properties(value)


def _reject_forbidden_datablocks() -> None:
    if bpy.data.use_autopack:
        raise ValueError("Runtime scene source enables automatic external-file packing")
    for collection_name in _FORBIDDEN_DATA_COLLECTIONS:
        collection = getattr(bpy.data, collection_name, ())
        if len(collection):
            raise ValueError(
                f"Runtime scene source contains forbidden {collection_name} datablocks"
            )
    for image in bpy.data.images:
        # Blender creates non-file-backed viewer buffers internally.  They are
        # not material inputs and are not serialized environmental content.
        if image.source != "VIEWER" or image.filepath or image.packed_file is not None:
            raise ValueError("Runtime scene source contains an image or external bitmap")


def _node_tree_audit(
    tree: Any,
    *,
    allowed_node_types: frozenset[str],
    audit_names: bool,
) -> None:
    _assert_local(tree, kind="node tree")
    _assert_static(tree, kind="node tree")
    _assert_no_properties(tree, kind="node-tree")
    # Blender 5.2 embeds owner node trees with a read-only generated name.
    # The owner datablock is opaque, and every mutable nested node name and
    # label is scrubbed below.
    for node in tree.nodes:
        if node.bl_idname not in allowed_node_types:
            raise ValueError(f"Runtime scene contains unsupported shader node {node.bl_idname}")
        if node.label:
            raise ValueError("Runtime scene contains a custom shader-node label")
        if audit_names and OPAQUE_NAME.fullmatch(node.name) is None:
            raise ValueError("Runtime scene contains a non-opaque shader-node name")
        _assert_no_properties(node, kind="shader-node")
        for socket in (*node.inputs, *node.outputs):
            _assert_no_properties(socket, kind="shader-socket")


def _scrub_node_tree(tree: Any, prefix: str) -> None:
    _strip_properties(tree)
    for index, node in enumerate(sorted(tree.nodes, key=lambda item: item.name)):
        node.name = f"{prefix}_{index:06d}"
        node.label = ""
        _strip_properties(node)
        for socket in (*node.inputs, *node.outputs):
            _strip_properties(socket)


def _object_audit(obj: Any, *, audit_names: bool) -> None:
    if obj.type not in _ALLOWED_OBJECT_TYPES:
        raise ValueError(f"Runtime scene contains unsupported object type {obj.type}")
    if audit_names and OPAQUE_NAME.fullmatch(obj.name) is None:
        raise ValueError("Runtime scene contains a non-opaque object name")
    _assert_local(obj, kind="object")
    _assert_static(obj, kind="object")
    _assert_no_properties(obj, kind="object")
    if len(obj.modifiers):
        raise ValueError("Runtime scene contains a geometry modifier")
    if len(obj.constraints):
        raise ValueError("Runtime scene contains an object constraint")
    if len(obj.vertex_groups):
        raise ValueError("Runtime scene contains named vertex groups")
    if len(obj.particle_systems):
        raise ValueError("Runtime scene contains a particle system")
    if obj.parent is not None or obj.instance_collection is not None or obj.instance_type != "NONE":
        raise ValueError("Runtime scene contains parenting or collection instancing")
    _finite((*obj.location, *obj.rotation_euler, *obj.scale), kind="object transform")


def _mesh_audit(mesh: Any, *, audit_names: bool) -> None:
    if audit_names and OPAQUE_NAME.fullmatch(mesh.name) is None:
        raise ValueError("Runtime scene contains a non-opaque mesh name")
    _assert_local(mesh, kind="mesh")
    _assert_static(mesh, kind="mesh")
    _assert_no_properties(mesh, kind="mesh")
    if mesh.shape_keys is not None:
        raise ValueError("Runtime scene contains mesh shape keys")
    if len(mesh.uv_layers) or len(mesh.color_attributes) or len(mesh.skin_vertices):
        raise ValueError("Runtime scene contains custom mesh UV/color/skin data")
    custom_attributes = [
        attribute.name
        for attribute in mesh.attributes
        if attribute.name not in _SAFE_MESH_ATTRIBUTES
        and not attribute.name.startswith(".select_")
        and not attribute.name.startswith(".uv_select_")
    ]
    if custom_attributes:
        raise ValueError("Runtime scene contains custom named mesh attributes")
    for vertex in mesh.vertices:
        _finite(vertex.co, kind="mesh coordinate")
    for material in mesh.materials:
        if material is None or material not in bpy.data.materials[:]:
            raise ValueError("Runtime scene contains a missing or external material slot")


def _material_audit(material: Any, *, audit_names: bool) -> int:
    if audit_names and OPAQUE_NAME.fullmatch(material.name) is None:
        raise ValueError("Runtime scene contains a non-opaque material name")
    _assert_local(material, kind="material")
    _assert_static(material, kind="material")
    _assert_no_properties(material, kind="material")
    _finite(material.diffuse_color, kind="material color")
    if material.use_nodes:
        if material.node_tree is None:
            raise ValueError("Runtime material is missing its numeric node tree")
        _node_tree_audit(
            material.node_tree,
            allowed_node_types=_ALLOWED_MATERIAL_NODES,
            audit_names=audit_names,
        )
        return 1
    return 0


def _light_audit(light: Any, *, audit_names: bool) -> int:
    if audit_names and OPAQUE_NAME.fullmatch(light.name) is None:
        raise ValueError("Runtime scene contains a non-opaque light name")
    _assert_local(light, kind="light")
    _assert_static(light, kind="light")
    _assert_no_properties(light, kind="light")
    if light.type not in _ALLOWED_LIGHT_TYPES:
        raise ValueError("Runtime scene contains an unsupported light configuration")
    _finite((*light.color, light.energy), kind="light")
    if light.node_tree is not None:
        _node_tree_audit(
            light.node_tree,
            allowed_node_types=_ALLOWED_LIGHT_NODES,
            audit_names=audit_names,
        )
        return 1
    return 0


def _world_audit(world: Any, *, audit_names: bool) -> int:
    if audit_names and OPAQUE_NAME.fullmatch(world.name) is None:
        raise ValueError("Runtime scene contains a non-opaque world name")
    _assert_local(world, kind="world")
    _assert_static(world, kind="world")
    _assert_no_properties(world, kind="world")
    _finite(world.color, kind="world color")
    if world.use_nodes:
        if world.node_tree is None:
            raise ValueError("Runtime world is missing its numeric node tree")
        _node_tree_audit(
            world.node_tree,
            allowed_node_types=_ALLOWED_WORLD_NODES,
            audit_names=audit_names,
        )
        return 1
    return 0


def audit_runtime_scene(*, audit_names: bool = True) -> dict[str, int | bool]:
    """Fail closed unless the loaded file is the narrow static v2 contract."""

    _reject_forbidden_datablocks()
    if len(bpy.data.scenes) != 1:
        raise ValueError("Runtime asset must contain exactly one scene")
    if len(bpy.data.cameras):
        raise ValueError("Runtime asset contains a persisted camera")
    scene = bpy.data.scenes[0]
    _assert_local(scene, kind="scene")
    _assert_static(scene, kind="scene")
    _assert_no_properties(scene, kind="scene")
    if audit_names and OPAQUE_NAME.fullmatch(scene.name) is None:
        raise ValueError("Runtime scene contains a non-opaque scene name")
    # The master scene collection, embedded owner node-tree names, and builtin
    # socket display names are fixed Blender schema strings rather than source
    # metadata; their mutable descendants and owner datablocks are audited.
    _assert_no_properties(scene.collection, kind="master scene collection")
    compositor_tree = getattr(scene, "compositing_node_group", None)
    if scene.camera is not None or scene.sequence_editor is not None or compositor_tree is not None:
        raise ValueError("Runtime scene contains a camera, sequencer, or compositor")
    if scene.rigidbody_world is not None or len(scene.timeline_markers):
        raise ValueError("Runtime scene contains simulation or timeline metadata")

    for obj in bpy.data.objects:
        _object_audit(obj, audit_names=audit_names)
    for mesh in bpy.data.meshes:
        _mesh_audit(mesh, audit_names=audit_names)
    light_node_trees = sum(
        _light_audit(light, audit_names=audit_names) for light in bpy.data.lights
    )
    node_trees = sum(
        _material_audit(material, audit_names=audit_names)
        for material in bpy.data.materials
    )
    node_trees += sum(_world_audit(world, audit_names=audit_names) for world in bpy.data.worlds)
    node_trees += light_node_trees
    for collection in bpy.data.collections:
        if audit_names and OPAQUE_NAME.fullmatch(collection.name) is None:
            raise ValueError("Runtime scene contains a non-opaque collection name")
        _assert_local(collection, kind="collection")
        _assert_no_properties(collection, kind="collection")
    for layer in scene.view_layers:
        if audit_names and OPAQUE_NAME.fullmatch(layer.name) is None:
            raise ValueError("Runtime scene contains a non-opaque view-layer name")
        _assert_no_properties(layer, kind="view-layer")
    ancillary = (
        bpy.data.workspaces,
        bpy.data.screens,
        bpy.data.window_managers,
        bpy.data.linestyles,
        bpy.data.palettes,
        bpy.data.brushes,
    )
    for values in ancillary:
        for value in values:
            if audit_names and OPAQUE_NAME.fullmatch(value.name) is None:
                raise ValueError("Runtime scene contains a non-opaque ancillary datablock name")
            _assert_local(value, kind="ancillary datablock")
            _assert_no_properties(value, kind="ancillary datablock")
    return {
        "object_names_opaque": bool(audit_names),
        "nested_names_opaque": bool(audit_names),
        "custom_properties_present": False,
        "external_assets_present": False,
        "automation_present": False,
        "animation_present": False,
        "unsupported_datablocks_present": False,
        "strict_nested_datablock_audit_passed": True,
        "mesh_objects": sum(obj.type == "MESH" for obj in bpy.data.objects),
        "light_objects": sum(obj.type == "LIGHT" for obj in bpy.data.objects),
        "materials": len(bpy.data.materials),
        "collections": len(bpy.data.collections),
        "node_trees": node_trees,
    }


def sanitize_source_scene() -> dict[str, int | bool]:
    """Strip permitted generation-only names and reject all active content."""

    _reject_forbidden_datablocks()
    if len(bpy.data.scenes) != 1:
        raise ValueError("Runtime export source must contain exactly one scene")
    # Source cameras are generation-only.  They are removed, never copied into
    # the runtime artifact, before the strict output audit.
    for obj in list(bpy.data.objects):
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)
        elif obj.type not in _ALLOWED_OBJECT_TYPES:
            raise ValueError(f"Runtime export source contains unsupported object type {obj.type}")
    for camera in list(bpy.data.cameras):
        bpy.data.cameras.remove(camera, do_unlink=True)
    try:
        bpy.ops.outliner.orphans_purge(do_recursive=True)
    except RuntimeError:
        pass

    # Custom metadata is inert but prohibited at runtime, so scrub it before
    # the strict audit. Active content (automation, animation, modifiers,
    # constraints, instancing) is never removed and therefore still fails.
    for mesh in bpy.data.meshes:
        for layer in list(mesh.uv_layers):
            mesh.uv_layers.remove(layer)
        for attribute in list(mesh.color_attributes):
            mesh.color_attributes.remove(attribute)
        _strip_properties(mesh)
    for obj in bpy.data.objects:
        _strip_properties(obj)
    for material in bpy.data.materials:
        _strip_properties(material)
        if material.use_nodes and material.node_tree is not None:
            _scrub_node_tree(material.node_tree, "d")
    for light in bpy.data.lights:
        _strip_properties(light)
        if light.node_tree is not None:
            _scrub_node_tree(light.node_tree, "d")
    for world in bpy.data.worlds:
        _strip_properties(world)
        if world.use_nodes and world.node_tree is not None:
            _scrub_node_tree(world.node_tree, "d")
    for scene in bpy.data.scenes:
        scene.camera = None
        if scene.sequence_editor is not None:
            scene.sequence_editor_clear()
        compositor_tree = getattr(scene, "compositing_node_group", None)
        if compositor_tree is not None:
            scene.compositing_node_group = None
        _strip_properties(scene)
        for layer in scene.view_layers:
            _strip_properties(layer)
    for collection in bpy.data.collections:
        _strip_properties(collection)

    # Automation and geometry generators are rejected before any names are
    # changed, so sanitization cannot conceal their presence.
    audit_runtime_scene(audit_names=False)

    _rename(bpy.data.objects, "r")
    _rename(bpy.data.meshes, "g")
    _rename(bpy.data.materials, "m")
    _rename(bpy.data.lights, "l")
    _rename(bpy.data.collections, "c")
    _rename(bpy.data.worlds, "w")
    _rename(bpy.data.scenes, "s")
    _rename(bpy.data.workspaces, "k")
    _rename(bpy.data.screens, "h")
    _rename(bpy.data.window_managers, "j")
    _rename(bpy.data.linestyles, "e")
    _rename(bpy.data.palettes, "p")
    _rename(bpy.data.brushes, "b")
    scene = bpy.data.scenes[0]
    for index, layer in enumerate(scene.view_layers):
        layer.name = f"v_{index:06d}"
        _strip_properties(layer)
    for material in bpy.data.materials:
        _strip_properties(material)
        if material.use_nodes and material.node_tree is not None:
            _scrub_node_tree(material.node_tree, "d")
    for light in bpy.data.lights:
        if light.node_tree is not None:
            _scrub_node_tree(light.node_tree, "d")
    for world in bpy.data.worlds:
        _strip_properties(world)
        if world.use_nodes and world.node_tree is not None:
            _scrub_node_tree(world.node_tree, "d")
    for obj in bpy.data.objects:
        _strip_properties(obj)
        obj.hide_render = False
        obj.hide_viewport = False
    scene.camera = None
    scene.render.filepath = ""
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    _strip_properties(scene)
    for collection in bpy.data.collections:
        _strip_properties(collection)
    bpy.context.preferences.filepaths.save_version = 0
    return audit_runtime_scene(audit_names=True)


def validate_manifest(manifest: object, *, scene_id: str, asset: Path) -> dict[str, Any]:
    if not isinstance(manifest, dict) or set(manifest) != set(MANIFEST_KEYS):
        raise ValueError("Runtime scene manifest fields changed")
    required_false = (
        "custom_properties_present",
        "external_assets_present",
        "automation_present",
        "animation_present",
        "unsupported_datablocks_present",
    )
    if (
        manifest.get("schema") != RUNTIME_SCENE_SCHEMA
        or manifest.get("scene_id") != scene_id
        or manifest.get("asset_file") != asset.name
        or manifest.get("object_names_opaque") is not True
        or manifest.get("nested_names_opaque") is not True
        or manifest.get("strict_nested_datablock_audit_passed") is not True
        or any(manifest.get(field) is not False for field in required_false)
    ):
        raise ValueError("Runtime scene manifest is not a strict v2 attestation")
    expected_hash = manifest.get("asset_sha256")
    if (
        not isinstance(expected_hash, str)
        or SHA256.fullmatch(expected_hash) is None
        or file_sha256(asset) != expected_hash
    ):
        raise ValueError("Runtime scene asset hash differs from its manifest")
    for field in ("mesh_objects", "light_objects", "materials", "collections", "node_trees"):
        value = manifest.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Runtime scene manifest contains an invalid numeric count")
    if int(manifest["mesh_objects"]) < 1:
        raise ValueError("Runtime scene manifest contains no mesh geometry")
    return manifest


__all__ = [
    "MANIFEST_KEYS",
    "RUNTIME_SCENE_SCHEMA",
    "audit_runtime_scene",
    "canonical_sha256",
    "file_sha256",
    "sanitize_source_scene",
    "validate_manifest",
]
