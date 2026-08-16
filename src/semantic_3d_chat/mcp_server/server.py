"""Official MCP Python SDK wrapper around the tested numerical robot actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import (
    PROJECT_ROOT,
    artifact_root,
    load_config,
    project_path,
    reports_root,
)
from semantic_3d_chat.robot.simulator import EmbodiedCameraSimulator
from semantic_3d_chat.robot.tools import tool_schemas

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
OpaqueSceneId = Annotated[str, StringConstraints(pattern=r"^scene_[0-9]{6}$")]
OpaqueObservationId = Annotated[str, StringConstraints(pattern=r"^o_[0-9]{6}$")]
ProtocolErrorCode = Annotated[str, StringConstraints(pattern=r"^E_[A-Z0-9_]+$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

_SCENE_ID = re.compile(r"scene_[0-9]{6}")
_OPAQUE_ASSET_FILE = re.compile(r"[a-z]_[0-9]{6}\.blend")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PREFLIGHT_SCHEMA = "semantic_3d_chat.embodied_mcp_preflight.v1"
_PROTECTED_RUNTIME_COMPONENTS = frozenset(
    {
        "experiments",
        "features",
        "oracle",
        "predictions",
        "qa",
        "questions",
        "rendered",
        "scorer",
        "scorer_only",
        "scorer-only",
        "training",
    }
)
# Runtime implementation code legitimately imports
# ``src/semantic_3d_chat/training/checkpointing.py`` for the checkpoint schema.
# Data/config paths still reject a ``training`` component, while the process
# audit blocks only the configured training-data roots so source imports remain
# possible.
_PROTECTED_AUDIT_COMPONENTS = _PROTECTED_RUNTIME_COMPONENTS - {"training"}
_V54_BASE_CHECKPOINT_ID = (
    "7c3e679702ccd204fa4d7ae4077b065f3d7a7fe36df7dbc45492d67566e97f59"
)
_RUNTIME_ASSET_MANIFEST_FIELDS = frozenset(
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
_BASE_RUNTIME_FILES = frozenset({"adapter.safetensors", "runtime_metadata.json"})
_BASE_CHECKPOINT_FILES = _BASE_RUNTIME_FILES | {"metadata.json"}
_EMBODIED_CONFIG_TOP_LEVEL = frozenset(
    {
        "seed",
        "runtime",
        "paths",
        "scene",
        "vision",
        "scene_encoder",
        "language",
        "training",
        "render",
        "mapping",
        "robot",
    }
)
_EMBODIED_VISION_FIELDS = frozenset(
    {
        "backend",
        "model_id",
        "revision",
        "input_size",
        "middle_layer",
        "late_layer",
        "feature_mode",
        "aligned_method",
        "dtype",
        "storage_dtype",
        "batch_size",
    }
)
_EMBODIED_RENDER_FIELDS = frozenset(
    {"resolution", "horizontal_fov_degrees", "engine", "samples"}
)
_EMBODIED_MAPPING_FIELDS = frozenset(
    {
        "voxel_size_m",
        "depth_min_m",
        "depth_max_m",
        "pixel_stride",
        "max_voxels",
        "confidence_distance_scale_m",
    }
)
_EMBODIED_ROBOT_FIELDS = frozenset(
    {
        "auto_scan_after_motion",
        "radius_m",
        "camera_height_m",
        "initial_position_xy_m",
        "initial_body_yaw_degrees",
        "max_move_m",
        "max_move_to_m",
        "max_turn_degrees",
        "max_look_delta_degrees",
        "max_camera_yaw_offset_degrees",
        "max_pitch_degrees",
        "collision_z_min_m",
        "collision_z_max_m",
        "surface_padding_m",
        "scan_resolution",
        "scan_horizontal_fov_degrees",
        "scan_depth_min_m",
        "scan_depth_max_m",
        "history_length",
        "state_token_count",
        "state_encoder_hidden_dim",
        "face_alignment_deadband_degrees",
        "face_alignment_stalled_turn_degrees",
        "approach_heading_deadband_degrees",
        "approach_target_standoff_m",
        "approach_minimum_progress_m",
        "approach_minimum_safe_step_m",
    }
)


class ToolResponse(BaseModel):
    """Common structured output containing protocol and numerical state only."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    success: bool
    error_code: ProtocolErrorCode | None
    scene_id: OpaqueSceneId
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)]
    scene_version: Annotated[int, Field(ge=0)]
    position_m: Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]
    camera_position_m: Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]
    body_yaw_degrees: FiniteFloat
    camera_yaw_degrees: FiniteFloat
    pitch_degrees: FiniteFloat
    linear_velocity_xy_m: Annotated[list[FiniteFloat], Field(min_length=2, max_length=2)]
    angular_velocity_degrees: FiniteFloat
    collision: bool
    last_movement_delta_m: Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]
    distance_moved: Annotated[FiniteFloat, Field(ge=0.0)]
    turn_degrees: FiniteFloat
    scan_coverage: Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
    scan_count: Annotated[int, Field(ge=0)]
    visible_voxels: Annotated[int, Field(ge=0)]
    valid_depth_pixels: Annotated[int, Field(ge=0)]
    observation_id: OpaqueObservationId | None
    clearance_m: FiniteFloat | None
    action_count: Annotated[int, Field(ge=0)]
    stopped: bool
    map_sha256: Sha256 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    binding_schema: Literal[
        "semantic_3d_chat.scene_prefix_binding.v1",
        "semantic_3d_chat.scene_prefix_binding.v2",
    ] | None = Field(
        default=None,
        alias="schema",
        exclude_if=lambda value: value is None,
    )
    map_version: Annotated[int, Field(ge=0)] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    scene_prefix_sha256: Sha256 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    scene_control_signature_sha256: Sha256 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    source_voxels: Annotated[int, Field(ge=1)] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    processed_voxels: Annotated[int, Field(ge=1)] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    binding_sha256: Sha256 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    active_prefix_sha256: Sha256 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    robot_state_sha256: Sha256 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    robot_tokens_sha256: Sha256 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    robot_state_encoder_sha256: Sha256 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    active_binding_sha256: Sha256 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


def _response(payload: dict[str, Any]) -> ToolResponse:
    return ToolResponse.model_validate(payload)


def _runtime_config(runtime: Any) -> dict[str, Any]:
    config = getattr(runtime, "config", None)
    if config is None:
        simulator = getattr(runtime, "simulator", None)
        config = getattr(simulator, "config", None)
    if not isinstance(config, dict):
        raise TypeError("MCP robot runtime must expose a numerical action configuration")
    return config


def _harden_input_schemas(server: MCPServer[None], runtime: Any) -> None:
    """Apply the direct protocol's strict, configured schemas to MCP tools.

    MCP SDK 2.0 derives function-argument models with Pydantic's default
    ``extra='ignore'`` policy. That would advertise no action bounds and would
    silently discard unexpected fields. The project pins this SDK version, so
    we explicitly forbid extras on each generated argument model and expose the
    same configured limits used by the already-tested direct protocol. Runtime
    methods still validate every value independently before changing state.
    """

    schemas = {item["name"]: item["inputSchema"] for item in tool_schemas(_runtime_config(runtime))}
    for name, input_schema in schemas.items():
        registered = server._tool_manager.get_tool(name)
        if registered is None:  # pragma: no cover - programming error during registration
            raise RuntimeError(f"MCP tool was not registered: {name}")
        argument_model = registered.fn_metadata.arg_model
        argument_model.model_config["extra"] = "forbid"
        argument_model.model_rebuild(force=True)
        registered.parameters = {
            **input_schema,
            "title": f"{name}Arguments",
        }


def build_server(simulator: Any) -> MCPServer[None]:
    """Build an in-process server so schemas and actions can be tested directly."""

    server: MCPServer[None] = MCPServer(
        "semantic-3d-robot",
        version="0.1.0",
        description="Bounded numerical embodied-camera actions over continuous scene memory.",
        instructions=(
            "Tool results contain only protocol status, opaque identifiers, numerical pose, "
            "collision state, scan coverage, and scene version."
        ),
    )

    @server.tool(structured_output=True)
    def get_robot_state() -> ToolResponse:
        """Return the current numerical robot and camera state."""

        return _response(simulator.get_robot_state())

    @server.tool(structured_output=True)
    def look(yaw_delta_degrees: float, pitch_delta_degrees: float) -> ToolResponse:
        """Rotate the camera within the configured per-call and total limits."""

        return _response(simulator.look(yaw_delta_degrees, pitch_delta_degrees))

    @server.tool(structured_output=True)
    def turn(angle_degrees: float) -> ToolResponse:
        """Rotate the robot body by one bounded angle."""

        return _response(simulator.turn(angle_degrees))

    @server.tool(structured_output=True)
    def move_forward(distance_meters: float) -> ToolResponse:
        """Attempt one bounded forward translation with swept collision checking."""

        return _response(simulator.move_forward(distance_meters))

    @server.tool(structured_output=True)
    def move_backward(distance_meters: float) -> ToolResponse:
        """Attempt one bounded backward translation with swept collision checking."""

        return _response(simulator.move_backward(distance_meters))

    @server.tool(structured_output=True)
    def move_to(x: float, y: float) -> ToolResponse:
        """Attempt one bounded straight-line movement to a numerical world coordinate."""

        return _response(simulator.move_to(x, y))

    @server.tool(structured_output=True)
    def scan() -> ToolResponse:
        """Capture numerical RGB-D, update observation counts, and advance scene version."""

        return _response(simulator.scan())

    @server.tool(structured_output=True)
    def stop() -> ToolResponse:
        """Stop the current episode; reset_scene starts a new one."""

        return _response(simulator.stop())

    @server.tool(structured_output=True)
    def reset_scene(scene_id: str, seed: int) -> ToolResponse:
        """Reset to an opaque scene ID and deterministic numerical start state."""

        return _response(simulator.reset_scene(scene_id, seed))

    _harden_input_schemas(server, simulator)
    return server


def _serve(
    server: MCPServer[None],
    *,
    transport: str,
    host: str,
    port: int,
) -> None:
    if transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=host, port=port)


def _save_audit_durably(audit: FileAccessAudit, destination: str | Path) -> None:
    """Atomically persist the complete lifetime audit after serving stops."""

    output = Path(destination).expanduser()
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        audit.save(temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        directory_descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _runtime_audit() -> FileAccessAudit:
    """Deny environmental supervision while allowing implementation imports."""

    return FileAccessAudit(
        [
            PROJECT_ROOT / "data" / "oracle",
            PROJECT_ROOT / "data" / "qa",
            PROJECT_ROOT / "data_gemma4" / "training",
            PROJECT_ROOT / "reports" / "gemma4" / "scorer_only",
            PROJECT_ROOT / "reports" / "gemma4" / "scorer",
            PROJECT_ROOT / "reports" / "gemma4" / "questions",
            PROJECT_ROOT / "reports" / "gemma4" / "predictions",
        ],
        forbidden_component_names=_PROTECTED_AUDIT_COMPONENTS,
        block_forbidden=True,
    )


def _extend_runtime_audit(audit: FileAccessAudit, config: dict[str, Any]) -> None:
    """Apply the same deny policy to configured derived-data roots."""

    roots = [artifact_root(config, kind).resolve() for kind in ("oracle", "qa")]
    roots.extend(
        [
            artifact_root(config, "checkpoints").resolve().parent / "training",
            reports_root(config).resolve() / "scorer_only",
        ]
    )
    for root in roots:
        if root not in audit.forbidden_roots:
            audit.forbidden_roots.append(root)


def _safe_runtime_path(
    path: str | Path,
    *,
    purpose: str,
    expected: Literal["file", "directory"] | None = None,
) -> Path:
    """Resolve one runtime path without following a symlink or supervision root."""

    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    unresolved = Path(os.path.abspath(rooted))
    current = Path(unresolved.anchor)
    for component in unresolved.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{purpose} cannot contain symbolic-link components")
    if _PROTECTED_RUNTIME_COMPONENTS.intersection(
        part.casefold() for part in unresolved.parts
    ):
        raise ValueError(f"{purpose} cannot be stored with environmental supervision")
    if expected == "file" and not unresolved.is_file():
        raise FileNotFoundError(f"{purpose} is unavailable: {unresolved}")
    if expected == "directory" and not unresolved.is_dir():
        raise FileNotFoundError(f"{purpose} is unavailable: {unresolved}")
    return unresolved


def _strict_json_object(
    path: Path,
    *,
    purpose: str,
    audit: FileAccessAudit,
) -> dict[str, Any]:
    """Read one audited JSON object while rejecting duplicate fields."""

    audit.record(path)

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{purpose} repeats field {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{purpose} is invalid JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{purpose} must be a JSON object")
    return value


def _audited_sha256(path: Path, audit: FileAccessAudit) -> str:
    audit.record(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _forbid_adjacent_training_metadata(
    checkpoint: str | Path,
    audit: FileAccessAudit,
) -> None:
    """Block the training/evaluation sidecar beside an inference checkpoint."""

    root = _safe_runtime_path(
        checkpoint,
        purpose="semantic adapter checkpoint",
    )
    metadata = (root / "metadata.json").resolve()
    if metadata not in audit.forbidden_roots:
        audit.forbidden_roots.append(metadata)


def _validate_configured_action_protocol(config: dict[str, Any]) -> dict[str, Any]:
    """Validate every bound without constructing a stateful simulator."""

    schemas = tool_schemas(config)
    names = [item["name"] for item in schemas]
    if len(names) != 9 or len(set(names)) != len(names):
        raise ValueError("MCP action protocol must contain nine unique bounded tools")
    return {
        "tool_count": len(names),
        "tools": names,
        "strict_input_schemas": all(
            item.get("inputSchema", {}).get("additionalProperties") is False
            for item in schemas
        ),
    }


def _validate_semantic_config_isolation(config: dict[str, Any]) -> None:
    """Reject experiment/evaluation YAML before semantic runtime construction."""

    observed_top_level = {key for key in config if not key.startswith("_")}
    if observed_top_level != set(_EMBODIED_CONFIG_TOP_LEVEL):
        raise ValueError(
            "Semantic MCP configuration is not the isolated embodied-runtime schema"
        )
    exact_nested_fields = {
        "vision": _EMBODIED_VISION_FIELDS,
        "render": _EMBODIED_RENDER_FIELDS,
        "mapping": _EMBODIED_MAPPING_FIELDS,
        "robot": _EMBODIED_ROBOT_FIELDS,
    }
    for field, expected in exact_nested_fields.items():
        value = config.get(field)
        if not isinstance(value, dict) or set(value) != set(expected):
            raise ValueError(f"Semantic MCP {field} configuration fields changed")
    scene = config.get("scene")
    if not isinstance(scene, dict) or set(scene) != {"room_size_m"}:
        raise ValueError("Semantic MCP scene config may contain only numeric room bounds")
    # Reuse the strict standalone validator for every field shared with chat.
    # The embodied file adds only allowlisted render/mapping/robot and dense-
    # vision settings above; no evaluation, split, label, or oracle surface is
    # admitted to this process.
    from semantic_3d_chat.chat.runtime_config import validate_runtime_config

    runtime_view = {
        key: config[key]
        for key in (
            "runtime",
            "paths",
            "scene",
            "scene_encoder",
            "language",
            "training",
        )
    }
    runtime_view["vision"] = {
        key: config["vision"][key]
        for key in ("backend", "model_id", "revision")
    }
    validate_runtime_config(runtime_view)


def _validate_embodied_render_contract(config: dict[str, Any]) -> None:
    render = config.get("render")
    mapping = config.get("mapping")
    if not isinstance(render, dict) or not isinstance(mapping, dict):
        raise TypeError("Semantic MCP requires render and mapping configurations")
    resolution = render.get("resolution")
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 2 for value in resolution)
    ):
        raise ValueError("Semantic MCP render resolution must contain two positive integers")
    fov = render.get("horizontal_fov_degrees")
    samples = render.get("samples")
    depth_max = mapping.get("depth_max_m")
    if (
        isinstance(fov, bool)
        or not isinstance(fov, (int, float))
        or not math.isfinite(float(fov))
        or not 1.0 <= float(fov) < 179.0
    ):
        raise ValueError("Semantic MCP horizontal field of view is invalid")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("Semantic MCP render sample count must be positive")
    if (
        isinstance(depth_max, bool)
        or not isinstance(depth_max, (int, float))
        or not math.isfinite(float(depth_max))
        or float(depth_max) <= 0.0
    ):
        raise ValueError("Semantic MCP maximum depth must be finite and positive")
    engine = render.get("engine")
    if not isinstance(engine, str) or not engine:
        raise ValueError("Semantic MCP render engine must be a nonempty identifier")


def _validate_navigation_safety_metadata(config: dict[str, Any]) -> None:
    """Validate allowlisted convergence settings without using them as MCP semantics."""

    robot = config.get("robot")
    if not isinstance(robot, dict):
        raise TypeError("Semantic MCP requires a robot configuration")
    numeric_fields = (
        "face_alignment_deadband_degrees",
        "face_alignment_stalled_turn_degrees",
        "approach_heading_deadband_degrees",
        "approach_target_standoff_m",
        "approach_minimum_progress_m",
        "approach_minimum_safe_step_m",
    )
    values: dict[str, float] = {}
    for field in numeric_fields:
        value = robot.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"Semantic MCP {field} must be finite and positive")
        values[field] = float(value)
    maximum_turn = float(robot["max_turn_degrees"])
    maximum_move = float(robot["max_move_m"])
    if any(
        values[field] >= maximum_turn
        for field in (
            "face_alignment_deadband_degrees",
            "face_alignment_stalled_turn_degrees",
            "approach_heading_deadband_degrees",
        )
    ):
        raise ValueError("Semantic MCP navigation angular safety metadata exceeds turn bound")
    if any(
        values[field] > maximum_move
        for field in ("approach_minimum_progress_m", "approach_minimum_safe_step_m")
    ):
        raise ValueError("Semantic MCP navigation distance safety metadata exceeds move bound")


def _map_preflight(
    config: dict[str, Any],
    scene_id: str,
    *,
    audit: FileAccessAudit,
    path: str | Path | None = None,
    require_semantics: bool,
) -> dict[str, Any]:
    """Validate map structure without materializing the high-dimensional matrix."""

    raw_path = path or project_path(config, "maps", scene_id, "voxel_map.npz")
    source = _safe_runtime_path(
        raw_path,
        purpose="sanitized numerical scene map",
        expected="file",
    )
    audit.record(source)
    from semantic_3d_chat.scene_encoder.map_io import validate_runtime_map_sidecars

    validate_runtime_map_sidecars(source)
    with np.load(source, allow_pickle=False) as archive:
        required = {"centers_world", "mean_rgb", "observation_count"}
        if require_semantics:
            required.update(
                {"semantic_features", "normal", "confidence", "metadata_json"}
            )
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Numerical map is missing fields: {sorted(missing)}")
        centers = archive["centers_world"]
        rgb = archive["mean_rgb"]
        observations = archive["observation_count"]
        count = int(centers.shape[0]) if centers.ndim == 2 else 0
        if (
            count < 1
            or centers.shape != (count, 3)
            or rgb.shape != (count, 3)
            or observations.shape != (count,)
            or centers.dtype.kind not in {"f", "i", "u"}
            or rgb.dtype.kind not in {"f", "i", "u"}
            or observations.dtype.kind not in {"i", "u"}
            or not np.isfinite(centers).all()
            or not np.isfinite(rgb).all()
            or np.any(observations < 0)
        ):
            raise ValueError("Numerical map has invalid geometry, RGB, or observation arrays")
        feature_dim: int | None = None
        header: dict[str, Any] | None = None
        if "metadata_json" in archive.files:
            raw_header = archive["metadata_json"]
            serialized = raw_header.item()
            if isinstance(serialized, bytes):
                serialized = serialized.decode("utf-8")
            header_value = json.loads(str(serialized))
            if not isinstance(header_value, dict):
                raise TypeError("Numerical map header must be a JSON object")
            header = header_value
            declared_count = header.get("occupied_voxels")
            declared_dim = header.get("feature_dim")
            if declared_count != count:
                raise ValueError("Numerical map header voxel count differs from geometry")
            if isinstance(declared_dim, bool) or not isinstance(declared_dim, int) or declared_dim < 1:
                raise ValueError("Numerical map header feature dimension is invalid")
            feature_dim = declared_dim
            metadata = header.get("metadata")
            if (
                isinstance(metadata, dict)
                and "scene_id" in metadata
                and metadata.get("scene_id") != scene_id
            ):
                raise ValueError("Numerical map header belongs to another opaque scene")
    return {
        "path": str(source),
        "sha256": _audited_sha256(source, audit),
        "size_bytes": source.stat().st_size,
        "occupied_voxels": count,
        "feature_dim": feature_dim,
        "semantic_payload_required": require_semantics,
        "high_dimensional_features_loaded": False,
        "runtime_sidecars_validated": True,
    }


def _runtime_asset_preflight(
    scene_id: str,
    path: str | Path,
    *,
    audit: FileAccessAudit,
) -> dict[str, Any]:
    """Authenticate the sanitized Blender payload without importing or starting Blender."""

    asset = _safe_runtime_path(
        path,
        purpose="sanitized runtime scene asset",
        expected="file",
    )
    if _OPAQUE_ASSET_FILE.fullmatch(asset.name) is None:
        raise ValueError("Sanitized runtime scene asset filename must be opaque")
    manifest_path = _safe_runtime_path(
        asset.with_suffix(".json"),
        purpose="sanitized runtime scene manifest",
        expected="file",
    )
    manifest = _strict_json_object(
        manifest_path,
        purpose="sanitized runtime scene manifest",
        audit=audit,
    )
    if set(manifest) != set(_RUNTIME_ASSET_MANIFEST_FIELDS):
        raise ValueError("Runtime scene manifest fields changed")
    unsafe_flags = (
        "custom_properties_present",
        "external_assets_present",
        "automation_present",
        "animation_present",
        "unsupported_datablocks_present",
    )
    if (
        manifest.get("schema") != "semantic_3d_chat.runtime_scene.v2"
        or manifest.get("scene_id") != scene_id
        or manifest.get("asset_file") != asset.name
        or manifest.get("object_names_opaque") is not True
        or manifest.get("nested_names_opaque") is not True
        or manifest.get("strict_nested_datablock_audit_passed") is not True
        or any(manifest.get(field) is not False for field in unsafe_flags)
    ):
        raise ValueError("Runtime scene manifest is not a strict v2 attestation")
    counts: dict[str, int] = {}
    for field in ("mesh_objects", "light_objects", "materials", "collections", "node_trees"):
        value = manifest.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Runtime scene manifest contains an invalid numerical count")
        counts[field] = value
    if counts["mesh_objects"] < 1:
        raise ValueError("Runtime scene manifest contains no mesh geometry")
    expected_hash = manifest.get("asset_sha256")
    observed_hash = _audited_sha256(asset, audit)
    if (
        not isinstance(expected_hash, str)
        or _SHA256.fullmatch(expected_hash) is None
        or observed_hash != expected_hash
    ):
        raise ValueError("Runtime scene asset hash differs from its manifest")
    return {
        "asset": str(asset),
        "manifest": str(manifest_path),
        "asset_sha256": observed_hash,
        "asset_size_bytes": asset.stat().st_size,
        "strict_manifest": True,
        "blender_loaded": False,
        **counts,
    }


def _base_checkpoint_preflight(
    checkpoint: str | Path,
    config: dict[str, Any],
    *,
    semantic_dim: int,
    audit: FileAccessAudit,
) -> dict[str, Any]:
    """Bind sanitized adapter inputs while never opening training metadata."""

    root = _safe_runtime_path(
        checkpoint,
        purpose="semantic adapter checkpoint",
        expected="directory",
    )
    inventory = {item.name for item in root.iterdir()}
    if inventory not in {_BASE_RUNTIME_FILES, _BASE_CHECKPOINT_FILES}:
        raise ValueError(
            "Semantic adapter checkpoint must contain the exact runtime files, "
            "optionally plus offline training metadata"
        )
    adapter = root / "adapter.safetensors"
    runtime_metadata_path = root / "runtime_metadata.json"
    training_metadata = (root / "metadata.json").resolve()
    for member in (adapter, runtime_metadata_path):
        if member.is_symlink() or not member.is_file():
            raise ValueError("Semantic adapter checkpoint members must be regular files")
    if training_metadata.exists() and (
        training_metadata.is_symlink() or not training_metadata.is_file()
    ):
        raise ValueError("Semantic adapter training metadata is unsafe")
    if training_metadata.exists() and training_metadata not in audit.forbidden_roots:
        audit.forbidden_roots.append(training_metadata)
    metadata = _strict_json_object(
        runtime_metadata_path,
        purpose="semantic adapter runtime metadata",
        audit=audit,
    )
    from semantic_3d_chat.training.checkpointing import (
        validate_runtime_checkpoint_metadata,
    )

    validate_runtime_checkpoint_metadata(metadata)
    scene_encoder = config.get("scene_encoder")
    language = config.get("language")
    if not isinstance(scene_encoder, dict) or not isinstance(language, dict):
        raise TypeError("Semantic MCP configuration lacks scene/language contracts")
    expected = {
        "semantic_dim": semantic_dim,
        "language_hidden_dim": int(scene_encoder["language_aligned_tail_dim"]),
        "language_model_id": str(language["model_id"]),
        "language_revision": str(language["revision"]),
        "scene_latents": int(scene_encoder["global_latents"]),
        "scene_model_dim": int(scene_encoder["model_dim"]),
    }
    mismatches = {
        field: {"checkpoint": metadata.get(field), "runtime": value}
        for field, value in expected.items()
        if metadata.get(field) != value
    }
    if metadata.get("question_dependent_scene_processing") is not False:
        mismatches["question_dependent_scene_processing"] = {
            "checkpoint": metadata.get("question_dependent_scene_processing"),
            "runtime": False,
        }
    if mismatches:
        raise ValueError(f"Semantic adapter checkpoint contract mismatch: {mismatches}")
    adapter_hash = _audited_sha256(adapter, audit)
    runtime_metadata_hash = _audited_sha256(runtime_metadata_path, audit)
    # The current Gemma 4 V54 base has a separately frozen release identity.
    # Enforce it whenever that exact model/revision/prefix contract is selected.
    exact_v54 = (
        language.get("model_id") == "google/gemma-4-E2B-it"
        and language.get("revision") == "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
        and scene_encoder.get("global_latents") == 256
    )
    if exact_v54:
        from semantic_3d_chat.chat.fixed_prefix_ple_reader_runtime import (
            validate_v54_checkpoint,
        )

        validate_v54_checkpoint(root, audit=audit)
    return {
        "path": str(root),
        "adapter_sha256": adapter_hash,
        "runtime_metadata_sha256": runtime_metadata_hash,
        "adapter_size_bytes": adapter.stat().st_size,
        "semantic_dim": semantic_dim,
        "language_hidden_dim": expected["language_hidden_dim"],
        "scene_latents": expected["scene_latents"],
        "question_dependent_scene_processing": False,
        "training_metadata_opened": False,
        "exact_v54_release_authenticated": exact_v54,
        "checkpoint_identity_sha256": (
            _V54_BASE_CHECKPOINT_ID if exact_v54 else None
        ),
    }


def _robot_state_checkpoint_preflight(
    checkpoint: str | Path,
    *,
    hidden_size: int,
    audit: FileAccessAudit,
) -> dict[str, Any]:
    from semantic_3d_chat.robot.state_checkpoint import load_robot_state_checkpoint

    root = _safe_runtime_path(
        checkpoint,
        purpose="numeric robot-state checkpoint",
        expected="directory",
    )
    _encoder, state_sha256, metadata = load_robot_state_checkpoint(
        root,
        expected_output_dim=hidden_size,
        device="cpu",
        audit=audit,
    )
    return {
        "path": str(root),
        "state_sha256": state_sha256,
        "output_dim": int(metadata["output_dim"]),
        "token_count": int(metadata["token_count"]),
        "numeric_inputs_only": True,
        "environmental_text_inputs": [],
        "task_trained": bool(metadata["task_trained"]),
    }


def _control_checkpoint_preflight(
    checkpoint: str | Path,
    runtime_config_path: str | Path,
    *,
    hidden_size: int,
    expected_base_checkpoint_sha256: str | None,
    audit: FileAccessAudit,
) -> dict[str, Any]:
    """Authenticate the optional small controller without loading Gemma weights."""

    from semantic_3d_chat.chat.question_control_runtime import _load_control_head
    from semantic_3d_chat.chat.runtime_config import (
        effective_runtime_config_sha256,
        load_runtime_config,
    )

    control_config = load_runtime_config(
        runtime_config_path,
        record_file=audit.record,
    )
    _control, metadata = _load_control_head(
        checkpoint,
        hidden_size=hidden_size,
        device=_torch_cpu_device(),
        audit=audit,
    )
    observed_runtime_hash = effective_runtime_config_sha256(control_config)
    if metadata.get("base_runtime_config_sha256") != observed_runtime_hash:
        raise ValueError("Question controller belongs to another runtime configuration")
    if (
        expected_base_checkpoint_sha256 is None
        or metadata.get("base_checkpoint_sha256")
        != expected_base_checkpoint_sha256
    ):
        raise ValueError("Question controller belongs to another base checkpoint")
    return {
        "path": str(_safe_runtime_path(checkpoint, purpose="question controller", expected="directory")),
        "architecture": metadata.get("architecture"),
        "weights_sha256": metadata.get("weights_sha256"),
        "runtime_config_sha256": observed_runtime_hash,
        "complete_scene_prefix_required": metadata.get("complete_scene_prefix_required"),
        "question_dependent_scene_retrieval": metadata.get(
            "question_dependent_scene_retrieval"
        ),
        "environmental_text_inputs": metadata.get("environmental_text_inputs"),
    }


def _torch_cpu_device() -> Any:
    """Import torch only for an explicitly requested small controller preflight."""

    import torch

    return torch.device("cpu")


def _persistent_map_preflight(
    config: dict[str, Any],
    scene_id: str,
    path: str | Path | None,
    *,
    audit: FileAccessAudit,
) -> dict[str, Any]:
    candidate = _safe_runtime_path(
        path or project_path(config, "robot", scene_id, "semantic_map.npz"),
        purpose="persistent semantic-map destination",
    )
    if candidate.exists():
        if not candidate.is_file():
            raise ValueError("Persistent semantic-map destination is not a regular file")
        return {
            "exists": True,
            **_map_preflight(
                config,
                scene_id,
                audit=audit,
                path=candidate,
                require_semantics=True,
            ),
        }
    if candidate.parent.exists() and not candidate.parent.is_dir():
        raise ValueError("Persistent semantic-map parent is not a directory")
    return {
        "path": str(candidate),
        "exists": False,
        "will_be_created_only_after_scan": True,
    }


def _scan_output_preflight(
    config: dict[str, Any],
    scene_id: str,
    path: str | Path | None,
) -> dict[str, Any]:
    candidate = _safe_runtime_path(
        path or project_path(config, "robot", scene_id, "scans"),
        purpose="sanitized RGB-D scan output",
    )
    if candidate.exists() and not candidate.is_dir():
        raise ValueError("Sanitized scan output destination is not a directory")
    if not candidate.exists() and candidate.parent.exists() and not candidate.parent.is_dir():
        raise ValueError("Sanitized scan output parent is not a directory")
    return {
        "path": str(candidate),
        "exists": candidate.is_dir(),
        "will_be_created_only_during_live_observation": not candidate.exists(),
    }


def _semantic_preflight(
    args: argparse.Namespace,
    config: dict[str, Any],
    *,
    audit: FileAccessAudit,
) -> dict[str, Any]:
    _validate_semantic_config_isolation(config)
    _validate_embodied_render_contract(config)
    _validate_navigation_safety_metadata(config)
    protocol = _validate_configured_action_protocol(config)
    base_map = _map_preflight(
        config,
        args.scene,
        audit=audit,
        require_semantics=True,
    )
    feature_dim = base_map.get("feature_dim")
    if isinstance(feature_dim, bool) or not isinstance(feature_dim, int):
        raise TypeError("Semantic map must declare its high-dimensional feature width")
    v96_candidate = (
        None
        if args.v96_candidate_bridge_hook is None
        else _v96_candidate_server_contract(args, audit=audit)
    )
    checkpoint = (
        _base_checkpoint_preflight(
            args.checkpoint,
            config,
            semantic_dim=feature_dim,
            audit=audit,
        )
        if v96_candidate is None
        else dict(v96_candidate["checkpoint"])
    )
    if checkpoint.get("semantic_dim") != feature_dim:
        raise ValueError("V96 candidate checkpoint semantic dimension differs from map")
    asset = _runtime_asset_preflight(args.scene, args.runtime_asset, audit=audit)
    hidden_size = int(checkpoint["language_hidden_dim"])
    robot_state = (
        None
        if args.robot_state_checkpoint is None
        else _robot_state_checkpoint_preflight(
            args.robot_state_checkpoint,
            hidden_size=hidden_size,
            audit=audit,
        )
    )
    control = None
    if args.control_checkpoint is not None:
        assert args.control_runtime_config is not None
        control = _control_checkpoint_preflight(
            args.control_checkpoint,
            args.control_runtime_config,
            hidden_size=hidden_size,
            expected_base_checkpoint_sha256=checkpoint.get(
                "checkpoint_identity_sha256"
            ),
            audit=audit,
        )
    return {
        "schema": _PREFLIGHT_SCHEMA,
        "phase": "embodied_mcp_preflight",
        "passed": True,
        "mode": (
            "semantic_continuous_map"
            if v96_candidate is None
            else "semantic_continuous_map_v96_explicit_candidate"
        ),
        "scene_id": args.scene,
        "config": str(Path(config["_config_path"]).resolve()),
        "base_map": base_map,
        "persistent_map": _persistent_map_preflight(
            config,
            args.scene,
            args.persistent_map,
            audit=audit,
        ),
        "scan_output": _scan_output_preflight(
            config,
            args.scene,
            args.scan_output_directory,
        ),
        "runtime_asset": asset,
        "base_checkpoint": checkpoint,
        "question_controller": control,
        "v96_explicit_candidate": (
            None if v96_candidate is None else v96_candidate["summary"]
        ),
        "robot_state_checkpoint": robot_state,
        "action_protocol": protocol,
        "loads_language_model": False,
        "loads_blender": False,
        "starts_transport": False,
        "changes_robot_or_map_state": False,
        "scene_prefix_computation_deferred_to_live_startup": True,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
    }


def _v96_candidate_server_contract(
    args: argparse.Namespace,
    *,
    audit: FileAccessAudit,
) -> dict[str, Any]:
    """Authenticate explicit V96 inputs without loading Gemma or a controller.

    Known-development evidence is read only by the isolated authorizer child.
    The MCP process receives a hash-only authorization and immediately blocks
    those evidence/config paths before it loads any runtime model.
    """

    from semantic_3d_chat.chat.runtime_config import (
        effective_runtime_config_sha256,
        load_runtime_config,
    )
    from semantic_3d_chat.chat.v96_explicit_candidate_cli import (
        load_v96_runtime_hook,
        run_isolated_v96_authorization,
    )
    from semantic_3d_chat.chat.v96_explicit_candidate_runtime import (
        _checkpoint_fingerprint,
        validate_v96_scene_memory_contract,
        validate_v96_v85_base_checkpoint_contract,
    )
    from semantic_3d_chat.robot.v96_candidate_refresh import (
        _checkpoint_fingerprint as compiler_checkpoint_fingerprint,
    )
    from semantic_3d_chat.robot.v96_candidate_refresh import (
        load_v96_candidate_mcp_hook,
        run_isolated_v96_release_verification,
    )
    from semantic_3d_chat.scene_encoder.v81_scene_memory_artifact import (
        FIXED_MEMORY_TOKENS,
        HIDDEN_SIZE,
        load_v81_scene_memory,
    )

    bridge_hook_path = _safe_runtime_path(
        args.v96_candidate_bridge_hook,
        purpose="V96 explicit-candidate MCP hook",
        expected="file",
    )
    audit.record(bridge_hook_path)
    bridge_hook = load_v96_candidate_mcp_hook(bridge_hook_path)
    audit.record(bridge_hook.candidate_hook)
    candidate_hook = load_v96_runtime_hook(bridge_hook.candidate_hook)
    if candidate_hook.path != bridge_hook.candidate_hook:
        raise ValueError("V96 MCP bridge resolved another candidate hook")

    authorization = run_isolated_v96_authorization(
        candidate_hook.authorization_config
    )
    # Known-development PASS alone is deliberately insufficient for embodied
    # startup. The separate model-free child must also authenticate the sealed
    # deferred-final result and the six-scene oracle-unavailable release smoke.
    release_verification = run_isolated_v96_release_verification()
    for forbidden in (
        authorization.authorization_config_path,
        authorization.final_score_path,
        authorization.evidence_path,
    ):
        blocked = Path(forbidden).resolve()
        if blocked not in audit.forbidden_roots:
            audit.forbidden_roots.append(blocked)
    if (
        authorization.pass_evidence_authenticated is not True
        or authorization.known_development_gate_passed is not True
        or authorization.runtime_promotion_authorized is not False
        or authorization.explicit_candidate_flag_required is not True
        or Path(authorization.runtime_config_path).resolve()
        != candidate_hook.runtime_config
        or release_verification["candidate_fingerprint_sha256"]
        != authorization.candidate_fingerprint_sha256
        or release_verification["v95_state_sha256"]
        != authorization.v95_state_sha256
        or release_verification["v96_state_sha256"]
        != authorization.v96_state_sha256
    ):
        raise ValueError(
            "V96 MCP bridge lacks a mutually bound deferred-final release PASS"
        )

    checkpoint = _safe_runtime_path(
        args.checkpoint,
        purpose="V96 authenticated V85 checkpoint",
        expected="directory",
    )
    if checkpoint != Path(authorization.v85_checkpoint_path).resolve():
        raise ValueError("V96 MCP checkpoint differs from the hash-only authorization")
    if {item.name for item in checkpoint.iterdir()} != _BASE_RUNTIME_FILES:
        raise ValueError("V96 MCP V85 checkpoint inventory changed")
    adapter = checkpoint / "adapter.safetensors"
    metadata_path = checkpoint / "runtime_metadata.json"
    if (
        _audited_sha256(adapter, audit) != authorization.v85_adapter_sha256
        or _audited_sha256(metadata_path, audit)
        != authorization.v85_metadata_sha256
    ):
        raise ValueError("V96 MCP V85 checkpoint bytes changed")
    metadata = _strict_json_object(
        metadata_path,
        purpose="V96 V85 runtime metadata",
        audit=audit,
    )
    validate_v96_v85_base_checkpoint_contract(metadata)

    runtime_config = load_runtime_config(
        candidate_hook.runtime_config,
        record_file=audit.record,
    )
    if (
        effective_runtime_config_sha256(runtime_config)
        != authorization.runtime_config_effective_sha256
    ):
        raise ValueError("V96 MCP runtime config changed after authorization")
    checkpoint_identity = _checkpoint_fingerprint(checkpoint, audit)
    memory_path = (
        _safe_runtime_path(
            args.v96_scene_memory,
            purpose="V96 explicit-candidate scene memory",
            expected="directory",
        )
        if args.v96_scene_memory is not None
        else _safe_runtime_path(
            candidate_hook.scene_memory_root / args.scene,
            purpose="V96 explicit-candidate scene memory",
            expected="directory",
        )
    )
    loaded = load_v81_scene_memory(
        memory_path,
        expected_scene_id=args.scene,
        expected_base_checkpoint_sha256=checkpoint_identity,
        expected_runtime_config_sha256=authorization.runtime_config_effective_sha256,
        expected_model_device="cpu",
        record_file=audit.record,
    )
    validate_v96_scene_memory_contract(scene_id=args.scene, loaded=loaded)

    control = _safe_runtime_path(
        bridge_hook.atlas_control_checkpoint,
        purpose="V96 question-free compiler checkpoint",
        expected="directory",
    )
    control_fingerprint = compiler_checkpoint_fingerprint(control, audit)
    control_metadata = _strict_json_object(
        control / "runtime_metadata.json",
        purpose="V96 question-free compiler metadata",
        audit=audit,
    )
    control_weights = control_metadata.get("weights_sha256")
    memory_control = loaded.metadata["source_control_checkpoint_sha256"]
    if (
        not isinstance(control_weights, str)
        or _SHA256.fullmatch(control_weights) is None
        or memory_control not in {control_fingerprint, control_weights}
        or control_metadata.get("environmental_text_inputs") != []
        or control_metadata.get("question_dependent_scene_retrieval") is not False
        or control_metadata.get("latent_selection_or_top_k_used") is not False
        or control_metadata.get("oracle_runtime_loaded") is not False
        or control_metadata.get("question_or_answer_text_serialized") is not False
    ):
        raise ValueError("V96 MCP question-free compiler binding changed")

    probe = _safe_runtime_path(
        bridge_hook.atlas_probe_bank,
        purpose="V96 numeric atlas probe bank",
        expected="directory",
    )
    if {item.name for item in probe.iterdir()} != {
        "probes.safetensors",
        "runtime_metadata.json",
    }:
        raise ValueError("V96 MCP numeric probe inventory changed")
    probe_metadata = _strict_json_object(
        probe / "runtime_metadata.json",
        purpose="V96 numeric probe metadata",
        audit=audit,
    )
    probe_tensor_sha256 = probe_metadata.get("probe_tensor_sha256")
    if (
        not isinstance(probe_tensor_sha256, str)
        or _SHA256.fullmatch(probe_tensor_sha256) is None
        or loaded.metadata["source_probe_tensor_sha256"] != probe_tensor_sha256
        or probe_metadata.get("questions_or_answers_serialized") is not False
        or probe_metadata.get("environmental_text_serialized") is not False
        or probe_metadata.get("oracle_loaded") is not False
    ):
        raise ValueError("V96 MCP numeric probe binding changed")
    probe_file_sha256 = probe_metadata.get("probe_file_sha256")
    if (
        not isinstance(probe_file_sha256, str)
        or _SHA256.fullmatch(probe_file_sha256) is None
        or _audited_sha256(probe / "probes.safetensors", audit)
        != probe_file_sha256
    ):
        raise ValueError("V96 MCP numeric probe bytes changed")

    semantic_dim = metadata.get("semantic_dim")
    hidden_size = metadata.get("language_hidden_dim")
    if (
        isinstance(semantic_dim, bool)
        or not isinstance(semantic_dim, int)
        or semantic_dim < 1
        or isinstance(hidden_size, bool)
        or not isinstance(hidden_size, int)
        or hidden_size != HIDDEN_SIZE
    ):
        raise ValueError("V96 MCP V85 dimensional contract changed")
    summary = {
        "artifact": "gemma4_v96_explicit_candidate_mcp_server_preflight_v1",
        "mode": "explicit_candidate_only_not_default",
        "candidate_fingerprint_sha256": (
            authorization.candidate_fingerprint_sha256
        ),
        "candidate_state_sha256": authorization.v96_state_sha256,
        "gate_results_sha256": authorization.gate_results_sha256,
        "known_development_gate_passed": True,
        "pass_evidence_authenticated": True,
        "deferred_final_gate_passed": release_verification[
            "deferred_final_binding_exact"
        ],
        "runtime_leakage_smoke_passed": release_verification[
            "runtime_smoke_binding_exact"
        ],
        "promoted_runtime_release_verified": release_verification[
            "promoted_runtime_release_verified"
        ],
        "release_verification_check_count": release_verification["check_count"],
        "deferred_final_evidence_sha256": release_verification[
            "deferred_final_evidence_sha256"
        ],
        "runtime_leakage_smoke_sha256": release_verification[
            "runtime_smoke_sha256"
        ],
        "verified_release_checkpoint_sha256": release_verification[
            "release_checkpoint_sha256"
        ],
        "verified_release_adapter_sha256": release_verification[
            "release_adapter_sha256"
        ],
        "runtime_implementation_inventory_sha256": release_verification[
            "runtime_implementation_inventory_sha256"
        ],
        "scene_memory_path": str(memory_path),
        "scene_memory_sha256": loaded.metadata["canonical_prefix_sha256"],
        "base_scene_prefix_sha256": loaded.metadata["base_prefix_sha256"],
        "fixed_memory_tokens": FIXED_MEMORY_TOKENS,
        "frozen_lora_bank_count": 10,
        "question_free_compiler_checkpoint_sha256": control_fingerprint,
        "question_free_compiler_weights_sha256": control_weights,
        "numeric_probe_tensor_sha256": probe_tensor_sha256,
        "numeric_tool_outputs_only": True,
        "full_memory_recompiled_before_map_commit": True,
        "direct_v96_answer_robot_tokens_authenticated": False,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
        "automatic_runtime_promotion": False,
        "runtime_promotion_authorized": False,
    }
    return {
        "authorization": authorization,
        "candidate_hook": candidate_hook,
        "bridge_hook": bridge_hook,
        "runtime_config": runtime_config,
        "scene_memory_path": memory_path,
        "checkpoint": {
            "path": str(checkpoint),
            "adapter_sha256": authorization.v85_adapter_sha256,
            "runtime_metadata_sha256": authorization.v85_metadata_sha256,
            "adapter_size_bytes": adapter.stat().st_size,
            "semantic_dim": semantic_dim,
            "language_hidden_dim": hidden_size,
            "scene_latents": metadata.get("scene_latents"),
            "question_dependent_scene_processing": False,
            "training_metadata_opened": False,
            "exact_v54_release_authenticated": False,
            "checkpoint_identity_sha256": checkpoint_identity,
            "frozen_lora_bank_count_after_v96_extension": 10,
        },
        "summary": summary,
    }


def _numeric_preflight(
    args: argparse.Namespace,
    config: dict[str, Any],
    *,
    audit: FileAccessAudit,
) -> dict[str, Any]:
    return {
        "schema": _PREFLIGHT_SCHEMA,
        "phase": "embodied_mcp_preflight",
        "passed": True,
        "mode": "numeric_only",
        "scene_id": args.scene,
        "config": str(Path(config["_config_path"]).resolve()),
        "base_map": _map_preflight(
            config,
            args.scene,
            audit=audit,
            require_semantics=False,
        ),
        "action_protocol": _validate_configured_action_protocol(config),
        "loads_language_model": False,
        "loads_blender": False,
        "starts_transport": False,
        "changes_robot_or_map_state": False,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
    }


def _run_semantic_server_lifetime(
    server_factory: Callable[[], MCPServer[None]],
    *,
    audit: FileAccessAudit,
    audit_report: str | Path | Callable[[], str | Path],
    transport: str,
    host: str,
    port: int,
) -> None:
    """Audit semantic runtime construction and every served MCP request.

    ``server.run`` is intentionally inside the audit context.  The shutdown
    report is written in ``finally`` so a blocked request, transport failure,
    or operator interrupt still leaves an inspectable access record.
    """

    try:
        with audit:
            server = server_factory()
            _serve(server, transport=transport, host=host, port=port)
    finally:
        destination = audit_report() if callable(audit_report) else audit_report
        _save_audit_durably(audit, destination)
    audit.assert_clean()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="configs/default.yaml")
    result.add_argument("--scene", default="scene_000001")
    result.add_argument("--seed", type=int)
    result.add_argument(
        "--checkpoint",
        help="Local adapter checkpoint for the continuous-map embodied runtime.",
    )
    result.add_argument(
        "--control-checkpoint",
        help=(
            "Optional sealed continuous question-controller checkpoint. When supplied, "
            "robot scans refresh the same full-scene signature used by static chat."
        ),
    )
    result.add_argument(
        "--control-runtime-config",
        help=(
            "Standalone sanitized runtime YAML authenticated by the controller. "
            "Required with --control-checkpoint; robot/render settings remain in --config."
        ),
    )
    result.add_argument(
        "--runtime-asset",
        help="Authenticated sanitized Blender asset outside oracle/QA directories.",
    )
    result.add_argument(
        "--persistent-map",
        help="Optional persistent embodied semantic-map path.",
    )
    result.add_argument(
        "--scan-output-directory",
        help="Optional sanitized RGB-D observation directory.",
    )
    result.add_argument(
        "--robot-state-checkpoint",
        help=(
            "Optional sanitized numeric state encoder. Production controlled mode "
            "requires it so pose and motion reach the LM as continuous tokens."
        ),
    )
    result.add_argument(
        "--v96-candidate-bridge-hook",
        help=(
            "Optional explicit, unpromoted V96 candidate bridge. This mode requires "
            "authenticated PASS evidence and cannot be combined with a question "
            "controller. It never changes the default runtime."
        ),
    )
    result.add_argument(
        "--v96-scene-memory",
        help=(
            "Optional sanitized two-file V96 scene memory override. Valid only with "
            "--v96-candidate-bridge-hook."
        ),
    )
    result.add_argument(
        "--allow-explicit-v96-candidate",
        action="store_true",
        help="Acknowledge that V96 is an authenticated but unpromoted candidate.",
    )
    result.add_argument(
        "--audit-report",
        help=(
            "Runtime file-audit report; --check saves one in both numeric and semantic modes. "
            "Defaults below the configured reports/metrics directory."
        ),
    )
    result.add_argument(
        "--check",
        action="store_true",
        help=(
            "Authenticate local MCP inputs and exit without loading Gemma or Blender, "
            "starting transport, or changing robot/map state."
        ),
    )
    result.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8766)
    return result


def _validate_arguments(args: argparse.Namespace) -> bool:
    if not isinstance(args.scene, str) or _SCENE_ID.fullmatch(args.scene) is None:
        raise SystemExit("--scene must be opaque and match scene_ followed by six digits")
    if not 1 <= args.port <= 65_535:
        raise SystemExit("--port must be between 1 and 65535")
    semantic_mode = args.checkpoint is not None or args.runtime_asset is not None
    v96_mode = args.v96_candidate_bridge_hook is not None
    if v96_mode != bool(args.allow_explicit_v96_candidate):
        raise SystemExit(
            "V96 candidate MCP mode requires both --v96-candidate-bridge-hook and "
            "--allow-explicit-v96-candidate"
        )
    if args.v96_scene_memory is not None and not v96_mode:
        raise SystemExit(
            "--v96-scene-memory requires --v96-candidate-bridge-hook"
        )
    if v96_mode and not semantic_mode:
        raise SystemExit(
            "V96 candidate MCP mode requires --checkpoint and --runtime-asset"
        )
    if v96_mode and args.control_checkpoint is not None:
        raise SystemExit(
            "V96 fixed scene memory cannot be combined with --control-checkpoint"
        )
    if v96_mode and args.robot_state_checkpoint is None:
        raise SystemExit(
            "V96 candidate MCP mode requires --robot-state-checkpoint for numeric "
            "active-prefix binding"
        )
    if semantic_mode and (args.checkpoint is None or args.runtime_asset is None):
        raise SystemExit("--checkpoint and --runtime-asset must be supplied together")
    if args.control_checkpoint is not None and not semantic_mode:
        raise SystemExit(
            "--control-checkpoint requires --checkpoint and --runtime-asset"
        )
    if (args.control_checkpoint is None) != (args.control_runtime_config is None):
        raise SystemExit(
            "--control-checkpoint and --control-runtime-config must be supplied together"
        )
    if args.control_checkpoint is not None and args.robot_state_checkpoint is None:
        raise SystemExit(
            "Controlled semantic MCP requires --robot-state-checkpoint"
        )
    if args.robot_state_checkpoint is not None and not semantic_mode:
        raise SystemExit(
            "--robot-state-checkpoint requires --checkpoint and --runtime-asset"
        )
    if args.scan_output_directory is not None and not semantic_mode:
        raise SystemExit(
            "--scan-output-directory requires --checkpoint and --runtime-asset"
        )
    if semantic_mode and args.seed is not None:
        raise SystemExit("--seed is not supported by the fail-closed refreshing runtime")
    return semantic_mode


def _absolute_report_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _default_audit_report(config: dict[str, Any], scene_id: str) -> Path:
    return reports_root(config) / "metrics" / f"embodied_mcp_file_access_{scene_id}.json"


def _run_check(args: argparse.Namespace, *, semantic_mode: bool) -> int:
    """Run a finite audited preflight and always leave a machine-readable audit."""

    audit = _runtime_audit()
    fallback = PROJECT_ROOT / "reports" / "metrics" / (
        f"embodied_mcp_file_access_{args.scene}.json"
    )
    audit_report = (
        _absolute_report_path(args.audit_report)
        if args.audit_report is not None
        else fallback
    )
    summary: dict[str, Any]
    status = 0
    try:
        with audit:
            # This is intentionally the first runtime input read. Recursive
            # ``_base_`` YAML reads are captured by the same process audit.
            config = load_config(args.config)
            _extend_runtime_audit(audit, config)
            if args.audit_report is None:
                audit_report = _default_audit_report(config, args.scene)
            summary = (
                _semantic_preflight(args, config, audit=audit)
                if semantic_mode
                else _numeric_preflight(args, config, audit=audit)
            )
        audit.assert_clean()
    except Exception as error:  # noqa: BLE001 - finite CLI reports failures as JSON
        status = 2
        summary = {
            "schema": _PREFLIGHT_SCHEMA,
            "phase": "embodied_mcp_preflight",
            "passed": False,
            "mode": (
                "semantic_continuous_map_v96_explicit_candidate"
                if semantic_mode and args.v96_candidate_bridge_hook is not None
                else "semantic_continuous_map"
                if semantic_mode
                else "numeric_only"
            ),
            "scene_id": args.scene,
            "loads_language_model": False,
            "loads_blender": False,
            "starts_transport": False,
            "changes_robot_or_map_state": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    finally:
        _save_audit_durably(audit, audit_report)
    summary.update(
        {
            "audit_report": str(audit_report.resolve()),
            "loaded_file_count": len(audit.unique_paths),
            "forbidden_access_count": len(audit.forbidden_accesses()),
        }
    )
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
    return status


def _run_semantic_live(args: argparse.Namespace) -> int:
    """Construct and serve semantic MCP with config included in lifetime auditing."""

    audit = _runtime_audit()
    report_holder = {
        "path": (
            _absolute_report_path(args.audit_report)
            if args.audit_report is not None
            else PROJECT_ROOT
            / "reports"
            / "metrics"
            / f"embodied_mcp_file_access_{args.scene}.json"
        )
    }

    def semantic_server_factory() -> MCPServer[None]:
        # Configuration, runtime imports, authenticated scene-manifest reads,
        # model/map construction, tool registration, and every later request
        # share one uninterrupted process-wide access audit.
        config = load_config(args.config)
        _extend_runtime_audit(audit, config)
        _validate_semantic_config_isolation(config)
        _validate_embodied_render_contract(config)
        _validate_navigation_safety_metadata(config)
        _validate_configured_action_protocol(config)
        _forbid_adjacent_training_metadata(args.checkpoint, audit)
        if args.audit_report is None:
            report_holder["path"] = _default_audit_report(config, args.scene)
        from semantic_3d_chat.robot.blender_scanner import SanitizedBlenderScanner
        from semantic_3d_chat.robot.runtime_refresh import build_refreshing_embodied_runtime

        chat_runtime = None
        v96_contract = None
        if args.v96_candidate_bridge_hook is not None:
            v96_contract = _v96_candidate_server_contract(args, audit=audit)
            from semantic_3d_chat.chat.v96_explicit_candidate_runtime import (
                V96ExplicitCandidateChatRuntime,
            )

            chat_runtime = V96ExplicitCandidateChatRuntime.load(
                v96_contract["runtime_config"],
                args.scene,
                authorization=v96_contract["authorization"],
                scene_memory=v96_contract["scene_memory_path"],
                audit=audit,
                local_files_only=True,
            )
        elif args.control_checkpoint is not None:
            from semantic_3d_chat.chat.question_control_runtime import (
                QuestionControlledChatRuntime,
            )
            from semantic_3d_chat.chat.runtime_config import load_runtime_config

            assert args.control_runtime_config is not None
            control_config = load_runtime_config(
                args.control_runtime_config,
                record_file=audit.record,
            )
            chat_runtime = QuestionControlledChatRuntime.load(
                control_config,
                args.scene,
                base_checkpoint=args.checkpoint,
                control_checkpoint=args.control_checkpoint,
                audit=audit,
            )

        resolution = tuple(int(value) for value in config["render"]["resolution"])
        scanner = SanitizedBlenderScanner(
            args.scene,
            args.runtime_asset,
            resolution=resolution,
            horizontal_fov_degrees=float(config["render"]["horizontal_fov_degrees"]),
            engine=str(config["render"]["engine"]),
            samples=int(config["render"]["samples"]),
            max_depth_m=float(config["mapping"]["depth_max_m"]),
            output_directory=_safe_runtime_path(
                args.scan_output_directory
                or project_path(config, "robot", args.scene, "scans"),
                purpose="sanitized RGB-D scan output",
            ),
        )
        common_runtime_arguments = {
            "persistent_map_path": (
                None if args.persistent_map is None else Path(args.persistent_map)
            ),
            "observation_scanner": scanner,
            "robot_state_checkpoint": args.robot_state_checkpoint,
            "audit": audit,
            "local_files_only": True,
        }
        if v96_contract is None:
            simulator = build_refreshing_embodied_runtime(
                config,
                args.scene,
                checkpoint=args.checkpoint,
                chat_runtime=chat_runtime,
                **common_runtime_arguments,
            )
        else:
            from semantic_3d_chat.robot.v96_candidate_refresh import (
                V75QuestionFreeV96MemoryCompiler,
                build_v96_candidate_refreshing_embodied_runtime,
            )

            assert isinstance(chat_runtime, V96ExplicitCandidateChatRuntime)
            compiler = V75QuestionFreeV96MemoryCompiler.load(
                v96_contract["bridge_hook"].atlas_control_checkpoint,
                v96_contract["bridge_hook"].atlas_probe_bank,
                device=chat_runtime.base.language.device,
                audit=audit,
            )
            simulator = build_v96_candidate_refreshing_embodied_runtime(
                config,
                args.scene,
                checkpoint=args.checkpoint,
                authorization=v96_contract["authorization"],
                chat_runtime=chat_runtime,
                memory_compiler=compiler,
                allow_explicit_candidate=True,
                **common_runtime_arguments,
            )
        return build_server(simulator)

    _run_semantic_server_lifetime(
        semantic_server_factory,
        audit=audit,
        audit_report=lambda: report_holder["path"],
        transport=args.transport,
        host=args.host,
        port=args.port,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    semantic_mode = _validate_arguments(args)
    if args.check:
        return _run_check(args, semantic_mode=semantic_mode)
    if semantic_mode:
        return _run_semantic_live(args)

    config = load_config(args.config)
    simulator = EmbodiedCameraSimulator(config, args.scene, seed=args.seed)
    _serve(
        build_server(simulator),
        transport=args.transport,
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
