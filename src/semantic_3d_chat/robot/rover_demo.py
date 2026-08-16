"""One-process, loopback-only browser demo for the local toy rover.

The browser is a human control surface.  Gemma, the continuous 3D scene
memory, and the kinematic rover remain in this process on the local Mac. The
prepared map image and scan montage are display-only and are never model
inputs. Control keeps the precomputed scene memory fixed and refreshes only
numeric robot-state tokens after motion.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import tempfile
import threading
import webbrowser
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from semantic_3d_chat.config import PROJECT_ROOT, load_config

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_OPAQUE_SCENE_ID = re.compile(r"scene_[0-9]{6}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_COMPONENTS = frozenset({"oracle", "qa", "training", "scorer_only"})


@dataclass(frozen=True)
class RoverDemoSettings:
    config: Path
    control_runtime_config: Path
    scene_id: str
    base_checkpoint: Path
    control_checkpoint: Path
    runtime_asset: Path
    robot_state_checkpoint: Path
    navigation_checkpoint: Path
    map_path: Path
    map_visual: Path
    scan_visual: Path
    audit_output: Path
    host: str
    port: int
    open_browser: bool


def _rooted(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _safe_path(path: Path, purpose: str, *, directory: bool = False) -> Path:
    forbidden = {part.casefold() for part in path.parts} & _FORBIDDEN_COMPONENTS
    if forbidden:
        raise ValueError(f"Refusing {purpose} below a prohibited runtime directory: {path}")
    if path.is_symlink():
        raise ValueError(f"Refusing symbolic-link {purpose}: {path}")
    predicate = path.is_dir if directory else path.is_file
    if not predicate():
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"Required {purpose} {kind} is unavailable: {path}")
    return path


def _require_inventory(path: Path, names: frozenset[str], purpose: str) -> None:
    _safe_path(path, purpose, directory=True)
    observed = {member.name for member in path.iterdir()}
    if observed != names:
        raise ValueError(
            f"{purpose} inventory changed: expected={sorted(names)}, observed={sorted(observed)}"
        )
    for name in names:
        _safe_path(path / name, f"{purpose} member {name}")


def _blender_executable() -> str:
    requested = os.environ.get("BLENDER", "blender")
    candidate = (
        str(Path(requested).expanduser().resolve())
        if os.sep in requested
        else shutil.which(requested)
    )
    if not candidate or not os.access(candidate, os.X_OK):
        raise FileNotFoundError(
            "Blender is unavailable; install Blender or set BLENDER to its executable"
        )
    return candidate


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _backend_preflight(settings: RoverDemoSettings) -> dict[str, Any]:
    from semantic_3d_chat.robot.practical_rover import practical_rover_preflight

    return practical_rover_preflight(
        config=settings.config,
        control_config=settings.control_runtime_config,
        scene_id=settings.scene_id,
        base_checkpoint=settings.base_checkpoint,
        control_checkpoint=settings.control_checkpoint,
        runtime_asset=settings.runtime_asset,
        robot_state_checkpoint=settings.robot_state_checkpoint,
        navigation_checkpoint=settings.navigation_checkpoint,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/runtime/embodied_live.yaml")
    parser.add_argument(
        "--control-runtime-config",
        default="configs/runtime/gemma4_v56_question_control.yaml",
    )
    parser.add_argument("--scene", default="scene_000001")
    parser.add_argument(
        "--base-checkpoint",
        default="data_gemma4/runtime/checkpoints/gemma4_v54_release_v1",
    )
    parser.add_argument(
        "--control-checkpoint",
        default="data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1",
    )
    parser.add_argument("--runtime-asset")
    parser.add_argument(
        "--robot-state-checkpoint",
        default="data_gemma4/checkpoints/robot_state_numeric_v1",
    )
    parser.add_argument(
        "--navigation-checkpoint",
        default=(
            "data_gemma4/checkpoints/"
            "gemma_waypoint_policy_v2_operator_dagger_v14_runtime_aligned"
        ),
    )
    parser.add_argument("--map")
    parser.add_argument("--map-visual")
    parser.add_argument("--scan-visual", default="reports/gemma4/figures/scan_montage.png")
    parser.add_argument(
        "--audit-report",
        help="File-access audit written on clean local-server shutdown.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the default browser after the local server starts.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the complete local launch surface without loading Gemma or Blender.",
    )
    return parser


def _settings(args: argparse.Namespace) -> RoverDemoSettings:
    if args.host not in _LOOPBACK_HOSTS:
        raise ValueError("The rover demo refuses every non-loopback host")
    if not 1 <= args.port <= 65_535:
        raise ValueError("--port must be between 1 and 65535")
    if _OPAQUE_SCENE_ID.fullmatch(args.scene) is None:
        raise ValueError("--scene must be opaque and match scene_ followed by six digits")
    compact_id = args.scene.removeprefix("scene_")
    runtime_asset = args.runtime_asset or f"data/runtime_assets/{args.scene}/s_{compact_id}.blend"
    map_path = args.map or f"data_gemma4/maps/{args.scene}/voxel_map.npz"
    map_visual = args.map_visual or f"reports/gemma4/figures/{args.scene}/map_rgb.png"
    audit_output = args.audit_report or (
        f"reports/gemma4/metrics/practical_rover_access_{args.scene}.json"
    )
    return RoverDemoSettings(
        config=_rooted(args.config),
        control_runtime_config=_rooted(args.control_runtime_config),
        scene_id=args.scene,
        base_checkpoint=_rooted(args.base_checkpoint),
        control_checkpoint=_rooted(args.control_checkpoint),
        runtime_asset=_rooted(runtime_asset),
        robot_state_checkpoint=_rooted(args.robot_state_checkpoint),
        navigation_checkpoint=_rooted(args.navigation_checkpoint),
        map_path=_rooted(map_path),
        map_visual=_rooted(map_visual),
        scan_visual=_rooted(args.scan_visual),
        audit_output=_rooted(audit_output),
        host=args.host,
        port=args.port,
        open_browser=not args.no_open,
    )


def check_rover_demo(settings: RoverDemoSettings) -> dict[str, Any]:
    """Perform the finite preflight without loading model tensors or Blender."""

    _safe_path(settings.config, "embodied runtime configuration")
    _safe_path(settings.control_runtime_config, "continuous-control configuration")
    _require_inventory(
        settings.base_checkpoint,
        frozenset({"adapter.safetensors", "runtime_metadata.json"}),
        "sanitized base checkpoint",
    )
    _require_inventory(
        settings.control_checkpoint,
        frozenset({"control.safetensors", "runtime_metadata.json"}),
        "sanitized continuous controller",
    )
    _require_inventory(
        settings.robot_state_checkpoint,
        frozenset({"state.safetensors", "runtime_metadata.json"}),
        "numeric robot-state checkpoint",
    )
    _require_inventory(
        settings.navigation_checkpoint,
        frozenset({"policy.safetensors", "runtime_metadata.json"}),
        "task-trained navigation checkpoint",
    )
    _safe_path(settings.runtime_asset, "sanitized Blender room")
    _safe_path(settings.map_path, "continuous semantic voxel map")
    _safe_path(settings.map_visual, "human-only map preview")
    _safe_path(settings.scan_visual, "human-only scan montage")
    blender = _blender_executable()
    missing_modules = [
        name
        for name in ("starlette", "uvicorn", "torch", "transformers", "safetensors")
        if not _module_available(name)
    ]
    if missing_modules:
        raise ModuleNotFoundError(f"Missing local rover dependencies: {missing_modules}")
    if not _module_available("semantic_3d_chat.robot.practical_rover"):
        raise ModuleNotFoundError("The practical rover backend is unavailable")
    if not _module_available("semantic_3d_chat.robot.rover_web_app"):
        raise ModuleNotFoundError("The rover browser UI is unavailable")

    config = load_config(settings.config)
    room_size = config.get("scene", {}).get("room_size_m")
    if (
        not isinstance(room_size, Sequence)
        or isinstance(room_size, (str, bytes))
        or len(room_size) != 3
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in room_size)
        or any(float(value) <= 0.0 for value in room_size)
    ):
        raise ValueError("Runtime config has no valid three-dimensional room size")

    import torch

    backend = _backend_preflight(settings)
    if backend.get("ready") is not True:
        raise RuntimeError("Practical rover backend preflight did not pass")
    configured_map = backend.get("continuous_map")
    if not isinstance(configured_map, str) or _rooted(configured_map) != settings.map_path:
        raise ValueError(
            "The requested continuous map differs from the precomputed static base map "
            "authenticated by the runtime configuration"
        )
    navigation_checkpoint_sha256 = backend.get("navigation_checkpoint_sha256")
    gemma_runtime_binding_sha256 = backend.get("gemma_runtime_binding_sha256")
    if (
        not isinstance(navigation_checkpoint_sha256, str)
        or _SHA256.fullmatch(navigation_checkpoint_sha256) is None
        or not isinstance(gemma_runtime_binding_sha256, str)
        or _SHA256.fullmatch(gemma_runtime_binding_sha256) is None
    ):
        raise ValueError("Backend preflight has no authenticated navigation identity")

    return {
        "schema_version": 1,
        "artifact": "semantic_3d_chat_local_rover_demo_preflight_v1",
        "passed": True,
        "scene_id": settings.scene_id,
        "host": settings.host,
        "port": settings.port,
        "loopback_only": True,
        "room_size_m": [float(value) for value in room_size],
        "continuous_map": str(settings.map_path.relative_to(PROJECT_ROOT)),
        "runtime_asset": str(settings.runtime_asset.relative_to(PROJECT_ROOT)),
        "audit_output": str(settings.audit_output.relative_to(PROJECT_ROOT)),
        "human_only_visuals": [
            str(settings.map_visual.relative_to(PROJECT_ROOT)),
            str(settings.scan_visual.relative_to(PROJECT_ROOT)),
        ],
        "human_visuals_are_model_inputs": False,
        "starts_from_precomputed_static_map": True,
        "inherits_prior_persistent_map": False,
        "initial_rgbd_scan": False,
        "runtime_rgbd_scans": False,
        "navigation_checkpoint": str(
            settings.navigation_checkpoint.relative_to(PROJECT_ROOT)
        ),
        "navigation_checkpoint_sha256": navigation_checkpoint_sha256,
        "gemma_runtime_binding_sha256": gemma_runtime_binding_sha256,
        "task_trained_navigation": True,
        "actual_gemma_causal_waypoint_policy": True,
        "model_selects_every_waypoint_heading_and_stop": True,
        "deterministic_route_planner_used": False,
        "navigation_fallback": None,
        "high_level_natural_language_only": True,
        "untrained_json_backend_enabled": False,
        "environmental_text_inputs": [],
        "local_inference": True,
        "mps_available": bool(torch.backends.mps.is_available()),
        "blender": blender,
        "backend_preflight": backend,
        "loads_model": False,
        "runs_blender": False,
        "starts_server": False,
    }


def _build_session(
    settings: RoverDemoSettings,
    *,
    persistent_map: Path,
) -> Any:
    from semantic_3d_chat.robot.practical_rover import build_local_practical_rover

    persistent_map = persistent_map.resolve()
    if persistent_map.exists() or persistent_map.is_symlink():
        raise FileExistsError(
            "A fresh rover session cannot inherit an existing persistent semantic map"
        )
    return build_local_practical_rover(
        config=settings.config,
        control_config=settings.control_runtime_config,
        scene_id=settings.scene_id,
        base_checkpoint=settings.base_checkpoint,
        control_checkpoint=settings.control_checkpoint,
        runtime_asset=settings.runtime_asset,
        robot_state_checkpoint=settings.robot_state_checkpoint,
        navigation_checkpoint=settings.navigation_checkpoint,
        persistent_map=persistent_map,
        audit_output=settings.audit_output,
        initial_scan=False,
    )


def _serve(settings: RoverDemoSettings, preflight: dict[str, Any]) -> None:
    from semantic_3d_chat.robot.rover_web_app import (
        create_rover_web_app,
        serve_rover_web_app,
    )

    config = load_config(settings.config)
    room_size = tuple(float(value) for value in config["scene"]["room_size_m"])
    # A unique empty map path makes each operator session start from the
    # authenticated, question-independent static room memory. Camera scans are
    # disabled during control, and the entire empty update tree is removed on
    # shutdown.
    with tempfile.TemporaryDirectory(
        prefix=f"semantic_3d_chat_rover_{settings.scene_id}_"
    ) as runtime_directory:
        persistent_map = Path(runtime_directory) / "semantic_map.npz"
        session = _build_session(settings, persistent_map=persistent_map)
        try:
            app = create_rover_web_app(
                session,
                room_size_m=room_size,
                visual_assets={"overview": settings.scan_visual, "map": settings.map_visual},
            )
            url = f"http://{settings.host}:{settings.port}/"
            print(
                json.dumps(
                    {
                        **preflight,
                        "starts_server": True,
                        "url": url,
                        "fresh_session_map": True,
                    },
                    sort_keys=True,
                )
            )
            if settings.open_browser:
                opener = threading.Timer(1.0, webbrowser.open, args=(url,))
                opener.daemon = True
                opener.start()
            serve_rover_web_app(app, host=settings.host, port=settings.port)
        finally:
            session.close()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = _settings(args)
    preflight = check_rover_demo(settings)
    if args.check:
        print(json.dumps(preflight, indent=2, sort_keys=True, allow_nan=False))
        return 0
    _serve(settings, preflight)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
