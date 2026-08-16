#!/usr/bin/env python3
"""Fail-closed, no-model preflight for the promoted V75 static demo.

The check authenticates only inference inputs: one standalone runtime config,
the exact two-file V54 scene-prefix release, the exact two-file V75 continuous
controller, one sanitized high-dimensional voxel map, and human-only visuals.
It does not import or load Gemma, Blender, QA, oracle, render, feature-cache, or
training artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from semantic_3d_chat.chat.runtime_config import (
    effective_runtime_config_sha256,
    load_runtime_config,
)
from semantic_3d_chat.config import PROJECT_ROOT, project_path, reports_root
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.evaluation.v75_official_validation_contract import (
    EXPECTED_BASE_CHECKPOINT_SHA256,
    EXPECTED_RUNTIME_CONFIG_SHA256,
    authenticate_v75_control_checkpoint,
)

DEFAULT_CONFIG = Path("configs/runtime/gemma4_v56_question_control.yaml")
DEFAULT_BASE_CHECKPOINT = Path(
    "data_gemma4/runtime/checkpoints/gemma4_v54_release_v1"
)
DEFAULT_CONTROL_CHECKPOINT = Path(
    "data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1"
)
_OPAQUE_SCENE_ID = re.compile(r"scene_[0-9]{6}")
_BASE_INVENTORY = frozenset({"adapter.safetensors", "runtime_metadata.json"})


def _rooted(path: str | Path) -> Path:
    value = Path(path).expanduser()
    rooted = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(rooted))


def _reject_symlink_components(path: Path, purpose: str) -> Path:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"V75 demo {purpose} path contains a symbolic link: {current}")
    return path


def _safe_regular(path: Path, purpose: str) -> Path:
    source = _reject_symlink_components(path, purpose)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"V75 demo {purpose} is unavailable: {source}")
    return source


def _safe_base_checkpoint(path: str | Path) -> tuple[Path, str]:
    source = _reject_symlink_components(_rooted(path), "base checkpoint")
    if not source.is_dir() or source.is_symlink():
        raise FileNotFoundError(f"V75 demo base checkpoint is unavailable: {source}")
    inventory = {item.name for item in source.iterdir()}
    if inventory != _BASE_INVENTORY:
        raise ValueError(
            "V75 demo base checkpoint must be the exact two-file inference release; "
            f"observed={sorted(inventory)}"
        )
    for name in _BASE_INVENTORY:
        _safe_regular(source / name, f"base checkpoint {name}")
    fingerprint, entries = checkpoint_fingerprint(source)
    if (
        fingerprint != EXPECTED_BASE_CHECKPOINT_SHA256
        or {str(entry["path"]) for entry in entries} != _BASE_INVENTORY
    ):
        raise ValueError("V75 demo base checkpoint identity changed")
    return source, fingerprint


def validate_v75_demo_inputs(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    scene_id: str = "scene_000001",
    base_checkpoint: str | Path = DEFAULT_BASE_CHECKPOINT,
    control_checkpoint: str | Path = DEFAULT_CONTROL_CHECKPOINT,
) -> dict[str, Any]:
    """Authenticate the complete static demo surface without loading a model."""

    if _OPAQUE_SCENE_ID.fullmatch(scene_id) is None:
        raise ValueError("V75 demo scene ID must be opaque")
    config = load_runtime_config(config_path)
    runtime_config_sha256 = effective_runtime_config_sha256(config)
    if runtime_config_sha256 != EXPECTED_RUNTIME_CONFIG_SHA256:
        raise ValueError("V75 demo effective runtime configuration changed")

    base_path, base_sha256 = _safe_base_checkpoint(base_checkpoint)
    control = authenticate_v75_control_checkpoint(control_checkpoint)
    metadata: Mapping[str, Any] = control.metadata
    if (
        metadata.get("schema_version") != 75
        or metadata.get("base_checkpoint_sha256") != base_sha256
        or metadata.get("base_runtime_config_sha256") != runtime_config_sha256
        or metadata.get("saved_runtime_training_gate_passed") is not True
    ):
        raise ValueError("V75 demo controller is not bound to the selected base and config")

    map_path = _safe_regular(
        project_path(config, "maps", scene_id, "voxel_map.npz"),
        "sanitized voxel map",
    )
    figures = reports_root(config) / "figures" / scene_id
    preview_path = _safe_regular(figures / "map_rgb.png", "RGB map preview")
    point_cloud_path = _safe_regular(figures / "map_rgb.ply", "point-cloud preview")

    return {
        "schema_version": 1,
        "artifact": "promoted_v75_static_demo_preflight_v1",
        "passed": True,
        "loads_model": False,
        "runs_blender": False,
        "scene_id": scene_id,
        "runtime_config": str(_rooted(config_path)),
        "runtime_config_effective_sha256": runtime_config_sha256,
        "base_checkpoint": str(base_path),
        "base_checkpoint_sha256": base_sha256,
        "base_checkpoint_inventory": sorted(_BASE_INVENTORY),
        "control_checkpoint": str(control.path),
        "control_checkpoint_sha256": control.sha256,
        "control_weights_sha256": control.weights_sha256,
        "control_runtime_metadata_sha256": control.runtime_metadata_sha256,
        "control_schema_version": 75,
        "control_architecture": metadata["architecture"],
        "saved_runtime_training_gate_passed": True,
        "scene_latents": metadata["environment_latents"],
        "control_tokens": metadata["control_tokens"],
        "complete_scene_prefix_required": True,
        "prequestion_scene_key_value_cache": True,
        "all_environment_latents_attended": True,
        "question_dependent_scene_retrieval": False,
        "environmental_text_inputs": [],
        "training_or_evaluation_artifacts_loaded": False,
        "voxel_map": str(map_path),
        "voxel_map_size_bytes": map_path.stat().st_size,
        "rgb_map_preview": str(preview_path),
        "point_cloud_preview": str(point_cloud_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--scene", default="scene_000001")
    parser.add_argument("--base-checkpoint", default=str(DEFAULT_BASE_CHECKPOINT))
    parser.add_argument("--control-checkpoint", default=str(DEFAULT_CONTROL_CHECKPOINT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = validate_v75_demo_inputs(
            config_path=args.config,
            scene_id=args.scene,
            base_checkpoint=args.base_checkpoint,
            control_checkpoint=args.control_checkpoint,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"V75 demo preflight refused: {error}", file=os.sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_BASE_CHECKPOINT",
    "DEFAULT_CONFIG",
    "DEFAULT_CONTROL_CHECKPOINT",
    "main",
    "validate_v75_demo_inputs",
]
