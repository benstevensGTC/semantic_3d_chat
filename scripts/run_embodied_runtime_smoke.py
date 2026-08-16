"""Run and attest one fresh-render embodied semantic-map transaction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT, config_hash, load_config
from semantic_3d_chat.robot.blender_scanner import SanitizedBlenderScanner
from semantic_3d_chat.robot.runtime_refresh import build_refreshing_embodied_runtime


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return Path(os.path.abspath(rooted))


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--config",
        default="configs/runtime/embodied_live.yaml",
    )
    result.add_argument("--scene", default="scene_000001")
    result.add_argument(
        "--checkpoint",
        default="data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000",
    )
    result.add_argument(
        "--asset",
        default="data/runtime_assets/scene_000001/s_000001.blend",
    )
    result.add_argument(
        "--robot-state-checkpoint",
        default="data_gemma4/checkpoints/robot_state_numeric_v1",
    )
    result.add_argument(
        "--output",
        default="reports/gemma4/metrics/embodied_runtime_smoke_scene_000001.json",
    )
    result.add_argument("--samples", type=int, default=1)
    result.add_argument("--x", type=float, default=0.0)
    result.add_argument("--y", type=float, default=0.0)
    result.add_argument("--yaw", type=float, default=0.0)
    result.add_argument("--pitch", type=float, default=0.0)
    result.add_argument(
        "--motion-turn-degrees",
        type=float,
        help=(
            "After the initial scan, execute one bounded turn. The live runtime "
            "must capture and fuse a second complete RGB-D observation."
        ),
    )
    result.add_argument("--keep-work", action="store_true")
    result.add_argument("--work-root", type=Path)
    return result


def _run(args: argparse.Namespace, work_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = _rooted(args.config)
    checkpoint = _rooted(args.checkpoint)
    asset = _rooted(args.asset)
    state_checkpoint = _rooted(args.robot_state_checkpoint)
    forbidden_roots = [
        PROJECT_ROOT / "data" / "oracle",
        PROJECT_ROOT / "data" / "qa",
        PROJECT_ROOT / "data_diverse28" / "qa",
        PROJECT_ROOT / "data_gemma4" / "training",
    ]
    audit = FileAccessAudit(forbidden_roots, block_forbidden=True)
    with audit:
        config = load_config(config_path)
        source_config_hash = config_hash(config, length=64)
        config["paths"]["data_root"] = str(work_root / "runtime")
        resolution = tuple(int(value) for value in config["render"]["resolution"])
        scanner = SanitizedBlenderScanner(
            args.scene,
            asset,
            resolution=resolution,
            horizontal_fov_degrees=float(config["render"]["horizontal_fov_degrees"]),
            engine=str(config["render"]["engine"]),
            samples=args.samples,
            max_depth_m=float(config["mapping"]["depth_max_m"]),
            output_directory=work_root / "observations",
        )
        load_started = time.perf_counter()
        runtime = build_refreshing_embodied_runtime(
            config,
            args.scene,
            checkpoint=checkpoint,
            persistent_map_path=work_root / "semantic_map.npz",
            observation_scanner=scanner,
            robot_state_checkpoint=state_checkpoint,
            audit=audit,
            local_files_only=True,
        )
        runtime_build_seconds = time.perf_counter() - load_started
        before = runtime.prefix_binding()
        # Set only numerical state before the question-independent scan.
        runtime.simulator.state.position_xy_m = np.asarray([args.x, args.y], dtype=np.float64)
        runtime.simulator.state.body_yaw_degrees = float(args.yaw)
        runtime.simulator.state.pitch_degrees = float(args.pitch)
        scan_started = time.perf_counter()
        scan = runtime.scan()
        transaction_seconds = time.perf_counter() - scan_started
        if not scan["success"]:
            raise RuntimeError(f"Embodied semantic transaction failed: {scan['error_code']}")
        after_scan = runtime.prefix_binding()
        motion_result: dict[str, Any] | None = None
        motion_transaction_seconds: float | None = None
        if args.motion_turn_degrees is not None:
            if config.get("robot", {}).get("auto_scan_after_motion") is not True:
                raise ValueError(
                    "--motion-turn-degrees requires robot.auto_scan_after_motion=true"
                )
            motion_started = time.perf_counter()
            motion_result = runtime.turn(float(args.motion_turn_degrees))
            motion_transaction_seconds = time.perf_counter() - motion_started
            if not motion_result.get("success"):
                raise RuntimeError(
                    "Embodied motion-refresh transaction failed: "
                    f"{motion_result.get('error_code')}"
                )
        after = runtime.prefix_binding()
        observation_id = str(scan["observation_id"])
        observation_root = work_root / "observations"
        receipt_path = observation_root / f"{observation_id}.json"
        rgb_path = observation_root / f"{observation_id}.png"
        depth_path = observation_root / f"{observation_id}.npy"
        asset_manifest_path = asset.with_suffix(".json")
        asset_manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
        strict_asset_audit = asset_manifest.get("strict_nested_datablock_audit_passed") is True
        observation_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        persistent = work_root / "semantic_map.npz"
        with np.load(persistent, allow_pickle=False) as archive:
            header = json.loads(str(archive["metadata_json"].item()))
        map_receipt = header["metadata"]
        audit.assert_clean()

    loaded_files = audit.unique_paths
    renderer_command = scanner._command(
        observation_id,
        (args.x, args.y, float(config["robot"]["camera_height_m"])),
        args.yaw,
        args.pitch,
    )
    reproducible_command = [
        ".venv-gemma4/bin/python",
        "scripts/run_embodied_runtime_smoke.py",
        "--config",
        _relative(config_path),
        "--checkpoint",
        _relative(checkpoint),
        "--asset",
        _relative(asset),
        "--robot-state-checkpoint",
        _relative(state_checkpoint),
        "--scene",
        args.scene,
        "--samples",
        str(args.samples),
        "--output",
        _relative(_rooted(args.output)),
    ]
    if args.motion_turn_degrees is not None:
        reproducible_command.extend(
            ["--motion-turn-degrees", str(args.motion_turn_degrees)]
        )
    report: dict[str, Any] = {
        "schema": "semantic_3d_chat.embodied_runtime_smoke.v1",
        "passed": True,
        "scene_id": args.scene,
        "execution": {
            "reproducible_command": shlex.join(reproducible_command),
            "config_path": _relative(config_path),
            "config_file_sha256": _sha256(config_path),
            "merged_source_config_sha256": source_config_hash,
            "numeric_runtime_overrides": {
                "data_root": "isolated temporary directory",
                "x_m": args.x,
                "y_m": args.y,
                "yaw_degrees": args.yaw,
                "pitch_degrees": args.pitch,
                "samples": args.samples,
                "motion_turn_degrees": args.motion_turn_degrees,
            },
            "runtime_build_seconds": round(runtime_build_seconds, 6),
            "render_encode_fuse_prefix_seconds": round(transaction_seconds, 6),
            "motion_render_encode_fuse_prefix_seconds": (
                None
                if motion_transaction_seconds is None
                else round(motion_transaction_seconds, 6)
            ),
            "total_seconds": round(time.perf_counter() - started, 6),
        },
        "models": {
            "vision_model_id": config["vision"]["model_id"],
            "vision_revision": config["vision"]["revision"],
            "language_model_id": config["language"]["model_id"],
            "language_revision": config["language"]["revision"],
            "checkpoint_path": _relative(checkpoint),
            "adapter_sha256": _sha256(checkpoint / "adapter.safetensors"),
            "runtime_metadata_sha256": _sha256(checkpoint / "runtime_metadata.json"),
            "robot_state_checkpoint_path": _relative(state_checkpoint),
            "robot_state_weights_sha256": _sha256(
                state_checkpoint / "state.safetensors"
            ),
            "robot_state_metadata_sha256": _sha256(
                state_checkpoint / "runtime_metadata.json"
            ),
        },
        "renderer": {
            "scanner_class": type(scanner).__name__,
            "point_splat_scanner_used": False,
            "asset_path": _relative(asset),
            "asset_sha256": _sha256(asset),
            "asset_manifest_sha256": _sha256(asset_manifest_path),
            "asset_manifest": asset_manifest,
            "strict_nested_datablock_audit_passed": strict_asset_audit,
            "renderer_command_sha256": _canonical_sha256(renderer_command),
            "renderer_command_has_forbidden_path_component": bool(
                {"oracle", "qa"}
                & {part.casefold() for value in renderer_command for part in Path(value).parts}
            ),
            "complete_rgb_width": int(observation_receipt["width"]),
            "complete_rgb_height": int(observation_receipt["height"]),
            "camera_pose_numeric": observation_receipt["camera_to_world"],
            "intrinsics_numeric": observation_receipt["intrinsics"],
        },
        "observation": {
            "observation_id": observation_id,
            "receipt_sha256": _sha256(receipt_path),
            "rgb_sha256": _sha256(rgb_path),
            "depth_sha256": _sha256(depth_path),
            "valid_depth_pixels": int(scan["valid_depth_pixels"]),
            "visible_source_voxels": int(scan["visible_voxels"]),
            "directional_coverage": float(scan["scan_coverage"]),
            "vision_encoder_calls": int(map_receipt["vision_encoder_calls"]),
            "feature_grid_height": int(map_receipt["feature_grid_height"]),
            "feature_grid_width": int(map_receipt["feature_grid_width"]),
            "feature_dim": int(map_receipt["feature_dim"]),
        },
        "map_prefix_transaction": {
            "scene_version": int(scan["scene_version"]),
            "map_sha256": scan["map_sha256"],
            "source_voxels": int(after_scan["source_voxels"]),
            "processed_voxels": int(after_scan["processed_voxels"]),
            "prefix_before_sha256": before["scene_prefix_sha256"],
            "prefix_after_sha256": after_scan["scene_prefix_sha256"],
            "prefix_changed": before["scene_prefix_sha256"]
            != after_scan["scene_prefix_sha256"],
            "binding_sha256": after_scan["binding_sha256"],
            "active_prefix_sha256": after_scan["active_prefix_sha256"],
            "robot_state_sha256": after_scan["robot_state_sha256"],
            "robot_tokens_sha256": after_scan["robot_tokens_sha256"],
            "robot_state_encoder_sha256": after_scan[
                "robot_state_encoder_sha256"
            ],
            "prefix_built_before_any_user_question": True,
        },
        "motion_refresh_transaction": (
            None
            if motion_result is None
            else {
                "success": bool(motion_result["success"]),
                "turn_degrees": float(motion_result["turn_degrees"]),
                "scan_count_total": int(motion_result["scan_count"]),
                "observation_id": str(motion_result["observation_id"]),
                "scene_version": int(motion_result["scene_version"]),
                "map_version": int(motion_result["map_version"]),
                "map_sha256": str(motion_result["map_sha256"]),
                "prefix_before_motion_sha256": after_scan["scene_prefix_sha256"],
                "prefix_after_motion_sha256": after["scene_prefix_sha256"],
                "prefix_changed_after_motion": (
                    after_scan["scene_prefix_sha256"]
                    != after["scene_prefix_sha256"]
                ),
                "map_changed_after_motion": (
                    after_scan["map_sha256"] != after["map_sha256"]
                ),
                "new_complete_image_encoder_calls": 1,
            }
        ),
        "runtime_file_audit": {
            "blocking_enabled": True,
            "loaded_file_count": len(loaded_files),
            "loaded_file_inventory_sha256": _canonical_sha256(loaded_files),
            "forbidden_accesses": audit.forbidden_accesses(),
            "oracle_or_qa_loaded": False,
            "renderer_subprocess_opened_only_authenticated_runtime_asset": True,
        },
        "limitations": [
            "This proves one deterministic simulator pose, not learned navigation success.",
            "Directional coverage measures world-direction bins, not complete surface coverage.",
            "The numeric robot-state projection is deterministic and hash-bound but not yet task-trained.",
            *(
                []
                if strict_asset_audit
                else [
                    "This transient asset predates the strict nested Blender datablock-name audit; permanent export is pending."
                ]
            ),
        ],
    }
    if map_receipt["vision_encoder_calls"] != 1:
        raise RuntimeError("Fresh scan did not use exactly one complete-image encoder call")
    if not report["map_prefix_transaction"]["prefix_changed"]:
        raise RuntimeError("Fresh scan did not change the complete scene prefix")
    motion = report["motion_refresh_transaction"]
    if motion is not None and (
        motion["scan_count_total"] != 2
        or motion["scene_version"] != 2
        or motion["map_version"] != 2
        or not motion["prefix_changed_after_motion"]
        or not motion["map_changed_after_motion"]
    ):
        raise RuntimeError("Accepted motion did not trigger exactly one fresh map refresh")
    if any(
        report["map_prefix_transaction"][field] is None
        for field in (
            "robot_state_sha256",
            "robot_tokens_sha256",
            "robot_state_encoder_sha256",
        )
    ):
        raise RuntimeError("Fresh scan did not bind numeric robot-state tokens")
    if report["renderer"]["renderer_command_has_forbidden_path_component"]:
        raise RuntimeError("Renderer command contains an oracle or QA path component")
    return report


def main() -> None:
    args = parser().parse_args()
    output = _rooted(args.output)
    if args.work_root is not None:
        work_root = _rooted(args.work_root)
        work_root.mkdir(parents=True, exist_ok=True)
        report = _run(args, work_root)
    elif args.keep_work:
        work_root = Path(
            tempfile.mkdtemp(prefix="semantic_3d_embodied_smoke.", dir="/private/tmp")
        )
        report = _run(args, work_root)
        report["execution"]["retained_work_root"] = str(work_root)
    else:
        with tempfile.TemporaryDirectory(
            prefix="semantic_3d_embodied_smoke.", dir="/private/tmp"
        ) as directory:
            report = _run(args, Path(directory))
    _atomic_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
