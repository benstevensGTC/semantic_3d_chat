"""Offline, non-inference preflight for the prepared Semantic 3D Chat demo."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import struct
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from semantic_3d_chat.config import (
    PROJECT_ROOT,
    artifact_root,
    config_hash,
    load_config,
    project_path,
    reports_root,
)

SCENE_ID = re.compile(r"scene_[0-9]{6}")
FRAME_ID = re.compile(r"f_[0-9]{6}")
SUPPORTED_CHECKPOINT_SCHEMAS = {1, 2, 3}
PROMOTION_SCHEMA_VERSION = 1
SCENE_TOKENIZER_CONTRACT_DEFAULTS: dict[str, int | float | None] = {
    "language_aligned_tail_dim": 0,
    "native_aligned_coverage_scale": 0.0,
    "learned_scene_token_scale": 1.0,
    "learned_scene_token_rms_target": None,
}
REQUIRED_PACKAGES = ("numpy", "safetensors", "torch", "transformers", "yaml")
MAP_ARRAYS = (
    "centers_world.npy",
    "semantic_features.npy",
    "mean_rgb.npy",
    "normal.npy",
    "confidence.npy",
    "observation_count.npy",
)
FORBIDDEN_MANIFEST_KEY_TERMS = (
    "bbox",
    "caption",
    "category",
    "color",
    "instance",
    "label",
    "name",
    "object",
    "relationship",
    "segmentation",
    "semantic",
    "support",
    "visibility",
)


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _equal_number(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    if left is None or right is None:
        return left is right
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _scene_tokenizer_contract(config: dict[str, Any]) -> dict[str, int | float | None]:
    settings = config["scene_encoder"]
    rms_target = settings.get("learned_scene_token_rms_target")
    return {
        "language_aligned_tail_dim": int(settings.get("language_aligned_tail_dim", 0)),
        "native_aligned_coverage_scale": float(settings.get("native_aligned_coverage_scale", 0.0)),
        "learned_scene_token_scale": float(settings.get("learned_scene_token_scale", 1.0)),
        "learned_scene_token_rms_target": (None if rms_target is None else float(rms_target)),
    }


def _checkpoint_contract_errors(metadata: dict[str, Any], config: dict[str, Any]) -> list[str]:
    required = {
        "config_hash",
        "input_voxel_size_m",
        "language_hidden_dim",
        "language_model_id",
        "language_revision",
        "scene_latents",
        "scene_model_dim",
        "schema_version",
        "semantic_dim",
    }
    scene_tokenizer_contract = _scene_tokenizer_contract(config)
    uses_aligned_bypass = any(
        not _equal_number(value, SCENE_TOKENIZER_CONTRACT_DEFAULTS[key])
        for key, value in scene_tokenizer_contract.items()
    )
    metadata_has_scene_tokenizer_contract = any(key in metadata for key in scene_tokenizer_contract)
    if uses_aligned_bypass or metadata_has_scene_tokenizer_contract:
        required.update(scene_tokenizer_contract)
    errors = [
        f"missing required metadata field: {key}" for key in sorted(required - metadata.keys())
    ]
    expected = {
        "language_model_id": str(config["language"]["model_id"]),
        "language_revision": str(config["language"]["revision"]),
        "scene_latents": int(config["scene_encoder"]["global_latents"]),
        "scene_model_dim": int(config["scene_encoder"]["model_dim"]),
    }
    architecture = config["scene_encoder"].get("architecture_version")
    if architecture is not None:
        expected["scene_encoder_architecture_version"] = str(architecture)
    errors.extend(
        f"{key}: checkpoint={metadata.get(key)!r}, config={value!r}"
        for key, value in expected.items()
        if metadata.get(key) != value
    )
    errors.extend(
        f"{key}: checkpoint={metadata[key]!r}, config={value!r}"
        for key, value in scene_tokenizer_contract.items()
        if key in metadata and not _equal_number(metadata[key], value)
    )
    configured_voxel_size = config["scene_encoder"].get("input_voxel_size_m")
    checkpoint_voxel_size = metadata.get("input_voxel_size_m")
    voxel_size_matches = _equal_number(checkpoint_voxel_size, configured_voxel_size)
    if not voxel_size_matches:
        errors.append(
            "input_voxel_size_m: "
            f"checkpoint={checkpoint_voxel_size!r}, config={configured_voxel_size!r}"
        )
    if metadata.get("schema_version") not in SUPPORTED_CHECKPOINT_SCHEMAS:
        errors.append(f"unsupported schema_version={metadata.get('schema_version')!r}")
    return errors


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_promotion(checkpoint: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Validate an explicit behavioral-promotion record for a checkpoint/config pair."""

    promotion_path = checkpoint / "promotion.json"
    promotion = _read_json_object(promotion_path)
    expected = {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "status": "accepted",
        "config_hash": config_hash(config),
        "checkpoint_metadata_sha256": _sha256_file(checkpoint / "metadata.json"),
        "checkpoint_adapter_sha256": _sha256_file(checkpoint / "adapter.safetensors"),
    }
    errors = [
        f"{key}: promotion={promotion.get(key)!r}, expected={value!r}"
        for key, value in expected.items()
        if promotion.get(key) != value
    ]
    evidence = promotion.get("evidence")
    if (
        not isinstance(evidence, list)
        or not evidence
        or not all(isinstance(item, str) and item.strip() for item in evidence)
    ):
        errors.append("evidence must be a non-empty list of artifact paths")
    if errors:
        raise ValueError("invalid behavioral promotion record: " + "; ".join(errors))
    return {
        "path": str(promotion_path.resolve()),
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "status": "accepted",
        "config_hash": expected["config_hash"],
        "checkpoint_metadata_sha256": expected["checkpoint_metadata_sha256"],
        "checkpoint_adapter_sha256": expected["checkpoint_adapter_sha256"],
        "evidence": evidence,
    }


def resolve_checkpoint(config: dict[str, Any], requested: str | Path | None = None) -> Path:
    """Select an architecture-compatible prepared checkpoint without loading tensors."""

    checkpoint_root = artifact_root(config, "checkpoints")
    if requested is not None:
        candidates = [_rooted(requested)]
    else:
        candidates: list[Path] = []
        namespace = config.get("training", {}).get("output_namespace")
        if namespace:
            candidates.append(checkpoint_root / str(namespace) / "best")
        candidates.append(checkpoint_root / "best")
        candidates.extend(sorted(checkpoint_root.glob("*/best")))
        candidates = list(dict.fromkeys(path.resolve() for path in candidates))

    compatible: list[tuple[tuple[int, int, int, int], Path]] = []
    rejected: list[str] = []
    configured_namespace = config.get("training", {}).get("output_namespace")
    for candidate in candidates:
        metadata_path = candidate / "metadata.json"
        adapter_path = candidate / "adapter.safetensors"
        if not metadata_path.is_file() or not adapter_path.is_file():
            rejected.append(f"{candidate}: checkpoint files are missing")
            continue
        try:
            metadata = _read_json_object(metadata_path)
        except (OSError, ValueError, TypeError) as exc:
            rejected.append(f"{candidate}: invalid metadata ({exc})")
            continue
        errors = _checkpoint_contract_errors(metadata, config)
        if errors:
            rejected.append(f"{candidate}: " + "; ".join(errors))
            continue
        score = (
            int(metadata.get("output_namespace") == configured_namespace),
            int(metadata.get("config_hash") == config_hash(config)),
            int(metadata.get("epoch", 0)),
            int(metadata_path.stat().st_mtime_ns),
        )
        compatible.append((score, candidate.resolve()))
    if not compatible:
        detail = "\n  ".join(rejected) if rejected else "no checkpoint candidates found"
        raise FileNotFoundError(
            "No prepared checkpoint is compatible with the selected config. Checked:\n  " + detail
        )
    return max(compatible, key=lambda item: item[0])[1]


def _npy_header(handle: Any) -> tuple[tuple[int, ...], np.dtype[Any]]:
    version = np.lib.format.read_magic(handle)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
    else:
        raise ValueError(f"Unsupported NPY header version: {version}")
    return tuple(int(value) for value in shape), np.dtype(dtype)


def inspect_map(path: Path) -> dict[str, Any]:
    """Read only NPZ headers; the 2048-D voxel payload is never materialized."""

    headers: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        missing = sorted(set(MAP_ARRAYS) - names)
        if missing:
            raise ValueError(f"Map is missing arrays: {missing}")
        for name in MAP_ARRAYS:
            with archive.open(name) as handle:
                shape, dtype = _npy_header(handle)
            headers[name] = {"shape": list(shape), "dtype": str(dtype)}
    voxel_count = headers["centers_world.npy"]["shape"][0]
    semantic_shape = headers["semantic_features.npy"]["shape"]
    expected_shapes = {
        "centers_world.npy": [voxel_count, 3],
        "mean_rgb.npy": [voxel_count, 3],
        "normal.npy": [voxel_count, 3],
        "confidence.npy": [voxel_count],
        "observation_count.npy": [voxel_count],
    }
    failures = [
        f"{name} has {headers[name]['shape']}, expected {shape}"
        for name, shape in expected_shapes.items()
        if headers[name]["shape"] != shape
    ]
    if len(semantic_shape) != 2 or semantic_shape[0] != voxel_count:
        failures.append(f"semantic_features.npy has {semantic_shape}, expected [{voxel_count}, D]")
    if voxel_count <= 0:
        failures.append("map has no occupied voxels")
    if failures:
        raise ValueError("; ".join(failures))
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "voxel_count": voxel_count,
        "semantic_dim": semantic_shape[1],
        "semantic_dtype": headers["semantic_features.npy"]["dtype"],
        "arrays": headers,
    }


def inspect_safetensors(path: Path) -> dict[str, Any]:
    """Validate the safetensors envelope without importing torch or reading tensor data."""

    file_size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors header")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size <= 2 or 8 + header_size > file_size:
            raise ValueError("invalid safetensors header length")
        header = json.loads(handle.read(header_size))
    tensors = [key for key in header if key != "__metadata__"]
    if not tensors:
        raise ValueError("checkpoint contains no tensors")
    return {"path": str(path), "bytes": file_size, "tensor_count": len(tensors)}


def inspect_manifest(path: Path, scene_id: str) -> dict[str, Any]:
    manifest = _read_json_object(path)
    pending: list[Any] = [manifest]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            for key, nested in value.items():
                lowered = str(key).casefold()
                if any(term in lowered for term in FORBIDDEN_MANIFEST_KEY_TERMS):
                    raise ValueError(f"Sanitized render manifest contains forbidden key: {key}")
                pending.append(nested)
        elif isinstance(value, list):
            pending.extend(value)
    if manifest.get("scene_id") != scene_id:
        raise ValueError(f"Manifest scene mismatch: {manifest.get('scene_id')!r} != {scene_id!r}")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Sanitized render manifest has no frames")
    root = path.parent.resolve()
    seen: set[str] = set()
    verified_artifact_count = 0
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise TypeError(f"Frame {index} is not an object")
        frame_id = frame.get("frame_id")
        if not isinstance(frame_id, str) or not FRAME_ID.fullmatch(frame_id):
            raise ValueError(f"Frame {index} has a non-opaque frame ID: {frame_id!r}")
        if frame_id in seen:
            raise ValueError(f"Duplicate frame ID: {frame_id}")
        seen.add(frame_id)
        intrinsics = np.asarray(frame.get("intrinsics"), dtype=np.float64)
        pose = np.asarray(frame.get("camera_to_world"), dtype=np.float64)
        if intrinsics.shape != (3, 3) or pose.shape != (4, 4):
            raise ValueError(f"Frame {frame_id} has invalid camera matrix shapes")
        if not np.isfinite(intrinsics).all() or not np.isfinite(pose).all():
            raise ValueError(f"Frame {frame_id} has non-finite camera matrices")
        for field in ("rgb_path", "depth_path"):
            relative = frame.get(field)
            if not isinstance(relative, str) or Path(relative).is_absolute():
                raise ValueError(f"Frame {frame_id} has invalid {field}")
            artifact = (root / relative).resolve()
            try:
                artifact.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"Frame {frame_id} {field} escapes the scene directory") from exc
            if not artifact.is_file():
                raise FileNotFoundError(f"Frame {frame_id} is missing {field}: {artifact}")
            expected_relative = (
                f"rgb/{frame_id}.png" if field == "rgb_path" else f"depth/{frame_id}.npy"
            )
            if relative != expected_relative:
                raise ValueError(
                    f"Frame {frame_id} {field} is not an opaque canonical filename: {relative}"
                )
            verified_artifact_count += 1
    return {
        "path": str(path),
        "frame_count": len(frames),
        "image_size": manifest.get("image_size"),
        "verified_frame_artifact_count": verified_artifact_count,
    }


def _hub_root() -> Path:
    if configured := os.environ.get("HF_HUB_CACHE"):
        return Path(configured).expanduser().resolve()
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return (home / "hub").expanduser().resolve()


def inspect_model_cache(model_id: str, revision: str) -> dict[str, Any]:
    snapshot = _hub_root() / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
    required = (snapshot / "config.json", snapshot / "tokenizer_config.json")
    missing = [str(path) for path in required if not path.is_file()]
    weights = [
        path
        for pattern in ("*.safetensors", "*.bin")
        for path in snapshot.glob(pattern)
        if path.is_file()
    ]
    if missing or not weights:
        detail = list(missing)
        if not weights:
            detail.append(f"weights below {snapshot}")
        raise FileNotFoundError("Missing cached model files: " + ", ".join(detail))
    model_config = _read_json_object(snapshot / "config.json")
    hidden_size = model_config.get("hidden_size")
    if hidden_size is None and isinstance(model_config.get("text_config"), dict):
        hidden_size = model_config["text_config"].get("hidden_size")
    return {
        "model_id": model_id,
        "revision": revision,
        "snapshot": str(snapshot),
        "weight_files": [str(path) for path in sorted(set(weights))],
        "hidden_size": hidden_size,
    }


def run_checks(
    config_path: str | Path,
    scene_id: str,
    checkpoint: str | Path | None = None,
    *,
    require_promotion: bool = False,
) -> dict[str, Any]:
    if not SCENE_ID.fullmatch(scene_id):
        raise ValueError("scene must be opaque and match scene_ followed by six digits")
    resolved_config = _rooted(config_path)
    config = load_config(resolved_config)
    checks: list[Check] = []
    artifacts: dict[str, Any] = {}

    for executable in ("blender", "uv"):
        location = shutil.which(executable)
        checks.append(
            Check(f"executable:{executable}", location is not None, location or "missing")
        )
    for package in REQUIRED_PACKAGES:
        found = importlib.util.find_spec(package) is not None
        checks.append(
            Check(f"python_package:{package}", found, "available" if found else "missing")
        )

    selected_checkpoint: Path | None = None
    checkpoint_metadata: dict[str, Any] | None = None
    try:
        selected_checkpoint = resolve_checkpoint(config, checkpoint)
        checkpoint_metadata = _read_json_object(selected_checkpoint / "metadata.json")
        errors = _checkpoint_contract_errors(checkpoint_metadata, config)
        if errors:
            raise ValueError("; ".join(errors))
        adapter = inspect_safetensors(selected_checkpoint / "adapter.safetensors")
        artifacts["checkpoint"] = {
            "path": str(selected_checkpoint),
            "schema_version": checkpoint_metadata.get("schema_version"),
            "epoch": checkpoint_metadata.get("epoch"),
            "output_namespace": checkpoint_metadata.get("output_namespace"),
            "architecture_version": checkpoint_metadata.get("scene_encoder_architecture_version"),
            "scene_tokenizer_contract": {
                key: checkpoint_metadata.get(key) for key in SCENE_TOKENIZER_CONTRACT_DEFAULTS
            },
            "config_hash_matches": checkpoint_metadata.get("config_hash") == config_hash(config),
            "adapter": adapter,
        }
        checks.append(Check("checkpoint_contract", True, str(selected_checkpoint)))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        selected_checkpoint = None
        checkpoint_metadata = None
        checks.append(Check("checkpoint_contract", False, str(exc)))

    if selected_checkpoint is not None and require_promotion:
        try:
            promotion = inspect_promotion(selected_checkpoint, config)
            artifacts["behavioral_promotion"] = promotion
            checks.append(Check("behavioral_promotion", True, promotion["path"]))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            checks.append(Check("behavioral_promotion", False, str(exc)))

    manifest_path = project_path(config, "rendered", scene_id, "manifest.json")
    try:
        artifacts["render_scan"] = inspect_manifest(manifest_path, scene_id)
        checks.append(Check("sanitized_render_scan", True, str(manifest_path)))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        checks.append(Check("sanitized_render_scan", False, str(exc)))

    map_path = project_path(config, "maps", scene_id, "voxel_map.npz")
    try:
        map_summary = inspect_map(map_path)
        artifacts["voxel_map"] = map_summary
        checks.append(Check("numeric_voxel_map", True, str(map_path)))
        if checkpoint_metadata is not None:
            expected_dim = int(checkpoint_metadata.get("semantic_dim", -1))
            actual_dim = int(map_summary["semantic_dim"])
            checks.append(
                Check(
                    "map_checkpoint_semantic_dim",
                    actual_dim == expected_dim,
                    f"map={actual_dim}, checkpoint={expected_dim}",
                )
            )
    except (OSError, ValueError, TypeError, zipfile.BadZipFile) as exc:
        checks.append(Check("numeric_voxel_map", False, str(exc)))

    for role in ("vision", "language"):
        selection = config[role]
        try:
            artifacts[f"{role}_model_cache"] = inspect_model_cache(
                str(selection["model_id"]), str(selection["revision"])
            )
            checks.append(Check(f"local_model_cache:{role}", True, str(selection["model_id"])))
            if role == "language" and checkpoint_metadata is not None:
                cached_hidden_size = artifacts["language_model_cache"].get("hidden_size")
                checkpoint_hidden_size = checkpoint_metadata.get("language_hidden_dim")
                try:
                    hidden_size_matches = int(cached_hidden_size) == int(checkpoint_hidden_size)
                except (TypeError, ValueError):
                    hidden_size_matches = False
                checks.append(
                    Check(
                        "language_cache_checkpoint_hidden_size",
                        hidden_size_matches,
                        f"cache={cached_hidden_size}, checkpoint={checkpoint_hidden_size}",
                    )
                )
        except (OSError, ValueError, TypeError) as exc:
            checks.append(Check(f"local_model_cache:{role}", False, str(exc)))

    configured_reports_root = reports_root(config)
    previews = {
        "overview": project_path(config, "rendered", scene_id, "p_000000.png"),
        "map_rgb": configured_reports_root / "figures" / scene_id / "map_rgb.png",
        "map_point_cloud": configured_reports_root / "figures" / scene_id / "map_rgb.ply",
    }
    artifacts["visuals"] = {
        name: str(path.resolve()) for name, path in previews.items() if path.is_file()
    }
    for name, path in previews.items():
        checks.append(Check(f"visual:{name}", path.is_file(), str(path)))

    inspected_paths = [str(resolved_config)]
    if selected_checkpoint is not None:
        inspected_paths.extend(
            [
                str(selected_checkpoint / "metadata.json"),
                str(selected_checkpoint / "adapter.safetensors"),
            ]
        )
        if require_promotion:
            inspected_paths.append(str(selected_checkpoint / "promotion.json"))
    if "render_scan" in artifacts:
        inspected_paths.append(str(manifest_path))
    if "voxel_map" in artifacts:
        inspected_paths.append(str(map_path))
    forbidden_parts = {"oracle", "qa", "features"}
    forbidden_inspected = [
        path for path in inspected_paths if forbidden_parts.intersection(Path(path).parts)
    ]
    passed = all(check.passed for check in checks) and not forbidden_inspected
    return {
        "schema_version": 1,
        "mode": "static_preflight",
        "passed": passed,
        "inference_performed": False,
        "device_tensor_created": False,
        "scene_id": scene_id,
        "config": str(resolved_config),
        "checkpoint": None if selected_checkpoint is None else str(selected_checkpoint),
        "checks": [asdict(check) for check in checks],
        "artifacts": artifacts,
        "inspected_paths": sorted(set(inspected_paths)),
        "forbidden_inspected_paths": forbidden_inspected,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="configs/default.yaml")
    result.add_argument("--scene", default="scene_000001")
    result.add_argument("--checkpoint")
    result.add_argument(
        "--resolve-checkpoint",
        action="store_true",
        help="Print the selected compatible checkpoint path and perform no artifact checks.",
    )
    result.add_argument(
        "--require-promotion",
        action="store_true",
        help=(
            "Require promotion.json beside the checkpoint, bound to the adapter, "
            "metadata, and exact selected config."
        ),
    )
    result.add_argument("--output", default="reports/metrics/demo_check.json")
    result.add_argument("--no-write", action="store_true")
    result.add_argument(
        "--verbose",
        action="store_true",
        help="Print the complete report instead of a compact terminal summary.",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_config(_rooted(args.config))
    if args.resolve_checkpoint:
        try:
            checkpoint = resolve_checkpoint(config, args.checkpoint)
            if args.require_promotion:
                inspect_promotion(checkpoint, config)
            print(checkpoint)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            print(str(exc), file=sys.stderr)
            return 4
        return 0
    report = run_checks(
        args.config,
        args.scene,
        args.checkpoint,
        require_promotion=args.require_promotion,
    )
    if not args.no_write:
        output = _rooted(args.output)
        report["output"] = str(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    failures = [check for check in report["checks"] if not check["passed"]]
    map_summary = report["artifacts"].get("voxel_map", {})
    render_summary = report["artifacts"].get("render_scan", {})
    terminal_report = (
        report
        if args.verbose
        else {
            "passed": report["passed"],
            "mode": report["mode"],
            "inference_performed": report["inference_performed"],
            "device_tensor_created": report["device_tensor_created"],
            "scene_id": report["scene_id"],
            "checkpoint": report["checkpoint"],
            "frame_count": render_summary.get("frame_count"),
            "voxel_count": map_summary.get("voxel_count"),
            "semantic_dim": map_summary.get("semantic_dim"),
            "checks_passed": len(report["checks"]) - len(failures),
            "checks_total": len(report["checks"]),
            "failed_checks": failures,
            "output": report.get("output"),
        }
    )
    print(json.dumps(terminal_report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
