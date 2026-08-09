"""Safe deterministic control-map generation without oracle or model access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

ABLATION_MODES: Final[tuple[str, ...]] = (
    "geometry_shuffle",
    "semantic_shuffle",
    "zero_semantics",
    "zero_rgb",
    "zero_normals",
    "zero_xyz",
    "xyz_shuffle",
)
REQUIRED_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "semantic_features",
        "centers_world",
        "mean_rgb",
        "normal",
        "confidence",
        "observation_count",
    }
)
GEOMETRY_FIELDS: Final[tuple[str, ...]] = (
    "voxel_coordinates",
    "centers_world",
    "normal",
    "normal_valid",
    "view_direction",
    "view_direction_valid",
)
XYZ_FIELDS: Final[tuple[str, ...]] = ("voxel_coordinates", "centers_world")
SEMANTIC_FIELDS: Final[tuple[str, ...]] = (
    "semantic_features",
    "semantic_feature_m2",
    "semantic_variance",
)


def _reject_oracle_path(path: str | Path, purpose: str) -> Path:
    resolved = Path(path).resolve()
    if "oracle" in {part.lower() for part in resolved.parts}:
        raise ValueError(f"{purpose} must not use the oracle directory: {resolved}")
    return resolved


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_safe_map_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Load every NPZ member with pickle disabled and validate its row schema."""

    source = _reject_oracle_path(path, "Source map")
    with np.load(source, allow_pickle=False) as archive:
        missing = REQUIRED_FIELDS - set(archive.files)
        if missing:
            raise ValueError(f"Map is missing required fields: {sorted(missing)}")
        arrays = {name: archive[name].copy() for name in archive.files}
    for name, value in arrays.items():
        if value.dtype.hasobject:
            raise TypeError(f"Unsafe object array in map field {name}")
    semantic = arrays["semantic_features"]
    centers = arrays["centers_world"]
    if semantic.ndim != 2 or semantic.shape[0] < 1:
        raise ValueError("semantic_features must be a non-empty [N, D] matrix")
    count = semantic.shape[0]
    if centers.shape != (count, 3) or not np.isfinite(centers).all():
        raise ValueError("centers_world must be finite with shape [N, 3]")
    expected_shapes = {
        "mean_rgb": (count, 3),
        "normal": (count, 3),
        "confidence": (count,),
        "observation_count": (count,),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {arrays[name].shape}")
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} contains NaN or infinite values")
    if not np.isfinite(semantic).all():
        raise ValueError("semantic_features contains NaN or infinite values")
    for name in (
        "voxel_coordinates",
        "normal_valid",
        "view_direction",
        "view_direction_valid",
        "semantic_feature_m2",
        "semantic_variance",
        "weight_sum",
        "last_frame",
    ):
        if name in arrays and arrays[name].shape[0] != count:
            raise ValueError(f"Optional field {name} does not share voxel count {count}")
    return arrays


def deterministic_permutation(count: int, seed: int) -> np.ndarray:
    if count < 1:
        raise ValueError("Permutation count must be positive")
    if seed < 0:
        raise ValueError("Ablation seed must be non-negative")
    permutation = np.random.Generator(np.random.PCG64(seed)).permutation(count)
    if count > 1 and np.array_equal(permutation, np.arange(count)):
        permutation = np.roll(permutation, 1)
    return permutation.astype(np.int64, copy=False)


def _permutation_hash(permutation: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(permutation, dtype="<i8").tobytes()).hexdigest()


def apply_ablation(
    source_arrays: Mapping[str, np.ndarray],
    mode: str,
    *,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Apply one control while retaining keys, shapes, and safe dtypes."""

    if mode not in ABLATION_MODES:
        raise ValueError(f"Unknown ablation {mode!r}; choose from {ABLATION_MODES}")
    arrays = {name: np.asarray(value).copy() for name, value in source_arrays.items()}
    missing = REQUIRED_FIELDS - set(arrays)
    if missing:
        raise ValueError(f"Map is missing required fields: {sorted(missing)}")
    count = int(arrays["semantic_features"].shape[0])
    permutation: np.ndarray | None = None
    affected_fields: list[str] = []

    if mode in {"geometry_shuffle", "semantic_shuffle", "xyz_shuffle"}:
        permutation = deterministic_permutation(count, seed)
    if mode == "geometry_shuffle":
        # Move the complete geometric bundle while leaving appearance and
        # semantic payload rows fixed, breaking their physical association.
        for name in GEOMETRY_FIELDS:
            if name in arrays:
                arrays[name] = arrays[name][permutation]
                affected_fields.append(name)
    elif mode == "semantic_shuffle":
        for name in SEMANTIC_FIELDS:
            if name in arrays:
                arrays[name] = arrays[name][permutation]
                affected_fields.append(name)
    elif mode == "xyz_shuffle":
        # Shuffle only spatial coordinates; normals/view directions remain as
        # non-positional payloads for a narrower spatial-reasoning control.
        for name in XYZ_FIELDS:
            if name in arrays:
                arrays[name] = arrays[name][permutation]
                affected_fields.append(name)
    elif mode == "zero_semantics":
        for name in SEMANTIC_FIELDS:
            if name in arrays:
                arrays[name] = np.zeros_like(arrays[name])
                affected_fields.append(name)
    elif mode == "zero_rgb":
        arrays["mean_rgb"] = np.zeros_like(arrays["mean_rgb"])
        affected_fields.append("mean_rgb")
    elif mode == "zero_normals":
        arrays["normal"] = np.zeros_like(arrays["normal"])
        affected_fields.append("normal")
        if "normal_valid" in arrays:
            arrays["normal_valid"] = np.zeros_like(arrays["normal_valid"], dtype=bool)
            affected_fields.append("normal_valid")
    elif mode == "zero_xyz":
        for name in XYZ_FIELDS:
            if name in arrays:
                arrays[name] = np.zeros_like(arrays[name])
                affected_fields.append(name)

    metadata = {
        "mode": mode,
        "seed": int(seed),
        "voxel_count": count,
        "affected_fields": affected_fields,
        "permutation_algorithm": "numpy.PCG64" if permutation is not None else None,
        "permutation_sha256": (_permutation_hash(permutation) if permutation is not None else None),
    }
    return arrays, metadata


def _update_metadata(
    arrays: dict[str, np.ndarray], metadata: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    result = {name: value.copy() for name, value in arrays.items()}
    header: dict[str, Any] = {}
    if "metadata_json" in result:
        raw = result["metadata_json"]
        if raw.shape != () or raw.dtype.kind not in {"U", "S"}:
            raise ValueError("metadata_json must be a scalar string")
        raw_value = raw.item()
        serialized = (
            raw_value.decode("utf-8")
            if isinstance(raw_value, (bytes, np.bytes_))
            else str(raw_value)
        )
        header = json.loads(serialized)
        if not isinstance(header, dict):
            raise TypeError("metadata_json must decode to an object")
    header["ablation"] = dict(metadata)
    result["metadata_json"] = np.asarray(json.dumps(header, sort_keys=True))
    return result


def save_safe_map_arrays(
    path: str | Path,
    arrays: Mapping[str, np.ndarray],
    *,
    overwrite: bool = False,
) -> Path:
    destination = _reject_oracle_path(path, "Ablation output")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing ablation: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        if np.asarray(value).dtype.hasobject:
            raise TypeError(f"Refusing to save unsafe object field {name}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination


def create_ablation_map(
    source_path: str | Path,
    destination_path: str | Path,
    mode: str,
    *,
    seed: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = _reject_oracle_path(source_path, "Source map")
    source_hash = file_sha256(source)
    source_arrays = load_safe_map_arrays(source)
    arrays, metadata = apply_ablation(source_arrays, mode, seed=seed)
    metadata = {**metadata, "source_sha256": source_hash}
    output = save_safe_map_arrays(
        destination_path,
        _update_metadata(arrays, metadata),
        overwrite=overwrite,
    )
    return {
        **metadata,
        "path": str(output),
        "sha256": file_sha256(output),
    }


def create_ablation_suite(
    source_path: str | Path,
    output_directory: str | Path,
    *,
    seed: int,
    modes: Sequence[str] = ABLATION_MODES,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not modes:
        raise ValueError("At least one ablation mode is required")
    if len(set(modes)) != len(modes):
        raise ValueError("Ablation modes must be unique")
    source = _reject_oracle_path(source_path, "Source map")
    output_root = _reject_oracle_path(output_directory, "Ablation output directory")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    destinations = [output_root / f"map_{mode}.npz" for mode in modes]
    if not overwrite:
        existing = [path for path in [*destinations, manifest_path] if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing partial ablation-suite overwrite: {existing}")
    results = [
        create_ablation_map(
            source,
            destination,
            mode,
            seed=seed,
            overwrite=overwrite,
        )
        for mode, destination in zip(modes, destinations, strict=True)
    ]
    manifest = {
        "schema_version": 1,
        "source_path": str(source),
        "source_sha256": file_sha256(source),
        "seed": seed,
        "ablations": results,
    }
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {**manifest, "manifest_path": str(manifest_path)}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--mode", action="append", choices=ABLATION_MODES)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = create_ablation_suite(
        args.map,
        args.output_dir,
        seed=args.seed,
        modes=args.mode or ABLATION_MODES,
        overwrite=args.force,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
