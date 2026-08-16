#!/usr/bin/env python3
"""Authenticate every local artifact required by the strict one-command demo.

This is a finite, read-only readiness check. It neither loads Gemma tensors nor
starts Blender, and it never opens oracle, QA, render, or training metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from huggingface_hub import try_to_load_from_cache

from semantic_3d_chat.config import PROJECT_ROOT

_SHA256 = frozenset("0123456789abcdef")
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "artifacts",
        "model",
        "distribution_url",
        "environmental_text_inputs",
    }
)
_ARTIFACT_FIELDS = frozenset({"path", "role", "size_bytes", "sha256"})
_MODEL_FIELDS = frozenset(
    {
        "model_id",
        "revision",
        "required_files",
        "weights_size_bytes",
        "weights_sha256",
    }
)
_PROHIBITED_PATH_COMPONENTS = frozenset({"oracle", "qa", "rendered", "features", "training"})


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return (candidate if candidate.is_absolute() else PROJECT_ROOT / candidate).resolve()


def _safe_regular(path: Path, purpose: str) -> Path:
    unresolved = Path(os.path.abspath(path))
    current = Path(unresolved.anchor)
    for component in unresolved.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{purpose} path contains a symbolic link: {current}")
    if not unresolved.is_file():
        raise FileNotFoundError(f"{purpose} is unavailable: {unresolved}")
    return unresolved


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value).issubset(_SHA256)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: str | Path) -> dict[str, Any]:
    source = _safe_regular(_rooted(path), "Demo artifact manifest")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != _TOP_LEVEL_FIELDS:
        raise ValueError("Demo artifact manifest fields changed")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact") != "semantic_3d_chat_local_demo_artifacts_v1"
        or payload.get("distribution_url") is not None
        or payload.get("environmental_text_inputs") != []
    ):
        raise ValueError("Demo artifact manifest contract changed")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Demo artifact inventory is empty")
    observed_paths: set[str] = set()
    for entry in artifacts:
        if not isinstance(entry, Mapping) or set(entry) != _ARTIFACT_FIELDS:
            raise ValueError("Demo artifact entry fields changed")
        path = entry.get("path")
        role = entry.get("role")
        size = entry.get("size_bytes")
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or path in observed_paths
            or _PROHIBITED_PATH_COMPONENTS & {part.casefold() for part in Path(path).parts}
            or not isinstance(role, str)
            or not role
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 1
            or not _is_sha256(entry.get("sha256"))
        ):
            raise ValueError(f"Demo artifact entry is invalid: {path!r}")
        observed_paths.add(path)
    model = payload.get("model")
    if not isinstance(model, Mapping) or set(model) != _MODEL_FIELDS:
        raise ValueError("Demo model contract fields changed")
    required = model.get("required_files")
    if (
        not isinstance(model.get("model_id"), str)
        or not isinstance(model.get("revision"), str)
        or not isinstance(required, list)
        or not required
        or any(not isinstance(name, str) or Path(name).name != name for name in required)
        or "model.safetensors" not in required
        or isinstance(model.get("weights_size_bytes"), bool)
        or not isinstance(model.get("weights_size_bytes"), int)
        or model["weights_size_bytes"] < 1
        or not _is_sha256(model.get("weights_sha256"))
    ):
        raise ValueError("Demo model contract is invalid")
    return dict(payload)


def check_readiness(
    manifest: Mapping[str, Any], *, verify_model_hash: bool = True
) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []
    ready = True
    for expected in manifest["artifacts"]:
        path = _rooted(str(expected["path"]))
        entry = {
            "path": str(expected["path"]),
            "role": str(expected["role"]),
            "required_size_bytes": int(expected["size_bytes"]),
            "required_sha256": str(expected["sha256"]),
            "exists": path.is_file() and not path.is_symlink(),
            "valid": False,
        }
        if entry["exists"]:
            entry["observed_size_bytes"] = path.stat().st_size
            entry["size_matches"] = path.stat().st_size == expected["size_bytes"]
            if entry["size_matches"]:
                entry["observed_sha256"] = _sha256(path)
                entry["sha256_matches"] = entry["observed_sha256"] == expected["sha256"]
            else:
                entry["sha256_matches"] = None
            entry["valid"] = bool(entry["size_matches"] and entry["sha256_matches"] is True)
        ready = ready and bool(entry["valid"])
        artifacts.append(entry)
        if not entry["valid"]:
            problems.append(
                {
                    "kind": "project_artifact",
                    "path": entry["path"],
                    "role": entry["role"],
                    "exists": entry["exists"],
                    "required_size_bytes": entry["required_size_bytes"],
                    "required_sha256": entry["required_sha256"],
                }
            )

    model = manifest["model"]
    model_files: list[dict[str, Any]] = []
    model_ready = True
    for filename in model["required_files"]:
        cached = try_to_load_from_cache(
            str(model["model_id"]),
            filename,
            revision=str(model["revision"]),
        )
        path = None if not isinstance(cached, str) else Path(cached)
        entry = {"filename": filename, "exists": bool(path and path.is_file()), "valid": False}
        if filename == "model.safetensors":
            entry["required_size_bytes"] = int(model["weights_size_bytes"])
            entry["required_sha256"] = str(model["weights_sha256"])
        if path and path.is_file():
            entry["observed_size_bytes"] = path.stat().st_size
            if filename == "model.safetensors":
                entry["size_matches"] = path.stat().st_size == model["weights_size_bytes"]
                if verify_model_hash and entry["size_matches"]:
                    entry["observed_sha256"] = _sha256(path)
                    entry["sha256_matches"] = entry["observed_sha256"] == model["weights_sha256"]
                else:
                    entry["sha256_matches"] = None
                entry["valid"] = bool(
                    entry["size_matches"]
                    and (entry["sha256_matches"] is True or not verify_model_hash)
                )
            else:
                entry["valid"] = True
        model_ready = model_ready and bool(entry["valid"])
        model_files.append(entry)
        if not entry["valid"]:
            problem = {
                "kind": "model_file",
                "model_id": model["model_id"],
                "revision": model["revision"],
                "filename": filename,
                "exists": entry["exists"],
            }
            for field in ("required_size_bytes", "required_sha256"):
                if field in entry:
                    problem[field] = entry[field]
            problems.append(problem)
    ready = ready and model_ready
    return {
        "schema_version": 1,
        "artifact": manifest["artifact"],
        "ready": ready,
        "artifact_hashes_verified": True,
        "model_weights_hash_verified": verify_model_hash,
        "loads_model": False,
        "runs_blender": False,
        "environmental_text_inputs": [],
        "distribution_url": None,
        "distribution_configured": False,
        "missing_or_invalid": problems,
        "artifacts": artifacts,
        "model": {
            "model_id": model["model_id"],
            "revision": model["revision"],
            "files": model_files,
            "ready": model_ready,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/runtime/demo_artifacts_v1.json")
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Hash every project artifact, but verify the 10.25 GB model by exact "
            "revision/path/size without rehashing it."
        ),
    )
    args = parser.parse_args(argv)
    result = check_readiness(load_manifest(args.manifest), verify_model_hash=not args.fast)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
