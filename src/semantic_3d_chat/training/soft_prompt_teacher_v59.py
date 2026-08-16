"""Cached numeric Gemma soft-prompt teachers for the V59 expansion gate."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from semantic_3d_chat.config import PROJECT_ROOT

_SHA256_LENGTH = 64


@dataclass(frozen=True)
class ExpansionTeacherTarget:
    scene_id: str
    question_id: str
    tokens: torch.Tensor

    def __post_init__(self) -> None:
        if not self.scene_id.startswith("scene_") or not self.question_id:
            raise ValueError("V59 teacher targets require opaque scene/question IDs")
        if self.tokens.ndim != 3 or self.tokens.shape[0] != 1:
            raise ValueError("V59 teacher tokens must have shape [1,C,H]")
        if not self.tokens.is_floating_point() or not torch.isfinite(self.tokens).all():
            raise ValueError("V59 teacher tokens must be finite floating point")

    @property
    def key(self) -> tuple[str, str]:
        return self.scene_id, self.question_id


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    rooted = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(rooted))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"V59 {field} must be a lowercase SHA-256 digest")
    return value


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"V59 teacher cache path contains a symlink: {current}")


def _path_guard(path: str | Path) -> Path:
    root = _resolve(path)
    _reject_symlink_components(root)
    try:
        scoped = root.relative_to(PROJECT_ROOT)
    except ValueError:
        scoped = Path(root.name)
    forbidden = {"oracle", "validation", "development", "test", "final", "v55"}
    tokens = {
        token
        for part in scoped.parts
        for token in part.casefold().replace("-", "_").split("_")
    }
    if forbidden & tokens:
        raise ValueError("V59 teacher cache path targets a forbidden split")
    return root


def save_expansion_teachers(
    destination: str | Path,
    *,
    targets: Sequence[ExpansionTeacherTarget],
    base_checkpoint_sha256: str,
    base_runtime_config_sha256: str,
    source_control_checkpoint_sha256: str,
    selection_sha256: str,
) -> dict[str, str]:
    root = _path_guard(destination)
    if root.exists():
        raise FileExistsError(f"V59 teacher cache already exists: {root}")
    ordered = sorted(targets, key=lambda target: target.key)
    if not ordered or len({target.key for target in ordered}) != len(ordered):
        raise ValueError("V59 teacher cache requires unique nonempty targets")
    if len({tuple(target.tokens.shape) for target in ordered}) != 1:
        raise ValueError("V59 teacher targets must share one tensor shape")
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()
    try:
        state = {
            f"prompt_{index:06d}": target.tokens.detach().cpu().float().contiguous()
            for index, target in enumerate(ordered)
        }
        weights = root / "teachers.safetensors"
        save_file(state, weights)
        metadata = {
            "schema_version": 1,
            "artifact": "v59_training_only_expansion_soft_prompt_cache",
            "weights_sha256": _sha256_file(weights),
            "base_checkpoint_sha256": _digest(
                base_checkpoint_sha256, "base checkpoint"
            ),
            "base_runtime_config_sha256": _digest(
                base_runtime_config_sha256, "runtime config"
            ),
            "source_control_checkpoint_sha256": _digest(
                source_control_checkpoint_sha256, "source controller"
            ),
            "selection_sha256": _digest(selection_sha256, "teacher selection"),
            "target_count": len(ordered),
            "environmental_text_inputs": [],
            "runtime_load_permitted": False,
            "records": [
                {
                    "tensor_key": f"prompt_{index:06d}",
                    "scene_id": target.scene_id,
                    "question_id": target.question_id,
                    "shape": list(target.tokens.shape),
                    "rms": float(
                        target.tokens.detach().float().square().mean().sqrt()
                    ),
                }
                for index, target in enumerate(ordered)
            ],
        }
        metadata_path = root / "metadata.json"
        with metadata_path.open("x", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        loaded, loaded_metadata = load_expansion_teachers(root)
        expected = {target.key: target.tokens.detach().cpu().float() for target in ordered}
        if set(loaded) != set(expected) or any(
            not torch.equal(loaded[key], expected[key]) for key in expected
        ):
            raise RuntimeError("V59 teacher cache failed exact reload")
        if loaded_metadata != metadata:
            raise RuntimeError("V59 teacher cache metadata changed on reload")
        return {
            "weights_sha256": metadata["weights_sha256"],
            "metadata_sha256": _sha256_file(metadata_path),
        }
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise


def load_expansion_teachers(
    source: str | Path,
) -> tuple[dict[tuple[str, str], torch.Tensor], dict[str, Any]]:
    root = _path_guard(source)
    expected_files = {"teachers.safetensors", "metadata.json"}
    if (
        not root.is_dir()
        or {item.name for item in root.iterdir()} != expected_files
        or any(item.is_symlink() for item in root.iterdir())
    ):
        raise ValueError("V59 teacher cache inventory changed")
    weights = root / "teachers.safetensors"
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "artifact",
        "weights_sha256",
        "base_checkpoint_sha256",
        "base_runtime_config_sha256",
        "source_control_checkpoint_sha256",
        "selection_sha256",
        "target_count",
        "environmental_text_inputs",
        "runtime_load_permitted",
        "records",
    }
    if not isinstance(metadata, Mapping) or set(metadata) != required:
        raise ValueError("V59 teacher cache metadata fields changed")
    if (
        metadata["schema_version"] != 1
        or metadata["artifact"] != "v59_training_only_expansion_soft_prompt_cache"
        or metadata["environmental_text_inputs"] != []
        or metadata["runtime_load_permitted"] is not False
        or metadata["weights_sha256"] != _sha256_file(weights)
    ):
        raise ValueError("V59 teacher cache contract mismatch")
    for field in (
        "weights_sha256",
        "base_checkpoint_sha256",
        "base_runtime_config_sha256",
        "source_control_checkpoint_sha256",
        "selection_sha256",
    ):
        _digest(metadata[field], field)
    records = metadata["records"]
    if not isinstance(records, list) or metadata["target_count"] != len(records):
        raise ValueError("V59 teacher cache record count changed")
    state = load_file(str(weights), device="cpu")
    if set(state) != {record.get("tensor_key") for record in records}:
        raise ValueError("V59 teacher cache tensor inventory changed")
    targets: dict[tuple[str, str], torch.Tensor] = {}
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "tensor_key",
            "scene_id",
            "question_id",
            "shape",
            "rms",
        }:
            raise ValueError("V59 teacher record fields changed")
        target = ExpansionTeacherTarget(
            str(record["scene_id"]),
            str(record["question_id"]),
            state[str(record["tensor_key"])],
        )
        if target.key in targets or list(target.tokens.shape) != record["shape"]:
            raise ValueError("V59 teacher record tensor mismatch")
        expected_rms = float(target.tokens.float().square().mean().sqrt())
        if abs(float(record["rms"]) - expected_rms) > 1e-7:
            raise ValueError("V59 teacher RMS metadata changed")
        targets[target.key] = target.tokens
    return targets, dict(metadata)


__all__ = [
    "ExpansionTeacherTarget",
    "load_expansion_teachers",
    "save_expansion_teachers",
]
