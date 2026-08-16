"""Create-once numeric cache and candidate contracts for the V82 reader."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from semantic_3d_chat.language.v82_dense_learned_reader import (
    ARCHITECTURE,
    ARTIFACT,
    CANDIDATE_TENSOR_NAMES,
    TRAINABLE_PARAMETER_COUNT,
    DenseLearnedSceneReaderV82,
)

CACHE_ARTIFACT: Final[str] = "v82_numeric_reader_cache_v1"
CACHE_TENSOR_FILENAME: Final[str] = "training_tensors.safetensors"
CACHE_METADATA_FILENAME: Final[str] = "metadata.json"
CANDIDATE_WEIGHTS_FILENAME: Final[str] = "reader.safetensors"
CANDIDATE_METADATA_FILENAME: Final[str] = "runtime_metadata.json"
SCHEMA_VERSION: Final[int] = 82

CACHE_TENSOR_NAMES: Final[frozenset[str]] = frozenset(
    {
        "scene_memories",
        "question_queries",
        "row_scene_indices",
        "row_paired_scene_indices",
        "row_query_indices",
        "row_expected_change",
        "target_controls",
        "paired_target_controls",
    }
)
_CACHE_SAFE_METADATA = {
    "artifact": CACHE_ARTIFACT,
    "schema_version": str(SCHEMA_VERSION),
    "questions_or_answers_serialized": "false",
    "environmental_text_serialized": "false",
    "oracle_serialized": "false",
}
_CANDIDATE_SAFE_METADATA = {
    "artifact": ARTIFACT,
    "architecture": ARCHITECTURE,
    "schema_version": str(SCHEMA_VERSION),
    "environmental_memory_serialized": "false",
    "questions_or_answers_serialized": "false",
    "runtime_promotion_authorized": "false",
}
_SCENE_ID = re.compile(r"scene_[0-9]{6}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class LoadedV82Cache:
    root: Path
    tensors: dict[str, torch.Tensor]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LoadedV82Candidate:
    root: Path
    model: DenseLearnedSceneReaderV82
    metadata: dict[str, Any]


def sha256_file_v82(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256_v82(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"V82 JSON contains duplicate field: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )
    if not isinstance(value, dict):
        raise TypeError("V82 metadata must be a JSON object")
    return value


def _validate_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"V82 {label} is not a SHA-256 digest")
    return value


def _guard_root(path: str | Path, *, must_exist: bool) -> Path:
    source = Path(path).expanduser()
    source = source if source.is_absolute() else Path.cwd() / source
    source = Path(os.path.abspath(source))
    current = Path(source.anchor)
    for component in source.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"V82 artifact path contains a symlink: {current}")
    if must_exist:
        if not source.is_dir():
            raise FileNotFoundError(source)
    elif source.exists():
        raise FileExistsError(source)
    return source


def validate_cache_tensors_v82(
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if set(tensors) != CACHE_TENSOR_NAMES:
        raise ValueError("V82 numeric cache tensor inventory changed")
    result = {name: value.detach().cpu().contiguous() for name, value in tensors.items()}
    memories = result["scene_memories"]
    queries = result["question_queries"]
    rows = result["row_scene_indices"]
    if (
        memories.ndim != 3
        or memories.shape[1:] != (738, 1536)
        or memories.shape[0] < 2
        or memories.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or not bool(torch.isfinite(memories).all())
    ):
        raise ValueError("V82 cache scene memories must be finite [S,738,1536]")
    if (
        queries.ndim != 2
        or queries.shape[1] != 1536
        or queries.shape[0] < 1
        or queries.dtype != torch.float32
        or not bool(torch.isfinite(queries).all())
        or bool(torch.any(queries.norm(dim=-1) <= 1e-8))
    ):
        raise ValueError("V82 cache question queries must be nonzero FP32 [Q,1536]")
    row_count = int(rows.numel())
    if rows.ndim != 1 or row_count < 1:
        raise ValueError("V82 cache must contain at least one row")
    for name in (
        "row_scene_indices",
        "row_paired_scene_indices",
        "row_query_indices",
    ):
        value = result[name]
        if value.shape != rows.shape or value.dtype != torch.int64:
            raise ValueError(f"V82 cache {name} must be int64 [R]")
    changed = result["row_expected_change"]
    if changed.shape != rows.shape or changed.dtype != torch.bool:
        raise ValueError("V82 cache row_expected_change must be bool [R]")
    for name in ("target_controls", "paired_target_controls"):
        value = result[name]
        if (
            value.shape != (row_count, 4, 1536)
            or value.dtype not in (torch.bfloat16, torch.float16, torch.float32)
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"V82 cache {name} must be finite [R,4,1536]")
    if (
        int(rows.min()) < 0
        or int(rows.max()) >= memories.shape[0]
        or int(result["row_paired_scene_indices"].min()) < 0
        or int(result["row_paired_scene_indices"].max()) >= memories.shape[0]
        or int(result["row_query_indices"].min()) < 0
        or int(result["row_query_indices"].max()) >= queries.shape[0]
    ):
        raise ValueError("V82 cache index tensor is out of bounds")
    if bool(torch.any(rows == result["row_paired_scene_indices"])):
        raise ValueError("V82 cache paired scene must differ from source scene")
    return result


def _cache_metadata(
    tensors: Mapping[str, torch.Tensor],
    *,
    split_role: str,
    scene_ids: list[str],
    tensor_file_sha256: str,
    source_qa_sha256: str,
    source_v73_config_sha256: str,
    source_prefix_manifest_sha256: str,
    source_controller_sha256: str,
    source_probe_tensor_sha256: str,
) -> dict[str, Any]:
    checked = validate_cache_tensors_v82(tensors)
    if split_role not in {
        "historical_optimization_fold",
        "historical_pair_scene_disjoint_development_fold",
    }:
        raise ValueError("V82 cache split role is invalid")
    if (
        len(scene_ids) != checked["scene_memories"].shape[0]
        or len(set(scene_ids)) != len(scene_ids)
        or any(_SCENE_ID.fullmatch(value) is None for value in scene_ids)
    ):
        raise ValueError("V82 cache scene inventory must be unique opaque IDs")
    for label, digest in {
        "tensor_file_sha256": tensor_file_sha256,
        "source_qa_sha256": source_qa_sha256,
        "source_v73_config_sha256": source_v73_config_sha256,
        "source_prefix_manifest_sha256": source_prefix_manifest_sha256,
        "source_controller_sha256": source_controller_sha256,
        "source_probe_tensor_sha256": source_probe_tensor_sha256,
    }.items():
        _validate_sha256(digest, label)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": CACHE_ARTIFACT,
        "status": "historical_numeric_diagnostic_not_runtime",
        "split_role": split_role,
        "tensor_file_sha256": tensor_file_sha256,
        "scene_ids": scene_ids,
        "scene_count": len(scene_ids),
        "question_query_count": int(checked["question_queries"].shape[0]),
        "row_count": int(checked["row_scene_indices"].numel()),
        "changed_row_count": int(checked["row_expected_change"].sum()),
        "fixed_memory_shape_per_scene": [1, 738, 1536],
        "fixed_memory_compiled_before_row_query_binding": True,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "semantic_or_spatial_top_k_selection": False,
        "source_qa_sha256": source_qa_sha256,
        "source_v73_config_sha256": source_v73_config_sha256,
        "source_prefix_manifest_sha256": source_prefix_manifest_sha256,
        "source_controller_sha256": source_controller_sha256,
        "source_probe_tensor_sha256": source_probe_tensor_sha256,
        "questions_or_answers_serialized": False,
        "environmental_text_serialized": False,
        "oracle_serialized": False,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "runtime_promotion_authorized": False,
    }


def save_v82_cache(
    destination: str | Path,
    tensors: Mapping[str, torch.Tensor],
    *,
    split_role: str,
    scene_ids: list[str],
    source_qa_sha256: str,
    source_v73_config_sha256: str,
    source_prefix_manifest_sha256: str,
    source_controller_sha256: str,
    source_probe_tensor_sha256: str,
) -> dict[str, Any]:
    output = _guard_root(destination, must_exist=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    checked = validate_cache_tensors_v82(tensors)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as raw:
        temporary = Path(raw)
        tensor_path = temporary / CACHE_TENSOR_FILENAME
        save_file(checked, str(tensor_path), metadata=_CACHE_SAFE_METADATA)
        metadata = _cache_metadata(
            checked,
            split_role=split_role,
            scene_ids=scene_ids,
            tensor_file_sha256=sha256_file_v82(tensor_path),
            source_qa_sha256=source_qa_sha256,
            source_v73_config_sha256=source_v73_config_sha256,
            source_prefix_manifest_sha256=source_prefix_manifest_sha256,
            source_controller_sha256=source_controller_sha256,
            source_probe_tensor_sha256=source_probe_tensor_sha256,
        )
        (temporary / CACHE_METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    return metadata


def load_v82_cache(root: str | Path) -> LoadedV82Cache:
    source = _guard_root(root, must_exist=True)
    if {item.name for item in source.iterdir()} != {
        CACHE_TENSOR_FILENAME,
        CACHE_METADATA_FILENAME,
    }:
        raise ValueError("V82 cache must contain exactly two files")
    tensor_path = source / CACHE_TENSOR_FILENAME
    metadata_path = source / CACHE_METADATA_FILENAME
    if any(path.is_symlink() or not path.is_file() for path in (tensor_path, metadata_path)):
        raise ValueError("V82 cache entries must be regular files")
    metadata = _strict_json(metadata_path)
    if (
        metadata.get("schema_version") != SCHEMA_VERSION
        or metadata.get("artifact") != CACHE_ARTIFACT
        or metadata.get("questions_or_answers_serialized") is not False
        or metadata.get("environmental_text_serialized") is not False
        or metadata.get("oracle_serialized") is not False
        or metadata.get("runtime_promotion_authorized") is not False
        or metadata.get("official_validation_loaded") is not False
        or metadata.get("official_test_loaded") is not False
        or metadata.get("deferred_final_loaded") is not False
        or metadata.get("question_dependent_retrieval") is not False
        or metadata.get("semantic_or_spatial_top_k_selection") is not False
    ):
        raise ValueError("V82 cache metadata contract changed")
    if sha256_file_v82(tensor_path) != metadata.get("tensor_file_sha256"):
        raise ValueError("V82 cache tensor file digest changed")
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != CACHE_TENSOR_NAMES or handle.metadata() != _CACHE_SAFE_METADATA:
            raise ValueError("V82 cache safetensors contract changed")
    tensors = validate_cache_tensors_v82(load_file(str(tensor_path), device="cpu"))
    if (
        metadata.get("scene_count") != tensors["scene_memories"].shape[0]
        or metadata.get("question_query_count") != tensors["question_queries"].shape[0]
        or metadata.get("row_count") != tensors["row_scene_indices"].numel()
        or metadata.get("changed_row_count")
        != int(tensors["row_expected_change"].sum())
    ):
        raise ValueError("V82 cache metadata counts changed")
    scene_ids = metadata.get("scene_ids")
    if (
        not isinstance(scene_ids, list)
        or len(scene_ids) != metadata["scene_count"]
        or any(not isinstance(value, str) or _SCENE_ID.fullmatch(value) is None for value in scene_ids)
    ):
        raise ValueError("V82 cache scene IDs changed")
    return LoadedV82Cache(source, tensors, metadata)


def _candidate_metadata(
    *,
    weights_sha256: str,
    training_cache_sha256: str,
    training_cache_metadata_sha256: str,
    fit_summary: Mapping[str, int | float | bool],
) -> dict[str, Any]:
    for label, digest in {
        "weights_sha256": weights_sha256,
        "training_cache_sha256": training_cache_sha256,
        "training_cache_metadata_sha256": training_cache_metadata_sha256,
    }.items():
        _validate_sha256(digest, label)
    if not fit_summary or any(
        isinstance(value, float) and not math.isfinite(value)
        for value in fit_summary.values()
    ):
        raise ValueError("V82 fit summary is empty or nonfinite")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "architecture": ARCHITECTURE,
        "status": "historical_training_diagnostic_not_promoted",
        "weights_sha256": weights_sha256,
        "training_cache_sha256": training_cache_sha256,
        "training_cache_metadata_sha256": training_cache_metadata_sha256,
        "trainable_parameter_count": TRAINABLE_PARAMETER_COUNT,
        "fixed_memory_tokens": 738,
        "hidden_size": 1536,
        "atlas_group_count": 96,
        "atlas_value_count": 384,
        "base_environment_latents": 256,
        "all_384_atlas_values_and_256_base_latents_positive_floor": True,
        "boi_eoi_and_96_probe_keys_are_not_payload": True,
        "question_dependent_retrieval": False,
        "semantic_or_spatial_top_k_selection": False,
        "environmental_memory_serialized": False,
        "questions_or_answers_serialized": False,
        "oracle_serialized": False,
        "training_split_role": "historical_optimization_fold",
        "fit_summary": dict(fit_summary),
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "runtime_promotion_authorized": False,
    }


def save_v82_candidate(
    destination: str | Path,
    model: DenseLearnedSceneReaderV82,
    *,
    training_cache_sha256: str,
    training_cache_metadata_sha256: str,
    fit_summary: Mapping[str, int | float | bool],
) -> dict[str, Any]:
    output = _guard_root(destination, must_exist=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    state = model.candidate_state_dict()
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as raw:
        temporary = Path(raw)
        weights_path = temporary / CANDIDATE_WEIGHTS_FILENAME
        save_file(state, str(weights_path), metadata=_CANDIDATE_SAFE_METADATA)
        metadata = _candidate_metadata(
            weights_sha256=sha256_file_v82(weights_path),
            training_cache_sha256=training_cache_sha256,
            training_cache_metadata_sha256=training_cache_metadata_sha256,
            fit_summary=fit_summary,
        )
        (temporary / CANDIDATE_METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    return metadata


def load_v82_candidate(
    root: str | Path,
    *,
    device: torch.device | str = "cpu",
    record_file: Any | None = None,
) -> LoadedV82Candidate:
    source = _guard_root(root, must_exist=True)
    if {item.name for item in source.iterdir()} != {
        CANDIDATE_WEIGHTS_FILENAME,
        CANDIDATE_METADATA_FILENAME,
    }:
        raise ValueError("V82 candidate must contain exactly two files")
    weights_path = source / CANDIDATE_WEIGHTS_FILENAME
    metadata_path = source / CANDIDATE_METADATA_FILENAME
    if any(path.is_symlink() or not path.is_file() for path in (weights_path, metadata_path)):
        raise ValueError("V82 candidate entries must be regular files")
    if record_file is not None:
        record_file(metadata_path)
    metadata = _strict_json(metadata_path)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "architecture": ARCHITECTURE,
        "status": "historical_training_diagnostic_not_promoted",
        "trainable_parameter_count": TRAINABLE_PARAMETER_COUNT,
        "fixed_memory_tokens": 738,
        "hidden_size": 1536,
        "atlas_group_count": 96,
        "atlas_value_count": 384,
        "base_environment_latents": 256,
        "all_384_atlas_values_and_256_base_latents_positive_floor": True,
        "boi_eoi_and_96_probe_keys_are_not_payload": True,
        "question_dependent_retrieval": False,
        "semantic_or_spatial_top_k_selection": False,
        "environmental_memory_serialized": False,
        "questions_or_answers_serialized": False,
        "oracle_serialized": False,
        "training_split_role": "historical_optimization_fold",
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "runtime_promotion_authorized": False,
    }
    if any(metadata.get(field) != value for field, value in expected.items()):
        raise ValueError("V82 candidate runtime metadata contract changed")
    _validate_sha256(metadata.get("weights_sha256"), "weights_sha256")
    _validate_sha256(metadata.get("training_cache_sha256"), "training_cache_sha256")
    _validate_sha256(
        metadata.get("training_cache_metadata_sha256"),
        "training_cache_metadata_sha256",
    )
    if record_file is not None:
        record_file(weights_path)
    if sha256_file_v82(weights_path) != metadata["weights_sha256"]:
        raise ValueError("V82 candidate weights digest changed")
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != CANDIDATE_TENSOR_NAMES or handle.metadata() != _CANDIDATE_SAFE_METADATA:
            raise ValueError("V82 candidate safetensors contract changed")
    model = DenseLearnedSceneReaderV82()
    state = load_file(str(weights_path), device="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=torch.float32).eval()
    model.requires_grad_(False)
    return LoadedV82Candidate(source, model, metadata)


__all__ = [
    "CACHE_ARTIFACT",
    "CACHE_METADATA_FILENAME",
    "CACHE_TENSOR_FILENAME",
    "CACHE_TENSOR_NAMES",
    "CANDIDATE_METADATA_FILENAME",
    "CANDIDATE_WEIGHTS_FILENAME",
    "LoadedV82Cache",
    "LoadedV82Candidate",
    "canonical_sha256_v82",
    "load_v82_cache",
    "load_v82_candidate",
    "save_v82_cache",
    "save_v82_candidate",
    "sha256_file_v82",
    "validate_cache_tensors_v82",
]
