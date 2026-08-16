"""Sealed numeric artifact for one immutable V81 continuous scene memory.

Compilation and chat are deliberately separate processes.  The compiler may
load a learned controller and a numeric probe bank, but the chat runtime loads
only this two-file artifact.  Neither file contains questions, answers, object
names, labels, captions, or oracle relationships.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas import tensor_sha256

ARTIFACT: Final[str] = "v81_fixed_continuous_scene_memory_v1"
SCHEMA_VERSION: Final[int] = 81
MEMORY_FILENAME: Final[str] = "memory.safetensors"
METADATA_FILENAME: Final[str] = "runtime_metadata.json"
TENSOR_NAME: Final[str] = "fixed_scene_memory"
FIXED_MEMORY_TOKENS: Final[int] = 738
HIDDEN_SIZE: Final[int] = 1536
BASE_PREFIX_TOKENS: Final[int] = 258
ATLAS_MEMORY_TOKENS: Final[int] = 480
PROBE_COUNT: Final[int] = 96
VALUES_PER_PROBE: Final[int] = 4
BASE_ENVIRONMENT_LATENTS: Final[int] = 256
READER_LOGIT_SCALE: Final[float] = 160.0
READER_UNIFORM_FLOOR_MASS: Final[float] = 0.05

_SCENE_ID = re.compile(r"scene_[0-9]{6}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_PATH_COMPONENTS = frozenset(
    {"oracle", "qa", "rendered", "features", "training", "scorer"}
)
_SAFE_TENSOR_METADATA = {
    "artifact": ARTIFACT,
    "schema_version": str(SCHEMA_VERSION),
    "tensor_name": TENSOR_NAME,
    "environmental_text_inputs": "false",
    "questions_or_answers_serialized": "false",
}
_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "status",
        "scene_id",
        "tensor_file_sha256",
        "tensor_sha256",
        "canonical_prefix_sha256",
        "dtype",
        "shape",
        "fixed_memory_tokens",
        "hidden_size",
        "atlas_memory_tokens",
        "probe_count",
        "values_per_probe",
        "base_environment_latents",
        "base_prefix_sha256",
        "base_prefix_tensor_sha256",
        "source_base_checkpoint_sha256",
        "runtime_config_sha256",
        "source_control_checkpoint_sha256",
        "source_probe_tensor_sha256",
        "source_compiler_architecture",
        "reader_architecture",
        "reader_logit_scale",
        "reader_uniform_floor_mass",
        "compiled_before_user_question",
        "question_inputs_used_for_compilation",
        "question_dependent_scene_processing",
        "question_dependent_retrieval",
        "semantic_or_spatial_top_k_selection",
        "environmental_text_inputs",
        "questions_or_answers_serialized",
        "oracle_loaded",
    }
)


@dataclass(frozen=True)
class LoadedV81SceneMemory:
    root: Path
    memory: torch.Tensor
    metadata: dict[str, Any]


def _sha256_file(path: Path, record_file: Any | None = None) -> str:
    if record_file is not None:
        record_file(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"V81 scene-memory path contains a symlink: {current}")


def _guard_root(path: str | Path, *, must_exist: bool) -> Path:
    source = Path(os.path.abspath(Path(path).expanduser()))
    _reject_symlink_components(source)
    components = tuple(component.casefold() for component in source.parts)
    forbidden = _FORBIDDEN_PATH_COMPONENTS.intersection(components)
    hidden_oracle = tuple(
        component for component in components if component.startswith(".oracle-unavailable-")
    )
    if forbidden or hidden_oracle:
        raise ValueError(
            "V81 scene memory must be separate from forbidden data: "
            f"{sorted((*forbidden, *hidden_oracle))}"
        )
    if must_exist and (source.is_symlink() or not source.is_dir()):
        raise FileNotFoundError(f"V81 scene memory is unavailable: {source}")
    return source


def _strict_json(path: Path, record_file: Any | None = None) -> dict[str, Any]:
    if record_file is not None:
        record_file(path)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"V81 metadata has a duplicate field: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("V81 scene-memory metadata is invalid JSON") from error
    if not isinstance(value, dict) or set(value) != _METADATA_FIELDS:
        raise ValueError("V81 scene-memory metadata fields changed")
    return value


def _validate_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"V81 {field} is not a lowercase SHA-256 digest")
    return value


def _base_prefix(memory: torch.Tensor) -> torch.Tensor:
    atlas_end = 1 + ATLAS_MEMORY_TOKENS
    return torch.cat((memory[:, :1], memory[:, atlas_end:-1], memory[:, -1:]), dim=1)


def validate_v81_scene_memory_tensor(memory: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(memory, torch.Tensor)
        or tuple(memory.shape) != (1, FIXED_MEMORY_TOKENS, HIDDEN_SIZE)
        or not memory.is_floating_point()
        or not bool(torch.isfinite(memory).all())
    ):
        raise ValueError(
            "V81 fixed scene memory must be finite floating point with shape "
            f"[1,{FIXED_MEMORY_TOKENS},{HIDDEN_SIZE}]"
        )
    base = _base_prefix(memory)
    if tuple(base.shape) != (1, BASE_PREFIX_TOKENS, HIDDEN_SIZE):
        raise RuntimeError("V81 fixed scene memory has an invalid structured layout")
    return memory.detach().contiguous()


def build_v81_scene_memory_metadata(
    memory: torch.Tensor,
    *,
    scene_id: str,
    tensor_file_sha256: str,
    source_base_checkpoint_sha256: str,
    runtime_config_sha256: str,
    source_control_checkpoint_sha256: str,
    source_probe_tensor_sha256: str,
) -> dict[str, Any]:
    """Return strict text-free runtime metadata for a compiled numeric memory."""

    memory = validate_v81_scene_memory_tensor(memory)
    if _SCENE_ID.fullmatch(scene_id) is None:
        raise ValueError("V81 scene ID must be opaque scene_NNNNNN")
    digests = {
        "tensor_file_sha256": tensor_file_sha256,
        "source_base_checkpoint_sha256": source_base_checkpoint_sha256,
        "runtime_config_sha256": runtime_config_sha256,
        "source_control_checkpoint_sha256": source_control_checkpoint_sha256,
        "source_probe_tensor_sha256": source_probe_tensor_sha256,
    }
    for field, value in digests.items():
        _validate_sha256(value, field)
    base = _base_prefix(memory)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "status": "local_research_primary_candidate",
        "scene_id": scene_id,
        **digests,
        "tensor_sha256": tensor_sha256(memory),
        "canonical_prefix_sha256": prefix_sha256(memory),
        "dtype": str(memory.dtype),
        "shape": list(memory.shape),
        "fixed_memory_tokens": FIXED_MEMORY_TOKENS,
        "hidden_size": HIDDEN_SIZE,
        "atlas_memory_tokens": ATLAS_MEMORY_TOKENS,
        "probe_count": PROBE_COUNT,
        "values_per_probe": VALUES_PER_PROBE,
        "base_environment_latents": BASE_ENVIRONMENT_LATENTS,
        "base_prefix_sha256": prefix_sha256(base),
        "base_prefix_tensor_sha256": tensor_sha256(base),
        "source_compiler_architecture": "fixed_scene_key_value_atlas_v75_v2",
        "reader_architecture": "normalized_query_probe_cosine_dense_read_v81",
        "reader_logit_scale": READER_LOGIT_SCALE,
        "reader_uniform_floor_mass": READER_UNIFORM_FLOOR_MASS,
        "compiled_before_user_question": True,
        "question_inputs_used_for_compilation": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "semantic_or_spatial_top_k_selection": False,
        "environmental_text_inputs": [],
        "questions_or_answers_serialized": False,
        "oracle_loaded": False,
    }
    if set(result) != _METADATA_FIELDS:
        raise RuntimeError("V81 scene-memory metadata implementation drifted")
    return result


def save_v81_scene_memory(
    destination: str | Path,
    memory: torch.Tensor,
    *,
    scene_id: str,
    source_base_checkpoint_sha256: str,
    runtime_config_sha256: str,
    source_control_checkpoint_sha256: str,
    source_probe_tensor_sha256: str,
) -> dict[str, Any]:
    """Atomically create a two-file V81 memory; existing output is never replaced."""

    output = _guard_root(destination, must_exist=False)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    memory_cpu = validate_v81_scene_memory_tensor(memory).cpu()
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as raw:
        temporary = Path(raw)
        tensor_path = temporary / MEMORY_FILENAME
        save_file(
            {TENSOR_NAME: memory_cpu},
            str(tensor_path),
            metadata=_SAFE_TENSOR_METADATA,
        )
        file_sha256 = _sha256_file(tensor_path)
        metadata = build_v81_scene_memory_metadata(
            memory_cpu,
            scene_id=scene_id,
            tensor_file_sha256=file_sha256,
            source_base_checkpoint_sha256=source_base_checkpoint_sha256,
            runtime_config_sha256=runtime_config_sha256,
            source_control_checkpoint_sha256=source_control_checkpoint_sha256,
            source_probe_tensor_sha256=source_probe_tensor_sha256,
        )
        (temporary / METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    return metadata


def load_v81_scene_memory(
    root: str | Path,
    *,
    expected_scene_id: str,
    expected_base_checkpoint_sha256: str,
    expected_runtime_config_sha256: str,
    expected_model_device: torch.device | str,
    record_file: Any | None = None,
) -> LoadedV81SceneMemory:
    """Authenticate and load the only environmental tensor used by V81 chat."""

    source = _guard_root(root, must_exist=True)
    if {item.name for item in source.iterdir()} != {
        MEMORY_FILENAME,
        METADATA_FILENAME,
    }:
        raise ValueError("V81 scene-memory inventory must contain exactly two files")
    tensor_path = source / MEMORY_FILENAME
    metadata_path = source / METADATA_FILENAME
    if any(path.is_symlink() or not path.is_file() for path in (tensor_path, metadata_path)):
        raise ValueError("V81 scene-memory entries must be regular, non-symlink files")
    metadata = _strict_json(metadata_path, record_file)
    expected_exact = {
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "status": "local_research_primary_candidate",
        "scene_id": expected_scene_id,
        "shape": [1, FIXED_MEMORY_TOKENS, HIDDEN_SIZE],
        "fixed_memory_tokens": FIXED_MEMORY_TOKENS,
        "hidden_size": HIDDEN_SIZE,
        "atlas_memory_tokens": ATLAS_MEMORY_TOKENS,
        "probe_count": PROBE_COUNT,
        "values_per_probe": VALUES_PER_PROBE,
        "base_environment_latents": BASE_ENVIRONMENT_LATENTS,
        "source_base_checkpoint_sha256": expected_base_checkpoint_sha256,
        "runtime_config_sha256": expected_runtime_config_sha256,
        "source_compiler_architecture": "fixed_scene_key_value_atlas_v75_v2",
        "reader_architecture": "normalized_query_probe_cosine_dense_read_v81",
        "reader_logit_scale": READER_LOGIT_SCALE,
        "reader_uniform_floor_mass": READER_UNIFORM_FLOOR_MASS,
        "compiled_before_user_question": True,
        "question_inputs_used_for_compilation": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "semantic_or_spatial_top_k_selection": False,
        "environmental_text_inputs": [],
        "questions_or_answers_serialized": False,
        "oracle_loaded": False,
    }
    if any(metadata.get(field) != value for field, value in expected_exact.items()):
        raise ValueError("V81 scene-memory runtime contract changed")
    for field in (
        "tensor_file_sha256",
        "tensor_sha256",
        "canonical_prefix_sha256",
        "base_prefix_sha256",
        "base_prefix_tensor_sha256",
        "source_base_checkpoint_sha256",
        "runtime_config_sha256",
        "source_control_checkpoint_sha256",
        "source_probe_tensor_sha256",
    ):
        _validate_sha256(metadata.get(field), field)
    if not isinstance(metadata.get("dtype"), str):
        raise TypeError("V81 scene-memory dtype metadata is invalid")
    if _sha256_file(tensor_path, record_file) != metadata["tensor_file_sha256"]:
        raise ValueError("V81 scene-memory file digest changed")
    if record_file is not None:
        record_file(tensor_path)
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {TENSOR_NAME} or handle.metadata() != _SAFE_TENSOR_METADATA:
            raise ValueError("V81 scene-memory safetensors contract changed")
    memory = validate_v81_scene_memory_tensor(
        load_file(str(tensor_path), device="cpu")[TENSOR_NAME]
    )
    base = _base_prefix(memory)
    checks = {
        "tensor_sha256": tensor_sha256(memory),
        "canonical_prefix_sha256": prefix_sha256(memory),
        "base_prefix_sha256": prefix_sha256(base),
        "base_prefix_tensor_sha256": tensor_sha256(base),
    }
    if any(metadata[field] != value for field, value in checks.items()):
        raise ValueError("V81 scene-memory tensor identity changed")
    if metadata["dtype"] != str(memory.dtype):
        raise ValueError("V81 scene-memory dtype changed")
    if not math.isclose(
        float(metadata["reader_uniform_floor_mass"]),
        READER_UNIFORM_FLOOR_MASS,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("V81 reader floor changed")
    return LoadedV81SceneMemory(
        root=source,
        memory=memory.to(device=expected_model_device).detach(),
        metadata=metadata,
    )


__all__ = [
    "ARTIFACT",
    "ATLAS_MEMORY_TOKENS",
    "BASE_PREFIX_TOKENS",
    "FIXED_MEMORY_TOKENS",
    "HIDDEN_SIZE",
    "MEMORY_FILENAME",
    "METADATA_FILENAME",
    "READER_LOGIT_SCALE",
    "READER_UNIFORM_FLOOR_MASS",
    "LoadedV81SceneMemory",
    "build_v81_scene_memory_metadata",
    "load_v81_scene_memory",
    "save_v81_scene_memory",
    "validate_v81_scene_memory_tensor",
]
