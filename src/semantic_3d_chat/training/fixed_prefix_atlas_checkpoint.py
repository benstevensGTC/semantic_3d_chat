"""Offline construction and strict loading of a fixed-prefix atlas checkpoint.

The checkpoint contains only learned numeric parameters: a sealed V7 value
function and a deterministic bank of continuous probe centroids.  It contains
no source questions, answers, labels, filenames, scene metadata, or object
vocabulary.  Runtime uses these tensors before the first user question and may
discard them immediately after compiling the immutable scene prefix.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from semantic_3d_chat.scene_encoder.fixed_prefix_atlas import (
    tensor_sha256,
    validate_probe_bank,
)
from semantic_3d_chat.scene_encoder.question_control_v7 import (
    AlwaysOnTeacherBasisFullSceneQuestionControlV7,
)
from semantic_3d_chat.training.question_control_v7_checkpoint import (
    v7_value_state_sha256,
)

ATLAS_WEIGHTS_FILENAME: Final[str] = "atlas.safetensors"
ATLAS_METADATA_FILENAME: Final[str] = "runtime_metadata.json"
ATLAS_ARCHITECTURE: Final[str] = "fixed_scene_key_value_atlas_v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_PATH_COMPONENTS = frozenset({"oracle", "qa", "scorer", "training"})
_FORBIDDEN_TEXT_FRAGMENTS = frozenset(
    {
        "caption",
        "category",
        "object_name",
        "instance_id",
        "scene_graph",
        "chair",
        "bowl",
        "book",
        "picture",
        "frame",
        "cube",
        "table",
        "lamp",
        "door",
        "plant",
        "cabinet",
    }
)
_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "architecture",
        "hidden_size",
        "environment_latents",
        "probe_count",
        "values_per_probe",
        "scene_moment_count",
        "interaction_dim",
        "trunk_dim",
        "output_basis_rank",
        "maximum_value_rms",
        "initial_value_rms",
        "atlas_memory_tokens",
        "fixed_prefix_tokens",
        "weights_sha256",
        "controller_state_sha256",
        "probe_bank_sha256",
        "source_controller_checkpoint_sha256",
        "source_controller_weights_sha256",
        "source_controller_metadata_sha256",
        "base_checkpoint_sha256",
        "base_runtime_config_sha256",
        "probe_construction",
        "source_vector_count",
        "source_vector_set_sha256",
        "complete_base_scene_prefix_required",
        "complete_base_scene_prefix_preserved",
        "all_probes_processed",
        "all_atlas_tokens_appended",
        "compiled_before_user_question",
        "user_question_inputs_used_for_compilation",
        "question_dependent_scene_processing",
        "question_dependent_retrieval",
        "semantic_or_spatial_top_k_selection",
        "runtime_source_records_loaded",
        "runtime_answer_metadata_loaded",
        "environmental_text_inputs",
        "local_inference_only",
        "structural_contract_verified",
        "behavioral_evaluation_status",
    }
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )


def _absolute_without_resolving(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return Path(os.path.abspath(candidate))


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(
                "Fixed-prefix atlas paths must not contain symlinks: "
                f"{current}"
            )


def two_file_checkpoint_fingerprint(path: str | Path) -> tuple[str, dict[str, str]]:
    root = _absolute_without_resolving(path)
    _reject_symlink_components(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Two-file checkpoint is unavailable: {root}")
    inventory = {item.name for item in root.iterdir()}
    if inventory == {"control.safetensors", ATLAS_METADATA_FILENAME}:
        names = ("control.safetensors", ATLAS_METADATA_FILENAME)
    elif inventory == {ATLAS_WEIGHTS_FILENAME, ATLAS_METADATA_FILENAME}:
        names = (ATLAS_WEIGHTS_FILENAME, ATLAS_METADATA_FILENAME)
    else:
        raise ValueError("Checkpoint must contain exactly one weights file and runtime metadata")
    entries: dict[str, str] = {}
    for name in names:
        source = root / name
        if source.is_symlink() or not source.is_file():
            raise ValueError("Checkpoint entries must be regular non-symlink files")
        entries[name] = sha256_file(source)
    return _canonical_sha256(entries), entries


def _tensor_set_sha256(values: torch.Tensor) -> str:
    return tensor_sha256(values.detach().cpu().float().contiguous())


def deterministic_spherical_probe_bank(
    source_vectors: torch.Tensor,
    *,
    probe_count: int,
    iterations: int = 12,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Cluster continuous source vectors without retaining their source records.

    Initialization is deterministic farthest-first.  Assignment uses cosine
    similarity and the returned probe for each cluster is the raw-vector mean,
    retaining the language model's native embedding scale.
    """

    if not isinstance(source_vectors, torch.Tensor) or source_vectors.ndim != 2:
        raise ValueError("Source vectors must have shape [N,H]")
    if not source_vectors.is_floating_point() or not bool(
        torch.isfinite(source_vectors).all().item()
    ):
        raise ValueError("Source vectors must be finite floating-point values")
    if isinstance(probe_count, bool) or not isinstance(probe_count, int):
        raise TypeError("probe_count must be an integer")
    if not 1 <= probe_count <= source_vectors.shape[0]:
        raise ValueError("probe_count must be between one and the source-vector count")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    raw = source_vectors.detach().cpu().float().contiguous()
    norms = raw.norm(dim=-1)
    if bool(torch.any(norms <= 1e-8).item()):
        raise ValueError("Source vectors must all be nonzero")
    normalized = F.normalize(raw, dim=-1, eps=1e-8)

    # Select the most globally distinct point first, then repeatedly add the
    # point with the lowest similarity to its nearest selected center.
    similarities = normalized @ normalized.T
    first = int(torch.argmin(similarities.mean(dim=1)).item())
    selected = [first]
    best_similarity = similarities[:, first].clone()
    while len(selected) < probe_count:
        for index in selected:
            best_similarity[index] = math.inf
        next_index = int(torch.argmin(best_similarity).item())
        selected.append(next_index)
        best_similarity = torch.maximum(best_similarity, similarities[:, next_index])
    centers = normalized[selected].clone()

    assignments = torch.zeros(raw.shape[0], dtype=torch.long)
    completed_iterations = 0
    for completed_iterations in range(1, iterations + 1):
        assignments = torch.argmax(normalized @ centers.T, dim=1)
        counts = torch.bincount(assignments, minlength=probe_count)
        if bool(torch.any(counts == 0).item()):
            nearest = torch.max(normalized @ centers.T, dim=1).values
            candidates = torch.argsort(nearest, stable=True)
            cursor = 0
            for empty in torch.nonzero(counts == 0, as_tuple=False).flatten().tolist():
                while int(candidates[cursor]) in selected:
                    cursor += 1
                replacement = int(candidates[cursor])
                cursor += 1
                centers[empty] = normalized[replacement]
                selected.append(replacement)
            continue
        updated = torch.stack(
            [F.normalize(normalized[assignments == index].mean(dim=0), dim=0, eps=1e-8)
             for index in range(probe_count)]
        )
        if torch.equal(assignments, torch.argmax(normalized @ updated.T, dim=1)):
            centers = updated
            break
        centers = updated

    assignments = torch.argmax(normalized @ centers.T, dim=1)
    counts = torch.bincount(assignments, minlength=probe_count)
    if bool(torch.any(counts == 0).item()):
        raise RuntimeError("Deterministic probe clustering produced an empty cluster")
    probes = torch.stack(
        [raw[assignments == index].mean(dim=0) for index in range(probe_count)]
    ).contiguous()
    probes = validate_probe_bank(probes, hidden_size=raw.shape[1])
    assigned_cosine = torch.sum(
        normalized * F.normalize(probes, dim=-1)[assignments], dim=-1
    )
    audit = {
        "algorithm": "deterministic_farthest_first_spherical_kmeans_v1",
        "probe_count": probe_count,
        "source_vector_count": raw.shape[0],
        "hidden_size": raw.shape[1],
        "iterations": completed_iterations,
        "cluster_minimum_size": int(counts.min()),
        "cluster_maximum_size": int(counts.max()),
        "mean_assigned_cosine": float(assigned_cosine.mean()),
        "minimum_assigned_cosine": float(assigned_cosine.min()),
        "source_vector_set_sha256": _tensor_set_sha256(raw),
        "probe_bank_sha256": tensor_sha256(probes),
        "source_records_retained": False,
        "source_text_retained": False,
    }
    return probes, audit


@dataclass(frozen=True)
class LoadedFixedPrefixAtlas:
    controller: AlwaysOnTeacherBasisFullSceneQuestionControlV7
    probe_embeddings: torch.Tensor
    metadata: dict[str, Any]
    checkpoint_path: Path


def _runtime_metadata(
    *,
    controller: AlwaysOnTeacherBasisFullSceneQuestionControlV7,
    probe_embeddings: torch.Tensor,
    weights_sha256: str,
    source_controller_checkpoint_sha256: str,
    source_controller_files: Mapping[str, str],
    base_checkpoint_sha256: str,
    base_runtime_config_sha256: str,
    probe_audit: Mapping[str, Any],
) -> dict[str, Any]:
    probe_count = int(probe_embeddings.shape[0])
    memory_tokens = probe_count * (controller.control_token_count + 1)
    expected_source_files = {"control.safetensors", ATLAS_METADATA_FILENAME}
    if set(source_controller_files) != expected_source_files:
        raise ValueError("Source controller must be the sealed two-file V7 checkpoint")
    if probe_audit.get("algorithm") != (
        "deterministic_farthest_first_spherical_kmeans_v1"
    ):
        raise ValueError("Fixed-prefix probes require the deterministic atlas algorithm")
    if probe_audit.get("probe_bank_sha256") != tensor_sha256(probe_embeddings):
        raise ValueError("Probe construction audit does not match the saved probe bank")
    if probe_audit.get("probe_count", probe_count) != probe_count:
        raise ValueError("Probe construction audit count does not match the saved bank")
    payload = {
        "schema_version": 1,
        "architecture": ATLAS_ARCHITECTURE,
        "hidden_size": controller.hidden_size,
        "environment_latents": controller.expected_environment_latents,
        "probe_count": probe_count,
        "values_per_probe": controller.control_token_count,
        "scene_moment_count": controller.moment_count,
        "interaction_dim": controller.interaction_dim,
        "trunk_dim": controller.trunk_dim,
        "output_basis_rank": controller.output_basis_rank,
        "maximum_value_rms": controller.maximum_control_rms,
        "initial_value_rms": controller.initial_control_rms,
        "atlas_memory_tokens": memory_tokens,
        "fixed_prefix_tokens": controller.expected_environment_latents + 2 + memory_tokens,
        "weights_sha256": weights_sha256,
        "controller_state_sha256": v7_value_state_sha256(controller),
        "probe_bank_sha256": tensor_sha256(probe_embeddings),
        "source_controller_checkpoint_sha256": source_controller_checkpoint_sha256,
        "source_controller_weights_sha256": source_controller_files["control.safetensors"],
        "source_controller_metadata_sha256": source_controller_files[
            ATLAS_METADATA_FILENAME
        ],
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "base_runtime_config_sha256": base_runtime_config_sha256,
        "probe_construction": str(probe_audit.get("algorithm")),
        "source_vector_count": int(probe_audit.get("source_vector_count", 0)),
        "source_vector_set_sha256": str(probe_audit.get("source_vector_set_sha256")),
        "complete_base_scene_prefix_required": True,
        "complete_base_scene_prefix_preserved": True,
        "all_probes_processed": True,
        "all_atlas_tokens_appended": True,
        "compiled_before_user_question": True,
        "user_question_inputs_used_for_compilation": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "semantic_or_spatial_top_k_selection": False,
        "runtime_source_records_loaded": False,
        "runtime_answer_metadata_loaded": False,
        "environmental_text_inputs": [],
        "local_inference_only": True,
        "structural_contract_verified": True,
        "behavioral_evaluation_status": "pending_fixed_prefix_held_evaluation",
    }
    validate_fixed_prefix_atlas_metadata(payload)
    return payload


def validate_fixed_prefix_atlas_metadata(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _METADATA_FIELDS:
        raise ValueError("Fixed-prefix atlas runtime metadata fields changed")
    payload = dict(value)
    integers = {
        "hidden_size": payload.get("hidden_size"),
        "environment_latents": payload.get("environment_latents"),
        "probe_count": payload.get("probe_count"),
        "values_per_probe": payload.get("values_per_probe"),
        "scene_moment_count": payload.get("scene_moment_count"),
        "interaction_dim": payload.get("interaction_dim"),
        "trunk_dim": payload.get("trunk_dim"),
        "output_basis_rank": payload.get("output_basis_rank"),
        "atlas_memory_tokens": payload.get("atlas_memory_tokens"),
        "fixed_prefix_tokens": payload.get("fixed_prefix_tokens"),
        "source_vector_count": payload.get("source_vector_count"),
    }
    if any(type(item) is not int or item < 1 for item in integers.values()):
        raise ValueError("Fixed-prefix atlas dimensions must be positive integers")
    for field in ("maximum_value_rms", "initial_value_rms"):
        item = payload.get(field)
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(
            float(item)
        ):
            raise ValueError(f"Fixed-prefix atlas {field} must be finite")
    if not 0.0 < float(payload["initial_value_rms"]) < float(
        payload["maximum_value_rms"]
    ) <= 1.0:
        raise ValueError("Fixed-prefix atlas value RMS bounds are invalid")
    expected_memory = int(payload["probe_count"]) * (
        int(payload["values_per_probe"]) + 1
    )
    expected_prefix = int(payload["environment_latents"]) + 2 + expected_memory
    if (
        payload.get("schema_version") != 1
        or payload.get("architecture") != ATLAS_ARCHITECTURE
        or payload.get("atlas_memory_tokens") != expected_memory
        or payload.get("fixed_prefix_tokens") != expected_prefix
        or payload.get("complete_base_scene_prefix_required") is not True
        or payload.get("complete_base_scene_prefix_preserved") is not True
        or payload.get("all_probes_processed") is not True
        or payload.get("all_atlas_tokens_appended") is not True
        or payload.get("compiled_before_user_question") is not True
        or payload.get("user_question_inputs_used_for_compilation") is not False
        or payload.get("question_dependent_scene_processing") is not False
        or payload.get("question_dependent_retrieval") is not False
        or payload.get("semantic_or_spatial_top_k_selection") is not False
        or payload.get("runtime_source_records_loaded") is not False
        or payload.get("runtime_answer_metadata_loaded") is not False
        or payload.get("environmental_text_inputs") != []
        or payload.get("local_inference_only") is not True
        or payload.get("structural_contract_verified") is not True
        or payload.get("behavioral_evaluation_status")
        != "pending_fixed_prefix_held_evaluation"
    ):
        raise ValueError("Fixed-prefix atlas runtime contract mismatch")
    for field in (
        "weights_sha256",
        "controller_state_sha256",
        "probe_bank_sha256",
        "source_controller_checkpoint_sha256",
        "source_controller_weights_sha256",
        "source_controller_metadata_sha256",
        "base_checkpoint_sha256",
        "base_runtime_config_sha256",
        "source_vector_set_sha256",
    ):
        if not isinstance(payload.get(field), str) or _SHA256.fullmatch(payload[field]) is None:
            raise ValueError(f"Fixed-prefix atlas {field} is not a SHA-256 digest")
    if payload.get("probe_construction") != (
        "deterministic_farthest_first_spherical_kmeans_v1"
    ):
        raise ValueError("Fixed-prefix atlas probe construction changed")
    serialized = json.dumps(payload, sort_keys=True, allow_nan=False).casefold()
    if any(fragment in serialized for fragment in _FORBIDDEN_TEXT_FRAGMENTS):
        raise ValueError("Fixed-prefix atlas metadata contains forbidden semantic text")
    return payload


def save_fixed_prefix_atlas_checkpoint(
    checkpoint_path: str | Path,
    *,
    controller: AlwaysOnTeacherBasisFullSceneQuestionControlV7,
    probe_embeddings: torch.Tensor,
    source_controller_checkpoint_sha256: str,
    source_controller_files: Mapping[str, str],
    base_checkpoint_sha256: str,
    base_runtime_config_sha256: str,
    probe_audit: Mapping[str, Any],
) -> dict[str, Any]:
    if type(controller) is not AlwaysOnTeacherBasisFullSceneQuestionControlV7:
        raise TypeError("Fixed-prefix atlas requires the exact V7 controller")
    probes = validate_probe_bank(probe_embeddings, hidden_size=controller.hidden_size)
    destination = _absolute_without_resolving(checkpoint_path)
    _reject_symlink_components(destination)
    if destination.exists():
        raise FileExistsError(f"Fixed-prefix atlas refuses to overwrite: {destination}")
    if _FORBIDDEN_PATH_COMPONENTS.intersection(
        component.casefold() for component in destination.parts
    ):
        raise ValueError("Fixed-prefix atlas must be stored outside runtime-forbidden roots")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.partial-", dir=destination.parent)
    )
    try:
        state = {
            "probe_embeddings": probes,
            **{
                f"controller.{name}": tensor.detach().cpu().float().contiguous()
                for name, tensor in controller.state_dict().items()
            },
        }
        if any(
            not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all().item())
            for tensor in state.values()
        ):
            raise ValueError("Fixed-prefix atlas state must be finite floating point")
        weights = temporary / ATLAS_WEIGHTS_FILENAME
        save_file(state, weights)
        reloaded = load_file(str(weights), device="cpu")
        if set(reloaded) != set(state) or any(
            not torch.equal(reloaded[name], state[name]) for name in state
        ):
            raise RuntimeError("Fixed-prefix atlas failed exact safetensors reload")
        metadata = _runtime_metadata(
            controller=controller,
            probe_embeddings=probes,
            weights_sha256=sha256_file(weights),
            source_controller_checkpoint_sha256=source_controller_checkpoint_sha256,
            source_controller_files=source_controller_files,
            base_checkpoint_sha256=base_checkpoint_sha256,
            base_runtime_config_sha256=base_runtime_config_sha256,
            probe_audit=probe_audit,
        )
        metadata_path = temporary / ATLAS_METADATA_FILENAME
        with metadata_path.open("x", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    fingerprint, files = two_file_checkpoint_fingerprint(destination)
    return {
        "checkpoint_sha256": fingerprint,
        "files": files,
        "metadata": metadata,
        "probe_audit": dict(probe_audit),
    }


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, nested in pairs:
            if key in output:
                raise ValueError(f"Duplicate fixed-prefix atlas metadata field: {key}")
            output[key] = nested
        return output

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    return validate_fixed_prefix_atlas_metadata(value)


def load_fixed_prefix_atlas_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    expected_hidden_size: int | None = None,
    expected_base_checkpoint_sha256: str | None = None,
    expected_base_runtime_config_sha256: str | None = None,
    record_file: Callable[[str | Path], None] | None = None,
) -> LoadedFixedPrefixAtlas:
    root = _absolute_without_resolving(checkpoint_path)
    _reject_symlink_components(root)
    if _FORBIDDEN_PATH_COMPONENTS.intersection(
        component.casefold() for component in root.parts
    ):
        raise ValueError("Fixed-prefix atlas path reaches a runtime-forbidden root")
    fingerprint, _files = two_file_checkpoint_fingerprint(root)
    del fingerprint
    metadata_path = root / ATLAS_METADATA_FILENAME
    weights_path = root / ATLAS_WEIGHTS_FILENAME
    if record_file is not None:
        record_file(metadata_path)
    metadata = _strict_json(metadata_path)
    if sha256_file(weights_path) != metadata["weights_sha256"]:
        raise ValueError("Fixed-prefix atlas weights changed")
    if expected_hidden_size is not None and metadata["hidden_size"] != expected_hidden_size:
        raise ValueError("Fixed-prefix atlas hidden size differs from the language model")
    if (
        expected_base_checkpoint_sha256 is not None
        and metadata["base_checkpoint_sha256"] != expected_base_checkpoint_sha256
    ):
        raise ValueError("Fixed-prefix atlas is bound to a different base checkpoint")
    if (
        expected_base_runtime_config_sha256 is not None
        and metadata["base_runtime_config_sha256"]
        != expected_base_runtime_config_sha256
    ):
        raise ValueError("Fixed-prefix atlas is bound to a different runtime config")
    if record_file is not None:
        record_file(weights_path)
    state = load_file(str(weights_path), device="cpu")
    if "probe_embeddings" not in state or "controller.output_basis" not in state:
        raise ValueError("Fixed-prefix atlas is missing required numeric tensors")
    controller_state = {
        name.removeprefix("controller."): tensor
        for name, tensor in state.items()
        if name.startswith("controller.")
    }
    if len(controller_state) + 1 != len(state):
        raise ValueError("Fixed-prefix atlas contains unexpected tensors")
    probes = validate_probe_bank(
        state["probe_embeddings"], hidden_size=int(metadata["hidden_size"])
    )
    if probes.shape[0] != metadata["probe_count"] or tensor_sha256(probes) != metadata[
        "probe_bank_sha256"
    ]:
        raise ValueError("Fixed-prefix atlas probe bank changed")
    controller = AlwaysOnTeacherBasisFullSceneQuestionControlV7(
        int(metadata["hidden_size"]),
        controller_state["output_basis"],
        control_tokens=int(metadata["values_per_probe"]),
        expected_environment_latents=int(metadata["environment_latents"]),
        moment_count=int(metadata["scene_moment_count"]),
        interaction_dim=int(metadata["interaction_dim"]),
        trunk_dim=int(metadata["trunk_dim"]),
        maximum_control_rms=float(metadata["maximum_value_rms"]),
        initial_control_rms=float(metadata["initial_value_rms"]),
    )
    if controller.output_basis_rank != metadata["output_basis_rank"]:
        raise ValueError("Fixed-prefix atlas output-basis rank changed")
    controller.load_state_dict(controller_state, strict=True)
    if v7_value_state_sha256(controller) != metadata["controller_state_sha256"]:
        raise ValueError("Fixed-prefix atlas controller state changed")
    controller = controller.to(device=device, dtype=torch.float32).eval()
    return LoadedFixedPrefixAtlas(
        controller=controller,
        probe_embeddings=probes,
        metadata=metadata,
        checkpoint_path=root,
    )


__all__ = [
    "ATLAS_ARCHITECTURE",
    "ATLAS_METADATA_FILENAME",
    "ATLAS_WEIGHTS_FILENAME",
    "LoadedFixedPrefixAtlas",
    "deterministic_spherical_probe_bank",
    "load_fixed_prefix_atlas_checkpoint",
    "save_fixed_prefix_atlas_checkpoint",
    "sha256_file",
    "two_file_checkpoint_fingerprint",
    "validate_fixed_prefix_atlas_metadata",
]
