"""Fail-closed authentication of the complete V94 evidence chain.

This is intentionally separate from V94's sealed preflight, trainer, and
evaluator.  It does not run a model, score answers, mutate an artifact, or open
the reserved answer-bearing validation file.  Instead it independently binds:

* the sealed config, preregistration, CPU preflight, training report, and
  fixed-final bridge;
* the sanitized three-field question manifest;
* all six cached continuous scene memories and their deterministic controls;
* every prediction row and the predictor's completed file-access audit; and
* the create-once aggregate score, if it is present.

The checks deliberately duplicate important evaluator checks.  Evidence is
only useful when its verifier does not trust arbitrary self-consistent hashes
emitted by the process being audited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch
import yaml
from safetensors.torch import load_file

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.question_manifest import (
    QuestionManifest,
    load_question_manifest,
)

ARTIFACT: Final[str] = "gemma4_v94_strict_multiscene_evidence_v1"
CONFIG: Final[Path] = Path(
    "configs/experiments/gemma4_v94_strict_multiscene_full40.yaml"
)
SCENE_IDS: Final[tuple[str, ...]] = tuple(
    f"scene_{index:06d}" for index in range(57, 63)
)
PAIR_SCENE: Final[dict[str, str]] = {
    "scene_000057": "scene_000058",
    "scene_000058": "scene_000057",
    "scene_000059": "scene_000060",
    "scene_000060": "scene_000059",
    "scene_000061": "scene_000062",
    "scene_000062": "scene_000061",
}
ARMS: Final[tuple[str, ...]] = (
    "v94",
    "v85_parent",
    "paired_wrong",
    "zero_payload",
    "shuffled_atlas",
)
MEMORY_SHAPE: Final[tuple[int, int, int]] = (1, 738, 1536)
QUESTION_COUNT: Final[int] = 216
COMPILATION_ATTESTATION_DIRECTORY: Final[str] = (
    "evaluation_cache_compilation_attestation"
)

_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_PREREG_KEYS: Final[frozenset[str]] = frozenset(
    {
        "artifact",
        "schema_version",
        "status",
        "config_path",
        "config_sha256",
        "authenticated_sources",
        "derived_contract",
        "strict_input_contract",
        "dataset_contract",
        "frozen_stack",
        "bridge",
        "training_protocol",
        "evaluation_protocol",
        "fixed_gates",
        "evaluation_label_file_opened",
        "full_gemma_model_loaded",
        "optimizer_constructed",
        "optimizer_updates",
        "behavior_scored",
        "oracle_loaded",
        "runtime_promotion_authorized",
    }
)
_CPU_KEYS: Final[frozenset[str]] = frozenset(
    {
        "artifact",
        "schema_version",
        "status",
        "passed",
        "config_sha256",
        "preregistration_sha256",
        "authenticated_sources",
        "derived_contract",
        "evaluation_label_file_opened",
        "full_gemma_model_loaded",
        "optimizer_constructed",
        "optimizer_updates",
        "behavior_scored",
        "protected_or_sealed_behavior_artifacts_opened",
        "oracle_loaded",
        "runtime_promotion_authorized",
    }
)
_CACHE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "artifact",
        "schema_version",
        "scene_ids",
        "scene_count",
        "shape_each",
        "dtype",
        "compiled_before_questions",
        "question_inputs_used",
        "question_dependent_retrieval",
        "all_memory_slots_retained",
        "environmental_text_inputs",
        "source_runtime_config_sha256",
        "source_v85_adapter_sha256",
        "source_v85_metadata_sha256",
        "source_controller_weights_sha256",
        "source_controller_metadata_sha256",
        "source_probe_weights_sha256",
        "source_probe_metadata_sha256",
        "scenes",
    }
)
_CACHE_SCENE_KEYS: Final[frozenset[str]] = frozenset(
    {"filename", "file_sha256", "file_size_bytes", "memory_sha256"}
)
_PROVENANCE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "artifact",
        "schema_version",
        "config_sha256",
        "question_manifest_sha256",
        "questions_sha256",
        "memory_manifest_sha256",
        "candidate_weights_sha256",
        "candidate_state_sha256",
        "scene_ids",
        "row_count",
        "arms",
        "labels_opened",
        "questions_opened_after_all_memories_bound",
        "question_dependent_retrieval",
        "environmental_text_inputs",
        "provenance_sha256",
    }
)
_PREDICTION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "artifact",
        "scene_id",
        "question_id",
        "paired_scene_id",
        "v94_prediction",
        "v85_parent_prediction",
        "paired_wrong_prediction",
        "zero_payload_prediction",
        "shuffled_atlas_prediction",
        "memory_sha256",
        "paired_memory_sha256",
        "zero_memory_sha256",
        "shuffled_memory_sha256",
        "prefix_hash_unchanged",
        "elapsed_seconds",
        "provenance_sha256",
    }
)
_ACCESS_KEYS: Final[frozenset[str]] = frozenset(
    {
        "loaded_files",
        "forbidden_roots",
        "forbidden_component_names",
        "block_forbidden",
        "forbidden_accesses",
        "passed",
    }
)
_CANDIDATE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "artifact",
        "schema_version",
        "status",
        "parent",
        "v86_through_v93_loaded",
        "bank_name",
        "target_module",
        "rank",
        "alpha",
        "dropout",
        "parameter_count",
        "state_sha256",
        "weights_sha256",
        "tensor_inventory",
        "environmental_memory_serialized",
        "questions_or_answers_serialized",
        "oracle_serialized",
        "evaluation_scored",
        "runtime_promotion_authorized",
        "bindings",
    }
)
_CANDIDATE_BINDING_KEYS: Final[frozenset[str]] = frozenset(
    {
        "config_sha256",
        "preregistration_sha256",
        "cpu_preflight_sha256",
        "frozen_v85_adapter_sha256",
        "row_order_sha256",
        "training_source_sha256",
        "fixed_final_optimizer_updates",
        "class_weight_inventory_sha256",
    }
)
_COMPILATION_PRE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "artifact",
        "schema_version",
        "status",
        "config_sha256",
        "evaluator_source_sha256",
        "numeric_compiler_source_sha256",
        "cache_path",
        "cache_absent_before_compile",
        "maps",
        "questions_opened",
        "labels_opened",
        "oracle_opened",
    }
)
_COMPILATION_POST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "artifact",
        "schema_version",
        "status",
        "pre_attestation_sha256",
        "config_sha256",
        "maps_before",
        "maps_after",
        "maps_unchanged",
        "cache_manifest_file_sha256",
        "cache_manifest_canonical_sha256",
        "memory_sha256",
        "loaded_files",
        "loaded_file_inventory_sha256",
        "forbidden_roots",
        "forbidden_component_names",
        "block_forbidden",
        "forbidden_accesses",
        "protected_read_count",
        "questions_opened",
        "labels_opened",
        "oracle_opened",
    }
)


def _sha256_file(path: Path) -> str:
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
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prefix_sha256(value: torch.Tensor) -> str:
    canonical = value.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(canonical).hexdigest()


def _tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        header = json.dumps(
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"V94 evidence expected a mapping for {label}")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"V94 evidence expected a list for {label}")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"V94 evidence found an invalid SHA-256 for {label}")
    return value


def _resolve(root: Path, raw: str | Path, *, label: str) -> Path:
    value = Path(raw).expanduser()
    if not value.is_absolute() and (".." in value.parts or not value.parts):
        raise ValueError(f"V94 evidence rejected unsafe relative path for {label}: {raw}")
    resolved = (value if value.is_absolute() else root / value).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"V94 evidence path escapes project root for {label}: {raw}") from error
    return resolved


def _regular_file(root: Path, raw: str | Path, *, label: str) -> Path:
    path = _resolve(root, raw, label=label)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"V94 evidence file is absent or linked for {label}: {path}")
    return path


def _json_file(root: Path, raw: str | Path, *, label: str) -> tuple[dict[str, Any], Path]:
    path = _regular_file(root, raw, label=label)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"V94 evidence JSON is not one object for {label}: {path}")
    return value, path


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"V94 evidence found a blank {label} row at {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"V94 evidence {label} row {line_number} is not an object")
            rows.append(value)
    return rows


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> str:
    """Create one immutable JSON record without a replace-existing path."""

    if path.exists() or path.is_symlink():
        raise FileExistsError(f"V94 create-once evidence already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(encoded).hexdigest()


def _load_sealed_config(root: Path, config_path: str | Path) -> tuple[dict[str, Any], Path]:
    path = _regular_file(root, config_path, label="sealed config")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"v94"}:
        raise ValueError("V94 evidence requires exactly one top-level v94 config")
    config = dict(_mapping(payload["v94"], "sealed config payload"))
    if (
        config.get("schema_version") != 94
        or config.get("artifact")
        != "gemma4_v94_strict_multiscene_full40_direct_memory_lora_v1"
        or config.get("status") != "sealed_before_full_model_load"
        or config.get("seed") != 940094
        or any(value == "TO_FILL" for value in _walk_values(config))
    ):
        raise ValueError("V94 evidence config is not the sealed V94 experiment")
    strict = _mapping(config.get("strict_input_contract"), "strict input contract")
    evaluation = _mapping(config.get("evaluation"), "evaluation contract")
    gates = _mapping(config.get("gates"), "gate contract")
    if (
        strict.get("shape_per_scene") != list(MEMORY_SHAPE)
        or strict.get("compiled_before_question") is not True
        or strict.get("reused_byte_identically_across_questions") is not True
        or strict.get("all_memory_slots_retained") is not True
        or strict.get("question_dependent_retrieval") is not False
        or strict.get("environmental_text_inputs") != []
        or evaluation.get("scene_ids") != list(SCENE_IDS)
        or evaluation.get("row_count") != QUESTION_COUNT
        or evaluation.get("labels_opened_by_question_only_predictor") is not False
        or evaluation.get("labels_opened_only_by_separate_scorer") is not True
        or gates.get("every_evaluation_memory_hash_retained_required") is not True
        or gates.get("exact_prefix_hash_invariance_required") is not True
        or gates.get("question_label_isolation_required") is not True
        or gates.get("protected_read_count_maximum") != 0
    ):
        raise ValueError("V94 evidence config weakened a sealed inference/evidence gate")
    return config, path


def _walk_values(value: object) -> Sequence[object]:
    if isinstance(value, Mapping):
        return [item for child in value.values() for item in _walk_values(child)]
    if isinstance(value, list):
        return [item for child in value for item in _walk_values(child)]
    return [value]


def _expected_authenticated_sources(config: Mapping[str, Any]) -> dict[str, str]:
    sources = _mapping(config.get("sources"), "source contract")
    expected: dict[str, str] = {}

    def direct(path_key: str, digest_key: str) -> None:
        raw = sources.get(path_key)
        if not isinstance(raw, str) or Path(raw).is_absolute():
            raise ValueError(f"V94 evidence requires relative source path {path_key}")
        expected[Path(raw).as_posix()] = _require_sha256(
            sources.get(digest_key), digest_key
        )

    for path_key, digest_key in (
        ("runtime_config", "runtime_config_sha256"),
        ("training_qa", "training_qa_sha256"),
        ("split_manifest", "split_manifest_sha256"),
        ("parent_v85_config", "parent_v85_config_sha256"),
        ("parent_v85_preregistration", "parent_v85_preregistration_sha256"),
        ("parent_v85_cpu_preflight", "parent_v85_cpu_preflight_sha256"),
        ("parent_v85_training", "parent_v85_training_sha256"),
        ("parent_v85_fixed_bridge", "parent_v85_fixed_bridge_sha256"),
        ("parent_v85_fixed_metadata", "parent_v85_fixed_metadata_sha256"),
        (
            "parent_v85_development_predictions",
            "parent_v85_development_predictions_sha256",
        ),
        ("parent_v85_development_score", "parent_v85_development_score_sha256"),
        ("sanitized_evaluation_questions", "sanitized_evaluation_questions_sha256"),
        ("preflight_source", "preflight_source_sha256"),
        ("training_source", "training_source_sha256"),
        ("evaluation_source", "evaluation_source_sha256"),
    ):
        direct(path_key, digest_key)

    def child(directory_key: str, filename: str, digest_key: str) -> None:
        raw = sources.get(directory_key)
        if not isinstance(raw, str) or Path(raw).is_absolute():
            raise ValueError(f"V94 evidence requires relative source path {directory_key}")
        expected[(Path(raw) / filename).as_posix()] = _require_sha256(
            sources.get(digest_key), digest_key
        )

    child("train_memory_cache", "training_tensors.safetensors", "train_memory_tensor_sha256")
    child("train_memory_cache", "metadata.json", "train_memory_metadata_sha256")
    child(
        "development_memory_cache",
        "training_tensors.safetensors",
        "development_memory_tensor_sha256",
    )
    child(
        "development_memory_cache", "metadata.json", "development_memory_metadata_sha256"
    )
    child("frozen_v85_checkpoint", "adapter.safetensors", "frozen_v85_adapter_sha256")
    child("frozen_v85_checkpoint", "runtime_metadata.json", "frozen_v85_metadata_sha256")
    child(
        "evaluation_memory_controller",
        "control.safetensors",
        "evaluation_memory_controller_weights_sha256",
    )
    child(
        "evaluation_memory_controller",
        "runtime_metadata.json",
        "evaluation_memory_controller_metadata_sha256",
    )
    child(
        "evaluation_probe_bank", "probes.safetensors", "evaluation_probe_tensor_sha256"
    )
    child(
        "evaluation_probe_bank",
        "runtime_metadata.json",
        "evaluation_probe_metadata_sha256",
    )
    expected["evaluation_labels_declared_sha256_not_opened"] = _require_sha256(
        sources.get("evaluation_qa_sha256"), "evaluation QA declaration"
    )
    return dict(sorted(expected.items()))


def _authenticate_declared_sources(
    root: Path, config: Mapping[str, Any], declared: object
) -> dict[str, str]:
    observed = dict(_mapping(declared, "authenticated source inventory"))
    expected = _expected_authenticated_sources(config)
    if observed != expected:
        raise ValueError("V94 authenticated source inventory changed")
    for raw, digest in expected.items():
        if raw == "evaluation_labels_declared_sha256_not_opened":
            continue
        path = _regular_file(root, raw, label=f"sealed source {raw}")
        if _sha256_file(path) != digest:
            raise ValueError(f"V94 current sealed source bytes changed: {raw}")
    return expected


def _authenticate_seal_chain(
    root: Path, config: Mapping[str, Any], config_path: Path
) -> dict[str, Any]:
    outputs = _mapping(config.get("outputs"), "outputs")
    prereg, prereg_path = _json_file(
        root, str(outputs.get("preregistration")), label="preregistration"
    )
    cpu, cpu_path = _json_file(
        root, str(outputs.get("cpu_preflight")), label="CPU preflight"
    )
    if set(prereg) != _PREREG_KEYS or set(cpu) != _CPU_KEYS:
        raise ValueError("V94 preregistration or CPU preflight fields changed")
    config_sha = _sha256_file(config_path)
    prereg_sha = _sha256_file(prereg_path)
    cpu_sha = _sha256_file(cpu_path)
    if (
        prereg.get("artifact")
        != "gemma4_v94_strict_multiscene_full40_preregistration_v1"
        or prereg.get("schema_version") != 94
        or prereg.get("status")
        != "sealed_before_first_v94_full_model_load_or_validation_label_read"
        or prereg.get("config_path") != config_path.relative_to(root).as_posix()
        or prereg.get("config_sha256") != config_sha
        or prereg.get("strict_input_contract") != config.get("strict_input_contract")
        or prereg.get("dataset_contract") != config.get("dataset")
        or prereg.get("frozen_stack") != config.get("frozen_stack")
        or prereg.get("bridge") != config.get("bridge")
        or prereg.get("training_protocol") != config.get("training")
        or prereg.get("evaluation_protocol") != config.get("evaluation")
        or prereg.get("fixed_gates") != config.get("gates")
        or prereg.get("evaluation_label_file_opened") is not False
        or prereg.get("full_gemma_model_loaded") is not False
        or prereg.get("optimizer_constructed") is not False
        or prereg.get("optimizer_updates") != 0
        or prereg.get("behavior_scored") is not False
        or prereg.get("oracle_loaded") is not False
        or prereg.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V94 sealed preregistration does not bind the current config")
    authenticated_sources = _authenticate_declared_sources(
        root, config, prereg.get("authenticated_sources")
    )
    if (
        cpu.get("artifact")
        != "gemma4_v94_strict_multiscene_full40_cpu_preflight_v1"
        or cpu.get("schema_version") != 94
        or cpu.get("status") != "passed"
        or cpu.get("passed") is not True
        or cpu.get("config_sha256") != config_sha
        or cpu.get("preregistration_sha256") != prereg_sha
        or cpu.get("authenticated_sources") != authenticated_sources
        or cpu.get("derived_contract") != prereg.get("derived_contract")
        or cpu.get("evaluation_label_file_opened") is not False
        or cpu.get("full_gemma_model_loaded") is not False
        or cpu.get("optimizer_constructed") is not False
        or cpu.get("optimizer_updates") != 0
        or cpu.get("behavior_scored") is not False
        or cpu.get("protected_or_sealed_behavior_artifacts_opened") != []
        or cpu.get("oracle_loaded") is not False
        or cpu.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V94 CPU preflight does not bind the sealed preregistration")
    return {
        "config_sha256": config_sha,
        "preregistration_sha256": prereg_sha,
        "cpu_preflight_sha256": cpu_sha,
        "authenticated_sources": authenticated_sources,
    }


def _authenticate_training_and_candidate(
    root: Path, config: Mapping[str, Any], seal: Mapping[str, Any]
) -> dict[str, Any]:
    outputs = _mapping(config.get("outputs"), "outputs")
    report, report_path = _json_file(
        root, str(outputs.get("training_report")), label="training report"
    )
    candidate_root = _resolve(
        root, str(outputs.get("fixed_final_candidate")), label="fixed-final candidate"
    )
    if candidate_root.is_symlink() or not candidate_root.is_dir():
        raise FileNotFoundError("V94 fixed-final candidate directory is absent or linked")
    metadata, metadata_path = _json_file(
        root,
        candidate_root.relative_to(root) / "runtime_metadata.json",
        label="candidate metadata",
    )
    weights_path = _regular_file(
        root,
        candidate_root.relative_to(root) / "bridge.safetensors",
        label="candidate weights",
    )
    if set(metadata) != _CANDIDATE_KEYS:
        raise ValueError("V94 candidate metadata fields changed")
    bindings = dict(_mapping(metadata.get("bindings"), "candidate bindings"))
    if set(bindings) != _CANDIDATE_BINDING_KEYS:
        raise ValueError("V94 candidate binding fields changed")
    sources = _mapping(config.get("sources"), "sources")
    dataset = _mapping(config.get("dataset"), "dataset")
    training = _mapping(config.get("training"), "training")
    expected_bindings = {
        "config_sha256": seal["config_sha256"],
        "preregistration_sha256": seal["preregistration_sha256"],
        "cpu_preflight_sha256": seal["cpu_preflight_sha256"],
        "frozen_v85_adapter_sha256": sources.get("frozen_v85_adapter_sha256"),
        "row_order_sha256": training.get("row_order_sha256"),
        "training_source_sha256": sources.get("training_source_sha256"),
        "fixed_final_optimizer_updates": 360,
        "class_weight_inventory_sha256": dataset.get(
            "inverse_sqrt_class_weight_inventory_sha256"
        ),
    }
    weights_sha = _sha256_file(weights_path)
    if (
        metadata.get("artifact")
        != "gemma4_v94_strict_multiscene_full40_fixed_final_v1"
        or metadata.get("schema_version") != 94
        or metadata.get("status")
        != "fixed_final_awaiting_preregistered_acceptance_gates"
        or metadata.get("parent") != "exact_v85_strict_runtime_candidate_only"
        or metadata.get("v86_through_v93_loaded") is not False
        or metadata.get("bank_name") != "v94_strict_multiscene_full40_bridge"
        or metadata.get("target_module")
        != "model.language_model.layers.34.mlp.gate_proj"
        or metadata.get("rank") != 8
        or metadata.get("alpha") != 16.0
        or metadata.get("dropout") != 0.0
        or metadata.get("parameter_count") != 110592
        or metadata.get("weights_sha256") != weights_sha
        or metadata.get("tensor_inventory") != ["lora_a", "lora_b"]
        or metadata.get("environmental_memory_serialized") is not False
        or metadata.get("questions_or_answers_serialized") is not False
        or metadata.get("oracle_serialized") is not False
        or metadata.get("evaluation_scored") is not False
        or metadata.get("runtime_promotion_authorized") is not False
        or bindings != expected_bindings
    ):
        raise ValueError("V94 fixed-final candidate metadata changed")
    tensors = load_file(str(weights_path), device="cpu")
    if (
        set(tensors) != {"lora_a", "lora_b"}
        or tuple(tensors["lora_a"].shape) != (8, 1536)
        or tuple(tensors["lora_b"].shape) != (12288, 8)
        or tensors["lora_a"].dtype != torch.float32
        or tensors["lora_b"].dtype != torch.float32
        or not all(torch.isfinite(value).all() for value in tensors.values())
    ):
        raise ValueError("V94 candidate tensor topology or values changed")
    candidate_state = _tensor_state_sha256(
        {
            "adapters.0.lora_a": tensors["lora_a"],
            "adapters.0.lora_b": tensors["lora_b"],
        }
    )
    if metadata.get("state_sha256") != candidate_state:
        raise ValueError("V94 candidate state digest does not match its tensors")
    report_candidate = _mapping(report.get("candidate"), "training candidate")
    scene_memories = _mapping(report.get("scene_memories"), "training memories")
    report_gates = dict(_mapping(report.get("gates"), "training gates"))
    expected_candidate_path = candidate_root.relative_to(root).as_posix()
    if (
        report.get("artifact")
        != "gemma4_v94_strict_multiscene_full40_training_v1"
        or report.get("schema_version") != 94
        or report.get("status") != "fixed_final_training_complete_not_promoted"
        or report.get("config_sha256") != seal["config_sha256"]
        or report.get("preregistration_sha256") != seal["preregistration_sha256"]
        or report.get("cpu_preflight_sha256") != seal["cpu_preflight_sha256"]
        or report.get("source_hashes") != seal["authenticated_sources"]
        or report.get("optimizer_updates") != 360
        or report.get("micro_rows_consumed") != 2880
        or report.get("unique_training_rows") != 960
        or report.get("training_scene_count") != 40
        or report.get("paired_wrong_margin_rows_consumed") != 396
        or report.get("causal_margin_rows_consumed") != 54
        or not report_gates
        or not all(value is True for value in report_gates.values())
        or scene_memories.get("hash_invariant") is not True
        or scene_memories.get("hashes_before") != scene_memories.get("hashes_after")
        or len(_mapping(scene_memories.get("hashes_before"), "training memory hashes"))
        != 40
        or report_candidate.get("path") != expected_candidate_path
        or report_candidate.get("weights_sha256") != weights_sha
        or report_candidate.get("metadata_canonical_sha256")
        != _canonical_sha256(metadata)
        or report_candidate.get("fixed_final") is not True
        or report_candidate.get("runtime_promotion_authorized") is not False
        or report.get("protected_read_count") != 0
        or report.get("official_validation_loaded") is not False
        or report.get("official_test_loaded") is not False
        or report.get("deferred_final_loaded") is not False
        or report.get("oracle_loaded") is not False
        or report.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V94 training report does not authenticate the fixed-final run")
    return {
        "training_report_sha256": _sha256_file(report_path),
        "candidate_weights_sha256": weights_sha,
        "candidate_metadata_sha256": _sha256_file(metadata_path),
        "candidate_metadata_canonical_sha256": _canonical_sha256(metadata),
        "candidate_state_sha256": candidate_state,
        "candidate_weights_path": weights_path,
        "candidate_metadata_path": metadata_path,
    }


def _authenticate_questions(
    root: Path, config: Mapping[str, Any]
) -> tuple[QuestionManifest, Path]:
    sources = _mapping(config.get("sources"), "sources")
    outputs = _mapping(config.get("outputs"), "outputs")
    source_path = _regular_file(
        root,
        str(sources.get("sanitized_evaluation_questions")),
        label="sanitized question manifest",
    )
    output_path = _resolve(
        root,
        str(outputs.get("evaluation_question_manifest")),
        label="question-manifest output",
    )
    if source_path != output_path:
        raise ValueError("V94 predictor question path differs from the sealed sanitized source")
    expected_sha = _require_sha256(
        sources.get("sanitized_evaluation_questions_sha256"),
        "sanitized question manifest",
    )
    if _sha256_file(source_path) != expected_sha:
        raise ValueError("V94 sanitized question-manifest bytes changed")
    manifest = load_question_manifest(source_path)
    if (
        manifest.manifest_sha256 != expected_sha
        or manifest.source_qa_sha256 != sources.get("evaluation_qa_sha256")
        or manifest.question_count != QUESTION_COUNT
        or manifest.scene_count != len(SCENE_IDS)
        or {row.scene_id for row in manifest.questions} != set(SCENE_IDS)
    ):
        raise ValueError("V94 sanitized question manifest changed")
    return manifest, source_path


def _numeric_compiler_sources(config: Mapping[str, Any]) -> dict[str, str]:
    sources = _mapping(config.get("sources"), "sources")
    return {
        "runtime_config_sha256": _require_sha256(
            sources.get("runtime_config_sha256"), "runtime config"
        ),
        "v85_adapter_sha256": _require_sha256(
            sources.get("frozen_v85_adapter_sha256"), "V85 adapter"
        ),
        "v85_metadata_sha256": _require_sha256(
            sources.get("frozen_v85_metadata_sha256"), "V85 metadata"
        ),
        "controller_weights_sha256": _require_sha256(
            sources.get("evaluation_memory_controller_weights_sha256"),
            "memory controller weights",
        ),
        "controller_metadata_sha256": _require_sha256(
            sources.get("evaluation_memory_controller_metadata_sha256"),
            "memory controller metadata",
        ),
        "probe_weights_sha256": _require_sha256(
            sources.get("evaluation_probe_tensor_sha256"), "probe weights"
        ),
        "probe_metadata_sha256": _require_sha256(
            sources.get("evaluation_probe_metadata_sha256"), "probe metadata"
        ),
    }


def _current_map_inventory(
    root: Path, config: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    sources = _mapping(config.get("sources"), "sources")
    runtime_path = _regular_file(
        root, str(sources.get("runtime_config")), label="runtime config"
    )
    runtime = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
    paths = _mapping(_mapping(runtime, "runtime config").get("paths"), "runtime paths")
    maps_root_raw = paths.get("maps_root")
    if not isinstance(maps_root_raw, str):
        raise TypeError("V94 runtime maps_root must be a path")
    maps_root = _resolve(root, maps_root_raw, label="runtime maps root")
    inventory: dict[str, dict[str, Any]] = {}
    for scene_id in SCENE_IDS:
        path = maps_root / scene_id / "voxel_map.npz"
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"V94 source voxel map is absent or linked: {path}")
        inventory[scene_id] = {
            "path": path.relative_to(root).as_posix(),
            "file_sha256": _sha256_file(path),
            "file_size_bytes": path.stat().st_size,
        }
    return inventory


def _precompile_seal_identity(
    root: Path, config: Mapping[str, Any], config_path: Path
) -> dict[str, Any]:
    """Authenticate the seal without opening questions, training QA, or labels."""

    outputs = _mapping(config.get("outputs"), "outputs")
    prereg, prereg_path = _json_file(
        root, str(outputs.get("preregistration")), label="preregistration"
    )
    cpu, _cpu_path = _json_file(
        root, str(outputs.get("cpu_preflight")), label="CPU preflight"
    )
    config_sha = _sha256_file(config_path)
    prereg_sha = _sha256_file(prereg_path)
    declared = _expected_authenticated_sources(config)
    if (
        prereg.get("artifact")
        != "gemma4_v94_strict_multiscene_full40_preregistration_v1"
        or prereg.get("status")
        != "sealed_before_first_v94_full_model_load_or_validation_label_read"
        or prereg.get("config_sha256") != config_sha
        or prereg.get("authenticated_sources") != declared
        or cpu.get("artifact")
        != "gemma4_v94_strict_multiscene_full40_cpu_preflight_v1"
        or cpu.get("passed") is not True
        or cpu.get("config_sha256") != config_sha
        or cpu.get("preregistration_sha256") != prereg_sha
        or cpu.get("authenticated_sources") != declared
    ):
        raise ValueError("V94 precompile wrapper does not have the exact sealed contract")
    sources = _mapping(config.get("sources"), "sources")
    numeric_paths = {
        "runtime_config_sha256": _regular_file(
            root, str(sources["runtime_config"]), label="runtime config"
        ),
        "v85_adapter_sha256": _regular_file(
            root,
            Path(str(sources["frozen_v85_checkpoint"])) / "adapter.safetensors",
            label="V85 adapter",
        ),
        "v85_metadata_sha256": _regular_file(
            root,
            Path(str(sources["frozen_v85_checkpoint"])) / "runtime_metadata.json",
            label="V85 metadata",
        ),
        "controller_weights_sha256": _regular_file(
            root,
            Path(str(sources["evaluation_memory_controller"])) / "control.safetensors",
            label="memory controller weights",
        ),
        "controller_metadata_sha256": _regular_file(
            root,
            Path(str(sources["evaluation_memory_controller"]))
            / "runtime_metadata.json",
            label="memory controller metadata",
        ),
        "probe_weights_sha256": _regular_file(
            root,
            Path(str(sources["evaluation_probe_bank"])) / "probes.safetensors",
            label="probe weights",
        ),
        "probe_metadata_sha256": _regular_file(
            root,
            Path(str(sources["evaluation_probe_bank"])) / "runtime_metadata.json",
            label="probe metadata",
        ),
    }
    expected = _numeric_compiler_sources(config)
    if any(_sha256_file(path) != expected[name] for name, path in numeric_paths.items()):
        raise ValueError("V94 numeric compiler source changed before cache compilation")
    return {"config_sha256": config_sha, "numeric_sources": expected}


def _attestation_root(root: Path, config: Mapping[str, Any]) -> Path:
    outputs = _mapping(config.get("outputs"), "outputs")
    cache_root = _resolve(
        root, str(outputs.get("evaluation_memory_cache")), label="evaluation cache"
    )
    return cache_root.parent / COMPILATION_ATTESTATION_DIRECTORY


def compile_evaluation_memory_cache_with_attestation_v94(
    config_path: str | Path = CONFIG,
    *,
    root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Create the V94 cache behind pre/post map and access attestations.

    The sealed compiler itself cannot be edited after preregistration.  This
    wrapper supplies the missing source-map provenance while retaining that
    compiler byte-for-byte.  It refuses an existing cache or attestation; an
    interrupted run therefore cannot be silently resumed without its original
    access inventory.
    """

    project_root = Path(root).expanduser().resolve()
    if project_root != PROJECT_ROOT.resolve():
        raise ValueError("V94 sealed compiler can only run at its pinned project root")
    config, resolved_config = _load_sealed_config(project_root, config_path)
    seal = _precompile_seal_identity(project_root, config, resolved_config)
    outputs = _mapping(config.get("outputs"), "outputs")
    sources = _mapping(config.get("sources"), "sources")
    cache_root = _resolve(
        project_root,
        str(outputs.get("evaluation_memory_cache")),
        label="evaluation cache",
    )
    attestation_root = _attestation_root(project_root, config)
    if cache_root.exists() or cache_root.is_symlink():
        raise FileExistsError("V94 attested compilation requires an absent cache")
    if attestation_root.exists() or attestation_root.is_symlink():
        raise FileExistsError("V94 create-once compilation attestation already exists")
    attestation_root.parent.mkdir(parents=True, exist_ok=True)
    attestation_root.mkdir()
    maps_before = _current_map_inventory(project_root, config)
    evaluator_path = _regular_file(
        project_root, str(sources["evaluation_source"]), label="sealed evaluator source"
    )
    if _sha256_file(evaluator_path) != sources["evaluation_source_sha256"]:
        raise ValueError("V94 sealed evaluator source changed before compilation")
    pre = {
        "artifact": "v94_evaluation_cache_precompile_attestation_v1",
        "schema_version": 1,
        "status": "sealed_before_cache_creation",
        "config_sha256": seal["config_sha256"],
        "evaluator_source_sha256": sources["evaluation_source_sha256"],
        "numeric_compiler_source_sha256": seal["numeric_sources"],
        "cache_path": cache_root.relative_to(project_root).as_posix(),
        "cache_absent_before_compile": True,
        "maps": maps_before,
        "questions_opened": False,
        "labels_opened": False,
        "oracle_opened": False,
    }
    pre_path = attestation_root / "pre.json"
    pre_sha = _atomic_create_json(pre_path, pre)

    question_path = _resolve(
        project_root,
        str(outputs["evaluation_question_manifest"]),
        label="sanitized questions",
    )
    forbidden_roots = sorted({*_expected_forbidden_roots(project_root), str(question_path)})
    audit = FileAccessAudit(
        forbidden_roots=[Path(path) for path in forbidden_roots],
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    with audit:
        from semantic_3d_chat.evaluation.evaluate_v94_strict_multiscene_full40 import (
            compile_evaluation_memory_cache_v94,
        )

        compile_result = compile_evaluation_memory_cache_v94(resolved_config)
        # Authenticate the renamed final paths while the outer audit is still
        # live.  The sealed compiler first verifies files in its temporary
        # directory and then atomically renames it.
        cache = _authenticate_memory_cache(project_root, config)
        maps_after = _current_map_inventory(project_root, config)
    audit.assert_clean()
    if compile_result.get("created") is not True or compile_result.get(
        "protected_read_count"
    ) != 0:
        raise RuntimeError("V94 sealed compiler did not create one clean cache")
    if maps_after != maps_before:
        raise RuntimeError("V94 source voxel maps changed during cache compilation")
    loaded = audit.unique_paths
    loaded_set = set(loaded)
    mandatory = {
        *(str(_resolve(project_root, row["path"], label=scene)) for scene, row in maps_before.items()),
        str(cache["manifest_path"]),
        *(str(path) for path in cache["memory_paths"].values()),
    }
    if (
        not mandatory <= loaded_set
        or str(question_path) in loaded_set
        or any({"qa", "oracle"} & {part.casefold() for part in Path(path).parts} for path in loaded)
    ):
        raise RuntimeError("V94 compiler access inventory is incomplete or protected")
    post = {
        "artifact": "v94_evaluation_cache_postcompile_attestation_v1",
        "schema_version": 1,
        "status": "complete_cache_bound_no_questions_or_labels",
        "pre_attestation_sha256": pre_sha,
        "config_sha256": seal["config_sha256"],
        "maps_before": maps_before,
        "maps_after": maps_after,
        "maps_unchanged": True,
        "cache_manifest_file_sha256": cache["manifest_file_sha256"],
        "cache_manifest_canonical_sha256": cache["manifest_canonical_sha256"],
        "memory_sha256": cache["memory_hashes"],
        "loaded_files": loaded,
        "loaded_file_inventory_sha256": _canonical_sha256(loaded),
        "forbidden_roots": forbidden_roots,
        "forbidden_component_names": ["oracle"],
        "block_forbidden": True,
        "forbidden_accesses": [],
        "protected_read_count": 0,
        "questions_opened": False,
        "labels_opened": False,
        "oracle_opened": False,
    }
    post_path = attestation_root / "post.json"
    post_sha = _atomic_create_json(post_path, post)
    return {
        "artifact": "v94_attested_evaluation_cache_compilation_v1",
        "pre_attestation_sha256": pre_sha,
        "post_attestation_sha256": post_sha,
        "cache_manifest_sha256": cache["manifest_file_sha256"],
        "memory_sha256": cache["memory_hashes"],
        "protected_read_count": 0,
    }


def _zero_payload(memory: torch.Tensor) -> torch.Tensor:
    result = memory.clone()
    result[:, 1:-1].zero_()
    return result


def _shuffle_atlas(memory: torch.Tensor) -> torch.Tensor:
    # [BOI][96 * (probe key + 4 values)][all 256 base latents][EOI]
    atlas = memory[:, 1:481].reshape(1, 96, 5, 1536)
    keys = atlas[:, :, :1]
    values = atlas[:, :, 1:].roll(shifts=1, dims=1)
    shuffled = torch.cat((keys, values), dim=2).reshape(1, 480, 1536)
    return torch.cat((memory[:, :1], shuffled, memory[:, 481:]), dim=1)


def _authenticate_memory_cache(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    outputs = _mapping(config.get("outputs"), "outputs")
    sources = _mapping(config.get("sources"), "sources")
    cache_root = _resolve(
        root, str(outputs.get("evaluation_memory_cache")), label="evaluation cache"
    )
    if cache_root.is_symlink() or not cache_root.is_dir():
        raise FileNotFoundError("V94 evaluation cache is absent or linked")
    manifest, manifest_path = _json_file(
        root, cache_root.relative_to(root) / "manifest.json", label="cache manifest"
    )
    if set(manifest) != _CACHE_KEYS:
        raise ValueError("V94 cache-manifest fields changed")
    exact = {
        "artifact": "v94_question_independent_evaluation_memory_cache_v1",
        "schema_version": 1,
        "scene_ids": list(SCENE_IDS),
        "scene_count": 6,
        "shape_each": list(MEMORY_SHAPE),
        "dtype": "bfloat16",
        "compiled_before_questions": True,
        "question_inputs_used": False,
        "question_dependent_retrieval": False,
        "all_memory_slots_retained": True,
        "environmental_text_inputs": [],
        "source_runtime_config_sha256": sources.get("runtime_config_sha256"),
        "source_v85_adapter_sha256": sources.get("frozen_v85_adapter_sha256"),
        "source_v85_metadata_sha256": sources.get("frozen_v85_metadata_sha256"),
        "source_controller_weights_sha256": sources.get(
            "evaluation_memory_controller_weights_sha256"
        ),
        "source_controller_metadata_sha256": sources.get(
            "evaluation_memory_controller_metadata_sha256"
        ),
        "source_probe_weights_sha256": sources.get("evaluation_probe_tensor_sha256"),
        "source_probe_metadata_sha256": sources.get(
            "evaluation_probe_metadata_sha256"
        ),
    }
    if any(manifest.get(key) != value for key, value in exact.items()):
        raise ValueError("V94 cache manifest is not bound to the current sealed sources")
    entries = _mapping(manifest.get("scenes"), "cache scenes")
    if set(entries) != set(SCENE_IDS):
        raise ValueError("V94 cache must contain exactly the six held-out scenes")
    expected_files = {"manifest.json", *(f"{scene}.safetensors" for scene in SCENE_IDS)}
    if {path.name for path in cache_root.iterdir()} != expected_files:
        raise ValueError("V94 evaluation cache contains unexpected files")
    memory_hashes: dict[str, str] = {}
    zero_hashes: dict[str, str] = {}
    shuffled_hashes: dict[str, str] = {}
    memory_paths: dict[str, Path] = {}
    for scene_id in SCENE_IDS:
        entry = dict(_mapping(entries[scene_id], f"cache entry {scene_id}"))
        if set(entry) != _CACHE_SCENE_KEYS or entry.get("filename") != f"{scene_id}.safetensors":
            raise ValueError(f"V94 cache entry changed for {scene_id}")
        path = _regular_file(
            root, cache_root.relative_to(root) / str(entry["filename"]), label=scene_id
        )
        if (
            path.stat().st_size != entry.get("file_size_bytes")
            or _sha256_file(path) != entry.get("file_sha256")
        ):
            raise ValueError(f"V94 cached memory bytes changed for {scene_id}")
        state = load_file(str(path), device="cpu")
        if set(state) != {"scene_memory"}:
            raise ValueError(f"V94 cached memory tensor inventory changed for {scene_id}")
        memory = state["scene_memory"].detach().contiguous()
        if (
            tuple(memory.shape) != MEMORY_SHAPE
            or memory.dtype != torch.bfloat16
            or not torch.isfinite(memory).all()
        ):
            raise ValueError(f"V94 cached memory tensor changed for {scene_id}")
        memory_sha = _prefix_sha256(memory)
        if memory_sha != entry.get("memory_sha256"):
            raise ValueError(f"V94 cached memory semantic hash changed for {scene_id}")
        memory_hashes[scene_id] = memory_sha
        zero_hashes[scene_id] = _prefix_sha256(_zero_payload(memory))
        shuffled_hashes[scene_id] = _prefix_sha256(_shuffle_atlas(memory))
        memory_paths[scene_id] = path
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_file_sha256": _sha256_file(manifest_path),
        "manifest_canonical_sha256": _canonical_sha256(manifest),
        "memory_hashes": memory_hashes,
        "zero_hashes": zero_hashes,
        "shuffled_hashes": shuffled_hashes,
        "memory_paths": memory_paths,
        "all_six_memory_hashes_retained": len(memory_hashes) == 6,
    }


def _authenticate_compilation_attestation(
    root: Path,
    config: Mapping[str, Any],
    *,
    config_sha256: str,
    cache: Mapping[str, Any],
) -> dict[str, str]:
    attestation_root = _attestation_root(root, config)
    if attestation_root.is_symlink() or not attestation_root.is_dir():
        raise FileNotFoundError("V94 cache compilation attestation is absent or linked")
    if {path.name for path in attestation_root.iterdir()} != {"pre.json", "post.json"}:
        raise ValueError("V94 cache compilation attestation has unexpected files")
    pre, pre_path = _json_file(
        root, attestation_root.relative_to(root) / "pre.json", label="precompile attestation"
    )
    post, post_path = _json_file(
        root,
        attestation_root.relative_to(root) / "post.json",
        label="postcompile attestation",
    )
    if set(pre) != _COMPILATION_PRE_KEYS or set(post) != _COMPILATION_POST_KEYS:
        raise ValueError("V94 cache compilation attestation fields changed")
    sources = _mapping(config.get("sources"), "sources")
    outputs = _mapping(config.get("outputs"), "outputs")
    cache_path = _resolve(root, str(outputs["evaluation_memory_cache"]), label="cache")
    maps = _current_map_inventory(root, config)
    expected_pre = {
        "artifact": "v94_evaluation_cache_precompile_attestation_v1",
        "schema_version": 1,
        "status": "sealed_before_cache_creation",
        "config_sha256": config_sha256,
        "evaluator_source_sha256": sources.get("evaluation_source_sha256"),
        "numeric_compiler_source_sha256": _numeric_compiler_sources(config),
        "cache_path": cache_path.relative_to(root).as_posix(),
        "cache_absent_before_compile": True,
        "maps": maps,
        "questions_opened": False,
        "labels_opened": False,
        "oracle_opened": False,
    }
    if pre != expected_pre:
        raise ValueError("V94 precompile attestation is not current")
    loaded = _list(post.get("loaded_files"), "compiler loaded files")
    forbidden_roots = _list(post.get("forbidden_roots"), "compiler forbidden roots")
    question_path = _resolve(
        root,
        str(outputs["evaluation_question_manifest"]),
        label="sanitized questions",
    )
    expected_forbidden = sorted({*_expected_forbidden_roots(root), str(question_path)})
    mandatory = {
        *(str(_resolve(root, row["path"], label=scene)) for scene, row in maps.items()),
        str(cache["manifest_path"]),
        *(str(path) for path in cache["memory_paths"].values()),
    }
    if (
        post.get("artifact")
        != "v94_evaluation_cache_postcompile_attestation_v1"
        or post.get("schema_version") != 1
        or post.get("status") != "complete_cache_bound_no_questions_or_labels"
        or post.get("pre_attestation_sha256") != _sha256_file(pre_path)
        or post.get("config_sha256") != config_sha256
        or post.get("maps_before") != maps
        or post.get("maps_after") != maps
        or post.get("maps_unchanged") is not True
        or post.get("cache_manifest_file_sha256") != cache["manifest_file_sha256"]
        or post.get("cache_manifest_canonical_sha256")
        != cache["manifest_canonical_sha256"]
        or post.get("memory_sha256") != cache["memory_hashes"]
        or loaded != sorted(set(loaded))
        or post.get("loaded_file_inventory_sha256") != _canonical_sha256(loaded)
        or forbidden_roots != expected_forbidden
        or post.get("forbidden_component_names") != ["oracle"]
        or post.get("block_forbidden") is not True
        or post.get("forbidden_accesses") != []
        or post.get("protected_read_count") != 0
        or post.get("questions_opened") is not False
        or post.get("labels_opened") is not False
        or post.get("oracle_opened") is not False
        or not mandatory <= set(loaded)
        or str(question_path) in set(loaded)
        or any(
            {"qa", "oracle"} & {part.casefold() for part in Path(path).parts}
            for path in loaded
        )
    ):
        raise ValueError("V94 postcompile attestation is incomplete or not current")
    return {
        "pre_attestation_sha256": _sha256_file(pre_path),
        "post_attestation_sha256": _sha256_file(post_path),
    }


def _authenticate_prediction_provenance(
    provenance: Mapping[str, Any],
    *,
    config_sha256: str,
    questions: QuestionManifest,
    cache: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> str:
    if set(provenance) != _PROVENANCE_KEYS:
        raise ValueError("V94 prediction provenance fields changed")
    supplied = provenance.get("provenance_sha256")
    unsigned = dict(provenance)
    unsigned.pop("provenance_sha256", None)
    expected = {
        "artifact": "v94_question_only_same216_predictions_v1",
        "schema_version": 1,
        "config_sha256": config_sha256,
        "question_manifest_sha256": questions.manifest_sha256,
        "questions_sha256": questions.questions_sha256,
        "memory_manifest_sha256": cache["manifest_canonical_sha256"],
        "candidate_weights_sha256": candidate["candidate_weights_sha256"],
        "candidate_state_sha256": candidate["candidate_state_sha256"],
        "scene_ids": list(SCENE_IDS),
        "row_count": QUESTION_COUNT,
        "arms": list(ARMS),
        "labels_opened": False,
        "questions_opened_after_all_memories_bound": True,
        "question_dependent_retrieval": False,
        "environmental_text_inputs": [],
    }
    if unsigned != expected or supplied != _canonical_sha256(expected):
        raise ValueError("V94 prediction provenance is self-consistent but not current")
    return str(supplied)


def _authenticate_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    questions: QuestionManifest,
    cache: Mapping[str, Any],
    provenance_sha256: str,
) -> dict[str, Any]:
    expected_keys = {(row.scene_id, row.question_id) for row in questions.questions}
    observed: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        if set(row) != _PREDICTION_KEYS:
            raise ValueError(f"V94 prediction fields changed at row {index}")
        scene_id = row.get("scene_id")
        question_id = row.get("question_id")
        key = (str(scene_id), str(question_id))
        paired = PAIR_SCENE.get(str(scene_id))
        elapsed = row.get("elapsed_seconds")
        if key in observed:
            raise ValueError(f"V94 duplicate prediction key: {key}")
        observed.add(key)
        if (
            key not in expected_keys
            or paired is None
            or row.get("artifact") != "v94_question_only_same216_predictions_v1"
            or row.get("paired_scene_id") != paired
            or row.get("provenance_sha256") != provenance_sha256
            or row.get("memory_sha256") != cache["memory_hashes"][str(scene_id)]
            or row.get("paired_memory_sha256") != cache["memory_hashes"][paired]
            or row.get("zero_memory_sha256") != cache["zero_hashes"][str(scene_id)]
            or row.get("shuffled_memory_sha256")
            != cache["shuffled_hashes"][str(scene_id)]
            or row.get("prefix_hash_unchanged") is not True
            or isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0.0
            or not all(
                isinstance(row.get(field), str)
                for field in (
                    "v94_prediction",
                    "v85_parent_prediction",
                    "paired_wrong_prediction",
                    "zero_payload_prediction",
                    "shuffled_atlas_prediction",
                )
            )
        ):
            raise ValueError(f"V94 prediction row is not bound to cached memory: {key}")
    if len(rows) != QUESTION_COUNT or observed != expected_keys:
        raise ValueError("V94 predictions do not cover the exact 216 sanitized keys")
    records = {
        (str(row["scene_id"]), str(row["question_id"])): row for row in rows
    }
    question_by_key = {
        (row.scene_id, row.question_id): row.question for row in questions.questions
    }
    key_by_scene_question = {
        (row.scene_id, row.question): (row.scene_id, row.question_id)
        for row in questions.questions
    }

    def normalized(value: object) -> str:
        return " ".join(str(value).strip().casefold().split())

    paired_matches = 0
    paired_comparable = 0
    for key, row in records.items():
        paired_key = key_by_scene_question.get(
            (PAIR_SCENE[key[0]], question_by_key[key])
        )
        if paired_key is None:
            continue
        paired_comparable += 1
        paired_matches += normalized(row["paired_wrong_prediction"]) == normalized(
            records[paired_key]["v94_prediction"]
        )
    return {
        "diagnostic_scope": "posthoc_non_preregistered_not_a_promotion_gate",
        "paired_wrong_raw_prediction_change_count": sum(
            normalized(row["v94_prediction"])
            != normalized(row["paired_wrong_prediction"])
            for row in rows
        ),
        "zero_payload_raw_prediction_change_count": sum(
            normalized(row["v94_prediction"])
            != normalized(row["zero_payload_prediction"])
            for row in rows
        ),
        "atlas_value_shuffle_raw_prediction_change_count": sum(
            normalized(row["v94_prediction"])
            != normalized(row["shuffled_atlas_prediction"])
            for row in rows
        ),
        "paired_wrong_direct_pair_comparable_count": paired_comparable,
        "paired_wrong_matches_direct_paired_scene_prediction_count": paired_matches,
    }


def _expected_forbidden_roots(root: Path) -> set[str]:
    roots = [root / "data" / "oracle"]
    roots.extend(root.glob("data*/oracle"))
    roots.extend(root.glob("data*/qa"))
    return {str(path.resolve()) for path in roots}


def _authenticate_access_log(
    root: Path,
    access: Mapping[str, Any],
    *,
    config_path: Path,
    questions_path: Path,
    cache: Mapping[str, Any],
    candidate: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    if set(access) != _ACCESS_KEYS:
        raise ValueError("V94 predictor access-log fields changed")
    loaded = _list(access.get("loaded_files"), "loaded files")
    forbidden_roots = _list(access.get("forbidden_roots"), "forbidden roots")
    forbidden_names = _list(
        access.get("forbidden_component_names"), "forbidden component names"
    )
    if (
        access.get("block_forbidden") is not True
        or access.get("passed") is not True
        or access.get("forbidden_accesses") != []
        or forbidden_names != ["oracle"]
        or set(forbidden_roots) != _expected_forbidden_roots(root)
        or loaded != sorted(set(loaded))
        or not all(isinstance(value, str) for value in loaded)
    ):
        raise ValueError("V94 predictor access audit is incomplete or not fail-closed")
    loaded_set = set(loaded)
    for raw in loaded:
        path = Path(raw)
        if not path.is_absolute() or str(path.resolve()) != raw:
            raise ValueError("V94 predictor access log contains a noncanonical path")
        if {part.casefold() for part in path.parts} & {"oracle", "qa"}:
            raise ValueError(f"V94 predictor access log contains protected data: {path}")
    sources = _mapping(config.get("sources"), "sources")
    outputs = _mapping(config.get("outputs"), "outputs")
    mandatory = {
        str(config_path),
        str(questions_path),
        str(cache["manifest_path"]),
        *(str(path) for path in cache["memory_paths"].values()),
        str(candidate["candidate_weights_path"]),
        str(candidate["candidate_metadata_path"]),
        str(_resolve(root, str(sources["runtime_config"]), label="runtime config")),
        str(
            _resolve(root, str(sources["frozen_v85_checkpoint"]), label="V85 parent")
            / "adapter.safetensors"
        ),
        str(
            _resolve(root, str(sources["frozen_v85_checkpoint"]), label="V85 parent")
            / "runtime_metadata.json"
        ),
        str(
            _resolve(
                root,
                str(sources["evaluation_memory_controller"]),
                label="controller",
            )
            / "control.safetensors"
        ),
        str(
            _resolve(
                root,
                str(sources["evaluation_memory_controller"]),
                label="controller",
            )
            / "runtime_metadata.json"
        ),
        str(
            _resolve(root, str(sources["evaluation_probe_bank"]), label="probes")
            / "probes.safetensors"
        ),
        str(
            _resolve(root, str(sources["evaluation_probe_bank"]), label="probes")
            / "runtime_metadata.json"
        ),
    }
    label_path = _resolve(
        root,
        str(sources.get("evaluation_qa_reserved_for_label_scorer")),
        label="reserved validation labels",
    )
    if not mandatory <= loaded_set or str(label_path) in loaded_set:
        missing = sorted(mandatory - loaded_set)
        raise ValueError(
            "V94 predictor access log lacks required reads or includes labels: "
            f"missing={missing}"
        )
    prediction_path = _resolve(
        root, str(outputs.get("evaluation_predictions")), label="predictions"
    )
    if str(prediction_path) in loaded_set:
        # A resume may legitimately read its partial predictions through the
        # JSONL helper, but it must still have the exact completed provenance
        # and access log validated by this module.  No special exemption is
        # needed here; this branch documents that a resume read is permitted.
        pass


def _authenticate_predictions(
    root: Path,
    config: Mapping[str, Any],
    *,
    config_path: Path,
    questions: QuestionManifest,
    questions_path: Path,
    cache: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    outputs = _mapping(config.get("outputs"), "outputs")
    predictions_path = _regular_file(
        root, str(outputs.get("evaluation_predictions")), label="predictions"
    )
    provenance_path = _regular_file(
        root,
        predictions_path.relative_to(root).with_name(
            f"{predictions_path.name}.provenance.json"
        ),
        label="prediction provenance",
    )
    access_path = _regular_file(
        root,
        predictions_path.relative_to(root).with_name(f"{predictions_path.name}.access.json"),
        label="completed prediction access audit",
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    access = json.loads(access_path.read_text(encoding="utf-8"))
    if not isinstance(provenance, dict) or not isinstance(access, dict):
        raise TypeError("V94 prediction provenance/access evidence must be JSON objects")
    provenance_sha = _authenticate_prediction_provenance(
        provenance,
        config_sha256=candidate.get("config_sha256", "")
        or _sha256_file(config_path),
        questions=questions,
        cache=cache,
        candidate=candidate,
    )
    rows = _read_jsonl(predictions_path, label="prediction")
    diagnostics = _authenticate_prediction_rows(
        rows,
        questions=questions,
        cache=cache,
        provenance_sha256=provenance_sha,
    )
    _authenticate_access_log(
        root,
        access,
        config_path=config_path,
        questions_path=questions_path,
        cache=cache,
        candidate=candidate,
        config=config,
    )
    return {
        "predictions_sha256": _sha256_file(predictions_path),
        "prediction_provenance_sha256": provenance_sha,
        "prediction_provenance_file_sha256": _sha256_file(provenance_path),
        "prediction_access_sha256": _sha256_file(access_path),
        "prediction_row_count": len(rows),
        "prediction_keys_sha256": _canonical_sha256(
            sorted((str(row["scene_id"]), str(row["question_id"])) for row in rows)
        ),
        "posthoc_non_preregistered_diagnostics": diagnostics,
    }


def _authenticate_score(
    root: Path,
    config: Mapping[str, Any],
    *,
    questions: QuestionManifest,
    predictions: Mapping[str, Any],
    require_score: bool,
    require_behavior_pass: bool,
) -> dict[str, Any] | None:
    outputs = _mapping(config.get("outputs"), "outputs")
    score_path = _resolve(root, str(outputs.get("evaluation_score")), label="score")
    if not score_path.exists():
        if require_score or require_behavior_pass:
            raise FileNotFoundError("V94 sealed aggregate score is required but absent")
        return None
    if score_path.is_symlink() or not score_path.is_file():
        raise FileNotFoundError("V94 aggregate score is linked or not a regular file")
    score = json.loads(score_path.read_text(encoding="utf-8"))
    if not isinstance(score, dict):
        raise TypeError("V94 score must be one JSON object")
    metrics = _mapping(score.get("metrics"), "score metrics")
    gate_results = dict(_mapping(metrics.get("runtime_candidate_gates"), "score gates"))
    if not gate_results or not all(isinstance(value, bool) for value in gate_results.values()):
        raise ValueError("V94 score has missing or non-boolean gate results")
    behavior_passed = all(gate_results.values())
    sources = _mapping(config.get("sources"), "sources")
    if (
        score.get("artifact") != "v94_label_isolated_same216_score_v1"
        or score.get("schema_version") != 94
        or score.get("row_count") != QUESTION_COUNT
        or score.get("scene_count") != 6
        or score.get("question_manifest_sha256") != questions.manifest_sha256
        or score.get("reference_sha256") != sources.get("evaluation_qa_sha256")
        or score.get("predictions_sha256") != predictions["predictions_sha256"]
        or score.get("prediction_provenance_sha256")
        != predictions["prediction_provenance_sha256"]
        or score.get("labels_opened_only_by_this_scorer") is not True
        or score.get("answers_or_questions_serialized") is not False
        or metrics.get("runtime_candidate_gate_passed") is not behavior_passed
        or metrics.get("automatic_runtime_promotion") is not False
        or score.get("runtime_promotion_authorized") is not False
        or gate_results.get("prefix_hash_invariance") is not True
        or gate_results.get("protected_read_count_zero") is not True
        or score.get("status")
        != (
            "passed_awaiting_separate_leakage_packaging"
            if behavior_passed
            else "measured_gate_not_passed"
        )
    ):
        raise ValueError("V94 aggregate score is not bound to the authenticated predictions")
    if require_behavior_pass and not behavior_passed:
        raise ValueError("V94 aggregate behavior gates did not all pass")
    v94_accuracy = float(_mapping(metrics.get("v94"), "V94 metrics")["accuracy"])
    zero_accuracy = float(
        _mapping(metrics.get("zero_payload"), "zero-payload metrics")["accuracy"]
    )
    atlas_shuffle_accuracy = float(
        _mapping(metrics.get("shuffled_atlas"), "atlas-value-shuffle metrics")[
            "accuracy"
        ]
    )
    if not all(
        math.isfinite(value)
        for value in (v94_accuracy, zero_accuracy, atlas_shuffle_accuracy)
    ):
        raise ValueError("V94 score contains a nonfinite control accuracy")
    return {
        "score_sha256": _sha256_file(score_path),
        "behavior_gate_passed": behavior_passed,
        "behavior_gate_results_sha256": _canonical_sha256(gate_results),
        "status": score["status"],
        "posthoc_non_preregistered_diagnostics": {
            "diagnostic_scope": "posthoc_non_preregistered_not_a_promotion_gate",
            "v94_minus_zero_payload_accuracy": v94_accuracy - zero_accuracy,
            "v94_minus_atlas_value_shuffle_accuracy": v94_accuracy
            - atlas_shuffle_accuracy,
            "atlas_value_shuffle_prediction_change_count": metrics.get(
                "shuffled_atlas_prediction_change_count"
            ),
            "paired_wrong": predictions.get(
                "posthoc_non_preregistered_diagnostics"
            ),
            "paired_wrong_label_accuracy_available": False,
            "paired_wrong_label_accuracy_omission": (
                "the sealed aggregate score did not retain this metric and the "
                "evidence process does not open per-row labels"
            ),
        },
    }


def authenticate_v94_evidence(
    config_path: str | Path = CONFIG,
    *,
    root: str | Path = PROJECT_ROOT,
    require_score: bool = False,
    require_behavior_pass: bool = False,
) -> dict[str, Any]:
    """Authenticate V94 without opening validation labels or running Gemma.

    A score is optional only while scoring has not yet happened.  If a score
    file exists, it is always authenticated.  ``require_score`` closes that
    transitional allowance; ``require_behavior_pass`` additionally makes the
    preregistered behavioral result a required success rather than an
    authenticated negative result.
    """

    project_root = Path(root).expanduser().resolve()
    config, resolved_config = _load_sealed_config(project_root, config_path)
    seal = _authenticate_seal_chain(project_root, config, resolved_config)
    candidate = _authenticate_training_and_candidate(project_root, config, seal)
    questions, questions_path = _authenticate_questions(project_root, config)
    cache = _authenticate_memory_cache(project_root, config)
    compilation = _authenticate_compilation_attestation(
        project_root,
        config,
        config_sha256=seal["config_sha256"],
        cache=cache,
    )
    predictions = _authenticate_predictions(
        project_root,
        config,
        config_path=resolved_config,
        questions=questions,
        questions_path=questions_path,
        cache=cache,
        candidate={**candidate, "config_sha256": seal["config_sha256"]},
    )
    score = _authenticate_score(
        project_root,
        config,
        questions=questions,
        predictions=predictions,
        require_score=require_score,
        require_behavior_pass=require_behavior_pass,
    )
    gates = {
        "sealed_config_prereg_cpu_chain": True,
        "current_sources_match_sealed_digests": True,
        "training_and_fixed_final_candidate_bound": True,
        "sanitized_exact_216_question_keys": True,
        "six_complete_memories_authenticated": True,
        "source_voxel_maps_and_compiler_access_attested": True,
        # This is the independently enforced counterpart of the preregistered
        # every_evaluation_memory_hash_retained_required gate.
        "every_evaluation_memory_hash_retained": cache[
            "all_six_memory_hashes_retained"
        ],
        "all_prediction_rows_bound_to_exact_memory_and_controls": True,
        "prefix_hash_invariance": True,
        "predictor_access_log_complete_and_label_free": True,
        "score_authenticated_if_present": score is not None or not require_score,
    }
    if not all(gates.values()):
        raise RuntimeError(f"V94 evidence gate failed: {gates}")
    evidence = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "passed": True,
        "behavior_score_present": score is not None,
        "behavior_gate_passed": None if score is None else score["behavior_gate_passed"],
        "config_sha256": seal["config_sha256"],
        "preregistration_sha256": seal["preregistration_sha256"],
        "cpu_preflight_sha256": seal["cpu_preflight_sha256"],
        "training_report_sha256": candidate["training_report_sha256"],
        "candidate_weights_sha256": candidate["candidate_weights_sha256"],
        "candidate_metadata_sha256": candidate["candidate_metadata_sha256"],
        "candidate_state_sha256": candidate["candidate_state_sha256"],
        "question_manifest_sha256": questions.manifest_sha256,
        "questions_sha256": questions.questions_sha256,
        "cache_manifest_sha256": cache["manifest_file_sha256"],
        "cache_precompile_attestation_sha256": compilation[
            "pre_attestation_sha256"
        ],
        "cache_postcompile_attestation_sha256": compilation[
            "post_attestation_sha256"
        ],
        "memory_sha256": cache["memory_hashes"],
        **predictions,
        "score": score,
        "gates": gates,
    }
    evidence["bundle_sha256"] = _canonical_sha256(evidence)
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    parser.add_argument("--compile-cache", action="store_true")
    parser.add_argument("--require-score", action="store_true")
    parser.add_argument("--require-behavior-pass", action="store_true")
    args = parser.parse_args(argv)
    if args.compile_cache:
        if args.require_score or args.require_behavior_pass:
            parser.error("--compile-cache cannot be combined with score requirements")
        result = compile_evaluation_memory_cache_with_attestation_v94(
            args.config, root=args.root
        )
    else:
        result = authenticate_v94_evidence(
            args.config,
            root=args.root,
            require_score=args.require_score,
            require_behavior_pass=args.require_behavior_pass,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT",
    "CONFIG",
    "authenticate_v94_evidence",
    "compile_evaluation_memory_cache_with_attestation_v94",
    "main",
]
