"""Package, isolate-smoke, and promote V96 after its deferred-final PASS.

This is a post-evaluation release surface.  Candidate creation is impossible
until both V96's known-development evidence and its sealed deferred-final
evidence authenticate and pass.  Promotion is a separate command and remains
impossible until six child chat processes run with every oracle directory
physically unavailable.

The chat children receive only a standalone runtime YAML, a two-file frozen
ten-bank checkpoint, one two-file numeric V81 scene memory, and one sanitized
numeric voxel map.  They never receive labels, questions from evaluation,
captions, scene graphs, or oracle metadata.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from semantic_3d_chat.chat.runtime_config import (
    effective_runtime_config_sha256,
    load_runtime_config,
)
from semantic_3d_chat.chat.v96_explicit_candidate_authorize import (
    authorize_v96_explicit_candidate,
)
from semantic_3d_chat.chat.v96_explicit_candidate_runtime import (
    BASE_BANKS,
    EXPECTED_BANKS,
    TOTAL_PARAMETER_COUNT,
    V94_BANK,
    V94_STATE_SHA256,
    V95_BANK,
    V96_BANK,
    V96CandidateAuthorization,
)
from semantic_3d_chat.chat.v96_strict_multiscene_runtime import (
    PENDING_DECISION,
    PROMOTED_DECISION,
    RUNTIME_IMPLEMENTATION_FILES,
    runtime_implementation_inventory_v96,
    validate_v96_release_runtime_contract,
)
from semantic_3d_chat.config import PROJECT_ROOT, config_hash
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.evaluation.seal_v96_deferred_final import (
    authenticate_deferred_final_evidence_v96,
)
from semantic_3d_chat.evaluation.strict_direct_release_core import (
    base_bank_order,
    sha256_file,
    validate_runtime_bank_inventory,
)
from semantic_3d_chat.evaluation.v96_deferred_final_common import (
    PRIMARY,
    authenticate_fixed_inputs_before_questions_v96_final,
)
from semantic_3d_chat.evaluation.v96_deferred_final_evaluation import (
    authenticate_preregistration_v96_final,
    authenticate_stage_receipt_v96_final,
    authenticate_unlock_blind_v96_final,
    output_paths_v96_final,
)
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.scene_encoder.map_io import validate_runtime_map_sidecars
from semantic_3d_chat.scene_encoder.v81_scene_memory_artifact import (
    MEMORY_FILENAME,
    METADATA_FILENAME,
    load_v81_scene_memory,
    save_v81_scene_memory,
)
from semantic_3d_chat.training.checkpointing import (
    RUNTIME_METADATA_FILENAME,
    validate_runtime_checkpoint_metadata,
)

SCHEMA_VERSION: Final[int] = 96
ARTIFACT: Final[str] = "gemma4_v96_strict_runtime_release_v1"
SMOKE_ARTIFACT: Final[str] = "gemma4_v96_strict_runtime_smoke_v1"
SCENE_IDS: Final[tuple[str, ...]] = tuple(f"scene_{index:06d}" for index in range(25, 31))
MEMORY_SHAPE: Final[tuple[int, int, int]] = (1, 738, 1536)

PARENT_RUNTIME_CONFIG: Final[Path] = (
    PROJECT_ROOT / "configs/runtime/gemma4_v85_strict_multiscene.yaml"
)
PARENT_CHECKPOINT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v85_strict_runtime_candidate"
)
RUNTIME_CONFIG: Final[Path] = (
    PROJECT_ROOT / "configs/runtime/gemma4_v96_strict_multiscene.yaml"
)
CANDIDATE_CHECKPOINT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v96_strict_runtime_candidate"
)
CANDIDATE_MEMORY_ROOT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v96_strict_runtime_candidate_memories"
)
SMOKE_ROOT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v96_strict_runtime_smoke"
)
SMOKE_REPORT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/gemma4_v96_strict_runtime_smoke.json"
)
RELEASE_CHECKPOINT: Final[Path] = (
    PROJECT_ROOT / "data_gemma4/runtime/checkpoints/gemma4_v96_strict_multiscene_release_v1"
)
RELEASE_MEMORY_ROOT: Final[Path] = (
    PROJECT_ROOT / "data_gemma4/runtime/scene_memories/v96"
)
RUNTIME_MAP_ROOT: Final[Path] = PROJECT_ROOT / "data_gemma4/runtime/maps/v96"
RELEASE_REPORT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/gemma4_v96_strict_runtime_release.json"
)
PRIMARY_POINTER: Final[Path] = PROJECT_ROOT / "configs/runtime/primary.json"
ORACLE_JOURNAL: Final[Path] = SMOKE_ROOT / "oracle_move_journal.json"

_RELEASE_IMPLEMENTATION_FILES: Final[tuple[str, ...]] = (
    "src/semantic_3d_chat/chat/v96_explicit_candidate_authorize.py",
    "src/semantic_3d_chat/evaluation/v96_strict_runtime_release.py",
    "scripts/run_v96_strict_multiscene_demo.sh",
)

_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_SMOKE_QUESTIONS: Final[tuple[str, ...]] = (
    "What is in the room?",
    "What is closest to the camera?",
)
_SAFE_TENSOR_METADATA: Final[dict[str, str]] = {
    "environmental_memory_serialized": "false",
    "questions_or_answers_serialized": "false",
    "oracle_serialized": "false",
}
_CHECKPOINT_FILES: Final[frozenset[str]] = frozenset(
    {"adapter.safetensors", RUNTIME_METADATA_FILENAME}
)
_REQUIRED_SMOKE_GATES: Final[frozenset[str]] = frozenset(
    {
        "known_development_and_deferred_final_gates_passed",
        "all_six_runtime_processes_exit_zero",
        "all_oracle_directories_physically_unavailable",
        "all_oracle_directories_restored",
        "all_six_children_report_oracle_unavailable",
        "all_six_children_use_exact_ten_frozen_banks",
        "all_six_children_report_candidate_mode",
        "all_twelve_questions_return_nonempty_answers",
        "every_scene_prefix_is_invariant",
        "every_scene_prefix_matches_attested_memory",
        "direct_memory_layout_retained_for_every_answer",
        "file_audit_forbidden_read_count_zero",
        "file_audit_protected_read_count_zero",
        "candidate_adapter_bytes_unchanged",
        "candidate_memory_tensor_bytes_unchanged",
        "candidate_map_bytes_unchanged",
        "runtime_implementation_bytes_unchanged",
        "oracle_move_journal_restored_and_bound",
        "no_expectation_channel_in_child_protocol",
        "default_runtime_pointer_unchanged",
    }
)


@dataclass(frozen=True)
class _BridgeSpec:
    root: Path
    artifact: str
    status: str
    schema_version: int
    bank_name: str
    targets: tuple[str, ...]
    rank: int
    alpha: float
    parameter_count: int
    state_sha256: str
    weights_sha256: str
    metadata_sha256: str


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"V96 release requires lowercase SHA-256 for {label}")
    return value


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise TypeError(f"Expected one JSON object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace one recovery journal without following links."""

    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"V96 atomic JSON target must be a physical file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        encoded = json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json_create_once_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Create complete evidence atomically and refuse any existing target."""

    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        encoded = json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _release_implementation_inventory() -> dict[str, Any]:
    """Bind smoke/promotion code in addition to the self-checking runtime."""

    runtime = runtime_implementation_inventory_v96()
    release_files: list[dict[str, Any]] = []
    for relative in _RELEASE_IMPLEMENTATION_FILES:
        path = PROJECT_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"V96 release source must be physical: {path}")
        payload = path.read_bytes()
        release_files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    binding = {
        "runtime_inventory_sha256": runtime["inventory_sha256"],
        "runtime_file_count": len(RUNTIME_IMPLEMENTATION_FILES),
        "release_files": release_files,
    }
    return {
        **binding,
        "inventory_sha256": _canonical_sha256(binding),
    }


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _require_exact_checkpoint_package(root: Path) -> None:
    """Reject links, missing entries, and unaccounted files in a runtime checkpoint."""

    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"V96 runtime checkpoint must be a physical directory: {root}")
    entries = {item.name for item in root.iterdir()}
    if entries != _CHECKPOINT_FILES:
        raise ValueError(
            "V96 runtime checkpoint must contain exactly adapter.safetensors and "
            f"{RUNTIME_METADATA_FILENAME}: {sorted(entries)}"
        )
    for name in _CHECKPOINT_FILES:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"V96 runtime checkpoint entry must be physical: {path}")


def _require_exact_scene_bundle(root: Path, *, label: str) -> None:
    """Require exactly the six opaque physical scene directories at one bundle root."""

    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"V96 {label} must be a physical directory: {root}")
    entries = {item.name for item in root.iterdir()}
    if entries != set(SCENE_IDS):
        raise ValueError(f"V96 {label} is not exactly six scenes: {sorted(entries)}")
    for scene_id in SCENE_IDS:
        scene_root = root / scene_id
        if scene_root.is_symlink() or not scene_root.is_dir():
            raise ValueError(f"V96 {label} scene root must be physical: {scene_root}")


def _default_runtime_snapshot() -> dict[str, Any]:
    """Capture the operator default without importing or changing it."""

    if PRIMARY_POINTER.is_symlink():
        raise ValueError("V96 release refuses a symbolic-link primary runtime pointer")
    pointer_exists = PRIMARY_POINTER.is_file()
    if PRIMARY_POINTER.exists() and not pointer_exists:
        raise ValueError("V96 primary runtime pointer is not a regular file")
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    if "\ndemo:\n" not in makefile:
        raise ValueError("V96 release could not identify the default demo recipe")
    demo_body = makefile.split("\ndemo:\n", 1)[1].split("\n\n", 1)[0]
    return {
        "primary_pointer_exists": pointer_exists,
        "primary_pointer_sha256": (
            sha256_file(PRIMARY_POINTER) if pointer_exists else None
        ),
        "default_demo_uses_v89": "run_v89_strict_scene1_demo.sh" in demo_body,
        "default_demo_uses_v96": "run_v96_strict_multiscene_demo.sh" in demo_body,
    }


def authenticate_v96_release_gate() -> dict[str, Any]:
    """Require mutually bound known-development and deferred-final PASS evidence."""

    authorization = authorize_v96_explicit_candidate()
    final = authenticate_deferred_final_evidence_v96()
    final_paths = output_paths_v96_final()
    score = _read_json(final_paths["final_score"])
    gates = score.get("gate_results")
    if (
        final.get("artifact") != "gemma4_v96_deferred_final_evidence_v1"
        or final.get("schema_version") != SCHEMA_VERSION
        or final.get("status") != "passed_deferred_final_not_runtime_promoted"
        or final.get("deferred_final_gate_passed") is not True
        or final.get("candidate_fingerprint_sha256")
        != authorization.candidate_fingerprint_sha256
        or final.get("candidate_attestation_file_sha256")
        != authorization.candidate_attestation_file_sha256
        or final.get("candidate_attestation_identity_sha256")
        != authorization.candidate_attestation_identity_sha256
        or final.get("v1_implementation_seal_sha256")
        != authorization.v1_implementation_seal_sha256
        or final.get("v2_implementation_seal_sha256")
        != authorization.v2_implementation_seal_sha256
        or final.get("question_label_isolation_proven") is not True
        or final.get("prefix_hash_invariant") is not True
        or final.get("protected_read_count") != 0
        or final.get("row_level_content_serialized") is not False
        or final.get("runtime_packaging_requires_separate_leakage_gate") is not True
        or final.get("runtime_promotion_authorized") is not False
        or final.get("automatic_runtime_promotion") is not False
        or final.get("authenticated") is not True
        or score.get("artifact") != "gemma4_v96_deferred_final_gate_v1"
        or score.get("status") != "passed_deferred_final_not_runtime_promoted"
        or score.get("passed") is not True
        or score.get("candidate_fingerprint_sha256")
        != authorization.candidate_fingerprint_sha256
        or score.get("candidate_attestation_file_sha256")
        != authorization.candidate_attestation_file_sha256
        or score.get("candidate_attestation_identity_sha256")
        != authorization.candidate_attestation_identity_sha256
        or score.get("v1_implementation_seal_sha256")
        != authorization.v1_implementation_seal_sha256
        or score.get("v2_implementation_seal_sha256")
        != authorization.v2_implementation_seal_sha256
        or score.get("eligible_for_separate_runtime_leakage_evaluation") is not True
        or score.get("runtime_promotion_authorized") is not False
        or not isinstance(gates, Mapping)
        or not gates
        or any(value is not True for value in gates.values())
        or final.get("final_score_sha256") != sha256_file(final_paths["final_score"])
        or final.get("evidence_file_sha256") != sha256_file(final_paths["evidence"])
    ):
        raise ValueError("V96 held-out release gate is incomplete or failed")
    return {
        "authorization": authorization,
        "final": dict(final),
        "score": score,
        "final_score_sha256": str(final["final_score_sha256"]),
        "deferred_final_evidence_sha256": str(final["evidence_file_sha256"]),
        "gate_results_sha256": str(final["gate_results_sha256"]),
        "known_development_gate_passed": True,
        "deferred_final_gate_passed": True,
        "eligible_for_separate_runtime_leakage_evaluation": True,
        "runtime_promotion_authorized": False,
    }


def _bridge_specs(authorization: V96CandidateAuthorization) -> tuple[_BridgeSpec, ...]:
    authorization.validate()
    return (
        _BridgeSpec(
            root=Path(authorization.v94_bridge_path),
            artifact="gemma4_v94_strict_multiscene_full40_fixed_final_v1",
            status="fixed_final_awaiting_preregistered_acceptance_gates",
            schema_version=94,
            bank_name=V94_BANK,
            targets=("model.language_model.layers.34.mlp.gate_proj",),
            rank=8,
            alpha=16.0,
            parameter_count=110_592,
            state_sha256=V94_STATE_SHA256,
            weights_sha256=authorization.v94_weights_sha256,
            metadata_sha256=authorization.v94_metadata_sha256,
        ),
        _BridgeSpec(
            root=Path(authorization.v95_bridge_path),
            artifact="gemma4_v95_strict_causal_successor_fixed_final_v1",
            status="fixed_final_awaiting_known_development_gate",
            schema_version=95,
            bank_name=V95_BANK,
            targets=(
                "model.language_model.layers.9.self_attn.k_proj",
                "model.language_model.layers.9.self_attn.v_proj",
                "model.language_model.layers.34.mlp.up_proj",
            ),
            rank=8,
            alpha=16.0,
            parameter_count=143_360,
            state_sha256=authorization.v95_state_sha256,
            weights_sha256=authorization.v95_weights_sha256,
            metadata_sha256=authorization.v95_metadata_sha256,
        ),
        _BridgeSpec(
            root=Path(authorization.v96_candidate_path),
            artifact="gemma4_v96_atomic_pair_repair_fixed_final_v1",
            status="fixed_final_awaiting_known_development_gate",
            schema_version=96,
            bank_name=V96_BANK,
            targets=("model.language_model.layers.9.self_attn.q_proj",),
            rank=8,
            alpha=16.0,
            parameter_count=45_056,
            state_sha256=authorization.v96_state_sha256,
            weights_sha256=authorization.v96_weights_sha256,
            metadata_sha256=authorization.v96_metadata_file_sha256,
        ),
    )


def _load_bridge(spec: _BridgeSpec) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    root = spec.root.resolve()
    if root.is_symlink() or not root.is_dir() or {
        item.name for item in root.iterdir()
    } != {"bridge.safetensors", RUNTIME_METADATA_FILENAME}:
        raise ValueError(f"V96 release bridge inventory changed: {spec.bank_name}")
    weights = root / "bridge.safetensors"
    metadata_path = root / RUNTIME_METADATA_FILENAME
    if any(path.is_symlink() or not path.is_file() for path in (weights, metadata_path)):
        raise ValueError(f"V96 release bridge files must be physical: {spec.bank_name}")
    if (
        sha256_file(weights) != spec.weights_sha256
        or sha256_file(metadata_path) != spec.metadata_sha256
    ):
        raise ValueError(f"V96 release bridge bytes changed: {spec.bank_name}")
    metadata = _read_json(metadata_path)
    target_matches = (
        metadata.get("target_module") == spec.targets[0]
        if spec.schema_version == 94
        else metadata.get("target_modules") == list(spec.targets)
    )
    score_field = "evaluation_scored" if spec.schema_version == 94 else "known_development_scored"
    if (
        metadata.get("artifact") != spec.artifact
        or metadata.get("schema_version") != spec.schema_version
        or metadata.get("status") != spec.status
        or metadata.get("bank_name") != spec.bank_name
        or not target_matches
        or metadata.get("rank") != spec.rank
        or float(metadata.get("alpha", -1.0)) != spec.alpha
        or float(metadata.get("dropout", -1.0)) != 0.0
        or metadata.get("parameter_count") != spec.parameter_count
        or metadata.get("state_sha256") != spec.state_sha256
        or metadata.get("weights_sha256") != spec.weights_sha256
        or metadata.get("environmental_memory_serialized") is not False
        or metadata.get("questions_or_answers_serialized") is not False
        or metadata.get("oracle_serialized") is not False
        or metadata.get(score_field) is not False
        or metadata.get("runtime_promotion_authorized") is not False
        or (spec.schema_version >= 95 and metadata.get("deferred_final_generated") is not False)
    ):
        raise ValueError(f"V96 release bridge metadata changed: {spec.bank_name}")
    with safe_open(str(weights), framework="pt", device="cpu") as archive:
        tensor_metadata = archive.metadata()
        if not isinstance(tensor_metadata, Mapping) or any(
            tensor_metadata.get(key) != value
            for key, value in _SAFE_TENSOR_METADATA.items()
        ):
            raise ValueError(f"V96 release bridge tensor metadata is unsafe: {spec.bank_name}")
        raw = {key: archive.get_tensor(key) for key in list(archive.keys())}
    if spec.schema_version == 94:
        if set(raw) != {"lora_a", "lora_b"}:
            raise ValueError("V96 release V94 tensor inventory changed")
        state = {f"adapters.0.{key}": value for key, value in raw.items()}
    else:
        expected = {
            f"adapters.{index}.{suffix}"
            for index in range(len(spec.targets))
            for suffix in ("lora_a", "lora_b")
        }
        if set(raw) != expected:
            raise ValueError(f"V96 release tensor inventory changed: {spec.bank_name}")
        state = raw
    counts: dict[str, int] = {}
    for index, target in enumerate(spec.targets):
        a = state[f"adapters.{index}.lora_a"]
        b = state[f"adapters.{index}.lora_b"]
        if (
            a.ndim != 2
            or b.ndim != 2
            or a.shape[0] != spec.rank
            or b.shape[1] != spec.rank
            or a.dtype != torch.float32
            or b.dtype != torch.float32
            or not bool(torch.isfinite(a).all())
            or not bool(torch.isfinite(b).all())
        ):
            raise ValueError(f"V96 release bridge tensor changed: {spec.bank_name}")
        counts[target] = a.numel() + b.numel()
    if sum(counts.values()) != spec.parameter_count or tensor_state_sha256(state) != spec.state_sha256:
        raise ValueError(f"V96 release bridge state changed: {spec.bank_name}")
    return {name: value.detach().cpu().contiguous() for name, value in state.items()}, counts


def _parent_metadata(authorization: V96CandidateAuthorization) -> dict[str, Any]:
    if (
        Path(authorization.runtime_config_path).resolve()
        != PARENT_RUNTIME_CONFIG.resolve()
        or Path(authorization.v85_checkpoint_path).resolve()
        != PARENT_CHECKPOINT.resolve()
        or sha256_file(PARENT_RUNTIME_CONFIG)
        != authorization.runtime_config_file_sha256
        or sha256_file(PARENT_CHECKPOINT / "adapter.safetensors")
        != authorization.v85_adapter_sha256
        or sha256_file(PARENT_CHECKPOINT / RUNTIME_METADATA_FILENAME)
        != authorization.v85_metadata_sha256
    ):
        raise ValueError("V96 release parent bytes changed")
    metadata = _read_json(PARENT_CHECKPOINT / RUNTIME_METADATA_FILENAME)
    if base_bank_order(metadata) != BASE_BANKS:
        raise ValueError("V96 release parent bank order changed")
    return metadata


def _states_from_parent(metadata: Mapping[str, Any]) -> dict[str, str]:
    raw = metadata.get("lora_bank_state_sha256")
    if not isinstance(raw, Mapping) or set(raw) != set(BASE_BANKS):
        raise ValueError("V96 release parent state inventory changed")
    return {name: _require_sha256(raw[name], f"parent bank {name}") for name in BASE_BANKS}


def build_runtime_config_payload(gate: Mapping[str, Any]) -> dict[str, Any]:
    authorization = gate.get("authorization")
    if not isinstance(authorization, V96CandidateAuthorization):
        raise TypeError("V96 release gate lacks candidate authorization")
    parent_metadata = _parent_metadata(authorization)
    states = _states_from_parent(parent_metadata)
    parent = load_runtime_config(PARENT_RUNTIME_CONFIG)
    parent.pop("_config_path", None)
    parent.pop("_runtime_safe_config", None)
    paths = parent.get("paths")
    if not isinstance(paths, dict):
        raise TypeError("V96 release parent runtime paths are malformed")
    paths["maps_root"] = _relative(RUNTIME_MAP_ROOT)
    configured = parent.get("language", {}).get("lora_banks")
    if not isinstance(configured, dict) or tuple(configured) != BASE_BANKS:
        raise ValueError("V96 release parent runtime bank order changed")
    for name in BASE_BANKS:
        configured[name]["expected_initial_state_sha256"] = states[name]
    for spec in _bridge_specs(authorization):
        configured[spec.bank_name] = {
            "trainable": False,
            "rank": spec.rank,
            "alpha": spec.alpha,
            "dropout": 0.0,
            "initialization_algorithm": "checkpoint_overwrite",
            "initialization_seed": None,
            "expected_initial_state_sha256": spec.state_sha256,
            "target_modules": list(spec.targets),
        }
    if tuple(configured) != EXPECTED_BANKS or any(
        row.get("trainable") is not False for row in configured.values()
    ):
        raise RuntimeError("V96 runtime config lost exact frozen ten-bank order")
    return parent


def materialize_runtime_config(gate: Mapping[str, Any]) -> dict[str, Any]:
    payload = build_runtime_config_payload(gate)
    encoded = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
    if RUNTIME_CONFIG.exists():
        if RUNTIME_CONFIG.is_symlink() or RUNTIME_CONFIG.read_text(encoding="utf-8") != encoded:
            raise ValueError("Existing V96 runtime config differs from authenticated gate")
    else:
        RUNTIME_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        with RUNTIME_CONFIG.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    return load_runtime_config(RUNTIME_CONFIG)


def _composed_adapter(
    authorization: V96CandidateAuthorization,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, int]]]:
    _parent_metadata(authorization)
    parent = load_file(str(PARENT_CHECKPOINT / "adapter.safetensors"), device="cpu")
    result = {name: value.detach().cpu().contiguous() for name, value in parent.items()}
    counts: dict[str, dict[str, int]] = {}
    for spec in _bridge_specs(authorization):
        state, bank_counts = _load_bridge(spec)
        counts[spec.bank_name] = bank_counts
        for name, value in state.items():
            key = f"lora_banks.{spec.bank_name}.{name}"
            if key in result:
                raise ValueError(f"V96 release duplicate adapter key: {key}")
            result[key] = value
    retained = {name: result[name] for name in parent}
    if tensor_state_sha256(retained) != tensor_state_sha256(parent):
        raise RuntimeError("V96 release changed a parent adapter tensor")
    return result, counts


def build_runtime_metadata(
    gate: Mapping[str, Any], *, promotion: str, smoke_report_sha256: str | None
) -> dict[str, Any]:
    if promotion not in {PENDING_DECISION, PROMOTED_DECISION}:
        raise ValueError("Unknown V96 runtime promotion state")
    promoted = promotion == PROMOTED_DECISION
    if promoted != (smoke_report_sha256 is not None):
        raise ValueError("V96 promotion and smoke binding disagree")
    if smoke_report_sha256 is not None:
        _require_sha256(smoke_report_sha256, "smoke report")
    authorization = gate.get("authorization")
    final = gate.get("final")
    if not isinstance(authorization, V96CandidateAuthorization) or not isinstance(final, Mapping):
        raise TypeError("V96 release gate payload is malformed")
    parent = copy.deepcopy(_parent_metadata(authorization))
    states = _states_from_parent(parent)
    tensors, added_counts = _composed_adapter(authorization)
    lora = parent["lora"]
    banks = list(lora["banks"])
    for row in banks:
        row["expected_initial_state_sha256"] = states[str(row["name"])]
    modules = dict(parent["lora_bank_wrapped_modules"])
    counts = dict(parent["lora_bank_parameter_counts"])
    for spec in _bridge_specs(authorization):
        states[spec.bank_name] = spec.state_sha256
        modules[spec.bank_name] = list(spec.targets)
        counts[spec.bank_name] = added_counts[spec.bank_name]
        banks.append(
            {
                "name": spec.bank_name,
                "trainable": False,
                "rank": spec.rank,
                "alpha": spec.alpha,
                "dropout": 0.0,
                "target_modules": list(spec.targets),
                "initialization_algorithm": "checkpoint_overwrite",
                "initialization_seed": None,
                "expected_initial_state_sha256": spec.state_sha256,
                "adapter_parameter_count": spec.parameter_count,
            }
        )
    parent["lora"] = {
        "schema_version": 2,
        "enabled": True,
        "banks": banks,
        "adapter_parameter_count": TOTAL_PARAMETER_COUNT,
        "trainable_adapter_parameter_count": 0,
    }
    parent["lora_bank_state_sha256"] = states
    parent["lora_bank_wrapped_modules"] = modules
    parent["lora_bank_parameter_counts"] = counts
    parent["lora_parameter_count"] = TOTAL_PARAMETER_COUNT
    parent["lora_trainable_parameter_count"] = 0
    config = build_runtime_config_payload(gate)
    parent["config_hash"] = config_hash(config)
    runtime_implementation = runtime_implementation_inventory_v96()
    provenance = copy.deepcopy(dict(parent.get("initialization_provenance", {})))
    provenance["v96_strict_runtime_release"] = {
        "schema_version": SCHEMA_VERSION,
        "candidate_fingerprint_sha256": authorization.candidate_fingerprint_sha256,
        "candidate_attestation_file_sha256": authorization.candidate_attestation_file_sha256,
        "candidate_attestation_identity_sha256": (
            authorization.candidate_attestation_identity_sha256
        ),
        "v1_implementation_seal_sha256": authorization.v1_implementation_seal_sha256,
        "v2_implementation_seal_sha256": authorization.v2_implementation_seal_sha256,
        "deferred_final_evidence_sha256": gate["deferred_final_evidence_sha256"],
        "deferred_final_score_sha256": gate["final_score_sha256"],
        "deferred_final_gate_results_sha256": gate["gate_results_sha256"],
        "runtime_implementation_inventory_sha256": runtime_implementation[
            "inventory_sha256"
        ],
        "known_development_gate_passed": True,
        "deferred_final_gate_passed": True,
        "deferred_final_evidence_authenticated": True,
        "supervision_isolation_proven": final["question_label_isolation_proven"],
        "prefix_hash_invariant_in_evaluation": final["prefix_hash_invariant"],
        "v94_state_sha256": V94_STATE_SHA256,
        "v95_state_sha256": authorization.v95_state_sha256,
        "v96_state_sha256": authorization.v96_state_sha256,
        "promotion_decision": promotion,
        "runtime_promotion_authorized": promoted,
        "smoke_report_sha256": smoke_report_sha256,
        "held_out_generalization_claim": True,
        "environmental_text_inputs": [],
    }
    parent["initialization_provenance"] = provenance
    validate_runtime_checkpoint_metadata(parent)
    validate_runtime_bank_inventory(
        runtime_config=config,
        checkpoint_metadata=parent,
        expected_bank_order=EXPECTED_BANKS,
        expected_states=states,
    )
    validate_v96_release_runtime_contract(
        runtime_config=config,
        checkpoint_metadata=parent,
    )
    if len(tensors) <= 181 or len(tensors) != 191:
        raise RuntimeError("V96 release adapter tensor count changed")
    return parent


def _atomic_checkpoint(
    destination: Path,
    *,
    metadata: Mapping[str, Any],
    authorization: V96CandidateAuthorization,
    source_adapter: Path | None = None,
) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        if source_adapter is None:
            tensors, _counts = _composed_adapter(authorization)
            save_file(tensors, str(temporary / "adapter.safetensors"))
        else:
            if source_adapter.is_symlink() or not source_adapter.is_file():
                raise FileNotFoundError(source_adapter)
            shutil.copyfile(source_adapter, temporary / "adapter.safetensors")
        _write_json(temporary / RUNTIME_METADATA_FILENAME, metadata)
        if {item.name for item in temporary.iterdir()} != {
            "adapter.safetensors",
            RUNTIME_METADATA_FILENAME,
        }:
            raise RuntimeError("V96 checkpoint is not an exact two-file package")
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    fingerprint, files = checkpoint_fingerprint(destination)
    return {
        "checkpoint_sha256": fingerprint,
        "checkpoint_files": files,
        "adapter_sha256": sha256_file(destination / "adapter.safetensors"),
        "runtime_metadata_sha256": sha256_file(
            destination / RUNTIME_METADATA_FILENAME
        ),
        "exact_two_file_checkpoint": True,
    }


def _package_memories(
    destination: Path,
    *,
    checkpoint_sha256: str,
    runtime_config_sha256: str,
) -> dict[str, dict[str, Any]]:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    fixed = authenticate_fixed_inputs_before_questions_v96_final()
    if tuple(fixed.memory_paths) != SCENE_IDS:
        raise ValueError("V96 release source memory inventory changed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    summaries: dict[str, dict[str, Any]] = {}
    try:
        for scene_id in SCENE_IDS:
            source = fixed.memory_paths[scene_id]
            source_metadata = _read_json(source / METADATA_FILENAME)
            memory = fixed.memories[PRIMARY][scene_id]
            metadata = save_v81_scene_memory(
                temporary / scene_id,
                memory,
                scene_id=scene_id,
                source_base_checkpoint_sha256=checkpoint_sha256,
                runtime_config_sha256=runtime_config_sha256,
                source_control_checkpoint_sha256=source_metadata[
                    "source_control_checkpoint_sha256"
                ],
                source_probe_tensor_sha256=source_metadata["source_probe_tensor_sha256"],
            )
            if (
                metadata["canonical_prefix_sha256"]
                != fixed.memory_hashes[PRIMARY][scene_id]
                or metadata["tensor_file_sha256"]
                != sha256_file(source / MEMORY_FILENAME)
            ):
                raise RuntimeError(f"V96 release changed numeric memory: {scene_id}")
            summaries[scene_id] = {
                "source_tensor_file_sha256": sha256_file(source / MEMORY_FILENAME),
                "packaged_tensor_file_sha256": metadata["tensor_file_sha256"],
                "canonical_prefix_sha256": metadata["canonical_prefix_sha256"],
                "tensor_bytes_reused_exactly": True,
                "environmental_text_inputs": [],
            }
        if tuple(sorted(item.name for item in temporary.iterdir())) != SCENE_IDS:
            raise RuntimeError("V96 release memory bundle is not exactly six scenes")
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summaries


def _authenticated_source_maps() -> dict[str, dict[str, Any]]:
    """Authenticate only numeric map outputs from the sealed maps receipt."""

    preregistration = authenticate_preregistration_v96_final()
    unlock = authenticate_unlock_blind_v96_final(preregistration)
    receipt = authenticate_stage_receipt_v96_final(
        preregistration,
        unlock,
        "maps",
        hash_outputs=True,
    )
    outputs = receipt.get("output_sha256")
    if not isinstance(outputs, Mapping):
        raise TypeError("V96 authenticated maps receipt has no output inventory")
    result: dict[str, dict[str, Any]] = {}
    for scene_id in SCENE_IDS:
        relative = f"data_gemma4/maps/{scene_id}/voxel_map.npz"
        source = PROJECT_ROOT / relative
        expected = _require_sha256(outputs.get(relative), f"source map {scene_id}")
        if source.is_symlink() or not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"V96 authenticated numeric map changed: {scene_id}")
        validate_runtime_map_sidecars(source)
        result[scene_id] = {
            "source_path": relative,
            "source_sha256": expected,
            "size_bytes": source.stat().st_size,
            "maps_receipt_sha256": receipt["receipt_file_sha256"],
        }
    return result


def _package_runtime_maps(destination: Path) -> dict[str, dict[str, Any]]:
    """Copy six sanitized numeric maps into the standalone runtime surface."""

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    sources = _authenticated_source_maps()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    summaries: dict[str, dict[str, Any]] = {}
    try:
        for scene_id in SCENE_IDS:
            scene_root = temporary / scene_id
            scene_root.mkdir()
            target = scene_root / "voxel_map.npz"
            source = PROJECT_ROOT / sources[scene_id]["source_path"]
            shutil.copyfile(source, target)
            packaged_sha = sha256_file(target)
            if packaged_sha != sources[scene_id]["source_sha256"]:
                raise RuntimeError(f"V96 runtime map bytes changed: {scene_id}")
            validate_runtime_map_sidecars(target)
            summaries[scene_id] = {
                **sources[scene_id],
                "runtime_path": _relative(destination / scene_id / "voxel_map.npz"),
                "packaged_sha256": packaged_sha,
                "numeric_bytes_reused_exactly": True,
                "environmental_text_inputs": [],
            }
        if tuple(sorted(item.name for item in temporary.iterdir())) != SCENE_IDS:
            raise RuntimeError("V96 runtime map bundle is not exactly six scenes")
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summaries


def _verify_runtime_maps() -> dict[str, dict[str, Any]]:
    sources = _authenticated_source_maps()
    if RUNTIME_MAP_ROOT.is_symlink() or not RUNTIME_MAP_ROOT.is_dir():
        raise FileNotFoundError("V96 standalone runtime map root is unavailable")
    if tuple(sorted(item.name for item in RUNTIME_MAP_ROOT.iterdir())) != SCENE_IDS:
        raise ValueError("V96 runtime map root is not exactly six scenes")
    result: dict[str, dict[str, Any]] = {}
    for scene_id in SCENE_IDS:
        scene_root = RUNTIME_MAP_ROOT / scene_id
        target = scene_root / "voxel_map.npz"
        if (
            scene_root.is_symlink()
            or not scene_root.is_dir()
            or {item.name for item in scene_root.iterdir()} != {"voxel_map.npz"}
            or target.is_symlink()
            or not target.is_file()
        ):
            raise ValueError(f"V96 runtime map inventory changed: {scene_id}")
        packaged_sha = sha256_file(target)
        if packaged_sha != sources[scene_id]["source_sha256"]:
            raise ValueError(f"V96 runtime map bytes changed: {scene_id}")
        validate_runtime_map_sidecars(target)
        result[scene_id] = {
            **sources[scene_id],
            "runtime_path": _relative(target),
            "packaged_sha256": packaged_sha,
            "numeric_bytes_reused_exactly": True,
            "environmental_text_inputs": [],
        }
    return result


def prepare_candidate() -> dict[str, Any]:
    gate = authenticate_v96_release_gate()
    if any(
        path.exists() or path.is_symlink()
        for path in (CANDIDATE_CHECKPOINT, CANDIDATE_MEMORY_ROOT, RUNTIME_MAP_ROOT)
    ):
        raise FileExistsError("V96 runtime candidate destination already exists")
    config = materialize_runtime_config(gate)
    runtime_sha = effective_runtime_config_sha256(config)
    authorization = gate["authorization"]
    metadata = build_runtime_metadata(
        gate, promotion=PENDING_DECISION, smoke_report_sha256=None
    )
    checkpoint = _atomic_checkpoint(
        CANDIDATE_CHECKPOINT,
        metadata=metadata,
        authorization=authorization,
    )
    maps = _package_runtime_maps(RUNTIME_MAP_ROOT)
    memories = _package_memories(
        CANDIDATE_MEMORY_ROOT,
        checkpoint_sha256=checkpoint["checkpoint_sha256"],
        runtime_config_sha256=runtime_sha,
    )
    return {
        "phase": "v96_strict_runtime_candidate_prepared",
        "checkpoint": checkpoint,
        "runtime_maps": maps,
        "scene_memories": memories,
        "scene_count": len(SCENE_IDS),
        "runtime_config_sha256": runtime_sha,
        "default_runtime_pointer_modified": False,
        "passed": True,
    }


def verify_candidate() -> dict[str, Any]:
    gate = authenticate_v96_release_gate()
    authorization = gate["authorization"]
    _require_exact_checkpoint_package(CANDIDATE_CHECKPOINT)
    _require_exact_scene_bundle(CANDIDATE_MEMORY_ROOT, label="candidate memory root")
    config = load_runtime_config(RUNTIME_CONFIG)
    runtime_sha = effective_runtime_config_sha256(config)
    maps = _verify_runtime_maps()
    metadata = _read_json(CANDIDATE_CHECKPOINT / RUNTIME_METADATA_FILENAME)
    expected_metadata = build_runtime_metadata(
        gate, promotion=PENDING_DECISION, smoke_report_sha256=None
    )
    if metadata != expected_metadata:
        raise ValueError("V96 candidate runtime metadata changed")
    fingerprint, files = checkpoint_fingerprint(CANDIDATE_CHECKPOINT)
    actual = load_file(str(CANDIDATE_CHECKPOINT / "adapter.safetensors"), device="cpu")
    expected, _counts = _composed_adapter(authorization)
    memory: dict[str, dict[str, Any]] = {}
    for scene_id in SCENE_IDS:
        loaded = load_v81_scene_memory(
            CANDIDATE_MEMORY_ROOT / scene_id,
            expected_scene_id=scene_id,
            expected_base_checkpoint_sha256=fingerprint,
            expected_runtime_config_sha256=runtime_sha,
            expected_model_device="cpu",
        )
        memory[scene_id] = {
            "tensor_file_sha256": sha256_file(
                CANDIDATE_MEMORY_ROOT / scene_id / MEMORY_FILENAME
            ),
            "metadata_sha256": sha256_file(
                CANDIDATE_MEMORY_ROOT / scene_id / METADATA_FILENAME
            ),
            "canonical_prefix_sha256": loaded.metadata["canonical_prefix_sha256"],
        }
    checks = {
        "exact_two_file_checkpoint": {row["path"] for row in files}
        == {"adapter.safetensors", RUNTIME_METADATA_FILENAME},
        "exact_ten_bank_order": tuple(
            row["name"] for row in metadata["lora"]["banks"]
        )
        == EXPECTED_BANKS,
        "exact_ten_added_bridge_tensors": len(actual) - 181 == 10,
        "exact_tensor_inventory": set(actual) == set(expected),
        "all_tensor_values_equal": set(actual) == set(expected)
        and all(torch.equal(actual[name], expected[name]) for name in actual),
        "all_ten_banks_frozen": metadata["lora"]["adapter_parameter_count"]
        == TOTAL_PARAMETER_COUNT
        and metadata["lora"]["trainable_adapter_parameter_count"] == 0,
        "exact_six_v81_memories": tuple(sorted(memory)) == SCENE_IDS,
        "exact_six_sanitized_numeric_maps": tuple(sorted(maps)) == SCENE_IDS
        and config["paths"]["maps_root"] == _relative(RUNTIME_MAP_ROOT),
        "default_runtime_pointer_unchanged": _default_runtime_snapshot()[
            "default_demo_uses_v89"
        ]
        is True
        and _default_runtime_snapshot()["default_demo_uses_v96"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V96 strict candidate verification failed: {checks}")
    return {
        "phase": "v96_strict_runtime_candidate_verified",
        "checkpoint_sha256": fingerprint,
        "adapter_sha256": sha256_file(CANDIDATE_CHECKPOINT / "adapter.safetensors"),
        "runtime_metadata_sha256": sha256_file(
            CANDIDATE_CHECKPOINT / RUNTIME_METADATA_FILENAME
        ),
        "runtime_config_sha256": runtime_sha,
        "runtime_maps": maps,
        "scene_memories": memory,
        "checks": checks,
        "passed": True,
    }


def _smoke_command(scene_id: str, *, audit_path: Path, chat_path: Path) -> list[str]:
    python = PROJECT_ROOT / ".venv-gemma4/bin/python"
    command = [
        str(python),
        "-m",
        "semantic_3d_chat.chat.v96_strict_multiscene_cli",
        "--config",
        str(RUNTIME_CONFIG),
        "--scene",
        scene_id,
        "--base-checkpoint",
        str(CANDIDATE_CHECKPOINT),
        "--scene-memory",
        str(CANDIDATE_MEMORY_ROOT / scene_id),
        "--audit-log",
        str(audit_path),
        "--chat-log",
        str(chat_path),
        "--allow-candidate",
    ]
    for question in _SMOKE_QUESTIONS:
        command.extend(("--question", question))
    return command


def _oracle_directories() -> tuple[Path, ...]:
    candidates = tuple(PROJECT_ROOT.glob("data*/oracle"))
    unsafe = [
        path for path in candidates if path.is_symlink() or (path.exists() and not path.is_dir())
    ]
    if unsafe:
        raise ValueError(f"V96 oracle roots must be physical directories: {unsafe}")
    return tuple(sorted({path.resolve() for path in candidates if path.is_dir()}))


def _oracle_journal_payload(moves: Sequence[tuple[Path, Path]], status: str) -> dict[str, Any]:
    rows = [
        {"source": _relative(source), "hidden": _relative(hidden)}
        for source, hidden in moves
    ]
    return {
        "artifact": "gemma4_v96_oracle_move_recovery_journal_v1",
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "moves": rows,
        "moves_sha256": _canonical_sha256(rows),
    }


def _validated_oracle_journal_moves(
    journal: Mapping[str, Any],
) -> tuple[tuple[Path, Path], ...]:
    rows = journal.get("moves")
    if (
        journal.get("artifact") != "gemma4_v96_oracle_move_recovery_journal_v1"
        or journal.get("schema_version") != SCHEMA_VERSION
        or not isinstance(journal.get("status"), str)
        or not isinstance(rows, list)
        or not rows
        or journal.get("moves_sha256") != _canonical_sha256(rows)
    ):
        raise ValueError("V96 oracle recovery journal is malformed")
    moves: list[tuple[Path, Path]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("V96 oracle recovery journal row is malformed")
        source_raw = row.get("source")
        hidden_raw = row.get("hidden")
        if not isinstance(source_raw, str) or not isinstance(hidden_raw, str):
            raise TypeError("V96 oracle recovery paths are malformed")
        source = Path(os.path.abspath(PROJECT_ROOT / source_raw))
        hidden = Path(os.path.abspath(PROJECT_ROOT / hidden_raw))
        if (
            source.parent.parent != PROJECT_ROOT.resolve()
            or not source.parent.name.startswith("data")
            or source.name != "oracle"
            or hidden.parent != source.parent
            or re.fullmatch(r"\.oracle-unavailable-v96-[0-9]+-[0-9]+", hidden.name)
            is None
        ):
            raise ValueError("V96 oracle recovery journal path escaped its exact scope")
        moves.append((source, hidden))
    if len(set(moves)) != len(moves):
        raise ValueError("V96 oracle recovery journal has duplicate moves")
    return tuple(moves)


def _restore_oracle_moves(
    moves: Sequence[tuple[Path, Path]], *, status: str
) -> dict[str, Any]:
    for source, hidden in reversed(tuple(moves)):
        source_present = source.exists() or source.is_symlink()
        hidden_present = hidden.exists() or hidden.is_symlink()
        if source_present and hidden_present:
            raise RuntimeError(f"V96 oracle recovery found both paths: {source}")
        if source_present:
            if source.is_symlink() or not source.is_dir():
                raise ValueError(f"V96 oracle recovery source is unsafe: {source}")
            continue
        if not hidden_present or hidden.is_symlink() or not hidden.is_dir():
            raise RuntimeError(f"V96 oracle recovery cannot locate moved directory: {source}")
        os.rename(hidden, source)
    if not all(
        source.is_dir() and not source.is_symlink() and not hidden.exists()
        for source, hidden in moves
    ):
        raise RuntimeError("V96 oracle recovery did not restore every directory")
    payload = _oracle_journal_payload(moves, status)
    _write_json_atomic(ORACLE_JOURNAL, payload)
    return payload


def recover_oracle_roots() -> dict[str, Any]:
    """Recover a smoke interrupted after physically renaming oracle roots."""

    if not ORACLE_JOURNAL.exists() and not ORACLE_JOURNAL.is_symlink():
        return {
            "phase": "v96_oracle_recovery_not_needed",
            "recovered": False,
            "passed": True,
        }
    journal = _read_json(ORACLE_JOURNAL)
    moves = _validated_oracle_journal_moves(journal)
    if journal.get("status") == "restored_after_isolated_smoke":
        if not all(source.is_dir() and not hidden.exists() for source, hidden in moves):
            raise RuntimeError("V96 restored oracle journal disagrees with the filesystem")
        return {
            "phase": "v96_oracle_recovery_not_needed",
            "recovered": False,
            "passed": True,
        }
    restored = _restore_oracle_moves(moves, status="recovered_after_interrupted_smoke")
    return {
        "phase": "v96_oracle_recovery_complete",
        "recovered": True,
        "journal_sha256": _canonical_sha256(restored),
        "passed": True,
    }


def _parse_json_objects(stdout: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def _protected_smoke_reads(audit: Mapping[str, Any]) -> list[str]:
    loaded = audit.get("loaded_files")
    if not isinstance(loaded, list):
        return ["<missing-loaded-file-inventory>"]
    explicit = {
        (PROJECT_ROOT / "configs/experiments").resolve(),
        (PROJECT_ROOT / "reports/gemma4/questions").resolve(),
        (PROJECT_ROOT / "reports/gemma4/predictions").resolve(),
        (PROJECT_ROOT / "reports/gemma4/artifacts/v95_deferred_final").resolve(),
        (PROJECT_ROOT / "reports/gemma4/artifacts/v96_atomic_pair_repair_final").resolve(),
    }
    violations: list[str] = []
    for raw in loaded:
        if not isinstance(raw, str):
            violations.append(str(raw))
            continue
        path = Path(raw).resolve()
        components = {part.casefold() for part in path.parts}
        protected = bool(components & {"oracle", "qa", "scorer", "predictions"})
        protected = protected or any(
            path == root or root in path.parents for root in explicit
        )
        if protected:
            violations.append(str(path))
    return sorted(set(violations))


def validate_runtime_smoke_report_v96(
    smoke: Mapping[str, Any], gate: Mapping[str, Any]
) -> None:
    candidate = verify_candidate()
    implementation = _release_implementation_inventory()
    journal = _read_json(ORACLE_JOURNAL)
    journal_moves = _validated_oracle_journal_moves(journal)
    records = smoke.get("scenes")
    gates = smoke.get("gates")
    scene_records_valid = isinstance(records, Mapping) and tuple(sorted(records)) == SCENE_IDS
    if scene_records_valid:
        for scene_id in SCENE_IDS:
            record = records[scene_id]
            chat_path = SMOKE_ROOT / "chat" / f"{scene_id}.jsonl"
            audit_path = SMOKE_ROOT / "audit" / f"{scene_id}.json"
            if (
                not isinstance(record, Mapping)
                or chat_path.is_symlink()
                or audit_path.is_symlink()
                or not chat_path.is_file()
                or not audit_path.is_file()
            ):
                scene_records_valid = False
                break
            rows = [
                json.loads(line)
                for line in chat_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            audit = _read_json(audit_path)
            expected_hash = candidate["scene_memories"][scene_id][
                "canonical_prefix_sha256"
            ]
            expected_hashes = [expected_hash] * len(_SMOKE_QUESTIONS)
            if (
                record.get("returncode") != 0
                or record.get("chat_sha256") != sha256_file(chat_path)
                or record.get("audit_sha256") != sha256_file(audit_path)
                or not _require_sha256(record.get("stdout_sha256"), "smoke stdout")
                or not _require_sha256(record.get("stderr_sha256"), "smoke stderr")
                or record.get("prefix_hashes") != expected_hashes
                or record.get("environment_conditioned_input_hashes")
                != expected_hashes
                or record.get("protected_reads") != []
                or len(rows) != len(_SMOKE_QUESTIONS)
                or any(
                    not isinstance(row.get("answer"), str)
                    or not row["answer"].strip()
                    or row.get("prefix_hash") != expected_hash
                    or row.get("environment_conditioned_input_sha256")
                    != expected_hash
                    for row in rows
                )
                or audit.get("passed") is not True
                or audit.get("forbidden_accesses") != []
                or _protected_smoke_reads(audit) != []
            ):
                scene_records_valid = False
                break
    if (
        smoke.get("artifact") != SMOKE_ARTIFACT
        or smoke.get("schema_version") != SCHEMA_VERSION
        or smoke.get("candidate_fingerprint_sha256")
        != gate["authorization"].candidate_fingerprint_sha256
        or smoke.get("candidate_attestation_file_sha256")
        != gate["authorization"].candidate_attestation_file_sha256
        or smoke.get("candidate_attestation_identity_sha256")
        != gate["authorization"].candidate_attestation_identity_sha256
        or smoke.get("deferred_final_evidence_sha256")
        != gate["deferred_final_evidence_sha256"]
        or smoke.get("candidate_checkpoint_sha256") != candidate["checkpoint_sha256"]
        or smoke.get("candidate_adapter_sha256") != candidate["adapter_sha256"]
        or smoke.get("runtime_map_sha256")
        != {
            scene: candidate["runtime_maps"][scene]["packaged_sha256"]
            for scene in SCENE_IDS
        }
        or smoke.get("runtime_implementation_inventory") != implementation
        or smoke.get("oracle_move_journal_sha256") != sha256_file(ORACLE_JOURNAL)
        or journal.get("status") != "restored_after_isolated_smoke"
        or not all(
            source.is_dir() and not hidden.exists()
            for source, hidden in journal_moves
        )
        or smoke.get("scene_ids") != list(SCENE_IDS)
        or smoke.get("questions") != list(_SMOKE_QUESTIONS)
        or smoke.get("expected_answers_supplied_to_children") is not False
        or smoke.get("behavior_assertions_in_children") is not False
        or not scene_records_valid
        or not isinstance(gates, Mapping)
        or set(gates) != _REQUIRED_SMOKE_GATES
        or any(value is not True for value in gates.values())
        or smoke.get("passed") is not True
        or smoke.get("promotion_authorized") is not True
        or smoke.get("default_runtime_pointer_modified") is not False
        or smoke.get("default_runtime_snapshot") != _default_runtime_snapshot()
        or sha256_file(CANDIDATE_CHECKPOINT / "adapter.safetensors")
        != smoke.get("candidate_adapter_sha256")
        or any(
            sha256_file(CANDIDATE_MEMORY_ROOT / scene / MEMORY_FILENAME)
            != candidate["scene_memories"][scene]["tensor_file_sha256"]
            for scene in SCENE_IDS
        )
        or any(
            sha256_file(RUNTIME_MAP_ROOT / scene / "voxel_map.npz")
            != candidate["runtime_maps"][scene]["packaged_sha256"]
            for scene in SCENE_IDS
        )
        or any(
            flag
            in _smoke_command(
                scene,
                audit_path=SMOKE_ROOT / "audit" / f"{scene}.json",
                chat_path=SMOKE_ROOT / "chat" / f"{scene}.jsonl",
            )
            for scene in SCENE_IDS
            for flag in ("--expected", "--answer", "--reference")
        )
    ):
        raise ValueError("V96 isolated runtime smoke did not pass exactly")


def run_smoke() -> dict[str, Any]:
    """Run every held-out runtime scene while all oracle roots are renamed."""

    gate = authenticate_v96_release_gate()
    if SMOKE_REPORT.is_file():
        existing = _read_json(SMOKE_REPORT)
        validate_runtime_smoke_report_v96(existing, gate)
        return existing
    if SMOKE_ROOT.exists() or SMOKE_ROOT.is_symlink():
        raise FileExistsError("V96 smoke work root already exists")
    candidate = verify_candidate()
    python = PROJECT_ROOT / ".venv-gemma4/bin/python"
    if not python.is_file():
        raise FileNotFoundError("V96 local Gemma Python environment is unavailable")
    oracle_roots = _oracle_directories()
    if not oracle_roots:
        raise FileNotFoundError("V96 smoke requires at least one physical oracle directory")
    moves: list[tuple[Path, Path]] = []
    for index, source in enumerate(oracle_roots):
        hidden = source.parent / f".oracle-unavailable-v96-{os.getpid()}-{index}"
        if hidden.exists() or hidden.is_symlink():
            raise FileExistsError(hidden)
        moves.append((source, hidden))
    SMOKE_ROOT.mkdir(parents=True)
    (SMOKE_ROOT / "chat").mkdir()
    (SMOKE_ROOT / "audit").mkdir()
    _write_json_atomic(
        ORACLE_JOURNAL,
        _oracle_journal_payload(moves, "prepared_for_physical_oracle_isolation"),
    )
    before_adapter = sha256_file(CANDIDATE_CHECKPOINT / "adapter.safetensors")
    before_memories = {
        scene: sha256_file(CANDIDATE_MEMORY_ROOT / scene / MEMORY_FILENAME)
        for scene in SCENE_IDS
    }
    before_maps = {
        scene: sha256_file(RUNTIME_MAP_ROOT / scene / "voxel_map.npz")
        for scene in SCENE_IDS
    }
    implementation_before = _release_implementation_inventory()
    default_before = _default_runtime_snapshot()
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
        }
    )
    completed: dict[str, subprocess.CompletedProcess[str]] = {}
    physically_unavailable = False
    try:
        for source, hidden in moves:
            os.rename(source, hidden)
        physically_unavailable = all(
            not source.exists() and hidden.is_dir() for source, hidden in moves
        )
        _write_json_atomic(
            ORACLE_JOURNAL,
            _oracle_journal_payload(moves, "oracles_physically_unavailable"),
        )
        for scene_id in SCENE_IDS:
            completed[scene_id] = subprocess.run(
                _smoke_command(
                    scene_id,
                    audit_path=SMOKE_ROOT / "audit" / f"{scene_id}.json",
                    chat_path=SMOKE_ROOT / "chat" / f"{scene_id}.jsonl",
                ),
                cwd=PROJECT_ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
    finally:
        _restore_oracle_moves(moves, status="restored_after_isolated_smoke")
    failures = {
        scene: {
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
        }
        for scene, process in completed.items()
        if process.returncode != 0
    }
    if failures or tuple(sorted(completed)) != SCENE_IDS:
        raise RuntimeError(f"V96 strict runtime child failed: {failures}")

    scenes: dict[str, dict[str, Any]] = {}
    child_oracle_unavailable = True
    exact_banks = True
    candidate_mode = True
    nonempty = True
    prefixes_invariant = True
    prefixes_match = True
    direct_layout = True
    forbidden_clean = True
    protected_clean = True
    for scene_id in SCENE_IDS:
        process = completed[scene_id]
        objects = _parse_json_objects(process.stdout)
        startups = [row for row in objects if row.get("phase") == "v96_strict_multiscene_ready"]
        completions = [row for row in objects if row.get("phase") == "v96_chat_audit_complete"]
        startup = startups[0] if len(startups) == 1 else {}
        completion = completions[0] if len(completions) == 1 else {}
        chat_path = SMOKE_ROOT / "chat" / f"{scene_id}.jsonl"
        audit_path = SMOKE_ROOT / "audit" / f"{scene_id}.json"
        rows = [
            json.loads(line)
            for line in chat_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        audit = _read_json(audit_path)
        expected_hash = candidate["scene_memories"][scene_id]["canonical_prefix_sha256"]
        prefix_hashes = [row.get("prefix_hash") for row in rows]
        input_hashes = [row.get("environment_conditioned_input_sha256") for row in rows]
        child_oracle_unavailable &= startup.get(
            "oracle_directory_available_at_runtime_start"
        ) is False
        exact_banks &= (
            startup.get("frozen_lora_bank_count") == 10
            and startup.get("frozen_lora_parameter_count") == TOTAL_PARAMETER_COUNT
            and startup.get("trainable_runtime_parameter_count") == 0
            and startup.get("lora_bank_order") == list(EXPECTED_BANKS)
        )
        candidate_mode &= (
            startup.get("runtime_package_mode") == "candidate"
            and completion.get("runtime_package_mode") == "candidate"
            and startup.get("runtime_promotion_authorized") is False
        )
        nonempty &= len(rows) == len(_SMOKE_QUESTIONS) and all(
            isinstance(row.get("answer"), str) and bool(row["answer"].strip())
            for row in rows
        )
        prefixes_invariant &= (
            prefix_hashes == input_hashes
            and len(set(prefix_hashes)) == 1
            and completion.get("prefix_hash_invariant") is True
        )
        prefixes_match &= set(prefix_hashes) == {expected_hash}
        direct_layout &= (
            all(
                isinstance(row.get("prepared_layout_audit"), Mapping)
                and row["prepared_layout_audit"].get(
                    "fixed_scene_memory_tokens_supplied_to_gemma"
                )
                == 738
                and row["prepared_layout_audit"].get(
                    "question_derived_environmental_tokens"
                )
                == 0
                for row in rows
            )
            and completion.get("exact_738_token_memory_supplied_directly_to_gemma") is True
        )
        forbidden_clean &= (
            audit.get("passed") is True
            and audit.get("forbidden_accesses") == []
            and completion.get("forbidden_access_count") == 0
        )
        protected = _protected_smoke_reads(audit)
        protected_clean &= protected == []
        scenes[scene_id] = {
            "returncode": process.returncode,
            "chat_sha256": sha256_file(chat_path),
            "audit_sha256": sha256_file(audit_path),
            "stdout_sha256": hashlib.sha256(process.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(process.stderr.encode()).hexdigest(),
            "prefix_hashes": prefix_hashes,
            "environment_conditioned_input_hashes": input_hashes,
            "protected_reads": protected,
        }
    after_memories = {
        scene: sha256_file(CANDIDATE_MEMORY_ROOT / scene / MEMORY_FILENAME)
        for scene in SCENE_IDS
    }
    after_maps = {
        scene: sha256_file(RUNTIME_MAP_ROOT / scene / "voxel_map.npz")
        for scene in SCENE_IDS
    }
    implementation_after = _release_implementation_inventory()
    restored_journal = _read_json(ORACLE_JOURNAL)
    default_after = _default_runtime_snapshot()
    gates = {
        "known_development_and_deferred_final_gates_passed": gate[
            "known_development_gate_passed"
        ]
        is True
        and gate["deferred_final_gate_passed"] is True,
        "all_six_runtime_processes_exit_zero": all(
            row.returncode == 0 for row in completed.values()
        ),
        "all_oracle_directories_physically_unavailable": physically_unavailable,
        "all_oracle_directories_restored": all(source.is_dir() for source, _ in moves),
        "all_six_children_report_oracle_unavailable": child_oracle_unavailable,
        "all_six_children_use_exact_ten_frozen_banks": exact_banks,
        "all_six_children_report_candidate_mode": candidate_mode,
        "all_twelve_questions_return_nonempty_answers": nonempty,
        "every_scene_prefix_is_invariant": prefixes_invariant,
        "every_scene_prefix_matches_attested_memory": prefixes_match,
        "direct_memory_layout_retained_for_every_answer": direct_layout,
        "file_audit_forbidden_read_count_zero": forbidden_clean,
        "file_audit_protected_read_count_zero": protected_clean,
        "candidate_adapter_bytes_unchanged": before_adapter
        == sha256_file(CANDIDATE_CHECKPOINT / "adapter.safetensors")
        == candidate["adapter_sha256"],
        "candidate_memory_tensor_bytes_unchanged": before_memories == after_memories,
        "candidate_map_bytes_unchanged": before_maps == after_maps,
        "runtime_implementation_bytes_unchanged": implementation_before
        == implementation_after,
        "oracle_move_journal_restored_and_bound": restored_journal.get("status")
        == "restored_after_isolated_smoke"
        and all(source.is_dir() and not hidden.exists() for source, hidden in moves),
        "no_expectation_channel_in_child_protocol": all(
            all(
                flag
                not in _smoke_command(
                    scene,
                    audit_path=SMOKE_ROOT / "audit" / f"{scene}.json",
                    chat_path=SMOKE_ROOT / "chat" / f"{scene}.jsonl",
                )
                for flag in ("--expected", "--answer", "--reference")
            )
            for scene in SCENE_IDS
        ),
        "default_runtime_pointer_unchanged": default_before == default_after
        and default_after["default_demo_uses_v89"] is True
        and default_after["default_demo_uses_v96"] is False,
    }
    report = {
        "artifact": SMOKE_ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "candidate_fingerprint_sha256": gate[
            "authorization"
        ].candidate_fingerprint_sha256,
        "candidate_attestation_file_sha256": gate[
            "authorization"
        ].candidate_attestation_file_sha256,
        "candidate_attestation_identity_sha256": gate[
            "authorization"
        ].candidate_attestation_identity_sha256,
        "deferred_final_evidence_sha256": gate["deferred_final_evidence_sha256"],
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "candidate_adapter_sha256": candidate["adapter_sha256"],
        "runtime_map_sha256": after_maps,
        "runtime_implementation_inventory": implementation_after,
        "oracle_move_journal_sha256": sha256_file(ORACLE_JOURNAL),
        "scene_ids": list(SCENE_IDS),
        "questions": list(_SMOKE_QUESTIONS),
        "expected_answers_supplied_to_children": False,
        "behavior_assertions_in_children": False,
        "scenes": scenes,
        "gates": gates,
        "passed": all(gates.values()),
        "promotion_authorized": all(gates.values()),
        "default_runtime_pointer_modified": False,
        "default_runtime_snapshot": default_after,
    }
    _write_json_create_once_atomic(SMOKE_REPORT, report)
    return report


def _copy_rebound_memory(
    source: Path,
    destination: Path,
    *,
    scene_id: str,
    source_checkpoint_sha256: str,
    destination_checkpoint_sha256: str,
    runtime_config_sha256: str,
) -> dict[str, Any]:
    loaded = load_v81_scene_memory(
        source,
        expected_scene_id=scene_id,
        expected_base_checkpoint_sha256=source_checkpoint_sha256,
        expected_runtime_config_sha256=runtime_config_sha256,
        expected_model_device="cpu",
    )
    metadata = save_v81_scene_memory(
        destination,
        loaded.memory,
        scene_id=scene_id,
        source_base_checkpoint_sha256=destination_checkpoint_sha256,
        runtime_config_sha256=runtime_config_sha256,
        source_control_checkpoint_sha256=loaded.metadata[
            "source_control_checkpoint_sha256"
        ],
        source_probe_tensor_sha256=loaded.metadata["source_probe_tensor_sha256"],
    )
    if (
        sha256_file(source / MEMORY_FILENAME)
        != sha256_file(destination / MEMORY_FILENAME)
        or metadata["canonical_prefix_sha256"]
        != loaded.metadata["canonical_prefix_sha256"]
    ):
        raise RuntimeError(f"V96 promoted memory bytes changed: {scene_id}")
    return {
        "candidate_tensor_file_sha256": sha256_file(source / MEMORY_FILENAME),
        "release_tensor_file_sha256": sha256_file(destination / MEMORY_FILENAME),
        "canonical_prefix_sha256": metadata["canonical_prefix_sha256"],
        "tensor_bytes_reused_exactly": True,
        "metadata_only_rebinding": True,
    }


def _promote_release_once() -> dict[str, Any]:
    gate = authenticate_v96_release_gate()
    if any(
        path.exists() or path.is_symlink()
        for path in (RELEASE_CHECKPOINT, RELEASE_MEMORY_ROOT, RELEASE_REPORT)
    ):
        raise FileExistsError("V96 strict release destination already exists")
    smoke = _read_json(SMOKE_REPORT)
    validate_runtime_smoke_report_v96(smoke, gate)
    candidate = verify_candidate()
    if smoke.get("default_runtime_snapshot") != _default_runtime_snapshot():
        raise ValueError("Default runtime changed after the V96 leakage smoke")
    smoke_sha = sha256_file(SMOKE_REPORT)
    metadata = build_runtime_metadata(
        gate,
        promotion=PROMOTED_DECISION,
        smoke_report_sha256=smoke_sha,
    )
    checkpoint = _atomic_checkpoint(
        RELEASE_CHECKPOINT,
        metadata=metadata,
        authorization=gate["authorization"],
        source_adapter=CANDIDATE_CHECKPOINT / "adapter.safetensors",
    )
    if checkpoint["adapter_sha256"] != candidate["adapter_sha256"]:
        raise RuntimeError("Promoted V96 adapter differs from smoked candidate")
    runtime_sha = effective_runtime_config_sha256(load_runtime_config(RUNTIME_CONFIG))
    RELEASE_MEMORY_ROOT.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{RELEASE_MEMORY_ROOT.name}.", dir=RELEASE_MEMORY_ROOT.parent)
    )
    memories: dict[str, dict[str, Any]] = {}
    try:
        for scene_id in SCENE_IDS:
            memories[scene_id] = _copy_rebound_memory(
                CANDIDATE_MEMORY_ROOT / scene_id,
                temporary / scene_id,
                scene_id=scene_id,
                source_checkpoint_sha256=candidate["checkpoint_sha256"],
                destination_checkpoint_sha256=checkpoint["checkpoint_sha256"],
                runtime_config_sha256=runtime_sha,
            )
        os.rename(temporary, RELEASE_MEMORY_ROOT)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    release = {
        "artifact": ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "promotion_decision": PROMOTED_DECISION,
        "promotion_scope": "strict_direct_continuous_scene_memory_scenes_25_through_30",
        "scene_ids": list(SCENE_IDS),
        "scene_count": len(SCENE_IDS),
        "held_out_generalization_claim": True,
        "strict_input_contract": {
            "shape_each": list(MEMORY_SHAPE),
            "continuous_environment_payload_tokens": 736,
            "native_boi_eoi_retained": True,
            "compiled_before_question": True,
            "same_exact_memory_reused_for_every_question_per_scene": True,
            "question_derived_environmental_tokens": 0,
            "question_conditioned_environmental_readout": False,
            "question_dependent_scene_processing": False,
            "question_dependent_retrieval": False,
            "environmental_text_inputs": [],
        },
        "runtime_config": _relative(RUNTIME_CONFIG),
        "runtime_config_sha256": runtime_sha,
        "checkpoint": checkpoint,
        "runtime_maps": candidate["runtime_maps"],
        "scene_memories": memories,
        "bindings": {
            "candidate_fingerprint_sha256": gate[
                "authorization"
            ].candidate_fingerprint_sha256,
            "candidate_attestation_file_sha256": gate[
                "authorization"
            ].candidate_attestation_file_sha256,
            "candidate_attestation_identity_sha256": gate[
                "authorization"
            ].candidate_attestation_identity_sha256,
            "v1_implementation_seal_sha256": gate[
                "authorization"
            ].v1_implementation_seal_sha256,
            "v2_implementation_seal_sha256": gate[
                "authorization"
            ].v2_implementation_seal_sha256,
            "deferred_final_evidence_sha256": gate[
                "deferred_final_evidence_sha256"
            ],
            "deferred_final_score_sha256": gate["final_score_sha256"],
            "runtime_smoke_sha256": smoke_sha,
            "runtime_implementation_inventory_sha256": smoke[
                "runtime_implementation_inventory"
            ]["runtime_inventory_sha256"],
            "release_implementation_inventory_sha256": smoke[
                "runtime_implementation_inventory"
            ]["inventory_sha256"],
            "v94_state_sha256": V94_STATE_SHA256,
            "v95_state_sha256": gate["authorization"].v95_state_sha256,
            "v96_state_sha256": gate["authorization"].v96_state_sha256,
        },
        "chat_runtime_loads_training_or_evaluation_reports": False,
        "runtime_checkpoint_contains_environmental_text": False,
        "runtime_checkpoint_contains_supervision": False,
        "scene_memory_tensor_bytes_unchanged_from_smoked_candidate": True,
        "adapter_bytes_unchanged_from_smoked_candidate": True,
        "default_runtime_pointer_modified": False,
        "default_runtime_snapshot": _default_runtime_snapshot(),
        "all_release_gates_passed": True,
    }
    _write_json_create_once_atomic(RELEASE_REPORT, release)
    return release


def _remove_partial_release_outputs() -> None:
    """Remove only exact unreported V96 promotion destinations."""

    if RELEASE_REPORT.exists() or RELEASE_REPORT.is_symlink():
        raise RuntimeError("Refusing to remove a V96 release with a release report")
    for root in (RELEASE_CHECKPOINT, RELEASE_MEMORY_ROOT):
        if root.is_symlink():
            raise ValueError(f"Refusing to remove symbolic-link V96 release output: {root}")
        if root.is_dir():
            shutil.rmtree(root)
        elif root.exists():
            raise ValueError(f"V96 release output is not a directory: {root}")


def promote_release() -> dict[str, Any]:
    """Promote transactionally; a failed attempt leaves no partial package."""

    if any(
        path.exists() or path.is_symlink()
        for path in (RELEASE_CHECKPOINT, RELEASE_MEMORY_ROOT, RELEASE_REPORT)
    ):
        raise FileExistsError("V96 strict release destination already exists")
    try:
        return _promote_release_once()
    except BaseException:
        _remove_partial_release_outputs()
        raise


def cleanup_partial_release() -> None:
    """Recover only a pre-report promotion interrupted by process termination."""

    _remove_partial_release_outputs()


def verify_release() -> dict[str, Any]:
    gate = authenticate_v96_release_gate()
    smoke = _read_json(SMOKE_REPORT)
    validate_runtime_smoke_report_v96(smoke, gate)
    candidate = verify_candidate()
    runtime_maps = _verify_runtime_maps()
    implementation = _release_implementation_inventory()
    release = _read_json(RELEASE_REPORT)
    _require_exact_checkpoint_package(RELEASE_CHECKPOINT)
    _require_exact_scene_bundle(RELEASE_MEMORY_ROOT, label="release memory root")
    smoke_sha = sha256_file(SMOKE_REPORT)
    metadata = _read_json(RELEASE_CHECKPOINT / RUNTIME_METADATA_FILENAME)
    expected_metadata = build_runtime_metadata(
        gate,
        promotion=PROMOTED_DECISION,
        smoke_report_sha256=smoke_sha,
    )
    if metadata != expected_metadata:
        raise ValueError("V96 promoted runtime metadata changed")
    fingerprint, files = checkpoint_fingerprint(RELEASE_CHECKPOINT)
    candidate_fingerprint, _candidate_files = checkpoint_fingerprint(CANDIDATE_CHECKPOINT)
    runtime_sha = effective_runtime_config_sha256(load_runtime_config(RUNTIME_CONFIG))
    memory_equal = True
    memory_bound = True
    for scene_id in SCENE_IDS:
        loaded = load_v81_scene_memory(
            RELEASE_MEMORY_ROOT / scene_id,
            expected_scene_id=scene_id,
            expected_base_checkpoint_sha256=fingerprint,
            expected_runtime_config_sha256=runtime_sha,
            expected_model_device="cpu",
        )
        memory_equal &= sha256_file(
            RELEASE_MEMORY_ROOT / scene_id / MEMORY_FILENAME
        ) == sha256_file(CANDIDATE_MEMORY_ROOT / scene_id / MEMORY_FILENAME)
        memory_bound &= (
            loaded.metadata["canonical_prefix_sha256"]
            == release["scene_memories"][scene_id]["canonical_prefix_sha256"]
        )
    provenance = metadata["initialization_provenance"]["v96_strict_runtime_release"]
    checks = {
        "release_report_identity": release.get("artifact") == ARTIFACT
        and release.get("schema_version") == SCHEMA_VERSION
        and release.get("all_release_gates_passed") is True,
        "release_report_promoted": release.get("promotion_decision")
        == PROMOTED_DECISION,
        "exact_two_file_checkpoint": {row["path"] for row in files}
        == {"adapter.safetensors", RUNTIME_METADATA_FILENAME},
        "checkpoint_fingerprint_matches_release": fingerprint
        == release.get("checkpoint", {}).get("checkpoint_sha256"),
        "adapter_byte_identical_to_smoked_candidate": sha256_file(
            RELEASE_CHECKPOINT / "adapter.safetensors"
        )
        == sha256_file(CANDIDATE_CHECKPOINT / "adapter.safetensors")
        == smoke["candidate_adapter_sha256"],
        "all_six_memory_tensor_files_byte_identical_to_candidate": memory_equal,
        "all_six_memories_bound_to_attested_prefixes": memory_bound,
        "all_six_runtime_maps_bound_to_smoked_bytes": release.get("runtime_maps")
        == runtime_maps
        == candidate["runtime_maps"],
        "exact_ten_frozen_final_state_banks": tuple(
            row["name"] for row in metadata["lora"]["banks"]
        )
        == EXPECTED_BANKS
        and metadata["lora"]["adapter_parameter_count"] == TOTAL_PARAMETER_COUNT
        and metadata["lora"]["trainable_adapter_parameter_count"] == 0,
        "deferred_final_binding_exact": provenance[
            "deferred_final_evidence_sha256"
        ]
        == gate["deferred_final_evidence_sha256"],
        "candidate_attestation_binding_exact": provenance[
            "candidate_attestation_file_sha256"
        ]
        == release["bindings"]["candidate_attestation_file_sha256"]
        == gate["authorization"].candidate_attestation_file_sha256
        and provenance["candidate_attestation_identity_sha256"]
        == release["bindings"]["candidate_attestation_identity_sha256"]
        == gate["authorization"].candidate_attestation_identity_sha256,
        "evaluator_implementation_binding_exact": provenance[
            "v1_implementation_seal_sha256"
        ]
        == release["bindings"]["v1_implementation_seal_sha256"]
        == gate["authorization"].v1_implementation_seal_sha256
        and provenance["v2_implementation_seal_sha256"]
        == release["bindings"]["v2_implementation_seal_sha256"]
        == gate["authorization"].v2_implementation_seal_sha256,
        "runtime_smoke_binding_exact": provenance["smoke_report_sha256"]
        == smoke_sha
        == release["bindings"]["runtime_smoke_sha256"],
        "runtime_implementation_binding_exact": provenance[
            "runtime_implementation_inventory_sha256"
        ]
        == implementation["runtime_inventory_sha256"]
        == release["bindings"]["runtime_implementation_inventory_sha256"],
        "release_implementation_binding_exact": implementation[
            "inventory_sha256"
        ]
        == release["bindings"]["release_implementation_inventory_sha256"],
        "runtime_promotion_authorized": provenance["runtime_promotion_authorized"]
        is True,
        "candidate_checkpoint_identity_retained_in_smoke": candidate_fingerprint
        == smoke["candidate_checkpoint_sha256"],
        "default_runtime_pointer_unchanged": release[
            "default_runtime_pointer_modified"
        ]
        is False
        and release.get("default_runtime_snapshot") == _default_runtime_snapshot()
        and smoke.get("default_runtime_snapshot") == _default_runtime_snapshot(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"V96 strict release verification failed: {checks}")
    return {
        "phase": "v96_strict_runtime_release_verified",
        "passed": True,
        "candidate_fingerprint_sha256": gate[
            "authorization"
        ].candidate_fingerprint_sha256,
        "candidate_attestation_file_sha256": gate[
            "authorization"
        ].candidate_attestation_file_sha256,
        "candidate_attestation_identity_sha256": gate[
            "authorization"
        ].candidate_attestation_identity_sha256,
        "v1_implementation_seal_sha256": gate[
            "authorization"
        ].v1_implementation_seal_sha256,
        "v2_implementation_seal_sha256": gate[
            "authorization"
        ].v2_implementation_seal_sha256,
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "candidate_adapter_sha256": candidate["adapter_sha256"],
        "deferred_final_evidence_sha256": gate["deferred_final_evidence_sha256"],
        "runtime_smoke_sha256": smoke_sha,
        "release_report_sha256": sha256_file(RELEASE_REPORT),
        "release_checkpoint_sha256": fingerprint,
        "release_adapter_sha256": sha256_file(
            RELEASE_CHECKPOINT / "adapter.safetensors"
        ),
        "v95_state_sha256": gate["authorization"].v95_state_sha256,
        "v96_state_sha256": gate["authorization"].v96_state_sha256,
        "runtime_implementation_inventory_sha256": implementation[
            "runtime_inventory_sha256"
        ],
        "release_implementation_inventory_sha256": implementation[
            "inventory_sha256"
        ],
        "scene_ids": list(SCENE_IDS),
        "checks": checks,
    }


def cleanup_failed_candidate() -> None:
    """Remove only an un-smoked, unpromoted partial V96 candidate."""

    if ORACLE_JOURNAL.exists() or ORACLE_JOURNAL.is_symlink():
        recover_oracle_roots()
    if any(
        path.exists() or path.is_symlink()
        for path in (SMOKE_REPORT, RELEASE_REPORT, RELEASE_CHECKPOINT, RELEASE_MEMORY_ROOT)
    ):
        raise RuntimeError("Refusing V96 cleanup after smoke or release evidence exists")
    for root in (
        CANDIDATE_CHECKPOINT,
        CANDIDATE_MEMORY_ROOT,
        RUNTIME_MAP_ROOT,
        SMOKE_ROOT,
    ):
        if root.is_symlink():
            raise ValueError(f"Refusing to clean symbolic-link V96 candidate: {root}")
        if root.exists():
            shutil.rmtree(root)
    if RUNTIME_CONFIG.exists():
        if RUNTIME_CONFIG.is_symlink():
            raise ValueError("Refusing to clean symbolic-link V96 runtime config")
        RUNTIME_CONFIG.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "authenticate",
            "prepare",
            "verify-candidate",
            "smoke",
            "promote",
            "verify",
            "recover-oracles",
            "cleanup-partial-release",
            "cleanup-failed-candidate",
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    functions = {
        "authenticate": authenticate_v96_release_gate,
        "prepare": prepare_candidate,
        "verify-candidate": verify_candidate,
        "smoke": run_smoke,
        "promote": promote_release,
        "verify": verify_release,
        "recover-oracles": recover_oracle_roots,
        "cleanup-partial-release": lambda: (
            cleanup_partial_release()
            or {"phase": "v96_partial_release_cleaned", "passed": True}
        ),
        "cleanup-failed-candidate": lambda: (
            cleanup_failed_candidate()
            or {"phase": "v96_failed_candidate_cleaned", "passed": True}
        ),
    }
    try:
        result = functions[args.command]()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"V96 strict runtime {args.command} refused: {error}", file=sys.stderr)
        return 2
    serializable = {
        key: value.to_payload() if isinstance(value, V96CandidateAuthorization) else value
        for key, value in result.items()
    }
    print(json.dumps(serializable, indent=2, sort_keys=True, allow_nan=False))
    return 2 if args.command == "smoke" and result.get("passed") is not True else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT",
    "CANDIDATE_CHECKPOINT",
    "CANDIDATE_MEMORY_ROOT",
    "ORACLE_JOURNAL",
    "RELEASE_CHECKPOINT",
    "RELEASE_MEMORY_ROOT",
    "RELEASE_REPORT",
    "RUNTIME_CONFIG",
    "RUNTIME_MAP_ROOT",
    "SCENE_IDS",
    "SMOKE_REPORT",
    "authenticate_v96_release_gate",
    "build_runtime_config_payload",
    "build_runtime_metadata",
    "cleanup_failed_candidate",
    "cleanup_partial_release",
    "main",
    "materialize_runtime_config",
    "prepare_candidate",
    "promote_release",
    "recover_oracle_roots",
    "run_smoke",
    "validate_runtime_smoke_report_v96",
    "verify_candidate",
    "verify_release",
]
