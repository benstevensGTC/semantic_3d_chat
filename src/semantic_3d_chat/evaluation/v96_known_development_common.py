"""Shared fail-closed contracts for V96 known-development evaluation.

V96 keeps Gemma and the exact nine-bank V95 parent frozen, adds only the
fresh V96 bank, and evaluates the resulting fixed-final candidate through four
complete-memory arms.  This module contains no Gemma loader and no reference
loader.  Questions are opened only by the predictor after all memories and
the fixed-final candidate have been authenticated; labels are opened only by
the two dedicated aggregate-only scorers.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.baseline_io import read_jsonl
from semantic_3d_chat.evaluation.question_manifest import QuestionManifest
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import sha256_file_v85
from semantic_3d_chat.evaluation.v95_known_development_common import (
    MEMORY_CACHE,
    PAIR_SCENE,
    QUESTION_MANIFEST,
    QUESTION_MANIFEST_SHA256,
    QUESTIONS_PER_SCENE,
    QUESTIONS_SHA256,
    REFERENCE_SHA256,
    SCENE_IDS,
    authenticate_memory_cache_v95,
    load_known_questions_v95,
    protected_v94_behavior_paths_v95,
)
from semantic_3d_chat.evaluation.v95_strict_causal_successor_preflight import (
    permuted_payload_memory_v95,
    zero_payload_memory_v95,
)
from semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight import (
    CONFIG,
    FRESH_BANK_NAME,
    FRESH_PARAMETER_COUNT,
    TARGET_MODULES,
    authenticate_parent_v95_v96,
    load_config_v96,
)
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.language.prefix_injection import prefix_sha256

SCHEMA_VERSION: Final[int] = 96
QUESTION_COUNT: Final[int] = 216
CHANGED_SIDE_COUNT: Final[int] = 24
CHANGED_UNIT_COUNT: Final[int] = 12
INVARIANT_SIDE_COUNT: Final[int] = 192
INVARIANT_UNIT_COUNT: Final[int] = 96
MEMORY_SHAPE: Final[tuple[int, int, int]] = (1, 738, 1536)
MEMORY_DTYPE: Final[torch.dtype] = torch.bfloat16
# Keep V95's sealed payload permutation byte-identical for an apples-to-apples
# causal control; 960096 remains V96's optimization/schedule seed only.
PERMUTATION_SEED: Final[int] = 950095

PRIMARY: Final[str] = "primary"
ZERO_PAYLOAD: Final[str] = "zero_payload"
FULL_INTERIOR_PERMUTATION: Final[str] = "full_interior_permutation"
PAIRED_WRONG_SCENE: Final[str] = "paired_wrong_scene"
ARMS: Final[tuple[str, ...]] = (
    PRIMARY,
    ZERO_PAYLOAD,
    FULL_INTERIOR_PERMUTATION,
    PAIRED_WRONG_SCENE,
)

PREDICTION_ARTIFACT: Final[str] = (
    "gemma4_v96_known_development_question_only_predictions_v1"
)
PREDICTION_COMPLETION_ARTIFACT: Final[str] = (
    "gemma4_v96_known_development_prediction_completion_v1"
)
STRUCTURED_SCORE_ARTIFACT: Final[str] = (
    "gemma4_v96_known_development_structured_score_v1"
)
NLL_ARTIFACT: Final[str] = "gemma4_v96_known_development_nll_aggregate_v1"
NLL_COMPLETION_ARTIFACT: Final[str] = (
    "gemma4_v96_known_development_nll_completion_v1"
)
FINAL_SCORE_ARTIFACT: Final[str] = "gemma4_v96_known_development_gate_v1"
EVIDENCE_ARTIFACT: Final[str] = "gemma4_v96_known_development_evidence_v1"

EXPECTED_CANDIDATE_TENSORS: Final[dict[str, tuple[int, ...]]] = {
    "adapters.0.lora_a": (8, 1536),
    "adapters.0.lora_b": (4096, 8),
}
_HEX64: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_PREDICTION_FIELD: Final[dict[str, str]] = {
    arm: f"{arm}_prediction" for arm in ARMS
}
_MEMORY_FIELD: Final[dict[str, str]] = {
    arm: f"{arm}_memory_sha256" for arm in ARMS
}
PREDICTION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "artifact",
        "schema_version",
        "scene_id",
        "question_id",
        "paired_scene_id",
        *(_PREDICTION_FIELD[arm] for arm in ARMS),
        *(_MEMORY_FIELD[arm] for arm in ARMS),
        "all_memory_hashes_unchanged",
        "provenance_sha256",
    }
)


@dataclass(frozen=True)
class EvaluationPathsV96:
    predictions: Path
    provenance: Path
    prediction_access: Path
    prediction_completion: Path
    structured_score: Path
    nll: Path
    nll_access: Path
    nll_completion: Path
    final_score: Path
    evidence: Path


@dataclass(frozen=True)
class FixedInputsV96:
    candidate: Mapping[str, Any]
    memories: Mapping[str, Mapping[str, torch.Tensor]]
    memory_hashes: Mapping[str, Mapping[str, str]]
    memory_manifest_sha256: str
    memory_paths: Mapping[str, Path]


def resolve_v96(path: str | Path) -> Path:
    value = Path(path).expanduser()
    rooted = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(rooted))


def canonical_sha256_v96(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assert_bound_config_path_v96(path: str | Path) -> None:
    """Reject evaluator execution against a config not covered by the seal."""

    if resolve_v96(path) != resolve_v96(CONFIG):
        raise ValueError("V96 known-development config path is not the sealed default")


def require_sha256_v96(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"V96 known-development {label} is not a SHA-256")
    return value


def read_json_strict_v96(path: str | Path) -> dict[str, Any]:
    source = resolve_v96(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate V96 JSON key in {source}: {key}")
            result[key] = value
        return result

    value = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise TypeError(f"V96 JSON must contain exactly one object: {source}")
    return value


def _atomic_create(path: Path, encoded: bytes) -> None:
    """Publish bytes once; interrupted publication leaves no partial target."""

    import tempfile

    destination = resolve_v96(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_create_once_v96(path: str | Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_create(resolve_v96(path), encoded)


def write_jsonl_create_once_v96(
    path: str | Path, records: Sequence[Mapping[str, Any]]
) -> None:
    encoded = "".join(
        json.dumps(dict(record), sort_keys=True, allow_nan=False) + "\n"
        for record in records
    ).encode("utf-8")
    _atomic_create(resolve_v96(path), encoded)


def evaluation_paths_v96(config: Mapping[str, Any]) -> EvaluationPathsV96:
    del config  # Paths are part of the evaluator implementation seal.
    predictions = (
        PROJECT_ROOT
        / "reports/gemma4/predictions"
        / "gemma4_v96_atomic_pair_repair_known_development_question_only.jsonl"
    )
    metrics = PROJECT_ROOT / "reports/gemma4/metrics"
    stem = metrics / "gemma4_v96_atomic_pair_repair_known_development"
    return EvaluationPathsV96(
        predictions=predictions,
        provenance=predictions.with_name(f"{predictions.name}.provenance.json"),
        prediction_access=predictions.with_name(f"{predictions.name}.access.json"),
        prediction_completion=predictions.with_name(f"{predictions.name}.completion.json"),
        structured_score=stem.with_name(f"{stem.name}_structured.json"),
        nll=stem.with_name(f"{stem.name}_nll.json"),
        nll_access=stem.with_name(f"{stem.name}_nll_access.json"),
        nll_completion=stem.with_name(f"{stem.name}_nll_completion.json"),
        final_score=stem.with_suffix(".json"),
        evidence=stem.with_name(f"{stem.name}_evidence.json"),
    )


def assert_output_bundle_state_v96(paths: Sequence[Path], *, complete: bool) -> None:
    states = [path.exists() or path.is_symlink() for path in paths]
    if complete and not all(states):
        raise FileNotFoundError("V96 evidence bundle is incomplete")
    if not complete and any(states):
        raise FileExistsError("V96 create-once output bundle is asymmetric or already exists")


def assert_same_candidate_v96(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> None:
    if before != after:
        raise RuntimeError("V96 fixed-final candidate changed during evaluation")


def _candidate_binding_contract_v96(
    config: Mapping[str, Any],
    preflight: Mapping[str, str],
    *,
    v95_parent_evidence_sha256: str,
) -> dict[str, Any]:
    return {
        "config_sha256": preflight["config_sha256"],
        "preregistration_sha256": preflight["preregistration_sha256"],
        "cpu_preflight_sha256": preflight["cpu_preflight_sha256"],
        "topology_smoke_sha256": preflight["topology_smoke_sha256"],
        "trainer_source_sha256": config["sources"]["trainer_source_sha256"],
        "schedule_sha256": config["training"]["schedule_sha256"],
        "invariant_subset_sha256": config["training"]["invariant_subset_sha256"],
        "v95_parent_evidence_sha256": v95_parent_evidence_sha256,
        "fixed_final_optimizer_updates": 285,
        "class_weight_inventory_sha256": config["training_pool"][
            "balanced_class_weight_inventory_sha256"
        ],
        "changed_family_weight_inventory_sha256": config["training_pool"][
            "changed_family_weight_inventory_sha256"
        ],
        "invariant_family_weight_inventory_sha256": config["training_pool"][
            "invariant_family_weight_inventory_sha256"
        ],
        "known_development_labels_opened": False,
        "known_development_questions_opened": False,
        "deferred_final_generated": False,
    }


def authenticate_training_report_files_v96(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    """Authenticate training evidence without importing the model-bearing trainer."""

    trainer = load_future_trainer_v96()
    authenticate = getattr(trainer, "authenticate_training_report_v96", None)
    if not callable(authenticate):
        raise TypeError("V96 trainer lacks its model-free training authenticator")
    preflight = authenticate(config, config_path=config_path)
    report_path = resolve_v96(config["outputs"]["training_report"])
    report = read_json_strict_v96(report_path)
    candidate = report.get("candidate")
    if (
        report.get("artifact") != "gemma4_v96_atomic_pair_repair_training_v1"
        or report.get("schema_version") != SCHEMA_VERSION
        or report.get("status") != "fixed_final_training_complete_not_promoted"
        or report.get("config_sha256") != preflight["config_sha256"]
        or report.get("preregistration_sha256") != preflight["preregistration_sha256"]
        or report.get("cpu_preflight_sha256") != preflight["cpu_preflight_sha256"]
        or report.get("topology_smoke_sha256") != preflight["topology_smoke_sha256"]
        or report.get("optimizer_updates") != 285
        or report.get("micro_steps_consumed") != 2280
        or report.get("retention_steps_consumed") != 1920
        or report.get("changed_pair_steps_consumed") != 264
        or report.get("invariant_pair_steps_consumed") != 96
        or report.get("total_nll_forwards") != 3168
        or report.get("protected_read_count") != 0
        or report.get("known_development_labels_loaded") is not False
        or report.get("known_development_questions_loaded") is not False
        or report.get("deferred_final_generated") is not False
        or report.get("oracle_loaded") is not False
        or report.get("runtime_promotion_authorized") is not False
        or not isinstance(candidate, Mapping)
        or candidate.get("fixed_final") is not True
        or candidate.get("known_development_scored") is not False
        or candidate.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V96 fixed-final training report authentication failed")
    return {
        **preflight,
        "training_report_sha256": sha256_file_v85(report_path),
        "training_report_candidate_metadata_sha256": require_sha256_v96(
            candidate.get("metadata_canonical_sha256"), "training candidate metadata"
        ),
        "training_report_candidate_weights_sha256": require_sha256_v96(
            candidate.get("weights_sha256"), "training candidate weights"
        ),
    }


def authenticate_fixed_final_candidate_v96(
    config: Mapping[str, Any],
    *,
    config_path: str | Path = CONFIG,
    audit: FileAccessAudit | None = None,
) -> dict[str, Any]:
    """Authenticate the immutable two-tensor V96 bank without loading Gemma."""

    training = authenticate_training_report_files_v96(config, config_path=config_path)
    parent = authenticate_parent_v95_v96(config)
    root = resolve_v96(config["outputs"]["fixed_final_candidate"])
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(root)
    entries = sorted(path.name for path in root.iterdir())
    if entries != ["bridge.safetensors", "runtime_metadata.json"]:
        raise ValueError("V96 fixed-final directory inventory changed")
    weights = root / "bridge.safetensors"
    metadata_path = root / "runtime_metadata.json"
    if audit is not None:
        audit.record(weights)
        audit.record(metadata_path)
    metadata = read_json_strict_v96(metadata_path)
    expected_fields = {
        "artifact",
        "schema_version",
        "status",
        "parent",
        "bank_name",
        "target_modules",
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
        "known_development_scored",
        "deferred_final_generated",
        "runtime_promotion_authorized",
        "bindings",
    }
    if (
        set(metadata) != expected_fields
        or metadata.get("artifact") != "gemma4_v96_atomic_pair_repair_fixed_final_v1"
        or metadata.get("schema_version") != SCHEMA_VERSION
        or metadata.get("status") != "fixed_final_awaiting_known_development_gate"
        or metadata.get("parent") != "v95_fixed_final_nonpromoted_optimization_parent"
        or metadata.get("bank_name") != FRESH_BANK_NAME
        or metadata.get("target_modules") != list(TARGET_MODULES)
        or metadata.get("rank") != 8
        or metadata.get("alpha") != 16.0
        or metadata.get("dropout") != 0.0
        or metadata.get("parameter_count") != FRESH_PARAMETER_COUNT
        or metadata.get("tensor_inventory") != sorted(EXPECTED_CANDIDATE_TENSORS)
        or metadata.get("bindings")
        != _candidate_binding_contract_v96(
            config,
            training,
            v95_parent_evidence_sha256=parent["v95_evidence_sha256"],
        )
        or any(
            metadata.get(field) is not False
            for field in (
                "environmental_memory_serialized",
                "questions_or_answers_serialized",
                "oracle_serialized",
                "known_development_scored",
                "deferred_final_generated",
                "runtime_promotion_authorized",
            )
        )
    ):
        raise ValueError("V96 fixed-final metadata contract changed")
    weights_sha256 = sha256_file_v85(weights)
    if (
        metadata.get("weights_sha256") != weights_sha256
        or training["training_report_candidate_weights_sha256"] != weights_sha256
        or training["training_report_candidate_metadata_sha256"]
        != canonical_sha256_v96(metadata)
    ):
        raise ValueError("V96 fixed-final bytes differ from the training report")
    with safe_open(str(weights), framework="pt", device="cpu") as archive:
        keys = sorted(archive.keys())
        observed = {
            key: (tuple(archive.get_slice(key).get_shape()), archive.get_tensor(key).dtype)
            for key in keys
        }
    if keys != sorted(EXPECTED_CANDIDATE_TENSORS) or any(
        observed[key][0] != shape or observed[key][1] != torch.float32
        for key, shape in EXPECTED_CANDIDATE_TENSORS.items()
    ):
        raise ValueError("V96 fixed-final tensor inventory, shape, or dtype changed")
    state = load_file(str(weights), device="cpu")
    if (
        any(not bool(torch.isfinite(tensor).all()) for tensor in state.values())
        or tensor_state_sha256(state) != metadata.get("state_sha256")
    ):
        raise ValueError("V96 fixed-final tensor state changed")
    fingerprint = {
        "artifact": "gemma4_v96_fixed_final_fingerprint_v1",
        "directory_inventory": entries,
        "weights_sha256": weights_sha256,
        "metadata_file_sha256": sha256_file_v85(metadata_path),
        "metadata_canonical_sha256": canonical_sha256_v96(metadata),
        "state_sha256": metadata["state_sha256"],
        "tensor_inventory_sha256": canonical_sha256_v96(
            {key: [*observed[key][0], str(observed[key][1])] for key in keys}
        ),
        "training_report_sha256": training["training_report_sha256"],
        "config_sha256": training["config_sha256"],
        "preregistration_sha256": training["preregistration_sha256"],
        "cpu_preflight_sha256": training["cpu_preflight_sha256"],
        "fixed_final_optimizer_updates": 285,
        "frozen_v95_state_sha256": parent["v95_state_sha256"],
        "known_development_scored": False,
        "deferred_final_generated": False,
        "runtime_promotion_authorized": False,
    }
    fingerprint["fingerprint_sha256"] = canonical_sha256_v96(fingerprint)
    return fingerprint


def bind_all_memories_v96(
    source: Mapping[str, torch.Tensor], *, permutation_seed: int = PERMUTATION_SEED
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict[str, str]]]:
    """Bind every scene/control combination before question I/O."""

    if tuple(source) != SCENE_IDS:
        raise ValueError("V96 memory source must contain six scenes in sealed order")
    primary = {scene_id: source[scene_id] for scene_id in SCENE_IDS}
    zero = {scene_id: zero_payload_memory_v95(source[scene_id]) for scene_id in SCENE_IDS}
    permutation = {
        scene_id: permuted_payload_memory_v95(source[scene_id], seed=permutation_seed)
        for scene_id in SCENE_IDS
    }
    wrong = {scene_id: source[PAIR_SCENE[scene_id]] for scene_id in SCENE_IDS}
    memories = {
        PRIMARY: primary,
        ZERO_PAYLOAD: zero,
        FULL_INTERIOR_PERMUTATION: permutation,
        PAIRED_WRONG_SCENE: wrong,
    }
    hashes = {
        arm: {
            scene_id: prefix_sha256(memories[arm][scene_id]) for scene_id in SCENE_IDS
        }
        for arm in ARMS
    }
    if tuple(memories) != ARMS or any(
        tuple(values) != SCENE_IDS for values in memories.values()
    ):
        raise RuntimeError("V96 did not bind the complete arm/scene product")
    return memories, hashes


def authenticate_fixed_inputs_before_questions_v96(
    config: Mapping[str, Any],
    *,
    config_path: str | Path = CONFIG,
    audit: FileAccessAudit | None = None,
) -> FixedInputsV96:
    candidate = authenticate_fixed_final_candidate_v96(
        config, config_path=config_path, audit=audit
    )
    source, paths, manifest_sha256 = authenticate_memory_cache_v95(audit=audit)
    memories, hashes = bind_all_memories_v96(source)
    return FixedInputsV96(
        candidate=candidate,
        memories=memories,
        memory_hashes=hashes,
        memory_manifest_sha256=manifest_sha256,
        memory_paths=paths,
    )


def load_known_questions_v96() -> QuestionManifest:
    return load_known_questions_v95()


def _oracle_roots_v96() -> list[Path]:
    roots = [PROJECT_ROOT / "data/oracle"]
    roots.extend(PROJECT_ROOT.glob("data*/oracle"))
    return [path.resolve() for path in roots]


def _protected_prior_behavior_paths_v96(
    config: Mapping[str, Any],
) -> list[Path]:
    # The pinned, row-free V95 structured/final/evidence aggregates are part
    # of V96's sealed parent authentication and are therefore allowed.  Raw
    # V94/V95 prediction rows and unrelated historical behavior remain
    # forbidden to the label-blind V96 predictor.
    roots = list(protected_v94_behavior_paths_v95())
    own = evaluation_paths_v96(config)
    own_prediction_bundle = {
        own.predictions,
        own.provenance,
        own.prediction_access,
        own.prediction_completion,
    }
    prediction_root = PROJECT_ROOT / "reports/gemma4/predictions"
    if prediction_root.is_dir():
        roots.extend(
            path
            for path in prediction_root.iterdir()
            if path.resolve() not in own_prediction_bundle
        )
    roots.extend(
        PROJECT_ROOT.glob(
            "reports/gemma4/metrics/gemma4_v95_strict_causal_successor_"
            "known_development_nll*"
        )
    )
    return [path.resolve() for path in roots]


def _other_sanitized_question_paths_v96() -> list[Path]:
    root = PROJECT_ROOT / "reports/gemma4/questions"
    if not root.is_dir():
        return []
    return [
        path.resolve()
        for path in root.iterdir()
        if path.resolve() != QUESTION_MANIFEST.resolve()
    ]


def prediction_forbidden_roots_v96(config: Mapping[str, Any]) -> list[Path]:
    roots = _oracle_roots_v96()
    roots.extend(path.resolve() for path in PROJECT_ROOT.glob("data*/qa"))
    roots.append(resolve_v96(config["known_development_gate"]["labels_path"]))
    roots.extend(_protected_prior_behavior_paths_v96(config))
    roots.extend(_other_sanitized_question_paths_v96())
    return list(dict.fromkeys(roots))


def nll_forbidden_roots_v96(config: Mapping[str, Any]) -> list[Path]:
    allowed = resolve_v96(config["known_development_gate"]["labels_path"])
    roots = _oracle_roots_v96() + _protected_prior_behavior_paths_v96(config)
    roots.extend(_other_sanitized_question_paths_v96())
    for directory in PROJECT_ROOT.glob("data*/qa"):
        directory = directory.resolve()
        if allowed.parent != directory:
            roots.append(directory)
        else:
            roots.extend(
                path.resolve() for path in directory.iterdir() if path.resolve() != allowed
            )
    roots.append(resolve_v96(config["sources"]["training_qa"]))
    roots.extend(
        resolve_v96(path) for path in config["deferred_final_lock"]["empty_qa_placeholders"]
    )
    return list(dict.fromkeys(roots))


def structured_score_forbidden_roots_v96(
    config: Mapping[str, Any],
) -> list[Path]:
    return nll_forbidden_roots_v96(config)


def audit_report_v96(audit: FileAccessAudit) -> dict[str, Any]:
    violations = audit.forbidden_accesses()
    return {
        "artifact": "gemma4_v96_file_access_audit_v1",
        "schema_version": SCHEMA_VERSION,
        "loaded_files": audit.unique_paths,
        "loaded_file_inventory_sha256": canonical_sha256_v96(audit.unique_paths),
        "forbidden_roots": [str(path) for path in audit.forbidden_roots],
        "forbidden_component_names": sorted(audit.forbidden_component_names),
        "block_forbidden": audit.block_forbidden,
        "forbidden_accesses": violations,
        "protected_read_count": len(violations),
        "passed": not violations,
    }


def prediction_provenance_v96(
    config_path: str | Path,
    fixed: FixedInputsV96,
    questions: QuestionManifest,
) -> dict[str, Any]:
    from semantic_3d_chat.evaluation.v96_known_development_implementation import (
        authenticate_evaluation_implementation_v96,
    )

    implementation = authenticate_evaluation_implementation_v96()
    value = {
        "artifact": PREDICTION_ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "config_sha256": sha256_file_v85(resolve_v96(config_path)),
        "implementation_seal_sha256": implementation["seal_sha256"],
        "implementation_source_inventory_sha256": implementation[
            "source_inventory_sha256"
        ],
        "candidate_fingerprint_sha256": fixed.candidate["fingerprint_sha256"],
        "candidate_state_sha256": fixed.candidate["state_sha256"],
        "frozen_v95_state_sha256": fixed.candidate["frozen_v95_state_sha256"],
        "memory_manifest_sha256": fixed.memory_manifest_sha256,
        "bound_memory_sha256": {
            arm: dict(fixed.memory_hashes[arm]) for arm in ARMS
        },
        "question_manifest_sha256": questions.manifest_sha256,
        "questions_sha256": questions.questions_sha256,
        "source_qa_sha256": questions.source_qa_sha256,
        "scene_ids": list(SCENE_IDS),
        "row_count": QUESTION_COUNT,
        "arms": list(ARMS),
        "all_memories_bound_before_questions": True,
        "all_736_interior_tokens_permuted": True,
        "labels_opened": False,
        "questions_or_answers_from_qa_serialized": False,
        "question_dependent_retrieval": False,
        "environmental_text_inputs": [],
        "runtime_promotion_authorized": False,
    }
    value["provenance_sha256"] = canonical_sha256_v96(value)
    return value


def prediction_row_v96(
    *,
    scene_id: str,
    question_id: str,
    predictions: Mapping[str, str],
    memory_hashes: Mapping[str, str],
    provenance_sha256: str,
    unchanged: bool,
) -> dict[str, Any]:
    if (
        scene_id not in SCENE_IDS
        or set(predictions) != set(ARMS)
        or set(memory_hashes) != set(ARMS)
        or any(_HEX64.fullmatch(str(memory_hashes[arm])) is None for arm in ARMS)
        or _HEX64.fullmatch(provenance_sha256) is None
    ):
        raise ValueError("V96 prediction row has an invalid arm or scene inventory")
    row = {
        "artifact": PREDICTION_ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "scene_id": scene_id,
        "question_id": question_id,
        "paired_scene_id": PAIR_SCENE[scene_id],
        **{_PREDICTION_FIELD[arm]: predictions[arm] for arm in ARMS},
        **{_MEMORY_FIELD[arm]: memory_hashes[arm] for arm in ARMS},
        "all_memory_hashes_unchanged": unchanged,
        "provenance_sha256": provenance_sha256,
    }
    if set(row) != PREDICTION_FIELDS:
        raise AssertionError("V96 prediction row field contract changed")
    return row


def validate_prediction_rows_v96(
    rows: Sequence[Mapping[str, Any]],
    *,
    questions: QuestionManifest,
    memory_hashes: Mapping[str, Mapping[str, str]],
    provenance_sha256: str,
) -> None:
    expected = {(row.scene_id, row.question_id) for row in questions.questions}
    seen: set[tuple[str, str]] = set()
    scene_counts: Counter[str] = Counter()
    for row in rows:
        key = (str(row.get("scene_id")), str(row.get("question_id")))
        scene_id = key[0]
        if (
            set(row) != PREDICTION_FIELDS
            or key not in expected
            or key in seen
            or row.get("artifact") != PREDICTION_ARTIFACT
            or row.get("schema_version") != SCHEMA_VERSION
            or row.get("paired_scene_id") != PAIR_SCENE.get(scene_id)
            or row.get("provenance_sha256") != provenance_sha256
            or row.get("all_memory_hashes_unchanged") is not True
            or any(not isinstance(row.get(_PREDICTION_FIELD[arm]), str) for arm in ARMS)
            or any(
                _HEX64.fullmatch(str(row.get(_MEMORY_FIELD[arm]))) is None
                for arm in ARMS
            )
            or any(
                row.get(_MEMORY_FIELD[arm]) != memory_hashes[arm][scene_id]
                for arm in ARMS
            )
        ):
            raise ValueError(f"V96 prediction row contract changed: {key}")
        seen.add(key)
        scene_counts[scene_id] += 1
    if (
        len(rows) != QUESTION_COUNT
        or seen != expected
        or scene_counts
        != Counter({scene_id: QUESTIONS_PER_SCENE for scene_id in SCENE_IDS})
    ):
        raise ValueError("V96 predictions do not have exact 216-row coverage")


def mandatory_fixed_input_reads_v96(
    config: Mapping[str, Any],
    fixed: FixedInputsV96,
    *,
    config_path: str | Path = CONFIG,
) -> set[str]:
    candidate = resolve_v96(config["outputs"]["fixed_final_candidate"])
    return {
        str(resolve_v96(config_path)),
        str(resolve_v96(config["outputs"]["training_report"])),
        str(resolve_v96(config["outputs"]["preregistration"])),
        str(resolve_v96(config["outputs"]["cpu_preflight"])),
        str(resolve_v96(config["outputs"]["topology_smoke"])),
        str((candidate / "bridge.safetensors").resolve()),
        str((candidate / "runtime_metadata.json").resolve()),
        str((MEMORY_CACHE / "manifest.json").resolve()),
        str(QUESTION_MANIFEST.resolve()),
        *(str(path.resolve()) for path in fixed.memory_paths.values()),
    }


def authenticate_prediction_bundle_v96(
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    assert_bound_config_path_v96(config_path)
    initial = load_config_v96(config_path, allow_draft=False)
    audit = FileAccessAudit(
        prediction_forbidden_roots_v96(initial),
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    with audit:
        config = load_config_v96(config_path, allow_draft=False)
        fixed = authenticate_fixed_inputs_before_questions_v96(
            config, config_path=config_path, audit=audit
        )
        questions = load_known_questions_v96()
        paths = evaluation_paths_v96(config)
        assert_output_bundle_state_v96(
            (
                paths.predictions,
                paths.provenance,
                paths.prediction_access,
                paths.prediction_completion,
            ),
            complete=True,
        )
        provenance = read_json_strict_v96(paths.provenance)
        expected_provenance = prediction_provenance_v96(config_path, fixed, questions)
        if provenance != expected_provenance:
            raise ValueError("V96 prediction provenance changed")
        rows = read_jsonl(paths.predictions)
        validate_prediction_rows_v96(
            rows,
            questions=questions,
            memory_hashes=fixed.memory_hashes,
            provenance_sha256=provenance["provenance_sha256"],
        )
        access = read_json_strict_v96(paths.prediction_access)
        completion = read_json_strict_v96(paths.prediction_completion)
        current = authenticate_fixed_final_candidate_v96(
            config, config_path=config_path, audit=audit
        )
    audit.assert_clean()
    assert_same_candidate_v96(fixed.candidate, current)
    mandatory = mandatory_fixed_input_reads_v96(
        config, fixed, config_path=config_path
    )
    if (
        access.get("artifact") != "gemma4_v96_file_access_audit_v1"
        or access.get("passed") is not True
        or access.get("forbidden_accesses") != []
        or access.get("protected_read_count") != 0
        or not mandatory <= set(access.get("loaded_files", []))
        or access.get("loaded_file_inventory_sha256")
        != canonical_sha256_v96(access.get("loaded_files"))
        or completion.get("artifact") != PREDICTION_COMPLETION_ARTIFACT
        or completion.get("schema_version") != SCHEMA_VERSION
        or completion.get("candidate_fingerprint_before")
        != fixed.candidate["fingerprint_sha256"]
        or completion.get("candidate_fingerprint_after")
        != fixed.candidate["fingerprint_sha256"]
        or completion.get("candidate_immutable") is not True
        or completion.get("frozen_v95_parent_immutable") is not True
        or completion.get("memory_manifest_sha256") != fixed.memory_manifest_sha256
        or completion.get("bound_memory_inventory_sha256")
        != canonical_sha256_v96(fixed.memory_hashes)
        or completion.get("question_manifest_sha256") != questions.manifest_sha256
        or completion.get("questions_sha256") != questions.questions_sha256
        or completion.get("prediction_provenance_sha256")
        != provenance["provenance_sha256"]
        or completion.get("prediction_provenance_file_sha256")
        != sha256_file_v85(paths.provenance)
        or completion.get("prediction_sha256") != sha256_file_v85(paths.predictions)
        or completion.get("prediction_access_sha256")
        != sha256_file_v85(paths.prediction_access)
        or completion.get("row_count") != QUESTION_COUNT
        or completion.get("scene_count") != 6
        or completion.get("arms") != list(ARMS)
        or completion.get("all_memories_bound_before_questions") is not True
        or completion.get("all_memory_hashes_invariant") is not True
        or completion.get("labels_opened") is not False
        or completion.get("oracle_opened") is not False
        or completion.get("protected_read_count") != 0
        or completion.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V96 prediction completion/access evidence changed")
    return {
        "config": config,
        "fixed": fixed,
        "questions": questions,
        "paths": paths,
        "provenance": provenance,
        "rows": rows,
        "access": access,
        "completion": completion,
        "prediction_sha256": sha256_file_v85(paths.predictions),
        "provenance_file_sha256": sha256_file_v85(paths.provenance),
        "access_sha256": sha256_file_v85(paths.prediction_access),
        "completion_sha256": sha256_file_v85(paths.prediction_completion),
    }


_ROW_CONTENT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "question",
        "answer",
        "prediction",
        "predictions",
        "reference",
        "references",
        "rows",
        "scene_id",
        "question_id",
        "target_xyz",
        "target_instance",
    }
)


def assert_aggregate_only_v96(value: object, *, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in _ROW_CONTENT_KEYS:
                raise ValueError(
                    "V96 aggregate attempted to serialize row-level content at "
                    + ".".join((*path, str(key)))
                )
            assert_aggregate_only_v96(item, path=(*path, str(key)))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            assert_aggregate_only_v96(item, path=(*path, str(index)))


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"V96 {label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"V96 {label} must be finite")
    return result


def validate_structured_metrics_v96(metrics: Mapping[str, Any]) -> None:
    if set(metrics) != {"arms", "counterfactual", "comparisons", "stable_invariant"}:
        raise ValueError("V96 structured aggregate field inventory changed")
    arms = metrics["arms"]
    changed = metrics["counterfactual"]
    comparisons = metrics["comparisons"]
    stable = metrics["stable_invariant"]
    if not all(isinstance(value, Mapping) for value in (arms, changed, comparisons, stable)):
        raise TypeError("V96 structured aggregate sections are missing")
    if set(arms) != set(ARMS):
        raise ValueError("V96 structured arm inventory changed")
    for arm in ARMS:
        item = arms[arm]
        if (
            not isinstance(item, Mapping)
            or item.get("total") != QUESTION_COUNT
            or not isinstance(item.get("correct"), int)
            or isinstance(item.get("correct"), bool)
            or not 0 <= item["correct"] <= QUESTION_COUNT
        ):
            raise ValueError("V96 structured arm counts changed")
        accuracy = _finite_number(item.get("accuracy"), f"{arm} accuracy")
        if abs(accuracy - item["correct"] / QUESTION_COUNT) > 1e-12:
            raise ValueError("V96 structured accuracy is inconsistent")
    for key, upper in {
        "canonical_correct_sides": CHANGED_SIDE_COUNT,
        "canonical_complete_units": CHANGED_UNIT_COUNT,
        "canonical_prediction_changed_units": CHANGED_UNIT_COUNT,
    }.items():
        value = changed.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= upper:
            raise ValueError(f"V96 structured counterfactual {key} changed")
    if set(comparisons) != set(ARMS[1:]):
        raise ValueError("V96 structured comparison inventory changed")
    expected_stable = {
        "side_count",
        "unit_count",
        "invariant_false_change_count",
        "invariant_false_change_rate",
    }
    false_count = stable.get("invariant_false_change_count")
    if (
        set(stable) != expected_stable
        or stable.get("side_count") != INVARIANT_SIDE_COUNT
        or stable.get("unit_count") != INVARIANT_UNIT_COUNT
        or not isinstance(false_count, int)
        or isinstance(false_count, bool)
        or not 0 <= false_count <= INVARIANT_SIDE_COUNT
        or abs(
            _finite_number(
                stable.get("invariant_false_change_rate"),
                "invariant false-change rate",
            )
            - false_count / INVARIANT_SIDE_COUNT
        )
        > 1e-12
    ):
        raise ValueError("V96 stable-invariant aggregate changed")
    assert_aggregate_only_v96(metrics)


def validate_nll_metrics_v96(metrics: Mapping[str, Any]) -> None:
    expected = {
        "primary_mean_nll",
        "paired_wrong_scene_mean_nll",
        "zero_payload_mean_nll",
        "full_interior_permutation_mean_nll",
        "mean_wrong_minus_primary_nll",
        "mean_changed_wrong_minus_primary_nll",
        "zero_payload_mean_nll_gap",
        "full_interior_permutation_mean_nll_gap",
        "row_count_per_arm",
        "changed_row_count",
    }
    if set(metrics) != expected:
        raise ValueError("V96 NLL aggregate field inventory changed")
    if (
        metrics.get("row_count_per_arm") != QUESTION_COUNT
        or metrics.get("changed_row_count") != CHANGED_SIDE_COUNT
    ):
        raise ValueError("V96 NLL aggregate row counts changed")
    for key in expected - {"row_count_per_arm", "changed_row_count"}:
        _finite_number(metrics.get(key), key)
    assert_aggregate_only_v96(metrics)


def authenticate_structured_score_v96(
    config_path: str | Path = CONFIG,
    *,
    prediction_bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    assert_bound_config_path_v96(config_path)
    bundle = dict(prediction_bundle or authenticate_prediction_bundle_v96(config_path))
    path = bundle["paths"].structured_score
    report = read_json_strict_v96(path)
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise TypeError("V96 structured metrics are missing")
    validate_structured_metrics_v96(metrics)
    expected_fields = {
        "artifact",
        "schema_version",
        "status",
        "candidate_fingerprint_sha256",
        "frozen_v95_state_sha256",
        "memory_manifest_sha256",
        "bound_memory_inventory_sha256",
        "question_manifest_sha256",
        "questions_sha256",
        "reference_sha256",
        "prediction_sha256",
        "prediction_provenance_sha256",
        "prediction_access_sha256",
        "prediction_completion_sha256",
        "row_count",
        "scene_count",
        "prediction_bundle_authenticated_before_labels_opened",
        "labels_opened_only_by_separate_scorer",
        "scorer_loaded_model",
        "row_level_content_serialized",
        "metrics",
        "runtime_promotion_authorized",
    }
    if (
        set(report) != expected_fields
        or report.get("artifact") != STRUCTURED_SCORE_ARTIFACT
        or report.get("schema_version") != SCHEMA_VERSION
        or report.get("status") != "measured_aggregate_only_not_yet_gated"
        or report.get("candidate_fingerprint_sha256")
        != bundle["fixed"].candidate["fingerprint_sha256"]
        or report.get("frozen_v95_state_sha256")
        != bundle["fixed"].candidate["frozen_v95_state_sha256"]
        or report.get("memory_manifest_sha256")
        != bundle["fixed"].memory_manifest_sha256
        or report.get("bound_memory_inventory_sha256")
        != canonical_sha256_v96(bundle["fixed"].memory_hashes)
        or report.get("question_manifest_sha256")
        != bundle["questions"].manifest_sha256
        or report.get("questions_sha256") != bundle["questions"].questions_sha256
        or report.get("reference_sha256") != REFERENCE_SHA256
        or report.get("prediction_sha256") != bundle["prediction_sha256"]
        or report.get("prediction_provenance_sha256")
        != bundle["provenance"]["provenance_sha256"]
        or report.get("prediction_access_sha256") != bundle["access_sha256"]
        or report.get("prediction_completion_sha256") != bundle["completion_sha256"]
        or report.get("row_count") != QUESTION_COUNT
        or report.get("scene_count") != 6
        or report.get("prediction_bundle_authenticated_before_labels_opened") is not True
        or report.get("labels_opened_only_by_separate_scorer") is not True
        or report.get("scorer_loaded_model") is not False
        or report.get("row_level_content_serialized") is not False
        or report.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V96 structured aggregate authentication failed")
    assert_aggregate_only_v96(report)
    return {"report": report, "sha256": sha256_file_v85(path), "bundle": bundle}


def authenticate_nll_bundle_v96(
    config_path: str | Path = CONFIG,
    *,
    fixed: FixedInputsV96 | None = None,
    questions: QuestionManifest | None = None,
) -> dict[str, Any]:
    assert_bound_config_path_v96(config_path)
    config = load_config_v96(config_path, allow_draft=False)
    current = fixed or authenticate_fixed_inputs_before_questions_v96(
        config, config_path=config_path
    )
    manifest = questions or load_known_questions_v96()
    paths = evaluation_paths_v96(config)
    assert_output_bundle_state_v96(
        (paths.nll, paths.nll_access, paths.nll_completion), complete=True
    )
    report = read_json_strict_v96(paths.nll)
    access = read_json_strict_v96(paths.nll_access)
    completion = read_json_strict_v96(paths.nll_completion)
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise TypeError("V96 NLL aggregate metrics are missing")
    validate_nll_metrics_v96(metrics)
    expected_report_fields = {
        "artifact",
        "schema_version",
        "status",
        "candidate_fingerprint_sha256",
        "frozen_v95_state_sha256",
        "memory_manifest_sha256",
        "bound_memory_inventory_sha256",
        "question_manifest_sha256",
        "questions_sha256",
        "reference_sha256",
        "row_count",
        "scene_count",
        "arms",
        "fixed_final_and_memories_authenticated_before_labels_opened",
        "labels_opened_only_by_separate_nll_evaluator",
        "row_level_content_serialized",
        "metrics",
        "runtime_promotion_authorized",
    }
    if (
        set(report) != expected_report_fields
        or report.get("artifact") != NLL_ARTIFACT
        or report.get("schema_version") != SCHEMA_VERSION
        or report.get("status") != "measured_aggregate_only_not_yet_gated"
        or report.get("candidate_fingerprint_sha256")
        != current.candidate["fingerprint_sha256"]
        or report.get("frozen_v95_state_sha256")
        != current.candidate["frozen_v95_state_sha256"]
        or report.get("memory_manifest_sha256") != current.memory_manifest_sha256
        or report.get("bound_memory_inventory_sha256")
        != canonical_sha256_v96(current.memory_hashes)
        or report.get("question_manifest_sha256") != manifest.manifest_sha256
        or report.get("questions_sha256") != manifest.questions_sha256
        or report.get("reference_sha256") != REFERENCE_SHA256
        or report.get("row_count") != QUESTION_COUNT
        or report.get("scene_count") != 6
        or report.get("arms") != list(ARMS)
        or report.get("fixed_final_and_memories_authenticated_before_labels_opened")
        is not True
        or report.get("labels_opened_only_by_separate_nll_evaluator") is not True
        or report.get("row_level_content_serialized") is not False
        or report.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V96 NLL aggregate authentication failed")
    assert_aggregate_only_v96(report)
    expected_label = resolve_v96(config["known_development_gate"]["labels_path"])
    mandatory = mandatory_fixed_input_reads_v96(
        config, current, config_path=config_path
    ) | {str(expected_label)}
    if (
        access.get("artifact") != "gemma4_v96_file_access_audit_v1"
        or access.get("passed") is not True
        or access.get("forbidden_accesses") != []
        or access.get("protected_read_count") != 0
        or str(expected_label) not in set(access.get("loaded_files", []))
        or not mandatory <= set(access.get("loaded_files", []))
        or access.get("loaded_file_inventory_sha256")
        != canonical_sha256_v96(access.get("loaded_files"))
        or completion.get("artifact") != NLL_COMPLETION_ARTIFACT
        or completion.get("schema_version") != SCHEMA_VERSION
        or completion.get("candidate_fingerprint_before")
        != current.candidate["fingerprint_sha256"]
        or completion.get("candidate_fingerprint_after")
        != current.candidate["fingerprint_sha256"]
        or completion.get("candidate_immutable") is not True
        or completion.get("frozen_v95_parent_immutable") is not True
        or completion.get("memory_hashes_invariant") is not True
        or completion.get("nll_sha256") != sha256_file_v85(paths.nll)
        or completion.get("nll_access_sha256") != sha256_file_v85(paths.nll_access)
        or completion.get("row_count_per_arm") != QUESTION_COUNT
        or completion.get("changed_row_count") != CHANGED_SIDE_COUNT
        or completion.get("row_level_content_serialized") is not False
        or completion.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V96 NLL completion/access authentication failed")
    return {
        "report": report,
        "access": access,
        "completion": completion,
        "sha256": sha256_file_v85(paths.nll),
        "access_sha256": sha256_file_v85(paths.nll_access),
        "completion_sha256": sha256_file_v85(paths.nll_completion),
        "fixed": current,
        "questions": manifest,
        "config": config,
        "paths": paths,
    }


def known_development_gate_results_v96(
    structured_metrics: Mapping[str, Any],
    nll_metrics: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    immutable_fixed_final: bool,
    frozen_v95_parent_immutable: bool,
    prefix_invariant: bool,
    label_isolation_proven: bool,
    protected_read_count: int,
) -> dict[str, bool]:
    """Apply every V96 preregistered known-development acceptance gate."""

    validate_structured_metrics_v96(structured_metrics)
    validate_nll_metrics_v96(nll_metrics)
    primary = structured_metrics["arms"][PRIMARY]
    zero = structured_metrics["arms"][ZERO_PAYLOAD]
    permutation = structured_metrics["arms"][FULL_INTERIOR_PERMUTATION]
    changed = structured_metrics["counterfactual"]
    stable = structured_metrics["stable_invariant"]
    gates = {
        "post_fixed_final_go_no_go_role_exact": contract.get("role")
        == "post_fixed_final_go_no_go_not_checkpoint_selection",
        "exact_six_scene_216_row_scope": contract.get("scene_count") == 6
        and contract.get("row_count") == QUESTION_COUNT,
        "exact_changed_and_invariant_scope": contract.get("changed_side_total")
        == CHANGED_SIDE_COUNT
        and contract.get("changed_unit_total") == CHANGED_UNIT_COUNT
        and contract.get("invariant_side_total") == INVARIANT_SIDE_COUNT,
        "labels_opened_after_fixed_final_only": bool(
            contract.get("labels_opened_after_fixed_final_only")
        )
        and label_isolation_proven,
        "labels_opened_by_separate_label_isolated_evaluators_only": bool(
            contract.get("labels_opened_by_separate_label_isolated_evaluator_only")
        )
        and label_isolation_proven,
        "v96_correct_minimum": primary["correct"]
        >= int(contract["v96_correct_minimum"]),
        "changed_side_correct_minimum": changed["canonical_correct_sides"]
        >= int(contract["changed_side_correct_minimum"]),
        "complete_changed_units_minimum": changed["canonical_complete_units"]
        >= int(contract["complete_changed_units_minimum"]),
        "prediction_changed_units_minimum": changed[
            "canonical_prediction_changed_units"
        ]
        >= int(contract["prediction_changed_units_minimum"]),
        "invariant_false_change_maximum": stable["invariant_false_change_count"]
        <= int(contract["invariant_false_change_maximum"]),
        "mean_changed_side_wrong_minus_correct_nll_minimum": nll_metrics[
            "mean_changed_wrong_minus_primary_nll"
        ]
        >= float(contract["mean_changed_side_wrong_minus_correct_nll_minimum"]),
        "correct_accuracy_above_zero_payload": (
            not bool(contract["correct_accuracy_above_zero_payload_required"])
            or primary["accuracy"] > zero["accuracy"]
        ),
        "correct_accuracy_above_full_interior_permutation": (
            not bool(contract["correct_accuracy_above_full_interior_permutation_required"])
            or primary["accuracy"] > permutation["accuracy"]
        ),
        "correct_nll_below_zero_payload": (
            not bool(contract["correct_nll_below_zero_payload_required"])
            or nll_metrics["zero_payload_mean_nll_gap"] > 0.0
        ),
        "correct_nll_below_full_interior_permutation": (
            not bool(contract["correct_nll_below_full_interior_permutation_required"])
            or nll_metrics["full_interior_permutation_mean_nll_gap"] > 0.0
        ),
        "fixed_final_checkpoint_immutable": immutable_fixed_final
        and bool(contract["fixed_final_checkpoint_may_not_change_after_gate"]),
        "frozen_v95_parent_immutable": frozen_v95_parent_immutable,
        "question_independent_prefix_hash_invariance": prefix_invariant,
        "protected_read_count_zero": protected_read_count == 0,
        "full_known_development_coverage": primary["total"]
        == int(contract["row_count"]),
        "pass_required_before_deferred_final_unlock_contract": bool(
            contract.get("pass_required_before_deferred_final_unlock")
        ),
    }
    if not all(isinstance(value, bool) for value in gates.values()):
        raise AssertionError("V96 gate results must be booleans")
    return gates


def load_future_trainer_v96() -> Any:
    """Import the V96 trainer only inside model-bearing evaluator processes."""

    try:
        return importlib.import_module(
            "semantic_3d_chat.training.train_v96_atomic_pair_repair"
        )
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "V96 evaluator is implemented but training source is not yet materialized"
        ) from error


__all__ = [
    "ARMS",
    "CHANGED_SIDE_COUNT",
    "CHANGED_UNIT_COUNT",
    "EVIDENCE_ARTIFACT",
    "FINAL_SCORE_ARTIFACT",
    "FULL_INTERIOR_PERMUTATION",
    "INVARIANT_SIDE_COUNT",
    "INVARIANT_UNIT_COUNT",
    "MEMORY_DTYPE",
    "MEMORY_SHAPE",
    "NLL_ARTIFACT",
    "NLL_COMPLETION_ARTIFACT",
    "PAIRED_WRONG_SCENE",
    "PAIR_SCENE",
    "PERMUTATION_SEED",
    "PREDICTION_ARTIFACT",
    "PREDICTION_COMPLETION_ARTIFACT",
    "PREDICTION_FIELDS",
    "PRIMARY",
    "QUESTIONS_SHA256",
    "QUESTION_COUNT",
    "QUESTION_MANIFEST",
    "QUESTION_MANIFEST_SHA256",
    "REFERENCE_SHA256",
    "SCENE_IDS",
    "SCHEMA_VERSION",
    "STRUCTURED_SCORE_ARTIFACT",
    "ZERO_PAYLOAD",
    "EvaluationPathsV96",
    "FixedInputsV96",
    "assert_aggregate_only_v96",
    "assert_bound_config_path_v96",
    "assert_output_bundle_state_v96",
    "assert_same_candidate_v96",
    "audit_report_v96",
    "authenticate_fixed_final_candidate_v96",
    "authenticate_fixed_inputs_before_questions_v96",
    "authenticate_nll_bundle_v96",
    "authenticate_prediction_bundle_v96",
    "authenticate_structured_score_v96",
    "bind_all_memories_v96",
    "canonical_sha256_v96",
    "evaluation_paths_v96",
    "known_development_gate_results_v96",
    "load_future_trainer_v96",
    "load_known_questions_v96",
    "mandatory_fixed_input_reads_v96",
    "nll_forbidden_roots_v96",
    "prediction_forbidden_roots_v96",
    "prediction_provenance_v96",
    "prediction_row_v96",
    "read_json_strict_v96",
    "require_sha256_v96",
    "structured_score_forbidden_roots_v96",
    "validate_nll_metrics_v96",
    "validate_prediction_rows_v96",
    "validate_structured_metrics_v96",
    "write_json_create_once_v96",
    "write_jsonl_create_once_v96",
]
