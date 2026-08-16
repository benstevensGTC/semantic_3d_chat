"""Shared, model-free contracts for V95's post-fixed-final development gate.

This module deliberately contains no language-model loader and no reference
loader.  It authenticates the immutable V95 fixed-final adapter, binds the six
numeric scene memories and their causal controls, validates label-blind
prediction evidence, and authenticates aggregate-only score artifacts.

The command modules keep the security-relevant process boundaries explicit:

* :mod:`predict_v95_known_development` may open sanitized questions, never QA;
* :mod:`score_v95_known_development` may open labels, never a model;
* :mod:`nll_v95_known_development` is the separately authorized label-aware
  model process and may serialize aggregate NLL values only;
* :mod:`seal_v95_known_development` opens neither labels nor a model.
"""

from __future__ import annotations

import hashlib
import json
import math
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
from semantic_3d_chat.evaluation.question_manifest import (
    QuestionManifest,
    load_question_manifest,
)
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import (
    sha256_file_v85,
)
from semantic_3d_chat.evaluation.v95_strict_causal_successor_preflight import (
    CONFIG,
    FRESH_BANK_NAME,
    FRESH_PARAMETER_COUNT,
    PRIOR_EVALUATION_SCENES,
    TARGET_MODULES,
    authenticate_cpu_preflight_v95,
    load_config_v95,
    permuted_payload_memory_v95,
    zero_payload_memory_v95,
)
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.language.prefix_injection import prefix_sha256

SCHEMA_VERSION: Final[int] = 95
QUESTION_COUNT: Final[int] = 216
QUESTIONS_PER_SCENE: Final[int] = 36
SCENE_IDS: Final[tuple[str, ...]] = PRIOR_EVALUATION_SCENES
PAIR_SCENE: Final[dict[str, str]] = {
    "scene_000057": "scene_000058",
    "scene_000058": "scene_000057",
    "scene_000059": "scene_000060",
    "scene_000060": "scene_000059",
    "scene_000061": "scene_000062",
    "scene_000062": "scene_000061",
}
MEMORY_SHAPE: Final[tuple[int, int, int]] = (1, 738, 1536)
MEMORY_DTYPE: Final[torch.dtype] = torch.bfloat16

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
    "gemma4_v95_known_development_question_only_predictions_v1"
)
PREDICTION_COMPLETION_ARTIFACT: Final[str] = (
    "gemma4_v95_known_development_prediction_completion_v1"
)
STRUCTURED_SCORE_ARTIFACT: Final[str] = (
    "gemma4_v95_known_development_structured_score_v1"
)
NLL_ARTIFACT: Final[str] = "gemma4_v95_known_development_nll_aggregate_v1"
NLL_COMPLETION_ARTIFACT: Final[str] = (
    "gemma4_v95_known_development_nll_completion_v1"
)
FINAL_SCORE_ARTIFACT: Final[str] = "gemma4_v95_known_development_gate_v1"
EVIDENCE_ARTIFACT: Final[str] = "gemma4_v95_known_development_evidence_v1"

QUESTION_MANIFEST: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/questions/v56_fresh_development_validation.json"
)
QUESTION_MANIFEST_SHA256: Final[str] = (
    "74fcfb181bbf809dd6dc3b07800de728558298149e9d76325870c6b4d665b0a2"
)
QUESTIONS_SHA256: Final[str] = (
    "e468d851e46ad606c9599ac1a8016ed10fa974f9985dfc3add6250f3403f8b25"
)
REFERENCE_SHA256: Final[str] = (
    "30ed9006ed442198b3e2444e0c3cdda73cb77c01e7285f31000709b94bb8acad"
)
MEMORY_CACHE: Final[Path] = (
    PROJECT_ROOT
    / "reports/gemma4/artifacts/v94_strict_multiscene_full40/evaluation_cache"
)
MEMORY_MANIFEST_SHA256: Final[str] = (
    "d03bb0b0b1da545d2eef40582fc494f3aea0ec6009e41fb72a9e96e270840591"
)
EXPECTED_MEMORY_SHA256: Final[dict[str, str]] = {
    "scene_000057": "bb74113fe6961ba94b8aab7e9e6b24567ce465a8f2708e7f34c882511d31c059",
    "scene_000058": "7309d0724ffbe5b8f537331e71f52fbdd25f17b39ec710cafe8a48e295d977fd",
    "scene_000059": "bf37aa1365738605aa8fedd8fe65a88b45efb464aaf17744feeabbc34986053c",
    "scene_000060": "99763806c2495e7926a38a9e23632104a4dc691f0b8a266b407c030dc9a207b4",
    "scene_000061": "0d13267813aa387c64cb1206bd899ca6c2c29f8e24f969f0eb97991019b5c77e",
    "scene_000062": "63032491ffbc5dd0499cdcd376cba7d816fbb3d3fa1bdbfc86967db2feb91da4",
}
EXPECTED_MEMORY_FILE_SHA256: Final[dict[str, str]] = {
    "scene_000057": "43af95079f20de329065cce0f6810de2fef60789725916a177dcb0f8d87431d6",
    "scene_000058": "59ff270bd033c7bf0392a1dd177ecd45f7aa8f047336df78f648678c31c6ead5",
    "scene_000059": "83cba4c8d0eb5b866bf6eea23657f2b203834cfba19b62ca208b5aca9c5c4b0f",
    "scene_000060": "9b90c8e6a0c15af42e34dabe784fdfb1d7349cba7249dcee138ccf58d3a4c889",
    "scene_000061": "a172abab1882ca1494c8d67c436bea4fda1c93712b9ff45441a868c879771d35",
    "scene_000062": "49d93ebd157cc357f135a7266c2b33b1b74a3ed446c4f811f944f26dec23211b",
}
EXPECTED_CANDIDATE_TENSORS: Final[dict[str, tuple[int, ...]]] = {
    "adapters.0.lora_a": (8, 1536),
    "adapters.0.lora_b": (512, 8),
    "adapters.1.lora_a": (8, 1536),
    "adapters.1.lora_b": (512, 8),
    "adapters.2.lora_a": (8, 1536),
    "adapters.2.lora_b": (12288, 8),
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
class EvaluationPathsV95:
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
class FixedInputsV95:
    candidate: Mapping[str, Any]
    memories: Mapping[str, Mapping[str, torch.Tensor]]
    memory_hashes: Mapping[str, Mapping[str, str]]
    memory_manifest_sha256: str
    memory_paths: Mapping[str, Path]


def resolve_v95(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def canonical_sha256_v95(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_sha256_v95(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"V95 known-development {label} is not a SHA-256")
    return value


def read_json_strict_v95(path: str | Path) -> dict[str, Any]:
    source = resolve_v95(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate V95 JSON key in {source}: {key}")
            result[key] = value
        return result

    value = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise TypeError(f"V95 JSON must contain exactly one object: {source}")
    return value


def write_json_create_once_v95(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = resolve_v95(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_jsonl_create_once_v95(
    path: str | Path, records: Sequence[Mapping[str, Any]]
) -> None:
    destination = resolve_v95(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True, allow_nan=False) + "\n")


def evaluation_paths_v95(config: Mapping[str, Any]) -> EvaluationPathsV95:
    final_score = resolve_v95(str(config["outputs"]["known_development_score"]))
    predictions = (
        PROJECT_ROOT
        / "reports/gemma4/predictions"
        / "gemma4_v95_strict_causal_successor_known_development_question_only.jsonl"
    )
    stem = final_score.with_suffix("")
    return EvaluationPathsV95(
        predictions=predictions,
        provenance=predictions.with_name(f"{predictions.name}.provenance.json"),
        prediction_access=predictions.with_name(f"{predictions.name}.access.json"),
        prediction_completion=predictions.with_name(
            f"{predictions.name}.completion.json"
        ),
        structured_score=stem.with_name(f"{stem.name}_structured.json"),
        nll=stem.with_name(f"{stem.name}_nll.json"),
        nll_access=stem.with_name(f"{stem.name}_nll_access.json"),
        nll_completion=stem.with_name(f"{stem.name}_nll_completion.json"),
        final_score=final_score,
        evidence=stem.with_name(f"{stem.name}_evidence.json"),
    )


def assert_output_bundle_state_v95(paths: Sequence[Path], *, complete: bool) -> None:
    states = [path.exists() or path.is_symlink() for path in paths]
    if complete and not all(states):
        raise FileNotFoundError("V95 evidence bundle is incomplete")
    if not complete and any(states):
        raise FileExistsError("V95 create-once output bundle is asymmetric or already exists")


def assert_same_candidate_v95(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> None:
    if before != after:
        raise RuntimeError("V95 fixed-final candidate changed during evaluation")


def _candidate_binding_contract(
    config: Mapping[str, Any], preflight: Mapping[str, str]
) -> dict[str, Any]:
    training = config["training"]
    return {
        "config_sha256": preflight["config_sha256"],
        "preregistration_sha256": preflight["preregistration_sha256"],
        "cpu_preflight_sha256": preflight["cpu_preflight_sha256"],
        "trainer_source_sha256": config["sources"]["trainer_source_sha256"],
        "row_order_sha256": training["row_order_sha256"],
        "cross_scene_schedule_sha256": training["cross_scene_schedule_sha256"],
        "zero_payload_schedule_sha256": training["zero_payload_schedule_sha256"],
        "permutation_control_schedule_sha256": training[
            "permutation_control_schedule_sha256"
        ],
        "fixed_final_optimizer_updates": 480,
        "class_weight_inventory_sha256": config["training_pool"][
            "balanced_class_weight_inventory_sha256"
        ],
        "known_development_labels_opened": False,
        "deferred_final_generated": False,
    }


def authenticate_training_report_files_v95(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    """Authenticate V95 training without importing any model-bearing trainer."""

    preflight = authenticate_cpu_preflight_v95(config, config_path=config_path)
    report_path = resolve_v95(config["outputs"]["training_report"])
    report = read_json_strict_v95(report_path)
    if (
        report.get("artifact") != "gemma4_v95_strict_causal_successor_training_v1"
        or report.get("schema_version") != 95
        or report.get("status") != "fixed_final_training_complete_not_promoted"
        or report.get("config_sha256") != preflight["config_sha256"]
        or report.get("preregistration_sha256")
        != preflight["preregistration_sha256"]
        or report.get("cpu_preflight_sha256") != preflight["cpu_preflight_sha256"]
        or report.get("optimizer_updates") != 480
        or report.get("micro_rows_consumed") != 3840
        or report.get("wrong_memory_rows_consumed") != 996
        or report.get("zero_payload_rows_consumed") != 500
        or report.get("permutation_rows_consumed") != 500
        or report.get("total_nll_forwards") != 5836
        or report.get("protected_read_count") != 0
        or report.get("known_development_labels_loaded") is not False
        or report.get("deferred_final_generated") is not False
        or report.get("oracle_loaded") is not False
        or report.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V95 fixed-final training report authentication failed")
    return {
        **preflight,
        "training_report_sha256": sha256_file_v85(report_path),
        "training_report_candidate_metadata_sha256": require_sha256_v95(
            report.get("candidate", {}).get("metadata_canonical_sha256"),
            "training candidate metadata",
        ),
        "training_report_candidate_weights_sha256": require_sha256_v95(
            report.get("candidate", {}).get("weights_sha256"),
            "training candidate weights",
        ),
    }


def authenticate_fixed_final_candidate_v95(
    config: Mapping[str, Any],
    *,
    config_path: str | Path = CONFIG,
    audit: FileAccessAudit | None = None,
) -> dict[str, Any]:
    """Authenticate the immutable six-tensor candidate without loading Gemma."""

    training = authenticate_training_report_files_v95(config, config_path=config_path)
    root = resolve_v95(config["outputs"]["fixed_final_candidate"])
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(root)
    entries = sorted(path.name for path in root.iterdir())
    if entries != ["bridge.safetensors", "runtime_metadata.json"]:
        raise ValueError("V95 fixed-final directory inventory changed")
    weights = root / "bridge.safetensors"
    metadata_path = root / "runtime_metadata.json"
    if audit is not None:
        audit.record(weights)
        audit.record(metadata_path)
    metadata = read_json_strict_v95(metadata_path)
    expected_metadata_fields = {
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
    bindings = metadata.get("bindings")
    expected_bindings = _candidate_binding_contract(config, training)
    if (
        set(metadata) != expected_metadata_fields
        or metadata.get("artifact")
        != "gemma4_v95_strict_causal_successor_fixed_final_v1"
        or metadata.get("schema_version") != 95
        or metadata.get("status") != "fixed_final_awaiting_known_development_gate"
        or metadata.get("parent") != "fixed_final_nonpromoted_optimization_parent"
        or metadata.get("bank_name") != FRESH_BANK_NAME
        or metadata.get("target_modules") != list(TARGET_MODULES)
        or metadata.get("rank") != 8
        or metadata.get("alpha") != 16.0
        or metadata.get("dropout") != 0.0
        or metadata.get("parameter_count") != FRESH_PARAMETER_COUNT
        or metadata.get("tensor_inventory") != sorted(EXPECTED_CANDIDATE_TENSORS)
        or not isinstance(bindings, Mapping)
        or dict(bindings) != expected_bindings
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
        raise ValueError("V95 fixed-final metadata contract changed")
    weights_sha256 = sha256_file_v85(weights)
    if (
        metadata.get("weights_sha256") != weights_sha256
        or training["training_report_candidate_weights_sha256"] != weights_sha256
        or training["training_report_candidate_metadata_sha256"]
        != canonical_sha256_v95(metadata)
    ):
        raise ValueError("V95 fixed-final bytes differ from the training report")
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
        raise ValueError("V95 fixed-final tensor inventory, shape, or dtype changed")
    state = load_file(str(weights), device="cpu")
    if (
        any(not bool(torch.isfinite(tensor).all()) for tensor in state.values())
        or tensor_state_sha256(state) != metadata.get("state_sha256")
    ):
        raise ValueError("V95 fixed-final tensor state changed")
    fingerprint = {
        "artifact": "gemma4_v95_fixed_final_fingerprint_v1",
        "directory_inventory": entries,
        "weights_sha256": weights_sha256,
        "metadata_file_sha256": sha256_file_v85(metadata_path),
        "metadata_canonical_sha256": canonical_sha256_v95(metadata),
        "state_sha256": metadata["state_sha256"],
        "tensor_inventory_sha256": canonical_sha256_v95(
            {key: [*observed[key][0], str(observed[key][1])] for key in keys}
        ),
        "training_report_sha256": training["training_report_sha256"],
        "config_sha256": training["config_sha256"],
        "preregistration_sha256": training["preregistration_sha256"],
        "cpu_preflight_sha256": training["cpu_preflight_sha256"],
        "fixed_final_optimizer_updates": bindings["fixed_final_optimizer_updates"],
        "known_development_scored": False,
        "deferred_final_generated": False,
        "runtime_promotion_authorized": False,
    }
    fingerprint["fingerprint_sha256"] = canonical_sha256_v95(fingerprint)
    return fingerprint


def authenticate_memory_cache_v95(
    *, audit: FileAccessAudit | None = None
) -> tuple[dict[str, torch.Tensor], dict[str, Path], str]:
    """Load the exact six numeric V94-compiled memories, never V94 behavior rows."""

    manifest_path = MEMORY_CACHE / "manifest.json"
    if audit is not None:
        audit.record(manifest_path)
    if sha256_file_v85(manifest_path) != MEMORY_MANIFEST_SHA256:
        raise ValueError("V95 known-development memory manifest bytes changed")
    manifest = read_json_strict_v95(manifest_path)
    if (
        manifest.get("artifact") != "v94_question_independent_evaluation_memory_cache_v1"
        or manifest.get("scene_ids") != list(SCENE_IDS)
        or manifest.get("scene_count") != 6
        or manifest.get("shape_each") != list(MEMORY_SHAPE)
        or manifest.get("dtype") != "bfloat16"
        or manifest.get("compiled_before_questions") is not True
        or manifest.get("question_inputs_used") is not False
        or manifest.get("question_dependent_retrieval") is not False
        or manifest.get("all_memory_slots_retained") is not True
        or manifest.get("environmental_text_inputs") != []
        or set(manifest.get("scenes", {})) != set(SCENE_IDS)
    ):
        raise ValueError("V95 known-development memory manifest contract changed")
    memories: dict[str, torch.Tensor] = {}
    paths: dict[str, Path] = {}
    for scene_id in SCENE_IDS:
        record = manifest["scenes"][scene_id]
        expected_filename = f"{scene_id}.safetensors"
        path = MEMORY_CACHE / expected_filename
        if (
            set(record) != {
                "filename",
                "file_sha256",
                "file_size_bytes",
                "memory_sha256",
            }
            or record.get("filename") != expected_filename
            or record.get("file_sha256") != EXPECTED_MEMORY_FILE_SHA256[scene_id]
            or record.get("memory_sha256") != EXPECTED_MEMORY_SHA256[scene_id]
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record.get("file_size_bytes")
        ):
            raise ValueError(f"V95 numeric memory record changed: {scene_id}")
        if audit is not None:
            audit.record(path)
        if sha256_file_v85(path) != EXPECTED_MEMORY_FILE_SHA256[scene_id]:
            raise ValueError(f"V95 numeric memory bytes changed: {scene_id}")
        state = load_file(str(path), device="cpu")
        if set(state) != {"scene_memory"}:
            raise ValueError("V95 numeric memory tensor inventory changed")
        memory = state["scene_memory"].detach().cpu().contiguous()
        if (
            tuple(memory.shape) != MEMORY_SHAPE
            or memory.dtype != MEMORY_DTYPE
            or not bool(torch.isfinite(memory).all())
            or prefix_sha256(memory) != EXPECTED_MEMORY_SHA256[scene_id]
        ):
            raise ValueError(f"V95 numeric memory tensor changed: {scene_id}")
        memories[scene_id] = memory
        paths[scene_id] = path
    return memories, paths, MEMORY_MANIFEST_SHA256


def bind_all_memories_v95(
    source: Mapping[str, torch.Tensor], *, permutation_seed: int
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict[str, str]]]:
    """Construct every complete causal arm before any question is opened."""

    if tuple(source) != SCENE_IDS:
        raise ValueError("V95 memory source must contain the six scenes in sealed order")
    primary = {scene_id: source[scene_id] for scene_id in SCENE_IDS}
    zero = {
        scene_id: zero_payload_memory_v95(source[scene_id]) for scene_id in SCENE_IDS
    }
    permutation = {
        scene_id: permuted_payload_memory_v95(
            source[scene_id], seed=permutation_seed
        )
        for scene_id in SCENE_IDS
    }
    wrong = {
        scene_id: source[PAIR_SCENE[scene_id]] for scene_id in SCENE_IDS
    }
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
    if tuple(memories) != ARMS or any(tuple(values) != SCENE_IDS for values in memories.values()):
        raise RuntimeError("V95 did not bind the complete arm/scene Cartesian product")
    return memories, hashes


def authenticate_fixed_inputs_before_questions_v95(
    config: Mapping[str, Any],
    *,
    config_path: str | Path = CONFIG,
    audit: FileAccessAudit | None = None,
) -> FixedInputsV95:
    """Authenticate candidate and bind all 24 memories before question I/O."""

    candidate = authenticate_fixed_final_candidate_v95(
        config, config_path=config_path, audit=audit
    )
    source, paths, manifest_sha256 = authenticate_memory_cache_v95(audit=audit)
    memories, hashes = bind_all_memories_v95(
        source, permutation_seed=int(config["training"]["payload_permutation_seed"])
    )
    return FixedInputsV95(
        candidate=candidate,
        memories=memories,
        memory_hashes=hashes,
        memory_manifest_sha256=manifest_sha256,
        memory_paths=paths,
    )


def load_known_questions_v95() -> QuestionManifest:
    if sha256_file_v85(QUESTION_MANIFEST) != QUESTION_MANIFEST_SHA256:
        raise ValueError("V95 sanitized question manifest bytes changed")
    manifest = load_question_manifest(QUESTION_MANIFEST)
    counts = Counter(row.scene_id for row in manifest.questions)
    if (
        manifest.manifest_sha256 != QUESTION_MANIFEST_SHA256
        or manifest.questions_sha256 != QUESTIONS_SHA256
        or manifest.source_qa_sha256 != REFERENCE_SHA256
        or manifest.question_count != QUESTION_COUNT
        or manifest.scene_count != 6
        or counts != Counter({scene_id: QUESTIONS_PER_SCENE for scene_id in SCENE_IDS})
    ):
        raise ValueError("V95 sanitized question contract changed")
    return manifest


def protected_v94_behavior_paths_v95() -> list[Path]:
    predictions = PROJECT_ROOT / "reports/gemma4/predictions"
    metrics = PROJECT_ROOT / "reports/gemma4/metrics"
    paths = [
        predictions / "gemma4_v94_strict_multiscene_full40_validation_question_only.jsonl",
        predictions
        / "gemma4_v94_strict_multiscene_full40_validation_question_only.jsonl.provenance.json",
        predictions
        / "gemma4_v94_strict_multiscene_full40_validation_question_only.jsonl.access.json",
        metrics / "gemma4_v94_strict_multiscene_full40_validation.json",
        predictions / "v94_strong_causal_question_only.jsonl",
        predictions / "v94_strong_causal_representative_core_question_only.jsonl",
        metrics / "v94_strong_causal_ablations.json",
        metrics / "v94_strong_causal_representative_core_ablations.json",
    ]
    return [path.resolve() for path in paths]


def _oracle_roots_v95() -> list[Path]:
    roots = [PROJECT_ROOT / "data/oracle"]
    roots.extend(PROJECT_ROOT.glob("data*/oracle"))
    return [path.resolve() for path in roots]


def prediction_forbidden_roots_v95(config: Mapping[str, Any]) -> list[Path]:
    roots = _oracle_roots_v95()
    roots.extend(path.resolve() for path in PROJECT_ROOT.glob("data*/qa"))
    roots.append(resolve_v95(config["known_development_gate"]["labels_path"]))
    roots.extend(protected_v94_behavior_paths_v95())
    return list(dict.fromkeys(roots))


def nll_forbidden_roots_v95(config: Mapping[str, Any]) -> list[Path]:
    allowed = resolve_v95(config["known_development_gate"]["labels_path"])
    roots = _oracle_roots_v95() + protected_v94_behavior_paths_v95()
    for directory in PROJECT_ROOT.glob("data*/qa"):
        directory = directory.resolve()
        if allowed.parent != directory:
            roots.append(directory)
            continue
        roots.extend(path.resolve() for path in directory.iterdir() if path.resolve() != allowed)
    roots.append(resolve_v95(config["sources"]["training_qa"]))
    roots.extend(
        resolve_v95(path) for path in config["deferred_final_lock"]["empty_qa_placeholders"]
    )
    return list(dict.fromkeys(roots))


def structured_score_forbidden_roots_v95(
    config: Mapping[str, Any],
) -> list[Path]:
    """Allow the pinned development labels and block every other QA source.

    Structured scoring is intentionally model-free, but it is still a
    label-bearing process.  Give it the same narrow read boundary as the NLL
    evaluator so a future scorer edit cannot silently consult training QA,
    another split, V94 behavior outputs, deferred-final placeholders, or an
    oracle directory.
    """

    return nll_forbidden_roots_v95(config)


def audit_report_v95(audit: FileAccessAudit) -> dict[str, Any]:
    violations = audit.forbidden_accesses()
    return {
        "artifact": "gemma4_v95_file_access_audit_v1",
        "schema_version": SCHEMA_VERSION,
        "loaded_files": audit.unique_paths,
        "loaded_file_inventory_sha256": canonical_sha256_v95(audit.unique_paths),
        "forbidden_roots": [str(path) for path in audit.forbidden_roots],
        "forbidden_component_names": sorted(audit.forbidden_component_names),
        "block_forbidden": audit.block_forbidden,
        "forbidden_accesses": violations,
        "protected_read_count": len(violations),
        "passed": not violations,
    }


def prediction_provenance_v95(
    config_path: str | Path,
    fixed: FixedInputsV95,
    questions: QuestionManifest,
) -> dict[str, Any]:
    value = {
        "artifact": PREDICTION_ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "config_sha256": sha256_file_v85(resolve_v95(config_path)),
        "candidate_fingerprint_sha256": fixed.candidate["fingerprint_sha256"],
        "candidate_state_sha256": fixed.candidate["state_sha256"],
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
    value["provenance_sha256"] = canonical_sha256_v95(value)
    return value


def prediction_row_v95(
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
        raise ValueError("V95 prediction row has an invalid arm or scene inventory")
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
        raise AssertionError("V95 prediction row field contract changed")
    return row


def validate_prediction_rows_v95(
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
            raise ValueError(f"V95 prediction row contract changed: {key}")
        seen.add(key)
        scene_counts[scene_id] += 1
    if (
        len(rows) != QUESTION_COUNT
        or seen != expected
        or scene_counts
        != Counter({scene_id: QUESTIONS_PER_SCENE for scene_id in SCENE_IDS})
    ):
        raise ValueError("V95 predictions do not have exact 216-row coverage")


def mandatory_fixed_input_reads_v95(
    config: Mapping[str, Any],
    fixed: FixedInputsV95,
    *,
    config_path: str | Path = CONFIG,
) -> set[str]:
    candidate = resolve_v95(config["outputs"]["fixed_final_candidate"])
    return {
        str(resolve_v95(config_path)),
        str(resolve_v95(config["outputs"]["training_report"])),
        str(resolve_v95(config["outputs"]["preregistration"])),
        str(resolve_v95(config["outputs"]["cpu_preflight"])),
        str((candidate / "bridge.safetensors").resolve()),
        str((candidate / "runtime_metadata.json").resolve()),
        str((MEMORY_CACHE / "manifest.json").resolve()),
        str(QUESTION_MANIFEST.resolve()),
        *(str(path.resolve()) for path in fixed.memory_paths.values()),
    }


def authenticate_prediction_bundle_v95(
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    """Independently authenticate prediction/access evidence without labels/model."""

    initial = load_config_v95(config_path, allow_draft=False)
    audit = FileAccessAudit(
        prediction_forbidden_roots_v95(initial),
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    with audit:
        config = load_config_v95(config_path, allow_draft=False)
        fixed = authenticate_fixed_inputs_before_questions_v95(
            config, config_path=config_path, audit=audit
        )
        questions = load_known_questions_v95()
        paths = evaluation_paths_v95(config)
        assert_output_bundle_state_v95(
            (
                paths.predictions,
                paths.provenance,
                paths.prediction_access,
                paths.prediction_completion,
            ),
            complete=True,
        )
        provenance = read_json_strict_v95(paths.provenance)
        expected_provenance = prediction_provenance_v95(
            config_path, fixed, questions
        )
        if provenance != expected_provenance:
            raise ValueError("V95 prediction provenance changed")
        rows = read_jsonl(paths.predictions)
        validate_prediction_rows_v95(
            rows,
            questions=questions,
            memory_hashes=fixed.memory_hashes,
            provenance_sha256=provenance["provenance_sha256"],
        )
        access = read_json_strict_v95(paths.prediction_access)
        completion = read_json_strict_v95(paths.prediction_completion)
        current = authenticate_fixed_final_candidate_v95(
            config, config_path=config_path, audit=audit
        )
    audit.assert_clean()
    assert_same_candidate_v95(fixed.candidate, current)
    mandatory = mandatory_fixed_input_reads_v95(
        config, fixed, config_path=config_path
    )
    if (
        access.get("artifact") != "gemma4_v95_file_access_audit_v1"
        or access.get("passed") is not True
        or access.get("forbidden_accesses") != []
        or access.get("protected_read_count") != 0
        or not mandatory <= set(access.get("loaded_files", []))
        or access.get("loaded_file_inventory_sha256")
        != canonical_sha256_v95(access.get("loaded_files"))
        or completion.get("artifact") != PREDICTION_COMPLETION_ARTIFACT
        or completion.get("schema_version") != SCHEMA_VERSION
        or completion.get("candidate_fingerprint_before")
        != fixed.candidate["fingerprint_sha256"]
        or completion.get("candidate_fingerprint_after")
        != fixed.candidate["fingerprint_sha256"]
        or completion.get("candidate_immutable") is not True
        or completion.get("memory_manifest_sha256") != fixed.memory_manifest_sha256
        or completion.get("bound_memory_inventory_sha256")
        != canonical_sha256_v95(fixed.memory_hashes)
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
        raise ValueError("V95 prediction completion/access evidence changed")
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


def assert_aggregate_only_v95(value: object, *, path: tuple[str, ...] = ()) -> None:
    """Reject row-level content before an aggregate score is serialized."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in _ROW_CONTENT_KEYS:
                raise ValueError(
                    "V95 aggregate attempted to serialize row-level content at "
                    + ".".join((*path, str(key)))
                )
            assert_aggregate_only_v95(item, path=(*path, str(key)))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            assert_aggregate_only_v95(item, path=(*path, str(index)))


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"V95 {label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"V95 {label} must be finite")
    return result


def validate_structured_metrics_v95(metrics: Mapping[str, Any]) -> None:
    arms = metrics.get("arms")
    changed = metrics.get("counterfactual")
    comparisons = metrics.get("comparisons")
    if not all(isinstance(value, Mapping) for value in (arms, changed, comparisons)):
        raise TypeError("V95 structured aggregate sections are missing")
    if set(arms) != set(ARMS):
        raise ValueError("V95 structured arm inventory changed")
    for arm in ARMS:
        item = arms[arm]
        if (
            not isinstance(item, Mapping)
            or item.get("total") != QUESTION_COUNT
            or not isinstance(item.get("correct"), int)
            or isinstance(item.get("correct"), bool)
            or not 0 <= item["correct"] <= QUESTION_COUNT
        ):
            raise ValueError("V95 structured arm counts changed")
        accuracy = _finite_number(item.get("accuracy"), f"{arm} accuracy")
        if abs(accuracy - item["correct"] / QUESTION_COUNT) > 1e-12:
            raise ValueError("V95 structured accuracy is inconsistent")
    for key, upper in {
        "canonical_correct_sides": 24,
        "canonical_complete_units": 12,
        "canonical_prediction_changed_units": 12,
    }.items():
        value = changed.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= upper:
            raise ValueError(f"V95 structured counterfactual {key} changed")
    if set(comparisons) != set(ARMS[1:]):
        raise ValueError("V95 structured comparison inventory changed")
    assert_aggregate_only_v95(metrics)


def validate_nll_metrics_v95(metrics: Mapping[str, Any]) -> None:
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
        raise ValueError("V95 NLL aggregate field inventory changed")
    if metrics.get("row_count_per_arm") != QUESTION_COUNT or metrics.get(
        "changed_row_count"
    ) != 24:
        raise ValueError("V95 NLL aggregate row counts changed")
    for key in expected - {"row_count_per_arm", "changed_row_count"}:
        _finite_number(metrics.get(key), key)
    assert_aggregate_only_v95(metrics)


def authenticate_structured_score_v95(
    config_path: str | Path = CONFIG,
    *,
    prediction_bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    bundle = dict(prediction_bundle or authenticate_prediction_bundle_v95(config_path))
    path = bundle["paths"].structured_score
    report = read_json_strict_v95(path)
    expected_fields = {
        "artifact",
        "schema_version",
        "status",
        "candidate_fingerprint_sha256",
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
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise TypeError("V95 structured metrics are missing")
    validate_structured_metrics_v95(metrics)
    if (
        set(report) != expected_fields
        or report.get("artifact") != STRUCTURED_SCORE_ARTIFACT
        or report.get("schema_version") != SCHEMA_VERSION
        or report.get("status") != "measured_aggregate_only_not_yet_gated"
        or report.get("candidate_fingerprint_sha256")
        != bundle["fixed"].candidate["fingerprint_sha256"]
        or report.get("memory_manifest_sha256")
        != bundle["fixed"].memory_manifest_sha256
        or report.get("bound_memory_inventory_sha256")
        != canonical_sha256_v95(bundle["fixed"].memory_hashes)
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
        raise ValueError("V95 structured aggregate authentication failed")
    assert_aggregate_only_v95(report)
    return {"report": report, "sha256": sha256_file_v85(path), "bundle": bundle}


def authenticate_nll_bundle_v95(
    config_path: str | Path = CONFIG,
    *,
    fixed: FixedInputsV95 | None = None,
    questions: QuestionManifest | None = None,
) -> dict[str, Any]:
    """Authenticate aggregate NLL evidence without reopening labels or a model."""

    config = load_config_v95(config_path, allow_draft=False)
    current = fixed or authenticate_fixed_inputs_before_questions_v95(
        config, config_path=config_path
    )
    manifest = questions or load_known_questions_v95()
    paths = evaluation_paths_v95(config)
    assert_output_bundle_state_v95(
        (paths.nll, paths.nll_access, paths.nll_completion), complete=True
    )
    report = read_json_strict_v95(paths.nll)
    access = read_json_strict_v95(paths.nll_access)
    completion = read_json_strict_v95(paths.nll_completion)
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise TypeError("V95 NLL aggregate metrics are missing")
    validate_nll_metrics_v95(metrics)
    expected_report_fields = {
        "artifact",
        "schema_version",
        "status",
        "candidate_fingerprint_sha256",
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
        or report.get("memory_manifest_sha256") != current.memory_manifest_sha256
        or report.get("bound_memory_inventory_sha256")
        != canonical_sha256_v95(current.memory_hashes)
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
        raise ValueError("V95 NLL aggregate authentication failed")
    assert_aggregate_only_v95(report)
    expected_label = resolve_v95(config["known_development_gate"]["labels_path"])
    mandatory = mandatory_fixed_input_reads_v95(
        config, current, config_path=config_path
    ) | {str(expected_label)}
    if (
        access.get("artifact") != "gemma4_v95_file_access_audit_v1"
        or access.get("passed") is not True
        or access.get("forbidden_accesses") != []
        or access.get("protected_read_count") != 0
        or str(expected_label) not in set(access.get("loaded_files", []))
        or not mandatory <= set(access.get("loaded_files", []))
        or access.get("loaded_file_inventory_sha256")
        != canonical_sha256_v95(access.get("loaded_files"))
        or completion.get("artifact") != NLL_COMPLETION_ARTIFACT
        or completion.get("schema_version") != SCHEMA_VERSION
        or completion.get("candidate_fingerprint_before")
        != current.candidate["fingerprint_sha256"]
        or completion.get("candidate_fingerprint_after")
        != current.candidate["fingerprint_sha256"]
        or completion.get("candidate_immutable") is not True
        or completion.get("memory_hashes_invariant") is not True
        or completion.get("nll_sha256") != sha256_file_v85(paths.nll)
        or completion.get("nll_access_sha256") != sha256_file_v85(paths.nll_access)
        or completion.get("row_count_per_arm") != QUESTION_COUNT
        or completion.get("changed_row_count") != 24
        or completion.get("row_level_content_serialized") is not False
        or completion.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V95 NLL completion/access authentication failed")
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


def known_development_gate_results_v95(
    structured_metrics: Mapping[str, Any],
    nll_metrics: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    immutable_fixed_final: bool,
    prefix_invariant: bool,
    label_isolation_proven: bool,
    protected_read_count: int,
) -> dict[str, bool]:
    """Apply every V95 known-development preregistered acceptance condition."""

    validate_structured_metrics_v95(structured_metrics)
    validate_nll_metrics_v95(nll_metrics)
    primary = structured_metrics["arms"][PRIMARY]
    zero = structured_metrics["arms"][ZERO_PAYLOAD]
    permutation = structured_metrics["arms"][FULL_INTERIOR_PERMUTATION]
    changed = structured_metrics["counterfactual"]
    reference_accuracy = float(contract["v94_reference_correct"]) / float(
        contract["v94_reference_total"]
    )
    gates = {
        "post_fixed_final_go_no_go_role_exact": contract.get("role")
        == "post_fixed_final_go_no_go_not_checkpoint_selection",
        "exact_six_scene_216_row_scope": contract.get("scene_count") == 6
        and contract.get("row_count") == QUESTION_COUNT,
        "labels_opened_after_fixed_final_only": bool(
            contract.get("labels_opened_after_fixed_final_only")
        )
        and label_isolation_proven,
        "labels_opened_by_separate_label_isolated_evaluators_only": bool(
            contract.get("labels_opened_by_separate_label_isolated_evaluator_only")
        )
        and label_isolation_proven,
        "v95_correct_minimum": primary["correct"]
        >= int(contract["v95_correct_minimum"]),
        "v95_accuracy_margin_over_v94_minimum": primary["accuracy"]
        >= reference_accuracy + float(contract["v95_accuracy_margin_over_v94_minimum"]),
        "changed_side_correct_minimum": changed["canonical_correct_sides"]
        >= int(contract["changed_side_correct_minimum"]),
        "changed_side_above_v94_reference": changed["canonical_correct_sides"]
        > int(contract["v94_reference_changed_side_correct"]),
        "complete_changed_units_minimum": changed["canonical_complete_units"]
        >= int(contract["complete_changed_units_minimum"]),
        "complete_changed_units_above_v94_reference": changed[
            "canonical_complete_units"
        ]
        > int(contract["v94_reference_complete_changed_units"]),
        "prediction_changed_units_minimum": changed[
            "canonical_prediction_changed_units"
        ]
        >= int(contract["prediction_changed_units_minimum"]),
        "prediction_changed_units_above_v94_reference": changed[
            "canonical_prediction_changed_units"
        ]
        > int(contract["v94_reference_prediction_changed_units"]),
        "mean_changed_side_wrong_minus_correct_nll_minimum": nll_metrics[
            "mean_changed_wrong_minus_primary_nll"
        ]
        >= float(contract["mean_changed_side_wrong_minus_correct_nll_minimum"]),
        "correct_accuracy_above_zero_payload": (
            not bool(contract["correct_accuracy_above_zero_payload_required"])
            or primary["accuracy"] > zero["accuracy"]
        ),
        "correct_accuracy_above_full_interior_permutation": (
            not bool(
                contract["correct_accuracy_above_full_interior_permutation_required"]
            )
            or primary["accuracy"] > permutation["accuracy"]
        ),
        "correct_nll_below_zero_payload": (
            not bool(contract["correct_nll_below_zero_payload_required"])
            or nll_metrics["zero_payload_mean_nll_gap"] > 0.0
        ),
        "correct_nll_below_full_interior_permutation": (
            not bool(
                contract["correct_nll_below_full_interior_permutation_required"]
            )
            or nll_metrics["full_interior_permutation_mean_nll_gap"] > 0.0
        ),
        "fixed_final_checkpoint_immutable": immutable_fixed_final
        and bool(contract["fixed_final_checkpoint_may_not_change_after_gate"]),
        "question_independent_prefix_hash_invariance": prefix_invariant,
        "protected_read_count_zero": protected_read_count == 0,
        "full_known_development_coverage": primary["total"]
        == int(contract["row_count"]),
        "pass_required_before_deferred_final_unlock_contract": bool(
            contract.get("pass_required_before_deferred_final_unlock")
        ),
    }
    if not all(isinstance(value, bool) for value in gates.values()):
        raise AssertionError("V95 gate results must be booleans")
    return gates


__all__ = [
    "ARMS",
    "EVIDENCE_ARTIFACT",
    "FINAL_SCORE_ARTIFACT",
    "FULL_INTERIOR_PERMUTATION",
    "MEMORY_SHAPE",
    "NLL_ARTIFACT",
    "NLL_COMPLETION_ARTIFACT",
    "PAIRED_WRONG_SCENE",
    "PAIR_SCENE",
    "PREDICTION_ARTIFACT",
    "PREDICTION_COMPLETION_ARTIFACT",
    "PREDICTION_FIELDS",
    "PRIMARY",
    "QUESTION_COUNT",
    "QUESTION_MANIFEST",
    "REFERENCE_SHA256",
    "SCENE_IDS",
    "SCHEMA_VERSION",
    "STRUCTURED_SCORE_ARTIFACT",
    "ZERO_PAYLOAD",
    "EvaluationPathsV95",
    "FixedInputsV95",
    "assert_aggregate_only_v95",
    "assert_output_bundle_state_v95",
    "assert_same_candidate_v95",
    "audit_report_v95",
    "authenticate_fixed_final_candidate_v95",
    "authenticate_fixed_inputs_before_questions_v95",
    "authenticate_nll_bundle_v95",
    "authenticate_prediction_bundle_v95",
    "authenticate_structured_score_v95",
    "bind_all_memories_v95",
    "canonical_sha256_v95",
    "evaluation_paths_v95",
    "known_development_gate_results_v95",
    "load_known_questions_v95",
    "mandatory_fixed_input_reads_v95",
    "nll_forbidden_roots_v95",
    "prediction_forbidden_roots_v95",
    "prediction_provenance_v95",
    "prediction_row_v95",
    "read_json_strict_v95",
    "require_sha256_v95",
    "resolve_v95",
    "structured_score_forbidden_roots_v95",
    "validate_nll_metrics_v95",
    "validate_prediction_rows_v95",
    "validate_structured_metrics_v95",
    "write_json_create_once_v95",
    "write_jsonl_create_once_v95",
]
