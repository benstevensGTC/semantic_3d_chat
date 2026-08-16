"""Shared label-blind contracts for V96's deferred final evaluation."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.runtime_config import (
    effective_runtime_config_sha256,
    load_runtime_config,
)
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.evaluation.question_manifest import (
    QuestionManifest,
    load_question_manifest,
)
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import (
    sha256_file_v85,
)
from semantic_3d_chat.evaluation.v95_deferred_final_materialization import (
    MEMORY_ROOT,
    QUESTION_MANIFEST,
    SELECTION_MANIFEST,
    WORK_ROOT,
)
from semantic_3d_chat.evaluation.v95_deferred_final_qa import FINAL_QA, RAW_QA
from semantic_3d_chat.evaluation.v95_strict_causal_successor_preflight import (
    permuted_payload_memory_v95,
    zero_payload_memory_v95,
)
from semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight import (
    CONFIG,
    load_config_v96,
)
from semantic_3d_chat.evaluation.v96_deferred_final_evaluation import (
    PAIR_SCENE,
    QUESTION_COUNT,
    ROWS_PER_SCENE,
    SCENE_IDS,
    authenticate_materialized_inputs_v96_final,
    output_paths_v96_final,
)
from semantic_3d_chat.evaluation.v96_known_development_common_v2 import (
    ARMS,
    FULL_INTERIOR_PERMUTATION,
    PAIRED_WRONG_SCENE,
    PERMUTATION_SEED,
    PRIMARY,
    ZERO_PAYLOAD,
    assert_aggregate_only_v96,
    assert_output_bundle_state_v96,
    authenticate_fixed_final_candidate_v96,
    canonical_sha256_v96,
    read_json_strict_v96,
    write_json_create_once_v96,
    write_jsonl_create_once_v96,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.scene_encoder.v81_scene_memory_artifact import (
    MEMORY_FILENAME,
    METADATA_FILENAME,
    load_v81_scene_memory,
)

SCHEMA_VERSION: Final[int] = 96
MEMORY_SHAPE: Final[tuple[int, int, int]] = (1, 738, 1536)
V85_CHECKPOINT: Final[Path] = PROJECT_ROOT / "reports/gemma4/artifacts/v85_strict_runtime_candidate"
RUNTIME_CONFIG: Final[Path] = PROJECT_ROOT / "configs/runtime/gemma4_v85_strict_multiscene.yaml"

V96_PREDICTION_ARTIFACT: Final[str] = "gemma4_v96_deferred_final_question_only_predictions_v1"
V94_PREDICTION_ARTIFACT: Final[str] = (
    "gemma4_v94_deferred_final_same_rows_question_only_predictions_v1"
)
PREDICTION_COMPLETION_ARTIFACT: Final[str] = "gemma4_v96_deferred_final_prediction_completion_v1"
V94_COMPLETION_ARTIFACT: Final[str] = "gemma4_v94_deferred_final_same_rows_completion_v1"
STRUCTURED_SCORE_ARTIFACT: Final[str] = "gemma4_v96_deferred_final_structured_score_v1"
NLL_ARTIFACT: Final[str] = "gemma4_v96_deferred_final_nll_aggregate_v1"
NLL_COMPLETION_ARTIFACT: Final[str] = "gemma4_v96_deferred_final_nll_completion_v1"
FINAL_SCORE_ARTIFACT: Final[str] = "gemma4_v96_deferred_final_gate_v1"
EVIDENCE_ARTIFACT: Final[str] = "gemma4_v96_deferred_final_evidence_v1"

V96_ROW_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "artifact",
        "schema_version",
        "scene_id",
        "question_id",
        "paired_scene_id",
        *(f"{arm}_prediction" for arm in ARMS),
        *(f"{arm}_memory_sha256" for arm in ARMS),
        "all_memory_hashes_unchanged",
        "provenance_sha256",
    }
)
V94_ROW_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "artifact",
        "schema_version",
        "scene_id",
        "question_id",
        "memory_sha256",
        "prediction",
        "provenance_sha256",
    }
)


@dataclass(frozen=True)
class FixedFinalInputs:
    candidate: Mapping[str, Any]
    memories: Mapping[str, Mapping[str, torch.Tensor]]
    memory_hashes: Mapping[str, Mapping[str, str]]
    memory_inventory_sha256: str
    memory_paths: Mapping[str, Path]
    materialized: Mapping[str, Any]


def _strict_memory_metadata(path: Path) -> dict[str, Any]:
    return read_json_strict_v96(path)


def bind_all_memories_v96_final(
    source: Mapping[str, torch.Tensor],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict[str, str]]]:
    if tuple(source) != SCENE_IDS:
        raise ValueError("V96 final memory source is not the six sealed scenes")
    primary = {scene: source[scene] for scene in SCENE_IDS}
    zero = {scene: zero_payload_memory_v95(source[scene]) for scene in SCENE_IDS}
    permutation = {
        scene: permuted_payload_memory_v95(source[scene], seed=PERMUTATION_SEED)
        for scene in SCENE_IDS
    }
    wrong = {scene: source[PAIR_SCENE[scene]] for scene in SCENE_IDS}
    memories = {
        PRIMARY: primary,
        ZERO_PAYLOAD: zero,
        FULL_INTERIOR_PERMUTATION: permutation,
        PAIRED_WRONG_SCENE: wrong,
    }
    hashes = {
        arm: {scene: prefix_sha256(memories[arm][scene]) for scene in SCENE_IDS} for arm in ARMS
    }
    if (
        tuple(memories) != ARMS
        or any(tuple(values) != SCENE_IDS for values in memories.values())
        or any(
            tuple(memory.shape) != MEMORY_SHAPE
            or memory.dtype != torch.bfloat16
            or not bool(torch.isfinite(memory).all())
            for values in memories.values()
            for memory in values.values()
        )
        or any(
            torch.equal(permutation[scene][:, 1:-1], primary[scene][:, 1:-1]) for scene in SCENE_IDS
        )
        or any(
            not torch.equal(primary[scene][:, :1], permutation[scene][:, :1])
            or not torch.equal(primary[scene][:, -1:], permutation[scene][:, -1:])
            for scene in SCENE_IDS
        )
    ):
        raise RuntimeError("V96 final four-arm memory binding changed")
    return memories, hashes


def authenticate_fixed_inputs_before_questions_v96_final(
    *, audit: FileAccessAudit | None = None
) -> FixedFinalInputs:
    """Load all six memories and bind all 24 arms before question I/O."""

    materialized = authenticate_materialized_inputs_v96_final(label_process=False)
    config = load_config_v96(CONFIG, allow_draft=False)
    candidate = authenticate_fixed_final_candidate_v96(config, config_path=CONFIG, audit=audit)
    checkpoint_sha, _files = checkpoint_fingerprint(V85_CHECKPOINT)
    runtime_sha = effective_runtime_config_sha256(load_runtime_config(RUNTIME_CONFIG))
    source: dict[str, torch.Tensor] = {}
    paths: dict[str, Path] = {}
    inventory: dict[str, Any] = {}
    for scene in SCENE_IDS:
        root = MEMORY_ROOT / scene
        metadata_path = root / METADATA_FILENAME
        metadata = _strict_memory_metadata(metadata_path)
        if (
            metadata.get("source_base_checkpoint_sha256") != checkpoint_sha
            or metadata.get("runtime_config_sha256") != runtime_sha
        ):
            raise ValueError(f"V96 final memory compiler binding changed: {scene}")
        loaded = load_v81_scene_memory(
            root,
            expected_scene_id=scene,
            expected_base_checkpoint_sha256=checkpoint_sha,
            expected_runtime_config_sha256=runtime_sha,
            expected_model_device="cpu",
            record_file=audit.record if audit is not None else None,
        )
        source[scene] = loaded.memory
        paths[scene] = root
        inventory[scene] = {
            "tensor_file_sha256": sha256_file_v85(root / MEMORY_FILENAME),
            "metadata_file_sha256": sha256_file_v85(metadata_path),
            "canonical_prefix_sha256": loaded.metadata["canonical_prefix_sha256"],
        }
    memories, hashes = bind_all_memories_v96_final(source)
    if any(
        hashes[PRIMARY][scene] != inventory[scene]["canonical_prefix_sha256"] for scene in SCENE_IDS
    ):
        raise ValueError("V96 final primary prefixes differ from V81 metadata")
    return FixedFinalInputs(
        candidate=candidate,
        memories=memories,
        memory_hashes=hashes,
        memory_inventory_sha256=canonical_sha256_v96(inventory),
        memory_paths=paths,
        materialized=materialized,
    )


def load_questions_v96_final(
    fixed: FixedFinalInputs | None = None,
) -> QuestionManifest:
    if fixed is None:
        raise ValueError("V96 final questions require pre-bound memories")
    manifest = load_question_manifest(QUESTION_MANIFEST)
    counts = Counter(row.scene_id for row in manifest.questions)
    receipt = fixed.materialized["receipts"]["questions"]
    expected_sha = receipt["output_sha256"][QUESTION_MANIFEST.relative_to(PROJECT_ROOT).as_posix()]
    if (
        manifest.manifest_sha256 != expected_sha
        or manifest.question_count != QUESTION_COUNT
        or manifest.scene_count != len(SCENE_IDS)
        or tuple(dict.fromkeys(row.scene_id for row in manifest.questions)) != SCENE_IDS
        or counts != Counter({scene: ROWS_PER_SCENE for scene in SCENE_IDS})
    ):
        raise ValueError("V96 final sanitized question manifest changed")
    return manifest


def _other_question_manifests() -> list[Path]:
    root = PROJECT_ROOT / "reports/gemma4/questions"
    if not root.is_dir():
        return []
    return [
        path.resolve() for path in root.iterdir() if path.resolve() != QUESTION_MANIFEST.resolve()
    ]


def prediction_forbidden_roots_v96_final() -> list[Path]:
    roots = [path.resolve() for path in PROJECT_ROOT.glob("data*/oracle")]
    roots.extend(path.resolve() for path in PROJECT_ROOT.glob("data*/qa"))
    roots.extend(
        (PROJECT_ROOT / value).resolve()
        for value in (
            "data/rendered",
            "data_gemma4/features",
            "data_gemma4/maps",
            "configs/experiments/gemma4_v95_deferred_final_materialization.yaml",
            "configs/experiments/diverse20.yaml",
            "configs/experiments/diverse28.yaml",
            "configs/experiments/diverse52.yaml",
            str(RAW_QA.relative_to(PROJECT_ROOT)),
            str(SELECTION_MANIFEST.relative_to(PROJECT_ROOT)),
        )
    )
    roots.extend(_other_question_manifests())
    roots.append((WORK_ROOT / "qa_raw").resolve())
    return list(dict.fromkeys(roots))


def score_forbidden_roots_v96_final() -> list[Path]:
    allowed = FINAL_QA.resolve()
    roots = [path.resolve() for path in PROJECT_ROOT.glob("data*/oracle")]
    for directory in PROJECT_ROOT.glob("data*/qa"):
        resolved = directory.resolve()
        if allowed.parent != resolved:
            roots.append(resolved)
        else:
            roots.extend(
                path.resolve() for path in directory.iterdir() if path.resolve() != allowed
            )
    roots.extend(
        (PROJECT_ROOT / value).resolve()
        for value in (
            "data/rendered",
            "data_gemma4/features",
            "data_gemma4/maps",
        )
    )
    return list(dict.fromkeys(roots))


def audit_report_v96_final(
    audit: FileAccessAudit, *, question_path: Path | None = None, memory_paths: Sequence[Path] = ()
) -> dict[str, Any]:
    violations = audit.forbidden_accesses()
    ordered = list(audit.paths)
    memory_strings = {str(path.resolve()) for path in memory_paths}
    question_string = str(question_path.resolve()) if question_path is not None else None
    all_before = True
    if question_string is not None and question_string in ordered:
        first_question = ordered.index(question_string)
        all_before = all(
            any(item == memory for item in ordered[:first_question]) for memory in memory_strings
        )
    elif question_string is not None:
        all_before = False
    return {
        "artifact": "gemma4_v96_deferred_final_file_access_audit_v1",
        "schema_version": SCHEMA_VERSION,
        "loaded_files": audit.unique_paths,
        "loaded_files_in_order": ordered,
        "loaded_file_inventory_sha256": canonical_sha256_v96(audit.unique_paths),
        "forbidden_roots": [str(path) for path in audit.forbidden_roots],
        "forbidden_component_names": sorted(audit.forbidden_component_names),
        "block_forbidden": audit.block_forbidden,
        "forbidden_accesses": violations,
        "protected_read_count": len(violations),
        "all_six_memory_tensors_opened_before_question_manifest": all_before,
        "passed": not violations and all_before,
    }


def prediction_provenance_v96_final(
    *, model: str, fixed: FixedFinalInputs, questions: QuestionManifest
) -> dict[str, Any]:
    if model not in {"v96", "v94"}:
        raise ValueError("Unknown V96 final predictor model")
    model_identity = (
        {"v96_state_sha256": fixed.candidate["state_sha256"]}
        if model == "v96"
        else dict(fixed.materialized["preregistration"]["v94_same_row_comparator"])
    )
    payload = {
        "artifact": f"gemma4_{model}_deferred_final_prediction_provenance_v1",
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "model_identity": model_identity,
        "candidate_fingerprint_sha256": fixed.candidate["fingerprint_sha256"],
        "candidate_attestation_file_sha256": fixed.candidate[
            "attestation_file_sha256"
        ],
        "candidate_attestation_identity_sha256": fixed.candidate[
            "attestation_identity_sha256"
        ],
        "v2_implementation_seal_sha256": fixed.candidate[
            "v2_implementation_seal_sha256"
        ],
        "memory_inventory_sha256": fixed.memory_inventory_sha256,
        "bound_memory_inventory_sha256": canonical_sha256_v96(fixed.memory_hashes),
        "question_manifest_sha256": questions.manifest_sha256,
        "questions_sha256": questions.questions_sha256,
        "source_qa_sha256": questions.source_qa_sha256,
        "scene_ids": list(SCENE_IDS),
        "row_count": QUESTION_COUNT,
        "all_six_memories_bound_before_questions": True,
        "question_dependent_retrieval": False,
        "environmental_text_inputs": [],
        "labels_loaded": False,
        "oracle_loaded": False,
        "runtime_promotion_authorized": False,
    }
    payload["provenance_sha256"] = canonical_sha256_v96(payload)
    return payload


def validate_prediction_rows_v96_final(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: str,
    fixed: FixedFinalInputs,
    questions: QuestionManifest,
    provenance_sha256: str,
) -> None:
    expected = [(row.scene_id, row.question_id) for row in questions.questions]
    if len(rows) != QUESTION_COUNT or len({tuple(item) for item in expected}) != QUESTION_COUNT:
        raise ValueError("V96 final prediction question inventory changed")
    for raw, key in zip(rows, expected, strict=True):
        if model == "v96":
            if (
                set(raw) != V96_ROW_FIELDS
                or raw.get("artifact") != V96_PREDICTION_ARTIFACT
                or raw.get("schema_version") != SCHEMA_VERSION
                or (raw.get("scene_id"), raw.get("question_id")) != key
                or raw.get("paired_scene_id") != PAIR_SCENE[key[0]]
                or raw.get("all_memory_hashes_unchanged") is not True
                or any(
                    raw.get(f"{arm}_memory_sha256") != fixed.memory_hashes[arm][key[0]]
                    for arm in ARMS
                )
                or any(
                    not isinstance(raw.get(f"{arm}_prediction"), str)
                    or not raw[f"{arm}_prediction"].strip()
                    for arm in ARMS
                )
            ):
                raise ValueError(f"V96 final prediction row changed: {key}")
        elif model == "v94":
            if (
                set(raw) != V94_ROW_FIELDS
                or raw.get("artifact") != V94_PREDICTION_ARTIFACT
                or raw.get("schema_version") != SCHEMA_VERSION
                or (raw.get("scene_id"), raw.get("question_id")) != key
                or raw.get("memory_sha256") != fixed.memory_hashes[PRIMARY][key[0]]
                or not isinstance(raw.get("prediction"), str)
                or not raw["prediction"].strip()
            ):
                raise ValueError(f"V94 same-row comparator row changed: {key}")
        else:
            raise ValueError("Unknown V96 final prediction model")
        if raw.get("provenance_sha256") != provenance_sha256:
            raise ValueError("V96 final prediction provenance changed")


def _prediction_bundle_paths(model: str) -> tuple[Path, Path, Path, Path]:
    paths = output_paths_v96_final()
    prefix = "v96" if model == "v96" else "v94"
    return (
        paths[f"{prefix}_predictions"],
        paths[f"{prefix}_prediction_provenance"],
        paths[f"{prefix}_prediction_access"],
        paths[f"{prefix}_prediction_completion"],
    )


def authenticate_prediction_bundle_v96_final(model: str) -> dict[str, Any]:
    fixed = authenticate_fixed_inputs_before_questions_v96_final()
    questions = load_questions_v96_final(fixed)
    predictions, provenance_path, access_path, completion_path = _prediction_bundle_paths(model)
    assert_output_bundle_state_v96(
        (predictions, provenance_path, access_path, completion_path), complete=True
    )
    provenance = read_json_strict_v96(provenance_path)
    expected_provenance = prediction_provenance_v96_final(
        model=model, fixed=fixed, questions=questions
    )
    if provenance != expected_provenance:
        raise ValueError(f"V96 final {model} prediction provenance changed")
    rows = []
    for line in predictions.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("V96 final prediction row is not an object")
            rows.append(value)
    validate_prediction_rows_v96_final(
        rows,
        model=model,
        fixed=fixed,
        questions=questions,
        provenance_sha256=provenance["provenance_sha256"],
    )
    access = read_json_strict_v96(access_path)
    completion = read_json_strict_v96(completion_path)
    completion_artifact = (
        PREDICTION_COMPLETION_ARTIFACT if model == "v96" else V94_COMPLETION_ARTIFACT
    )
    if (
        access.get("artifact") != "gemma4_v96_deferred_final_file_access_audit_v1"
        or access.get("passed") is not True
        or access.get("protected_read_count") != 0
        or access.get("all_six_memory_tensors_opened_before_question_manifest") is not True
        or completion.get("artifact") != completion_artifact
        or completion.get("schema_version") != SCHEMA_VERSION
        or completion.get("model") != model
        or completion.get("prediction_sha256") != sha256_file_v85(predictions)
        or completion.get("provenance_file_sha256") != sha256_file_v85(provenance_path)
        or completion.get("access_sha256") != sha256_file_v85(access_path)
        or completion.get("candidate_fingerprint_before") != fixed.candidate["fingerprint_sha256"]
        or completion.get("candidate_fingerprint_after") != fixed.candidate["fingerprint_sha256"]
        or completion.get("candidate_attestation_file_sha256")
        != fixed.candidate["attestation_file_sha256"]
        or completion.get("candidate_attestation_identity_sha256")
        != fixed.candidate["attestation_identity_sha256"]
        or completion.get("v2_implementation_seal_sha256")
        != fixed.candidate["v2_implementation_seal_sha256"]
        or completion.get("all_memory_hashes_invariant") is not True
        or completion.get("all_memories_bound_before_questions") is not True
        or completion.get("row_count") != QUESTION_COUNT
        or completion.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError(f"V96 final {model} prediction completion changed")
    return {
        "model": model,
        "fixed": fixed,
        "questions": questions,
        "rows": rows,
        "provenance": provenance,
        "access": access,
        "completion": completion,
        "prediction_sha256": sha256_file_v85(predictions),
        "provenance_file_sha256": sha256_file_v85(provenance_path),
        "access_sha256": sha256_file_v85(access_path),
        "completion_sha256": sha256_file_v85(completion_path),
        "paths": _prediction_bundle_paths(model),
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


def assert_aggregate_only_v96_final(value: object, *, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in _ROW_CONTENT_KEYS:
                raise ValueError(
                    "V96 final aggregate serialized row content at " + ".".join((*path, str(key)))
                )
            assert_aggregate_only_v96_final(child, path=(*path, str(key)))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            assert_aggregate_only_v96_final(child, path=(*path, str(index)))
    assert_aggregate_only_v96(value)


def finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"V96 final {label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"V96 final {label} must be finite")
    return result


__all__ = [
    "ARMS",
    "EVIDENCE_ARTIFACT",
    "FINAL_SCORE_ARTIFACT",
    "FULL_INTERIOR_PERMUTATION",
    "NLL_ARTIFACT",
    "NLL_COMPLETION_ARTIFACT",
    "PAIRED_WRONG_SCENE",
    "PREDICTION_COMPLETION_ARTIFACT",
    "PRIMARY",
    "SCHEMA_VERSION",
    "STRUCTURED_SCORE_ARTIFACT",
    "V94_COMPLETION_ARTIFACT",
    "V94_PREDICTION_ARTIFACT",
    "V96_PREDICTION_ARTIFACT",
    "ZERO_PAYLOAD",
    "FixedFinalInputs",
    "assert_aggregate_only_v96_final",
    "audit_report_v96_final",
    "authenticate_fixed_inputs_before_questions_v96_final",
    "authenticate_prediction_bundle_v96_final",
    "bind_all_memories_v96_final",
    "finite_number",
    "load_questions_v96_final",
    "prediction_forbidden_roots_v96_final",
    "prediction_provenance_v96_final",
    "score_forbidden_roots_v96_final",
    "validate_prediction_rows_v96_final",
    "write_json_create_once_v96",
    "write_jsonl_create_once_v96",
]
