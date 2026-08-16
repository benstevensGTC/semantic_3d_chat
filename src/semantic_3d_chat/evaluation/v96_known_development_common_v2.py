"""Inference-safe shared contracts for V96 known-development evaluation v2.

The scientific experiment and every threshold remain V96.  This module only
repairs authentication: the predictor consumes a create-once aggregate
attestation and directly revalidates candidate bytes instead of recursively
rehashing training QA while an inference file audit is active.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.evaluation import v96_known_development_common as v1
from semantic_3d_chat.evaluation.score_v96_known_development import (
    load_references_v96 as load_references_v96_v1,
)
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import sha256_file_v85
from semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight import (
    CONFIG,
    FRESH_BANK_NAME,
)
from semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight import (
    load_config_v96 as load_config_v96_v1,
)
from semantic_3d_chat.evaluation.v96_evaluation_io_v2 import (
    physical_path_v96_v2,
    read_json_strict_v96_v2,
    read_jsonl_strict_v96_v2,
    write_json_create_once_v96_v2,
    write_jsonl_create_once_v96_v2,
)
from semantic_3d_chat.evaluation.v96_known_development_candidate_attestation import (
    ATTESTATION,
    authenticate_candidate_attestation_v96,
)
from semantic_3d_chat.evaluation.v96_known_development_implementation_v2 import (
    IMPLEMENTATION_SEAL_V2,
    authenticate_evaluation_implementation_v96_v2,
)

# Exact experiment constants and pure validators stay byte-for-byte V96 v1.
ARMS = v1.ARMS
CHANGED_SIDE_COUNT = v1.CHANGED_SIDE_COUNT
CHANGED_UNIT_COUNT = v1.CHANGED_UNIT_COUNT
EVIDENCE_ARTIFACT = v1.EVIDENCE_ARTIFACT
FINAL_SCORE_ARTIFACT = v1.FINAL_SCORE_ARTIFACT
FULL_INTERIOR_PERMUTATION = v1.FULL_INTERIOR_PERMUTATION
INVARIANT_SIDE_COUNT = v1.INVARIANT_SIDE_COUNT
INVARIANT_UNIT_COUNT = v1.INVARIANT_UNIT_COUNT
MEMORY_DTYPE = v1.MEMORY_DTYPE
MEMORY_SHAPE = v1.MEMORY_SHAPE
NLL_ARTIFACT = v1.NLL_ARTIFACT
NLL_COMPLETION_ARTIFACT = v1.NLL_COMPLETION_ARTIFACT
PAIRED_WRONG_SCENE = v1.PAIRED_WRONG_SCENE
PAIR_SCENE = v1.PAIR_SCENE
PERMUTATION_SEED = v1.PERMUTATION_SEED
PREDICTION_ARTIFACT = v1.PREDICTION_ARTIFACT
PREDICTION_COMPLETION_ARTIFACT = v1.PREDICTION_COMPLETION_ARTIFACT
PREDICTION_FIELDS = v1.PREDICTION_FIELDS
PRIMARY = v1.PRIMARY
QUESTION_COUNT = v1.QUESTION_COUNT
QUESTIONS_SHA256 = v1.QUESTIONS_SHA256
QUESTION_MANIFEST = v1.QUESTION_MANIFEST
QUESTION_MANIFEST_SHA256 = v1.QUESTION_MANIFEST_SHA256
QUESTIONS_PER_SCENE = v1.QUESTIONS_PER_SCENE
REFERENCE_SHA256 = v1.REFERENCE_SHA256
SCENE_IDS = v1.SCENE_IDS
SCHEMA_VERSION = v1.SCHEMA_VERSION
STRUCTURED_SCORE_ARTIFACT = v1.STRUCTURED_SCORE_ARTIFACT
ZERO_PAYLOAD = v1.ZERO_PAYLOAD
EvaluationPathsV96 = v1.EvaluationPathsV96
FixedInputsV96 = v1.FixedInputsV96

assert_aggregate_only_v96 = v1.assert_aggregate_only_v96
assert_bound_config_path_v96 = v1.assert_bound_config_path_v96
assert_output_bundle_state_v96 = v1.assert_output_bundle_state_v96
assert_same_candidate_v96 = v1.assert_same_candidate_v96
audit_report_v96 = v1.audit_report_v96
bind_all_memories_v96 = v1.bind_all_memories_v96
canonical_sha256_v96 = v1.canonical_sha256_v96
evaluation_paths_v96 = v1.evaluation_paths_v96
known_development_gate_results_v96 = v1.known_development_gate_results_v96
load_future_trainer_v96 = v1.load_future_trainer_v96
nll_forbidden_roots_v96 = v1.nll_forbidden_roots_v96
prediction_forbidden_roots_v96 = v1.prediction_forbidden_roots_v96
prediction_row_v96 = v1.prediction_row_v96
read_json_strict_v96 = read_json_strict_v96_v2
require_sha256_v96 = v1.require_sha256_v96
resolve_v96 = v1.resolve_v96
structured_score_forbidden_roots_v96 = v1.structured_score_forbidden_roots_v96
validate_nll_metrics_v96 = v1.validate_nll_metrics_v96
validate_prediction_rows_v96 = v1.validate_prediction_rows_v96
validate_structured_metrics_v96 = v1.validate_structured_metrics_v96
write_json_create_once_v96 = write_json_create_once_v96_v2
write_jsonl_create_once_v96 = write_jsonl_create_once_v96_v2

_ACCESS_RECEIPT_FIELDS = frozenset(
    {
        "artifact",
        "schema_version",
        "loaded_files",
        "loaded_file_inventory_sha256",
        "forbidden_roots",
        "forbidden_component_names",
        "block_forbidden",
        "forbidden_accesses",
        "protected_read_count",
        "passed",
    }
)
_PREDICTION_COMPLETION_FIELDS = frozenset(
    {
        "artifact",
        "schema_version",
        "candidate_fingerprint_before",
        "candidate_fingerprint_after",
        "candidate_attestation_file_sha256",
        "model_snapshot_inventory_sha256",
        "model_snapshot_hashes_invariant",
        "lora_bank_state_inventory_sha256",
        "lora_bank_topology_sha256",
        "lora_bank_states_invariant",
        "candidate_immutable",
        "frozen_v95_parent_immutable",
        "memory_manifest_sha256",
        "bound_memory_inventory_sha256",
        "question_manifest_sha256",
        "questions_sha256",
        "prediction_provenance_sha256",
        "prediction_provenance_file_sha256",
        "prediction_sha256",
        "prediction_access_sha256",
        "row_count",
        "scene_count",
        "arms",
        "all_memories_bound_before_questions",
        "all_memory_hashes_invariant",
        "implementation_reauthenticated_inside_audit",
        "training_qa_opened",
        "labels_opened",
        "oracle_opened",
        "protected_read_count",
        "runtime_promotion_authorized",
        "elapsed_seconds",
    }
)
_NLL_COMPLETION_FIELDS = frozenset(
    {
        "artifact",
        "schema_version",
        "candidate_fingerprint_before",
        "candidate_fingerprint_after",
        "candidate_attestation_file_sha256",
        "model_snapshot_inventory_sha256",
        "model_snapshot_hashes_invariant",
        "lora_bank_state_inventory_sha256",
        "lora_bank_topology_sha256",
        "lora_bank_states_invariant",
        "candidate_immutable",
        "frozen_v95_parent_immutable",
        "memory_hashes_invariant",
        "implementation_reauthenticated_inside_audit",
        "nll_sha256",
        "nll_access_sha256",
        "row_count_per_arm",
        "changed_row_count",
        "row_level_content_serialized",
        "runtime_promotion_authorized",
    }
)


def validate_access_receipt_v96_v2(
    access: Mapping[str, Any],
    *,
    forbidden_roots: list[Path],
    mandatory: set[str],
) -> None:
    loaded = access.get("loaded_files")
    expected_forbidden = [str(Path(path).resolve()) for path in forbidden_roots]
    if (
        set(access) != _ACCESS_RECEIPT_FIELDS
        or access.get("artifact") != "gemma4_v96_file_access_audit_v1"
        or access.get("schema_version") != SCHEMA_VERSION
        or not isinstance(loaded, list)
        or any(not isinstance(path, str) for path in loaded)
        or loaded != sorted(set(loaded))
        or any(not Path(path).is_absolute() or str(Path(path).resolve()) != path for path in loaded)
        or access.get("loaded_file_inventory_sha256")
        != canonical_sha256_v96(loaded)
        or access.get("forbidden_roots") != expected_forbidden
        or access.get("forbidden_component_names") != ["oracle"]
        or access.get("block_forbidden") is not True
        or access.get("forbidden_accesses") != []
        or access.get("protected_read_count") != 0
        or access.get("passed") is not True
        or not mandatory <= set(loaded)
    ):
        raise ValueError("V96 v2 file-access receipt changed")
    assert_aggregate_only_v96(access)


def validate_prediction_completion_schema_v96_v2(
    completion: Mapping[str, Any],
) -> None:
    elapsed = completion.get("elapsed_seconds")
    if (
        set(completion) != _PREDICTION_COMPLETION_FIELDS
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise ValueError("V96 v2 prediction completion schema changed")
    assert_aggregate_only_v96(completion)


def validate_nll_completion_schema_v96_v2(
    completion: Mapping[str, Any],
) -> None:
    if set(completion) != _NLL_COMPLETION_FIELDS:
        raise ValueError("V96 v2 NLL completion schema changed")
    assert_aggregate_only_v96(completion)


def load_config_v96_v2(
    path: str | Path = CONFIG, *, allow_draft: bool = False
) -> dict[str, Any]:
    return load_config_v96_v1(
        physical_path_v96_v2(path), allow_draft=allow_draft
    )


def load_known_questions_v96() -> Any:
    physical_path_v96_v2(QUESTION_MANIFEST)
    return v1.load_known_questions_v95()


def load_references_v96_v2(
    config: Mapping[str, Any], questions: Any
) -> list[dict[str, Any]]:
    """Strictly prevalidate the label bytes before V1's semantic validator."""

    source = physical_path_v96_v2(
        config["known_development_gate"]["labels_path"]
    )
    if (
        not source.is_file()
        or sha256_file_v85(source) != REFERENCE_SHA256
        or questions.source_qa_sha256 != REFERENCE_SHA256
    ):
        raise ValueError("V96 v2 known-development reference bytes changed")
    strict_rows = read_jsonl_strict_v96_v2(source)
    validated = load_references_v96_v1(config, questions)
    if strict_rows != validated or len(strict_rows) != QUESTION_COUNT:
        raise ValueError("V96 v2 known-development reference schema changed")
    return strict_rows


def authenticate_memory_cache_v96(
    *, audit: FileAccessAudit | None = None
) -> tuple[dict[str, Any], dict[str, Path], str]:
    physical_path_v96_v2(v1.MEMORY_CACHE / "manifest.json")
    for scene_id in SCENE_IDS:
        physical_path_v96_v2(v1.MEMORY_CACHE / f"{scene_id}.safetensors")
    return v1.authenticate_memory_cache_v95(audit=audit)


def authenticate_fixed_final_candidate_v96(
    config: Mapping[str, Any],
    *,
    config_path: str | Path = CONFIG,
    audit: FileAccessAudit | None = None,
    implementation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Revalidate attestation/candidate only; never walk training sources."""

    current = implementation or authenticate_evaluation_implementation_v96_v2(
        config_path=config_path
    )
    return authenticate_candidate_attestation_v96(
        config_path,
        audit=audit,
        authenticate_implementation_sources=False,
        expected_implementation_seal_sha256=str(current["seal_sha256"]),
    )


def authenticate_fixed_inputs_before_questions_v96(
    config: Mapping[str, Any],
    *,
    config_path: str | Path = CONFIG,
    audit: FileAccessAudit | None = None,
    implementation: Mapping[str, Any] | None = None,
) -> FixedInputsV96:
    candidate = authenticate_fixed_final_candidate_v96(
        config,
        config_path=config_path,
        audit=audit,
        implementation=implementation,
    )
    source, paths, manifest_sha256 = authenticate_memory_cache_v96(audit=audit)
    memories, hashes = bind_all_memories_v96(source)
    return FixedInputsV96(
        candidate=candidate,
        memories=memories,
        memory_hashes=hashes,
        memory_manifest_sha256=manifest_sha256,
        memory_paths=paths,
    )


def expected_lora_bank_states_v96_v2(
    fixed: FixedInputsV96,
    implementation: Mapping[str, Any],
) -> dict[str, str]:
    frozen = implementation.get("frozen_bank_expected_states")
    if (
        not isinstance(frozen, Mapping)
        or len(frozen) != 9
        or fixed.candidate.get("frozen_bank_expected_state_inventory_sha256")
        != canonical_sha256_v96(frozen)
        or implementation.get("frozen_bank_expected_state_inventory_sha256")
        != canonical_sha256_v96(frozen)
        or fixed.candidate.get("lora_bank_topology_sha256")
        != implementation.get("lora_bank_topology_sha256")
    ):
        raise ValueError("V96 v2 frozen LoRA-bank evidence changed")
    return {
        **{str(name): str(digest) for name, digest in frozen.items()},
        FRESH_BANK_NAME: str(fixed.candidate["state_sha256"]),
    }


def prediction_provenance_v96(
    config_path: str | Path,
    fixed: FixedInputsV96,
    questions: Any,
    *,
    implementation: Mapping[str, Any],
) -> dict[str, Any]:
    value = {
        "artifact": PREDICTION_ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "config_sha256": sha256_file_v85(resolve_v96(config_path)),
        "implementation_seal_sha256": implementation["seal_sha256"],
        "implementation_source_inventory_sha256": implementation[
            "source_inventory_sha256"
        ],
        "v1_implementation_seal_sha256": implementation[
            "v1_implementation_seal_sha256"
        ],
        "candidate_attestation_file_sha256": fixed.candidate[
            "attestation_file_sha256"
        ],
        "candidate_attestation_identity_sha256": fixed.candidate[
            "attestation_identity_sha256"
        ],
        "candidate_fingerprint_sha256": fixed.candidate["fingerprint_sha256"],
        "candidate_state_sha256": fixed.candidate["state_sha256"],
        "model_snapshot_inventory_sha256": fixed.candidate[
            "model_snapshot_inventory_sha256"
        ],
        "frozen_bank_expected_state_inventory_sha256": fixed.candidate[
            "frozen_bank_expected_state_inventory_sha256"
        ],
        "lora_bank_state_inventory_sha256": canonical_sha256_v96(
            expected_lora_bank_states_v96_v2(fixed, implementation)
        ),
        "lora_bank_topology_sha256": fixed.candidate[
            "lora_bank_topology_sha256"
        ],
        "weights_hashed_not_model_loaded": fixed.candidate[
            "weights_hashed_not_model_loaded"
        ],
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
        "candidate_authenticated_from_pre_question_aggregate_attestation": True,
        "training_qa_opened_by_predictor": False,
        "labels_opened": False,
        "questions_or_answers_from_qa_serialized": False,
        "question_dependent_retrieval": False,
        "environmental_text_inputs": [],
        "runtime_promotion_authorized": False,
    }
    value["provenance_sha256"] = canonical_sha256_v96(value)
    return value


def mandatory_fixed_input_reads_v96(
    config: Mapping[str, Any],
    fixed: FixedInputsV96,
    *,
    config_path: str | Path = CONFIG,
    implementation: Mapping[str, Any],
) -> set[str]:
    candidate = resolve_v96(config["outputs"]["fixed_final_candidate"])
    source_paths = implementation.get("mandatory_source_paths")
    model_paths = implementation.get("mandatory_model_paths")
    if (
        not isinstance(source_paths, list)
        or not source_paths
        or any(not isinstance(path, str) for path in source_paths)
        or not isinstance(model_paths, list)
        or not model_paths
        or any(not isinstance(path, str) for path in model_paths)
    ):
        raise ValueError("V96 v2 mandatory path inventory changed")
    return {
        str(resolve_v96(config_path)),
        str(IMPLEMENTATION_SEAL_V2.resolve()),
        str(ATTESTATION.resolve()),
        str(resolve_v96(config["outputs"]["training_report"])),
        str(resolve_v96(config["outputs"]["preregistration"])),
        str(resolve_v96(config["outputs"]["cpu_preflight"])),
        str(resolve_v96(config["outputs"]["topology_smoke"])),
        str((candidate / "bridge.safetensors").resolve()),
        str((candidate / "runtime_metadata.json").resolve()),
        str((v1.MEMORY_CACHE / "manifest.json").resolve()),
        str(QUESTION_MANIFEST.resolve()),
        *(str(path.resolve()) for path in fixed.memory_paths.values()),
        *source_paths,
        *model_paths,
    }


def authenticate_prediction_bundle_v96(
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    assert_bound_config_path_v96(config_path)
    implementation = authenticate_evaluation_implementation_v96_v2(
        config_path=config_path
    )
    initial = load_config_v96_v2(config_path, allow_draft=False)
    audit = FileAccessAudit(
        prediction_forbidden_roots_v96(initial),
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    with audit:
        implementation_inside = authenticate_evaluation_implementation_v96_v2(
            config_path=config_path
        )
        if implementation_inside != implementation:
            raise RuntimeError("V96 v2 implementation changed across audit boundary")
        config = load_config_v96_v2(config_path, allow_draft=False)
        fixed = authenticate_fixed_inputs_before_questions_v96(
            config,
            config_path=config_path,
            audit=audit,
            implementation=implementation_inside,
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
        expected_provenance = prediction_provenance_v96(
            config_path, fixed, questions, implementation=implementation_inside
        )
        if provenance != expected_provenance:
            raise ValueError("V96 v2 prediction provenance changed")
        rows = read_jsonl_strict_v96_v2(paths.predictions)
        validate_prediction_rows_v96(
            rows,
            questions=questions,
            memory_hashes=fixed.memory_hashes,
            provenance_sha256=provenance["provenance_sha256"],
        )
        access = read_json_strict_v96(paths.prediction_access)
        completion = read_json_strict_v96(paths.prediction_completion)
        current = authenticate_fixed_final_candidate_v96(
            config,
            config_path=config_path,
            audit=audit,
            implementation=implementation_inside,
        )
    audit.assert_clean()
    assert_same_candidate_v96(fixed.candidate, current)
    mandatory = mandatory_fixed_input_reads_v96(
        config,
        fixed,
        config_path=config_path,
        implementation=implementation_inside,
    )
    validate_access_receipt_v96_v2(
        access,
        forbidden_roots=prediction_forbidden_roots_v96(config),
        mandatory=mandatory,
    )
    validate_prediction_completion_schema_v96_v2(completion)
    if (
        completion.get("artifact") != PREDICTION_COMPLETION_ARTIFACT
        or completion.get("schema_version") != SCHEMA_VERSION
        or completion.get("candidate_fingerprint_before")
        != fixed.candidate["fingerprint_sha256"]
        or completion.get("candidate_fingerprint_after")
        != fixed.candidate["fingerprint_sha256"]
        or completion.get("candidate_attestation_file_sha256")
        != fixed.candidate["attestation_file_sha256"]
        or completion.get("model_snapshot_inventory_sha256")
        != fixed.candidate["model_snapshot_inventory_sha256"]
        or completion.get("model_snapshot_hashes_invariant") is not True
        or completion.get("lora_bank_state_inventory_sha256")
        != canonical_sha256_v96(
            expected_lora_bank_states_v96_v2(fixed, implementation_inside)
        )
        or completion.get("lora_bank_topology_sha256")
        != implementation_inside.get("lora_bank_topology_sha256")
        or completion.get("lora_bank_states_invariant") is not True
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
        or completion.get("implementation_reauthenticated_inside_audit") is not True
        or completion.get("training_qa_opened") is not False
        or completion.get("labels_opened") is not False
        or completion.get("oracle_opened") is not False
        or completion.get("protected_read_count") != 0
        or completion.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V96 v2 prediction completion/access evidence changed")
    return {
        "config": config,
        "fixed": fixed,
        "questions": questions,
        "paths": paths,
        "provenance": provenance,
        "rows": rows,
        "access": access,
        "completion": completion,
        "implementation": implementation_inside,
        "prediction_sha256": sha256_file_v85(paths.predictions),
        "provenance_file_sha256": sha256_file_v85(paths.provenance),
        "access_sha256": sha256_file_v85(paths.prediction_access),
        "completion_sha256": sha256_file_v85(paths.prediction_completion),
    }


def authenticate_structured_score_v96(
    config_path: str | Path = CONFIG,
    *,
    prediction_bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    bundle = dict(prediction_bundle or authenticate_prediction_bundle_v96(config_path))
    report = read_json_strict_v96(bundle["paths"].structured_score)
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise TypeError("V96 v2 structured metrics are missing")
    validate_structured_metrics_v96(metrics)
    expected_fields = {
        "artifact",
        "schema_version",
        "status",
        "candidate_fingerprint_sha256",
        "candidate_attestation_file_sha256",
        "model_snapshot_inventory_sha256",
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
        or report.get("candidate_attestation_file_sha256")
        != bundle["fixed"].candidate["attestation_file_sha256"]
        or report.get("model_snapshot_inventory_sha256")
        != bundle["fixed"].candidate["model_snapshot_inventory_sha256"]
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
        raise ValueError("V96 v2 structured aggregate authentication failed")
    assert_aggregate_only_v96(report)
    return {
        "report": report,
        "sha256": sha256_file_v85(bundle["paths"].structured_score),
        "bundle": bundle,
    }


def authenticate_nll_bundle_v96(
    config_path: str | Path = CONFIG,
    *,
    fixed: FixedInputsV96 | None = None,
    questions: Any | None = None,
    implementation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    assert_bound_config_path_v96(config_path)
    current_implementation = implementation or authenticate_evaluation_implementation_v96_v2(
        config_path=config_path
    )
    config = load_config_v96_v2(config_path, allow_draft=False)
    current = fixed or authenticate_fixed_inputs_before_questions_v96(
        config,
        config_path=config_path,
        implementation=current_implementation,
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
        raise TypeError("V96 v2 NLL aggregate metrics are missing")
    validate_nll_metrics_v96(metrics)
    expected_fields = {
        "artifact",
        "schema_version",
        "status",
        "candidate_fingerprint_sha256",
        "candidate_attestation_file_sha256",
        "model_snapshot_inventory_sha256",
        "lora_bank_state_inventory_sha256",
        "lora_bank_topology_sha256",
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
        set(report) != expected_fields
        or report.get("artifact") != NLL_ARTIFACT
        or report.get("schema_version") != SCHEMA_VERSION
        or report.get("status") != "measured_aggregate_only_not_yet_gated"
        or report.get("candidate_fingerprint_sha256")
        != current.candidate["fingerprint_sha256"]
        or report.get("candidate_attestation_file_sha256")
        != current.candidate["attestation_file_sha256"]
        or report.get("model_snapshot_inventory_sha256")
        != current.candidate["model_snapshot_inventory_sha256"]
        or report.get("lora_bank_state_inventory_sha256")
        != canonical_sha256_v96(
            expected_lora_bank_states_v96_v2(current, current_implementation)
        )
        or report.get("lora_bank_topology_sha256")
        != current_implementation.get("lora_bank_topology_sha256")
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
        raise ValueError("V96 v2 NLL aggregate authentication failed")
    assert_aggregate_only_v96(report)
    expected_label = resolve_v96(config["known_development_gate"]["labels_path"])
    mandatory = mandatory_fixed_input_reads_v96(
        config,
        current,
        config_path=config_path,
        implementation=current_implementation,
    ) | {str(expected_label)}
    validate_access_receipt_v96_v2(
        access,
        forbidden_roots=nll_forbidden_roots_v96(config),
        mandatory=mandatory,
    )
    validate_nll_completion_schema_v96_v2(completion)
    if (
        completion.get("artifact") != NLL_COMPLETION_ARTIFACT
        or completion.get("schema_version") != SCHEMA_VERSION
        or completion.get("candidate_fingerprint_before")
        != current.candidate["fingerprint_sha256"]
        or completion.get("candidate_fingerprint_after")
        != current.candidate["fingerprint_sha256"]
        or completion.get("candidate_attestation_file_sha256")
        != current.candidate["attestation_file_sha256"]
        or completion.get("model_snapshot_inventory_sha256")
        != current.candidate["model_snapshot_inventory_sha256"]
        or completion.get("model_snapshot_hashes_invariant") is not True
        or completion.get("lora_bank_state_inventory_sha256")
        != canonical_sha256_v96(
            expected_lora_bank_states_v96_v2(current, current_implementation)
        )
        or completion.get("lora_bank_topology_sha256")
        != current_implementation.get("lora_bank_topology_sha256")
        or completion.get("lora_bank_states_invariant") is not True
        or completion.get("candidate_immutable") is not True
        or completion.get("frozen_v95_parent_immutable") is not True
        or completion.get("memory_hashes_invariant") is not True
        or completion.get("implementation_reauthenticated_inside_audit") is not True
        or completion.get("nll_sha256") != sha256_file_v85(paths.nll)
        or completion.get("nll_access_sha256") != sha256_file_v85(paths.nll_access)
        or completion.get("row_count_per_arm") != QUESTION_COUNT
        or completion.get("changed_row_count") != CHANGED_SIDE_COUNT
        or completion.get("row_level_content_serialized") is not False
        or completion.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V96 v2 NLL completion/access authentication failed")
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
        "implementation": current_implementation,
    }


__all__ = [name for name in globals() if not name.startswith("_")]
