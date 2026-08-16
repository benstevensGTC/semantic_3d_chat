"""Fail-closed V96 authorization for the sealed V95 deferred recipe.

V95 preregistered the outcome-independent physical recipe for opaque scenes
25--30 before any of those scenes existed.  That recipe is immutable.  V96
reuses seven physical children byte-for-byte and authorization-repairs only
the QA-selection child while preserving its pure selector.  V96 also changes
the model authorization boundary:
an authenticated, completely passing V96 known-development gate and a separate
explicit create-once unlock are required before the V96 stage wrapper may run
one of the preregistered child commands.

This module is model-free and materialization-free.  Importing it, running its
preflight, or authenticating an unlock never starts Blender, loads Gemma,
opens deferred oracle/QA content, or creates a deferred scene.  The unlock
also binds both V96 wrapper source files so editing either file after explicit
authorization invalidates every later stage.
"""

from __future__ import annotations

import argparse
import functools
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final, ParamSpec, TypeVar

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.seal_v96_known_development_v2 import (
    authenticate_final_evidence_v96,
)
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import (
    sha256_file_v85,
)
from semantic_3d_chat.evaluation.v95_deferred_final_materialization import (
    ARTIFACT as V95_MATERIALIZATION_ARTIFACT,
)
from semantic_3d_chat.evaluation.v95_deferred_final_materialization import (
    PREREGISTRATION as V95_MATERIALIZATION_PREREGISTRATION,
)
from semantic_3d_chat.evaluation.v95_deferred_final_materialization import (
    QUESTION_MANIFEST as V95_QUESTION_MANIFEST,
)
from semantic_3d_chat.evaluation.v95_deferred_final_materialization import (
    RECEIPT_ROOT as V95_RECEIPT_ROOT,
)
from semantic_3d_chat.evaluation.v95_deferred_final_materialization import (
    STATUS as V95_MATERIALIZATION_STATUS,
)
from semantic_3d_chat.evaluation.v95_deferred_final_materialization import (
    WORK_ROOT as V95_WORK_ROOT,
)
from semantic_3d_chat.evaluation.v95_deferred_final_materialization import (
    authenticate_materialization_preregistration_v95,
)
from semantic_3d_chat.evaluation.v95_strict_causal_successor_preflight import (
    DEFERRED_FINAL_SCENES,
)
from semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight import (
    CONFIG,
    assert_deferred_final_absent_v96,
    load_config_v96,
)
from semantic_3d_chat.evaluation.v96_known_development_common_v2 import (
    assert_bound_config_path_v96,
    authenticate_fixed_final_candidate_v96,
    canonical_sha256_v96,
    read_json_strict_v96,
    require_sha256_v96,
    resolve_v96,
    write_json_create_once_v96,
)
from semantic_3d_chat.evaluation.v96_known_development_implementation import (
    exclusive_evaluation_lock_v96,
)
from semantic_3d_chat.evaluation.v96_known_development_implementation_v2 import (
    authenticate_evaluation_implementation_v96_v2,
)

SCHEMA_VERSION: Final[int] = 96
UNLOCK_ARTIFACT: Final[str] = "gemma4_v96_deferred_final_unlock_v1"
UNLOCK_STATUS: Final[str] = "unlocked_for_separate_v96_materializer_not_generated"
PREFLIGHT_STATUS: Final[str] = "eligible_for_explicit_unlock_not_generated"
DEFAULT_UNLOCK: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/gemma4_v96_atomic_pair_repair_deferred_final_unlock.json"
)

# These identities were sealed before any deferred scene or label existed.
# V96 may reuse the recipe but may not silently substitute another one.
PINNED_V95_MATERIALIZATION_FILE_SHA256: Final[str] = (
    "1e37dcc8e34791864907512bb6eab454b5cf6bb6448773258869a95f50cf3e8a"
)
PINNED_V95_MATERIALIZATION_IDENTITY_SHA256: Final[str] = (
    "88fd0d268b9e25199bd4c43aa368f31b5f2598305eedd26631de445c4893215e"
)
MATERIALIZATION_STAGE_ORDER: Final[tuple[str, ...]] = (
    "generate",
    "render",
    "features",
    "maps",
    "memory",
    "qa_raw",
    "qa_select",
    "questions",
)
_IMPLEMENTATION_INPUTS: Final[tuple[str, ...]] = (
    "src/semantic_3d_chat/evaluation/v96_deferred_final.py",
    "src/semantic_3d_chat/evaluation/v96_deferred_final_materialization.py",
    "src/semantic_3d_chat/evaluation/v96_deferred_final_qa.py",
)
_EXPECTED_DEFERRED_LOCK: Final[dict[str, Any]] = {
    "scene_ids": list(DEFERRED_FINAL_SCENES),
    "physical_artifact_roots": [
        "data/oracle",
        "data/rendered",
        "data_gemma4/features",
        "data_gemma4/maps",
    ],
    "empty_qa_placeholders": [
        "data_diverse20/qa/test.jsonl",
        "data_diverse28/qa/test.jsonl",
        "data_diverse52/qa/test.jsonl",
    ],
    "legacy_plan_files_never_opened": [
        "configs/experiments/diverse20.yaml",
        "configs/experiments/diverse28.yaml",
        "configs/experiments/diverse52.yaml",
    ],
    "physical_artifacts_required_absent_through_fixed_final": True,
    "qa_placeholders_required_zero_bytes_through_fixed_final": True,
    "generation_before_fixed_final_authorized": False,
    "rendering_before_fixed_final_authorized": False,
    "feature_extraction_before_fixed_final_authorized": False,
    "map_building_before_fixed_final_authorized": False,
    "qa_generation_before_fixed_final_authorized": False,
    "only_opaque_ids_and_absence_locks_available_to_v96": True,
}

_P = ParamSpec("_P")
_R = TypeVar("_R")


class DeferredFinalBlockedError(RuntimeError):
    """Raised when V96 has not earned the explicit deferred transition."""


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError as error:
        raise ValueError(f"V96 deferred path escapes the project: {path}") from error


def _unlock_path(output: str | Path | None) -> Path:
    expected = resolve_v96(DEFAULT_UNLOCK)
    destination = expected if output is None else resolve_v96(output)
    if destination != expected:
        raise ValueError(
            "V96 deferred final has one fixed unlock path; arbitrary output paths are not permitted"
        )
    return destination


def _validate_deferred_lock_config(config: Mapping[str, Any]) -> None:
    if config.get("deferred_final_lock") != _EXPECTED_DEFERRED_LOCK:
        raise ValueError("V96 deferred-final absence lock changed")


def _validate_materialization_preregistration_v96(
    materialization: Mapping[str, Any],
) -> None:
    """Require the exact already-sealed V95 physical recipe and no substitute."""

    stages = materialization.get("stages")
    if not isinstance(stages, Mapping):
        raise TypeError("V96 reused materialization stages are missing")
    if (
        materialization.get("authenticated") is not True
        or materialization.get("artifact") != V95_MATERIALIZATION_ARTIFACT
        or materialization.get("schema_version") != 95
        or materialization.get("status") != V95_MATERIALIZATION_STATUS
        or materialization.get("preregistration_file_sha256")
        != PINNED_V95_MATERIALIZATION_FILE_SHA256
        or materialization.get("preregistration_identity_sha256")
        != PINNED_V95_MATERIALIZATION_IDENTITY_SHA256
        or materialization.get("stage_order") != list(MATERIALIZATION_STAGE_ORDER)
        or set(stages) != set(MATERIALIZATION_STAGE_ORDER)
        or materialization.get("scene_count") != 6
        or materialization.get("pair_count") != 3
        or materialization.get("intended_row_count") != 216
        or materialization.get("intended_changed_unit_count") != 12
        or materialization.get("intended_changed_side_count") != 24
        or materialization.get("generation_requires_authenticated_unlock") is not True
        or materialization.get("every_execution_stage_requires_authenticated_unlock") is not True
        or materialization.get("legacy_plan_files_opened") != []
        or materialization.get("known_development_labels_opened") is not False
        or materialization.get("deferred_labels_opened") is not False
        or materialization.get("deferred_oracle_opened") is not False
        or materialization.get("deferred_artifacts_generated") is not False
        or materialization.get("model_loaded") is not False
        or materialization.get("optimizer_constructed") is not False
        or materialization.get("protected_read_count") != 0
        or materialization.get("automatic_runtime_promotion") is not False
    ):
        raise ValueError("V96 rejected the reused V95 materialization preregistration")

    all_outputs: list[str] = []
    for stage in MATERIALIZATION_STAGE_ORDER:
        contract = stages.get(stage)
        if not isinstance(contract, Mapping):
            raise TypeError(f"V96 reused stage contract is missing: {stage}")
        entrypoint = contract.get("authorized_entrypoint")
        child_argv = contract.get("child_argv")
        outputs = contract.get("expected_outputs")
        expected_receipt = _relative(V95_RECEIPT_ROOT / f"{stage}.json")
        if (
            not isinstance(entrypoint, list)
            or len(entrypoint) != 6
            or entrypoint[1:4]
            != [
                "-m",
                "semantic_3d_chat.evaluation.v95_deferred_final_materialization",
                "run-stage",
            ]
            or entrypoint[4:] != ["--stage", stage]
            or not isinstance(entrypoint[0], str)
            or not entrypoint[0]
            or not isinstance(child_argv, list)
            or not child_argv
            or any(
                not isinstance(argv, list)
                or not argv
                or any(not isinstance(value, str) or not value for value in argv)
                for argv in child_argv
            )
            or not isinstance(outputs, list)
            or not outputs
            or any(not isinstance(path, str) or not path for path in outputs)
            or contract.get("receipt") != expected_receipt
        ):
            raise ValueError(f"V96 reused materialization stage changed: {stage}")
        all_outputs.extend(outputs)
    if len(all_outputs) != len(set(all_outputs)):
        raise ValueError("V96 reused materialization output paths overlap")


def _implementation_identity() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in _IMPLEMENTATION_INPUTS:
        candidate = PROJECT_ROOT / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise FileNotFoundError(f"V96 deferred-final implementation input changed: {candidate}")
        result[relative] = sha256_file_v85(candidate)
    return result


def _forbidden_roots(config: Mapping[str, Any]) -> list[Path]:
    roots = [path.resolve() for path in PROJECT_ROOT.glob("data*/oracle")]
    roots.extend(path.resolve() for path in PROJECT_ROOT.glob("data*/qa"))
    roots.extend(
        resolve_v96(path)
        for path in config["deferred_final_lock"]["legacy_plan_files_never_opened"]
    )
    roots.extend(
        resolve_v96(root) / scene_id
        for root in config["deferred_final_lock"]["physical_artifact_roots"]
        for scene_id in DEFERRED_FINAL_SCENES
    )
    roots.extend((V95_WORK_ROOT.resolve(), V95_QUESTION_MANIFEST.resolve()))
    roots.extend(
        (
            PROJECT_ROOT
            / "reports/gemma4/metrics"
            / f"v95_deferred_final_memory_access_{scene_id}.json"
        ).resolve()
        for scene_id in DEFERRED_FINAL_SCENES
    )
    return list(dict.fromkeys(path.resolve() for path in roots))


def _assert_passing_gate(
    gate: Mapping[str, Any],
    candidate: Mapping[str, Any],
    evaluator: Mapping[str, Any],
) -> None:
    required_true = (
        "authenticated",
        "known_development_gate_passed",
        "deferred_final_unlock_eligible",
        "fixed_final_checkpoint_immutable",
        "frozen_v95_parent_immutable",
        "scene_prefix_question_independent",
    )
    matching = {
        "candidate_fingerprint_sha256": "fingerprint_sha256",
        "candidate_state_sha256": "state_sha256",
        "frozen_v95_state_sha256": "frozen_v95_state_sha256",
        "config_sha256": "config_sha256",
        "preregistration_sha256": "preregistration_sha256",
        "cpu_preflight_sha256": "cpu_preflight_sha256",
        "training_report_sha256": "training_report_sha256",
    }
    if (
        any(gate.get(field) is not True for field in required_true)
        or gate.get("status") != "passed_deferred_final_explicit_unlock_eligible"
        or gate.get("protected_read_count") != 0
        or gate.get("automatic_runtime_promotion") is not False
        or gate.get("runtime_promotion_authorized") is not False
        or any(gate.get(left) != candidate.get(right) for left, right in matching.items())
        or gate.get("candidate_attestation_file_sha256")
        != candidate.get("attestation_file_sha256")
        or gate.get("candidate_attestation_identity_sha256")
        != candidate.get("attestation_identity_sha256")
        or gate.get("candidate_attestation_immutable") is not True
        or gate.get("implementation_seal_sha256") != evaluator.get("seal_sha256")
        or gate.get("implementation_source_inventory_sha256")
        != evaluator.get("source_inventory_sha256")
        or gate.get("v1_implementation_seal_sha256")
        != evaluator.get("v1_implementation_seal_sha256")
        or candidate.get("v2_implementation_seal_sha256")
        != evaluator.get("seal_sha256")
        or not isinstance(gate.get("gate_results"), Mapping)
        or not gate["gate_results"]
        or not all(value is True for value in gate["gate_results"].values())
    ):
        raise DeferredFinalBlockedError(
            "V96 deferred final remains locked: authenticated known-development "
            "evidence is not a complete pass for the immutable fixed final"
        )


def _materialization_absence(
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    outputs = [
        path
        for stage in MATERIALIZATION_STAGE_ORDER
        for path in materialization["stages"][stage]["expected_outputs"]
    ]
    present_outputs = [
        path for path in outputs if resolve_v96(path).exists() or resolve_v96(path).is_symlink()
    ]
    receipts = [
        materialization["stages"][stage]["receipt"] for stage in MATERIALIZATION_STAGE_ORDER
    ]
    present_receipts = [
        path for path in receipts if resolve_v96(path).exists() or resolve_v96(path).is_symlink()
    ]
    work_root_present = V95_WORK_ROOT.exists() or V95_WORK_ROOT.is_symlink()
    if present_outputs or present_receipts or work_root_present:
        raise RuntimeError(
            "V96 refuses a preexisting deferred materialization footprint: "
            f"outputs={present_outputs}, receipts={present_receipts}, "
            f"work_root_present={work_root_present}"
        )
    return {
        "materialization_expected_output_count_checked": len(outputs),
        "materialization_outputs_present": [],
        "materialization_receipt_count_checked": len(receipts),
        "materialization_receipts_present": [],
        "materialization_work_root": _relative(V95_WORK_ROOT),
        "materialization_work_root_present": False,
    }


def _absence_without_recheck(
    config: Mapping[str, Any], materialization: Mapping[str, Any]
) -> dict[str, Any]:
    outputs = [
        path
        for stage in MATERIALIZATION_STAGE_ORDER
        for path in materialization["stages"][stage]["expected_outputs"]
    ]
    return {
        "scene_ids": list(DEFERRED_FINAL_SCENES),
        "physical_path_count_checked": len(DEFERRED_FINAL_SCENES)
        * len(config["deferred_final_lock"]["physical_artifact_roots"]),
        "physical_artifacts_present": [],
        "empty_qa_placeholders": {
            Path(path).as_posix(): 0
            for path in config["deferred_final_lock"]["empty_qa_placeholders"]
        },
        "legacy_plan_file_count_opened": 0,
        "generation_performed": False,
        "materialization_expected_output_count_checked": len(outputs),
        "materialization_outputs_present": [],
        "materialization_receipt_count_checked": len(MATERIALIZATION_STAGE_ORDER),
        "materialization_receipts_present": [],
        "materialization_work_root": _relative(V95_WORK_ROOT),
        "materialization_work_root_present": False,
    }


def _validate_absence_record(
    absence: Mapping[str, Any],
    config: Mapping[str, Any],
    materialization: Mapping[str, Any],
) -> None:
    expected = _absence_without_recheck(config, materialization)
    if dict(absence) != expected:
        raise ValueError("V96 recorded deferred-final absence attestation changed")


def _validate_access(access: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    expected_roots = [str(path) for path in _forbidden_roots(config)]
    loaded = access.get("loaded_files")
    if (
        access.get("artifact") != "gemma4_v96_deferred_final_unlock_access_v1"
        or access.get("schema_version") != SCHEMA_VERSION
        or access.get("protected_read_count") != 0
        or access.get("forbidden_accesses") != []
        or access.get("passed") is not True
        or access.get("block_forbidden") is not True
        or not isinstance(loaded, list)
        or access.get("loaded_file_inventory_sha256") != canonical_sha256_v96(loaded)
        or access.get("forbidden_roots") != expected_roots
        or access.get("forbidden_component_names") != ["oracle", "qa"]
    ):
        raise ValueError("V96 deferred-final unlock access evidence changed")


def _execution_contract(
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    original_child = {
        stage: materialization["stages"][stage]["child_argv"]
        for stage in MATERIALIZATION_STAGE_ORDER
    }
    child = dict(original_child)
    support_python = original_child["qa_select"][0][0]
    child["qa_select"] = [
        [
            support_python,
            "-m",
            "semantic_3d_chat.evaluation.v96_deferred_final_qa",
            "select",
            "--config",
            str(CONFIG),
        ]
    ]
    outputs = {
        stage: materialization["stages"][stage]["expected_outputs"]
        for stage in MATERIALIZATION_STAGE_ORDER
    }
    original_entrypoints = {
        stage: materialization["stages"][stage]["authorized_entrypoint"]
        for stage in MATERIALIZATION_STAGE_ORDER
    }
    wrapper_entrypoints = [
        [
            ".venv-gemma4/bin/python",
            "-m",
            "semantic_3d_chat.evaluation.v96_deferred_final_materialization",
            "run-stage",
            "--stage",
            stage,
        ]
        for stage in MATERIALIZATION_STAGE_ORDER
    ]
    return {
        "status": "fixed_v95_recipe_available_only_through_v96_unlock_wrapper",
        "reused_v95_preregistration_path": _relative(V95_MATERIALIZATION_PREREGISTRATION),
        "reused_v95_preregistration_file_sha256": materialization["preregistration_file_sha256"],
        "reused_v95_preregistration_identity_sha256": materialization[
            "preregistration_identity_sha256"
        ],
        "reused_child_argv_byte_for_byte": False,
        "reused_expected_outputs_byte_for_byte": True,
        "only_qa_select_authorization_child_changed_from_v95": True,
        "qa_select_v95_unlock_dependency_removed": True,
        "qa_select_authenticates_v96_unlock_before_label_read": True,
        "unchanged_stage_child_argv_byte_for_byte": [
            stage for stage in MATERIALIZATION_STAGE_ORDER if stage != "qa_select"
        ],
        "original_v95_child_argv_sha256": canonical_sha256_v96(original_child),
        "original_v95_entrypoints_sha256": canonical_sha256_v96(original_entrypoints),
        "v96_authorized_entrypoints": wrapper_entrypoints,
        "materialization_child_argv_sha256": canonical_sha256_v96(child),
        "materialization_stage_child_argv_sha256": {
            stage: canonical_sha256_v96(child[stage]) for stage in MATERIALIZATION_STAGE_ORDER
        },
        "materialization_output_contract_sha256": canonical_sha256_v96(outputs),
        "materialization_stage_output_contract_sha256": {
            stage: canonical_sha256_v96(outputs[stage]) for stage in MATERIALIZATION_STAGE_ORDER
        },
        "materialization_source_inventory_sha256": canonical_sha256_v96(
            materialization["source_sha256"]
        ),
        "numeric_compiler_source_inventory_sha256": canonical_sha256_v96(
            materialization["numeric_compiler_source_sha256"]
        ),
        "generation_recipe_sha256": materialization["recipe"]["recipe_sha256"],
        "required_scene_ids": list(DEFERRED_FINAL_SCENES),
        "required_scene_count": 6,
        "required_pair_count": 3,
        "required_question_count": 216,
        "required_changed_unit_count": 12,
        "required_changed_side_count": 24,
        "required_materialization_stage_order": list(MATERIALIZATION_STAGE_ORDER),
        "every_stage_reauthenticates_v96_unlock_before_child_execution": True,
        "every_stage_reauthenticates_v95_preregistration_before_child_execution": True,
        "every_stage_requires_final_evaluation_preregistration": True,
        "preexisting_unreceipted_outputs_rejected": True,
        "predecessor_receipts_and_output_hashes_required": True,
        "prediction_allowed_inputs_after_materialization": [
            "continuous_scene_memory",
            "memory_hashes",
            "numeric_geometry",
            "sanitized_questions_only_manifest",
            "immutable_model_parameters",
        ],
        "prediction_forbidden_inputs": [
            "oracle",
            "qa_or_answers",
            "labels",
            "semantic_scene_plans",
            "object_names_or_categories",
            "scene_captions",
            "textual_scene_graphs",
            "known_development_row_content",
        ],
        "all_scene_memories_must_be_bound_before_question_open": True,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "environmental_text_inputs": [],
        "deferred_prediction_and_scoring_harness_implemented": True,
        "deferred_prediction_or_scoring_authorized_by_this_unlock": False,
        "automatic_runtime_promotion": False,
    }


def _base_payload(
    *,
    config: Mapping[str, Any],
    gate: Mapping[str, Any],
    candidate: Mapping[str, Any],
    absence: Mapping[str, Any],
    access: Mapping[str, Any],
    materialization: Mapping[str, Any],
    evaluator: Mapping[str, Any],
    final_evaluation: Mapping[str, Any],
    status: str,
    explicit_unlock_created: bool,
) -> dict[str, Any]:
    legacy = config["deferred_final_lock"]["legacy_plan_files_never_opened"]
    sources = _implementation_identity()
    payload = {
        "artifact": UNLOCK_ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "config_sha256": candidate["config_sha256"],
        "preregistration_sha256": candidate["preregistration_sha256"],
        "cpu_preflight_sha256": candidate["cpu_preflight_sha256"],
        "training_report_sha256": candidate["training_report_sha256"],
        "candidate_fingerprint_sha256": candidate["fingerprint_sha256"],
        "candidate_state_sha256": candidate["state_sha256"],
        "candidate_weights_sha256": candidate["weights_sha256"],
        "candidate_attestation_file_sha256": candidate["attestation_file_sha256"],
        "candidate_attestation_identity_sha256": candidate[
            "attestation_identity_sha256"
        ],
        "frozen_v95_state_sha256": candidate["frozen_v95_state_sha256"],
        "known_development_final_score_sha256": require_sha256_v96(
            gate.get("final_score_sha256"), "deferred unlock final score"
        ),
        "known_development_evidence_sha256": require_sha256_v96(
            gate.get("evidence_sha256"), "deferred unlock evidence"
        ),
        "known_development_gate_results_sha256": canonical_sha256_v96(gate["gate_results"]),
        "known_development_implementation_seal_sha256": evaluator["seal_sha256"],
        "known_development_v1_implementation_seal_sha256": evaluator[
            "v1_implementation_seal_sha256"
        ],
        "known_development_implementation_source_inventory_sha256": evaluator[
            "source_inventory_sha256"
        ],
        "materialization_preregistration_file_sha256": materialization[
            "preregistration_file_sha256"
        ],
        "materialization_preregistration_identity_sha256": materialization[
            "preregistration_identity_sha256"
        ],
        "final_evaluation_preregistration_file_sha256": final_evaluation[
            "preregistration_file_sha256"
        ],
        "final_evaluation_preregistration_identity_sha256": final_evaluation[
            "preregistration_identity_sha256"
        ],
        "final_evaluation_gate_contract_sha256": final_evaluation["v95_gate_source"][
            "contract_sha256"
        ],
        "final_evaluation_implementation_source_inventory_sha256": final_evaluation[
            "implementation_source_inventory_sha256"
        ],
        "final_evaluation_preregistered_before_unlock": True,
        "known_development_gate_passed": True,
        "deferred_final_unlock_eligible": True,
        "explicit_separate_unlock_required": True,
        "explicit_unlock_created": explicit_unlock_created,
        "materialization_stage_execution_authorized": explicit_unlock_created,
        "deferred_scene_ids": list(DEFERRED_FINAL_SCENES),
        "deferred_absence_before_unlock": dict(absence),
        "deferred_absence_attestation_sha256": canonical_sha256_v96(absence),
        "legacy_plan_paths_declared_but_unopened": [Path(path).as_posix() for path in legacy],
        "legacy_plan_file_count_opened": 0,
        "deferred_label_file_count_opened": 0,
        "deferred_oracle_file_count_opened": 0,
        "scene_generation_performed": False,
        "rendering_performed": False,
        "feature_extraction_performed": False,
        "map_building_performed": False,
        "qa_generation_performed": False,
        "stage_execution_performed": False,
        "model_loaded": False,
        "optimizer_constructed": False,
        "protected_read_count": 0,
        "unlock_access": dict(access),
        "implementation_source_sha256": sources,
        "implementation_source_inventory_sha256": canonical_sha256_v96(sources),
        "execution_contract": _execution_contract(materialization),
        "runtime_promotion_authorized": False,
        "automatic_runtime_promotion": False,
    }
    payload["unlock_identity_sha256"] = canonical_sha256_v96(payload)
    return payload


def _authenticate_final_evaluation_preregistration_v96() -> dict[str, Any]:
    """Late import avoids a module cycle while making the unlock fail closed."""

    from semantic_3d_chat.evaluation.v96_deferred_final_evaluation import (
        authenticate_preregistration_v96_final,
    )

    preregistration = authenticate_preregistration_v96_final()
    if (
        preregistration.get("authenticated") is not True
        or preregistration.get("all_thresholds_sealed_before_deferred_labels") is not True
        or preregistration.get("predictors_are_label_blind") is not True
        or preregistration.get("automatic_runtime_promotion") is not False
    ):
        raise DeferredFinalBlockedError(
            "V96 deferred final requires its authenticated evaluator preregistration before unlock"
        )
    return preregistration


def _prerequisites(
    config_path: str | Path,
    *,
    require_absence: bool,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    assert_bound_config_path_v96(config_path)
    initial = load_config_v96(config_path, allow_draft=False)
    _validate_deferred_lock_config(initial)
    audit = FileAccessAudit(
        _forbidden_roots(initial),
        forbidden_component_names={"oracle", "qa"},
        block_forbidden=True,
    )
    with audit:
        config = load_config_v96(config_path, allow_draft=False)
        _validate_deferred_lock_config(config)
        materialization = authenticate_materialization_preregistration_v95()
        _validate_materialization_preregistration_v96(materialization)
        evaluator = authenticate_evaluation_implementation_v96_v2(
            config_path=config_path
        )
        gate = authenticate_final_evidence_v96(config_path)
        candidate = authenticate_fixed_final_candidate_v96(
            config,
            config_path=config_path,
            audit=audit,
            implementation=evaluator,
        )
        _assert_passing_gate(gate, candidate, evaluator)
        if require_absence:
            physical = assert_deferred_final_absent_v96(config)
            materialization_absence = _materialization_absence(materialization)
            absence = {**physical, **materialization_absence}
        else:
            absence = _absence_without_recheck(config, materialization)
    audit.assert_clean()
    access = {
        "artifact": "gemma4_v96_deferred_final_unlock_access_v1",
        "schema_version": SCHEMA_VERSION,
        "loaded_files": audit.unique_paths,
        "loaded_file_inventory_sha256": canonical_sha256_v96(audit.unique_paths),
        "forbidden_roots": [str(path) for path in audit.forbidden_roots],
        "forbidden_component_names": sorted(audit.forbidden_component_names),
        "block_forbidden": audit.block_forbidden,
        "forbidden_accesses": audit.forbidden_accesses(),
        "protected_read_count": len(audit.forbidden_accesses()),
        "passed": not audit.forbidden_accesses(),
    }
    _validate_absence_record(absence, config, materialization)
    _validate_access(access, config)
    return (
        config,
        gate,
        candidate,
        dict(absence),
        access,
        materialization,
        evaluator,
    )


@contextmanager
def deferred_final_guard_v96(
    config_path: str | Path = CONFIG,
) -> Iterator[Mapping[str, Any]]:
    """Hold the V96 process lock and authenticate the sealed evaluator."""

    assert_bound_config_path_v96(config_path)
    with exclusive_evaluation_lock_v96():
        implementation = authenticate_evaluation_implementation_v96_v2(
            config_path=config_path
        )
        yield implementation


def hardened_deferred_stage_v96(
    function: Callable[_P, _R],
) -> Callable[_P, _R]:
    """Guard a default-config deferred public stage for static verification."""

    @functools.wraps(function)
    def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        config_path = kwargs.get("config_path", CONFIG)
        if args and function.__name__ != "run_materialization_stage_v96":
            config_path = args[0]
        with deferred_final_guard_v96(config_path):
            return function(*args, **kwargs)

    return guarded


def _preflight_under_guard_v96(
    config_path: str | Path,
) -> dict[str, Any]:
    final_evaluation = _authenticate_final_evaluation_preregistration_v96()
    config, gate, candidate, absence, access, materialization, evaluator = _prerequisites(
        config_path, require_absence=True
    )
    return _base_payload(
        config=config,
        gate=gate,
        candidate=candidate,
        absence=absence,
        access=access,
        materialization=materialization,
        evaluator=evaluator,
        final_evaluation=final_evaluation,
        status=PREFLIGHT_STATUS,
        explicit_unlock_created=False,
    )


@hardened_deferred_stage_v96
def preflight_deferred_final_v96(
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    """Prove eligibility without writing an unlock or executing a stage."""

    return _preflight_under_guard_v96(config_path)


def _authenticate_deferred_final_unlock_under_guard_v96(
    config_path: str | Path = CONFIG,
    output: str | Path | None = None,
) -> dict[str, Any]:
    final_evaluation = _authenticate_final_evaluation_preregistration_v96()
    source = _unlock_path(output)
    payload = read_json_strict_v96(source)
    if (
        payload.get("artifact") != UNLOCK_ARTIFACT
        or payload.get("status") != UNLOCK_STATUS
        or payload.get("explicit_unlock_created") is not True
        or payload.get("materialization_stage_execution_authorized") is not True
    ):
        raise ValueError("V96 deferred-final unlock artifact/status changed")
    absence = payload.get("deferred_absence_before_unlock")
    stored_access = payload.get("unlock_access")
    if not isinstance(absence, Mapping) or not isinstance(stored_access, Mapping):
        raise TypeError("V96 deferred-final unlock evidence sections are missing")
    (
        config,
        gate,
        candidate,
        _unused_absence,
        current_access,
        materialization,
        evaluator,
    ) = _prerequisites(config_path, require_absence=False)
    _validate_absence_record(absence, config, materialization)
    _validate_access(stored_access, config)
    _validate_access(current_access, config)
    expected = _base_payload(
        config=config,
        gate=gate,
        candidate=candidate,
        absence=absence,
        access=stored_access,
        materialization=materialization,
        evaluator=evaluator,
        final_evaluation=final_evaluation,
        status=UNLOCK_STATUS,
        explicit_unlock_created=True,
    )
    if payload != expected:
        raise ValueError("V96 deferred-final unlock bytes no longer match its contract")
    return {
        **payload,
        "unlock_file_sha256": sha256_file_v85(source),
        "authentication_protected_read_count": current_access["protected_read_count"],
        "authenticated": True,
    }


@hardened_deferred_stage_v96
def unlock_deferred_final_v96(
    config_path: str | Path = CONFIG,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Create or authenticate the sole explicit V96 unlock."""

    destination = _unlock_path(output)
    if destination.exists() or destination.is_symlink():
        result = _authenticate_deferred_final_unlock_under_guard_v96(config_path, destination)
        return {**result, "reused_authenticated_unlock": True}
    preflight = _preflight_under_guard_v96(config_path)
    payload = {
        **preflight,
        "status": UNLOCK_STATUS,
        "explicit_unlock_created": True,
        "materialization_stage_execution_authorized": True,
    }
    payload.pop("unlock_identity_sha256")
    payload["unlock_identity_sha256"] = canonical_sha256_v96(payload)
    write_json_create_once_v96(destination, payload)
    result = _authenticate_deferred_final_unlock_under_guard_v96(config_path, destination)
    return {**result, "reused_authenticated_unlock": False}


@hardened_deferred_stage_v96
def authenticate_deferred_final_unlock_v96(
    config_path: str | Path = CONFIG,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Authenticate an unlock without requiring generated outputs to stay absent."""

    return _authenticate_deferred_final_unlock_under_guard_v96(config_path, output)


@hardened_deferred_stage_v96
def materialization_template_v96(
    config_path: str | Path = CONFIG,
    output: str | Path | None = None,
) -> dict[str, Any]:
    unlock = _authenticate_deferred_final_unlock_under_guard_v96(config_path, output)
    return {
        "artifact": "gemma4_v96_deferred_final_materialization_template_v1",
        "schema_version": SCHEMA_VERSION,
        "status": unlock["execution_contract"]["status"],
        "unlock_file_sha256": unlock["unlock_file_sha256"],
        "unlock_identity_sha256": unlock["unlock_identity_sha256"],
        "candidate_fingerprint_sha256": unlock["candidate_fingerprint_sha256"],
        "execution_contract": unlock["execution_contract"],
        "legacy_plan_file_count_opened": 0,
        "deferred_label_file_count_opened": 0,
        "model_loaded": False,
        "stage_execution_performed": False,
        "runtime_promotion_authorized": False,
    }


@hardened_deferred_stage_v96
def materialization_preflight_v96(
    config_path: str | Path = CONFIG,
    output: str | Path | None = None,
) -> dict[str, Any]:
    template = materialization_template_v96.__wrapped__(config_path, output)
    return {
        **template,
        "status": "materialization_preflight_passed_no_stage_executed",
        "stage_execution_performed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "preflight",
            "unlock",
            "authenticate",
            "template",
            "materialization-preflight",
        ),
    )
    parser.add_argument("--config", default=str(CONFIG))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = preflight_deferred_final_v96(args.config)
    elif args.command == "unlock":
        result = unlock_deferred_final_v96(args.config)
    elif args.command == "authenticate":
        result = authenticate_deferred_final_unlock_v96(args.config)
    elif args.command == "template":
        result = materialization_template_v96(args.config)
    else:
        result = materialization_preflight_v96(args.config)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_UNLOCK",
    "MATERIALIZATION_STAGE_ORDER",
    "PINNED_V95_MATERIALIZATION_FILE_SHA256",
    "PINNED_V95_MATERIALIZATION_IDENTITY_SHA256",
    "PREFLIGHT_STATUS",
    "UNLOCK_ARTIFACT",
    "UNLOCK_STATUS",
    "DeferredFinalBlockedError",
    "authenticate_deferred_final_unlock_v96",
    "deferred_final_guard_v96",
    "hardened_deferred_stage_v96",
    "main",
    "materialization_preflight_v96",
    "materialization_template_v96",
    "preflight_deferred_final_v96",
    "unlock_deferred_final_v96",
]
