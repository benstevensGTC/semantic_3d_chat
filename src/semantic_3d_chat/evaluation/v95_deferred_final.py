"""Fail-closed V95 deferred-final unlock and execution contract.

The generic :mod:`semantic_3d_chat.evaluation.final_once` controller predates
V95.  It expects a promoted ``adapter.safetensors`` checkpoint and a legacy
selector report, while V95 deliberately has an unpromoted six-tensor
``bridge.safetensors`` fixed final and a separately sealed known-development
gate.  This module is therefore the only V95 transition that may authorize
future materialization of opaque scenes 25--30.

This module is intentionally model-free.  It never opens a legacy semantic
plan, QA/answer file, oracle directory, or deferred-scene artifact.  ``unlock``
authenticates the immutable V95 fixed final, the complete passing
known-development evidence, and a separately sealed outcome-independent
materialization preregistration.  It proves the deferred footprint is still
absent and atomically creates one launch seal.  Generation remains possible
only through the preregistered stage runner, which reauthenticates this unlock
before every fixed child command.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.seal_v95_known_development import (
    authenticate_final_evidence_v95,
)
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import (
    sha256_file_v85,
)
from semantic_3d_chat.evaluation.v95_deferred_final_materialization import (
    authenticate_materialization_preregistration_v95,
)
from semantic_3d_chat.evaluation.v95_known_development_common import (
    authenticate_fixed_final_candidate_v95,
    canonical_sha256_v95,
    read_json_strict_v95,
    require_sha256_v95,
    resolve_v95,
)
from semantic_3d_chat.evaluation.v95_strict_causal_successor_preflight import (
    CONFIG,
    DEFERRED_FINAL_SCENES,
    assert_deferred_final_absent_v95,
    load_config_v95,
)

SCHEMA_VERSION: Final[int] = 95
UNLOCK_ARTIFACT: Final[str] = "gemma4_v95_deferred_final_unlock_v1"
UNLOCK_STATUS: Final[str] = "unlocked_for_separate_v95_materializer_not_generated"
PREFLIGHT_STATUS: Final[str] = "eligible_for_explicit_unlock_not_generated"
DEFAULT_UNLOCK: Final[Path] = (
    PROJECT_ROOT
    / "reports/gemma4/metrics/gemma4_v95_strict_causal_successor_deferred_final_unlock.json"
)

DEFERRED_WORK_ROOT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v95_deferred_final"
)
MATERIALIZATION_MANIFEST: Final[Path] = DEFERRED_WORK_ROOT / "materialization.json"
MEMORY_CACHE_ROOT: Final[Path] = DEFERRED_WORK_ROOT / "memory_cache"
QUESTION_MANIFEST: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/questions/gemma4_v95_deferred_final.json"
)
PREDICTION_ROOT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/predictions/v95_deferred_final"
)
FINAL_METRICS_ROOT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/v95_deferred_final"
)

_MATERIALIZATION_STAGES: Final[tuple[str, ...]] = (
    "authenticate_pre_outcome_materialization_preregistration",
    "generate",
    "render",
    "features",
    "maps",
    "memory",
    "qa_raw",
    "qa_select",
    "questions",
)
_EVALUATION_STAGES: Final[tuple[str, ...]] = (
    "authenticate_prediction_inputs_before_questions",
    "predict_v95_label_blind",
    "predict_fixed_v94_comparator_label_blind",
    "authenticate_predictions_without_labels",
    "score_structured_in_separate_label_process",
    "measure_nll_in_separate_label_aware_model_process",
    "seal_aggregate_final_evidence_without_labels_or_model",
    "run_separate_runtime_leakage_gate",
    "package_only_after_all_final_gates_pass",
)
_IMPLEMENTATION_INPUTS: Final[tuple[str, ...]] = (
    "src/semantic_3d_chat/evaluation/v95_known_development_common.py",
    "src/semantic_3d_chat/evaluation/predict_v95_known_development.py",
    "src/semantic_3d_chat/evaluation/authenticate_v95_known_development.py",
    "src/semantic_3d_chat/evaluation/score_v95_known_development.py",
    "src/semantic_3d_chat/evaluation/nll_v95_known_development.py",
    "src/semantic_3d_chat/evaluation/seal_v95_known_development.py",
    "src/semantic_3d_chat/evaluation/v95_deferred_final_qa.py",
    "src/semantic_3d_chat/evaluation/v95_deferred_final_materialization.py",
    "src/semantic_3d_chat/evaluation/v95_deferred_final.py",
)


class DeferredFinalBlockedError(RuntimeError):
    """Raised when the next deferred-final boundary is not implemented/sealed."""


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _unlock_path(output: str | Path | None) -> Path:
    """Resolve the sole production unlock path; arbitrary parallel seals are forbidden."""

    expected = DEFAULT_UNLOCK.resolve()
    destination = expected if output is None else resolve_v95(output)
    if destination != expected:
        raise ValueError(
            "V95 deferred final has one fixed unlock path; arbitrary output paths are "
            "not permitted"
        )
    return destination


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically create an immutable JSON seal without replacing a prior file."""

    destination = path.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"V95 deferred-final seal already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise FileExistsError(
                f"V95 deferred-final seal raced with another writer: {destination}"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _forbidden_roots(config: Mapping[str, Any]) -> list[Path]:
    """Return every path the unlock process must never open.

    Path existence is checked separately with metadata-only filesystem calls.
    The audit roots protect content reads, including after a future process has
    materialized one of these paths.
    """

    roots: list[Path] = []
    roots.extend(path.resolve() for path in PROJECT_ROOT.glob("data*/oracle"))
    roots.extend(path.resolve() for path in PROJECT_ROOT.glob("data*/qa"))
    roots.extend(
        resolve_v95(path)
        for path in config["deferred_final_lock"]["legacy_plan_files_never_opened"]
    )
    roots.extend(
        resolve_v95(root) / scene_id
        for root in config["deferred_final_lock"]["physical_artifact_roots"]
        for scene_id in DEFERRED_FINAL_SCENES
    )
    return list(dict.fromkeys(path.resolve() for path in roots))


def _assert_passing_gate(
    gate: Mapping[str, Any], candidate: Mapping[str, Any]
) -> None:
    required_true = (
        "authenticated",
        "known_development_gate_passed",
        "deferred_final_unlock_eligible",
        "fixed_final_checkpoint_immutable",
        "scene_prefix_question_independent",
    )
    if (
        any(gate.get(field) is not True for field in required_true)
        or gate.get("status") != "passed_deferred_final_explicit_unlock_eligible"
        or gate.get("protected_read_count") != 0
        or gate.get("automatic_runtime_promotion") is not False
        or gate.get("runtime_promotion_authorized") is not False
        or gate.get("candidate_fingerprint_sha256")
        != candidate.get("fingerprint_sha256")
        or gate.get("candidate_state_sha256") != candidate.get("state_sha256")
        or gate.get("config_sha256") != candidate.get("config_sha256")
        or gate.get("training_report_sha256")
        != candidate.get("training_report_sha256")
        or not isinstance(gate.get("gate_results"), Mapping)
        or not gate["gate_results"]
        or not all(value is True for value in gate["gate_results"].values())
    ):
        raise DeferredFinalBlockedError(
            "V95 deferred final remains locked: sealed known-development gate did not "
            "authenticate as a complete pass for the immutable fixed final"
        )


def _validate_absence_record(
    absence: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    expected_placeholders = {
        Path(path).as_posix(): 0
        for path in config["deferred_final_lock"]["empty_qa_placeholders"]
    }
    expected_physical_count = len(DEFERRED_FINAL_SCENES) * len(
        config["deferred_final_lock"]["physical_artifact_roots"]
    )
    if (
        absence.get("scene_ids") != list(DEFERRED_FINAL_SCENES)
        or absence.get("physical_path_count_checked") != expected_physical_count
        or absence.get("physical_artifacts_present") != []
        or absence.get("empty_qa_placeholders") != expected_placeholders
        or absence.get("legacy_plan_file_count_opened") != 0
        or any(
            absence.get(field) is not False
            for field in (
                "scene_generation_performed",
                "rendering_performed",
                "feature_extraction_performed",
                "map_building_performed",
                "qa_generation_performed",
            )
        )
    ):
        raise ValueError("V95 recorded deferred-final absence attestation changed")


def _execution_contract(
    config: Mapping[str, Any], materialization: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a semantics-free, outcome-independent future stage template."""

    deferred = config["deferred_evaluation"]
    return {
        "status": "fixed_preregistered_commands_available_only_after_unlock",
        "materialization_commands": [
            materialization["stages"][stage]["authorized_entrypoint"]
            for stage in materialization["stage_order"]
        ],
        "materialization_child_argv_sha256": canonical_sha256_v95(
            {
                stage: materialization["stages"][stage]["child_argv"]
                for stage in materialization["stage_order"]
            }
        ),
        "materialization_output_contract_sha256": canonical_sha256_v95(
            {
                stage: materialization["stages"][stage]["expected_outputs"]
                for stage in materialization["stage_order"]
            }
        ),
        "materialization_preregistration_file_sha256": materialization[
            "preregistration_file_sha256"
        ],
        "materialization_preregistration_identity_sha256": materialization[
            "preregistration_identity_sha256"
        ],
        "generation_recipe_sha256": materialization["recipe"]["recipe_sha256"],
        "generation_inputs_derived_without_legacy_plans": True,
        "every_stage_reauthenticates_unlock_before_execution": True,
        "required_scene_ids": list(DEFERRED_FINAL_SCENES),
        "required_scene_count": int(deferred["scene_count"]),
        "required_pair_count": int(deferred["pair_count"]),
        "required_question_count": int(deferred["expected_row_count_after_unlock"]),
        "required_changed_unit_count": int(
            deferred["expected_changed_unit_count_after_unlock"]
        ),
        "required_changed_side_count": int(
            deferred["expected_changed_side_count_after_unlock"]
        ),
        "required_materialization_stage_order": list(_MATERIALIZATION_STAGES),
        "required_evaluation_stage_order": list(_EVALUATION_STAGES),
        "fixed_output_paths": {
            "work_root": _relative(DEFERRED_WORK_ROOT),
            "materialization_manifest": _relative(MATERIALIZATION_MANIFEST),
            "memory_cache": _relative(MEMORY_CACHE_ROOT),
            "questions_only_manifest": _relative(QUESTION_MANIFEST),
            "predictions_root": _relative(PREDICTION_ROOT),
            "metrics_root": _relative(FINAL_METRICS_ROOT),
        },
        "prediction_allowed_inputs": [
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
            "v94_or_v95_scored_behavior",
        ],
        "required_prediction_arms": [
            "v95_primary",
            "v95_zero_payload",
            "v95_full_interior_permutation",
            "v95_paired_wrong_scene",
            "fixed_v94_primary_same_rows",
        ],
        "labels_may_be_opened_only_after_prediction_bundle_authentication": True,
        "labels_may_be_opened_only_by_separate_structured_and_nll_processes": True,
        "all_scene_memories_must_be_bound_before_question_open": True,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "environmental_text_inputs": [],
        "final_gates": dict(config["gates"]),
        "separate_runtime_leakage_gate_required": bool(
            config["gates"]["runtime_packaging_requires_separate_leakage_gate"]
        ),
        "automatic_runtime_promotion": False,
    }


def _implementation_identity() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in _IMPLEMENTATION_INPUTS:
        path = (PROJECT_ROOT / relative).resolve()
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"V95 deferred-final implementation input changed: {path}")
        result[relative] = sha256_file_v85(path)
    return result


def _base_payload(
    *,
    config: Mapping[str, Any],
    gate: Mapping[str, Any],
    candidate: Mapping[str, Any],
    absence: Mapping[str, Any],
    access: Mapping[str, Any],
    materialization: Mapping[str, Any],
    status: str,
) -> dict[str, Any]:
    paths = config["deferred_final_lock"]["legacy_plan_files_never_opened"]
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
        "known_development_final_score_sha256": require_sha256_v95(
            gate.get("final_score_sha256"), "deferred unlock known-development score"
        ),
        "known_development_evidence_sha256": require_sha256_v95(
            gate.get("evidence_sha256"), "deferred unlock known-development evidence"
        ),
        "known_development_gate_results_sha256": canonical_sha256_v95(
            gate["gate_results"]
        ),
        "materialization_preregistration_file_sha256": materialization[
            "preregistration_file_sha256"
        ],
        "materialization_preregistration_identity_sha256": materialization[
            "preregistration_identity_sha256"
        ],
        "known_development_gate_passed": True,
        "deferred_final_unlock_eligible": True,
        "deferred_scene_ids": list(DEFERRED_FINAL_SCENES),
        "deferred_absence_before_unlock": dict(absence),
        "deferred_absence_attestation_sha256": canonical_sha256_v95(absence),
        "legacy_plan_paths_declared_but_unopened": [Path(path).as_posix() for path in paths],
        "legacy_plan_file_count_opened": 0,
        "final_label_file_count_opened": 0,
        "oracle_file_count_opened": 0,
        "scene_generation_performed": False,
        "rendering_performed": False,
        "feature_extraction_performed": False,
        "map_building_performed": False,
        "qa_generation_performed": False,
        "model_loaded": False,
        "optimizer_constructed": False,
        "protected_read_count": 0,
        "unlock_access": dict(access),
        "implementation_source_sha256": _implementation_identity(),
        "execution_contract": _execution_contract(config, materialization),
        "runtime_promotion_authorized": False,
        "automatic_runtime_promotion": False,
    }
    payload["unlock_identity_sha256"] = canonical_sha256_v95(payload)
    return payload


def _validate_access(
    access: Mapping[str, Any], config: Mapping[str, Any] | None = None
) -> None:
    expected_roots = (
        None
        if config is None
        else [str(path) for path in _forbidden_roots(config)]
    )
    if (
        access.get("artifact") != "gemma4_v95_deferred_final_unlock_access_v1"
        or access.get("protected_read_count") != 0
        or access.get("forbidden_accesses") != []
        or access.get("passed") is not True
        or access.get("block_forbidden") is not True
        or not isinstance(access.get("loaded_files"), list)
        or access.get("loaded_file_inventory_sha256")
        != canonical_sha256_v95(access["loaded_files"])
        or (
            expected_roots is not None
            and access.get("forbidden_roots") != expected_roots
        )
        or access.get("forbidden_component_names") != ["oracle", "qa"]
    ):
        raise ValueError("V95 deferred-final unlock access evidence changed")


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
]:
    initial = load_config_v95(config_path, allow_draft=False)
    audit = FileAccessAudit(
        _forbidden_roots(initial),
        forbidden_component_names={"oracle", "qa"},
        block_forbidden=True,
    )
    with audit:
        config = load_config_v95(config_path, allow_draft=False)
        materialization = authenticate_materialization_preregistration_v95()
        gate = authenticate_final_evidence_v95(config_path)
        candidate = authenticate_fixed_final_candidate_v95(
            config, config_path=config_path, audit=audit
        )
        _assert_passing_gate(gate, candidate)
        absence = (
            assert_deferred_final_absent_v95(config)
            if require_absence
            else {
                "scene_ids": list(DEFERRED_FINAL_SCENES),
                "physical_path_count_checked": len(DEFERRED_FINAL_SCENES)
                * len(config["deferred_final_lock"]["physical_artifact_roots"]),
                "physical_artifacts_present": [],
                "empty_qa_placeholders": {
                    Path(path).as_posix(): 0
                    for path in config["deferred_final_lock"]["empty_qa_placeholders"]
                },
                "legacy_plan_file_count_opened": 0,
                "scene_generation_performed": False,
                "rendering_performed": False,
                "feature_extraction_performed": False,
                "map_building_performed": False,
                "qa_generation_performed": False,
            }
        )
    audit.assert_clean()
    access = {
        "artifact": "gemma4_v95_deferred_final_unlock_access_v1",
        "schema_version": SCHEMA_VERSION,
        "loaded_files": audit.unique_paths,
        "loaded_file_inventory_sha256": canonical_sha256_v95(audit.unique_paths),
        "forbidden_roots": [str(path) for path in audit.forbidden_roots],
        "forbidden_component_names": sorted(audit.forbidden_component_names),
        "block_forbidden": audit.block_forbidden,
        "forbidden_accesses": audit.forbidden_accesses(),
        "protected_read_count": len(audit.forbidden_accesses()),
        "passed": not audit.forbidden_accesses(),
    }
    _validate_absence_record(absence, config)
    _validate_access(access, config)
    return config, gate, candidate, dict(absence), access, materialization


def preflight_deferred_final_v95(
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    """Prove unlock eligibility without writing or opening protected content."""

    config, gate, candidate, absence, access, materialization = _prerequisites(
        config_path, require_absence=True
    )
    return _base_payload(
        config=config,
        gate=gate,
        candidate=candidate,
        absence=absence,
        access=access,
        materialization=materialization,
        status=PREFLIGHT_STATUS,
    )


def unlock_deferred_final_v95(
    config_path: str | Path = CONFIG,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Atomically create or authenticate the sole V95 deferred-final unlock."""

    destination = _unlock_path(output)
    if destination.exists() or destination.is_symlink():
        result = authenticate_deferred_final_unlock_v95(config_path, destination)
        return {**result, "reused_authenticated_unlock": True}
    preflight = preflight_deferred_final_v95(config_path)
    payload = {**preflight, "status": UNLOCK_STATUS}
    payload.pop("unlock_identity_sha256")
    payload["unlock_identity_sha256"] = canonical_sha256_v95(payload)
    _atomic_create_json(destination, payload)
    result = authenticate_deferred_final_unlock_v95(config_path, destination)
    return {**result, "reused_authenticated_unlock": False}


def authenticate_deferred_final_unlock_v95(
    config_path: str | Path = CONFIG,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Authenticate an existing unlock without requiring scenes to remain absent."""

    source = _unlock_path(output)
    payload = read_json_strict_v95(source)
    if payload.get("artifact") != UNLOCK_ARTIFACT or payload.get("status") != UNLOCK_STATUS:
        raise ValueError("V95 deferred-final unlock artifact/status changed")
    absence = payload.get("deferred_absence_before_unlock")
    access = payload.get("unlock_access")
    if not isinstance(absence, Mapping) or not isinstance(access, Mapping):
        raise TypeError("V95 deferred-final unlock evidence sections are missing")
    (
        config,
        gate,
        candidate,
        _unused_absence,
        current_access,
        materialization,
    ) = _prerequisites(
        config_path, require_absence=False
    )
    _validate_absence_record(absence, config)
    _validate_access(access, config)
    if dict(access) != current_access:
        raise ValueError("V95 deferred-final unlock access evidence is not reproducible")
    expected = _base_payload(
        config=config,
        gate=gate,
        candidate=candidate,
        absence=absence,
        access=current_access,
        materialization=materialization,
        status=UNLOCK_STATUS,
    )
    if payload != expected:
        raise ValueError("V95 deferred-final unlock bytes no longer match its contract")
    return {
        **payload,
        "unlock_file_sha256": sha256_file_v85(source),
        "authenticated": True,
    }


def materialization_template_v95(
    config_path: str | Path = CONFIG,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Return the authenticated future contract without opening generation inputs."""

    unlock = authenticate_deferred_final_unlock_v95(config_path, output)
    return {
        "artifact": "gemma4_v95_deferred_final_materialization_template_v1",
        "schema_version": SCHEMA_VERSION,
        "status": unlock["execution_contract"]["status"],
        "unlock_file_sha256": unlock["unlock_file_sha256"],
        "unlock_identity_sha256": unlock["unlock_identity_sha256"],
        "candidate_fingerprint_sha256": unlock["candidate_fingerprint_sha256"],
        "execution_contract": unlock["execution_contract"],
        "legacy_plan_file_count_opened": 0,
        "final_label_file_count_opened": 0,
        "model_loaded": False,
        "runtime_promotion_authorized": False,
    }


def materialization_preflight_v95(
    config_path: str | Path = CONFIG,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Authenticate the reviewed fixed plan without executing any stage."""

    template = materialization_template_v95(config_path, output)
    return {
        **template,
        "status": "materialization_preflight_passed_no_stage_executed",
        "stage_execution_performed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("preflight", "unlock", "authenticate", "template", "materialization-preflight"),
    )
    parser.add_argument("--config", default=str(CONFIG))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = preflight_deferred_final_v95(args.config)
    elif args.command == "unlock":
        result = unlock_deferred_final_v95(args.config)
    elif args.command == "authenticate":
        result = authenticate_deferred_final_unlock_v95(args.config)
    elif args.command == "template":
        result = materialization_template_v95(args.config)
    else:
        result = materialization_preflight_v95(args.config)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_UNLOCK",
    "PREFLIGHT_STATUS",
    "UNLOCK_ARTIFACT",
    "UNLOCK_STATUS",
    "DeferredFinalBlockedError",
    "authenticate_deferred_final_unlock_v95",
    "main",
    "materialization_preflight_v95",
    "materialization_template_v95",
    "preflight_deferred_final_v95",
    "unlock_deferred_final_v95",
]
