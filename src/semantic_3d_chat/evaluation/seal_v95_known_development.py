"""Hash-only final authenticator and gate/evidence sealer for V95.

This process opens neither labels nor a language model.  It independently
authenticates the label-blind predictions, aggregate structured score, and
aggregate NLL evidence, then applies every preregistered known-development
condition from the sealed V95 config.  Passing makes deferred-final generation
eligible; it never authorizes automatic runtime promotion.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import sha256_file_v85
from semantic_3d_chat.evaluation.v95_known_development_common import (
    EVIDENCE_ARTIFACT,
    FINAL_SCORE_ARTIFACT,
    QUESTION_COUNT,
    REFERENCE_SHA256,
    SCHEMA_VERSION,
    assert_aggregate_only_v95,
    assert_output_bundle_state_v95,
    authenticate_fixed_final_candidate_v95,
    authenticate_nll_bundle_v95,
    authenticate_prediction_bundle_v95,
    authenticate_structured_score_v95,
    canonical_sha256_v95,
    evaluation_paths_v95,
    known_development_gate_results_v95,
    read_json_strict_v95,
    write_json_create_once_v95,
)
from semantic_3d_chat.evaluation.v95_known_development_implementation import (
    hardened_evaluation_stage_v95,
)
from semantic_3d_chat.evaluation.v95_strict_causal_successor_preflight import CONFIG


def _build_final_reports_v95(
    *,
    bundle: Mapping[str, Any],
    structured: Mapping[str, Any],
    nll: Mapping[str, Any],
    config_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = bundle["config"]
    candidate = bundle["fixed"].candidate
    current = authenticate_fixed_final_candidate_v95(config, config_path=config_path)
    immutable = (
        current["fingerprint_sha256"]
        == candidate["fingerprint_sha256"]
        == structured["report"]["candidate_fingerprint_sha256"]
        == nll["report"]["candidate_fingerprint_sha256"]
    )
    protected_read_count = int(bundle["access"]["protected_read_count"]) + int(
        nll["access"]["protected_read_count"]
    )
    prefix_invariant = (
        bundle["completion"].get("all_memory_hashes_invariant") is True
        and bundle["completion"].get("all_memories_bound_before_questions") is True
        and bundle["provenance"].get("all_memories_bound_before_questions") is True
    )
    label_isolation = (
        structured["report"].get("prediction_bundle_authenticated_before_labels_opened") is True
        and structured["report"].get("labels_opened_only_by_separate_scorer") is True
        and structured["report"].get("scorer_loaded_model") is False
        and nll["report"].get("fixed_final_and_memories_authenticated_before_labels_opened") is True
        and nll["report"].get("labels_opened_only_by_separate_nll_evaluator") is True
    )
    gates = known_development_gate_results_v95(
        structured["report"]["metrics"],
        nll["report"]["metrics"],
        config["known_development_gate"],
        immutable_fixed_final=immutable,
        prefix_invariant=prefix_invariant,
        label_isolation_proven=label_isolation,
        protected_read_count=protected_read_count,
    )
    passed = all(gates.values())
    unlock_required = bool(
        config["known_development_gate"]["pass_required_before_deferred_final_unlock"]
    )
    final = {
        "artifact": FINAL_SCORE_ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "status": (
            "passed_deferred_final_explicit_unlock_eligible"
            if passed
            else "measured_preregistered_gate_not_passed"
        ),
        "candidate_fingerprint_sha256": candidate["fingerprint_sha256"],
        "candidate_state_sha256": candidate["state_sha256"],
        "config_sha256": candidate["config_sha256"],
        "preregistration_sha256": candidate["preregistration_sha256"],
        "cpu_preflight_sha256": candidate["cpu_preflight_sha256"],
        "training_report_sha256": candidate["training_report_sha256"],
        "memory_manifest_sha256": bundle["fixed"].memory_manifest_sha256,
        "bound_memory_inventory_sha256": canonical_sha256_v95(bundle["fixed"].memory_hashes),
        "question_manifest_sha256": bundle["questions"].manifest_sha256,
        "questions_sha256": bundle["questions"].questions_sha256,
        "reference_sha256": REFERENCE_SHA256,
        "prediction_sha256": bundle["prediction_sha256"],
        "prediction_provenance_file_sha256": bundle["provenance_file_sha256"],
        "prediction_access_sha256": bundle["access_sha256"],
        "prediction_completion_sha256": bundle["completion_sha256"],
        "structured_score_sha256": structured["sha256"],
        "nll_sha256": nll["sha256"],
        "nll_access_sha256": nll["access_sha256"],
        "nll_completion_sha256": nll["completion_sha256"],
        "row_count": QUESTION_COUNT,
        "scene_count": 6,
        "structured_metrics": structured["report"]["metrics"],
        "nll_metrics": nll["report"]["metrics"],
        "gate_results": gates,
        "known_development_gate_passed": passed,
        "fixed_final_checkpoint_immutable": immutable,
        "scene_prefix_question_independent": prefix_invariant,
        "protected_read_count": protected_read_count,
        "row_level_content_serialized": False,
        "deferred_final_unlock_requires_explicit_separate_command": True,
        "deferred_final_unlock_eligible": passed if unlock_required else True,
        "automatic_runtime_promotion": False,
        "runtime_promotion_authorized": False,
    }
    assert_aggregate_only_v95(final)
    evidence = {
        "artifact": EVIDENCE_ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "status": "sealed_aggregate_evidence",
        "candidate_fingerprint_sha256": candidate["fingerprint_sha256"],
        "fixed_final_checkpoint_immutable": immutable,
        "config_sha256": candidate["config_sha256"],
        "preregistration_sha256": candidate["preregistration_sha256"],
        "cpu_preflight_sha256": candidate["cpu_preflight_sha256"],
        "training_report_sha256": candidate["training_report_sha256"],
        "memory_manifest_sha256": bundle["fixed"].memory_manifest_sha256,
        "question_manifest_sha256": bundle["questions"].manifest_sha256,
        "prediction_sha256": bundle["prediction_sha256"],
        "prediction_provenance_file_sha256": bundle["provenance_file_sha256"],
        "prediction_access_sha256": bundle["access_sha256"],
        "prediction_completion_sha256": bundle["completion_sha256"],
        "structured_score_sha256": structured["sha256"],
        "nll_sha256": nll["sha256"],
        "nll_access_sha256": nll["access_sha256"],
        "nll_completion_sha256": nll["completion_sha256"],
        "known_development_gate_results_sha256": canonical_sha256_v95(gates),
        "known_development_gate_passed": passed,
        "row_level_content_serialized": False,
        "deferred_final_unlock_eligible": passed if unlock_required else True,
        "automatic_runtime_promotion": False,
        "runtime_promotion_authorized": False,
    }
    assert_aggregate_only_v95(evidence)
    return final, evidence


@hardened_evaluation_stage_v95
def seal_known_development_v95(
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    """Authenticate all inputs and create the aggregate final score/evidence once."""

    # Each authenticator is label-free/model-free. Missing or tampered NLL
    # evidence raises before either final artifact can be created.
    bundle = authenticate_prediction_bundle_v95(config_path)
    paths = evaluation_paths_v95(bundle["config"])
    states = [
        paths.final_score.exists() or paths.final_score.is_symlink(),
        paths.evidence.exists() or paths.evidence.is_symlink(),
    ]
    if all(states):
        return authenticate_final_evidence_v95(config_path)
    assert_output_bundle_state_v95((paths.final_score, paths.evidence), complete=False)
    structured = authenticate_structured_score_v95(config_path, prediction_bundle=bundle)
    nll = authenticate_nll_bundle_v95(
        config_path, fixed=bundle["fixed"], questions=bundle["questions"]
    )
    final, evidence = _build_final_reports_v95(
        bundle=bundle,
        structured=structured,
        nll=nll,
        config_path=config_path,
    )
    write_json_create_once_v95(paths.final_score, final)
    evidence = {**evidence, "final_score_sha256": sha256_file_v85(paths.final_score)}
    assert_aggregate_only_v95(evidence)
    write_json_create_once_v95(paths.evidence, evidence)
    return {**final, "evidence_sha256": sha256_file_v85(paths.evidence)}


@hardened_evaluation_stage_v95
def authenticate_final_evidence_v95(
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    """Rebuild expected aggregate evidence without labels/model and compare exactly."""

    bundle = authenticate_prediction_bundle_v95(config_path)
    paths = evaluation_paths_v95(bundle["config"])
    assert_output_bundle_state_v95((paths.final_score, paths.evidence), complete=True)
    structured = authenticate_structured_score_v95(config_path, prediction_bundle=bundle)
    nll = authenticate_nll_bundle_v95(
        config_path, fixed=bundle["fixed"], questions=bundle["questions"]
    )
    expected_final, expected_evidence = _build_final_reports_v95(
        bundle=bundle,
        structured=structured,
        nll=nll,
        config_path=config_path,
    )
    final = read_json_strict_v95(paths.final_score)
    evidence = read_json_strict_v95(paths.evidence)
    expected_evidence = {
        **expected_evidence,
        "final_score_sha256": sha256_file_v85(paths.final_score),
    }
    if final != expected_final or evidence != expected_evidence:
        raise ValueError("V95 final known-development evidence changed")
    assert_aggregate_only_v95(final)
    assert_aggregate_only_v95(evidence)
    return {
        **final,
        "final_score_sha256": sha256_file_v85(paths.final_score),
        "evidence_sha256": sha256_file_v85(paths.evidence),
        "authenticated": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("command", choices=("seal", "authenticate"), nargs="?", default="seal")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = (
        seal_known_development_v95(args.config)
        if args.command == "seal"
        else authenticate_final_evidence_v95(args.config)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "authenticate_final_evidence_v95",
    "main",
    "seal_known_development_v95",
]
