from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from semantic_3d_chat.evaluation import v62_pair_disjoint_preregistration as boundary
from semantic_3d_chat.evaluation import v66_internal_validation_preregistration as prereg
from semantic_3d_chat.evaluation import v66_internal_validation_terminal_gate as gate
from semantic_3d_chat.evaluation.predict_question_control import (
    _control_checkpoint_sha256,
)
from semantic_3d_chat.evaluation.question_manifest import QuestionManifest, QuestionRecord


def _v7_audit() -> dict[str, Any]:
    return {
        "architecture": gate.ARCHITECTURE,
        "scene_token_count": 258,
        "environment_latent_count": 256,
        "control_token_count": 4,
        "scene_moment_count": 8,
        "every_scene_token_influenced_output": True,
        "question_dependent_scene_retrieval": False,
        "softmax_scene_attention_used": False,
        "control_values_scene_question_bilinear": True,
        "gate_scene_question_conditioned": False,
        "inherited_v60_state_frozen": False,
        "separate_question_scene_route_projections": False,
        "normalized_route_factors": False,
        "all_scene_moments_consumed_by_route": False,
        "low_rank_bilinear_route": False,
        "route_uses_inherited_value_trunk": True,
        "route_factor_rank": None,
        "gate_probability": 0.999999997,
        "control_used": True,
        "maximum_control_rms": 0.05,
        "exact_no_control_route": False,
        "activation_rms": None,
        "activation_rms_threshold": None,
        "exact_no_control_below_threshold": False,
        "always_on_continuous_control": True,
        "legacy_route_parameters_ignored": True,
        "saved_runtime_training_gate_required": True,
    }


def _candidate() -> tuple[gate.CandidateInputs, tuple[dict[str, Any], ...]]:
    specs_by_id = {spec.pair_id: spec for spec in boundary.PAIR_INVENTORY}
    specs = [specs_by_id[pair_id] for pair_id in boundary.INTERNAL_VALIDATION_PAIR_IDS]
    prefixes = {
        scene: hashlib.sha256(f"prefix:{scene}".encode()).hexdigest()
        for spec in specs
        for scene in spec.scene_ids
    }
    signatures = {
        scene: hashlib.sha256(f"signature:{scene}".encode()).hexdigest()
        for spec in specs
        for scene in spec.scene_ids
    }
    questions: list[QuestionRecord] = []
    references: list[dict[str, Any]] = []
    natural: list[dict[str, Any]] = []
    swaps: list[dict[str, Any]] = []
    ordinal = 1
    for spec in specs:
        for unit_index in range(24):
            changed = unit_index < spec.changed_unit_count
            question = "Is there a chair?"
            question_key = f"cfq_{ordinal:016x}"
            sides: list[dict[str, Any]] = []
            for side_index, scene_id in enumerate(spec.scene_ids):
                question_id = f"q_{ordinal:06d}"
                ordinal += 1
                answer = ("yes" if side_index == 0 else "no") if changed else "yes"
                questions.append(QuestionRecord(scene_id, question_id, question))
                reference = {
                    "scene_id": scene_id,
                    "question_id": question_id,
                    "answer": answer,
                    "answer_items": None,
                    "answer_type": "presence",
                    "route_label": changed,
                    "counterfactual_pair_id": spec.pair_id,
                    "counterfactual_paired_scene_id": spec.scene_ids[1 - side_index],
                    "counterfactual_question_key": question_key,
                    "counterfactual_change_type": spec.change_type,
                    "counterfactual_role": ("reference" if side_index == 0 else "counterfactual"),
                }
                references.append(reference)
                sides.append(reference)
                natural.append(
                    {
                        "scene_id": scene_id,
                        "question_id": question_id,
                        "predicted_answer": answer,
                        "prefix_hash": prefixes[scene_id],
                        "scene_control_signature_sha256": signatures[scene_id],
                        "control_checkpoint_sha256": "c" * 64,
                        "control_audit": _v7_audit(),
                        "provenance_sha256": "d" * 64,
                    }
                )
            for side_index, reference in enumerate(sides):
                opposite = sides[1 - side_index]
                injected = str(opposite["scene_id"])
                swaps.append(
                    {
                        "scene_id": reference["scene_id"],
                        "question_id": reference["question_id"],
                        "injected_scene_id": injected,
                        "predicted_answer": opposite["answer"],
                        "prefix_hash": prefixes[injected],
                        "scene_control_signature_sha256": signatures[injected],
                        "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                        "control_checkpoint_sha256": "c" * 64,
                        "control_audit": _v7_audit(),
                        "provenance_sha256": "e" * 64,
                    }
                )
    manifest = QuestionManifest(
        questions=tuple(questions),
        questions_sha256="1" * 64,
        source_qa_sha256="2" * 64,
    )
    internal = {
        "thresholds": {
            "internal_validation": prereg.INTERNAL_VALIDATION_THRESHOLDS,
            "same_question_different_scene": prereg.SAME_QUESTION_THRESHOLDS,
            "scene_swap": prereg.SCENE_SWAP_THRESHOLDS,
            "training": prereg.TRAINING_THRESHOLDS,
        }
    }
    candidate = gate.CandidateInputs(
        internal_preregistration=internal,
        internal_preregistration_sha256="3" * 64,
        parent_preregistration={
            "artifacts": {
                "scorer_references": {
                    "sha256": prereg.PINNED_SCORER_REFERENCES_SHA256,
                    "records_sha256": prereg.PINNED_SCORER_RECORDS_SHA256,
                }
            }
        },
        parent_preregistration_sha256="4" * 64,
        training_preregistration_sha256="5" * 64,
        questions=manifest,
        questions_manifest_sha256="6" * 64,
        baseline={},
        baseline_sha256="7" * 64,
        natural_rows=tuple(natural),
        natural_sha256="8" * 64,
        natural_provenance_sha256="9" * 64,
        swap_rows=tuple(swaps),
        swap_sha256="a" * 64,
        swap_provenance_sha256="b" * 64,
        base_checkpoint_sha256="c" * 64,
        base_checkpoint_files=(),
        control_checkpoint_sha256="d" * 64,
        control_files={},
        control_metadata={
            "schema_version": 7,
            "architecture": gate.ARCHITECTURE,
            "saved_runtime_training_gate_attestation_sha256": "e" * 64,
        },
        runtime_config_file_sha256="f" * 64,
        runtime_config_effective_sha256="0" * 64,
        training_report_sha256="1" * 64,
        training_report_artifact=gate.TRAINING_REPORT_ARTIFACT,
        natural_prefixes=prefixes,
        natural_signatures=signatures,
    )
    return candidate, tuple(references)


def test_v7_prediction_audit_is_strict_and_always_on() -> None:
    metadata = {"moment_count": 8, "maximum_control_rms": 0.2}
    audit = _v7_audit()

    assert gate._valid_v7_audit(audit, metadata)
    audit["control_used"] = False
    assert not gate._valid_v7_audit(audit, metadata)


def test_control_fingerprint_matches_prediction_runner(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "control.safetensors").write_bytes(b"numeric weights")
    (checkpoint / "runtime_metadata.json").write_bytes(b"{}\n")

    observed, files = gate._control_checkpoint_fingerprint(checkpoint)

    assert observed == _control_checkpoint_sha256(checkpoint)
    assert set(files) == {"control.safetensors", "runtime_metadata.json"}


def test_v66_score_passes_exact_synthetic_population() -> None:
    candidate, references = _candidate()

    result = gate.score_populations(candidate, references)

    assert result["passed"] is True
    assert result["metrics"]["natural"]["canonical_exact"] == 384
    assert result["metrics"]["natural"]["changed_side_exact"] == 52
    assert result["metrics"]["scene_swap"]["answer_follows_injected_scene"] == 52
    assert (
        result["metrics"]["same_question_different_scene"]["distinct_scene_signature_hashes"] == 26
    )


def test_prefix_and_signature_are_invariant_but_control_is_question_conditioned() -> None:
    candidate, _references = _candidate()
    rows = [dict(row) for row in candidate.natural_rows]
    first_scene = str(rows[0]["scene_id"])
    same_scene = [row for row in rows if row["scene_id"] == first_scene]
    same_scene[0]["control_audit"] = {
        **same_scene[0]["control_audit"],
        "maximum_control_rms": 0.02,
    }
    same_scene[1]["control_audit"] = {
        **same_scene[1]["control_audit"],
        "maximum_control_rms": 0.08,
    }

    prefixes, signatures = gate._validate_natural_predictions(
        rows,
        manifest=candidate.questions,
        provenance_sha256="d" * 64,
        control_checkpoint_sha256="c" * 64,
        control_metadata={"moment_count": 8, "maximum_control_rms": 0.2},
        baseline={"scene_prefix_hashes": candidate.natural_prefixes},
    )

    assert prefixes[first_scene] == candidate.natural_prefixes[first_scene]
    assert signatures[first_scene] == candidate.natural_signatures[first_scene]
    assert (
        same_scene[0]["control_audit"]["maximum_control_rms"]
        != same_scene[1]["control_audit"]["maximum_control_rms"]
    )


def test_v66_training_checks_require_paired_scene_dependence() -> None:
    thresholds = prereg.TRAINING_THRESHOLDS
    metrics = {
        "answer_follows_injected_scene": 60,
        "paired_opposite_side_total": 80,
        "answer_follows_injected_scene_complete_units": 25,
        "paired_opposite_unit_total": 40,
        "answer_matches_original_reference": 20,
        "answer_matches_original_reference_complete_units": 5,
        "question_identity_count": 80,
        "exact_paired_scene_count": 80,
        "exact_paired_scene_prefix_count": 80,
        "exact_paired_scene_signature_count": 80,
        "differing_reference_count": 80,
        "cross_swap_complete_units": 40,
        "answer_or_question_text_stored": False,
    }

    assert all(gate._paired_opposite_training_checks(metrics, thresholds).values())
    metrics["exact_paired_scene_signature_count"] = 79
    assert not all(gate._paired_opposite_training_checks(metrics, thresholds).values())


def _synthetic_training_report() -> tuple[dict[str, Any], dict[str, Any]]:
    thresholds = prereg.TRAINING_THRESHOLDS
    per_type = {
        hashlib.sha256(answer_type.encode()).hexdigest(): {
            "exact": minimum,
            "total": minimum,
        }
        for answer_type, minimum in thresholds["per_type_minimum_exact"]
    }
    cv_metrics = {
        "supported_exact": 300,
        "supported_total": 571,
        "unsupported_total": 5,
        "inventory_total": 576,
        "eligible_fold_total": 12,
        "eligible_folds_with_exact_hit": 12,
        "changed_side_exact": 45,
        "changed_side_total": 75,
        "complete_changed_units": 15,
        "changed_unit_total": 35,
        "prediction_change_units": 20,
        "per_type_by_sha256": per_type,
    }
    final_metrics = {
        "supported_exact": 520,
        "supported_total": 576,
        "unsupported_total": 0,
        "inventory_total": 576,
        "complete_changed_units": 36,
        "changed_unit_total": 40,
    }
    dependence_metrics = {
        "answer_follows_injected_scene": 60,
        "paired_opposite_side_total": 80,
        "answer_follows_injected_scene_complete_units": 25,
        "paired_opposite_unit_total": 40,
        "answer_matches_original_reference": 20,
        "answer_matches_original_reference_complete_units": 5,
        "question_identity_count": 80,
        "exact_paired_scene_count": 80,
        "exact_paired_scene_prefix_count": 80,
        "exact_paired_scene_signature_count": 80,
        "differing_reference_count": 80,
        "cross_swap_complete_units": 40,
        "answer_or_question_text_stored": False,
    }
    metadata = {
        "source_v66_training_fit_state_sha256": "a" * 64,
        "base_checkpoint_sha256": "c" * 64,
        "base_runtime_config_sha256": "d" * 64,
    }
    saved_checks = gate._final_training_checks(final_metrics, thresholds)
    attestation = gate._canonical_sha256(
        {
            "schema_version": 1,
            "artifact": "v66_saved_runtime_training_gate_attestation",
            "training_fit_state_sha256": "a" * 64,
            "production_device": "cpu",
            "raw_question_token_embeddings_used": True,
            "behavior": final_metrics,
            "checks": saved_checks,
            "answer_or_question_text_stored": False,
        }
    )
    metadata["saved_runtime_training_gate_attestation_sha256"] = attestation
    report = {
        "schema_version": 1,
        "artifact": gate.TRAINING_REPORT_ARTIFACT,
        "promotion_eligible": False,
        "terminal_reason": ("training_and_paired_dependence_gates_passed_checkpoint_saved"),
        "authorization": {
            "baseline_lock_sha256": "b" * 64,
            "preregistration_sha256": "9" * 64,
            "filtered_training_qa_sha256": "f" * 64,
            "training_baseline_lock_sha256": (gate.PINNED_TRAINING_BASELINE_LOCK_SHA256),
        },
        "architecture": {
            "name": gate.ARCHITECTURE,
            "complete_scene_prefix": True,
            "scene_latents": 256,
            "question_dependent_scene_retrieval": False,
            "environmental_text_inputs": [],
            "runtime_answer_codebook": False,
        },
        "teacher_audit": {"every_answer_class_has_verified_teacher": True},
        "work_manifest_sha256": "8" * 64,
        "thresholds": thresholds,
        "cv": {
            "protocol": "leave_one_counterfactual_pair_out_all_576_rows",
            "metrics": cv_metrics,
            "checks": gate._cv_training_checks(cv_metrics, thresholds),
            "passed": True,
            "folds": [
                {
                    "held_pair_id": pair_id,
                    "held_rows_used_for_optimization": False,
                    "held_teacher_sources_used": False,
                    "behavior": {},
                    "fit": {},
                }
                for pair_id in boundary.TRAIN_PAIR_IDS
            ],
        },
        "scope": {
            "training_only": True,
            "gemma_frozen": True,
            "gemma_backward_used": False,
            "validation_inputs_used": False,
            "scorer_inputs_used": False,
            "oracle_loaded": False,
            "fresh_development_loaded": False,
            "deferred_final_loaded": False,
            "question_or_answer_text_stored": False,
        },
        "final_fit": {
            "metrics": final_metrics,
            "checks": saved_checks,
            "passed": True,
            "fit_state_sha256": "7" * 64,
        },
        "paired_opposite_scene_dependence": {
            "protocol": (
                "exact_counterfactual_paired_opposite_scene_prefix_and_signature_"
                "same_byte_identical_question"
            ),
            "metrics": dependence_metrics,
            "checks": gate._paired_opposite_training_checks(dependence_metrics, thresholds),
            "passed": True,
        },
        "checkpoint": {
            "weights_sha256": "1" * 64,
            "runtime_metadata_sha256": "2" * 64,
            "source_v66_training_fit_state_sha256": "a" * 64,
        },
        "saved_runtime_reload": {
            "strict_loader_passed": True,
            "architecture": gate.ARCHITECTURE,
            "training_fit_state_sha256": "a" * 64,
            "gate_attestation_sha256": attestation,
            "reloaded_state_exact": True,
            "raw_question_token_embeddings_used": True,
            "production_device": "cpu",
            "metrics": final_metrics,
            "checks": saved_checks,
            "passed_before_publication": True,
        },
    }
    return report, metadata


def test_training_report_authenticates_all_v66b_gates(tmp_path: Path) -> None:
    report, metadata = _synthetic_training_report()
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report) + "\n", encoding="utf-8")

    digest, artifact = gate._validate_training_report(
        path,
        training_preregistration={"thresholds": prereg.TRAINING_THRESHOLDS},
        training_preregistration_sha256="9" * 64,
        baseline_sha256="b" * 64,
        parent_preregistration={"artifacts": {"filtered_training": {"sha256": "f" * 64}}},
        base_checkpoint_sha256="c" * 64,
        runtime_config_sha256="d" * 64,
        control_files={
            "control.safetensors": {"sha256": "1" * 64},
            "runtime_metadata.json": {"sha256": "2" * 64},
        },
        control_metadata=metadata,
    )

    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert artifact == gate.TRAINING_REPORT_ARTIFACT


def test_training_report_rejects_failed_paired_scene_gate(tmp_path: Path) -> None:
    report, metadata = _synthetic_training_report()
    report["paired_opposite_scene_dependence"]["passed"] = False
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="paired-opposite scene-dependence"):
        gate._validate_training_report(
            path,
            training_preregistration={"thresholds": prereg.TRAINING_THRESHOLDS},
            training_preregistration_sha256="9" * 64,
            baseline_sha256="b" * 64,
            parent_preregistration={"artifacts": {"filtered_training": {"sha256": "f" * 64}}},
            base_checkpoint_sha256="c" * 64,
            runtime_config_sha256="d" * 64,
            control_files={
                "control.safetensors": {"sha256": "1" * 64},
                "runtime_metadata.json": {"sha256": "2" * 64},
            },
            control_metadata=metadata,
        )


def test_terminal_claim_exists_before_reference_validator(monkeypatch: Any, tmp_path: Path) -> None:
    candidate, references = _candidate()
    claim = tmp_path / "claim.json"
    output = tmp_path / "terminal.json"
    scorer = tmp_path / "scorer_only" / "references.json"
    scorer.parent.mkdir()
    scorer.write_text("synthetic", encoding="utf-8")

    monkeypatch.setattr(gate, "authenticate_candidate_inputs", lambda **_kwargs: candidate)

    def validate_after_claim(*_args: Any, **_kwargs: Any) -> tuple[Any, str]:
        assert claim.is_file()
        assert not output.exists()
        return references, prereg.PINNED_SCORER_REFERENCES_SHA256

    monkeypatch.setattr(gate.common, "_validate_scorer_records", validate_after_claim)

    result = gate.seal_terminal(
        candidate_predictions="unused",
        scene_swap_predictions="unused",
        scorer_references=scorer,
        internal_preregistration="unused",
        parent_preregistration="unused",
        training_preregistration="unused",
        baseline_lock="unused",
        questions_manifest="unused",
        base_checkpoint="unused",
        control_checkpoint="unused",
        runtime_config="unused",
        training_report="unused",
        launch_claim=claim,
        output=output,
    )

    assert result["passed"] is True
    assert claim.is_file()
    assert output.is_file()
    assert result["access_boundary"]["scorer_reference_open_count_in_terminal_process"] == 1
