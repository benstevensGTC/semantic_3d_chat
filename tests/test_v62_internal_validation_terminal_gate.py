from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from semantic_3d_chat.evaluation import v62_internal_validation_terminal_gate as gate
from semantic_3d_chat.evaluation import v62_pair_disjoint_preregistration as boundary
from semantic_3d_chat.evaluation.predict_v62_scene_swap import paired_scene_ids
from semantic_3d_chat.evaluation.question_manifest import QuestionManifest, QuestionRecord


def _audit(control_used: bool) -> dict[str, Any]:
    activation = 0.02 if control_used else 0.005
    probability = 1.0 / (1.0 + math.exp(-((activation - 0.01) / 0.01)))
    return {
        "architecture": gate.V65_ARCHITECTURE,
        "environment_latent_count": 256,
        "every_scene_token_influenced_output": True,
        "question_dependent_scene_retrieval": False,
        "softmax_scene_attention_used": False,
        "control_values_scene_question_bilinear": True,
        "gate_scene_question_conditioned": True,
        "exact_no_control_below_threshold": True,
        "saved_runtime_training_gate_required": True,
        "activation_rms": activation,
        "activation_rms_threshold": 0.01,
        "maximum_control_rms": activation,
        "gate_probability": probability,
        "control_used": control_used,
        "exact_no_control_route": not control_used,
    }


def _candidate() -> tuple[gate.CandidateInputs, tuple[dict[str, Any], ...]]:
    specs_by_id = {spec.pair_id: spec for spec in boundary.PAIR_INVENTORY}
    specs = [specs_by_id[pair_id] for pair_id in boundary.INTERNAL_VALIDATION_PAIR_IDS]
    prefixes = {
        scene_id: hashlib.sha256(scene_id.encode()).hexdigest()
        for spec in specs
        for scene_id in spec.scene_ids
    }
    questions: list[QuestionRecord] = []
    references: list[dict[str, Any]] = []
    natural: list[dict[str, Any]] = []
    swaps: list[dict[str, Any]] = []
    baseline_outputs: list[dict[str, str]] = []
    ordinal = 1
    control_sha = "c" * 64
    provenance_sha = "d" * 64
    swap_provenance_sha = "e" * 64
    for spec in specs:
        for unit_index in range(24):
            changed = unit_index < spec.changed_unit_count
            question = "Is there a chair?"
            question_key = f"cfq_{ordinal:016x}"
            side_rows: list[dict[str, Any]] = []
            for scene_index, scene_id in enumerate(spec.scene_ids):
                question_id = f"q_{ordinal:06d}"
                ordinal += 1
                answer = ("yes" if scene_index == 0 else "no") if changed else "yes"
                questions.append(QuestionRecord(scene_id, question_id, question))
                reference = {
                    "scene_id": scene_id,
                    "question_id": question_id,
                    "answer": answer,
                    "answer_items": None,
                    "answer_type": "presence",
                    "route_label": changed,
                    "counterfactual_pair_id": spec.pair_id,
                    "counterfactual_paired_scene_id": spec.scene_ids[1 - scene_index],
                    "counterfactual_question_key": question_key,
                    "counterfactual_change_type": spec.change_type,
                    "counterfactual_role": ("reference" if scene_index == 0 else "counterfactual"),
                }
                references.append(reference)
                side_rows.append(reference)
                natural.append(
                    {
                        "scene_id": scene_id,
                        "question_id": question_id,
                        "predicted_answer": answer,
                        "prefix_hash": prefixes[scene_id],
                        "control_checkpoint_sha256": control_sha,
                        "control_audit": _audit(changed),
                        "provenance_sha256": provenance_sha,
                    }
                )
                baseline_outputs.append(
                    {
                        "scene_id": scene_id,
                        "question_id": question_id,
                        "raw_output_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                    }
                )
            for scene_index, reference in enumerate(side_rows):
                opposite = side_rows[1 - scene_index]
                source_scene = str(reference["scene_id"])
                injected_scene = str(opposite["scene_id"])
                swaps.append(
                    {
                        "scene_id": source_scene,
                        "question_id": reference["question_id"],
                        "injected_scene_id": injected_scene,
                        "predicted_answer": opposite["answer"],
                        "prefix_hash": prefixes[injected_scene],
                        "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                        "control_checkpoint_sha256": control_sha,
                        "control_audit": _audit(changed),
                        "provenance_sha256": swap_provenance_sha,
                    }
                )
    manifest = QuestionManifest(
        questions=tuple(questions),
        questions_sha256="1" * 64,
        source_qa_sha256="2" * 64,
    )
    preregistration = {
        "thresholds": {
            "internal_validation": gate._EXPECTED_INTERNAL_VALIDATION_THRESHOLDS,
            "same_question_different_prefix_control": gate._EXPECTED_SAME_PREFIX_THRESHOLDS,
            "scene_swap_control": gate._EXPECTED_SWAP_THRESHOLDS,
        },
        "artifacts": {
            "scorer_references": {
                "sha256": "3" * 64,
                "records_sha256": "4" * 64,
            }
        },
    }
    candidate = gate.CandidateInputs(
        preregistration=preregistration,
        preregistration_sha256="5" * 64,
        questions=manifest,
        questions_manifest_sha256="6" * 64,
        baseline={
            "scene_prefix_hashes": prefixes,
            "required_output_hashes": baseline_outputs,
        },
        baseline_sha256="7" * 64,
        natural_rows=tuple(natural),
        natural_sha256="8" * 64,
        natural_provenance={},
        natural_provenance_sha256="9" * 64,
        swap_rows=tuple(swaps),
        swap_sha256="a" * 64,
        swap_provenance={},
        swap_provenance_sha256="b" * 64,
        base_checkpoint_sha256="c" * 64,
        base_checkpoint_files=(),
        control_checkpoint_sha256=control_sha,
        control_files={},
        control_metadata={
            "architecture": gate.V65_ARCHITECTURE,
            "schema_version": 6,
            "saved_runtime_training_gate_attestation_sha256": "0" * 64,
        },
        runtime_config_file_sha256="d" * 64,
        runtime_config_effective_sha256="e" * 64,
        training_report_sha256="f" * 64,
        training_report_artifact="v65_magnitude_gated_canonical_answer_distillation",
    )
    return candidate, tuple(references)


def _sealed_v65_report() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    retention = {
        "retention_inventory_exact": True,
        "every_retention_row_exact_no_control": True,
        "base_output_identity_by_construction": True,
    }
    final_behavior = {
        "side_exact": 76,
        "side_total": 80,
        "complete_units": 36,
        "unit_total": 40,
    }
    final_checks = {**gate._v65_final_checks(final_behavior), **retention}
    saved_reload: dict[str, Any] = {
        "strict_loader_passed": True,
        "architecture": gate.V65_ARCHITECTURE,
        "training_fit_state_sha256": "1" * 64,
        "reloaded_state_exact": True,
        "raw_question_token_embeddings_used": True,
        "production_device": "mps",
        "changed_behavior": final_behavior,
        "retention": retention,
        "checks": final_checks,
        "passed_before_publication": True,
    }
    attestation_payload = {
        "schema_version": 1,
        "artifact": "v65_saved_runtime_training_gate_attestation",
        "training_fit_state_sha256": "1" * 64,
        "production_device": "mps",
        "raw_question_token_embeddings_used": True,
        "changed_behavior": final_behavior,
        "retention": retention,
        "checks": final_checks,
        "answer_or_question_text_stored": False,
    }
    attestation = hashlib.sha256(gate._canonical_json_bytes(attestation_payload)).hexdigest()
    saved_reload["gate_attestation_sha256"] = attestation
    metadata = {
        "activation_rms_threshold": 0.01,
        "source_v65_training_fit_state_sha256": "1" * 64,
        "source_v65_value_state_sha256": "1" * 64,
        "saved_runtime_training_gate_attestation_sha256": attestation,
    }
    files = {
        "control.safetensors": {"sha256": "2" * 64, "size_bytes": 12},
        "runtime_metadata.json": {"sha256": "3" * 64, "size_bytes": 34},
    }
    cv_behavior = {
        "supported_side_exact": 45,
        "supported_side_total": 60,
        "unsupported_side_total": 20,
        "fully_supported_complete_units": 19,
        "fully_supported_unit_total": 28,
        "eligible_folds_with_exact_hit": 7,
        "eligible_fold_count": 8,
        "side_total": 80,
        "unit_total": 40,
        "pair_count": 12,
    }
    cv_checks = gate._v65_cv_checks(cv_behavior)
    folds = [
        {
            "held_pair_id": pair_id,
            "training_pair_count": 11,
            "held_scene_question_examples_used_for_optimization": False,
            "held_teacher_used_in_codebook_or_basis": False,
            "generation_semantics": gate._V65_GENERATION_SEMANTICS,
            "retention": retention,
        }
        for pair_id in boundary.TRAIN_PAIR_IDS
    ]
    report = {
        "schema_version": 2,
        "artifact": "v65_magnitude_gated_canonical_answer_distillation",
        "offline_checks_passed": True,
        "promotion_eligible": False,
        "successor_factorized_route_required": False,
        "terminal_reason": "training_behavior_gates_passed_checkpoint_saved",
        "authorization": {
            "baseline_lock_sha256": "4" * 64,
            "training_baseline_lock_sha256": (gate._PINNED_V65_TRAINING_BASELINE_LOCK_SHA256),
            "filtered_training_qa_sha256": "5" * 64,
            "baseline_validated_before_training_data": True,
            "training_v54_hash_inventory_count": 576,
        },
        "inputs": {
            "training_record_count": 576,
            "training_scene_count": 24,
            "training_pair_count": 12,
            "changed_teacher_side_count": 80,
            "changed_paired_unit_count": 40,
        },
        "base": {
            "checkpoint_sha256": "6" * 64,
            "runtime_config_effective_sha256": "7" * 64,
        },
        "scope": {
            "training_answers_used_only_to_build_numeric_codebook_and_score_training": True,
            "runtime_answer_strings": False,
            "training_v54_output_hashes_only": True,
            "gemma_backward_used": False,
            "validation_inputs_used": False,
            "scorer_inputs_used": False,
            "prediction_inputs_used": False,
            "oracle_loaded": False,
            "fresh_development_loaded": False,
            "deferred_final_loaded": False,
        },
        "architecture": {
            "name": gate.V65_ARCHITECTURE,
            "runtime_schema_version": 6,
            "hidden_size": 1536,
            "control_tokens": 4,
            "global_scene_latents": 256,
            "activation_rms_aggregation": "maximum_over_control_tokens",
            "activation_rms_threshold": 0.01,
            "exact_no_control_below_threshold": True,
            "unified_scene_question_value_and_route": True,
            "question_dependent_scene_retrieval": False,
            "complete_scene_prefix_retained": True,
        },
        "codebook": {
            "final_all_training_only": True,
            "folds_use_separate_training_only_codebooks": True,
            "held_fold_label_codebook_visible": False,
            "held_teacher_used_in_fold_codebook_or_basis": False,
            "answer_strings_serialized_in_report_or_runtime": False,
        },
        "checkpoint": {
            "weights_sha256": "2" * 64,
            "runtime_metadata_sha256": "3" * 64,
            "source_v65_training_fit_state_sha256": "1" * 64,
            "source_v65_value_state_sha256": "1" * 64,
        },
        "cross_validation": {
            "protocol": "deterministic_leave_one_counterfactual_pair_out",
            "pair_count": 12,
            "each_changed_training_side_generated_exactly_once": True,
            "fold_specific_training_only_codebook_and_basis": True,
            "held_teacher_used_in_fold_codebook_or_basis": False,
            "unsupported_closed_vocabulary_sides_excluded_from_primary_cv_gate": True,
            "held_scene_question_examples_used_for_fold_optimization": False,
            "thresholds": gate._EXPECTED_V65_BEHAVIOR_THRESHOLDS,
            "behavior": cv_behavior,
            "checks": cv_checks,
            "retention": retention,
            "passed": True,
            "folds": folds,
        },
        "final_fit": {
            "behavior": final_behavior,
            "checks": {
                **gate._v65_final_checks(final_behavior),
                "source_v60_question_norm_exact": True,
                "source_v60_question_norm_frozen": True,
                **retention,
            },
            "passed": True,
            "retention": retention,
        },
        "saved_runtime_reload": saved_reload,
    }
    return report, metadata, files


def test_perfect_population_passes_every_preregistered_control() -> None:
    candidate, references = _candidate()
    report = gate.score_populations(candidate, references)

    assert report["passed"] is True
    assert all(report["checks"].values())
    assert report["metrics"]["natural"]["exact"] == 384
    assert report["metrics"]["natural"]["changed_side_exact"] == 52
    assert report["metrics"]["natural"]["changed_paired_unit_complete"] == 26
    assert report["metrics"]["natural"]["changed_paired_unit_correct_direction"] == 26
    assert report["metrics"]["natural"]["retention_exact_raw_identity"] == 332
    assert report["metrics"]["same_question_different_prefix"] == {
        "complete_unit_coverage": 26,
        "question_text_identity": 26,
        "distinct_scene_prefix_hashes": 26,
        "changed_side_exact": 52,
        "changed_paired_unit_complete": 26,
        "correct_changed_direction": 26,
    }
    assert report["metrics"]["scene_swap"]["blind_supplied_side_count"] == 384
    assert report["metrics"]["scene_swap"]["answer_follows_injected_scene"] == 52
    assert report["metrics"]["scene_swap"]["bidirectional_unit_complete"] == 26


def test_one_changed_pair_flipped_loses_direction_and_completeness() -> None:
    candidate, references = _candidate()
    rows = [dict(row) for row in candidate.natural_rows]
    changed = [row for row in references if row["route_label"]]
    first, second = changed[:2]
    by_key = {(row["scene_id"], row["question_id"]): row for row in rows}
    by_key[(first["scene_id"], first["question_id"])]["predicted_answer"] = second["answer"]
    by_key[(second["scene_id"], second["question_id"])]["predicted_answer"] = first["answer"]
    candidate = gate.CandidateInputs(**{**candidate.__dict__, "natural_rows": tuple(rows)})

    report = gate.score_populations(candidate, references)

    assert report["metrics"]["natural"]["changed_paired_unit_complete"] == 25
    assert report["metrics"]["natural"]["changed_paired_unit_correct_direction"] == 25


def test_retention_requires_raw_byte_identity_and_exact_no_control_route() -> None:
    candidate, references = _candidate()
    rows = [dict(row) for row in candidate.natural_rows]
    retention_index = next(
        index for index, reference in enumerate(references) if not reference["route_label"]
    )
    rows[retention_index]["predicted_answer"] = "Yes."
    rows[retention_index]["control_audit"] = _audit(True)
    candidate = gate.CandidateInputs(**{**candidate.__dict__, "natural_rows": tuple(rows)})

    report = gate.score_populations(candidate, references)

    assert report["metrics"]["natural"]["retention_side_exact"] == 332
    assert report["metrics"]["natural"]["retention_exact_raw_identity"] == 331
    assert report["metrics"]["natural"]["retention_exact_no_control_route"] == 331
    assert report["passed"] is False


def test_prediction_audit_requires_schema6_magnitude_gate_semantics() -> None:
    assert gate._valid_prediction_audit(_audit(True)) is True
    assert gate._valid_prediction_audit(_audit(False)) is True

    ungated = _audit(True)
    ungated["saved_runtime_training_gate_required"] = False
    assert gate._valid_prediction_audit(ungated) is False

    threshold_mismatch = _audit(False)
    threshold_mismatch["activation_rms"] = 0.02
    threshold_mismatch["maximum_control_rms"] = 0.02
    assert gate._valid_prediction_audit(threshold_mismatch) is False

    legacy = _audit(True)
    legacy["architecture"] = "normalized_factorized_scene_question_route_v5"
    assert gate._valid_prediction_audit(legacy) is False


def test_terminal_accepts_only_attested_saved_runtime_v65_report(tmp_path: Path) -> None:
    report, metadata, files = _sealed_v65_report()
    report_path = tmp_path / "v65.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    digest, artifact = gate._validate_training_report(
        report_path,
        baseline_sha256="4" * 64,
        preregistration={"artifacts": {"filtered_training": {"sha256": "5" * 64}}},
        base_checkpoint_sha256="6" * 64,
        runtime_config_sha256="7" * 64,
        control_files=files,
        control_metadata=metadata,
    )

    assert len(digest) == 64
    assert artifact == "v65_magnitude_gated_canonical_answer_distillation"


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("saved_runtime_reload", "passed_before_publication"), False),
        (("saved_runtime_reload", "gate_attestation_sha256"), "8" * 64),
        (("cross_validation", "passed"), False),
        (("final_fit", "passed"), False),
        (("checkpoint", "weights_sha256"), "9" * 64),
    ),
)
def test_terminal_rejects_ungated_or_tampered_v65_report(
    tmp_path: Path,
    path: tuple[str, str],
    value: object,
) -> None:
    report, metadata, files = _sealed_v65_report()
    report[path[0]][path[1]] = value
    report_path = tmp_path / "tampered.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="V65"):
        gate._validate_training_report(
            report_path,
            baseline_sha256="4" * 64,
            preregistration={"artifacts": {"filtered_training": {"sha256": "5" * 64}}},
            base_checkpoint_sha256="6" * 64,
            runtime_config_sha256="7" * 64,
            control_files=files,
            control_metadata=metadata,
        )


def test_blind_scene_swap_pairing_contains_only_reciprocal_opaque_ids() -> None:
    pairing = paired_scene_ids()
    assert len(pairing) == 16
    assert all(pairing[value] == key for key, value in pairing.items())
    assert all(
        key.startswith("scene_") and value.startswith("scene_") for key, value in pairing.items()
    )


def test_swap_validator_accepts_sealed_52_or_blind_384_but_scorer_checks_subset() -> None:
    candidate, references = _candidate()
    gate._validate_swap_predictions(
        candidate.swap_rows,
        manifest=candidate.questions,
        provenance_sha256="e" * 64,
        control_checkpoint_sha256="c" * 64,
    )
    changed_keys = {
        (row["scene_id"], row["question_id"]) for row in references if row["route_label"]
    }
    selected = tuple(
        row for row in candidate.swap_rows if (row["scene_id"], row["question_id"]) in changed_keys
    )
    gate._validate_swap_predictions(
        selected,
        manifest=candidate.questions,
        provenance_sha256="e" * 64,
        control_checkpoint_sha256="c" * 64,
    )
    selected_candidate = gate.CandidateInputs(**{**candidate.__dict__, "swap_rows": selected})
    assert gate.score_populations(selected_candidate, references)["passed"] is True

    wrong_subset = gate.CandidateInputs(
        **{**candidate.__dict__, "swap_rows": candidate.swap_rows[:52]}
    )
    with pytest.raises(ValueError, match="does not cover"):
        gate.score_populations(wrong_subset, references)


def test_launch_claim_is_written_before_scorer_open_and_closes_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, references = _candidate()
    scorer = tmp_path / "scorer_only" / "references.json"
    scorer.parent.mkdir()
    scorer.write_text("sealed", encoding="utf-8")
    claim = tmp_path / "claim.json"
    output = tmp_path / "terminal.json"
    opened = 0

    monkeypatch.setattr(gate, "authenticate_candidate_inputs", lambda **_kwargs: candidate)

    def scorer_open(*_args: Any, **_kwargs: Any) -> tuple[tuple[dict[str, Any], ...], str]:
        nonlocal opened
        opened += 1
        assert claim.is_file()
        assert not output.exists()
        return references, "3" * 64

    monkeypatch.setattr(gate, "_validate_scorer_records", scorer_open)
    report = gate.seal_terminal(
        candidate_predictions="unused",
        scene_swap_predictions="unused",
        scorer_references=scorer,
        preregistration="unused",
        baseline_lock="unused",
        questions_manifest="unused",
        base_checkpoint="unused",
        control_checkpoint="unused",
        runtime_config="unused",
        training_report="unused",
        launch_claim=claim,
        output=output,
    )

    assert opened == 1
    assert report["passed"] is True
    assert output.is_file()
    assert report["authorization"]["fresh_development_57_62_one_shot_authorized"] is True
    assert report["authorization"]["deferred_final_25_30_authorized"] is False

    monkeypatch.setattr(
        gate,
        "authenticate_candidate_inputs",
        lambda **_kwargs: pytest.fail("retry reached input authentication"),
    )
    with pytest.raises(FileExistsError, match="launch claim"):
        gate.seal_terminal(
            candidate_predictions="unused",
            scene_swap_predictions="unused",
            scorer_references=scorer,
            preregistration="unused",
            baseline_lock="unused",
            questions_manifest="unused",
            base_checkpoint="unused",
            control_checkpoint="unused",
            runtime_config="unused",
            training_report="unused",
            launch_claim=claim,
            output=output,
        )


def test_failed_internal_gate_does_not_authorize_fresh_development(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, references = _candidate()
    rows = [dict(row) for row in candidate.natural_rows]
    for index, reference in enumerate(references):
        if reference["route_label"]:
            rows[index]["predicted_answer"] = "unknown"
    candidate = gate.CandidateInputs(**{**candidate.__dict__, "natural_rows": tuple(rows)})
    scorer = tmp_path / "scorer_only" / "references.json"
    scorer.parent.mkdir()
    scorer.write_text("sealed", encoding="utf-8")
    monkeypatch.setattr(gate, "authenticate_candidate_inputs", lambda **_kwargs: candidate)
    monkeypatch.setattr(
        gate,
        "_validate_scorer_records",
        lambda *_args, **_kwargs: (references, "3" * 64),
    )

    report = gate.seal_terminal(
        candidate_predictions="unused",
        scene_swap_predictions="unused",
        scorer_references=scorer,
        preregistration="unused",
        baseline_lock="unused",
        questions_manifest="unused",
        base_checkpoint="unused",
        control_checkpoint="unused",
        runtime_config="unused",
        training_report="unused",
        launch_claim=tmp_path / "failed.claim.json",
        output=tmp_path / "failed.terminal.json",
    )

    assert report["passed"] is False
    assert report["authorization"]["fresh_development_57_62_one_shot_authorized"] is False
    assert report["authorization"]["fresh_development_authorization_scope"] == []


def test_cli_has_no_tuning_or_reference_selection_surface() -> None:
    destinations = {action.dest for action in gate._parser()._actions}
    assert {
        "candidate_predictions",
        "scene_swap_predictions",
        "scorer_references",
        "launch_claim",
        "output",
    } <= destinations
    assert destinations.isdisjoint(
        {
            "threshold",
            "retry",
            "candidate",
            "scene_ids",
            "changed_questions",
            "reference_subset",
        }
    )
