from __future__ import annotations

import hashlib
import json
import os
import runpy
import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = runpy.run_path(str(ROOT / "scripts/build_current_report.py"))


def _summary() -> dict[str, Any]:
    return BUILDER["build_summary"]()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_sealed_schema7(root: Path) -> Path:
    checkpoint = root / "sealed"
    checkpoint.mkdir()
    weights = checkpoint / "control.safetensors"
    weights.write_bytes(b"numeric-test-weights")
    metadata = {
        "schema_version": 7,
        "architecture": "always_on_teacher_basis_full_scene_control_v7",
        "always_on_continuous_control": True,
        "complete_scene_prefix_required": True,
        "question_dependent_scene_retrieval": False,
        "environmental_text_inputs": [],
        "training_answers_runtime_loaded": False,
        "answer_class_codebook_runtime_loaded": False,
        "saved_runtime_training_gate_required": True,
        "saved_runtime_training_gate_passed": True,
        "saved_runtime_training_gate_attestation_sha256": "4" * 64,
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
    }
    (checkpoint / "runtime_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return checkpoint


def _demo_check(checkpoint: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["RESEARCH_CONTROL_CHECKPOINT"] = str(checkpoint)
    return subprocess.run(
        ["./scripts/run_research_demo.sh", "--check"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _strict_demo_check() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["./scripts/run_strict_fixed_prefix_demo.sh", "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_current_summary_is_claim_bounded_and_matches_measured_artifacts() -> None:
    summary = _summary()

    assert summary["schema"] == "semantic_3d_chat.current_metrics.v1"
    assert summary["claim_scope"]["final_acceptance_claimed"] is False
    assert summary["geometry"]["passed"] is True
    assert summary["reference_map"]["feature_dim"] == 3072
    assert summary["reference_map"]["occupied_voxels"] == 74699
    semantic = summary["zero_shot_semantic_localization"]
    assert semantic["development_scene_count"] == 17
    assert semantic["scorable_query_count"] == 221
    assert semantic["mean_top1_localization_accuracy"] == pytest.approx(0.48416289592760187)
    qa = summary["static_qa_development"]
    assert qa["question_count"] == 216
    assert qa["normalized_exact_accuracy"] == pytest.approx(89 / 216)
    assert qa["acceptance_gate_passed"] is False

    sources = tuple(summary["source_artifacts"])
    assert sources
    assert not any(
        forbidden in path
        for path in sources
        for forbidden in (
            "/oracle/",
            "/qa/",
            "scorer_only",
            "final_once",
            "scene_000025",
            "scene_000026",
            "scene_000027",
            "scene_000028",
            "scene_000029",
            "scene_000030",
        )
    )


def test_current_markdown_is_explicit_about_pending_schema7() -> None:
    markdown = BUILDER["render_markdown"](_summary())

    assert "not a final acceptance claim" in markdown
    assert "V66b completed its preregistered pair-disjoint training gate and failed" in markdown
    assert "37/75 changed sides" in markdown
    assert "No V66b oracle-deletion success is claimed" in markdown
    assert "48.42%" in markdown
    assert "make research-demo-check" in markdown


def test_current_summary_records_failed_v66b_gate_without_a_checkpoint() -> None:
    summary = _summary()
    training = summary["schema7_preregistered_training"]
    result = training["result"]

    assert training["training_result_measured"] is True
    assert result["status"] == "pair_disjoint_gate_failed_checkpoint_not_published"
    assert result["passed"] is False
    assert result["checkpoint_published"] is False
    assert result["supported_exact"] == 409
    assert result["supported_total"] == 571
    assert result["changed_side_exact"] == 37
    assert result["changed_side_total"] == 75
    assert result["complete_changed_units"] == 5
    assert result["changed_unit_total"] == 35
    assert result["prediction_change_units"] == 16
    assert result["failed_checks"] == [
        "held_changed_side_exact",
        "held_complete_units",
        "held_prediction_change_units",
        "per_type_spatial_relation",
    ]
    assert all(result["contract_checks"].values())


def test_current_summary_authenticates_failed_v67_screen_without_publication() -> None:
    summary = _summary()
    screen = summary["v67_pair_objective_numeric_screen"]

    assert screen["status"] == "authenticated_numeric_screen_failed_no_publication"
    assert screen["measurement_status"] == "authenticated_screen_failed"
    assert screen["measurement_authenticated"] is True
    assert screen["passed"] is False
    assert screen["promotion_eligible"] is False
    assert screen["gemma_generation_used"] is False
    assert screen["full_behavioral_run_executed"] is False
    assert screen["checkpoint_published"] is False
    assert screen["checkpoint_absent"] is True
    assert screen["fold_count"] == 12
    assert screen["metrics"]["supported_class_exact"] == 482
    assert screen["metrics"]["supported_total"] == 571
    assert screen["metrics"]["changed_class_exact"] == 52
    assert screen["metrics"]["changed_total"] == 75
    assert screen["metrics"]["complete_class_units"] == 13
    assert screen["metrics"]["complete_unit_total"] == 35
    assert screen["metrics"]["prediction_change_units"] == 14
    assert screen["metrics"]["positive_own_over_opposite_sides"] == 47
    assert screen["metrics"]["fully_supported_pair_sides"] == 70
    assert screen["thresholds"]["held_complete_class_units_minimum"] == 15
    assert screen["thresholds"]["held_prediction_change_units_minimum"] == 20
    assert screen["thresholds"]["positive_own_over_opposite_sides_minimum"] == 53
    assert screen["failed_checks"] == [
        "held_complete_class_units",
        "held_prediction_change_units",
        "positive_own_over_opposite_sides",
    ]
    assert all(screen["authentication_checks"].values())
    assert len(screen["preregistration_sha256"]) == 64
    assert len(screen["numeric_screen_sha256"]) == 64
    assert len(screen["training_identity_sha256"]) == 64
    for path, digest in screen["implementation_source_sha256"].items():
        assert summary["source_artifacts"][path] == digest
    assert (
        "reports/gemma4/metrics/v67_pair_objective_preregistration.json"
        in summary["source_artifacts"]
    )
    assert (
        "reports/gemma4/metrics/v67_pair_objective_numeric_screen.json"
        in summary["source_artifacts"]
    )

    markdown = BUILDER["render_markdown"](summary)
    assert "13/35 (minimum 15)" in markdown
    assert "14/35 (minimum 20)" in markdown
    assert "47/70 (minimum 53)" in markdown
    assert "authorized no Gemma generation or full run" in markdown
    assert "no V67 checkpoint was published" in markdown


def test_v67_screen_authentication_fails_closed_on_artifact_or_source_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inspector = BUILDER["_inspect_v67_numeric_screen"]
    inspector_globals = inspector.__globals__
    preregistration = tmp_path / "preregistration.json"
    screen = tmp_path / "screen.json"
    preregistration_bytes = (
        ROOT / "reports/gemma4/metrics/v67_pair_objective_preregistration.json"
    ).read_bytes()
    screen_bytes = (
        ROOT / "reports/gemma4/metrics/v67_pair_objective_numeric_screen.json"
    ).read_bytes()
    preregistration.write_bytes(preregistration_bytes)
    screen.write_bytes(screen_bytes)
    checkpoint = tmp_path / "checkpoint"
    full_report = tmp_path / "full.json"
    replacements = {
        "V67_PREREGISTRATION": preregistration,
        "V67_NUMERIC_SCREEN": screen,
        "V67_CHECKPOINT": checkpoint,
        "V67_FULL_TRAINING_REPORT": full_report,
    }
    for name, value in replacements.items():
        monkeypatch.setitem(inspector_globals, name, value)

    authenticated = inspector()
    assert authenticated["measurement_authenticated"] is True

    screen.write_bytes(screen.read_bytes() + b" ")
    rejected_artifact = inspector()
    assert rejected_artifact["measurement_status"] == ("artifact_present_authentication_failed")
    assert rejected_artifact["measurement_authenticated"] is False
    assert "numeric-screen digest differs" in rejected_artifact["measurement_evidence_error"]

    screen.write_bytes(screen_bytes)
    source_hashes = dict(BUILDER["V67_SOURCE_SHA256"])
    original_source = next(iter(source_hashes))
    copied_source = tmp_path / original_source.name
    copied_source.write_bytes((ROOT / original_source).read_bytes())
    source_hashes[copied_source] = source_hashes.pop(original_source)
    monkeypatch.setitem(inspector_globals, "V67_SOURCE_SHA256", source_hashes)
    assert inspector()["measurement_authenticated"] is True

    copied_source.write_bytes(copied_source.read_bytes() + b"\n# tampered\n")
    rejected_source = inspector()
    assert rejected_source["measurement_authenticated"] is False
    assert "implementation source digest differs" in rejected_source["measurement_evidence_error"]


def test_v67_screen_is_optional_and_checkpoint_presence_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inspector = BUILDER["_inspect_v67_numeric_screen"]
    inspector_globals = inspector.__globals__
    preregistration = tmp_path / "missing_preregistration.json"
    screen = tmp_path / "missing_screen.json"
    preregistration_bytes = (
        ROOT / "reports/gemma4/metrics/v67_pair_objective_preregistration.json"
    ).read_bytes()
    screen_bytes = (
        ROOT / "reports/gemma4/metrics/v67_pair_objective_numeric_screen.json"
    ).read_bytes()
    monkeypatch.setitem(inspector_globals, "V67_PREREGISTRATION", preregistration)
    monkeypatch.setitem(inspector_globals, "V67_NUMERIC_SCREEN", screen)
    monkeypatch.setitem(inspector_globals, "V67_CHECKPOINT", tmp_path / "checkpoint")
    monkeypatch.setitem(inspector_globals, "V67_FULL_TRAINING_REPORT", tmp_path / "full.json")

    missing = inspector()
    assert missing["measurement_status"] == "not_measured"
    assert missing["measurement_authenticated"] is False

    preregistration.write_bytes(preregistration_bytes)
    screen.write_bytes(screen_bytes)
    checkpoint = inspector_globals["V67_CHECKPOINT"]
    checkpoint.mkdir()
    rejected = inspector()
    assert rejected["measurement_authenticated"] is False
    assert "checkpoint exists after a failed screen" in rejected["measurement_evidence_error"]


def test_current_summary_authenticates_failed_v68_grid_without_publication() -> None:
    summary = _summary()
    grid = summary["v68_regularized_pair_numeric_grid"]

    assert grid["status"] == ("authenticated_all_arm_numeric_grid_failed_no_publication")
    assert grid["measurement_status"] == "authenticated_grid_failed"
    assert grid["measurement_authenticated"] is True
    assert grid["passed"] is False
    assert grid["promotion_eligible"] is False
    assert grid["gemma_generation_used"] is False
    assert grid["full_behavioral_run_executed"] is False
    assert grid["checkpoint_published"] is False
    assert grid["checkpoint_absent"] is True
    assert grid["selected_arm_id"] is None
    assert grid["arm_count"] == 3
    assert [arm["arm_id"] for arm in grid["arms"]] == [
        "balanced_all_value_anchor",
        "interaction_only_anchor",
        "strong_all_value_anchor",
    ]
    expected = [
        (
            489,
            13,
            14,
            47,
            {
                "held_complete_class_units": 2,
                "held_prediction_change_units": 6,
                "positive_own_over_opposite_sides": 6,
            },
        ),
        (
            484,
            13,
            14,
            50,
            {
                "held_complete_class_units": 2,
                "held_prediction_change_units": 6,
                "positive_own_over_opposite_sides": 3,
            },
        ),
        (
            489,
            14,
            17,
            50,
            {
                "held_complete_class_units": 1,
                "held_prediction_change_units": 3,
                "positive_own_over_opposite_sides": 3,
            },
        ),
    ]
    for arm, (supported, complete, changed, margins, gaps) in zip(
        grid["arms"], expected, strict=True
    ):
        assert arm["fold_count"] == 12
        assert arm["metrics"]["supported_class_exact"] == supported
        assert arm["metrics"]["supported_total"] == 571
        assert arm["metrics"]["complete_class_units"] == complete
        assert arm["metrics"]["prediction_change_units"] == changed
        assert arm["metrics"]["positive_own_over_opposite_sides"] == margins
        assert arm["failed_checks"] == [
            "held_complete_class_units",
            "held_prediction_change_units",
            "positive_own_over_opposite_sides",
        ]
        assert arm["fail_gaps"] == gaps
        assert arm["gate_checks"]["no_generation_used"] is True
    assert all(grid["authentication_checks"].values())
    assert len(grid["preregistration_sha256"]) == 64
    assert len(grid["numeric_grid_sha256"]) == 64
    assert len(grid["training_identity_sha256"]) == 64
    assert (
        "reports/gemma4/metrics/v68_regularized_pair_preregistration.json"
        in summary["source_artifacts"]
    )
    assert (
        "reports/gemma4/metrics/v68_regularized_pair_numeric_grid.json"
        in summary["source_artifacts"]
    )

    markdown = BUILDER["render_markdown"](summary)
    assert "Every arm executed all twelve pair-held-out folds" in markdown
    assert "| strong_all_value_anchor | 489/571 | 14/35 | 17/35 | 50/70 |" in markdown
    assert "missed complete units by\n1" in markdown
    assert "used\nno Gemma generation" in markdown
    assert "published no V68 checkpoint" in markdown


def test_v68_inspector_authenticates_real_all_arm_fold_evidence() -> None:
    grid = BUILDER["_inspect_v68_numeric_grid"]()

    assert grid["measurement_authenticated"] is True
    assert grid["measurement_status"] == "authenticated_grid_failed"
    assert grid["gemma_generation_used"] is False
    assert grid["checkpoint_absent"] is True
    assert grid["arm_count"] == 3
    assert [arm["fold_count"] for arm in grid["arms"]] == [12, 12, 12]
    assert [arm["fail_gaps"] for arm in grid["arms"]] == [
        {
            "held_complete_class_units": 2,
            "held_prediction_change_units": 6,
            "positive_own_over_opposite_sides": 6,
        },
        {
            "held_complete_class_units": 2,
            "held_prediction_change_units": 6,
            "positive_own_over_opposite_sides": 3,
        },
        {
            "held_complete_class_units": 1,
            "held_prediction_change_units": 3,
            "positive_own_over_opposite_sides": 3,
        },
    ]
    assert all(grid["authentication_checks"].values())


def test_v68_grid_fails_closed_on_fold_tamper_missing_files_and_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inspector = BUILDER["_inspect_v68_numeric_grid"]
    inspector_globals = inspector.__globals__
    original_grid_sha256 = inspector_globals["V68_NUMERIC_GRID_SHA256"]
    preregistration = tmp_path / "preregistration.json"
    screen = tmp_path / "screen.json"
    checkpoint = tmp_path / "checkpoint"
    full_report = tmp_path / "full.json"
    preregistration_bytes = (
        ROOT / "reports/gemma4/metrics/v68_regularized_pair_preregistration.json"
    ).read_bytes()
    screen_bytes = (
        ROOT / "reports/gemma4/metrics/v68_regularized_pair_numeric_grid.json"
    ).read_bytes()
    preregistration.write_bytes(preregistration_bytes)
    screen.write_bytes(screen_bytes)
    for name, value in {
        "V68_PREREGISTRATION": preregistration,
        "V68_NUMERIC_GRID": screen,
        "V68_CHECKPOINT": checkpoint,
        "V68_FULL_TRAINING_REPORT": full_report,
    }.items():
        monkeypatch.setitem(inspector_globals, name, value)

    authenticated = inspector()
    assert authenticated["measurement_authenticated"] is True

    tampered = json.loads(screen.read_text(encoding="utf-8"))
    tampered["arm_results"][0]["folds"][0]["numeric_screen"]["supported_class_exact"] += 1
    _write_json(screen, tampered)
    monkeypatch.setitem(inspector_globals, "V68_NUMERIC_GRID_SHA256", _sha256(screen))
    rejected_fold = inspector()
    assert rejected_fold["measurement_authenticated"] is False
    assert "integer aggregates differ from folds" in rejected_fold["measurement_evidence_error"]

    screen.write_bytes(screen_bytes)
    tampered_checks = json.loads(screen.read_text(encoding="utf-8"))
    tampered_checks["arm_results"][1]["checks"]["held_prediction_change_units"] = True
    _write_json(screen, tampered_checks)
    monkeypatch.setitem(inspector_globals, "V68_NUMERIC_GRID_SHA256", _sha256(screen))
    rejected_gate = inspector()
    assert rejected_gate["measurement_authenticated"] is False
    assert "gate checks differ from recomputation" in rejected_gate["measurement_evidence_error"]

    screen.write_bytes(screen_bytes)
    monkeypatch.setitem(
        inspector_globals,
        "V68_NUMERIC_GRID_SHA256",
        original_grid_sha256,
    )
    screen.unlink()
    preregistered_only = inspector()
    assert preregistered_only["measurement_status"] == "not_measured"
    assert preregistered_only["preregistration_authenticated"] is True

    screen.write_bytes(screen_bytes)
    preregistration.unlink()
    missing_preregistration = inspector()
    assert missing_preregistration["measurement_authenticated"] is False
    assert "preregistration is missing" in missing_preregistration["measurement_evidence_error"]

    screen.unlink()
    missing_both = inspector()
    assert missing_both["measurement_status"] == "not_measured"
    assert missing_both["measurement_authenticated"] is False

    preregistration.write_bytes(preregistration_bytes)
    screen.write_bytes(screen_bytes)
    checkpoint.mkdir()
    unexpected_checkpoint = inspector()
    assert unexpected_checkpoint["measurement_authenticated"] is False
    assert (
        "checkpoint exists after a failed grid"
        in unexpected_checkpoint["measurement_evidence_error"]
    )


def _fake_v69_passing_screen() -> dict[str, Any]:
    thresholds = BUILDER["_read_object"](BUILDER["V69_PREREGISTRATION"])["numeric_screen"][
        "thresholds"
    ]
    aggregate: dict[str, Any] = {
        "inventory_total": 576,
        "supported_class_exact": thresholds["held_supported_class_exact_minimum"],
        "supported_total": thresholds["held_supported_total"],
        "unsupported_total": thresholds["held_unsupported_total"],
        "changed_class_exact": thresholds["held_changed_class_exact_minimum"],
        "changed_total": thresholds["held_changed_total"],
        "complete_class_units": thresholds["held_complete_class_units_minimum"],
        "complete_unit_total": thresholds["held_complete_unit_total"],
        "prediction_change_units": thresholds["held_prediction_change_units_minimum"],
        "pair_delta_cosine_sum": (
            thresholds["mean_pair_delta_cosine_minimum"] * thresholds["held_complete_unit_total"]
        ),
        "positive_pair_delta_units": thresholds["positive_pair_delta_units_minimum"],
        "own_over_opposite_margin_sum": (
            thresholds["mean_own_over_opposite_margin_minimum"]
            * thresholds["fully_supported_pair_sides"]
        ),
        "positive_own_over_opposite_sides": thresholds["positive_own_over_opposite_sides_minimum"],
        "fully_supported_pair_sides": thresholds["fully_supported_pair_sides"],
        "mean_pair_delta_cosine": thresholds["mean_pair_delta_cosine_minimum"],
        "mean_own_over_opposite_margin": thresholds["mean_own_over_opposite_margin_minimum"],
        "answer_or_question_text_stored": False,
        "gemma_generation_used": False,
    }
    integer_fields = (
        "inventory_total",
        "supported_class_exact",
        "supported_total",
        "unsupported_total",
        "changed_class_exact",
        "changed_total",
        "complete_class_units",
        "complete_unit_total",
        "prediction_change_units",
        "positive_pair_delta_units",
        "positive_own_over_opposite_sides",
        "fully_supported_pair_sides",
    )
    float_fields = ("pair_delta_cosine_sum", "own_over_opposite_margin_sum")
    scope = [
        "scene_projection.weight",
        "question_projection.weight",
        "control_trunk.0.weight",
        "control_trunk.0.bias",
        "control_trunk.1.weight",
        "control_trunk.1.bias",
        "coefficient_output.weight",
        "coefficient_output.bias",
        "magnitude_output.weight",
        "magnitude_output.bias",
    ]
    folds: list[dict[str, Any]] = []
    for index, pair_id in enumerate(BUILDER["V68_PAIR_IDS"]):
        numeric = {field: aggregate[field] if index == 0 else 0 for field in integer_fields}
        numeric.update({field: aggregate[field] if index == 0 else 0.0 for field in float_fields})
        numeric.update(
            {
                "question_or_answer_text_stored": False,
                "gemma_generation_used": False,
            }
        )
        foundation = {
            "base_elapsed_seconds": 1.0,
            "pair_refinement_elapsed_seconds": 1.0,
            "base_optimizer_steps": 1,
            "base_classification_optimizer_steps": 1,
            "pair_refinement_optimizer_steps": 1,
            "question_norm_frozen": True,
            "question_norm_sha256": "1" * 64,
            "anchor_state_sha256": "2" * 64,
            "optimizer_parameter_names": scope,
        }
        folds.append(
            {
                "held_pair_id": pair_id,
                "held_rows_used_for_optimization": False,
                "held_teacher_sources_used": False,
                "fold_codebook_sha256": "3" * 64,
                "fold_basis_sha256": "4" * 64,
                "fit": {
                    "foundation": foundation,
                    "augmentation_optimizer_steps": 1,
                    "augmentation_elapsed_seconds": 0.5,
                    "optimizer_parameter_names": scope,
                    "anchor_state_sha256": "5" * 64,
                    "transition_bucket_sizes": [2],
                    "balanced_unit_count": 2,
                    "question_partner_inventory_sha256": "6" * 64,
                    "question_norm_frozen": True,
                    "question_norm_sha256": "1" * 64,
                },
                "numeric_screen": numeric,
            }
        )
    checks = BUILDER["_v68_gate_checks"](aggregate, thresholds)
    arms = BUILDER["V69_ARM_GRID"]
    result = {
        "arm_id": arms[0]["arm_id"],
        "arm_sha256": BUILDER["_v68_canonical_sha256"](dict(arms[0])),
        "status": "passed",
        "passed": True,
        "metrics": aggregate,
        "checks": checks,
        "gemma_generation_used": False,
        "folds": folds,
    }
    skipped = [
        {
            "arm_id": arm["arm_id"],
            "arm_sha256": BUILDER["_v68_canonical_sha256"](dict(arm)),
            "status": "skipped_after_first_pass",
            "passed": None,
            "metrics": None,
            "checks": None,
            "gemma_generation_used": False,
            "folds": [],
        }
        for arm in arms[1:]
    ]
    return {
        "schema_version": 1,
        "artifact": "v69_pair_augmentation_numeric_grid_v1",
        "passed": True,
        "promotion_eligible": False,
        "checkpoint_published": False,
        "gemma_generation_used": False,
        "terminal_reason": "numeric_grid_passed_selected_arm_authorized",
        "preregistration_sha256": BUILDER["V69_PREREGISTRATION_SHA256"],
        "training_identity_sha256": "7" * 64,
        "implementation_source_hashes": {
            path.as_posix(): digest for path, digest in BUILDER["V69_SOURCE_SHA256"].items()
        },
        "authorization": {
            "baseline_lock_sha256": (
                "ff9aef64c85e243219216638163ab308d8aaf6492be7209ae43775fecd283d66"
            ),
            "filtered_training_qa_sha256": (
                "84b99385fadc5d06e44465ada5902f56131192298ca1539373dc3b334608cbf1"
            ),
            "teacher_audit_sha256": (
                "62bdec366c512ff01f6ff215379087ef3bf9fa7d1f4cef291fbd5b5f36166834"
            ),
        },
        "selection": {
            "rule": "run_in_declared_order_and_select_first_all_gate_pass",
            "selected_arm_id": arms[0]["arm_id"],
            "selected_arm_sha256": BUILDER["_v68_canonical_sha256"](dict(arms[0])),
            "later_arms_skipped_after_first_pass": True,
        },
        "thresholds": thresholds,
        "arm_results": [result, *skipped],
        "scope": {
            "training_only": True,
            "numeric_teacher_and_prefix_cache_only": True,
            "question_or_answer_text_stored": False,
            "validation_inputs_used": False,
            "scorer_inputs_used": False,
            "oracle_loaded": False,
            "fresh_development_loaded": False,
            "internal_validation_loaded": False,
            "deferred_final_loaded": False,
        },
    }


def test_current_summary_authenticates_v69_preregistration_while_screen_runs() -> None:
    v69 = _summary()["v69_pair_augmentation_numeric_grid"]

    if v69["measurement_status"] == "not_measured":
        assert v69["status"] == "preregistered_numeric_grid_running_or_not_measured"
        assert v69["preregistration_authenticated"] is True
        assert v69["preregistration_sha256"] == BUILDER["V69_PREREGISTRATION_SHA256"]
    else:
        assert v69["measurement_authenticated"] is True
        assert v69["measurement_status"] in {
            "authenticated_grid_passed",
            "authenticated_grid_failed",
        }
    for path, digest in v69["implementation_source_sha256"].items():
        assert _summary()["source_artifacts"][path] == digest

    markdown = BUILDER["render_markdown"](_summary())
    assert "V69 grid state" in markdown
    assert "V69" in markdown


def test_current_summary_authenticates_sealed_v69_all_arm_failure() -> None:
    v69 = _summary()["v69_pair_augmentation_numeric_grid"]

    assert v69["status"] == "authenticated_all_arm_numeric_grid_failed_no_publication"
    assert v69["measurement_status"] == "authenticated_grid_failed"
    assert v69["measurement_authenticated"] is True
    assert v69["passed"] is False
    assert v69["promotion_eligible"] is False
    assert v69["selected_arm_id"] is None
    assert v69["arm_count"] == 3
    assert v69["executed_arm_count"] == 3
    assert sum(arm["fold_count"] for arm in v69["arms"]) == 36
    assert [arm["metrics"]["prediction_change_units"] for arm in v69["arms"]] == [
        18,
        17,
        16,
    ]
    assert [arm["metrics"]["positive_own_over_opposite_sides"] for arm in v69["arms"]] == [
        50,
        49,
        49,
    ]
    assert all(arm["status"] == "failed" for arm in v69["arms"])
    assert all(arm["passed"] is False for arm in v69["arms"])
    assert v69["gemma_generation_used"] is False
    assert v69["full_behavioral_run_executed"] is False
    assert v69["checkpoint_published"] is False
    assert v69["checkpoint_absent"] is True
    assert v69["numeric_grid_sha256"] == BUILDER["V69_NUMERIC_GRID_SHA256"]
    assert v69["training_identity_sha256"] == BUILDER["V69_TRAINING_IDENTITY_SHA256"]
    assert all(v69["authentication_checks"].values())


def test_v69_inspector_recomputes_first_pass_and_fails_closed_on_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER["_inspect_v69_numeric_grid"]
    globals_ = inspector.__globals__
    preregistration = tmp_path / "preregistration.json"
    screen = tmp_path / "screen.json"
    preregistration.write_bytes(
        (ROOT / "reports/gemma4/metrics/v69_pair_augmentation_preregistration.json").read_bytes()
    )
    payload = _fake_v69_passing_screen()
    _write_json(screen, payload)
    for name, value in {
        "V69_PREREGISTRATION": preregistration,
        "V69_NUMERIC_GRID": screen,
        "V69_CHECKPOINT": tmp_path / "checkpoint",
        "V69_FULL_TRAINING_REPORT": tmp_path / "full.json",
        "V69_NUMERIC_GRID_SHA256": _sha256(screen),
        "V69_TRAINING_IDENTITY_SHA256": "7" * 64,
    }.items():
        monkeypatch.setitem(globals_, name, value)

    authenticated = inspector()
    assert authenticated["measurement_authenticated"] is True
    assert authenticated["measurement_status"] == "authenticated_grid_passed"
    assert authenticated["selected_arm_id"] == "balanced_extrapolation_010"
    assert authenticated["executed_arm_count"] == 1
    assert [arm["status"] for arm in authenticated["arms"]] == [
        "passed",
        "skipped_after_first_pass",
        "skipped_after_first_pass",
    ]
    assert all(authenticated["authentication_checks"].values())

    tampered = json.loads(screen.read_text(encoding="utf-8"))
    tampered["arm_results"][0]["folds"][0]["numeric_screen"]["supported_class_exact"] += 1
    _write_json(screen, tampered)
    monkeypatch.setitem(globals_, "V69_NUMERIC_GRID_SHA256", _sha256(screen))
    rejected = inspector()
    assert rejected["measurement_authenticated"] is False
    assert "integer aggregates differ from folds" in rejected["measurement_evidence_error"]

    _write_json(screen, payload)
    monkeypatch.setitem(globals_, "V69_NUMERIC_GRID_SHA256", _sha256(screen))
    globals_["V69_CHECKPOINT"].mkdir()
    checkpoint_rejected = inspector()
    assert checkpoint_rejected["measurement_authenticated"] is False
    assert "checkpoint exists" in checkpoint_rejected["measurement_evidence_error"]


def test_v69_screen_result_requires_immutable_result_pins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER["_inspect_v69_numeric_grid"]
    globals_ = inspector.__globals__
    preregistration = tmp_path / "preregistration.json"
    screen = tmp_path / "screen.json"
    preregistration.write_bytes(
        (ROOT / "reports/gemma4/metrics/v69_pair_augmentation_preregistration.json").read_bytes()
    )
    _write_json(screen, _fake_v69_passing_screen())
    for name, value in {
        "V69_PREREGISTRATION": preregistration,
        "V69_NUMERIC_GRID": screen,
        "V69_CHECKPOINT": tmp_path / "checkpoint",
        "V69_FULL_TRAINING_REPORT": tmp_path / "full.json",
        "V69_NUMERIC_GRID_SHA256": None,
        "V69_TRAINING_IDENTITY_SHA256": None,
    }.items():
        monkeypatch.setitem(globals_, name, value)

    rejected = inspector()
    assert rejected["measurement_authenticated"] is False
    assert "immutable result pins are not installed" in rejected["measurement_evidence_error"]


def test_current_summary_authenticates_sealed_v70_failure() -> None:
    summary = _summary()
    v70 = summary["v70_low_frequency_moments_numeric_screen"]

    assert v70["status"] == "authenticated_numeric_screen_failed_no_publication"
    assert v70["measurement_status"] == "authenticated_screen_failed"
    assert v70["measurement_authenticated"] is True
    assert v70["passed"] is False
    assert v70["promotion_eligible"] is False
    assert v70["gemma_generation_used"] is False
    assert v70["full_behavioral_run_executed"] is False
    assert v70["atlas_compilation_executed"] is False
    assert v70["checkpoint_published"] is False
    assert v70["checkpoint_absent"] is True
    assert v70["fold_count"] == 12
    assert v70["architecture"] == {
        "candidate_moment_count": 32,
        "complete_scene_prefix": True,
        "environment_latent_count": 256,
        "environmental_text_inputs": [],
        "moment_family": "fixed_low_frequency_dct",
        "question_dependent_scene_retrieval": False,
        "question_independent_scene_prefix": True,
        "source_moment_count": 8,
    }
    assert v70["metrics"]["supported_class_exact"] == 484
    assert v70["metrics"]["supported_total"] == 571
    assert v70["metrics"]["changed_class_exact"] == 55
    assert v70["metrics"]["changed_total"] == 75
    assert v70["metrics"]["complete_class_units"] == 15
    assert v70["metrics"]["complete_unit_total"] == 35
    assert v70["metrics"]["prediction_change_units"] == 16
    assert v70["metrics"]["positive_pair_delta_units"] == 30
    assert v70["metrics"]["positive_own_over_opposite_sides"] == 51
    assert v70["metrics"]["fully_supported_pair_sides"] == 70
    assert v70["metrics"]["mean_pair_delta_cosine"] == pytest.approx(
        0.4998284623026848
    )
    assert v70["metrics"]["mean_own_over_opposite_margin"] == pytest.approx(
        0.11894646883010865
    )
    assert v70["failed_checks"] == [
        "held_prediction_change_units",
        "positive_own_over_opposite_sides",
    ]
    assert v70["fail_gaps"] == {
        "held_prediction_change_units": 4,
        "positive_own_over_opposite_sides": 2,
    }
    assert v70["preregistration_sha256"] == BUILDER["V70_PREREGISTRATION_SHA256"]
    assert v70["numeric_screen_sha256"] == BUILDER["V70_NUMERIC_SCREEN_SHA256"]
    assert v70["training_identity_sha256"] == BUILDER["V70_TRAINING_IDENTITY_SHA256"]
    assert v70["numeric_fit_seconds"] == pytest.approx(385.39903008402325)
    assert v70["total_wall_time_seconds"] == pytest.approx(393.3133014580235)
    assert all(v70["authentication_checks"].values())
    for path, digest in v70["implementation_source_sha256"].items():
        assert summary["source_artifacts"][path] == digest

    markdown = BUILDER["render_markdown"](summary)
    assert "32-low-frequency-moment" in markdown
    assert "16/35" in markdown
    assert "51/70" in markdown
    assert "failed exactly two locked gates" in markdown
    assert "published no V70 checkpoint" in markdown


def test_v70_inspector_recomputes_folds_and_fails_closed_on_tamper_or_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER["_inspect_v70_numeric_screen"]
    globals_ = inspector.__globals__
    preregistration = tmp_path / "preregistration.json"
    screen = tmp_path / "screen.json"
    preregistration.write_bytes(
        (
            ROOT
            / "reports/gemma4/metrics/v70_low_frequency_moments_preregistration.json"
        ).read_bytes()
    )
    screen_bytes = (
        ROOT / "reports/gemma4/metrics/v70_low_frequency_moments_numeric_screen.json"
    ).read_bytes()
    screen.write_bytes(screen_bytes)
    checkpoint = tmp_path / "checkpoint"
    for name, value in {
        "V70_PREREGISTRATION": preregistration,
        "V70_NUMERIC_SCREEN": screen,
        "V70_CHECKPOINT": checkpoint,
        "V70_NUMERIC_SCREEN_SHA256": _sha256(screen),
    }.items():
        monkeypatch.setitem(globals_, name, value)

    authenticated = inspector()
    assert authenticated["measurement_authenticated"] is True
    assert all(authenticated["authentication_checks"].values())

    tampered = json.loads(screen.read_text(encoding="utf-8"))
    tampered["result"]["folds"][0]["numeric_screen"]["supported_class_exact"] += 1
    _write_json(screen, tampered)
    monkeypatch.setitem(globals_, "V70_NUMERIC_SCREEN_SHA256", _sha256(screen))
    rejected = inspector()
    assert rejected["measurement_authenticated"] is False
    assert "integer aggregates differ from fold evidence" in rejected[
        "measurement_evidence_error"
    ]

    screen.write_bytes(screen_bytes)
    monkeypatch.setitem(globals_, "V70_NUMERIC_SCREEN_SHA256", _sha256(screen))
    checkpoint.mkdir()
    checkpoint_rejected = inspector()
    assert checkpoint_rejected["measurement_authenticated"] is False
    assert "checkpoint exists" in checkpoint_rejected["measurement_evidence_error"]


def test_current_summary_reports_only_source_level_extended_qa_capability() -> None:
    summary = _summary()
    capability = summary["offline_qa_generator"]

    assert capability["status"] == "implementation_and_synthetic_test_contract_present"
    assert capability["scope"] == "offline_dataset_generation_only"
    assert capability["families"] == [
        "object_location",
        "containment",
        "viewpoint_relative",
        "metric",
        "uncertainty",
    ]
    assert capability["exact_visibility_evidence"] is True
    assert capability["grounding_coordinates_and_references"] is True
    assert capability["unseen_objects_removed_from_answerable_rows"] is True
    assert capability["existing_datasets_regenerated_by_builder"] is False
    assert capability["model_accuracy_on_new_families_measured"] is False
    assert all(capability["checks"].values())
    assert capability["source_path"] in summary["source_artifacts"]
    assert capability["synthetic_test_path"] in summary["source_artifacts"]

    markdown = BUILDER["render_markdown"](summary)
    normalized_markdown = " ".join(markdown.split())
    assert "offline generator now has source and synthetic-test coverage" in markdown
    assert "builder regenerated no dataset" in markdown
    assert "model accuracy on these new families remains unmeasured" in normalized_markdown


def test_current_summary_reports_runnable_strict_fixed_prefix_path() -> None:
    summary = _summary()
    strict = summary["strict_fixed_prefix_runtime"]

    assert strict["status"] == ("live_chat_and_oracle_deletion_passed_below_acceptance_behavior")
    assert strict["strict_fixed_environment_embedding_input"] is True
    assert strict["question_conditioned_scene_readout_tokens"] is False
    assert strict["question_dependent_scene_retrieval"] is False
    assert strict["behaviorally_promoted"] is False
    assert strict["acceptance_gate_passed"] is False
    assert all(strict["checks"].values())
    assert all(strict["live_checks"].values())
    assert strict["live_chat_passed"] is True
    assert strict["oracle_deletion_passed"] is True
    assert strict["live_question_count"] == 3
    assert len(strict["environment_conditioned_input_sha256"]) == 64
    assert strict["source_path"] in summary["source_artifacts"]
    assert strict["launcher_path"] in summary["source_artifacts"]
    assert strict["synthetic_test_path"] in summary["source_artifacts"]
    assert strict["sample_chat_path"] in summary["source_artifacts"]
    assert len(strict["sample_chat"]) == 3
    assert strict["sample_chat_is_demo_not_accuracy_evidence"] is True

    markdown = BUILDER["render_markdown"](summary)
    assert "strict fixed-prefix CLI is runnable" in markdown
    assert "proof of mechanism" in markdown
    assert "make strict-demo-leakage" in markdown
    assert "demo transcript, not representative held-out accuracy" in markdown
    assert "strict_prefix_chat.jsonl" in markdown


def test_strict_demo_preflight_exposes_human_visualization_paths() -> None:
    result = _strict_demo_check()

    assert result.returncode == 0, result.stderr
    assert "Strict fixed-prefix demo preflight: PASS" in result.stdout
    assert "RGB map preview:" in result.stdout
    assert "point cloud:" in result.stdout
    assert "macOS viewer command: open" in result.stdout


def test_summary_tracks_strict_atlas_and_current_learned_llm_tool_policy() -> None:
    summary = _summary()
    atlas = summary["strict_fixed_prefix_atlas"]
    policy = summary["llm_tool_policy"]

    assert atlas["status"] == (
        "authenticated_v75_structural_mechanism_behavior_negative_not_promoted"
    )
    assert atlas["evidence_authenticated"] is True
    assert atlas["controller_architecture"] == "dense_full_scene_continuous_control_v75"
    assert atlas["controller_weights_sha256"] == (
        "bb112f42ca5df71b88b4cd7721b9107f9be9b0dc01b612a4ace6212548da669c"
    )
    assert atlas["fixed_prefix_token_count"] == 738
    assert atlas["atlas_memory_tokens"] == 480
    assert atlas["all_base_latents_preserved"] is True
    assert atlas["all_atlas_tokens_preserved"] is True
    assert atlas["every_probe_processed"] is True
    assert atlas["strict_fixed_environment_embedding_input"] is True
    assert atlas["question_conditioned_scene_readout_tokens"] is False
    assert atlas["behavioral_accuracy_measured"] is True
    assert atlas["runtime_promotion_authorized"] is False
    assert all(atlas["checks"].values())
    for path, digest in BUILDER["V75_FIXED_ATLAS_MECHANISM_EVIDENCE_SHA256"].items():
        assert summary["source_artifacts"][path.as_posix()] == digest
    for path, digest in BUILDER["V75_FIXED_ATLAS_BEHAVIOR_EVIDENCE_SHA256"].items():
        assert summary["source_artifacts"][path.as_posix()] == digest
    behavior = atlas["behavioral_result"]
    assert behavior["status"] == "authenticated_historical_internal_negative_not_promoted"
    assert behavior["fixed_v75_atlas"]["correct"] == 6
    assert behavior["frozen_v54"]["correct"] == 6
    assert behavior["direct_exact_v75"]["correct"] == 9
    assert behavior["prediction_change_units"] == {
        "fixed_v75_atlas": 1,
        "frozen_v54": 1,
        "direct_exact_v75": 2,
        "total": 8,
    }
    assert behavior["pair_disjoint"] is True
    assert behavior["scene_disjoint"] is True
    assert behavior["question_disjoint"] is False
    assert behavior["question_overlap_with_training"] == 12
    assert behavior["predictor_loaded_file_count"] == 119
    assert behavior["predictor_forbidden_access_count"] == 0
    assert behavior["runtime_promotion_authorized"] is False
    assert all(behavior["checks"].values())
    assert policy["status"] == "authenticated_historical_v3_partial_v4_1_rejected"
    assert policy["tool_count"] == 9
    assert policy["maximum_retries"] == 2
    assert policy["navigation_benchmark_structure_status"] == "structurally_verified"
    assert all(policy["navigation_structural_checks"].values())
    assert policy["navigation_measurement_status"] == "authenticated_complete"
    assert policy["live_navigation_accuracy_measured"] is True
    assert policy["claimed_trained_navigation_policy"] is True
    assert policy["complete_navigation_success_claimed"] is False
    assert policy["navigation_metrics"]["task_count"] == 6
    assert policy["navigation_metrics"]["success_count"] == 5
    assert policy["navigation_benchmark_passed"] is False
    assert all(policy["checks"].values())

    markdown = BUILDER["render_markdown"](summary)
    collapsed_markdown = " ".join(markdown.split())
    assert "738-token scene-only input" in markdown
    assert "exact sealed V75 controller" in collapsed_markdown
    assert "removes the stale dependency on rejected V66b" in collapsed_markdown
    assert "supplied no behavioral gain" in collapsed_markdown
    assert policy["status"] in markdown
    assert "make gemma4-embodied-chat-llm" in markdown


def test_current_summary_authenticates_navigation_history_and_v41_rejection() -> None:
    summary = _summary()
    learned = summary["learned_navigation_policy"]
    policy = summary["llm_tool_policy"]

    assert learned["status"] == "authenticated_historical_v3_partial_v4_1_rejected"
    assert learned["measurement_authenticated"] is True
    assert learned["claimed_trained_navigation_policy"] is True
    assert learned["complete_success_claimed"] is False
    assert learned["current_version"] == "v3_historical"

    v1 = learned["history"]["v1"]
    v2 = learned["history"]["v2"]
    assert v1["measurement_authenticated"] is True
    assert v1["metrics"]["success_count"] == 3
    assert v1["metrics"]["task_count"] == 6
    assert v1["benchmark_passed"] is False
    assert v2["measurement_authenticated"] is True
    assert v2["metrics"] == {
        "action_failure_count": 0,
        "collision_count": 0,
        "executed_action_count": 24,
        "policy_rejection_count": 0,
        "success_count": 4,
        "success_rate": 4 / 6,
        "task_count": 6,
    }
    assert v2["benchmark_passed"] is False
    assert v2["weak_scene_controls"] is True
    assert v2["causal_control_accuracy_deltas"] == {
        "wrong_scene_prefix": 0.0,
        "zero_robot_tokens": pytest.approx(0.4691357910633087),
        "zero_scene_prefix": pytest.approx(0.007495582103729248),
    }
    assert v2["offline_validation"] == {
        "action_accuracy": pytest.approx(0.8298059701919556),
        "argument_mae": pytest.approx(0.3123275339603424),
        "stop_recall": pytest.approx(0.8421052694320679),
        "turn_sign_accuracy": pytest.approx(0.7581018805503845),
    }
    feasibility = v2["benchmark_feasibility"]
    assert feasibility["benchmark_version"] == 2
    assert feasibility["preregistered_numeric_start"] is True
    assert feasibility["expected_initial_position_xy_m"] == [0.0, -0.5]
    assert feasibility["criteria_changed_from_v1"] is False
    assert feasibility["all_progress_criteria_feasible"] is True
    assert all(row["feasible"] for row in feasibility["progress_tasks"])

    v3 = learned["current"]
    assert v3["measurement_authenticated"] is True
    assert v3["metrics"]["success_count"] == 5
    assert v3["metrics"]["executed_action_count"] == 23
    assert v3["metrics"]["collision_count"] == 0
    assert v3["benchmark_passed"] is False
    assert v3["weak_scene_controls"] is True
    assert v3["causal_control_accuracy_deltas"]["zero_target_state"] == pytest.approx(
        0.6481481492519379
    )
    assert v3["target_control_turn_sign_deltas"]["wrong_target_state"] == pytest.approx(
        0.09837961196899414
    )
    assert v3["evidence_scope"] == "historical_sealed_run"
    assert v3["current_runtime_compatibility_claimed"] is False

    v41 = learned["successor_v4_1"]
    assert v41["measurement_authenticated"] is True
    assert v41["status"] == (
        "historical_evidence_authenticated_current_runtime_compatibility_not_claimed"
    )
    assert v41["passed_gate_count"] == 13
    assert v41["gate_count"] == 14
    assert v41["failed_gates"] == ["shuffled_clearance_family_drop"]
    assert v41["shuffled_clearance_family_drop"] == pytest.approx(
        0.049565017223358154
    )
    assert v41["required_shuffled_clearance_family_drop"] == pytest.approx(0.1)
    assert v41["checkpoint_absent"] is True
    assert v41["live_benchmark_executed"] is False
    assert v41["promotion_eligible"] is False
    assert v41["historical_source_inventory_authenticated"] is True
    assert v41["current_runtime_compatibility_claimed"] is False
    assert isinstance(v41["current_source_drift_paths"], list)

    untrained = policy["untrained_navigation_control"]
    assert untrained["navigation_measurement_status"] == "authenticated_complete"
    assert untrained["claimed_trained_navigation_policy"] is False
    assert untrained["navigation_metrics"]["success_count"] == 0
    assert untrained["navigation_metrics"]["task_count"] == 6

    for checks_name in (
        "run_contract_checks",
        "score_checks",
        "inference_audit_checks",
        "checkpoint_metadata_checks",
        "runtime_audit_checks",
    ):
        assert all(v2[checks_name].values())
    for path, digest in v2["implementation_source_sha256"].items():
        assert summary["source_artifacts"][path] == digest
    for path in v2["evidence_paths"]:
        assert path in summary["source_artifacts"]
    assert not any("/oracle/" in path for path in v2["evidence_paths"])
    for key, path in v3["implementation_source_paths"].items():
        assert summary["source_artifacts"][path] == v3["implementation_source_sha256"][key]
    for path in v3["evidence_paths"]:
        assert path in summary["source_artifacts"]
    assert not any("/oracle/" in path for path in v3["evidence_paths"])

    markdown = BUILDER["render_markdown"](summary)
    assert "historical hash-authenticated supervised V3 controller passed 5/6" in markdown
    assert "V2 completed 4/6" in markdown
    assert "V1 completed 3/6" in markdown
    assert "untrained seam completed 0/6" in markdown
    assert "wrong-scene action-accuracy delta 0.003527" in markdown
    assert "zero-scene delta -0.000441" in markdown
    assert "not complete multi-scene navigation success" in markdown
    assert "single preregistered V4.1 successor passed 13/14" in markdown
    assert "No V4.1 checkpoint was published and no live V4.1 benchmark ran" in markdown


def test_summary_authenticates_terminal_ple_and_tool_decoder_negatives() -> None:
    summary = _summary()
    ple = summary["v54_fixed_prefix_ple_reader_v1_v5"]
    tool = summary["gemma4_tool_decoder_v2_2"]

    assert ple["status"] == "authenticated_terminal_negative_no_checkpoint"
    assert ple["evidence_authenticated"] is True
    assert ple["checkpoint_published"] is False
    assert ple["versions"]["v5"]["updates"] == 80
    assert ple["versions"]["v5"]["positive_wrong_prefix_sides_after"] == 28
    assert ple["versions"]["v5"]["complete_changed_units_after"] == 9
    assert ple["versions"]["v5"]["greedy_executed"] is False

    assert tool["status"] == "authenticated_terminal_negative_no_runtime_checkpoint"
    assert tool["evidence_authenticated"] is True
    assert tool["optimizer_updates"] == 64
    assert tool["training_loss_first"] == pytest.approx(2.414295881986618)
    assert tool["training_loss_final"] == pytest.approx(0.2341814790852368)
    assert tool["heldout_answer_token_nll"] == pytest.approx(0.37775762747489017)
    assert tool["heldout_answer_token_accuracy"] == pytest.approx(0.8712881694434225)
    assert tool["heldout_exact_sequence_accuracy"] == pytest.approx(0.17416225749559083)
    assert tool["heldout_valid_schema_rate"] == pytest.approx(0.2641093474426808)
    assert tool["heldout_tool_accuracy"] == pytest.approx(0.24118165784832452)
    assert tool["greedy_generation_executed"] is False
    assert tool["runtime_checkpoint_published"] is False
    assert tool["runtime_checkpoint_absent"] is True

    for evidence in (ple, tool):
        assert all(evidence["checks"].values())
        for path, digest in evidence["evidence_sha256"].items():
            assert summary["source_artifacts"][path] == digest

    markdown = BUILDER["render_markdown"](summary)
    assert "V54 PLE-reader V1--V5 chain" in markdown
    assert "28/52 positive wrong-prefix sides" in markdown
    assert "9/26 complete changed units" in markdown
    assert "Gemma tool-decoder V2.2" in markdown
    assert "17.42% exact sequences" in markdown
    assert "published no runtime checkpoint" in markdown


def test_summary_authenticates_v6_smoke_failure_before_gradients() -> None:
    summary = _summary()
    evidence = summary["fixed_prefix_decoder_reader_v6"]

    assert evidence["status"] == (
        "authenticated_terminal_smoke_failure_no_training_no_checkpoint"
    )
    assert evidence["evidence_authenticated"] is True
    assert evidence["passed"] is False
    assert evidence["promotion_eligible"] is False
    assert evidence["failure_stage"] == (
        "byte_exact_full_vs_tail_answer_logit_equivalence"
    )
    assert evidence["gradient_computation_executed"] is False
    assert evidence["optimizer_constructed"] is False
    assert evidence["optimizer_steps"] == 0
    assert evidence["training_executed"] is False
    assert evidence["checkpoint_published"] is False
    assert evidence["checkpoint_absent"] is True
    assert evidence["forbidden_file_read_count"] == 0
    assert evidence["deferred_or_final_qa_accessed"] is False
    assert evidence["single_smoke_attempt_consumed"] is True
    assert evidence["loaded_file_count"] == 233
    assert all(evidence["checks"].values())
    for path, digest in evidence["evidence_sha256"].items():
        assert summary["source_artifacts"][path] == digest

    markdown = BUILDER["render_markdown"](summary)
    assert "fixed-prefix upper-decoder\nreader V6" in markdown
    assert "byte-exact full-sequence-versus-answer-tail selected-logit" in markdown
    assert "computed no V6 gradient, constructed no\noptimizer" in markdown
    assert "zero forbidden reads" in markdown
    assert "planned 576\ntraining rows, 384 validation rows,\nand 96 updates were never executed" in markdown


def test_learned_navigation_v2_authentication_fails_closed_on_path_or_hash_tamper(
    tmp_path: Path,
) -> None:
    inspector = BUILDER["_inspect_learned_navigation_run"]
    artifacts = dict(BUILDER["LEARNED_NAVIGATION_V2"])
    pins = dict(BUILDER["LEARNED_NAVIGATION_V2_SHA256"])

    tampered_score = tmp_path / "score.json"
    original_score = ROOT / artifacts["score"]
    tampered_score.write_bytes(original_score.read_bytes() + b" ")
    score_artifacts = {**artifacts, "score": tampered_score}
    rejected_hash = inspector("v2", score_artifacts, pins)
    assert rejected_hash["measurement_authenticated"] is False
    assert rejected_hash["claimed_trained_navigation_policy"] is False
    assert "digest differs: score" in rejected_hash["evidence_error"]

    checkpoint_link = tmp_path / "checkpoint-link"
    checkpoint_link.symlink_to(ROOT / artifacts["checkpoint"], target_is_directory=True)
    path_artifacts = {**artifacts, "checkpoint": checkpoint_link}
    rejected_path = inspector("v2", path_artifacts, pins)
    assert rejected_path["measurement_authenticated"] is False
    assert rejected_path["claimed_trained_navigation_policy"] is False
    assert "exact two-file tree" in rejected_path["evidence_error"]


def test_summary_tracks_authenticated_causal_control_pipeline() -> None:
    summary = _summary()
    controls = summary["causal_control_suite"]

    assert controls["implementation_status"] == ("implementation_and_synthetic_verification_passed")
    assert controls["conditions"] == [
        "primary",
        "empty_scene_prefix",
        "wrong_scene_prefix",
        "semantic_shuffle",
        "position_shuffle",
        "geometry_only",
        "semantics_without_xyz",
        "remove_rgb",
        "remove_normals",
    ]
    assert controls["question_dependent_retrieval"] is False
    assert controls["complete_map_rows_retained"] is True
    assert controls["current_gemma_measurement_status"] in {
        "not_measured",
        "authenticated_complete",
    }
    assert controls["measurement_authenticated"] is (
        controls["current_gemma_measurement_status"] == "authenticated_complete"
    )
    if controls["measurement_authenticated"]:
        assert controls["status"] == (
            "authenticated_nine_condition_development_measurement_complete"
        )
        assert controls["measurement_question_count"] == 216
        assert controls["measurement_results"]["primary"][
            "normalized_exact_accuracy"
        ] == pytest.approx(89 / 216)
        assert (
            controls["measurement_results"]["remove_normals"]["normalized_exact_accuracy"]
            == controls["measurement_results"]["primary"]["normalized_exact_accuracy"]
        )
        assert controls["measurement_results"]["semantic_shuffle"][
            "normalized_exact_accuracy"
        ] == pytest.approx(26 / 216)
        assert controls["measurement_results"]["position_shuffle"][
            "normalized_exact_accuracy"
        ] == pytest.approx(29 / 216)
        assert controls["measurement_results"]["semantics_without_xyz"][
            "normalized_exact_accuracy"
        ] == pytest.approx(22 / 216)
        assert len(controls["measurement_evidence_paths"]) == 20
        assert (
            "reports/gemma4/metrics/controls/v55_development_summary.json"
            in summary["source_artifacts"]
        )
    assert all(controls["checks"].values())

    markdown = BUILDER["render_markdown"](summary)
    assert "all nine prediction files" in markdown
    assert "normal channel is unpopulated" in " ".join(markdown.split())
    for condition in controls["conditions"]:
        assert f"| {condition} |" in markdown


def test_control_measurement_authenticates_all_nine_files_and_fails_on_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conditions = tuple(BUILDER["CONTROL_CONDITIONS"])
    prediction_root = tmp_path / "predictions"
    metrics_root = tmp_path / "metrics"
    manifest_path = prediction_root / "manifest.json"
    summary_path = tmp_path / "summary.json"
    references = tmp_path / "references.jsonl"
    manifest_rows: dict[str, Any] = {}
    result_rows: dict[str, Any] = {}
    for index, condition in enumerate(conditions):
        prediction = prediction_root / f"{condition}.jsonl"
        prediction.parent.mkdir(parents=True, exist_ok=True)
        prediction.write_text(
            json.dumps(
                {
                    "scene_id": "scene_000019",
                    "question_id": "q_000001",
                    "predicted_answer": "yes",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        prediction_sha256 = _sha256(prediction)
        accuracy = 0.5 - index / 100.0
        metric = metrics_root / f"{condition}.json"
        _write_json(
            metric,
            {
                "normalized_exact_accuracy": accuracy,
                "spatial_relation_accuracy": 0.4,
                "count": {"accuracy": 0.3},
                "grounding": {"mean_coordinate_error_m": 1.5},
                "counterfactual": {"changed_when_expected_rate": 0.2},
                "predictions_sha256": prediction_sha256,
            },
        )
        manifest_rows[condition] = {
            "path": str(prediction),
            "sha256": prediction_sha256,
            "prediction_count": 1,
        }
        result_rows[condition] = {
            "normalized_exact_accuracy": accuracy,
            "spatial_relation_accuracy": 0.4,
            "count_accuracy": 0.3,
            "grounding_mean_error_m": 1.5,
            "counterfactual_changed_rate": 0.2,
            "prediction_count": 1,
            "prediction_sha256": prediction_sha256,
            "metrics_path": str(metric),
            "metrics_sha256": _sha256(metric),
            "exact_accuracy_delta_vs_primary": accuracy - 0.5,
        }
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "question_dependent_retrieval": False,
            "one_prefix_per_scene_condition": True,
            "conditions": manifest_rows,
            "questions": {"question_count": 1},
        },
    )
    _write_json(
        summary_path,
        {
            "schema_version": 1,
            "artifact": "continuous_scene_control_scores",
            "control_manifest_path": str(manifest_path),
            "control_manifest_sha256": _sha256(manifest_path),
            "references_path": str(references),
            "references_sha256": "a" * 64,
            "complete_prediction_coverage_required": True,
            "question_dependent_retrieval": False,
            "one_prefix_per_scene_condition": True,
            "results": result_rows,
        },
    )
    replacements = {
        "CONTROL_SUMMARY": summary_path,
        "CONTROL_MANIFEST": manifest_path,
        "CONTROL_PREDICTIONS_DIRECTORY": prediction_root,
        "CONTROL_METRICS_DIRECTORY": metrics_root,
        "CONTROL_REFERENCES": references,
    }
    inspector_globals = BUILDER["_inspect_control_measurement"].__globals__
    for name, value in replacements.items():
        monkeypatch.setitem(inspector_globals, name, value)

    measurement = BUILDER["_inspect_control_measurement"]()
    assert measurement["measurement_authenticated"] is True
    assert measurement["current_gemma_measurement_status"] == "authenticated_complete"
    assert list(measurement["measurement_results"]) == list(conditions)
    assert len(measurement["measurement_evidence_paths"]) == 20

    primary = prediction_root / "primary.jsonl"
    primary.write_text(primary.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    rejected = BUILDER["_inspect_control_measurement"]()
    assert rejected["measurement_authenticated"] is False
    assert rejected["current_gemma_measurement_status"] == (
        "artifact_present_authentication_failed"
    )
    assert "digest differs" in rejected["measurement_evidence_error"]


def test_llm_navigation_measurement_reports_authenticated_failure_without_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    families = tuple(BUILDER["LLM_NAVIGATION_FAMILIES"])
    tasks_path = tmp_path / "tasks.json"
    journal_path = tmp_path / "journal.json"
    audit_path = tmp_path / "audit.json"
    score_path = tmp_path / "score.json"
    tasks = {
        "schema": "semantic_3d_chat.llm_navigation_tasks.v1",
        "scene_id": "scene_000001",
        "seed": 7,
        "tasks": [
            {
                "task_id": f"nav_{index:03d}",
                "family": family,
                "instruction": f"Perform development task {index}.",
                "max_steps": 4,
            }
            for index, family in enumerate(families)
        ],
    }
    _write_json(tasks_path, tasks)
    _write_json(audit_path, {"passed": True, "forbidden_accesses": []})
    episodes = []
    score_tasks = []
    for index, family in enumerate(families):
        body = {
            "task_id": f"nav_{index:03d}",
            "family": family,
            "steps": [],
            "task_success_scored_during_inference": False,
            "oracle_or_labels_available": False,
            "environmental_text_inputs": [],
        }
        episode_sha256 = BUILDER["_canonical_sha256"](body)
        episodes.append({**body, "episode_sha256": episode_sha256})
        score_tasks.append(
            {
                "task_id": f"nav_{index:03d}",
                "family": family,
                "passed": index < 2,
                "metrics": {"collision_count": 0},
                "episode_sha256": episode_sha256,
            }
        )
    journal_body = {
        "schema": "semantic_3d_chat.llm_navigation_journal.v1",
        "status": "complete",
        "header": {
            "scene_id": "scene_000001",
            "task_count": len(families),
            "task_manifest_sha256": BUILDER["_canonical_sha256"](tasks),
            "local_inference": True,
            "environment_input": "continuous_scene_and_numeric_robot_prefix",
            "feedback_input": "bounded_numeric_tool_receipts",
            "oracle_or_labels_available": False,
            "claimed_learned_navigation_success": False,
            "run_contract": {
                "model_id": "google/gemma-4-E2B-it",
                "strict_fixed_environment_embedding_input": True,
                "question_conditioned_scene_readout_tokens": False,
                "fallback_policy": "fail_closed",
            },
        },
        "episodes": episodes,
        "runtime_file_audit": {
            "passed": True,
            "blocking_enabled": True,
            "forbidden_accesses": [],
            "audit_report_sha256": _sha256(audit_path),
        },
    }
    journal = {
        **journal_body,
        "journal_sha256": BUILDER["_canonical_sha256"](journal_body),
    }
    _write_json(journal_path, journal)
    by_family = {
        family: {
            "task_count": 1,
            "success_count": int(index < 2),
            "success_rate": float(index < 2),
        }
        for index, family in enumerate(families)
    }
    _write_json(
        score_path,
        {
            "schema": "semantic_3d_chat.llm_navigation_score.v1",
            "scene_id": "scene_000001",
            "passed": False,
            "policy_status": "untrained_tool_selection_seam",
            "claimed_trained_navigation_policy": False,
            "separation": {
                "inference_journal_validated_before_oracle_open": True,
                "inference_received_oracle_or_labels": False,
                "oracle_used_only_by_post_inference_scorer": True,
                "prediction_journal_sha256": journal["journal_sha256"],
                "scoring_spec_sha256": "b" * 64,
                "scene_oracle_sha256": "c" * 64,
            },
            "metrics": {
                "task_count": 6,
                "success_count": 2,
                "success_rate": 2 / 6,
                "collision_count": 0,
                "action_failure_count": 0,
                "policy_rejection_count": 4,
                "executed_action_count": 2,
            },
            "by_family": by_family,
            "tasks": score_tasks,
        },
    )
    replacements = {
        "LLM_NAVIGATION_TASKS": tasks_path,
        "LLM_NAVIGATION_JOURNAL": journal_path,
        "LLM_NAVIGATION_AUDIT": audit_path,
        "LLM_NAVIGATION_SCORE": score_path,
    }
    inspector_globals = BUILDER["_inspect_llm_navigation_measurement"].__globals__
    for name, value in replacements.items():
        monkeypatch.setitem(inspector_globals, name, value)

    measurement = BUILDER["_inspect_llm_navigation_measurement"]()
    assert measurement["navigation_measurement_status"] == "authenticated_complete"
    assert measurement["live_navigation_accuracy_measured"] is True
    assert measurement["navigation_benchmark_passed"] is False
    assert measurement["claimed_trained_navigation_policy"] is False
    assert measurement["navigation_metrics"]["success_count"] == 2
    assert all(measurement["navigation_audit_checks"].values())

    audit_path.write_text("tampered\n", encoding="utf-8")
    rejected = BUILDER["_inspect_llm_navigation_measurement"]()
    assert rejected["live_navigation_accuracy_measured"] is False
    assert rejected["navigation_measurement_status"] == ("artifact_present_authentication_failed")


def test_summary_tracks_strict_loopback_web_ui() -> None:
    summary = _summary()
    web = summary["strict_web"]

    assert web["status"] == "preflight_and_synthetic_request_invariance_passed"
    assert web["host"] == "127.0.0.1"
    assert web["port"] == 8766
    assert web["strict_fixed_environment_embedding_input"] is True
    assert web["human_visuals_are_model_inputs"] is False
    assert web["forbidden_access_count"] == 0
    assert web["live_server_started_by_builder"] is False
    assert all(web["checks"].values())


def test_summary_tracks_prohibited_oracle_text_upper_bound_without_promotion() -> None:
    summary = _summary()
    oracle_text = summary["oracle_text_upper_bound"]

    assert oracle_text["role"] == "prohibited_evaluation_only_upper_bound"
    assert oracle_text["primary_path_eligible"] is False
    assert oracle_text["prohibited_primary_input"] is True
    assert oracle_text["question_independent_scene_text"] is True
    assert oracle_text["answer_bearing_references_available_only_to_scorer"] is True
    assert oracle_text["prepared_scene_text_present"] is True
    assert len(oracle_text["prepared_scene_text_sha256"]) == 64
    assert all(oracle_text["checks"].values())
    if oracle_text["measurement_status"] == "authenticated_complete":
        assert oracle_text["measurement_authenticated"] is True
        assert oracle_text["question_count"] == 216
        assert oracle_text["scene_count"] == 6
        assert oracle_text["local_gemma_inference_authenticated"] is True
    elif oracle_text["measurement_status"] == "artifact_present_stale_implementation":
        assert oracle_text["measurement_authenticated"] is False
        assert oracle_text["local_gemma_inference_authenticated"] is False
        assert oracle_text["artifact_chain_authenticated_except_implementation_source"] is True
        assert oracle_text["historical_artifacts_preserved_unmodified"] is True
        assert oracle_text["historical_metric_claim_permitted"] is False
        assert oracle_text["requires_local_gemma_rerun"] is True
        assert oracle_text["question_count"] == 216
        assert oracle_text["scene_count"] == 6
        assert oracle_text["implementation_hash_drift"]
        for hashes in oracle_text["implementation_hash_drift"].values():
            assert len(hashes["recorded_sha256"]) == 64
            assert len(hashes["current_sha256"]) == 64
            assert hashes["recorded_sha256"] != hashes["current_sha256"]
        assert "metrics" not in oracle_text
    else:
        assert oracle_text["measurement_status"] == "not_measured"
        assert oracle_text["measurement_authenticated"] is False
    for path in (
        "src/semantic_3d_chat/evaluation/oracle_text_artifacts.py",
        "src/semantic_3d_chat/evaluation/oracle_text_prepare.py",
        "src/semantic_3d_chat/evaluation/oracle_text_predict.py",
        "src/semantic_3d_chat/evaluation/oracle_text_score.py",
        "tests/test_oracle_text_upper_bound.py",
        "configs/experiments/gemma4_oracle_text_v55.yaml",
        "scripts/run_oracle_text_upper_bound_v55.sh",
    ):
        assert path in summary["source_artifacts"]

    markdown = BUILDER["render_markdown"](summary)
    assert any(
        marker in markdown
        for marker in (
            "categorically prohibited",
            "remains unmeasured",
            "inference implementation evidence is stale",
        )
    )


def test_oracle_text_measurement_classifies_live_source_drift_as_stale() -> None:
    result = BUILDER["_inspect_oracle_text_measurement"]()

    assert result["measurement_status"] == "artifact_present_stale_implementation"
    assert result["measurement_authenticated"] is False
    assert result["artifact_chain_authenticated_except_implementation_source"] is True
    assert result["historical_artifacts_preserved_unmodified"] is True
    assert result["historical_metric_claim_permitted"] is False
    assert result["requires_local_gemma_rerun"] is True
    assert set(result["implementation_hash_drift"]) == {"language/local_lm.py"}
    assert "metrics" not in result


def test_summary_authenticates_complete_direct_multiview_baseline() -> None:
    summary = _summary()
    direct = summary["direct_multiview_baseline"]

    assert direct["status"] == "authenticated_complete_evaluation_only_baseline"
    assert direct["measurement_authenticated"] is True
    assert direct["primary_path_eligible"] is False
    assert direct["prohibited_primary_substitute"] is True
    assert direct["question_independent_scene_cache"] is True
    assert direct["scene_count"] == 6
    assert direct["question_count"] == 216
    assert direct["exact_count"] == 100
    assert direct["normalized_exact_accuracy"] == pytest.approx(100 / 216)
    assert direct["complete_views_per_scene"] == 24
    assert direct["scene_cache_build_count"] == 6
    assert len(set(direct["scene_cache_sha256_by_scene"].values())) == 6
    assert direct["count_accuracy"] == 0.0
    assert direct["spatial_relation_accuracy"] == pytest.approx(22 / 48)
    assert direct["presence_f1"] == pytest.approx(44 / 46)
    for path in (
        "reports/gemma4/metrics/direct_multiview_diverse20_validation.json",
        "reports/gemma4/predictions/direct_multiview_diverse20_validation.jsonl",
        "src/semantic_3d_chat/evaluation/direct_multiview_baseline.py",
        "tests/test_direct_multiview_cache.py",
        "configs/experiments/gemma4_diverse20_direct_multiview_baseline.yaml",
    ):
        assert path in summary["source_artifacts"]

    markdown = BUILDER["render_markdown"](summary)
    assert "100/216 exact" in markdown
    assert "all\n24 complete RGB views" in markdown
    assert "only two questions" not in markdown


def test_direct_multiview_evidence_fails_closed_on_prediction_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inspector = BUILDER["_inspect_direct_multiview_baseline"]
    inspector_globals = inspector.__globals__
    source_predictions = ROOT / BUILDER["DIRECT_IMAGE_PREDICTIONS"]
    source_metrics = ROOT / BUILDER["DIRECT_IMAGE_METRICS"]
    predictions = tmp_path / "predictions.jsonl"
    metrics = tmp_path / "metrics.json"
    predictions.write_bytes(source_predictions.read_bytes())
    metric_document = json.loads(source_metrics.read_text(encoding="utf-8"))
    metric_document["predictions_path"] = str(predictions)
    _write_json(metrics, metric_document)
    replacements = {
        "DIRECT_IMAGE_PREDICTIONS": predictions,
        "DIRECT_IMAGE_METRICS": metrics,
        "DIRECT_IMAGE_PREDICTIONS_SHA256": _sha256(predictions),
        "DIRECT_IMAGE_METRICS_SHA256": _sha256(metrics),
    }
    for name, value in replacements.items():
        monkeypatch.setitem(inspector_globals, name, value)

    authenticated = inspector()
    assert authenticated["measurement_authenticated"] is True

    predictions.write_bytes(predictions.read_bytes() + b"\n")
    rejected = inspector()
    assert rejected["status"] == "artifact_present_authentication_failed"
    assert rejected["measurement_authenticated"] is False
    assert "prediction artifact digest differs" in rejected["measurement_evidence_error"]


def test_oracle_text_measurement_fails_closed_on_malformed_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score = tmp_path / "oracle_text_score.json"
    _write_json(score, {"schema": "tampered"})
    inspector = BUILDER["_inspect_oracle_text_measurement"]
    monkeypatch.setitem(inspector.__globals__, "ORACLE_TEXT_SCORE", score)

    result = inspector()
    assert result["measurement_status"] == "artifact_present_authentication_failed"
    assert result["measurement_authenticated"] is False
    assert "ValueError" in result["measurement_evidence_error"]
    assert "score flags or schema differ" in result["measurement_evidence_error"]


def test_research_demo_fails_closed_without_schema7_checkpoint(tmp_path: Path) -> None:
    result = _demo_check(tmp_path / "missing")

    assert result.returncode == 2
    assert "sealed V66b schema-7 checkpoint has not been published" in result.stderr
    assert "Research demo preflight: PASS" not in result.stdout


def test_research_demo_check_accepts_exact_sealed_schema7_structure(
    tmp_path: Path,
) -> None:
    result = _demo_check(_fake_sealed_schema7(tmp_path))

    assert result.returncode == 0, result.stderr
    assert "Research demo preflight: PASS" in result.stdout
    assert "RGB map preview:" in result.stdout
    assert "point cloud:" in result.stdout


def test_research_demo_rejects_extra_checkpoint_files(tmp_path: Path) -> None:
    checkpoint = _fake_sealed_schema7(tmp_path)
    (checkpoint / "training_answers.json").write_text("{}", encoding="utf-8")
    result = _demo_check(checkpoint)

    assert result.returncode == 2
    assert "checkpoint inventory is not exact" in result.stderr


def test_make_and_readme_use_v66b_fail_closed_entrypoint() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    launcher = (ROOT / "scripts/run_research_demo.sh").read_text(encoding="utf-8")

    assert "research-demo-check:" in makefile
    assert "gemma4_v66b_allrow_always_on_control" in makefile
    assert "make research-demo" in readme
    assert "does not fall back" in readme
    assert "run_schema7_question_control_demo.sh" in launcher
    assert "run_full_demo.sh" not in launcher
