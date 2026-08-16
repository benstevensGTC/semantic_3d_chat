from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from semantic_3d_chat.training import train_question_control_v63 as v63


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "baseline_lock": "baseline.json",
        "filtered_train_qa": "train.jsonl",
        "teacher_cache": "teachers",
        "prefix_cache": "prefixes",
        "base_runtime_config": "configs/runtime/gemma4_v54.yaml",
        "base_checkpoint": "base",
        "source_v60_checkpoint": "v60",
        "output_checkpoint": "output",
        "training_report": "report.json",
        "diagnostics_output": None,
        "device": "cpu",
        "seed": 1,
        "basis_rank": 128,
        "moment_count": 8,
        "interaction_dim": 32,
        "trunk_dim": 192,
        "maximum_control_rms": 0.25,
        "initial_control_rms": 0.075,
        "gate_threshold": 0.5,
        "epochs": 1,
        "batch_size": 16,
        "changed_repeats": 1,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "coefficient_weight": 3.0,
        "log_rms_weight": 1.0,
        "reconstruction_weight": 2.0,
        "relative_mse_weight": 0.25,
        "pair_delta_weight": 1.0,
        "route_weight": 0.25,
        "log_every": 1,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _synthetic_inventory() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ordinal = 0
    for spec in v63._train_specs():
        for scene_index, scene_id in enumerate(spec.scene_ids):
            for question_index in range(24):
                ordinal += 1
                rows.append(
                    {
                        "scene_id": scene_id,
                        "question_id": f"q_{ordinal:06d}",
                        "question": f"Opaque paired question {question_index}",
                        "answer": f"answer {question_index}",
                        "counterfactual_pair_id": spec.pair_id,
                        "counterfactual_question_key": (
                            f"cfq_{int(spec.pair_id[-6:]):06x}{question_index:010x}"
                        ),
                        "counterfactual_expected_change": (
                            question_index < spec.changed_unit_count
                        ),
                        "counterfactual_role": (
                            "reference" if scene_index == 0 else "counterfactual"
                        ),
                    }
                )
    return rows


def test_parser_has_one_training_data_boundary_and_no_evaluation_paths() -> None:
    parser = v63._parser()
    destinations = {
        action.dest for action in parser._actions if action.dest != "help"
    }

    assert "filtered_train_qa" in destinations
    assert "baseline_lock" in destinations
    assert "source_v60_checkpoint" in destinations
    assert "diagnostics_output" in destinations
    assert {
        "validation_questions",
        "internal_validation_questions",
        "scorer_references",
        "scorer_sidecar",
        "preregistration",
        "baseline_predictions",
        "source_train_qa",
        "train_qa",
    }.isdisjoint(destinations)


def test_training_inventory_is_exact_pair_disjoint_population() -> None:
    rows = v63._validated_rows(_synthetic_inventory())

    assert len(rows) == 576
    assert len(v63.training_scene_ids()) == 24
    assert {row.pair_id for row in rows} == set(v63.TRAIN_PAIR_IDS)
    assert sum(row.route_label for row in rows) == 80
    assert len(v63._changed_units(rows)) == 40


def test_baseline_lock_is_validated_before_training_file_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}\n", encoding="utf-8")
    filtered = tmp_path / "filtered.jsonl"
    filtered.write_text("not opened first\n", encoding="utf-8")
    observed: list[str] = []

    def validate(path: str | Path) -> dict[str, str]:
        assert Path(path) == baseline
        observed.append("baseline")
        return {"v54_checkpoint_sha256": "a" * 64}

    def stop(_path: str | Path) -> tuple[dict[str, object], ...]:
        observed.append("training")
        raise RuntimeError("ordering proven")

    monkeypatch.setattr(v63, "validate_baseline_lock", validate)
    monkeypatch.setattr(v63, "load_filtered_training_qa", stop)
    args = _args(baseline_lock=baseline, filtered_train_qa=filtered)

    with pytest.raises(RuntimeError, match="ordering proven"):
        v63.build_v63_preflight(args)
    assert observed == ["baseline", "training"]


def test_preregistered_cv_gate_requires_every_pair_and_teacher() -> None:
    passing = {
        "mean_prompt_cosine": 0.95,
        "minimum_prompt_cosine": 0.75,
        "prompt_root_mean_square_error": 0.01,
        "mean_prompt_rms_absolute_error": 0.005,
        "mean_pair_delta_cosine": 0.70,
        "positive_pair_delta_fraction": 0.90,
        "teacher_side_count": 80,
        "changed_pair_unit_count": 40,
    }

    checks = v63.evaluate_cv_checks(passing, fold_mean_cosines=[0.90] * 12)
    assert all(checks.values())

    incomplete = dict(passing, teacher_side_count=79)
    checks = v63.evaluate_cv_checks(incomplete, fold_mean_cosines=[0.90] * 12)
    assert checks["complete_teacher_side_coverage"] is False

    weak_fold = v63.evaluate_cv_checks(passing, fold_mean_cosines=[0.90] * 11 + [0.7])
    assert weak_fold["minimum_fold_mean_prompt_cosine"] is False


def test_failed_cross_validation_writes_no_checkpoint_or_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    report = tmp_path / "report.json"
    preflight = SimpleNamespace(
        output_checkpoint=checkpoint,
        training_report=report,
    )
    monkeypatch.setattr(v63, "build_v63_preflight", lambda _args: preflight)
    monkeypatch.setattr(
        v63,
        "_compute_frozen_question_embeddings",
        lambda *_args, **_kwargs: ({}, {"gemma_backward_used": False}),
    )
    monkeypatch.setattr(
        v63,
        "run_pair_disjoint_cross_validation",
        lambda **_kwargs: {"passed": False, "checks": {"mean_prompt_cosine": False}},
    )
    monkeypatch.setattr(v63, "_failure_provenance", lambda _preflight, _args: {})

    with pytest.raises(
        v63.V63OfflineGateError,
        match="no checkpoint or training report published",
    ):
        v63.train_v63(_args(output_checkpoint=checkpoint, training_report=report))
    assert not checkpoint.exists()
    assert not report.exists()


def test_failure_diagnostics_are_atomic_create_once_and_publish_no_model_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    training_report = tmp_path / "training_report.json"
    diagnostics = tmp_path / "metrics" / "v63_failure.json"
    error = v63.V63OfflineGateError(
        "pair-held-out gate failed",
        {
            "cross_validation": {
                "passed": False,
                "aggregate": {"mean_prompt_cosine": 0.81},
            }
        },
        failure_stage="pair_disjoint_cross_validation",
        provenance={
            "authorization": {
                "baseline_lock_sha256": "a" * 64,
                "preregistration_sha256": "b" * 64,
            },
            "inputs": {"filtered_training_qa_sha256": "c" * 64},
        },
        scope=v63._failure_scope(),
        gemma_audit={
            "gemma_backward_used": False,
            "gemma_generation_used": False,
        },
    )

    def fail(_args: argparse.Namespace) -> dict[str, object]:
        raise error

    monkeypatch.setattr(v63, "train_v63", fail)
    status = v63.main(
        [
            "--baseline-lock",
            str(tmp_path / "baseline.json"),
            "--filtered-train-qa",
            str(tmp_path / "train.jsonl"),
            "--teacher-cache",
            str(tmp_path / "teachers"),
            "--prefix-cache",
            str(tmp_path / "prefixes"),
            "--base-runtime-config",
            str(tmp_path / "runtime.yaml"),
            "--base-checkpoint",
            str(tmp_path / "base"),
            "--source-v60-checkpoint",
            str(tmp_path / "v60"),
            "--output-checkpoint",
            str(checkpoint),
            "--training-report",
            str(training_report),
            "--diagnostics-output",
            str(diagnostics),
        ]
    )

    assert status == 2
    assert not checkpoint.exists()
    assert not training_report.exists()
    payload = json.loads(diagnostics.read_text(encoding="utf-8"))
    assert payload["artifact"] == (
        "v63_pair_disjoint_expanded_value_distillation_failure"
    )
    assert payload["failure_stage"] == "pair_disjoint_cross_validation"
    assert payload["diagnostics"] == error.diagnostics
    assert payload["provenance"] == error.provenance
    assert payload["scope"]["training_only"] is True
    assert payload["scope"]["validation_inputs_used"] is False
    assert payload["scope"]["scorer_inputs_used"] is False
    assert payload["scope"]["oracle_loaded"] is False
    assert payload["publication"] == {
        "checkpoint_published": False,
        "failure_diagnostics_create_once": True,
        "training_report_published": False,
    }
    original = diagnostics.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        v63._publish_failure_report_create_once(
            diagnostics,
            v63._failure_report(error),
        )
    assert diagnostics.read_bytes() == original
    assert not list(diagnostics.parent.glob(".v63_failure.json.*.tmp"))


def test_diagnostics_destination_must_be_disjoint_from_training_outputs(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    args = _args(
        output_checkpoint=checkpoint,
        diagnostics_output=checkpoint / "failure.json",
    )

    with pytest.raises(ValueError, match="overlaps an input/output"):
        v63._validate_diagnostics_destination(args)


def test_output_isolation_rejects_teacher_cache_ancestor(tmp_path: Path) -> None:
    teacher_cache = tmp_path / "teachers"
    teacher_cache.mkdir()

    with pytest.raises(ValueError, match="overlaps an input"):
        v63._validate_output_isolation(
            output_checkpoint=teacher_cache / "new_checkpoint",
            training_report=tmp_path / "report.json",
            inputs=(teacher_cache,),
        )


def test_final_gate_is_independent_of_provisional_route() -> None:
    summary = {
        "mean_prompt_cosine": 0.99,
        "minimum_prompt_cosine": 0.90,
        "prompt_root_mean_square_error": 0.005,
        "mean_prompt_rms_absolute_error": 0.002,
        "mean_pair_delta_cosine": 0.90,
        "teacher_side_count": 80,
        "changed_pair_unit_count": 40,
    }

    checks = v63.evaluate_final_checks(summary)

    assert all(checks.values())
    assert not any("route" in name for name in checks)


def test_every_fresh_fit_copies_and_freezes_source_v60_question_norm() -> None:
    torch.manual_seed(6301)
    basis = torch.linalg.qr(torch.randn(1536, 4)).Q.T.contiguous()
    source = v63.TeacherBasisFullSceneQuestionControlV3(
        1536,
        basis,
        control_tokens=4,
        expected_environment_latents=256,
        moment_count=2,
        interaction_dim=2,
        trunk_dim=8,
        maximum_control_rms=0.25,
        initial_control_rms=0.075,
    )
    with torch.no_grad():
        source.question_norm.weight.uniform_(0.5, 1.5)
        source.question_norm.bias.uniform_(-0.25, 0.25)
    norm_state = v63._question_norm_state(source)
    expected_sha256 = v63._tensor_state_sha256(norm_state)
    rows = (
        v63.V63Row("scene_000011", "q_000001", "Q?", "pair", "unit", True),
        v63.V63Row("scene_000012", "q_000002", "Q?", "pair", "unit", True),
        v63.V63Row("scene_000011", "q_000003", "N?", "pair", "keep", False),
        v63.V63Row("scene_000012", "q_000004", "N?", "pair", "keep", False),
    )
    generator = torch.Generator().manual_seed(6302)
    prefixes = {
        scene: torch.randn(1, 258, 1536, generator=generator) * 0.01
        for scene in ("scene_000011", "scene_000012")
    }
    questions = {
        row.key: torch.randn(1, 1, 1536, generator=generator) for row in rows
    }
    targets = {
        rows[0].key: torch.randn(1, 4, 1536, generator=generator) * 0.05,
        rows[1].key: torch.randn(1, 4, 1536, generator=generator) * 0.05,
    }
    args = _args(
        basis_rank=4,
        moment_count=2,
        interaction_dim=2,
        trunk_dim=8,
        epochs=1,
        batch_size=2,
        changed_repeats=1,
        log_every=2,
    )

    fit = v63._fit_controller(
        rows=rows,
        targets=targets,
        prefixes=prefixes,
        questions=questions,
        source_question_norm_state=norm_state,
        args=args,
        seed=6303,
        log_phase="v63_test",
        fixed_output_basis=basis,
    )

    assert torch.equal(fit.control.output_basis, basis)
    assert fit.question_norm_sha256 == expected_sha256
    assert fit.question_norm_frozen is True
    assert v63._question_norm_state(fit.control).keys() == norm_state.keys()
    assert all(
        torch.equal(v63._question_norm_state(fit.control)[name], value)
        for name, value in norm_state.items()
    )
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in fit.control.question_norm.parameters()
    )
