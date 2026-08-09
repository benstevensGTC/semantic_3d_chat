from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.lr_sweep import (
    main,
    parse_named_report,
    summarize_lr_sweep,
    write_summary,
)

EMPTY_HASH = hashlib.sha256(b"").hexdigest()
SCENE_HASH = "690bd890bfda024dbb5c7d3c68087b8113bc3b8ee81dd6143c7eb2a884e7245b"
FROZEN_HASH = "dec768bed654c8c4e16da0318857543ad54d8f5f68f4d24a9a87cd19ec706594"
SOURCE_ADAPTER_HASH = "a4c85c14a214e4e594992e489a784cb4bacb64d3dfda519ad3da18b1595d9f22"
SOURCE_METADATA_HASH = "f097c6477546460440e77a3d225afb55818cb13abf9cbb4a90500f75a879b0f5"
TRAINABLE_INITIAL_HASH = "b4ec0518e4759dda33fc93c9c1d4c76f52f1024fd5b8b1667ad1b4ef5da198af"
TRAINABLE_FINAL_HASH = "6" * 64
SELECTION_HASH = "7f0714e3151c9ddb57c1da95a457820a833e490c070881a88a9fee4a9168f933"
MEMBERSHIP_HASH = "99ee448c23fb71b7269a353a54b2156ac55701847af170597dcc351af15cbcbe"


def _source_provenance(commit: str = "a" * 40) -> dict[str, object]:
    return {
        "schema_version": 1,
        "scope": "repository_excluding_generated_artifacts_v1",
        "available": True,
        "head_commit": commit,
        "head_tree": "b" * 40,
        "is_clean": True,
        "tracked_diff_sha256": EMPTY_HASH,
    }


def _pair_gate(
    *,
    full_unit: float,
    full_side: float,
    candidate_unit: float,
    candidate_side: float,
    full_mean: float,
    full_minimum: float,
    candidate_mean: float,
    candidate_minimum: float,
    full_hinge: float = 10.0,
    candidate_hinge: float = 1.0,
) -> dict[str, object]:
    return {
        "evaluation_type": "teacher_forced_same_distribution_candidate_logit_ranking",
        "ranking_mode": "candidate_logit",
        "free_generation_evaluated": False,
        "shared_candidate_tokens_excluded": True,
        "same_next_token_distribution": True,
        "unit_count": 6,
        "side_count": 12,
        "changed_unit_accuracy": candidate_unit,
        "side_accuracy": candidate_side,
        "first_answer_token_full_vocab_evaluated": True,
        "first_answer_token_top1_accuracy": full_side,
        "first_answer_token_top1_unit_accuracy": full_unit,
        "mean_first_answer_token_target_vs_best_other_logit_margin": full_mean,
        "minimum_first_answer_token_target_vs_best_other_logit_margin": full_minimum,
        "mean_own_vs_alternate_candidate_logit_margin": candidate_mean,
        "minimum_own_vs_alternate_candidate_logit_margin": candidate_minimum,
        "ranking_hinge_at_configured_margin": candidate_hinge,
        "first_answer_token_target_vs_best_other_hinge": full_hinge,
    }


def _report(
    learning_rate: float,
    *,
    mirror: dict[str, float] | None = None,
    color_full_side: float = 1.0,
    color_full_minimum: float = 0.5,
    color_candidate_minimum: float = 1.0,
) -> dict[str, object]:
    source = _source_provenance()
    color = _pair_gate(
        full_unit=1.0,
        full_side=color_full_side,
        candidate_unit=1.0,
        candidate_side=1.0,
        full_mean=1.5,
        full_minimum=color_full_minimum,
        candidate_mean=2.0,
        candidate_minimum=color_candidate_minimum,
    )
    mirror_values = {
        "full_unit": 0.0,
        "full_side": 0.0,
        "candidate_unit": 0.0,
        "candidate_side": 0.5,
        "full_mean": -10.0,
        "full_minimum": -20.0,
        "candidate_mean": 0.0,
        "candidate_minimum": -4.0,
    }
    if mirror:
        mirror_values.update(mirror)
    mirror_gate = _pair_gate(**mirror_values)
    final_gate = {"by_pair": {"pair_000001": color, "pair_000003": mirror_gate}}
    initialization = {
        "schema_version": 2,
        "mode": "legacy_lora_into_frozen_named_bank",
        "checkpoint": "data_gemma4/checkpoints/source/epoch_008",
        "adapter_sha256": SOURCE_ADAPTER_HASH,
        "metadata_sha256": SOURCE_METADATA_HASH,
        "expected_adapter_sha256": SOURCE_ADAPTER_HASH,
        "expected_metadata_sha256": SOURCE_METADATA_HASH,
        "checkpoint_epoch": 8,
        "checkpoint_output_namespace": "source_v12",
        "checkpoint_source_provenance": _source_provenance("c" * 40),
        "optimizer_state_loaded": False,
        "history_loaded": False,
        "target_bank": "inherited_v12",
        "target_bank_state_sha256": FROZEN_HASH,
        "new_trainable_banks_zero_output": True,
    }
    lora_banks = [
        {
            "name": "inherited_v12",
            "trainable": False,
            "rank": 4,
            "alpha": 8.0,
            "dropout": 0.0,
            "target_modules": ["layer.34.q_proj"],
            "initialization_algorithm": "checkpoint_overwrite",
            "initialization_seed": None,
            "expected_initial_state_sha256": FROZEN_HASH,
            "adapter_parameter_count": 10,
        },
        {
            "name": "extension_v14",
            "trainable": True,
            "rank": 8,
            "alpha": 16.0,
            "dropout": 0.0,
            "target_modules": ["layer.33.q_proj"],
            "initialization_algorithm": "cpu_kaiming_uniform_a_exact_zero_b",
            "initialization_seed": 14008,
            "expected_initial_state_sha256": TRAINABLE_INITIAL_HASH,
            "adapter_parameter_count": 20,
            "learning_rate": learning_rate,
            "weight_decay": 0.0,
        },
    ]
    selection = {
        "schema_version": 1,
        "strategy": "paired_expected_change_then_least_represented_answer_type_v1",
        "train": {"selected_ids_sha256": SELECTION_HASH},
        "training_counterfactual_pair_membership_sha256": MEMBERSHIP_HASH,
        "source_provenance": deepcopy(source),
        "initialize_expected_adapter_sha256": SOURCE_ADAPTER_HASH,
        "initialize_expected_metadata_sha256": SOURCE_METADATA_HASH,
        "train_scene_ids": ["scene_000003", "scene_000004", "scene_000007", "scene_000008"],
        "test_scene_ids": ["scene_000005", "scene_000006"],
    }
    return {
        "target_epochs": 4,
        "epochs": 4,
        "steps": 48,
        "optimizer_steps": 4,
        "gradient_accumulation": 12,
        "train_scene_ids": selection["train_scene_ids"],
        "test_scene_ids": selection["test_scene_ids"],
        "source_provenance": source,
        "initialization_provenance": initialization,
        "initialize_expected_adapter_sha256": SOURCE_ADAPTER_HASH,
        "initialize_expected_metadata_sha256": SOURCE_METADATA_HASH,
        "freeze_scene_adapter": True,
        "frozen_scene_state_sha256": SCENE_HASH,
        "frozen_lora_bank_state_sha256": {"inherited_v12": FROZEN_HASH},
        "lora_bank_state_sha256": {
            "inherited_v12": FROZEN_HASH,
            "extension_v14": TRAINABLE_FINAL_HASH,
        },
        "training_counterfactual_pair_membership_sha256": MEMBERSHIP_HASH,
        "selection": selection,
        "lora": {"schema_version": 2, "enabled": True, "banks": lora_banks},
        "lora_optimizer": {"learning_rate": learning_rate, "weight_decay": 0.0},
        "pair_candidate_gate": final_gate,
        "history": [
            {"epoch": 1, "pair_candidate_gate": deepcopy(final_gate)},
            {"epoch": 2, "pair_candidate_gate": deepcopy(final_gate)},
            {"epoch": 3, "pair_candidate_gate": deepcopy(final_gate)},
            {"epoch": 4, "pair_candidate_gate": deepcopy(final_gate)},
        ],
    }


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
    return path


def test_sweep_uses_declared_lexicographic_ranking_and_lower_lr_tiebreak(tmp_path: Path) -> None:
    # Full-vocabulary unit accuracy dominates every later, even much larger, metric.
    weaker_full_unit = _report(
        1e-4,
        mirror={
            "full_unit": 0.5,
            "full_side": 1.0,
            "candidate_unit": 1.0,
            "candidate_side": 1.0,
            "full_mean": 100.0,
            "full_minimum": 100.0,
            "candidate_mean": 100.0,
            "candidate_minimum": 100.0,
        },
    )
    # These arms are scientifically tied, so the lower LR must win.
    tied_metrics = {
        "full_unit": 2 / 3,
        "full_side": 0.75,
        "candidate_unit": 0.5,
        "candidate_side": 0.75,
        "full_mean": -2.0,
        "full_minimum": -7.0,
        "candidate_mean": 0.5,
        "candidate_minimum": -1.0,
    }
    reports = {
        "high_lr": _write(tmp_path / "high.json", _report(2e-3, mirror=tied_metrics)),
        "low_lr": _write(tmp_path / "low.json", _report(5e-4, mirror=tied_metrics)),
        "weak": _write(tmp_path / "weak.json", weaker_full_unit),
    }

    summary = summarize_lr_sweep(reports)

    assert summary["selected_arm"] == "low_lr"
    assert summary["ranked_winner"] == "low_lr"
    assert summary["extension_qualified_count"] == 3
    assert [row["name"] for row in summary["ranking"]] == ["low_lr", "high_lr", "weak"]
    assert summary["eligible_count"] == 3
    assert summary["rejected_count"] == 0


@pytest.mark.parametrize(
    ("mutate", "expected_reason"),
    [
        (
            lambda report: report["history"][-1].pop("pair_candidate_gate"),
            "missing_final_pair_gate",
        ),
        (
            lambda report: report["lora_bank_state_sha256"].update({"inherited_v12": "9" * 64}),
            "frozen_state_violation",
        ),
        (
            lambda report: report["source_provenance"].update({"is_clean": False}),
            "invalid_provenance",
        ),
        (
            lambda report: report["pair_candidate_gate"]["by_pair"]["pair_000003"].update(
                {"mean_own_vs_alternate_candidate_logit_margin": float("nan")}
            ),
            "non_finite_value",
        ),
        (
            lambda report: report.update({"steps": 47}),
            "protocol_violation",
        ),
        (
            lambda report: report["lora_bank_state_sha256"].update(
                {"extension_v14": TRAINABLE_INITIAL_HASH}
            ),
            "frozen_state_violation",
        ),
        (
            lambda report: report["lora"]["banks"][0].update({"trainable": True}),
            "invalid_provenance",
        ),
    ],
)
def test_integrity_violations_are_hard_rejected(
    tmp_path: Path,
    mutate: object,
    expected_reason: str,
) -> None:
    report = _report(1e-3)
    mutate(report)  # type: ignore[operator]
    path = _write(tmp_path / "arm.json", report)

    summary = summarize_lr_sweep({"arm": path})

    assert summary["selected_arm"] is None
    assert summary["eligible_count"] == 0
    assert summary["arms"][0]["rejection_reasons"] == [expected_reason]


def test_color_regression_is_visible_but_never_ranked(tmp_path: Path) -> None:
    good = _report(5e-4, mirror={"full_unit": 0.5})
    regressed = _report(
        2e-3,
        mirror={
            "full_unit": 1.0,
            "full_side": 1.0,
            "candidate_unit": 1.0,
            "candidate_side": 1.0,
        },
        color_full_side=11 / 12,
    )
    reports = {
        "good": _write(tmp_path / "good.json", good),
        "regressed": _write(tmp_path / "regressed.json", regressed),
    }

    summary = summarize_lr_sweep(reports)

    assert summary["selected_arm"] == "good"
    assert [row["name"] for row in summary["ranking"]] == ["good"]
    rejected = next(arm for arm in summary["arms"] if arm["name"] == "regressed")
    assert rejected["rejection_reasons"] == ["color_integrity_failed"]
    assert rejected["mirror_metrics"]["full_vocab_unit_accuracy"] == 1.0


def test_ranked_arm_cannot_be_selected_for_extension_without_response_threshold(
    tmp_path: Path,
) -> None:
    report = _report(
        1e-3,
        mirror={
            "candidate_hinge": 2.0455729961395264,
            "full_hinge": 19.0,
            "candidate_minimum": -4.0,
            "full_minimum": -24.0,
        },
    )

    summary = summarize_lr_sweep({"flat": _write(tmp_path / "flat.json", report)})

    assert summary["eligible_count"] == 1
    assert summary["ranked_winner"] == "flat"
    assert summary["extension_qualified_count"] == 0
    assert summary["selected_arm"] is None
    assert summary["ranking"][0]["extension_checks"] == {
        "candidate_hinge_better_than_v13": False,
        "full_vocab_hinge_better_than_v13": True,
        "candidate_minimum_not_worse_than_v12": True,
        "full_vocab_minimum_not_worse_than_v12": True,
    }


@pytest.mark.parametrize(
    "minimum_kwargs",
    [
        {"color_full_minimum": 0.0},
        {"color_candidate_minimum": 0.0},
    ],
)
def test_color_accuracy_alone_does_not_hide_zero_minimum_margin(
    tmp_path: Path, minimum_kwargs: dict[str, float]
) -> None:
    report = _report(1e-3, **minimum_kwargs)

    summary = summarize_lr_sweep({"arm": _write(tmp_path / "arm.json", report)})

    assert summary["selected_arm"] is None
    assert summary["arms"][0]["color_integrity"]["full_vocab_side_accuracy"] == 1.0
    assert summary["arms"][0]["rejection_reasons"] == ["color_integrity_failed"]


def test_unique_majority_contract_rejects_mismatched_arm(tmp_path: Path) -> None:
    reports = {name: _report(lr) for name, lr in (("a", 5e-4), ("b", 1e-3), ("c", 2e-3))}
    reports["c"]["source_provenance"]["head_commit"] = "d" * 40  # type: ignore[index]
    reports["c"]["selection"]["source_provenance"]["head_commit"] = "d" * 40  # type: ignore[index]
    paths = {name: _write(tmp_path / f"{name}.json", report) for name, report in reports.items()}

    summary = summarize_lr_sweep(paths)

    assert summary["eligible_count"] == 2
    assert next(arm for arm in summary["arms"] if arm["name"] == "c")["rejection_reasons"] == [
        "cross_arm_provenance_mismatch"
    ]


def test_machine_readable_output_exposes_pinned_protocol(tmp_path: Path) -> None:
    path = _write(tmp_path / "arm.json", _report(1e-3))

    contract = summarize_lr_sweep({"arm": path})["expected_sweep_contract"]

    assert contract["epochs"] == 4
    assert contract["steps"] == 48
    assert contract["optimizer_steps"] == 4
    assert contract["gradient_accumulation"] == 12
    assert contract["frozen_scene_state_sha256"] == SCENE_HASH
    assert contract["frozen_bank_state_sha256"] == FROZEN_HASH
    assert contract["source_adapter_sha256"] == SOURCE_ADAPTER_HASH
    assert contract["source_metadata_sha256"] == SOURCE_METADATA_HASH
    assert contract["selection_sha256"] == SELECTION_HASH
    assert contract["pair_membership_sha256"] == MEMBERSHIP_HASH
    assert contract["trainable_initial_state_sha256"] == TRAINABLE_INITIAL_HASH
    assert contract["v13_candidate_hinge"] == pytest.approx(2.0455729961395264)
    assert contract["v13_full_vocab_hinge"] == pytest.approx(19.23177146911621)


def test_writer_and_cli_emit_strict_machine_readable_json(tmp_path: Path, capsys: object) -> None:
    report_path = _write(tmp_path / "arm.json", _report(1e-3))
    direct_output = tmp_path / "direct.json"
    summary = summarize_lr_sweep({"arm": report_path})
    write_summary(summary, direct_output)
    assert json.loads(direct_output.read_text(encoding="utf-8"))["selected_arm"] == "arm"

    cli_output = tmp_path / "cli.json"
    assert main(["--report", f"arm={report_path}", "--output", str(cli_output)]) == 0
    assert json.loads(cli_output.read_text(encoding="utf-8"))["selected_arm"] == "arm"
    assert '"selected_arm": "arm"' in capsys.readouterr().out  # type: ignore[attr-defined]
    assert parse_named_report(f"arm={report_path}") == ("arm", report_path)
