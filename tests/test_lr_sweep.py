from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from semantic_3d_chat.evaluation.lr_sweep import (
    EXPECTED_V14_ARMS,
    attest_v14_lr_sweep,
    main,
    parse_named_report,
    summarize_lr_sweep,
    write_summary,
)
from semantic_3d_chat.language.lora import tensor_state_sha256

EMPTY_HASH = hashlib.sha256(b"").hexdigest()
SCENE_HASH = "690bd890bfda024dbb5c7d3c68087b8113bc3b8ee81dd6143c7eb2a884e7245b"
FROZEN_HASH = "dec768bed654c8c4e16da0318857543ad54d8f5f68f4d24a9a87cd19ec706594"
SOURCE_ADAPTER_HASH = "a4c85c14a214e4e594992e489a784cb4bacb64d3dfda519ad3da18b1595d9f22"
SOURCE_METADATA_HASH = "f097c6477546460440e77a3d225afb55818cb13abf9cbb4a90500f75a879b0f5"
TRAINABLE_INITIAL_HASH = "b4ec0518e4759dda33fc93c9c1d4c76f52f1024fd5b8b1667ad1b4ef5da198af"
TRAINABLE_FINAL_HASH = "6" * 64
SELECTION_HASH = "7f0714e3151c9ddb57c1da95a457820a833e490c070881a88a9fee4a9168f933"
MEMBERSHIP_HASH = "99ee448c23fb71b7269a353a54b2156ac55701847af170597dcc351af15cbcbe"
SOURCE_HEAD = "1ee8b5d13777e74ebdfe1f87e7d8320403ad5fbf"
SOURCE_TREE = "b606e85cbb5a786ba2e00f971cf07c174bc5cbef"


def _source_provenance(
    commit: str = SOURCE_HEAD, tree: str = SOURCE_TREE
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "scope": "repository_excluding_generated_artifacts_v1",
        "available": True,
        "head_commit": commit,
        "head_tree": tree,
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
            "name": "extension_v13",
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
            "extension_v13": TRAINABLE_FINAL_HASH,
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


def _v14_attestation_fixture(
    tmp_path: Path,
) -> tuple[dict[str, Path], dict[str, Path], Path]:
    checkpoints: dict[str, Path] = {}
    selections: dict[str, Path] = {}
    historical_reports: dict[str, Path] = {}
    for index, (name, contract) in enumerate(sorted(EXPECTED_V14_ARMS.items())):
        mirror = {"full_unit": 0.5} if name == "lr2e3" else {"full_unit": 0.0}
        report = _report(float(contract["learning_rate"]), mirror=mirror)
        checkpoint = tmp_path / f"checkpoint_{name}" / "epoch_004"
        checkpoint.mkdir(parents=True)
        trainable_state = {
            "adapters.0.lora_a": torch.tensor(
                [[1.0 + index, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32
            ),
            "adapters.0.lora_b": torch.tensor(
                [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8 + index]],
                dtype=torch.float32,
            ),
        }
        final_hash = tensor_state_sha256(trainable_state)
        report["lora_bank_state_sha256"]["extension_v13"] = final_hash  # type: ignore[index]
        report["lora"]["banks"][1]["adapter_parameter_count"] = 14  # type: ignore[index]
        adapter_tensors = {
            f"lora_banks.extension_v13.{key}": value for key, value in trainable_state.items()
        }
        save_file(adapter_tensors, checkpoint / "adapter.safetensors")
        optimizer = {
            "state": {
                0: {"step": torch.tensor(4.0)},
                1: {"step": torch.tensor(4.0)},
            },
            "param_groups": [
                {
                    "params": [0, 1],
                    "lr": float(contract["learning_rate"]),
                    "weight_decay": 0.0,
                }
            ],
        }
        torch.save(optimizer, checkpoint / "optimizer.pt")
        metadata = deepcopy(report)
        metadata.pop("selection")
        metadata.pop("lora_optimizer")
        metadata.update(
            {
                "schema_version": 3,
                "config_hash": contract["config_hash"],
                "output_namespace": contract["output_namespace"],
                "epoch": 4,
                "global_step": 48,
                "optimizer_step": 4,
            }
        )
        _write(checkpoint / "metadata.json", metadata)
        selection = deepcopy(report["selection"])
        selection.update(
            {
                "lora": deepcopy(report["lora"]),
                "lora_optimizer": deepcopy(report["lora_optimizer"]),
                "gradient_accumulation": 12,
                "freeze_scene_adapter": True,
            }
        )
        selection_path = _write(tmp_path / f"selection_{name}.json", selection)
        historical_path = _write(tmp_path / f"historical_{name}.json", report)
        checkpoints[name] = checkpoint
        selections[name] = selection_path
        historical_reports[name] = historical_path
    summary_path = tmp_path / "historical_sweep.json"
    write_summary(summarize_lr_sweep(historical_reports), summary_path)
    return checkpoints, selections, summary_path


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
                {"extension_v13": TRAINABLE_INITIAL_HASH}
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
    assert contract["source_head_commit"] == SOURCE_HEAD
    assert contract["source_head_tree"] == SOURCE_TREE
    assert contract["v13_candidate_hinge"] == pytest.approx(2.0455729961395264)
    assert contract["v13_full_vocab_hinge"] == pytest.approx(19.23177146911621)


def test_writer_and_cli_emit_strict_machine_readable_json(tmp_path: Path, capsys: object) -> None:
    report_path = _write(tmp_path / "arm.json", _report(1e-3))
    expected_input_hash = hashlib.sha256(report_path.read_bytes()).hexdigest()
    direct_output = tmp_path / "direct.json"
    summary = summarize_lr_sweep({"arm": report_path})
    assert summary["arms"][0]["input_report_sha256"] == expected_input_hash
    assert summary["ranking"][0]["input_report_sha256"] == expected_input_hash
    assert summary["selected_report_sha256"] == expected_input_hash
    write_summary(summary, direct_output)
    assert json.loads(direct_output.read_text(encoding="utf-8"))["selected_arm"] == "arm"

    cli_output = tmp_path / "cli.json"
    assert main(["--report", f"arm={report_path}", "--output", str(cli_output)]) == 0
    assert json.loads(cli_output.read_text(encoding="utf-8"))["selected_arm"] == "arm"
    assert '"selected_arm": "arm"' in capsys.readouterr().out  # type: ignore[attr-defined]
    assert parse_named_report(f"arm={report_path}") == ("arm", report_path)


def test_ranked_summary_keeps_content_hash_when_report_path_is_overwritten(tmp_path: Path) -> None:
    report_path = _write(tmp_path / "mutable.json", _report(1e-3))
    summary = summarize_lr_sweep({"arm": report_path})
    original_hash = summary["selected_report_sha256"]

    _write(report_path, _report(2e-3, mirror={"full_unit": 1.0}))

    assert hashlib.sha256(report_path.read_bytes()).hexdigest() != original_hash
    assert summary["arms"][0]["input_report_sha256"] == original_hash
    assert summarize_lr_sweep({"arm": report_path})["selected_report_sha256"] != original_hash


def test_rejected_malformed_report_still_records_raw_input_hash(tmp_path: Path) -> None:
    path = tmp_path / "malformed.json"
    path.write_bytes(b'{"incomplete":')

    summary = summarize_lr_sweep({"bad": path})

    assert summary["arms"][0]["rejection_reasons"] == ["unreadable_report"]
    assert summary["arms"][0]["input_report_sha256"] == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_v14_checkpoint_attestation_binds_all_raw_evidence_and_recomputed_bank(
    tmp_path: Path,
) -> None:
    checkpoints, selections, sweep_summary = _v14_attestation_fixture(tmp_path)

    attestation = attest_v14_lr_sweep(checkpoints, selections, sweep_summary)

    assert attestation["all_arms_attested"] is True
    assert attestation["historical_training_reports_loaded"] is False
    assert attestation["validated_arm_count"] == 4
    assert attestation["ranking"][0] == "lr2e3"
    assert attestation["selected_arm"] == "lr2e3"
    assert attestation["historical_sweep_summary"]["sha256"] == hashlib.sha256(
        sweep_summary.read_bytes()
    ).hexdigest()
    for arm in attestation["arms"]:
        name = arm["name"]
        raw_hashes = arm["raw_file_sha256"]
        assert raw_hashes == {
            "metadata.json": hashlib.sha256(
                (checkpoints[name] / "metadata.json").read_bytes()
            ).hexdigest(),
            "adapter.safetensors": hashlib.sha256(
                (checkpoints[name] / "adapter.safetensors").read_bytes()
            ).hexdigest(),
            "optimizer.pt": hashlib.sha256(
                (checkpoints[name] / "optimizer.pt").read_bytes()
            ).hexdigest(),
            "selection.json": hashlib.sha256(selections[name].read_bytes()).hexdigest(),
        }
        assert arm["recomputed_trainable_bank_state_sha256"]["extension_v13"] == (
            json.loads((checkpoints[name] / "metadata.json").read_text(encoding="utf-8"))[
                "lora_bank_state_sha256"
            ]["extension_v13"]
        )
        assert arm["optimizer_validation"]["slot_optimizer_steps"] == [4]
        assert arm["sweep_summary_correspondence"] is True


@pytest.mark.parametrize(
    ("evidence", "key_path", "replacement", "expected_code"),
    [
        ("metadata", ("epoch",), 3, "protocol_violation"),
        ("metadata", ("global_step",), 47, "protocol_violation"),
        ("metadata", ("optimizer_step",), 3, "protocol_violation"),
        ("metadata", ("gradient_accumulation",), 11, "protocol_violation"),
        ("metadata", ("source_provenance", "head_commit"), "0" * 40, "invalid_provenance"),
        ("metadata", ("config_hash",), "bad", "invalid_provenance"),
        ("metadata", ("frozen_scene_state_sha256",), "0" * 64, "frozen_state_violation"),
        (
            "metadata",
            ("frozen_lora_bank_state_sha256", "inherited_v12"),
            "0" * 64,
            "frozen_state_violation",
        ),
        ("selection", ("train", "selected_ids_sha256"), "0" * 64, "invalid_provenance"),
        (
            "selection",
            ("training_counterfactual_pair_membership_sha256",),
            "0" * 64,
            "invalid_provenance",
        ),
    ],
)
def test_v14_checkpoint_attestation_fails_closed_on_protocol_and_provenance_mutation(
    tmp_path: Path,
    evidence: str,
    key_path: tuple[str, ...],
    replacement: object,
    expected_code: str,
) -> None:
    checkpoints, selections, sweep_summary = _v14_attestation_fixture(tmp_path)
    target = (
        checkpoints["lr2e3"] / "metadata.json"
        if evidence == "metadata"
        else selections["lr2e3"]
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    cursor = payload
    for key in key_path[:-1]:
        cursor = cursor[key]
    cursor[key_path[-1]] = replacement
    _write(target, payload)

    with pytest.raises(ValueError) as exc_info:
        attest_v14_lr_sweep(checkpoints, selections, sweep_summary)

    assert getattr(exc_info.value, "code", None) == expected_code


def test_v14_checkpoint_attestation_recomputes_adapter_and_optimizer_state(
    tmp_path: Path,
) -> None:
    checkpoints, selections, sweep_summary = _v14_attestation_fixture(tmp_path)
    checkpoint = checkpoints["lr2e3"]
    tensors = {
        "lora_banks.extension_v13.adapters.0.lora_a": torch.zeros((2, 3)),
        "lora_banks.extension_v13.adapters.0.lora_b": torch.zeros((4, 2)),
    }
    save_file(tensors, checkpoint / "adapter.safetensors")

    with pytest.raises(ValueError) as adapter_error:
        attest_v14_lr_sweep(checkpoints, selections, sweep_summary)
    assert getattr(adapter_error.value, "code", None) == "adapter_state_hash_mismatch"

    checkpoints, selections, sweep_summary = _v14_attestation_fixture(tmp_path / "optimizer")
    checkpoint = checkpoints["lr2e3"]
    optimizer = torch.load(checkpoint / "optimizer.pt", weights_only=True)
    for slot in optimizer["state"].values():
        slot["step"] = torch.tensor(3.0)
    torch.save(optimizer, checkpoint / "optimizer.pt")

    with pytest.raises(ValueError) as optimizer_error:
        attest_v14_lr_sweep(checkpoints, selections, sweep_summary)
    assert getattr(optimizer_error.value, "code", None) == "invalid_optimizer_checkpoint"


def test_v14_checkpoint_attestation_rejects_historical_gate_mutation(tmp_path: Path) -> None:
    checkpoints, selections, sweep_summary = _v14_attestation_fixture(tmp_path)
    summary = json.loads(sweep_summary.read_text(encoding="utf-8"))
    next(arm for arm in summary["arms"] if arm["name"] == "lr2e3")["mirror_metrics"][
        "candidate_mean_margin"
    ] += 1.0
    _write(sweep_summary, summary)

    with pytest.raises(ValueError) as exc_info:
        attest_v14_lr_sweep(checkpoints, selections, sweep_summary)

    assert getattr(exc_info.value, "code", None) == "sweep_summary_mismatch"
