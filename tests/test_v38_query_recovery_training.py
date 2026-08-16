from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from semantic_3d_chat.config import load_config
from semantic_3d_chat.language.lora import lora_banks_settings, tensor_state_sha256
from semantic_3d_chat.training.pair_curriculum import build_exact_question_pair_units
from semantic_3d_chat.training.train_block_cross_v35 import load_v35_train_qa_records
from semantic_3d_chat.training.train_query_recovery_v38 import (
    _HYBRID_FROZEN_STATE_SHA256,
    _HYBRID_FULL_STATE_SHA256,
    _HYBRID_V23_STATE_SHA256,
    _PAIR_SCHEDULE_SHA256,
    _QUERY_BANK,
    _QUERY_CONSTRUCTION_STATE_SHA256,
    _QUERY_PARAMETER_NAMES,
    _QUERY_SOURCE_STATE_SHA256,
    _ROLLED_BACK_V_NAMES,
    _TARGET_INITIAL_STATE_SHA256,
    _V23_BANK,
    _V37_TERMINAL_SHA256,
    OPTIMIZER_AUDIT_FILENAME,
    _optimizer_payload_audit,
    assemble_v38_hybrid_tensors,
    build_v38_schedule,
    optimizer_step_audit,
    preflight_v38,
    priority_side_deficit,
    require_v37_terminal_gate,
    v38_contract,
    v38_loader_config,
    v38_loss_values,
    v38_settings,
    v38_update8_gate,
    v38_update16_gate,
    v38_update41_gate,
    validate_per_unit_nll_diagnostics,
)

CONFIG = Path("configs/experiments/gemma4_diverse28_query_recovery_v38.yaml")


def _config() -> dict:
    return load_config(CONFIG)


def test_v38_contract_locks_exact_hybrid_and_query_only_runtime() -> None:
    config = _config()
    contract = v38_contract(config)
    settings = v38_settings(config)
    banks = lora_banks_settings(config)
    assert contract.terminal_report_sha256 == _V37_TERMINAL_SHA256
    assert contract.hybrid_tensor_state_sha256 == _HYBRID_FULL_STATE_SHA256
    assert contract.hybrid_v23_state_sha256 == _HYBRID_V23_STATE_SHA256
    assert contract.frozen_state_sha256 == _HYBRID_FROZEN_STATE_SHA256
    assert settings.saved_optimizer_steps == (0, 8, 16, 24, 32, 40, 41)
    assert banks.bank(_V23_BANK).trainable is False
    assert banks.bank(_V23_BANK).expected_initial_state_sha256 == _HYBRID_V23_STATE_SHA256
    assert banks.bank(_QUERY_BANK).trainable is True
    assert banks.bank(_QUERY_BANK).expected_initial_state_sha256 == (
        _QUERY_SOURCE_STATE_SHA256
    )


def test_v38_loader_copy_restores_only_construction_contract() -> None:
    actual = _config()
    loader = v38_loader_config(actual)
    actual_banks = lora_banks_settings(actual)
    loader_banks = lora_banks_settings(loader)
    assert actual_banks.bank(_QUERY_BANK).initialization_algorithm == "checkpoint_overwrite"
    assert loader_banks.bank(_QUERY_BANK).initialization_algorithm == (
        "cpu_kaiming_uniform_a_exact_zero_b"
    )
    assert loader_banks.bank(_QUERY_BANK).initialization_seed == 30030
    assert loader_banks.bank(_QUERY_BANK).expected_initial_state_sha256 == (
        _QUERY_CONSTRUCTION_STATE_SHA256
    )
    assert loader_banks.bank(_V23_BANK).trainable is False
    assert loader_banks.bank(_V23_BANK).expected_initial_state_sha256 == (
        _TARGET_INITIAL_STATE_SHA256
    )
    assert loader["training"]["lora_learning_rate"] == 0.0002


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("training.v38_query_recovery", "learning_rate", 2e-4),
        ("v38_query_recovery", "v37_optimizer_file_opened", True),
        ("v38_query_recovery", "validation_qa_loaded_during_training", True),
        ("v38_query_recovery", "hybrid_full_tensor_state_sha256", "0" * 64),
        ("v38_query_recovery", "schedule_sha256", "0" * 64),
        ("v38_query_recovery", "target_bank_source_state_sha256", "0" * 64),
    ],
)
def test_v38_contract_fails_closed(section: str, field: str, value: object) -> None:
    config = copy.deepcopy(_config())
    target = config
    for key in section.split("."):
        target = target[key]
    target[field] = value
    with pytest.raises((TypeError, ValueError)):
        v38_contract(config)


def test_v38_terminal_and_hybrid_replay_exact_real_files() -> None:
    config = _config()
    terminal = require_v37_terminal_gate(config)
    hybrid, audit = assemble_v38_hybrid_tensors(config)
    assert terminal["sha256"] == _V37_TERMINAL_SHA256
    assert tensor_state_sha256(hybrid) == _HYBRID_FULL_STATE_SHA256
    assert tuple(audit["differs_from_v37_only_tensor_names"]) == tuple(
        sorted(_ROLLED_BACK_V_NAMES)
    )
    assert audit["v37_optimizer_file_opened"] is False
    assert audit["v36_optimizer_file_opened"] is False


def test_v38_preflight_never_opens_source_adam_validation_or_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden_optimizers = {
        (
            Path("data_gemma4/checkpoints/gemma4_v37_diverse28_scene_ingress_kv/update_016")
            / "optimizer.pt"
        ).resolve(),
        (
            Path("data_gemma4/checkpoints/gemma4_v36_diverse28_joint_block_cross/update_016")
            / "optimizer.pt"
        ).resolve(),
    }
    original_open = Path.open
    original_read_text = Path.read_text

    def guarded_open(path: Path, *args, **kwargs):
        if path.resolve() in forbidden_optimizers:
            raise AssertionError("V38 preflight opened inherited Adam state")
        return original_open(path, *args, **kwargs)

    def guarded_read_text(path: Path, *args, **kwargs):
        lowered = {part.casefold() for part in path.parts}
        if path.name == "validation.jsonl" or "oracle" in lowered:
            raise AssertionError("V38 preflight crossed its train-only data boundary")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    report = preflight_v38(_config())
    assert report["passed"] is True
    assert report["source_optimizer_files_opened"] is False
    assert report["scene_maps_loaded"] is False
    assert report["validation_qa_loaded"] is False


def test_v38_schedule_is_exact_priority_twice_then_canonical_cycle() -> None:
    config = _config()
    records, _audit = load_v35_train_qa_records(v38_loader_config(config))
    units = build_exact_question_pair_units(records)
    schedule, audit = build_v38_schedule(records, units, seed=int(config["seed"]))
    assert len(schedule) == 41
    assert audit["pair_schedule_sha256"] == _PAIR_SCHEDULE_SHA256
    assert audit["schedule_sha256"] == (
        "2e7a2a8136c968739a8aaaa1138be87bec322fefa0fb340e6fb9bc6e07278bea"
    )
    assert [item.pair_unit.question_key for item in schedule[:8]] == [
        "cfq_13b1138d14c52a7c",
        "cfq_1c8b8cd72fcde904",
        "cfq_163eb92339ad35a5",
        "cfq_66aab89cee5bef49",
        "cfq_a1c673a1197a0961",
        "cfq_d469c4ac156ac42d",
        "cfq_ac7ac024c40aaddc",
        "cfq_fa3601dfffa80a0e",
    ]
    assert [item.pair_unit.question_key for item in schedule[:8]] == [
        item.pair_unit.question_key for item in schedule[8:16]
    ]
    assert audit["saved_optimizer_steps"] == [0, 8, 16, 24, 32, 40, 41]


def _metric_and_nll_fixture() -> tuple[dict, list[dict]]:
    rows = []
    diagnostics = []
    families = ["book_support"] * 4 + ["picture_support"] * 4 + ["mirror_lr"] * 4
    families += ["other"] * 13
    pair_ids = ["pair_000015"] * 4 + ["pair_000017"] * 4 + ["pair_000016"] * 4
    pair_ids += [f"pair_{index:06d}" for index in range(100, 113)]
    for index, (family, pair_id) in enumerate(zip(families, pair_ids, strict=True)):
        identity = f"key_{index:02d}"
        scene_ids = [f"scene_{2 * index:06d}", f"scene_{2 * index + 1:06d}"]
        correct = [1.0, 1.0]
        swapped = [2.0, 2.0]
        side = [1.0, 1.0]
        cross = [1.0, 1.0]
        rows.append(
            {
                "pair_id": pair_id,
                "question_key": identity,
                "scene_ids": scene_ids,
                "family": family,
                "side_margins": side,
                "cross_prefix_margins": cross,
                "complete": True,
                "cross_prefix_complete": True,
            }
        )
        diagnostics.append(
            {
                "pair_id": pair_id,
                "question_key": identity,
                "family": family,
                "scene_ids": scene_ids,
                "correct_answer_nll_mean": 1.0,
                "correct_answer_nll": [1.0, 1.0],
                "correct_ranking_nll": correct,
                "swapped_ranking_nll": swapped,
                "side_margins": side,
                "cross_prefix_margins": cross,
                "side_correct": [True, True],
                "cross_prefix_correct": [True, True],
                "side_complete": True,
                "cross_prefix_complete": True,
            }
        )
    metrics = {
        "complete_units": 25,
        "positive_sides": 50,
        "cross_prefix_complete_units": 25,
        "complete_physical_pair_coverage": 16,
        "complete_units_by_family": {
            "book_support": 4,
            "picture_support": 4,
            "mirror_lr": 4,
        },
        "cross_prefix_complete_units_by_family": {
            "book_support": 4,
            "picture_support": 4,
            "mirror_lr": 4,
        },
        "units": rows,
    }
    return metrics, diagnostics


def test_v38_per_unit_diagnostics_are_2x2_score_bound_and_gate_replayable() -> None:
    contract = v38_contract(_config())
    metrics, diagnostics = _metric_and_nll_fixture()
    audit = validate_per_unit_nll_diagnostics(diagnostics, metrics)
    assert audit["unique_pair_question_key_count"] == 25
    assert priority_side_deficit(metrics)["combined"] == 0.0
    gate8 = v38_update8_gate(
        pair_metrics=metrics,
        broad_nll=2.90,
        source_broad_nll=2.9013306349515915,
        source_priority_deficit=31.1137291,
        query_state_sha256="changed",
        frozen_state_sha256=_HYBRID_FROZEN_STATE_SHA256,
        scene_state_exact=True,
        per_unit_nll_diagnostics=diagnostics,
        contract=contract,
    )
    gate16 = v38_update16_gate(
        update8_gate=gate8,
        pair_metrics=metrics,
        broad_nll=2.90,
        source_broad_nll=2.9013306349515915,
        source_priority_deficit=31.1137291,
        query_state_sha256="changed",
        frozen_state_sha256=_HYBRID_FROZEN_STATE_SHA256,
        scene_state_exact=True,
        per_unit_nll_diagnostics=diagnostics,
        contract=contract,
    )
    gate41 = v38_update41_gate(
        update16_gate=gate16,
        pair_metrics=metrics,
        greedy_metrics={
            "complete_units": 6,
            "complete_units_by_family": {
                "book_support": 1,
                "picture_support": 1,
                "mirror_lr": 1,
            },
            "broad_exact_correct": 23,
            "broad_row_count": 48,
        },
        broad_nll=2.90,
        source_broad_nll=2.9013306349515915,
        source_priority_deficit=31.1137291,
        query_state_sha256="changed",
        frozen_state_sha256=_HYBRID_FROZEN_STATE_SHA256,
        scene_state_exact=True,
        per_unit_nll_diagnostics=diagnostics,
        contract=contract,
    )
    assert gate8["passed"] is True
    assert gate16["passed"] is True
    assert gate41["passed"] is True
    assert all(gate["checks"] for gate in (gate8, gate16, gate41))

    tampered = copy.deepcopy(diagnostics)
    tampered[0]["correct_answer_nll"][0] += 1.0
    with pytest.raises(ValueError, match="2x2 scores"):
        validate_per_unit_nll_diagnostics(tampered, metrics)


def _optimizer_fixture(step: int) -> tuple[dict, dict[str, torch.Tensor]]:
    parameters = [torch.nn.Parameter(torch.zeros(index + 1)) for index in range(8)]
    optimizer = torch.optim.AdamW(
        [
            {
                "name": f"lora_banks.{_QUERY_BANK}",
                "params": parameters,
                "parameter_names": list(_QUERY_PARAMETER_NAMES),
                "lr": 2e-5,
                "weight_decay": 0.0,
            }
        ]
    )
    for _ in range(step):
        optimizer.zero_grad(set_to_none=True)
        sum(parameter.sum() for parameter in parameters).backward()
        optimizer.step()
    tensors = {
        name: parameter.detach().clone()
        for name, parameter in zip(_QUERY_PARAMETER_NAMES, parameters, strict=True)
    }
    return optimizer.state_dict(), tensors


def test_v38_optimizer_audit_locks_group_schema_moments_and_file_hash(
    tmp_path: Path,
) -> None:
    payload, tensors = _optimizer_fixture(2)
    checkpoint = tmp_path / "update_002"
    checkpoint.mkdir()
    optimizer_path = checkpoint / "optimizer.pt"
    torch.save(payload, optimizer_path)
    digest = hashlib.sha256(optimizer_path.read_bytes()).hexdigest()
    (checkpoint / OPTIMIZER_AUDIT_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact": "v38_optimizer_integrity_manifest",
                "optimizer_step": 2,
                "optimizer_filename": "optimizer.pt",
                "optimizer_sha256": digest,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    assert optimizer_step_audit(
        checkpoint, expected_step=2, tensors=tensors
    )["self_hash_linkage_verified"] is True
    tampered = copy.deepcopy(payload)
    tampered["param_groups"][0]["maximize"] = True
    with pytest.raises(ValueError, match="group identity/order/settings"):
        _optimizer_payload_audit(tampered, expected_step=2, tensors=tensors)


def test_v38_loss_optimizes_only_authorized_terms() -> None:
    settings = v38_settings(_config())
    optimized, reported = v38_loss_values(
        settings=settings,
        broad_nll=2.0,
        pair_correct_nll=3.0,
        side_hinge=4.0,
        cross_prefix_hinge=5.0,
        frozen_normalized_residual=999.0,
    )
    assert optimized == pytest.approx(0.5 * 2 + 3 + 8 * 4 + 5)
    assert reported == optimized
