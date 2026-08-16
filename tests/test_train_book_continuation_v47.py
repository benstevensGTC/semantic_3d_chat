from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.training import train_book_continuation_v47 as v47
from semantic_3d_chat.training.pair_curriculum import build_exact_question_pair_units
from semantic_3d_chat.training.train_projected_gradient_v41 import v41_loader_config

CONFIG = PROJECT_ROOT / "configs/experiments/gemma4_diverse28_book_continuation_v47.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gate_metrics(*, lost_positive: bool = True) -> dict[str, Any]:
    required = {
        "cfq_a578dc166be9a217": ("pair_000005", "other"),
        "cfq_0a79d507273195ef": ("pair_000006", "other"),
        "cfq_5c84a2c27d2be251": ("pair_000006", "other"),
        "cfq_736067b51ce93c49": ("pair_000007", "other"),
        "cfq_997610c185204121": ("pair_000007", "other"),
        "cfq_699675ceeaf65406": ("pair_000016", "mirror_lr"),
        "cfq_90b3d9852a93ce2a": ("pair_000018", "other"),
        "cfq_13b1138d14c52a7c": ("pair_000015", "book_support"),
        "cfq_a1c673a1197a0961": ("pair_000015", "book_support"),
    }
    rows: list[dict[str, Any]] = [
        {
            "pair_id": pair_id,
            "question_key": key,
            "family": family,
            "side_margins": [0.5, 0.5],
            "cross_prefix_margins": [0.5, 0.5],
        }
        for key, (pair_id, family) in required.items()
    ]
    for index in range(16):
        family = "book_support" if index < 2 else "picture_support" if index < 6 else "other"
        rows.append(
            {
                "pair_id": f"pair_fill_{index:02d}",
                "question_key": f"cfq_fill_{index:02d}",
                "family": family,
                "side_margins": [0.5, 0.5],
                "cross_prefix_margins": [0.5, 0.5],
            }
        )
    assert len(rows) == 25
    if not lost_positive:
        next(row for row in rows if row["question_key"] == "cfq_699675ceeaf65406")[
            "side_margins"
        ] = [0.5, 0.0]
    return {
        "unit_count": 25,
        "complete_units": 10,
        "positive_sides": 35,
        "cross_prefix_complete_units": 17,
        "complete_physical_pair_coverage": 5,
        "complete_units_by_family": {
            "book_support": 1,
            "mirror_lr": 2,
            "picture_support": 0,
        },
        "cross_prefix_complete_units_by_family": {
            "book_support": 1,
            "mirror_lr": 2,
            "picture_support": 0,
        },
        "units": rows,
    }


def _terminal_payload() -> dict[str, Any]:
    authorization = {
        "authorization_id": v47._AUTHORIZATION_ID,
        "authorized": True,
        "only_exact_action": "one_bounded_four_step_v47_book_support_continuation",
        "authorized_config": str(v47.V47_CONFIG),
        "authorized_trainer": str(v47.V47_TRAINER),
        "authorized_test": str(v47.V47_TEST),
        "authorized_output": str(v47.DEFAULT_OUTPUT),
        "explicit_terminal_sha256_cli_required": True,
        "implementation_integrity": {
            "config_sha256": _sha256(PROJECT_ROOT / v47.V47_CONFIG),
            "trainer_sha256": _sha256(PROJECT_ROOT / v47.V47_TRAINER),
            "test_sha256": _sha256(PROJECT_ROOT / v47.V47_TEST),
        },
        "source": {
            "v46_report_sha256": v47._V46_REPORT_SHA256,
            "base_checkpoint": str(v47._BASE_CHECKPOINT),
            "candidate_id": v47._CANDIDATE_ID,
            "candidate_full_tensor_state_sha256": v47._CANDIDATE_FULL_SHA256,
            "candidate_authorized_surface_sha256": (v47._CANDIDATE_AUTHORIZED_SHA256),
            "candidate_frozen_state_sha256": v47._CANDIDATE_FROZEN_SHA256,
        },
        "training": {
            "optimizer_steps": 4,
            "checkpoint_steps": [0, 2, 4],
            "target_question_keys": [v47._TARGET_QUESTION_KEY] * 4,
            "broad_question_ids": list(v47._BROAD_QUESTION_IDS),
            "fresh_adamw": True,
            "same_v45_objective": True,
            "update2_integrity_only": True,
            "update4_original_v45_final_gate": True,
        },
        "scope": {
            "train_only": True,
            "validation_access_authorized": False,
            "oracle_access_authorized": False,
            "final_test_access_authorized": False,
            "selector_execution_authorized": False,
            "runtime_promotion_authorized": False,
        },
    }
    return {
        "schema_version": 1,
        "artifact": "v46_v45_u4_lost_side_terminal_gate",
        "passed": True,
        "only_exact_successor_authorized": v47._AUTHORIZATION_ID,
        "conditional_successor_authorization": authorization,
    }


def test_exact_config_settings_source_schedule_and_gates_are_locked() -> None:
    config = load_config(CONFIG)
    settings = v47.v47_settings(config)
    contract = v47.v47_contract(config)
    assert _sha256(CONFIG) == v47._CONFIG_FILE_SHA256
    assert settings.optimizer_steps == 4
    assert settings.checkpoint_steps == (0, 2, 4)
    assert settings.scene_readout_learning_rate == 1.0e-5
    assert settings.query_learning_rate == 8.0e-6
    assert settings.retention_weight == 8.0
    assert contract["update2_gate"]["no_behavioral_count_or_lost_side_fail_stop"]
    assert contract["update4_gate"]["book_cross_prefix_complete_units_minimum"] == 1
    assert v47._V45_U4_CONSTRUCTION_FULL_SHA256 == (
        "468f493a746c6125f8ebc62d57ca8ae0419160f6e13ce903dd9f40c64aa772c2"
    )
    assert v47._CANDIDATE_FULL_SHA256 == (
        "1d5adc1fb0d7a895056b77d38c8a12aba95c9997ec8a94edf68673f9c58fb954"
    )


def test_contract_rejects_optimizer_schedule_candidate_and_gate_changes() -> None:
    config = copy.deepcopy(load_config(CONFIG))
    config["training"]["v47_book_continuation"]["optimizer_steps"] = 5
    with pytest.raises(ValueError, match="settings changed"):
        v47.v47_settings(config)

    config = copy.deepcopy(load_config(CONFIG))
    config["v47_book_continuation"]["reconstructed_candidate_alpha"] = 0.5
    with pytest.raises(ValueError, match="contract changed"):
        v47.v47_contract(config)

    config = copy.deepcopy(load_config(CONFIG))
    config["v47_book_continuation"]["update2_gate"][
        "no_behavioral_count_or_lost_side_fail_stop"
    ] = False
    with pytest.raises(ValueError, match="contract changed"):
        v47.v47_contract(config)


def test_schedule_is_exact_original_v45_steps_five_through_eight() -> None:
    config = load_config(CONFIG)
    loader = v41_loader_config(config)
    records, _audit = v47.load_v35_train_qa_records(loader)
    units = build_exact_question_pair_units(records)
    schedule, audit, broad = v47.build_v47_schedule(records, units, config=config)
    assert len(broad) == 48
    assert [row.optimizer_step for row in schedule] == [1, 2, 3, 4]
    assert [row.target_unit.pair_id for row in schedule] == ["pair_000015"] * 4
    assert [row.target_unit.question_key for row in schedule] == [v47._TARGET_QUESTION_KEY] * 4
    assert [row.broad_record.question_id for row in schedule] == list(v47._BROAD_QUESTION_IDS)
    assert [row["broad_row_number"] for row in audit["rows"]] == [13, 14, 15, 16]
    assert audit["fixed_nonadaptive"] is True


def test_v46_report_authenticates_unique_exact_candidate_without_authorizing_it() -> None:
    evidence = v47.require_v46_report()
    candidate = evidence["candidate"]
    assert evidence["sha256"] == v47._V46_REPORT_SHA256
    assert candidate["candidate_id"] == "g5_both_sign_alpha_1p0"
    assert candidate["full_tensor_state_sha256"] == v47._CANDIDATE_FULL_SHA256
    assert candidate["authorized_surface_state_sha256"] == (v47._CANDIDATE_AUTHORIZED_SHA256)
    assert candidate["candidate_authorized"] is False
    assert candidate["threshold_diagnostic"]["all_numeric_thresholds_met"] is True


def test_terminal_authorization_is_explicit_hash_bound_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "terminal.json"
    path.write_text(json.dumps(_terminal_payload(), sort_keys=True), encoding="utf-8")
    digest = _sha256(path)
    monkeypatch.setattr(v47, "DEFAULT_TERMINAL", path)
    terminal = v47.require_v46_terminal(digest)
    assert terminal["sha256"] == digest
    assert all(terminal["checks"].values())
    with pytest.raises(ValueError, match="explicit invocation"):
        v47.require_v46_terminal("0" * 64)

    changed = _terminal_payload()
    changed["conditional_successor_authorization"]["training"]["optimizer_steps"] = 5
    path.write_text(json.dumps(changed, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization changed"):
        v47.require_v46_terminal(_sha256(path))


def test_update2_gate_ignores_behavioral_counts_and_lost_side_but_not_integrity() -> None:
    metrics = _gate_metrics(lost_positive=False)
    metrics.update(
        {
            "complete_units": 0,
            "positive_sides": 0,
            "cross_prefix_complete_units": 0,
            "complete_physical_pair_coverage": 0,
        }
    )
    gate = v47.v47_update2_gate(
        pair_metrics=metrics,
        broad_nll=3.0,
        scene_changed=True,
        query_changed=True,
        frozen_exact=True,
        trust_rms=0.01,
    )
    assert gate["passed"] is True
    assert gate["complete_units"] == 0
    assert gate["retention_diagnostics"]["both_lost_sides_strictly_positive"] is False
    assert gate["behavioral_counts_are_diagnostic_only"] is True

    assert (
        v47.v47_update2_gate(
            pair_metrics=metrics,
            broad_nll=3.051,
            scene_changed=True,
            query_changed=True,
            frozen_exact=True,
            trust_rms=0.0,
        )["passed"]
        is False
    )
    assert (
        v47.v47_update2_gate(
            pair_metrics=metrics,
            broad_nll=3.0,
            scene_changed=False,
            query_changed=True,
            frozen_exact=True,
            trust_rms=0.0,
        )["passed"]
        is False
    )


def test_update4_gate_is_the_exact_strict_final_train_only_gate() -> None:
    metrics = _gate_metrics()
    greedy = {"complete_units": 5, "broad_exact_correct": 23, "broad_row_count": 48}
    gate = v47.v47_update4_gate(
        update2_gate={"passed": True},
        pair_metrics=metrics,
        broad_nll=2.9,
        greedy_metrics=greedy,
        scene_changed=True,
        query_changed=True,
        frozen_exact=True,
        trust_rms=0.002,
    )
    assert gate["passed"] is True
    assert gate["full_train_pair_unit_count"] == 25
    assert gate["full_broad_nll_row_count"] == 48
    assert gate["scene_readout_state_changed"] is True
    assert gate["query_state_changed"] is True

    failed = v47.v47_update4_gate(
        update2_gate={"passed": True},
        pair_metrics=_gate_metrics(lost_positive=False),
        broad_nll=2.9,
        greedy_metrics=greedy,
        scene_changed=True,
        query_changed=True,
        frozen_exact=True,
        trust_rms=0.001,
    )
    assert failed["both_lost_side_margins_remain_strictly_positive"] is False
    assert failed["passed"] is False


def test_candidate_hash_and_prefix_reference_precede_every_write_and_optimizer() -> None:
    source = Path(v47.__file__).read_text(encoding="utf-8")
    run = source[source.index("def _run_impl") : source.index("def run_v47")]
    hash_gate = run.index("reconstructed candidate hash attestation failed before write")
    prefix_reference = run.index("candidate_scene_tokens =")
    optimizer = run.index("optimizer = v45_optimizer(")
    output_write = run.index("output_path.mkdir(")
    assert hash_gate < prefix_reference < optimizer < output_write
    assert "load_optimizer" not in run
    assert "references=candidate_scene_tokens" in run
    assert "step in (2, 4)" in run
    assert "step == 2 and gate2 is not None" in run
    assert "step == 4 and gate4 is not None" in run


def test_cli_requires_explicit_v46_terminal_sha256(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["train_book_continuation_v47"])
    with pytest.raises(SystemExit) as raised:
        v47.main()
    assert raised.value.code == 2
