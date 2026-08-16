from __future__ import annotations

import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = runpy.run_path(str(ROOT / "scripts/build_current_report.py"))


@pytest.fixture(scope="module")
def summary() -> dict[str, object]:
    return BUILDER["build_summary"]()


def test_v95_training_and_negative_gate_are_exactly_authenticated(
    summary: dict[str, object],
) -> None:
    result = summary["v95_strict_causal_successor"]
    assert isinstance(result, dict)
    assert result["status"] == "measured_preregistered_gate_not_passed"
    assert result["evidence_authenticated"] is True
    assert result["known_development_gate_passed"] is False
    assert result["runtime_promotion_authorized"] is False
    assert result["deferred_final_unlock_eligible"] is False
    assert result["deferred_final_remains_locked"] is True
    assert result["default_runtime"] == "v89"
    assert result["held_out_or_generalization_claim"] is False
    assert result["official_or_final_evaluation_claim"] is False

    training = result["training"]
    assert training["elapsed_seconds"] == pytest.approx(13234.774727290962)
    assert training["elapsed_seconds_rounded"] == pytest.approx(13234.8)
    assert training["fresh_parameter_count"] == 143360
    assert training["scene_count"] == 40
    assert training["row_count"] == 960
    assert training["epochs"] == 4
    assert training["known_development_labels_loaded"] is False
    assert training["oracle_loaded"] is False

    development = result["known_development"]
    assert development["arms"] == {
        "primary": {
            "correct": 167,
            "total": 216,
            "accuracy": pytest.approx(167 / 216),
        },
        "zero_payload": {
            "correct": 36,
            "total": 216,
            "accuracy": pytest.approx(36 / 216),
        },
        "full_interior_permutation": {
            "correct": 127,
            "total": 216,
            "accuracy": pytest.approx(127 / 216),
        },
        "paired_wrong_scene": {
            "correct": 164,
            "total": 216,
            "accuracy": pytest.approx(164 / 216),
        },
    }
    assert development["nll"]["zero_payload_mean_nll_gap"] == pytest.approx(2.296439275215894)
    assert development["nll"]["full_interior_permutation_mean_nll_gap"] == pytest.approx(
        0.616878268558505
    )
    assert development["nll"]["mean_wrong_minus_primary_nll"] == pytest.approx(0.05409157463280415)
    assert development["nll"]["mean_changed_wrong_minus_primary_nll"] == pytest.approx(
        0.48682417169523734
    )
    assert development["counterfactual"] == {
        "correct_sides": 13,
        "side_count": 24,
        "complete_units": 1,
        "unit_count": 12,
        "prediction_changed_units": 2,
    }


def test_v95_hash_chain_and_report_boundaries_are_serialized(
    summary: dict[str, object],
) -> None:
    result = summary["v95_strict_causal_successor"]
    hashes = result["hashes"]
    assert hashes["config_sha256"] == (
        "9115c36b417d03bec935257b42e30597170d5acbf6c4683b5c021a8e4d9bbea2"
    )
    assert hashes["preregistration_sha256"] == (
        "d60df9a9a04843fefbb46e8f2845613e5d887dc4f06665fe015c0aafcc7cf03d"
    )
    assert hashes["cpu_preflight_sha256"] == (
        "5ac211be59df4083588a776f4eb7d5a1b8ea38c9d635284b6452e45a5cb549ad"
    )
    assert hashes["candidate_state_sha256"] == (
        "53404c733586ebd25caa440f822a4d4af6cc3dbb71bf4f6b6f94af23f3a2492a"
    )
    assert hashes["candidate_fingerprint_sha256"] == (
        "3c499d0f519766dea3185f4342fa6738776101cf5882cb77f4e43985586c2c1b"
    )
    assert hashes["artifact_sha256"] == {
        path.as_posix(): digest for path, digest in BUILDER["V95_REPORT_EVIDENCE_SHA256"].items()
    }
    assert summary["claim_scope"]["v95_held_out_generalization_measured"] is False
    assert summary["claim_scope"]["v95_runtime_promoted"] is False
    assert summary["claim_scope"]["v95_deferred_final_unlock_eligible"] is False
    assert summary["claim_scope"]["operator_default_remains_v89"] is True


def test_v95_markdown_states_result_without_promotion_or_final_claim(
    summary: dict[str, object],
) -> None:
    markdown = BUILDER["render_markdown"](summary)
    collapsed = " ".join(markdown.split())

    assert "took 13234.8 s on MPS" in collapsed
    assert "143,360-parameter unmerged bridge" in collapsed
    assert "primary at 167/216 (77.31%)" in collapsed
    assert "zero payload at 36/216" in collapsed
    assert "full interior-token permutation at 127/216" in collapsed
    assert "paired wrong-scene memory at 164/216" in collapsed
    assert "+2.296439 for zero payload" in collapsed
    assert "+0.616878 for permutation" in collapsed
    assert "+0.054092 for paired wrong scenes" in collapsed
    assert "changed-side paired-wrong gap was +0.486824" in collapsed
    assert "13/24 correct sides, 1/12 complete units, and 2/12" in collapsed
    assert "The gate therefore **failed**" in markdown
    assert "V95 was not promoted" in collapsed
    assert "deferred-final materialization remains locked" in collapsed
    assert "V89 remains the operator-default runtime" in collapsed
    assert "not held-out-final, official, generalization, or final-acceptance" in collapsed


def test_v95_report_inspector_fails_closed_on_digest_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER["_inspect_v95_strict_causal_successor"]
    hashes = dict(inspector.__globals__["V95_REPORT_EVIDENCE_SHA256"])
    first_path = next(iter(hashes))
    hashes[first_path] = "0" * 64
    monkeypatch.setitem(inspector.__globals__, "V95_REPORT_EVIDENCE_SHA256", hashes)

    rejected = inspector()
    assert rejected["evidence_authenticated"] is False
    assert rejected["known_development_gate_passed"] is False
    assert rejected["runtime_promotion_authorized"] is False
    assert rejected["deferred_final_unlock_eligible"] is False
    assert rejected["deferred_final_remains_locked"] is True
    assert rejected["default_runtime"] == "v89"
    assert rejected["held_out_or_generalization_claim"] is False
    assert rejected["official_or_final_evaluation_claim"] is False
