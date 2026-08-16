from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = runpy.run_path(str(ROOT / "scripts/build_current_report.py"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def summary() -> dict[str, Any]:
    return BUILDER["build_summary"]()


def test_current_summary_authenticates_v84_and_v84_1_exact_evidence(
    summary: dict[str, Any],
) -> None:
    evidence = summary["v84_strict_immutable_memory_bridge"]

    assert evidence["status"] == (
        "authenticated_v84_nll_wiring_pass_v84_1_two_scene_causal_overfit_passed_not_promoted"
    )
    assert evidence["evidence_authenticated"] is True
    assert evidence["runtime_promotion_authorized"] is False
    assert evidence["official_validation_measured"] is False
    assert evidence["held_out_generalization_measured"] is False
    assert evidence["development_behavior_scored"] is False
    assert evidence["oracle_loaded"] is False
    assert evidence["oracle_or_qa_files_opened_by_report_builder"] is False
    assert all(evidence["checks"].values())

    memory = evidence["strict_immutable_memory_contract"]
    assert memory["shape_per_scene"] == [1, 738, 1536]
    assert memory["all_memory_tokens_retained"] is True
    assert memory["compiled_before_question"] is True
    assert memory["memory_hash_invariant"] is True
    assert memory["question_derived_environmental_tokens"] == 0
    assert memory["question_conditioned_environmental_readout"] is False
    assert memory["question_dependent_retrieval"] is False
    assert memory["semantic_or_spatial_top_k_selection"] is False
    assert memory["control_tokens"] == 0
    assert memory["environmental_text_inputs"] == []

    v84 = evidence["v84_four_update_wiring"]
    assert v84["status"] == "nll_wiring_pass_greedy_pair_not_separated"
    assert v84["optimizer_updates"] == 4
    assert v84["initial_mean_correct_scene_nll"] == pytest.approx(5.284232020378113)
    assert v84["final_mean_correct_scene_nll"] == pytest.approx(3.1652339696884155)
    assert v84["both_rows_nll_improved"] is True
    assert v84["final_canonical_or_raw_greedy_predictions"] == [
        "under the table",
        "under the table",
    ]
    assert v84["final_greedy_pair_separated"] is False
    assert v84["final_greedy_exact_count"] == 0

    v84_1 = evidence["v84_1_two_scene_causal_overfit"]
    assert v84_1["status"] == "preregistered_two_scene_causal_overfit_passed"
    assert v84_1["optimizer_updates"] == 32
    assert v84_1["optimization_scene_count"] == 2
    assert v84_1["initial_mean_correct_scene_nll"] == pytest.approx(5.284232020378113)
    assert v84_1["final_mean_correct_scene_nll"] == pytest.approx(0.030677192844450474)
    assert [row["canonical_greedy_prediction"] for row in v84_1["final_rows"]] == [
        "on",
        "under",
    ]
    assert [row["correct_scene_nll"] for row in v84_1["final_rows"]] == pytest.approx(
        [0.03221847489476204, 0.02913591079413891]
    )
    assert [row["wrong_minus_correct_nll"] for row in v84_1["final_rows"]] == pytest.approx(
        [2.5907809548079967, 1.3780144769698381]
    )
    assert v84_1["both_final_greedy_exact"] is True
    assert v84_1["final_greedy_pair_separated"] is True
    assert v84_1["memory_hash_invariant"] is True
    assert v84_1["development_behavior_scored"] is False
    assert v84_1["official_validation_loaded"] is False
    assert v84_1["oracle_loaded"] is False
    assert v84_1["runtime_promotion_authorized"] is False

    for path, digest in BUILDER["V84_EVIDENCE_SHA256"].items():
        assert summary["source_artifacts"][path.as_posix()] == digest


def test_current_markdown_bounds_v84_and_v84_1_claims(
    summary: dict[str, Any],
) -> None:
    markdown = BUILDER["render_markdown"](summary)
    collapsed = " ".join(markdown.split())

    assert "V84 and V84.1 now provide authenticated causal-wiring" in collapsed
    assert "complete `[1, 738, 1536]` continuous memories" in collapsed
    assert "question-derived environmental tokens is exactly 0" in collapsed
    assert "original V84 four-update answer-NLL wiring smoke passed" in collapsed
    assert "did **not** separate the pair" in collapsed
    assert "both final greedy answers remained `under the table`" in collapsed
    assert "Mean correct-scene NLL fell from 5.284232 to 0.030677" in collapsed
    assert "`scene_000019` produced exact `on`" in collapsed
    assert "`scene_000020` produced exact `under`" in collapsed
    assert "preregistered **two-scene causal overfit/wiring result**" in collapsed
    assert "not held-out scene understanding" in collapsed
    assert "No development behavior, sealed historical behavior set" in collapsed
    assert "Runtime promotion is not authorized for either checkpoint" in collapsed
    assert "no held-scene generalization is claimed" in collapsed


def test_v84_inspector_fails_closed_on_checkpoint_digest_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER["_inspect_v84_strict_immutable_memory_bridge"]
    globals_ = inspector.__globals__
    original = ROOT / BUILDER["V84_1_CHECKPOINT_WEIGHTS"]
    tampered = tmp_path / "bridge.safetensors"
    tampered.write_bytes(original.read_bytes() + b"tamper")
    evidence = {
        (tampered if path == BUILDER["V84_1_CHECKPOINT_WEIGHTS"] else path): digest
        for path, digest in BUILDER["V84_EVIDENCE_SHA256"].items()
    }
    monkeypatch.setitem(globals_, "V84_1_CHECKPOINT_WEIGHTS", tampered)
    monkeypatch.setitem(globals_, "V84_EVIDENCE_SHA256", evidence)

    result = inspector()

    assert result["status"] == "artifact_present_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert "evidence digest differs" in result["measurement_evidence_error"]


def test_v84_inspector_rejects_rehashed_false_parent_greedy_separation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER["_inspect_v84_strict_immutable_memory_bridge"]
    globals_ = inspector.__globals__
    original = ROOT / BUILDER["V84_WIRING_REPORT"]
    payload = json.loads(original.read_text(encoding="utf-8"))
    payload["final_rows"][0]["greedy_prediction"] = "on"
    payload["final_rows"][0]["greedy_normalized_exact"] = True
    tampered = tmp_path / original.name
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    evidence = {
        (tampered if path == BUILDER["V84_WIRING_REPORT"] else path): (
            _sha256(tampered) if path == BUILDER["V84_WIRING_REPORT"] else digest
        )
        for path, digest in BUILDER["V84_EVIDENCE_SHA256"].items()
    }
    monkeypatch.setitem(globals_, "V84_WIRING_REPORT", tampered)
    monkeypatch.setitem(globals_, "V84_EVIDENCE_SHA256", evidence)

    result = inspector()

    assert result["status"] == "artifact_present_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert "v84_four_update_wiring" in result["measurement_evidence_error"]


def test_v84_inspector_rejects_rehashed_followup_scope_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER["_inspect_v84_strict_immutable_memory_bridge"]
    globals_ = inspector.__globals__
    original = ROOT / BUILDER["V84_1_WIRING_REPORT"]
    payload = json.loads(original.read_text(encoding="utf-8"))
    payload["development_behavior_scored"] = True
    tampered = tmp_path / original.name
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    evidence = {
        (tampered if path == BUILDER["V84_1_WIRING_REPORT"] else path): (
            _sha256(tampered) if path == BUILDER["V84_1_WIRING_REPORT"] else digest
        )
        for path, digest in BUILDER["V84_EVIDENCE_SHA256"].items()
    }
    monkeypatch.setitem(globals_, "V84_1_WIRING_REPORT", tampered)
    monkeypatch.setitem(globals_, "V84_EVIDENCE_SHA256", evidence)

    result = inspector()

    assert result["status"] == "artifact_present_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert "v84_1_fixed_update_wiring" in result["measurement_evidence_error"]
