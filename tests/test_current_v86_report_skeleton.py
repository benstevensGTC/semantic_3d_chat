from __future__ import annotations

import hashlib
import runpy
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
BUILDER = runpy.run_path(str(ROOT / "scripts/build_current_report.py"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def summary() -> dict[str, Any]:
    return BUILDER["build_summary"]()


def test_v86_terminal_negative_is_authenticated_and_not_promoted() -> None:
    result = BUILDER["_inspect_v86_scene1_strict_demo_skeleton"]()

    assert result["status"] == (
        "authenticated_single_scene_overfit_model_gate_failed_86_of_138_not_promoted"
    )
    assert result["design_contract_authenticated"] is True
    assert result["preregistration_authenticated"] is True
    assert result["cpu_preflight_authenticated"] is True
    assert result["result_evidence_authenticated"] is True
    assert result["evidence_authenticated"] is True
    assert result["skeleton_ready"] is True
    assert result["draft_status"] == "preregistered_before_full_model_load"
    assert result["scene_id"] == "scene_000001"
    assert result["strict_input_contract"]["shape"] == [1, 738, 1536]
    assert result["strict_input_contract"]["compiled_before_question"] is True
    assert result["strict_input_contract"]["question_dependent_retrieval"] is False
    assert result["strict_input_contract"]["control_tokens"] == 0
    assert result["strict_input_contract"]["environmental_text_inputs"] == []
    assert result["planned_gates"]["live_smoke_expected_answers"] == [
        "yes",
        "red",
        "left",
    ]
    assert result["training_measured"] is True
    assert result["evaluation_measured"] is True
    assert result["runtime_smoke_measured"] is False
    assert result["runtime_smoke_blocked_by_model_gate"] is True
    assert result["terminal_result"] is True
    assert result["model_acceptance_gate_passed"] is False
    assert result["held_out_generalization_measured"] is False
    assert result["official_validation_measured"] is False
    assert result["runtime_promotion_authorized"] is False
    assert result["default_runtime_changed"] is False
    assert result["single_scene_overfit_only"] is True
    assert result["training_result"]["optimizer_updates"] == 92
    assert result["training_result"]["micro_rows_consumed"] == 552
    assert result["training_result"]["protected_read_count"] == 0
    assert result["single_scene_evaluation"]["canonical_exact"] == {
        "accuracy": 86 / 138,
        "correct": 86,
        "total": 138,
    }
    assert result["single_scene_evaluation"]["acceptance_threshold"] == 0.8
    assert result["generic_scene1_smoke"]["observed_answers"] == [
        "yes",
        "red",
        "left",
    ]
    assert result["generic_scene1_smoke"]["runtime_oracle_unavailable_audit_run"] is False
    assert result["zero_payload_causal_control"]["mean_zero_minus_correct_nll"] == (
        pytest.approx(1.4776708508531253)
    )
    assert result["zero_payload_causal_control"]["canonical_prediction_changes"] == 2
    assert result["strict_input_invariance"]["prefix_hash_invariant"] is True
    assert result["strict_input_invariance"]["question_derived_environmental_tokens"] == 0
    assert result["evaluation_isolation"]["loaded_file_count"] == 81
    assert result["evaluation_isolation"]["protected_read_count"] == 0
    assert result["evaluation_isolation"]["runtime_smoke_blocked"] is True
    assert all(result["checks"].values())
    assert result["measurement_evidence_paths"]
    assert set(result["pending_evidence_hash_paths"]) == {
        BUILDER["V86_RUNTIME_SMOKE"].as_posix()
    }


def test_v86_only_sealed_design_evidence_enters_current_source_artifacts(
    summary: dict[str, Any],
) -> None:
    path = BUILDER["V86_DRAFT_CONFIG"]
    digest = BUILDER["V86_DRAFT_CONFIG_SHA256"]

    assert _sha256(ROOT / path) == digest
    assert summary["source_artifacts"][path.as_posix()] == digest
    result = summary["v86_scene1_strict_demo"]
    assert result["evidence_authenticated"] is True
    for evidence_path, evidence_digest in BUILDER["V86_SEALED_EVIDENCE_SHA256"].items():
        if evidence_digest is None:
            assert evidence_path.as_posix() not in summary["source_artifacts"]
        else:
            assert summary["source_artifacts"][evidence_path.as_posix()] == evidence_digest


def test_v86_semantic_change_fails_even_when_tampered_file_is_rehashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER["_inspect_v86_scene1_strict_demo_skeleton"]
    globals_ = inspector.__globals__
    payload = yaml.safe_load((ROOT / BUILDER["V86_DRAFT_CONFIG"]).read_text())
    payload["v86"]["gates"]["live_smoke_questions"][2]["expected"] = "right"
    tampered = tmp_path / "v86_tampered.yaml"
    tampered.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    monkeypatch.setitem(globals_, "V86_DRAFT_CONFIG", tampered)
    monkeypatch.setitem(globals_, "V86_DRAFT_CONFIG_SHA256", _sha256(tampered))

    result = inspector()

    assert result["status"] == "draft_contract_authentication_failed"
    assert result["design_contract_authenticated"] is False
    assert result["evidence_authenticated"] is False
    assert result["runtime_promotion_authorized"] is False
    assert "semantic contract differs" in result["draft_contract_error"]


def test_v86_fake_sealed_hashes_cannot_authenticate_result_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER["_inspect_v86_scene1_strict_demo_skeleton"]
    globals_ = inspector.__globals__
    fake_hashes = {path: "0" * 64 for path in BUILDER["V86_SEALED_EVIDENCE_SHA256"]}
    monkeypatch.setitem(globals_, "V86_SEALED_EVIDENCE_SHA256", fake_hashes)

    result = inspector()

    assert result["status"] == "draft_contract_authentication_failed"
    assert result["design_contract_authenticated"] is False
    assert result["evidence_authenticated"] is False
    assert result["training_measured"] is False
    assert result["runtime_promotion_authorized"] is False
    assert "V86 sealed evidence differs" in result["draft_contract_error"]


def test_v86_markdown_reports_terminal_negative_and_causal_positive(
    summary: dict[str, Any],
) -> None:
    markdown = BUILDER["render_markdown"](summary)
    collapsed = " ".join(markdown.split())

    assert "V86 completed the preregistered strict single-scene" in collapsed
    assert "terminal negative for its locked acceptance gate" in collapsed
    assert "86/ 138 (62.32%) canonical exact" in collapsed
    assert "below the locked 80% minimum" in collapsed
    assert "answered all 3/3 predeclared questions correctly" in collapsed
    assert "positive 1.477671 gap" in collapsed
    assert "blocked the independent oracle-unavailable runtime smoke" in collapsed
    assert "no V86 runtime package was promoted" in collapsed
    assert "V75 remains the default" in collapsed
