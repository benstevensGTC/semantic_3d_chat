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


def test_v87_terminal_negative_is_authenticated_and_not_promoted() -> None:
    result = BUILDER["_inspect_v87_scene1_balanced_terminal"]()

    assert result["status"] == (
        "authenticated_balanced_single_scene_model_gates_failed_103_of_138_not_promoted"
    )
    assert result["evidence_authenticated"] is True
    assert result["terminal_result"] is True
    assert result["training_measured"] is True
    assert result["evaluation_measured"] is True
    assert result["runtime_smoke_measured"] is False
    assert result["runtime_smoke_blocked_by_model_gates"] is True
    assert result["model_acceptance_gate_passed"] is False
    assert result["held_out_generalization_measured"] is False
    assert result["runtime_promotion_authorized"] is False
    assert result["default_runtime_changed"] is False
    assert result["training_result"]["optimizer_updates"] == 184
    assert result["training_result"]["micro_rows_consumed"] == 1104
    assert result["training_result"]["protected_read_count"] == 0
    assert result["single_scene_evaluation"]["canonical_exact"] == {
        "accuracy": 103 / 138,
        "correct": 103,
        "total": 138,
    }
    assert result["single_scene_evaluation"]["failed_model_gates"] == [
        "all_scene1_canonical_accuracy_at_least_0_80",
        "attribute_accuracy_at_least_0_50",
        "generic_live_smoke_exactly_3_of_3",
    ]
    assert result["generic_scene1_smoke"]["correct"] == 0
    assert result["generic_scene1_smoke"]["observed_answers"] == [
        "no",
        "wood",
        "right",
    ]
    assert result["generic_scene1_smoke"]["expected_answers"] == [
        "yes",
        "red",
        "left",
    ]
    assert result["generic_scene1_smoke"]["all_observed_answers_incorrect"] is True
    assert result["zero_payload_causal_control"]["mean_zero_minus_correct_nll"] == (
        pytest.approx(1.6181515057881672)
    )
    assert result["zero_payload_causal_control"]["canonical_prediction_changes"] == 2
    assert result["strict_input_invariance"]["prefix_hash_invariant"] is True
    assert result["strict_input_invariance"]["question_derived_environmental_tokens"] == 0
    assert result["evaluation_isolation"]["loaded_file_count"] == 83
    assert result["evaluation_isolation"]["protected_read_count"] == 0
    assert result["comparison_to_v86"]["correct_answer_gain"] == 17
    assert result["comparison_to_v86"]["smoke_regressed"] is True
    assert all(result["checks"].values())


def test_v87_sealed_evidence_enters_current_source_artifacts(
    summary: dict[str, Any],
) -> None:
    result = summary["v87_scene1_balanced"]
    assert result["evidence_authenticated"] is True
    for evidence_path, evidence_digest in BUILDER["V87_SEALED_EVIDENCE_SHA256"].items():
        if evidence_digest is None:
            assert evidence_path.as_posix() not in summary["source_artifacts"]
            assert not (ROOT / evidence_path).exists()
        else:
            assert _sha256(ROOT / evidence_path) == evidence_digest
            assert summary["source_artifacts"][evidence_path.as_posix()] == evidence_digest


def test_v87_rejects_semantically_tampered_evaluation_even_if_rehashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER["_inspect_v87_scene1_balanced_terminal"]
    globals_ = inspector.__globals__
    source = ROOT / BUILDER["V87_EVALUATION"]
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["metrics"]["generic_smoke"]["records"][2]["exact_correct"] = True
    tampered = tmp_path / "v87_evaluation.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    hashes = dict(BUILDER["V87_SEALED_EVIDENCE_SHA256"])
    hashes[BUILDER["V87_EVALUATION"]] = _sha256(tampered)
    monkeypatch.setitem(globals_, "V87_EVALUATION", tampered)
    hashes[tampered] = hashes.pop(source.relative_to(ROOT))
    monkeypatch.setitem(globals_, "V87_SEALED_EVIDENCE_SHA256", hashes)

    result = inspector()

    assert result["status"] == "artifact_present_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert result["runtime_promotion_authorized"] is False
    assert "single_scene_evaluation" in result["measurement_evidence_error"]


def test_v87_markdown_reports_improvement_and_all_failed_gates(
    summary: dict[str, Any],
) -> None:
    markdown = BUILDER["render_markdown"](summary)
    collapsed = " ".join(markdown.split())

    assert "V87 then tested a preregistered class-balanced successor" in collapsed
    assert "103/ 138 (74.64%) canonical exact" in collapsed
    assert "17 more correct answers" in collapsed
    assert "Attribute accuracy remained only 7/18 (38.89%)" in collapsed
    assert "generic smoke regressed from V86's 3/3 to 0/3" in collapsed
    assert "`right`—were **all incorrect**" in collapsed
    assert "`right` is not reported anywhere as a success" in collapsed
    assert "no V87 runtime package was built or promoted" in collapsed
    assert "not held-out generalization or official validation" in collapsed
