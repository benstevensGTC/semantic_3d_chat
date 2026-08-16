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


def test_v88_terminal_negative_is_authenticated_and_not_promoted() -> None:
    result = BUILDER["_inspect_v88_scene1_augmented_terminal"]()

    assert result["status"] == (
        "authenticated_augmented_single_scene_overall_gate_failed_107_of_138_not_promoted"
    )
    assert result["evidence_authenticated"] is True
    assert result["terminal_result"] is True
    assert result["post_v87_training_set_development"] is True
    assert result["development_known_smoke_trained"] is True
    assert result["held_out_smoke"] is False
    assert result["training_measured"] is True
    assert result["evaluation_measured"] is True
    assert result["runtime_smoke_measured"] is False
    assert result["runtime_smoke_blocked_by_model_gate"] is True
    assert result["model_acceptance_gate_passed"] is False
    assert result["held_out_generalization_measured"] is False
    assert result["official_validation_measured"] is False
    assert result["runtime_promotion_authorized"] is False
    assert result["default_runtime_changed"] is False
    assert result["training_result"]["optimizer_updates"] == 188
    assert result["training_result"]["micro_rows_consumed"] == 1128
    assert result["training_result"]["causal_margin_rows_consumed"] == 20
    assert result["training_result"]["protected_read_count"] == 0
    assert result["single_scene_evaluation"]["canonical_exact"] == {
        "accuracy": 107 / 138,
        "correct": 107,
        "total": 138,
    }
    assert result["single_scene_evaluation"]["failed_model_gates"] == [
        "all_scene1_canonical_accuracy_at_least_0_80"
    ]
    assert result["single_scene_evaluation"]["overall_acceptance_shortfall_correct"] == 4
    assert result["generic_scene1_smoke"]["correct"] == 3
    assert result["generic_scene1_smoke"]["observed_answers"] == [
        "yes",
        "red",
        "left",
    ]
    assert result["generic_scene1_smoke"]["development_known_and_trained"] is True
    assert result["generic_scene1_smoke"]["held_out"] is False
    assert result["generic_scene1_smoke"]["runtime_oracle_unavailable_audit_run"] is False
    assert result["zero_payload_causal_control"]["mean_zero_minus_correct_nll"] == (
        pytest.approx(1.8954026500384014)
    )
    assert result["zero_payload_causal_control"]["canonical_prediction_changes"] == 2
    assert result["strict_input_invariance"]["shape"] == [1, 738, 1536]
    assert result["strict_input_invariance"]["prefix_hash_invariant"] is True
    assert result["strict_input_invariance"]["question_derived_environmental_tokens"] == 0
    assert result["evaluation_isolation"]["loaded_file_count"] == 85
    assert result["evaluation_isolation"]["protected_read_count"] == 0
    assert result["evaluation_isolation"]["runtime_smoke_blocked"] is True
    assert result["comparison_to_v87"]["correct_answer_gain"] == 4
    assert result["comparison_to_v87"]["smoke_was_trained_in_v88"] is True
    assert all(result["checks"].values())


def test_v88_sealed_evidence_enters_current_source_artifacts(
    summary: dict[str, Any],
) -> None:
    result = summary["v88_scene1_augmented"]
    assert result["evidence_authenticated"] is True
    for evidence_path, evidence_digest in BUILDER["V88_SEALED_EVIDENCE_SHA256"].items():
        if evidence_digest is None:
            assert evidence_path.as_posix() not in summary["source_artifacts"]
            assert not (ROOT / evidence_path).exists()
        else:
            assert _sha256(ROOT / evidence_path) == evidence_digest
            assert summary["source_artifacts"][evidence_path.as_posix()] == evidence_digest


def test_v88_rejects_trained_smoke_relabelled_as_held_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER["_inspect_v88_scene1_augmented_terminal"]
    globals_ = inspector.__globals__
    source_path = BUILDER["V88_EVALUATION"]
    payload = json.loads((ROOT / source_path).read_text(encoding="utf-8"))
    payload["metrics"]["generic_smoke"]["held_out"] = True
    tampered = tmp_path / "v88_evaluation.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    hashes = dict(BUILDER["V88_SEALED_EVIDENCE_SHA256"])
    hashes[tampered] = _sha256(tampered)
    hashes.pop(source_path)
    monkeypatch.setitem(globals_, "V88_EVALUATION", tampered)
    monkeypatch.setitem(globals_, "V88_SEALED_EVIDENCE_SHA256", hashes)

    result = inspector()

    assert result["status"] == "artifact_present_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert result["runtime_promotion_authorized"] is False
    assert "single_scene_evaluation" in result["measurement_evidence_error"]


def test_v88_markdown_reports_measured_negative_without_smoke_overclaim(
    summary: dict[str, Any],
) -> None:
    markdown = BUILDER["render_markdown"](summary)
    collapsed = " ".join(markdown.split())

    assert "V88 followed with a preregistered, retention-aware" in collapsed
    assert "107/ 138 (77.54%) canonical exact" in collapsed
    assert "4 more correct answers" in collapsed
    assert "4 correct answers short of the locked 80% overall gate" in collapsed
    assert "Attribute accuracy improved to 11/18 (61.11%)" in collapsed
    assert "`yes`, `red`, and the physically correct `left`" in collapsed
    assert "explicitly represented in the V88 training schedule" in collapsed
    assert "not held-out evidence or a generalization result" in collapsed
    assert "Every preregistered model gate except overall accuracy passed" in collapsed
    assert "no V88 runtime package was built or promoted" in collapsed
    assert "No held-out or official-validation claim is made" in collapsed
