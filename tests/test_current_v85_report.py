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


def test_v85_inspector_authenticates_development_and_denies_promotion() -> None:
    result = BUILDER["_inspect_v85_strict_multiscene_candidate"]()

    assert result["status"] == (
        "authenticated_development_gate_passed_runtime_packaging_equivalent_"
        "scene1_behavior_failed_not_promoted"
    )
    assert result["evidence_authenticated"] is True
    assert result["development_only"] is True
    assert result["official_validation_measured"] is False
    assert result["held_out_final_measured"] is False
    assert result["runtime_promotion_authorized"] is False
    assert result["default_runtime_changed"] is False
    assert result["oracle_or_qa_files_opened_by_report_builder"] is False

    development = result["scene_disjoint_development"]
    assert development["scene_count"] == 16
    assert development["question_count"] == 384
    assert development["canonical_exact"] == {
        "accuracy": 214 / 384,
        "correct": 214,
        "total": 384,
    }
    assert development["answer_frequency_majority_baseline"] == {
        "accuracy": 62 / 384,
        "correct": 62,
        "total": 384,
    }
    assert development["spatial_relation_accuracy"] == pytest.approx(42 / 80)
    assert development["prediction_changing_units"] == 8
    assert development["complete_changed_units"] == 4
    assert development["changed_unit_count"] == 26
    assert development["automatic_runtime_promotion"] is False
    assert development["official_validation"] is False


def test_v85_runtime_correction_preserves_failed_smoke_and_invariance() -> None:
    result = BUILDER["_inspect_v85_strict_multiscene_candidate"]()
    runtime = result["strict_scene1_runtime_smoke"]

    assert runtime["observed_answers"] == ["no", "blue", "left"]
    assert runtime["original_predeclared_expected_answers"] == ["yes", "red", "right"]
    assert runtime["original_artifact_score"] == {"correct": 0, "total": 3}
    assert runtime["corrected_expected_answers"] == ["yes", "red", "left"]
    assert runtime["corrected_passes"] == [False, False, True]
    assert runtime["corrected_score"] == {
        "correct": 1,
        "total": 3,
        "accuracy": 1 / 3,
    }
    assert "reversed subject and reference" in runtime["original_artifact_evaluator_error"]
    assert runtime["prefix_hash"] == (
        "a428f5147c815839ae7315a0adab952ab210814fb21dcdc5bf13b167f28a6e37"
    )
    assert runtime["same_prefix_every_question"] is True
    assert runtime["same_total_environment_conditioned_input_every_question"] is True
    assert runtime["question_derived_environmental_tokens"] == 0
    assert runtime["loaded_file_count"] == 5208
    assert runtime["forbidden_access_count"] == 0
    assert runtime["oracle_physically_unavailable"] is True
    assert runtime["behavior_gate_passed"] is False
    assert runtime["promotion_authorized"] is False


def test_v85_scene39_replay_proves_packaging_equivalence_not_accuracy() -> None:
    result = BUILDER["_inspect_v85_strict_multiscene_candidate"]()
    equivalence = result["scene39_packaging_equivalence"]

    assert equivalence["correct"] == equivalence["total"] == 24
    assert equivalence["packaging_numerically_equivalent"] is True
    assert equivalence["prefix_and_total_environment_input_invariant"] is True
    assert equivalence["forbidden_access_count"] == 0
    assert equivalence["oracle_physically_unavailable"] is True
    assert equivalence["selection_or_promotion_changed"] is False
    caveat = equivalence["caveat"].lower()
    assert "sealed development predictions only" in caveat
    assert "not a ground-truth rescore" in caveat
    assert "not" in caveat and "promotion evidence" in caveat


def test_v85_figure_and_every_evidence_file_are_hard_hash_bound(
    summary: dict[str, Any],
) -> None:
    result = summary["v85_strict_multiscene_candidate"]
    figure = result["development_accuracy_figure"]

    assert figure == {
        "status": "authenticated_posthoc_development_only_visualization",
        "path": "reports/gemma4/figures/v85_development_accuracy_by_type.png",
        "sha256": "4fee17cd5581663d5b01b0bddcfa08cf5c89d067d36dda7c83b330fa4c90b95f",
        "summary_path": "reports/gemma4/examples/v85_development_accuracy_by_type.json",
        "summary_sha256": ("89e96c572eb1b35a0cd9838c49012caf2f50e323e9c4e63822c4dfffcc5e5734"),
        "source_report_sha256": (
            "202134d8900e105d63f23d1cc1d19d68a882c4464382b7a63b7aa007f2714828"
        ),
        "new_inference": False,
        "official_validation": False,
        "runtime_promotion_evidence": False,
    }
    for path, digest in BUILDER["V85_EVIDENCE_SHA256"].items():
        assert _sha256(ROOT / path) == digest
        assert summary["source_artifacts"][path.as_posix()] == digest
    assert not any("data/oracle" in path for path in result["measurement_evidence_paths"])
    assert not any("data/qa" in path for path in result["measurement_evidence_paths"])


def test_v85_markdown_is_explicit_about_scope_and_runtime_failure(
    summary: dict[str, Any],
) -> None:
    markdown = BUILDER["render_markdown"](summary)
    collapsed = " ".join(markdown.split())

    assert "214/ 384 (55.73%)" in collapsed
    assert "62/ 384 (16.15%)" in collapsed
    assert "pair- and scene-disjoint **development** split" in collapsed
    assert "24/24 sealed development predictions exactly" in collapsed
    assert "not** a ground-truth rescore" in collapsed
    assert "corrected score as 1/ 3" in collapsed
    assert "V85 is **not promoted**" in collapsed
    assert "later accepted strict V89 release is the current operator default" in collapsed
    assert "not official validation" in collapsed.lower()
    assert "4fee17cd5581663d5b01b0bddcfa08cf5c89d067d36dda7c83b330fa4c90b95f" in markdown


def test_v85_inspector_fails_closed_on_raw_digest_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER["_inspect_v85_strict_multiscene_candidate"]
    pinned = inspector.__globals__["V85_EVIDENCE_SHA256"]
    path = BUILDER["V85_RUNTIME_SMOKE"]
    monkeypatch.setitem(pinned, path, "0" * 64)

    result = inspector()

    assert result["status"] == "artifact_present_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert result["runtime_promotion_authorized"] is False
    assert "V85 evidence digest differs" in result["measurement_evidence_error"]


def test_v85_inspector_fails_closed_on_rehashed_runtime_promotion_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER["_inspect_v85_strict_multiscene_candidate"]
    globals_ = inspector.__globals__
    original = ROOT / BUILDER["V85_RUNTIME_SMOKE"]
    payload = json.loads(original.read_text(encoding="utf-8"))
    payload["promotion_authorized"] = True
    tampered = tmp_path / original.name
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    evidence = dict(BUILDER["V85_EVIDENCE_SHA256"])
    del evidence[BUILDER["V85_RUNTIME_SMOKE"]]
    evidence[tampered] = _sha256(tampered)
    monkeypatch.setitem(globals_, "V85_RUNTIME_SMOKE", tampered)
    monkeypatch.setitem(globals_, "V85_EVIDENCE_SHA256", evidence)

    result = inspector()

    assert result["status"] == "artifact_present_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert result["runtime_promotion_authorized"] is False
    assert "definitive_runtime_smoke" in result["measurement_evidence_error"]


def test_v85_inspector_fails_closed_on_rehashed_equivalence_scope_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER["_inspect_v85_strict_multiscene_candidate"]
    globals_ = inspector.__globals__
    original = ROOT / BUILDER["V85_RUNTIME_EQUIVALENCE"]
    payload = json.loads(original.read_text(encoding="utf-8"))
    payload["selection_or_promotion_changed"] = True
    tampered = tmp_path / original.name
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    evidence = dict(BUILDER["V85_EVIDENCE_SHA256"])
    del evidence[BUILDER["V85_RUNTIME_EQUIVALENCE"]]
    evidence[tampered] = _sha256(tampered)
    monkeypatch.setitem(globals_, "V85_RUNTIME_EQUIVALENCE", tampered)
    monkeypatch.setitem(globals_, "V85_EVIDENCE_SHA256", evidence)

    result = inspector()

    assert result["status"] == "artifact_present_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert result["runtime_promotion_authorized"] is False
    assert "scene39_packaging_equivalence" in result["measurement_evidence_error"]
