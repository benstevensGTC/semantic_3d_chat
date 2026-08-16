from __future__ import annotations

import hashlib
import runpy
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from semantic_3d_chat.evaluation import v89_reporting as reporting

ROOT = Path(__file__).resolve().parents[1]
BUILDER = runpy.run_path(str(ROOT / "scripts/build_current_report.py"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _passing_evaluation() -> dict[str, Any]:
    return {
        "artifact": "gemma4_v89_scene1_retention_evaluation_v1",
        "schema_version": 89,
        "status": "model_gates_pass_separate_runtime_packaging_required",
        "metrics": {
            "model_acceptance_gate_passed": True,
            "model_acceptance_gates": {name: True for name in reporting._REQUIRED_MODEL_GATES},
            "canonical_type_specific": {
                "accuracy": 122 / 138,
                "correct": 122,
                "total": 138,
            },
            "generic_smoke": {
                "correct": 3,
                "total": 3,
                "development_known_and_trained": True,
                "held_out": False,
            },
        },
        "development_known_smoke_trained": True,
        "held_out_smoke_claim": False,
        "held_out_generalization_claim": False,
        "parent_v85_v86_v87_v88_mutated": False,
        "separate_runtime_packaging_authorized": True,
        "automatic_runtime_promotion": False,
        "runtime_promotion_authorized": False,
        "oracle_loaded": False,
    }


def _passing_runtime_smoke() -> dict[str, Any]:
    return {
        "artifact": "gemma4_v89_strict_runtime_smoke_v1",
        "schema_version": 89,
        "gates": {name: True for name in reporting._REQUIRED_RUNTIME_GATES},
        "passed": True,
        "promotion_authorized": True,
        "expected_behavior_not_loaded_by_chat_runtime": True,
        "held_out_generalization_claim": False,
    }


@pytest.fixture(scope="module")
def v89() -> dict[str, Any]:
    return reporting.inspect_v89_reporting_state(ROOT)


@pytest.fixture(scope="module")
def summary() -> dict[str, Any]:
    return BUILDER["build_summary"]()


def test_v89_exact_result_and_runtime_evidence_are_authenticated(
    v89: dict[str, Any],
) -> None:
    assert v89["status"] == "authenticated_runtime_ready_single_scene_122_of_138_promoted"
    assert v89["source_bundle_authenticated"] is True
    assert v89["result_evidence_authenticated"] is True
    assert v89["training_measured"] is True
    assert v89["evaluation_measured"] is True
    assert v89["runtime_smoke_measured"] is True
    assert v89["runtime_release_authenticated"] is True
    assert v89["runtime_ready"] is True
    assert v89["runtime_promotion_authorized"] is True
    assert v89["canonical_report_generation_authorized"] is True
    assert all(v89["checks"].values())
    assert v89["single_scene_evaluation"]["canonical_exact"] == {
        "accuracy": 122 / 138,
        "correct": 122,
        "total": 138,
    }
    assert v89["single_scene_evaluation"]["model_acceptance_gate_passed"] is True
    assert v89["generic_scene1_smoke"]["observed_answers"] == ["yes", "red", "left"]
    assert v89["generic_scene1_smoke"]["records"] == [
        {"question": "Is there a chair?", "answer": "yes", "exact_correct": True},
        {"question": "What color is the bowl?", "answer": "red", "exact_correct": True},
        {
            "question": "Is the bowl left or right of the chair?",
            "answer": "left",
            "exact_correct": True,
        },
    ]
    assert v89["generic_scene1_smoke"]["development_known_and_trained"] is True
    assert v89["generic_scene1_smoke"]["held_out"] is False
    assert v89["comparison_to_v88"]["correct_answer_gain"] == 15


def test_v89_all_pinned_bytes_match(v89: dict[str, Any]) -> None:
    expected = {
        **reporting.V89_SEALED_SOURCE_SHA256,
        **reporting.V89_SEALED_RESULT_SHA256,
        **reporting.V89_SEALED_RUNTIME_SHA256,
    }
    for path, digest in expected.items():
        assert _sha256(ROOT / path) == digest
    assert set(v89["measurement_evidence_sha256"].values()) == set(
        reporting.V89_SEALED_RESULT_SHA256.values()
    ) | set(reporting.V89_SEALED_RUNTIME_SHA256.values())


def test_v89_inspector_opens_only_authenticated_json_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[Path] = []
    original = reporting._read_object

    def audited(path: Path) -> dict[str, Any]:
        opened.append(path.resolve())
        return original(path)

    monkeypatch.setattr(reporting, "_read_object", audited)
    result = reporting.inspect_v89_reporting_state(ROOT)
    expected_json = {
        (ROOT / path).resolve()
        for path in (
            reporting.V89_PREREGISTRATION,
            reporting.V89_CPU_PREFLIGHT,
            reporting.V89_TRAINING,
            reporting.V89_FIXED_FINAL_METADATA,
            reporting.V89_EVALUATION_PREDICTIONS,
            reporting.V89_EVALUATION,
            reporting.V89_ACCURACY_FIGURE_SUMMARY,
            reporting.V89_RUNTIME_SMOKE,
            reporting.V89_RUNTIME_FILE_AUDIT,
            reporting.V89_RUNTIME_RELEASE,
            reporting.V89_RELEASE_METADATA,
            reporting.V89_RELEASE_MEMORY_METADATA,
        )
    }
    assert result["evidence_authenticated"] is True
    assert set(opened) == expected_json
    assert not any("data/oracle" in path.as_posix() for path in opened)
    assert not any("data/qa" in path.as_posix() for path in opened)


def test_v89_result_fails_closed_on_digest_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        reporting.V89_SEALED_RESULT_SHA256,
        reporting.V89_EVALUATION,
        "0" * 64,
    )
    result = reporting.inspect_v89_reporting_state(ROOT)
    assert result["status"] == "source_bundle_authentication_failed"
    assert result["result_evidence_authenticated"] is False
    assert result["runtime_ready"] is False
    assert result["canonical_report_generation_authorized"] is False


def test_v89_passing_model_gate_alone_is_not_runtime_ready() -> None:
    result = reporting.classify_v89_runtime_readiness(
        evaluation=_passing_evaluation(),
        runtime_smoke=None,
        evaluation_authenticated=True,
        runtime_smoke_authenticated=False,
    )
    assert result["model_acceptance_gate_passed"] is True
    assert result["separate_runtime_smoke_passed"] is False
    assert result["runtime_ready"] is False


def test_v89_runtime_gate_inventory_is_exact_and_oracle_unavailable() -> None:
    evaluation = _passing_evaluation()
    smoke = _passing_runtime_smoke()
    assert (
        reporting.classify_v89_runtime_readiness(
            evaluation=evaluation,
            runtime_smoke=smoke,
            evaluation_authenticated=True,
            runtime_smoke_authenticated=True,
        )["runtime_ready"]
        is True
    )

    smoke["gates"]["unexpected_gate"] = True
    assert (
        reporting.classify_v89_runtime_readiness(
            evaluation=evaluation,
            runtime_smoke=smoke,
            evaluation_authenticated=True,
            runtime_smoke_authenticated=True,
        )["runtime_ready"]
        is False
    )

    smoke = _passing_runtime_smoke()
    smoke["gates"]["oracle_physically_unavailable"] = False
    assert (
        reporting.classify_v89_runtime_readiness(
            evaluation=evaluation,
            runtime_smoke=smoke,
            evaluation_authenticated=True,
            runtime_smoke_authenticated=True,
        )["runtime_ready"]
        is False
    )


def test_v89_trained_smoke_cannot_be_relabelled_as_held_out() -> None:
    evaluation = _passing_evaluation()
    evaluation["metrics"]["generic_smoke"]["held_out"] = True
    result = reporting.classify_v89_runtime_readiness(
        evaluation=evaluation,
        runtime_smoke=_passing_runtime_smoke(),
        evaluation_authenticated=True,
        runtime_smoke_authenticated=True,
    )
    assert result["model_acceptance_gate_passed"] is False
    assert result["runtime_ready"] is False


def test_v89_reporting_is_integrated_into_current_summary(
    summary: dict[str, Any],
) -> None:
    result = summary["v89_scene1_retention"]
    assert result["result_evidence_authenticated"] is True
    assert result["runtime_ready"] is True
    assert summary["status"]["v89_scene1_retention"] == result["status"]
    assert summary["claim_scope"]["v89_is_post_v88_single_scene_training_set_development"] is True
    assert summary["claim_scope"]["v89_smoke_is_trained_and_not_held_out"] is True
    assert summary["claim_scope"]["v89_runtime_ready"] is True
    assert summary["claim_scope"]["v89_held_out_generalization_measured"] is False


def test_canonical_report_gate_fails_closed_without_runtime_smoke(
    summary: dict[str, Any],
) -> None:
    authorize = BUILDER["canonical_report_generation_authorized"]
    assert authorize(summary) is True

    missing_smoke = deepcopy(summary)
    missing_smoke["v89_scene1_retention"]["runtime_smoke_authenticated"] = False
    assert authorize(missing_smoke) is False

    missing_release = deepcopy(summary)
    missing_release["v89_scene1_retention"]["runtime_release_authenticated"] = False
    assert authorize(missing_release) is False


def test_v89_markdown_reports_measured_scope_and_runtime_ready(
    summary: dict[str, Any],
) -> None:
    markdown = BUILDER["render_markdown"](summary)
    collapsed = " ".join(markdown.split())
    assert "122/138" in collapsed
    assert "88.41%" in collapsed
    assert "all 11" in collapsed
    assert "single-scene **training-set development**" in collapsed
    assert "explicitly included in training" in collapsed
    assert "**not held out**" in collapsed
    assert "oracle physically unavailable" in collapsed
    assert "strict scene-one experimental runtime" in collapsed
    assert "not held-out generalization" in collapsed
    assert "current operator default is the promoted strict V89" in collapsed
    assert "`make demo`, `make chat`" in collapsed
    assert "The runnable strict primary is V89" in collapsed
    assert "**User:** Is there a chair? **V89:** `yes`" in collapsed
    assert "**User:** What color is the bowl? **V89:** `red`" in collapsed
    assert "**User:** Is the bowl left or right of the chair? **V89:** `left`" in collapsed
    assert "These three questions and answers were explicitly included in V89 training" in collapsed
    assert "V75 is not the operator default" in collapsed
    assert "legacy strict-prefix V54 comparator" in collapsed
    assert "make demo-smoke # finite promoted strict V89 three-question proof" in collapsed
    assert "The default promoted V75 demo" not in collapsed
    assert "The runnable strict primary path uses the V54 adapter" not in collapsed
