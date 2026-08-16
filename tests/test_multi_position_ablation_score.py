from __future__ import annotations

import hashlib
import json
import runpy
import shutil
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCORER_PATH = PROJECT_ROOT / "scripts" / "score_center_vs_multi_position.py"
BUILDER_PATH = PROJECT_ROOT / "scripts" / "build_current_report.py"
REPORT_PATH = (
    PROJECT_ROOT / "reports" / "gemma4_multi_position" / "metrics" / "center_vs_multi_position.json"
)


def _load(path: Path) -> dict[str, Any]:
    return runpy.run_path(str(path))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_scorer_reproduces_authenticated_report_and_create_or_verify(tmp_path: Path) -> None:
    scorer = _load(SCORER_PATH)
    expected = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    observed = scorer["build_comparison"]()

    assert observed == expected
    assert observed["artifact_sha256"] == (
        "eb0e5d9cc697d5dfd12c83fe41933cc3d5e947ea0a089fc878c0b6eada6e1fb8"
    )
    assert observed["contract"]["model_free"] is True
    assert observed["contract"]["report_only"] is True
    assert observed["contract"]["data_input_allowlist_exact"] == [
        "center_map",
        "center_semantic",
        "multi_position_map",
        "multi_position_semantic",
    ]
    assert observed["delta_multi_position_minus_center"]["coverage"]["occupied_voxels"] == 23_377
    assert observed["delta_multi_position_minus_center"]["coverage"][
        "multiview_voxel_fraction"
    ] == pytest.approx(0.21497925826581255)
    assert observed["delta_multi_position_minus_center"]["semantic_localization"][
        "top1_localization_accuracy"
    ] == pytest.approx(-0.23076923076923078)
    assert (
        observed["delta_multi_position_minus_center"]["semantic_localization"][
            "top_k_localization_accuracy"
        ]
        == 0.0
    )
    assert observed["delta_multi_position_minus_center"]["semantic_localization"][
        "mean_precision_at_k"
    ] == pytest.approx(-0.041538461538461525)
    assert observed["delta_multi_position_minus_center"]["view_consistency"][
        "same_minus_different_mean_cosine"
    ] == pytest.approx(-0.14035462503743623)
    assert observed["directional_summary"]["occupied_voxel_coverage_increased"] is True
    assert observed["directional_summary"]["top1_localization_improved"] is False
    assert observed["directional_summary"]["precision_at_k_improved"] is False

    output = tmp_path / "comparison.json"
    assert scorer["_write_create_or_verify"](output, observed) == "created"
    assert scorer["_write_create_or_verify"](output, observed) == "verified_existing"
    assert output.read_bytes() == REPORT_PATH.read_bytes()


def test_scorer_rejects_a_tampered_allowlisted_input(tmp_path: Path) -> None:
    scorer = _load(SCORER_PATH)
    copied: dict[str, Path] = {}
    for role, relative in scorer["DEFAULT_INPUTS"].items():
        destination = tmp_path / f"{role}.json"
        shutil.copyfile(PROJECT_ROOT / relative, destination)
        copied[role] = destination

    tampered = json.loads(copied["center_map"].read_text(encoding="utf-8"))
    tampered["occupied_voxels"] += 1
    copied["center_map"].write_text(json.dumps(tampered, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="input digest differs for center_map"):
        scorer["build_comparison"](
            center_map_path=copied["center_map"],
            center_semantic_path=copied["center_semantic"],
            multi_position_map_path=copied["multi_position_map"],
            multi_position_semantic_path=copied["multi_position_semantic"],
        )


def test_report_builder_authenticates_and_renders_honest_ablation() -> None:
    builder = _load(BUILDER_PATH)
    measurement = builder["_inspect_multi_position_ablation"]()

    assert measurement["measurement_status"] == "authenticated_complete"
    assert measurement["measurement_authenticated"] is True
    assert all(measurement["authentication_checks"].values())
    assert measurement["measurements"]["center_24_view"]["scan"]["frame_count"] == 24
    assert measurement["measurements"]["multi_position_96_view"]["scan"]["frame_count"] == 96
    assert measurement["delta_multi_position_minus_center"]["coverage"][
        "occupied_voxel_relative_change"
    ] == pytest.approx(0.31294930320352354)

    summary = json.loads(
        (PROJECT_ROOT / "reports" / "metrics" / "current_metrics.json").read_text(encoding="utf-8")
    )
    summary["v67_pair_objective_numeric_screen"] = {
        "status": "not_measured",
        "measurement_status": "not_measured",
    }
    summary["v68_regularized_pair_numeric_grid"] = {
        "status": "not_measured",
        "measurement_status": "not_measured",
    }
    summary["center_vs_multi_position_ablation"] = measurement
    summary["oracle_text_upper_bound"] = {
        "status": "not_measured",
        "measurement_status": "not_measured",
        "prepared_scene_text_present": False,
    }
    summary["direct_multiview_baseline"] = {
        "status": "authenticated_complete",
        "exact_count": 0,
        "question_count": 1,
        "normalized_exact_accuracy": 0.0,
        "scene_count": 1,
        "complete_views_per_scene": 24,
        "spatial_relation_accuracy": 0.0,
        "count_accuracy": 0.0,
        "presence_f1": 0.0,
    }
    summary["llm_tool_policy"]["navigation_measurement_status"] = "not_measured"
    markdown = builder["render_markdown"](summary)
    compact_markdown = " ".join(markdown.split())
    assert "24-view center scan" in compact_markdown
    assert "74,699 to 98,076" in compact_markdown
    assert "+31.29%" in compact_markdown
    assert "61.54% to 38.46%" in compact_markdown
    assert "top-k remained 84.62%" in compact_markdown
    assert "45.23% to 41.08%" in compact_markdown
    assert "includes no downstream QA or navigation run" in compact_markdown


def test_report_builder_fails_closed_on_rehashed_wrong_delta(tmp_path: Path) -> None:
    builder = _load(BUILDER_PATH)
    inspector_globals = builder["_inspect_multi_position_ablation"].__globals__
    copied_inputs: dict[str, Path] = {}
    for role, relative in builder["MULTI_POSITION_INPUT_PATHS"].items():
        destination = tmp_path / f"{role}.json"
        shutil.copyfile(PROJECT_ROOT / relative, destination)
        copied_inputs[role] = destination

    tampered_report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    tampered_report["input_artifacts"] = {
        role: {
            "path": path.as_posix(),
            "sha256": builder["MULTI_POSITION_INPUT_SHA256"][role],
        }
        for role, path in copied_inputs.items()
    }
    tampered_report["delta_multi_position_minus_center"]["coverage"]["occupied_voxels"] += 1
    body = dict(tampered_report)
    body.pop("artifact_sha256")
    tampered_report["artifact_sha256"] = builder["_canonical_newline_sha256"](body)
    tampered_path = tmp_path / "comparison.json"
    tampered_path.write_text(
        json.dumps(tampered_report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    inspector_globals["MULTI_POSITION_ABLATION_REPORT"] = tampered_path
    inspector_globals["MULTI_POSITION_ABLATION_REPORT_SHA256"] = _sha256(tampered_path)
    inspector_globals["MULTI_POSITION_ABLATION_ARTIFACT_SHA256"] = tampered_report[
        "artifact_sha256"
    ]
    inspector_globals["MULTI_POSITION_INPUT_PATHS"] = copied_inputs
    result = builder["_inspect_multi_position_ablation"]()
    assert result["measurement_status"] == "artifact_present_authentication_failed"
    assert "deltas differ from recomputation" in result["measurement_evidence_error"]


def test_report_builder_treats_absent_optional_artifact_as_unmeasured(
    tmp_path: Path,
) -> None:
    builder = _load(BUILDER_PATH)
    builder["_inspect_multi_position_ablation"].__globals__["MULTI_POSITION_ABLATION_REPORT"] = (
        tmp_path / "absent.json"
    )
    result = builder["_inspect_multi_position_ablation"]()
    assert result["measurement_status"] == "not_measured"
    assert result["measurement_authenticated"] is False


def test_makefile_exposes_reproducible_ablation_target() -> None:
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    assert ".PHONY: score-multi-position-ablation" in makefile
    assert "score-multi-position-ablation:" in makefile
    assert "scripts/score_center_vs_multi_position.py" in makefile
