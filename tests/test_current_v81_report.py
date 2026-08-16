from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = runpy.run_path(str(ROOT / "scripts/build_current_report.py"))


@pytest.fixture(scope="module")
def summary() -> dict[str, Any]:
    return BUILDER["build_summary"]()


def test_current_summary_bounds_v81_fixed_memory_claim(summary: dict[str, Any]) -> None:
    evidence = summary["v81_sealed_scene_memory"]

    assert evidence["status"] == (
        "authenticated_experimental_runtime_historical_gate_failed_not_promoted"
    )
    assert evidence["evidence_authenticated"] is True
    assert evidence["runtime_promotion_authorized"] is False
    assert evidence["official_validation_measured"] is False
    assert evidence["fixed_memory"] == {
        "scene_id": "scene_000001",
        "shape": [1, 738, 1536],
        "dtype": "torch.bfloat16",
        "canonical_prefix_sha256": (
            "a428f5147c815839ae7315a0adab952ab210814fb21dcdc5bf13b167f28a6e37"
        ),
        "tensor_sha256": (
            "b9e6503e652b18514f157b0b9a5580e84c5fe19b18e4f294e3be010438c530ab"
        ),
        "tensor_file_sha256": (
            "3427851432d8f2a3609b6205b18b8c0d9a0fcf68d8f3bf0c98c758ac64209ffb"
        ),
        "base_environment_latents": 256,
        "atlas_memory_tokens": 480,
        "probe_groups": 96,
        "scene_value_tokens": 384,
        "compiled_before_question": True,
        "question_dependent_retrieval": False,
        "semantic_or_spatial_top_k_selection": False,
        "environmental_text_inputs": [],
        "exact_two_file_release": True,
    }
    historical = evidence["historical_development"]
    assert historical["scores"] == {
        "v81": {"correct": 8, "total": 16},
        "frozen_v54": {"correct": 6, "total": 16},
        "shuffled_atlas": {"correct": 3, "total": 16},
        "zero_environment": {"correct": 1, "total": 16},
        "wrong_scene": {"correct": 9, "total": 16},
    }
    assert historical["pair_disjoint"] is True
    assert historical["scene_disjoint"] is True
    assert historical["question_disjoint"] is False
    assert historical["passed"] is False
    assert historical["gates"]["candidate_correct_at_least_9"] is False
    assert historical["gates"]["correct_minus_wrong_scene_at_least_2"] is False
    assert historical["gates"]["gain_over_frozen_v54_at_least_3"] is False

    isolation = evidence["live_isolation"]
    assert isolation["passed"] is True
    assert isolation["fixed_738_memory_invariant"] is True
    assert isolation["base_258_prefix_invariant"] is True
    assert isolation["oracle_unavailable_during_inference"] is True
    assert isolation["oracle_restored"] is True
    assert isolation["compiler_or_probe_reads"] == 0
    assert isolation["forbidden_access_count"] == 0
    assert isolation["loaded_file_count"] == 4204

    grounding = evidence["live_grounding_mechanism"]
    assert grounding["answer"] == "unknown"
    assert grounding["all_scene_tokens_scored"] is True
    assert grounding["scene_latent_count"] == 256
    assert grounding["accuracy_claimed"] is False
    assert grounding["forbidden_access_count"] == 0
    assert all(evidence["checks"].values())


def test_current_summary_authenticates_two_scene_conversation_mcp(
    summary: dict[str, Any],
) -> None:
    evidence = summary["conversation_mcp_stdio"]

    assert evidence["status"] == "passed_two_scene_live_official_mcp_stdio_integration"
    assert evidence["evidence_authenticated"] is True
    assert evidence["scene_ids"] == ["scene_000001", "scene_000031"]
    assert evidence["transport"] == "official_sdk_stdio"
    assert evidence["agent"] == "ConversationalEmbodiedAgent"
    assert evidence["action_commands"] == ["scan", "turn", "stop"]
    assert evidence["map_versions"] == [0, 1, 2, 2]
    assert evidence["binding_refresh_count_per_scene"] == 4
    assert evidence["numeric_structured_receipts_only"] is True
    assert evidence["environmental_text_inputs"] == []
    assert evidence["forbidden_access_count"] == 0
    assert len(evidence["runs"]) == 2
    assert {row["loaded_file_count"] for row in evidence["runs"]} == {4182}
    assert len({row["final_scene_prefix_sha256"] for row in evidence["runs"]}) == 2


def test_current_markdown_reports_v81_and_mcp_without_promotion(
    summary: dict[str, Any],
) -> None:
    markdown = BUILDER["render_markdown"](summary)
    collapsed = " ".join(markdown.split())

    assert "V81 sealed scene-memory state" in markdown
    assert (
        "a428f5147c815839ae7315a0adab952ab210814fb21dcdc5bf13b167f28a6e37"
        in markdown
    )
    assert "V81 scored 8/16 versus 6/16 for frozen V54" in collapsed
    assert "3/16 after atlas-value shuffling" in collapsed
    assert "1/16 with an exactly zero environmental payload" in collapsed
    assert "9/16 with the paired wrong scene" in collapsed
    assert "V81 remains experimental and runtime promotion is false" in collapsed
    assert "passed_two_scene_live_official_mcp_stdio_integration" in markdown
    assert "scene_000001` and `scene_000031" in markdown
    assert "This is real two-scene transport/state-refresh evidence" in collapsed
    assert "not semantic instruction-following accuracy" in collapsed
    assert "make v81-scene-memory-chat" in markdown
    assert "make conversation-mcp-smoke" in markdown


def test_v81_inspector_fails_closed_on_historical_score_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER["_inspect_v81_sealed_scene_memory_runtime"]
    original = ROOT / BUILDER["V81_HISTORICAL_SCORE"]
    tampered = json.loads(original.read_text(encoding="utf-8"))
    tampered["arms"]["v81"]["correct"] = 9
    path = tmp_path / original.name
    path.write_text(json.dumps(tampered), encoding="utf-8")
    monkeypatch.setitem(inspector.__globals__, "V81_HISTORICAL_SCORE", path)

    result = inspector()

    assert result["status"] == "artifact_present_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert result["runtime_promotion_authorized"] is False
    assert "historical_controls" in result["measurement_evidence_error"]


def test_conversation_mcp_inspector_fails_closed_on_text_leakage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER["_inspect_conversation_mcp_stdio_smokes"]
    original_paths = BUILDER["CONVERSATION_MCP_STDIO_REPORTS"]
    first = json.loads((ROOT / original_paths[0]).read_text(encoding="utf-8"))
    first["environmental_text_inputs"] = ["forbidden caption"]
    tampered = tmp_path / original_paths[0].name
    tampered.write_text(json.dumps(first), encoding="utf-8")
    monkeypatch.setitem(
        inspector.__globals__,
        "CONVERSATION_MCP_STDIO_REPORTS",
        (tampered, original_paths[1]),
    )

    result = inspector()

    assert result["status"] == "artifact_present_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert "scene_000001" in result["measurement_evidence_error"]
