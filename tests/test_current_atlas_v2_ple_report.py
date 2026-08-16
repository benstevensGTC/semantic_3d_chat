from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = runpy.run_path(str(ROOT / "scripts/build_current_report.py"))


def _inspect(name: str) -> dict[str, Any]:
    return BUILDER[name]()


def test_atlas_v2_report_authenticates_only_structural_exposure() -> None:
    result = _inspect("_inspect_fixed_prefix_atlas_v2")

    assert result["status"] == "authenticated_structural_only_compilation_disabled"
    assert result["evidence_authenticated"] is True
    assert result["fixed_prefix_token_count"] == 738
    assert result["base_scene_latent_count"] == 256
    assert result["atlas_memory_token_count"] == 480
    assert result["prompt_token_range_including_bos"] == [57, 64]
    assert result["v1_visible_base_latents"] == 0
    assert result["v2_visible_base_latents"] == 256
    assert len(result["exposure_rows"]) == 8
    assert result["direct_sliding_attention_exposure_only"] is True
    assert result["periodic_full_attention_layers_outside_calculation"] is True
    assert result["compilation_enabled"] is False
    assert result["checkpoint_present"] is False
    assert result["behavioral_accuracy_measured"] is False
    assert result["behavioral_improvement_claimed"] is False
    assert all(result["config_checks"].values())
    assert set(result["implementation_source_sha256"]) == {
        "src/semantic_3d_chat/scene_encoder/fixed_prefix_atlas_v2.py",
        "src/semantic_3d_chat/evaluation/fixed_prefix_atlas_v2_exposure.py",
        "configs/experiments/gemma4_strict_fixed_prefix_atlas_v2.yaml",
        "tests/test_fixed_prefix_atlas_v2.py",
    }


def test_ple_reader_report_is_authenticated_design_only() -> None:
    result = _inspect("_inspect_ple_reader_preregistration")

    assert result["status"] == "authenticated_design_only_training_not_authorized"
    assert result["evidence_authenticated"] is True
    assert result["design_only"] is True
    assert result["training_authorized"] is False
    assert result["training_executed"] is False
    assert result["gemma_generation_executed"] is False
    assert result["checkpoint_published"] is False
    assert result["behavioral_accuracy_measured"] is False
    assert result["rank"] == 4
    assert result["trainable_parameter_count"] == 41_984
    assert result["projection_shape"] == [8960, 1536]
    assert result["atlas_v2_acceptance_required_before_training"] is True
    assert result["environmental_text_inputs"] == []
    assert all(result["checks"].values())
    assert len(result["preregistration_sha256"]) == 64
    assert set(result["implementation_source_sha256"]) == {
        "src/semantic_3d_chat/evaluation/ple_reader_preregistration.py",
        "configs/experiments/gemma4_fixed_prefix_ple_reader_v1.yaml",
        "tests/test_ple_reader_preregistration.py",
        "reports/gemma4/metrics/gemma4_fixed_prefix_ple_reader_preregistration_v1.json",
    }


@pytest.mark.parametrize(
    ("inspector_name", "hashes_name", "expected_status"),
    [
        (
            "_inspect_fixed_prefix_atlas_v2",
            "FIXED_ATLAS_V2_SOURCE_SHA256",
            "structural_evidence_authentication_failed",
        ),
        (
            "_inspect_ple_reader_preregistration",
            "PLE_READER_SOURCE_SHA256",
            "preregistration_evidence_authentication_failed",
        ),
    ],
)
def test_structural_evidence_fails_closed_on_digest_drift(
    inspector_name: str,
    hashes_name: str,
    expected_status: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER[inspector_name]
    pinned = inspector.__globals__[hashes_name]
    first_path = next(iter(pinned))
    monkeypatch.setitem(pinned, first_path, "0" * 64)

    result = inspector()

    assert result["status"] == expected_status
    assert result["evidence_authenticated"] is False
    if "training_authorized" in result:
        assert result["training_authorized"] is False
    assert result["behavioral_accuracy_measured"] is False
    assert "digest changed" in result["measurement_evidence_error"]


def test_current_summary_and_markdown_keep_both_claims_bounded() -> None:
    summary = BUILDER["build_summary"]()

    atlas = summary["strict_fixed_prefix_atlas_v2"]
    reader = summary["fixed_prefix_ple_reader_preregistration"]
    assert atlas["evidence_authenticated"] is True
    assert reader["evidence_authenticated"] is True
    for path, digest in atlas["implementation_source_sha256"].items():
        assert summary["source_artifacts"][path] == digest
    for path, digest in reader["implementation_source_sha256"].items():
        assert summary["source_artifacts"][path] == digest

    markdown = BUILDER["render_markdown"](summary)
    assert "0/256 base latents in V1 to 256/256 in V2" in markdown
    assert "structural local-window" in markdown
    assert "evidence only; V2 compilation is disabled" in markdown
    assert "41,984 trainable" in markdown
    assert "That earlier artifact remains a design-only preregistration" in markdown
    assert "make strict-atlas-v2-auth" in markdown
    assert "make ple-reader-prereg-auth" in markdown
