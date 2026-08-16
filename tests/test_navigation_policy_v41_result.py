from __future__ import annotations

from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.navigation_policy_v41_result import (
    TRAINING_REPORT_SHA256,
    V41ResultAuthenticationError,
    authenticate_navigation_policy_v41_result,
    inspect_navigation_policy_v41_result,
)


def test_packaged_v4_1_rejection_authenticates_without_checkpoint_or_live_run() -> None:
    result = authenticate_navigation_policy_v41_result()
    assert result["measurement_authenticated"] is True
    assert result["status"] == (
        "historical_evidence_authenticated_current_runtime_compatibility_not_claimed"
    )
    assert result["training_report_sha256"] == TRAINING_REPORT_SHA256
    assert result["checkpoint_absent"] is True
    assert result["live_benchmark_executed"] is False
    assert result["oracle_or_scorer_opened"] is False
    assert result["failed_gates"] == ["shuffled_clearance_family_drop"]
    assert result["passed_gate_count"] == 13
    assert result["gate_count"] == 14
    assert result["promotion_eligible"] is False
    assert result["historical_source_inventory_authenticated"] is True
    assert result["current_runtime_compatibility_claimed"] is False


def test_v4_1_historical_result_does_not_claim_current_runtime_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from semantic_3d_chat.evaluation import navigation_policy_v41_result as evidence

    original_sha256 = evidence._sha256

    def drift_one_current_source(path: Path) -> str:
        if path.name == "conversation_cli.py":
            return "0" * 64
        return original_sha256(path)

    monkeypatch.setattr(evidence, "_sha256", drift_one_current_source)
    result = evidence.authenticate_navigation_policy_v41_result()

    assert result["measurement_authenticated"] is True
    assert result["current_source_snapshot_matches_sealed"] is False
    assert any(path.endswith("conversation_cli.py") for path in result["current_source_drift_paths"])
    assert result["current_runtime_compatibility_claimed"] is False


def test_v4_1_result_authentication_fails_closed_on_report_tamper(
    tmp_path: Path,
) -> None:
    tampered = tmp_path / "training.json"
    source = Path("reports/gemma4/metrics/navigation_policy_v4_1_training.json")
    tampered.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(V41ResultAuthenticationError, match="digest differs"):
        authenticate_navigation_policy_v41_result(training_report=tampered)
    inspected = inspect_navigation_policy_v41_result(training_report=tampered)
    assert inspected["measurement_authenticated"] is False
    assert inspected["promotion_eligible"] is False


def test_v4_1_result_authentication_fails_closed_if_checkpoint_exists(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "unexpected"
    checkpoint.mkdir()
    with pytest.raises(V41ResultAuthenticationError, match="unexpectedly exists"):
        authenticate_navigation_policy_v41_result(checkpoint=checkpoint)
