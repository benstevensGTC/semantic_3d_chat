from __future__ import annotations

from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.navigation_policy_v3_evidence import (
    EvidenceAuthenticationError,
    authenticate_navigation_policy_v3,
    inspect_navigation_policy_v3,
)


def test_packaged_v3_evidence_authenticates() -> None:
    result = authenticate_navigation_policy_v3()
    assert result["measurement_authenticated"] is True
    assert result["evidence_version"] == "v3"
    assert result["current_version"] == "v3_historical"
    assert result["evidence_scope"] == "historical_sealed_run"
    assert result["current_runtime_compatibility_claimed"] is False
    assert result["historical_source_snapshot"]["exact_original_bytes_available"] is True
    assert result["historical_source_snapshot"]["current_runtime_source_claimed"] is False
    assert result["live_benchmark"]["metrics"]["success_count"] == 5
    assert result["live_benchmark"]["metrics"]["collision_count"] == 0
    assert result["offline_training"]["weak_direct_scene_prefix_controls"] is True
    assert result["runtime_checkpoint_audit"]["oracle_directory_unavailable"] is True


def test_tampered_v3_artifact_fails_closed(tmp_path: Path) -> None:
    tampered = tmp_path / "score.json"
    source = Path("reports/gemma4/metrics/llm_navigation_scene_000001_learned_v3.json")
    tampered.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(EvidenceAuthenticationError, match="artifact digest differs: score"):
        authenticate_navigation_policy_v3(file_overrides={"score": tampered})
    result = inspect_navigation_policy_v3(file_overrides={"score": tampered})
    assert result["measurement_authenticated"] is False
    assert result["claimed_trained_navigation_policy"] is False


def test_tampered_v3_source_fails_closed(tmp_path: Path) -> None:
    tampered = tmp_path / "navigation_policy_v3.py"
    source = Path(
        "reports/gemma4/evidence/navigation_policy_v3_sources/navigation_policy_v3.py"
    )
    tampered.write_bytes(source.read_bytes() + b"# tamper\n")
    with pytest.raises(EvidenceAuthenticationError, match="source digest differs: policy"):
        authenticate_navigation_policy_v3(source_overrides={"policy": tampered})


def test_v3_evidence_inspector_never_opens_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.open

    def guarded(path: Path, *args: object, **kwargs: object):
        assert "oracle" not in {part.casefold() for part in path.parts}
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    assert authenticate_navigation_policy_v3()["measurement_authenticated"] is True
