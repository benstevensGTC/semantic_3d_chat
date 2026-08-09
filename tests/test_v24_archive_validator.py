from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import v24_archive_validator as archive

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / archive.ARCHIVE_RELATIVE_PATH
FINAL_PATH = PROJECT_ROOT / "reports/gemma4/metrics/v24_extension_final.json"
CHECKPOINT_PATH = PROJECT_ROOT / "data_gemma4/checkpoints/gemma4_v24_shared_query/epoch_001"
EVIDENCE_AVAILABLE = FINAL_PATH.is_file() and CHECKPOINT_PATH.is_dir()


def _summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def test_tracked_v24_summary_has_exact_source_independent_seal() -> None:
    result = archive.validate_archive(
        SUMMARY_PATH,
        repo_root=PROJECT_ROOT,
        verify_bound_files=False,
    )
    assert result["valid"] is True
    assert result["summary_sha256"] == archive.EXPECTED_SUMMARY_SHA256
    assert result["training_source_commit"] == archive.EXPECTED_TRAINING_COMMIT
    assert result["controller_source_commit"] == archive.EXPECTED_CONTROLLER_COMMIT
    assert result["current_source_head_checked"] is False
    assert result["bound_files_verified"] is False
    assert result["denial_absence_verified"] is False
    assert result["selected_epoch"] == 1
    assert result["greedy_audit_authorized"] is False


def test_v24_summary_byte_tamper_fails_before_artifact_validation(tmp_path: Path) -> None:
    value = _summary()
    value["outcome"]["greedy_audit_authorized"] = True
    tampered = tmp_path / "v24_final_summary.json"
    tampered.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(archive.V24ArchiveViolation, match="archive summary SHA-256"):
        archive.validate_archive(
            tampered,
            repo_root=PROJECT_ROOT,
            verify_bound_files=False,
        )


@pytest.mark.skipif(not EVIDENCE_AVAILABLE, reason="generated V24 evidence is not present")
def test_complete_v24_archive_validates_after_source_tree_changes() -> None:
    result = archive.validate_archive(SUMMARY_PATH, repo_root=PROJECT_ROOT)
    assert result["bound_files_verified"] is True
    assert result["denial_absence_verified"] is True
    assert result["authoritative_artifact_count"] == 11
    assert result["checkpoint_epoch_count"] == 8
    assert result["decision"] == "conditional_limit_reached_no_greedy_audit"
    assert result["current_source_head_checked"] is False


@pytest.mark.skipif(not EVIDENCE_AVAILABLE, reason="generated V24 evidence is not present")
def test_bound_v24_final_hash_tamper_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = archive._file_sha256

    def mismatched_final(path: Path, field: str) -> str:
        if field == "final_selector":
            return "0" * 64
        return original(path, field)

    monkeypatch.setattr(archive, "_file_sha256", mismatched_final)
    with pytest.raises(archive.V24ArchiveViolation, match="final_selector SHA-256"):
        archive.validate_archive(SUMMARY_PATH, repo_root=PROJECT_ROOT)


@pytest.mark.parametrize("forbidden_kind", ["promotion", "greedy", "static", "leakage", "robot"])
def test_closed_v24_rejects_forbidden_downstream_outputs(
    forbidden_kind: str, tmp_path: Path
) -> None:
    checkpoint_root = tmp_path / "data_gemma4/checkpoints/v24"
    checkpoint_root.mkdir(parents=True)
    metrics = tmp_path / "reports/gemma4/metrics"
    metrics.mkdir(parents=True)
    forbidden = {
        "checkpoint_roots": ["data_gemma4/checkpoints/v24"],
        "forbidden_checkpoint_filename": "promotion.json",
        "forbidden_metric_globs": [
            "*v24*greedy*.json",
            "*v24*static*qa*.json",
            "*v24*leakage*.json",
            "*v24*robot*.json",
        ],
    }
    if forbidden_kind == "promotion":
        (checkpoint_root / "promotion.json").write_text("{}\n", encoding="utf-8")
    else:
        filename = {
            "greedy": "v24_greedy.json",
            "static": "v24_static_qa.json",
            "leakage": "v24_leakage.json",
            "robot": "v24_robot.json",
        }[forbidden_kind]
        (metrics / filename).write_text("{}\n", encoding="utf-8")

    with pytest.raises(archive.V24ArchiveViolation, match="denied"):
        archive._validate_denial_absence(tmp_path, forbidden)


def test_v24_archive_path_cannot_escape_repository(tmp_path: Path) -> None:
    with pytest.raises(archive.V24ArchiveViolation, match="must remain inside"):
        archive._candidate_path(tmp_path, "../outside.json", "test.path")
