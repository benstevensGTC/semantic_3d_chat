from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import v21_archive_validator as archive

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / archive.ARCHIVE_RELATIVE_PATH
FINAL_REPORT_PATH = PROJECT_ROOT / "reports/gemma4/metrics/v21_extension_final.json"
EVIDENCE_AVAILABLE = FINAL_REPORT_PATH.is_file()


def test_tracked_summary_has_exact_seal_and_needs_no_current_source_head() -> None:
    result = archive.validate_archive(
        SUMMARY_PATH,
        repo_root=PROJECT_ROOT,
        verify_bound_files=False,
    )
    assert result["valid"] is True
    assert result["summary_sha256"] == archive.EXPECTED_SUMMARY_SHA256
    assert result["sealed_source_commit"] == archive.EXPECTED_SOURCE_COMMIT
    assert result["current_source_head_checked"] is False
    assert result["bound_files_verified"] is False


def test_summary_byte_tamper_is_rejected_before_artifact_validation(tmp_path: Path) -> None:
    value = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    value["outcome"]["greedy_audit_authorized"] = True
    tampered = tmp_path / "v21_final_summary.json"
    tampered.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(archive.V21ArchiveViolation, match="archive summary SHA-256"):
        archive.validate_archive(
            tampered,
            repo_root=PROJECT_ROOT,
            verify_bound_files=False,
        )


@pytest.mark.skipif(not EVIDENCE_AVAILABLE, reason="generated V21 evidence is not present")
def test_complete_local_archive_validates_after_source_tree_changes() -> None:
    result = archive.validate_archive(SUMMARY_PATH, repo_root=PROJECT_ROOT)
    assert result["bound_files_verified"] is True
    assert result["authoritative_artifact_count"] == 10
    assert result["superseded_artifact_count"] == 10
    assert result["selected_epoch"] == 8
    assert result["current_source_head_checked"] is False


@pytest.mark.skipif(not EVIDENCE_AVAILABLE, reason="generated V21 evidence is not present")
def test_bound_artifact_hash_tamper_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    original = archive._file_sha256

    def mismatched_final_report(path: Path, field: str) -> str:
        if field == "final_selector_report":
            return "0" * 64
        return original(path, field)

    monkeypatch.setattr(archive, "_file_sha256", mismatched_final_report)
    with pytest.raises(archive.V21ArchiveViolation, match="final_selector_report SHA-256"):
        archive.validate_archive(SUMMARY_PATH, repo_root=PROJECT_ROOT)


@pytest.mark.skipif(not EVIDENCE_AVAILABLE, reason="generated V21 evidence is not present")
def test_denied_v21_rejects_any_promotion_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    denied_checkpoint = tmp_path / "epoch_008"
    denied_checkpoint.mkdir()
    (denied_checkpoint / "promotion.json").write_text("{}\n", encoding="utf-8")
    original = archive._bound_path

    def redirected_checkpoint(repo_root: Path, relative: object, field: str) -> Path:
        if field == "selected checkpoint path":
            return denied_checkpoint
        return original(repo_root, relative, field)

    monkeypatch.setattr(archive, "_bound_path", redirected_checkpoint)
    with pytest.raises(archive.V21ArchiveViolation, match="denied promotion"):
        archive.validate_archive(SUMMARY_PATH, repo_root=PROJECT_ROOT)


def test_archive_path_cannot_escape_repository(tmp_path: Path) -> None:
    with pytest.raises(archive.V21ArchiveViolation, match="must remain inside"):
        archive._bound_path(tmp_path, "../outside.json", "test.path")
