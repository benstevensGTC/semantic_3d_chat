from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import v22_archive_validator as archive

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / archive.ARCHIVE_RELATIVE_PATH
SCREEN_PATH = PROJECT_ROOT / "reports/gemma4/metrics/v22_epoch_screen.json"
EVIDENCE_AVAILABLE = SCREEN_PATH.is_file()


def _summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def test_tracked_summary_has_exact_seal_and_no_current_source_dependency() -> None:
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
    assert result["denial_absence_verified"] is False


def test_summary_byte_tamper_is_rejected_before_artifact_validation(tmp_path: Path) -> None:
    value = _summary()
    value["outcome"]["extension_authorized"] = True
    tampered = tmp_path / "v22_final_summary.json"
    tampered.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(archive.V22ArchiveViolation, match="archive summary SHA-256"):
        archive.validate_archive(
            tampered,
            repo_root=PROJECT_ROOT,
            verify_bound_files=False,
        )


@pytest.mark.skipif(not EVIDENCE_AVAILABLE, reason="generated V22 evidence is not present")
def test_complete_local_archive_validates_after_source_tree_changes() -> None:
    result = archive.validate_archive(SUMMARY_PATH, repo_root=PROJECT_ROOT)
    assert result["bound_files_verified"] is True
    assert result["denial_absence_verified"] is True
    assert result["authoritative_artifact_count"] == 6
    assert result["checkpoint_epoch_count"] == 4
    assert result["selected_epoch"] == 3
    assert result["current_source_head_checked"] is False


@pytest.mark.skipif(not EVIDENCE_AVAILABLE, reason="generated V22 evidence is not present")
def test_bound_screen_hash_tamper_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    original = archive._file_sha256

    def mismatched_screen(path: Path, field: str) -> str:
        if field == "epoch_screen":
            return "0" * 64
        return original(path, field)

    monkeypatch.setattr(archive, "_file_sha256", mismatched_screen)
    with pytest.raises(archive.V22ArchiveViolation, match="epoch_screen SHA-256"):
        archive.validate_archive(SUMMARY_PATH, repo_root=PROJECT_ROOT)


@pytest.mark.parametrize(
    ("forbidden_kind", "message"),
    [
        ("extension", "extension was denied"),
        ("promotion", "denied promotion"),
        ("greedy", "greedy/promotion output was denied"),
    ],
)
def test_closed_screen_rejects_later_forbidden_outputs(
    forbidden_kind: str, message: str, tmp_path: Path
) -> None:
    forbidden = _summary()["forbidden_outputs"]
    primary = tmp_path / forbidden["primary_checkpoint_root"]
    primary.mkdir(parents=True)
    reports = tmp_path / "reports/gemma4/metrics"
    reports.mkdir(parents=True)

    if forbidden_kind == "extension":
        (tmp_path / forbidden["extension_checkpoint_root"]).mkdir(parents=True)
    elif forbidden_kind == "promotion":
        epoch = primary / "epoch_003"
        epoch.mkdir()
        (epoch / "promotion.json").write_text("{}\n", encoding="utf-8")
    else:
        (reports / "greedy_audit_gemma4_v22.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(archive.V22ArchiveViolation, match=message):
        archive._validate_denial_absence(tmp_path, forbidden)


def test_archive_path_cannot_escape_repository(tmp_path: Path) -> None:
    with pytest.raises(archive.V22ArchiveViolation, match="must remain inside"):
        archive._candidate_path(tmp_path, "../outside.json", "test.path")
