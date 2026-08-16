from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import v51_query_alpha_grid as v51
from semantic_3d_chat.evaluation import v52_query_alpha_refinement as v52
from semantic_3d_chat.evaluation import v54_semantic_greedy_gate as v54


def _row(
    answer_type: str,
    expected: str,
    generated: str,
    *,
    legacy: bool = False,
    list_correct: bool = False,
) -> dict[str, object]:
    return {
        "answer_type": answer_type,
        "expected_normalized_answer": expected,
        "generated_normalized_answer": generated,
        "legacy_exact_correct": legacy,
        "type_aware_correct": list_correct,
    }


def test_canonical_type_specific_match_uses_existing_metric_contract() -> None:
    assert v54.canonical_type_specific_match(
        _row("presence", "yes", "yes there is floor lamp in room")
    )
    assert v54.canonical_type_specific_match(_row("count", "2", "there are two"))
    assert v54.canonical_type_specific_match(
        _row("spatial_relation", "left", "it is to left")
    )
    assert v54.canonical_type_specific_match(
        _row("support", "book, cube", "cube, book", list_correct=True)
    )
    assert not v54.canonical_type_specific_match(
        _row("attribute", "yellow", "brown")
    )
    assert not v54.canonical_type_specific_match(
        _row("presence", "yes", "yes or no")
    )


def test_exact_v53_rows_recompute_only_two_known_rescues() -> None:
    report = json.loads(v54._resolve(v54.V53_REPORT).read_text(encoding="utf-8"))
    detail = report["detailed_greedy"]
    result = v54.recompute_semantic_metrics(detail)
    assert result["legacy_exact"] == {
        "changed_rows_correct": 24,
        "complete_units": 4,
        "broad_rows_correct": 23,
    }
    assert result["canonical_type_specific"] == {
        "changed_rows_correct": 25,
        "complete_units": 5,
        "broad_rows_correct": 24,
    }
    assert result["changed_rescues"] == [v54._EXPECTED_CHANGED_RESCUE]
    assert result["broad_rescues"] == [v54._EXPECTED_BROAD_RESCUE]
    assert result["regressions"] == []


def test_exact_predecessor_and_candidate_hashes_are_authenticated() -> None:
    result = v54.authenticate_predecessor(v54.V53_REPORT_SHA256)
    assert result["sha256"] == v54.V53_REPORT_SHA256
    assert all(result["checks"].values())
    assert result["candidate_reconstruction"]["full_tensor_state_sha256"] == (
        v54.CANDIDATE_FULL_SHA256
    )
    with pytest.raises(ValueError, match="pinned V53 report"):
        v54.authenticate_predecessor("0" * 64)


def test_metrics_implementation_is_hashed_before_scorer_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer_executed = False

    def reject_metrics(_path: Path, _expected: str, field: str) -> None:
        assert field == "V54 metrics module"
        raise ValueError("simulated metrics integrity failure")

    def forbidden_recompute(_detail: object) -> dict[str, object]:
        nonlocal scorer_executed
        scorer_executed = True
        return {}

    monkeypatch.setattr(v54, "_locked_hash", reject_metrics)
    monkeypatch.setattr(v54, "recompute_semantic_metrics", forbidden_recompute)
    with pytest.raises(ValueError, match="simulated metrics integrity"):
        v54.authenticate_predecessor(v54.V53_REPORT_SHA256)
    assert scorer_executed is False


def test_model_free_preflight_and_pinned_paths(tmp_path: Path) -> None:
    if v54._resolve(v54.DEFAULT_REPORT).exists():
        with pytest.raises(FileExistsError, match="one-shot"):
            v54.preflight(expected_v53_report_sha256=v54.V53_REPORT_SHA256)
    else:
        result = v54.preflight(expected_v53_report_sha256=v54.V53_REPORT_SHA256)
        assert result["passed"] is True
        assert result["model_loaded"] is False
        assert result["qa_loaded"] is False
        assert result["maps_loaded"] is False
        assert result["generation_executed"] is False
        assert result["checkpoint_written"] is False
    with pytest.raises(ValueError, match="paths are pinned"):
        v54.preflight(
            expected_v53_report_sha256=v54.V53_REPORT_SHA256,
            paths=v54.GatePaths(report=tmp_path / "other.json"),
        )


def test_inherited_v52_provenance_and_v54_metadata_rewrite(tmp_path: Path) -> None:
    predecessor = v54.authenticate_predecessor(v54.V53_REPORT_SHA256)
    provenance = v54.inherited_v52_staging_provenance(predecessor)
    assert provenance["terminal_path"] == str(v52.V51_REPORT)
    assert provenance["terminal_sha256"] == v52.V51_REPORT_SHA256
    assert provenance["authorization_id"] == v52.AUTHORIZATION_ID

    source_metadata = v54._resolve(v52.SOURCE_CHECKPOINT) / "metadata.json"
    metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
    metadata[v52.AUTHORIZATION_ID] = {
        "artifact": v52.AUTHORIZATION_ID,
        "authenticated_predecessor_sha256": v52.V51_REPORT_SHA256,
    }
    (tmp_path / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True), encoding="utf-8"
    )
    state = {
        "full_tensor_state_sha256": v54.CANDIDATE_FULL_SHA256,
        "authorized_surface_state_sha256": v54.CANDIDATE_AUTHORIZED_SHA256,
        "frozen_state_sha256": v51._FROZEN_SHA256,
    }
    stage = v54._rewrite_v54_metadata(
        tmp_path,
        predecessor=predecessor,
        preparation={"prefix_reference_hash_inventory_sha256": "a" * 64},
        state=state,
    )
    rewritten = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    runtime = json.loads(
        (tmp_path / "runtime_metadata.json").read_text(encoding="utf-8")
    )
    assert stage["artifact"] == v54.AUTHORIZATION_ID
    assert stage["write_authorization"] == {
        "source": "user_standing_build_request",
        "scope": "materialize_exact_authenticated_v52_candidate_only",
        "v53_is_authenticated_evidence_not_write_authority": True,
    }
    assert v52.AUTHORIZATION_ID not in rewritten
    assert rewritten[v54.AUTHORIZATION_ID] == stage
    assert v54.AUTHORIZATION_ID not in runtime
    assert stage["authenticated_predecessors"]["v53_report_sha256"] == (
        v54.V53_REPORT_SHA256
    )


def test_final_report_requires_checkpoint_restoration_and_clean_access() -> None:
    predecessor = v54.authenticate_predecessor(v54.V53_REPORT_SHA256)
    reconstruction = dict(predecessor["candidate_reconstruction"])
    restoration = {
        "passed": True,
        "full_tensor_state_sha256": v51._SOURCE_FULL_SHA256,
        "authorized_surface_state_sha256": v51._SOURCE_AUTHORIZED_SHA256,
        "frozen_state_sha256": v51._FROZEN_SHA256,
        "all_parameter_gradients_absent": True,
    }
    access = {
        "passed": True,
        "training_map_count": 16,
        "optimizer_file_reads": [],
        "forbidden_file_accesses": [],
        "validation_qa_loaded": False,
        "oracle_loaded": False,
        "final_test_loaded": False,
    }
    passed = v54._report_summary(
        predecessor=predecessor,
        preparation={},
        reconstruction=reconstruction,
        checkpoint={"authenticated": True},
        restoration=restoration,
        access=access,
        errors=[],
        written=True,
    )
    assert passed["passed"] is True
    denied = v54._report_summary(
        predecessor=predecessor,
        preparation={},
        reconstruction=reconstruction,
        checkpoint={"authenticated": True},
        restoration=restoration,
        access={**access, "oracle_loaded": True},
        errors=[],
        written=True,
    )
    assert denied["passed"] is False


def test_dirty_gate_publishes_no_checkpoint_and_report_failure_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predecessor = v54.authenticate_predecessor(v54.V53_REPORT_SHA256)
    reconstruction = dict(predecessor["candidate_reconstruction"])
    restoration = {
        "passed": True,
        "full_tensor_state_sha256": v51._SOURCE_FULL_SHA256,
        "authorized_surface_state_sha256": v51._SOURCE_AUTHORIZED_SHA256,
        "frozen_state_sha256": v51._FROZEN_SHA256,
        "all_parameter_gradients_absent": True,
    }
    clean_access = {
        "passed": True,
        "training_map_count": 16,
        "optimizer_file_reads": [],
        "forbidden_file_accesses": [],
        "validation_qa_loaded": False,
        "oracle_loaded": False,
        "final_test_loaded": False,
    }
    dirty = v54._report_summary(
        predecessor=predecessor,
        preparation={},
        reconstruction=reconstruction,
        checkpoint=None,
        restoration=restoration,
        access={**clean_access, "oracle_loaded": True},
        errors=[{"type": "RuntimeError", "message": "dirty access"}],
        written=False,
    )
    dirty_root = tmp_path / "dirty-checkpoint"
    dirty_root.mkdir()
    dirty_report = tmp_path / "dirty-report.json"
    v54._publish_report_or_rollback(
        report_path=dirty_report,
        checkpoint_root=dirty_root,
        report=dirty,
    )
    assert dirty_report.is_file()
    assert not dirty_root.exists()

    passing = v54._report_summary(
        predecessor=predecessor,
        preparation={},
        reconstruction=reconstruction,
        checkpoint={"authenticated": True},
        restoration=restoration,
        access=clean_access,
        errors=[],
        written=True,
    )
    passing_root = tmp_path / "passing-checkpoint"
    (passing_root / "update_000").mkdir(parents=True)

    def fail_write(_path: Path, _value: dict[str, object]) -> None:
        raise OSError("simulated publication failure")

    monkeypatch.setattr(v51, "_atomic_json", fail_write)
    with pytest.raises(OSError, match="simulated publication"):
        v54._publish_report_or_rollback(
            report_path=tmp_path / "passing-report.json",
            checkpoint_root=passing_root,
            report=passing,
        )
    assert not passing_root.exists()


def test_module_has_no_generation_optimizer_or_heldout_execution_imports() -> None:
    source = Path(v54.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "torch.optim" not in imported
    assert "_question_logits_and_answer" not in source
    assert ".evaluate_greedy(" not in source
    assert ".detailed_greedy(" not in source
    assert "load_optimizer_checkpoint" not in source
    assert "save_optimizer_checkpoint" not in source
    assert "data/oracle" not in source
