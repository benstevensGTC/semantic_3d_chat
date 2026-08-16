from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from semantic_3d_chat.evaluation import v51_query_alpha_grid as v51
from semantic_3d_chat.evaluation import v53_v52_greedy_failure_diagnostic as v53


def _teacher_units() -> list[dict[str, Any]]:
    result = []
    for index in range(25):
        pair_id = (
            "pair_000016"
            if index < 2
            else "pair_000005"
            if index == 2
            else "pair_000007"
            if index == 3
            else "pair_000015"
            if index == 4
            else f"pair_{index + 100:06d}"
        )
        result.append(
            {
                "pair_id": pair_id,
                "question_key": f"cfq_test_{index:04d}",
                "scene_ids": [f"scene_{index * 2 + 11:06d}", f"scene_{index * 2 + 12:06d}"],
                "family": v53._family(pair_id),
                "side_margins": [0.25, 0.25],
                "cross_prefix_margins": [0.1, 0.2],
                "complete": index < 10,
                "cross_prefix_complete": True,
            }
        )
    return result


def _pair_metrics() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "unit_count": 25,
        "complete_units": 10,
        "positive_sides": 35,
        "cross_prefix_complete_units": 18,
        "complete_physical_pair_coverage": 5,
        "complete_units_by_family": {
            "book_support": 1,
            "mirror_lr": 2,
            "picture_support": 1,
        },
        "cross_prefix_complete_units_by_family": {
            "book_support": 1,
            "mirror_lr": 4,
            "picture_support": 2,
        },
        "units": _teacher_units(),
    }


def _non_greedy() -> dict[str, Any]:
    hashes = {
        scene_id: f"{index + 1:064x}"
        for index, scene_id in enumerate(v51._TRAIN_SCENE_IDS)
    }
    return {
        "pair_metrics": _pair_metrics(),
        "per_unit_nll_diagnostics": [
            {"pair_id": f"pair_{index:06d}", "nll": 1.0}
            for index in range(25)
        ],
        "broad_nll": 2.9172720114390054,
        "broad_row_count": 48,
        "priority_side_deficit": 29.67397975921631,
        "retention_diagnostics": {"both_lost_sides_strictly_positive": True},
        "original_v46_candidate_relative_prefix_trust_rms": (
            v51._EXPECTED_ORIGINAL_PREFIX_RMS
        ),
        "candidate_prefix_sha256_by_train_scene": hashes,
        "candidate_prefix_hash_inventory_sha256": v51._canonical_sha256(hashes),
        "candidate_prefix_matches_all_prior_query_alphas": True,
        "candidate_prefix_scene_count": 16,
    }


def _reconstruction() -> dict[str, Any]:
    return {
        **v53.TARGET_SPEC,
        "source_v47_u4_exact_before_reconstruction": True,
        "reconstructed_candidate_full_tensor_state_exact": True,
        "reconstructed_candidate_authorized_surface_state_exact": True,
        "scene_readout_state_changed": True,
        "query_state_changed": True,
        "frozen_state_sha256": v51._FROZEN_SHA256,
    }


def _detailed_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    units: list[dict[str, Any]] = []
    for index, teacher in enumerate(_teacher_units()):
        legacy = [True, True] if index < 4 else [True, False] if index < 20 else [False, False]
        type_aware = list(legacy)
        rescued = index == 4
        if rescued:
            type_aware[1] = True
        legacy_count = sum(legacy)
        aware_count = sum(type_aware)
        classify = lambda count: (
            "complete_success"
            if count == 2
            else "one_sided_failure"
            if count == 1
            else "complete_failure"
        )
        sides = []
        for side_index in range(2):
            sides.append(
                {
                    "side_index": side_index,
                    "scene_id": teacher["scene_ids"][side_index],
                    "question_id": f"q_{index:06x}{side_index}",
                    "expected_normalized_answer": "red blue",
                    "generated_normalized_answer": (
                        "blue red" if rescued and side_index == 1 else "red blue"
                    ),
                    "answer_type": "list" if rescued else "spatial_relation",
                    "legacy_exact_correct": legacy[side_index],
                    "type_aware_correct": type_aware[side_index],
                    "reordered_list_rescue": rescued and side_index == 1,
                    "teacher_side_margin": 0.25,
                    "teacher_cross_prefix_margin": 0.1,
                    "teacher_side_positive": True,
                }
            )
        units.append(
            {
                "pair_id": teacher["pair_id"],
                "question_key": teacher["question_key"],
                "scene_ids": teacher["scene_ids"],
                "family": teacher["family"],
                "teacher_complete": teacher["complete"],
                "teacher_cross_prefix_complete": True,
                "legacy_greedy_complete": legacy_count == 2,
                "type_aware_greedy_complete": aware_count == 2,
                "legacy_failure_classification": classify(legacy_count),
                "type_aware_failure_classification": classify(aware_count),
                "sides": sides,
            }
        )
    broad = []
    for index in range(48):
        legacy = index < 23
        rescue = index == 23
        broad.append(
            {
                "scene_id": f"scene_{index % 16 + 11:06d}",
                "question_id": f"q_broad{index:06x}",
                "expected_normalized_answer": "red blue",
                "generated_normalized_answer": "blue red" if rescue else "red blue",
                "answer_type": "support" if rescue else "presence",
                "legacy_exact_correct": legacy,
                "type_aware_correct": legacy or rescue,
                "reordered_list_rescue": rescue,
            }
        )
    return units, broad


def _detail() -> dict[str, Any]:
    units, broad = _detailed_rows()
    return {
        "summary": v53.summarize_detailed_rows(units=units, broad_rows=broad),
        "pair_units": units,
        "broad_rows": broad,
    }


class FakeBackend:
    def __init__(self, *, access_passed: bool = True) -> None:
        self.access_passed = access_passed
        self.calls: list[str] = []

    def authenticate_and_prepare(self) -> Mapping[str, Any]:
        self.calls.append("prepare")
        return {"candidate_grid": [dict(value) for value in v53.v52.CANDIDATE_GRID]}

    def reconstruct_candidate(self, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls.append(f"reconstruct:{candidate['candidate_id']}")
        assert dict(candidate) == v53.TARGET_SPEC
        return _reconstruction()

    def evaluate_non_greedy(self, candidate_id: str) -> Mapping[str, Any]:
        self.calls.append(f"non_greedy:{candidate_id}")
        return _non_greedy()

    def detailed_greedy(
        self, candidate_id: str, teacher_metrics: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        self.calls.append(f"detail:{candidate_id}")
        assert teacher_metrics["unit_count"] == 25
        return _detail()

    def restore_source(self) -> Mapping[str, Any]:
        self.calls.append("restore")
        return {
            "passed": True,
            "full_tensor_state_sha256": v51._SOURCE_FULL_SHA256,
            "authorized_surface_state_sha256": v51._SOURCE_AUTHORIZED_SHA256,
            "frozen_state_sha256": v51._FROZEN_SHA256,
            "all_parameter_gradients_absent": True,
        }

    def access_audit(self) -> Mapping[str, Any]:
        self.calls.append("access")
        return {
            "passed": self.access_passed,
            "training_map_count": 16,
            "optimizer_file_reads": [],
            "forbidden_file_accesses": [],
            "validation_qa_loaded": False,
            "oracle_loaded": False,
            "final_test_loaded": False,
        }

    def close(self) -> None:
        self.calls.append("close")


def _predecessor() -> dict[str, Any]:
    non_greedy = _non_greedy()
    return {
        "path": str(v53.V52_REPORT),
        "sha256": v53.V52_REPORT_SHA256,
        "checks": {"authenticated": True},
        "target": {
            "recorded_reconstruction": _reconstruction(),
            "recorded_non_greedy": {
                "evidence": {
                    "pair_metrics_sha256": v53._canonical_sha256(
                        non_greedy["pair_metrics"]
                    ),
                    "per_unit_nll_sha256": v53._canonical_sha256(
                        non_greedy["per_unit_nll_diagnostics"]
                    ),
                    "broad_nll": non_greedy["broad_nll"],
                }
            },
        },
    }


def test_exact_target_and_predecessor_contract() -> None:
    assert v53.TARGET_SPEC == {
        "candidate_id": "guarded_scene_alpha_1p0_query_alpha_2p0625",
        "declared_order": 1,
        "scene_alpha": 1.0,
        "query_alpha": 2.0625,
    }
    result = v53.authenticate_predecessor(v53.V52_REPORT_SHA256)
    assert result["target"]["candidate"] == v53.TARGET_SPEC
    assert all(result["checks"].values())
    with pytest.raises(ValueError, match="pinned V52 report"):
        v53.authenticate_predecessor("0" * 64)


def test_dual_scoring_reproduces_v52_and_records_list_rescues() -> None:
    units, broad = _detailed_rows()
    summary = v53.summarize_detailed_rows(units=units, broad_rows=broad)
    assert summary["complete_units"] == 4
    assert summary["changed_rows_exact_correct"] == 24
    assert summary["broad_exact_correct"] == 23
    assert summary["legacy_exact"]["complete_units"] == 4
    assert summary["type_aware"]["complete_units"] == 5
    assert summary["reordered_list_rescue_rows"] == 1
    assert summary["reordered_list_rescue_units"] == 1
    assert summary["broad_reordered_list_rescue_rows"] == 1


def test_execute_reconstructs_only_target_and_is_report_only() -> None:
    backend = FakeBackend()
    report = v53.execute_diagnostic(predecessor=_predecessor(), backend=backend)
    assert report["passed"] is True
    assert backend.calls == [
        "prepare",
        f"reconstruct:{v53.TARGET_CANDIDATE_ID}",
        f"non_greedy:{v53.TARGET_CANDIDATE_ID}",
        f"detail:{v53.TARGET_CANDIDATE_ID}",
        "restore",
        "access",
        "close",
    ]
    assert report["scope"]["only_one_candidate_reconstructed"] is True
    assert report["checkpoint_written"] is False
    assert report["optimizer_constructed_or_loaded"] is False


def test_access_failure_fails_closed_without_checkpoint() -> None:
    report = v53.execute_diagnostic(
        predecessor=_predecessor(), backend=FakeBackend(access_passed=False)
    )
    assert report["passed"] is False
    assert report["checkpoint_written"] is False
    assert report["access_audit"]["passed"] is False


def test_model_free_preflight_or_completed_one_shot_and_paths_are_pinned(
    tmp_path: Path,
) -> None:
    if v53._resolve(v53.DEFAULT_REPORT).exists():
        with pytest.raises(FileExistsError, match="one-shot"):
            v53.preflight(expected_v52_report_sha256=v53.V52_REPORT_SHA256)
    else:
        result = v53.preflight(expected_v52_report_sha256=v53.V52_REPORT_SHA256)
        assert result["passed"] is True
        assert result["model_loaded"] is False
        assert result["qa_loaded"] is False
        assert result["maps_loaded"] is False
    with pytest.raises(ValueError, match="paths are pinned"):
        v53.run_diagnostic(
            expected_v52_report_sha256=v53.V52_REPORT_SHA256,
            paths=v53.DiagnosticPaths(report=tmp_path / "other.json"),
        )


def test_module_has_no_optimizer_checkpoint_or_heldout_execution_imports() -> None:
    source = Path(v53.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "torch.optim" not in imported
    assert "load_optimizer_checkpoint" not in source
    assert "save_optimizer_checkpoint" not in source
    assert "stage_checkpoint(" not in source
    assert "data/oracle" not in source
