from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from semantic_3d_chat.evaluation import v51_query_alpha_grid as v51
from semantic_3d_chat.evaluation import v52_query_alpha_refinement as v52


def _candidate_ids() -> list[str]:
    return [str(value["candidate_id"]) for value in v52.CANDIDATE_GRID]


def _pair_metrics(*, passes: bool = True) -> dict[str, Any]:
    units = [
        {
            "pair_id": pair_id,
            "question_key": key,
            "side_margins": margins,
            "cross_prefix_margins": [0.1, 0.2],
        }
        for pair_id, key, margins in (
            ("pair_000015", "cfq_163eb92339ad35a5", [0.17, 0.20]),
            ("pair_000016", "cfq_699675ceeaf65406", [0.50, 0.37]),
            ("pair_000006", "cfq_5c84a2c27d2be251", [0.06, 0.25]),
        )
    ]
    units.extend(
        {
            "pair_id": f"pair_{index:06d}",
            "question_key": f"cfq_test_{index:04d}",
            "side_margins": [0.5, 0.5],
            "cross_prefix_margins": [0.25, 0.25],
        }
        for index in range(3, 25)
    )
    return {
        "schema_version": 1,
        "unit_count": 25,
        "complete_units": 10 if passes else 9,
        "positive_sides": 35,
        "cross_prefix_complete_units": 18,
        "complete_physical_pair_coverage": 5,
        "complete_units_by_family": {
            "book_support": 1,
            "mirror_lr": 2,
            "picture_support": 1,
        },
        "cross_prefix_complete_units_by_family": {
            "book_support": 2,
            "mirror_lr": 4,
            "picture_support": 2,
        },
        "units": units,
    }


def _non_greedy(*, passes: bool = True) -> dict[str, Any]:
    prefix_hashes = {
        scene_id: f"{index + 1:064x}"
        for index, scene_id in enumerate(v51._TRAIN_SCENE_IDS)
    }
    return {
        "pair_metrics": _pair_metrics(passes=passes),
        "per_unit_nll_diagnostics": [
            {"pair_id": f"pair_{index:06d}", "nll": 1.0 + index / 100.0}
            for index in range(25)
        ],
        "broad_nll": 2.920,
        "broad_row_count": 48,
        "priority_side_deficit": 29.8,
        "retention_diagnostics": {"both_lost_sides_strictly_positive": True},
        "original_v46_candidate_relative_prefix_trust_rms": (
            v51._EXPECTED_ORIGINAL_PREFIX_RMS
        ),
        "candidate_prefix_sha256_by_train_scene": prefix_hashes,
        "candidate_prefix_hash_inventory_sha256": v51._canonical_sha256(prefix_hashes),
        "candidate_prefix_matches_all_prior_query_alphas": True,
        "candidate_prefix_scene_count": 16,
    }


def _greedy(*, passes: bool = True) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "changed_unit_count": 25,
        "changed_row_count": 50,
        "changed_rows_exact_correct": 30,
        "complete_units": 5 if passes else 4,
        "complete_physical_pair_coverage": 5,
        "complete_units_by_family": {
            "book_support": 1,
            "mirror_lr": 1,
            "picture_support": 1,
        },
        "broad_row_count": 48,
        "broad_exact_correct": 23 if passes else 22,
        "broad_exact_accuracy": (23 if passes else 22) / 48,
    }


class FakeBackend:
    def __init__(
        self,
        *,
        pre_pass_ids: set[str] | None = None,
        greedy_pass_ids: set[str] | None = None,
        access_passed: bool = True,
    ) -> None:
        all_ids = set(_candidate_ids())
        self.pre_pass_ids = all_ids if pre_pass_ids is None else pre_pass_ids
        self.greedy_pass_ids = all_ids if greedy_pass_ids is None else greedy_pass_ids
        self.access_passed = access_passed
        self.calls: list[str] = []

    def authenticate_and_prepare(self) -> dict[str, Any]:
        self.calls.append("prepare")
        return {"candidate_grid": list(v51.CANDIDATE_GRID)}

    def reconstruct_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        candidate_id = str(candidate["candidate_id"])
        self.calls.append(f"reconstruct:{candidate_id}")
        return {
            **candidate,
            "source_v47_u4_exact_before_reconstruction": True,
            "reconstructed_candidate_full_tensor_state_exact": True,
            "reconstructed_candidate_authorized_surface_state_exact": True,
            "scene_readout_state_sha256": "a" * 64,
            "scene_readout_state_changed": True,
            "query_state_changed": True,
            "frozen_state_sha256": v51._FROZEN_SHA256,
        }

    def evaluate_non_greedy(self, candidate_id: str) -> dict[str, Any]:
        self.calls.append(f"non_greedy:{candidate_id}")
        return _non_greedy(passes=candidate_id in self.pre_pass_ids)

    def evaluate_greedy(self, candidate_id: str) -> dict[str, Any]:
        self.calls.append(f"greedy:{candidate_id}")
        return _greedy(passes=candidate_id in self.greedy_pass_ids)

    def restore_source(self) -> dict[str, Any]:
        self.calls.append("restore")
        return {
            "passed": True,
            "full_tensor_state_sha256": v51._SOURCE_FULL_SHA256,
            "authorized_surface_state_sha256": v51._SOURCE_AUTHORIZED_SHA256,
            "frozen_state_sha256": v51._FROZEN_SHA256,
        }

    def stage_checkpoint(
        self,
        directory: Path,
        candidate: dict[str, Any],
        provenance: dict[str, Any],
    ) -> dict[str, Any]:
        candidate_id = str(candidate["candidate_id"])
        self.calls.append(f"stage:{candidate_id}")
        assert provenance["authorization_id"] == v52.AUTHORIZATION_ID
        (directory / "adapter.safetensors").write_bytes(b"adapter")
        (directory / "metadata.json").write_text("{}\n", encoding="utf-8")
        (directory / "runtime_metadata.json").write_text("{}\n", encoding="utf-8")
        return {"candidate_state_authenticated": True}

    def access_audit(self) -> dict[str, Any]:
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
    return {
        "path": str(v52.V51_REPORT),
        "sha256": v52.V51_REPORT_SHA256,
        "checks": {"authenticated": True},
    }


def test_fixed_boundary_grid_is_exact_and_excludes_seen_anchors() -> None:
    assert v52.QUERY_ALPHAS == (2.03125, 2.0625, 2.125, 2.1875)
    assert [row["query_alpha"] for row in v52.CANDIDATE_GRID] == list(
        v52.QUERY_ALPHAS
    )
    assert [row["scene_alpha"] for row in v52.CANDIDATE_GRID] == [1.0] * 4
    assert [row["declared_order"] for row in v52.CANDIDATE_GRID] == [0, 1, 2, 3]
    assert 2.0 not in v52.QUERY_ALPHAS
    assert 2.25 not in v52.QUERY_ALPHAS


def test_exact_v51_report_authenticates_and_wrong_hash_fails() -> None:
    result = v52.authenticate_predecessor(v52.V51_REPORT_SHA256)
    assert result["sha256"] == v52.V51_REPORT_SHA256
    assert all(result["checks"].values())
    with pytest.raises(ValueError, match="pinned V51 report"):
        v52.authenticate_predecessor("0" * 64)


def test_scoped_v51_parameterization_restores_every_global() -> None:
    fields = (
        "AUTHORIZATION_ID",
        "_QUERY_ALPHAS",
        "CANDIDATE_GRID",
        "DEFAULT_REPORT",
        "DEFAULT_CHECKPOINT_ROOT",
        "DEFAULT_CHECKPOINT",
    )
    before = {name: getattr(v51, name) for name in fields}
    with v52.scoped_v51_refinement():
        assert v51.AUTHORIZATION_ID == v52.AUTHORIZATION_ID
        assert v51._QUERY_ALPHAS == v52.QUERY_ALPHAS
        assert v51.CANDIDATE_GRID == v52.CANDIDATE_GRID
    assert {name: getattr(v51, name) for name in fields} == before


def test_complete_grid_precedes_selection_and_checkpoint_is_optimizer_free(
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    checkpoint = tmp_path / "update_000"
    report = v52.execute_refinement_gate(
        predecessor=_predecessor(), backend=backend, checkpoint_path=checkpoint
    )
    ids = _candidate_ids()
    assert report["artifact"] == v52.AUTHORIZATION_ID
    assert report["candidate_grid"]["evaluated_ids"] == ids
    assert report["selection"]["winner"]["candidate_id"] == ids[0]
    assert backend.calls.index(f"stage:{ids[0]}") > backend.calls.index(
        f"greedy:{ids[-1]}"
    )
    assert report["checkpoint"]["written"] is True
    assert not (checkpoint / "optimizer.pt").exists()
    assert v51.AUTHORIZATION_ID != v52.AUTHORIZATION_ID


def test_greedy_runs_iff_own_pre_gate_and_grid_does_not_short_circuit(
    tmp_path: Path,
) -> None:
    ids = _candidate_ids()
    backend = FakeBackend(
        pre_pass_ids={ids[1], ids[2], ids[3]},
        greedy_pass_ids={ids[2], ids[3]},
    )
    report = v52.execute_refinement_gate(
        predecessor=_predecessor(),
        backend=backend,
        checkpoint_path=tmp_path / "update_000",
    )
    rows = report["candidate_grid"]["candidates"]
    assert [row["greedy_gate"]["executed"] for row in rows] == [
        False,
        True,
        True,
        True,
    ]
    assert report["selection"]["winner"]["candidate_id"] == ids[2]
    assert [row["candidate"]["candidate_id"] for row in rows] == ids


def test_no_winner_or_failed_access_writes_no_checkpoint(tmp_path: Path) -> None:
    ids = set(_candidate_ids())
    no_winner_path = tmp_path / "none" / "update_000"
    no_winner = v52.execute_refinement_gate(
        predecessor=_predecessor(),
        backend=FakeBackend(greedy_pass_ids=set()),
        checkpoint_path=no_winner_path,
    )
    assert no_winner["selection"]["winner"] is None
    assert no_winner["checkpoint"]["written"] is False
    assert not no_winner_path.exists()

    denied_path = tmp_path / "denied" / "update_000"
    denied = v52.execute_refinement_gate(
        predecessor=_predecessor(),
        backend=FakeBackend(greedy_pass_ids=ids, access_passed=False),
        checkpoint_path=denied_path,
    )
    assert denied["selection"]["winner"] is not None
    assert denied["checkpoint"]["written"] is False
    assert not denied_path.exists()


def test_real_preflight_is_model_free_or_refuses_completed_one_shot_and_paths_are_pinned(
    tmp_path: Path,
) -> None:
    if v52._resolve(v52.DEFAULT_REPORT).exists():
        with pytest.raises(FileExistsError, match="one-shot"):
            v52.preflight(expected_v51_report_sha256=v52.V51_REPORT_SHA256)
    else:
        result = v52.preflight(expected_v51_report_sha256=v52.V51_REPORT_SHA256)
        assert result["passed"] is True
        assert result["candidate_count"] == 4
        assert result["model_loaded"] is False
        assert result["qa_loaded"] is False
        assert result["maps_loaded"] is False
    with pytest.raises(ValueError, match="paths are pinned"):
        v52.run_grid(
            expected_v51_report_sha256=v52.V51_REPORT_SHA256,
            paths=v52.RefinementPaths(report=tmp_path / "other.json"),
        )


def test_report_publication_failure_rolls_back_new_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_root = tmp_path / "checkpoint"
    (checkpoint_root / "update_000").mkdir(parents=True)
    report = {
        "artifact": v52.AUTHORIZATION_ID,
        "authorization": {"predecessor_sha256": v52.V51_REPORT_SHA256},
        "refinement": {
            "query_alpha_grid_declared_order": list(v52.QUERY_ALPHAS)
        },
        "checkpoint": {"written": True},
    }

    def fail_write(_path: Path, _value: dict[str, Any]) -> None:
        raise OSError("simulated publication failure")

    monkeypatch.setattr(v51, "_atomic_json", fail_write)
    with pytest.raises(OSError, match="simulated"):
        v52._publish_report_or_rollback(
            report_path=tmp_path / "report.json",
            checkpoint_root=checkpoint_root,
            report=report,
        )
    assert not checkpoint_root.exists()


def test_module_has_no_optimizer_selector_or_heldout_execution_imports() -> None:
    source = Path(v52.__file__).read_text(encoding="utf-8")
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
    assert "optimizer.pt\").read" not in source
    assert "selector." not in source
    assert "data/oracle" not in source
