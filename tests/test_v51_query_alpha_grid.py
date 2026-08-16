from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any

import pytest

from semantic_3d_chat.evaluation import v51_query_alpha_grid as grid


def _pair_metrics(*, complete_units: int = 10) -> dict[str, Any]:
    focus = [
        ("pair_000015", "cfq_163eb92339ad35a5", [0.17, 0.20]),
        ("pair_000016", "cfq_699675ceeaf65406", [0.50, 0.37]),
        ("pair_000006", "cfq_5c84a2c27d2be251", [0.06, 0.25]),
    ]
    units = [
        {
            "pair_id": pair_id,
            "question_key": key,
            "side_margins": margins,
            "cross_prefix_margins": [0.1, 0.2],
        }
        for pair_id, key, margins in focus
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
        "complete_units": complete_units,
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
        for index, scene_id in enumerate(grid._TRAIN_SCENE_IDS)
    }
    return {
        "pair_metrics": _pair_metrics(complete_units=10 if passes else 9),
        "per_unit_nll_diagnostics": [
            {"pair_id": f"pair_{index:06d}", "nll": 1.0 + index / 100.0}
            for index in range(25)
        ],
        "broad_nll": 2.920,
        "broad_row_count": 48,
        "priority_side_deficit": 29.8,
        "retention_diagnostics": {"both_lost_sides_strictly_positive": True},
        "original_v46_candidate_relative_prefix_trust_rms": (
            grid._EXPECTED_ORIGINAL_PREFIX_RMS
        ),
        "candidate_prefix_sha256_by_train_scene": prefix_hashes,
        "candidate_prefix_hash_inventory_sha256": grid._canonical_sha256(prefix_hashes),
        "candidate_prefix_matches_all_prior_query_alphas": True,
        "candidate_prefix_scene_count": 16,
    }


def _greedy(*, passes: bool = True) -> dict[str, Any]:
    complete = 5 if passes else 4
    broad = 23 if passes else 22
    return {
        "schema_version": 1,
        "changed_unit_count": 25,
        "changed_row_count": 50,
        "changed_rows_exact_correct": 30,
        "complete_units": complete,
        "complete_physical_pair_coverage": 5,
        "complete_units_by_family": {
            "book_support": 1,
            "mirror_lr": 1,
            "picture_support": 1,
        },
        "broad_row_count": 48,
        "broad_exact_correct": broad,
        "broad_exact_accuracy": broad / 48,
    }


def _reconstruction(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        **spec,
        "source_v47_u4_exact_before_reconstruction": True,
        "full_tensor_state_sha256": f"{int(spec['declared_order']) + 1:064x}",
        "authorized_surface_state_sha256": f"{int(spec['declared_order']) + 4:064x}",
        "scene_readout_state_sha256": "a" * 64,
        "query_state_sha256": f"{int(spec['declared_order']) + 16:064x}",
        "frozen_state_sha256": grid._FROZEN_SHA256,
        "reconstructed_candidate_full_tensor_state_exact": True,
        "reconstructed_candidate_authorized_surface_state_exact": True,
        "scene_readout_state_changed": True,
        "query_state_changed": True,
        "reconstructed_directly_from_v47_u4": True,
    }


class FakeBackend:
    def __init__(
        self,
        *,
        pre_pass_ids: set[str] | None = None,
        greedy_pass_ids: set[str] | None = None,
        restoration_passed: bool = True,
        access_passed: bool = True,
    ) -> None:
        all_ids = {str(value["candidate_id"]) for value in grid.CANDIDATE_GRID}
        self.pre_pass_ids = all_ids if pre_pass_ids is None else pre_pass_ids
        self.greedy_pass_ids = all_ids if greedy_pass_ids is None else greedy_pass_ids
        self.restoration_passed = restoration_passed
        self.access_passed = access_passed
        self.calls: list[str] = []

    def authenticate_and_prepare(self) -> dict[str, Any]:
        self.calls.append("prepare")
        return {
            "candidate_grid": [dict(value) for value in grid.CANDIDATE_GRID],
            "all_16_training_maps_cached": True,
        }

    def reconstruct_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        candidate_id = str(candidate["candidate_id"])
        self.calls.append(f"reconstruct:{candidate_id}")
        return _reconstruction(dict(candidate))

    def evaluate_non_greedy(self, candidate_id: str) -> dict[str, Any]:
        self.calls.append(f"non_greedy:{candidate_id}")
        return _non_greedy(passes=candidate_id in self.pre_pass_ids)

    def evaluate_greedy(self, candidate_id: str) -> dict[str, Any]:
        self.calls.append(f"greedy:{candidate_id}")
        return _greedy(passes=candidate_id in self.greedy_pass_ids)

    def restore_source(self) -> dict[str, Any]:
        self.calls.append("restore")
        return {
            "passed": self.restoration_passed,
            "full_tensor_state_sha256": grid._SOURCE_FULL_SHA256,
            "authorized_surface_state_sha256": grid._SOURCE_AUTHORIZED_SHA256,
            "frozen_state_sha256": grid._FROZEN_SHA256,
        }

    def stage_checkpoint(
        self,
        directory: Path,
        candidate: dict[str, Any],
        provenance: dict[str, Any],
    ) -> dict[str, Any]:
        candidate_id = str(candidate["candidate_id"])
        self.calls.append(f"stage:{candidate_id}")
        assert provenance["winner"]["candidate_id"] == candidate_id
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


def _terminal() -> dict[str, Any]:
    return {"path": str(grid.V50_TERMINAL), "sha256": "a" * 64, "checks": {"all": True}}


def _candidate_ids() -> list[str]:
    return [str(value["candidate_id"]) for value in grid.CANDIDATE_GRID]


def test_fixed_grid_inventory_and_declared_order() -> None:
    assert [value["scene_alpha"] for value in grid.CANDIDATE_GRID] == [1.0] * 4
    assert [value["query_alpha"] for value in grid.CANDIDATE_GRID] == [
        1.75,
        2.25,
        1.5,
        2.5,
    ]
    assert [value["declared_order"] for value in grid.CANDIDATE_GRID] == [0, 1, 2, 3]
    assert len(set(_candidate_ids())) == 4
    assert grid._NONSELECTABLE_QUERY_ALPHA_ANCHOR == 2.0
    assert 2.0 not in [value["query_alpha"] for value in grid.CANDIDATE_GRID]
    assert grid._SCENE_LR == 1.0e-5
    assert grid._QUERY_LR == 8.0e-6


def test_split_scaling_reconstructs_each_candidate_directly_from_source() -> None:
    import torch

    from semantic_3d_chat.training.train_joint_scene_readout_v44 import (
        _PARAMETER_NAMES,
        _PARAMETER_SHAPES,
    )

    source = {
        name: torch.full(shape, 0.5, dtype=torch.float32)
        for name, shape in zip(_PARAMETER_NAMES, _PARAMETER_SHAPES)
    }
    direction = {
        name: torch.ones(shape, dtype=torch.float32)
        for name, shape in zip(_PARAMETER_NAMES, _PARAMETER_SHAPES)
    }
    candidates = [
        grid.candidate_from_split_alphas(
            source,
            direction,
            scene_alpha=float(spec["scene_alpha"]),
            query_alpha=float(spec["query_alpha"]),
        )
        for spec in grid.CANDIDATE_GRID
    ]
    assert torch.equal(source[_PARAMETER_NAMES[0]], torch.full_like(source[_PARAMETER_NAMES[0]], 0.5))
    for candidate, query_alpha in zip(candidates, grid._QUERY_ALPHAS):
        scene_delta = source[_PARAMETER_NAMES[0]] - candidate[_PARAMETER_NAMES[0]]
        assert torch.allclose(
            scene_delta,
            torch.full_like(scene_delta, grid._SCENE_ALPHA * grid._SCENE_LR),
            rtol=0.0,
            atol=1e-7,
        )
        for name in _PARAMETER_NAMES[1:]:
            query_delta = source[name] - candidate[name]
            assert torch.allclose(
                query_delta,
                torch.full_like(query_delta, query_alpha * grid._QUERY_LR),
                rtol=0.0,
                atol=1e-7,
            )
    assert all(
        torch.equal(candidates[0][_PARAMETER_NAMES[0]], candidate[_PARAMETER_NAMES[0]])
        for candidate in candidates[1:]
    )
    assert not torch.equal(candidates[0][_PARAMETER_NAMES[1]], candidates[1][_PARAMETER_NAMES[1]])


def test_complete_grid_is_evaluated_before_first_passing_candidate_is_selected(
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    checkpoint = tmp_path / "update_000"
    report = grid.execute_grid_gate(
        terminal=_terminal(), backend=backend, checkpoint_path=checkpoint
    )
    ids = _candidate_ids()
    assert report["candidate_grid"]["evaluated_ids"] == ids
    assert report["candidate_grid"]["complete_fixed_grid_evaluated_before_selection"] is True
    assert report["selection"]["winner"]["candidate_id"] == ids[0]
    assert backend.calls.index(f"stage:{ids[0]}") > backend.calls.index(f"greedy:{ids[3]}")
    assert report["passed"] is True
    assert report["checkpoint"]["written"] is True
    assert sorted(path.name for path in checkpoint.iterdir()) == [
        "adapter.safetensors",
        "metadata.json",
        "runtime_metadata.json",
    ]
    assert not (checkpoint / "optimizer.pt").exists()


def test_greedy_is_conditional_for_each_candidate_but_grid_never_short_circuits(
    tmp_path: Path,
) -> None:
    ids = _candidate_ids()
    backend = FakeBackend(
        pre_pass_ids={ids[1], ids[2], ids[3]}, greedy_pass_ids={ids[2], ids[3]}
    )
    report = grid.execute_grid_gate(
        terminal=_terminal(), backend=backend, checkpoint_path=tmp_path / "update_000"
    )
    rows = report["candidate_grid"]["candidates"]
    assert [row["non_greedy_pre_gate"]["evaluated"] for row in rows] == [
        True,
        True,
        True,
        True,
    ]
    assert [row["greedy_gate"]["executed"] for row in rows] == [
        False,
        True,
        True,
        True,
    ]
    assert f"greedy:{ids[0]}" not in backend.calls
    assert report["selection"]["winner"]["candidate_id"] == ids[2]


def test_winner_is_first_full_pass_in_declared_order_not_best_or_last(tmp_path: Path) -> None:
    ids = _candidate_ids()
    backend = FakeBackend(greedy_pass_ids={ids[1], ids[2]})
    report = grid.execute_grid_gate(
        terminal=_terminal(), backend=backend, checkpoint_path=tmp_path / "update_000"
    )
    assert report["selection"]["passing_candidate_ids"] == [ids[1], ids[2]]
    assert report["selection"]["winner"]["candidate_id"] == ids[1]
    assert f"stage:{ids[1]}" in backend.calls
    assert f"stage:{ids[2]}" not in backend.calls


def test_no_full_winner_writes_zero_checkpoint(tmp_path: Path) -> None:
    backend = FakeBackend(greedy_pass_ids=set())
    checkpoint = tmp_path / "update_000"
    report = grid.execute_grid_gate(
        terminal=_terminal(), backend=backend, checkpoint_path=checkpoint
    )
    assert report["selection"]["winner"] is None
    assert report["passed"] is False
    assert report["checkpoint"]["written"] is False
    assert not checkpoint.exists()
    assert not list(tmp_path.glob(".update_000.staged.*"))


def test_each_candidate_and_final_state_are_restored_and_access_audited(
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    report = grid.execute_grid_gate(
        terminal=_terminal(), backend=backend, checkpoint_path=tmp_path / "update_000"
    )
    assert backend.calls.count("restore") == 5
    assert backend.calls[-3:] == ["restore", "access", "close"]
    assert all(
        row["source_restoration"]["passed"]
        for row in report["candidate_grid"]["candidates"]
    )
    assert report["final_source_restoration"]["passed"] is True
    assert report["access_audit"]["passed"] is True


def test_failed_restoration_or_access_audit_discards_staged_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "update_000"
    backend = FakeBackend(access_passed=False)
    report = grid.execute_grid_gate(
        terminal=_terminal(), backend=backend, checkpoint_path=checkpoint
    )
    assert report["selection"]["winner"] is not None
    assert report["checkpoint"]["written"] is False
    assert report["passed"] is False
    assert not checkpoint.exists()
    assert not list(tmp_path.glob(".update_000.staged.*"))


def test_query_only_scene_and_prefix_invariance_is_mandatory(tmp_path: Path) -> None:
    class ChangedPrefixBackend(FakeBackend):
        def evaluate_non_greedy(self, candidate_id: str) -> dict[str, Any]:
            evidence = super().evaluate_non_greedy(candidate_id)
            if candidate_id == _candidate_ids()[2]:
                hashes = dict(evidence["candidate_prefix_sha256_by_train_scene"])
                hashes[grid._TRAIN_SCENE_IDS[0]] = "f" * 64
                evidence["candidate_prefix_sha256_by_train_scene"] = hashes
                evidence["candidate_prefix_hash_inventory_sha256"] = (
                    grid._canonical_sha256(hashes)
                )
            return evidence

    checkpoint = tmp_path / "update_000"
    report = grid.execute_grid_gate(
        terminal=_terminal(),
        backend=ChangedPrefixBackend(),
        checkpoint_path=checkpoint,
    )
    invariant = report["query_only_scene_prefix_invariance"]
    assert invariant["all_candidate_scene_tensors_bit_identical"] is True
    assert invariant["all_candidate_scene_prefixes_bit_identical"] is False
    assert invariant["passed"] is False
    assert report["selection"]["winner"] is None
    assert report["checkpoint"]["written"] is False
    assert not checkpoint.exists()


def test_exact_v50_anchor_prefix_rms_is_required(tmp_path: Path) -> None:
    class ChangedRmsBackend(FakeBackend):
        def evaluate_non_greedy(self, candidate_id: str) -> dict[str, Any]:
            evidence = super().evaluate_non_greedy(candidate_id)
            if candidate_id == _candidate_ids()[1]:
                evidence["original_v46_candidate_relative_prefix_trust_rms"] += 1e-6
            return evidence

    report = grid.execute_grid_gate(
        terminal=_terminal(),
        backend=ChangedRmsBackend(),
        checkpoint_path=tmp_path / "update_000",
    )
    assert report["query_only_scene_prefix_invariance"][
        "all_candidate_prefix_rms_matches_v50_anchor"
    ] is False
    assert report["passed"] is False
    assert report["checkpoint"]["written"] is False


def test_exact_terminal_authorization_is_accepted_and_mutation_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from semantic_3d_chat.evaluation import (
        v50_scene_query_alpha_terminal_gate as terminal,
    )

    monkeypatch.setattr(
        terminal,
        "_V51_SCRIPT_SHA256",
        grid._sha256(grid._resolve(grid.V51_SCRIPT)),
    )
    monkeypatch.setattr(
        terminal,
        "_V51_TEST_SHA256",
        grid._sha256(grid._resolve(grid.V51_TEST)),
    )
    authorization = terminal.v51_authorization_template()
    assert authorization["authorized"] is True
    assert authorization["candidate_grid"]["candidates_declared_order"] == [
        dict(value) for value in grid.CANDIDATE_GRID
    ]
    assert authorization["per_candidate_gate"]["non_greedy_check_names"] == list(
        grid._NON_GREEDY_CHECK_NAMES
    )
    assert authorization["per_candidate_gate"]["greedy_check_names"] == list(
        grid._GREEDY_CHECK_NAMES
    )
    report = {
        "artifact": "v50_scene_query_alpha_terminal_gate",
        "passed": True,
        "terminal_materialization_authorized": True,
        "only_exact_successor_authorized": grid.AUTHORIZATION_ID,
        "conditional_successor_authorization": authorization,
    }
    checks = grid._validate_terminal_authorization(report, authorization)
    assert all(checks.values())

    changed = copy.deepcopy(authorization)
    changed["candidate_grid"]["query_alpha_grid_declared_order"] = [
        1.5,
        1.75,
        2.25,
        2.5,
    ]
    with pytest.raises(ValueError, match="terminal authorization changed"):
        grid._validate_terminal_authorization(report, changed)

    changed = copy.deepcopy(authorization)
    changed["scope"]["validation_access_authorized"] = True
    with pytest.raises(ValueError, match="terminal authorization changed"):
        grid._validate_terminal_authorization(report, changed)


def test_real_backend_constructor_is_inert() -> None:
    backend = grid.RealGridBackend(_terminal(), grid.GridPaths())
    assert backend._prepared is False
    assert backend._delegate._prepared is False
    assert backend._delegate._bundle is None
    backend.close()


def test_real_backend_reuses_v49_three_probes_without_second_gradient_diagnostic() -> None:
    source = Path(grid.__file__).read_text(encoding="utf-8")
    assert source.count("self._delegate.authenticate_and_reconstruct()") == 1
    assert "_gradient_diagnostics" not in source
    assert (
        "exact_three_autograd_grad_probes_reused_for_all_candidates" in source
    )


def test_per_candidate_check_names_match_terminal_contract_exactly() -> None:
    checks = grid.non_greedy_pre_gate_checks(
        _reconstruction(dict(grid.CANDIDATE_GRID[0])), _non_greedy()
    )
    assert list(checks) == list(grid._NON_GREEDY_CHECK_NAMES)
    assert list(grid.greedy_final_gate_checks(_greedy())) == list(grid._GREEDY_CHECK_NAMES)


def test_production_paths_are_pinned_before_terminal_or_model_access(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="paths are pinned"):
        grid.run_grid(
            expected_v50_terminal_sha256="0" * 64,
            paths=grid.GridPaths(report=tmp_path / "other.json"),
        )


def test_module_has_no_optimizer_or_selector_execution_imports() -> None:
    source = Path(grid.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "torch.optim" not in imported_names
    assert "load_optimizer_checkpoint" not in source
    assert "save_optimizer_checkpoint" not in source
    assert "optimizer.pt\").read" not in source
    assert "selector." not in source
