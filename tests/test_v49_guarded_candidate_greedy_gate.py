from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from semantic_3d_chat.evaluation import v48_dual_margin_terminal_gate as v48_terminal
from semantic_3d_chat.evaluation import v49_guarded_candidate_greedy_gate as gate


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


def _non_greedy(*, complete_units: int = 10, prefix_rms: float = 0.001) -> dict[str, Any]:
    return {
        "pair_metrics": _pair_metrics(complete_units=complete_units),
        "per_unit_nll_diagnostics": [
            {"pair_id": f"pair_{index:06d}", "nll": 1.0 + index / 100.0} for index in range(25)
        ],
        "broad_nll": 2.920,
        "broad_row_count": 48,
        "priority_side_deficit": 29.8,
        "retention_diagnostics": {"both_lost_sides_strictly_positive": True},
        "original_v46_candidate_relative_prefix_trust_rms": prefix_rms,
    }


def _reconstruction() -> dict[str, Any]:
    return {
        "candidate_id": gate._CANDIDATE_ID,
        "source_v47_u4_exact_before_reconstruction": True,
        "full_tensor_state_sha256": gate._CANDIDATE_FULL_SHA256,
        "authorized_surface_state_sha256": gate._CANDIDATE_AUTHORIZED_SHA256,
        "frozen_state_sha256": gate._FROZEN_SHA256,
        "scene_readout_state_changed": True,
        "query_state_changed": True,
    }


def _greedy(*, complete_units: int = 5, broad_correct: int = 23) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "changed_unit_count": 25,
        "changed_row_count": 50,
        "changed_rows_exact_correct": 30,
        "complete_units": complete_units,
        "complete_physical_pair_coverage": 5,
        "complete_units_by_family": {
            "book_support": 1,
            "mirror_lr": 1,
            "picture_support": 1,
        },
        "broad_row_count": 48,
        "broad_exact_correct": broad_correct,
        "broad_exact_accuracy": broad_correct / 48,
    }


class FakeBackend:
    def __init__(
        self,
        *,
        non_greedy: dict[str, Any] | None = None,
        greedy: dict[str, Any] | None = None,
        restoration_passed: bool = True,
        access_passed: bool = True,
        reconstruction_error: Exception | None = None,
    ) -> None:
        self.non_greedy = _non_greedy() if non_greedy is None else non_greedy
        self.greedy = _greedy() if greedy is None else greedy
        self.restoration_passed = restoration_passed
        self.access_passed = access_passed
        self.reconstruction_error = reconstruction_error
        self.calls: list[str] = []

    def authenticate_and_reconstruct(self) -> dict[str, Any]:
        self.calls.append("reconstruct")
        if self.reconstruction_error is not None:
            raise self.reconstruction_error
        return _reconstruction()

    def evaluate_non_greedy(self) -> dict[str, Any]:
        self.calls.append("non_greedy")
        return copy.deepcopy(self.non_greedy)

    def evaluate_greedy(self) -> dict[str, Any]:
        self.calls.append("greedy")
        return copy.deepcopy(self.greedy)

    def stage_checkpoint(self, directory: Path, provenance: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("stage")
        assert provenance["candidate_id"] == gate._CANDIDATE_ID
        (directory / "adapter.safetensors").write_bytes(b"adapter")
        (directory / "metadata.json").write_text("{}\n", encoding="utf-8")
        (directory / "runtime_metadata.json").write_text("{}\n", encoding="utf-8")
        return {"candidate_state_authenticated": True}

    def restore_source(self) -> dict[str, Any]:
        self.calls.append("restore")
        return {
            "passed": self.restoration_passed,
            "full_tensor_state_sha256": gate._SOURCE_FULL_SHA256,
            "authorized_surface_state_sha256": gate._SOURCE_AUTHORIZED_SHA256,
            "frozen_state_sha256": gate._FROZEN_SHA256,
        }

    def access_audit(self) -> dict[str, Any]:
        self.calls.append("access")
        return {
            "passed": self.access_passed,
            "optimizer_file_reads": [],
            "forbidden_file_accesses": [],
        }

    def close(self) -> None:
        self.calls.append("close")


def _terminal() -> dict[str, Any]:
    return {"path": str(gate.V48_TERMINAL), "sha256": "a" * 64, "checks": {"all": True}}


def test_pre_gate_failure_skips_greedy_and_writes_no_checkpoint(tmp_path: Path) -> None:
    backend = FakeBackend(non_greedy=_non_greedy(complete_units=9))
    checkpoint = tmp_path / "update_000"
    report = gate.execute_staged_gate(
        terminal=_terminal(), backend=backend, checkpoint_path=checkpoint
    )
    assert report["passed"] is False
    assert report["non_greedy_pre_gate"]["passed"] is False
    assert report["greedy_gate"] == {
        "authorized": False,
        "executed": False,
        "greedy_skipped_due_pre_gate": True,
        "checks": {},
        "passed": False,
        "evidence": None,
    }
    assert backend.calls == ["reconstruct", "non_greedy", "restore", "access", "close"]
    assert report["checkpoint"]["written"] is False
    assert not checkpoint.exists()


def test_pre_gate_pass_makes_full_greedy_mandatory_but_greedy_failure_writes_nothing(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(greedy=_greedy(complete_units=4))
    checkpoint = tmp_path / "update_000"
    report = gate.execute_staged_gate(
        terminal=_terminal(), backend=backend, checkpoint_path=checkpoint
    )
    assert report["non_greedy_pre_gate"]["passed"] is True
    assert report["greedy_gate"]["authorized"] is True
    assert report["greedy_gate"]["executed"] is True
    assert report["greedy_gate"]["passed"] is False
    assert "stage" not in backend.calls
    assert report["passed"] is False
    assert not checkpoint.exists()


def test_checkpoint_is_staged_then_source_restored_then_atomically_published(
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    checkpoint = tmp_path / "update_000"
    report = gate.execute_staged_gate(
        terminal=_terminal(), backend=backend, checkpoint_path=checkpoint
    )
    assert report["passed"] is True
    assert backend.calls == [
        "reconstruct",
        "non_greedy",
        "greedy",
        "stage",
        "restore",
        "access",
        "close",
    ]
    assert report["checkpoint"]["written"] is True
    assert report["checkpoint"]["write_iff_final_gate_passed"] is True
    assert sorted(path.name for path in checkpoint.iterdir()) == [
        "adapter.safetensors",
        "metadata.json",
        "runtime_metadata.json",
    ]
    assert not (checkpoint / "optimizer.pt").exists()


def test_failed_source_restoration_discards_staged_checkpoint(tmp_path: Path) -> None:
    backend = FakeBackend(restoration_passed=False)
    checkpoint = tmp_path / "update_000"
    report = gate.execute_staged_gate(
        terminal=_terminal(), backend=backend, checkpoint_path=checkpoint
    )
    assert "stage" in backend.calls
    assert report["final_train_gate"]["source_restored_exact"] is False
    assert report["passed"] is False
    assert report["checkpoint"]["written"] is False
    assert not checkpoint.exists()
    assert not list(tmp_path.glob(".update_000.staged.*"))


def test_reconstruction_exception_is_reported_after_restore_attempt(tmp_path: Path) -> None:
    backend = FakeBackend(reconstruction_error=RuntimeError("synthetic failure"))
    report = gate.execute_staged_gate(
        terminal=_terminal(),
        backend=backend,
        checkpoint_path=tmp_path / "update_000",
    )
    assert report["passed"] is False
    assert report["final_train_gate"]["execution_error"] == {
        "type": "RuntimeError",
        "message": "synthetic failure",
    }
    assert backend.calls == ["reconstruct", "restore", "access", "close"]


def test_exact_v48_authorization_template_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_sha = gate._sha256(gate._resolve(gate.V49_SCRIPT))
    test_sha = gate._sha256(gate._resolve(gate.V49_TEST))
    monkeypatch.setattr(v48_terminal, "_V49_SCRIPT_SHA256", script_sha)
    monkeypatch.setattr(v48_terminal, "_V49_TEST_SHA256", test_sha)
    authorization = v48_terminal.v49_authorization_template()
    report = {
        "artifact": "v48_v47_u4_dual_margin_terminal_gate",
        "passed": True,
        "terminal_materialization_authorized": True,
        "only_exact_successor_authorized": gate.AUTHORIZATION_ID,
        "conditional_successor_authorization": authorization,
    }
    checks = gate._validate_terminal_authorization(report, authorization)
    assert all(checks.values())

    changed = copy.deepcopy(authorization)
    changed["measurements"]["non_greedy_pre_gate_evaluated_first"] = False
    with pytest.raises(ValueError, match="terminal authorization changed"):
        gate._validate_terminal_authorization(report, changed)


def test_require_terminal_uses_explicit_sha_and_rejects_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    terminal_path = tmp_path / "terminal.json"
    monkeypatch.setattr(gate, "V48_TERMINAL", terminal_path)
    monkeypatch.setattr(
        v48_terminal,
        "_V49_SCRIPT_SHA256",
        gate._sha256(gate._resolve(gate.V49_SCRIPT)),
    )
    monkeypatch.setattr(
        v48_terminal,
        "_V49_TEST_SHA256",
        gate._sha256(gate._resolve(gate.V49_TEST)),
    )
    authorization = v48_terminal.v49_authorization_template()
    authorization["invocation_contract"]["terminal_path"] = str(terminal_path)
    payload = {
        "artifact": "v48_v47_u4_dual_margin_terminal_gate",
        "passed": True,
        "terminal_materialization_authorized": True,
        "only_exact_successor_authorized": gate.AUTHORIZATION_ID,
        "conditional_successor_authorization": authorization,
    }
    terminal_path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    digest = hashlib.sha256(terminal_path.read_bytes()).hexdigest()
    result = gate.require_terminal(digest, terminal_path)
    assert result["sha256"] == digest
    with pytest.raises(ValueError, match="differs from explicit"):
        gate.require_terminal("0" * 64, terminal_path)


def test_preflight_authenticates_files_without_opening_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("seed: 1\n", encoding="utf-8")
    report = tmp_path / "report.json"
    checkpoint = tmp_path / "candidate" / "update_000"
    v48_report = tmp_path / "v48.json"
    v48_report.write_text("{}\n", encoding="utf-8")
    protected = tmp_path / "protected.json"
    protected.write_text("{}\n", encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    for name in gate._SOURCE_FILES:
        (source / name).write_bytes(name.encode())
    prefix = tmp_path / "prefix"
    prefix.mkdir()
    for name in gate._PREFIX_REFERENCE_FILES:
        (prefix / name).write_bytes(name.encode())

    monkeypatch.setattr(gate, "DEFAULT_CONFIG", config)
    monkeypatch.setattr(gate, "DEFAULT_REPORT", report)
    monkeypatch.setattr(gate, "DEFAULT_CHECKPOINT", checkpoint)
    monkeypatch.setattr(gate, "V48_REPORT", v48_report)
    monkeypatch.setattr(gate, "PROTECTED_REPORT", protected)
    monkeypatch.setattr(gate, "SOURCE_CHECKPOINT", source)
    monkeypatch.setattr(gate, "PREFIX_REFERENCE_CHECKPOINT", prefix)
    monkeypatch.setattr(gate, "_CONFIG_SHA256", gate._sha256(config))
    monkeypatch.setattr(gate, "_V48_REPORT_SHA256", gate._sha256(v48_report))
    monkeypatch.setattr(gate, "_PROTECTED_REPORT_SHA256", gate._sha256(protected))
    monkeypatch.setattr(
        gate,
        "_SOURCE_FILES",
        {name: gate._sha256(source / name) for name in gate._SOURCE_FILES},
    )
    monkeypatch.setattr(
        gate,
        "_PREFIX_REFERENCE_FILES",
        {name: gate._sha256(prefix / name) for name in gate._PREFIX_REFERENCE_FILES},
    )
    monkeypatch.setattr(
        gate,
        "require_terminal",
        lambda *_args, **_kwargs: {
            "path": "terminal.json",
            "sha256": "a" * 64,
            "checks": {"all": True},
        },
    )
    opened: list[str] = []
    real_locked_hash = gate._locked_hash

    def tracked(path: Path, expected: str, field: str) -> None:
        opened.append(path.name)
        real_locked_hash(path, expected, field)

    monkeypatch.setattr(gate, "_locked_hash", tracked)
    result = gate.preflight(
        expected_v48_terminal_sha256="a" * 64,
        paths=gate.GatePaths(
            terminal=gate.V48_TERMINAL,
            report=report,
            checkpoint=checkpoint,
            config=config,
        ),
    )
    assert result["passed"] is True
    assert result["model_loaded"] is False
    assert result["qa_loaded"] is False
    assert result["maps_loaded"] is False
    assert result["source"]["optimizer_file_opened"] is False
    assert "optimizer.pt" not in opened


def test_production_backend_constructor_is_inert() -> None:
    backend = gate.RealGateBackend(_terminal(), gate.GatePaths())
    assert backend._prepared is False
    assert backend._audit is None
    assert backend._bundle is None
    backend.close()


def test_production_paths_are_pinned_before_terminal_or_model_access(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="paths are pinned"):
        gate.run_gate(
            expected_v48_terminal_sha256="0" * 64,
            paths=gate.GatePaths(report=tmp_path / "other.json"),
        )


def test_module_contains_no_optimizer_or_selector_execution_imports() -> None:
    tree = ast.parse(Path(gate.__file__).read_text(encoding="utf-8"))
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    source = Path(gate.__file__).read_text(encoding="utf-8")
    assert "torch.optim" not in imported_names
    assert "load_optimizer_checkpoint" not in source
    assert "save_optimizer_checkpoint" not in source
    assert "selector." not in source
