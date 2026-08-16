from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation import v43_aggregate_projected_screen as screen


def test_terminal_authorizes_exact_aggregate_projection_screen() -> None:
    terminal = screen.require_terminal()
    auth = terminal["authorization"]
    assert terminal["sha256"] == screen._TERMINAL_SHA256
    assert auth["only_exact_action"] == (
        "one_v43_aggregate_projected_response_screen"
    )
    assert auth["gradient_surface"]["broad_component"] == (
        "mean_48_unchanged_rows_times_1"
    )
    assert auth["gradient_surface"]["cross_component"] == (
        "mean_all_25_pair_cross_hinge_times_56"
    )
    assert auth["projection"]["fixed_scalar_steps"] == list(screen._STEPS)
    assert auth["diagnostic_scope"]["optimizer_step_authorized"] is False
    assert auth["diagnostic_scope"]["checkpoint_write_authorized"] is False


def test_candidate_formula_uses_positive_step_as_descent() -> None:
    source = torch.zeros((4096, 4), dtype=torch.float32)
    direction = torch.ones_like(source)
    assert torch.equal(
        screen.candidate_from_direction(source, direction, 0.0), source
    )
    positive = screen.candidate_from_direction(source, direction, 0.004)
    negative = screen.candidate_from_direction(source, direction, -0.004)
    assert torch.all(positive < 0)
    assert torch.all(negative > 0)
    with pytest.raises(ValueError, match="fixed grid"):
        screen.candidate_from_direction(source, direction, 0.003)


def test_candidate_ranking_uses_the_v43_step_grid() -> None:
    def row(step: float, complete: int) -> dict:
        return {
            "scalar_step": step,
            "pair_metrics": {
                "complete_units": complete,
                "positive_sides": 34,
                "cross_prefix_complete_units": 17,
                "complete_units_by_family": {
                    "book_support": 1,
                    "mirror_lr": 1,
                    "picture_support": 1,
                },
            },
            "priority_side_deficit": 30.0,
            "broad_nll": 2.9,
        }

    candidates = [row(-0.008, 9), row(0.004, 10), row(0.002, 10)]
    selected = min(candidates, key=screen.candidate_rank_key)
    assert selected["scalar_step"] == 0.002
    with pytest.raises(ValueError, match="fixed grid"):
        screen.candidate_rank_key(row(0.003, 10))


def test_bundle_state_attestation_is_exact_and_gradient_sensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(2, 2)
    bundle = SimpleNamespace(
        checkpoint_modules={"model": model},
        language=SimpleNamespace(model=model),
    )
    monkeypatch.setattr(screen, "target_v41_state_sha256", lambda _bundle: "target")
    monkeypatch.setattr(
        screen, "module_collection_state_sha256", lambda _modules: "full"
    )
    monkeypatch.setattr(screen, "frozen_v41_state_sha256", lambda _bundle: "frozen")
    clean = screen.bundle_state_attestation(
        bundle,
        expected_target_sha256="target",
        expected_full_sha256="full",
        expected_frozen_sha256="frozen",
    )
    assert clean["passed"] is True
    model.weight.grad = torch.zeros_like(model.weight)
    dirty = screen.bundle_state_attestation(
        bundle,
        expected_target_sha256="target",
        expected_full_sha256="full",
        expected_frozen_sha256="frozen",
    )
    assert dirty["all_gradients_absent"] is False
    assert dirty["passed"] is False


def test_cpu_aggregate_is_float32_mean_and_count_locked() -> None:
    values = [
        torch.full((4096, 4), float(index), dtype=torch.float32)
        for index in range(1, 4)
    ]
    aggregate = screen._aggregate_cpu_gradients(
        values, expected_count=3, name="test"
    )
    assert aggregate.dtype == torch.float32
    assert aggregate.device.type == "cpu"
    assert torch.equal(aggregate, torch.full_like(aggregate, 2.0))
    with pytest.raises(RuntimeError, match="count"):
        screen._aggregate_cpu_gradients(values, expected_count=4, name="test")


def test_preflight_uses_no_model_maps_optimizer_or_restricted_data() -> None:
    result = screen._screen_preflight()
    assert result["passed"] is True
    assert result["fixed_scalar_steps"] == list(screen._STEPS)
    assert result["train_question_count"] == 384
    assert result["changed_train_unit_count"] == 25
    assert result["gemma_loaded"] is False
    assert result["scene_maps_loaded"] is False
    assert result["optimizer_loaded_or_constructed"] is False
    assert result["validation_qa_loaded"] is False
    assert result["oracle_loaded"] is False
    assert result["final_test_scenes_touched"] is False
    assert result["forbidden_file_accesses"] == []


def test_source_has_autograd_grad_but_no_optimizer_backward_or_checkpoint_api() -> None:
    path = (
        PROJECT_ROOT
        / "src/semantic_3d_chat/evaluation/v43_aggregate_projected_screen.py"
    )
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "grad" in attributes
    assert "backward" not in attributes
    assert "step" not in attributes
    assert "torch.optim" not in source
    assert "save_adapter_checkpoint" not in source
    assert "save_optimizer_checkpoint" not in source


def test_write_report_refuses_existing_output_before_live_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing.json"
    output.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(screen, "DEFAULT_OUTPUT", output)
    called = False

    def forbidden(_config: object) -> dict:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(screen, "run_screen", forbidden)
    with pytest.raises(FileExistsError, match="one-shot"):
        screen.write_report(output)
    assert called is False


def test_write_report_refuses_fresh_unauthorized_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def forbidden(_config: object) -> dict:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(screen, "run_screen", forbidden)
    with pytest.raises(ValueError, match="pinned"):
        screen.write_report(tmp_path / "unauthorized.json")
    assert called is False
