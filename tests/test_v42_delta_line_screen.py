from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation import v42_delta_line_screen as screen


def _endpoints() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    root = PROJECT_ROOT / screen.CHECKPOINT_ROOT
    return (
        load_file(root / "update_000/adapter.safetensors", device="cpu"),
        load_file(root / "update_008/adapter.safetensors", device="cpu"),
    )


def test_terminal_revision2_authorizes_only_fixed_temporary_screen() -> None:
    terminal = screen.require_terminal()
    authorization = terminal["authorization"]
    scope = authorization["diagnostic_scope"]
    assert terminal["sha256"] == screen._TERMINAL_SHA256
    assert authorization["only_exact_action"] == (
        "one_train_only_no_step_diagnostic_screen"
    )
    assert scope["temporary_target_b_substitution_authorized"] is True
    assert scope["fixed_alpha_grid"] == list(screen._ALPHAS)
    assert scope["fixed_candidate_state_sha256"] == dict(
        screen._CANDIDATE_HASHES
    )
    assert scope["gradient_measurement_authorized"] is False
    assert scope["optimizer_step_authorized"] is False
    assert scope["checkpoint_write_authorized"] is False


def test_candidate_grid_and_full_envelopes_are_exactly_prehashed() -> None:
    source, stopped = _endpoints()
    candidates, audit = screen.build_candidate_inventory(source, stopped)
    assert tuple(candidates) == screen._ALPHAS
    assert audit["fixed_alpha_grid"] == list(screen._ALPHAS)
    assert audit["endpoint_zero_exact_clone"] is True
    assert audit["endpoint_one_exact_clone"] is True
    assert torch.equal(candidates[0.0], source[screen._TARGET])
    assert torch.equal(candidates[1.0], stopped[screen._TARGET])
    assert [row["target_state_sha256"] for row in audit["candidate_rows"]] == [
        screen._CANDIDATE_HASHES[screen.alpha_key(alpha)]["target"]
        for alpha in screen._ALPHAS
    ]


def test_candidate_constructor_rejects_unsealed_alpha_or_shape() -> None:
    source, stopped = _endpoints()
    with pytest.raises(ValueError, match="sealed grid"):
        screen.candidate_tensor(source[screen._TARGET], stopped[screen._TARGET], 2.0)
    with pytest.raises(ValueError, match="shape"):
        screen.candidate_tensor(torch.zeros(2), torch.ones(2), 0.0)


def test_teacher_ranking_is_deterministic_and_uses_declared_order() -> None:
    def row(alpha: float, complete: int, family: int, positive: int, cross: int):
        families = {
            "book_support": int(family >= 1),
            "mirror_lr": int(family >= 2),
            "picture_support": int(family >= 3),
        }
        return {
            "alpha": alpha,
            "priority_side_deficit": 30.0,
            "broad_nll": 2.9,
            "pair_metrics": {
                "complete_units": complete,
                "complete_units_by_family": families,
                "positive_sides": positive,
                "cross_prefix_complete_units": cross,
            },
        }

    rows = [
        row(-1.0, 9, 3, 34, 17),
        row(-0.5, 10, 1, 34, 17),
        row(-0.25, 10, 2, 34, 17),
    ]
    assert min(rows, key=screen.candidate_rank_key)["alpha"] == -0.25


def test_preflight_loads_no_model_map_or_restricted_data() -> None:
    result = screen.preflight()
    assert result["passed"] is True
    assert result["train_question_count"] == 384
    assert result["changed_train_unit_count"] == 25
    assert result["gemma_loaded"] is False
    assert result["scene_maps_loaded"] is False
    assert result["validation_qa_loaded"] is False
    assert result["oracle_loaded"] is False
    assert result["final_test_scenes_touched"] is False
    assert result["optimizer_loaded_or_constructed"] is False
    assert result["forbidden_file_accesses"] == []
    assert not any("/maps/" in path for path in result["loaded_files"])


def test_screen_source_contains_no_optimizer_gradient_or_checkpoint_api() -> None:
    path = PROJECT_ROOT / "src/semantic_3d_chat/evaluation/v42_delta_line_screen.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "backward" not in attributes
    assert "step" not in attributes
    assert "autograd" not in source
    assert "torch.optim" not in source
    assert "save_adapter_checkpoint" not in source
    assert "save_optimizer_checkpoint" not in source


def test_write_report_refuses_to_overwrite_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing.json"
    output.write_text("{}\n", encoding="utf-8")
    called = False

    def forbidden_run(_config: object) -> dict:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(screen, "run_screen", forbidden_run)
    with pytest.raises(FileExistsError, match="one-shot"):
        screen.write_report(output)
    assert called is False
