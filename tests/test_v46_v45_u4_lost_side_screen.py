from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest
import torch

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation import v45_retention_repair_terminal_gate as seal
from semantic_3d_chat.evaluation import v46_v45_u4_lost_side_screen as screen


def _surface(fill: float) -> dict[str, torch.Tensor]:
    return {
        name: torch.full(shape, fill, dtype=torch.float32)
        for name, shape in zip(screen._PARAMETER_NAMES, screen._PARAMETER_SHAPES)
    }


def _authorized_payload() -> dict:
    authorization = copy.deepcopy(seal._authorization())
    authorization["explicit_terminal_sha256_cli_required"] = True
    authorization["implementation_integrity"] = {
        "script_sha256": screen._sha256(screen._resolve(screen.V46_SCRIPT)),
        "test_sha256": screen._sha256(screen._resolve(screen.V46_TEST)),
        "config_sha256": screen._CONFIG_FILE_SHA256,
    }
    return {
        "schema_version": 1,
        "artifact": "v45_retention_repair_terminal_gate",
        "passed": True,
        "only_exact_successor_authorized": screen._AUTHORIZATION_ID,
        "conditional_successor_authorization": authorization,
    }


def test_candidate_grid_is_exact_nonadaptive_three_by_five() -> None:
    assert screen._DIRECTION_IDS == (
        "g5_scene_sign",
        "g5_query_sign",
        "g5_both_sign",
    )
    assert screen._ALPHA_GRID == (0.125, 0.25, 0.5, 1.0, 2.0)
    source = _surface(0.0)
    gradient = _surface(1.0)
    rows, candidates, inventory_hash = screen.build_candidate_inventory(source, source, gradient)
    assert len(rows) == len(candidates) == 15
    assert len(inventory_hash) == 64
    assert [(row["direction_id"], row["alpha"]) for row in rows] == [
        (direction, alpha) for direction in screen._DIRECTION_IDS for alpha in screen._ALPHA_GRID
    ]
    assert len({row["authorized_surface_state_sha256"] for row in rows}) == 15


def test_fresh_adam_sign_candidate_changes_only_its_fixed_groups() -> None:
    source = _surface(0.0)
    gradient = _surface(1.0)
    scene = screen.candidate_from_sign_line(
        source, gradient, direction_id="g5_scene_sign", alpha=0.5
    )
    assert torch.all(scene[screen._PARAMETER_NAMES[0]] == -0.5 * screen._SCENE_LR)
    assert torch.equal(scene[screen._PARAMETER_NAMES[1]], source[screen._PARAMETER_NAMES[1]])
    assert torch.equal(scene[screen._PARAMETER_NAMES[2]], source[screen._PARAMETER_NAMES[2]])
    query = screen.candidate_from_sign_line(
        source, gradient, direction_id="g5_query_sign", alpha=1.0
    )
    assert torch.equal(query[screen._PARAMETER_NAMES[0]], source[screen._PARAMETER_NAMES[0]])
    assert torch.all(query[screen._PARAMETER_NAMES[1]] == -screen._QUERY_LR)
    assert torch.all(query[screen._PARAMETER_NAMES[2]] == -screen._QUERY_LR)
    both = screen.candidate_from_sign_line(source, gradient, direction_id="g5_both_sign", alpha=2.0)
    assert torch.all(both[screen._PARAMETER_NAMES[0]] == -2.0 * screen._SCENE_LR)
    assert torch.all(both[screen._PARAMETER_NAMES[1]] == -2.0 * screen._QUERY_LR)
    assert torch.all(both[screen._PARAMETER_NAMES[2]] == -2.0 * screen._QUERY_LR)


def test_candidate_constructor_rejects_grid_shape_dtype_and_device_drift() -> None:
    source = _surface(0.0)
    gradient = _surface(1.0)
    with pytest.raises(ValueError, match="three-direction"):
        screen.candidate_from_sign_line(source, gradient, direction_id="adaptive", alpha=0.5)
    with pytest.raises(ValueError, match="five-value"):
        screen.candidate_from_sign_line(source, gradient, direction_id="g5_both_sign", alpha=0.75)
    wrong_shape = dict(source)
    wrong_shape[screen._PARAMETER_NAMES[0]] = torch.zeros(1, dtype=torch.float32)
    with pytest.raises(ValueError, match="source tensor changed"):
        screen.candidate_from_sign_line(
            wrong_shape, gradient, direction_id="g5_scene_sign", alpha=0.5
        )
    wrong_dtype = dict(gradient)
    wrong_dtype[screen._PARAMETER_NAMES[2]] = wrong_dtype[screen._PARAMETER_NAMES[2]].double()
    with pytest.raises(ValueError, match="gradient tensor changed"):
        screen.candidate_from_sign_line(
            source, wrong_dtype, direction_id="g5_query_sign", alpha=0.5
        )


def test_terminal_hash_is_explicit_and_exact_authorization_is_consumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _authorized_payload()
    terminal = tmp_path / "terminal.json"
    monkeypatch.setattr(screen, "DEFAULT_TERMINAL", terminal)
    report["conditional_successor_authorization"]["invocation_contract"]["terminal_path"] = str(
        terminal
    )
    terminal.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    observed = screen._sha256(terminal)
    result = screen.require_terminal(observed)
    assert result["sha256"] == observed
    assert result["authorization_id"] == screen._AUTHORIZATION_ID
    with pytest.raises(ValueError, match="differs from the explicit invocation"):
        screen.require_terminal("0" * 64)
    with pytest.raises(ValueError, match="64 lowercase hex"):
        screen.require_terminal("not-a-sha")


def test_authorization_rejects_adaptive_or_diagnostic_direction_broadening() -> None:
    report = _authorized_payload()
    authorization = report["conditional_successor_authorization"]
    bad = copy.deepcopy(authorization)
    bad["fresh_adam_sign_line"]["adaptive_direction_or_scalar_selection"] = True
    with pytest.raises(ValueError, match="authorization changed"):
        screen._validate_authorization(report, bad)
    bad = copy.deepcopy(authorization)
    bad["fresh_adam_sign_line"]["diagnostic_gradient_q699_and_q0a79_used_as_directions"] = True
    with pytest.raises(ValueError, match="authorization changed"):
        screen._validate_authorization(report, bad)


def test_source_uses_autograd_grad_but_no_training_selection_or_checkpoint_write() -> None:
    path = PROJECT_ROOT / screen.V46_SCRIPT
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "grad" in attributes
    assert "backward" not in attributes
    assert "step" not in attributes
    assert "torch.optim" not in source
    assert "load_optimizer_checkpoint" not in source
    assert "save_adapter_checkpoint" not in source
    assert "save_optimizer_checkpoint" not in source
    assert "training_greedy_metrics" not in source
    assert "candidate_rank_key" not in source
    assert "selected_candidate" not in source
    assert "--expected-v45-terminal-sha256" in source


def test_write_report_refuses_existing_output_before_live_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing.json"
    output.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(screen, "DEFAULT_OUTPUT", output)
    called = False

    def forbidden(**_kwargs: object) -> dict:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(screen, "run_screen", forbidden)
    with pytest.raises(FileExistsError, match="one-shot"):
        screen.write_report(output, expected_v45_terminal_sha256="0" * 64)
    assert called is False


def test_write_report_refuses_fresh_unauthorized_output_before_live_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def forbidden(**_kwargs: object) -> dict:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(screen, "run_screen", forbidden)
    with pytest.raises(ValueError, match="pinned"):
        screen.write_report(
            tmp_path / "unauthorized.json",
            expected_v45_terminal_sha256="0" * 64,
        )
    assert called is False


def test_materialized_terminal_and_preflight_are_train_only() -> None:
    terminal = screen._resolve(screen.DEFAULT_TERMINAL)
    if not terminal.is_file():
        pytest.skip("V45 terminal is intentionally materialized after V46 hash pinning")
    terminal_sha256 = screen._sha256(terminal)
    authenticated = screen.require_terminal(terminal_sha256)
    assert authenticated["authorization_id"] == screen._AUTHORIZATION_ID
    result = screen._preflight(expected_v45_terminal_sha256=terminal_sha256)
    assert result["passed"] is True
    assert result["candidate_grid"]["candidate_count"] == 15
    assert result["gemma_loaded"] is False
    assert result["scene_maps_loaded"] is False
    assert result["optimizer_constructed_or_loaded"] is False
    assert result["validation_qa_loaded"] is False
    assert result["oracle_loaded"] is False
    assert result["final_test_scenes_touched"] is False
    assert result["selector_executed"] is False
    assert result["runtime_promotion_executed"] is False
