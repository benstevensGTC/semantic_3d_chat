from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation import v48_v47_u4_dual_margin_screen as screen


def _surface(fill: float) -> dict[str, torch.Tensor]:
    return {
        name: torch.full(shape, fill, dtype=torch.float32)
        for name, shape in zip(screen._PARAMETER_NAMES, screen._PARAMETER_SHAPES)
    }


def _gradients() -> dict[str, dict[str, torch.Tensor]]:
    book = _surface(1.0)
    mirror: dict[str, torch.Tensor] = {}
    guard: dict[str, torch.Tensor] = {}
    for name, shape in zip(screen._PARAMETER_NAMES, screen._PARAMETER_SHAPES):
        count = int(torch.tensor(shape).prod())
        alternating = torch.where(
            torch.arange(count) % 2 == 0,
            torch.tensor(1.0),
            torch.tensor(-1.0),
        ).reshape(shape)
        mirror[name] = alternating.float()
        guard[name] = torch.where(alternating > 0, -torch.ones_like(alternating), alternating)
    return {"g_book": book, "g_mirror": mirror, "g5_guard": guard}


def _terminal_payload() -> dict[str, Any]:
    authorization = {
        "authorization_id": screen._AUTHORIZATION_ID,
        "authorized": True,
        "only_exact_action": ("one_bounded_read_only_v48_train_checkpoint_dual_margin_diagnostic"),
        "authorized_script": str(screen.V48_SCRIPT),
        "authorized_test": str(screen.V48_TEST),
        "authorized_report": str(screen.DEFAULT_OUTPUT),
        "authorized_config": str(screen.DEFAULT_CONFIG),
        "explicit_terminal_sha256_cli_required": True,
        "invocation_contract": {
            "terminal_path": str(screen.DEFAULT_TERMINAL),
            "required_cli_argument": "--expected-v47-terminal-sha256",
            "v48_must_not_embed_terminal_sha256": True,
        },
        "implementation_integrity": {
            "script_sha256": screen._sha256(screen._resolve(screen.V48_SCRIPT)),
            "test_sha256": screen._sha256(screen._resolve(screen.V48_TEST)),
            "config_sha256": screen.v47._CONFIG_FILE_SHA256,
        },
        "source": {
            "checkpoint": str(screen.DEFAULT_SOURCE),
            "file_sha256": dict(screen._SOURCE_FILES),
            "full_tensor_state_sha256": screen._SOURCE_FULL_SHA256,
            "authorized_surface_state_sha256": screen._SOURCE_AUTHORIZED_SHA256,
            "frozen_state_sha256": screen._SOURCE_FROZEN_SHA256,
            "optimizer_file_open_authorized": False,
        },
        "measurements": {
            "isolated_side_gradient_specs": screen._expected_gradient_specs(),
            "normalize_each_nonzero_component": (
                "unit_l2_within_each_scene_or_query_group_before_combination"
            ),
            "report_raw_norms_and_pairwise_cosines_by_group": True,
        },
        "candidate_grid": {
            "direction_ids": list(screen._DIRECTION_IDS),
            "alpha_grid": list(screen._ALPHA_GRID),
            "candidate_formula": ("float32_P0-alpha*lr_group*sign(normalized_component_sum)"),
            "scene_readout_learning_rate": screen._SCENE_LR,
            "query_learning_rate": screen._QUERY_LR,
            "exact_candidate_count": 15,
            "full_25_unit_teacher_metrics_per_candidate": True,
            "full_fixed_48_row_broad_nll_per_candidate": True,
            "candidate_relative_prefix_trust_per_candidate": True,
            "exact_source_restoration_before_and_after_every_probe": True,
            "prehash_all_candidates_before_candidate_forward": True,
        },
        "scope": {
            "train_only": True,
            "report_only_output": True,
            "candidate_checkpoint_write_authorized": False,
            "optimizer_construction_or_step_authorized": False,
            "candidate_selection_authorized": False,
            "greedy_generation_authorized": False,
            "validation_access_authorized": False,
            "oracle_access_authorized": False,
            "final_test_access_authorized": False,
            "selector_execution_authorized": False,
            "runtime_promotion_authorized": False,
        },
    }
    return {
        "schema_version": 1,
        "artifact": "v47_book_continuation_terminal_gate",
        "passed": True,
        "v47_final_train_only_gate_passed": False,
        "only_exact_successor_authorized": screen._AUTHORIZATION_ID,
        "conditional_successor_authorization": authorization,
    }


def _gate_metrics(*, lost_positive: bool = True) -> dict[str, Any]:
    required = {
        "cfq_a578dc166be9a217": ("pair_000005", "other"),
        "cfq_0a79d507273195ef": ("pair_000006", "other"),
        "cfq_5c84a2c27d2be251": ("pair_000006", "other"),
        "cfq_736067b51ce93c49": ("pair_000007", "other"),
        "cfq_997610c185204121": ("pair_000007", "other"),
        "cfq_699675ceeaf65406": ("pair_000016", "mirror_lr"),
        "cfq_90b3d9852a93ce2a": ("pair_000018", "other"),
        "cfq_13b1138d14c52a7c": ("pair_000015", "book_support"),
        "cfq_a1c673a1197a0961": ("pair_000015", "book_support"),
    }
    rows = [
        {
            "pair_id": pair_id,
            "question_key": key,
            "family": family,
            "side_margins": [0.5, 0.5],
            "cross_prefix_margins": [0.5, 0.5],
        }
        for key, (pair_id, family) in required.items()
    ]
    for index in range(16):
        family = "book_support" if index < 2 else "picture_support" if index < 6 else "other"
        rows.append(
            {
                "pair_id": f"pair_fill_{index:02d}",
                "question_key": f"cfq_fill_{index:02d}",
                "family": family,
                "side_margins": [0.5, 0.5],
                "cross_prefix_margins": [0.5, 0.5],
            }
        )
    if not lost_positive:
        next(row for row in rows if row["question_key"] == "cfq_699675ceeaf65406")[
            "side_margins"
        ] = [0.5, 0.0]
    return {
        "unit_count": 25,
        "complete_units": 10,
        "positive_sides": 35,
        "cross_prefix_complete_units": 17,
        "complete_physical_pair_coverage": 5,
        "complete_units_by_family": {
            "book_support": 1,
            "mirror_lr": 2,
            "picture_support": 0,
        },
        "cross_prefix_complete_units_by_family": {
            "book_support": 1,
            "mirror_lr": 2,
            "picture_support": 0,
        },
        "units": rows,
    }


def test_fixed_gradient_direction_and_candidate_grid_contract() -> None:
    assert screen._GRADIENT_SPECS == (
        ("g_book", "pair_000015", "cfq_163eb92339ad35a5", 0),
        ("g_mirror", "pair_000016", "cfq_699675ceeaf65406", 1),
        ("g5_guard", "pair_000006", "cfq_5c84a2c27d2be251", 0),
    )
    assert screen._DIRECTION_IDS == (
        "dual_query_sign",
        "dual_both_sign",
        "guarded_both_sign",
    )
    assert screen._ALPHA_GRID == (0.125, 0.25, 0.5, 1.0, 2.0)
    assert screen._SCENE_LR == 1.0e-5
    assert screen._QUERY_LR == 8.0e-6


def test_each_gradient_component_is_unit_l2_normalized_within_each_group() -> None:
    gradients = _gradients()
    normalized, raw_norms = screen.normalize_gradient_components_by_group(gradients)
    for gradient_id in gradients:
        for group in ("scene_readout", "query"):
            assert raw_norms[gradient_id][group] > 0.0
            assert screen._group_l2(normalized[gradient_id], group) == pytest.approx(
                1.0, abs=1.0e-6
            )
    geometry = screen.gradient_geometry(gradients)
    assert set(geometry["groups"]) == {"scene_readout", "query"}
    for group in geometry["groups"].values():
        assert len(group["pairwise_cosines"]) == 3
        assert all(
            row["raw_cosine"] == pytest.approx(row["normalized_cosine"])
            for row in group["pairwise_cosines"].values()
        )


def test_normalized_directions_use_exact_components_and_query_only_mask() -> None:
    gradients = _gradients()
    normalized, _norms = screen.normalize_gradient_components_by_group(gradients)
    directions, audit = screen.build_normalized_directions(gradients)
    scene_name = screen._PARAMETER_NAMES[0]
    assert torch.count_nonzero(directions["dual_query_sign"][scene_name]).item() == 0
    for name in screen._PARAMETER_NAMES[1:]:
        expected = normalized["g_book"][name] + normalized["g_mirror"][name]
        assert torch.equal(directions["dual_query_sign"][name], expected)
        assert torch.equal(directions["dual_both_sign"][name], expected)
    for name in screen._PARAMETER_NAMES:
        expected = (
            normalized["g_book"][name] + normalized["g_mirror"][name] + normalized["g5_guard"][name]
        )
        assert torch.equal(directions["guarded_both_sign"][name], expected)
    assert audit["inactive_scene_group_exact_zero_for_dual_query_sign"] is True


def test_candidate_grid_is_prehashed_nonadaptive_three_by_five() -> None:
    source = _surface(0.0)
    directions, _audit = screen.build_normalized_directions(_gradients())
    rows, candidates, inventory_hash = screen.build_candidate_inventory(source, source, directions)
    assert len(rows) == len(candidates) == 15
    assert len(inventory_hash) == 64
    assert [(row["direction_id"], row["alpha"]) for row in rows] == [
        (direction, alpha) for direction in screen._DIRECTION_IDS for alpha in screen._ALPHA_GRID
    ]


def test_fresh_adam_sign_candidate_changes_only_fixed_active_groups() -> None:
    source = _surface(0.0)
    directions, _audit = screen.build_normalized_directions(_gradients())
    query = screen.candidate_from_normalized_direction(
        source,
        directions["dual_query_sign"],
        direction_id="dual_query_sign",
        alpha=0.5,
    )
    assert torch.equal(query[screen._PARAMETER_NAMES[0]], source[screen._PARAMETER_NAMES[0]])
    for name in screen._PARAMETER_NAMES[1:]:
        expected = -0.5 * screen._QUERY_LR * torch.sign(directions["dual_query_sign"][name])
        assert torch.equal(query[name], expected)
    both = screen.candidate_from_normalized_direction(
        source,
        directions["guarded_both_sign"],
        direction_id="guarded_both_sign",
        alpha=2.0,
    )
    assert torch.equal(
        both[screen._PARAMETER_NAMES[0]],
        -2.0
        * screen._SCENE_LR
        * torch.sign(directions["guarded_both_sign"][screen._PARAMETER_NAMES[0]]),
    )


def test_candidate_constructor_rejects_direction_alpha_shape_and_dtype_drift() -> None:
    source = _surface(0.0)
    directions, _audit = screen.build_normalized_directions(_gradients())
    with pytest.raises(ValueError, match="outside the fixed grid"):
        screen.candidate_from_normalized_direction(
            source, directions["dual_both_sign"], direction_id="adaptive", alpha=0.5
        )
    with pytest.raises(ValueError, match="outside the fixed grid"):
        screen.candidate_from_normalized_direction(
            source,
            directions["dual_both_sign"],
            direction_id="dual_both_sign",
            alpha=0.75,
        )
    wrong = dict(directions["dual_both_sign"])
    wrong[screen._PARAMETER_NAMES[0]] = torch.zeros(1, dtype=torch.float32)
    with pytest.raises(ValueError, match="direction tensor changed"):
        screen.candidate_from_normalized_direction(
            source, wrong, direction_id="dual_both_sign", alpha=0.5
        )
    wrong = dict(directions["dual_both_sign"])
    wrong[screen._PARAMETER_NAMES[2]] = wrong[screen._PARAMETER_NAMES[2]].double()
    with pytest.raises(ValueError, match="direction tensor changed"):
        screen.candidate_from_normalized_direction(
            source, wrong, direction_id="dual_both_sign", alpha=0.5
        )


def test_threshold_is_exact_strict_v47_teacher_gate_without_greedy() -> None:
    passed = screen.candidate_threshold_diagnostic(
        _gate_metrics(), broad_nll=2.9, prefix_trust_rms=0.002
    )
    assert passed["all_numeric_thresholds_met"] is True
    assert len(passed["checks"]) == 11
    assert "greedy" not in json.dumps(passed)

    lost = screen.candidate_threshold_diagnostic(
        _gate_metrics(lost_positive=False), broad_nll=2.9, prefix_trust_rms=0.001
    )
    assert lost["checks"]["both_lost_sides_strictly_positive"] is False
    assert lost["all_numeric_thresholds_met"] is False

    drift = screen.candidate_threshold_diagnostic(
        _gate_metrics(), broad_nll=2.9, prefix_trust_rms=0.002001
    )
    assert drift["checks"]["candidate_relative_prefix_trust_rms_at_most_0_002"] is False
    assert drift["all_numeric_thresholds_met"] is False


def test_exact_v47_u4_source_is_authenticated_without_optimizer_read() -> None:
    optimizer = screen._resolve(screen.DEFAULT_SOURCE) / "optimizer.pt"
    audit = FileAccessAudit([optimizer], block_forbidden=True)
    with audit:
        tensors, metadata, source = screen._source_evidence()
    audit.assert_clean()
    assert len(tensors) == 179
    assert metadata["optimizer_step"] == 4
    assert source["full_tensor_state_sha256"] == screen._SOURCE_FULL_SHA256
    assert source["authorized_surface_state_sha256"] == (screen._SOURCE_AUTHORIZED_SHA256)
    assert source["frozen_state_sha256"] == screen._SOURCE_FROZEN_SHA256
    assert source["v47_final_train_only_gate_passed"] is False
    assert not any(path.endswith("/optimizer.pt") for path in audit.unique_paths)


def test_terminal_hash_is_explicit_and_exact_authorization_is_consumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    terminal = tmp_path / "terminal.json"
    monkeypatch.setattr(screen, "DEFAULT_TERMINAL", terminal)
    report = _terminal_payload()
    report["conditional_successor_authorization"]["invocation_contract"]["terminal_path"] = str(
        terminal
    )
    terminal.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    digest = screen._sha256(terminal)
    result = screen.require_terminal(digest)
    assert result["sha256"] == digest
    assert all(result["checks"].values())
    with pytest.raises(ValueError, match="explicit invocation"):
        screen.require_terminal("0" * 64)
    with pytest.raises(ValueError, match="lowercase hex"):
        screen.require_terminal("not-a-sha")


def test_authorization_rejects_grid_normalization_and_scope_broadening() -> None:
    report = _terminal_payload()
    authorization = report["conditional_successor_authorization"]
    bad = copy.deepcopy(authorization)
    bad["candidate_grid"]["alpha_grid"].append(4.0)
    with pytest.raises(ValueError, match="authorization changed"):
        screen._validate_authorization(report, bad)
    bad = copy.deepcopy(authorization)
    bad["measurements"]["normalize_each_nonzero_component"] = "global_average"
    with pytest.raises(ValueError, match="authorization changed"):
        screen._validate_authorization(report, bad)
    bad = copy.deepcopy(authorization)
    bad["scope"]["greedy_generation_authorized"] = True
    with pytest.raises(ValueError, match="authorization changed"):
        screen._validate_authorization(report, bad)


def test_source_contains_no_optimizer_step_checkpoint_greedy_or_selection_path() -> None:
    path = PROJECT_ROOT / screen.V48_SCRIPT
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "backward" not in attributes
    assert "step" not in attributes
    assert "torch.optim" not in source
    assert "load_optimizer_checkpoint" not in source
    assert "save_adapter_checkpoint" not in source
    assert "save_optimizer_checkpoint" not in source
    assert "training_greedy_metrics" not in source
    assert "candidate_rank_key" not in source
    assert "selected_candidate" not in source
    assert "v46._selected_side_gradient" in source
    assert "--expected-v47-terminal-sha256" in source


def test_candidate_relative_prefix_reference_and_prehash_order_are_explicit() -> None:
    source = Path(screen.__file__).read_text(encoding="utf-8")
    run = source[source.index("def run_screen") : source.index("def _atomic_json")]
    reference = run.index("source_scene_tokens =")
    gradients = run.index("gradients, gradient_audit =")
    inventory = run.index("build_candidate_inventory(")
    candidate_loop = run.index("for specification in candidate_inventory")
    trust = run.index("references=source_scene_tokens")
    assert reference < gradients < inventory < candidate_loop < trust
    assert "candidate_hashes_fixed_before_candidate_forward_evaluation" in source
    assert "candidate_relative_prefix_trust_rms" in source


def test_write_report_refuses_existing_or_unpinned_output_before_live_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = tmp_path / "existing.json"
    existing.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(screen, "DEFAULT_OUTPUT", existing)
    called = False

    def forbidden(**_kwargs: object) -> dict[str, Any]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(screen, "run_screen", forbidden)
    with pytest.raises(FileExistsError, match="one-shot"):
        screen.write_report(existing, expected_v47_terminal_sha256="0" * 64)
    assert called is False

    monkeypatch.setattr(screen, "DEFAULT_OUTPUT", Path("reports/pinned.json"))
    with pytest.raises(ValueError, match="pinned"):
        screen.write_report(
            tmp_path / "unapproved.json",
            expected_v47_terminal_sha256="0" * 64,
        )
    assert called is False


def test_materialized_terminal_preflight_is_train_only_if_available() -> None:
    terminal = screen._resolve(screen.DEFAULT_TERMINAL)
    if not terminal.is_file():
        pytest.skip("V47 terminal is intentionally materialized after V48 hash pinning")
    digest = screen._sha256(terminal)
    authenticated = screen.require_terminal(digest)
    assert authenticated["authorization_id"] == screen._AUTHORIZATION_ID
    result = screen._preflight(expected_v47_terminal_sha256=digest)
    assert result["passed"] is True
    assert result["candidate_grid"]["candidate_count"] == 15
    assert result["gemma_loaded"] is False
    assert result["scene_maps_loaded"] is False
    assert result["optimizer_constructed_or_loaded"] is False
    assert result["candidate_checkpoint_written"] is False
    assert result["validation_qa_loaded"] is False
    assert result["oracle_loaded"] is False
    assert result["final_test_scenes_touched"] is False
    assert result["selector_executed"] is False
    assert result["runtime_promotion_executed"] is False
