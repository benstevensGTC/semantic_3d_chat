from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import v49_guarded_candidate_terminal_gate as gate

_V49_REPORT_SHA256 = "7d82a503a5402dcfd80816459eea5d653849e89a49fa8ce7b585dc806ff7acc9"


@pytest.fixture(scope="module")
def report_payload() -> dict[str, object]:
    return json.loads((gate.PROJECT_ROOT / gate.V49_REPORT).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def scaffold() -> dict[str, object]:
    return gate.build_terminal_scaffold(_V49_REPORT_SHA256)


def test_exact_v49_result_is_authenticated_and_v50_is_authorized_after_hash_pin(
    scaffold: dict[str, object],
) -> None:
    assert scaffold["passed"] is True
    assert scaffold["artifact"] == "v49_guarded_candidate_terminal_gate_scaffold"
    assert scaffold["terminal_materialization_authorized"] is True
    assert scaffold["only_exact_successor_authorized"] == "v50_scene_query_alpha_grid"
    authorization = scaffold["conditional_successor_authorization"]
    assert isinstance(authorization, dict)
    assert authorization["authorized"] is True
    assert scaffold["v49_checkpoint_write_authorized"] is False
    assert scaffold["v49_report_reference"] == {
        "path": str(gate.V49_REPORT),
        "sha256": _V49_REPORT_SHA256,
        "authenticated": True,
    }


def test_v49_failed_exactly_original_prefix_trust_radius(
    scaffold: dict[str, object],
) -> None:
    review = scaffold["v49_result_review"]
    assert isinstance(review, dict)
    non_greedy = review["non_greedy_review"]
    assert isinstance(non_greedy, dict)
    assert non_greedy["failed_checks"] == [gate._FAILED_CHECK]
    assert non_greedy["observed_prefix_trust_rms"] == pytest.approx(
        0.0020444965921342373,
        abs=1.0e-15,
    )
    assert non_greedy["prefix_trust_rms_maximum"] == 0.002
    assert non_greedy["excess"] == pytest.approx(0.00004449659213423726)
    checks = non_greedy["checks"]
    assert isinstance(checks, dict)
    assert checks[gate._FAILED_CHECK] is False
    assert all(value is True for key, value in checks.items() if key != gate._FAILED_CHECK)


def test_v49_greedy_checkpoint_restoration_and_access_are_sealed(
    scaffold: dict[str, object],
) -> None:
    review = scaffold["v49_result_review"]
    assert isinstance(review, dict)
    assert review["v49_final_train_gate_passed"] is False
    assert review["v49_greedy_executed"] is False
    assert review["v49_checkpoint_written"] is False
    assert review["source_restored_exact"] is True
    assert review["access_audit_passed"] is True
    assert not (gate.PROJECT_ROOT / gate.V49_CHECKPOINT).exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("optimizer_constructed_or_loaded", True),
        ("optimizer_state_file_opened", True),
        ("optimizer_step_executed", True),
        ("candidate_selection_performed", True),
        ("validation_qa_loaded", True),
        ("validation_environment_maps_loaded", True),
        ("oracle_loaded", True),
        ("final_test_scenes_touched", True),
        ("selector_executed", True),
        ("runtime_promotion_executed", True),
        ("chat_promotion_executed", True),
        ("embodied_promotion_executed", True),
    ],
)
def test_restricted_or_stateful_top_level_mutations_fail_closed(
    report_payload: dict[str, object], field: str, value: object
) -> None:
    changed = copy.deepcopy(report_payload)
    changed[field] = value
    with pytest.raises(ValueError, match="fixed result envelope"):
        gate.review_report_payload(changed)


def test_a_second_pre_gate_failure_or_false_success_fails_closed(
    report_payload: dict[str, object],
) -> None:
    changed = copy.deepcopy(report_payload)
    changed["non_greedy_pre_gate"]["checks"]["teacher_positive_sides_at_least_35"] = False
    with pytest.raises(ValueError, match="fail exactly"):
        gate.review_report_payload(changed)

    changed = copy.deepcopy(report_payload)
    changed["non_greedy_pre_gate"]["checks"][gate._FAILED_CHECK] = True
    with pytest.raises(ValueError, match="fail exactly"):
        gate.review_report_payload(changed)


def test_prefix_trust_measurement_mutation_fails_closed(
    report_payload: dict[str, object],
) -> None:
    changed = copy.deepcopy(report_payload)
    changed["non_greedy_pre_gate"]["evidence"][
        "original_v46_candidate_relative_prefix_trust_rms"
    ] = 0.0019
    with pytest.raises(ValueError, match="non-greedy evidence changed"):
        gate.review_report_payload(changed)


def test_greedy_execution_or_checkpoint_persistence_mutation_fails_closed(
    report_payload: dict[str, object],
) -> None:
    changed = copy.deepcopy(report_payload)
    changed["greedy_gate"]["executed"] = True
    with pytest.raises(ValueError, match="fixed result envelope"):
        gate.review_report_payload(changed)

    changed = copy.deepcopy(report_payload)
    changed["checkpoint"]["written"] = True
    with pytest.raises(ValueError, match="fixed result envelope"):
        gate.review_report_payload(changed)


def test_wrong_v49_report_hash_is_rejected_before_review() -> None:
    with pytest.raises(ValueError, match="differs from the fixed reviewed result"):
        gate.load_and_review_report("0" * 64)


def test_authenticated_file_reader_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(FileNotFoundError, match="must be a real file"):
        gate._locked_file(gate._resolve(link), gate._sha256(target), "test input")


def test_placeholder_mode_never_opens_unstable_v50_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gate, "_V50_SCRIPT_SHA256", gate.V50_SCRIPT_SHA256_PLACEHOLDER)
    monkeypatch.setattr(gate, "_V50_TEST_SHA256", gate.V50_TEST_SHA256_PLACEHOLDER)
    monkeypatch.setattr(gate, "V50_SCRIPT", tmp_path / "missing_v50.py")
    monkeypatch.setattr(gate, "V50_TEST", tmp_path / "missing_test_v50.py")
    result = gate.build_terminal_scaffold(_V49_REPORT_SHA256)
    implementation = result["v50_implementation_review"]
    assert isinstance(implementation, dict)
    assert implementation["ready"] is False
    assert implementation["status"] == "pending_stable_v50_module_and_test_hashes"
    assert implementation["no_v50_file_opened_in_placeholder_mode"] is True
    with pytest.raises(RuntimeError, match="pending stable V50"):
        gate.build_terminal_report(_V49_REPORT_SHA256)


def test_v50_grid_is_fixed_and_uses_guarded_gradients() -> None:
    authorization = gate.v50_authorization_template()
    assert authorization["authorization_id"] == "v50_scene_query_alpha_grid"
    assert authorization["authorized"] is True
    assert authorization["authorized_script"] == (
        "src/semantic_3d_chat/evaluation/v50_scene_query_alpha_grid.py"
    )
    assert authorization["authorized_test"] == "tests/test_v50_scene_query_alpha_grid.py"
    assert authorization["authorized_report"] == (
        "reports/gemma4/metrics/v50_scene_query_alpha_grid.json"
    )
    grid = authorization["candidate_grid"]
    assert grid["candidate_count"] == 3
    assert grid["direction_components"] == ["g_book", "g_mirror", "g5_guard"]
    assert grid["scene_alpha_grid_declared_order"] == [1.0, 0.5, 0.25]
    assert grid["query_alpha_fixed"] == 2.0
    assert [row["declared_order"] for row in grid["candidates_declared_order"]] == [0, 1, 2]
    assert [row["scene_alpha"] for row in grid["candidates_declared_order"]] == [
        1.0,
        0.5,
        0.25,
    ]
    assert all(row["query_alpha"] == 2.0 for row in grid["candidates_declared_order"])
    assert grid["adaptive_grid_or_candidate_mutation"] is False


def test_v50_evaluates_all_pre_gates_and_every_eligible_greedy_gate() -> None:
    authorization = gate.v50_authorization_template()
    workflow = authorization["evaluation_and_selection"]
    assert workflow["all_candidates_receive_full_non_greedy_gate_before_selection"] is True
    assert workflow["greedy_runs_for_every_pre_gate_passing_candidate"] is True
    assert workflow["greedy_runs_only_for_pre_gate_passing_candidates"] is True
    assert workflow["every_pre_gate_passing_candidate_receives_full_changed_25_greedy"] is True
    assert workflow["every_pre_gate_passing_candidate_receives_full_broad_48_greedy"] is True
    assert workflow["winner_selection"] == (
        "first_full_gate_passing_candidate_in_fixed_declared_order_after_all_evaluations"
    )
    assert workflow["no_early_stop_after_first_full_gate_pass"] is True


def test_v50_thresholds_are_unchanged_and_persistence_is_optimizer_free() -> None:
    authorization = gate.v50_authorization_template()
    threshold = authorization["per_candidate_gate"]
    assert threshold["non_greedy_check_names"] == list(gate._NON_GREEDY_CHECKS)
    assert threshold["greedy_check_names"] == list(gate._GREEDY_CHECKS)
    assert threshold["teacher_complete_units_minimum"] == 10
    assert threshold["teacher_positive_sides_minimum"] == 35
    assert threshold["teacher_cross_prefix_complete_units_minimum"] == 17
    assert threshold["broad_nll_maximum"] == 2.9213306349515915
    assert threshold["original_v46_candidate_relative_prefix_trust_rms_maximum"] == 0.002
    assert threshold["train_greedy_complete_units_minimum"] == 5
    assert threshold["broad_greedy_exact_correct_minimum"] == 23
    assert threshold["broad_greedy_row_count_exact"] == 48
    assert threshold["thresholds_unchanged_from_v49"] is True

    persistence = authorization["conditional_persistence"]
    assert persistence["checkpoint_write_iff_full_gate_winner_exists"] is True
    assert persistence["checkpoint_contains_declared_order_winner_only"] is True
    assert persistence["failed_grid_writes_no_checkpoint"] is True
    assert persistence["optimizer_file_in_checkpoint"] is False


def test_v50_scope_forbids_restricted_data_optimizer_and_promotion() -> None:
    scope = gate.v50_authorization_template()["scope"]
    assert scope["train_only"] is True
    for field in (
        "backward_or_parameter_gradient_accumulation_authorized",
        "optimizer_construction_authorized",
        "optimizer_state_file_open_authorized",
        "optimizer_state_loading_authorized",
        "optimizer_step_authorized",
        "validation_access_authorized",
        "oracle_access_authorized",
        "final_test_access_authorized",
        "selector_execution_authorized",
        "runtime_promotion_authorized",
        "chat_promotion_authorized",
        "embodied_promotion_authorized",
    ):
        assert scope[field] is False


def test_v50_pinned_hashes_materialize_terminal_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = gate.build_terminal_report(_V49_REPORT_SHA256)
    assert report["passed"] is True
    assert report["terminal_materialization_authorized"] is True
    assert report["only_exact_successor_authorized"] == "v50_scene_query_alpha_grid"

    output = tmp_path / gate.DEFAULT_OUTPUT.name
    monkeypatch.setattr(gate, "DEFAULT_OUTPUT", output)
    written = gate.write_report(output, expected_v49_report_sha256=_V49_REPORT_SHA256)
    assert written["passed"] is True
    assert output.is_file()
    with pytest.raises(FileExistsError, match="one-shot"):
        gate.write_report(output, expected_v49_report_sha256=_V49_REPORT_SHA256)


def test_terminal_module_has_no_model_training_or_tensor_imports() -> None:
    tree = ast.parse(Path(gate.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported == {
        "__future__",
        "argparse",
        "collections.abc",
        "hashlib",
        "json",
        "math",
        "os",
        "pathlib",
        "re",
        "semantic_3d_chat.config",
        "tempfile",
        "typing",
    }


def test_module_and_test_hashes_are_stable_inputs_for_later_pin() -> None:
    module_path = Path(gate.__file__)
    test_path = Path(__file__)
    assert len(hashlib.sha256(module_path.read_bytes()).hexdigest()) == 64
    assert len(hashlib.sha256(test_path.read_bytes()).hexdigest()) == 64
