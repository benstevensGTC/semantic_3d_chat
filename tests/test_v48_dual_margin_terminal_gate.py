from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import v48_dual_margin_terminal_gate as gate

_V48_REPORT_SHA256 = "7abd2fa7741f84ea56933383199ec449d47dd99361def15d6a3874b9e154e02c"


@pytest.fixture(scope="module")
def report_payload() -> dict[str, object]:
    return json.loads((gate.PROJECT_ROOT / gate.V48_REPORT).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def scaffold() -> dict[str, object]:
    return gate.build_terminal_scaffold(_V48_REPORT_SHA256)


def test_exact_v48_result_is_authenticated_without_authorizing_v48_candidate(
    scaffold: dict[str, object],
) -> None:
    assert scaffold["passed"] is True
    assert scaffold["artifact"] == "v48_v47_u4_dual_margin_terminal_gate_scaffold"
    assert scaffold["terminal_materialization_authorized"] is True
    assert scaffold["only_exact_successor_authorized"] == (
        "v49_guarded_alpha2_train_candidate_gate"
    )
    authorization = scaffold["conditional_successor_authorization"]
    assert isinstance(authorization, dict)
    assert authorization["authorized"] is True
    assert scaffold["v48_candidate_checkpoint_write_authorized"] is False

    reference = scaffold["v48_report_reference"]
    assert isinstance(reference, dict)
    assert reference == {
        "path": str(gate.V48_REPORT),
        "sha256": _V48_REPORT_SHA256,
        "authenticated": True,
    }
    review = scaffold["v48_result_review"]
    assert isinstance(review, dict)
    assert review["candidate_count"] == 15
    assert review["restoration_count"] == 30
    assert review["teacher_threshold_passing_candidate_count"] == 1
    assert review["v48_candidate_selection_performed"] is False
    assert review["v48_candidate_authorization_granted"] is False
    assert review["v48_candidate_checkpoint_written"] is False


def test_unique_threshold_candidate_and_exact_evidence_are_sealed(
    scaffold: dict[str, object],
) -> None:
    review = scaffold["v48_result_review"]
    assert isinstance(review, dict)
    authentication = review["unique_candidate_authentication"]
    assert isinstance(authentication, dict)
    assert authentication["passed"] is True
    assert all(authentication["checks"].values())
    observed = authentication["observed"]
    assert observed == gate._EXPECTED_SELECTION
    assert observed["candidate_id"] == "guarded_both_sign_alpha_2p0"
    assert observed["complete_units"] == 10
    assert observed["positive_sides"] == 35
    assert observed["cross_prefix_complete_units"] == 18
    assert observed["complete_physical_pair_coverage"] == 5
    assert observed["mirror_complete_units"] == 2
    assert observed["book_complete_units"] == 1
    assert observed["book_cross_prefix_complete_units"] == 2


def test_all_candidate_thresholds_are_recomputed_from_full_evidence(
    report_payload: dict[str, object],
) -> None:
    results = report_payload["candidate_results"]
    assert isinstance(results, list)
    passing = []
    for result in results:
        pair = result["pair_metrics"]
        checks = gate.candidate_threshold_checks(
            pair,
            result["broad_nll"],
            result["candidate_relative_prefix_trust_rms"],
        )
        assert result["threshold_diagnostic"]["checks"] == checks
        assert result["threshold_diagnostic"]["all_numeric_thresholds_met"] is all(checks.values())
        if all(checks.values()):
            passing.append(result["candidate_id"])
    assert passing == ["guarded_both_sign_alpha_2p0"]


def test_v48_result_review_rejects_selection_authorization_or_persistence(
    report_payload: dict[str, object],
) -> None:
    mutations = (
        ("candidate_selection_performed", True),
        ("candidate_authorization_granted", True),
        ("candidate_checkpoint_written", True),
        ("parameter_state_persisted", True),
        ("optimizer_constructed_or_loaded", True),
        ("greedy_generation_executed", True),
        ("validation_qa_loaded", True),
        ("oracle_loaded", True),
        ("final_test_scenes_touched", True),
    )
    for field, value in mutations:
        changed = copy.deepcopy(report_payload)
        changed[field] = value
        with pytest.raises(ValueError, match="fixed report envelope"):
            gate.review_report_payload(changed)


def test_v48_result_review_rejects_second_threshold_passing_candidate(
    report_payload: dict[str, object],
) -> None:
    changed = copy.deepcopy(report_payload)
    selected = copy.deepcopy(changed["candidate_results"][-1])
    inventory_identity = changed["candidate_inventory"]["candidates"][0]
    for field in (
        "candidate_id",
        "direction_id",
        "alpha",
        "authorized_surface_state_sha256",
        "full_tensor_state_sha256",
    ):
        selected[field] = inventory_identity[field]
    selected["candidate_state_before_forward"]["authorized_surface_state_sha256"] = (
        inventory_identity["authorized_surface_state_sha256"]
    )
    selected["candidate_state_before_forward"]["full_tensor_state_sha256"] = inventory_identity[
        "full_tensor_state_sha256"
    ]
    changed["candidate_results"][0] = selected
    with pytest.raises(ValueError, match="exactly one teacher-threshold passing"):
        gate.review_report_payload(changed)


def test_restoration_and_candidate_hash_mutations_fail_closed(
    report_payload: dict[str, object],
) -> None:
    changed = copy.deepcopy(report_payload)
    changed["restoration_audit"][0]["full_tensor_state_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="exact source restoration"):
        gate.review_report_payload(changed)

    changed = copy.deepcopy(report_payload)
    changed["candidate_results"][-1]["full_tensor_state_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity differs from inventory"):
        gate.review_report_payload(changed)


def test_wrong_v48_report_hash_is_rejected_before_review() -> None:
    with pytest.raises(ValueError, match="differs from the fixed reviewed result"):
        gate.load_and_review_report("0" * 64)


def test_authenticated_file_reader_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(FileNotFoundError, match="must be a real file"):
        gate._locked_file(gate._resolve(link), gate._sha256(target), "test input")


def test_placeholder_mode_never_opens_unstable_v49_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        gate, "_V49_SCRIPT_SHA256", gate.V49_SCRIPT_SHA256_PLACEHOLDER
    )
    monkeypatch.setattr(gate, "_V49_TEST_SHA256", gate.V49_TEST_SHA256_PLACEHOLDER)
    monkeypatch.setattr(gate, "V49_SCRIPT", tmp_path / "missing_v49.py")
    monkeypatch.setattr(gate, "V49_TEST", tmp_path / "missing_test_v49.py")
    result = gate.build_terminal_scaffold(_V48_REPORT_SHA256)
    review = result["v49_implementation_review"]
    assert isinstance(review, dict)
    assert review["ready"] is False
    assert review["status"] == "pending_stable_v49_module_and_test_hashes"
    assert review["no_v49_file_opened_in_placeholder_mode"] is True


def test_v49_authorization_is_strictly_staged_and_fail_closed() -> None:
    authorization = gate.v49_authorization_template()
    assert authorization["authorized"] is True
    assert authorization["authorization_id"] == "v49_guarded_alpha2_train_candidate_gate"
    measurements = authorization["measurements"]
    assert measurements["non_greedy_pre_gate_evaluated_first"] is True
    assert measurements["full_greedy_mandatory_iff_pre_gate_passes"] is True
    assert measurements["greedy_skipped_due_pre_gate_required_if_failed"] is True
    assert measurements["pre_gate_failure_forces_final_failure_and_no_checkpoint"] is True
    gate_contract = authorization["final_train_gate"]
    assert gate_contract["pre_gate_passed_equals_all_non_greedy_checks"] is True
    assert gate_contract["pre_gate_failure_forbids_any_greedy_generation"] is True
    assert gate_contract["pre_gate_failure_forces_final_gate_failure"] is True
    assert gate_contract["pre_gate_pass_requires_exhaustive_greedy_evaluation"] is True
    assert gate_contract["final_gate_passed_equals_pre_gate_passed_and_all_greedy_checks"]
    assert gate_contract["non_greedy_pre_gate_check_names"] == list(
        gate._NON_GREEDY_PRE_GATE_CHECKS
    )
    assert gate_contract["greedy_final_gate_check_names"] == list(gate._GREEDY_FINAL_GATE_CHECKS)
    persistence = authorization["conditional_persistence"]
    assert persistence["candidate_checkpoint_write_iff_every_final_gate_check_passes"]
    assert persistence["pre_gate_failure_writes_no_checkpoint"] is True
    assert persistence["optimizer_file_in_checkpoint"] is False


def test_v49_source_prefix_reference_and_candidate_are_exact() -> None:
    authorization = gate.v49_authorization_template()
    source = authorization["source"]
    assert source["v47_u4"]["optimizer_file_open_authorized"] is False
    assert source["original_v46_candidate_prefix_reference"] == {
        "checkpoint": str(gate._PREFIX_REFERENCE_CHECKPOINT),
        "file_sha256": gate._PREFIX_REFERENCE_FILES,
        "candidate_id": "g5_both_sign_alpha_1p0",
        "full_tensor_state_sha256": gate._PREFIX_REFERENCE_FULL_SHA256,
        "authorized_surface_state_sha256": gate._PREFIX_REFERENCE_AUTHORIZED_SHA256,
        "frozen_state_sha256": gate._FROZEN_SHA256,
        "scene_count": 16,
        "question_free_global_scene_prefix": True,
    }
    candidate = authorization["candidate_reconstruction"]
    assert candidate["candidate_id"] == "guarded_both_sign_alpha_2p0"
    assert candidate["direction_components"] == ["g_book", "g_mirror", "g5_guard"]
    assert candidate["expected_full_tensor_state_sha256"] == (
        "69c5471d141ab56397969e2aac5f1097676f6ea328a3d5577e816c3aef6f3387"
    )
    assert candidate["expected_authorized_surface_state_sha256"] == (
        "f43e1b2a84006c8188adeff9f206ba864ae3d51b6567cad679f6d2e5f3610cf2"
    )


def test_terminal_materialization_and_write_are_one_shot_after_hashes_are_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = gate.build_terminal_report(_V48_REPORT_SHA256)
    assert report["passed"] is True
    assert report["terminal_materialization_authorized"] is True
    assert report["only_exact_successor_authorized"] == (
        "v49_guarded_alpha2_train_candidate_gate"
    )

    output = tmp_path / gate.DEFAULT_OUTPUT.name
    monkeypatch.setattr(gate, "DEFAULT_OUTPUT", output)
    written = gate.write_report(output, expected_v48_report_sha256=_V48_REPORT_SHA256)
    assert written["passed"] is True
    assert output.is_file()
    with pytest.raises(FileExistsError, match="one-shot"):
        gate.write_report(output, expected_v48_report_sha256=_V48_REPORT_SHA256)


def test_terminal_module_has_no_model_or_training_imports() -> None:
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
