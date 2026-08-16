from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import v50_scene_query_alpha_terminal_gate as gate

_REPORT_SHA = "158cedd46c73e29fc4cd5e412b6ddd260bb6be187c967c8c4489e5f610cc46f1"


@pytest.fixture(scope="module")
def payload() -> dict[str, object]:
    return json.loads((gate.PROJECT_ROOT / gate.V50_REPORT).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def scaffold() -> dict[str, object]:
    return gate.build_terminal_scaffold(_REPORT_SHA)


def test_exact_v50_report_is_sealed_and_v51_authorized_after_hash_pin(
    scaffold: dict[str, object],
) -> None:
    assert scaffold["passed"] is True
    assert scaffold["artifact"] == "v50_scene_query_alpha_terminal_gate_scaffold"
    assert scaffold["terminal_materialization_authorized"] is True
    assert scaffold["only_exact_successor_authorized"] == "v51_query_alpha_grid"
    authorization = scaffold["conditional_successor_authorization"]
    assert isinstance(authorization, dict)
    assert authorization["authorized"] is True
    assert scaffold["v50_checkpoint_write_authorized"] is False
    assert scaffold["v50_report_reference"] == {
        "path": str(gate.V50_REPORT),
        "sha256": _REPORT_SHA,
        "authenticated": True,
    }


def test_query_alpha_two_anchor_failed_only_positive_side(
    scaffold: dict[str, object],
) -> None:
    review = scaffold["v50_result_review"]
    assert review["candidate_count"] == 3
    assert review["anchor_candidate_id"] == "guarded_scene_alpha_1p0_query_alpha_2p0"
    assert review["anchor_only_failed_check"] == "teacher_positive_sides_at_least_35"
    assert review["anchor_positive_sides"] == 34
    assert review["anchor_prefix_rms"] == pytest.approx(0.0016845178324729204)
    assert review["greedy_executed"] is False
    assert review["checkpoint_written"] is False
    assert review["source_restored_exact"] is True
    assert review["access_audit_passed"] is True


def test_all_three_candidate_failures_are_exact(scaffold: dict[str, object]) -> None:
    reviews = scaffold["v50_result_review"]["candidate_reviews"]
    assert [row["candidate_id"] for row in reviews] == [
        value[0] for value in gate._V50_CANDIDATES
    ]
    for row in reviews:
        assert set(row["failed_checks"]) == gate._EXPECTED_FAILED_CHECKS[
            row["candidate_id"]
        ]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("passed",), True),
        (("selection", "winner"), {"candidate_id": "forbidden"}),
        (("selection", "passing_candidate_ids"), ["forbidden"]),
        (("final_train_gate", "execution_errors"), [{"type": "bad"}]),
        (("checkpoint", "written"), True),
        (("access_audit", "optimizer_file_reads"), ["optimizer.pt"]),
        (("access_audit", "oracle_loaded"), True),
        (("validation_qa_loaded",), True),
    ],
)
def test_report_mutations_fail_closed(
    payload: dict[str, object], path: tuple[str, ...], value: object
) -> None:
    changed = copy.deepcopy(payload)
    target = changed
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        gate.review_report_payload(changed)


def test_anchor_metric_mutation_fails_closed(payload: dict[str, object]) -> None:
    changed = copy.deepcopy(payload)
    changed["candidate_grid"]["candidates"][0]["non_greedy_pre_gate"]["evidence"][
        "positive_sides"
    ] = 35
    with pytest.raises(ValueError, match="anchor changed"):
        gate.review_report_payload(changed)


def test_wrong_report_hash_is_rejected() -> None:
    with pytest.raises(ValueError, match="differs from the fixed result"):
        gate.load_and_review_report("0" * 64)


def test_locked_file_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(FileNotFoundError, match="real file"):
        gate._locked_file(link, gate._sha256(target), "test")


def test_placeholder_mode_does_not_open_missing_v51_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        gate, "_V51_SCRIPT_SHA256", gate.V51_SCRIPT_SHA256_PLACEHOLDER
    )
    monkeypatch.setattr(gate, "_V51_TEST_SHA256", gate.V51_TEST_SHA256_PLACEHOLDER)
    monkeypatch.setattr(gate, "V51_SCRIPT", tmp_path / "missing.py")
    monkeypatch.setattr(gate, "V51_TEST", tmp_path / "missing_test.py")
    result = gate.build_terminal_scaffold(_REPORT_SHA)
    assert result["v51_implementation_review"] == {
        "ready": False,
        "status": "pending_stable_v51_module_and_test_hashes",
        "script": {
            "path": str(tmp_path / "missing.py"),
            "sha256": gate.V51_SCRIPT_SHA256_PLACEHOLDER,
        },
        "test": {
            "path": str(tmp_path / "missing_test.py"),
            "sha256": gate.V51_TEST_SHA256_PLACEHOLDER,
        },
        "no_v51_file_opened_in_placeholder_mode": True,
    }


def test_v51_grid_and_prefix_invariance_contract_are_fixed() -> None:
    authorization = gate.v51_authorization_template()
    assert authorization["authorized"] is True
    assert authorization["authorization_id"] == "v51_query_alpha_grid"
    grid = authorization["candidate_grid"]
    assert grid["candidate_count"] == 4
    assert grid["scene_alpha_fixed"] == 1.0
    assert grid["query_alpha_grid_declared_order"] == [1.75, 2.25, 1.5, 2.5]
    assert grid["query_alpha_two_anchor_nonselectable"] is True
    assert [row["query_alpha"] for row in grid["candidates_declared_order"]] == [
        1.75,
        2.25,
        1.5,
        2.5,
    ]
    assert grid["all_candidate_scene_tensors_bit_identical"] is True
    assert grid["all_candidate_scene_prefixes_bit_identical"] is True
    assert grid["expected_original_prefix_trust_rms"] == pytest.approx(
        0.0016845178324729204
    )


def test_v51_thresholds_scope_and_persistence_are_fail_closed() -> None:
    authorization = gate.v51_authorization_template()
    threshold = authorization["per_candidate_gate"]
    assert threshold["non_greedy_check_names"] == list(gate._NON_GREEDY_CHECKS)
    assert threshold["greedy_check_names"] == list(gate._GREEDY_CHECKS)
    assert threshold["teacher_positive_sides_minimum"] == 35
    assert threshold["broad_nll_maximum"] == 2.9213306349515915
    assert threshold["original_v46_candidate_relative_prefix_trust_rms_maximum"] == 0.002
    persistence = authorization["conditional_persistence"]
    assert persistence["checkpoint_write_iff_full_gate_winner_exists"] is True
    assert persistence["optimizer_file_in_checkpoint"] is False
    scope = authorization["scope"]
    assert scope["train_only"] is True
    assert scope["fixed_four_candidate_query_grid"] is True
    for name in (
        "optimizer_construction_authorized",
        "optimizer_state_file_open_authorized",
        "optimizer_step_authorized",
        "validation_access_authorized",
        "oracle_access_authorized",
        "final_test_access_authorized",
        "selector_execution_authorized",
        "runtime_promotion_authorized",
        "chat_promotion_authorized",
        "embodied_promotion_authorized",
    ):
        assert scope[name] is False


def test_terminal_materialization_and_write_are_one_shot_after_hash_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = gate.build_terminal_report(_REPORT_SHA)
    assert report["passed"] is True
    assert report["terminal_materialization_authorized"] is True
    assert report["only_exact_successor_authorized"] == "v51_query_alpha_grid"

    output = tmp_path / "terminal.json"
    monkeypatch.setattr(gate, "DEFAULT_OUTPUT", output)
    written = gate.write_report(output, expected_v50_report_sha256=_REPORT_SHA)
    assert written["passed"] is True
    assert output.is_file()
    with pytest.raises(FileExistsError, match="one-shot"):
        gate.write_report(output, expected_v50_report_sha256=_REPORT_SHA)


def test_terminal_module_imports_no_model_training_or_tensor_code() -> None:
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
