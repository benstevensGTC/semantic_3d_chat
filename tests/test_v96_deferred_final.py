from __future__ import annotations

import hashlib
import inspect
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from semantic_3d_chat.evaluation import v96_deferred_final as deferred
from semantic_3d_chat.evaluation import v96_deferred_final_evaluation as final_evaluation
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import sha256_file_v85
from semantic_3d_chat.evaluation.v95_deferred_final_materialization import (
    authenticate_materialization_preregistration_v95,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_deferred_final_inventory_binds_v2_evaluation_io_exactly() -> None:
    relative = "src/semantic_3d_chat/evaluation/v96_evaluation_io_v2.py"

    assert relative in final_evaluation._IMPLEMENTATION_SOURCES
    inventory = final_evaluation._source_inventory()
    assert set(inventory) == set(final_evaluation._IMPLEMENTATION_SOURCES)
    assert inventory[relative] == sha256_file_v85(final_evaluation.PROJECT_ROOT / relative)


def _config(tmp_path: Path) -> dict[str, Any]:
    return {
        "deferred_final_lock": {
            **deferred._EXPECTED_DEFERRED_LOCK,
            "physical_artifact_roots": [
                str(tmp_path / "oracle"),
                str(tmp_path / "rendered"),
                str(tmp_path / "features"),
                str(tmp_path / "maps"),
            ],
            "empty_qa_placeholders": [
                str(tmp_path / "d20/qa/test.jsonl"),
                str(tmp_path / "d28/qa/test.jsonl"),
                str(tmp_path / "d52/qa/test.jsonl"),
            ],
            "legacy_plan_files_never_opened": [
                str(tmp_path / "diverse20.yaml"),
                str(tmp_path / "diverse28.yaml"),
                str(tmp_path / "diverse52.yaml"),
            ],
        },
    }


def _candidate() -> dict[str, Any]:
    return {
        "config_sha256": _digest("config"),
        "preregistration_sha256": _digest("prereg"),
        "cpu_preflight_sha256": _digest("preflight"),
        "training_report_sha256": _digest("training"),
        "fingerprint_sha256": _digest("candidate"),
        "state_sha256": _digest("state"),
        "weights_sha256": _digest("weights"),
        "attestation_file_sha256": _digest("candidate-attestation-file"),
        "attestation_identity_sha256": _digest("candidate-attestation-identity"),
        "v2_implementation_seal_sha256": _digest("evaluator-seal"),
        "frozen_v95_state_sha256": _digest("v95-state"),
    }


def _evaluator() -> dict[str, Any]:
    return {
        "authenticated": True,
        "seal_sha256": _digest("evaluator-seal"),
        "source_inventory_sha256": _digest("evaluator-sources"),
        "v1_implementation_seal_sha256": _digest("v1-evaluator-seal"),
    }


def _final_evaluation() -> dict[str, Any]:
    return {
        "authenticated": True,
        "all_thresholds_sealed_before_deferred_labels": True,
        "predictors_are_label_blind": True,
        "automatic_runtime_promotion": False,
        "preregistration_file_sha256": _digest("final-evaluation-file"),
        "preregistration_identity_sha256": _digest("final-evaluation-identity"),
        "v95_gate_source": {"contract_sha256": _digest("final-gates")},
        "implementation_source_inventory_sha256": _digest("final-evaluation-sources"),
    }


def _gate(candidate: dict[str, Any], evaluator: dict[str, Any]) -> dict[str, Any]:
    return {
        "authenticated": True,
        "status": "passed_deferred_final_explicit_unlock_eligible",
        "known_development_gate_passed": True,
        "deferred_final_unlock_eligible": True,
        "fixed_final_checkpoint_immutable": True,
        "frozen_v95_parent_immutable": True,
        "scene_prefix_question_independent": True,
        "protected_read_count": 0,
        "automatic_runtime_promotion": False,
        "runtime_promotion_authorized": False,
        "candidate_fingerprint_sha256": candidate["fingerprint_sha256"],
        "candidate_state_sha256": candidate["state_sha256"],
        "frozen_v95_state_sha256": candidate["frozen_v95_state_sha256"],
        "config_sha256": candidate["config_sha256"],
        "preregistration_sha256": candidate["preregistration_sha256"],
        "cpu_preflight_sha256": candidate["cpu_preflight_sha256"],
        "training_report_sha256": candidate["training_report_sha256"],
        "implementation_seal_sha256": evaluator["seal_sha256"],
        "implementation_source_inventory_sha256": evaluator["source_inventory_sha256"],
        "v1_implementation_seal_sha256": evaluator[
            "v1_implementation_seal_sha256"
        ],
        "candidate_attestation_file_sha256": candidate[
            "attestation_file_sha256"
        ],
        "candidate_attestation_identity_sha256": candidate[
            "attestation_identity_sha256"
        ],
        "candidate_attestation_immutable": True,
        "final_score_sha256": _digest("known-score"),
        "evidence_sha256": _digest("known-evidence"),
        "gate_results": {
            "accuracy": True,
            "causal_controls": True,
            "prefix_invariance": True,
        },
    }


def _materialization() -> dict[str, Any]:
    stages = {
        stage: {
            "authorized_entrypoint": [
                "/fixed/python",
                "-m",
                "semantic_3d_chat.evaluation.v95_deferred_final_materialization",
                "run-stage",
                "--stage",
                stage,
            ],
            "child_argv": [["fixed-child", stage]],
            "expected_outputs": [f"fixed/{stage}.artifact"],
            "receipt": f"reports/gemma4/artifacts/v95_deferred_final/receipts/{stage}.json",
        }
        for stage in deferred.MATERIALIZATION_STAGE_ORDER
    }
    return {
        "artifact": deferred.V95_MATERIALIZATION_ARTIFACT,
        "schema_version": 95,
        "status": deferred.V95_MATERIALIZATION_STATUS,
        "authenticated": True,
        "preregistration_file_sha256": (deferred.PINNED_V95_MATERIALIZATION_FILE_SHA256),
        "preregistration_identity_sha256": (deferred.PINNED_V95_MATERIALIZATION_IDENTITY_SHA256),
        "stage_order": list(deferred.MATERIALIZATION_STAGE_ORDER),
        "stages": stages,
        "scene_count": 6,
        "pair_count": 3,
        "intended_row_count": 216,
        "intended_changed_unit_count": 12,
        "intended_changed_side_count": 24,
        "source_sha256": {"source.py": _digest("source")},
        "numeric_compiler_source_sha256": {"compiler.py": _digest("compiler")},
        "recipe": {"recipe_sha256": _digest("recipe")},
        "generation_requires_authenticated_unlock": True,
        "every_execution_stage_requires_authenticated_unlock": True,
        "legacy_plan_files_opened": [],
        "known_development_labels_opened": False,
        "deferred_labels_opened": False,
        "deferred_oracle_opened": False,
        "deferred_artifacts_generated": False,
        "model_loaded": False,
        "optimizer_constructed": False,
        "protected_read_count": 0,
        "automatic_runtime_promotion": False,
    }


def _absence(config: dict[str, Any], materialization: dict[str, Any]) -> dict[str, Any]:
    return deferred._absence_without_recheck(config, materialization)


def _access(config: dict[str, Any]) -> dict[str, Any]:
    loaded: list[str] = []
    return {
        "artifact": "gemma4_v96_deferred_final_unlock_access_v1",
        "schema_version": 96,
        "loaded_files": loaded,
        "loaded_file_inventory_sha256": deferred.canonical_sha256_v96(loaded),
        "forbidden_roots": [str(path) for path in deferred._forbidden_roots(config)],
        "forbidden_component_names": ["oracle", "qa"],
        "block_forbidden": True,
        "forbidden_accesses": [],
        "protected_read_count": 0,
        "passed": True,
    }


@contextmanager
def _unguarded(*_args: Any, **_kwargs: Any) -> Any:
    yield {"authenticated": True}


def _patch_prerequisites(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _config(tmp_path)
    candidate = _candidate()
    evaluator = _evaluator()
    gate = _gate(candidate, evaluator)
    materialization = _materialization()
    absence = _absence(config, materialization)
    access = _access(config)

    def prerequisites(
        _config_path: str | Path, *, require_absence: bool
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        del require_absence
        return config, gate, candidate, absence, access, materialization, evaluator

    monkeypatch.setattr(deferred, "_prerequisites", prerequisites)
    monkeypatch.setattr(deferred, "deferred_final_guard_v96", _unguarded)
    monkeypatch.setattr(
        deferred,
        "_authenticate_final_evaluation_preregistration_v96",
        _final_evaluation,
    )
    return config, gate, candidate


def test_v96_reuses_exact_authenticated_v95_materialization_contract() -> None:
    materialization = authenticate_materialization_preregistration_v95()

    deferred._validate_materialization_preregistration_v96(materialization)

    assert (
        materialization["preregistration_file_sha256"]
        == deferred.PINNED_V95_MATERIALIZATION_FILE_SHA256
    )
    assert (
        materialization["preregistration_identity_sha256"]
        == deferred.PINNED_V95_MATERIALIZATION_IDENTITY_SHA256
    )
    assert materialization["stage_order"] == list(deferred.MATERIALIZATION_STAGE_ORDER)


@pytest.mark.parametrize(
    "field,value",
    [
        ("preregistration_file_sha256", _digest("substitute")),
        ("preregistration_identity_sha256", _digest("substitute")),
        ("generation_requires_authenticated_unlock", False),
        ("every_execution_stage_requires_authenticated_unlock", False),
        ("deferred_artifacts_generated", True),
        ("protected_read_count", 1),
        ("automatic_runtime_promotion", True),
    ],
)
def test_reused_materialization_contract_fails_closed_on_tamper(field: str, value: Any) -> None:
    materialization = {**_materialization(), field: value}
    with pytest.raises(ValueError, match="rejected the reused"):
        deferred._validate_materialization_preregistration_v96(materialization)


@pytest.mark.parametrize(
    "mutation",
    [
        {"known_development_gate_passed": False},
        {"deferred_final_unlock_eligible": False},
        {"fixed_final_checkpoint_immutable": False},
        {"frozen_v95_parent_immutable": False},
        {"scene_prefix_question_independent": False},
        {"protected_read_count": 1},
        {"runtime_promotion_authorized": True},
        {"status": "measured_preregistered_gate_not_passed"},
        {"gate_results": {"accuracy": True, "causal_controls": False}},
    ],
)
def test_v96_unlock_rejects_every_incomplete_gate_condition(
    mutation: dict[str, Any],
) -> None:
    candidate = _candidate()
    evaluator = _evaluator()
    gate = {**_gate(candidate, evaluator), **mutation}
    with pytest.raises(deferred.DeferredFinalBlockedError, match="remains locked"):
        deferred._assert_passing_gate(gate, candidate, evaluator)


def test_v96_unlock_rejects_a_different_candidate_or_evaluator() -> None:
    candidate = _candidate()
    evaluator = _evaluator()
    gate = _gate(candidate, evaluator)
    gate["candidate_fingerprint_sha256"] = _digest("other-candidate")
    with pytest.raises(deferred.DeferredFinalBlockedError, match="remains locked"):
        deferred._assert_passing_gate(gate, candidate, evaluator)
    gate = _gate(candidate, evaluator)
    evaluator["seal_sha256"] = _digest("other-seal")
    with pytest.raises(deferred.DeferredFinalBlockedError, match="remains locked"):
        deferred._assert_passing_gate(gate, candidate, evaluator)


def test_preflight_proves_eligibility_but_does_not_authorize_or_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config_value, _gate_value, candidate = _patch_prerequisites(monkeypatch, tmp_path)
    unlock = tmp_path / "unlock.json"
    monkeypatch.setattr(deferred, "DEFAULT_UNLOCK", unlock)

    result = deferred.preflight_deferred_final_v96("synthetic.yaml")

    assert result["status"] == deferred.PREFLIGHT_STATUS
    assert result["candidate_fingerprint_sha256"] == candidate["fingerprint_sha256"]
    assert result["explicit_unlock_created"] is False
    assert result["materialization_stage_execution_authorized"] is False
    assert result["stage_execution_performed"] is False
    assert result["scene_generation_performed"] is False
    assert result["model_loaded"] is False
    assert not unlock.exists()
    contract = result["execution_contract"]
    assert contract["reused_child_argv_byte_for_byte"] is False
    assert contract["only_qa_select_authorization_child_changed_from_v95"] is True
    assert contract["qa_select_v95_unlock_dependency_removed"] is True
    assert contract["qa_select_authenticates_v96_unlock_before_label_read"] is True
    assert contract["unchanged_stage_child_argv_byte_for_byte"] == [
        stage for stage in deferred.MATERIALIZATION_STAGE_ORDER if stage != "qa_select"
    ]
    assert contract["deferred_prediction_or_scoring_authorized_by_this_unlock"] is False
    assert contract["automatic_runtime_promotion"] is False


def test_unlock_is_explicit_create_once_authenticated_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_prerequisites(monkeypatch, tmp_path)
    unlock = tmp_path / "unlock.json"
    monkeypatch.setattr(deferred, "DEFAULT_UNLOCK", unlock)

    created = deferred.unlock_deferred_final_v96("synthetic.yaml")
    reused = deferred.unlock_deferred_final_v96("synthetic.yaml")

    assert created["authenticated"] is True
    assert created["reused_authenticated_unlock"] is False
    assert created["explicit_unlock_created"] is True
    assert created["materialization_stage_execution_authorized"] is True
    assert reused["reused_authenticated_unlock"] is True
    assert created["unlock_file_sha256"] == reused["unlock_file_sha256"]


def test_unlock_authentication_detects_payload_and_wrapper_source_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_prerequisites(monkeypatch, tmp_path)
    unlock = tmp_path / "unlock.json"
    monkeypatch.setattr(deferred, "DEFAULT_UNLOCK", unlock)
    source_a = {"wrapper.py": _digest("source-a")}
    monkeypatch.setattr(deferred, "_implementation_identity", lambda: source_a)
    deferred.unlock_deferred_final_v96("synthetic.yaml")

    monkeypatch.setattr(
        deferred,
        "_implementation_identity",
        lambda: {"wrapper.py": _digest("source-b")},
    )
    with pytest.raises(ValueError, match="no longer match"):
        deferred.authenticate_deferred_final_unlock_v96("synthetic.yaml")

    monkeypatch.setattr(deferred, "_implementation_identity", lambda: source_a)
    payload = json.loads(unlock.read_text(encoding="utf-8"))
    payload["model_loaded"] = True
    unlock.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="no longer match"):
        deferred.authenticate_deferred_final_unlock_v96("synthetic.yaml")


def test_existing_unlock_authentication_does_not_reassert_current_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    candidate = _candidate()
    evaluator = _evaluator()
    gate = _gate(candidate, evaluator)
    materialization = _materialization()
    absence = _absence(config, materialization)
    access = _access(config)
    calls: list[bool] = []

    def prerequisites(
        _config_path: str | Path, *, require_absence: bool
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        calls.append(require_absence)
        return config, gate, candidate, absence, access, materialization, evaluator

    monkeypatch.setattr(deferred, "_prerequisites", prerequisites)
    monkeypatch.setattr(deferred, "deferred_final_guard_v96", _unguarded)
    monkeypatch.setattr(
        deferred,
        "_authenticate_final_evaluation_preregistration_v96",
        _final_evaluation,
    )
    unlock = tmp_path / "unlock.json"
    monkeypatch.setattr(deferred, "DEFAULT_UNLOCK", unlock)
    deferred.unlock_deferred_final_v96("synthetic.yaml")
    calls.clear()

    deferred.authenticate_deferred_final_unlock_v96("synthetic.yaml")

    assert calls == [False]


def test_gate_is_checked_before_any_deferred_absence_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    candidate = _candidate()
    evaluator = _evaluator()
    gate = {**_gate(candidate, evaluator), "known_development_gate_passed": False}
    materialization = _materialization()
    absence_called = False

    def mark_absence(_config_value: dict[str, Any]) -> dict[str, Any]:
        nonlocal absence_called
        absence_called = True
        return {}

    monkeypatch.setattr(deferred, "assert_bound_config_path_v96", lambda *_a: None)
    monkeypatch.setattr(deferred, "load_config_v96", lambda *_a, **_k: config)
    monkeypatch.setattr(deferred, "_validate_deferred_lock_config", lambda *_a: None)
    monkeypatch.setattr(
        deferred,
        "authenticate_materialization_preregistration_v95",
        lambda: materialization,
    )
    monkeypatch.setattr(deferred, "_validate_materialization_preregistration_v96", lambda *_a: None)
    monkeypatch.setattr(
        deferred,
        "authenticate_evaluation_implementation_v96_v2",
        lambda **_k: evaluator,
    )
    monkeypatch.setattr(deferred, "authenticate_final_evidence_v96", lambda *_a: gate)
    monkeypatch.setattr(
        deferred,
        "authenticate_fixed_final_candidate_v96",
        lambda *_a, **_k: candidate,
    )
    monkeypatch.setattr(deferred, "assert_deferred_final_absent_v96", mark_absence)

    with pytest.raises(deferred.DeferredFinalBlockedError, match="remains locked"):
        deferred._prerequisites("synthetic.yaml", require_absence=True)
    assert absence_called is False


def test_materialization_absence_rejects_any_unreceipted_footprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    materialization = _materialization()
    monkeypatch.setattr(deferred, "V95_WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(
        deferred,
        "resolve_v96",
        lambda raw: tmp_path / Path(raw),
    )
    output = tmp_path / "fixed/generate.artifact"
    output.parent.mkdir(parents=True)
    output.write_text("partial", encoding="utf-8")

    with pytest.raises(RuntimeError, match="preexisting deferred materialization"):
        deferred._materialization_absence(materialization)


def test_arbitrary_parallel_unlock_paths_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="one fixed unlock path"):
        deferred._unlock_path(tmp_path / "parallel.json")


def test_authorization_module_has_no_model_loader_or_generation_process() -> None:
    source = inspect.getsource(deferred)
    forbidden = (
        "import subprocess",
        "load_local_language_model",
        "Gemma4ForConditionalGeneration",
        "run_materialization_stage_v95",
    )
    assert not any(value in source for value in forbidden)
    assert "authenticate_final_evidence_v96" in source
    assert "authenticate_materialization_preregistration_v95" in source
    assert "explicit_separate_unlock_required" in source


def test_v96_deferred_make_targets_are_explicit_and_do_not_auto_unlock() -> None:
    makefile = (deferred.PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    for target in (
        "v96-deferred-final-check:",
        "v96-deferred-final-preflight:",
        "v96-deferred-final-unlock:",
        "v96-deferred-final-authenticate:",
        "v96-deferred-final-template:",
        "v96-deferred-final-materialization-preflight:",
        "v96-deferred-final-generate:",
        "v96-deferred-final-questions:",
    ):
        assert target in makefile
    section = makefile[makefile.index("v96-deferred-final-check:") :]
    assert "-m semantic_3d_chat.evaluation.v96_deferred_final" in section
    assert "gemma4-final-once" not in section
    generate_line = next(
        line for line in section.splitlines() if line.startswith("v96-deferred-final-generate:")
    )
    assert "v96-deferred-final-unlock" not in generate_line
