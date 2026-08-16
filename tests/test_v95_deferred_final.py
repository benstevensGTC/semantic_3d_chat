from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from semantic_3d_chat.evaluation import v95_deferred_final as deferred


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _config(tmp_path: Path) -> dict[str, Any]:
    return {
        "deferred_final_lock": {
            "physical_artifact_roots": [
                str(tmp_path / "oracle"),
                str(tmp_path / "rendered"),
                str(tmp_path / "features"),
                str(tmp_path / "maps"),
            ],
            "empty_qa_placeholders": [
                str(tmp_path / "data_diverse20/qa/test.jsonl"),
                str(tmp_path / "data_diverse28/qa/test.jsonl"),
                str(tmp_path / "data_diverse52/qa/test.jsonl"),
            ],
            "legacy_plan_files_never_opened": [
                str(tmp_path / "diverse20.yaml"),
                str(tmp_path / "diverse28.yaml"),
                str(tmp_path / "diverse52.yaml"),
            ],
        },
        "deferred_evaluation": {
            "scene_count": 6,
            "pair_count": 3,
            "expected_row_count_after_unlock": 216,
            "expected_changed_unit_count_after_unlock": 12,
            "expected_changed_side_count_after_unlock": 24,
        },
        "gates": {
            "canonical_accuracy_minimum": 0.65,
            "canonical_accuracy_margin_over_fixed_v94_same_rows": 0.03,
            "runtime_packaging_requires_separate_leakage_gate": True,
            "automatic_runtime_promotion": False,
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
    }


def _gate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "authenticated": True,
        "status": "passed_deferred_final_explicit_unlock_eligible",
        "known_development_gate_passed": True,
        "deferred_final_unlock_eligible": True,
        "fixed_final_checkpoint_immutable": True,
        "scene_prefix_question_independent": True,
        "protected_read_count": 0,
        "automatic_runtime_promotion": False,
        "runtime_promotion_authorized": False,
        "candidate_fingerprint_sha256": candidate["fingerprint_sha256"],
        "candidate_state_sha256": candidate["state_sha256"],
        "config_sha256": candidate["config_sha256"],
        "training_report_sha256": candidate["training_report_sha256"],
        "final_score_sha256": _digest("known-score"),
        "evidence_sha256": _digest("known-evidence"),
        "gate_results": {
            "accuracy": True,
            "causal_control": True,
            "prefix_invariance": True,
        },
    }


def _absence(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene_ids": list(deferred.DEFERRED_FINAL_SCENES),
        "physical_path_count_checked": 24,
        "physical_artifacts_present": [],
        "empty_qa_placeholders": {
            Path(path).as_posix(): 0
            for path in config["deferred_final_lock"]["empty_qa_placeholders"]
        },
        "legacy_plan_file_count_opened": 0,
        "scene_generation_performed": False,
        "rendering_performed": False,
        "feature_extraction_performed": False,
        "map_building_performed": False,
        "qa_generation_performed": False,
    }


def _access(config: dict[str, Any]) -> dict[str, Any]:
    files: list[str] = []
    return {
        "artifact": "gemma4_v95_deferred_final_unlock_access_v1",
        "schema_version": 95,
        "loaded_files": files,
        "loaded_file_inventory_sha256": deferred.canonical_sha256_v95(files),
        "forbidden_roots": [str(path) for path in deferred._forbidden_roots(config)],
        "forbidden_component_names": ["oracle", "qa"],
        "block_forbidden": True,
        "forbidden_accesses": [],
        "protected_read_count": 0,
        "passed": True,
    }


def _materialization() -> dict[str, Any]:
    stage_order = (
        "generate",
        "render",
        "features",
        "maps",
        "memory",
        "qa_raw",
        "qa_select",
        "questions",
    )
    stages = {
        stage: {
            "authorized_entrypoint": ["python", "-m", "fixed", stage],
            "child_argv": [["fixed-child", stage]],
            "expected_outputs": [f"fixed/{stage}.artifact"],
        }
        for stage in stage_order
    }
    return {
        "authenticated": True,
        "preregistration_file_sha256": _digest("materialization-prereg"),
        "preregistration_identity_sha256": _digest("materialization-identity"),
        "stage_order": list(stage_order),
        "stages": stages,
        "recipe": {"recipe_sha256": _digest("recipe")},
    }


def _patch_prerequisites(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _config(tmp_path)
    candidate = _candidate()
    gate = _gate(candidate)
    absence = _absence(config)
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
    ]:
        del require_absence
        return config, gate, candidate, absence, access, _materialization()

    monkeypatch.setattr(deferred, "_prerequisites", prerequisites)
    return config, gate, candidate


def test_passing_preflight_is_model_free_and_does_not_authorize_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config_value, _gate_value, candidate = _patch_prerequisites(
        monkeypatch, tmp_path
    )

    result = deferred.preflight_deferred_final_v95("synthetic.yaml")

    assert result["status"] == deferred.PREFLIGHT_STATUS
    assert result["candidate_fingerprint_sha256"] == candidate["fingerprint_sha256"]
    assert result["known_development_gate_passed"] is True
    assert result["legacy_plan_file_count_opened"] == 0
    assert result["final_label_file_count_opened"] == 0
    assert result["scene_generation_performed"] is False
    assert result["model_loaded"] is False
    contract = result["execution_contract"]
    assert len(contract["materialization_commands"]) == 8
    assert contract["generation_inputs_derived_without_legacy_plans"] is True
    assert contract["every_stage_reauthenticates_unlock_before_execution"] is True
    assert contract["required_scene_ids"] == list(deferred.DEFERRED_FINAL_SCENES)
    assert contract["required_prediction_arms"] == [
        "v95_primary",
        "v95_zero_payload",
        "v95_full_interior_permutation",
        "v95_paired_wrong_scene",
        "fixed_v94_primary_same_rows",
    ]
    assert contract["labels_may_be_opened_only_after_prediction_bundle_authentication"]
    assert contract["automatic_runtime_promotion"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        {"known_development_gate_passed": False},
        {"deferred_final_unlock_eligible": False},
        {"fixed_final_checkpoint_immutable": False},
        {"scene_prefix_question_independent": False},
        {"protected_read_count": 1},
        {"runtime_promotion_authorized": True},
        {"status": "measured_preregistered_gate_not_passed"},
        {"gate_results": {"accuracy": True, "causal_control": False}},
    ],
)
def test_unlock_gate_fails_closed_on_every_required_condition(
    mutation: dict[str, Any],
) -> None:
    candidate = _candidate()
    gate = {**_gate(candidate), **mutation}
    with pytest.raises(deferred.DeferredFinalBlockedError, match="remains locked"):
        deferred._assert_passing_gate(gate, candidate)


def test_unlock_gate_rejects_a_different_fixed_final() -> None:
    candidate = _candidate()
    gate = _gate(candidate)
    gate["candidate_fingerprint_sha256"] = _digest("different")
    with pytest.raises(deferred.DeferredFinalBlockedError, match="remains locked"):
        deferred._assert_passing_gate(gate, candidate)


def test_prerequisite_order_checks_gate_before_deferred_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    candidate = _candidate()
    gate = {**_gate(candidate), "known_development_gate_passed": False}
    absence_called = False

    def mark_absence(_config_value: dict[str, Any]) -> dict[str, Any]:
        nonlocal absence_called
        absence_called = True
        return _absence(config)

    monkeypatch.setattr(deferred, "load_config_v95", lambda *_a, **_k: config)
    monkeypatch.setattr(
        deferred, "authenticate_final_evidence_v95", lambda *_a, **_k: gate
    )
    monkeypatch.setattr(
        deferred,
        "authenticate_fixed_final_candidate_v95",
        lambda *_a, **_k: candidate,
    )
    monkeypatch.setattr(
        deferred,
        "authenticate_materialization_preregistration_v95",
        lambda *_a, **_k: _materialization(),
    )
    monkeypatch.setattr(deferred, "assert_deferred_final_absent_v95", mark_absence)

    with pytest.raises(deferred.DeferredFinalBlockedError, match="remains locked"):
        deferred._prerequisites("synthetic.yaml", require_absence=True)
    assert absence_called is False


def test_unlock_is_atomic_authenticatable_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_prerequisites(monkeypatch, tmp_path)
    output = tmp_path / "unlock.json"
    monkeypatch.setattr(deferred, "DEFAULT_UNLOCK", output)

    created = deferred.unlock_deferred_final_v95("synthetic.yaml")
    reused = deferred.unlock_deferred_final_v95("synthetic.yaml")

    assert created["authenticated"] is True
    assert created["reused_authenticated_unlock"] is False
    assert reused["authenticated"] is True
    assert reused["reused_authenticated_unlock"] is True
    assert created["unlock_file_sha256"] == reused["unlock_file_sha256"]
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == deferred.UNLOCK_STATUS


def test_unlock_authentication_detects_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_prerequisites(monkeypatch, tmp_path)
    output = tmp_path / "unlock.json"
    monkeypatch.setattr(deferred, "DEFAULT_UNLOCK", output)
    deferred.unlock_deferred_final_v95("synthetic.yaml")
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["model_loaded"] = True
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="no longer match"):
        deferred.authenticate_deferred_final_unlock_v95("synthetic.yaml")


def test_existing_unlock_authentication_does_not_reassert_current_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    candidate = _candidate()
    gate = _gate(candidate)
    absence = _absence(config)
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
    ]:
        calls.append(require_absence)
        return config, gate, candidate, absence, access, _materialization()

    monkeypatch.setattr(deferred, "_prerequisites", prerequisites)
    output = tmp_path / "unlock.json"
    monkeypatch.setattr(deferred, "DEFAULT_UNLOCK", output)
    deferred.unlock_deferred_final_v95("synthetic.yaml")
    calls.clear()

    deferred.authenticate_deferred_final_unlock_v95("synthetic.yaml")

    assert calls == [False]


def test_materialization_preflight_authenticates_without_executing_a_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_prerequisites(monkeypatch, tmp_path)
    output = tmp_path / "unlock.json"
    monkeypatch.setattr(deferred, "DEFAULT_UNLOCK", output)
    deferred.unlock_deferred_final_v95("synthetic.yaml")

    template = deferred.materialization_template_v95("synthetic.yaml")
    assert template["status"].startswith("fixed_preregistered")
    assert template["legacy_plan_file_count_opened"] == 0
    assert template["final_label_file_count_opened"] == 0
    result = deferred.materialization_preflight_v95("synthetic.yaml")
    assert result["status"] == "materialization_preflight_passed_no_stage_executed"
    assert result["stage_execution_performed"] is False


def test_arbitrary_parallel_unlock_paths_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="one fixed unlock path"):
        deferred._unlock_path(tmp_path / "parallel-unlock.json")


def test_absence_record_rejects_generated_or_nonempty_claims(tmp_path: Path) -> None:
    config = _config(tmp_path)
    absence = _absence(config)
    absence["scene_generation_performed"] = True
    with pytest.raises(ValueError, match="absence attestation"):
        deferred._validate_absence_record(absence, config)


def test_file_audit_blocks_legacy_plan_and_qa_reads(tmp_path: Path) -> None:
    config = _config(tmp_path)
    plan = Path(config["deferred_final_lock"]["legacy_plan_files_never_opened"][0])
    label = Path(config["deferred_final_lock"]["empty_qa_placeholders"][0])
    plan.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("forbidden", encoding="utf-8")
    label.write_text("forbidden", encoding="utf-8")
    audit = deferred.FileAccessAudit(
        deferred._forbidden_roots(config),
        forbidden_component_names={"oracle", "qa"},
        block_forbidden=True,
    )
    with audit:
        with pytest.raises(PermissionError, match="Blocked forbidden"):
            plan.read_text(encoding="utf-8")
        with pytest.raises(PermissionError, match="Blocked forbidden"):
            label.read_text(encoding="utf-8")


def test_module_has_no_generation_model_or_legacy_final_once_dependency() -> None:
    source = inspect.getsource(deferred)
    forbidden_imports = (
        "import blender",
        "import subprocess",
        "from semantic_3d_chat.data.scene_variants",
        "from semantic_3d_chat.evaluation.final_once",
        "load_local_language_model",
        "Gemma4ForConditionalGeneration",
    )
    assert not any(value in source for value in forbidden_imports)
    assert "authenticate_materialization_preregistration_v95" in source
    assert "every_stage_reauthenticates_unlock_before_execution" in source


def test_makefile_exposes_only_v95_specific_deferred_transition() -> None:
    makefile = (deferred.PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    for target in (
        "v95-deferred-final-check:",
        "v95-deferred-final-preflight:",
        "v95-deferred-final-unlock:",
        "v95-deferred-final-authenticate:",
        "v95-deferred-final-template:",
        "v95-deferred-final-materialization-preflight:",
        "v95-deferred-final-preregister-materialization:",
        "v95-deferred-final-authenticate-materialization-preregistration:",
    ):
        assert target in makefile
    section = makefile[makefile.index("v95-deferred-final-check:") :]
    assert "-m semantic_3d_chat.evaluation.v95_deferred_final" in section
    assert "$(MAKE) gemma4-final-once" not in section
