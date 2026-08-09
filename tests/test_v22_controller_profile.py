from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from semantic_3d_chat.config import load_config
from semantic_3d_chat.evaluation import (
    v21_epoch_selector as shared_selector,
)
from semantic_3d_chat.evaluation import (
    v21_extension_controller as shared_extension,
)
from semantic_3d_chat.evaluation import (
    v21_update1_verifier as shared_update1,
)
from semantic_3d_chat.evaluation.phase_aware_local_field_profile import (
    V21_LOCAL_FIELD_PROFILE,
    V22_LOCAL_FIELD_PROFILE,
)
from semantic_3d_chat.evaluation.v21_epoch_selector import (
    _validate_config,
)
from semantic_3d_chat.evaluation.v21_extension_controller import (
    V21ExtensionViolation,
    _require_extension_decision,
)
from semantic_3d_chat.evaluation.v21_structural_preflight import (
    V21StructuralPreflightViolation,
    validate_v21_config_contract,
)
from semantic_3d_chat.evaluation.v22_extension_controller import (
    prepare_v22_extension_launch,
    select_v22_final_extension,
)


def _v22_config():
    return load_config(V22_LOCAL_FIELD_PROFILE.config_path)


def _continuation_screen() -> dict[str, object]:
    color = {
        "full_vocab_sides": 12,
        "full_vocab_units": 6,
        "minimum_candidate_margin": 0.5,
        "minimum_full_vocab_margin": 0.5,
    }
    mirror = {
        "full_vocab_sides": 8,
        "full_vocab_units": 2,
        "minimum_candidate_margin": -0.5,
        "minimum_full_vocab_margin": -0.5,
    }
    row = {
        "epoch": 3,
        "color": color,
        "mirror": mirror,
    }
    return {
        "selector_type": V22_LOCAL_FIELD_PROFILE.selector_type,
        "report_only": True,
        "model_inference_executed": False,
        "gemma_model_loaded": False,
        "checkpoint_tensor_state_loaded": True,
        "checkpoint_tensor_state_safely_inspected": True,
        "optimizer_deserialized": True,
        "optimizer_deserialization_weights_only": True,
        "model_dtype": "bfloat16",
        "continuation_authorized": True,
        "continuation_gate_passed": True,
        "full_teacher_gate_passed": False,
        "greedy_audit_authorized": False,
        "greedy_audit_forbidden": True,
        "decision": "continue_selected_epoch_no_greedy_audit",
        "conditional_max_optimizer_updates": 8,
        "selected_epoch": 3,
        "selection_policy": {
            "continuation_requires": {
                "mirror_minimum_full_vocab_sides": 8,
                "mirror_minimum_full_vocab_units": 2,
            }
        },
        "epochs": [
            {**copy.deepcopy(row), "epoch": epoch} for epoch in (1, 2, 3, 4)
        ],
    }


def test_v22_profile_pins_exact_hash_margins_roles_and_namespaces() -> None:
    contract = validate_v21_config_contract(_v22_config(), profile=V22_LOCAL_FIELD_PROFILE)
    assert contract["contract_sha256"] == V22_LOCAL_FIELD_PROFILE.normalized_contract_sha256
    assert contract["resolved_config_hash"] == V22_LOCAL_FIELD_PROFILE.resolved_config_hash
    assert contract["role"] == V22_LOCAL_FIELD_PROFILE.preflight_role
    assert contract[V22_LOCAL_FIELD_PROFILE.screen_key]["role"] == (
        V22_LOCAL_FIELD_PROFILE.screen_role
    )
    assert contract["pair_objective_policy"]["by_pair"]["pair_000003"] == {
        "role": "signed_target",
        "language_nll_weight": 0.0,
        "candidate_hinge_weight": 8.0,
        "candidate_margin": 0.25,
        "full_vocab_hinge_weight": 2.0,
        "full_vocab_margin": 0.25,
    }
    selector = _validate_config(_v22_config(), profile=V22_LOCAL_FIELD_PROFILE)
    assert selector["objective_policy"] == contract["pair_objective_policy"]
    assert selector["screen"]["primary_output_namespace"] == (
        V22_LOCAL_FIELD_PROFILE.output_namespace
    )
    assert selector["screen"]["extension_output_namespace"] == (
        V22_LOCAL_FIELD_PROFILE.extension_namespace
    )


def test_v21_and_v22_authorization_profiles_are_not_interchangeable() -> None:
    v21 = load_config(V21_LOCAL_FIELD_PROFILE.config_path)
    with pytest.raises(V21StructuralPreflightViolation, match="config hash"):
        validate_v21_config_contract(v21, profile=V22_LOCAL_FIELD_PROFILE)
    with pytest.raises(V21StructuralPreflightViolation, match="config hash"):
        validate_v21_config_contract(_v22_config(), profile=V21_LOCAL_FIELD_PROFILE)


def _nested_reduction(profile):
    values = {
        "sources": {"implementation": "shared-v21-family-source"},
        "structural": {"parameter_count": 196_608},
        "dependence": {"local": True},
        "ranks": {"minimum": 2},
        "centered": {"verified": True},
        "raw_scene": {"scene": "raw"},
        "effective_scene": {"scene": "effective"},
        "precision": {"dtype": "bfloat16"},
        "raw_pair": {"pair": "raw"},
        "effective_pair": {"pair": "effective"},
        "phase": {"algorithm": "phase-aware"},
        "functional": {"passed": True},
        "structural_gate": {"passed": True},
    }
    return (
        shared_update1._build_rich_preflight_reduction(**values)
        if profile is V21_LOCAL_FIELD_PROFILE
        else shared_update1._build_rich_preflight_reduction(**values, profile=profile)
    ), values["sources"]


def test_nested_rich_reduction_is_bound_to_v22_and_not_interchangeable() -> None:
    v22_reduction, sources = _nested_reduction(V22_LOCAL_FIELD_PROFILE)
    assert v22_reduction["preflight_contract_sha256"] == (
        V22_LOCAL_FIELD_PROFILE.normalized_contract_sha256
    )
    assert (
        shared_selector._validate_rich_preflight_reduction(
            v22_reduction,
            implementation_sources=sources,
            profile=V22_LOCAL_FIELD_PROFILE,
        )
        == v22_reduction
    )
    with pytest.raises(
        shared_selector.V21EpochSelectorViolation,
        match="preflight_contract_sha256",
    ):
        shared_selector._validate_rich_preflight_reduction(
            v22_reduction,
            implementation_sources=sources,
        )

    v21_reduction, sources = _nested_reduction(V21_LOCAL_FIELD_PROFILE)
    assert (
        shared_selector._validate_rich_preflight_reduction(
            v21_reduction,
            implementation_sources=sources,
        )
        == v21_reduction
    )
    with pytest.raises(
        shared_selector.V21EpochSelectorViolation,
        match="preflight_contract_sha256",
    ):
        shared_selector._validate_rich_preflight_reduction(
            v21_reduction,
            implementation_sources=sources,
            profile=V22_LOCAL_FIELD_PROFILE,
        )


def _functional_measurements(mirror_margin: float) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for pair_id, scenes in {
        "pair_000001": ("scene_000003", "scene_000004"),
        "pair_000003": ("scene_000007", "scene_000008"),
    }.items():
        margin = 0.5 if pair_id == "pair_000001" else mirror_margin
        for index in range(6):
            result.append(
                {
                    "pair_id": pair_id,
                    "question_key": f"q{index}",
                    "sides": [
                        {
                            "scene_id": scene_id,
                            "candidate_margin": margin,
                            "full_vocab_margin": margin,
                        }
                        for scene_id in scenes
                    ],
                }
            )
    return result


def test_v22_executed_nested_policies_reject_self_consistent_v21_targets() -> None:
    v22_contract = validate_v21_config_contract(
        _v22_config(),
        profile=V22_LOCAL_FIELD_PROFILE,
    )
    v21_contract = validate_v21_config_contract(
        load_config(V21_LOCAL_FIELD_PROFILE.config_path)
    )
    v22_policies = shared_update1._contract_pair_policies(v22_contract)
    v21_policies = shared_update1._contract_pair_policies(v21_contract)
    coverage = shared_update1._expected_pair_policy_coverage(v22_contract)
    microsteps = [
        {
            "pair_id": "pair_000003",
            "pair_objective_policy": copy.deepcopy(v22_policies["pair_000003"]),
        }
        for _ in range(12)
    ]
    preflight = {
        "pair_objective_policy": copy.deepcopy(v22_contract["pair_objective_policy"]),
        "pair_objective_policy_coverage": copy.deepcopy(coverage),
    }
    shared_update1._validate_executed_pair_policy_evidence(
        preflight,
        v22_contract,
        microsteps,
    )

    stale_microsteps = copy.deepcopy(microsteps)
    stale_microsteps[0]["pair_objective_policy"] = copy.deepcopy(
        v21_policies["pair_000003"]
    )
    with pytest.raises(shared_update1.V21Update1Violation, match="microstep 1"):
        shared_update1._validate_executed_pair_policy_evidence(
            preflight,
            v22_contract,
            stale_microsteps,
        )

    stale_coverage = copy.deepcopy(coverage)
    stale_coverage["resolved_by_pair"]["pair_000003"] = copy.deepcopy(
        v21_policies["pair_000003"]
    )
    coverage_body = {
        key: value for key, value in stale_coverage.items() if key != "coverage_sha256"
    }
    stale_coverage["coverage_sha256"] = shared_update1.canonical_sha256(coverage_body)
    stale_preflight = {**preflight, "pair_objective_policy_coverage": stale_coverage}
    with pytest.raises(shared_update1.V21Update1Violation, match="coverage"):
        shared_update1._validate_executed_pair_policy_evidence(
            stale_preflight,
            v22_contract,
            microsteps,
        )

    v22_functional = shared_update1._expected_functional_policies(v22_contract)
    audit = shared_update1.evaluate_v21_predicted_update(
        _functional_measurements(-0.5),
        _functional_measurements(-0.25),
        policies=v22_functional,
        color_pair_id="pair_000001",
        mirror_pair_id="pair_000003",
    )
    assert shared_update1._validate_functional_audit(audit, v22_contract) == audit

    # Recompute every derived functional field under stale V21 targets. This
    # is internally self-consistent evidence, but it is not V22 evidence.
    stale_audit = shared_update1.evaluate_v21_predicted_update(
        _functional_measurements(-0.5),
        _functional_measurements(-0.25),
        policies=shared_update1._expected_functional_policies(v21_contract),
        color_pair_id="pair_000001",
        mirror_pair_id="pair_000003",
    )
    with pytest.raises(shared_update1.V21Update1Violation, match="functional policies"):
        shared_update1._validate_functional_audit(stale_audit, v22_contract)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda profile: replace(profile, resolved_config_hash="f" * 12),
        lambda profile: replace(profile, normalized_contract_sha256="f" * 64),
        lambda profile: replace(profile, mirror_candidate_margin=1.0),
        lambda profile: replace(profile, mirror_full_vocab_margin=1.0),
        lambda profile: replace(profile, output_namespace="wrong_namespace"),
    ],
)
def test_v22_profile_mutation_fails_closed_before_execution(mutation) -> None:
    with pytest.raises(V21StructuralPreflightViolation):
        validate_v21_config_contract(_v22_config(), profile=mutation(V22_LOCAL_FIELD_PROFILE))


def test_v22_continuation_requires_its_own_selector_identity() -> None:
    screen = _continuation_screen()
    assert _require_extension_decision(screen, profile=V22_LOCAL_FIELD_PROFILE) == 3
    stale = copy.deepcopy(screen)
    stale["selector_type"] = V21_LOCAL_FIELD_PROFILE.selector_type
    with pytest.raises(V21ExtensionViolation, match="selector_type"):
        _require_extension_decision(stale, profile=V22_LOCAL_FIELD_PROFILE)
    with pytest.raises(V21ExtensionViolation, match="selector_type"):
        _require_extension_decision(screen)


def test_v22_wrappers_pass_only_the_immutable_v22_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []

    def prepare(config_path, screen_path, *, current_provenance, profile):
        calls.append(("prepare", profile))
        return {"config": str(config_path), "screen": str(screen_path)}

    def select(manifest_path, *, current_provenance, profile):
        calls.append(("select", profile))
        return {"manifest": str(manifest_path)}

    monkeypatch.setattr(shared_extension, "prepare_extension_launch", prepare)
    monkeypatch.setattr(shared_extension, "select_final_extension", select)
    # The wrapper binds imported functions, so patch those bindings as well.
    monkeypatch.setattr(
        "semantic_3d_chat.evaluation.v22_extension_controller.prepare_extension_launch",
        prepare,
    )
    monkeypatch.setattr(
        "semantic_3d_chat.evaluation.v22_extension_controller.select_final_extension",
        select,
    )
    prepare_v22_extension_launch("config", "screen", current_provenance={"clean": True})
    select_v22_final_extension("manifest", current_provenance={"clean": True})
    assert calls == [
        ("prepare", V22_LOCAL_FIELD_PROFILE),
        ("select", V22_LOCAL_FIELD_PROFILE),
    ]


def test_v22_make_targets_use_only_v22_controller_modules() -> None:
    makefile = (V22_LOCAL_FIELD_PROFILE.config_path.parents[2] / "Makefile").read_text(
        encoding="utf-8"
    )
    for target in (
        "gemma4-v22-preflight",
        "gemma4-v22-stage1",
        "gemma4-v22-verify-update1",
        "gemma4-v22-select",
        "gemma4-v22-prepare-extension",
        "gemma4-v22-select-extension",
    ):
        assert f"{target}:" in makefile
    assert "semantic_3d_chat.evaluation.v22_structural_preflight" in makefile
    assert "semantic_3d_chat.evaluation.v22_update1_verifier" in makefile
    assert "semantic_3d_chat.evaluation.v22_epoch_selector" in makefile
    assert "semantic_3d_chat.evaluation.v22_extension_controller" in makefile
