from __future__ import annotations

import ast
import inspect
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from semantic_3d_chat.chat import v96_explicit_candidate_cli as cli
from semantic_3d_chat.chat import v96_explicit_candidate_runtime as runtime
from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.scene_encoder.v81_scene_memory_artifact import (
    LoadedV81SceneMemory,
)


def _digest(character: str = "a") -> str:
    return character * 64


def _candidate() -> dict:
    return {
        "artifact": "gemma4_v96_fixed_final_fingerprint_v1",
        "directory_inventory": ["bridge.safetensors", "runtime_metadata.json"],
        "weights_sha256": _digest("1"),
        "metadata_file_sha256": _digest("2"),
        "metadata_canonical_sha256": _digest("3"),
        "state_sha256": _digest("4"),
        "tensor_inventory_sha256": _digest("5"),
        "training_report_sha256": _digest("6"),
        "config_sha256": _digest("7"),
        "preregistration_sha256": _digest("8"),
        "cpu_preflight_sha256": _digest("9"),
        "attestation_file_sha256": _digest("c"),
        "attestation_identity_sha256": _digest("d"),
        "v2_implementation_seal_sha256": _digest("e"),
        "fixed_final_optimizer_updates": 285,
        "frozen_v95_state_sha256": _digest("a"),
        "known_development_scored": False,
        "deferred_final_generated": False,
        "runtime_promotion_authorized": False,
        "fingerprint_sha256": _digest("b"),
    }


def _evidence(candidate: dict | None = None) -> dict:
    candidate = _candidate() if candidate is None else candidate
    return {
        "artifact": runtime.FINAL_GATE_ARTIFACT,
        "schema_version": 96,
        "status": runtime.FINAL_GATE_PASS_STATUS,
        "candidate_fingerprint_sha256": candidate["fingerprint_sha256"],
        "candidate_state_sha256": candidate["state_sha256"],
        "frozen_v95_state_sha256": candidate["frozen_v95_state_sha256"],
        "config_sha256": candidate["config_sha256"],
        "preregistration_sha256": candidate["preregistration_sha256"],
        "cpu_preflight_sha256": candidate["cpu_preflight_sha256"],
        "training_report_sha256": candidate["training_report_sha256"],
        "implementation_seal_sha256": candidate["v2_implementation_seal_sha256"],
        "implementation_source_inventory_sha256": _digest("f"),
        "v1_implementation_seal_sha256": _digest("0"),
        "candidate_attestation_file_sha256": candidate[
            "attestation_file_sha256"
        ],
        "candidate_attestation_identity_sha256": candidate[
            "attestation_identity_sha256"
        ],
        "candidate_attestation_immutable": True,
        "final_score_sha256": _digest("e"),
        "evidence_sha256": _digest("f"),
        "gate_results": {
            "correct_minimum": True,
            "causal_nll_minimum": True,
            "prefix_invariant": True,
        },
        "known_development_gate_passed": True,
        "deferred_final_unlock_eligible": True,
        "deferred_final_unlock_requires_explicit_separate_command": True,
        "scene_prefix_question_independent": True,
        "fixed_final_checkpoint_immutable": True,
        "frozen_v95_parent_immutable": True,
        "protected_read_count": 0,
        "row_level_content_serialized": False,
        "automatic_runtime_promotion": False,
        "runtime_promotion_authorized": False,
        "authenticated": True,
    }


def _authorization_payload() -> dict:
    root = Path.cwd().resolve()
    return {
        "artifact": runtime.AUTHORIZATION_ARTIFACT,
        "schema_version": 96,
        "status": runtime.AUTHORIZATION_STATUS,
        "authorization_config_path": str(
            root / "configs/experiments/gemma4_v96_atomic_pair_repair.yaml"
        ),
        "authorization_config_sha256": _digest("1"),
        "runtime_config_path": str(
            root / "configs/runtime/gemma4_v85_strict_multiscene.yaml"
        ),
        "runtime_config_file_sha256": _digest("2"),
        "runtime_config_effective_sha256": _digest("3"),
        "v85_checkpoint_path": str(
            root / "reports/gemma4/artifacts/v85_strict_runtime_candidate"
        ),
        "v85_adapter_sha256": _digest("4"),
        "v85_metadata_sha256": _digest("5"),
        "v94_bridge_path": str(
            root / "reports/gemma4/artifacts/v94_strict_multiscene_full40_final"
        ),
        "v94_weights_sha256": _digest("6"),
        "v94_metadata_sha256": _digest("7"),
        "v94_state_sha256": runtime.V94_STATE_SHA256,
        "v95_bridge_path": str(
            root / "reports/gemma4/artifacts/v95_strict_causal_successor_final"
        ),
        "v95_weights_sha256": _digest("8"),
        "v95_metadata_sha256": _digest("9"),
        "v95_state_sha256": _digest("a"),
        "v96_candidate_path": str(
            root / "reports/gemma4/artifacts/v96_atomic_pair_repair_final"
        ),
        "v96_weights_sha256": _digest("b"),
        "v96_metadata_file_sha256": _digest("c"),
        "v96_metadata_canonical_sha256": _digest("d"),
        "v96_state_sha256": _digest("e"),
        "candidate_fingerprint_sha256": _digest("f"),
        "config_sha256": _digest("1"),
        "preregistration_sha256": _digest("2"),
        "cpu_preflight_sha256": _digest("3"),
        "training_report_sha256": _digest("4"),
        "final_score_path": str(
            root
            / "reports/gemma4/metrics/gemma4_v96_atomic_pair_repair_known_development.json"
        ),
        "final_score_sha256": _digest("5"),
        "evidence_path": str(
            root
            / "reports/gemma4/metrics/gemma4_v96_atomic_pair_repair_known_development_evidence.json"
        ),
        "evidence_sha256": _digest("6"),
        "implementation_seal_sha256": _digest("7"),
        "implementation_source_inventory_sha256": _digest("8"),
        "v1_implementation_seal_sha256": _digest("a"),
        "v2_implementation_seal_sha256": _digest("7"),
        "candidate_attestation_file_sha256": _digest("b"),
        "candidate_attestation_identity_sha256": _digest("c"),
        "candidate_attestation_immutable": True,
        "gate_results_sha256": _digest("9"),
        "gate_count": 3,
        "all_gate_results_passed": True,
        "candidate_authenticated": True,
        "pass_evidence_authenticated": True,
        "known_development_gate_passed": True,
        "scene_prefix_question_independent": True,
        "row_level_content_serialized": False,
        "environmental_text_inputs": [],
        "deferred_final_unlock_eligible": True,
        "automatic_runtime_promotion": False,
        "runtime_promotion_authorized": False,
        "explicit_candidate_flag_required": True,
    }


def _authorization() -> runtime.V96CandidateAuthorization:
    return runtime.V96CandidateAuthorization.from_payload(_authorization_payload())


def _loaded_memory(scene_id: str = "scene_000039") -> LoadedV81SceneMemory:
    return LoadedV81SceneMemory(
        root=Path("numeric-memory"),
        memory=torch.zeros((1, 738, 1536), dtype=torch.bfloat16),
        metadata={
            "scene_id": scene_id,
            "shape": [1, 738, 1536],
            "fixed_memory_tokens": 738,
            "hidden_size": 1536,
            "compiled_before_user_question": True,
            "question_inputs_used_for_compilation": False,
            "question_dependent_scene_processing": False,
            "question_dependent_retrieval": False,
            "semantic_or_spatial_top_k_selection": False,
            "environmental_text_inputs": [],
            "questions_or_answers_serialized": False,
            "oracle_loaded": False,
        },
    )


def test_v96_pass_contract_requires_every_gate_and_exact_candidate_binding() -> None:
    candidate = _candidate()
    result = runtime.validate_v96_pass_evidence(
        candidate=candidate,
        evidence=_evidence(candidate),
    )

    assert result["all_gate_results_passed"] is True
    assert result["gate_count"] == 3
    assert result["candidate_state_sha256"] == candidate["state_sha256"]


@pytest.mark.parametrize("mutation", ("failed_gate", "wrong_candidate", "promoted"))
def test_v96_pass_contract_fails_closed(mutation: str) -> None:
    candidate = _candidate()
    evidence = _evidence(candidate)
    if mutation == "failed_gate":
        evidence["gate_results"]["causal_nll_minimum"] = False
    elif mutation == "wrong_candidate":
        evidence["candidate_fingerprint_sha256"] = _digest("0")
    else:
        evidence["runtime_promotion_authorized"] = True

    with pytest.raises(ValueError):
        runtime.validate_v96_pass_evidence(candidate=candidate, evidence=evidence)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("pass_evidence_authenticated", False),
        ("known_development_gate_passed", False),
        ("runtime_promotion_authorized", True),
        ("environmental_text_inputs", ["forbidden"]),
        ("row_level_content_serialized", True),
    ),
)
def test_v96_authorization_rejects_gate_or_leakage_drift(
    field: str, value: object
) -> None:
    payload = _authorization_payload()
    payload[field] = value
    with pytest.raises(ValueError):
        runtime.V96CandidateAuthorization.from_payload(payload)


def test_v96_extension_is_exact_three_frozen_banks_after_v85() -> None:
    settings = runtime.frozen_v96_extension_settings(_authorization())

    assert tuple(bank.name for bank in settings.banks) == (
        runtime.V94_BANK,
        runtime.V95_BANK,
        runtime.V96_BANK,
    )
    assert all(bank.trainable is False for bank in settings.banks)
    assert settings.banks[0].adapter.target_modules == runtime.V94_TARGETS
    assert settings.banks[1].adapter.target_modules == runtime.V95_TARGETS
    assert settings.banks[2].adapter.target_modules == runtime.V96_TARGETS
    assert runtime.EXPECTED_BANKS[-3:] == tuple(bank.name for bank in settings.banks)
    assert runtime.TOTAL_PARAMETER_COUNT == 864_256


def test_v96_validates_real_local_v85_seven_bank_metadata_without_model() -> None:
    path = Path("reports/gemma4/artifacts/v85_strict_runtime_candidate/runtime_metadata.json")
    metadata = json.loads(path.read_text(encoding="utf-8"))

    runtime.validate_v96_v85_base_checkpoint_contract(metadata)

    metadata["lora_bank_state_sha256"][runtime.BASE_BANKS[0]] = _digest("0")
    with pytest.raises(ValueError, match="bank changed"):
        runtime.validate_v96_v85_base_checkpoint_contract(metadata)


def test_v96_memory_accepts_only_complete_bfloat16_question_free_tensor() -> None:
    loaded = _loaded_memory()
    runtime.validate_v96_scene_memory_contract(scene_id="scene_000039", loaded=loaded)

    loaded.metadata["question_dependent_retrieval"] = True
    with pytest.raises(ValueError, match="oracle-free"):
        runtime.validate_v96_scene_memory_contract(
            scene_id="scene_000039", loaded=loaded
        )


def test_v96_runtime_source_has_no_v96_evaluation_or_training_import() -> None:
    tree = ast.parse(inspect.getsource(runtime))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any(
        module.startswith(("semantic_3d_chat.evaluation", "semantic_3d_chat.training"))
        for module in imported
    )


def test_v96_hook_is_explicit_and_does_not_modify_default_pointer() -> None:
    hook = cli.load_v96_runtime_hook()
    raw = yaml_safe_load(hook.path)
    contract = raw["v96_candidate_runtime"]

    assert hook.default_scene == "scene_000039"
    assert hook.runtime_config.name == "gemma4_v85_strict_multiscene.yaml"
    assert contract["mode"] == cli.HOOK_MODE
    assert contract["require_explicit_candidate_flag"] is True
    assert contract["default_runtime_pointer_modified"] is False
    assert contract["runtime_promotion_authorized"] is False
    assert contract["environmental_text_inputs"] == []


def yaml_safe_load(path: Path) -> dict:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_v96_isolated_authorization_accepts_one_strict_hash_only_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _authorization_payload()
    completed = subprocess.CompletedProcess(
        args=["python"],
        returncode=0,
        stdout=json.dumps(payload) + "\n",
        stderr="",
    )
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: completed)

    result = cli.run_isolated_v96_authorization(
        payload["authorization_config_path"]
    )

    assert result.pass_evidence_authenticated is True
    assert result.environmental_text_inputs == ()


def test_v96_cli_requires_explicit_candidate_flag_before_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def forbidden(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(cli, "run_isolated_v96_authorization", forbidden)

    assert cli.main([]) == 2
    assert called is False


def test_v96_invalid_authorization_is_rejected_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization = replace(_authorization(), pass_evidence_authenticated=False)
    called = False

    def forbidden(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(runtime.StaticChatRuntime, "load", forbidden)
    config = load_runtime_config("configs/runtime/gemma4_v85_strict_multiscene.yaml")

    with pytest.raises(ValueError, match="authenticated PASS"):
        runtime.V96ExplicitCandidateChatRuntime.load(
            config,
            "scene_000039",
            authorization=authorization,
            scene_memory="unused",
        )
    assert called is False
