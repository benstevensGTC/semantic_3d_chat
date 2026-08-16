from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from semantic_3d_chat.evaluation import v94_strict_runtime_release as release
from semantic_3d_chat.evaluation.strict_direct_release_core import BridgeSourceContract


def _evidence(*, behavior: bool = True) -> dict[str, Any]:
    value: dict[str, Any] = {
        "artifact": release.v94_strict_multiscene_evidence.ARTIFACT,
        "passed": True,
        "behavior_score_present": True,
        "behavior_gate_passed": behavior,
        "candidate_state_sha256": release.V94_TRAINED_STATE_SHA256,
        "candidate_weights_sha256": "1" * 64,
        "candidate_metadata_sha256": "2" * 64,
        "memory_sha256": {
            scene: f"{index:x}" * 64 for index, scene in enumerate(release.SCENE_IDS, 3)
        },
        "score": {
            "score_sha256": "9" * 64,
            "behavior_gate_passed": behavior,
            "status": (
                "passed_awaiting_separate_leakage_packaging"
                if behavior
                else "measured_gate_not_passed"
            ),
        },
    }
    value["bundle_sha256"] = release._canonical_sha256(value)
    return value


def test_v94_authentication_always_requires_score_and_behavior_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def authenticate(*args: object, **kwargs: object) -> dict[str, Any]:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return _evidence()

    monkeypatch.setattr(
        release.v94_strict_multiscene_evidence,
        "authenticate_v94_evidence",
        authenticate,
    )
    monkeypatch.setattr(release, "load_bridge_source", lambda _contract: object())

    result = release.authenticate_v94_model_gate()

    assert result["behavior_gate_passed"] is True
    assert observed["kwargs"] == {
        "root": release.PROJECT_ROOT,
        "require_score": True,
        "require_behavior_pass": True,
    }
    assert observed["args"] == (release.EXPERIMENT_CONFIG,)


def test_v94_authentication_fails_closed_on_negative_behavior_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        release.v94_strict_multiscene_evidence,
        "authenticate_v94_evidence",
        lambda *_args, **_kwargs: _evidence(behavior=False),
    )
    with pytest.raises(ValueError, match="behavior gate"):
        release.authenticate_v94_model_gate()


def test_v94_runtime_payload_is_exact_v85_seven_plus_v94_and_rebinds_v28() -> None:
    payload = release.build_runtime_config_payload(_evidence())
    banks = payload["language"]["lora_banks"]
    parent_metadata = json.loads(
        (release.PARENT_CHECKPOINT / release.RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8")
    )

    assert tuple(banks) == release.EXPECTED_BANKS
    assert len(banks) == 8
    assert all(row["trainable"] is False for row in banks.values())
    assert all(
        banks[name]["expected_initial_state_sha256"]
        == parent_metadata["lora_bank_state_sha256"][name]
        for name in release.PARENT_BANKS
    )
    assert (
        banks["extension_v28_stage_b_query"]["expected_initial_state_sha256"]
        == "ac90fc60e944b792d41fc18a21daca3ed87a7ec634a7a5c8594339371b0631e9"
    )
    assert banks[release.V94_BANK] == {
        "trainable": False,
        "rank": 8,
        "alpha": 16.0,
        "dropout": 0.0,
        "initialization_algorithm": "checkpoint_overwrite",
        "initialization_seed": None,
        "expected_initial_state_sha256": release.V94_TRAINED_STATE_SHA256,
        "target_modules": [release.V94_TARGET],
    }


def test_v94_composition_delegates_to_strict_core_with_one_two_tensor_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def compose(**kwargs: object) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        observed.update(kwargs)
        return {"sentinel": torch.ones(1)}, {"added_tensor_count": 2}

    monkeypatch.setattr(release, "compose_exact_bank_archive", compose)
    tensors, metadata = release._composed_adapter(_evidence())
    bridges = observed["added_bridges"]

    assert set(tensors) == {"sentinel"}
    assert metadata["added_tensor_count"] == 2
    assert observed["base_checkpoint"] == release.PARENT_CHECKPOINT
    assert observed["expected_base_banks"] == release.PARENT_BANKS
    assert observed["expected_final_banks"] == release.EXPECTED_BANKS
    assert isinstance(bridges, tuple) and len(bridges) == 1
    assert isinstance(bridges[0], BridgeSourceContract)
    assert bridges[0].bank_name == release.V94_BANK
    assert bridges[0].parameter_count == 110_592


def test_v94_runtime_metadata_rebinds_every_bank_and_exposes_runtime_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        release,
        "_composed_adapter",
        lambda _evidence: (
            {
                "scene_encoder.weight": torch.ones(1),
                "block_cross_residual.weight": torch.zeros(1),
            },
            {},
        ),
    )

    metadata = release.build_runtime_metadata(
        _evidence(),
        promotion=release.PENDING_DECISION,
        smoke_report_sha256=None,
    )
    provenance = metadata["initialization_provenance"]["v94_strict_runtime_release"]

    assert tuple(row["name"] for row in metadata["lora"]["banks"]) == release.EXPECTED_BANKS
    assert metadata["lora"]["adapter_parameter_count"] == 675_840
    assert metadata["lora"]["trainable_adapter_parameter_count"] == 0
    assert all(
        row["expected_initial_state_sha256"] == metadata["lora_bank_state_sha256"][row["name"]]
        for row in metadata["lora"]["banks"]
    )
    assert metadata["question_dependent_scene_processing"] is False
    assert provenance == {
        "schema_version": 94,
        "source_v94_evidence_sha256": _evidence()["bundle_sha256"],
        "source_v94_score_sha256": "9" * 64,
        "v94_bridge_state_sha256": release.V94_TRAINED_STATE_SHA256,
        "model_acceptance_gate_passed": True,
        "model_gate_report_authenticated": True,
        "promotion_decision": release.PENDING_DECISION,
        "runtime_promotion_authorized": False,
        "smoke_report_sha256": None,
        "held_out_generalization_claim": True,
    }


def test_v94_candidate_memory_packager_visits_exact_six_attested_scenes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evidence = _evidence()
    observed: list[str] = []
    manifest = {"scenes": {scene: {} for scene in release.SCENE_IDS}}
    monkeypatch.setattr(release, "_cache_manifest", lambda _evidence: manifest)
    monkeypatch.setattr(
        release,
        "checkpoint_fingerprint",
        lambda _path: ("a" * 64, []),
    )

    def load_memory(
        scene_id: str, _evidence: object, _manifest: object
    ) -> tuple[torch.Tensor, dict[str, str]]:
        observed.append(scene_id)
        return torch.zeros(1), {"file_sha256": "b" * 64}

    def save_memory(destination: Path, _memory: torch.Tensor, **kwargs: object) -> dict[str, Any]:
        destination.mkdir()
        (destination / release.MEMORY_FILENAME).write_bytes(b"numeric")
        (destination / release.METADATA_FILENAME).write_text("{}", encoding="utf-8")
        scene_id = str(kwargs["scene_id"])
        return {
            "tensor_file_sha256": "c" * 64,
            "canonical_prefix_sha256": evidence["memory_sha256"][scene_id],
        }

    monkeypatch.setattr(release, "_load_attested_cache_memory", load_memory)
    monkeypatch.setattr(release, "save_v81_scene_memory", save_memory)
    destination = tmp_path / "bundle"

    result = release._package_candidate_memories(
        destination,
        evidence=evidence,
        checkpoint_sha256="d" * 64,
        runtime_config_sha256="e" * 64,
    )

    assert tuple(observed) == release.SCENE_IDS
    assert tuple(sorted(result)) == release.SCENE_IDS
    assert tuple(sorted(path.name for path in destination.iterdir())) == release.SCENE_IDS


def test_v94_rebound_promotion_copies_memory_tensor_bytes_exactly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "candidate" / "scene_000057"
    destination = tmp_path / "release" / "scene_000057"
    source.mkdir(parents=True)
    original = b"opaque-continuous-memory-bytes\x00\x01"
    (source / release.MEMORY_FILENAME).write_bytes(original)
    (source / release.METADATA_FILENAME).write_text("{}", encoding="utf-8")
    metadata = {
        "scene_id": "scene_000057",
        "source_base_checkpoint_sha256": "a" * 64,
        "runtime_config_sha256": "c" * 64,
        "canonical_prefix_sha256": "d" * 64,
    }

    def load(_root: Path, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(metadata=dict(metadata), memory=torch.zeros(1))

    monkeypatch.setattr(release, "load_v81_scene_memory", load)
    result = release._copy_rebound_v81_memory(
        source,
        destination,
        scene_id="scene_000057",
        source_checkpoint_sha256="a" * 64,
        destination_checkpoint_sha256="b" * 64,
        runtime_config_sha256="c" * 64,
    )

    assert (destination / release.MEMORY_FILENAME).read_bytes() == original
    assert result["candidate_tensor_file_sha256"] == result["release_tensor_file_sha256"]
    assert result["tensor_bytes_reused_exactly"] is True
    rebound = json.loads((destination / release.METADATA_FILENAME).read_text())
    assert rebound["source_base_checkpoint_sha256"] == "b" * 64


def test_v94_smoke_protocol_has_no_expected_answer_channel() -> None:
    command = release._smoke_command(
        "scene_000057",
        audit_path=Path("audit.json"),
        chat_path=Path("chat.jsonl"),
    )

    assert command.count("--question") == 2
    assert [
        command[index + 1] for index, token in enumerate(command) if token == "--question"
    ] == list(release._SMOKE_QUESTIONS)
    assert all(flag not in command for flag in ("--expected", "--answer", "--reference"))
    assert "--allow-candidate" in command


def test_v94_postprocess_audit_rejects_oracle_evidence_bridge_and_cache_reads() -> None:
    safe = {
        "loaded_files": [
            str(release.RUNTIME_CONFIG.resolve()),
            str((release.CANDIDATE_CHECKPOINT / "adapter.safetensors").resolve()),
            str(
                (release.CANDIDATE_MEMORY_ROOT / "scene_000057" / release.MEMORY_FILENAME).resolve()
            ),
        ]
    }
    unsafe = {
        "loaded_files": [
            *safe["loaded_files"],
            str((release.PROJECT_ROOT / "data/oracle/scene_000057.json").resolve()),
            str((release.V94_BRIDGE_CANDIDATE / "bridge.safetensors").resolve()),
            str((release.EVALUATION_CACHE / "scene_000057.safetensors").resolve()),
            str(release.EXPERIMENT_CONFIG.resolve()),
        ]
    }

    assert release._protected_smoke_reads(safe) == []
    assert len(release._protected_smoke_reads(unsafe)) == 4


def test_v94_prepare_fails_before_any_write_when_hardened_score_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def absent() -> dict[str, Any]:
        raise FileNotFoundError("V94 sealed aggregate score is required but absent")

    monkeypatch.setattr(release, "authenticate_v94_model_gate", absent)
    with pytest.raises(FileNotFoundError, match="score is required"):
        release.prepare_candidate()


def test_v94_release_module_does_not_import_training_or_label_scorer() -> None:
    source = inspect.getsource(release).casefold()

    assert "train_v94_strict_multiscene_full40" not in source
    assert "score_label_isolated_v94" not in source
    assert "evaluation_qa_reserved_for_label_scorer" not in source
    assert "require_score=true" in source
    assert "require_behavior_pass=true" in source
