from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.evaluation import predict_v96_known_development_v2 as predictor_v2
from semantic_3d_chat.evaluation import v96_known_development_candidate_attestation as attestation
from semantic_3d_chat.evaluation import v96_known_development_common as common_v1
from semantic_3d_chat.evaluation import v96_known_development_common_v2 as common_v2
from semantic_3d_chat.evaluation import v96_known_development_implementation_v2 as implementation
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import sha256_file_v85
from semantic_3d_chat.evaluation.v96_evaluation_io_v2 import (
    read_json_strict_v96_v2,
    read_jsonl_strict_v96_v2,
    write_json_create_once_v96_v2,
)
from semantic_3d_chat.evaluation.v96_known_development_common import (
    canonical_sha256_v96,
    evaluation_paths_v96,
)
from semantic_3d_chat.evaluation.v96_known_development_common_v2 import (
    prediction_forbidden_roots_v96,
)
from semantic_3d_chat.evaluation.v96_known_development_implementation import (
    authenticate_evaluation_implementation_v96,
)
from semantic_3d_chat.language.lora import tensor_state_sha256


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _synthetic_attested_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, object], Path, Path]:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("v96: {}\n", encoding="utf-8")
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    state = {
        "adapters.0.lora_a": torch.zeros((8, 1536), dtype=torch.float32),
        "adapters.0.lora_b": torch.zeros((4096, 8), dtype=torch.float32),
    }
    weights = candidate_root / "bridge.safetensors"
    save_file(state, str(weights))
    weights_sha256 = sha256_file_v85(weights)
    state_sha256 = tensor_state_sha256(state)
    metadata = {
        "artifact": "gemma4_v96_atomic_pair_repair_fixed_final_v1",
        "schema_version": 96,
        "status": "fixed_final_awaiting_known_development_gate",
        "parent": "v95_fixed_final_nonpromoted_optimization_parent",
        "bank_name": "v96_atomic_pair_repair_bridge",
        "target_modules": ["model.language_model.layers.9.self_attn.q_proj"],
        "rank": 8,
        "alpha": 16.0,
        "dropout": 0.0,
        "parameter_count": 45_056,
        "state_sha256": state_sha256,
        "weights_sha256": weights_sha256,
        "tensor_inventory": ["adapters.0.lora_a", "adapters.0.lora_b"],
        "environmental_memory_serialized": False,
        "questions_or_answers_serialized": False,
        "oracle_serialized": False,
        "known_development_scored": False,
        "deferred_final_generated": False,
        "runtime_promotion_authorized": False,
        "bindings": {"synthetic": _digest("binding")},
    }
    metadata_path = candidate_root / "runtime_metadata.json"
    _write_json(metadata_path, metadata)
    aggregate_paths = {
        "training_report_sha256": tmp_path / "training.json",
        "preregistration_sha256": tmp_path / "prereg.json",
        "cpu_preflight_sha256": tmp_path / "cpu.json",
        "topology_smoke_sha256": tmp_path / "topology.json",
    }
    for index, path in enumerate(aggregate_paths.values()):
        _write_json(path, {"value": index})
    v1_seal = tmp_path / "v1-seal.json"
    _write_json(v1_seal, {"artifact": "synthetic-v1"})
    v2_seal = tmp_path / "v2-seal.json"
    training_qa = tmp_path / "data_diverse52/qa/train.jsonl"
    hub = tmp_path / "hub"
    model_root = hub / "models--synthetic--model"
    blobs = model_root / "blobs"
    snapshot = model_root / "snapshots/revision"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    model_blob = blobs / "model-blob"
    tokenizer_blob = blobs / "tokenizer-blob"
    model_blob.write_bytes(b"synthetic model weights")
    tokenizer_blob.write_bytes(b'{"synthetic":"tokenizer"}\n')
    (snapshot / "model.safetensors").symlink_to("../../blobs/model-blob")
    (snapshot / "tokenizer.json").symlink_to("../../blobs/tokenizer-blob")
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    inventory = {
        "adapters.0.lora_a": [8, 1536, "torch.float32"],
        "adapters.0.lora_b": [4096, 8, "torch.float32"],
    }
    pins: dict[str, object] = {
        "weights_sha256": weights_sha256,
        "metadata_file_sha256": sha256_file_v85(metadata_path),
        "metadata_canonical_sha256": canonical_sha256_v96(metadata),
        "state_sha256": state_sha256,
        "tensor_inventory_sha256": canonical_sha256_v96(inventory),
        **{key: sha256_file_v85(path) for key, path in aggregate_paths.items()},
        "config_sha256": sha256_file_v85(config_path),
        "frozen_v95_state_sha256": _digest("parent"),
        "fixed_final_optimizer_updates": 285,
        "known_development_scored": False,
        "deferred_final_generated": False,
        "runtime_promotion_authorized": False,
    }
    fingerprint = {
        "artifact": "gemma4_v96_fixed_final_fingerprint_v1",
        "directory_inventory": ["bridge.safetensors", "runtime_metadata.json"],
        "weights_sha256": pins["weights_sha256"],
        "metadata_file_sha256": pins["metadata_file_sha256"],
        "metadata_canonical_sha256": pins["metadata_canonical_sha256"],
        "state_sha256": pins["state_sha256"],
        "tensor_inventory_sha256": pins["tensor_inventory_sha256"],
        "training_report_sha256": pins["training_report_sha256"],
        "config_sha256": pins["config_sha256"],
        "preregistration_sha256": pins["preregistration_sha256"],
        "cpu_preflight_sha256": pins["cpu_preflight_sha256"],
        "fixed_final_optimizer_updates": 285,
        "frozen_v95_state_sha256": pins["frozen_v95_state_sha256"],
        "known_development_scored": False,
        "deferred_final_generated": False,
        "runtime_promotion_authorized": False,
    }
    pins["fingerprint_sha256"] = canonical_sha256_v96(fingerprint)
    access_paths = sorted(
        {
            str(config_path.resolve()),
            str(training_qa.resolve()),
            str(weights.resolve()),
            str(metadata_path.resolve()),
            *(str(path.resolve()) for path in aggregate_paths.values()),
        }
    )
    access_pins = {
        "unique_paths": access_paths,
        "unique_path_count": len(access_paths),
        "unique_path_inventory_sha256": canonical_sha256_v96(access_paths),
        "protected_read_count": 0,
        "known_development_questions_opened": False,
        "known_development_labels_opened": False,
        "oracle_opened": False,
        "model_loaded": False,
    }
    config = {
        "sources": {
            "training_qa": str(training_qa),
            "model_id": "synthetic/model",
            "model_revision": "revision",
            "model_blob_sha256_identity": sha256_file_v85(model_blob),
        },
        "known_development_gate": {
            "labels_path": str(tmp_path / "known-development-labels.jsonl")
        },
        "outputs": {
            "fixed_final_candidate": str(candidate_root),
            "training_report": str(aggregate_paths["training_report_sha256"]),
            "preregistration": str(aggregate_paths["preregistration_sha256"]),
            "cpu_preflight": str(aggregate_paths["cpu_preflight_sha256"]),
            "topology_smoke": str(aggregate_paths["topology_smoke_sha256"]),
        },
    }
    model_snapshot_binding = implementation.build_model_snapshot_binding_v96_v2(
        config
    )
    frozen_bank_expected_states = {
        name: _digest(f"sealed-{name}")
        for name in (*implementation.V94_BANKS, implementation.V95_BANK_NAME)
    }
    frozen_bank_inventory_sha256 = canonical_sha256_v96(
        frozen_bank_expected_states
    )
    topology_names = (
        *implementation.V94_BANKS,
        implementation.V95_BANK_NAME,
        implementation.V96_BANK_NAME,
    )
    lora_bank_topology = {
        "schema_version": 2,
        "enabled": True,
        "banks": [
            {
                "name": name,
                "trainable": index == 9,
                "rank": 8,
                "alpha": 16.0,
                "dropout": 0.0,
                "target_modules": [f"synthetic.target.{index}"],
                "initialization_algorithm": "checkpoint_overwrite",
                "initialization_seed": None,
                "expected_initial_state_sha256": (
                    frozen_bank_expected_states.get(name, state_sha256)
                ),
            }
            for index, name in enumerate(topology_names)
        ],
    }
    lora_bank_topology_sha256 = canonical_sha256_v96(lora_bank_topology)
    _write_json(
        v2_seal,
        {
            "candidate_pins": pins,
            "candidate_auth_access": access_pins,
            "model_snapshot_binding": model_snapshot_binding,
            "frozen_bank_expected_states": frozen_bank_expected_states,
            "frozen_bank_expected_state_inventory_sha256": (
                frozen_bank_inventory_sha256
            ),
            "lora_bank_topology": lora_bank_topology,
            "lora_bank_topology_sha256": lora_bank_topology_sha256,
        },
    )
    attestation_path = tmp_path / "attestation.json"
    payload = {
        "artifact": attestation.ARTIFACT,
        "schema_version": 96,
        "status": attestation.STATUS,
        "config_sha256": pins["config_sha256"],
        "candidate_fingerprint_sha256": pins["fingerprint_sha256"],
        "candidate_state_sha256": pins["state_sha256"],
        "candidate_weights_sha256": pins["weights_sha256"],
        "candidate_metadata_file_sha256": pins["metadata_file_sha256"],
        "candidate_metadata_canonical_sha256": pins["metadata_canonical_sha256"],
        "candidate_tensor_inventory_sha256": pins["tensor_inventory_sha256"],
        "training_report_sha256": pins["training_report_sha256"],
        "preregistration_sha256": pins["preregistration_sha256"],
        "cpu_preflight_sha256": pins["cpu_preflight_sha256"],
        "topology_smoke_sha256": pins["topology_smoke_sha256"],
        "frozen_v95_state_sha256": pins["frozen_v95_state_sha256"],
        "fixed_final_optimizer_updates": 285,
        "v1_implementation_seal_sha256": sha256_file_v85(v1_seal),
        "v2_implementation_seal_sha256": sha256_file_v85(v2_seal),
        "historical_v1_attempt_failed_before_question_io": True,
        "historical_v1_output_count": 0,
        "training_chain_authenticated_in_separate_model_free_process": True,
        "training_qa_content_serialized": False,
        "known_development_questions_opened": False,
        "known_development_labels_opened": False,
        "oracle_opened": False,
        "model_loaded": False,
        "row_level_content_serialized": False,
        "runtime_promotion_authorized": False,
        "base_model_snapshot_inventory_sha256": model_snapshot_binding[
            "inventory_sha256"
        ],
        "frozen_bank_expected_state_inventory_sha256": (
            frozen_bank_inventory_sha256
        ),
        "lora_bank_topology_sha256": lora_bank_topology_sha256,
        "weights_hashed_not_model_loaded": True,
        "candidate_auth_unique_paths": access_pins["unique_paths"],
        "candidate_auth_unique_path_count": access_pins["unique_path_count"],
        "candidate_auth_unique_path_inventory_sha256": access_pins[
            "unique_path_inventory_sha256"
        ],
        "protected_read_count": 0,
    }
    payload["attestation_identity_sha256"] = canonical_sha256_v96(payload)
    _write_json(attestation_path, payload)
    monkeypatch.setattr(attestation, "load_config_v96", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(attestation, "V1_IMPLEMENTATION_SEAL", v1_seal)
    monkeypatch.setattr(implementation, "IMPLEMENTATION_SEAL_V2", v2_seal)
    return config, config_path, attestation_path


def test_v1_seal_unchanged_and_failed_attempt_has_no_outputs() -> None:
    authenticated = authenticate_evaluation_implementation_v96()
    assert authenticated["seal_sha256"] == (
        "a22e436d8ea2d106e5a1b33a0ddcbf76e49ae9d5c71825bdf212e28cdb67cef6"
    )
    paths = evaluation_paths_v96({})
    assert not [
        path for path in paths.__dict__.values() if path.exists() or path.is_symlink()
    ]


def test_prediction_boundary_explicitly_contains_training_qa() -> None:
    config = {
        "known_development_gate": {
            "labels_path": "data_diverse52/qa/validation.jsonl"
        }
    }
    training = Path("data_diverse52/qa/train.jsonl").resolve()
    forbidden = prediction_forbidden_roots_v96(config)
    assert any(training == root or root in training.parents for root in forbidden)


def test_attested_candidate_auth_never_opens_training_qa(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config, config_path, attestation_path = _synthetic_attested_candidate(
        tmp_path, monkeypatch
    )
    training_qa = tmp_path / "data_diverse52/qa/train.jsonl"
    training_qa.parent.mkdir(parents=True)
    training_qa.write_text("training-only\n", encoding="utf-8")
    audit = FileAccessAudit([training_qa], block_forbidden=True)
    with audit:
        result = attestation.authenticate_candidate_attestation_v96(
            config_path,
            audit=audit,
            authenticate_implementation_sources=False,
            expected_implementation_seal_sha256=sha256_file_v85(
                implementation.IMPLEMENTATION_SEAL_V2
            ),
            attestation_path=attestation_path,
        )
    audit.assert_clean()
    assert result["artifact"] == "gemma4_v96_fixed_final_fingerprint_v1"
    assert str(training_qa.resolve()) not in audit.unique_paths
    assert result["directory_inventory"] == [
        "bridge.safetensors",
        "runtime_metadata.json",
    ]


def test_attested_candidate_auth_detects_aggregate_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, config_path, attestation_path = _synthetic_attested_candidate(
        tmp_path, monkeypatch
    )
    Path(str(config["outputs"]["training_report"])).write_text(
        '{"changed":true}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="attestation authentication failed"):
        attestation.authenticate_candidate_attestation_v96(
            config_path,
            authenticate_implementation_sources=False,
            expected_implementation_seal_sha256=sha256_file_v85(
                implementation.IMPLEMENTATION_SEAL_V2
            ),
            attestation_path=attestation_path,
        )


@pytest.mark.parametrize("logical_name", ["model.safetensors", "tokenizer.json"])
def test_model_snapshot_auth_records_every_target_and_rejects_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, logical_name: str
) -> None:
    config, _config_path, _attestation_path = _synthetic_attested_candidate(
        tmp_path, monkeypatch
    )
    sealed = read_json_strict_v96_v2(implementation.IMPLEMENTATION_SEAL_V2)
    expected = sealed["model_snapshot_binding"]
    audit = FileAccessAudit()
    with audit:
        observed = implementation.authenticate_model_snapshot_v96_v2(
            config, expected=expected, audit=audit
        )
    expected_targets = {
        entry["physical_target"] for entry in expected["logical_entries"].values()
    }
    assert expected_targets <= set(audit.unique_paths)
    assert observed == expected
    target = Path(expected["logical_entries"][logical_name]["physical_target"])
    target.write_bytes(target.read_bytes() + b"mutated")
    with pytest.raises(ValueError, match="snapshot|weight identity"):
        implementation.authenticate_model_snapshot_v96_v2(
            config, expected=expected
        )


def test_loaded_lora_bank_auth_rejects_joint_live_and_settings_tamper() -> None:
    frozen = {
        name: _digest(f"sealed-{name}")
        for name in (*implementation.V94_BANKS, implementation.V95_BANK_NAME)
    }
    candidate = _digest("sealed-v96-candidate")
    topology_names = (*frozen, predictor_v2.FRESH_BANK_NAME)
    topology = {
        "schema_version": 2,
        "enabled": True,
        "banks": [
            {
                "name": name,
                "trainable": index == 9,
                "rank": 8,
                "alpha": 16.0,
                "dropout": 0.0,
                "target_modules": [f"synthetic.target.{index}"],
                "initialization_algorithm": "checkpoint_overwrite",
                "initialization_seed": None,
                "expected_initial_state_sha256": None,
            }
            for index, name in enumerate(topology_names)
        ],
    }

    def collection_for(
        states: dict[str, str], *, contract: dict[str, object] = topology
    ) -> SimpleNamespace:
        ordered_names = (*frozen, predictor_v2.FRESH_BANK_NAME)
        banks = tuple(
            SimpleNamespace(
                settings=SimpleNamespace(
                    name=name,
                    trainable=name == predictor_v2.FRESH_BANK_NAME,
                    # Simulate jointly rewritten mutable settings metadata.
                    expected_initial_state_sha256=states[name],
                )
            )
            for name in ordered_names
        )
        return SimpleNamespace(
            banks=banks,
            settings=SimpleNamespace(contract=lambda: contract),
            validate_state=lambda: None,
            state_sha256=lambda: dict(states),
        )

    sealed_states = {**frozen, predictor_v2.FRESH_BANK_NAME: candidate}
    assert predictor_v2.authenticate_loaded_lora_bank_states_v96_v2(
        collection_for(sealed_states),
        expected_frozen_bank_state_sha256=frozen,
        expected_candidate_state_sha256=candidate,
        expected_lora_bank_topology=topology,
    ) == sealed_states

    jointly_tampered = dict(sealed_states)
    jointly_tampered[implementation.V94_BANKS[0]] = _digest(
        "jointly-tampered-weight-and-metadata"
    )
    with pytest.raises(ValueError, match="sealed inventory"):
        predictor_v2.authenticate_loaded_lora_bank_states_v96_v2(
            collection_for(jointly_tampered),
            expected_frozen_bank_state_sha256=frozen,
            expected_candidate_state_sha256=candidate,
            expected_lora_bank_topology=topology,
        )

    tampered_topology = json.loads(json.dumps(topology))
    tampered_topology["banks"][0]["target_modules"] = ["tampered.target"]
    with pytest.raises(ValueError, match="sealed inventory"):
        predictor_v2.authenticate_loaded_lora_bank_states_v96_v2(
            collection_for(sealed_states, contract=tampered_topology),
            expected_frozen_bank_state_sha256=frozen,
            expected_candidate_state_sha256=candidate,
            expected_lora_bank_topology=topology,
        )


def test_frozen_bank_inventory_rejects_pinned_v85_metadata_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    v95_config_path = tmp_path / "v95.yaml"
    v95_config_path.write_text("synthetic-v95-contract\n", encoding="utf-8")
    checkpoint = tmp_path / "v85"
    checkpoint.mkdir()
    metadata_path = checkpoint / "runtime_metadata.json"
    frozen_names = implementation.V94_BANKS[:-1]
    frozen_states = {name: _digest(f"v85-{name}") for name in frozen_names}
    metadata = {
        "lora": {
            "adapter_parameter_count": 565_248,
            "trainable_adapter_parameter_count": 0,
            "banks": [{"name": name} for name in frozen_names],
        },
        "lora_bank_state_sha256": frozen_states,
    }
    _write_json(metadata_path, metadata)
    v95 = {
        "sources": {
            "frozen_v85_checkpoint": str(checkpoint),
            "frozen_v85_metadata_sha256": sha256_file_v85(metadata_path),
        },
        "frozen_stack": {
            "v94_bank_name": implementation.V94_BANKS[-1],
            "v94_bank_state_sha256": _digest("v94-final"),
        },
        "bridge": {"bank_name": implementation.V95_BANK_NAME},
    }
    config = {
        "sources": {
            "frozen_v95_config": str(v95_config_path),
            "frozen_v95_config_sha256": sha256_file_v85(v95_config_path),
        },
        "frozen_stack": {
            "v95_bank_name": implementation.V95_BANK_NAME,
            "v95_bank_state_sha256": _digest("v95-final"),
        },
    }
    monkeypatch.setattr(implementation, "V95_CONFIG", v95_config_path)
    monkeypatch.setattr(implementation, "V85_RUNTIME_METADATA", metadata_path)
    monkeypatch.setattr(
        implementation,
        "load_config_v95",
        lambda *_args, **_kwargs: v95,
    )
    observed = implementation.build_frozen_bank_expected_states_v96_v2(config)
    assert observed == {
        **frozen_states,
        implementation.V94_BANKS[-1]: _digest("v94-final"),
        implementation.V95_BANK_NAME: _digest("v95-final"),
    }

    metadata["lora_bank_state_sha256"][frozen_names[0]] = _digest(
        "tampered-v85-bank"
    )
    _write_json(metadata_path, metadata)
    with pytest.raises(ValueError, match="metadata bytes changed"):
        implementation.build_frozen_bank_expected_states_v96_v2(config)


def test_candidate_forbidden_roots_include_active_custom_model_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = tmp_path / "custom-hub"
    model_root = hub / "models--synthetic--model"
    model_root.mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    config = {
        "sources": {"model_id": "synthetic/model"},
        "known_development_gate": {"labels_path": tmp_path / "labels.jsonl"},
    }
    assert model_root.resolve() in {
        path.resolve() for path in implementation._candidate_forbidden_roots(config)
    }


def test_attested_candidate_auth_rejects_self_rehashed_access_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config, config_path, attestation_path = _synthetic_attested_candidate(
        tmp_path, monkeypatch
    )
    payload = read_json_strict_v96_v2(attestation_path)
    payload["candidate_auth_unique_path_inventory_sha256"] = _digest("forged")
    payload["attestation_identity_sha256"] = canonical_sha256_v96(
        {
            key: value
            for key, value in payload.items()
            if key != "attestation_identity_sha256"
        }
    )
    _write_json(attestation_path, payload)
    with pytest.raises(ValueError, match="attestation authentication failed"):
        attestation.authenticate_candidate_attestation_v96(
            config_path,
            authenticate_implementation_sources=False,
            expected_implementation_seal_sha256=sha256_file_v85(
                implementation.IMPLEMENTATION_SEAL_V2
            ),
            attestation_path=attestation_path,
        )


def test_attested_candidate_auth_rejects_removed_unverifiable_access_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config, config_path, attestation_path = _synthetic_attested_candidate(
        tmp_path, monkeypatch
    )
    payload = read_json_strict_v96_v2(attestation_path)
    payload["comprehensive_access_inventory_sha256"] = _digest("self-attested")
    payload["attestation_identity_sha256"] = canonical_sha256_v96(
        {
            key: value
            for key, value in payload.items()
            if key != "attestation_identity_sha256"
        }
    )
    _write_json(attestation_path, payload)
    with pytest.raises(ValueError, match="attestation authentication failed"):
        attestation.authenticate_candidate_attestation_v96(
            config_path,
            authenticate_implementation_sources=False,
            expected_implementation_seal_sha256=sha256_file_v85(
                implementation.IMPLEMENTATION_SEAL_V2
            ),
            attestation_path=attestation_path,
        )


def test_v2_strict_json_and_jsonl_reject_duplicates_nonfinite_and_symlinks(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        read_json_strict_v96_v2(duplicate)
    nonfinite = tmp_path / "nonfinite.jsonl"
    nonfinite.write_text('{"value":NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Non-finite"):
        read_jsonl_strict_v96_v2(nonfinite)
    physical = tmp_path / "physical.json"
    _write_json(physical, {"ok": True})
    linked = tmp_path / "linked.json"
    linked.symlink_to(physical)
    with pytest.raises(FileNotFoundError, match="symlink"):
        read_json_strict_v96_v2(linked)


@pytest.mark.parametrize(
    ("suffix", "content", "reader"),
    [
        ("json", '{"nested":{"values":[1e999]}}\n', read_json_strict_v96_v2),
        ("jsonl", '{"nested":[{"value":-1e999}]}\n', read_jsonl_strict_v96_v2),
    ],
)
def test_v2_strict_readers_reject_nested_exponent_overflow(
    tmp_path: Path, suffix: str, content: str, reader: object
) -> None:
    source = tmp_path / f"overflow.{suffix}"
    source.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="Non-finite"):
        reader(source)


def test_v2_create_once_rejects_symlinked_ancestor(tmp_path: Path) -> None:
    physical = tmp_path / "physical"
    physical.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(physical, target_is_directory=True)
    with pytest.raises(FileNotFoundError, match="symlink"):
        write_json_create_once_v96_v2(linked / "artifact.json", {"ok": True})
    assert not (physical / "artifact.json").exists()


def test_v2_seal_staging_rejects_symlinked_ancestor_before_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    physical = tmp_path / "physical"
    physical.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(physical, target_is_directory=True)

    def must_not_build(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("seal payload built before staging-path validation")

    monkeypatch.setattr(
        implementation,
        "build_evaluation_implementation_seal_v96_v2",
        must_not_build,
    )
    with pytest.raises(FileNotFoundError, match="symlink"):
        implementation.seal_evaluation_implementation_v96_v2(
            seal_path=linked / "seal.json"
        )
    assert list(physical.iterdir()) == []


def test_question_manifest_rejects_symlinked_ancestor_before_delegate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    physical = tmp_path / "physical"
    physical.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(physical, target_is_directory=True)
    monkeypatch.setattr(common_v2, "QUESTION_MANIFEST", linked / "questions.json")

    def must_not_run() -> object:
        raise AssertionError("V1 question loader ran after a linked-ancestor check")

    monkeypatch.setattr(common_v1, "load_known_questions_v95", must_not_run)
    with pytest.raises(FileNotFoundError, match="symlink"):
        common_v2.load_known_questions_v96()


def test_reference_loader_rejects_symlinked_ancestor_before_delegate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    physical = tmp_path / "physical"
    physical.mkdir()
    (physical / "labels.jsonl").write_text("{}\n", encoding="utf-8")
    linked = tmp_path / "linked"
    linked.symlink_to(physical, target_is_directory=True)

    def must_not_run(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("V1 reference loader ran after a linked-ancestor check")

    monkeypatch.setattr(common_v2, "load_references_v96_v1", must_not_run)
    config = {"known_development_gate": {"labels_path": linked / "labels.jsonl"}}
    with pytest.raises(FileNotFoundError, match="symlink"):
        common_v2.load_references_v96_v2(
            config, SimpleNamespace(source_qa_sha256=_digest("unused"))
        )


@pytest.mark.parametrize(
    "content",
    [
        '{"scene_id":"a","scene_id":"b"}\n',
        '{"nested":{"value":1e999}}\n',
    ],
)
def test_reference_loader_rejects_duplicate_and_nonfinite_jsonl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content: str,
) -> None:
    labels = tmp_path / "labels.jsonl"
    labels.write_text(content, encoding="utf-8")
    digest = sha256_file_v85(labels)
    monkeypatch.setattr(common_v2, "REFERENCE_SHA256", digest)

    def must_not_run(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("V1 reference loader ran before strict prevalidation")

    monkeypatch.setattr(common_v2, "load_references_v96_v1", must_not_run)
    config = {"known_development_gate": {"labels_path": labels}}
    with pytest.raises(ValueError, match="Duplicate|Non-finite"):
        common_v2.load_references_v96_v2(
            config, SimpleNamespace(source_qa_sha256=digest)
        )


@pytest.mark.parametrize("linked_leaf", ["manifest", "tensor"])
def test_memory_cache_rejects_symlinked_leaf_before_delegate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, linked_leaf: str
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    physical = tmp_path / "physical.bin"
    physical.write_bytes(b"physical")
    (cache / "manifest.json").write_text("{}\n", encoding="utf-8")
    for scene_id in common_v2.SCENE_IDS:
        (cache / f"{scene_id}.safetensors").write_bytes(b"tensor")
    target = (
        cache / "manifest.json"
        if linked_leaf == "manifest"
        else cache / f"{common_v2.SCENE_IDS[0]}.safetensors"
    )
    target.unlink()
    target.symlink_to(physical)
    monkeypatch.setattr(common_v1, "MEMORY_CACHE", cache)

    def must_not_run(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("V1 memory auth ran after a linked-leaf check")

    monkeypatch.setattr(common_v1, "authenticate_memory_cache_v95", must_not_run)
    with pytest.raises(FileNotFoundError, match="symlink"):
        common_v2.authenticate_memory_cache_v96()


def test_memory_cache_rejects_symlinked_ancestor_before_delegate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    physical = tmp_path / "physical"
    physical.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(physical, target_is_directory=True)
    monkeypatch.setattr(common_v1, "MEMORY_CACHE", linked)

    def must_not_run(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("V1 memory auth ran after a linked-ancestor check")

    monkeypatch.setattr(common_v1, "authenticate_memory_cache_v95", must_not_run)
    with pytest.raises(FileNotFoundError, match="symlink"):
        common_v2.authenticate_memory_cache_v96()


@pytest.mark.parametrize("extra_field", ["question", "answer"])
def test_access_receipt_rejects_row_content_fields(
    tmp_path: Path, extra_field: str
) -> None:
    loaded = tmp_path / "loaded.json"
    loaded.write_text("{}\n", encoding="utf-8")
    audit = FileAccessAudit([tmp_path / "forbidden"], block_forbidden=True)
    with audit:
        audit.record(loaded)
    receipt = common_v2.audit_report_v96(audit)
    receipt[extra_field] = "forbidden row content"
    with pytest.raises(ValueError, match="receipt changed"):
        common_v2.validate_access_receipt_v96_v2(
            receipt,
            forbidden_roots=[tmp_path / "forbidden"],
            mandatory={str(loaded.resolve())},
        )


@pytest.mark.parametrize(
    ("validator", "field_inventory"),
    [
        (
            common_v2.validate_prediction_completion_schema_v96_v2,
            common_v2._PREDICTION_COMPLETION_FIELDS,
        ),
        (
            common_v2.validate_nll_completion_schema_v96_v2,
            common_v2._NLL_COMPLETION_FIELDS,
        ),
    ],
)
@pytest.mark.parametrize("extra_field", ["question", "answer"])
def test_completion_receipts_reject_row_content_fields(
    validator: object, field_inventory: frozenset[str], extra_field: str
) -> None:
    receipt = {field: False for field in field_inventory}
    if "elapsed_seconds" in receipt:
        receipt["elapsed_seconds"] = 0.0
    receipt[extra_field] = "forbidden row content"
    with pytest.raises(ValueError, match="schema changed"):
        validator(receipt)


def test_v2_source_closure_is_transitive_and_excludes_unrelated_runtimes() -> None:
    closure = implementation.transitive_implementation_sources_v96_v2()
    assert len(closure) == 120
    assert implementation.IMPLEMENTATION_SOURCES_V2 == {
        **closure,
        "static:runtime_config": implementation.PROJECT_ROOT
        / "configs/runtime/gemma4_v85_strict_multiscene.yaml",
        "static:v95_config": implementation.V95_CONFIG,
        "static:v85_runtime_metadata": implementation.V85_RUNTIME_METADATA,
    }
    names = {path.name for path in closure.values()}
    assert {
        "score_v96_known_development.py",
        "train_v96_atomic_pair_repair.py",
        "local_lm.py",
        "gemma4_backend.py",
    } <= names
    relative = [path.relative_to(implementation.PROJECT_ROOT).parts for path in closure.values()]
    assert all("robot" not in parts and "mcp_server" not in parts for parts in relative)


def test_runtime_config_binding_rejects_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text("language:\n  model_id: fixed\n", encoding="utf-8")
    config = {
        "sources": {
            "runtime_config": str(runtime),
            "runtime_config_sha256": sha256_file_v85(runtime),
        }
    }
    monkeypatch.setattr(implementation, "PROJECT_ROOT", tmp_path)
    assert implementation.authenticate_runtime_config_input_v96_v2(config) == {
        "path": "runtime.yaml",
        "sha256": sha256_file_v85(runtime),
    }
    runtime.write_text("language:\n  model_id: mutated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="runtime config changed"):
        implementation.authenticate_runtime_config_input_v96_v2(config)


def test_v2_seal_checks_output_absence_before_historical_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = tmp_path / "prediction.jsonl"
    existing.write_text("existing\n", encoding="utf-8")
    monkeypatch.setattr(implementation, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(implementation, "_known_outputs", lambda: (existing,))
    monkeypatch.setattr(implementation, "CANDIDATE_ATTESTATION", tmp_path / "absent.json")

    def forbidden_auth(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("historical auth ran before absence check")

    monkeypatch.setattr(
        implementation, "authenticate_evaluation_implementation_v96", forbidden_auth
    )
    with pytest.raises(RuntimeError, match="must precede"):
        implementation.build_evaluation_implementation_seal_v96_v2()
