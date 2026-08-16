from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch

from semantic_3d_chat.config import artifact_root, load_config
from semantic_3d_chat.language.lora import lora_banks_settings
from semantic_3d_chat.training.pair_curriculum import build_exact_question_pair_units
from semantic_3d_chat.training.train_block_cross_v35 import load_v35_train_qa_records
from semantic_3d_chat.training.train_scene_ingress_kv_v37 import (
    _QUERY_CONSTRUCTION_STATE_SHA256,
    _QUERY_MODULES,
    _SOURCE_QUERY_STATE_SHA256,
    _TARGET_BANK,
    _TARGET_PARAMETER_NAMES,
    OPTIMIZER_AUDIT_FILENAME,
    _optimizer_payload_audit,
    _source_replay_attestation,
    build_v37_schedule,
    optimizer_step_audit,
    preflight_v37,
    require_exact_v36_source,
    v37_contract,
    v37_loader_config,
    v37_loss_values,
    v37_settings,
    validate_v37_training_cache_boundary,
)

CONFIG = Path("configs/experiments/gemma4_diverse28_scene_ingress_kv_v37.yaml")


def _config() -> dict:
    return load_config(CONFIG)


def test_v37_contract_locks_learned_target_and_frozen_query_runtime() -> None:
    config = _config()
    contract = v37_contract(config)
    settings = v37_settings(config)
    banks = lora_banks_settings(config)
    target = banks.bank(_TARGET_BANK)
    query = banks.bank("extension_v30_joint_pair_query")
    assert settings.saved_optimizer_steps == (0, 8, 16, 24, 32, 40, 48, 56, 64)
    assert settings.learning_rate == 2e-5
    assert contract.source_query_state_sha256 == _SOURCE_QUERY_STATE_SHA256
    assert target.trainable is True
    assert target.initialization_algorithm == "checkpoint_overwrite"
    assert target.initialization_seed is None
    assert query.trainable is False
    assert query.adapter.target_modules == _QUERY_MODULES
    assert query.initialization_algorithm == "checkpoint_overwrite"
    assert query.initialization_seed is None
    assert query.expected_initial_state_sha256 == _SOURCE_QUERY_STATE_SHA256


def test_v37_loader_copy_restores_only_query_construction_contract() -> None:
    actual = _config()
    loader = v37_loader_config(actual)
    actual_banks = lora_banks_settings(actual)
    loader_banks = lora_banks_settings(loader)
    query = loader_banks.bank("extension_v30_joint_pair_query")
    assert actual_banks.bank(_TARGET_BANK).trainable is True
    assert loader_banks.bank(_TARGET_BANK).trainable is False
    assert query.trainable is True
    assert query.initialization_algorithm == "cpu_kaiming_uniform_a_exact_zero_b"
    assert query.initialization_seed == 30030
    assert query.expected_initial_state_sha256 == _QUERY_CONSTRUCTION_STATE_SHA256
    assert actual_banks.bank("extension_v30_joint_pair_query").trainable is False


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("training.v37_scene_ingress_kv", "learning_rate", 2e-4),
        ("v37_scene_ingress_kv", "source_optimizer_state_loaded", True),
        ("v37_scene_ingress_kv", "validation_qa_loaded_during_training", True),
        ("v37_scene_ingress_kv", "target_bank_source_state_sha256", "0" * 64),
        ("v37_scene_ingress_kv", "schedule_sha256", "0" * 64),
    ],
)
def test_v37_contract_fails_closed(section: str, field: str, value: object) -> None:
    config = copy.deepcopy(_config())
    target = config
    for key in section.split("."):
        target = target[key]
    target[field] = value
    with pytest.raises((TypeError, ValueError)):
        v37_contract(config)


def test_v37_preflight_never_opens_source_adam_validation_maps_or_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_optimizer = (
        Path("data_gemma4/checkpoints/gemma4_v36_diverse28_joint_block_cross/update_016")
        / "optimizer.pt"
    ).resolve()
    original_open = Path.open
    original_read_text = Path.read_text

    def guarded_open(path: Path, *args, **kwargs):
        if path.resolve() == source_optimizer:
            raise AssertionError("V37 preflight opened inherited V36 Adam state")
        return original_open(path, *args, **kwargs)

    def guarded_read_text(path: Path, *args, **kwargs):
        if path.name == "validation.jsonl" or "oracle" in {
            part.casefold() for part in path.parts
        }:
            raise AssertionError("V37 preflight crossed its data boundary")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    report = preflight_v37(_config())
    assert report["passed"] is True
    assert report["source_optimizer_file_opened"] is False
    assert report["scene_maps_loaded"] is False
    assert report["validation_qa_loaded"] is False


def test_v37_schedule_is_exact_priority_then_two_cycles_then_priority_tail() -> None:
    config = _config()
    records, _ = load_v35_train_qa_records(v37_loader_config(config))
    units = build_exact_question_pair_units(records)
    schedule, audit = build_v37_schedule(records, units, seed=int(config["seed"]))
    assert len(schedule) == 64
    assert audit["schedule_sha256"] == (
        "76a123412d4bd3aeee012515b37095c22d9cbf9eb56934b622d715daca45fa2b"
    )
    assert audit["appearance_counts_by_family"] == {
        "book_support": 15,
        "mirror_lr": 8,
        "other": 26,
        "picture_support": 15,
    }
    assert audit["broad_answer_type_counts"] == {
        "attribute": 10,
        "count": 9,
        "metric": 9,
        "orientation": 9,
        "presence": 9,
        "spatial_relation": 9,
        "support": 9,
    }


def test_v37_training_cache_boundary_requires_exact_16_train_maps() -> None:
    config = v37_loader_config(_config())
    contract = v37_contract(_config())
    train = tuple(_config()["v37_scene_ingress_kv"]["train_scene_ids"])
    validation = tuple(_config()["v37_scene_ingress_kv"]["validation_scene_ids"])
    loaded = [
        str((artifact_root(config, "maps") / scene / "voxel_map.npz").resolve())
        for scene in train
    ]
    audit = {
        "scene_count": 16,
        "scene_ids": list(train),
        "scene_scope": "training_only",
        "authenticated_manifest_scene_count": 22,
        "authenticated_manifest_train_subset_count": 16,
        "validation_scene_ids_loaded": [],
        "validation_environment_maps_loaded": False,
        "deferred_final_scene_ids_loaded": [],
        "loaded_environment_files": loaded,
    }
    caches = {scene: object() for scene in train}
    result = validate_v37_training_cache_boundary(
        cache_audit=audit,
        caches=caches,
        config=config,
        train_scene_ids=train,
        validation_scene_ids=validation,
    )
    assert result["validation_environment_maps_loaded"] is False
    assert contract.saved_optimizer_steps[-1] == 64
    bad = copy.deepcopy(audit)
    bad["scene_ids"][-1] = validation[0]
    with pytest.raises(RuntimeError, match="train-map boundary"):
        validate_v37_training_cache_boundary(
            cache_audit=bad,
            caches=caches,
            config=config,
            train_scene_ids=train,
            validation_scene_ids=validation,
        )


def test_v37_source_replay_requires_deterministic_current_prefix_hashes() -> None:
    config = _config()
    _, metadata, _ = require_exact_v36_source(config)
    row = metadata["history"][-1]
    pair = copy.deepcopy(row["training_pair_metrics"])
    pair["cross_prefix_complete_units_by_family"] = {}
    pair["complete_physical_pair_coverage"] = 7
    train = tuple(config["v37_scene_ingress_kv"]["train_scene_ids"])
    current_hashes = {
        scene: hashlib.sha256(f"current-{scene}".encode()).hexdigest() for scene in train
    }
    prefix = {
        "source_prefix_scene_count": 16,
        "source_prefix_scene_ids": list(train),
        "source_prefix_sha256_by_scene": dict(current_hashes),
        "replayed_prefix_sha256_by_scene": dict(current_hashes),
        "source_prefixes_replayed_bit_exact": True,
        "current_v36_u16_prefixes_recomputed_deterministically_from_exact_tensors": True,
        "external_prefix_manifest_used": False,
    }
    report = _source_replay_attestation(
        source_metadata=metadata,
        pair_metrics=pair,
        broad_nll=float(row["training_broad_nll"]),
        residual=row["training_residual_diagnostics"],
        prefix_replay=prefix,
        expected_scene_ids=train,
    )
    assert (
        report[
            "current_v36_u16_prefixes_recomputed_deterministically_from_exact_tensors"
        ]
        is True
    )
    assert report["external_prefix_manifest_used"] is False
    prefix["source_prefix_sha256_by_scene"][train[0]] = "0" * 64
    with pytest.raises(ValueError, match="prefixes did not replay"):
        _source_replay_attestation(
            source_metadata=metadata,
            pair_metrics=pair,
            broad_nll=float(row["training_broad_nll"]),
            residual=row["training_residual_diagnostics"],
            prefix_replay=prefix,
            expected_scene_ids=train,
        )


def _optimizer_fixture(step: int) -> tuple[dict, dict[str, torch.Tensor]]:
    parameters = [torch.nn.Parameter(torch.zeros(index + 1)) for index in range(8)]
    optimizer = torch.optim.AdamW(
        [
            {
                "name": f"lora_banks.{_TARGET_BANK}",
                "params": parameters,
                "parameter_names": list(_TARGET_PARAMETER_NAMES),
                "lr": 2e-5,
                "weight_decay": 0.0,
            }
        ]
    )
    for _ in range(step):
        optimizer.zero_grad(set_to_none=True)
        sum(parameter.sum() for parameter in parameters).backward()
        optimizer.step()
    tensors = {
        name: parameter.detach().clone()
        for name, parameter in zip(_TARGET_PARAMETER_NAMES, parameters, strict=True)
    }
    return optimizer.state_dict(), tensors


def _write_optimizer_arm(checkpoint: Path, payload: dict, step: int) -> None:
    checkpoint.mkdir()
    optimizer_path = checkpoint / "optimizer.pt"
    torch.save(payload, optimizer_path)
    digest = hashlib.sha256(optimizer_path.read_bytes()).hexdigest()
    (checkpoint / OPTIMIZER_AUDIT_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact": "v37_optimizer_integrity_manifest",
                "optimizer_step": step,
                "optimizer_filename": "optimizer.pt",
                "optimizer_sha256": digest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_v37_optimizer_audit_locks_full_adamw_schema_and_self_hash(tmp_path: Path) -> None:
    payload, tensors = _optimizer_fixture(2)
    checkpoint = tmp_path / "update_002"
    _write_optimizer_arm(checkpoint, payload, 2)
    audit = optimizer_step_audit(checkpoint, expected_step=2, tensors=tensors)
    assert audit["exact_adamw_group_schema_verified"] is True
    assert audit["self_hash_linkage_verified"] is True

    tampered = copy.deepcopy(payload)
    tampered["param_groups"][0]["maximize"] = True
    with pytest.raises(ValueError, match="group identity/order/settings"):
        _optimizer_payload_audit(tampered, expected_step=2, tensors=tensors)
    tampered = copy.deepcopy(payload)
    tampered["param_groups"][0]["betas"] = (0.8, 0.999)
    with pytest.raises(ValueError, match="group identity/order/settings"):
        _optimizer_payload_audit(tampered, expected_step=2, tensors=tensors)

    # A finite moment edit is detected by the separately persisted file hash.
    changed = torch.load(checkpoint / "optimizer.pt", weights_only=True)
    changed["state"][0]["exp_avg"].add_(1.0)
    torch.save(changed, checkpoint / "optimizer.pt")
    with pytest.raises(ValueError, match="integrity manifest"):
        optimizer_step_audit(checkpoint, expected_step=2, tensors=tensors)


def test_v37_loss_reports_frozen_residual_without_optimizing_it() -> None:
    settings = v37_settings(_config())
    optimized, reported = v37_loss_values(
        settings=settings,
        broad_nll=2.0,
        pair_correct_nll=3.0,
        side_hinge=4.0,
        cross_prefix_hinge=5.0,
        frozen_normalized_residual=6.0,
    )
    assert optimized == pytest.approx(0.25 * 2 + 0.5 * 3 + 4 * 4 + 8 * 5)
    assert reported == pytest.approx(optimized + 0.001 * 6)
    source = inspect.getsource(
        __import__(
            "semantic_3d_chat.training.train_scene_ingress_kv_v37",
            fromlist=["run_v37"],
        ).run_v37
    )
    assert "residual_penalty(" not in source
    assert "train_optimized_loss" in source
    assert "residual_penalty_contributes_gradient" in source
