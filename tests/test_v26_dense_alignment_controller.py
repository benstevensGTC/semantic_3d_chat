from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from semantic_3d_chat.config import config_hash, load_config
from semantic_3d_chat.evaluation import v26_dense_alignment_controller as controller
from semantic_3d_chat.scene_encoder.dense_alignment import construct_dense_alignment
from semantic_3d_chat.training.checkpointing import runtime_checkpoint_metadata

_CLEAN_PROVENANCE = {
    "schema_version": 1,
    "scope": "repository_excluding_generated_artifacts_v1",
    "available": True,
    "head_commit": "0" * 40,
    "head_tree": "1" * 40,
    "is_clean": True,
    "tracked_diff_sha256": "2" * 64,
}


def _calibration_audit() -> dict[str, object]:
    path = Path(__file__).resolve().parents[1] / controller.CALIBRATION_REPORT_PATH
    audit = json.loads(path.read_text(encoding="utf-8"))
    audit.update(
        {
            "pair_optimizer_state_empty_before_warmup": True,
            "pair_optimizer_rebuilt_after_warmup": True,
            "pair_optimizer_state_empty_after_warmup": True,
            "pair_optimizer_steps_before_qa": 0,
            "held_out_scene_gradient_access": False,
            "category_text_prototypes_serialized": False,
            "oracle_payload_retained": False,
        }
    )
    return audit


def _pair_gate(*, full_teacher: bool = False) -> dict[str, object]:
    source = controller.v25._load_json(
        controller.v25.SOURCE_CHECKPOINT / "metadata.json", "V24 source metadata"
    )
    gate = deepcopy(source["pair_candidate_gate"])
    mirror_units = gate["detail"]["units"][6:]
    for unit in mirror_units:
        for side in unit["sides"]:
            full = float(side["first_token_target_vs_best_other_logit_margin"])
            candidate = float(side["own_vs_alternate_candidate_logit_margin"])
            if full <= 0.0:
                side["first_token_target_vs_best_other_logit_margin"] = (
                    0.125 if full_teacher else full + 0.0625
                )
            if candidate <= 0.0:
                side["own_vs_alternate_candidate_logit_margin"] = (
                    0.125 if full_teacher else candidate + 0.0625
                )
            side["full_vocab_top1_passed"] = (
                side["first_token_target_vs_best_other_logit_margin"] > 0.0
            )
            side["own_preference_passed"] = (
                side["own_vs_alternate_candidate_logit_margin"] > 0.0
            )
    full_values = [
        float(side["first_token_target_vs_best_other_logit_margin"])
        for unit in mirror_units
        for side in unit["sides"]
    ]
    candidate_values = [
        float(side["own_vs_alternate_candidate_logit_margin"])
        for unit in mirror_units
        for side in unit["sides"]
    ]
    mirror = gate["by_pair"][controller.v25.MIRROR_PAIR_ID]
    mirror["first_answer_token_top1_accuracy"] = sum(
        value > 0.0 for value in full_values
    ) / 12
    mirror["first_answer_token_top1_unit_accuracy"] = sum(
        all(value > 0.0 for value in full_values[index : index + 2])
        for index in range(0, 12, 2)
    ) / 6
    mirror["minimum_first_answer_token_target_vs_best_other_logit_margin"] = min(
        full_values
    )
    mirror["mean_first_answer_token_target_vs_best_other_logit_margin"] = sum(
        full_values
    ) / 12
    mirror["minimum_own_vs_alternate_candidate_logit_margin"] = min(candidate_values)
    mirror["mean_own_vs_alternate_candidate_logit_margin"] = sum(candidate_values) / 12
    return gate


def _metadata(epoch: int = 1, *, dense_hash: str = "3" * 64) -> dict[str, object]:
    config = load_config(controller.CONFIG_PATH)
    source_metadata = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / controller.v25.SOURCE_CHECKPOINT
            / "metadata.json"
        ).read_text(encoding="utf-8")
    )
    return {
        "schema_version": 3,
        "epoch": epoch,
        "optimizer_step": epoch,
        "config_hash": config_hash(config),
        "output_namespace": controller.PRIMARY_NAMESPACE,
        "source_provenance": deepcopy(_CLEAN_PROVENANCE),
        "scene_ids": list(controller.PAIRED_QA_TRAIN_SCENES),
        "train_scene_ids": list(controller.PAIRED_QA_TRAIN_SCENES),
        "validation_scene_ids": [],
        "test_scene_ids": list(controller.FINAL_QA_TEST_SCENES),
        "gradient_accumulation": 12,
        "counterfactual_pair_unit_count": 12,
        "counterfactual_pair_unit_selection_sha256": (
            controller.EXPECTED_PAIR_UNIT_SELECTION_SHA256
        ),
        "training_counterfactual_pair_count": 2,
        "training_counterfactual_pair_membership_sha256": (
            controller.EXPECTED_PAIR_MEMBERSHIP_SHA256
        ),
        "pair_curriculum": {
            "enabled": True,
            "pair_only": True,
            "pair_only_scene_ids": list(controller.PAIRED_QA_TRAIN_SCENES),
            "steps_per_epoch": 12,
            "gate_enabled": True,
        },
        "dense_alignment_optimizer": {
            "name": "AdamW",
            "learning_rate": 0.0003,
            "weight_decay": 0.0,
        },
        "train_dense_alignment_only": True,
        "freeze_scene_adapter": True,
        "dense_alignment": controller.dense_alignment_settings(config).contract(),
        "dense_alignment_parameter_count": controller.EXPECTED_DENSE_PARAMETER_COUNT,
        "dense_alignment_initial_state_sha256": (
            controller.EXPECTED_DENSE_INITIAL_SHA256
        ),
        "dense_alignment_state_sha256": dense_hash,
        "dense_alignment_calibration": _calibration_audit(),
        "initialization_provenance": {
            "schema_version": 7,
            "mode": "frozen_named_lora_scene_stack_plus_zero_output_dense_alignment",
            "checkpoint": str(controller.v25.SOURCE_CHECKPOINT),
            "adapter_sha256": controller.v25.EXPECTED_SOURCE_ARTIFACTS[
                "adapter_sha256"
            ],
            "metadata_sha256": controller.v25.EXPECTED_SOURCE_ARTIFACTS[
                "metadata_sha256"
            ],
            "expected_adapter_sha256": controller.v25.EXPECTED_SOURCE_ARTIFACTS[
                "adapter_sha256"
            ],
            "expected_metadata_sha256": controller.v25.EXPECTED_SOURCE_ARTIFACTS[
                "metadata_sha256"
            ],
            "checkpoint_epoch": 1,
            "checkpoint_output_namespace": "gemma4_v24_shared_query",
            "checkpoint_config_hash": source_metadata["config_hash"],
            "checkpoint_source_provenance": source_metadata["source_provenance"],
            "initialize_named_lora_freeze_for_dense_alignment_transition": True,
            "optimizer_state_loaded": False,
            "history_loaded": False,
            "source_lora_bank_state_sha256": {
                name: controller.EXPECTED_FROZEN_HASHES[name]
                for name in controller.FROZEN_BANKS
            },
            "source_scene_state_sha256": controller.EXPECTED_FROZEN_HASHES[
                "scene"
            ],
            "expected_source_scene_state_sha256": (
                controller.EXPECTED_FROZEN_HASHES["scene"]
            ),
            "source_global_scene_residual_state_sha256": (
                controller.EXPECTED_FROZEN_HASHES["global"]
            ),
            "expected_source_global_scene_residual_state_sha256": (
                controller.EXPECTED_FROZEN_HASHES["global"]
            ),
            "source_signed_x_scene_residual_state_sha256": (
                controller.EXPECTED_FROZEN_HASHES["signed_x"]
            ),
            "expected_source_signed_x_scene_residual_state_sha256": (
                controller.EXPECTED_FROZEN_HASHES["signed_x"]
            ),
            "all_source_modules_frozen": True,
            "dense_alignment_initial_state_sha256": (
                controller.EXPECTED_DENSE_INITIAL_SHA256
            ),
            "expected_dense_alignment_initial_state_sha256": (
                controller.EXPECTED_DENSE_INITIAL_SHA256
            ),
            "dense_alignment_zero_output": True,
            "source_checkpoint_loaded_dense_alignment": False,
            "dense_alignment_calibration_authorized": True,
            "dense_alignment_calibration_final_state_sha256": (
                controller.EXPECTED_CALIBRATION_FINAL_SHA256
            ),
            "pair_optimizer_rebuilt_after_dense_alignment_calibration": True,
        },
        "pair_candidate_gate": _pair_gate(),
        "frozen_scene_state_sha256": controller.EXPECTED_FROZEN_HASHES["scene"],
        "frozen_global_scene_residual_state_sha256": (
            controller.EXPECTED_FROZEN_HASHES["global"]
        ),
        "frozen_signed_x_scene_residual_state_sha256": (
            controller.EXPECTED_FROZEN_HASHES["signed_x"]
        ),
        "global_scene_residual_state_sha256": controller.EXPECTED_FROZEN_HASHES[
            "global"
        ],
        "signed_x_scene_residual_state_sha256": controller.EXPECTED_FROZEN_HASHES[
            "signed_x"
        ],
        "lora_bank_state_sha256": {
            name: controller.EXPECTED_FROZEN_HASHES[name]
            for name in controller.FROZEN_BANKS
        },
        "question_dependent_scene_processing": False,
        "all_voxels_transformed": True,
    }


def _write_checkpoint(path: Path) -> dict[str, object]:
    config = load_config(controller.CONFIG_PATH)
    module = construct_dense_alignment(config, semantic_dim=3072)
    assert module is not None
    with torch.no_grad():
        module.alignment_b[0, 0] = 0.125
    dense_hash = module.state_sha256()
    assert dense_hash not in {
        controller.EXPECTED_DENSE_INITIAL_SHA256,
        controller.EXPECTED_CALIBRATION_FINAL_SHA256,
    }
    metadata = _metadata(dense_hash=dense_hash)
    path.mkdir()
    (path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    runtime = runtime_checkpoint_metadata(metadata)
    (path / "runtime_metadata.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    source_adapter = (
        Path(__file__).resolve().parents[1]
        / controller.v25.SOURCE_CHECKPOINT
        / "adapter.safetensors"
    )
    tensors = load_file(source_adapter, device="cpu")
    tensors.update(
        {
            f"dense_aligner.{name}": value.detach().cpu().contiguous()
            for name, value in module.state_dict().items()
        }
    )
    save_file(tensors, path / "adapter.safetensors")
    torch.save({"state": {}, "param_groups": []}, path / "optimizer.pt")
    return metadata


def test_v26_config_profile_split_optimizer_and_gates_are_hash_pinned() -> None:
    config = load_config(controller.CONFIG_PATH)
    contract = controller.v26_contract(config)

    assert config_hash(config, length=64) == controller.EXPECTED_CONFIG_SHA256
    assert (
        controller._canonical_sha256(contract)
        == controller.EXPECTED_CONTRACT_SHA256
    )
    assert contract["profile_version"] == 26
    assert tuple(contract["split_isolation"]["calibration_scene_ids"]) == (
        controller.CALIBRATION_SCENES
    )
    assert tuple(contract["split_isolation"]["final_qa_test_scene_ids"]) == (
        controller.FINAL_QA_TEST_SCENES
    )
    assert contract["warmup"]["max_optimizer_steps"] == 40
    assert contract["warmup"]["learning_rate"] == 0.005
    assert contract["warmup"]["early_stop_top1_accuracy"] == 1.0
    assert contract["warmup"]["early_stop_minimum_margin"] == 0.10
    assert contract["warmup"]["delta_rms_cap"] == 1.0
    assert contract["warmup"]["delta_abs_max_cap"] == 3.5


def test_preregistered_and_production_reports_are_bit_pinned_and_test_isolated() -> None:
    root = Path(__file__).resolve().parents[1]
    screen_raw = (root / controller.CALIBRATION_SCREEN_PATH).read_bytes()
    production_raw = (root / controller.CALIBRATION_REPORT_PATH).read_bytes()

    assert hashlib.sha256(screen_raw).hexdigest() == (
        controller.EXPECTED_SCREEN_REPORT_SHA256
    )
    assert hashlib.sha256(production_raw).hexdigest() == (
        controller.EXPECTED_CALIBRATION_REPORT_SHA256
    )
    production = json.loads(production_raw)
    assert production["final_state_sha256"] == (
        controller.EXPECTED_CALIBRATION_FINAL_SHA256
    )
    assert production["training"]["optimizer_steps"] == 13
    assert production["scene_access_audit"]["forbidden_map_access_count"] == 0
    assert production["scene_access_audit"]["forbidden_oracle_access_count"] == 0
    assert production["scene_access_audit"]["qa_final_test_map_access_count"] == 0
    assert production["scene_access_audit"]["qa_final_test_oracle_access_count"] == 0


def test_preflight_and_exact_calibration_authorize_only_stage_one(tmp_path: Path) -> None:
    preflight_path = tmp_path / "preflight.json"
    decision_path = tmp_path / "decision.json"
    preflight = controller.run_preflight(
        output=preflight_path, source_provenance=_CLEAN_PROVENANCE
    )
    decision = controller.verify_calibration_report(
        config_path=controller.CONFIG_PATH,
        preflight_path=preflight_path,
        calibration_path=controller.CALIBRATION_REPORT_PATH,
        bridge_path=controller.CALIBRATION_BRIDGE_PATH,
        output=decision_path,
    )

    assert preflight["paired_qa_stage_authorized"] is False
    assert preflight["qa_split_isolation"]["final_test_in_forbidden_set"] is True
    assert decision["paired_qa_stage_authorized"] is True
    assert decision["final_qa_test_untouched"] is True
    assert decision["qa_run_count"] == 0
    assert decision["greedy_audit_authorized"] is False


def test_contract_and_calibration_drift_fail_closed(tmp_path: Path) -> None:
    config = load_config(controller.CONFIG_PATH)
    config["training"]["dense_alignment_warmup"]["learning_rate"] = 0.01
    with pytest.raises(controller.V26ControlViolation, match="config SHA-256"):
        controller._validate_contract(config)

    preflight_path = tmp_path / "preflight.json"
    controller.run_preflight(output=preflight_path, source_provenance=_CLEAN_PROVENANCE)
    drifted = json.loads(
        (Path(__file__).resolve().parents[1] / controller.CALIBRATION_REPORT_PATH).read_text(
            encoding="utf-8"
        )
    )
    drifted["scene_access_audit"]["qa_final_test_map_access_count"] = 1
    drifted_path = tmp_path / "drifted.json"
    drifted_path.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(controller.V26ControlViolation, match="report SHA-256"):
        controller.verify_calibration_report(
            config_path=controller.CONFIG_PATH,
            preflight_path=preflight_path,
            calibration_path=drifted_path,
            bridge_path=controller.CALIBRATION_BRIDGE_PATH,
        )


def test_update1_verifies_full_metadata_adapter_and_sanitized_runtime_sidecar(
    tmp_path: Path,
) -> None:
    preflight_path = tmp_path / "preflight.json"
    decision_path = tmp_path / "decision.json"
    checkpoint = tmp_path / "epoch_001"
    controller.run_preflight(output=preflight_path, source_provenance=_CLEAN_PROVENANCE)
    controller.verify_calibration_report(
        config_path=controller.CONFIG_PATH,
        preflight_path=preflight_path,
        calibration_path=controller.CALIBRATION_REPORT_PATH,
        bridge_path=controller.CALIBRATION_BRIDGE_PATH,
        output=decision_path,
    )
    _write_checkpoint(checkpoint)

    report = controller.verify_update1(
        config_path=controller.CONFIG_PATH,
        preflight_path=preflight_path,
        calibration_decision_path=decision_path,
        checkpoint=checkpoint,
    )

    assert report["stage_2_authorized"] is True
    assert set(report["artifact_hashes"]) == {
        "adapter_sha256",
        "metadata_sha256",
        "runtime_metadata_sha256",
        "optimizer_sha256",
    }
    assert report["runtime_sidecar"]["sanitized_projection_exact"] is True
    assert report["runtime_sidecar"]["training_only_field_count"] == 0
    assert report["runtime_sidecar"]["calibration_payload_present"] is False
    assert report["runtime_sidecar"]["question_ids_present"] is False


def test_update1_rejects_training_payload_in_runtime_sidecar(tmp_path: Path) -> None:
    checkpoint = tmp_path / "epoch_001"
    metadata = _write_checkpoint(checkpoint)
    runtime = runtime_checkpoint_metadata(metadata)
    runtime["history"] = []

    with pytest.raises(controller.V26ControlViolation, match="runtime sidecar"):
        controller._runtime_sidecar_audit(metadata, runtime)


def test_adapter_audit_rejects_dense_only_payload(tmp_path: Path) -> None:
    config = load_config(controller.CONFIG_PATH)
    module = construct_dense_alignment(config, semantic_dim=3072)
    assert module is not None
    with torch.no_grad():
        module.alignment_b[0, 0] = 0.125
    adapter = tmp_path / "adapter.safetensors"
    save_file(
        {
            f"dense_aligner.{name}": value.detach().cpu().contiguous()
            for name, value in module.state_dict().items()
        },
        adapter,
    )
    metadata = _metadata(dense_hash=module.state_sha256())

    with pytest.raises(controller.V26ControlViolation, match="complete adapter tensor inventory"):
        controller._adapter_audit(config, adapter, metadata)


def test_calibration_decision_is_bound_to_exact_preflight(tmp_path: Path) -> None:
    preflight_path = tmp_path / "preflight.json"
    decision_path = tmp_path / "decision.json"
    controller.run_preflight(output=preflight_path, source_provenance=_CLEAN_PROVENANCE)
    controller.verify_calibration_report(
        config_path=controller.CONFIG_PATH,
        preflight_path=preflight_path,
        calibration_path=controller.CALIBRATION_REPORT_PATH,
        bridge_path=controller.CALIBRATION_BRIDGE_PATH,
        output=decision_path,
    )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["preflight_sha256"] = "f" * 64
    decision_path.write_text(json.dumps(decision), encoding="utf-8")

    with pytest.raises(controller.V26ControlViolation, match="decision preflight"):
        controller._validate_calibration_decision(
            decision_path,
            preflight_path=preflight_path,
            config_path=controller.CONFIG_PATH,
        )


def test_selector_supports_only_the_primary_four_update_screen() -> None:
    config = load_config(controller.CONFIG_PATH)
    four = {
        epoch: _metadata(epoch, dense_hash=f"{epoch + 2:x}" * 64)
        for epoch in range(1, 5)
    }
    report = controller.select_epoch_metadata(four, config)

    assert report["evaluated_optimizer_updates"] == 4
    assert report["conditional_extension_authorized"] is False
    assert report["extension_controller_required"] is True
    assert report["greedy_audit_authorized"] is False
    assert report["final_qa_test_untouched"] is True

    with pytest.raises(controller.V26ControlViolation, match="exactly epochs 1--4"):
        controller.select_epoch_metadata(
            {
                epoch: _metadata(epoch, dense_hash=f"{epoch + 2:x}" * 64)
                for epoch in range(1, 9)
            },
            config,
        )


def test_minimal_training_metadata_fails_closed() -> None:
    config = load_config(controller.CONFIG_PATH)
    with pytest.raises(controller.V26ControlViolation, match="trainer metadata schema"):
        controller.verify_update1_metadata(
            {
                "schema_version": 1,
                "epoch": 1,
                "optimizer_step": 1,
                "dense_alignment_state_sha256": "3" * 64,
            },
            config,
        )


def test_selector_rejects_epoch1_artifact_not_bound_to_update1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(controller.CONFIG_PATH)
    epoch_metadata = {
        epoch: _metadata(epoch, dense_hash=f"{epoch + 2:x}" * 64)
        for epoch in range(1, 5)
    }
    update_hashes = {
        "adapter_sha256": "a" * 64,
        "metadata_sha256": "b" * 64,
        "runtime_metadata_sha256": "c" * 64,
        "optimizer_sha256": "d" * 64,
    }
    update = {
        "schema_version": 1,
        "audit_type": "v26_dense_alignment_update1_verification",
        "match": True,
        "stage_2_authorized": True,
        "greedy_audit_authorized": False,
        "report_only": True,
        "model_loaded": False,
        "oracle_loaded": False,
        "question_files_loaded": False,
        "question_dependent_scene_processing": False,
        "final_qa_test_untouched": True,
        "config_sha256": controller.EXPECTED_CONFIG_SHA256,
        "contract_sha256": controller.EXPECTED_CONTRACT_SHA256,
        "preflight_sha256": "e" * 64,
        "preflight_contract_sha256": controller.EXPECTED_CONTRACT_SHA256,
        "calibration_decision_sha256": "f" * 64,
        "calibration_chain": {
            "source_report_sha256": controller.EXPECTED_CALIBRATION_REPORT_SHA256,
            "final_state_sha256": controller.EXPECTED_CALIBRATION_FINAL_SHA256,
            "bridge_sha256": controller.EXPECTED_CALIBRATION_BRIDGE_SHA256,
            "preflight_sha256": "e" * 64,
            "contract_sha256": controller.EXPECTED_CONTRACT_SHA256,
        },
        "qa_data_artifacts": controller._qa_split_audit(config),
        "checkpoint": "epoch_001",
        "artifact_hashes": update_hashes,
        "dense_alignment": {
            "state_sha256": epoch_metadata[1]["dense_alignment_state_sha256"]
        },
    }
    update_path = tmp_path / "update1.json"
    update_path.write_text(json.dumps(update), encoding="utf-8")

    def fake_checkpoint(
        _config: dict[str, object], _path: Path, *, expected_epoch: int
    ) -> tuple[dict[str, object], dict[str, str], dict[str, object]]:
        hashes = dict(update_hashes)
        if expected_epoch == 1:
            hashes["optimizer_sha256"] = "0" * 64
        return epoch_metadata[expected_epoch], hashes, {"sanitized_projection_exact": True}

    monkeypatch.setattr(controller, "_checkpoint_epoch_artifacts", fake_checkpoint)
    with pytest.raises(controller.V26ControlViolation, match="artifact binding"):
        controller.select_epochs(
            config_path=controller.CONFIG_PATH,
            update1_report_path=update_path,
            epoch_bindings=[
                (epoch, Path(f"epoch_{epoch:03d}")) for epoch in range(1, 5)
            ],
        )
