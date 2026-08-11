from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from semantic_3d_chat.config import config_hash, load_config
from semantic_3d_chat.evaluation import v25_dense_alignment_controller as controller

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
    config = load_config(controller.CONFIG_PATH)
    supervision = controller.dense_alignment_supervision_settings(config)
    return {
        "schema_version": 1,
        "config_sha256": controller.EXPECTED_CONFIG_SHA256,
        "calibration_split_sha256": controller._canonical_sha256(
            list(supervision.calibration_scene_ids)
        ),
        "held_out_split_sha256": controller._canonical_sha256(
            list(supervision.held_out_scene_ids)
        ),
        "category_vocabulary_sha256": "4" * 64,
        "category_embedding_sha256": "5" * 64,
        "calibration_summary_sha256": "6" * 64,
        "initial_state_sha256": controller.EXPECTED_DENSE_INITIAL_SHA256,
        "final_state_sha256": "3" * 64,
        "calibration_scene_count": 8,
        "held_out_scene_count": 2,
        "category_count": 9,
        "region_count": 72,
        "skipped_underfilled_region_count": 3,
        "summarized_region_voxel_count": 12_000,
        "selective_token_row_count": 9,
        "loaded_parameter_count": 1,
        "cpu_only": True,
        "local_files_only": True,
        "raw_map_write_count": 0,
        "raw_maps_preserved": True,
        "question_dependent_selection": False,
        "training": {
            "learning_rate": 0.01,
            "weight_decay": 0.0001,
            "delta_mse_regularization_weight": 0.01,
            "maximum_optimizer_steps": 20,
            "optimizer_steps": 1,
            "stopped_at_first_pass": True,
            "calibration_passed": True,
            "history": [
                {
                    "optimizer_step": 1,
                    "region_count": 72,
                    "category_count": 9,
                    "contrastive_loss": 0.025,
                    "total_loss": 0.030,
                    "top1_accuracy": 1.0,
                    "minimum_cosine_margin": 0.118,
                    "mean_cosine_margin": 0.25,
                    "delta_mean_squared": 0.508,
                    "delta_rms": 0.713,
                    "delta_abs_max": 2.82,
                    "passed": True,
                }
            ],
            "final_state_sha256": "3" * 64,
        },
        "held_out_localization": {
            "target_region_count": 4,
            "top_k": 100,
            "minimum_precision_required": 0.10,
            "maximum_mirror_centroid_error_required_m": 0.15,
            "all_target_hit_at_k": True,
            "minimum_precision_at_k": 0.95,
            "minimum_region_margin": 0.11,
            "minimum_correct_vs_distractor_margin": 0.12,
            "maximum_mirror_centroid_error_m": 0.05,
            "mirror_centroid_errors_m": [0.04, 0.05],
            "targets": [],
            "passed": True,
        },
        "qa_update_authorized": True,
        "bridge_written": False,
        "bridge_sha256": None,
        "pair_optimizer_state_empty_before_warmup": True,
        "pair_optimizer_rebuilt_after_warmup": True,
        "pair_optimizer_state_empty_after_warmup": True,
        "pair_optimizer_steps_before_qa": 0,
        "held_out_scene_gradient_access": False,
        "category_text_prototypes_serialized": False,
        "oracle_payload_retained": False,
    }


def _pair_gate(*, full_teacher: bool = False) -> dict[str, object]:
    source = controller._load_json(
        controller.SOURCE_CHECKPOINT / "metadata.json", "test V24 source metadata"
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
    mirror = gate["by_pair"][controller.MIRROR_PAIR_ID]
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


def _update1_metadata() -> dict[str, object]:
    return {
        "epoch": 1,
        "optimizer_step": 1,
        "output_namespace": controller.PRIMARY_NAMESPACE,
        "train_dense_alignment_only": True,
        "freeze_scene_adapter": True,
        "dense_alignment_parameter_count": controller.EXPECTED_DENSE_PARAMETER_COUNT,
        "dense_alignment_calibration": _calibration_audit(),
        "pair_candidate_gate": _pair_gate(),
        "frozen_scene_state_sha256": controller.EXPECTED_FROZEN_HASHES["scene"],
        "global_scene_residual_state_sha256": controller.EXPECTED_FROZEN_HASHES["global"],
        "signed_x_scene_residual_state_sha256": controller.EXPECTED_FROZEN_HASHES[
            "signed_x"
        ],
        "lora_bank_state_sha256": {
            name: controller.EXPECTED_FROZEN_HASHES[name]
            for name in controller.FROZEN_BANKS
        },
    }


def _epoch_metadata(epoch: int, *, full_teacher: bool = False) -> dict[str, object]:
    metadata = _update1_metadata()
    metadata["epoch"] = epoch
    metadata["optimizer_step"] = epoch
    metadata["dense_alignment_state_sha256"] = f"{epoch + 6:x}" * 64
    metadata["pair_candidate_gate"] = _pair_gate(full_teacher=full_teacher)
    return metadata


def test_v25_config_and_normalized_contract_are_hash_pinned() -> None:
    config = load_config(controller.CONFIG_PATH)

    assert config_hash(config, length=64) == controller.EXPECTED_CONFIG_SHA256
    assert (
        controller._canonical_sha256(controller.v25_contract(config))
        == controller.EXPECTED_CONTRACT_SHA256
    )
    assert controller.v25_contract(config)["paired_qa"]["pair_steps_per_epoch"] == 12
    assert controller.v25_contract(config)["warmup"]["max_optimizer_steps"] == 20


def test_preflight_binds_source_zero_output_isolation_and_stage_boundary(
    tmp_path: Path,
) -> None:
    output = tmp_path / "v25_preflight.json"
    report = controller.run_preflight(
        output=output,
        source_provenance=_CLEAN_PROVENANCE,
    )

    assert report["authorized"] is True
    assert report["calibration_stage_authorized"] is True
    assert report["paired_qa_stage_authorized"] is False
    assert report["model_loaded"] is False
    assert report["oracle_loaded"] is False
    assert report["optimizer_steps"] == 0
    assert report["dense_alignment"]["exact_zero_forward_bitwise_identity"] is True
    assert report["dense_alignment"]["voxel_locality_probe_passed"] is True
    assert report["runtime_isolation"]["all_voxels_transformed"] is True
    assert report["runtime_isolation"]["runtime_oracle_access"] is False
    assert report["source"]["artifact_hashes"] == controller.EXPECTED_SOURCE_ARTIFACTS
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert "output" not in persisted
    validated = controller.validate_preflight(output)
    assert validated["contract_sha256"] == controller.EXPECTED_CONTRACT_SHA256


def test_preflight_rejects_dirty_source_and_runtime_contract_drift() -> None:
    dirty = {**_CLEAN_PROVENANCE, "is_clean": False}
    with pytest.raises(controller.V25ControlViolation, match="clean committed"):
        controller.run_preflight(source_provenance=dirty)

    config = load_config(controller.CONFIG_PATH)
    config["experiment"]["runtime_oracle_access"] = True
    with pytest.raises(controller.V25ControlViolation, match="config SHA-256"):
        controller._validate_contract(config)


def test_deterministic_calibration_failure_is_verified_as_terminal_stop(
    tmp_path: Path,
) -> None:
    output = tmp_path / "calibration_decision.json"
    decision = controller.verify_calibration_report(
        config_path=controller.CONFIG_PATH,
        calibration_path=Path(
            "reports/gemma4/metrics/v25_dense_alignment_calibration.json"
        ),
        output=output,
    )

    assert decision["decision"] == "bounded_calibration_failed_stop_before_paired_qa"
    assert decision["calibration_authorized"] is False
    assert decision["paired_qa_stage_authorized"] is False
    assert decision["terminal_stop"] is True
    assert decision["thresholds_preserved"] is True
    assert decision["threshold_relaxation_permitted"] is False
    assert decision["optimizer_steps"] == 20
    assert decision["held_out_localization_passed"] is True
    assert decision["qa_update_authorized"] is False
    assert decision["greedy_audit_authorized"] is False
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert "output" not in persisted


def test_update1_metadata_requires_successful_warmup_before_pair_authorization() -> None:
    config = load_config(controller.CONFIG_PATH)
    verified = controller.verify_update1_metadata(_update1_metadata(), config)

    assert verified["match"] is True
    assert verified["calibration_gate"]["passed"] is True
    assert verified["teacher_forced_gate"]["stage_2_passed"] is True
    assert (
        verified["teacher_forced_gate"][
            "both_source_negative_margins_strictly_improved"
        ]
        is True
    )
    assert verified["paired_qa_optimizer_step"] == 1
    assert verified["stage_2_authorized"] is True
    assert verified["greedy_audit_authorized"] is False


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("training", "history", 0, "top1_accuracy"), 0.99, "top1_accuracy"),
        (("training", "history", 0, "minimum_cosine_margin"), 0.099, "minimum_margin"),
        (("training", "history", 0, "delta_rms"), 1.01, "delta_rms"),
        (("training", "history", 0, "delta_abs_max"), 3.51, "delta_abs_max"),
        (("held_out_localization", "passed"), False, "held_out_localization_passed"),
        (("question_dependent_selection",), True, "question_dependent_selection"),
        (("pair_optimizer_steps_before_qa",), 1, "pair_optimizer_steps_before_qa"),
        (("oracle_payload_retained",), True, "oracle_payload_retained"),
    ],
)
def test_update1_metadata_rejects_failed_or_leaky_warmup(
    path: tuple[str | int, ...], value: object, message: str
) -> None:
    config = load_config(controller.CONFIG_PATH)
    metadata = _update1_metadata()
    audit = deepcopy(metadata["dense_alignment_calibration"])
    cursor = audit
    for component in path[:-1]:
        cursor = cursor[component]
    cursor[path[-1]] = value
    metadata["dense_alignment_calibration"] = audit

    with pytest.raises(controller.V25ControlViolation, match=message):
        controller.verify_update1_metadata(metadata, config)


def test_update1_metadata_rejects_any_frozen_surface_drift() -> None:
    config = load_config(controller.CONFIG_PATH)
    metadata = _update1_metadata()
    metadata["lora_bank_state_sha256"] = {
        **metadata["lora_bank_state_sha256"],
        "extension_v24_shared_query": "f" * 64,
    }

    with pytest.raises(controller.V25ControlViolation, match="frozen LoRA"):
        controller.verify_update1_metadata(metadata, config)


def test_four_update_selector_authorizes_only_conditional_extension_without_full_gate() -> None:
    config = load_config(controller.CONFIG_PATH)
    report = controller.select_epoch_metadata(
        {epoch: _epoch_metadata(epoch) for epoch in range(1, 5)},
        config,
    )

    assert report["evaluated_optimizer_updates"] == 4
    assert report["decision"] == "conditional_extension_to_update_8_authorized"
    assert report["conditional_extension_authorized"] is True
    assert report["greedy_audit_authorized"] is False
    assert report["static_chat_authorized"] is False
    assert report["selected_epoch"] == 1


def test_selector_authorizes_greedy_audit_only_after_complete_teacher_gate() -> None:
    config = load_config(controller.CONFIG_PATH)
    epochs = {epoch: _epoch_metadata(epoch) for epoch in range(1, 5)}
    epochs[4] = _epoch_metadata(4, full_teacher=True)
    report = controller.select_epoch_metadata(epochs, config)

    assert report["decision"] == "full_teacher_gate_passed_greedy_audit_authorized"
    assert report["full_teacher_first_pass_epoch"] == 4
    assert report["selected_epoch"] == 4
    assert report["greedy_audit_authorized"] is True
    assert report["static_chat_authorized"] is False


def test_eight_update_selector_stops_at_hard_limit_without_full_gate() -> None:
    config = load_config(controller.CONFIG_PATH)
    report = controller.select_epoch_metadata(
        {epoch: _epoch_metadata(epoch) for epoch in range(1, 9)},
        config,
    )

    assert report["evaluated_optimizer_updates"] == 8
    assert report["hard_optimizer_update_limit"] == 8
    assert report["conditional_limit_reached"] is True
    assert report["conditional_extension_authorized"] is False
    assert report["greedy_audit_authorized"] is False


def test_selector_rejects_unregistered_epoch_counts() -> None:
    config = load_config(controller.CONFIG_PATH)
    with pytest.raises(controller.V25ControlViolation, match="exactly epochs"):
        controller.select_epoch_metadata(
            {epoch: _epoch_metadata(epoch) for epoch in range(1, 6)},
            config,
        )
