from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
import transformers
from safetensors.torch import save_file

from semantic_3d_chat.config import config_hash, load_config
from semantic_3d_chat.evaluation import v24_shared_query_controller as controller


def _clean_provenance() -> dict[str, object]:
    return {
        "schema_version": 1,
        "scope": "repository_excluding_generated_artifacts_v1",
        "available": True,
        "is_clean": True,
        "head_commit": "a" * 40,
        "head_tree": "b" * 40,
        "tracked_diff_sha256": hashlib.sha256(b"").hexdigest(),
    }


def _pair_metrics(
    sides: int = 12,
    units: int = 6,
    mean: float = 1.0,
    minimum: float = 0.25,
) -> dict[str, float | int]:
    return {
        "full_vocab_sides": sides,
        "full_vocab_units": units,
        "mean_candidate_margin": mean,
        "minimum_candidate_margin": minimum,
        "mean_full_vocab_margin": mean,
        "minimum_full_vocab_margin": minimum,
    }


def _epoch(epoch: int, mirror: dict[str, float | int]) -> dict[str, Any]:
    bank_hash = hashlib.sha256(f"v24-{epoch}".encode()).hexdigest()
    detail = {
        "schema_version": 1,
        "source": "checkpoint_metadata_only_no_model_inference",
        "contains_environment_text": False,
        "by_pair": {},
        "source_detail_sha256": "d" * 64,
    }
    return {
        "epoch": epoch,
        "optimizer_step": epoch,
        "cumulative_microsteps": epoch * 12,
        "metadata_path": (
            f"data_gemma4/checkpoints/{controller.PRIMARY_NAMESPACE}/"
            f"epoch_{epoch:03d}/metadata.json"
        ),
        "metadata_sha256": f"metadata-{epoch}",
        "adapter_sha256": f"adapter-{epoch}",
        "optimizer_sha256": f"optimizer-{epoch}",
        "new_bank_state_sha256": bank_hash,
        "recomputed_payload_hashes": {"tensor_count": 1},
        "optimizer_manifest": {"optimizer": "AdamW", "expected_step": epoch},
        "color": _pair_metrics(),
        "mirror": mirror,
        "opaque_unit_margin_detail": detail,
    }


def _update1(path: Path, epoch1: dict[str, Any]) -> Path:
    value = {
        "schema_version": 1,
        "audit_type": "v24_shared_query_update1_verifier",
        "match": True,
        "stage_2_authorized": True,
        "report_only": True,
        "model_loaded": False,
        "oracle_loaded": False,
        "preflight_sha256": "1" * 64,
        "config_sha256": controller.EXPECTED_CONFIG_SHA256,
        "contract_sha256": controller.EXPECTED_CONTRACT_SHA256,
        "checkpoint": f"data_gemma4/checkpoints/{controller.PRIMARY_NAMESPACE}/epoch_001",
        "checkpoint_artifact_hashes": {
            "adapter_sha256": epoch1["adapter_sha256"],
            "metadata_sha256": epoch1["metadata_sha256"],
            "optimizer_sha256": epoch1["optimizer_sha256"],
        },
        "new_bank_state_sha256": epoch1["new_bank_state_sha256"],
        "ordered_parameter_shapes": [list(shape) for shape in controller.EXPECTED_PARAMETER_SHAPES],
        "a_tensors_unchanged": True,
        "b_tensors_all_changed": True,
        "all_prior_tensors_frozen": True,
        "optimizer_manifest": epoch1["optimizer_manifest"],
        "recomputed_payload_hashes": epoch1["recomputed_payload_hashes"],
        "color": epoch1["color"],
        "mirror": epoch1["mirror"],
        "opaque_unit_margin_detail": epoch1["opaque_unit_margin_detail"],
        "source_provenance": _clean_provenance(),
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _optimizer_state() -> dict[str, Any]:
    state: dict[int, dict[str, torch.Tensor]] = {}
    for index, shape in enumerate(controller.EXPECTED_PARAMETER_SHAPES):
        is_a = index % 2 == 0
        state[index] = {
            "step": torch.tensor(1.0),
            "exp_avg": torch.zeros(shape) if is_a else torch.ones(shape),
            "exp_avg_sq": torch.zeros(shape) if is_a else torch.full(shape, 0.5),
        }
    return {
        "state": state,
        "param_groups": [
            {
                "name": "language_lora",
                "lr": 0.0003,
                "weight_decay": 0.0,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "amsgrad": False,
                "maximize": False,
                "foreach": False,
                "capturable": False,
                "differentiable": False,
                "fused": False,
                "decoupled_weight_decay": True,
                "params": list(range(4)),
            }
        ],
    }


def _gate_with_detail() -> dict[str, Any]:
    pair_margins = {
        "pair_000001": ([1.0] * 12, [2.0] * 12),
        "pair_000003": ([0.5] * 12, [0.75] * 12),
    }
    units: list[dict[str, Any]] = []
    by_pair: dict[str, Any] = {}
    global_index = 0
    for pair_index, (pair_id, (candidate, full)) in enumerate(pair_margins.items()):
        by_pair[pair_id] = {
            "first_answer_token_top1_accuracy": 1.0,
            "first_answer_token_top1_unit_accuracy": 1.0,
            "mean_own_vs_alternate_candidate_logit_margin": sum(candidate) / 12,
            "minimum_own_vs_alternate_candidate_logit_margin": min(candidate),
            "mean_first_answer_token_target_vs_best_other_logit_margin": sum(full) / 12,
            "minimum_first_answer_token_target_vs_best_other_logit_margin": min(full),
        }
        for local in range(6):
            scene_ids = [f"scene_{pair_index * 2 + side + 1:06x}" for side in range(2)]
            question_ids = [f"q_{global_index * 2 + side + 1:06x}" for side in range(2)]
            sides = []
            for side in range(2):
                offset = local * 2 + side
                sides.append(
                    {
                        "side_index": side,
                        "scene_id": scene_ids[side],
                        "question_id": question_ids[side],
                        "own_vs_alternate_candidate_logit_margin": candidate[offset],
                        "first_token_target_vs_best_other_logit_margin": full[offset],
                        "own_preference_passed": True,
                        "full_vocab_top1_passed": True,
                        "own_candidate_token_id": 10 + side,
                        "alternate_candidate_token_id": 11 - side,
                    }
                )
            units.append(
                {
                    "unit_index": global_index,
                    "scene_ids": scene_ids,
                    "question_ids": question_ids,
                    "sides": sides,
                }
            )
            global_index += 1
    return {
        "by_pair": by_pair,
        "detail": {
            "schema_version": 1,
            "artifact": "training_candidate_gate_detail",
            "training_only": True,
            "free_generation_evaluated": False,
            "candidate_representation": "candidate_token_ids",
            "contains_question_text": False,
            "contains_oracle_geometry": False,
            "contains_canonical_training_targets": False,
            "full_vocab_first_token_evaluated": True,
            "unit_count": 12,
            "side_count": 24,
            "units": units,
        },
    }


def test_v24_config_contract_and_real_query_shapes_are_exact() -> None:
    config = load_config(controller.CONFIG_PATH)
    assert config_hash(config, length=64) == controller.EXPECTED_CONFIG_SHA256
    assert controller._canonical_sha256(controller.v24_contract(config)) == (
        controller.EXPECTED_CONTRACT_SHA256
    )
    assert config["training"]["initialize_named_lora_freeze_and_extend_transition"] is True
    collection = controller._install_shape_only(config)
    bank = collection.bank(controller.NEW_BANK)
    assert bank.installation.parameter_count == 36_864
    assert [tuple(parameter.shape) for parameter in bank.installation.parameters()] == list(
        controller.EXPECTED_PARAMETER_SHAPES
    )
    assert bank.installation.state_sha256() == controller.EXPECTED_NEW_BANK_INITIAL_SHA256
    assert not any(torch.count_nonzero(adapter.lora_b) for adapter in bank.installation.adapters)


def test_v24_composed_transition_accepts_only_exact_sealed_v23_banks() -> None:
    config = load_config(controller.CONFIG_PATH)
    collection = controller._install_shape_only(config)
    metadata = controller._load_json(controller.SOURCE_CHECKPOINT / "metadata.json", "source")
    assert controller._source_to_v24_transition_mismatch(metadata, collection) is None
    tampered = deepcopy(metadata)
    tampered["lora_bank_state_sha256"]["extension_v23_shared_kv"] = "0" * 64
    mismatch = controller._source_to_v24_transition_mismatch(tampered, collection)
    assert mismatch is not None
    assert "extension_v23_shared_kv.state" in mismatch


def test_v24_preflight_is_no_step_and_binds_sequence_audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provenance = _clean_provenance()
    sequence = {"audit_type": "test_recomputed_sequence", "model_loaded": False}
    monkeypatch.setattr(controller, "capture_git_source_provenance", lambda _root: provenance)
    monkeypatch.setattr(controller, "_sequence_length_audit", lambda _config: sequence)
    report = controller.run_preflight(controller.CONFIG_PATH, tmp_path / "preflight.json")
    assert report["authorized"] is True
    assert report["model_loaded"] is False
    assert report["optimizer_constructed"] is False
    assert report["optimizer_step_executed"] is False
    assert report["sequence_length_audit"] == sequence
    assert report["new_bank"]["parameter_count"] == 36_864


@pytest.mark.skipif(
    int(transformers.__version__.split(".")[0]) < 5,
    reason="exact pinned Gemma 4 tokenizer audit requires the .venv-gemma4 runtime",
)
def test_v24_sequence_audit_recomputes_all_pinned_lengths_and_weight_headers() -> None:
    report = controller._sequence_length_audit(load_config(controller.CONFIG_PATH))
    assert report["record_count"] == 24
    assert report["pair_unit_count"] == 12
    assert report["prompt_token_range_inclusive"] == [57, 62]
    assert report["answer_plus_eos_tokens"] == 2
    assert report["total_sequence_token_range_inclusive"] == [317, 322]
    assert report["prefix_plus_prompt_token_range_inclusive"] == [315, 320]
    assert report["maximum_final_query_to_boi_distance"] == 318
    assert report["maximum_final_query_to_first_scene_latent_distance"] == 317
    assert report["maximum_final_query_to_boi_inclusive_span"] == 319
    assert report["maximum_final_query_to_first_scene_latent_inclusive_span"] == 318
    assert report["all_final_prompt_queries_reach_entire_scene_prefix"] is True
    assert report["target_q_proj_weights"]["28"]["shape"] == [2048, 1536]
    assert report["target_q_proj_weights"]["29"]["shape"] == [4096, 1536]


def test_v24_optimizer_manifest_has_exact_four_parameter_surface(tmp_path: Path) -> None:
    path = tmp_path / "optimizer.pt"
    torch.save(_optimizer_state(), path)
    manifest = controller._optimizer_manifest(path)
    assert manifest["group"]["params"] == [0, 1, 2, 3]
    assert [row["shape"] for row in manifest["parameter_states"]] == [
        list(shape) for shape in controller.EXPECTED_PARAMETER_SHAPES
    ]
    assert [row["role"] for row in manifest["parameter_states"]] == ["A", "B", "A", "B"]


def test_v24_optimizer_manifest_rejects_nonzero_first_step_a_moment(tmp_path: Path) -> None:
    state = _optimizer_state()
    state["state"][0]["exp_avg"].reshape(-1)[0] = 1.0
    path = tmp_path / "optimizer.pt"
    torch.save(state, path)
    with pytest.raises(controller.V24ControlViolation, match="LoRA-A optimizer moments"):
        controller._optimizer_manifest(path)


def test_v24_adapter_payload_requires_all_prior_banks_and_two_new_targets(tmp_path: Path) -> None:
    tensors = {
        "scene_model.weight": torch.ones(1),
        "global_scene_residual.weight": torch.ones(1),
        "signed_x_scene_residual.weight": torch.ones(1),
        **{
            f"lora_banks.{name}.adapters.0.lora_a": torch.ones(1)
            for name in controller.FROZEN_BANKS
        },
    }
    for index, (a_shape, b_shape) in enumerate(
        zip(
            controller.EXPECTED_PARAMETER_SHAPES[::2],
            controller.EXPECTED_PARAMETER_SHAPES[1::2],
            strict=True,
        )
    ):
        tensors[f"lora_banks.{controller.NEW_BANK}.adapters.{index}.lora_a"] = torch.zeros(a_shape)
        tensors[f"lora_banks.{controller.NEW_BANK}.adapters.{index}.lora_b"] = torch.zeros(b_shape)
    path = tmp_path / "adapter.safetensors"
    save_file(tensors, path)
    payload = controller._adapter_payload(path)
    assert set(payload["lora_bank_state_sha256"]) == {
        *controller.FROZEN_BANKS,
        controller.NEW_BANK,
    }
    assert set(payload["new_bank_state"]) == {
        f"adapters.{index}.{suffix}" for index in range(2) for suffix in ("lora_a", "lora_b")
    }


def test_v24_opaque_unit_detail_recomputes_aggregates_and_rejects_text_leak() -> None:
    metadata = {"pair_candidate_gate": _gate_with_detail()}
    detail = controller._opaque_unit_margin_detail(metadata)
    assert detail["contains_environment_text"] is False
    assert len(detail["by_pair"]["pair_000001"]["units"]) == 6
    assert len(detail["by_pair"]["pair_000003"]["units"]) == 6
    leaked = deepcopy(metadata)
    leaked["pair_candidate_gate"]["detail"]["units"][0]["sides"][0]["own_canonical_target"] = "left"
    with pytest.raises(controller.V24ControlViolation, match="prohibited field"):
        controller._opaque_unit_margin_detail(leaked)


def test_v24_selector_ranks_units_then_sides_without_loading_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    records = {
        1: _epoch(1, _pair_metrics(sides=10, units=2, mean=2.0, minimum=-0.5)),
        2: _epoch(2, _pair_metrics(sides=8, units=3, mean=0.2, minimum=-1.0)),
        3: _epoch(3, _pair_metrics(sides=9, units=3, mean=0.5, minimum=-0.8)),
        4: _epoch(4, _pair_metrics(sides=9, units=3, mean=0.5, minimum=-0.8)),
    }
    monkeypatch.setattr(
        controller,
        "_epoch_record",
        lambda _config, epoch, _path, _source: deepcopy(records[epoch]),
    )
    monkeypatch.setattr(
        controller, "capture_git_source_provenance", lambda _root: _clean_provenance()
    )
    report = controller.select_epochs(
        controller.CONFIG_PATH,
        _update1(tmp_path / "update1.json", records[1]),
        {epoch: Path(f"unused-{epoch}") for epoch in range(1, 5)},
        tmp_path / "selection.json",
    )
    assert report["selected_epoch"] == 3
    assert report["continuation_authorized"] is True
    assert report["greedy_audit_authorized"] is False
    assert report["model_loaded"] is False
    assert [row["epoch"] for row in report["ranking"]] == [3, 4, 2, 1]


def test_v24_selector_authorizes_greedy_only_after_complete_teacher_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    records = {
        epoch: _epoch(
            epoch,
            _pair_metrics(
                sides=12 if epoch == 2 else 8,
                units=6 if epoch == 2 else 2,
                mean=1.0 if epoch == 2 else 0.1,
                minimum=0.25 if epoch == 2 else -0.5,
            ),
        )
        for epoch in range(1, 5)
    }
    monkeypatch.setattr(
        controller,
        "_epoch_record",
        lambda _config, epoch, _path, _source: deepcopy(records[epoch]),
    )
    monkeypatch.setattr(
        controller, "capture_git_source_provenance", lambda _root: _clean_provenance()
    )
    report = controller.select_epochs(
        controller.CONFIG_PATH,
        _update1(tmp_path / "update1.json", records[1]),
        {epoch: Path(f"unused-{epoch}") for epoch in range(1, 5)},
        tmp_path / "selection.json",
    )
    assert report["selected_epoch"] == 2
    assert report["full_teacher_gate_passed"] is True
    assert report["greedy_audit_authorized"] is True
    assert report["static_chat_authorized"] is False


def test_v24_selector_rejects_forged_minimal_update1(tmp_path: Path) -> None:
    path = tmp_path / "forged.json"
    path.write_text(json.dumps({"match": True, "stage_2_authorized": True}), encoding="utf-8")
    with pytest.raises(controller.V24ControlViolation, match="update-1 report root keys"):
        controller.select_epochs(
            controller.CONFIG_PATH,
            path,
            {epoch: Path(f"unused-{epoch}") for epoch in range(1, 5)},
            tmp_path / "selection.json",
        )
