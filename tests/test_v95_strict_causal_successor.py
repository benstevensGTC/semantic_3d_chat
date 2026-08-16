from __future__ import annotations

import builtins
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import torch

from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import resolve_v85
from semantic_3d_chat.evaluation.v95_strict_causal_successor_preflight import (
    DEFERRED_FINAL_SCENES,
    EXPECTED_INITIAL_STATE_SHA256,
    EXPECTED_TOTAL_ADAPTER_PARAMETER_COUNT,
    FRESH_PARAMETER_COUNT,
    PRIOR_EVALUATION_SCENES,
    TARGET_MODULES,
    TRAINING_SCENES,
    assert_deferred_final_absent_v95,
    build_preregistration_v95,
    causal_control_schedule_v95,
    cross_scene_schedule_v95,
    derive_contract_v95,
    derive_preregistration_v95,
    forbidden_training_roots_v95,
    load_config_v95,
    load_scene_memories_v95,
    load_training_rows_v95,
    lora_preflight_v95,
    payload_permutation_v95,
    permuted_payload_memory_v95,
    seal_parent_evidence_v95,
    training_schedule_v95,
    validate_parent_evidence_seal_v95,
    zero_payload_memory_v95,
)


def _identity(row: Any) -> str:
    return " ".join(row.question.casefold().split())


def test_v95_exact_training_only_inventory_and_main_schedule() -> None:
    config = load_config_v95()
    rows = load_training_rows_v95(config)
    schedule = training_schedule_v95(rows)

    assert len(rows) == 960
    assert tuple(sorted({row.scene_id for row in rows})) == TRAINING_SCENES
    assert not set(PRIOR_EVALUATION_SCENES).intersection(row.scene_id for row in rows)
    assert not set(DEFERRED_FINAL_SCENES).intersection(row.scene_id for row in rows)
    assert len({row.pair_id for row in rows}) == 20
    assert len(schedule) == 3840
    assert Counter(epoch for epoch, _row in schedule) == Counter({0: 960, 1: 960, 2: 960, 3: 960})


def test_v95_wrong_memory_schedule_covers_every_eligible_row_twice() -> None:
    rows = load_training_rows_v95(load_config_v95())
    schedule = cross_scene_schedule_v95(rows)
    exposures = Counter(row.key for _epoch, row, _wrong in schedule)

    assert len(schedule) == 996
    assert len(exposures) == 498
    assert set(exposures.values()) == {2}
    assert Counter(epoch for epoch, _row, _wrong in schedule) == Counter(
        {0: 249, 1: 249, 2: 249, 3: 249}
    )
    assert all(_identity(row) == _identity(wrong) for _epoch, row, wrong in schedule)
    assert all(row.scene_id != wrong.scene_id for _epoch, row, wrong in schedule)
    assert all(row.answer_class != wrong.answer_class for _epoch, row, wrong in schedule)


@pytest.mark.parametrize("arm", ["zero_payload", "full_interior_permutation"])
def test_v95_control_rotation_covers_all_498_with_only_two_repeats(arm: str) -> None:
    rows = load_training_rows_v95(load_config_v95())
    first = causal_control_schedule_v95(rows, arm=arm)
    second = causal_control_schedule_v95(rows, arm=arm)
    exposures = Counter(row.key for _epoch, row in first)

    assert [row.key for _epoch, row in first] == [row.key for _epoch, row in second]
    assert len(first) == 500
    assert len(exposures) == 498
    assert Counter(exposures.values()) == Counter({1: 496, 2: 2})
    assert Counter(epoch for epoch, _row in first) == Counter({0: 125, 1: 125, 2: 125, 3: 125})
    assert len({row.answer_type for _epoch, row in first}) == 7


def test_v95_fresh_bank_targets_shared_attention_ingress_and_is_exact_zero() -> None:
    config = load_config_v95()
    preflight = lora_preflight_v95(config)

    assert TARGET_MODULES == (
        "model.language_model.layers.9.self_attn.k_proj",
        "model.language_model.layers.9.self_attn.v_proj",
        "model.language_model.layers.34.mlp.up_proj",
    )
    assert preflight["target_modules"] == list(TARGET_MODULES)
    assert preflight["parameter_count"] == FRESH_PARAMETER_COUNT == 143_360
    assert preflight["initial_state_sha256"] == EXPECTED_INITIAL_STATE_SHA256
    assert preflight["exact_zero_output_at_initialization"] is True
    assert EXPECTED_TOTAL_ADAPTER_PARAMETER_COUNT == 819_200


def test_v95_controls_retain_native_boundaries_and_permute_all_payload_slots() -> None:
    config = load_config_v95()
    rows = load_training_rows_v95(config)
    memories, _hashes = load_scene_memories_v95(config, rows)
    memory = memories[min(memories)]
    zero = zero_payload_memory_v95(memory)
    permuted = permuted_payload_memory_v95(memory)
    permutation = payload_permutation_v95()

    assert memory.shape == (1, 738, 1536)
    assert memory.dtype == torch.bfloat16
    assert torch.equal(zero[:, :1], memory[:, :1])
    assert torch.equal(zero[:, -1:], memory[:, -1:])
    assert torch.count_nonzero(zero[:, 1:-1]).item() == 0
    assert torch.equal(permuted[:, :1], memory[:, :1])
    assert torch.equal(permuted[:, -1:], memory[:, -1:])
    assert not torch.equal(permuted[:, 1:-1], memory[:, 1:-1])
    assert sorted(permutation.tolist()) == list(range(736))


def test_v95_deferred_final_is_physically_absent_and_semantically_opaque() -> None:
    config = load_config_v95()
    result = assert_deferred_final_absent_v95(config)
    serialized = json.dumps(config["deferred_final_lock"], sort_keys=True)

    assert result["physical_path_count_checked"] == 24
    assert result["physical_artifacts_present"] == []
    assert set(result["empty_qa_placeholders"].values()) == {0}
    assert result["legacy_plan_file_count_opened"] == 0
    assert all(
        token not in serialized
        for token in (
            "color_variant",
            "chair_orientation",
            "bowl_placement",
            "picture_placement",
            "change_type",
            "seed_offset",
        )
    )


def test_v95_parent_seal_contains_hashes_and_aggregates_not_qa_rows() -> None:
    config = load_config_v95()
    evidence = json.loads(
        resolve_v85(config["sources"]["v94_hardened_scored_evidence"]).read_text(encoding="utf-8")
    )

    validate_parent_evidence_seal_v95(evidence)
    assert evidence["behavior_gate_passed"] is False
    assert evidence["score"]["status"] == "measured_gate_not_passed"
    assert evidence["prediction_row_count"] == 216
    assert "rows" not in evidence
    assert "predictions" not in evidence

    leaked = copy.deepcopy(evidence)
    leaked["question"] = "prohibited serialized question"
    with pytest.raises(ValueError, match="serializes questions or answers"):
        validate_parent_evidence_seal_v95(leaked)


def test_v95_parent_evidence_and_preregistration_are_create_once() -> None:
    with pytest.raises(FileExistsError, match="already exists"):
        seal_parent_evidence_v95()
    with pytest.raises(FileExistsError, match="create-once output exists"):
        build_preregistration_v95()


def test_v95_normal_derivation_blocks_direct_v94_and_all_held_out_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config_v95()
    forbidden = set(forbidden_training_roots_v95(config))
    aggregate = resolve_v85(config["sources"]["v94_hardened_scored_evidence"])
    assert aggregate.resolve() not in forbidden
    original_open = builtins.open
    original_path_open = Path.open

    def rejects(path: object) -> bool:
        candidate = Path(path).resolve()  # type: ignore[arg-type]
        return any(candidate == root or root in candidate.parents for root in forbidden)

    def guarded_open(file: object, *args: object, **kwargs: object) -> object:
        if rejects(file):
            raise AssertionError(f"V95 opened protected source: {file}")
        return original_open(file, *args, **kwargs)  # type: ignore[arg-type]

    def guarded_path_open(path: Path, *args: object, **kwargs: object) -> object:
        if rejects(path):
            raise AssertionError(f"V95 opened protected source: {path}")
        return original_path_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    result = derive_contract_v95()

    assert result["file_audit_forbidden_reads"] == []
    assert result["prior_evaluation_labels_opened"] is False
    assert result["deferred_final_semantic_plans_opened"] is False
    assert result["deferred_final_artifacts_generated"] is False
    assert result["frozen_parent"]["status"] == "authenticated"
    assert result["frozen_parent"]["v94_behavior_gate_passed"] is False
    assert result["total_nll_forward_evaluations"] == 5836
    assert result["full_gemma_model_loaded"] is False
    assert result["optimizer_updates_performed"] == 0


def test_v95_preregistration_derivation_is_nonmutating_and_has_both_gates() -> None:
    config = load_config_v95()
    result = derive_preregistration_v95()

    assert result["parent_authenticated"] is True
    assert result["training_authorized"] is False
    assert result["known_development_protocol"] == config["known_development_gate"]
    assert result["deferred_evaluation_protocol"] == config["deferred_evaluation"]
    assert result["prior_evaluation_labels_opened"] is False
    assert result["deferred_final_labels_opened"] is False
    assert result["deferred_final_artifacts_generated"] is False
    assert result["full_gemma_model_loaded"] is False
    assert result["optimizer_constructed"] is False


def test_v95_known_development_is_fixed_final_gate_not_checkpoint_selection() -> None:
    config = load_config_v95()
    gate = config["known_development_gate"]
    training = config["training"]

    assert gate["role"] == "post_fixed_final_go_no_go_not_checkpoint_selection"
    assert gate["scene_ids"] == list(PRIOR_EVALUATION_SCENES)
    assert gate["v94_reference_correct"] == 143
    assert gate["v95_correct_minimum"] == 150
    assert gate["fixed_final_checkpoint_may_not_change_after_gate"] is True
    assert gate["pass_required_before_deferred_final_unlock"] is True
    assert training["checkpoint_selection"].startswith("fixed_final_update_480_before")
    assert training["intermediate_behavior_selection"] is False
