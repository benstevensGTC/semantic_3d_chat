from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from semantic_3d_chat.evaluation.fixed_prefix_ple_v54_v4_preregistration import (
    V3_ABORT_SHA256,
    V3_PREREGISTRATION_SHA256,
    V3_SMOKE_SHA256,
    authenticate_v3_abort,
    build_preregistration,
    v4_implementation_hashes,
    write_preregistration,
)
from semantic_3d_chat.training.pair_curriculum import token_normalized_nll
from semantic_3d_chat.training.train_fixed_prefix_ple_v54_v4 import (
    _synthetic_equivalence,
    selected_answer_nll_from_full_logits,
)


def test_v4_authenticates_exact_zero_update_v3_oom() -> None:
    abort = authenticate_v3_abort()
    contract = build_preregistration()

    assert abort["failure_scope"]["adapter_update_count"] == 0
    assert abort["failure_scope"]["optimizer_constructed"] is False
    assert abort["checkpoint_absent"] is True
    assert abort["error"]["memory"] == {
        "attempted_allocation_mib": 642.0,
        "mps_allocated_gib": 20.32,
        "mps_max_allowed_gib": 30.19,
        "other_allocations_gib": 9.51,
    }
    assert contract["v3_abort"]["preregistration_sha256"] == V3_PREREGISTRATION_SHA256
    assert contract["v3_abort"]["smoke_sha256"] == V3_SMOKE_SHA256
    assert contract["v3_abort"]["abort_sha256"] == V3_ABORT_SHA256


def test_v4_resource_amendment_preserves_numeric_objective() -> None:
    resource = build_preregistration()["resource_only_changes"]

    assert resource["teacher_forcing_microbatch_size"] == {"v3": 2, "v4": 1}
    assert resource["retention_next_token_logits_to_keep"] == 1
    assert resource["examples_streamed_without_batch_padding"] is True
    assert resource["each_example_token_normalized_before_any_mean"] is True
    assert resource["same_answer_labels"] is True
    assert resource["same_fp32_cross_entropy"] is True
    assert resource["same_per_example_answer_token_normalization"] is True
    assert resource["same_correct_wrong_prefix_objective"] is True
    assert resource["same_gradient_accumulation_divisor"] is True


def test_v4_selected_answer_nll_exactly_matches_full_reference() -> None:
    generator = torch.Generator().manual_seed(54)
    logits = torch.randn(4, 9, 23, generator=generator)
    labels = torch.full((4, 9), -100, dtype=torch.long)
    labels[0, -1:] = torch.tensor([2])
    labels[1, -2:] = torch.tensor([4, 8])
    labels[2, -3:] = torch.tensor([1, 7, 9])
    labels[3, -4:] = torch.tensor([3, 5, 11, 6])

    full = token_normalized_nll(logits, labels)
    selected = selected_answer_nll_from_full_logits(logits, labels)

    assert torch.equal(full, selected)
    assert _synthetic_equivalence()["exact"] is True


def test_v4_selected_answer_nll_rejects_non_suffix_labels() -> None:
    logits = torch.randn(1, 6, 9)
    labels = torch.full((1, 6), -100, dtype=torch.long)
    labels[0, 2] = 1
    labels[0, 4] = 2

    with pytest.raises(ValueError, match="contiguous suffix"):
        selected_answer_nll_from_full_logits(logits, labels)


def test_v4_locks_every_training_field_and_runtime_leakage_rule() -> None:
    locked = build_preregistration()["locked_unchanged_training_fields"]

    assert locked["optimization"]["seed"] == 720054
    assert locked["optimization"]["learning_rate"] == 0.0003
    assert locked["optimization"]["maximum_updates"] == 40
    assert locked["objective"]["same_question_wrong_prefix_hinge_weight"] == 1.0
    assert locked["selection_except_resource_microbatch"]["all_gates_required"] is True
    assert locked["runtime_contract"]["question_dependent_retrieval"] is False
    assert locked["runtime_contract"]["environmental_text_inputs"] == []
    assert locked["publication"]["failed_run_publishes_no_checkpoint"] is True


def test_v4_preregistration_is_create_once_and_source_locked(tmp_path: Path) -> None:
    contract = build_preregistration()
    destination = tmp_path / "v4.json"
    path, digest = write_preregistration(destination)

    assert path == destination.resolve()
    assert len(digest) == 64
    assert json.loads(destination.read_text(encoding="utf-8")) == contract
    assert contract["v4_implementation_source_hashes"] == v4_implementation_hashes()
    with pytest.raises(FileExistsError, match="exists"):
        write_preregistration(destination)


def test_v4_training_paths_all_use_streamed_tail_logits() -> None:
    source = Path(
        "src/semantic_3d_chat/training/train_fixed_prefix_ple_v54_v4.py"
    ).read_text(encoding="utf-8")
    launcher = Path("scripts/run_gemma4_v54_fixed_prefix_ple_reader_v4.sh").read_text(
        encoding="utf-8"
    )

    assert 'v1.answer_nlls = streamed_answer_nlls' in source
    assert 'v1.retention_logits = bounded_retention_logits' in source
    assert '"labels": None' in source
    assert '"logits_to_keep": causal_positions' in source
    assert "logits_to_keep=1" in source
    assert "learning_rate" not in source
    assert "maximum_updates" not in source
    assert "preregister|preflight|smoke|train|authenticate" in launcher
