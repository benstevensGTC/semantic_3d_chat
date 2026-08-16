from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import pytest
import torch

import semantic_3d_chat.evaluation.v35_block_cross_selector as selector
from semantic_3d_chat.config import load_config
from semantic_3d_chat.evaluation.v30_joint_pair_selector import (
    GenerationEvidence,
    PairMarginEvidence,
)

CONFIG = Path("configs/experiments/gemma4_diverse28_block_cross_v35.yaml")


def _prefix(factor: float) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "tensor": "fake_continuous_prefix",
        "rms_by_validation_family": {
            "book_support": factor,
            "mirror_lr": factor,
            "picture_support": factor,
        },
        "weak_pair_mean_rms": factor,
        "unrelated_pair_count": 12,
        "unrelated_mean_rms": 1.0,
        "question_inputs_used": False,
        "all_validation_scenes_processed": True,
    }


def _pair(*, passed: int, mean: float) -> PairMarginEvidence:
    margins = tuple((mean, mean) for _ in range(12))
    return PairMarginEvidence(
        unit_keys=tuple((f"pair_{index // 4}", f"q_{index}") for index in range(12)),
        margins=margins,
        passed_units=passed,
        passed_sides=2 * passed,
        mean_margin=mean,
        minimum_margin=mean,
    )


def _teacher(*, improved: bool) -> selector.V35TeacherEvidence:
    factor = 1.03 if improved else 1.0
    complete = 1 if improved else 0
    return selector.V35TeacherEvidence(
        validation_answer_token_nll=2.0 if improved else 3.0,
        pair_margins=_pair(passed=2 if improved else 1, mean=0.2 if improved else 0.1),
        family_teacher={
            family: {"unit_count": 4, "complete_units": complete, "mean_margin": 0.2}
            for family in ("book_support", "mirror_lr", "picture_support")
        },
        prefix_diagnostics=_prefix(factor),
        color_full_vocab_sides=12,
        mirror_full_vocab_sides=10,
        negative_sides=frozenset(),
        prefix_sha256_by_scene={f"scene_{index:06d}": "a" * 64 for index in range(19, 25)},
    )


def _greedy(complete: int) -> selector.V35GreedyEvidence:
    return selector.V35GreedyEvidence(
        generation=GenerationEvidence(
            changed_row_count=24,
            changed_unit_count=12,
            exact_correct_sides=2 * complete,
            exact_complete_units_correct=complete,
            prediction_changed_units=complete,
            broad_row_count=12,
            broad_exact_correct=8,
        ),
        complete_by_family={
            "book_support": int(complete >= 1),
            "mirror_lr": int(complete >= 1),
            "picture_support": int(complete >= 1),
        },
        prediction_changed_by_family={
            "book_support": int(complete >= 1),
            "mirror_lr": int(complete >= 1),
            "picture_support": int(complete >= 1),
        },
    )


class _FakeEvaluator:
    validation_scene_ids = tuple(f"scene_{index:06d}" for index in range(19, 25))
    cache_audit: ClassVar[Mapping[str, Any]] = {
        "scene_count": 22,
        "all_voxels_covered": True,
        "all_block_tokens_cached": True,
        "question_inputs_to_scene_cache": False,
    }

    def __init__(self) -> None:
        self.step = 0
        self.approved = False
        self.greedy_calls: list[int] = []

    def install(
        self, tensors: Mapping[str, torch.Tensor], *, approved_v29: bool = False
    ) -> None:
        self.approved = approved_v29
        self.step = int(tensors["block_cross_residual.w_q"].reshape(-1)[0])

    def evaluate_teacher(self) -> selector.V35TeacherEvidence:
        return _teacher(improved=not self.approved and self.step > 0)

    def evaluate_greedy(self) -> selector.V35GreedyEvidence:
        self.greedy_calls.append(-1 if self.approved else self.step)
        complete = {-1: 0, 32: 1, 64: 3, 100: 6}.get(
            -1 if self.approved else self.step, 0
        )
        return _greedy(complete)

    def evaluate_aggregate_exact(self) -> tuple[int, int]:
        return (216, 80 if self.approved else 81)

    def attest_prefix_invariance(self) -> Mapping[str, Any]:
        return {
            "passed": True,
            "environment_built_before_questions": True,
            "oracle_environment_files_loaded": False,
            "deferred_final_scenes_loaded": False,
            "question_dependent_scene_processing": False,
            "question_dependent_retrieval": False,
        }


def _adapter(step: int) -> dict[str, torch.Tensor]:
    values = {
        "base": torch.ones(1),
        "block_cross_residual.w_q": torch.tensor([float(step)]),
        "block_cross_residual.w_k": torch.zeros(1),
        "block_cross_residual.w_v": torch.zeros(1),
        "block_cross_residual.w_o": torch.zeros(1),
        "block_cross_residual.architecture_marker": torch.tensor(35),
        "block_cross_residual.architecture_dimensions": torch.tensor(
            [1536, 384, 256, 256, 4, 64]
        ),
        "block_cross_residual.initialization_seed_state": torch.tensor(35035),
        "block_cross_residual.latent_anchors": torch.zeros(1, 3),
        "block_cross_residual.spatial_temperature": torch.tensor(0.2),
        "block_cross_residual.uniform_floor": torch.tensor(0.01),
        "block_cross_residual.residual_scale": torch.tensor(0.25),
    }
    for index in range(4):
        values[
            f"lora_banks.extension_v30_joint_pair_query.adapters.{index}.lora_a"
        ] = torch.ones(1)
        values[
            f"lora_banks.extension_v30_joint_pair_query.adapters.{index}.lora_b"
        ] = torch.zeros(1)
    return values


def test_v35_selector_uses_fake_evaluator_only_after_complete_envelope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = load_config(CONFIG)
    contract = selector.v35_contract(config)
    checkpoints = tuple(
        tmp_path / f"update_{step:03d}" for step in contract.saved_optimizer_steps
    )
    monkeypatch.setattr(
        selector,
        "validate_v35_checkpoint_envelope",
        lambda *_args: (checkpoints, [{"inspected": True}] * len(checkpoints)),
    )
    monkeypatch.setattr(
        selector,
        "_source_v29_evidence",
        lambda _metadata: {"checkpoint": str(tmp_path / "approved")},
    )
    monkeypatch.setattr(selector, "_validate_source_against_config", lambda *_args: None)
    monkeypatch.setattr(
        selector,
        "require_v34_terminal_gate",
        lambda _config: {"path": "terminal.json", "sha256": "a" * 64},
    )
    monkeypatch.setattr(selector, "_metadata", lambda _path: {})

    def fake_load(path: Path, *, device: str) -> dict[str, torch.Tensor]:
        assert device == "cpu"
        if path.parent.name == "approved":
            return {"base": torch.ones(1)}
        return _adapter(int(path.parent.name.removeprefix("update_")))

    monkeypatch.setattr(selector, "load_file", fake_load)
    monkeypatch.setattr(
        selector,
        "_approved_v29_runtime_tensor_envelope",
        lambda update0, _approved, **_kwargs: update0,
    )
    fake = _FakeEvaluator()
    report = selector.select_v35(
        CONFIG,
        tmp_path,
        evaluator_factory=lambda *_args: fake,
    )
    assert report["selected_optimizer_step"] == 100
    assert fake.greedy_calls == [-1, 32, 64, 100]
    assert report["chat_promotion_eligible"] is True
    assert set(report["chat_promotion"]["checks"]) == {
        "development_checkpoint_selected",
        "changed_complete_pair_threshold_met",
        "aggregate_validation_exact_accuracy_retained",
    }
    assert all(report["chat_promotion"]["audited_internal_requirements"].values())


def test_v35_envelope_fails_before_model_or_validation_when_arm_set_is_incomplete(
    tmp_path: Path,
) -> None:
    config = load_config(CONFIG)
    contract = selector.v35_contract(config)
    with pytest.raises(FileNotFoundError, match="complete saved-arm envelope"):
        selector.validate_v35_checkpoint_envelope(config, tmp_path, contract)


def test_v35_approved_v29_envelope_allows_only_zero_compatibility_routes() -> None:
    update0 = _adapter(0)
    approved = {"base": torch.ones(1)}
    expected = selector.tensor_state_sha256(selector._core_state(update0))
    merged = selector._approved_v29_runtime_tensor_envelope(
        update0, approved, expected_core_state_sha256=expected
    )
    assert set(merged) == set(update0)
    forged = dict(update0)
    forged["block_cross_residual.w_o"] = torch.ones(1)
    with pytest.raises(ValueError, match="block route is not exact-zero"):
        selector._approved_v29_runtime_tensor_envelope(
            forged, approved, expected_core_state_sha256=expected
        )
    forged = dict(update0)
    forged["surprise"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unauthorized compatibility tensor"):
        selector._approved_v29_runtime_tensor_envelope(
            forged, approved, expected_core_state_sha256=expected
        )
    forged = dict(update0)
    forged["block_cross_residual.latent_anchors"] = torch.ones(1, 3)
    with pytest.raises(ValueError, match="differs from its exact initialization"):
        selector._approved_v29_runtime_tensor_envelope(
            forged, approved, expected_core_state_sha256=expected
        )


def test_v35_promotion_exact_outward_shape_and_internal_attestations() -> None:
    selected = {
        "checkpoint": "/tmp/update_100",
        "optimizer_step": 100,
        "greedy_exact_complete_units_correct": 6,
        "greedy_complete_units_by_family": {
            "book_support": 1,
            "mirror_lr": 4,
            "picture_support": 1,
        },
        "color_full_vocab_sides": 12,
        "mirror_full_vocab_sides": 10,
        "new_negative_sides_vs_approved_v29": [],
        "checks": {"broad_retention_vs_approved_v29": True},
    }
    attestation = {
        "passed": True,
        "environment_built_before_questions": True,
        "oracle_environment_files_loaded": False,
        "deferred_final_scenes_loaded": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
    }
    promotion = selector._promotion(
        selected,
        approved_v29_aggregate=(216, 80),
        selected_aggregate=(216, 80),
        prefix_attestation=attestation,
    )
    assert set(promotion["checks"]) == {
        "development_checkpoint_selected",
        "changed_complete_pair_threshold_met",
        "aggregate_validation_exact_accuracy_retained",
    }
    assert promotion["eligible"] is True


def test_v35_selector_docs_and_make_target_have_no_final_bypass() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "gemma4-v35-select-block-cross" in makefile
    assert "gemma4-v35-evaluate-final" not in makefile
    assert "V35 post-training selector" in readme
