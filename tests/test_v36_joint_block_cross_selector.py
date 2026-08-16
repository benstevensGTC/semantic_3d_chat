from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
import torch

import semantic_3d_chat.evaluation.v36_joint_block_cross_selector as selector
from semantic_3d_chat.evaluation.v30_joint_pair_selector import (
    GenerationEvidence,
    PairMarginEvidence,
)


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
    complete = 1 if improved else 0
    return selector.V35TeacherEvidence(
        validation_answer_token_nll=2.0 if improved else 3.0,
        pair_margins=_pair(passed=2 if improved else 1, mean=0.2 if improved else 0.1),
        family_teacher={
            family: {"unit_count": 4, "complete_units": complete, "mean_margin": 0.2}
            for family in ("book_support", "mirror_lr", "picture_support")
        },
        prefix_diagnostics=_prefix(1.01 if improved else 1.0),
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

    def install(self, tensors: Mapping[str, torch.Tensor], *, approved_v29: bool = False) -> None:
        self.approved = approved_v29
        self.step = int(tensors["block_cross_residual.w_q"].reshape(-1)[0])

    def evaluate_teacher(self) -> selector.V35TeacherEvidence:
        return _teacher(improved=not self.approved and self.step > 0)

    def evaluate_greedy(self) -> selector.V35GreedyEvidence:
        key = -1 if self.approved else self.step
        self.greedy_calls.append(key)
        return _greedy({-1: 0, 32: 1, 64: 3, 100: 6}.get(key, 0))

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


def _fake_contract() -> SimpleNamespace:
    return SimpleNamespace(
        saved_optimizer_steps=(*range(0, 97, 8), 100),
        decoder_bank_initial_state_sha256="b" * 64,
    )


def _fake_adapter(step: int) -> dict[str, torch.Tensor]:
    return {"block_cross_residual.w_q": torch.tensor([float(step)])}


def test_selector_opens_validation_only_after_complete_envelope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _fake_contract()
    split = SimpleNamespace(
        train_scene_ids=tuple(f"scene_{index:06d}" for index in (*range(11, 19), *range(31, 39))),
        validation_scene_ids=tuple(f"scene_{index:06d}" for index in range(19, 25)),
        deferred_final_scene_ids=tuple(f"scene_{index:06d}" for index in range(25, 31)),
    )
    checkpoints = tuple(tmp_path / f"update_{step:03d}" for step in contract.saved_optimizer_steps)
    monkeypatch.setattr(selector, "load_config", lambda _path: {})
    monkeypatch.setattr(selector, "v36_contract", lambda _config: contract)
    monkeypatch.setattr(selector, "v31_contract", lambda _config: split)
    monkeypatch.setattr(selector, "_selection_requirements", lambda _config: object())
    monkeypatch.setattr(selector, "_retention_control_config", lambda _config: {})
    envelope_complete = False

    def validate(*_args: object) -> tuple[tuple[Path, ...], list[dict[str, Any]]]:
        nonlocal envelope_complete
        envelope_complete = True
        return checkpoints, [{"inspected": True}] * len(checkpoints)

    monkeypatch.setattr(selector, "validate_v36_checkpoint_envelope", validate)
    monkeypatch.setattr(
        selector,
        "_source_v29_evidence",
        lambda _metadata: {"checkpoint": str(tmp_path / "approved")},
    )
    monkeypatch.setattr(selector, "_validate_source_against_config", lambda *_args: None)
    monkeypatch.setattr(selector, "_metadata", lambda _path: {})
    monkeypatch.setattr(
        selector,
        "require_v35_terminal_gate",
        lambda _config: {"path": "terminal.json", "sha256": "a" * 64},
    )
    monkeypatch.setattr(
        selector,
        "_approved_v29_runtime_tensor_envelope",
        lambda update0, _approved, **_kwargs: update0,
    )

    def fake_load(path: Path, *, device: str) -> dict[str, torch.Tensor]:
        assert device == "cpu"
        if path.parent.name == "approved":
            return {"approved": torch.ones(1)}
        return _fake_adapter(int(path.parent.name.removeprefix("update_")))

    monkeypatch.setattr(selector, "load_file", fake_load)
    fake = _FakeEvaluator()

    def factory(*_args: object) -> _FakeEvaluator:
        assert envelope_complete, "evaluator constructed before envelope validation"
        return fake

    report = selector.select_v36(Path("unused.yaml"), tmp_path, evaluator_factory=factory)
    assert report["selected_optimizer_step"] == 100
    assert fake.greedy_calls == [-1, 32, 64, 100]
    assert report["teacher_scored_steps"] == list(contract.saved_optimizer_steps)
    assert report["chat_promotion_eligible"] is True
    assert set(report["chat_promotion"]["checks"]) == {
        "development_checkpoint_selected",
        "changed_complete_pair_threshold_met",
        "aggregate_validation_exact_accuracy_retained",
    }


def test_incomplete_envelope_fails_before_source_or_evaluator(tmp_path: Path) -> None:
    contract = _fake_contract()
    with pytest.raises(FileNotFoundError, match="complete saved-arm envelope"):
        selector.validate_v36_checkpoint_envelope({}, tmp_path, contract)


def test_checkpoint_root_symlink_fails_before_traversal(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="checkpoint root must be a real directory"):
        selector.validate_v36_checkpoint_envelope({}, alias, _fake_contract())


def _source_baseline_fixture() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    pair = {
        "complete_units": 9,
        "cross_prefix_complete_units": 15,
        "positive_sides": 32,
        "mean_cross_prefix_margin": 1.32265043258667,
        "complete_units_by_family": {
            "book_support": 0,
            "mirror_lr": 1,
            "picture_support": 0,
        },
    }
    residual = {"aggregate_rms": 0.041}
    greedy = {"complete_units": 2, "broad_exact_accuracy": 0.75}
    stage = {
        "source_pair_metrics": copy.deepcopy(pair),
        "source_broad_train_nll": 3.25,
        "source_train_greedy_metrics": copy.deepcopy(greedy),
        "source_replay_attestation": {
            "source_complete_units": 9,
            "source_cross_prefix_complete_units": 15,
            "source_positive_sides": 32,
            "source_mean_cross_prefix_margin": 1.32265043258667,
            "source_complete_units_by_family": copy.deepcopy(pair["complete_units_by_family"]),
            "source_residual_rms": 0.041,
        },
    }
    history = [
        {
            "source_pair_metrics": copy.deepcopy(pair),
            "source_broad_train_nll": 3.25,
            "source_train_greedy_metrics": copy.deepcopy(greedy),
            "training_residual_diagnostics": copy.deepcopy(residual),
        }
    ]
    source_rows = [{} for _ in range(32)]
    source_rows.append(
        {
            "training_pair_metrics": copy.deepcopy(pair),
            "training_residual_diagnostics": copy.deepcopy(residual),
        }
    )
    return stage, history, {"history": source_rows}


@pytest.mark.parametrize(
    "tamper",
    [
        "stage_pair",
        "history_pair",
        "history_broad",
        "history_greedy",
        "source_pair",
        "replay_pair",
        "source_residual",
        "replay_residual",
    ],
)
def test_source_baselines_are_bound_to_update_zero_and_exact_v35(
    tamper: str,
) -> None:
    stage, history, source = _source_baseline_fixture()
    selector._validate_source_baseline_provenance(
        stage=stage, history=history, source_metadata=source
    )
    if tamper == "stage_pair":
        stage["source_pair_metrics"]["complete_units"] += 1
    elif tamper == "history_pair":
        history[0]["source_pair_metrics"]["positive_sides"] += 1
    elif tamper == "history_broad":
        history[0]["source_broad_train_nll"] += 0.1
    elif tamper == "history_greedy":
        history[0]["source_train_greedy_metrics"]["complete_units"] += 1
    elif tamper == "source_pair":
        source["history"][-1]["training_pair_metrics"]["complete_units"] += 1
    elif tamper == "replay_pair":
        stage["source_replay_attestation"]["source_complete_units"] += 1
    elif tamper == "source_residual":
        source["history"][-1]["training_residual_diagnostics"]["aggregate_rms"] += 0.1
    elif tamper == "replay_residual":
        stage["source_replay_attestation"]["source_residual_rms"] += 0.1
    with pytest.raises(ValueError, match="V36 source"):
        selector._validate_source_baseline_provenance(
            stage=stage, history=history, source_metadata=source
        )


def test_synthetic_update_zero_envelope_requires_exact_v35_source_and_two_proofs() -> None:
    surfaces = {
        "exact_stopped_v35_update32_loaded": True,
        "fresh_v35_optimizer_state_loaded": False,
        "decoder_bank_exact_zero_output": True,
        "learned_block_core_active": True,
        "joint_update_zero_equivalent_to_v35_update32": True,
    }
    stage = {
        "source_replay_attestation": {
            "exact_stopped_v35_update32_loaded": True,
            "v35_optimizer_state_loaded": False,
            "fresh_adam_state": True,
            "validation_qa_loaded": False,
        },
        "update_zero_equivalence": surfaces,
    }
    contract = SimpleNamespace(
        core_source_state_sha256="c" * 64,
        decoder_bank_initial_state_sha256="b" * 64,
    )
    tensors = {"source": torch.tensor([1.0])}
    selector._validate_update_zero_envelope(
        tensors=tensors,
        source_tensors={"source": torch.tensor([1.0])},
        surface={
            "core_state_sha256": "c" * 64,
            "decoder_bank_state_sha256": "b" * 64,
        },
        stage=stage,
        history=[{"update_zero_surfaces": surfaces}],
        contract=contract,
    )
    forged = dict(stage)
    forged["update_zero_equivalence"] = {**surfaces, "learned_block_core_active": False}
    with pytest.raises(ValueError, match="differs between metadata locations"):
        selector._validate_update_zero_envelope(
            tensors=tensors,
            source_tensors=tensors,
            surface={
                "core_state_sha256": "c" * 64,
                "decoder_bank_state_sha256": "b" * 64,
            },
            stage=forged,
            history=[{"update_zero_surfaces": surfaces}],
            contract=contract,
        )


def _optimizer_tensors() -> dict[str, torch.Tensor]:
    tensors = {
        f"block_cross_residual.{name}": torch.zeros(1) for name in ("w_q", "w_k", "w_v", "w_o")
    }
    tensors.update(
        {
            name: torch.zeros(index + 1)
            for index, name in enumerate(selector._BANK_OPTIMIZER_PARAMETER_NAMES)
        }
    )
    return tensors


def _optimizer_payload(step: int, tensors: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    group_names = {
        "block_cross_residual.qkv": [
            "block_cross_residual.w_q",
            "block_cross_residual.w_k",
            "block_cross_residual.w_v",
        ],
        "block_cross_residual.output": ["block_cross_residual.w_o"],
        f"lora_banks.{selector._BANK_NAME}": list(selector._BANK_OPTIMIZER_PARAMETER_NAMES),
    }
    groups = []
    state: dict[int, dict[str, Any]] = {}
    next_id = 0
    for name, names in group_names.items():
        ids = list(range(next_id, next_id + len(names)))
        next_id += len(names)
        groups.append(
            {
                "name": name,
                "parameter_names": names,
                "params": ids,
                "lr": (
                    2e-5
                    if name.startswith("lora_banks")
                    else (0.0 if step <= 8 else (1e-4 if name.endswith("qkv") else 2.5e-5))
                ),
                "weight_decay": 0.0,
            }
        )
        for parameter_id, tensor_name in zip(ids, names, strict=True):
            if step <= 8 and tensor_name.startswith("block_cross_residual"):
                continue
            tensor_step = step if tensor_name.startswith("lora_banks") else step - 8
            state[parameter_id] = {
                "step": torch.tensor(float(tensor_step)),
                "exp_avg": torch.zeros_like(tensors[tensor_name]),
                "exp_avg_sq": torch.zeros_like(tensors[tensor_name]),
            }
    return {"state": state, "param_groups": groups}


@pytest.mark.parametrize("step", [8, 16, 100])
def test_optimizer_audit_proves_fresh_three_group_staging(tmp_path: Path, step: int) -> None:
    tensors = _optimizer_tensors()
    checkpoint = tmp_path / f"update_{step:03d}"
    checkpoint.mkdir()
    torch.save(_optimizer_payload(step, tensors), checkpoint / "optimizer.pt")
    audit = selector._optimizer_step_audit(checkpoint, expected_step=step, tensors=tensors)
    assert audit["lora_optimizer_step"] == step
    assert audit["block_core_optimizer_step"] == (None if step == 8 else step - 8)
    assert audit["fresh_v36_adam_staging_verified"] is True


def test_optimizer_audit_rejects_interleaved_bank_shape_alias(tmp_path: Path) -> None:
    tensors = _optimizer_tensors()
    checkpoint = tmp_path / "update_016"
    checkpoint.mkdir()
    payload = _optimizer_payload(16, tensors)
    bank = payload["param_groups"][2]
    bank["params"][1], bank["params"][4] = bank["params"][4], bank["params"][1]
    torch.save(payload, checkpoint / "optimizer.pt")
    with pytest.raises(ValueError, match="parameter identifiers changed"):
        selector._optimizer_step_audit(checkpoint, expected_step=16, tensors=tensors)


def test_optimizer_audit_rejects_same_shape_name_reordering(tmp_path: Path) -> None:
    tensors = _optimizer_tensors()
    checkpoint = tmp_path / "update_016"
    checkpoint.mkdir()
    payload = _optimizer_payload(16, tensors)
    qkv = payload["param_groups"][0]
    qkv["parameter_names"][1], qkv["parameter_names"][2] = (
        qkv["parameter_names"][2],
        qkv["parameter_names"][1],
    )
    torch.save(payload, checkpoint / "optimizer.pt")
    with pytest.raises(ValueError, match="ordered names changed"):
        selector._optimizer_step_audit(checkpoint, expected_step=16, tensors=tensors)


def test_promotion_keeps_exact_three_key_outward_contract() -> None:
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


def test_docs_and_make_expose_selector_without_final_bypass() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "gemma4-v36-select-joint-block-cross" in makefile
    assert "gemma4-v36-evaluate-final" not in makefile
    assert "V36 post-training selector" in readme
