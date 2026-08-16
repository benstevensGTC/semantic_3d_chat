from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
import torch

import semantic_3d_chat.evaluation.v37_scene_ingress_kv_selector as selector
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


def _pair(*, count: int = 12, passed: int = 2, margin: float = 0.2) -> PairMarginEvidence:
    margins = tuple((margin, margin) for _ in range(count))
    return PairMarginEvidence(
        unit_keys=tuple((f"pair_{index // 4}", f"q_{index}") for index in range(count)),
        margins=margins,
        passed_units=passed,
        passed_sides=2 * passed,
        mean_margin=margin,
        minimum_margin=margin,
    )


def _teacher(*, improved: bool = False, unit_count: int = 12) -> selector.V35TeacherEvidence:
    return selector.V35TeacherEvidence(
        validation_answer_token_nll=2.0 if improved else 3.0,
        pair_margins=_pair(
            count=unit_count,
            passed=2 if improved else 1,
            margin=0.2 if improved else 0.1,
        ),
        family_teacher={
            family: {
                "unit_count": 4,
                "complete_units": int(improved),
                "mean_margin": 0.2,
            }
            for family in selector._PRIORITY_FAMILIES
        },
        prefix_diagnostics=_prefix(1.01 if improved else 1.0),
        color_full_vocab_sides=12,
        mirror_full_vocab_sides=10,
        negative_sides=frozenset(),
        prefix_sha256_by_scene={f"scene_{index:06d}": "a" * 64 for index in range(19, 25)},
    )


def _greedy(
    complete: int,
    *,
    family_complete: Mapping[str, int] | None = None,
    broad_correct: int = 8,
) -> selector.V35GreedyEvidence:
    by_family = (
        dict(family_complete)
        if family_complete is not None
        else {family: int(complete > 0) for family in selector._PRIORITY_FAMILIES}
    )
    return selector.V35GreedyEvidence(
        generation=GenerationEvidence(
            changed_row_count=24,
            changed_unit_count=12,
            exact_correct_sides=2 * complete,
            exact_complete_units_correct=complete,
            prediction_changed_units=complete,
            broad_row_count=12,
            broad_exact_correct=broad_correct,
        ),
        complete_by_family=by_family,
        prediction_changed_by_family={
            family: int(complete > 0) for family in selector._PRIORITY_FAMILIES
        },
    )


class _FakeEvaluator:
    validation_scene_ids = tuple(f"scene_{index:06d}" for index in range(19, 25))
    cache_audit: ClassVar[Mapping[str, Any]] = {
        "scene_count": 6,
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
        self.step = int(tensors["step"].item())

    def evaluate_teacher(self) -> selector.V35TeacherEvidence:
        return _teacher(improved=not self.approved and self.step > 0)

    def evaluate_greedy(self) -> selector.V35GreedyEvidence:
        key = -1 if self.approved else self.step
        self.greedy_calls.append(key)
        return _greedy({-1: 0, 16: 5, 32: 6, 64: 7}.get(key, 0))

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
    return SimpleNamespace(saved_optimizer_steps=selector._EXPECTED_STEPS)


def _fake_split() -> SimpleNamespace:
    return SimpleNamespace(
        train_scene_ids=tuple(f"scene_{index:06d}" for index in (*range(11, 19), *range(31, 39))),
        validation_scene_ids=tuple(f"scene_{index:06d}" for index in range(19, 25)),
        deferred_final_scene_ids=tuple(f"scene_{index:06d}" for index in range(25, 31)),
    )


def test_selector_opens_validation_only_after_complete_u64_envelope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _fake_contract()
    checkpoints = tuple(tmp_path / f"update_{step:03d}" for step in contract.saved_optimizer_steps)
    envelope_complete = False

    monkeypatch.setattr(selector, "load_config", lambda _path: {})
    monkeypatch.setattr(selector, "v37_contract", lambda _config: contract)

    def validate(*_args: object) -> tuple[tuple[Path, ...], list[dict[str, Any]]]:
        nonlocal envelope_complete
        envelope_complete = True
        return checkpoints, [{"inspected": True}] * len(checkpoints)

    monkeypatch.setattr(selector, "validate_v37_checkpoint_envelope", validate)
    monkeypatch.setattr(selector, "_selection_requirements", lambda _config: object())
    monkeypatch.setattr(selector, "_retention_control_config", lambda _config: {})
    monkeypatch.setattr(selector, "_metadata", lambda _path: {})
    monkeypatch.setattr(
        selector,
        "_source_v29_evidence",
        lambda _metadata: {"checkpoint": str(tmp_path / "approved")},
    )
    monkeypatch.setattr(selector, "_validate_source_against_config", lambda *_args: None)
    monkeypatch.setattr(
        selector,
        "_approved_v29_envelope",
        lambda _tensors, **_kwargs: {"step": torch.tensor(-1)},
    )
    monkeypatch.setattr(selector, "v37_loader_config", lambda _config: {})
    monkeypatch.setattr(selector, "v31_contract", lambda _config: _fake_split())
    monkeypatch.setattr(
        selector,
        "require_v36_terminal_gate",
        lambda _config: {"path": "terminal.json", "sha256": "b" * 64},
    )

    def fake_load(path: Path, *, device: str) -> dict[str, torch.Tensor]:
        assert device == "cpu"
        if path.parent.name == "approved":
            return {"approved": torch.ones(1)}
        return {
            "step": torch.tensor(int(path.parent.name.removeprefix("update_")), dtype=torch.int64)
        }

    monkeypatch.setattr(selector, "load_file", fake_load)
    fake = _FakeEvaluator()

    def factory(*_args: object) -> _FakeEvaluator:
        assert envelope_complete, "validation evaluator constructed before envelope audit"
        return fake

    report = selector.select_v37(Path("unused.yaml"), tmp_path, evaluator_factory=factory)

    assert report["selected_optimizer_step"] == 64
    assert report["teacher_scored_steps"] == list(selector._EXPECTED_STEPS)
    assert fake.greedy_calls == [-1, 16, 32, 64]
    assert report["development_selection_passed"] is True
    assert report["chat_promotion_eligible"] is True
    assert report["runtime_promotion_written"] is False
    assert report["final_evaluation_ran"] is False
    assert report["final_test_scenes_touched"] is False
    assert set(report["chat_promotion"]["checks"]) == {
        "development_checkpoint_selected",
        "changed_complete_pair_threshold_met",
        "aggregate_validation_exact_accuracy_retained",
    }


def test_envelope_failure_prevents_validation_factory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    constructed = False
    monkeypatch.setattr(selector, "load_config", lambda _path: {})
    monkeypatch.setattr(selector, "v37_contract", lambda _config: _fake_contract())

    def fail(*_args: object) -> tuple[tuple[Path, ...], list[dict[str, Any]]]:
        raise ValueError("missing passed update-64 train gate")

    def factory(*_args: object) -> _FakeEvaluator:
        nonlocal constructed
        constructed = True
        return _FakeEvaluator()

    monkeypatch.setattr(selector, "validate_v37_checkpoint_envelope", fail)
    with pytest.raises(ValueError, match="update-64"):
        selector.select_v37(Path("unused.yaml"), tmp_path, evaluator_factory=factory)
    assert constructed is False


def test_incomplete_or_aliased_checkpoint_root_fails_before_model_load(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="exact completed update-64 envelope"):
        selector.validate_v37_checkpoint_envelope({}, tmp_path, _fake_contract())

    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="checkpoint root must be a real directory"):
        selector.validate_v37_checkpoint_envelope({}, alias, _fake_contract())


def _train_only_stage(provenance: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_optimizer_state_loaded": False,
        "source_optimizer_file_opened": False,
        "validation_qa_loaded": False,
        "oracle_environment_files_loaded": False,
        "deferred_final_scene_ids_loaded": [],
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "independent_selector_required": True,
        "train_qa_dataset": {
            "loaded_files": provenance["qa_loaded_files"],
            "train_question_count": 384,
            "train_scene_ids": provenance["train_scene_ids"],
            "train_changed_pair_unit_count": 25,
            "validation_scene_ids_from_pinned_contract": provenance["validation_scene_ids"],
            "validation_qa_path": provenance["validation_qa_path"],
            "validation_qa_loaded": False,
            "deferred_final_qa_loaded": False,
            "oracle_environment_files_loaded": False,
        },
        "scene_cache": {
            "scene_count": 16,
            "scene_ids": provenance["train_scene_ids"],
            "scene_scope": "training_only",
            "authenticated_manifest_scene_count": 22,
            "authenticated_manifest_train_subset_count": 16,
            "validation_environment_maps_loaded": False,
            "validation_scene_ids_loaded": [],
            "deferred_final_scene_ids_loaded": [],
            "loaded_environment_files": provenance["map_loaded_files"],
            "all_voxels_covered": True,
            "all_occupied_blocks_processed": True,
            "all_block_tokens_cached": True,
            "question_inputs_to_scene_cache": False,
            "answer_inputs_to_scene_cache": False,
            "question_dependent_scene_processing": False,
            "question_dependent_retrieval": False,
            "oracle_environment_files_loaded": False,
            "validation_qa_loaded": False,
        },
    }


def test_training_boundary_locks_exact_qa_and_sixteen_map_paths() -> None:
    config = selector.load_config(selector.DEFAULT_CONFIG)
    provenance = selector._training_provenance(config)
    stage = _train_only_stage(provenance)
    selector._validate_training_boundaries(stage, "synthetic", provenance)

    forged = copy.deepcopy(stage)
    forged["scene_cache"]["loaded_environment_files"][-1] = str(
        Path(forged["scene_cache"]["loaded_environment_files"][-1]).with_name("aliased_map.npz")
    )
    with pytest.raises(ValueError, match="map provenance"):
        selector._validate_training_boundaries(forged, "synthetic", provenance)

    forged = copy.deepcopy(stage)
    forged["train_qa_dataset"]["loaded_files"][-1] = str(
        Path(forged["train_qa_dataset"]["loaded_files"][-1]).with_name("validation.jsonl")
    )
    with pytest.raises(ValueError, match="QA provenance"):
        selector._validate_training_boundaries(forged, "synthetic", provenance)


def test_prefix_replay_locks_exact_train_hashes_and_current_recomputation() -> None:
    config = selector.load_config(selector.DEFAULT_CONFIG)
    provenance = selector._training_provenance(config)
    hashes = {scene_id: "a" * 64 for scene_id in provenance["train_scene_ids"]}
    stage = {
        "prefix_replay_attestation": {
            "source_prefix_scene_count": 16,
            "source_prefix_scene_ids": provenance["train_scene_ids"],
            "source_prefix_sha256_by_scene": hashes,
            "replayed_prefix_sha256_by_scene": dict(hashes),
            "source_prefixes_replayed_bit_exact": True,
            "current_v36_u16_prefixes_recomputed_deterministically_from_exact_tensors": True,
            "external_prefix_manifest_used": False,
            "scene_prefixes_built_before_questions": True,
            "training_scene_prefixes_question_free": True,
            "validation_environment_maps_loaded": False,
            "validation_qa_loaded": False,
            "question_dependent_scene_processing": False,
            "question_dependent_retrieval": False,
        }
    }
    selector._validate_prefix_replay(stage, "synthetic", provenance)

    forged = copy.deepcopy(stage)
    forged["prefix_replay_attestation"]["external_prefix_manifest_used"] = True
    with pytest.raises(ValueError, match="prefix replay"):
        selector._validate_prefix_replay(forged, "synthetic", provenance)

    forged = copy.deepcopy(stage)
    forged["prefix_replay_attestation"]["replayed_prefix_sha256_by_scene"][
        provenance["train_scene_ids"][0]
    ] = "b" * 64
    with pytest.raises(ValueError, match="prefix replay"):
        selector._validate_prefix_replay(forged, "synthetic", provenance)


def test_exact_development_evidence_rejects_missing_changed_unit() -> None:
    with pytest.raises(ValueError, match="exactly 12 changed validation units"):
        selector._validate_exact_development_evidence(_teacher(unit_count=11), _greedy(6))

    malformed = selector.V35GreedyEvidence(
        generation=GenerationEvidence(
            changed_row_count=22,
            changed_unit_count=11,
            exact_correct_sides=12,
            exact_complete_units_correct=6,
            prediction_changed_units=6,
            broad_row_count=12,
            broad_exact_correct=8,
        ),
        complete_by_family={family: 1 for family in selector._PRIORITY_FAMILIES},
        prediction_changed_by_family={family: 1 for family in selector._PRIORITY_FAMILIES},
    )
    with pytest.raises(ValueError, match="exactly 12 changed validation units"):
        selector._validate_exact_development_evidence(_teacher(), malformed)


def test_development_gate_requires_six_all_families_and_locked_retention() -> None:
    approved_teacher = _teacher()
    approved_greedy = _greedy(0)
    missing_family = {
        "book_support": 2,
        "mirror_lr": 4,
        "picture_support": 0,
    }
    checks, negatives = selector._development_checks(
        teacher=_teacher(improved=True),
        greedy=_greedy(6, family_complete=missing_family),
        source=_teacher(),
        approved=approved_teacher,
        approved_greedy=approved_greedy,
    )
    assert negatives == []
    assert checks["greedy_complete_units_at_least_6_of_12"] is True
    assert checks["each_priority_family_greedy_complete"] is False
    assert checks["v36_u16_teacher_complete_units_improved_by_at_least_1"] is True
    assert checks["v36_u16_validation_answer_nll_no_worse"] is True
    assert all(
        checks[key]
        for key in (
            "approved_v29_color_12_sides_retained",
            "approved_v29_mirror_10_sides_retained",
            "approved_v29_controls_no_new_negatives",
            "broad_retention_vs_approved_v29",
        )
    )


def test_target_state_is_exact_eight_tensor_shared_kv_surface() -> None:
    tensors = {
        name: torch.zeros(shape, dtype=torch.float32)
        for name, shape in zip(
            selector._TARGET_PARAMETER_NAMES, selector._TARGET_SHAPES, strict=True
        )
    }
    tensors["frozen.weight"] = torch.ones(1)
    target = selector._target_state(tensors)
    assert tuple(f"{selector._TARGET_PREFIX}{name}" for name in target) == (
        selector._TARGET_PARAMETER_NAMES
    )
    assert sum(value.numel() for value in target.values()) == 30_720
    assert set(selector._frozen_complement(tensors)) == {"frozen.weight"}

    reordered = {"frozen.weight": torch.ones(1)}
    for name in reversed(selector._TARGET_PARAMETER_NAMES):
        reordered[name] = tensors[name]
    with pytest.raises(ValueError, match="order or inventory"):
        selector._target_state(reordered)

    wrong_shape = dict(tensors)
    wrong_shape[selector._TARGET_PARAMETER_NAMES[0]] = torch.zeros(1)
    with pytest.raises(ValueError, match="shapes changed"):
        selector._target_state(wrong_shape)


class _SyntheticAdapter(torch.nn.Module):
    def __init__(self, a_shape: tuple[int, ...], b_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.lora_a = torch.nn.Parameter(torch.zeros(a_shape))
        self.lora_b = torch.nn.Parameter(torch.zeros(b_shape))


class _SyntheticTargetState(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapters = torch.nn.ModuleList(
            _SyntheticAdapter(selector._TARGET_SHAPES[index], selector._TARGET_SHAPES[index + 1])
            for index in range(0, len(selector._TARGET_SHAPES), 2)
        )


class _SyntheticCollection:
    def __init__(self, target: torch.nn.Module, query: torch.nn.Module) -> None:
        self._states = {
            selector._TARGET_BANK: target,
            selector._QUERY_BANK: query,
        }

    def bank(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(installation=SimpleNamespace(state_module=self._states[name]))


def test_runtime_evaluator_uses_loader_copy_then_installs_exact_frozen_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual = {"identity": "actual-v37"}
    loader = {"identity": "v30-construction-copy"}
    target = _SyntheticTargetState()
    query = torch.nn.Linear(2, 2, bias=False)
    core = torch.nn.Linear(2, 2, bias=False)
    language = torch.nn.Linear(2, 2, bias=False)
    collection = _SyntheticCollection(target, query)
    captured: dict[str, Any] = {}

    monkeypatch.setattr(selector, "v37_loader_config", lambda config: loader)

    def fake_base_init(
        evaluator: object,
        config: dict[str, Any],
        _control: dict[str, Any],
        _checkpoint: Path,
        _requirements: object,
    ) -> None:
        captured["construction_config"] = config
        evaluator.bundle = SimpleNamespace(
            lora_installation=collection,
            language=SimpleNamespace(model=language),
            checkpoint_modules={
                "target": target,
                "query": query,
                "block_cross_residual": core,
            },
        )
        evaluator.block_cross_residual = core

    def fake_base_install(
        _evaluator: object,
        tensors: Mapping[str, torch.Tensor],
        *,
        approved_v29: bool = False,
    ) -> None:
        assert approved_v29 is False
        query.load_state_dict(
            selector._prefixed_state(tensors, selector._QUERY_PREFIX), strict=True
        )
        core.load_state_dict(selector._prefixed_state(tensors, selector._CORE_PREFIX), strict=True)

    monkeypatch.setattr(selector._V35RuntimeEvaluator, "__init__", fake_base_init)
    monkeypatch.setattr(selector._V35RuntimeEvaluator, "install", fake_base_install)

    def fake_retag(_bundle: object, config: Mapping[str, Any]) -> dict[str, Any]:
        captured["retag_config"] = config
        return {"retagged": True}

    monkeypatch.setattr(selector, "retag_bundle_for_v37", fake_retag)
    query_hash = selector.tensor_state_sha256(query.state_dict())
    core_hash = selector.tensor_state_sha256(core.state_dict())
    monkeypatch.setattr(
        selector,
        "v37_contract",
        lambda _config: SimpleNamespace(
            source_query_state_sha256=query_hash,
            source_core_state_sha256=core_hash,
        ),
    )

    evaluator = selector._V37RuntimeEvaluator(actual, {}, Path("update_000"), object())
    assert captured["construction_config"] is loader
    assert captured["retag_config"] == actual
    assert evaluator.config == actual
    assert evaluator.loader_transition == {"retagged": True}

    tensors = {
        f"{selector._TARGET_PREFIX}{name}": value.detach().clone() + 1
        for name, value in target.state_dict().items()
    }
    tensors.update(
        {
            f"{selector._QUERY_PREFIX}{name}": value.detach().clone()
            for name, value in query.state_dict().items()
        }
    )
    tensors.update(
        {
            f"{selector._CORE_PREFIX}{name}": value.detach().clone()
            for name, value in core.state_dict().items()
        }
    )
    evaluator.install(tensors)
    assert all(torch.equal(value, torch.ones_like(value)) for value in target.state_dict().values())
    assert not any(
        parameter.requires_grad
        for module in evaluator.bundle.checkpoint_modules.values()
        for parameter in module.parameters()
    )

    forged = dict(tensors)
    forged[f"{selector._QUERY_PREFIX}weight"] = forged[f"{selector._QUERY_PREFIX}weight"] + 1
    with pytest.raises(RuntimeError, match="frozen learned query/core"):
        evaluator.install(forged)


def test_real_loader_copy_has_v30_construction_and_v37_runtime_trainability() -> None:
    actual = selector.load_config(selector.DEFAULT_CONFIG)
    loader = selector.v37_loader_config(actual)
    actual_banks = actual["language"]["lora_banks"]
    loader_banks = loader["language"]["lora_banks"]
    assert actual_banks[selector._TARGET_BANK]["trainable"] is True
    assert actual_banks[selector._QUERY_BANK]["trainable"] is False
    assert loader_banks[selector._TARGET_BANK]["trainable"] is False
    assert loader_banks[selector._QUERY_BANK]["trainable"] is True
    assert actual_banks[selector._QUERY_BANK]["initialization_algorithm"] == (
        "checkpoint_overwrite"
    )
    assert loader_banks[selector._QUERY_BANK]["initialization_algorithm"] == (
        "cpu_kaiming_uniform_a_exact_zero_b"
    )


def test_real_v37_contract_exposes_only_exact_saved_steps() -> None:
    config = selector.load_config(selector.DEFAULT_CONFIG)
    contract = selector.v37_contract(config)
    assert contract.saved_optimizer_steps == selector._EXPECTED_STEPS
    assert contract.saved_optimizer_steps[-1] == 64
    assert contract.target_source_state_sha256 == (
        "91a9eea577cab5a37e840cdf4007722a398415846af91280713bcb2cda0f045c"
    )
