from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
import torch

import semantic_3d_chat.evaluation.v38_query_recovery_selector as selector
from semantic_3d_chat.evaluation.v30_joint_pair_selector import (
    GenerationEvidence,
    PairMarginEvidence,
)


def _prefix(factor: float) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "tensor": "continuous_prefix",
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
            family: {"unit_count": 4, "complete_units": int(improved), "mean_margin": 0.2}
            for family in selector._PRIORITY_FAMILIES
        },
        prefix_diagnostics=_prefix(1.01 if improved else 1.0),
        color_full_vocab_sides=12,
        mirror_full_vocab_sides=10,
        negative_sides=frozenset(),
        prefix_sha256_by_scene={
            f"scene_{index:06d}": "a" * 64 for index in range(19, 25)
        },
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

    def install(
        self, tensors: Mapping[str, torch.Tensor], *, approved_v29: bool = False
    ) -> None:
        self.approved = approved_v29
        self.step = int(tensors["step"].item())

    def evaluate_teacher(self) -> selector.V35TeacherEvidence:
        return _teacher(improved=not self.approved and self.step > 0)

    def evaluate_greedy(self) -> selector.V35GreedyEvidence:
        key = -1 if self.approved else self.step
        self.greedy_calls.append(key)
        return _greedy({-1: 0, 16: 6, 24: 6, 32: 7, 40: 7, 41: 8}.get(key, 0))

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
        train_scene_ids=tuple(
            f"scene_{index:06d}" for index in (*range(11, 19), *range(31, 39))
        ),
        validation_scene_ids=tuple(f"scene_{index:06d}" for index in range(19, 25)),
        deferred_final_scene_ids=tuple(f"scene_{index:06d}" for index in range(25, 31)),
    )


def test_selector_crosses_validation_boundary_only_after_passed_u41_envelope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _fake_contract()
    checkpoints = tuple(tmp_path / f"update_{step:03d}" for step in selector._EXPECTED_STEPS)
    envelope_complete = False
    monkeypatch.setattr(selector, "load_config", lambda _path: {})
    monkeypatch.setattr(selector, "v38_contract", lambda _config: contract)

    def validate(*_args: object) -> tuple[tuple[Path, ...], list[dict[str, Any]]]:
        nonlocal envelope_complete
        envelope_complete = True
        return checkpoints, [{"inspected": True}] * len(checkpoints)

    monkeypatch.setattr(selector, "validate_v38_checkpoint_envelope", validate)
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
    monkeypatch.setattr(selector, "v38_loader_config", lambda _config: {})
    monkeypatch.setattr(selector, "v31_contract", lambda _config: _fake_split())
    monkeypatch.setattr(
        selector,
        "require_v37_terminal_gate",
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
        assert envelope_complete, "validation evaluator was constructed before u41 audit"
        return fake

    report = selector.select_v38(Path("unused.yaml"), tmp_path, evaluator_factory=factory)
    assert report["selected_optimizer_step"] == 41
    assert report["training_completed_through_update41_before_validation_loaded"] is True
    assert report["train_only_update41_gate_passed"] is True
    assert fake.greedy_calls == [-1, 16, 24, 32, 40, 41]
    assert report["development_selection_passed"] is True
    assert report["chat_promotion_eligible"] is True
    assert report["runtime_promotion_written"] is False
    assert report["final_test_scenes_touched"] is False


def test_failed_terminal_envelope_prevents_validation_factory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    constructed = False
    monkeypatch.setattr(selector, "load_config", lambda _path: {})
    monkeypatch.setattr(selector, "v38_contract", lambda _config: _fake_contract())

    def fail(*_args: object) -> tuple[tuple[Path, ...], list[dict[str, Any]]]:
        raise ValueError("missing passed update-41 train-only gate")

    def factory(*_args: object) -> _FakeEvaluator:
        nonlocal constructed
        constructed = True
        return _FakeEvaluator()

    monkeypatch.setattr(selector, "validate_v38_checkpoint_envelope", fail)
    with pytest.raises(ValueError, match="update-41"):
        selector.select_v38(Path("unused.yaml"), tmp_path, evaluator_factory=factory)
    assert constructed is False


def test_refusal_writes_no_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "selection.json"

    def fail(*_args: object, **_kwargs: object) -> dict[str, Any]:
        raise ValueError("failed u41")

    monkeypatch.setattr(selector, "select_v38", fail)
    with pytest.raises(ValueError, match="failed u41"):
        selector.run_v38_selector(Path("config.yaml"), tmp_path, output)
    assert not output.exists()


def test_incomplete_root_is_rejected_before_artifact_load(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="exact completed update-41 envelope"):
        selector._checkpoint_paths_or_raise(tmp_path, _fake_contract())

    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="checkpoint root must be a real directory"):
        selector._checkpoint_paths_or_raise(alias, _fake_contract())


def test_optimizer_manifest_must_self_link_to_exact_optimizer_bytes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "update_041"
    checkpoint.mkdir()
    (checkpoint / "optimizer.pt").write_bytes(b"tampered optimizer payload")
    (checkpoint / selector.OPTIMIZER_AUDIT_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact": "v38_optimizer_integrity_manifest",
                "optimizer_step": 41,
                "optimizer_filename": "optimizer.pt",
                "optimizer_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest or file hash changed"):
        selector.optimizer_step_audit(
            checkpoint,
            expected_step=41,
            tensors={name: torch.zeros(shape) for name, shape in zip(
                selector._QUERY_PARAMETER_NAMES, selector._QUERY_SHAPES, strict=True
            )},
        )


def test_per_unit_diagnostic_tamper_is_rejected() -> None:
    pairs = {
        "units": [
            {
                "pair_id": f"pair_{index:06d}",
                "question_key": f"q_{index}",
                "family": "other",
                "scene_ids": [
                    f"scene_{2 * index:06d}",
                    f"scene_{2 * index + 1:06d}",
                ],
                "side_margins": [1.0, 1.0],
                "cross_prefix_margins": [1.1, 0.9],
                "complete": True,
                "cross_prefix_complete": True,
            }
            for index in range(25)
        ]
    }
    diagnostics = [
        {
            "pair_id": f"pair_{index:06d}",
            "question_key": f"q_{index}",
            "family": "other",
            "scene_ids": [f"scene_{2 * index:06d}", f"scene_{2 * index + 1:06d}"],
            "correct_answer_nll": [1.0, 1.1],
            "correct_answer_nll_mean": 1.05,
            "correct_ranking_nll": [1.0, 1.1],
            "swapped_ranking_nll": [2.0, 2.1],
            "side_margins": [1.0, 1.0],
            "cross_prefix_margins": [1.1, 0.9],
            "side_correct": [True, True],
            "cross_prefix_correct": [True, True],
            "side_complete": True,
            "cross_prefix_complete": True,
        }
        for index in range(25)
    ]
    audit = selector.validate_per_unit_nll_diagnostics(diagnostics, pairs)
    assert audit["unit_count"] == 25

    forged = [dict(row) for row in diagnostics]
    forged[-1]["pair_id"] = forged[0]["pair_id"]
    forged[-1]["question_key"] = forged[0]["question_key"]
    with pytest.raises(ValueError, match="duplicate identity"):
        selector.validate_per_unit_nll_diagnostics(forged, pairs)


def test_query_state_is_exact_existing_v30_eight_tensor_surface() -> None:
    tensors = {
        name: torch.zeros(shape)
        for name, shape in zip(
            selector._QUERY_PARAMETER_NAMES, selector._QUERY_SHAPES, strict=True
        )
    }
    tensors["frozen.weight"] = torch.ones(1)
    query = selector._query_state(tensors)
    assert tuple(f"{selector._QUERY_PREFIX}{name}" for name in query) == (
        selector._QUERY_PARAMETER_NAMES
    )
    assert sum(value.numel() for value in query.values()) == 131_072
    assert set(selector._frozen_excluding_query(tensors)) == {"frozen.weight"}

    reordered = {"frozen.weight": torch.ones(1)}
    for name in reversed(selector._QUERY_PARAMETER_NAMES):
        reordered[name] = tensors[name]
    with pytest.raises(ValueError, match="order or inventory"):
        selector._query_state(reordered)

    wrong = dict(tensors)
    wrong[selector._QUERY_PARAMETER_NAMES[0]] = torch.zeros(1)
    with pytest.raises(ValueError, match="shapes changed"):
        selector._query_state(wrong)


class _SyntheticAdapter(torch.nn.Module):
    def __init__(self, a_shape: tuple[int, ...], b_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.lora_a = torch.nn.Parameter(torch.zeros(a_shape))
        self.lora_b = torch.nn.Parameter(torch.zeros(b_shape))


class _SyntheticQueryState(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapters = torch.nn.ModuleList(
            _SyntheticAdapter(
                selector._QUERY_SHAPES[index], selector._QUERY_SHAPES[index + 1]
            )
            for index in range(0, len(selector._QUERY_SHAPES), 2)
        )


class _SyntheticCollection:
    def __init__(self, query: torch.nn.Module, v23: torch.nn.Module) -> None:
        self._states = {selector._QUERY_BANK: query, selector._V23_BANK: v23}

    def bank(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(installation=SimpleNamespace(state_module=self._states[name]))


def test_runtime_evaluator_uses_loader_copy_and_installs_frozen_hybrid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual = {"identity": "actual-v38"}
    loader = {"identity": "v30-construction-copy"}
    query = _SyntheticQueryState()
    v23 = torch.nn.Linear(2, 2, bias=False)
    core = torch.nn.Linear(2, 2, bias=False)
    language = torch.nn.Linear(2, 2, bias=False)
    collection = _SyntheticCollection(query, v23)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(selector, "v38_loader_config", lambda _config: loader)

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
                "query": query,
                "v23": v23,
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
        core.load_state_dict(selector._prefixed_state(tensors, selector._CORE_PREFIX))

    monkeypatch.setattr(selector._V35RuntimeEvaluator, "__init__", fake_base_init)
    monkeypatch.setattr(selector._V35RuntimeEvaluator, "install", fake_base_install)

    def fake_retag(_bundle: object, config: Mapping[str, Any]) -> dict[str, Any]:
        captured["retag_config"] = config
        return {"retagged": True}

    monkeypatch.setattr(selector, "retag_bundle_for_v38", fake_retag)
    monkeypatch.setattr(
        selector,
        "v38_contract",
        lambda _config: SimpleNamespace(
            hybrid_v23_state_sha256=selector.tensor_state_sha256(v23.state_dict()),
            core_state_sha256=selector.tensor_state_sha256(core.state_dict()),
        ),
    )
    evaluator = selector._V38RuntimeEvaluator(actual, {}, Path("update_000"), object())
    assert captured == {
        "construction_config": loader,
        "retag_config": actual,
    }
    tensors = {
        f"{selector._QUERY_PREFIX}{name}": value.detach().clone() + 1
        for name, value in query.state_dict().items()
    }
    tensors.update(
        {
            f"{selector._V23_PREFIX}{name}": value.detach().clone()
            for name, value in v23.state_dict().items()
        }
    )
    tensors.update(
        {
            f"{selector._CORE_PREFIX}{name}": value.detach().clone()
            for name, value in core.state_dict().items()
        }
    )
    evaluator.install(tensors)
    assert all(torch.equal(value, torch.ones_like(value)) for value in query.state_dict().values())
    assert not any(
        parameter.requires_grad
        for module in evaluator.bundle.checkpoint_modules.values()
        for parameter in module.parameters()
    )

    forged = dict(tensors)
    forged[f"{selector._V23_PREFIX}weight"] += 1
    with pytest.raises(RuntimeError, match="frozen V23/core"):
        evaluator.install(forged)


def test_development_gate_requires_six_all_families_and_retention() -> None:
    missing_family = {"book_support": 2, "mirror_lr": 4, "picture_support": 0}
    checks, negatives = selector._development_checks(
        teacher=_teacher(improved=True),
        greedy=_greedy(6, family_complete=missing_family),
        source=_teacher(),
        approved=_teacher(),
        approved_greedy=_greedy(0),
    )
    assert negatives == []
    assert checks["greedy_complete_units_at_least_6_of_12"] is True
    assert checks["each_priority_family_greedy_complete"] is False
    assert checks["hybrid_u0_teacher_complete_units_improved_by_at_least_1"] is True
    assert checks["hybrid_u0_validation_answer_nll_no_worse"] is True


def test_exact_development_evidence_rejects_missing_unit() -> None:
    with pytest.raises(ValueError, match="exactly 12 changed validation units"):
        selector._validate_exact_development_evidence(_teacher(unit_count=11), _greedy(6))


def test_real_v38_contract_has_exact_terminal_envelope() -> None:
    config = selector.load_config(selector.DEFAULT_CONFIG)
    contract = selector.v38_contract(config)
    assert contract.saved_optimizer_steps == selector._EXPECTED_STEPS
    assert contract.saved_optimizer_steps[-1] == 41
    assert contract.query_source_state_sha256 == (
        "050706c300e6fb0ac8e4cc02e26c565b54a9a89505104302d4ffcedc02124c64"
    )
    assert contract.hybrid_v23_state_sha256 == (
        "5c9233bd96b381e2f63443f8a739a868a21997b28061d90fb407a46d9de2d4cb"
    )
