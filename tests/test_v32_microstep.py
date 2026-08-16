from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest
import torch

from semantic_3d_chat.config import config_hash, load_config
from semantic_3d_chat.data.dataset import QARecord
from semantic_3d_chat.evaluation import v32_microstep_selector as selector
from semantic_3d_chat.evaluation.v30_joint_pair_selector import (
    GenerationEvidence,
    PairMarginEvidence,
    RuntimeArmEvidence,
)
from semantic_3d_chat.evaluation.v32_microstep_selector import (
    validate_v32_checkpoint_envelope,
)
from semantic_3d_chat.training.checkpointing import runtime_checkpoint_metadata
from semantic_3d_chat.training.pair_curriculum import CounterfactualPairUnit
from semantic_3d_chat.training.train_joint_pair_v30 import v30_contract
from semantic_3d_chat.training.train_microstep_v32 import (
    build_v32_microstep_schedule,
    latest_v32_resume_checkpoint,
    require_v31_rejection,
    v31_rejection_status,
    v32_contract,
    v32_settings,
    validate_v32_resume_checkpoint,
)

V32_CONFIG = Path("configs/experiments/gemma4_diverse28_microstep_v32.yaml")


def _record(
    index: int,
    *,
    scene_id: str,
    answer_type: str,
    pair_id: str | None = None,
    question_key: str | None = None,
    role: str | None = None,
) -> QARecord:
    return QARecord(
        scene_id=scene_id,
        question_id=f"q_{index:04d}",
        question=f"opaque question {question_key or index}",
        answer="left" if role != "counterfactual" else "right",
        answer_type=answer_type,
        target_xyz=None,
        counterfactual_pair_id=pair_id,
        counterfactual_question_key=question_key,
        counterfactual_expected_change=pair_id is not None,
        counterfactual_role=role,
        counterfactual_change_type="mirror_lr" if pair_id is not None else None,
    )


def _schedule_fixture() -> tuple[list[QARecord], list[CounterfactualPairUnit]]:
    records = [
        _record(
            index,
            scene_id=f"scene_{11 + index % 8:06d}",
            answer_type=("presence", "count", "attribute", "spatial_relation")[index % 4],
        )
        for index in range(100)
    ]
    units: list[CounterfactualPairUnit] = []
    for index in range(25):
        pair_id = f"pair_{index:03d}"
        question_key = f"unit_{index:03d}"
        first = _record(
            1_000 + 2 * index,
            scene_id=f"scene_{11 + (2 * index) % 8:06d}",
            answer_type="spatial_relation",
            pair_id=pair_id,
            question_key=question_key,
            role="reference",
        )
        second = _record(
            1_001 + 2 * index,
            scene_id=f"scene_{11 + (2 * index + 1) % 8:06d}",
            answer_type="spatial_relation",
            pair_id=pair_id,
            question_key=question_key,
            role="counterfactual",
        )
        units.append(CounterfactualPairUnit(pair_id, question_key, first, second))
        records.extend((first, second))
    return records, units


def test_v32_locks_true_microsteps_exact_surface_and_deferred_final() -> None:
    config = load_config(V32_CONFIG)
    settings = v32_settings(config)
    contract = v32_contract(config)
    inherited = v30_contract(config)

    assert settings.optimizer_steps == 80
    assert settings.checkpoint_interval_steps == 8
    assert settings.saved_optimizer_steps == tuple(range(0, 81, 8))
    assert settings.sidecar_learning_rate == 2.5e-5
    assert settings.decoder_learning_rate == 2.0e-5
    assert settings.gradient_clip_norm == 1.0
    assert inherited["joint_trainable_parameter_count"] == 329_216
    assert inherited["source_selected_update"] == 4
    assert contract.v31.validation_scene_ids == tuple(
        f"scene_{index:06d}" for index in range(19, 25)
    )
    assert contract.v31.deferred_final_scene_ids == tuple(
        f"scene_{index:06d}" for index in range(25, 31)
    )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("training.v32_microstep", "optimizer_steps", 8),
        ("training.v32_microstep", "checkpoint_interval_steps", 1),
        ("training.v32_microstep", "sidecar_learning_rate", 1.0e-4),
        ("training.v32_microstep", "gradient_clip_norm", 5.0),
        ("v32_microstep", "validation_scene_ids", ["scene_000025"]),
        ("v32_microstep", "inspect_every_saved_arm", False),
        ("v32_microstep", "development_changed_complete_pairs_minimum", 0),
        ("v32_microstep", "chat_promotion_changed_complete_pairs_minimum", 1),
        ("v32_microstep", "training_requires_v31_rejection", False),
    ],
)
def test_v32_contract_fails_closed(section: str, field: str, value: object) -> None:
    config = copy.deepcopy(load_config(V32_CONFIG))
    target = config
    for key in section.split("."):
        target = target[key]
    target[field] = value
    with pytest.raises((TypeError, ValueError)):
        v32_contract(config)


def test_v32_schedule_has_80_real_balanced_steps_and_recurs_all_25_units() -> None:
    records, units = _schedule_fixture()
    settings = v32_settings(load_config(V32_CONFIG))
    schedule, audit = build_v32_microstep_schedule(
        records,
        units,
        settings=settings,
        seed=17,
    )

    assert [step.optimizer_step for step in schedule] == list(range(1, 81))
    assert all(len(step.broad_records) == 1 for step in schedule)
    assert all(len(step.pair_units) == 1 for step in schedule)
    appearances = Counter(
        (unit.pair_id, unit.question_key) for step in schedule for unit in step.pair_units
    )
    assert len(appearances) == 25
    assert set(appearances.values()) == {3, 4}
    assert audit["true_optimizer_step_per_schedule_row"] is True
    assert audit["pair_unit_minimum_recurrence"] == 3
    assert sum(audit["broad_answer_type_counts"].values()) == 80


def test_v32_schedule_is_deterministic_but_seed_sensitive() -> None:
    records, units = _schedule_fixture()
    settings = v32_settings(load_config(V32_CONFIG))
    _, first = build_v32_microstep_schedule(records, units, settings=settings, seed=17)
    _, repeat = build_v32_microstep_schedule(records, units, settings=settings, seed=17)
    _, changed = build_v32_microstep_schedule(records, units, settings=settings, seed=18)
    assert first["schedule_sha256"] == repeat["schedule_sha256"]
    assert first["schedule_sha256"] != changed["schedule_sha256"]


def _config_with_selection_report(tmp_path: Path, *, passed: bool) -> dict:
    config = copy.deepcopy(load_config(V32_CONFIG))
    contract = v32_contract(config)
    tmp_path.mkdir(parents=True, exist_ok=True)
    report_path = tmp_path / "v31_selection.json"
    selected_update = 1 if passed else None
    report_path.write_text(
        json.dumps(
            {
                "artifact": "v31_diverse28_joint_pair_development_selection",
                "all_intermediate_checkpoints_inspected": True,
                "final_test_scenes_touched": False,
                "development_validation_model_selection_only": True,
                "training_evaluation_only": True,
                "oracle_loaded": False,
                "question_dependent_scene_processing": False,
                "question_dependent_retrieval": False,
                "development_progress_is_not_chat_promotion": True,
                "train_scene_ids": list(contract.v31.train_scene_ids),
                "validation_scene_ids": list(contract.v31.validation_scene_ids),
                "deferred_final_scene_ids": list(contract.v31.deferred_final_scene_ids),
                "arms": [
                    {"update": update, "eligible": passed and update == selected_update}
                    for update in range(9)
                ],
                "selected_update": selected_update,
                "selected_checkpoint": (
                    None if selected_update is None else f"/opaque/update_{selected_update:03d}"
                ),
                "development_selection_passed": passed,
                "chat_promotion": {"eligible": False},
                "chat_promotion_eligible": False,
                "passed": passed,
            }
        ),
        encoding="utf-8",
    )
    config["v32_microstep"]["v31_selection_report"] = str(report_path)
    return config


def test_v32_training_is_authorized_only_after_audited_v31_rejection(tmp_path: Path) -> None:
    rejected = _config_with_selection_report(tmp_path / "rejected", passed=False)
    assert v31_rejection_status(rejected)["training_authorized"] is True
    assert require_v31_rejection(rejected)["status"] == "rejected"

    passed = _config_with_selection_report(tmp_path / "passed", passed=True)
    assert v31_rejection_status(passed)["training_authorized"] is False
    with pytest.raises(RuntimeError, match="conditional on an audited V31 rejection"):
        require_v31_rejection(passed)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda report: report["arms"].pop(), "all nine"),
        (
            lambda report: report["arms"][3].update({"eligible": True}),
            "eligible or selected",
        ),
        (
            lambda report: report.update({"oracle_loaded": True}),
            "leakage/development contract",
        ),
    ],
)
def test_v32_rejects_forged_or_incomplete_v31_rejection(
    tmp_path: Path, mutation, match: str
) -> None:
    config = _config_with_selection_report(tmp_path, passed=False)
    path = Path(config["v32_microstep"]["v31_selection_report"])
    report = json.loads(path.read_text(encoding="utf-8"))
    mutation(report)
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises((TypeError, ValueError), match=match):
        v31_rejection_status(config)


def _write_checkpoint_envelope(root: Path, config: dict) -> None:
    contract = v32_contract(config)
    expected_scene_ids = (*contract.v31.train_scene_ids, *contract.v31.validation_scene_ids)
    derived_scene_ids = tuple(f"scene_{index:06d}" for index in range(31, 39))
    pinned_scene_ids = tuple(
        scene_id for scene_id in expected_scene_ids if scene_id not in derived_scene_ids
    )
    for step in contract.saved_optimizer_steps:
        checkpoint = root / f"update_{step:03d}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter.safetensors").touch()
        if step > 0:
            optimizer_state = {
                parameter_id: {
                    "step": torch.tensor(float(step)),
                    "exp_avg": torch.zeros(1),
                    "exp_avg_sq": torch.zeros(1),
                }
                for parameter_id in range(10)
            }
            torch.save(
                {
                    "state": optimizer_state,
                    "param_groups": [
                        {
                            "name": "dense_sidecar_adapter.output_surfaces",
                            "lr": 2.5e-5,
                            "weight_decay": 0.0,
                            "params": [0, 1],
                        },
                        {
                            "name": "extension_v30_joint_pair_query",
                            "lr": 2.0e-5,
                            "weight_decay": 0.0,
                            "params": list(range(2, 10)),
                        },
                    ],
                },
                checkpoint / "optimizer.pt",
            )
        history = []
        for index in range(step + 1):
            saved = index % 8 == 0
            history.append(
                {
                    "optimizer_update": index,
                    **({} if index == 0 else {"true_optimizer_step": True}),
                    "validation_answer_token_nll": (
                        3.7 - index / 10_000 if saved else None
                    ),
                    "validation_pair_metrics": {} if saved else None,
                }
            )
        metadata = {
            "optimizer_step": step,
            "best_epoch": step,
            "best_monitor_loss": 3.7 - step / 10_000,
            "config_hash": config_hash(config),
            "history": history,
            "v30_joint_pair": {
                "train_scene_ids": list(contract.v31.train_scene_ids),
                "validation_scene_ids": list(contract.v31.validation_scene_ids),
                "train_question_count": contract.v31.train_question_count,
                "validation_question_count": contract.v31.validation_question_count,
                "final_test_scene_ids_loaded": [],
                "oracle_environment_files_loaded": False,
                "scene_cache": {
                    "scene_count": len(expected_scene_ids),
                    "exact_source_scene_prefixes": True,
                    "derived_source_prefixes_recomputed_bit_exact": True,
                    "deterministically_derived_source_scene_ids": list(derived_scene_ids),
                    "historically_pinned_source_scene_ids": list(pinned_scene_ids),
                    "source_prefix_sha256_by_scene": {
                        scene_id: "c" * 64 for scene_id in expected_scene_ids
                    },
                    "loaded_environment_files": [
                        f"/maps/{scene_id}/voxel_map.npz" for scene_id in expected_scene_ids
                    ],
                },
            },
            "v32_microstep": {
                "optimizer_step": step,
                "exact_trainable_parameter_count": 329_216,
                "train_scene_ids": list(contract.v31.train_scene_ids),
                "validation_scene_ids": list(contract.v31.validation_scene_ids),
                "deferred_final_scene_ids_loaded": [],
                "source_is_approved_v29_update_004": True,
                "every_saved_arm_requires_independent_selection": True,
                "conditional_v31_rejection": {
                    "status": "rejected",
                    "report": str(contract.v31_selection_report),
                    "report_sha256": "a" * 64,
                    "training_authorized": True,
                },
                "schedule": {
                    "optimizer_step_count": 80,
                    "schedule_sha256": "b" * 64,
                    "true_optimizer_step_per_schedule_row": True,
                    "pair_units_atomic": True,
                    "every_pair_unit_recurred": True,
                    "pair_unit_minimum_recurrence": 3,
                },
            },
        }
        (checkpoint / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (checkpoint / "runtime_metadata.json").write_text(
            json.dumps(runtime_checkpoint_metadata(metadata)), encoding="utf-8"
        )


def test_v32_checkpoint_envelope_inspects_all_eleven_saved_arms(tmp_path: Path) -> None:
    config = load_config(V32_CONFIG)
    contract = v32_contract(config)
    _write_checkpoint_envelope(tmp_path, config)
    paths = validate_v32_checkpoint_envelope(config, tmp_path, contract)
    assert [path.name for path in paths] == [f"update_{step:03d}" for step in range(0, 81, 8)]

    (tmp_path / "update_040" / "runtime_metadata.json").unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        validate_v32_checkpoint_envelope(config, tmp_path, contract)


def test_v32_checkpoint_envelope_rejects_final_scene_and_fake_step(tmp_path: Path) -> None:
    config = load_config(V32_CONFIG)
    contract = v32_contract(config)
    _write_checkpoint_envelope(tmp_path, config)
    path = tmp_path / "update_080" / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["v32_microstep"]["validation_scene_ids"][-1] = "scene_000025"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="validation split mismatch"):
        validate_v32_checkpoint_envelope(config, tmp_path, contract)


def test_v32_resume_discovers_latest_contiguous_complete_arm_and_validates_it(
    tmp_path: Path,
) -> None:
    config = load_config(V32_CONFIG)
    contract = v32_contract(config)
    settings = v32_settings(config)
    _write_checkpoint_envelope(tmp_path, config)
    (tmp_path / "update_080" / "optimizer.pt").unlink()

    resume = latest_v32_resume_checkpoint(tmp_path, contract)
    assert resume == tmp_path / "update_072"
    resume_cache = json.loads(
        (resume / "metadata.json").read_text(encoding="utf-8")
    )["v30_joint_pair"]["scene_cache"]
    metadata = validate_v32_resume_checkpoint(
        config=config,
        output=tmp_path,
        resume=resume,
        contract=contract,
        settings=settings,
        condition={"report_sha256": "a" * 64},
        schedule_audit={"schedule_sha256": "b" * 64},
        cache_audit=resume_cache,
    )
    assert metadata["optimizer_step"] == 72
    changed_cache = copy.deepcopy(resume_cache)
    changed_cache["source_prefix_sha256_by_scene"][contract.v31.train_scene_ids[0]] = "d" * 64
    with pytest.raises(ValueError, match="cache provenance changed"):
        validate_v32_resume_checkpoint(
            config=config,
            output=tmp_path,
            resume=resume,
            contract=contract,
            settings=settings,
            condition={"report_sha256": "a" * 64},
            schedule_audit={"schedule_sha256": "b" * 64},
            cache_audit=changed_cache,
        )


def test_v32_resume_rejects_a_noncontiguous_or_fake_adam_history(tmp_path: Path) -> None:
    config = load_config(V32_CONFIG)
    contract = v32_contract(config)
    settings = v32_settings(config)
    _write_checkpoint_envelope(tmp_path, config)
    (tmp_path / "update_016" / "runtime_metadata.json").unlink()
    with pytest.raises(ValueError, match="not a contiguous"):
        latest_v32_resume_checkpoint(tmp_path, contract)

    _write_checkpoint_envelope(tmp_path / "fresh", config)
    optimizer_path = tmp_path / "fresh" / "update_080" / "optimizer.pt"
    optimizer = torch.load(optimizer_path, map_location="cpu", weights_only=True)
    optimizer["state"][0]["step"] = torch.tensor(79.0)
    torch.save(optimizer, optimizer_path)
    resume_cache = json.loads(
        (tmp_path / "fresh" / "update_080" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )["v30_joint_pair"]["scene_cache"]
    with pytest.raises(ValueError, match="does not prove step 80"):
        validate_v32_resume_checkpoint(
            config=config,
            output=tmp_path / "fresh",
            resume=tmp_path / "fresh" / "update_080",
            contract=contract,
            settings=settings,
            condition={"report_sha256": "a" * 64},
            schedule_audit={"schedule_sha256": "b" * 64},
            cache_audit=resume_cache,
        )


def test_v32_selector_inspects_every_saved_arm_and_keeps_chat_gate_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(V32_CONFIG)
    contract = v32_contract(config)
    checkpoints = tuple(tmp_path / f"update_{step:03d}" for step in contract.saved_optimizer_steps)

    def metadata(path: Path) -> dict:
        step = int(path.name.removeprefix("update_"))
        return {
            "optimizer_step": step,
            "history": [{"validation_answer_token_nll": 3.7 if step == 0 else 3.6}],
            "v30_joint_pair": {"frozen_inherited_state_sha256": "frozen"},
            "v32_microstep": {
                "conditional_v31_rejection": {
                    "status": "rejected",
                    "report": str(contract.v31_selection_report),
                    "report_sha256": "a" * 64,
                    "training_authorized": True,
                }
            },
        }

    def pair_evidence(metadata_value: dict, *, expected_unit_count: int) -> PairMarginEvidence:
        passed = metadata_value["optimizer_step"] > 0
        margins = tuple(
            (0.1, 0.1) if passed and index == 0 else (-0.1, -0.1) for index in range(12)
        )
        return PairMarginEvidence(
            unit_keys=tuple((f"pair_{index}", f"unit_{index}") for index in range(12)),
            margins=margins,
            passed_units=1 if passed else 0,
            passed_sides=2 if passed else 0,
            mean_margin=sum(value for row in margins for value in row) / 24,
            minimum_margin=-0.1,
        )

    class FakeEvaluator:
        validation_scene_ids = contract.v31.validation_scene_ids

        def __init__(self) -> None:
            self.step = 0

        def install(self, tensors: dict) -> None:
            self.step = int(tensors["step"])

        def evaluate(self) -> RuntimeArmEvidence:
            pair = pair_evidence({"optimizer_step": self.step}, expected_unit_count=12)
            generation = GenerationEvidence(
                changed_row_count=24,
                changed_unit_count=12,
                exact_correct_sides=2 if self.step else 0,
                exact_complete_units_correct=1 if self.step else 0,
                prediction_changed_units=1 if self.step else 0,
                broad_row_count=48,
                broad_exact_correct=24,
            )
            return RuntimeArmEvidence(
                color_full_vocab_sides=12,
                color_full_vocab_units=6,
                mirror_full_vocab_sides=10,
                mirror_full_vocab_units=5,
                negative_sides=frozenset(),
                pair_margins=pair,
                generation=generation,
                prefix_sha256_by_scene={scene: "a" * 64 for scene in self.validation_scene_ids},
            )

        def evaluate_aggregate_exact(self) -> tuple[int, int]:
            return 216, 81

    fake_evaluator = FakeEvaluator()
    monkeypatch.setattr(
        selector,
        "validate_v32_checkpoint_envelope",
        lambda *_args, **_kwargs: checkpoints,
    )
    monkeypatch.setattr(selector, "_metadata", metadata)
    monkeypatch.setattr(
        selector,
        "v31_rejection_status",
        lambda _config: {
            "status": "rejected",
            "report": str(contract.v31_selection_report),
            "report_sha256": "a" * 64,
            "training_authorized": True,
        },
    )
    monkeypatch.setattr(
        selector,
        "_source_v29_evidence",
        lambda _metadata: {"validation_answer_token_nll": 3.7},
    )
    monkeypatch.setattr(selector, "_validate_source_against_config", lambda *_args: None)
    monkeypatch.setattr(selector, "_validate_runtime_metadata", lambda *_args: None)
    monkeypatch.setattr(selector, "_validate_no_leakage_or_final_scenes", lambda *_args: None)
    monkeypatch.setattr(
        selector,
        "_validate_trainable_surface",
        lambda *_args: {"fresh_bank_state_sha256": "fresh"},
    )
    monkeypatch.setattr(selector, "_frozen_tensor_sha256", lambda _tensors: "frozen")
    monkeypatch.setattr(selector, "_validate_update_zero", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(selector, "_pair_margin_evidence", pair_evidence)
    monkeypatch.setattr(
        selector,
        "load_file",
        lambda path, **_kwargs: {"step": int(path.parent.name.removeprefix("update_"))},
    )

    report = selector.select_v32(
        V32_CONFIG,
        tmp_path,
        evaluator_factory=lambda *_args, **_kwargs: fake_evaluator,
    )

    assert report["all_saved_arms_inspected"] is True
    assert [arm["optimizer_step"] for arm in report["arms"]] == list(range(0, 81, 8))
    assert report["development_selection_passed"] is True
    assert report["selected_update"] == 8
    assert report["selected_optimizer_step"] == 8
    assert report["chat_promotion"]["evaluated"] is True
    assert report["chat_promotion_eligible"] is False
    assert report["requirements"]["minimum_greedy_complete_units_correct"] == 1
    assert report["requirements"]["chat_promotion_changed_complete_pairs_minimum"] == 6


def test_v32_docs_and_make_targets_are_conditional_and_offer_no_final_target() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "gemma4-v32-preflight-microstep" in makefile
    assert "gemma4-v32-train-microstep" in makefile
    assert "gemma4-v32-select-microstep" in makefile
    assert "gemma4-v32-evaluate-final" not in makefile
    assert "conditional on an independently" in readme
    assert "rejected V31 selector report" in readme
    assert "There is intentionally no V32-specific" in readme
    assert "final-test bypass target" in readme
