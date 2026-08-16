from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from semantic_3d_chat.config import load_config
from semantic_3d_chat.evaluation import v30_joint_pair_selector as selector
from semantic_3d_chat.evaluation.v30_joint_pair_selector import (
    FRESH_BANK_NAME,
    FRESH_BANK_PREFIX,
    GenerationEvidence,
    PairMarginEvidence,
    RuntimeArmEvidence,
    _frozen_tensor_sha256,
    _generation_evidence,
    _pair_margin_evidence,
    _selection_requirements,
    _source_v29_evidence,
    _validate_trainable_surface,
)


def _config(*, updates: int = 1) -> dict:
    return {
        "seed": 30030,
        "training": {"v30_joint_pair": {"max_optimizer_steps": updates}},
        "v30_joint_pair": {
            "schema_version": 1,
            "source_selection_report": "/approved/v29-selection.json",
            "source_selection_report_sha256": "c" * 64,
            "source_checkpoint_root": "/approved/v29",
            "source_selected_update": 4,
            "source_adapter_sha256": "a" * 64,
            "source_runtime_metadata_sha256": "b" * 64,
            "fresh_bank": FRESH_BANK_NAME,
            "fresh_bank_parameter_count": selector.FRESH_BANK_PARAMETER_COUNT,
            "fresh_bank_initial_state_sha256": selector.FRESH_BANK_INITIAL_STATE_SHA256,
            "sidecar_trainable_parameter_names": [
                "output_projection.weight",
                "channel_gain",
            ],
            "sidecar_trainable_parameter_count": selector.SIDECAR_PARAMETER_COUNT,
            "joint_trainable_parameter_count": selector.TOTAL_TRAINABLE_PARAMETER_COUNT,
            "update_zero_validation_nll_absolute_tolerance": 1.0e-7,
            "selection_requires": {
                "color_full_vocab_sides": 12,
                "mirror_full_vocab_sides": 10,
                "no_new_negative_sides": True,
                "source_v29_validation_nll_must_improve": True,
                "validation_pair_unit_count": 12,
                "minimum_pair_margin": -10.0,
                "minimum_mean_margin_improvement": 0.01,
                "minimum_passed_unit_improvement": 1,
                "greedy_changed_row_count": 24,
                "minimum_greedy_complete_units_correct": 1,
                "broad_retention_subset_size": 48,
                "broad_exact_accuracy_no_regression": True,
            },
            "promotion_requires": {
                "validation_changed_complete_pairs_minimum": 6,
                "aggregate_validation_exact_accuracy_no_regression": True,
                "label": "chat_promotion_not_merely_development_progress",
            },
        },
    }


def _state(*, b_value: float = 0.0) -> dict[str, torch.Tensor]:
    return {
        "scene_model.inherited": torch.tensor([7.0]),
        "dense_sidecar_adapter.output_projection.weight": torch.zeros(196_608),
        "dense_sidecar_adapter.channel_gain": torch.zeros(1_536),
        "dense_sidecar_adapter.input_projection.weight": torch.tensor([3.0, 4.0]),
        f"{FRESH_BANK_PREFIX}adapters.0.lora_a": torch.zeros(131_068),
        f"{FRESH_BANK_PREFIX}adapters.0.lora_b": torch.full((4,), b_value),
    }


def _margin_metrics(margin: float) -> dict:
    rows = [
        {
            "pair_id": f"pair_{index:06d}",
            "question_key": f"cfq_{index:016x}",
            "scene_ids": [
                f"scene_{19 + 2 * (index % 3):06d}",
                f"scene_{20 + 2 * (index % 3):06d}",
            ],
            "margins": [margin, margin],
        }
        for index in range(12)
    ]
    passed = 12 if margin > 0 else 0
    return {
        "unit_count": 12,
        "side_count": 24,
        "passed_units": passed,
        "side_accuracy": 1.0 if margin > 0 else 0.0,
        "unit_accuracy": 1.0 if margin > 0 else 0.0,
        "mean_margin": margin,
        "minimum_margin": margin,
        "margins_by_unit": rows,
    }


def _metadata(update: int, state: dict[str, torch.Tensor], *, margin: float, nll: float) -> dict:
    fresh = {
        name.removeprefix(FRESH_BANK_PREFIX): value
        for name, value in state.items()
        if name.startswith(FRESH_BANK_PREFIX)
    }
    frozen_hash = _frozen_tensor_sha256(state)
    return {
        "optimizer_step": update,
        "history": [
            {
                "validation_answer_token_nll": nll,
                "validation_pair_metrics": _margin_metrics(margin),
            }
        ],
        "lora_bank_state_sha256": {FRESH_BANK_NAME: selector.tensor_state_sha256(fresh)},
        "v30_joint_pair": {
            "source_v29_checkpoint": "/approved/v29/update_004",
            "source_v29_adapter_sha256": "a" * 64,
            "source_v29_runtime_metadata_sha256": "b" * 64,
            "source_v29_selection_report": "/approved/v29-selection.json",
            "source_v29_selection_report_sha256": "c" * 64,
            "source_v29_selected_update": 4,
            "frozen_inherited_state_sha256": frozen_hash,
            "trainable_surface": {
                "sidecar_parameter_names": sorted(selector.SIDECAR_PARAMETER_NAMES),
                "sidecar_parameter_count": selector.SIDECAR_PARAMETER_COUNT,
                "fresh_bank": FRESH_BANK_NAME,
                "fresh_bank_parameter_names": sorted(
                    name for name in state if name.startswith(FRESH_BANK_PREFIX)
                ),
                "fresh_bank_parameter_count": selector.FRESH_BANK_PARAMETER_COUNT,
                "fresh_bank_target_modules": list(selector.FRESH_BANK_TARGET_MODULES),
                "total_parameter_count": selector.TOTAL_TRAINABLE_PARAMETER_COUNT,
                "every_other_parameter_frozen": True,
            },
            "update_zero_equivalence": {
                "approved_v29_source": True,
                "fresh_bank_exact_zero_output": True,
                "target_outputs_bit_exact": {
                    target: True for target in selector.FRESH_BANK_TARGET_MODULES
                },
                "exact_source_scene_prefixes": True,
                "exact_source_validation_nll": True,
                "question_dependent_scene_processing": False,
                "question_dependent_retrieval": False,
                "oracle_environment_files_loaded": False,
                "source_validation_answer_token_nll": nll,
                "observed_validation_answer_token_nll": nll,
                "validation_nll_absolute_tolerance": 1.0e-7,
            },
            "scene_cache": {
                "all_voxels_covered": True,
                "question_inputs_to_scene_cache": False,
                "question_dependent_scene_processing": False,
                "question_dependent_retrieval": False,
                "oracle_environment_files_loaded": False,
                "loaded_environment_files": [
                    "/maps/scene_000019/voxel_map.npz",
                    "/maps/scene_000024/voxel_map.npz",
                ],
                "exact_source_scene_prefixes": True,
            },
            "qa_dataset": {"deferred_test_scene_ids_loaded": []},
            "train_scene_ids": [f"scene_{value:06d}" for value in range(11, 19)],
            "validation_scene_ids": [f"scene_{value:06d}" for value in range(19, 25)],
            "final_test_scene_ids_loaded": [],
            "oracle_environment_files_loaded": False,
            "question_dependent_scene_processing": False,
            "question_dependent_retrieval": False,
            "development_validation_model_selection_only": True,
        },
    }


def _write_checkpoint(
    root: Path, update: int, *, margin: float, nll: float, b_value: float
) -> Path:
    path = root / f"update_{update:03d}"
    path.mkdir()
    state = _state(b_value=b_value)
    save_file(state, path / "adapter.safetensors")
    metadata = _metadata(update, state, margin=margin, nll=nll)
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    runtime = selector.runtime_checkpoint_metadata(metadata)
    (path / "runtime_metadata.json").write_text(json.dumps(runtime), encoding="utf-8")
    return path


def _pair_evidence(margin: float) -> PairMarginEvidence:
    return _pair_margin_evidence(
        {"history": [{"validation_pair_metrics": _margin_metrics(margin)}]},
        expected_unit_count=12,
    )


def test_checked_in_config_pins_exact_joint_surface_and_selection_gates() -> None:
    config = load_config(Path("configs/experiments/gemma4_diverse20_joint_pair_v30.yaml"))
    requirements = _selection_requirements(config)
    assert requirements.final_update == 4
    assert requirements.color_full_vocab_sides == 12
    assert requirements.mirror_full_vocab_sides == 10
    assert requirements.validation_pair_unit_count == 12
    assert requirements.minimum_mean_margin_improvement == pytest.approx(0.01)
    assert requirements.minimum_passed_unit_improvement == 1
    assert requirements.greedy_changed_row_count == 24
    assert requirements.broad_retention_subset_size == 48
    assert requirements.promotion_changed_complete_pairs_minimum == 6

    bank = selector.lora_banks_settings(config).bank(FRESH_BANK_NAME).adapter
    assert bank.rank == 8
    assert bank.alpha == pytest.approx(16.0)
    assert bank.target_modules == selector.FRESH_BANK_TARGET_MODULES


def test_tensor_audit_excludes_only_the_exact_joint_surface() -> None:
    state = _state()
    metadata = _metadata(0, state, margin=-0.1, nll=3.0)
    audit = _validate_trainable_surface(metadata, state)
    assert audit["fresh_bank_parameter_count"] == 131_072
    assert audit["sidecar_parameter_count"] == 198_144
    assert audit["total_parameter_count"] == 329_216

    changed_authorized = dict(state)
    changed_authorized[f"{FRESH_BANK_PREFIX}adapters.0.lora_b"] = torch.ones(4)
    assert _frozen_tensor_sha256(changed_authorized) == _frozen_tensor_sha256(state)
    changed_inherited = dict(state)
    changed_inherited["scene_model.inherited"] = torch.tensor([8.0])
    assert _frozen_tensor_sha256(changed_inherited) != _frozen_tensor_sha256(state)
    changed_hidden_sidecar = dict(state)
    changed_hidden_sidecar["dense_sidecar_adapter.input_projection.weight"] = torch.tensor(
        [5.0, 6.0]
    )
    assert _frozen_tensor_sha256(changed_hidden_sidecar) != _frozen_tensor_sha256(state)

    bad_metadata = json.loads(json.dumps(metadata))
    bad_metadata["v30_joint_pair"]["trainable_surface"]["every_other_parameter_frozen"] = False
    with pytest.raises(ValueError, match="every other"):
        _validate_trainable_surface(bad_metadata, state)


def test_pair_margin_evidence_is_recomputed_from_every_atomic_unit() -> None:
    evidence = _pair_evidence(0.25)
    assert len(evidence.unit_keys) == 12
    assert evidence.passed_units == 12
    assert evidence.passed_sides == 24
    assert evidence.mean_margin == pytest.approx(0.25)
    assert evidence.minimum_margin == pytest.approx(0.25)

    bad = _margin_metrics(0.25)
    bad["passed_units"] = 11
    with pytest.raises(ValueError, match="disagrees with raw margins"):
        _pair_margin_evidence(
            {"history": [{"validation_pair_metrics": bad}]},
            expected_unit_count=12,
        )

    duplicate = _margin_metrics(0.25)
    duplicate["margins_by_unit"][1] = duplicate["margins_by_unit"][0]
    with pytest.raises(ValueError, match="Duplicate"):
        _pair_margin_evidence(
            {"history": [{"validation_pair_metrics": duplicate}]},
            expected_unit_count=12,
        )


def test_greedy_evidence_requires_complete_units_and_exact_broad_subset() -> None:
    changed = []
    for unit in range(12):
        for side in range(2):
            changed.append(
                {
                    "pair_id": f"pair_{unit:06d}",
                    "question_key": f"cfq_{unit:016x}",
                    "scene_id": f"scene_{19 + 2 * (unit % 3) + side:06d}",
                    "prediction": "left" if side == 0 else "right",
                    "target": "left" if side == 0 else "right",
                }
            )
    broad = [
        {
            "scene_id": f"scene_{19 + index % 6:06d}",
            "question_id": f"q_{index:06d}",
            "prediction": "yes",
            "target": "yes" if index < 30 else "no",
        }
        for index in range(48)
    ]
    evidence = _generation_evidence(
        changed,
        broad,
        expected_changed_rows=24,
        expected_broad_rows=48,
    )
    assert evidence.exact_complete_units_correct == 12
    assert evidence.prediction_changed_units == 12
    assert evidence.broad_exact_correct == 30

    with pytest.raises(ValueError, match="two rows"):
        _generation_evidence(
            changed[:-1] + [{**changed[-1], "pair_id": changed[0]["pair_id"]}],
            broad,
            expected_changed_rows=24,
            expected_broad_rows=48,
        )
    final_scene = [dict(row) for row in changed]
    final_scene[0]["scene_id"] = "scene_000025"
    with pytest.raises(ValueError, match="deferred final"):
        _generation_evidence(
            final_scene,
            broad,
            expected_changed_rows=24,
            expected_broad_rows=48,
        )


def test_source_v29_provenance_is_bound_to_passed_selected_files(tmp_path: Path) -> None:
    source = tmp_path / "source" / "update_004"
    source.mkdir(parents=True)
    (source / "adapter.safetensors").write_bytes(b"approved adapter")
    (source / "runtime_metadata.json").write_text("{}", encoding="utf-8")
    (source / "metadata.json").write_text(
        json.dumps(
            {
                "optimizer_step": 4,
                "history": [{"validation_answer_token_nll": 3.25}],
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "selection.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact": "v28_post_stack_decoder_stage_b_selection",
                "training_evaluation_only": True,
                "question_text_serialized": False,
                "answer_text_serialized": False,
                "oracle_loaded": False,
                "question_dependent_scene_processing": False,
                "question_dependent_retrieval": False,
                "passed": True,
                "selected_update": 4,
                "selected_checkpoint": str(source),
                "arms": [
                    {
                        "update": 4,
                        "eligible": True,
                        "validation_answer_token_nll": 3.25,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    metadata = {
        "v30_joint_pair": {
            "source_v29_checkpoint": str(source),
            "source_v29_adapter_sha256": selector._file_sha256(source / "adapter.safetensors"),
            "source_v29_runtime_metadata_sha256": selector._file_sha256(
                source / "runtime_metadata.json"
            ),
            "source_v29_selection_report": str(report),
            "source_v29_selection_report_sha256": selector._file_sha256(report),
            "source_v29_selected_update": 4,
        }
    }
    evidence = _source_v29_evidence(metadata)
    assert evidence["checkpoint"] == str(source)
    assert evidence["selected_update"] == 4
    assert evidence["validation_answer_token_nll"] == pytest.approx(3.25)

    tampered = json.loads(json.dumps(metadata))
    tampered["v30_joint_pair"]["source_v29_adapter_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="adapter hash"):
        _source_v29_evidence(tampered)


def test_selector_applies_all_relative_retention_and_generation_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()
    first = _write_checkpoint(root, 0, margin=-0.1, nll=3.0, b_value=0.0)
    initial_tensors = selector.load_file(first / "adapter.safetensors", device="cpu")
    initial_hash = selector.tensor_state_sha256(selector._fresh_bank_state(initial_tensors))
    monkeypatch.setattr(selector, "FRESH_BANK_INITIAL_STATE_SHA256", initial_hash)
    config = _config()
    config["v30_joint_pair"]["fresh_bank_initial_state_sha256"] = initial_hash
    _write_checkpoint(root, 1, margin=0.2, nll=2.5, b_value=1.0)

    monkeypatch.setattr(selector, "load_config", lambda _path: config)
    monkeypatch.setattr(selector, "_retention_control_config", lambda _value: config)
    monkeypatch.setattr(
        selector,
        "_source_v29_evidence",
        lambda _metadata: {
            "checkpoint": "/approved/v29/update_004",
            "selected_update": 4,
            "selection_report": "/approved/v29-selection.json",
            "selection_report_sha256": "c" * 64,
            "adapter_sha256": "a" * 64,
            "runtime_metadata_sha256": "b" * 64,
            "validation_answer_token_nll": 3.0,
        },
    )

    class FakeEvaluator:
        validation_scene_ids = tuple(f"scene_{value:06d}" for value in range(19, 25))

        def __init__(self) -> None:
            self.index = -1
            self.aggregate_calls = 0

        def install(self, _tensors) -> None:
            self.index += 1

        def evaluate(self) -> RuntimeArmEvidence:
            margin = (-0.1, 0.2)[self.index]
            complete = (0, 1)[self.index]
            broad_correct = (24, 24)[self.index]
            return RuntimeArmEvidence(
                color_full_vocab_sides=12,
                color_full_vocab_units=6,
                mirror_full_vocab_sides=10,
                mirror_full_vocab_units=4,
                negative_sides=frozenset({("scene_000003", "q_000001")}),
                pair_margins=_pair_evidence(margin),
                generation=GenerationEvidence(
                    changed_row_count=24,
                    changed_unit_count=12,
                    exact_correct_sides=2 * complete,
                    exact_complete_units_correct=complete,
                    prediction_changed_units=complete,
                    broad_row_count=48,
                    broad_exact_correct=broad_correct,
                ),
                prefix_sha256_by_scene={
                    scene_id: f"{self.index + 1:064x}" for scene_id in self.validation_scene_ids
                },
            )

        def evaluate_aggregate_exact(self) -> tuple[int, int]:
            self.aggregate_calls += 1
            return 216, 70

    fake = FakeEvaluator()
    report = selector.select_joint_pair(
        tmp_path / "config.yaml",
        root,
        evaluator_factory=lambda *_args: fake,
    )
    assert report["passed"] is True
    assert report["development_selection_passed"] is True
    assert report["selected_update"] == 1
    assert report["arms"][0]["eligible"] is False
    assert report["arms"][1]["eligible"] is True
    assert all(report["arms"][1]["checks"].values())
    assert report["final_test_scenes_touched"] is False
    assert report["question_dependent_scene_processing"] is False
    assert report["question_dependent_retrieval"] is False
    assert report["oracle_loaded"] is False
    assert report["model_load_count"] == 1
    assert report["chat_promotion_eligible"] is False
    assert report["chat_promotion"]["evaluated"] is True
    assert report["chat_promotion"]["checks"] == {
        "development_checkpoint_selected": True,
        "changed_complete_pair_threshold_met": False,
        "aggregate_validation_exact_accuracy_retained": True,
    }
    serialized = json.dumps(report)
    assert 'question"' not in serialized
    assert 'target"' not in serialized


def test_better_nll_is_rejected_when_greedy_or_broad_retention_fails() -> None:
    requirements = _selection_requirements(_config())
    baseline = RuntimeArmEvidence(
        color_full_vocab_sides=12,
        color_full_vocab_units=6,
        mirror_full_vocab_sides=10,
        mirror_full_vocab_units=4,
        negative_sides=frozenset(),
        pair_margins=_pair_evidence(-0.1),
        generation=GenerationEvidence(24, 12, 0, 0, 0, 48, 24),
        prefix_sha256_by_scene={},
    )
    assert requirements.minimum_greedy_complete_units_correct == 1
    assert baseline.generation.broad_exact_accuracy == pytest.approx(0.5)
    # These are the two independent gates that a lower-NLL arm cannot bypass.
    failed_greedy = GenerationEvidence(24, 12, 0, 0, 0, 48, 24)
    failed_broad = GenerationEvidence(24, 12, 2, 1, 1, 48, 23)
    assert (
        failed_greedy.exact_complete_units_correct
        < requirements.minimum_greedy_complete_units_correct
    )
    assert failed_broad.broad_exact_accuracy < baseline.generation.broad_exact_accuracy
