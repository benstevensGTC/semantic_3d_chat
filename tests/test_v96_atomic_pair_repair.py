from __future__ import annotations

import builtins
import copy
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from semantic_3d_chat.evaluation import v96_atomic_pair_repair_preflight as v96_preflight
from semantic_3d_chat.evaluation.metrics import normalize_answer_items
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import canonical_sha256_v85
from semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight import (
    DEFERRED_FINAL_SCENES,
    EXPECTED_INITIAL_STATE_SHA256,
    EXPECTED_TOTAL_ADAPTER_PARAMETER_COUNT,
    FRESH_PARAMETER_COUNT,
    PRIOR_EVALUATION_SCENES,
    TARGET_MODULES,
    TRAINING_SCENES,
    assert_initial_outputs_absent_v96,
    authenticate_parent_v95_v96,
    derive_contract_v96,
    derive_preregistration_v96,
    forbidden_training_roots_v96,
    invariant_subset_v96,
    load_config_v96,
    load_training_rows_v96,
    lora_preflight_v96,
    pair_units_v96,
    training_schedule_v96,
)


def test_v96_raw_answers_round_trip_canonical_lists() -> None:
    rows = load_training_rows_v96(load_config_v96())
    lists = [row for row in rows if row.answer == "book, cube"]

    assert len(lists) == 22
    assert not any(row.answer == "book cube" for row in rows)
    assert all(normalize_answer_items(row.answer) == ("book", "cube") for row in lists)
    assert all(row.answer.encode("utf-8").decode("utf-8") == row.answer for row in rows)


def test_v96_exact_atomic_units_and_byte_identical_questions() -> None:
    rows = load_training_rows_v96(load_config_v96())
    changed, invariant = pair_units_v96(rows)

    assert len(rows) == 960
    assert tuple(sorted({row.scene_id for row in rows})) == TRAINING_SCENES
    assert not set(PRIOR_EVALUATION_SCENES).intersection(row.scene_id for row in rows)
    assert not set(DEFERRED_FINAL_SCENES).intersection(row.scene_id for row in rows)
    assert len(changed) == 66
    assert len(invariant) == 414
    assert all(unit.left.paired_scene_id == unit.right.scene_id for unit in changed)
    assert all(unit.right.paired_scene_id == unit.left.scene_id for unit in changed)
    assert all(
        unit.left.question.encode("utf-8") == unit.right.question.encode("utf-8")
        for unit in (*changed, *invariant)
    )
    assert all(unit.left.answer_class != unit.right.answer_class for unit in changed)
    assert all(unit.left.answer == unit.right.answer for unit in invariant)


def test_v96_invariant_subset_is_answer_independent_and_balanced() -> None:
    rows = load_training_rows_v96(load_config_v96())
    selected = invariant_subset_v96(rows)
    selected_keys = {unit.key for unit in selected}
    mutated = tuple(
        replace(row, answer="ignored canonical value", answer_class="ignored_class")
        if not row.expected_change
        else row
        for row in rows
    )
    mutated_keys = {unit.key for unit in invariant_subset_v96(mutated)}

    assert selected_keys == mutated_keys
    assert len(selected) == 96
    assert Counter(unit.answer_type for unit in selected) == Counter(
        {
            "attribute": 14,
            "count": 14,
            "metric": 13,
            "orientation": 13,
            "presence": 14,
            "spatial_relation": 14,
            "support": 14,
        }
    )
    assert len({unit.change_type for unit in selected}) == 9


def test_v96_schedule_has_exact_exposures_and_budget() -> None:
    config = load_config_v96()
    rows = load_training_rows_v96(config)
    changed, _invariant = pair_units_v96(rows)
    stable = invariant_subset_v96(rows)
    schedule = training_schedule_v96(rows)

    assert len(schedule) == 2_280
    assert Counter(step.kind for step in schedule) == Counter(
        {"retention": 1_920, "changed_pair": 264, "invariant_pair": 96}
    )
    assert Counter(step.row.key for step in schedule if step.row is not None) == Counter(
        {row.key: 2 for row in rows}
    )
    assert Counter(step.unit.key for step in schedule if step.kind == "changed_pair") == Counter(
        {unit.key: 4 for unit in changed}
    )
    assert Counter(step.unit.key for step in schedule if step.kind == "invariant_pair") == Counter(
        {unit.key: 1 for unit in stable}
    )
    assert (
        canonical_sha256_v85([step.identity() for step in schedule])
        == config["training"]["schedule_sha256"]
    )
    assert 1_920 + 264 * 4 + 96 * 2 == 3_168
    assert 2_280 // 8 == 285


def test_v96_q_projection_bank_is_exact_zero_and_correct_real_shape() -> None:
    config = load_config_v96()
    preflight = lora_preflight_v96(config)

    assert TARGET_MODULES == ("model.language_model.layers.9.self_attn.q_proj",)
    assert config["bridge"]["pinned_weight_shapes"] == {TARGET_MODULES[0] + ".weight": [4096, 1536]}
    assert preflight["parameter_count"] == FRESH_PARAMETER_COUNT == 45_056
    assert preflight["initial_state_sha256"] == EXPECTED_INITIAL_STATE_SHA256
    assert preflight["adapter_shapes"] == [{"lora_a": [8, 1536], "lora_b": [4096, 8]}]
    assert preflight["exact_zero_output_at_initialization"] is True
    assert EXPECTED_TOTAL_ADAPTER_PARAMETER_COUNT == 864_256


@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "broken_symlink"])
def test_v96_initial_output_gate_rejects_every_leaf_type(tmp_path: Path, kind: str) -> None:
    config = copy.deepcopy(load_config_v96())
    candidate = tmp_path / "candidate"
    config["outputs"] = {
        **config["outputs"],
        "work_root": str(tmp_path / "work"),
        "fixed_final_candidate": str(candidate),
        "training_report": str(tmp_path / "report.json"),
    }
    if kind == "file":
        candidate.write_text("occupied", encoding="utf-8")
    elif kind == "directory":
        candidate.mkdir()
    elif kind == "symlink":
        target = tmp_path / "target"
        target.write_text("occupied", encoding="utf-8")
        candidate.symlink_to(target)
    else:
        candidate.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError, match="initial output already exists"):
        assert_initial_outputs_absent_v96(config)


def _isolate_v96_initial_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> dict[str, object]:
    """Route pre-training derivation assertions away from live resume state."""

    original = v96_preflight.load_config_v96
    isolated_outputs = {
        "work_root": str(tmp_path / "work"),
        "fixed_final_candidate": str(tmp_path / "candidate"),
        "training_report": str(tmp_path / "training.json"),
    }

    def isolated_config(
        path: str | Path = v96_preflight.CONFIG,
        *,
        allow_draft: bool = True,
    ) -> dict[str, object]:
        config = copy.deepcopy(original(path, allow_draft=allow_draft))
        outputs = config.get("outputs")
        if not isinstance(outputs, dict):
            raise TypeError("V96 test config lost its output mapping")
        outputs.update(isolated_outputs)
        return config

    monkeypatch.setattr(v96_preflight, "load_config_v96", isolated_config)
    return isolated_outputs


def test_v96_derive_is_read_blocked_from_known_dev_and_deferred(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    isolated_outputs = _isolate_v96_initial_outputs(monkeypatch, tmp_path)
    config = v96_preflight.load_config_v96()
    forbidden = set(forbidden_training_roots_v96(config))
    original_open = builtins.open
    original_path_open = Path.open

    def rejects(path: object) -> bool:
        candidate = Path(path).resolve()  # type: ignore[arg-type]
        return any(candidate == root or root in candidate.parents for root in forbidden)

    def guarded_open(file: object, *args: object, **kwargs: object) -> object:
        if rejects(file):
            raise AssertionError(f"V96 opened protected source: {file}")
        return original_open(file, *args, **kwargs)  # type: ignore[arg-type]

    def guarded_path_open(path: Path, *args: object, **kwargs: object) -> object:
        if rejects(path):
            raise AssertionError(f"V96 opened protected source: {path}")
        return original_path_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    result = derive_contract_v96()

    assert result["file_audit_forbidden_reads"] == []
    assert result["known_development_labels_opened"] is False
    assert result["known_development_questions_opened"] is False
    assert result["deferred_final_artifacts_generated"] is False
    assert result["all_changed_pair_questions_byte_identical"] is True
    assert result["all_invariant_subset_questions_byte_identical"] is True
    assert result["initial_output_absence"]["work_root_absent"] is True
    assert result["initial_output_absence"]["checked_paths"] == {
        key: str(Path(value).resolve()) for key, value in isolated_outputs.items()
    }
    assert result["frozen_parent"]["v95_known_development_gate_passed"] is False
    assert result["full_gemma_model_loaded"] is False


def test_v96_preregistration_derivation_is_nonmutating_and_fixed_final(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    isolated_outputs = _isolate_v96_initial_outputs(monkeypatch, tmp_path)
    result = derive_preregistration_v96()

    assert result["training_authorized"] is False
    assert result["parent_authenticated"] is True
    assert result["known_development_protocol"]["v96_correct_minimum"] == 160
    assert result["known_development_protocol"]["changed_side_correct_minimum"] == 15
    assert result["known_development_protocol"]["complete_changed_units_minimum"] == 4
    assert result["known_development_protocol"]["prediction_changed_units_minimum"] == 7
    assert result["known_development_protocol"]["invariant_false_change_maximum"] == 20
    assert result["training_protocol"]["checkpoint_selection"].startswith("fixed_final_update_285")
    assert all(not Path(path).exists() for path in isolated_outputs.values())


def test_v96_pretraining_derivation_tests_do_not_touch_live_resume_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    live_work_root = Path(load_config_v96()["outputs"]["work_root"])
    live_inventory = sorted(path.name for path in live_work_root.glob("update_*"))
    _isolate_v96_initial_outputs(monkeypatch, tmp_path)

    result = derive_contract_v96()

    assert result["initial_output_absence"]["work_root_absent"] is True
    assert sorted(path.name for path in live_work_root.glob("update_*")) == live_inventory


def test_v96_config_contains_no_row_level_known_development_content() -> None:
    serialized = json.dumps(load_config_v96()["known_development_gate"], sort_keys=True)
    assert '"question"' not in serialized
    assert '"answer"' not in serialized


@pytest.mark.parametrize(
    "artifact",
    [
        "gemma4_v95_strict_causal_successor_training_v1",
        "gemma4_v95_strict_causal_successor_fixed_final_v1",
        "gemma4_v95_known_development_structured_score_v1",
        "gemma4_v95_known_development_gate_v1",
        "gemma4_v95_known_development_evidence_v1",
        "gemma4_v95_known_development_nll_aggregate_v1",
    ],
)
def test_v96_parent_chain_rejects_cross_artifact_hash_tamper(
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    original = v96_preflight._strict_json

    def tampered(path: str | Path) -> dict[str, object]:
        value = copy.deepcopy(original(path))
        if value.get("artifact") == artifact:
            if artifact == "gemma4_v95_strict_causal_successor_training_v1":
                value["candidate"]["metadata_canonical_sha256"] = "0" * 64  # type: ignore[index]
            elif artifact == "gemma4_v95_strict_causal_successor_fixed_final_v1":
                value["weights_sha256"] = "0" * 64
            elif artifact == "gemma4_v95_known_development_structured_score_v1":
                value["candidate_fingerprint_sha256"] = "0" * 64
            elif artifact == "gemma4_v95_known_development_gate_v1":
                value["nll_sha256"] = "0" * 64
            elif artifact == "gemma4_v95_known_development_evidence_v1":
                value["structured_score_sha256"] = "0" * 64
            else:
                value["candidate_fingerprint_sha256"] = "0" * 64
        return value

    monkeypatch.setattr(v96_preflight, "_strict_json", tampered)
    with pytest.raises(ValueError, match="V96"):
        authenticate_parent_v95_v96(load_config_v96())


def test_v96_parent_chain_rejects_current_v95_source_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config_v96()
    target = Path("src/semantic_3d_chat/training/train_v95_strict_causal_successor.py")
    original = v96_preflight.sha256_file_v85

    def tampered(path: str | Path) -> str:
        candidate = Path(path)
        if candidate.as_posix().endswith(target.as_posix()):
            return "0" * 64
        return original(path)

    monkeypatch.setattr(v96_preflight, "sha256_file_v85", tampered)
    with pytest.raises(ValueError, match="trainer_source chain changed"):
        authenticate_parent_v95_v96(config)


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("optimizer", "SGD"),
        ("learning_rate", 0.0001),
        ("weight_decay", 0.01),
        ("gradient_clip_norm", 2.0),
        ("retention_balanced_ce_weight", 0.5),
        ("balanced_ce_formula", "uniform"),
        ("balanced_ce_mean_over_rows", 2.0),
        ("changed_family_weight_formula", "uniform"),
        ("changed_family_weight_mean_over_units", 2.0),
        ("pair_correct_ce_weight", 0.5),
        ("within_memory_answer_margin_weight", 1.0),
        ("within_memory_answer_target_margin_nll", 0.5),
        ("across_memory_causal_margin_weight", 0.5),
        ("across_memory_causal_target_margin_nll", 0.25),
        ("pair_side_smoothmax_temperature", 0.5),
        ("invariant_family_weight_formula", "uniform"),
        ("invariant_family_weight_mean_over_units", 2.0),
        ("invariant_correct_ce_weight", 0.5),
        ("invariant_nll_consistency_weight", 0.25),
        ("invariant_nll_consistency_tolerance", 0.2),
        ("schedule_policy", "changed"),
        ("invariant_subset_policy", "changed"),
        ("controls_policy", "changed"),
        ("checkpoint_every_optimizer_updates", 30),
        ("checkpoint_selection", "best_validation_checkpoint"),
    ],
)
def test_v96_config_exact_pins_every_objective_field(
    tmp_path: Path,
    field: str,
    changed_value: object,
) -> None:
    config = copy.deepcopy(load_config_v96())
    config["training"][field] = changed_value
    path = tmp_path / f"changed_{field}.yaml"
    path.write_text(yaml.safe_dump({"v96": config}, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"V96 training {field} changed"):
        load_config_v96(path)
