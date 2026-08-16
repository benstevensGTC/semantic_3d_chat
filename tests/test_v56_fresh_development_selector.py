from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import v56_fresh_development_selector as selector


def _terminal_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact": selector.TERMINAL_ARTIFACT,
        "passed": True,
        "terminal_materialization_authorized": True,
        "authorization": {
            "authorization_id": selector.AUTHORIZATION_ID,
            "only_exact_action": "one_control_one_shot_fresh_development",
            "explicit_terminal_sha256_required": True,
            "control_checkpoint": {
                "path": "control",
                "sha256": "a" * 64,
            },
            "training_report": {
                "path": "training.json",
                "sha256": "b" * 64,
            },
            "development": {
                "reference_sha256": "c" * 64,
                "questions": {"manifest_sha256": "d" * 64},
            },
        },
    }


def _redirect_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replacements = {
        "DEFAULT_TERMINAL": tmp_path / "terminal.json",
        "CLAIM_PATH": tmp_path / "claim.json",
        "MODEL_SNAPSHOT_PATH": tmp_path / "model_snapshot.json",
        "QUESTIONS_PATH": tmp_path / "questions.json",
        "REFERENCE_PATH": tmp_path / "references.jsonl",
        "PREDICTIONS_PATH": tmp_path / "predictions.jsonl",
        "PREDICTION_PROVENANCE_PATH": tmp_path / "predictions.jsonl.provenance.json",
        "SCORE_PATH": tmp_path / "score.json",
        "SELECTOR_REPORT_PATH": tmp_path / "selector.json",
        "RUNTIME_CONFIG": tmp_path / "runtime.yaml",
        "V54_CHECKPOINT": tmp_path / "base_checkpoint",
    }
    for name, value in replacements.items():
        monkeypatch.setattr(selector, name, value)


def test_require_terminal_needs_explicit_exact_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _redirect_paths(tmp_path, monkeypatch)
    terminal = Path(selector.DEFAULT_TERMINAL)
    terminal.write_text(json.dumps(_terminal_payload()), encoding="utf-8")
    digest = hashlib.sha256(terminal.read_bytes()).hexdigest()

    report, observed = selector.require_terminal(digest, terminal)

    assert report["artifact"] == selector.TERMINAL_ARTIFACT
    assert observed == digest
    with pytest.raises(ValueError, match="differs from the explicit"):
        selector.require_terminal("0" * 64, terminal)


def test_launch_claim_is_create_once_and_exactly_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _redirect_paths(tmp_path, monkeypatch)
    payload = _terminal_payload()
    digest = "e" * 64

    created = selector.create_or_resume_claim(payload, digest)
    resumed = selector.create_or_resume_claim(payload, digest)

    assert created["created"] is True
    assert resumed["created"] is False
    assert created["sha256"] == resumed["sha256"]
    claim_path = Path(selector.CLAIM_PATH)
    altered = json.loads(claim_path.read_text(encoding="utf-8"))
    altered["one_candidate_only"] = False
    claim_path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs"):
        selector.create_or_resume_claim(payload, digest)


def test_unclaimed_partial_output_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _redirect_paths(tmp_path, monkeypatch)
    Path(selector.PREDICTIONS_PATH).write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="without a permanent claim"):
        selector.create_or_resume_claim(_terminal_payload(), "f" * 64)


def _passing_score() -> dict[str, object]:
    gates = {
        "full_validation_coverage_216": True,
        "three_atomic_fresh_pairs_present": True,
        "normalized_exact_accuracy_at_least_0_42": True,
        "canonical_correct_at_least_93_of_216": True,
        "spatial_relation_accuracy_at_least_0_60": True,
        "count_accuracy_at_least_0_80": True,
        "presence_f1_at_least_0_30": True,
        "canonical_changed_complete_units_at_least_2_of_12": True,
        "canonical_changed_correct_sides_at_least_12_of_24": True,
        "canonical_prediction_changed_units_at_least_2_of_12": True,
        "physical_change_families_at_least_2_of_3": True,
    }
    return {
        "schema_version": 1,
        "artifact": selector.SCORE_ARTIFACT,
        "passed": True,
        "gates": gates,
        "standard_metrics": {
            "normalized_exact_accuracy": 0.5,
            "spatial_relation_accuracy": 0.65,
            "count": {"accuracy": 0.9},
            "presence": {"f1": 0.4},
        },
        "canonical_type_specific": {"correct": 108, "total": 216, "accuracy": 0.5},
        "changed_counterfactual": {
            "canonical_complete_units": 3,
            "canonical_correct_sides": 14,
            "canonical_prediction_changed_units": 3,
        },
        "grounding": {"mean_coordinate_error_m": 1.0},
    }


def test_selector_report_selects_control_but_does_not_claim_final_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _redirect_paths(tmp_path, monkeypatch)
    Path(selector.SCORE_PATH).write_text("{}\n", encoding="utf-8")
    terminal = _terminal_payload()
    prefix = {
        scene_id: hashlib.sha256(scene_id.encode()).hexdigest()
        for scene_id in selector.EXPECTED_SCENE_IDS
    }

    report = selector.build_selector_report(
        terminal,
        "1" * 64,
        {"sha256": "2" * 64},
        {"sha256": "3" * 64, "tree_sha256": "4" * 64},
        {
            "sha256": "5" * 64,
            "prefix_sha256_by_scene": prefix,
            "scene_map_manifest": {},
        },
        _passing_score(),
    )

    assert report["passed"] is True
    assert report["selected_control_checkpoint"] == "control"
    assert report["final_evaluation_authorized"] is True
    assert report["chat_promotion_eligible"] is False
    assert report["deferred_final_scenes_touched"] is False
    serialized = json.dumps(report, sort_keys=True)
    assert '"question":' not in serialized
    assert '"answer":' not in serialized
    assert '"predicted_answer":' not in serialized


def test_prediction_command_uses_question_control_and_no_final_scene(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _redirect_paths(tmp_path, monkeypatch)
    commands: list[tuple[str, ...]] = []

    selector._run_predictions(
        _terminal_payload(),
        lambda command: commands.append(tuple(command)),
    )

    assert len(commands) == 1
    command = commands[0]
    assert "semantic_3d_chat.evaluation.predict_question_control" in command
    assert "--control-checkpoint" in command
    assert "--base-checkpoint" in command
    assert "validation" in command
    assert not any(f"scene_{index:06d}" in " ".join(command) for index in range(25, 31))


def test_run_selector_claims_before_any_fresh_input_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _redirect_paths(tmp_path, monkeypatch)
    events: list[str] = []
    payload = _terminal_payload()
    monkeypatch.setattr(
        selector,
        "require_terminal",
        lambda digest: (payload, digest),
    )

    def claim(_terminal: object, digest: str) -> dict[str, object]:
        events.append("claim")
        return {"sha256": digest, "created": True}

    monkeypatch.setattr(selector, "create_or_resume_claim", claim)
    monkeypatch.setattr(selector, "_execution_lock", nullcontext)
    monkeypatch.setattr(
        selector,
        "authenticate_claimed_inputs",
        lambda _: events.append("authenticate"),
    )
    snapshots = iter(
        (
            {"sha256": "1" * 64},
            {"sha256": "1" * 64},
        )
    )
    monkeypatch.setattr(
        selector,
        "_authenticate_or_create_model_snapshot",
        lambda *_: events.append("snapshot") or next(snapshots),
    )
    monkeypatch.setattr(
        selector,
        "_run_predictions",
        lambda *_: events.append("model"),
    )
    monkeypatch.setattr(
        selector,
        "_validate_predictions",
        lambda _: events.append("predictions") or {"sha256": "2" * 64},
    )
    monkeypatch.setattr(
        selector,
        "_run_score",
        lambda *_: events.append("score"),
    )
    monkeypatch.setattr(
        selector,
        "_validate_score",
        lambda *_: _passing_score(),
    )
    monkeypatch.setattr(
        selector,
        "build_selector_report",
        lambda *_: {"artifact": selector.ARTIFACT, "passed": True},
    )
    monkeypatch.setattr(
        selector,
        "_atomic_create",
        lambda *_: events.append("sealed"),
    )

    report = selector.run_selector("6" * 64, runner=lambda _: None)

    assert report["passed"] is True
    assert events == [
        "claim",
        "authenticate",
        "snapshot",
        "model",
        "snapshot",
        "authenticate",
        "predictions",
        "score",
        "sealed",
    ]


def test_real_claim_exists_before_first_fresh_input_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _redirect_paths(tmp_path, monkeypatch)
    digest = "7" * 64
    payload = _terminal_payload()
    monkeypatch.setattr(
        selector,
        "require_terminal",
        lambda supplied: (payload, supplied),
    )

    class AuditStop(RuntimeError):
        pass

    def first_input(_terminal: object) -> None:
        claim = Path(selector.CLAIM_PATH)
        assert claim.is_file()
        assert json.loads(claim.read_text(encoding="utf-8"))["terminal_sha256"] == digest
        raise AuditStop("claim observed before fresh input")

    monkeypatch.setattr(selector, "authenticate_claimed_inputs", first_input)
    with pytest.raises(AuditStop, match="claim observed"):
        selector.run_selector(digest, runner=lambda _: None)

    assert Path(selector.CLAIM_PATH).is_file()
