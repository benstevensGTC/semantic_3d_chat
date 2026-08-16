from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import v55_development_selector as selector


def _terminal_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact": selector.TERMINAL_ARTIFACT,
        "passed": True,
        "terminal_materialization_authorized": True,
        "authorization": {
            "authorization_id": selector.AUTHORIZATION_ID,
            "only_exact_action": "one_candidate_one_shot_development_selection",
            "explicit_terminal_sha256_required": True,
        },
    }


def test_require_terminal_needs_explicit_exact_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal = tmp_path / "terminal.json"
    terminal.write_text(json.dumps(_terminal_payload()), encoding="utf-8")
    digest = hashlib.sha256(terminal.read_bytes()).hexdigest()
    monkeypatch.setattr(selector, "DEFAULT_TERMINAL", terminal)

    report, observed = selector.require_terminal(digest, terminal)

    assert report["artifact"] == selector.TERMINAL_ARTIFACT
    assert observed == digest
    with pytest.raises(ValueError, match="differs from the explicit"):
        selector.require_terminal("0" * 64, terminal)


def _redirect_one_shot_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replacements = {
        "DEFAULT_TERMINAL": tmp_path / "terminal.json",
        "CLAIM_PATH": tmp_path / "claim.json",
        "MODEL_SNAPSHOT_PATH": tmp_path / "model_snapshot.json",
        "QUESTIONS_PATH": tmp_path / "questions.json",
        "PREDICTIONS_PATH": tmp_path / "predictions.jsonl",
        "PREDICTION_PROVENANCE_PATH": tmp_path / "predictions.jsonl.provenance.json",
        "SCORE_PATH": tmp_path / "score.json",
        "SELECTOR_REPORT_PATH": tmp_path / "selector.json",
        "REFERENCE_PATH": tmp_path / "validation.jsonl",
        "RUNTIME_CONFIG": tmp_path / "runtime.yaml",
        "V54_CHECKPOINT": tmp_path / "checkpoint" / "update_000",
    }
    for name, value in replacements.items():
        monkeypatch.setattr(selector, name, value)


def test_launch_claim_is_create_once_and_exactly_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _redirect_one_shot_paths(tmp_path, monkeypatch)
    digest = "a" * 64

    created = selector.create_or_resume_claim(digest)
    resumed = selector.create_or_resume_claim(digest)

    assert created["created"] is True
    assert resumed["created"] is False
    assert created["sha256"] == resumed["sha256"]
    claim_path = Path(selector.CLAIM_PATH)
    altered = json.loads(claim_path.read_text(encoding="utf-8"))
    altered["one_candidate_only"] = False
    claim_path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs"):
        selector.create_or_resume_claim(digest)


def test_unclaimed_partial_output_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _redirect_one_shot_paths(tmp_path, monkeypatch)
    Path(selector.PREDICTIONS_PATH).write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="without a permanent claim"):
        selector.create_or_resume_claim("b" * 64)


def _passing_score() -> dict[str, object]:
    gates = {
        "full_validation_coverage_216": True,
        "normalized_exact_accuracy_at_least_0_375": True,
        "spatial_relation_accuracy_at_least_0_55": True,
        "count_accuracy_at_least_0_80": True,
        "presence_f1_at_least_0_15": True,
        "canonical_changed_complete_units_at_least_2_of_12": True,
        "canonical_changed_correct_sides_at_least_12_of_24": True,
        "canonical_prediction_changed_units_at_least_2_of_12": True,
        "physical_change_families_at_least_2_of_3": True,
        "canonical_aggregate_correct_at_least_v29_91_of_216": True,
    }
    return {
        "schema_version": 1,
        "artifact": selector.SCORE_ARTIFACT,
        "passed": True,
        "gates": gates,
        "standard_metrics": {
            "normalized_exact_accuracy": 0.5,
            "spatial_relation_accuracy": 0.6,
            "count": {"accuracy": 0.9},
            "presence": {"f1": 0.2},
        },
        "canonical_type_specific": {"correct": 100, "total": 216, "accuracy": 100 / 216},
        "changed_counterfactual": {
            "canonical_complete_units": 3,
            "canonical_correct_sides": 14,
            "canonical_prediction_changed_units": 3,
        },
    }


def test_selector_report_has_exact_promotion_envelope_without_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _redirect_one_shot_paths(tmp_path, monkeypatch)
    Path(selector.SCORE_PATH).write_text("{}\n", encoding="utf-8")
    prefix = {scene_id: hashlib.sha256(scene_id.encode()).hexdigest() for scene_id in selector.EXPECTED_SCENES}
    report = selector.build_selector_report(
        "c" * 64,
        {"sha256": "d" * 64},
        {"sha256": "0" * 64, "tree_sha256": "1" * 64},
        {"path": "opaque", "sha256": "e" * 64},
        {
            "path": "opaque",
            "sha256": "f" * 64,
            "prefix_sha256_by_scene": prefix,
            "scene_map_manifest": {
                scene_id: {
                    "voxel_map_sha256": "0" * 64,
                    "voxel_map_size_bytes": 1,
                }
                for scene_id in selector.EXPECTED_SCENES
            },
        },
        _passing_score(),
    )

    assert report["passed"] is True
    assert report["development_selection_passed"] is True
    assert report["chat_promotion_eligible"] is True
    assert report["selected_update"] == 0
    assert report["selected_optimizer_step"] == 0
    assert report["final_test_scenes_touched"] is False
    assert set(report["chat_promotion"]["checks"]) == {
        "development_checkpoint_selected",
        "changed_complete_pair_threshold_met",
        "aggregate_validation_exact_accuracy_retained",
    }
    serialized = json.dumps(report, sort_keys=True)
    assert '"question":' not in serialized
    assert '"answer":' not in serialized
    assert '"predicted_answer":' not in serialized


def test_run_selector_claims_before_any_claimed_input_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _redirect_one_shot_paths(tmp_path, monkeypatch)
    events: list[str] = []
    monkeypatch.setattr(
        selector,
        "require_terminal",
        lambda digest: (_terminal_payload(), digest),
    )

    def claim(digest: str) -> dict[str, object]:
        events.append("claim")
        return {"sha256": digest, "created": True}

    monkeypatch.setattr(selector, "create_or_resume_claim", claim)
    monkeypatch.setattr(selector, "_execution_lock", nullcontext)
    monkeypatch.setattr(
        selector,
        "authenticate_claimed_inputs",
        lambda terminal: events.append("authenticate"),
    )
    monkeypatch.setattr(
        selector,
        "_authenticate_or_create_model_snapshot",
        lambda digest: events.append("snapshot") or {"sha256": "4" * 64},
    )
    monkeypatch.setattr(
        selector,
        "_prepare_questions",
        lambda runner: events.append("qa") or {"sha256": "1" * 64},
    )
    monkeypatch.setattr(
        selector,
        "_run_predictions",
        lambda runner: events.append("model"),
    )
    monkeypatch.setattr(
        selector,
        "_validate_predictions",
        lambda: events.append("maps") or {"sha256": "2" * 64},
    )
    monkeypatch.setattr(selector, "_run_score", lambda runner: events.append("score"))
    monkeypatch.setattr(selector, "_validate_score", lambda evidence: _passing_score())
    monkeypatch.setattr(
        selector,
        "build_selector_report",
        lambda *args: {"artifact": selector.ARTIFACT, "passed": True},
    )
    monkeypatch.setattr(
        selector,
        "_atomic_create",
        lambda path, value: events.append("sealed"),
    )

    report = selector.run_selector("3" * 64, runner=lambda command: None)

    assert report["passed"] is True
    assert events == [
        "claim",
        "authenticate",
        "snapshot",
        "qa",
        "model",
        "snapshot",
        "maps",
        "score",
        "sealed",
    ]


def test_real_claim_file_exists_before_first_claimed_input_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _redirect_one_shot_paths(tmp_path, monkeypatch)
    digest = "5" * 64
    monkeypatch.setattr(
        selector,
        "require_terminal",
        lambda supplied: (_terminal_payload(), supplied),
    )

    class AuditStop(RuntimeError):
        pass

    def first_claimed_input(_terminal: object) -> None:
        claim = Path(selector.CLAIM_PATH)
        assert claim.is_file()
        payload = json.loads(claim.read_text(encoding="utf-8"))
        assert payload["terminal_sha256"] == digest
        raise AuditStop("claim observed before claimed input")

    monkeypatch.setattr(selector, "authenticate_claimed_inputs", first_claimed_input)
    with pytest.raises(AuditStop, match="claim observed"):
        selector.run_selector(digest, runner=lambda command: None)

    assert Path(selector.CLAIM_PATH).is_file()
