from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import v61_route_generalization_gate as v61


def _baseline_answer(row: dict[str, object]) -> str:
    return f"private-baseline-output::{row['scene_id']}::{row['question_id']}"


def _write_baseline_predictions(
    path: Path,
    *,
    rows: tuple[dict[str, object], ...] = v61._ROWS,
) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "scene_id": row["scene_id"],
                        "question_id": row["question_id"],
                        "predicted_answer": _baseline_answer(row),
                        "prefix_hash": hashlib.sha256(str(row["scene_id"]).encode()).hexdigest(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def _write_candidate_predictions(
    path: Path,
    *,
    rows: tuple[dict[str, object], ...] = v61._ROWS,
    correct_routes: bool = True,
) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            route_expected = bool(row["route_expected"])
            control_used = route_expected if correct_routes else False
            answer = row["answer"] if route_expected else _baseline_answer(row)
            handle.write(
                json.dumps(
                    {
                        "scene_id": row["scene_id"],
                        "question_id": row["question_id"],
                        "predicted_answer": answer,
                        "control_audit": {
                            "architecture": ("scene_conditioned_gate_teacher_basis_control_v4"),
                            "environment_latent_count": 256,
                            "every_scene_token_influenced_output": True,
                            "question_dependent_scene_retrieval": False,
                            "softmax_scene_attention_used": False,
                            "control_used": control_used,
                            "exact_no_control_route": not control_used,
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def _prepare_and_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    questions = tmp_path / "questions.json"
    preregistration = tmp_path / "preregistration.json"
    v61.prepare(
        questions_output=questions,
        preregistration_output=preregistration,
    )
    predictions = tmp_path / "baseline.jsonl"
    _write_baseline_predictions(predictions)
    predictions.with_suffix(".jsonl.provenance.json").write_text(
        '{"synthetic": true}\n', encoding="utf-8"
    )
    monkeypatch.setattr(
        v61,
        "checkpoint_fingerprint",
        lambda _path: (
            "c" * 64,
            [
                {
                    "path": "adapter.safetensors",
                    "sha256": "d" * 64,
                    "size_bytes": 1,
                }
            ],
        ),
    )
    baseline = tmp_path / "baseline-lock.json"
    v61.lock_baseline(
        predictions=predictions,
        preregistration=preregistration,
        base_checkpoint=tmp_path / "synthetic-checkpoint-not-read",
        output=baseline,
    )
    return questions, preregistration, baseline


def test_prepare_emits_strict_questions_only_manifest(tmp_path: Path) -> None:
    questions = tmp_path / "questions.json"
    preregistration = tmp_path / "preregistration.json"

    locked = v61.prepare(
        questions_output=questions,
        preregistration_output=preregistration,
    )

    manifest = json.loads(questions.read_text(encoding="utf-8"))
    assert set(manifest) == {
        "schema",
        "schema_version",
        "question_count",
        "scene_count",
        "questions_sha256",
        "source_qa_sha256",
        "questions",
    }
    assert manifest["question_count"] == len(v61._ROWS) == 66
    assert manifest["scene_count"] == len(v61._SCENES) == 6
    assert all(
        set(record) == {"scene_id", "question_id", "question"} for record in manifest["questions"]
    )
    prohibited_fields = {
        "answer",
        "answer_type",
        "route_expected",
        "pair_group",
        "family_id",
    }
    assert all(not (set(record) & prohibited_fields) for record in manifest["questions"])
    assert locked["training_inputs_permitted"] is False


def test_baseline_lock_stores_only_answer_hashes_and_rejects_inventory_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _questions, preregistration, baseline_path = _prepare_and_lock(tmp_path, monkeypatch)

    lock = json.loads(baseline_path.read_text(encoding="utf-8"))
    serialized_lock = baseline_path.read_text(encoding="utf-8")
    assert lock["environmental_answer_text_stored"] is False
    assert lock["question_count"] == len(v61._ROWS)
    assert len(lock["required_output_hashes"]) == len(v61._ROWS)
    assert all(
        set(record) == {"scene_id", "question_id", "raw_output_sha256"}
        for record in lock["required_output_hashes"]
    )
    for row in v61._ROWS:
        assert _baseline_answer(row) not in serialized_lock
    expected_hashes = {
        (row["scene_id"], row["question_id"]): hashlib.sha256(
            _baseline_answer(row).encode()
        ).hexdigest()
        for row in v61._ROWS
    }
    assert {
        (record["scene_id"], record["question_id"]): record["raw_output_sha256"]
        for record in lock["required_output_hashes"]
    } == expected_hashes

    drifted_predictions = tmp_path / "drifted-baseline.jsonl"
    _write_baseline_predictions(drifted_predictions, rows=v61._ROWS[:-1])
    with pytest.raises(ValueError, match="inventory differs"):
        v61.lock_baseline(
            predictions=drifted_predictions,
            preregistration=preregistration,
            base_checkpoint=tmp_path / "synthetic-checkpoint-not-read",
            output=tmp_path / "drifted-lock.json",
        )


def test_score_enforces_thresholds_audits_and_locked_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _questions, preregistration, baseline = _prepare_and_lock(tmp_path, monkeypatch)

    perfect_predictions = tmp_path / "perfect.jsonl"
    _write_candidate_predictions(perfect_predictions)
    perfect = v61.score(
        predictions=perfect_predictions,
        preregistration=preregistration,
        baseline=baseline,
        output=tmp_path / "perfect-score.json",
    )
    assert perfect["passed"] is True
    assert all(perfect["checks"].values())
    assert perfect["metrics"] == {
        "route_positive": 22,
        "route_positive_total": 22,
        "route_negative": 44,
        "route_negative_total": 44,
        "route_positive_by_pair": {"anchor": 8, "mirror": 8, "removal": 6},
        "contradictory_complete_families": 11,
        "family_total": 11,
        "positive_answer_exact": 22,
        "positive_answer_total": 22,
        "positive_complete_families": 11,
        "positive_changed_families": 11,
        "negative_exact_v54_output_identity": 44,
        "negative_total": 44,
    }

    route_failure_predictions = tmp_path / "route-failure.jsonl"
    _write_candidate_predictions(route_failure_predictions, correct_routes=False)
    route_failure = v61.score(
        predictions=route_failure_predictions,
        preregistration=preregistration,
        baseline=baseline,
        output=tmp_path / "route-failure-score.json",
    )
    assert route_failure["passed"] is False
    assert route_failure["metrics"]["route_positive"] == 0
    assert route_failure["checks"]["route_positive"] is False
    assert route_failure["checks"]["route_anchor_positive"] is False
    assert route_failure["checks"]["route_mirror_positive"] is False
    assert route_failure["checks"]["route_removal_positive"] is False
    assert route_failure["checks"]["continuous_global_v4_audits_valid"] is False

    drifted_predictions = tmp_path / "drifted-candidate.jsonl"
    _write_candidate_predictions(drifted_predictions, rows=v61._ROWS[:-1])
    with pytest.raises(ValueError, match="inventory differs"):
        v61.score(
            predictions=drifted_predictions,
            preregistration=preregistration,
            baseline=baseline,
            output=tmp_path / "drifted-score.json",
        )


def test_cli_main_returns_two_when_score_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    received: dict[str, object] = {}

    def fake_score(**kwargs: object) -> dict[str, object]:
        received.update(kwargs)
        return {"artifact": "synthetic-v61-score", "passed": False}

    monkeypatch.setattr(v61, "score", fake_score)
    result = v61.main(
        [
            "score",
            "--predictions",
            str(tmp_path / "predictions.jsonl"),
            "--preregistration",
            str(tmp_path / "preregistration.json"),
            "--baseline",
            str(tmp_path / "baseline.json"),
            "--output",
            str(tmp_path / "score.json"),
        ]
    )

    assert result == 2
    assert received == {
        "predictions": str(tmp_path / "predictions.jsonl"),
        "preregistration": str(tmp_path / "preregistration.json"),
        "baseline": str(tmp_path / "baseline.json"),
        "output": str(tmp_path / "score.json"),
    }
    assert '"passed": false' in capsys.readouterr().out
