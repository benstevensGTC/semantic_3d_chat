from __future__ import annotations

import json
from pathlib import Path

from semantic_3d_chat.evaluation.v59_multiscene_train_gate import (
    LOCKED_SCENE_IDS,
    _read_train_qa,
    evaluate_candidate,
    locked_gate_rows,
    prepare_gate,
    seal_baseline,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_QA = PROJECT_ROOT / "data_diverse52/qa/train.jsonl"


def _write_predictions(
    path: Path,
    groups: dict[str, list[dict[str, object]]],
    *,
    correct_groups: set[str],
    v2_audits: bool = False,
) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for group_name, rows in groups.items():
            for row in rows:
                value = {
                    "scene_id": row["scene_id"],
                    "question_id": row["question_id"],
                    "predicted_answer": (
                        row["answer"] if group_name in correct_groups else "unknown"
                    ),
                }
                if v2_audits:
                    value["control_audit"] = {
                        "architecture": "bounded_global_scene_question_control_v2",
                        "environment_latent_count": 256,
                        "every_scene_token_influenced_output": True,
                        "question_dependent_scene_retrieval": False,
                        "softmax_scene_attention_used": False,
                        "maximum_control_rms": 0.1,
                        "control_used": group_name != "retention",
                        "exact_no_control_route": group_name == "retention",
                    }
                handle.write(json.dumps(value, sort_keys=True) + "\n")


def test_v59_inventory_is_train_only_and_locks_complete_anchor_retention() -> None:
    rows, _digest = _read_train_qa(TRAIN_QA)
    groups = locked_gate_rows(rows)
    assert set(LOCKED_SCENE_IDS) == {
        row["scene_id"] for values in groups.values() for row in values
    }
    assert {name: len(values) for name, values in groups.items()} == {
        "anchor_changed": 8,
        "expansion_changed": 14,
        "retention": 40,
    }
    assert {row["scene_id"] for row in groups["retention"]} == {
        "scene_000031",
        "scene_000032",
    }
    assert not ({*range(25, 31), *range(57, 63)} & {
        int(str(row["scene_id"]).removeprefix("scene_"))
        for values in groups.values()
        for row in values
    })


def test_v59_gate_uses_source_changed_baseline_and_no_control_retention(
    tmp_path: Path,
) -> None:
    rows, _digest = _read_train_qa(TRAIN_QA)
    groups = locked_gate_rows(rows)
    questions = tmp_path / "questions.json"
    preregistration = tmp_path / "preregistration.json"
    prepared = prepare_gate(
        train_qa=TRAIN_QA,
        questions_output=questions,
        preregistration_output=preregistration,
    )
    assert prepared["counts"] == {
        "anchor_changed": 8,
        "expansion_changed": 14,
        "retention": 40,
    }
    serialized_questions = questions.read_text(encoding="utf-8").casefold()
    assert '"answer"' not in serialized_questions

    source = tmp_path / "source.jsonl"
    no_control = tmp_path / "no_control.jsonl"
    _write_predictions(source, groups, correct_groups=set())
    _write_predictions(no_control, groups, correct_groups={"retention"})
    baseline_path = tmp_path / "baseline.json"
    baseline = seal_baseline(
        train_qa=TRAIN_QA,
        source_predictions=source,
        no_control_predictions=no_control,
        preregistration=preregistration,
        output=baseline_path,
    )
    assert baseline["thresholds"]["retention_exact"] == 40
    assert baseline["thresholds"]["expansion_exact"] == 9

    candidate = tmp_path / "candidate.jsonl"
    _write_predictions(candidate, groups, correct_groups=set(groups), v2_audits=True)
    score = evaluate_candidate(
        train_qa=TRAIN_QA,
        predictions=candidate,
        preregistration=preregistration,
        baseline=baseline_path,
        output=tmp_path / "score.json",
    )
    assert score["passed"] is True
    assert all(score["checks"].values())
