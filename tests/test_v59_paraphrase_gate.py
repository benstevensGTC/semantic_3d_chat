from __future__ import annotations

import json
from pathlib import Path

from semantic_3d_chat.evaluation.v59_paraphrase_gate import prepare, score


def test_v59_paraphrases_are_locked_without_answers_in_runtime_manifest(
    tmp_path: Path,
) -> None:
    questions = tmp_path / "questions.json"
    preregistration = tmp_path / "preregistration.json"
    locked = prepare(
        questions_output=questions,
        preregistration_output=preregistration,
    )
    assert locked["training_inputs_permitted"] is False
    assert locked["thresholds"] == {
        "exact": 6,
        "complete_units": 3,
        "changed_units": 3,
    }
    assert '"answer"' not in questions.read_text(encoding="utf-8").casefold()

    manifest = json.loads(questions.read_text(encoding="utf-8"))
    references = {
        "q_900001": "above",
        "q_900002": "below",
        "q_900003": "top",
        "q_900004": "beneath",
        "q_900005": "lower",
        "q_900006": "higher",
        "q_900007": "book, cube",
        "q_900008": "cube",
    }
    predictions = tmp_path / "predictions.jsonl"
    with predictions.open("x", encoding="utf-8") as handle:
        for row in manifest["questions"]:
            handle.write(
                json.dumps(
                    {
                        "scene_id": row["scene_id"],
                        "question_id": row["question_id"],
                        "predicted_answer": references[row["question_id"]],
                        "control_audit": {
                            "architecture": "bounded_global_scene_question_control_v2",
                            "environment_latent_count": 256,
                            "every_scene_token_influenced_output": True,
                            "question_dependent_scene_retrieval": False,
                            "softmax_scene_attention_used": False,
                            "control_used": True,
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    result = score(
        predictions=predictions,
        preregistration=preregistration,
        output=tmp_path / "score.json",
    )
    assert result["passed"] is True
    assert result["metrics"]["exact"] == 8
