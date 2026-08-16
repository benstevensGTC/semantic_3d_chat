from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation import v65_training_baseline_lock as lock
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.training import train_question_control_v65 as v65

QUESTIONS = PROJECT_ROOT / "reports/gemma4/questions/v65_training_natural.json"
PREDICTIONS = PROJECT_ROOT / "reports/gemma4/predictions/v65_v54_training_natural.jsonl"
PROVENANCE = PREDICTIONS.with_suffix(PREDICTIONS.suffix + ".provenance.json")
CHECKPOINT = PROJECT_ROOT / "data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000"
COMPLETED_LOCK = PROJECT_ROOT / "reports/gemma4/metrics/v65_v54_training_baseline_lock.json"


def _require_completed_artifacts() -> None:
    if not all(path.exists() for path in (QUESTIONS, PREDICTIONS, PROVENANCE, CHECKPOINT)):
        pytest.skip("completed local V65 baseline artifacts are unavailable")


def test_completed_sources_have_exact_pinned_training_inventory() -> None:
    _require_completed_artifacts()

    keys = lock._question_inventory(QUESTIONS)
    scene_counts = {
        scene_id: sum(item_scene == scene_id for item_scene, _question_id in keys)
        for scene_id in {scene_id for scene_id, _question_id in keys}
    }

    assert len(keys) == 576
    assert len(scene_counts) == 24
    assert set(scene_counts.values()) == {24}
    assert lock._sha256_file(PREDICTIONS) == v65._PINNED_V54_TRAINING_PREDICTIONS_SHA256
    assert lock._sha256_file(PROVENANCE) == v65._PINNED_V54_TRAINING_PROVENANCE_SHA256


def test_builder_is_deterministic_create_once_and_hash_only(tmp_path: Path) -> None:
    _require_completed_artifacts()
    output = tmp_path / "lock.json"

    payload = lock.build_training_baseline_lock(
        questions=QUESTIONS,
        predictions=PREDICTIONS,
        v54_checkpoint=CHECKPOINT,
        output=output,
    )

    assert lock._sha256_file(output) == v65._PINNED_V65_TRAINING_BASELINE_LOCK_SHA256
    assert payload["question_count"] == 576
    assert payload["scene_count"] == 24
    assert len(payload["scene_prefix_hashes"]) == 24
    assert len(set(payload["scene_prefix_hashes"].values())) == 24
    assert len(payload["required_output_hashes"]) == 576
    assert all(
        set(record) == {"scene_id", "question_id", "raw_output_sha256"}
        for record in payload["required_output_hashes"]
    )
    assert payload["answer_or_question_text_stored"] is False
    serialized = json.dumps(payload, sort_keys=True)
    assert "predicted_answer" not in serialized
    assert '"question"' not in serialized
    with pytest.raises(FileExistsError, match="create-once"):
        lock.build_training_baseline_lock(
            questions=QUESTIONS,
            predictions=PREDICTIONS,
            v54_checkpoint=CHECKPOINT,
            output=output,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scene_map_manifest_sha256", "0" * 64),
        ("v54_runtime_config_effective_sha256", "1" * 64),
        ("answer_or_question_text_stored", True),
    ],
)
def test_validator_rejects_any_rewritten_lock(tmp_path: Path, field: str, value: object) -> None:
    if not COMPLETED_LOCK.is_file():
        pytest.skip("completed local V65 baseline lock is unavailable")
    payload = json.loads(COMPLETED_LOCK.read_text(encoding="utf-8"))
    payload[field] = value
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="create-once pin"):
        v65.validate_training_baseline_lock(changed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("split", "validation"),
        ("config_sha256", "2" * 64),
        ("scene_map_manifest_sha256", "3" * 64),
    ],
)
def test_provenance_requires_exact_v54_runtime_identity(field: str, value: object) -> None:
    _require_completed_artifacts()
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    provenance[field] = value
    checkpoint_sha256, checkpoint_files = checkpoint_fingerprint(CHECKPOINT)

    with pytest.raises(ValueError, match="not bound"):
        lock._validate_prediction_provenance(
            provenance,
            question_path=QUESTIONS.resolve(),
            checkpoint_path=CHECKPOINT.resolve(),
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_files=checkpoint_files,
            expected_scenes={
                scene_id for scene_id, _question_id in lock._question_inventory(QUESTIONS)
            },
        )


@pytest.mark.parametrize("mutation", ["provenance", "order", "field"])
def test_prediction_inventory_is_exact_and_row_bound(tmp_path: Path, mutation: str) -> None:
    _require_completed_artifacts()
    rows = [json.loads(line) for line in PREDICTIONS.read_text(encoding="utf-8").splitlines()]
    if mutation == "provenance":
        rows[0]["provenance_sha256"] = "0" * 64
    elif mutation == "order":
        rows[0], rows[1] = rows[1], rows[0]
    else:
        rows[0]["question"] = "environmental text is forbidden here"
    changed = tmp_path / "predictions.jsonl"
    changed.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        lock._prediction_inventory(
            changed,
            expected_keys=lock._question_inventory(QUESTIONS),
            expected_provenance_sha256=json.loads(PROVENANCE.read_text(encoding="utf-8"))[
                "provenance_sha256"
            ],
        )
