from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_3d_chat.config import PROJECT_ROOT, artifact_root, load_config
from semantic_3d_chat.data.splits import split_fingerprint
from semantic_3d_chat.evaluation.v28_stage_b_selector import (
    _retention_control_config,
    _selection_requirements,
)
from semantic_3d_chat.training.pair_curriculum import pair_curriculum_settings
from semantic_3d_chat.training.train_post_stack_decoder import (
    load_stage_b_qa_records,
    stage_b_settings,
    v29_development_contract,
)

V29_CONFIG = Path("configs/experiments/gemma4_diverse20_post_stack_decoder_stage_b_v29.yaml")
V28_CONFIG = Path("configs/experiments/gemma4_color_mirror_post_stack_decoder_stage_b_v28.yaml")


def _record(scene_id: str, question_id: str) -> dict[str, object]:
    return {
        "scene_id": scene_id,
        "question_id": question_id,
        "question": "Training-only question?",
        "answer": "yes",
        "answer_type": "presence",
        "target_xyz": None,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _tiny_v29_config(tmp_path: Path) -> tuple[dict, Path]:
    qa_root = tmp_path / "data_diverse20" / "qa"
    qa_root.mkdir(parents=True)
    splits = {
        "train": ["scene_000011", "scene_000012"],
        "validation": ["scene_000019"],
        "test": [],
    }
    manifest = {
        "schema_version": 2,
        "splits": splits,
        "fingerprint": split_fingerprint(splits),
    }
    (qa_root / "splits.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_jsonl(
        qa_root / "train.jsonl",
        [
            _record("scene_000011", "q_000001"),
            _record("scene_000012", "q_000002"),
        ],
    )
    _write_jsonl(
        qa_root / "validation.jsonl",
        [_record("scene_000019", "q_000003")],
    )
    (qa_root / "test.jsonl").write_text("", encoding="utf-8")
    config = load_config(V29_CONFIG)
    config["paths"]["qa_root"] = str(qa_root)
    config["v29_development"].update(
        {
            "qa_root": str(qa_root),
            "split_fingerprint": split_fingerprint(splits),
            "train_scene_ids": splits["train"],
            "validation_scene_ids": splits["validation"],
            "deferred_test_scene_ids": ["scene_000025"],
            "question_counts": {"train": 2, "validation": 1},
        }
    )
    return config, qa_root


def test_v29_config_uses_diverse20_and_keeps_v28_unchanged() -> None:
    v29 = load_config(V29_CONFIG)
    v28 = load_config(V28_CONFIG)
    contract = v29_development_contract(v29)
    assert contract is not None

    assert artifact_root(v29, "qa").resolve() == (PROJECT_ROOT / "data_diverse20/qa").resolve()
    assert contract.qa_root == artifact_root(v29, "qa").resolve()
    assert contract.train_scene_ids == tuple(f"scene_{scene:06d}" for scene in range(11, 19))
    assert contract.validation_scene_ids == tuple(f"scene_{scene:06d}" for scene in range(19, 25))
    assert contract.deferred_test_scene_ids == tuple(
        f"scene_{scene:06d}" for scene in range(25, 31)
    )
    assert stage_b_settings(v29).gradient_accumulation == 48
    assert artifact_root(v28, "qa").resolve() == (PROJECT_ROOT / "data/qa").resolve()
    assert stage_b_settings(v28).gradient_accumulation == 12
    assert v29_development_contract(v28) is None


def test_v29_loader_reads_only_locked_train_and_validation_scenes(
    tmp_path: Path,
) -> None:
    config, qa_root = _tiny_v29_config(tmp_path)

    train, validation, audit = load_stage_b_qa_records(
        config,
        max_train_questions=None,
        max_validation_questions=None,
    )

    assert audit["qa_root"] == str(qa_root.resolve())
    assert audit["split_guarded"] is True
    assert audit["v29_development_contract"] is True
    assert audit["selected_train_scene_ids"] == ["scene_000011", "scene_000012"]
    assert audit["selected_validation_scene_ids"] == ["scene_000019"]
    assert audit["deferred_test_scene_ids_loaded"] == []
    assert {record.scene_id for record in train} == {
        "scene_000011",
        "scene_000012",
    }
    assert {record.scene_id for record in validation} == {"scene_000019"}


def test_v29_loader_rejects_deferred_scene_in_training_jsonl(tmp_path: Path) -> None:
    config, qa_root = _tiny_v29_config(tmp_path)
    _write_jsonl(
        qa_root / "train.jsonl",
        [
            _record("scene_000011", "q_000001"),
            _record("scene_000012", "q_000002"),
            _record("scene_000025", "q_000004"),
        ],
    )

    with pytest.raises(ValueError, match="outside splits.json train set"):
        load_stage_b_qa_records(
            config,
            max_train_questions=None,
            max_validation_questions=None,
        )


def test_v29_loader_rejects_nonempty_or_unlocked_test_split(tmp_path: Path) -> None:
    config, qa_root = _tiny_v29_config(tmp_path)
    manifest = json.loads((qa_root / "splits.json").read_text(encoding="utf-8"))
    manifest["splits"]["test"] = ["scene_000025"]
    manifest["fingerprint"] = split_fingerprint(manifest["splits"])
    (qa_root / "splits.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="persisted scene split differs"):
        load_stage_b_qa_records(
            config,
            max_train_questions=None,
            max_validation_questions=None,
        )


def test_v29_selector_retains_old_color_and_mirror_control_config() -> None:
    v29 = load_config(V29_CONFIG)
    controls = _retention_control_config(v29)
    curriculum = pair_curriculum_settings(controls)

    assert Path(controls["_config_path"]) == (PROJECT_ROOT / V28_CONFIG).resolve()
    assert artifact_root(controls, "qa").resolve() == (PROJECT_ROOT / "data/qa").resolve()
    assert set(curriculum.pair_only_scene_ids) == {
        "scene_000003",
        "scene_000004",
        "scene_000007",
        "scene_000008",
    }
    assert _selection_requirements(v29)[:3] == (12, 10, 4)


def test_makefile_exposes_v29_development_without_a_test_target() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert "gemma4-v29-train-diverse-stage-b" in makefile
    assert "gemma4-v29-select-diverse-stage-b" in makefile
    assert "gemma4-v29-evaluate-diverse-validation" in makefile
    assert "gemma4-v29-evaluate-diverse-test" not in makefile
