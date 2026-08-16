from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.evaluation import v75_fixed_atlas_artifacts as artifacts
from semantic_3d_chat.evaluation import v75_fixed_atlas_behavior as behavior
from semantic_3d_chat.evaluation import v75_fixed_atlas_preflight as preflight
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas import tensor_sha256
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas_v75 import (
    compile_fixed_scene_atlas_v75_v2,
)
from semantic_3d_chat.training.train_question_control_v73 import (
    load_config_v73,
    load_training_rows_v73,
    split_rows_v73,
)


def test_prepare_and_predictor_configs_lock_historical_only_boundaries() -> None:
    prepare = artifacts.load_prepare_config(
        "configs/experiments/gemma4_v75_fixed_prefix_atlas_prepare.yaml"
    )
    runtime = behavior.load_behavior_config(
        "configs/experiments/gemma4_v75_fixed_prefix_atlas_behavior.yaml"
    )
    assert prepare["scope"] == artifacts._SCOPE
    assert runtime["scope"] == behavior._SCOPE
    assert runtime["scene_ids"] == list(behavior.SCENE_IDS)
    assert runtime["probe_count"] == 96
    serialized = json.dumps(runtime, sort_keys=True).casefold()
    assert "training_qa" not in serialized
    assert "official_validation" in serialized  # explicit false contract
    assert "references" not in serialized
    assert "data/oracle" not in serialized


def test_historical_smoke_is_pair_and_scene_disjoint_but_not_question_disjoint() -> None:
    config = load_config_v73("configs/experiments/gemma4_v73_fullscene_controller.yaml")
    rows = load_training_rows_v73(config["training_qa"])
    train, held = split_rows_v73(rows)
    smoke = artifacts.select_historical_smoke_rows(held)
    assert len(smoke) == 16
    assert {row.scene_id for row in smoke} == set(behavior.SCENE_IDS)
    assert {row.scene_id for row in smoke}.isdisjoint({row.scene_id for row in train})
    assert {row.pair_id for row in smoke}.isdisjoint({row.pair_id for row in train})
    train_questions = {row.question for row in train}
    assert sum(row.question in train_questions for row in smoke) == 12


def test_numeric_probe_builder_matches_v75_mean_pooling_and_is_deterministic() -> None:
    questions = tuple(f"question-{index:03d}" for index in range(96))
    embeddings = {
        question: torch.arange((index % 4 + 1) * 1536, dtype=torch.float32).reshape(
            index % 4 + 1, 1536
        )
        for index, question in enumerate(questions)
    }
    first = artifacts.build_numeric_probe_tensor(questions, embeddings)
    second = artifacts.build_numeric_probe_tensor(questions, embeddings)
    assert first.shape == (96, 1536)
    assert torch.equal(first, second)
    assert torch.equal(first[7], embeddings[questions[7]].mean(dim=0))
    with pytest.raises(ValueError, match="inventory"):
        artifacts.build_numeric_probe_tensor(questions, dict(list(embeddings.items())[:-1]))


def _probe_metadata(probe_path: Path, probes: torch.Tensor) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact": "v75_fixed_atlas_numeric_probe_bank_v1",
        "status": "historical_internal_diagnostic_not_promoted",
        "probe_file_sha256": behavior._sha256_file(probe_path),
        "probe_tensor_sha256": tensor_sha256(probes),
        "probe_count": 96,
        "hidden_size": 1536,
        "dtype": "torch.float32",
        "source_scope": "v73_historical_optimization_fold_only",
        "source_train_pair_count": 12,
        "source_train_scene_count": 24,
        "source_train_row_count": 576,
        "source_unique_question_count": 96,
        "source_question_hash_inventory_sha256": "a" * 64,
        "source_qa_sha256": artifacts.SOURCE_QA_SHA256,
        "source_v73_config_sha256": artifacts.SOURCE_V73_CONFIG_SHA256,
        "model_revision": artifacts.GEMMA_REVISION,
        "model_file_sha256": artifacts.GEMMA_MODEL_FILE_SHA256,
        "embedding_tensor_name": "model.language_model.embed_tokens.weight",
        "pooling": "mean_of_complete_question_token_embedding_sequence",
        "probe_order": "ascending_sha256_of_question_text_not_serialized",
        "questions_or_answers_serialized": False,
        "answer_codebook_serialized": False,
        "environmental_text_serialized": False,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
        "runtime_promotion_authorized": False,
    }


def test_probe_runtime_loader_authenticates_numeric_only_two_file_artifact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "probe_bank"
    root.mkdir()
    probes = torch.randn(96, 1536, generator=torch.Generator().manual_seed(5))
    probe_path = root / "probes.safetensors"
    save_file(
        {"probe_embeddings": probes},
        probe_path,
        metadata=behavior._PROBE_SAFE_METADATA,
    )
    (root / "runtime_metadata.json").write_text(
        json.dumps(_probe_metadata(probe_path, probes)), encoding="utf-8"
    )
    audit = FileAccessAudit(block_forbidden=True)
    with audit:
        loaded, metadata = behavior._load_probe_bank(root, audit)
    assert torch.equal(loaded, probes)
    assert metadata["questions_or_answers_serialized"] is False
    assert {path.name for path in root.iterdir()} == {
        "probes.safetensors",
        "runtime_metadata.json",
    }
    assert audit.forbidden_accesses() == []

    metadata_path = root / "runtime_metadata.json"
    tampered = json.loads(metadata_path.read_text(encoding="utf-8"))
    tampered["question_text"] = "forbidden"
    metadata_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="fields changed"):
        behavior._load_probe_bank(root, FileAccessAudit())


def test_predictor_blocks_physically_separate_answer_references(tmp_path: Path) -> None:
    scorer = tmp_path / "scorer"
    scorer.mkdir()
    secret = scorer / "references.jsonl"
    secret.write_text('{"answer":"left"}\n', encoding="utf-8")
    audit = FileAccessAudit([scorer], block_forbidden=True)
    with audit, pytest.raises(PermissionError, match="before open"):
        secret.read_text(encoding="utf-8")
    assert audit.forbidden_accesses() == [str(secret.resolve())]


def test_predictor_compiles_before_opening_question_manifest_and_has_no_reference_arg() -> None:
    source = inspect.getsource(behavior.predict)
    assert source.index("compile_fixed_scene_atlas_v75_v2") < source.index(
        "_load_predictor_questions"
    )
    assert list(inspect.signature(behavior.predict).parameters) == ["config_path"]
    assert "reference" not in inspect.signature(behavior.predict).parameters
    compiler_parameters = inspect.signature(
        compile_fixed_scene_atlas_v75_v2
    ).parameters
    assert set(compiler_parameters) == {
        "base_scene_prefix",
        "controller",
        "probe_embeddings",
    }


def test_predictor_requires_prepared_artifacts_before_loading_gemma(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = behavior.load_behavior_config(
        "configs/experiments/gemma4_v75_fixed_prefix_atlas_behavior.yaml"
    )
    config["probe_bank"] = str(tmp_path / "missing-probe-bank")
    config["output_predictions"] = str(tmp_path / "predictions.json")
    monkeypatch.setattr(behavior, "load_behavior_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(
        behavior.StaticChatRuntime,
        "load",
        lambda *_args, **_kwargs: pytest.fail("Gemma runtime must not load"),
    )
    with pytest.raises(FileNotFoundError, match="probe bank"):
        behavior.predict("ignored.yaml")


def test_predictor_refuses_output_overwrite_before_loading_any_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = behavior.load_behavior_config(
        "configs/experiments/gemma4_v75_fixed_prefix_atlas_behavior.yaml"
    )
    output = tmp_path / "predictions.json"
    output.write_text("immutable", encoding="utf-8")
    config["output_predictions"] = str(output)
    monkeypatch.setattr(behavior, "load_behavior_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(
        behavior,
        "_load_probe_bank",
        lambda *_args, **_kwargs: pytest.fail("Artifacts must not be opened"),
    )
    with pytest.raises(FileExistsError):
        behavior.predict("ignored.yaml")
    assert output.read_text(encoding="utf-8") == "immutable"


def test_model_free_preflight_never_constructs_gemma_or_compiles_prefix() -> None:
    source = inspect.getsource(preflight.preflight)
    assert "StaticChatRuntime" not in source
    assert "load_local_language_model" not in source
    assert "compile_fixed_scene_atlas" not in source
    assert "gemma_model_loaded" in source
    assert '"scene_prefix_compiled": False' in source


def test_makefile_exposes_ordered_no_overwrite_diagnostic_targets() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    targets = (
        "v75-fixed-atlas-behavior-prepare",
        "v75-fixed-atlas-behavior-preflight",
        "v75-fixed-atlas-behavior-predict",
        "v75-fixed-atlas-behavior-score",
        "v75-fixed-atlas-behavior-full",
    )
    phony = next(
        line
        for line in makefile.splitlines()
        if line.startswith(".PHONY:") and targets[0] in line
    )
    assert all(target in phony for target in targets)
    assert (
        "v75-fixed-atlas-behavior-predict: v75-fixed-atlas-behavior-preflight"
        in makefile
    )
    assert "scripts/prepare_v75_fixed_atlas_behavior.py" in makefile
    assert "scripts/check_v75_fixed_atlas_behavior.py" in makefile
    assert "scripts/predict_v75_fixed_atlas_behavior.py" in makefile
    assert "scripts/score_v75_fixed_atlas_behavior.py" in makefile
    full = makefile[makefile.index(f"{targets[-1]}:") :]
    full = full[: full.index("\n\n")]
    assert full.index(targets[0]) < full.index(targets[2]) < full.index(targets[3])
    assert "official" not in full.casefold()


def _write_reference_artifact(root: Path) -> None:
    root.mkdir()
    rows = []
    for index in range(16):
        rows.append(
            {
                "row_id": f"row_{index:024x}",
                "answer": "yes" if index % 2 == 0 else "no",
                "answer_type": "presence",
                "change_type": f"family_{index // 2}",
                "unit_id": f"unit_{index // 2:024x}",
            }
        )
    reference_path = root / "references.jsonl"
    reference_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    metadata = {
        "schema_version": 1,
        "artifact": "v75_fixed_atlas_historical_smoke_scorer_references_v1",
        "status": "evaluation_only_never_loaded_by_predictor",
        "references_file_sha256": behavior._sha256_file(reference_path),
        "row_count": 16,
        "unit_count": 8,
        "change_family_count": 8,
        "model_or_runtime_loaded_by_scorer": False,
        "physically_separate_from_predictor_questions": True,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_model_free_scorer_compares_atlas_direct_v75_and_v54(tmp_path: Path) -> None:
    references = tmp_path / "scorer"
    _write_reference_artifact(references)
    records = []
    for index in range(16):
        answer = "yes" if index % 2 == 0 else "no"
        records.append(
            {
                "row_id": f"row_{index:024x}",
                "scene_id": behavior.SCENE_IDS[index],
                "atlas_prefix_sha256": "a" * 64,
                "base_prefix_sha256": "b" * 64,
                "atlas_prediction": answer,
                "direct_v75_prediction": answer if index < 12 else "unknown",
                "v54_prediction": answer if index < 8 else "unknown",
                "direct_v75_control_rms": 0.1,
                "elapsed_seconds": 0.2,
            }
        )
    prediction = {
        "artifact": "v75_fixed_atlas_historical_internal_predictions_v1",
        "execution_valid": True,
        "row_count": 16,
        "runtime_promotion_authorized": False,
        "behavioral_accuracy_scored_in_predictor": False,
        "scene_prefix": {
            "prefix_hashes_invariant": True,
            "all_scenes_compiled_before_question_manifest_opened": True,
            "same_compiled_prefix_reused_for_every_question": True,
            "question_inputs_used_for_compilation": False,
            "question_dependent_scene_processing": False,
            "question_dependent_retrieval": False,
        },
        "leakage": {
            "forbidden_access_count": 0,
            "scorer_reference_files_loaded": False,
        },
        "records": records,
    }
    prediction_path = tmp_path / "predictions.json"
    prediction_path.write_text(json.dumps(prediction), encoding="utf-8")
    output = tmp_path / "score.json"
    result = behavior.score(prediction_path, references, output)
    assert result["fixed_v75_atlas"]["correct"] == 16
    assert result["direct_exact_v75"]["correct"] == 12
    assert result["frozen_v54"]["correct"] == 8
    assert result["runtime_promotion_authorized"] is False
    assert result["structural_compiler_implies_behavioral_success"] is False
    assert output.is_file()
