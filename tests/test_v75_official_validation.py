from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import score_v75_official_validation as scorer
from semantic_3d_chat.evaluation import v75_official_validation_contract as contract


def _valid_audit() -> dict[str, object]:
    return {
        "answer_class_codebook_runtime_loaded": False,
        "answer_text_runtime_loaded": False,
        "architecture": contract.V75_RUNTIME_ARCHITECTURE,
        "bias_free_nonlinear_coefficient_decoder": True,
        "bilinear_question_scene_value_interaction": True,
        "coefficient_decoder_hidden_dimension": 768,
        "control_token_count": 4,
        "control_used": True,
        "environment_latent_count": 256,
        "every_scene_token_influenced_output": True,
        "immutable_full_prefix_retained_separately": True,
        "latent_selection_or_top_k_used": False,
        "maximum_control_rms": 0.1,
        "minimum_attention_weight": 0.05 / 256,
        "positive_attention_floor": True,
        "prequestion_scene_key_value_cache": True,
        "question_dependent_scene_retrieval": False,
        "question_only_output_path_exists": False,
        "saved_runtime_training_gate_required": True,
        "scene_token_count": 258,
        "softmax_scene_attention_used": True,
        "training_answers_runtime_loaded": False,
        "zero_preserving_coefficient_activation": True,
        "zero_scene_produces_exact_zero_controls": True,
    }


def test_frozen_questions_and_sealed_v75_checkpoint_authenticate() -> None:
    manifest = contract.validate_official_question_manifest(
        contract.DEFAULT_QUESTIONS_MANIFEST
    )
    control = contract.authenticate_v75_control_checkpoint(
        contract.DEFAULT_CONTROL_CHECKPOINT
    )

    assert manifest.question_count == 216
    assert sorted(manifest.by_scene()) == list(contract.EXPECTED_SCENE_IDS)
    assert control.metadata["source_v75_candidate_sha256"] == (
        contract.EXPECTED_SOURCE_V75_CANDIDATE_SHA256
    )
    assert control.metadata["environmental_text_inputs"] == []
    assert len(control.sha256) == 64


@pytest.mark.parametrize(
    "value,match",
    [
        ("data_diverse52/qa/validation.jsonl", "path tokens"),
        ("data/oracle/scene_000057.json", "path tokens"),
        ("reports/references/questions.json", "path tokens"),
    ],
)
def test_prediction_input_rejects_answer_or_oracle_paths(
    value: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        contract.safe_prediction_input(value, "test", kind="file")


def test_prediction_output_rejects_scorer_and_answer_paths() -> None:
    with pytest.raises(ValueError, match="output path tokens"):
        contract.safe_prediction_output("reports/scorer/answers.jsonl")


def test_control_audit_requires_v75_nonlinearity_and_all_latents() -> None:
    valid = _valid_audit()
    observed = contract.validate_v75_control_audit(valid)
    assert observed["coefficient_decoder_hidden_dimension"] == 768

    bad = dict(valid)
    bad["every_scene_token_influenced_output"] = False
    with pytest.raises(ValueError, match="full-scene"):
        contract.validate_v75_control_audit(bad)

    missing = dict(valid)
    missing.pop("zero_preserving_coefficient_activation")
    with pytest.raises(ValueError, match="fields changed"):
        contract.validate_v75_control_audit(missing)


def test_scorer_parser_is_the_only_process_with_a_reference_argument() -> None:
    prediction = argparse.ArgumentParser()
    prediction.add_argument("--config")
    prediction.add_argument("--questions-manifest")
    prediction.add_argument("--base-checkpoint")
    prediction.add_argument("--control-checkpoint")
    prediction.add_argument("--output")
    prediction_destinations = {action.dest for action in prediction._actions}
    scoring_destinations = {action.dest for action in scorer._parser()._actions}

    assert "references" not in prediction_destinations
    assert "qa" not in prediction_destinations
    assert "oracle" not in prediction_destinations
    assert "references" in scoring_destinations


def test_scorer_source_has_no_model_or_runtime_import() -> None:
    source = Path(scorer.__file__).read_text(encoding="utf-8")
    forbidden_imports = (
        "language.local_lm",
        "chat.question_control_runtime",
        "chat.runtime import",
        "torch",
        "transformers",
    )
    assert not any(value in source for value in forbidden_imports)


def test_official_prediction_provenance_authenticates_without_references() -> None:
    predictions = contract.resolve_path(contract.DEFAULT_PREDICTIONS)
    if not predictions.is_file():
        pytest.skip("official V75 predictions have not been generated")
    manifest = contract.validate_official_question_manifest(
        contract.DEFAULT_QUESTIONS_MANIFEST
    )
    control = contract.authenticate_v75_control_checkpoint(
        contract.DEFAULT_CONTROL_CHECKPOINT
    )
    assert manifest.manifest_path is not None
    provenance = scorer._load_provenance(
        predictions,
        manifest_path=manifest.manifest_path,
        control=control,
    )

    assert provenance.references_sha256 == contract.EXPECTED_QUESTION_MANIFEST_SHA256
    assert provenance.checkpoint_sha256 == contract.EXPECTED_BASE_CHECKPOINT_SHA256
    assert provenance.config_sha256 == contract.EXPECTED_RUNTIME_CONFIG_SHA256


def test_prediction_validator_rejects_a_tampered_audit() -> None:
    predictions = contract.resolve_path(contract.DEFAULT_PREDICTIONS)
    if not predictions.is_file():
        pytest.skip("official V75 predictions have not been generated")
    rows = scorer._load_jsonl(predictions, "predictions")
    manifest = contract.validate_official_question_manifest(
        contract.DEFAULT_QUESTIONS_MANIFEST
    )
    control = contract.authenticate_v75_control_checkpoint(
        contract.DEFAULT_CONTROL_CHECKPOINT
    )
    assert manifest.manifest_path is not None
    provenance = scorer._load_provenance(
        predictions,
        manifest_path=manifest.manifest_path,
        control=control,
    )
    tampered = [dict(row) for row in rows]
    tampered[0] = dict(tampered[0])
    tampered[0]["control_audit"] = dict(tampered[0]["control_audit"])
    tampered[0]["control_audit"]["question_dependent_scene_retrieval"] = True
    questions = {
        (record.scene_id, record.question_id): record.question
        for record in manifest.questions
    }

    with pytest.raises(ValueError, match="full-scene"):
        scorer._validate_predictions(
            tampered,
            questions,
            control_sha256=control.sha256,
            provenance_sha256=provenance.sha256,
        )


def test_score_report_contract_does_not_serialize_environmental_text() -> None:
    score_path = contract.resolve_path(contract.DEFAULT_SCORE)
    if not score_path.is_file():
        pytest.skip("official V75 score has not been generated")
    report = json.loads(score_path.read_text(encoding="utf-8"))
    serialized = json.dumps(report, sort_keys=True)

    assert report["scope"]["answer_references_loaded_only_by_isolated_scorer"]
    assert report["scope"]["prediction_process_accepts_answer_references"] is False
    assert '"question":' not in serialized
    assert '"answer":' not in serialized
    assert '"predicted_answer":' not in serialized


def test_control_identity_is_path_free_when_reduced_to_metrics() -> None:
    identity = contract.authenticate_v75_control_checkpoint(
        contract.DEFAULT_CONTROL_CHECKPOINT
    )
    reduced = {
        key: value
        for key, value in asdict(identity).items()
        if key != "path" and key != "metadata"
    }
    assert set(reduced) == {
        "sha256",
        "weights_sha256",
        "runtime_metadata_sha256",
    }
