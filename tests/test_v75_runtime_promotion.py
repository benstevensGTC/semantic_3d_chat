from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import torch

from semantic_3d_chat.chat.question_control_runtime import _load_control_head
from semantic_3d_chat.evaluation.v75_runtime_promotion import (
    DEFAULT_BASE_RELEASE_MANIFEST,
    DEFAULT_CANDIDATE,
    DEFAULT_CORRECT_REPORT,
    DEFAULT_RUNTIME_CONFIG,
    DEFAULT_SOURCE_REPORT,
    DEFAULT_WRONG_REPORT,
    EXPECTED_CANDIDATE_SHA256,
    build_v75_gate_attestation_payload,
    promote_v75_candidate,
    write_v75_gate_attestation,
)
from semantic_3d_chat.scene_encoder.question_control_v75 import (
    DenseFullSceneContinuousControlV75,
)


def _copy_json(source: Path, destination: Path) -> dict[str, object]:
    value = json.loads(source.read_text(encoding="utf-8"))
    destination.write_text(json.dumps(value), encoding="utf-8")
    return value


def test_v75_attestation_rederives_exact_behavior_and_contains_no_predictions() -> None:
    payload, control = build_v75_gate_attestation_payload()

    assert type(control) is DenseFullSceneContinuousControlV75
    assert payload["passed"] is True
    assert all(payload["gates"].values())
    assert payload["candidate"]["sha256"] == EXPECTED_CANDIDATE_SHA256
    assert payload["observed"] == {
        "full_correct_scene": {"correct": 295, "total": 384},
        "full_wrong_scene": {"correct": 278, "total": 384},
        "changed_side_correct": {"correct": 31, "total": 52},
        "wrong_scene_original_target": {"correct": 14, "total": 52},
        "wrong_scene_paired_target_follow": {"correct": 31, "total": 52},
        "complete_changed_units": {"correct": 6, "total": 26},
    }
    assert payload["runtime_binding"] == {
        "training_base_checkpoint_sha256": (
            "3e128b40c1b73bb32750285679cda6b1bea364e67465e986a94a81dfc95e81e8"
        ),
        "runtime_base_checkpoint_sha256": (
            "7c3e679702ccd204fa4d7ae4077b065f3d7a7fe36df7dbc45492d67566e97f59"
        ),
        "base_runtime_config_sha256": (
            "714c60ce9ccb1dff69c72f6618f8afb6f31bc60a830b5ee0fb794fedaa8a321e"
        ),
    }
    serialized = json.dumps(payload, sort_keys=True)
    for forbidden in (
        '"prediction"',
        '"reference"',
        '"question_id"',
        '"question_key"',
        '"answer_labels"',
        '"answer_items"',
        '"scene_description"',
    ):
        assert forbidden not in serialized


def test_v75_attestation_is_deterministic_and_promotion_packages_two_files(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first_result = write_v75_gate_attestation(first)
    second_result = write_v75_gate_attestation(second)
    assert first.read_bytes() == second.read_bytes()
    assert first_result["attestation_sha256"] == second_result["attestation_sha256"]

    checkpoint = tmp_path / "runtime"
    result = promote_v75_candidate(
        attestation_path=first,
        checkpoint_path=checkpoint,
    )
    assert result["passed"] is True
    assert {item.name for item in checkpoint.iterdir()} == {
        "control.safetensors",
        "runtime_metadata.json",
    }
    loaded, metadata = _load_control_head(
        checkpoint,
        hidden_size=1536,
        device=torch.device("cpu"),
    )
    assert type(loaded) is DenseFullSceneContinuousControlV75
    assert loaded.coefficient_decoder_hidden_dimension == 768
    assert metadata["source_v75_candidate_sha256"] == EXPECTED_CANDIDATE_SHA256
    assert metadata["base_checkpoint_sha256"] == (
        "7c3e679702ccd204fa4d7ae4077b065f3d7a7fe36df7dbc45492d67566e97f59"
    )
    assert metadata["saved_runtime_training_gate_attestation_sha256"] == hashlib.sha256(
        first.read_bytes()
    ).hexdigest()


def test_v75_gate_rejects_candidate_byte_tampering(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.safetensors"
    shutil.copyfile(DEFAULT_CANDIDATE, candidate)
    with candidate.open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(ValueError, match="candidate identity changed"):
        build_v75_gate_attestation_payload(candidate_path=candidate)


@pytest.mark.parametrize(
    ("source", "key", "message"),
    [
        (DEFAULT_CORRECT_REPORT, "official_validation_loaded", "contract mismatch"),
        (DEFAULT_WRONG_REPORT, "oracle_loaded", "contract mismatch"),
        (DEFAULT_SOURCE_REPORT, "official_test_loaded", "contract mismatch"),
    ],
)
def test_v75_gate_rejects_any_official_or_oracle_access_flag(
    tmp_path: Path,
    source: Path,
    key: str,
    message: str,
) -> None:
    target = tmp_path / source.name
    payload = _copy_json(source, target)
    payload[key] = True
    target.write_text(json.dumps(payload), encoding="utf-8")
    kwargs: dict[str, Path] = {}
    if source == DEFAULT_CORRECT_REPORT:
        kwargs["correct_report_path"] = target
    elif source == DEFAULT_WRONG_REPORT:
        kwargs["wrong_report_path"] = target
    else:
        kwargs["source_report_path"] = target
    with pytest.raises(ValueError, match=message):
        build_v75_gate_attestation_payload(**kwargs)


def test_v75_gate_rejects_record_and_paired_scene_tampering(tmp_path: Path) -> None:
    correct_path = tmp_path / "correct.json"
    correct = _copy_json(DEFAULT_CORRECT_REPORT, correct_path)
    records = correct["records"]
    assert isinstance(records, list) and isinstance(records[0], dict)
    records[0]["correct"] = not records[0]["correct"]
    correct_path.write_text(json.dumps(correct), encoding="utf-8")
    with pytest.raises(ValueError, match="record correctness disagrees"):
        build_v75_gate_attestation_payload(correct_report_path=correct_path)

    wrong_path = tmp_path / "wrong.json"
    wrong = _copy_json(DEFAULT_WRONG_REPORT, wrong_path)
    wrong_records = wrong["records"]
    assert isinstance(wrong_records, list) and isinstance(wrong_records[0], dict)
    wrong_records[0]["environment_scene_id"] = wrong_records[0]["scene_id"]
    wrong_path.write_text(json.dumps(wrong), encoding="utf-8")
    with pytest.raises(ValueError, match="did not use the paired scene"):
        build_v75_gate_attestation_payload(wrong_report_path=wrong_path)


def test_v75_promotion_rejects_attestation_tampering_without_output(tmp_path: Path) -> None:
    attestation = tmp_path / "attestation.json"
    write_v75_gate_attestation(attestation)
    payload = json.loads(attestation.read_text(encoding="utf-8"))
    payload["observed"]["changed_side_correct"]["correct"] = 30
    attestation.write_text(json.dumps(payload), encoding="utf-8")
    checkpoint = tmp_path / "denied"

    with pytest.raises(ValueError, match="does not authenticate current evidence"):
        promote_v75_candidate(
            attestation_path=attestation,
            checkpoint_path=checkpoint,
        )
    assert not checkpoint.exists()


def test_v75_gate_rejects_release_manifest_or_runtime_config_drift(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _copy_json(DEFAULT_BASE_RELEASE_MANIFEST, manifest)
    with pytest.raises(ValueError, match="manifest identity changed"):
        build_v75_gate_attestation_payload(base_release_manifest_path=manifest)

    alternative = Path("configs/runtime/gemma4_primary.yaml")
    assert alternative.is_file()
    with pytest.raises(ValueError, match="runtime configuration changed"):
        build_v75_gate_attestation_payload(runtime_config_path=alternative)


def test_actual_v75_release_matches_fresh_attestation_and_loads() -> None:
    attestation = Path(
        "reports/gemma4/metrics/v75_gemma_nll_runtime_promotion_attestation.json"
    )
    checkpoint = Path(
        "data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1"
    )
    expected, _control = build_v75_gate_attestation_payload(
        candidate_path=DEFAULT_CANDIDATE,
        source_report_path=DEFAULT_SOURCE_REPORT,
        correct_report_path=DEFAULT_CORRECT_REPORT,
        wrong_report_path=DEFAULT_WRONG_REPORT,
        runtime_config_path=DEFAULT_RUNTIME_CONFIG,
    )
    assert json.loads(attestation.read_text(encoding="utf-8")) == expected
    loaded, metadata = _load_control_head(
        checkpoint,
        hidden_size=1536,
        device=torch.device("cpu"),
    )
    assert type(loaded) is DenseFullSceneContinuousControlV75
    assert metadata["saved_runtime_training_gate_passed"] is True
