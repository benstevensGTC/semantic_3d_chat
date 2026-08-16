from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.check_v75_demo as demo_check

ROOT = Path(__file__).parents[1]


def test_v75_launcher_defaults_to_exact_sanitized_releases() -> None:
    launcher = (ROOT / "scripts" / "run_v75_question_control_demo.sh").read_text(
        encoding="utf-8"
    )

    assert "configs/runtime/gemma4_v56_question_control.yaml" in launcher
    assert "data_gemma4/runtime/checkpoints/gemma4_v54_release_v1" in launcher
    assert (
        "data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1"
        in launcher
    )
    assert "scripts/check_v75_demo.py" in launcher
    assert "semantic_3d_chat.chat.question_control_cli" in launcher
    assert "semantic_3d_chat.evaluation.question_control_leakage" in launcher
    assert "TRANSFORMERS_OFFLINE=1" in launcher
    assert "HF_HUB_OFFLINE=1" in launcher
    assert "data_gemma4/training" not in launcher
    assert "--training-artifact" not in launcher


def test_v75_demo_artifact_manifest_authenticates_both_two_file_releases() -> None:
    payload = json.loads(
        (ROOT / "configs" / "runtime" / "demo_artifacts_v1.json").read_text(
            encoding="utf-8"
        )
    )
    entries = {entry["path"]: entry for entry in payload["artifacts"]}

    assert entries[
        "data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1/"
        "control.safetensors"
    ] == {
        "path": (
            "data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1/"
            "control.safetensors"
        ),
        "role": "promoted_v75_continuous_controller",
        "sha256": "bb112f42ca5df71b88b4cd7721b9107f9be9b0dc01b612a4ace6212548da669c",
        "size_bytes": 8_356_368,
    }
    assert entries[
        "data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1/"
        "runtime_metadata.json"
    ]["sha256"] == "a45a192d27336329580612524d43f71f08e3f472e5fe833747ffc1395e2aa2be"


def test_v75_demo_preflight_refuses_nonopaque_scene_before_any_model_load() -> None:
    with pytest.raises(ValueError, match="opaque"):
        demo_check.validate_v75_demo_inputs(scene_id="room_with_chair")


@pytest.mark.skipif(
    not (ROOT / demo_check.DEFAULT_CONTROL_CHECKPOINT).is_dir(),
    reason="promoted local V75 runtime release is unavailable",
)
def test_actual_v75_demo_preflight_is_exact_and_no_model() -> None:
    report = demo_check.validate_v75_demo_inputs()

    assert report["passed"] is True
    assert report["loads_model"] is False
    assert report["runs_blender"] is False
    assert report["control_schema_version"] == 75
    assert report["scene_latents"] == 256
    assert report["complete_scene_prefix_required"] is True
    assert report["prequestion_scene_key_value_cache"] is True
    assert report["all_environment_latents_attended"] is True
    assert report["question_dependent_scene_retrieval"] is False
    assert report["environmental_text_inputs"] == []
    assert report["training_or_evaluation_artifacts_loaded"] is False
    assert report["base_checkpoint_inventory"] == [
        "adapter.safetensors",
        "runtime_metadata.json",
    ]
