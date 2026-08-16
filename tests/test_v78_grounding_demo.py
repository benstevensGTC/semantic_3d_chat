from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

import scripts.check_v78_grounding_demo as demo

ROOT = Path(__file__).parents[1]


def test_v78_launcher_is_explicit_optional_and_passes_checkpoint() -> None:
    launcher = (ROOT / "scripts/run_v78_grounding_demo.sh").read_text(encoding="utf-8")
    assert "gemma4_v75_nll_control_release_v1" in launcher
    assert "gemma4_v78_grounding_diagnostic_release_v1" in launcher
    assert "--grounding-checkpoint" in launcher
    assert "semantic_3d_chat.chat.question_control_cli" in launcher
    assert "semantic_3d_chat.evaluation.question_control_leakage" in launcher
    assert "TRANSFORMERS_OFFLINE=1" in launcher
    assert "HF_HUB_OFFLINE=1" in launcher
    assert "data_gemma4/training" not in launcher


def test_v78_embodied_make_targets_are_explicit_and_finite() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    check_start = makefile.index("v78-grounding-embodied-check:")
    check_body = makefile[check_start : makefile.find("\n\n", check_start)]
    once_start = makefile.index("v78-grounding-embodied-once:")
    once_body = makefile[once_start : makefile.find("\n\n", once_start)]

    assert "run_v78_grounding_demo.sh --check" in check_body
    assert "run_embodied_conversation.sh --check" in check_body
    for body in (check_body, once_body):
        assert (
            'EMBODIED_GROUNDING_CHECKPOINT="$(GEMMA4_V78_GROUNDING_CHECKPOINT)"'
            in body
        )
        assert 'EMBODIED_CONTROL_CHECKPOINT="$(GEMMA4_EMBODIED_CONTROL_CHECKPOINT)"' in body
    assert '--command "scan"' in once_body
    assert '--command "Where is the chair?"' in once_body
    assert "--audit-report" in once_body


def test_embodied_launcher_forwards_v78_checkpoint_without_loading_model(
    tmp_path: Path,
) -> None:
    capture = tmp_path / "capture.json"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

Path(os.environ["EMBODIED_CAPTURE"]).write_text(
    json.dumps(
        {
            "argv": sys.argv[1:],
            "grounding_env": os.environ.get("EMBODIED_GROUNDING_CHECKPOINT"),
            "offline": os.environ.get("TRANSFORMERS_OFFLINE"),
            "hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        },
        sort_keys=True,
    ),
    encoding="utf-8",
)
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    inputs = {
        name: tmp_path / name
        for name in (
            "embodied.yaml",
            "control.yaml",
            "base",
            "control",
            "grounding",
            "asset.blend",
            "robot_state",
        )
    }
    for path in inputs.values():
        if path.suffix:
            path.touch()
        else:
            path.mkdir()
    env = {
        **os.environ,
        "EMBODIED_CAPTURE": str(capture),
        "EMBODIED_PYTHON": str(fake_python),
        "EMBODIED_CONFIG": str(inputs["embodied.yaml"]),
        "EMBODIED_CONTROL_CONFIG": str(inputs["control.yaml"]),
        "EMBODIED_SCENE": "scene_000001",
        "EMBODIED_BASE_CHECKPOINT": str(inputs["base"]),
        "EMBODIED_CONTROL_CHECKPOINT": str(inputs["control"]),
        "EMBODIED_GROUNDING_CHECKPOINT": str(inputs["grounding"]),
        "EMBODIED_RUNTIME_ASSET": str(inputs["asset.blend"]),
        "EMBODIED_ROBOT_STATE_CHECKPOINT": str(inputs["robot_state"]),
    }

    subprocess.run(
        [str(ROOT / "scripts/run_embodied_conversation.sh"), "--check"],
        cwd=ROOT,
        env=env,
        check=True,
    )

    payload = json.loads(capture.read_text(encoding="utf-8"))
    arguments = payload["argv"]
    assert arguments[:2] == ["-m", "semantic_3d_chat.robot.conversation_cli"]
    assert arguments[arguments.index("--grounding-checkpoint") + 1] == str(
        inputs["grounding"]
    )
    assert arguments[arguments.index("--control-checkpoint") + 1] == str(
        inputs["control"]
    )
    assert arguments[-1] == "--check"
    assert payload["grounding_env"] == str(inputs["grounding"])
    assert payload["offline"] == "1"
    assert payload["hub_offline"] == "1"


@pytest.mark.skipif(
    not (ROOT / demo.DEFAULT_GROUNDING_CHECKPOINT).is_dir(),
    reason="local V78 runtime diagnostic release is unavailable",
)
def test_actual_v78_demo_preflight_keeps_v75_answer_path() -> None:
    report = demo.validate_v78_grounding_demo_inputs()
    assert report["passed"] is True
    assert report["answer_generation_unchanged"] is True
    assert report["optional_grounding_only"] is True
    assert report["official_validation_evidence"] is False
    assert report["runtime_promotion_authorized"] is False
    assert report["loads_oracle_or_qa"] is False
    grounding = report["v78_numeric_grounding"]
    assert grounding["checkpoint_inventory"] == [
        "grounding.safetensors",
        "metadata.json",
    ]
    assert grounding["all_scene_tokens_scored"] is True


def test_v78_runtime_release_metadata_contains_no_text_payloads() -> None:
    metadata_path = ROOT / demo.DEFAULT_GROUNDING_CHECKPOINT / "metadata.json"
    if not metadata_path.is_file():
        pytest.skip("local V78 runtime diagnostic release is unavailable")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, sort_keys=True).casefold()
    assert payload["answer_text_serialized"] is False
    assert payload["question_text_serialized"] is False
    assert payload["object_ids_serialized"] is False
    assert payload["target_coordinates_serialized"] is False
    assert payload["environmental_text_inputs"] == []
    assert "target_xyz" not in serialized
