from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

import scripts.check_v90_release as release_check

ROOT = Path(__file__).parents[1]
LAUNCHER = ROOT / "scripts" / "run_v90_strict_scene1_demo.sh"


def _quoted_questions(block: str) -> list[str]:
    return re.findall(r'^\s+"([^"]+\?)"$', block, flags=re.MULTILINE)


def test_v90_launcher_targets_only_the_future_strict_release() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert "configs/runtime/gemma4_v90_strict_scene1.yaml" in launcher
    assert "gemma4_v90_strict_scene1_release_v1" in launcher
    assert "runtime/scene_memories/v90/scene_000001" in launcher
    assert "scripts/check_v90_release.py" in launcher
    assert "semantic_3d_chat.chat.v90_strict_scene1_cli" in launcher
    assert "semantic_3d_chat.evaluation.v90_strict_runtime_release" in launcher
    assert "TRANSFORMERS_OFFLINE=1" in launcher
    assert "HF_HUB_OFFLINE=1" in launcher
    assert "run_v89_strict_scene1_demo.sh" not in launcher
    assert "question-dependent retrieval" in launcher


def test_v90_default_finite_demo_is_exact_six_core_plus_three_smoke() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    quick = launcher.split("elif [[ ${#V90_DEMO_QUESTIONS[@]} -eq 0 ]]; then", maxsplit=1)[1].split(
        "\nfi", maxsplit=1
    )[0]

    assert _quoted_questions(quick) == [
        "Is there a chair?",
        "What color is the bowl?",
        "Is the bowl left or right of the chair?",
        "What is on the table?",
        "What is underneath the table?",
        "What is hanging on the wall?",
        "Where is the red cube?",
        "What object could someone sit on?",
        "Is anything inside the bowl?",
    ]


def test_v90_all_primary_mode_has_exact_thirteen_question_inventory() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    full = launcher.split('if [[ "$V90_DEMO_ALL_PRIMARY" -eq 1 ]]; then', maxsplit=1)[1].split(
        "\nelif", maxsplit=1
    )[0]
    questions = _quoted_questions(full)

    assert len(questions) == 13
    assert len(set(questions)) == 13
    assert questions[0] == "What objects are around you?"
    assert questions[-1] == "Is anything inside the bowl?"


def test_v90_launcher_has_valid_shell_syntax_and_finite_help() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], cwd=ROOT, check=True)
    result = subprocess.run(
        [str(LAUNCHER), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--check" in result.stdout
    assert "--authenticate" in result.stdout
    assert "--interactive" in result.stdout
    assert "--leakage" in result.stdout
    assert "six core actionable questions" in result.stdout


def test_v90_launcher_refuses_conflicting_modes_before_preflight() -> None:
    result = subprocess.run(
        [str(LAUNCHER), "--check", "--interactive"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Choose exactly one" in result.stderr


def test_v90_model_free_checker_refuses_wrong_scene_before_file_reads() -> None:
    with pytest.raises(ValueError, match="only opaque scene_000001"):
        release_check.validate_v90_release(scene_id="scene_with_chair")


def test_v90_model_free_checker_fails_closed_without_release(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="runtime config is unavailable"):
        release_check.validate_v90_release(
            config_path=tmp_path / "missing.yaml",
            checkpoint_path=tmp_path / "missing-checkpoint",
            memory_path=tmp_path / "missing-memory",
            release_report_path=tmp_path / "missing-release.json",
            runtime_cli_source=tmp_path / "missing-cli.py",
            release_source=tmp_path / "missing-release.py",
            require_default_paths=False,
        )


def test_v90_model_free_checker_rejects_symlinked_release_inputs(
    tmp_path: Path,
) -> None:
    real = tmp_path / "runtime.yaml"
    real.write_text("_runtime_safe_config: true\n", encoding="utf-8")
    linked = tmp_path / "linked.yaml"
    linked.symlink_to(real)

    with pytest.raises(ValueError, match="symbolic link"):
        release_check.validate_v90_release(
            config_path=linked,
            checkpoint_path=tmp_path / "missing-checkpoint",
            memory_path=tmp_path / "missing-memory",
            release_report_path=tmp_path / "missing-release.json",
            runtime_cli_source=tmp_path / "missing-cli.py",
            release_source=tmp_path / "missing-release.py",
            require_default_paths=False,
        )


def test_v90_checker_import_graph_is_model_free() -> None:
    source = (ROOT / "scripts" / "check_v90_release.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert not any(name.startswith("torch") for name in imports)
    assert not any(name.startswith("transformers") for name in imports)
    assert not any("v90_strict_runtime_release" in name for name in imports)
    assert not any("v90_strict_scene1_cli" in name for name in imports)


@pytest.mark.skipif(
    not all(
        (ROOT / path).exists()
        for path in (
            release_check.DEFAULT_CONFIG,
            release_check.DEFAULT_CHECKPOINT,
            release_check.DEFAULT_MEMORY,
            release_check.DEFAULT_RELEASE_REPORT,
            release_check.DEFAULT_RUNTIME_CLI_SOURCE,
            release_check.DEFAULT_RELEASE_SOURCE,
        )
    ),
    reason="promoted V90 strict release is not available yet",
)
def test_actual_v90_release_preflight_is_exact_and_model_free() -> None:
    report = release_check.validate_v90_release()

    assert report["passed"] is True
    assert report["loads_model"] is False
    assert report["imports_torch"] is False
    assert report["imports_transformers"] is False
    assert report["frozen_lora_bank_count"] == 12
    assert report["trainable_runtime_parameter_count"] == 0
    assert report["scene_prefix_compiled_before_question"] is True
    assert report["same_exact_scene_prefix_for_every_question"] is True
    assert report["question_dependent_retrieval"] is False
    assert report["environmental_text_inputs"] == []
