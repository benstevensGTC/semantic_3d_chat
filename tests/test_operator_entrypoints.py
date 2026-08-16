from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]


def _recipe(makefile: str, target: str) -> str:
    match = re.search(
        rf"(?ms)^{re.escape(target)}(?:\s*:[^\n]*)?\n"
        rf"(?P<body>(?:\t[^\n]*\n|#[^\n]*\n|\n)*)",
        makefile,
    )
    assert match is not None, f"missing Make target: {target}"
    return match.group("body")


def test_default_demo_preflights_then_selects_chat_mode_from_tty() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    recipe = _recipe(makefile, "demo")

    doctor = recipe.index("./scripts/doctor.sh")
    preflight = recipe.index("demo-check")
    interactive = recipe.index("run_v89_strict_scene1_demo.sh --interactive")
    assert doctor < preflight < interactive
    assert 'if [ -t 0 ] && [ -t 1 ]' in recipe
    assert "No interactive TTY detected" in recipe
    finite = recipe.rindex('run_v89_strict_scene1_demo.sh --scene "$(SCENE)"')
    assert interactive < finite
    assert "map_rgb.png" in recipe
    assert "map_rgb.ply" in recipe
    assert "make strict-web" in recipe
    assert "make gemma4-embodied-mcp" in recipe


def test_finite_demo_has_an_explicit_smoke_name() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    recipe = _recipe(makefile, "demo-smoke")

    assert "run_v89_strict_scene1_demo.sh" in recipe
    assert "--interactive" not in recipe


def test_primary_phase_targets_use_gemma_and_legacy_targets_remain_explicit() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    for target in ("build-smoke-map", "semantic-sanity", "generate-dataset", "train"):
        recipe = _recipe(makefile, target)
        assert "$(GEMMA4_PYTHON)" in recipe
        assert "$(GEMMA4_CONFIG)" in recipe

    for target in (
        "legacy-build-smoke-map",
        "legacy-semantic-sanity",
        "legacy-generate-dataset",
        "legacy-train",
        "legacy-evaluate",
        "legacy-chat",
        "legacy-web",
        "legacy-robot",
        "legacy-mcp",
    ):
        assert _recipe(makefile, target)


def test_static_evaluate_requires_references_before_prediction() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    recipe = _recipe(makefile, "gemma4-evaluate-static")

    reference_guard = recipe.index('test -f "$(GEMMA4_STATIC_REFERENCES)"')
    predictor = recipe.index("gemma4-predict-static")
    scorer = recipe.index("semantic_3d_chat.evaluation.run")
    assert reference_guard < predictor < scorer
    assert '--references "$(GEMMA4_STATIC_REFERENCES)"' in recipe


def test_setup_uses_frozen_uv_and_has_a_standard_venv_fallback() -> None:
    setup = (ROOT / "scripts" / "setup.sh").read_text(encoding="utf-8")
    doctor = (ROOT / "scripts" / "doctor.sh").read_text(encoding="utf-8")

    assert "uv sync --frozen" in setup
    assert "python -m venv" not in setup  # Never accidentally target an unresolved name.
    assert '"$python" -m venv .venv' in setup
    assert '"$python" -m venv .venv-gemma4' in setup
    assert 'DOCTOR_PYTHON=.venv/bin/python' in doctor
    assert "python3.12 python3.11 python3" in doctor


def test_canonical_script_delegates_to_primary_demo_without_stack_options() -> None:
    launcher = (ROOT / "scripts" / "run_full_demo.sh").read_text(encoding="utf-8")

    assert 'exec make --no-print-directory demo SCENE="$SCENE"' in launcher
    assert 'exec make --no-print-directory demo-check SCENE="$SCENE"' in launcher
    assert 'exec make --no-print-directory demo-smoke SCENE="$SCENE"' in launcher
    assert 'exec make --no-print-directory demo-leakage SCENE="$SCENE"' in launcher


def test_v89_operator_wrapper_uses_only_the_promoted_strict_runtime() -> None:
    launcher = (ROOT / "scripts" / "run_v89_strict_scene1_demo.sh").read_text(
        encoding="utf-8"
    )

    assert "gemma4_v89_strict_scene1_release_v1" in launcher
    assert "semantic_3d_chat.chat.v89_strict_scene1_cli" in launcher
    assert "semantic_3d_chat.evaluation.v89_strict_runtime_release" in launcher
    assert "TRANSFORMERS_OFFLINE=1" in launcher
    assert "HF_HUB_OFFLINE=1" in launcher
    assert "run_v75_question_control_demo.sh" not in launcher
