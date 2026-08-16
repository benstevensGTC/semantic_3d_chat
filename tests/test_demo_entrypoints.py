from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]


def _recipe(makefile: str, target: str) -> str:
    match = re.search(
        rf"(?ms)^{re.escape(target)}(?:\s*:[^\n]*)?\n(?P<body>(?:\t[^\n]*\n|#[^\n]*\n|\n)*)",
        makefile,
    )
    assert match is not None, f"missing Make target: {target}"
    return match.group("body")


def test_semantic_embodied_mcp_has_a_finite_preflight_target() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    recipe = _recipe(makefile, "gemma4-embodied-mcp-check")

    assert "semantic_3d_chat.mcp_server.server" in recipe
    assert '--checkpoint "$(GEMMA4_EMBODIED_CHECKPOINT)"' in recipe
    assert '--runtime-asset "$(RUNTIME_SCENE_ASSET)"' in recipe
    assert '--robot-state-checkpoint "$(GEMMA4_ROBOT_STATE_CHECKPOINT)"' in recipe
    assert '--audit-report "$(GEMMA4_EMBODIED_MCP_CHECK_REPORT)"' in recipe
    assert "--check" in recipe


def test_live_semantic_embodied_mcp_always_loads_robot_state_tokens() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    recipe = _recipe(makefile, "gemma4-embodied-mcp")
    option = '--robot-state-checkpoint "$(GEMMA4_ROBOT_STATE_CHECKPOINT)"'

    assert recipe.count(option) == 1
    # The optional control-checkpoint expansion ends before the runtime asset;
    # state tokens remain mandatory even when that expansion is empty.
    assert option not in recipe.split('--runtime-asset "$(RUNTIME_SCENE_ASSET)"')[0]


def test_demo_check_covers_promoted_static_and_current_embodied_surfaces() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    header = next(line for line in makefile.splitlines() if line.startswith("demo-check:"))
    embodied_header = next(
        line for line in makefile.splitlines() if line.startswith("embodied-check:")
    )

    assert "v89-demo-check" in header
    assert "embodied-check" in header
    assert "v96-embodied-check" in embodied_header
    assert "embodied-demo-check" in embodied_header
    assert "navigation-policy-v3-3-check" not in embodied_header
    assert "held-out navigation acceptance: PENDING" in makefile


def test_mcp_stdio_smoke_output_is_overrideable() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    recipe = _recipe(makefile, "mcp-stdio-smoke")

    assert "MCP_STDIO_SMOKE_OUTPUT ?= reports/metrics/mcp_stdio_transport.json" in makefile
    assert '--output "$(MCP_STDIO_SMOKE_OUTPUT)"' in recipe


def test_strict_web_launcher_only_persists_check_audit_when_explicit(
    tmp_path: Path,
) -> None:
    launcher = ROOT / "scripts" / "run_strict_fixed_prefix_web.sh"
    fake_python = tmp_path / "fake-python"
    capture = tmp_path / "argv.txt"
    fake_python.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" > "$STRICT_WEB_TEST_CAPTURE"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    base = [
        str(launcher),
        "--python",
        str(fake_python),
        "--config",
        str(tmp_path / "gemma4_v54.yaml"),
        "--checkpoint",
        str(tmp_path / "checkpoint"),
    ]
    env = os.environ.copy()
    env.pop("STRICT_WEB_AUDIT", None)
    env["STRICT_WEB_TEST_CAPTURE"] = str(capture)

    def captured_args(*extra: str) -> list[str]:
        subprocess.run(
            [*base, *extra],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        return capture.read_text(encoding="utf-8").splitlines()

    implicit_check = captured_args("--check")
    explicit_path = tmp_path / "explicit-audit.json"
    explicit_check = captured_args("--check", "--audit-log", str(explicit_path))
    normal_serve = captured_args()

    assert "--audit-log" not in implicit_check
    explicit_index = explicit_check.index("--audit-log")
    assert explicit_check[explicit_index + 1] == str(explicit_path)
    normal_index = normal_serve.index("--audit-log")
    assert normal_serve[normal_index + 1] == (
        "reports/gemma4/metrics/strict_prefix_web_access.json"
    )


def test_default_demo_and_leakage_use_promoted_v89_but_preserve_v54_targets() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "run_v89_strict_scene1_demo.sh" in _recipe(makefile, "demo")
    assert "run_v89_strict_scene1_demo.sh" in _recipe(makefile, "demo-smoke")
    assert "run_v89_strict_scene1_demo.sh" in _recipe(makefile, "demo-leakage")
    assert "v89-demo-chat" in next(
        line for line in makefile.splitlines() if line.startswith("chat:")
    )
    assert "run_strict_fixed_prefix_demo.sh" in _recipe(makefile, "strict-demo")
    schema7 = ROOT / "scripts" / "run_schema7_question_control_demo.sh"
    assert schema7.is_file()
    assert "schema-7" in schema7.read_text(encoding="utf-8").casefold()


def test_strict_demo_advertises_finite_semantic_mcp_check_before_server() -> None:
    launcher = (ROOT / "scripts" / "run_strict_fixed_prefix_demo.sh").read_text(encoding="utf-8")

    check = "make gemma4-embodied-mcp-check SCENE="
    server = "make gemma4-embodied-mcp SCENE="
    assert check in launcher
    assert server in launcher
    assert launcher.index(check) < launcher.index(server)


def test_strict_launchers_prepare_the_two_file_runtime_release() -> None:
    for name in ("run_strict_fixed_prefix_demo.sh", "run_strict_fixed_prefix_web.sh"):
        launcher = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "data_gemma4/runtime/checkpoints/gemma4_v54_release_v1" in launcher
        assert "scripts/check_demo_artifacts.py --fast" in launcher
        assert "scripts/prepare_demo_runtime.py" in launcher


def test_prepare_demo_runtime_checks_complete_surface_after_atomic_copy() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    recipe = _recipe(makefile, "prepare-demo-runtime")

    assert "demo-artifacts-check-fast" in recipe
    assert "scripts/prepare_demo_runtime.py" in recipe
    assert recipe.index("scripts/prepare_demo_runtime.py") < recipe.index(
        "demo-artifacts-check-fast"
    )


def test_live_embodied_entrypoints_use_successor_config_not_sealed_v54() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    launcher = (ROOT / "scripts" / "run_embodied_conversation.sh").read_text(encoding="utf-8")
    cli = (ROOT / "src" / "semantic_3d_chat" / "robot" / "conversation_cli.py").read_text(
        encoding="utf-8"
    )

    assert "GEMMA4_EMBODIED_CONFIG ?= configs/runtime/embodied_live.yaml" in makefile
    assert 'EMBODIED_CONFIG="${EMBODIED_CONFIG:-configs/runtime/embodied_live.yaml}"' in launcher
    assert 'default="configs/runtime/embodied_live.yaml"' in cli
