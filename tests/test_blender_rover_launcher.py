from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
LAUNCHER = ROOT / "scripts/run_blender_rover_demo.sh"


def _source() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def test_launcher_is_executable_and_valid_shell() -> None:
    assert os.access(LAUNCHER, os.X_OK)
    result = subprocess.run(
        ["bash", "-n", str(LAUNCHER)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_help_is_finite_and_does_not_require_models_or_blender() -> None:
    result = subprocess.run(
        [str(LAUNCHER), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0
    assert "Blender's real 3D viewport" in result.stdout
    assert "--check" in result.stdout
    assert "--navigation-checkpoint PATH" in result.stdout


def test_launcher_rejects_non_loopback_and_invalid_scene_before_launch() -> None:
    non_loopback = subprocess.run(
        [str(LAUNCHER), "--host", "0.0.0.0"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    invalid_scene = subprocess.run(
        [str(LAUNCHER), "--scene", "chair_room"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert non_loopback.returncode == 2
    assert "only to 127.0.0.1" in non_loopback.stderr
    assert invalid_scene.returncode == 2
    assert "scene_ followed by six digits" in invalid_scene.stderr


def test_launcher_rejects_ambiguous_or_unavailable_explicit_checkpoint() -> None:
    missing_value = subprocess.run(
        [str(LAUNCHER), "--navigation-checkpoint"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    duplicate = subprocess.run(
        [
            str(LAUNCHER),
            "--navigation-checkpoint",
            "first",
            "--navigation-checkpoint",
            "second",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    unavailable = subprocess.run(
        [str(LAUNCHER), "--navigation-checkpoint", "does-not-exist"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert missing_value.returncode == 2
    assert "requires a nonempty path" in missing_value.stderr
    assert duplicate.returncode == 2
    assert "only once" in duplicate.stderr
    assert unavailable.returncode == 2
    assert "not a regular directory" in unavailable.stderr


def test_check_path_is_model_free_and_precedes_process_launch() -> None:
    source = _source()
    authenticated_preflight = source.index("authenticated_preflight_identity()")
    check_branch = source.index('if [[ "$BLENDER_ROVER_CHECK" == true ]]')
    backend_start = source.index(
        'echo "Starting local Gemma and the continuous 3D scene memory', check_branch
    )
    blender_start = source.rindex('"$BLENDER_ROVER_BLENDER" \\')

    assert "--check" in source[authenticated_preflight:check_branch]
    assert '"loads_model":false' in source[check_branch:backend_start]
    assert '"starts_backend":false' in source[check_branch:backend_start]
    assert '"starts_blender":false' in source[check_branch:backend_start]
    assert authenticated_preflight < check_branch < backend_start < blender_start


def test_launcher_uses_authenticated_loopback_api_and_exact_blender_contract() -> None:
    source = _source()

    assert 'BLENDER_ROVER_URL="http://$BLENDER_ROVER_HOST:$BLENDER_ROVER_PORT"' in source
    assert '"$BLENDER_ROVER_URL/api/state"' in source
    assert 'state.get("scene_id") != sys.argv[2]' in source
    assert '"high_level_natural_language_only": True' in source
    assert '"task_trained_navigation": True' in source
    assert '"model_selects_every_waypoint_and_heading": True' in source
    assert '"model_selects_stop": True' in source
    assert '"deterministic_route_planner_used": False' in source
    assert '"fallback_used": False' in source
    assert '"substitution_applied": False' in source
    assert '"synthetic_stop_applied": False' in source
    assert (
        'control.get("navigation_control_mode") != '
        '"actual_local_gemma_model_only_waypoint_policy"'
    ) in source
    assert 'control.get("navigation_checkpoint_sha256") != sys.argv[3]' in source
    assert 'control.get("gemma_runtime_binding_sha256") != sys.argv[4]' in source
    assert 'payload.get("navigation_checkpoint_sha256")' in source
    assert 'payload.get("gemma_runtime_binding_sha256")' in source
    assert 'backend.get("navigation_checkpoint_sha256") != checkpoint' in source
    assert 'backend.get("gemma_runtime_binding_sha256") != binding' in source
    assert '"camera_control_input": False' in source
    assert 'memory.get("tensor_shape") != [1, 258, 1536]' in source
    assert "--disable-autoexec" in source
    assert '"$BLENDER_ROVER_ASSET"' in source
    assert '--python "$BLENDER_ROVER_ADDON"' in source
    assert '--backend-url "$BLENDER_ROVER_URL"' in source
    assert '--project-root "$BLENDER_ROVER_ROOT"' in source


def test_launcher_has_scoped_port_and_child_cleanup_behavior() -> None:
    source = _source()

    assert "port_is_listening" in source
    assert "occupied by a different or unready service" in source
    assert 'kill -TERM "$BLENDER_ROVER_PID"' in source
    assert 'kill -KILL "$BLENDER_ROVER_PID"' in source
    assert "killall" not in source
    assert "pkill" not in source
    assert 'BLENDER_ROVER_OWNS_BACKEND=false' in source
    assert 'if [[ "$BLENDER_ROVER_OWNS_BACKEND" == true' in source


def test_make_exposes_preferred_real_3d_targets() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "BLENDER_ROVER_BACKEND_TIMEOUT ?= 900" in makefile
    assert "blender-rover-demo-check:" in makefile
    assert "run_blender_rover_demo.sh --check" in makefile
    assert "blender-rover-demo:" in makefile
    assert "run_blender_rover_demo.sh --scene" in makefile
    assert "rover-3d-check: blender-rover-demo-check" in makefile
    assert "rover-3d: blender-rover-demo" in makefile


def test_readme_leads_with_blender_as_the_real_3d_operator_surface() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "### Global-map semantic-goal Blender rover" in readme
    assert "make rover-3d" in readme
    assert "Gemma Rover" in readme
    assert "Closing Blender also" in readme


def test_live_launcher_starts_fake_backend_forwards_contract_and_cleans_child(
    tmp_path: Path,
) -> None:
    scripts = tmp_path / "scripts"
    addon_dir = tmp_path / "blender"
    asset_dir = tmp_path / "data/runtime_assets/scene_000001"
    python_dir = tmp_path / ".venv-gemma4/bin"
    for directory in (scripts, addon_dir, asset_dir, python_dir):
        directory.mkdir(parents=True, exist_ok=True)
    navigation_checkpoint = tmp_path / "candidate checkpoint [explicit]"
    navigation_checkpoint.mkdir()
    (navigation_checkpoint / "policy.safetensors").write_bytes(b"candidate-policy")
    (navigation_checkpoint / "runtime_metadata.json").write_text("{}\n", encoding="utf-8")
    launcher = scripts / LAUNCHER.name
    shutil.copy2(LAUNCHER, launcher)
    (addon_dir / "rover_control_ui.py").write_text("READY = True\n", encoding="utf-8")
    (asset_dir / "s_000001.blend").write_bytes(b"sanitized-room")
    (python_dir / "python").symlink_to(sys.executable)

    server = scripts / "fake_server.py"
    server.write_text(
        """
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

PAYLOAD = json.dumps({
    "state": {
        "scene_id": "scene_000001", "scene_prefix_hash": "0" * 64,
        "map_version": 0, "scan_count": 0, "action_count": 0,
    },
    "control": {
        "local_inference": True, "cloud_model_used": False,
        "high_level_natural_language_only": True,
        "task_trained_navigation": True,
        "model_selects_every_waypoint_and_heading": True,
        "model_selects_stop": True,
        "deterministic_route_planner_used": False,
        "fallback_used": False,
        "substitution_applied": False,
        "synthetic_stop_applied": False,
        "control_mode": "actual_local_gemma_model_only_waypoint_policy",
        "navigation_control_mode": "actual_local_gemma_model_only_waypoint_policy",
        "navigation_checkpoint_sha256": os.environ.get("FAKE_NAVIGATION_SHA", "d" * 64),
        "gemma_runtime_binding_sha256": os.environ.get("FAKE_BINDING_SHA", "e" * 64),
        "untrained_json_backend_enabled": False,
        "static_precomputed_scene_memory": True,
        "camera_control_input": False,
    },
    "scene_memory": {
        "tensor_shape": [1, 258, 1536], "map_version": 0,
        "question_dependent_scene_retrieval": False,
        "all_runtime_voxels_encoded": True,
    },
}).encode()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/api/state":
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(PAYLOAD)))
        self.end_headers()
        self.wfile.write(PAYLOAD)
    def log_message(self, *args):
        return

HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
""".lstrip(),
        encoding="utf-8",
    )
    backend = scripts / "run_local_rover_demo.sh"
    backend.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
check=false
port=8770
checkpoint=""
while (($#)); do
  case "$1" in
    --check) check=true; shift ;;
    --port) port="$2"; shift 2 ;;
    --navigation-checkpoint) checkpoint="$2"; shift 2 ;;
    *) shift ;;
  esac
done
printf '%s\t%s\n' "$check" "$checkpoint" >>"$FAKE_BACKEND_CALLS"
if [[ "$check" == true ]]; then
  printf '{"artifact":"semantic_3d_chat_local_rover_demo_preflight_v1","passed":true,"scene_id":"scene_000001","navigation_checkpoint_sha256":"%s","gemma_runtime_binding_sha256":"%s","backend_preflight":{"ready":true,"navigation_checkpoint":"%s","navigation_checkpoint_sha256":"%s","gemma_runtime_binding_sha256":"%s"}}\n' \
    "$(printf 'd%.0s' {1..64})" "$(printf 'e%.0s' {1..64})" "$checkpoint" \
    "$(printf 'd%.0s' {1..64})" "$(printf 'e%.0s' {1..64})"
  exit 0
fi
exec "$ROVER_DEMO_PYTHON" "$FAKE_SERVER" "$port"
""",
        encoding="utf-8",
    )
    backend.chmod(0o755)
    blender_args = tmp_path / "blender_args.txt"
    backend_calls = tmp_path / "backend_calls.txt"
    blender = tmp_path / "fake_blender"
    blender.write_text(
        """#!/usr/bin/env bash
printf '%s\n' "$@" >"$FAKE_BLENDER_ARGS"
""",
        encoding="utf-8",
    )
    blender.chmod(0o755)

    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    env = {
        **os.environ,
        "BLENDER": str(blender),
        "ROVER_DEMO_PYTHON": str(python_dir / "python"),
        "FAKE_SERVER": str(server),
        "FAKE_BLENDER_ARGS": str(blender_args),
        "FAKE_BACKEND_CALLS": str(backend_calls),
        "TMPDIR": str(tmp_path / "runtime"),
    }
    result = subprocess.run(
        [
            str(launcher),
            "--port",
            str(port),
            "--backend-timeout",
            "10",
            "--navigation-checkpoint",
            str(navigation_checkpoint),
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert "Local Gemma rover backend is ready" in result.stdout
    assert "Stopping the launcher-owned" in result.stdout
    observed = blender_args.read_text(encoding="utf-8").splitlines()
    assert observed == [
        "--disable-autoexec",
        str(asset_dir / "s_000001.blend"),
        "--python",
        str(addon_dir / "rover_control_ui.py"),
        "--",
        "--backend-url",
        f"http://127.0.0.1:{port}",
        "--project-root",
        str(tmp_path),
    ]
    assert backend_calls.read_text(encoding="utf-8").splitlines() == [
        f"true\t{navigation_checkpoint}",
        f"false\t{navigation_checkpoint}",
    ]
    with socket.socket() as probe:
        probe.settimeout(0.5)
        assert probe.connect_ex(("127.0.0.1", port)) != 0

    # Omitting the flag must remain a distinct, working passthrough mode. This
    # specifically guards macOS Bash 3.2 under `set -u`, where expanding an
    # empty array would abort before authenticated preflight.
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        default_port = reservation.getsockname()[1]
    default_result = subprocess.run(
        [
            str(launcher),
            "--port",
            str(default_port),
            "--backend-timeout",
            "10",
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert default_result.returncode == 0, default_result.stderr
    assert backend_calls.read_text(encoding="utf-8").splitlines() == [
        f"true\t{navigation_checkpoint}",
        f"false\t{navigation_checkpoint}",
        "true\t",
        "false\t",
    ]
    with socket.socket() as probe:
        probe.settimeout(0.5)
        assert probe.connect_ex(("127.0.0.1", default_port)) != 0

    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        stale_port = reservation.getsockname()[1]
    stale_env = {**env, "FAKE_NAVIGATION_SHA": "f" * 64}
    stale_server = subprocess.Popen(
        [str(python_dir / "python"), str(server), str(stale_port)],
        cwd=tmp_path,
        env=stale_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", stale_port)) == 0:
                    break
            time.sleep(0.02)
        else:
            pytest.fail("stale fake backend did not start")
        stale = subprocess.run(
            [
                str(launcher),
                "--port",
                str(stale_port),
                "--backend-timeout",
                "2",
                "--navigation-checkpoint",
                str(navigation_checkpoint),
            ],
            cwd=tmp_path,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    finally:
        stale_server.terminate()
        stale_server.wait(timeout=5)

    assert stale.returncode == 2
    assert "occupied by a different or unready service" in stale.stderr
