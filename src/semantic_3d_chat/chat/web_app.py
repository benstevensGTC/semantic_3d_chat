"""Local web UI for the oracle-isolated, continuous-scene chat runtime.

The server deliberately serves only an inline application shell and an allowlist
of human-facing figures below ``reports/figures``.  It never reads the oracle,
QA supervision, per-frame feature cache, or source render directory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from semantic_3d_chat.chat.file_audit import FileAccessAudit

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_OPAQUE_SCENE_ID = re.compile(r"scene_[0-9]{6}")
_FORBIDDEN_RUNTIME_DIRECTORIES = frozenset({"oracle", "qa", "features", "rendered"})
_VISUAL_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})
_MAX_REQUEST_BYTES = 16 * 1024
_MAX_QUESTION_CHARACTERS = 4_096


class ChatResult(Protocol):
    prefix_hash: str

    def to_dict(self) -> dict[str, Any]: ...


class ChatRuntime(Protocol):
    scene_id: str
    scene_prefix_hash: str

    @property
    def questions_answered(self) -> int: ...

    def answer(self, question: str) -> ChatResult: ...

    def assert_prefix_unchanged(self) -> None: ...

    def startup_summary(self) -> dict[str, Any]: ...


def _rooted(path: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()


def _forbidden_roots(data_root: Path) -> list[Path]:
    return [data_root / name for name in sorted(_FORBIDDEN_RUNTIME_DIRECTORIES)]


def _reject_forbidden_path(path: str | Path, purpose: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    forbidden = {part.casefold() for part in resolved.parts} & _FORBIDDEN_RUNTIME_DIRECTORIES
    if forbidden:
        raise ValueError(f"Refusing {purpose} in a forbidden runtime directory: {resolved}")
    return resolved


def _guard_visual_asset(path: str | Path, figure_root: Path) -> Path:
    """Require a visual to be a non-semantic image inside reports/figures."""

    resolved = _reject_forbidden_path(path, "web visual")
    safe_root = _reject_forbidden_path(figure_root, "web figure root")
    try:
        resolved.relative_to(safe_root)
    except ValueError as exc:
        raise ValueError(f"Web visual is outside the allowlisted figure root: {resolved}") from exc
    if resolved.suffix.casefold() not in _VISUAL_SUFFIXES:
        raise ValueError(f"Web visual must be a supported raster image: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Web visual does not exist: {resolved}")
    return resolved


def validate_visual_assets(
    assets: Mapping[str, str | Path], figure_root: Path
) -> dict[str, Path]:
    """Validate the small, fixed asset vocabulary used by the browser UI."""

    allowed_names = {"overview", "map"}
    unknown = set(assets) - allowed_names
    if unknown:
        raise ValueError(f"Unknown web visual names: {sorted(unknown)}")
    return {
        name: _guard_visual_asset(path, figure_root)
        for name, path in assets.items()
    }


def resolve_visual_assets(
    project_root: Path,
    reports_root: str | Path,
    scene_id: str,
) -> tuple[Path, dict[str, Path]]:
    """Resolve only precomputed report figures; never fall back to runtime data."""

    if not _OPAQUE_SCENE_ID.fullmatch(scene_id):
        raise ValueError("scene_id must be opaque and match scene_ followed by six digits")
    reports_path = _reject_forbidden_path(
        _rooted(reports_root, project_root), "web reports root"
    )
    figure_root = (reports_path / "figures").resolve()
    overview_choices = (figure_root / scene_id / "overview_rgb.png",)
    if scene_id == "scene_000001":
        overview_choices += (figure_root / "scan_montage.png",)
    candidates = {
        "overview": overview_choices,
        "map": (figure_root / scene_id / "map_rgb.png",),
    }
    selected: dict[str, Path] = {}
    for name, choices in candidates.items():
        for candidate in choices:
            if candidate.is_file():
                selected[name] = _guard_visual_asset(candidate, figure_root)
                break
    return figure_root, selected


def _reference_viewpoint(config: Mapping[str, Any]) -> dict[str, Any]:
    runtime_viewpoint = config.get("runtime", {}).get("reference_viewpoint")
    if isinstance(runtime_viewpoint, Mapping):
        return {
            "position_m": [float(value) for value in runtime_viewpoint["position_m"]],
            "yaw_degrees": float(runtime_viewpoint["yaw_degrees"]),
            "pitch_degrees": float(runtime_viewpoint["pitch_degrees"]),
            "scan_view_count": int(runtime_viewpoint["scan_view_count"]),
            "world_axes": {"x": "right", "y": "forward", "z": "up"},
        }
    render = config.get("render", {})
    position = [float(value) for value in render.get("camera_position_m", (0.0, 0.0, 0.0))]
    yaws = [float(value) for value in render.get("yaw_degrees", (0.0,))]
    pitches = [float(value) for value in render.get("pitch_degrees", (0.0,))]
    yaw = min(yaws, key=abs) if yaws else 0.0
    pitch = min(pitches, key=abs) if pitches else 0.0
    return {
        "position_m": position,
        "yaw_degrees": yaw,
        "pitch_degrees": pitch,
        "scan_view_count": len(yaws) * len(pitches),
        "world_axes": {"x": "right", "y": "forward", "z": "up"},
    }


def _public_runtime_state(
    runtime: ChatRuntime,
    config: Mapping[str, Any],
    asset_names: Mapping[str, Path],
    *,
    startup_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    runtime.assert_prefix_unchanged()
    # Freeze the runtime contract captured before the HTTP server accepts any
    # questions.  In particular, a question-conditioned research runtime must
    # never be mislabeled as the strict fixed-prefix primary path merely because
    # both runtimes preserve the same base scene-prefix hash.
    summary = dict(runtime.startup_summary() if startup_summary is None else startup_summary)
    permitted_summary_keys = (
        "scene_id",
        "prefix_hash",
        "prefix_shape",
        "scene_latents",
        "language_hidden_dim",
        "source_voxels",
        "processed_voxels",
        "occupied_blocks",
        "device",
        "prefix_build_seconds",
    )
    public_summary = {key: summary[key] for key in permitted_summary_keys if key in summary}
    strict_fixed = bool(summary.get("strict_fixed_environment_embedding_input", False))
    question_conditioned = summary.get("question_conditioned_scene_readout_tokens")
    question_retrieval = summary.get("question_dependent_scene_retrieval")
    environment_hash = str(
        summary.get("environment_conditioned_input_sha256", runtime.scene_prefix_hash)
    )
    return {
        **public_summary,
        "scene_id": runtime.scene_id,
        "prefix_hash": runtime.scene_prefix_hash,
        "environment_conditioned_input_sha256": environment_hash,
        "questions_answered": runtime.questions_answered,
        "prefix_built_before_questions": bool(
            summary.get("scene_prefix_computed_before_question", False)
        ),
        "strict_fixed_environment_embedding_input": strict_fixed,
        "question_conditioned_scene_readout_tokens": question_conditioned,
        "question_dependent_retrieval": question_retrieval,
        "human_visuals_are_model_inputs": False,
        "viewpoint": _reference_viewpoint(config),
        "visuals": {name: f"/assets/{name}" for name in asset_names},
    }


_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>Semantic 3D Chat</title>
  <style>
    :root { color-scheme: dark; --bg:#07111d; --panel:#0d1b29; --line:#23384a;
      --text:#e6f0f6; --muted:#93a9b8; --aqua:#58ddce; --blue:#72a8ff; --bad:#ff817b; }
    * { box-sizing:border-box; }
    body { margin:0; min-height:100vh; font:15px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;
      color:var(--text); background:radial-gradient(circle at 10% 0,#112b3f 0,var(--bg) 42%); }
    header { display:flex; justify-content:space-between; gap:20px; padding:22px 28px;
      border-bottom:1px solid var(--line); background:#07111dd9; position:sticky; top:0; z-index:2; }
    h1 { font:700 20px/1.2 system-ui,sans-serif; margin:0 0 5px; letter-spacing:.02em; }
    .sub,.muted { color:var(--muted); font-size:12px; }
    .dot { display:inline-block; width:9px; height:9px; border-radius:50%; background:var(--aqua);
      box-shadow:0 0 13px var(--aqua); margin-right:8px; }
    main { max-width:1500px; margin:auto; padding:22px; display:grid;
      grid-template-columns:minmax(0,1.45fr) minmax(340px,.8fr); gap:18px; }
    .panel { background:#0d1b29e8; border:1px solid var(--line); border-radius:14px; overflow:hidden;
      box-shadow:0 18px 50px #0005; }
    .panel-title { padding:12px 15px; border-bottom:1px solid var(--line); color:var(--muted);
      text-transform:uppercase; letter-spacing:.09em; font-size:11px; }
    .visuals { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
    .visual img { display:block; width:100%; aspect-ratio:4/3; object-fit:contain; background:#050b10; }
    .facts { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:1px; background:var(--line); }
    .fact { padding:12px; background:var(--panel); min-width:0; }
    .fact span { display:block; color:var(--muted); font-size:10px; text-transform:uppercase; }
    .fact strong { display:block; color:var(--aqua); margin-top:5px; overflow-wrap:anywhere; font-size:12px; }
    .chat { min-height:670px; display:flex; flex-direction:column; }
    #messages { flex:1; padding:15px; overflow:auto; max-height:580px; }
    .message { margin:0 0 13px; padding:11px 12px; border:1px solid var(--line); border-radius:10px;
      white-space:pre-wrap; overflow-wrap:anywhere; }
    .user { border-color:#365b89; background:#10243b; }
    .assistant { border-color:#216b64; background:#0d2b2a; }
    .ground { color:var(--muted); font-size:11px; margin-top:8px; }
    form { padding:14px; border-top:1px solid var(--line); display:grid; gap:9px; }
    textarea { width:100%; min-height:76px; resize:vertical; border:1px solid #365064; border-radius:9px;
      background:#07121d; color:var(--text); padding:11px; font:inherit; }
    button { justify-self:end; border:0; border-radius:8px; background:var(--aqua); color:#051312;
      font-weight:800; padding:9px 17px; cursor:pointer; }
    button:disabled { opacity:.5; cursor:wait; }
    #error { color:var(--bad); min-height:20px; }
    @media (max-width:950px) { main { grid-template-columns:1fr; } .chat { min-height:520px; } }
    @media (max-width:650px) { .visuals { grid-template-columns:1fr; } .facts { grid-template-columns:1fr 1fr; }
      header { padding:16px; } main { padding:12px; } }
  </style>
</head>
<body>
<header>
  <div><h1>Semantic 3D Chat</h1><div class="sub">continuous full-scene prefix · local inference</div></div>
  <div><span class="dot"></span><span id="status">loading scene memory</span></div>
</header>
<main>
  <section>
    <div class="visuals">
      <article class="panel visual" id="overview-card"><div class="panel-title">Complete RGB scan overview</div><img id="overview" alt="RGB scan overview"></article>
      <article class="panel visual" id="map-card"><div class="panel-title">Fused RGB point-map preview</div><img id="map" alt="Fused RGB point-map preview"></article>
    </div>
    <article class="panel" style="margin-top:18px">
      <div class="panel-title">Question-independent scene state</div>
      <div class="facts">
        <div class="fact"><span>Scene</span><strong id="scene">—</strong></div>
        <div class="fact"><span>Prefix</span><strong id="prefix">—</strong></div>
        <div class="fact"><span>Shape</span><strong id="shape">—</strong></div>
        <div class="fact"><span>Source voxels</span><strong id="voxels">—</strong></div>
        <div class="fact"><span>Reference position (m)</span><strong id="position">—</strong></div>
        <div class="fact"><span>Reference yaw / pitch</span><strong id="orientation">—</strong></div>
      </div>
    </article>
  </section>
  <aside class="panel chat">
    <div class="panel-title">Chat · numeric grounding shown below each answer</div>
    <div id="messages"><div class="message assistant">Scene prefix is ready. Ask a question about the continuous 3D memory.</div></div>
    <form id="chat-form">
      <textarea id="question" maxlength="4096" required placeholder="Ask about the room…"></textarea>
      <div id="error"></div><button id="send" type="submit">Ask locally</button>
    </form>
  </aside>
</main>
<script>
let initialPrefix = null;
const el = id => document.getElementById(id);
function message(kind, text, grounding) {
  const node = document.createElement('div'); node.className = `message ${kind}`; node.textContent = text;
  if (grounding) { const detail = document.createElement('div'); detail.className = 'ground';
    const xyz = grounding.grounding_xyz_m.map(v => Number(v).toFixed(3)).join(', ');
    detail.textContent = `grounding xyz [${xyz}] m · confidence ${Number(grounding.grounding_confidence).toFixed(3)} · support ${Number(grounding.grounding_support_distance_m).toFixed(3)} m · prefix reused ${grounding.prefix_reused}`;
    node.appendChild(detail); }
  el('messages').appendChild(node); el('messages').scrollTop = el('messages').scrollHeight;
}
async function initialize() {
  const state = await (await fetch('/api/state')).json(); initialPrefix = state.prefix_hash;
  el('scene').textContent = state.scene_id; el('prefix').textContent = state.prefix_hash.slice(0,16) + '…';
  el('shape').textContent = JSON.stringify(state.prefix_shape || []); el('voxels').textContent = state.source_voxels;
  el('position').textContent = state.viewpoint.position_m.map(v => Number(v).toFixed(2)).join(', ');
  el('orientation').textContent = `${state.viewpoint.yaw_degrees.toFixed(1)}° / ${state.viewpoint.pitch_degrees.toFixed(1)}°`;
  for (const key of ['overview','map']) { if (state.visuals[key]) el(key).src = state.visuals[key]; else el(`${key}-card`).hidden = true; }
  el('status').textContent = `ready · ${state.scene_latents} latents · prefix built before questions`;
}
el('chat-form').addEventListener('submit', async event => {
  event.preventDefault(); const question = el('question').value.trim(); if (!question) return;
  message('user', question); el('question').value = ''; el('send').disabled = true; el('error').textContent = '';
  try { const response = await fetch('/api/chat', {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify({question})});
    const body = await response.json(); if (!response.ok) throw new Error(body.error || 'request failed');
    if (body.prefix_hash !== initialPrefix) throw new Error('scene prefix changed unexpectedly');
    message('assistant', body.answer, body); el('status').textContent = `ready · ${body.questions_answered} questions · invariant prefix`;
  } catch (error) { el('error').textContent = String(error); }
  finally { el('send').disabled = false; el('question').focus(); }
});
initialize().catch(error => { el('status').textContent = 'startup error'; el('error').textContent = String(error); });
</script>
</body>
</html>
"""


def create_web_app(
    runtime: ChatRuntime,
    config: Mapping[str, Any],
    *,
    visual_assets: Mapping[str, str | Path] | None = None,
    project_root: Path = PROJECT_ROOT,
    audit: FileAccessAudit | None = None,
) -> Starlette:
    """Create an ASGI application around an already initialized scene runtime."""

    reports_root = config.get("paths", {}).get("reports_root", "reports")
    figure_root, discovered = resolve_visual_assets(project_root, reports_root, runtime.scene_id)
    assets = discovered if visual_assets is None else validate_visual_assets(visual_assets, figure_root)
    runtime.assert_prefix_unchanged()
    initial_prefix_hash = runtime.scene_prefix_hash
    startup_summary = dict(runtime.startup_summary())
    runtime_contract = _public_runtime_state(
        runtime,
        config,
        assets,
        startup_summary=startup_summary,
    )
    answer_lock = asyncio.Lock()

    async def index(_request: Request) -> Response:
        return HTMLResponse(
            _PAGE,
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
            },
        )

    async def state(_request: Request) -> Response:
        payload = _public_runtime_state(
            runtime,
            config,
            assets,
            startup_summary=startup_summary,
        )
        payload["prefix_reused"] = runtime.scene_prefix_hash == initial_prefix_hash
        return JSONResponse(payload, headers={"Cache-Control": "no-store"})

    async def chat(request: Request) -> Response:
        try:
            length = int(request.headers.get("content-length", "0"))
        except ValueError:
            length = _MAX_REQUEST_BYTES + 1
        if length > _MAX_REQUEST_BYTES:
            return JSONResponse({"error": "request_too_large"}, status_code=413)
        raw = await request.body()
        if len(raw) > _MAX_REQUEST_BYTES:
            return JSONResponse({"error": "request_too_large"}, status_code=413)
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return JSONResponse({"error": "invalid_json"}, status_code=400)
        question = payload.get("question") if isinstance(payload, dict) else None
        if not isinstance(question, str) or not question.strip():
            return JSONResponse({"error": "question_must_be_nonempty_text"}, status_code=400)
        if len(question) > _MAX_QUESTION_CHARACTERS:
            return JSONResponse({"error": "question_too_long"}, status_code=400)
        try:
            async with answer_lock:
                runtime.assert_prefix_unchanged()
                result = runtime.answer(question)
                runtime.assert_prefix_unchanged()
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        if result.prefix_hash != initial_prefix_hash:
            return JSONResponse({"error": "scene_prefix_changed"}, status_code=500)
        response = result.to_dict()
        response.update(
            {
                "scene_id": runtime.scene_id,
                "environment_conditioned_input_sha256": runtime_contract[
                    "environment_conditioned_input_sha256"
                ],
                "strict_fixed_environment_embedding_input": runtime_contract[
                    "strict_fixed_environment_embedding_input"
                ],
                "question_conditioned_scene_readout_tokens": runtime_contract[
                    "question_conditioned_scene_readout_tokens"
                ],
                "question_dependent_retrieval": runtime_contract[
                    "question_dependent_retrieval"
                ],
                "human_visuals_are_model_inputs": False,
                "prefix_reused": True,
                "questions_answered": runtime.questions_answered,
            }
        )
        return JSONResponse(response, headers={"Cache-Control": "no-store"})

    async def asset(request: Request) -> Response:
        name = str(request.path_params["asset_name"])
        path = assets.get(name)
        if path is None:
            return JSONResponse({"error": "unknown_asset"}, status_code=404)
        safe_path = _guard_visual_asset(path, figure_root)
        if audit is not None:
            audit.record(safe_path)
        return FileResponse(
            safe_path,
            headers={"Cache-Control": "public, max-age=300", "X-Content-Type-Options": "nosniff"},
        )

    app = Starlette(
        debug=False,
        routes=[
            Route("/", index, methods=["GET"]),
            Route("/api/state", state, methods=["GET"]),
            Route("/api/chat", chat, methods=["POST"]),
            Route("/assets/{asset_name:str}", asset, methods=["GET"]),
        ],
    )
    app.state.runtime = runtime
    app.state.initial_prefix_hash = initial_prefix_hash
    app.state.visual_assets = dict(assets)
    return app


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config")
    result.add_argument("--scene", default="scene_000001")
    result.add_argument("--checkpoint")
    result.add_argument("--primary-pointer")
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8765)
    result.add_argument(
        "--allow-network",
        action="store_true",
        help="Permit a non-loopback bind. The default server is local-only.",
    )
    result.add_argument("--audit-log")
    return result


def _run(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not 1 <= args.port <= 65_535:
        raise SystemExit("--port must be between 1 and 65535")
    loopback_hosts = {"127.0.0.1", "localhost", "::1"}
    if args.host not in loopback_hosts and not args.allow_network:
        raise SystemExit("Refusing a non-loopback bind without --allow-network")

    audit_path = _rooted(args.audit_log or "reports/metrics/web_file_access.json")
    default_data_root = PROJECT_ROOT / "data"
    audit = FileAccessAudit(
        _forbidden_roots(default_data_root),
        forbidden_component_names=_FORBIDDEN_RUNTIME_DIRECTORIES,
        block_forbidden=True,
    )
    runtime: ChatRuntime | None = None
    try:
        with audit:
            # Delay heavyweight imports until the file-access audit is active.
            import uvicorn

            from semantic_3d_chat.chat.launch import resolve_chat_launch
            from semantic_3d_chat.chat.runtime import StaticChatRuntime
            from semantic_3d_chat.config import (
                artifact_root,
                reports_root,
            )

            launch = resolve_chat_launch(
                config_path=args.config,
                checkpoint=args.checkpoint,
                primary_pointer=args.primary_pointer,
                audit=audit,
            )
            config = launch.config
            if args.audit_log is None:
                audit_path = reports_root(config) / "metrics" / "web_file_access.json"
            configured_forbidden_roots = [
                artifact_root(config, kind).resolve()
                for kind in sorted(_FORBIDDEN_RUNTIME_DIRECTORIES)
            ]
            for root in configured_forbidden_roots:
                if root not in audit.forbidden_roots:
                    audit.forbidden_roots.append(root)
            launch.verify_scene_map(args.scene, audit=audit)
            runtime = StaticChatRuntime.load(
                config,
                args.scene,
                checkpoint=launch.checkpoint_path,
                audit=audit,
                local_files_only=True,
            )
            launch.verify_scene_prefix(
                args.scene,
                loaded_scene_id=runtime.scene_id,
                prefix_sha256=runtime.scene_prefix_hash,
            )
            app = create_web_app(runtime, config, audit=audit)
            print(
                json.dumps(
                    {
                        "phase": "web_ready",
                        "url": f"http://{args.host}:{args.port}",
                        "scene_id": runtime.scene_id,
                        "prefix_hash": runtime.scene_prefix_hash,
                        "prefix_built_before_questions": True,
                        "behaviorally_promoted": launch.is_production_gemma,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            uvicorn.run(app, host=args.host, port=args.port, log_level="info", access_log=False)
            runtime.assert_prefix_unchanged()
    finally:
        audit.save(audit_path)
    audit.assert_clean()
    if runtime is not None:
        print(
            json.dumps(
                {
                    "phase": "web_audit_complete",
                    "passed": True,
                    "loaded_file_count": len(audit.unique_paths),
                    "prefix_hash": runtime.scene_prefix_hash,
                    "questions_answered": runtime.questions_answered,
                    "audit_log": str(audit_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(argv)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Web chat startup refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "create_web_app",
    "main",
    "resolve_visual_assets",
    "validate_visual_assets",
]
