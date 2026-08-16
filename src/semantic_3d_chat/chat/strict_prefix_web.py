"""Loopback-only web UI for the explicit V54 fixed-prefix Gemma baseline.

This is intentionally a research-demonstration launcher, not a promotion
bypass.  It authenticates the exact V54 adapter and sanitized runtime metadata,
builds the complete continuous scene prefix before starting the HTTP server,
and reuses that exact environmental embedding for every question.  It never
opens the adjacent training metadata.  The only browser visual is a precomputed
fused-map raster; it is served to the human and never passed to the language
model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.web_app import ChatRuntime, create_web_app
from semantic_3d_chat.config import PROJECT_ROOT

_OPAQUE_SCENE_ID = re.compile(r"scene_[0-9]{6}")
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_CHECKPOINT_FILES = frozenset({"adapter.safetensors", "runtime_metadata.json"})
_FORBIDDEN_COMPONENT_NAMES = frozenset(
    {"oracle", "qa", "rendered", "features", "scorer_only", "scorer-only"}
)
_V54_ARTIFACT = "v54_semantic_greedy_gate"
_V54_ADAPTER_SHA256 = "6c627f0a0d9efb7100489c24cbd4acafe10456b79280a64a85399a21cb541daf"
_V54_RUNTIME_METADATA_SHA256 = "807515461c71b08c08dfbd08a184a653e791413748530fa69402512eca6f6fdd"


class _PrefixRuntime(ChatRuntime, Protocol):
    def current_prefix_hash(self) -> str: ...


def _rooted(path: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()


def _configured_root(
    config: Mapping[str, Any],
    kind: str,
    *,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    paths = config["paths"]
    if kind == "reports":
        raw = paths["reports_root"]
    else:
        raw = paths.get(f"{kind}_root", Path(str(paths["data_root"])) / kind)
    return _rooted(str(raw), project_root)


def strict_forbidden_roots(
    project_root: Path = PROJECT_ROOT,
    config: Mapping[str, Any] | None = None,
) -> list[Path]:
    """Return inference-data deny roots without blocking runtime source modules."""

    roots = [
        *(project_root / "data" / name for name in ("oracle", "qa", "rendered", "features")),
        project_root / "data_gemma4" / "training",
        project_root / "reports" / "gemma4" / "scorer_only",
    ]
    if config is not None:
        roots.extend(
            _configured_root(config, name, project_root=project_root)
            for name in ("oracle", "qa", "rendered", "features")
        )
        roots.append(_configured_root(config, "reports", project_root=project_root) / "scorer_only")
    return sorted({path.resolve() for path in roots}, key=str)


def _strict_audit(project_root: Path = PROJECT_ROOT) -> FileAccessAudit:
    return FileAccessAudit(
        strict_forbidden_roots(project_root),
        forbidden_component_names=_FORBIDDEN_COMPONENT_NAMES,
        block_forbidden=True,
    )


def _extend_forbidden_roots(
    audit: FileAccessAudit,
    config: Mapping[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
) -> None:
    for root in strict_forbidden_roots(project_root, config):
        if root not in audit.forbidden_roots:
            audit.forbidden_roots.append(root)


def _validate_v54_config(config_path: Path, config: Mapping[str, Any]) -> None:
    if config_path.name != "gemma4_v54.yaml":
        raise ValueError("Strict web requires the explicit gemma4_v54.yaml runtime config")
    if config.get("language", {}).get("backend") != "gemma4":
        raise ValueError("Strict web requires the local Gemma 4 language backend")
    if config.get("vision", {}).get("backend") != "gemma4":
        raise ValueError("Strict web requires the Gemma 4 continuous visual feature contract")
    if int(config.get("scene_encoder", {}).get("global_latents", 0)) != 256:
        raise ValueError("The V54 baseline contract requires 256 full-scene latents")


def _audited_sha256(path: Path, audit: FileAccessAudit) -> str:
    audit.record(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_v54_checkpoint(checkpoint: Path, audit: FileAccessAudit) -> dict[str, Any]:
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        raise FileNotFoundError(f"V54 checkpoint is unavailable or unsafe: {checkpoint}")
    inventory = {item.name for item in checkpoint.iterdir()}
    if inventory != _CHECKPOINT_FILES:
        raise ValueError(f"V54 checkpoint inventory changed: {sorted(inventory)}")
    for name in _CHECKPOINT_FILES:
        path = checkpoint / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"V54 checkpoint member is unavailable or unsafe: {path}")
    # Training metadata is physically absent. Runtime authentication binds the
    # only two files inference is allowed to consume.
    adapter_sha256 = _audited_sha256(checkpoint / "adapter.safetensors", audit)
    runtime_metadata_sha256 = _audited_sha256(checkpoint / "runtime_metadata.json", audit)
    if adapter_sha256 != _V54_ADAPTER_SHA256:
        raise ValueError("V54 adapter digest changed")
    if runtime_metadata_sha256 != _V54_RUNTIME_METADATA_SHA256:
        raise ValueError("V54 sanitized runtime-metadata digest changed")
    return {
        "artifact": _V54_ARTIFACT,
        "runtime_promotion_authorized": False,
        "adapter_sha256": adapter_sha256,
        "runtime_metadata_sha256": runtime_metadata_sha256,
        "training_metadata_opened": False,
    }


def _validate_numeric_map(
    config: Mapping[str, Any],
    scene_id: str,
    audit: FileAccessAudit,
    *,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    path = _configured_root(config, "maps", project_root=project_root) / scene_id / "voxel_map.npz"
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Sanitized numeric voxel map is unavailable or unsafe: {path}")
    audit.record(path)
    return path


def _strict_map_visual(
    config: Mapping[str, Any],
    scene_id: str,
    *,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    path = (
        _configured_root(config, "reports", project_root=project_root)
        / "figures"
        / scene_id
        / "map_rgb.png"
    )
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Fused-map browser preview is unavailable or unsafe: {path}")
    return path.resolve()


def _strict_scan_visual(
    config: Mapping[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Resolve the precomputed human-only complete-scan montage."""

    path = (
        _configured_root(config, "reports", project_root=project_root)
        / "figures"
        / "scan_montage.png"
    )
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Complete-scan browser montage is unavailable or unsafe: {path}")
    return path.resolve()


def create_strict_web_app(
    runtime: _PrefixRuntime,
    config: Mapping[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
    audit: FileAccessAudit | None = None,
):
    """Create the strict UI with only the human-facing fused-map raster enabled."""

    runtime.assert_prefix_unchanged()
    initial_hash = runtime.scene_prefix_hash
    if runtime.questions_answered != 0 or runtime.current_prefix_hash() != initial_hash:
        raise RuntimeError("Complete scene prefix was not finalized before web startup")
    summary = runtime.startup_summary()
    required_contract = {
        "scene_prefix_computed_before_question": True,
        "strict_fixed_environment_embedding_input": True,
        "question_conditioned_scene_readout_tokens": False,
        "question_dependent_scene_retrieval": False,
    }
    for key, expected in required_contract.items():
        if summary.get(key) is not expected:
            raise RuntimeError(f"Strict web runtime contract mismatch for {key}")
    if summary.get("environment_conditioned_input_sha256") != initial_hash:
        raise RuntimeError("Strict web environment hash does not authenticate the fixed prefix")
    map_visual = _strict_map_visual(config, runtime.scene_id, project_root=project_root)
    scan_visual = _strict_scan_visual(config, project_root=project_root)
    app = create_web_app(
        runtime,
        config,
        visual_assets={"overview": scan_visual, "map": map_visual},
        project_root=project_root,
        audit=audit,
    )
    app.state.strict_fixed_environment_embedding_input = True
    app.state.environment_conditioned_input_sha256 = initial_hash
    app.state.question_conditioned_scene_readout_tokens = False
    app.state.human_visuals_are_model_inputs = False
    return app


def _load_runtime_config(path: Path, audit: FileAccessAudit) -> dict[str, Any]:
    from semantic_3d_chat.chat.runtime_config import load_runtime_config

    return load_runtime_config(path, record_file=audit.record)


def _load_static_runtime(
    config: dict[str, Any],
    scene_id: str,
    checkpoint: Path,
    audit: FileAccessAudit,
) -> _PrefixRuntime:
    # Heavy model imports and all model-file reads occur only after the audit is active.
    from semantic_3d_chat.chat.runtime import StaticChatRuntime

    return StaticChatRuntime.load(
        config,
        scene_id,
        checkpoint=checkpoint,
        audit=audit,
        local_files_only=True,
    )


def _serve(app: Any, host: str, port: int) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info", access_log=False)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", required=True)
    result.add_argument("--scene", default="scene_000001")
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8766)
    result.add_argument("--audit-log")
    result.add_argument(
        "--check",
        action="store_true",
        help="Validate the strict V54 inputs without loading Gemma or starting HTTP.",
    )
    return result


def _run(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.host not in _LOOPBACK_HOSTS:
        raise ValueError("Strict web refuses every non-loopback bind")
    if not 1 <= args.port <= 65_535:
        raise ValueError("--port must be between 1 and 65535")
    if _OPAQUE_SCENE_ID.fullmatch(args.scene) is None:
        raise ValueError("scene_id must be opaque and match scene_ followed by six digits")

    config_path = _rooted(args.config)
    checkpoint = _rooted(args.checkpoint)
    audit_path = (
        _rooted(args.audit_log)
        if args.audit_log is not None
        else (
            None if args.check else _rooted("reports/gemma4/metrics/strict_prefix_web_access.json")
        )
    )
    audit = _strict_audit()
    # The adjacent file is retained only for offline reproducibility.  Treat it
    # as forbidden supervision for the entire lifetime of this chat process.
    audit.forbidden_roots.append((checkpoint / "metadata.json").resolve())
    runtime: _PrefixRuntime | None = None
    completed = False
    try:
        with audit:
            config = _load_runtime_config(config_path, audit)
            _extend_forbidden_roots(audit, config)
            _validate_v54_config(config_path, config)
            marker = _validate_v54_checkpoint(checkpoint, audit)
            map_path = _validate_numeric_map(config, args.scene, audit)
            map_visual = _strict_map_visual(config, args.scene)
            scan_visual = _strict_scan_visual(config)
            preflight = {
                "phase": "strict_fixed_prefix_web_preflight",
                "passed": True,
                "scene_id": args.scene,
                "config": str(config_path),
                "checkpoint": str(checkpoint),
                "checkpoint_artifact": marker["artifact"],
                "behavioral_status": "development_checkpoint_acceptance_gate_failed",
                "runtime_promotion_authorized": False,
                "sanitized_numeric_map": str(map_path),
                "human_only_fused_map_preview": str(map_visual),
                "human_only_complete_scan_montage": str(scan_visual),
                "human_visuals_are_model_inputs": False,
                "loopback_only": True,
            }
            print(json.dumps(preflight, sort_keys=True), flush=True)
            if not args.check:
                runtime = _load_static_runtime(config, args.scene, checkpoint, audit)
                if runtime.questions_answered != 0:
                    raise RuntimeError("Runtime accepted questions before web startup")
                prefix_hash = runtime.scene_prefix_hash
                if runtime.current_prefix_hash() != prefix_hash:
                    raise RuntimeError("Scene prefix changed during startup")
                app = create_strict_web_app(runtime, config, audit=audit)
                print(
                    json.dumps(
                        {
                            "phase": "strict_fixed_prefix_web_ready",
                            "url": f"http://{args.host}:{args.port}",
                            "scene_id": runtime.scene_id,
                            "scene_prefix_computed_before_question": True,
                            "strict_fixed_environment_embedding_input": True,
                            "environment_conditioned_input_sha256": prefix_hash,
                            "question_conditioned_scene_readout_tokens": False,
                            "question_dependent_scene_retrieval": False,
                            "environmental_text_inputs": [],
                            "human_visuals_are_model_inputs": False,
                            "behavioral_status": "development_checkpoint_acceptance_gate_failed",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                _serve(app, args.host, args.port)
                runtime.assert_prefix_unchanged()
            completed = True
    finally:
        if audit_path is not None:
            audit.save(audit_path)
    audit.assert_clean()
    print(
        json.dumps(
            {
                "phase": "strict_fixed_prefix_web_audit_complete",
                "passed": completed,
                "strict_fixed_environment_embedding_input": True,
                "environment_conditioned_input_sha256": (
                    None if runtime is None else runtime.scene_prefix_hash
                ),
                "questions_answered": 0 if runtime is None else runtime.questions_answered,
                "loaded_file_count": len(audit.unique_paths),
                "forbidden_access_count": len(audit.forbidden_accesses()),
                "audit_log": None if audit_path is None else str(audit_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(argv)
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"Strict-prefix web refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "create_strict_web_app",
    "main",
    "parser",
    "strict_forbidden_roots",
]
