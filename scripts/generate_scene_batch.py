#!/usr/bin/env python3
"""Generate and/or render the deterministic ten-scene counterfactual batch."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from semantic_3d_chat.config import load_config
from semantic_3d_chat.data.scene_variants import ScenePlan, batch_scene_plans


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/multiscene.yaml")
    parser.add_argument("--stage", choices=("generate", "render", "all"), default="all")
    parser.add_argument("--blender", help="Blender executable; defaults to PATH lookup")
    parser.add_argument("--scene", action="append", help="Limit work to an opaque scene ID")
    parser.add_argument("--limit", type=int, help="Limit the selected plan for a small smoke run")
    parser.add_argument("--force", action="store_true", help="Regenerate existing artifacts")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without writing")
    return parser


def _resolve_blender(argument: str | None) -> str:
    requested = argument or "blender"
    resolved = shutil.which(requested)
    if resolved is None:
        candidate = Path(requested).expanduser()
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise FileNotFoundError(f"Blender executable is unavailable: {requested}")
        resolved = str(candidate.resolve())
    return resolved


def _base_config_path(config: dict[str, Any]) -> Path:
    raw_path = str(config["batch"].get("base_config", "configs/default.yaml"))
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Batch base configuration is unavailable: {path}")
    # The batch controls are passed as CLI flags. Keeping Blender on the base
    # config preserves scene_000001's original config hash and render manifest.
    if path.name != "default.yaml":
        raise ValueError("batch.base_config must point to the stable default.yaml")
    return path


def _data_root(config: dict[str, Any]) -> Path:
    path = Path(str(config["paths"]["data_root"]))
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _generate_command(
    blender: str,
    base_config: Path,
    plan: ScenePlan,
) -> list[str]:
    command = [
        blender,
        "--background",
        "--python",
        str(PROJECT_ROOT / "blender" / "generate_scene.py"),
        "--",
        "--config",
        str(base_config),
        "--scene",
        plan.scene_id,
        "--seed",
        str(plan.seed),
        "--color-variant",
        plan.color_variant,
        "--layout-variant",
        plan.layout_variant,
    ]
    for instance_id in plan.remove_instance_ids:
        command.extend(("--remove-instance", instance_id))
    if plan.pair_id is not None:
        command.extend(
            (
                "--pair-id",
                plan.pair_id,
                "--paired-scene",
                str(plan.paired_scene_id),
                "--change-type",
                str(plan.change_type),
                "--pair-role",
                str(plan.pair_role),
            )
        )
    return command


def _render_command(
    blender: str,
    base_config: Path,
    data_root: Path,
    plan: ScenePlan,
) -> list[str]:
    return [
        blender,
        "--background",
        str(data_root / "oracle" / plan.scene_id / "scene.blend"),
        "--python",
        str(PROJECT_ROOT / "blender" / "render_scan.py"),
        "--",
        "--config",
        str(base_config),
        "--scene",
        plan.scene_id,
    ]


def _run(command: Sequence[str], *, dry_run: bool) -> None:
    print(f"BATCH_COMMAND {shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _oracle_matches_plan(path: Path, plan: ScenePlan) -> bool:
    try:
        oracle = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if oracle.get("scene_id") != plan.scene_id or oracle.get("seed") != plan.seed:
        return False
    if plan.is_default_scene_one:
        return "generation" not in oracle
    generation = oracle.get("generation", {})
    expected = plan.oracle_metadata()
    return generation == expected and oracle.get("validation", {}).get("inside_room") is True


def _write_batch_oracle_manifest(
    path: Path,
    plans: Sequence[ScenePlan],
    *,
    base_config: Path,
) -> None:
    payload = {
        "schema_version": 1,
        "base_config": str(base_config.relative_to(PROJECT_ROOT)),
        "scene_count": len(plans),
        "scenes": [
            {"scene_id": plan.scene_id, **plan.oracle_metadata()}
            for plan in plans
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    all_plans = list(batch_scene_plans(config))
    expected_count = int(config["batch"].get("expected_scene_count", len(all_plans)))
    if len(all_plans) != expected_count:
        raise ValueError(f"Expected {expected_count} batch scenes, found {len(all_plans)}")
    plans = list(all_plans)
    if args.scene:
        requested = set(args.scene)
        available = {plan.scene_id for plan in plans}
        unknown = requested - available
        if unknown:
            raise ValueError(f"Requested scenes are not in the batch plan: {sorted(unknown)}")
        plans = [plan for plan in plans if plan.scene_id in requested]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        plans = plans[: args.limit]
    if not plans:
        raise ValueError("No scenes selected")

    blender = _resolve_blender(args.blender)
    base_config = _base_config_path(config)
    data_root = _data_root(config)
    generated = rendered = skipped = 0
    for plan in plans:
        oracle_path = data_root / "oracle" / plan.scene_id / "oracle.json"
        blend_path = data_root / "oracle" / plan.scene_id / "scene.blend"
        manifest_path = data_root / "rendered" / plan.scene_id / "manifest.json"

        generated_this_run = False
        if args.stage in {"generate", "all"}:
            generation_complete = blend_path.is_file() and _oracle_matches_plan(oracle_path, plan)
            if generation_complete and not args.force:
                print(f"BATCH_SKIP stage=generate scene={plan.scene_id} reason=cache", flush=True)
                skipped += 1
            else:
                _run(_generate_command(blender, base_config, plan), dry_run=args.dry_run)
                generated += 1
                generated_this_run = not args.dry_run

        if args.stage in {"render", "all"}:
            if not blend_path.is_file() and not args.dry_run and args.stage == "render":
                raise FileNotFoundError(f"Scene must be generated before rendering: {blend_path}")
            if manifest_path.is_file() and not args.force and not generated_this_run:
                print(f"BATCH_SKIP stage=render scene={plan.scene_id} reason=cache", flush=True)
                skipped += 1
            else:
                _run(
                    _render_command(blender, base_config, data_root, plan),
                    dry_run=args.dry_run,
                )
                rendered += 1

    if not args.dry_run:
        _write_batch_oracle_manifest(
            data_root / "oracle" / "batches" / "multiscene.json",
            all_plans,
            base_config=base_config,
        )
    print(
        "BATCH_COMPLETE "
        f"selected={len(plans)} generated={generated} rendered={rendered} "
        f"skipped={skipped} dry_run={str(args.dry_run).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
