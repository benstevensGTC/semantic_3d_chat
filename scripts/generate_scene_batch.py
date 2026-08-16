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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from semantic_3d_chat.config import load_config
from semantic_3d_chat.data.scene_variants import (
    ScenePlan,
    batch_scene_plans,
    batch_scene_splits,
    validate_visibility_evidence,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/multiscene.yaml")
    parser.add_argument("--stage", choices=("generate", "render", "all"), default="all")
    parser.add_argument("--blender", help="Blender executable; defaults to PATH lookup")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--scene", action="append", help="Limit work to an opaque scene ID"
    )
    selection.add_argument(
        "--split",
        action="append",
        choices=("train", "validation", "test"),
        help="Limit work to one or more explicit pair-atomic dataset splits",
    )
    parser.add_argument(
        "--include-deferred-test",
        action="store_true",
        help="Explicitly unlock splits declared in batch.deferred_splits",
    )
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
        "--plan-version",
        str(plan.plan_version),
        "--chair-count",
        str(plan.chair_count),
        "--chair-orientation",
        plan.chair_orientation,
        "--picture-placement",
        plan.picture_placement,
        "--bowl-placement",
        plan.bowl_placement,
        "--book-placement",
        plan.book_placement,
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


def _visibility_evidence_matches(path: Path, scene_id: str) -> bool:
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
        validate_visibility_evidence(evidence)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    return evidence.get("scene_id") == scene_id


def _require_generation_artifacts(
    blend_path: Path, oracle_path: Path, plan: ScenePlan
) -> None:
    """Fail closed when Blender masks an exception behind exit status zero."""

    if not (blend_path.is_file() and _oracle_matches_plan(oracle_path, plan)):
        raise RuntimeError(
            "Blender generation did not produce a validated scene: " f"{plan.scene_id}"
        )


def _require_render_artifacts(
    manifest_path: Path,
    visibility_path: Path,
    plan: ScenePlan,
    *,
    visibility_required: bool,
) -> None:
    """Require the sanitized manifest and, when configured, exact ray evidence."""

    if not (
        manifest_path.is_file()
        and (
            not visibility_required
            or _visibility_evidence_matches(visibility_path, plan.scene_id)
        )
    ):
        raise RuntimeError(
            "Blender rendering did not produce validated RGB-D artifacts: "
            f"{plan.scene_id}"
        )


def _select_plans(
    config: dict[str, Any],
    all_plans: Sequence[ScenePlan],
    *,
    requested_scenes: Sequence[str] | None,
    requested_splits: Sequence[str] | None,
    include_deferred: bool,
) -> list[ScenePlan]:
    """Select plans while requiring an explicit unlock for held-out splits."""

    splits = batch_scene_splits(config, tuple(all_plans))
    raw_deferred = config["batch"].get("deferred_splits", [])
    if isinstance(raw_deferred, str) or not isinstance(raw_deferred, (list, tuple)):
        raise TypeError("batch.deferred_splits must be a list")
    deferred = {str(value) for value in raw_deferred}
    if deferred - {"train", "validation", "test"}:
        raise ValueError(f"Unknown deferred split names: {sorted(deferred)}")
    if deferred and splits is None:
        raise ValueError("batch.deferred_splits requires explicit batch.splits")

    known = {plan.scene_id for plan in all_plans}
    if requested_scenes:
        requested = set(requested_scenes)
        unknown = requested - known
        if unknown:
            raise ValueError(f"Requested scenes are not in the batch plan: {sorted(unknown)}")
        if not include_deferred and splits is not None:
            locked = {
                scene_id
                for split_name in deferred
                for scene_id in splits[split_name]
            }
            forbidden = requested & locked
            if forbidden:
                raise ValueError(
                    "Deferred test scenes require --include-deferred-test: "
                    f"{sorted(forbidden)}"
                )
        return [plan for plan in all_plans if plan.scene_id in requested]

    if requested_splits:
        if splits is None:
            raise ValueError("--split requires explicit batch.splits in the configuration")
        selected_split_names = set(requested_splits)
        locked = selected_split_names & deferred
        if locked and not include_deferred:
            raise ValueError(
                "Deferred splits require --include-deferred-test: "
                f"{sorted(locked)}"
            )
        selected_ids = {
            scene_id
            for split_name in selected_split_names
            for scene_id in splits[split_name]
        }
        return [plan for plan in all_plans if plan.scene_id in selected_ids]

    if include_deferred or not deferred or splits is None:
        return list(all_plans)
    locked_ids = {
        scene_id for split_name in deferred for scene_id in splits[split_name]
    }
    return [plan for plan in all_plans if plan.scene_id not in locked_ids]


def _write_batch_oracle_manifest(
    path: Path,
    plans: Sequence[ScenePlan],
    *,
    base_config: Path,
    splits: Mapping[str, list[str]] | None = None,
    deferred_splits: Sequence[str] = (),
) -> None:
    payload = {
        "schema_version": 2 if splits is not None else 1,
        "base_config": str(base_config.relative_to(PROJECT_ROOT)),
        "scene_count": len(plans),
        "scenes": [
            {"scene_id": plan.scene_id, **plan.oracle_metadata()}
            for plan in plans
        ],
    }
    if splits is not None:
        payload["splits"] = dict(splits)
        payload["deferred_splits"] = list(deferred_splits)
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
    splits = batch_scene_splits(config, tuple(all_plans))
    plans = _select_plans(
        config,
        all_plans,
        requested_scenes=args.scene,
        requested_splits=args.split,
        include_deferred=args.include_deferred_test,
    )
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
        visibility_path = data_root / "oracle" / plan.scene_id / "visibility.json"
        manifest_path = data_root / "rendered" / plan.scene_id / "manifest.json"

        generated_this_run = False
        if args.stage in {"generate", "all"}:
            generation_complete = blend_path.is_file() and _oracle_matches_plan(oracle_path, plan)
            if generation_complete and not args.force:
                print(f"BATCH_SKIP stage=generate scene={plan.scene_id} reason=cache", flush=True)
                skipped += 1
            else:
                _run(_generate_command(blender, base_config, plan), dry_run=args.dry_run)
                # Blender can exit with status zero even when a Python script
                # raises.  Treat the validated artifacts, not its process exit
                # code, as the generation success signal.
                if not args.dry_run:
                    _require_generation_artifacts(blend_path, oracle_path, plan)
                generated += 1
                generated_this_run = not args.dry_run

        if args.stage in {"render", "all"}:
            if not blend_path.is_file() and not args.dry_run and args.stage == "render":
                raise FileNotFoundError(f"Scene must be generated before rendering: {blend_path}")
            visibility_required = bool(
                config["batch"].get("require_visibility_evidence", False)
            )
            render_complete = manifest_path.is_file() and (
                not visibility_required
                or _visibility_evidence_matches(visibility_path, plan.scene_id)
            )
            if render_complete and not args.force and not generated_this_run:
                print(f"BATCH_SKIP stage=render scene={plan.scene_id} reason=cache", flush=True)
                skipped += 1
            else:
                _run(
                    _render_command(blender, base_config, data_root, plan),
                    dry_run=args.dry_run,
                )
                if not args.dry_run:
                    _require_render_artifacts(
                        manifest_path,
                        visibility_path,
                        plan,
                        visibility_required=visibility_required,
                    )
                rendered += 1

    if not args.dry_run:
        manifest_name = str(config["batch"].get("manifest_name", "multiscene"))
        if not manifest_name or Path(manifest_name).name != manifest_name:
            raise ValueError("batch.manifest_name must be one plain path component")
        _write_batch_oracle_manifest(
            data_root / "oracle" / "batches" / f"{manifest_name}.json",
            all_plans,
            base_config=base_config,
            splits=splits,
            deferred_splits=tuple(config["batch"].get("deferred_splits", [])),
        )
    print(
        "BATCH_COMPLETE "
        f"selected={len(plans)} generated={generated} rendered={rendered} "
        f"skipped={skipped} dry_run={str(args.dry_run).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
