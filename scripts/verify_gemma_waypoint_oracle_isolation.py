#!/usr/bin/env python3
"""Run the production Gemma rover while every runtime oracle input is absent.

The parent process creates a minimal logical project root containing only the
selected runtime YAML files, sanitized checkpoints, opaque Blender asset, and
continuous voxel map.  It deliberately creates no oracle, QA, training-data,
question, prediction, or scorer directory.  A fresh child process patches the
project root before importing the rover runtime, installs a process-wide read
audit, and either performs a model-free preflight or a finite actual-Gemma
smoke. During that complete child execution, the parent also atomically renames
the exact source ``data/oracle`` directory to a unique sibling and restores it
in ``finally``. No oracle file or directory is deleted.

The live smoke builds the full scene prefix before any question, asks two
questions without changing the scene, then requests one bounded waypoint-policy
decision.  Navigation success is not an acceptance condition here; this script
tests isolation and prefix invariance, while the separate live rover verifier
tests task geometry.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

SOURCE_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT: Final[Path] = (
    SOURCE_PROJECT_ROOT
    / "reports/gemma4/metrics/gemma_waypoint_oracle_isolation.json"
)
_SCENE_ID: Final[re.Pattern[str]] = re.compile(r"scene_[0-9]{6}")
_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_COMPONENTS: Final[frozenset[str]] = frozenset(
    {"oracle", "qa", "scorer", "scorer_only", "scorer-only"}
)
_CHECKPOINT_INVENTORIES: Final[dict[str, frozenset[str]]] = {
    "base": frozenset({"adapter.safetensors", "runtime_metadata.json"}),
    "control": frozenset({"control.safetensors", "runtime_metadata.json"}),
    "robot_state": frozenset({"state.safetensors", "runtime_metadata.json"}),
    "navigation": frozenset({"policy.safetensors", "runtime_metadata.json"}),
}


def _has_forbidden_component(path: Path) -> bool:
    return any(
        component.casefold() in _FORBIDDEN_COMPONENTS
        or component.casefold().startswith(".oracle-unavailable-")
        for component in path.parts
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _safe_source(path: str | Path, *, purpose: str, directory: bool = False) -> Path:
    raw = Path(path).expanduser()
    candidate = (raw if raw.is_absolute() else SOURCE_PROJECT_ROOT / raw).resolve()
    if _has_forbidden_component(candidate):
        raise ValueError(f"{purpose} entered a forbidden supervision path")
    if candidate.is_symlink():
        raise ValueError(f"{purpose} cannot be a symbolic link")
    predicate = candidate.is_dir if directory else candidate.is_file
    if not predicate():
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"{purpose} {kind} is unavailable: {candidate}")
    return candidate


def _link_file(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Isolated-runtime source must be a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno not in {errno.EXDEV, errno.EPERM, errno.ENOTSUP}:
            raise
        shutil.copy2(source, destination)
    if destination.is_symlink() or not destination.is_file():
        raise RuntimeError(f"Failed to materialize isolated runtime file: {destination}")


def _link_checkpoint(source: Path, destination: Path, *, kind: str) -> None:
    expected = _CHECKPOINT_INVENTORIES[kind]
    observed = {item.name for item in source.iterdir()}
    if observed != expected:
        raise ValueError(
            f"{kind} checkpoint inventory changed: "
            f"expected={sorted(expected)} observed={sorted(observed)}"
        )
    for name in sorted(expected):
        _link_file(source / name, destination / name)


def _absence_paths(runtime_root: Path) -> tuple[Path, ...]:
    return (
        runtime_root / "data/oracle",
        runtime_root / "data/qa",
        runtime_root / "data_gemma4/oracle",
        runtime_root / "data_gemma4/qa",
        runtime_root / "data_gemma4/training",
        runtime_root / "reports/gemma4/questions",
        runtime_root / "reports/gemma4/predictions",
        runtime_root / "reports/gemma4/scorer_only",
    )


def _assert_absent_runtime_trees(runtime_root: Path) -> None:
    present = [str(path) for path in _absence_paths(runtime_root) if path.exists()]
    if present:
        raise RuntimeError(f"Isolated runtime contains forbidden trees: {present}")


@contextmanager
def _source_oracle_temporarily_unavailable(
    project_root: str | Path,
) -> Iterator[dict[str, Any]]:
    """Atomically hide only ``<project>/data/oracle`` and always restore it.

    The strict target checks make this unsuitable for any broader directory.
    SIGINT, SIGTERM, and SIGHUP are converted to an exception while the oracle
    is hidden so normal Python unwinding reaches the restoration block. SIGKILL
    and machine loss cannot be recovered from by any in-process mechanism.
    """

    raw_root = Path(project_root).expanduser()
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise ValueError("Oracle-isolation project root must be a real directory")
    root = raw_root.resolve()
    data = root / "data"
    oracle = data / "oracle"
    if (
        data.is_symlink()
        or not data.is_dir()
        or oracle.is_symlink()
        or not oracle.is_dir()
        or oracle.name != "oracle"
        or oracle.parent != data
        or data.parent != root
    ):
        raise ValueError(
            "Physical oracle isolation requires the exact real <project>/data/oracle directory"
        )
    resolved_oracle = oracle.resolve()
    if resolved_oracle != root / "data/oracle":
        raise ValueError("Resolved oracle-isolation target differs from <project>/data/oracle")
    hidden = data / f".oracle-unavailable-waypoint-{os.getpid()}-{uuid.uuid4().hex}"
    if hidden.exists() or hidden.is_symlink():
        raise FileExistsError("Unique hidden oracle destination unexpectedly exists")

    state: dict[str, Any] = {
        "original": str(resolved_oracle),
        "hidden": str(hidden),
        "renamed": False,
        "unavailable_during_child": False,
        "restored": False,
    }
    previous_handlers: dict[signal.Signals, Any] = {}

    def interrupt(signum: int, _frame: Any) -> None:
        raise InterruptedError(
            f"Oracle-isolated child interrupted by signal {signal.Signals(signum).name}"
        )

    for candidate in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if candidate is None:
            continue
        try:
            previous = signal.getsignal(candidate)
            signal.signal(candidate, interrupt)
            previous_handlers[candidate] = previous
        except ValueError:
            # Signal handlers can be installed only by the main thread. The
            # production CLI is main-threaded; unit-level callers still retain
            # the unconditional finally restoration below.
            for installed, handler in previous_handlers.items():
                signal.signal(installed, handler)
            previous_handlers.clear()
            break

    try:
        os.rename(resolved_oracle, hidden)
        state["renamed"] = True
        state["unavailable_during_child"] = (
            not resolved_oracle.exists() and hidden.is_dir() and not hidden.is_symlink()
        )
        if state["unavailable_during_child"] is not True:
            raise RuntimeError("Oracle directory remained available after atomic rename")
        yield state
    finally:
        # Ignore a second termination signal only during the tiny critical
        # restoration window; restore every previous handler immediately after.
        for candidate in previous_handlers:
            signal.signal(candidate, signal.SIG_IGN)
        try:
            hidden_holds_original = hidden.is_dir() and not hidden.is_symlink()
            if state["renamed"] or hidden_holds_original:
                if resolved_oracle.exists() or resolved_oracle.is_symlink():
                    raise RuntimeError(
                        "Cannot restore oracle because its exact original path was recreated; "
                        f"the original directory remains preserved at {hidden}"
                    )
                if not hidden_holds_original:
                    raise RuntimeError(
                        "Cannot restore oracle because the hidden original is unavailable"
                    )
                os.rename(hidden, resolved_oracle)
                state["restored"] = (
                    resolved_oracle.is_dir()
                    and not resolved_oracle.is_symlink()
                    and not hidden.exists()
                )
                if state["restored"] is not True:
                    raise RuntimeError("Oracle directory restoration could not be verified")
        finally:
            for candidate, handler in previous_handlers.items():
                signal.signal(candidate, handler)


def _materialize_runtime_copy(args: argparse.Namespace, runtime_root: Path) -> dict[str, str]:
    if _SCENE_ID.fullmatch(args.scene) is None:
        raise ValueError("--scene must be opaque and match scene_ followed by six digits")
    config = _safe_source(args.config, purpose="embodied runtime config")
    control_config = _safe_source(args.control_config, purpose="control runtime config")
    base = _safe_source(args.base_checkpoint, purpose="base checkpoint", directory=True)
    control = _safe_source(
        args.control_checkpoint,
        purpose="control checkpoint",
        directory=True,
    )
    robot = _safe_source(
        args.robot_state_checkpoint,
        purpose="robot-state checkpoint",
        directory=True,
    )
    navigation = _safe_source(
        args.navigation_checkpoint,
        purpose="navigation checkpoint",
        directory=True,
    )
    source_map = _safe_source(
        args.map
        or f"data_gemma4/maps/{args.scene}/voxel_map.npz",
        purpose="continuous semantic map",
    )
    asset = _safe_source(
        args.runtime_asset
        or f"data/runtime_assets/{args.scene}/{args.scene.replace('scene_', 's_', 1)}.blend",
        purpose="opaque runtime Blender asset",
    )
    asset_manifest = _safe_source(
        asset.with_suffix(".json"),
        purpose="sanitized runtime asset manifest",
    )
    render_script = _safe_source(
        "blender/render_runtime_observation.py",
        purpose="sanitized runtime render script",
    )

    # Runtime YAML inheritance stays self-contained, but no experiment or
    # evaluation configuration is copied into the child root.
    source_runtime_configs = SOURCE_PROJECT_ROOT / "configs/runtime"
    destination_runtime_configs = runtime_root / "configs/runtime"
    for source in sorted(source_runtime_configs.glob("*")):
        if source.is_file() and not source.is_symlink():
            _link_file(source, destination_runtime_configs / source.name)
    isolated_config = destination_runtime_configs / config.name
    isolated_control_config = destination_runtime_configs / control_config.name
    if not isolated_config.is_file() or not isolated_control_config.is_file():
        raise RuntimeError("Selected runtime configuration was not copied")

    checkpoint_root = runtime_root / "data_gemma4/runtime_checkpoints"
    isolated_base = checkpoint_root / "base"
    isolated_control = checkpoint_root / "control"
    isolated_robot = checkpoint_root / "robot_state"
    isolated_navigation = checkpoint_root / "navigation"
    _link_checkpoint(base, isolated_base, kind="base")
    _link_checkpoint(control, isolated_control, kind="control")
    _link_checkpoint(robot, isolated_robot, kind="robot_state")
    _link_checkpoint(navigation, isolated_navigation, kind="navigation")

    isolated_map = runtime_root / "data_gemma4/maps" / args.scene / "voxel_map.npz"
    isolated_asset = (
        runtime_root
        / "data/runtime_assets"
        / args.scene
        / f"{args.scene.replace('scene_', 's_', 1)}.blend"
    )
    _link_file(source_map, isolated_map)
    _link_file(asset, isolated_asset)
    _link_file(asset_manifest, isolated_asset.with_suffix(".json"))
    _link_file(render_script, runtime_root / "blender/render_runtime_observation.py")
    (runtime_root / "data").mkdir(parents=True, exist_ok=True)
    (runtime_root / "reports/gemma4/metrics").mkdir(parents=True, exist_ok=True)
    _assert_absent_runtime_trees(runtime_root)
    return {
        "runtime_root": str(runtime_root),
        "source_project_root": str(SOURCE_PROJECT_ROOT),
        "config": str(isolated_config),
        "control_config": str(isolated_control_config),
        "scene": args.scene,
        "base_checkpoint": str(isolated_base),
        "control_checkpoint": str(isolated_control),
        "runtime_asset": str(isolated_asset),
        "robot_state_checkpoint": str(isolated_robot),
        "navigation_checkpoint": str(isolated_navigation),
        "persistent_map": str(
            runtime_root / "data_gemma4/robot" / args.scene / "semantic_map.npz"
        ),
        "runtime_audit": str(runtime_root / "reports/gemma4/metrics/runtime_reads.json"),
        "mode": "preflight" if args.check else "live",
        "questions": list(args.question),
        "decision_instruction": args.decision_instruction,
        "decision_steps": args.decision_steps,
    }


class _WholeProcessReadAudit:
    """Record reads from before the first production-runtime import."""

    def __init__(self, runtime_root: Path, source_project_root: Path) -> None:
        self._lock = threading.Lock()
        self.paths: list[str] = []
        roots: list[Path] = []
        for root in (runtime_root, source_project_root):
            roots.extend(
                (
                    root / "data/oracle",
                    root / "data/qa",
                    root / "data_gemma4/oracle",
                    root / "data_gemma4/qa",
                    root / "data_gemma4/training",
                    root / "reports/gemma4/questions",
                    root / "reports/gemma4/predictions",
                    root / "reports/gemma4/scorer_only",
                )
            )
        self.forbidden_roots = tuple(path.resolve() for path in roots)
        self.forbidden_attempts: list[str] = []
        sys.addaudithook(self._hook)

    def _forbidden(self, path: Path) -> bool:
        if _has_forbidden_component(path):
            return True
        for root in self.forbidden_roots:
            try:
                path.relative_to(root)
            except ValueError:
                continue
            return True
        return False

    def _hook(self, event: str, values: tuple[Any, ...]) -> None:
        if event != "open" or not values:
            return
        raw = values[0]
        if not isinstance(raw, (str, bytes, os.PathLike)):
            return
        mode = values[1] if len(values) > 1 else None
        if isinstance(mode, str) and "r" not in mode and "+" not in mode:
            return
        try:
            candidate = Path(raw).expanduser().resolve()
            if not candidate.exists():
                return
            rendered = str(candidate)
        except (OSError, TypeError, ValueError):
            candidate = Path(os.fsdecode(raw))
            rendered = os.fsdecode(raw)
        forbidden = self._forbidden(candidate)
        with self._lock:
            self.paths.append(rendered)
            if forbidden:
                self.forbidden_attempts.append(rendered)
        if forbidden:
            raise PermissionError(f"Blocked forbidden child-runtime read before open: {rendered}")

    @property
    def unique_paths(self) -> list[str]:
        with self._lock:
            return sorted(set(self.paths))


def _read_json(path: Path, *, purpose: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{purpose} is unavailable or invalid") from error
    if not isinstance(value, dict):
        raise TypeError(f"{purpose} must be a JSON object")
    return value


def _binding_scene_hash(runtime: Any) -> str:
    binding = runtime.prefix_binding()
    if not isinstance(binding, Mapping):
        raise TypeError("Rover prefix binding must be an object")
    value = binding.get("scene_prefix_sha256")
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError("Rover scene-prefix hash is invalid")
    return value


def _run_child(request_path: Path, output_path: Path) -> int:
    # Read only the parent-authored root coordinates, then install the hook and
    # reread the complete request under audit.  No production runtime module is
    # imported before the hook is active.
    request_preview = _read_json(request_path, purpose="child request preview")
    runtime_root = Path(str(request_preview.get("runtime_root", ""))).resolve()
    source_root = Path(str(request_preview.get("source_project_root", ""))).resolve()
    audit = _WholeProcessReadAudit(runtime_root, source_root)
    request = _read_json(request_path, purpose="child request")
    _assert_absent_runtime_trees(runtime_root)

    # Patch the logical root before importing any rover, chat, mapping, or
    # language runtime module.  All project-relative data resolution now lands
    # in the oracle-absent copy.
    import semantic_3d_chat.config as config_module

    config_module.PROJECT_ROOT = runtime_root
    from semantic_3d_chat.robot.practical_rover import (
        build_local_practical_rover,
        practical_rover_preflight,
    )

    kwargs = {
        "config": request["config"],
        "control_config": request["control_config"],
        "scene_id": request["scene"],
        "base_checkpoint": request["base_checkpoint"],
        "control_checkpoint": request["control_checkpoint"],
        "runtime_asset": request["runtime_asset"],
        "robot_state_checkpoint": request["robot_state_checkpoint"],
        "navigation_checkpoint": request["navigation_checkpoint"],
    }
    mode = request.get("mode")
    runtime_explicit_reads: list[str] = []
    if mode == "preflight":
        preflight = practical_rover_preflight(**kwargs)
        _assert_absent_runtime_trees(runtime_root)
        result: dict[str, Any] = {
            "schema": "semantic_3d_chat.gemma_waypoint_oracle_isolation.v1",
            "passed": preflight.get("ready") is True,
            "mode": "preflight",
            "loads_gemma": False,
            "scene_id": request["scene"],
            "runtime_copy_oracle_and_qa_absent": True,
            "source_oracle_directory_mutated": False,
            "preflight": preflight,
        }
    elif mode == "live":
        runtime_audit_path = Path(str(request["runtime_audit"])).resolve()
        controller = build_local_practical_rover(
            **kwargs,
            persistent_map=request["persistent_map"],
            audit_output=runtime_audit_path,
            initial_scan=False,
        )
        try:
            startup = controller.startup()
            startup_hash = startup.get("scene_prefix_hash")
            startup_active_hash = startup.get("active_prefix_hash")
            if not isinstance(startup_hash, str) or _SHA256.fullmatch(startup_hash) is None:
                raise ValueError("Startup scene prefix hash is invalid")
            if (
                not isinstance(startup_active_hash, str)
                or _SHA256.fullmatch(startup_active_hash) is None
            ):
                raise ValueError("Startup active prefix hash is invalid")
            questions = request.get("questions")
            if (
                not isinstance(questions, list)
                or len(questions) < 2
                or any(not isinstance(item, str) or not item.strip() for item in questions)
            ):
                raise ValueError("Live isolation smoke requires at least two questions")
            question_receipts: list[dict[str, Any]] = []
            for question in questions:
                response = controller.handle_instruction(question)
                if (
                    response.get("decision_source")
                    != "local_gemma_continuous_scene_answer"
                    or response.get("gemma_attempted") is not True
                    or response.get("scene_prefix_hash") != startup_hash
                    or response.get("active_prefix_hash") != startup_active_hash
                    or response.get("environmental_text_inputs") != []
                ):
                    raise AssertionError("Question response violated the isolated Gemma contract")
                question_receipts.append(
                    {
                        "question_sha256": _sha256_text(question),
                        "scene_prefix_sha256": response["scene_prefix_hash"],
                        "active_prefix_sha256": response["active_prefix_hash"],
                        "actual_local_gemma_attempted": True,
                        "environmental_text_inputs": [],
                    }
                )

            waypoint = controller.gemma_waypoint_controller
            if waypoint is None:
                raise RuntimeError("Production Gemma waypoint controller is unavailable")
            goal = waypoint.run(
                str(request["decision_instruction"]),
                max_steps=int(request["decision_steps"]),
            )
            decisions = [receipt.as_dict() for receipt in goal.receipts]
            if not decisions or any(
                item.get("actual_gemma_causal_forward") is not True
                or item.get("scene_prefix_sha256") != startup_hash
                or item.get("deterministic_route_planner_used") is not False
                or item.get("substitution_applied") is not False
                or item.get("synthetic_stop_applied") is not False
                for item in decisions
            ):
                raise AssertionError("Waypoint decision violated prefix/provenance isolation")
            final_hash = _binding_scene_hash(controller.runtime)
            if final_hash != startup_hash:
                raise AssertionError("Static scene prefix changed during isolated inference")
        finally:
            controller.close()
        production_audit = _read_json(runtime_audit_path, purpose="runtime file audit")
        if (
            production_audit.get("passed") is not True
            or production_audit.get("forbidden_accesses") != []
        ):
            raise AssertionError("Production rover file-access audit failed")
        raw_runtime_reads = production_audit.get("loaded_files")
        if not isinstance(raw_runtime_reads, list) or any(
            not isinstance(item, str) for item in raw_runtime_reads
        ):
            raise TypeError("Production rover file-access inventory is invalid")
        runtime_explicit_reads = raw_runtime_reads
        _assert_absent_runtime_trees(runtime_root)
        result = {
            "schema": "semantic_3d_chat.gemma_waypoint_oracle_isolation.v1",
            "passed": True,
            "mode": "live",
            "loads_gemma": True,
            "scene_id": request["scene"],
            "runtime_copy_oracle_and_qa_absent": True,
            "source_oracle_directory_mutated": False,
            "scene_prefix_built_before_questions": True,
            "startup_scene_prefix_sha256": startup_hash,
            "startup_active_prefix_sha256": startup_active_hash,
            "question_count": len(question_receipts),
            "question_receipts": question_receipts,
            "all_question_scene_prefix_hashes_identical": True,
            "all_question_active_prefix_hashes_identical": True,
            "decision_instruction_sha256": _sha256_text(
                str(request["decision_instruction"])
            ),
            "decision_count": len(decisions),
            "decision_receipts": decisions,
            "all_decision_scene_prefix_hashes_identical": True,
            "final_scene_prefix_sha256": final_hash,
            "navigation_success_required_by_this_gate": False,
            "production_runtime_audit": production_audit,
        }
    else:
        raise ValueError("Child request mode must be preflight or live")

    # The CPython hook catches Python and ordinary native-library reads.  Merge
    # the production runtime's explicit records for native tensor readers that
    # may bypass that hook so the retained inventory is the complete union.
    loaded = sorted(set(audit.unique_paths).union(runtime_explicit_reads))
    forbidden = sorted(set(audit.forbidden_attempts))
    result["whole_process_read_audit"] = {
        "mechanism": "cpython_process_audit_hook_plus_runtime_explicit_records",
        "started_before_production_runtime_import": True,
        "loaded_files": loaded,
        "loaded_file_count": len(loaded),
        "loaded_file_inventory_sha256": _sha256_text("\n".join(loaded)),
        "forbidden_roots": [str(path) for path in audit.forbidden_roots],
        "forbidden_accesses": forbidden,
        "forbidden_access_count": len(forbidden),
        "passed": not forbidden,
    }
    result["passed"] = result.get("passed") is True and not forbidden
    _atomic_json(output_path, result)
    return 0 if result["passed"] else 1


def _validate_child_report(report: Mapping[str, Any], *, expected_mode: str) -> None:
    audit = report.get("whole_process_read_audit")
    if not isinstance(audit, Mapping):
        raise TypeError("Child report has no whole-process read audit")
    loaded = audit.get("loaded_files")
    if not isinstance(loaded, list) or not loaded or any(not isinstance(item, str) for item in loaded):
        raise ValueError("Child read inventory is empty or invalid")
    raw_forbidden_roots = audit.get("forbidden_roots")
    if not isinstance(raw_forbidden_roots, list) or any(
        not isinstance(item, str) for item in raw_forbidden_roots
    ):
        raise ValueError("Child read audit has no forbidden-root inventory")
    forbidden_roots = [Path(item).resolve() for item in raw_forbidden_roots]

    def under_forbidden_root(raw_path: str) -> bool:
        candidate = Path(raw_path).resolve()
        for root in forbidden_roots:
            try:
                candidate.relative_to(root)
            except ValueError:
                continue
            return True
        return False

    forbidden_by_inventory = [
        item
        for item in loaded
        if _has_forbidden_component(Path(item)) or under_forbidden_root(item)
    ]
    if (
        report.get("schema")
        != "semantic_3d_chat.gemma_waypoint_oracle_isolation.v1"
        or report.get("passed") is not True
        or report.get("mode") != expected_mode
        or report.get("runtime_copy_oracle_and_qa_absent") is not True
        or report.get("source_oracle_directory_mutated") is not False
        or audit.get("started_before_production_runtime_import") is not True
        or audit.get("forbidden_accesses") != []
        or audit.get("forbidden_access_count") != 0
        or audit.get("passed") is not True
        or forbidden_by_inventory
    ):
        raise AssertionError("Child runtime did not pass strict oracle/QA isolation")
    if expected_mode == "live":
        decisions = report.get("decision_receipts")
        questions = report.get("question_receipts")
        prefix = report.get("startup_scene_prefix_sha256")
        active_prefix = report.get("startup_active_prefix_sha256")
        if (
            not isinstance(prefix, str)
            or _SHA256.fullmatch(prefix) is None
            or not isinstance(active_prefix, str)
            or _SHA256.fullmatch(active_prefix) is None
            or not isinstance(decisions, list)
            or not decisions
            or not isinstance(questions, list)
            or len(questions) < 2
            or report.get("scene_prefix_built_before_questions") is not True
            or report.get("all_question_scene_prefix_hashes_identical") is not True
            or report.get("all_question_active_prefix_hashes_identical") is not True
            or report.get("all_decision_scene_prefix_hashes_identical") is not True
            or report.get("final_scene_prefix_sha256") != prefix
            or any(item.get("scene_prefix_sha256") != prefix for item in questions)
            or any(
                item.get("active_prefix_sha256") != active_prefix for item in questions
            )
            or any(item.get("scene_prefix_sha256") != prefix for item in decisions)
        ):
            raise AssertionError("Child live inference did not preserve one static scene prefix")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/runtime/embodied_live.yaml")
    parser.add_argument(
        "--control-config",
        default="configs/runtime/gemma4_v56_question_control.yaml",
    )
    parser.add_argument("--scene", default="scene_000001")
    parser.add_argument(
        "--base-checkpoint",
        default="data_gemma4/runtime/checkpoints/gemma4_v54_release_v1",
    )
    parser.add_argument(
        "--control-checkpoint",
        default="data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1",
    )
    parser.add_argument(
        "--robot-state-checkpoint",
        default="data_gemma4/checkpoints/robot_state_numeric_v1",
    )
    parser.add_argument(
        "--navigation-checkpoint",
        required=True,
        help="Explicit final model-only waypoint checkpoint; no stale default is selected.",
    )
    parser.add_argument("--runtime-asset")
    parser.add_argument("--map")
    parser.add_argument(
        "--question",
        action="append",
        default=None,
        help="Non-navigation question; repeat at least twice for the live smoke.",
    )
    parser.add_argument("--decision-instruction", default="Face the chair.")
    parser.add_argument("--decision-steps", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Build and preflight the oracle-absent copy without loading Gemma.",
    )
    parser.add_argument("--_child-request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_child-output", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args._child_request is not None or args._child_output is not None:
        if args._child_request is None or args._child_output is None:
            raise ValueError("Both internal child paths are required")
        return _run_child(args._child_request.resolve(), args._child_output.resolve())
    if (
        isinstance(args.decision_steps, bool)
        or not 1 <= args.decision_steps <= 8
        or not math.isfinite(args.timeout_seconds)
        or args.timeout_seconds <= 0.0
    ):
        raise ValueError("Decision steps must be 1..8 and timeout must be positive")
    args.question = args.question or ["Is there a chair?", "What is on the table?"]
    if len(args.question) < 2:
        raise ValueError("Repeat --question at least twice")

    with tempfile.TemporaryDirectory(prefix="semantic_3d_chat_isolated_") as raw:
        runtime_root = Path(raw).resolve()
        request = _materialize_runtime_copy(args, runtime_root)
        child_request = runtime_root / "child_request.json"
        child_output = runtime_root / "child_result.json"
        _atomic_json(child_request, request)
        environment = os.environ.copy()
        source_path = str(SOURCE_PROJECT_ROOT / "src")
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            source_path
            if not existing_pythonpath
            else f"{source_path}{os.pathsep}{existing_pythonpath}"
        )
        requested_python = args.python_executable.expanduser()
        child_python = (
            requested_python
            if requested_python.is_absolute()
            else SOURCE_PROJECT_ROOT / requested_python
        )
        # Do not resolve the venv launcher symlink: Python discovers its virtual
        # environment from that path. Dereferencing it would launch the base
        # interpreter without the project's installed dependencies.
        command = [
            os.path.abspath(child_python),
            str(Path(__file__).resolve()),
            "--navigation-checkpoint",
            request["navigation_checkpoint"],
            "--_child-request",
            str(child_request),
            "--_child-output",
            str(child_output),
        ]
        with _source_oracle_temporarily_unavailable(SOURCE_PROJECT_ROOT) as isolation:
            completed = subprocess.run(
                command,
                cwd=runtime_root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=args.timeout_seconds,
                check=False,
            )
            if completed.returncode != 0 or not child_output.is_file():
                detail = "\n".join(
                    part
                    for part in (completed.stdout[-4000:], completed.stderr[-4000:])
                    if part
                )
                raise RuntimeError(
                    f"Oracle-absent child failed with exit {completed.returncode}:\n{detail}"
                )
            report = _read_json(child_output, purpose="oracle-absent child report")
            _validate_child_report(report, expected_mode=request["mode"])
        if isolation["restored"] is not True:
            raise RuntimeError("Source oracle was not restored after child verification")
        report["isolated_runtime_copy_removed_after_verification"] = True
        report["source_oracle_directory_rename_attempted"] = True
        report["source_oracle_directory_delete_attempted"] = False
        report["source_oracle_directory"] = isolation["original"]
        report["source_oracle_directory_physically_unavailable_during_child"] = (
            isolation["unavailable_during_child"]
        )
        report["source_oracle_directory_restored"] = isolation["restored"]
        report["passed"] = bool(
            report.get("passed") is True
            and isolation["renamed"] is True
            and isolation["unavailable_during_child"] is True
            and isolation["restored"] is True
        )
        report["child_stdout_sha256"] = _sha256_text(completed.stdout)
        report["child_stderr_sha256"] = _sha256_text(completed.stderr)
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "mode": report["mode"],
                "scene_id": report["scene_id"],
                "loads_gemma": report["loads_gemma"],
                "runtime_copy_oracle_and_qa_absent": report[
                    "runtime_copy_oracle_and_qa_absent"
                ],
                "source_oracle_directory_physically_unavailable_during_child": report[
                    "source_oracle_directory_physically_unavailable_during_child"
                ],
                "source_oracle_directory_restored": report[
                    "source_oracle_directory_restored"
                ],
                "loaded_file_count": report["whole_process_read_audit"][
                    "loaded_file_count"
                ],
                "forbidden_access_count": report["whole_process_read_audit"][
                    "forbidden_access_count"
                ],
                "output": str(args.output.expanduser().resolve()),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
