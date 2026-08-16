"""Render hash-pinned, runtime-only V3 embodied-approach trajectories.

The two inputs are completed conversational-runtime result artifacts.  This
post-hoc visualization opens no simulator oracle, QA file, scene metadata, map,
or model.  Every plotted target is the continuous target XYZ recorded by the
live grounding path before its bounded action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOM_SIZE_M = (6.0, 5.0)
DEFAULT_FIGURE = Path("reports/gemma4/figures/embodied_approach_v3_trajectories.png")
DEFAULT_OUTPUT = Path("reports/gemma4/examples/embodied_approach_v3_trajectories.json")
PNG_METADATA = {
    "Software": "semantic_3d_chat deterministic V3 embodied trajectory plotter",
    "Creation Time": "",
}


@dataclass(frozen=True)
class SealedCase:
    """One immutable runtime-result input and its expected completion mode."""

    scene_id: str
    path: Path
    sha256: str
    completion_mode: str


DEFAULT_CASES = (
    SealedCase(
        scene_id="scene_000001",
        path=Path(
            "reports/gemma4/metrics/"
            "embodied_conversation_hybrid_approach_chair_v3_scene_000001.json"
        ),
        sha256="d12bad64211b16d55d905985cf78ff69db6a6f6e8183a0c18bb1e7f9be5a76ba",
        completion_mode="semantic_standoff",
    ),
    SealedCase(
        scene_id="scene_000031",
        path=Path(
            "reports/gemma4/metrics/"
            "embodied_conversation_hybrid_approach_chair_v3_scene_000031.json"
        ),
        sha256="495a03ca0385964c3694165c7f99f284dec2d1054593821606347c19f839c68e",
        completion_mode="collision_limited_safe_stop",
    ),
)


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _sequence(value: object, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return value


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _xy(value: object, name: str) -> list[float]:
    items = _sequence(value, name)
    if len(items) < 2:
        raise ValueError(f"{name} must contain at least X and Y")
    return [
        _finite_float(items[0], f"{name} X"),
        _finite_float(items[1], f"{name} Y"),
    ]


def _xyz(value: object, name: str) -> list[float]:
    items = _sequence(value, name)
    if len(items) != 3:
        raise ValueError(f"{name} must contain XYZ")
    return [_finite_float(item, f"{name}[{index}]") for index, item in enumerate(items)]


def load_sealed_result(case: SealedCase) -> dict[str, Any]:
    """Authenticate *case* before parsing and validate its runtime-only scope."""

    if not case.path.is_file() or case.path.is_symlink():
        raise FileNotFoundError(f"Sealed embodied result must be one regular file: {case.path}")
    observed_sha256 = file_sha256(case.path)
    if observed_sha256 != case.sha256:
        raise ValueError(
            f"{case.scene_id} embodied result digest differs: expected {case.sha256}, "
            f"observed {observed_sha256}"
        )
    payload = json.loads(case.path.read_bytes())
    if not isinstance(payload, dict):
        raise TypeError(f"{case.scene_id} embodied result must be a mapping")
    turns = _sequence(payload.get("turns"), f"{case.scene_id} turns")
    if not (
        payload.get("schema") == "semantic_3d_chat.embodied_conversation_result.v1"
        and payload.get("scene_id") == case.scene_id
        and payload.get("passed_runtime_audit") is True
        and payload.get("forbidden_access_count") == 0
        and payload.get("environmental_text_inputs") == []
        and len(turns) == 1
    ):
        raise ValueError(f"{case.scene_id} runtime identity or isolation differs")
    turn = _mapping(turns[0], f"{case.scene_id} turn")
    if not (
        turn.get("kind") == "learned_navigation_closed_loop"
        and turn.get("success") is True
        and turn.get("termination_reason") == "stop"
        and turn.get("prefix_refresh_verified") is True
        and turn.get("static_scene_prefix_question_independent") is True
        and turn.get("primary_static_scene_retrieval") is False
        and turn.get("question_dependent_navigation_grounding") is True
        and turn.get("environmental_text_inputs") == []
    ):
        raise ValueError(f"{case.scene_id} successful runtime trajectory contract differs")
    return payload


def _action_label(
    step: int, tool: str, arguments: Mapping[str, Any], approach: Mapping[str, Any]
) -> str:
    if tool == "turn":
        value = _finite_float(arguments.get("angle_degrees"), f"step {step} turn")
        return f"{step}. turn {value:+.1f}\N{DEGREE SIGN}"
    if tool in {"move_forward", "move_backward"}:
        value = _finite_float(arguments.get("distance_meters"), f"step {step} distance")
        suffix = " (clipped)" if approach.get("reason") == "collision_limited_safe_progress" else ""
        direction = "move" if tool == "move_forward" else "back"
        return f"{step}. {direction} {value:.3f} m{suffix}"
    if tool == "stop":
        mode = approach.get("completion_mode")
        suffix = " (collision-limited)" if mode == "collision_limited_safe_stop" else ""
        return f"{step}. stop{suffix}"
    return f"{step}. {tool}"


def extract_trajectory(case: SealedCase, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract finite numerical trajectory data from one authenticated result."""

    turn = _mapping(_sequence(payload["turns"], "turns")[0], f"{case.scene_id} turn")
    steps = _sequence(turn.get("steps"), f"{case.scene_id} steps")
    if len(steps) != turn.get("step_count") or not steps:
        raise ValueError(f"{case.scene_id} step count differs")

    output_steps: list[dict[str, Any]] = []
    positions: list[list[float]] = []
    path_distance_m = 0.0
    for expected_step, raw_step in enumerate(steps, start=1):
        step = _mapping(raw_step, f"{case.scene_id} step {expected_step}")
        if step.get("closed_loop_step") != expected_step or step.get("success") is not True:
            raise ValueError(f"{case.scene_id} step ordering or success differs")
        grounding = _mapping(
            step.get("continuous_grounding"),
            f"{case.scene_id} step {expected_step} grounding",
        )
        approach = _mapping(
            grounding.get("numeric_approach_interlock"),
            f"{case.scene_id} step {expected_step} approach",
        )
        collision_limit = _mapping(
            approach.get("collision_limited_interlock"),
            f"{case.scene_id} step {expected_step} collision limit",
        )
        selection = _mapping(
            step.get("tool_selection"), f"{case.scene_id} step {expected_step} selection"
        )
        call = _mapping(selection.get("call"), f"{case.scene_id} step {expected_step} call")
        arguments = _mapping(
            call.get("arguments"), f"{case.scene_id} step {expected_step} arguments"
        )
        tool = call.get("tool")
        if not isinstance(tool, str) or not tool:
            raise TypeError(f"{case.scene_id} step {expected_step} tool must be text")
        receipts = _sequence(
            step.get("action_receipts"), f"{case.scene_id} step {expected_step} receipts"
        )
        if len(receipts) != 1:
            raise ValueError(f"{case.scene_id} step {expected_step} must have one receipt")
        receipt = _mapping(receipts[0], f"{case.scene_id} step {expected_step} receipt")

        before_xy = _xy(
            approach.get("robot_position_xy_m"),
            f"{case.scene_id} step {expected_step} position before",
        )
        after_xy = _xy(
            receipt.get("position_m"), f"{case.scene_id} step {expected_step} position after"
        )
        if not positions:
            positions.append(before_xy)
        elif any(abs(left - right) > 1e-8 for left, right in zip(positions[-1], before_xy)):
            raise ValueError(f"{case.scene_id} trajectory positions are discontinuous")
        positions.append(after_xy)

        target_xyz = _xyz(
            grounding.get("target_xyz_m"), f"{case.scene_id} step {expected_step} target"
        )
        distance_moved = _finite_float(
            receipt.get("distance_moved"), f"{case.scene_id} step {expected_step} moved"
        )
        path_distance_m += distance_moved
        if not (
            grounding.get("all_map_voxels_scored") is True
            and grounding.get("oracle_inputs_at_runtime") is False
            and grounding.get("environmental_text_inputs") == []
            and receipt.get("collision") is False
            and receipt.get("success") is True
        ):
            raise ValueError(f"{case.scene_id} step {expected_step} runtime evidence differs")

        output_steps.append(
            {
                "step": expected_step,
                "learned_tool": approach.get("learned_tool"),
                "executed_tool": tool,
                "executed_arguments": dict(arguments),
                "action_label": _action_label(expected_step, tool, arguments, approach),
                "position_before_xy_m": before_xy,
                "position_after_xy_m": after_xy,
                "body_yaw_after_degrees": _finite_float(
                    receipt.get("body_yaw_degrees"),
                    f"{case.scene_id} step {expected_step} yaw",
                ),
                "distance_moved": distance_moved,
                "continuous_target_xyz_m": target_xyz,
                "continuous_target_distance_m": _finite_float(
                    approach.get("target_distance_m"),
                    f"{case.scene_id} step {expected_step} target distance",
                ),
                "actual_progress_m": _finite_float(
                    approach.get("actual_progress_m"),
                    f"{case.scene_id} step {expected_step} progress",
                ),
                "semantic_standoff_satisfied": approach.get("goal_satisfied") is True,
                "completion_mode": approach.get("completion_mode"),
                "completion_satisfied": approach.get("completion_satisfied") is True,
                "interlock_reason": approach.get("reason"),
                "collision_limited": {
                    "collision_predicted": collision_limit.get("collision_predicted") is True,
                    "requested_distance_m": collision_limit.get("requested_distance_m"),
                    "maximum_collision_free_distance_m": collision_limit.get(
                        "maximum_collision_free_distance_m"
                    ),
                    "executed_safe_distance_m": collision_limit.get("executed_safe_distance_m"),
                    "safe_closest_reachable": collision_limit.get("safe_closest_reachable") is True,
                    "reason": collision_limit.get("reason"),
                },
                "collision": False,
                "stopped": receipt.get("stopped") is True,
                "map_version": receipt.get("map_version"),
                "source_voxels": receipt.get("source_voxels"),
                "scored_voxels": grounding.get("scored_voxels"),
                "all_map_voxels_scored": True,
                "scene_prefix_sha256": receipt.get("scene_prefix_sha256"),
                "active_prefix_sha256": receipt.get("active_prefix_sha256"),
            }
        )

    final = output_steps[-1]
    completion_mode = final["completion_mode"]
    if completion_mode != case.completion_mode or final["completion_satisfied"] is not True:
        raise ValueError(f"{case.scene_id} completion mode differs")
    if case.completion_mode == "semantic_standoff" and not final["semantic_standoff_satisfied"]:
        raise ValueError(f"{case.scene_id} semantic standoff is not satisfied")
    if (
        case.completion_mode == "collision_limited_safe_stop"
        and not final["collision_limited"]["safe_closest_reachable"]
    ):
        raise ValueError(f"{case.scene_id} collision-limited completion is not reachable-safe")

    initial_xy = positions[0]
    final_xy = positions[-1]
    net_displacement_m = math.hypot(final_xy[0] - initial_xy[0], final_xy[1] - initial_xy[1])
    scene_prefixes = [step["scene_prefix_sha256"] for step in output_steps]
    active_prefixes = [step["active_prefix_sha256"] for step in output_steps]
    return {
        "scene_id": case.scene_id,
        "source": {
            "path": case.path.as_posix(),
            "sha256": case.sha256,
            "bytes": case.path.stat().st_size,
        },
        "instruction_family": "approach_then_stop",
        "completion_mode": completion_mode,
        "semantic_standoff_satisfied": final["semantic_standoff_satisfied"],
        "collision_limited_completion": completion_mode == "collision_limited_safe_stop",
        "step_count": len(output_steps),
        "initial_position_xy_m": initial_xy,
        "final_position_xy_m": final_xy,
        "positions_xy_m": positions,
        "path_distance_m": path_distance_m,
        "net_displacement_m": net_displacement_m,
        "final_continuous_target_distance_m": final["continuous_target_distance_m"],
        "target_standoff_m": 0.5,
        "stopped": final["stopped"],
        "collision_count": 0,
        "prefix_refresh_verified": True,
        "unique_scene_prefix_count": len(set(scene_prefixes)),
        "unique_active_prefix_count": len(set(active_prefixes)),
        "all_map_voxels_scored_every_step": all(
            step["all_map_voxels_scored"] for step in output_steps
        ),
        "steps": output_steps,
    }


def _plot_scene(axis: Any, scene: Mapping[str, Any]) -> None:
    width, depth = ROOM_SIZE_M
    axis.add_patch(
        Rectangle(
            (-width / 2.0, -depth / 2.0),
            width,
            depth,
            facecolor="#f8fafc",
            edgecolor="#334155",
            linewidth=1.5,
            zorder=0,
        )
    )
    positions = _sequence(scene["positions_xy_m"], "positions")
    x_values = [float(point[0]) for point in positions]
    y_values = [float(point[1]) for point in positions]
    targets = [step["continuous_target_xyz_m"] for step in scene["steps"]]
    target_x = [float(point[0]) for point in targets]
    target_y = [float(point[1]) for point in targets]

    axis.plot(
        target_x,
        target_y,
        linestyle="--",
        linewidth=1.0,
        color="#f59e0b",
        alpha=0.55,
        zorder=2,
    )
    axis.scatter(
        target_x,
        target_y,
        marker="x",
        s=38,
        linewidths=1.5,
        color="#d97706",
        zorder=3,
    )
    axis.plot(
        x_values,
        y_values,
        "o-",
        linewidth=2.2,
        markersize=4.5,
        color="#2563eb",
        zorder=4,
    )
    axis.scatter(
        [x_values[0]],
        [y_values[0]],
        marker="s",
        s=75,
        color="#0f172a",
        edgecolor="white",
        linewidth=0.8,
        zorder=6,
    )
    collision_limited = scene["completion_mode"] == "collision_limited_safe_stop"
    final_color = "#d97706" if collision_limited else "#16a34a"
    final_marker = "8" if collision_limited else "*"
    axis.scatter(
        [x_values[-1]],
        [y_values[-1]],
        marker=final_marker,
        s=165,
        color=final_color,
        edgecolor="white",
        linewidth=0.9,
        zorder=7,
    )
    axis.plot(
        [x_values[-1], target_x[-1]],
        [y_values[-1], target_y[-1]],
        linestyle=":",
        linewidth=1.4,
        color=final_color,
        zorder=3,
    )

    for index, step in enumerate(scene["steps"], start=1):
        before = step["position_before_xy_m"]
        after = step["position_after_xy_m"]
        if math.dist(before, after) > 1e-8:
            axis.annotate(
                "",
                xy=after,
                xytext=before,
                arrowprops={"arrowstyle": "->", "color": "#1d4ed8", "lw": 1.5},
                zorder=5,
            )
        offset = (5 + 4 * (index % 2), 5 + 7 * (index % 3))
        axis.annotate(
            str(index),
            xy=after,
            xytext=offset,
            textcoords="offset points",
            fontsize=7.5,
            color="#1e3a8a",
            weight="bold",
            zorder=8,
        )

    mode_label = (
        "collision-limited safe stop" if collision_limited else "semantic standoff satisfied"
    )
    action_lines = "\n".join(step["action_label"] for step in scene["steps"])
    axis.text(
        0.02,
        0.02,
        action_lines,
        transform=axis.transAxes,
        va="bottom",
        ha="left",
        fontsize=7.5,
        color="#0f172a",
        bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "#cbd5e1", "alpha": 0.9},
        zorder=10,
    )
    axis.text(
        0.98,
        0.02,
        f"{mode_label}\nfinal continuous distance: "
        f"{scene['final_continuous_target_distance_m']:.3f} m\n"
        f"net progress: {scene['net_displacement_m']:.3f} m",
        transform=axis.transAxes,
        va="bottom",
        ha="right",
        fontsize=8.2,
        color="#0f172a",
        bbox={"boxstyle": "round,pad=0.45", "fc": "white", "ec": final_color, "alpha": 0.94},
        zorder=10,
    )
    axis.set_xlim(-width / 2.0 - 0.15, width / 2.0 + 0.15)
    axis.set_ylim(-depth / 2.0 - 0.15, depth / 2.0 + 0.15)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.18, linewidth=0.6)
    axis.set_xlabel("World X — right (m)")
    axis.set_ylabel("World Y — forward (m)")
    axis.set_title(f"{scene['scene_id']} · {mode_label}", fontsize=11, weight="bold")


def plot_trajectories(scenes: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Render two runtime-derived XY trajectories with deterministic styling."""

    if len(scenes) != 2:
        raise ValueError("The V3 approach figure requires exactly two scenes")
    figure = Figure(figsize=(14.0, 6.0), dpi=160, facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    for axis, scene in zip(axes, scenes, strict=True):
        _plot_scene(axis, scene)
    handles = [
        Line2D([0], [0], color="#2563eb", marker="o", label="Robot trajectory"),
        Line2D([0], [0], color="#d97706", marker="x", linestyle="--", label="Continuous targets"),
        Line2D([0], [0], color="#0f172a", marker="s", linestyle="None", label="Start"),
        Line2D([0], [0], color="#16a34a", marker="*", linestyle="None", label="Semantic stop"),
        Line2D(
            [0],
            [0],
            color="#d97706",
            marker="8",
            linestyle="None",
            label="Collision-limited stop",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    figure.suptitle(
        "Continuous-grounded V3 successor approach trajectories",
        y=0.985,
        fontsize=14,
        weight="bold",
    )
    figure.text(
        0.5,
        0.025,
        "Post-hoc runtime visualization · continuous target XYZ and numeric robot receipts only · "
        "no oracle, QA, labels, scene metadata, map, or model loaded",
        ha="center",
        fontsize=8.5,
        color="#475569",
    )
    figure.subplots_adjust(left=0.055, right=0.985, bottom=0.14, top=0.82, wspace=0.16)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="png", dpi=160, metadata=PNG_METADATA)
    figure.clear()


def generate(
    cases: Sequence[SealedCase],
    figure_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Authenticate, extract, plot, and summarize sealed runtime results."""

    if len(cases) != 2 or len({case.scene_id for case in cases}) != 2:
        raise ValueError("Exactly two distinct sealed V3 approach cases are required")
    before = {case.scene_id: file_sha256(case.path) for case in cases}
    payloads = [load_sealed_result(case) for case in cases]
    scenes = [
        extract_trajectory(case, payload) for case, payload in zip(cases, payloads, strict=True)
    ]
    if {scene["completion_mode"] for scene in scenes} != {
        "semantic_standoff",
        "collision_limited_safe_stop",
    }:
        raise ValueError("The two cases must cover both V3 completion modes")
    plot_trajectories(scenes, figure_path)
    after = {case.scene_id: file_sha256(case.path) for case in cases}
    if before != after:
        raise RuntimeError("A sealed runtime result changed during trajectory generation")

    artifact = {
        "schema": "semantic_3d_chat.embodied_approach_v3_trajectories.v1",
        "artifact": "embodied_approach_v3_runtime_trajectory_visualization_v1",
        "scope": {
            "post_hoc_visualization_only": True,
            "new_inference": False,
            "source_file_count": 2,
            "runtime_result_files_loaded": 2,
            "oracle_files_loaded": False,
            "qa_files_loaded": False,
            "scene_metadata_files_loaded": False,
            "semantic_map_files_loaded": False,
            "model_files_loaded": False,
            "environmental_text_inputs": [],
            "continuous_target_xyz_only": True,
            "numeric_robot_receipts_only": True,
            "source_hashes_preserved": before == after,
        },
        "coordinate_convention": {"x": "right", "y": "forward", "unit": "meter"},
        "room_size_m": list(ROOM_SIZE_M),
        "completion_modes": {
            "semantic_standoff": "continuous target distance at or below 0.5 m",
            "collision_limited_safe_stop": (
                "minimum safe progress already met and no material collision-free step remains"
            ),
        },
        "figure": {
            "path": figure_path.as_posix(),
            "sha256": file_sha256(figure_path),
            "width_px": 2240,
            "height_px": 960,
        },
        "sources_sha256": before,
        "scenes": scenes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return artifact


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    artifact = generate(DEFAULT_CASES, args.figure, args.output)
    print(
        json.dumps(
            {
                "passed": True,
                "figure": artifact["figure"],
                "output": args.output.as_posix(),
                "scenes": [
                    {
                        "scene_id": scene["scene_id"],
                        "completion_mode": scene["completion_mode"],
                        "steps": scene["step_count"],
                        "net_displacement_m": scene["net_displacement_m"],
                    }
                    for scene in artifact["scenes"]
                ],
                "sources_sha256": artifact["sources_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CASES",
    "DEFAULT_FIGURE",
    "DEFAULT_OUTPUT",
    "SealedCase",
    "extract_trajectory",
    "file_sha256",
    "generate",
    "load_sealed_result",
    "plot_trajectories",
]
