"""Scripted numerical embodied-camera and collision benchmark (no scene labels)."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.mcp_server.server import build_server
from semantic_3d_chat.robot.simulator import EmbodiedCameraSimulator


def _normalize(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _turn_to(simulator: EmbodiedCameraSimulator, target_degrees: float) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    maximum = float(simulator.settings["max_turn_degrees"])
    while abs(_normalize(target_degrees - simulator.state.body_yaw_degrees)) > 1e-7:
        remaining = _normalize(target_degrees - simulator.state.body_yaw_degrees)
        step = math.copysign(min(abs(remaining), maximum), remaining)
        outputs.append(simulator.turn(step))
        if not outputs[-1]["success"]:
            break
    return outputs


def _heading_toward(delta_xy: np.ndarray) -> float:
    return math.degrees(math.atan2(-float(delta_xy[0]), float(delta_xy[1])))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _trajectory_figure(
    simulator: EmbodiedCameraSimulator, trajectory: list[dict[str, Any]], output: Path
) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    points = np.asarray([item["result"]["position_m"][:2] for item in trajectory], dtype=float)
    obstacles = simulator.collision_map.obstacle_points_xy_m
    if len(obstacles) > 5000:
        obstacles = obstacles[np.linspace(0, len(obstacles) - 1, 5000, dtype=np.int64)]
    figure = Figure(figsize=(6.5, 5.5), dpi=140, constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    axis.scatter(obstacles[:, 0], obstacles[:, 1], s=1, c="#8b95a5", alpha=0.35)
    axis.plot(points[:, 0], points[:, 1], "o-", color="#e85d3f", linewidth=2, markersize=4)
    axis.set(
        xlabel="X — right (m)",
        ylabel="Y — forward (m)",
        title="Scripted bounded robot trajectory over numerical occupancy",
    )
    axis.set_aspect("equal", adjustable="box")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    figure.clear()


def run_benchmark(config: dict[str, Any], scene_id: str) -> dict[str, Any]:
    simulator = EmbodiedCameraSimulator(config, scene_id)
    seed = int(config["seed"])
    trajectory: list[dict[str, Any]] = []
    checks: dict[str, bool] = {}

    def record(action: str, result: dict[str, Any]) -> dict[str, Any]:
        trajectory.append({"step": len(trajectory), "action": action, "result": result})
        return result

    start = record("get_robot_state", simulator.get_robot_state())
    checks["numeric_start_state"] = bool(start["success"] and not start["collision"])

    limited = record(
        "turn_limit_rejection",
        simulator.turn(float(config["robot"]["max_turn_degrees"]) + 1.0),
    )
    checks["turn_limit_rejected"] = limited["error_code"] == "E_LIMIT"

    turned = record("bounded_turn", simulator.turn(30.0))
    checks["bounded_turn_succeeded"] = bool(
        turned["success"] and math.isclose(turned["body_yaw_degrees"], 30.0, abs_tol=1e-7)
    )
    looked = record("bounded_look", simulator.look(10.0, -10.0))
    checks["bounded_look_succeeded"] = bool(
        looked["success"]
        and math.isclose(looked["camera_yaw_degrees"], 40.0, abs_tol=1e-7)
        and math.isclose(looked["pitch_degrees"], -10.0, abs_tol=1e-7)
    )

    scan = record("scan", simulator.scan())
    checks["scan_updated_scene"] = bool(
        scan["success"] and scan["visible_voxels"] > 0 and scan["scene_version"] == 1
    )

    # Select a free direction from geometry only, then execute it through the
    # same bounded public actions. This is not semantic target navigation.
    simulator.reset_scene(scene_id, seed)
    free_heading: float | None = None
    free_distance = min(0.10, float(config["robot"]["max_move_m"]))
    start_xy = simulator.state.position_xy_m.copy()
    for heading in np.arange(-180.0, 180.0, 15.0):
        yaw = math.radians(float(heading))
        delta = free_distance * np.array([-math.sin(yaw), math.cos(yaw)])
        if not simulator.collision_map.segment_check(start_xy, start_xy + delta).collision:
            free_heading = float(heading)
            break
    if free_heading is not None:
        for output in _turn_to(simulator, free_heading):
            record("free_move_turn", output)
        free_move = record("free_move", simulator.move_forward(free_distance))
        checks["free_space_move_succeeded"] = bool(
            free_move["success"] and math.isclose(free_move["distance_moved"], free_distance)
        )
    else:
        checks["free_space_move_succeeded"] = False

    # Aim at the closest anonymous occupied surface. The attempted pose is
    # rejected atomically, proving swept collision checking and no penetration.
    simulator.reset_scene(scene_id, seed)
    start_xy = simulator.state.position_xy_m.copy()
    offsets = simulator.collision_map.obstacle_points_xy_m.astype(float) - start_xy
    distances = np.linalg.norm(offsets, axis=1)
    nearest = offsets[int(np.argmin(distances))]
    collision_heading = _heading_toward(nearest)
    for output in _turn_to(simulator, collision_heading):
        record("collision_turn", output)
    collision_attempt = record(
        "collision_attempt", simulator.move_forward(float(config["robot"]["max_move_m"]))
    )
    checks["collision_rejected_atomically"] = bool(
        not collision_attempt["success"]
        and collision_attempt["error_code"] == "E_COLLISION"
        and np.allclose(collision_attempt["position_m"][:2], start_xy)
    )

    stopped = record("stop", simulator.stop())
    blocked_after_stop = record("move_after_stop", simulator.move_forward(0.05))
    checks["stop_blocks_motion"] = bool(
        stopped["success"] and blocked_after_stop["error_code"] == "E_STOPPED"
    )
    reset = record("reset_scene", simulator.reset_scene(scene_id, seed))
    checks["reset_restores_episode"] = bool(
        reset["success"] and not reset["stopped"] and reset["scene_version"] == 0
    )

    async def mcp_smoke() -> tuple[int, bool]:
        server = build_server(simulator)
        registered = await server.list_tools()
        response = await server.call_tool("get_robot_state", {})
        payload = response.structured_content
        return len(registered), bool(payload and payload.get("success"))

    mcp_tool_count, mcp_call_succeeded = asyncio.run(mcp_smoke())
    checks["mcp_tools_registered"] = mcp_tool_count == 9
    checks["mcp_structured_call_succeeded"] = mcp_call_succeeded

    passed = sum(checks.values())
    result = {
        "schema_version": 1,
        "scene_id": scene_id,
        "map_source": "numeric_voxel_map",
        "metadata_or_labels_loaded": False,
        "semantic_target_navigation_evaluated": False,
        "benchmark_scope": "bounded_numeric_actions_collision_scan_and_reset",
        "mcp_sdk_version": importlib.metadata.version("mcp"),
        "mcp_tool_count": mcp_tool_count,
        "checks": checks,
        "passed": passed,
        "total": len(checks),
        "pass_rate": passed / len(checks),
        "trajectory_steps": len(trajectory),
        "final_state": simulator.get_robot_state(),
    }
    reports = PROJECT_ROOT / str(config["paths"]["reports_root"])
    _write_json(reports / "metrics" / "robot_navigation.json", result)
    _write_json(reports / "examples" / "robot_trajectories.json", trajectory)
    _trajectory_figure(simulator, trajectory, reports / "figures" / "robot_trajectory.png")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="configs/default.yaml")
    result.add_argument("--scene", default="scene_000001")
    return result


def main() -> None:
    args = parser().parse_args()
    result = run_benchmark(load_config(args.config), args.scene)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
