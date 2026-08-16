#!/usr/bin/env python3
"""Render numeric robot trajectories from an authenticated navigation journal."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from semantic_3d_chat.evaluation.llm_navigation_benchmark import (
    file_sha256,
    validate_navigation_journal,
)


def numeric_trajectories(journal: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract only task IDs, families, and XY poses from a sealed journal."""

    validated = validate_navigation_journal(journal, require_complete=True)
    output: list[dict[str, Any]] = []
    for episode in validated["episodes"]:
        initial = episode.get("initial_state")
        if not isinstance(initial, Mapping):
            raise TypeError("Navigation episode lacks an initial numeric state")
        positions = [initial.get("position_m")]
        for step in episode["steps"]:
            receipt = step.get("receipt")
            if not isinstance(receipt, Mapping):
                raise TypeError("Navigation step lacks a numeric receipt")
            positions.append(receipt.get("position_m"))
        array = np.asarray(positions, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
            raise ValueError("Navigation trajectory contains invalid metric positions")
        output.append(
            {
                "task_id": episode["task_id"],
                "family": episode["family"],
                "termination": episode["termination"],
                "positions_m": array.tolist(),
            }
        )
    return output


def render(
    journal_path: Path,
    image_path: Path,
    data_path: Path,
    *,
    room_size_m: tuple[float, float] = (6.0, 5.0),
) -> dict[str, Any]:
    payload = json.loads(journal_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Navigation journal must contain one JSON object")
    trajectories = numeric_trajectories(payload)
    artifact = {
        "schema": "semantic_3d_chat.navigation_trajectories.v1",
        "source_journal_sha256": file_sha256(journal_path),
        "source_journal_root_sha256": payload["journal_sha256"],
        "scene_id": payload["header"]["scene_id"],
        "room_size_m": [float(room_size_m[0]), float(room_size_m[1])],
        "trajectories": trajectories,
    }

    image_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for index, trajectory in enumerate(trajectories):
        points = np.asarray(trajectory["positions_m"], dtype=np.float64)
        color = colors(index % 10)
        axis.plot(
            points[:, 0],
            points[:, 1],
            marker="o",
            markersize=4,
            linewidth=2,
            color=color,
            label=f"{trajectory['task_id']} · {trajectory['family']}",
        )
        axis.scatter(points[0, 0], points[0, 1], marker="s", s=45, color=color)
        axis.scatter(points[-1, 0], points[-1, 1], marker="X", s=60, color=color)
    width, depth = room_size_m
    axis.set_xlim(-width / 2.0, width / 2.0)
    axis.set_ylim(-depth / 2.0, depth / 2.0)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25)
    axis.set_xlabel("world X (m)")
    axis.set_ylabel("world Y (m)")
    axis.set_title("Learned V3 robot trajectories (square=start, X=stop)")
    axis.legend(loc="upper right", fontsize=8)
    figure.savefig(image_path, dpi=180)
    plt.close(figure)

    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = render(args.journal, args.image, args.output)
    print(
        json.dumps(
            {
                "passed": True,
                "scene_id": result["scene_id"],
                "trajectory_count": len(result["trajectories"]),
                "source_journal_sha256": result["source_journal_sha256"],
                "image": str(args.image),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["numeric_trajectories", "render"]
