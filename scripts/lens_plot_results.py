#!/usr/bin/env python3
"""Draw every measurement in the study as a figure.

One image per question, written to reports/figures. Each carries the thing that
decides whether the picture means anything -- the interval, the chance line, the
sample size -- because a bare bar chart of proportions invites a reader to take
a difference seriously that the data cannot support.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from semantic_3d_chat.config import PROJECT_ROOT

POINTS = PROJECT_ROOT / "reports" / "gemma4" / "metrics" / "point_grounding"
FIGURES = PROJECT_ROOT / "reports" / "figures"

INK = "#1b1b1f"
GRID = "#d8d8de"
SERIES = {
    "rope3d": "#1f6feb",
    "learned_absolute": "#d1691a",
    "none": "#6e7781",
    "rgb": "#8250df",
    "chance": "#b0141a",
}


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, color=INK, pad=12, loc="left")
    ax.set_xlabel(xlabel, fontsize=10, color=INK)
    ax.set_ylabel(ylabel, fontsize=10, color=INK)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)


def _save(figure, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    figure.tight_layout()
    figure.savefig(path, dpi=170, facecolor="white")
    plt.close(figure)
    return path


def _load(tag: str) -> dict | None:
    path = POINTS / f"{tag}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def plot_ablation(task: str, name: str, title: str) -> Path | None:
    prefix = "relational_" if task == "relational" else ""
    arms = [
        ("rope3d", f"{prefix}rope3d_rooms19", "3D rotary"),
        ("learned_absolute", f"{prefix}learned_absolute_rooms19", "learned absolute"),
        ("none", f"{prefix}none_rooms19", "no position"),
        ("rgb", f"{prefix}rgb_only_rooms19", "colour, not Gemma"),
    ]
    rows = [(key, label, _load(tag)) for key, tag, label in arms]
    rows = [(k, lab, run) for k, lab, run in rows if run]
    if not rows:
        return None

    figure, ax = plt.subplots(figsize=(7.4, 4.3))
    labels, values, lows, highs = [], [], [], []
    for _key, label, run in rows:
        held = run["held_out"]
        low, high = held.get("interval_95") or (held["hits_object"],) * 2
        labels.append(label)
        values.append(held["hits_object"])
        lows.append(held["hits_object"] - low)
        highs.append(high - held["hits_object"])
    colours = [SERIES[key] for key, _l, _r in rows]
    ax.bar(labels, values, color=colours, width=0.62,
           yerr=[lows, highs], capsize=6, ecolor=INK, error_kw={"linewidth": 1.2})
    chance = rows[0][2]["held_out"].get("chance_random_object")
    if chance:
        ax.axhline(chance, color=SERIES["chance"], linestyle="--", linewidth=1.4)
        ax.text(len(labels) - 0.45, chance + 0.012,
                f"chance {chance:.0%}", color=SERIES["chance"], fontsize=9, ha="right")
    held_rooms = len(rows[0][2]["held_out_rooms"])
    n = rows[0][2]["held_out"]["examples"]
    _style(ax, title, "", "lands on the object")
    ax.set_ylim(0, max(max(values) * 1.35, (chance or 0) * 1.6))
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")
    figure.text(0.012, 0.015,
                f"{held_rooms} held-out rooms, n={n}; bars are 95% Wilson intervals. "
                "Chance = a guesser that knows the answer is one of the room's objects.",
                fontsize=8, color="#5b5b66")
    return _save(figure, name)


def plot_scaling(metric: str, name: str, title: str, ylabel: str) -> Path | None:
    sizes = [2, 4, 8, 12, 16, 19]
    figure, ax = plt.subplots(figsize=(7.4, 4.3))
    drawn = False
    for mode, label in (("rope3d", "3D rotary"), ("learned_absolute", "learned absolute")):
        xs, ys, los, his = [], [], [], []
        for size in sizes:
            tag = f"scale19_{mode}" if size == 19 else f"{mode}_rooms{size}"
            run = _load(tag)
            if not run:
                continue
            held = run["held_out"]
            xs.append(size)
            ys.append(held[metric])
            if metric == "hits_object" and held.get("interval_95"):
                low, high = held["interval_95"]
                los.append(held[metric] - low)
                his.append(high - held[metric])
        if not xs:
            continue
        drawn = True
        if los:
            ax.errorbar(xs, ys, yerr=[los, his], color=SERIES[mode], marker="o",
                        linewidth=2.0, capsize=5, label=label, markersize=6)
        else:
            ax.plot(xs, ys, color=SERIES[mode], marker="o", linewidth=2.0,
                    label=label, markersize=6)
    if not drawn:
        plt.close(figure)
        return None
    if metric == "hits_object":
        run = _load("rope3d_rooms8") or _load("rope3d_rooms4")
        chance = run["held_out"].get("chance_random_object") if run else None
        if chance:
            ax.axhline(chance, color=SERIES["chance"], linestyle="--", linewidth=1.4)
            ax.text(sizes[-1], chance + 0.01, f"chance {chance:.0%}",
                    color=SERIES["chance"], fontsize=9, ha="right")
        ax.yaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")
    _style(ax, title, "training rooms", ylabel)
    ax.set_xticks(sizes)
    ax.legend(frameon=False, fontsize=9)
    figure.text(0.012, 0.015,
                "Every point trained to the same number of optimiser steps, so the axis is "
                "data rather than compute. One seed per point.",
                fontsize=8, color="#5b5b66")
    return _save(figure, name)


def plot_coverage(name: str) -> Path | None:
    rooms = sorted((PROJECT_ROOT / "data" / "spatial_lens").glob("*/scan_plan.json"))
    if not rooms:
        return None
    figure, ax = plt.subplots(figsize=(7.4, 4.3))
    finals = []
    for path in rooms[:40]:
        plan = json.loads(path.read_text(encoding="utf-8"))
        curve = plan["coverage_curve"]
        ax.plot(range(1, len(curve) + 1), curve, color=SERIES["rope3d"],
                alpha=0.28, linewidth=1.1)
        finals.append(len(curve))
    ax.axhline(0.99, color=SERIES["chance"], linestyle="--", linewidth=1.4)
    ax.text(1, 0.995, "99% target", color=SERIES["chance"], fontsize=9)
    _style(ax, "How many views a room actually needs",
           "views selected", "fraction of reachable surface seen")
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")
    figure.text(0.012, 0.015,
                f"{len(finals)} rooms; median {int(np.median(finals))} views to 99%. "
                "Reachable = visible from some camera standing in the room, so the "
                "inside of a drawer is not counted as a miss.",
                fontsize=8, color="#5b5b66")
    return _save(figure, name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    made = [
        plot_ablation("object", "ablation_object.png",
                      "Naming an object in a room the model has never seen"),
        plot_ablation("relational", "ablation_relational.png",
                      "Finding an object by its relation to another one"),
        plot_scaling("hits_object", "scaling_accuracy.png",
                     "Does more data help, and does the encoding change that?",
                     "lands on the object"),
        plot_scaling("median_gap_m", "scaling_precision.png",
                     "How far the answer lands from the object",
                     "median gap (m)"),
        plot_coverage("scan_coverage.png"),
    ]
    for path in made:
        if path:
            print(f"wrote {path.relative_to(PROJECT_ROOT)}")
    print(f"\n{sum(1 for p in made if p)} figures in {FIGURES.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
