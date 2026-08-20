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
ASSETS = PROJECT_ROOT / "reports" / "gemma4" / "metrics" / "point_grounding_assets"
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


def _load(tag: str, root: Path = POINTS) -> dict | None:
    path = root / f"{tag}.json"
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
    rooms = sorted((PROJECT_ROOT / "data" / "spatial_lens").glob("asset*/scan_plan.json"))
    if not rooms:
        return None
    figure, ax = plt.subplots(figsize=(7.4, 4.3))
    finals, surface = [], []
    for path in rooms:
        plan = json.loads(path.read_text(encoding="utf-8"))
        curve = plan["coverage_curve"]
        ax.plot(range(1, len(curve) + 1), curve, color=SERIES["rope3d"],
                alpha=0.24, linewidth=1.0)
        finals.append(len(curve))
        if plan.get("final_coverage_of_surface") is not None:
            surface.append(plan["final_coverage_of_surface"])
    ax.axhline(0.99, color=SERIES["chance"], linestyle="--", linewidth=1.4)
    ax.text(1, 0.995, "99% stopping threshold", color=SERIES["chance"], fontsize=9)
    _style(ax, "How many views a room actually needs",
           "views selected", "fraction of reachable surface seen")
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")
    tail = (f" Of the furniture's whole area, {np.mean(surface):.0%} is seen."
            if surface else "")
    figure.text(0.012, 0.015,
                f"{len(finals)} rooms; median {int(np.median(finals))} views. Reachable = "
                f"visible from some camera in the room, so a drawer interior is not a "
                f"miss -- but that denominator flatters the plan.{tail}",
                fontsize=8, color="#5b5b66")
    return _save(figure, name)


def plot_corpus_comparison(name: str) -> Path | None:
    """Coloured boxes against real furniture, same method both times."""

    tasks = [
        ("object", "naming an\nobject", "rope3d_rooms19", "object_rope3d"),
        ("object_rgb", "colour instead\nof Gemma", "rgb_only_rooms19", "object_rgb_only"),
        ("relational", "relational\n(unnamed target)", "relational_rope3d_rooms19",
         "relational_rope3d"),
    ]
    rows = []
    for _key, label, prim_tag, asset_tag in tasks:
        prim = _load(prim_tag, POINTS)
        asset = _load(asset_tag, ASSETS)
        if prim or asset:
            rows.append((label, prim, asset))
    if not rows:
        return None

    figure, ax = plt.subplots(figsize=(7.8, 4.4))
    width = 0.36
    spots = np.arange(len(rows))
    for offset, which, colour, label in (
        (-width / 2, 1, "#6e7781", "primitive rooms"),
        (width / 2, 2, "#1f6feb", "real assets"),
    ):
        values, lows, highs = [], [], []
        for row in rows:
            run = row[which]
            if run is None:
                values.append(0.0); lows.append(0.0); highs.append(0.0); continue
            held = run["held_out"]
            low, high = held.get("interval_95") or (held["hits_object"],) * 2
            values.append(held["hits_object"])
            lows.append(max(held["hits_object"] - low, 0.0))
            highs.append(max(high - held["hits_object"], 0.0))
        ax.bar(spots + offset, values, width=width, color=colour, label=label,
               yerr=[lows, highs], capsize=5, ecolor=INK, error_kw={"linewidth": 1.1})
    ax.set_xticks(spots)
    ax.set_xticklabels([row[0] for row in rows])
    _style(ax, "Does building the rooms properly change the answer?", "",
           "lands on the object")
    ax.legend(frameon=False, fontsize=9)
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")
    figure.text(0.012, 0.015,
                "Identical method, identical metric. Primitive rooms held one of each "
                "object and were flat-shaded, which is why colour did so well and why a "
                "relation was never needed to identify anything.",
                fontsize=8, color="#5b5b66")
    return _save(figure, name)


def plot_asset_ablation(task: str, name: str, title: str, note: str) -> Path | None:
    arms = [
        ("rope3d", f"{task}_rope3d", "3D rotary"),
        ("learned_absolute", f"{task}_learned_absolute", "learned absolute"),
        ("none", f"{task}_none", "no position"),
        ("rgb", f"{task}_rgb_only", "colour, not Gemma"),
        ("rope3d", f"{task}_rope3d_tokens", "3D rotary,\nword-level query"),
    ]
    rows = [(k, lab, _load(tag, ASSETS)) for k, tag, lab in arms]
    rows = [(k, lab, run) for k, lab, run in rows if run]
    if not rows:
        return None
    figure, ax = plt.subplots(figsize=(8.0, 4.4))
    labels, values, lows, highs = [], [], [], []
    for _k, label, run in rows:
        held = run["held_out"]
        low, high = held.get("interval_95") or (held["hits_object"],) * 2
        labels.append(label)
        values.append(held["hits_object"])
        lows.append(max(held["hits_object"] - low, 0.0))
        highs.append(max(high - held["hits_object"], 0.0))
    ax.bar(labels, values, color=[SERIES[k] for k, _l, _r in rows], width=0.6,
           yerr=[lows, highs], capsize=6, ecolor=INK, error_kw={"linewidth": 1.2})
    chance = rows[0][2]["held_out"].get("chance_random_object")
    if chance:
        ax.axhline(chance, color=SERIES["chance"], linestyle="--", linewidth=1.5)
        ax.text(len(labels) - 0.45, chance + 0.015, f"chance {chance:.0%}",
                color=SERIES["chance"], fontsize=9, ha="right")
    _style(ax, title, "", "lands on the object")
    ax.set_ylim(0, max(max(values) * 1.3, (chance or 0) * 1.5))
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")
    held_rooms = len(rows[0][2]["held_out_rooms"])
    effective = rows[0][2]["held_out"].get("effective_n")
    figure.text(0.012, 0.015,
                f"{held_rooms} held-out asset rooms; intervals over "
                f"{effective} independent targets. {note}",
                fontsize=8, color="#5b5b66")
    return _save(figure, name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    made = [
        plot_corpus_comparison("corpus_comparison.png"),
        plot_asset_ablation(
            "disambig", "asset_disambiguation.png",
            "\"The cabinet nearest the bookshelf\", in rooms holding two cabinets",
            "Semantics narrows to the cabinets; only distance chooses between them.",
        ),
        plot_asset_ablation(
            "object", "asset_object.png",
            "Naming an object in a real-furniture room the model has never seen",
            "Mostly a semantic task, which is why no-position does respectably.",
        ),
        plot_asset_ablation(
            "relational", "asset_relational.png",
            "Finding an unnamed object by its relation to a named one",
            "The strictest form: the phrase never says what the target is.",
        ),
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
