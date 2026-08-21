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
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from semantic_3d_chat.config import PROJECT_ROOT

POINTS = PROJECT_ROOT / "reports" / "gemma4" / "metrics" / "point_grounding"
ASSETS = PROJECT_ROOT / "reports" / "gemma4" / "metrics" / "point_grounding_assets"
CAPACITY = PROJECT_ROOT / "reports" / "gemma4" / "metrics" / "point_grounding_capacity"
# Its own folder: reports/figures already holds a decade of earlier experiments,
# and a reader cannot tell which pictures belong to which study once they are
# mixed together.
FIGURES = PROJECT_ROOT / "reports" / "figures" / "rope3d_study"

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


def _save(figure, name: str, footnote: str = "") -> Path:
    """Write the figure, wrapping the footnote so it cannot run off the edge."""

    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    if footnote:
        width = int(figure.get_size_inches()[0] * 15)
        lines = textwrap.wrap(" ".join(footnote.split()), width=width)
        figure.tight_layout(rect=(0, 0.03 + 0.035 * len(lines), 1, 1))
        figure.text(0.012, 0.012, "\n".join(lines), fontsize=8, color="#5b5b66",
                    va="bottom")
    else:
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
    return _save(figure, name, f"{held_rooms} held-out rooms, n={n}; bars are 95% Wilson intervals. " "Chance = a guesser that knows the answer is one of the room's objects.")


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
    return _save(figure, name, "Every point trained to the same number of optimiser steps, so the axis is " "data rather than compute. One seed per point.")


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
    return _save(figure, name, f"{len(finals)} rooms; median {int(np.median(finals))} views. Reachable = " f"visible from some camera in the room, so a drawer interior is not a " f"miss -- but that denominator flatters the plan.{tail}")


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
    return _save(figure, name, "Identical method, identical metric. Primitive rooms held one of each " "object and were flat-shaded, which is why colour did so well and why a " "relation was never needed to identify anything.")


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
    return _save(figure, name, f"{held_rooms} held-out asset rooms; intervals over " f"{effective} independent targets. {note}")


def plot_asset_scaling(name: str) -> Path | None:
    """Does more data help on the real-furniture corpus, and for which task?"""

    sizes = [4, 12, 24, 45]
    series = [
        ("scale_object_rope3d", "naming an object, 3D rotary", SERIES["rope3d"], "-"),
        ("scale_object_learned_absolute", "naming an object, absolute",
         SERIES["learned_absolute"], "-"),
        ("scale_disambig_rope3d", "\"which cabinet\", 3D rotary", SERIES["rgb"], "-"),
    ]
    figure, ax = plt.subplots(figsize=(7.6, 4.4))
    drawn = False
    for prefix, label, colour, style in series:
        xs, ys, los, his = [], [], [], []
        for size in sizes:
            run = _load(f"{prefix}_{size}", ASSETS)
            if not run:
                continue
            held = run["held_out"]
            xs.append(size)
            ys.append(held["hits_object"])
            low, high = held.get("interval_95") or (held["hits_object"],) * 2
            los.append(max(held["hits_object"] - low, 0.0))
            his.append(max(high - held["hits_object"], 0.0))
        if not xs:
            continue
        drawn = True
        ax.errorbar(xs, ys, yerr=[los, his], color=colour, marker="o", linestyle=style,
                    linewidth=2.0, capsize=5, label=label, markersize=6)
    if not drawn:
        plt.close(figure)
        return None
    # Two tasks with different chance lines share this axis, so both are drawn.
    for prefix, dash in (("scale_object_rope3d", (0, (4, 3))),
                         ("scale_disambig_rope3d", (0, (1, 2)))):
        run = _load(f"{prefix}_{sizes[-1]}", ASSETS) or _load(f"{prefix}_{sizes[0]}", ASSETS)
        chance = run["held_out"].get("chance_random_object") if run else None
        if chance:
            ax.axhline(chance, color=SERIES["chance"], linestyle=dash, linewidth=1.3)
            ax.text(sizes[-1], chance + 0.012, f"chance {chance:.0%}",
                    color=SERIES["chance"], fontsize=8, ha="right")
    _style(ax, "Real-furniture corpus: accuracy against training rooms",
           "training rooms", "lands on the object")
    ax.set_xticks(sizes)
    ax.legend(frameon=False, fontsize=9)
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")
    return _save(figure, name, "Fixed optimiser-step budget at every point, so the axis is data and not " "compute. 15 held-out rooms. One seed per point.")


def plot_capacity(name: str) -> Path | None:
    """Does a bigger reader solve the task a small one cannot?"""

    specs = [("disambig_dim256_layers4", "256x4"), ("disambig_dim512_layers6", "512x6"),
             ("disambig_dim768_layers8", "768x8")]
    rows = [(lab, _load(tag, CAPACITY)) for tag, lab in specs]
    rows = [(lab, r) for lab, r in rows if r]
    if len(rows) < 2:
        return None
    figure, ax = plt.subplots(figsize=(7.2, 4.3))
    xs = [r["parameters"] / 1e6 for _l, r in rows]
    ys = [r["held_out"]["hits_object"] for _l, r in rows]
    los = [max(y - (r["held_out"].get("interval_95") or [y, y])[0], 0) for y, (_l, r) in zip(ys, rows)]
    his = [max((r["held_out"].get("interval_95") or [y, y])[1] - y, 0) for y, (_l, r) in zip(ys, rows)]
    ax.errorbar(xs, ys, yerr=[los, his], color=SERIES["rope3d"], marker="o",
                linewidth=2.0, capsize=6, markersize=8)
    chance = rows[0][1]["held_out"].get("chance_random_object")
    if chance:
        ax.axhline(chance, color=SERIES["chance"], linestyle="--", linewidth=1.5)
        ax.text(xs[-1], chance + 0.015, f"chance {chance:.0%}", color=SERIES["chance"],
                fontsize=9, ha="right")
    for x, y, (lab, _r) in zip(xs, ys, rows):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(14, 4),
                    fontsize=8, color="#5b5b66", ha="left")
    _style(ax, "Fifteen times the reader does not buy the task",
           "reader parameters (millions)", "lands on the right cabinet")
    ax.set_xscale("log")
    # Plain parameter counts; nobody reads a model size as 6 x 10^1.
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:.0f}M" for x in xs])
    ax.minorticks_off()
    ax.set_xlim(min(xs) * 0.7, max(xs) * 1.45)
    ax.set_ylim(0, max(chance or 0.5, max(ys)) * 1.5)
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")
    return _save(figure, name, "Same data, same steps, same split. If depth or width were the missing " "ingredient for comparing two distances, it would appear here.")


def plot_wavelength(name: str) -> Path | None:
    """The one rotary hyperparameter nobody tuned, on both tasks."""

    bands = [2.0, 4.0, 8.0, 16.0, 32.0]
    figure, ax = plt.subplots(figsize=(7.4, 4.3))
    drawn = False
    for prefix, label, colour in (("object", "naming an object", SERIES["rope3d"]),
                                  ("disambig", "which cabbinet", SERIES["rgb"])):
        xs, ys = [], []
        for band in bands:
            run = _load(f"{prefix}_cycle{band}", CAPACITY)
            if run:
                xs.append(band)
                ys.append(run["held_out"]["hits_object"])
        if not xs:
            continue
        drawn = True
        pretty = label.replace("cabbinet", "cabinet")
        ax.plot(xs, ys, color=colour, marker="o", linewidth=2.0, markersize=7, label=pretty)
        run = _load(f"{prefix}_cycle8.0", CAPACITY)
        chance = run["held_out"].get("chance_random_object") if run else None
        if chance:
            ax.axhline(chance, color=colour, linestyle=":", linewidth=1.2, alpha=0.8)
            ax.text(bands[-1], chance + 0.012, f"chance {chance:.0%}", color=colour,
                    fontsize=8, ha="right")
    if not drawn:
        plt.close(figure)
        return None
    _style(ax, "The rotary band matters for one task and not the other",
           "metres per cycle (log)", "lands on the object")
    ax.set_xscale("log", base=2)
    ax.set_xticks(bands)
    ax.set_xticklabels([f"{b:g}" for b in bands])
    ax.legend(frameon=False, fontsize=9)
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")
    return _save(figure, name, "Naming peaks near room scale and falls off either side, so the knob is " "live. Disambiguation is flat and far below its own chance line at every " "band, so the knob has no purchase on it.")


def plot_model_comparison(name: str) -> Path | None:
    """Two model sizes, same rooms, same questions, same controls."""

    arms = []
    for tag, label in (("e2b_25", "Gemma E2B"), ("e4b_25", "Gemma E4B")):
        path = PROJECT_ROOT / "reports" / "gemma4" / "metrics" / f"rope3d_locate_{tag}.json"
        if path.is_file():
            arms.append((label, json.loads(path.read_text(encoding="utf-8"))))
    if len(arms) < 2:
        return None

    order = [
        ("raster", "raster order"),
        ("rope3d_xyz", "3D rotary"),
        ("rope3d_scrambled", "3D rotary,\nscrambled"),
        ("zeroed", "no scene"),
    ]
    figure, ax = plt.subplots(figsize=(8.0, 4.4))
    spots = np.arange(len(order))
    width = 0.36
    for offset, (label, run), colour in (
        (-width / 2, arms[0], "#6e7781"), (width / 2, arms[1], "#1f6feb")
    ):
        values, lows, highs = [], [], []
        for key, _pretty in order:
            cell = run["conditions"][key]
            value = cell["within_tolerance"]
            low, high = cell["interval_95"]
            values.append(value)
            lows.append(max(value - low, 0.0))
            highs.append(max(high - value, 0.0))
        ax.bar(spots + offset, values, width=width, color=colour, label=label,
               yerr=[lows, highs], capsize=5, ecolor=INK, error_kw={"linewidth": 1.1})
    baseline = arms[0][1]["random_baseline"]
    ax.axhline(baseline, color=SERIES["chance"], linestyle="--", linewidth=1.4)
    ax.text(len(order) - 0.4, baseline + 0.008, f"chance {baseline:.0%}",
            color=SERIES["chance"], fontsize=9, ha="right")
    ax.set_xticks(spots)
    ax.set_xticklabels([pretty for _k, pretty in order])
    _style(ax, "Does a bigger Gemma read the 3D field better?", "", "within 1 m")
    ax.legend(frameon=False, fontsize=9)
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")
    return _save(
        figure, name,
        f"Same {len(arms[0][1]['rooms'])} rooms and {arms[0][1]['queries_per_condition']} "
        "questions for both. Scale lifts every real condition by about seven points, "
        "but real positions still do not separate from scrambled ones (p = 0.35 for "
        "E2B, 0.18 for E4B), so neither model is using the rotary channel. E4B refuses "
        "every question when the scene is zeroed; E2B answers 76 of them.",
    )


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
        plot_asset_scaling("asset_scaling.png"),
        plot_capacity("capacity.png"),
        plot_model_comparison("model_comparison.png"),
        plot_wavelength("wavelength.png"),
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
