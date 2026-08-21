#!/usr/bin/env python3
"""Every number this study produced, in one table, with its baseline attached.

Nothing here is a bare proportion. Each task has its own chance line -- one in
however many objects for naming, one in k for "which cabinet" -- and quoting
accuracy without it invites reading 65% and 23% as though they were on the same
scale, when one is three times its baseline and the other is half of its own.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from semantic_3d_chat.config import PROJECT_ROOT

METRICS = PROJECT_ROOT / "reports" / "gemma4" / "metrics"
ASSETS = METRICS / "point_grounding_assets"
CAPACITY = METRICS / "point_grounding_capacity"


def load(tag: str, root: Path) -> dict | None:
    path = root / f"{tag}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def line(label: str, run: dict | None, width: int = 30) -> str | None:
    if not run:
        return None
    held = run["held_out"]
    low, high = held.get("interval_95") or (held["hits_object"],) * 2
    chance = held.get("chance_random_object") or 0.0
    lift = held["hits_object"] / chance if chance else float("nan")
    return (
        f"  {label:<{width}} {held['hits_object']:>6.1%}  "
        f"[{low:.2f},{high:.2f}]  chance {chance:>5.1%}  "
        f"lift {lift:>4.2f}x  gap {held['median_gap_m']:>5.2f}m"
    )


def block(title: str, rows: list[tuple[str, str]], root: Path = ASSETS) -> None:
    printed = [line(label, load(tag, root)) for label, tag in rows]
    printed = [p for p in printed if p]
    if not printed:
        return
    print(f"\n{title}")
    for row in printed:
        print(row)


def main() -> int:
    print("=" * 78)
    print("SEMANTIC 3D CHAT — 3D ROTARY POSITION STUDY")
    print("120 asset rooms · 90 training · 30 held out · fixed step budget")
    print("=" * 78)

    block("NAMING AN OBJECT  (the phrase says what to find)", [
        ("3D rotary", "object_rope3d"),
        ("learned absolute", "object_learned_absolute"),
        ("no position at all", "object_none"),
        ("colour instead of Gemma", "object_rgb_only"),
    ])
    block("WHICH CABINET  (two of a kind; only distance decides)", [
        ("3D rotary", "disambig_rope3d"),
        ("learned absolute", "disambig_learned_absolute"),
        ("no position at all", "disambig_none"),
        ("3D rotary + word-level query", "disambig_rope3d_tokens"),
        ("colour instead of Gemma", "disambig_rgb_only"),
    ])
    block("RELATIONAL  (the target is never named)", [
        ("3D rotary", "relational_rope3d"),
        ("learned absolute", "relational_learned_absolute"),
        ("no position at all", "relational_none"),
        ("3D rotary + word-level query", "relational_rope3d_tokens"),
    ])
    block("READER CAPACITY on the task that fails", [
        ("4.1M  (256 wide, 4 deep)", "disambig_dim256_layers4"),
        ("21.3M (512 wide, 6 deep)", "disambig_dim512_layers6"),
        ("60.8M (768 wide, 8 deep)", "disambig_dim768_layers8"),
    ], CAPACITY)
    block("READER CAPACITY on the task that works", [
        ("4.1M", "object_cycle8.0"),
        ("21.3M", "object_dim512_layers6"),
    ], CAPACITY)
    block("ROTARY BAND, naming", [(f"{b:g} m per cycle", f"object_cycle{b}")
                                 for b in (2.0, 4.0, 8.0, 16.0, 32.0)], CAPACITY)
    block("ROTARY BAND, which cabinet", [(f"{b:g} m per cycle", f"disambig_cycle{b}")
                                         for b in (2.0, 4.0, 8.0, 16.0, 32.0)], CAPACITY)
    block("POINTS PER ROOM, naming", [(f"{b} points", f"object_points{b}")
                                      for b in (512, 2048)], CAPACITY)
    block("POINTS PER ROOM, which cabinet", [(f"{b} points", f"disambig_points{b}")
                                             for b in (512, 2048)], CAPACITY)

    print("\nSCALING  (same optimiser steps at every point)")
    print(f"  {'rooms':>6}  {'rotary':>18}  {'absolute':>18}  {'which cabinet':>14}")
    for n in (6, 12, 24, 48, 90):
        a = load(f"scale_object_rope3d_{n}", ASSETS)
        b = load(f"scale_object_learned_absolute_{n}", ASSETS)
        c = load(f"scale_disambig_rope3d_{n}", ASSETS)
        def fmt(run: dict | None) -> str:
            if not run:
                return "-"
            held = run["held_out"]
            return f"{held['hits_object']:.1%} / {held['median_gap_m']:.2f}m"

        cc = f"{c['held_out']['hits_object']:.1%}" if c else "no capable rooms"
        print(f"  {n:>6}  {fmt(a):>18}  {fmt(b):>18}  {cc:>14}")

    for task, label in (("object", "naming"), ("disambig", "which cabinet")):
        runs = [load(f"{task}_seed{s}", CAPACITY) for s in (11, 22, 33)]
        base = load(f"{task}_cycle8.0", CAPACITY) or load(f"{task}_rope3d", ASSETS)
        values = [r["held_out"]["hits_object"] for r in [base, *runs] if r]
        gaps = [r["held_out"]["median_gap_m"] for r in [base, *runs] if r]
        if len(values) < 2:
            continue
        print(f"\nSEED SPREAD, {label} ({len(values)} seeds)")
        print(f"  accuracy {min(values):.1%}–{max(values):.1%} "
              f"(sd {statistics.pstdev(values):.3f}), "
              f"gap {min(gaps):.2f}–{max(gaps):.2f}m")
        print("  Differences smaller than this spread are not readable "
              "from a single seed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
