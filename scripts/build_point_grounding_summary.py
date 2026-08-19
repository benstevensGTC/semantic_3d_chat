#!/usr/bin/env python3
"""Collect the position ablation and the scaling curve into one table."""

from __future__ import annotations

import json

from semantic_3d_chat.config import PROJECT_ROOT

ROOT = PROJECT_ROOT / "reports" / "gemma4" / "metrics" / "point_grounding"


def main() -> int:
    runs = {}
    for path in sorted(ROOT.glob("*.json")):
        if path.name == "summary.json":
            continue
        runs[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    if not runs:
        raise SystemExit(f"no runs under {ROOT}")

    def row(name: str) -> dict[str, object] | None:
        run = runs.get(name)
        if run is None:
            return None
        held = run["held_out"]
        return {
            "rooms": run["train_room_count"],
            "hits_object": held["hits_object"],
            "chance": held["chance_hits_object"],
            "lift": held["lift_over_chance"],
            "median_gap_m": held["median_gap_m"],
            "gap_under_0p5m": held["gap_under_0p5m"],
            "parameters": run["parameters"],
        }

    ablation = {
        mode: row(f"{mode}_rooms19")
        for mode in ("rope3d", "learned_absolute", "none")
    }
    ablation["rope3d_no_augmentation"] = row("rope3d_rooms19_noaug")
    relational = {
        mode: row(f"relational_{mode}_rooms19")
        for mode in ("rope3d", "learned_absolute", "none")
    }
    scaling = {
        mode: {
            str(n): row(f"{mode}_rooms{n}")
            for n in (2, 4, 8, 12, 16, 19)
            if row(f"{mode}_rooms{n}")
        }
        for mode in ("rope3d", "learned_absolute", "relational_rope3d")
    }

    any_run = next(iter(runs.values()))
    summary = {
        "held_out_rooms": any_run["held_out_rooms"],
        "token_budget": any_run["token_budget"],
        "epochs": any_run["epochs"],
        "position_ablation": {k: v for k, v in ablation.items() if v},
        "relational_ablation": {k: v for k, v in relational.items() if v},
        "scaling": {k: v for k, v in scaling.items() if v},
    }
    destination = ROOT / "summary.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"held-out rooms: {', '.join(summary['held_out_rooms'])}\n")
    print("position ablation, 19 training rooms")
    print(f"  {'scheme':<24} {'on object':>10} {'chance':>8} {'lift':>7} {'median gap':>11}")
    for name, data in summary["position_ablation"].items():
        print(f"  {name:<24} {data['hits_object']:>9.1%} {data['chance']:>7.1%} "
              f"{data['lift']:>6.1f}x {data['median_gap_m']:>10.2f}m")
    if summary["relational_ablation"]:
        print("\nrelational phrases -- 'the chair nearest the shelf' -- 19 rooms")
        print(f"  {'scheme':<24} {'on object':>10} {'chance':>8} {'lift':>7} {'median gap':>11}")
        for name, data in summary["relational_ablation"].items():
            print(f"  {name:<24} {data['hits_object']:>9.1%} {data['chance']:>7.1%} "
                  f"{data['lift']:>6.1f}x {data['median_gap_m']:>10.2f}m")
    print("\nscaling: held-out accuracy against training rooms")
    header = sorted({n for mode in scaling.values() for n in mode}, key=int)
    print(f"  {'scheme':<24} " + " ".join(f"{n:>7}" for n in header))
    for mode, points in scaling.items():
        cells = " ".join(
            f"{points[n]['hits_object']:>6.1%}" if n in points else f"{'-':>7}"
            for n in header
        )
        print(f"  {mode:<24} {cells}")
    print(f"\nwrote {destination.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
