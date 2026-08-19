#!/usr/bin/env python3
"""Put both localization methods side by side on the same rooms and metric.

Method A needs no training: the semantic point cloud is rendered as a top-down
picture -- a layout the model already understands -- and Gemma is asked where
things are. Method B trains a 2.64M-parameter head to address the 3D field's
cells directly.

Both are scored by distance from the answer to the object's footprint, which is
zero on a hit, because that is what "could the rover drive there" means.
"""

from __future__ import annotations

import argparse
import json

from semantic_3d_chat.config import PROJECT_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topdown", default="reports/gemma4/metrics/spatial_lens_topdown_gemma.json")
    parser.add_argument("--grounding", default="reports/gemma4/metrics/spatial_lens_grounding.json")
    parser.add_argument("--output", default="reports/gemma4/metrics/spatial_lens_method_comparison.json")
    args = parser.parse_args()

    topdown = json.loads((PROJECT_ROOT / args.topdown).read_text(encoding="utf-8"))
    grounding = json.loads((PROJECT_ROOT / args.grounding).read_text(encoding="utf-8"))
    held = grounding["heldout"]

    report = {
        "schema": "semantic_3d_chat.spatial_lens.method_comparison.v1",
        "rooms": grounding["heldout_rooms"],
        "note": "both methods scored on the same held-out rooms and metric",
        "methods": {
            "zero_training_topdown_render": {
                "training_parameters": 0,
                "how": "semantic point cloud rendered top-down, read as an image",
                "lands_on_object": topdown["summary"]["lands_on_object"],
                "median_footprint_gap_m": topdown["summary"]["median_footprint_gap_m"],
                "within_0p5m_of_footprint": topdown["summary"]["footprint_gap_under_0p5m"],
                "answered": topdown["summary"]["answered"],
            },
            "trained_grounding_head": {
                "training_parameters": grounding["parameters"],
                "how": "head addresses the 3D field's cells from a phrase",
                "lands_on_object": held["hits_object_cell"],
                "median_footprint_gap_m": held["median_footprint_gap_m"],
                "within_0p5m_of_footprint": held["footprint_gap_under_0p5m"],
                "answered": held["examples"],
            },
        },
        "random_baseline_lands_on_object": grounding["random_baseline_hit_rate"],
    }
    a = report["methods"]["zero_training_topdown_render"]
    b = report["methods"]["trained_grounding_head"]
    report["trained_head_advantage"] = round(
        b["lands_on_object"] / max(a["lands_on_object"], 1e-9), 2
    )

    destination = PROJECT_ROOT / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"held-out rooms: {', '.join(report['rooms'])}\n")
    print(f"{'method':<34}{'on object':>11}{'gap':>9}{'<0.5 m':>9}{'params':>10}")
    for name, value in report["methods"].items():
        print(f"{name:<34}{value['lands_on_object']:>10.1%}"
              f"{value['median_footprint_gap_m']:>8.2f}m{value['within_0p5m_of_footprint']:>9.1%}"
              f"{value['training_parameters']:>10,}")
    print(f"{'random baseline':<34}"
          f"{report['random_baseline_lands_on_object']:>10.1%}")
    print(f"\ntrained head is {report['trained_head_advantage']}x better at landing on the object")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
