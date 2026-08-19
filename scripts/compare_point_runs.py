#!/usr/bin/env python3
"""Compare two sweep runs on the items they both scored.

Every run is evaluated on the same held-out examples in the same order, so the
right comparison is paired: how often did one scheme get an item the other
missed. Reading two overlapping Wilson intervals instead answers a question
nobody asked -- whether two independent samples differ -- and needs a far larger
effect before it can say anything at all.
"""

from __future__ import annotations

import argparse
import json

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.proportions import mcnemar_exact, wilson_interval

ROOT = PROJECT_ROOT / "reports" / "gemma4" / "metrics" / "point_grounding"


def load(tag: str) -> dict:
    path = ROOT / f"{tag}.json"
    if not path.is_file():
        raise SystemExit(f"no run named {tag}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first")
    parser.add_argument("second")
    args = parser.parse_args()

    a, b = load(args.first), load(args.second)
    first = a["held_out"].get("per_item")
    second = b["held_out"].get("per_item")
    if not first or not second:
        raise SystemExit("these runs predate per-item recording; re-run them")
    shared = sorted(set(first) & set(second))
    if not shared:
        raise SystemExit("the two runs share no items")

    left = [float(first[k]) for k in shared]
    right = [float(second[k]) for k in shared]
    test = mcnemar_exact(left, right)
    print(f"{args.first} vs {args.second}   on {len(shared)} shared items\n")
    for tag, values in ((args.first, left), (args.second, right)):
        hits = int(sum(values))
        low, high = wilson_interval(hits, len(values))
        print(
            f"  {tag:<36} {hits:>3}/{len(values)}  "
            f"{hits / len(values):.3f}  [{low}, {high}]"
        )
    print(
        f"\n  {args.first} only: {test['only_first']}"
        f"   {args.second} only: {test['only_second']}"
        f"   McNemar p = {test['p_value']}"
    )
    if test["p_value"] > 0.05:
        print("  Not distinguishable at this sample size.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
