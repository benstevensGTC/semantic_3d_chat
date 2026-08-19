#!/usr/bin/env python3
"""Generate a set of varied authored rooms for training and held-out testing.

Rooms differ in size, palette, which furniture is present, how many pieces, and
where everything sits. The point is breadth: a grounding head trained on a few
rooms has to work in rooms it has never seen, so the training rooms must not all
look alike.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SHAPES = ["table", "chair", "shelf", "cabinet", "lamp", "bed", "screen", "sphere", "box"]
COLORS = [
    "wood", "dark_wood", "charcoal", "cream", "teal", "red", "blue",
    "green", "orange", "purple", "gray", "white", "black", "terracotta",
]
SIZES = {
    "table": (1.2, 0.8, 0.74), "chair": (0.5, 0.5, 0.92), "shelf": (0.9, 0.35, 1.8),
    "cabinet": (1.0, 0.45, 1.1), "lamp": (0.36, 0.36, 1.6), "bed": (1.4, 2.0, 0.55),
    "screen": (1.1, 0.2, 0.7), "sphere": (0.36, 0.36, 0.36), "box": (0.5, 0.5, 0.5),
}


def make_room(name: str, seed: int) -> dict:
    rng = random.Random(seed)
    width = rng.choice([5.0, 5.5, 6.0, 6.5, 7.0])
    depth = rng.choice([4.5, 5.0, 5.5, 6.0])
    count = rng.randint(4, 7)
    shapes = rng.sample(SHAPES, count)
    placed: list[dict] = []
    for shape in shapes:
        base = SIZES[shape]
        scale = rng.uniform(0.85, 1.2)
        size = [round(base[0] * scale, 2), round(base[1] * scale, 2), round(base[2] * scale, 2)]
        for _ in range(200):
            x = round(rng.uniform(-width / 2 + size[0] / 2 + 0.3, width / 2 - size[0] / 2 - 0.3), 2)
            y = round(rng.uniform(-depth / 2 + size[1] / 2 + 0.3, depth / 2 - size[1] / 2 - 0.3), 2)
            clash = any(
                abs(x - o["position_m"][0]) < (size[0] + o["size_m"][0]) / 2 + 0.45
                and abs(y - o["position_m"][1]) < (size[1] + o["size_m"][1]) / 2 + 0.45
                for o in placed
            )
            if not clash:
                placed.append({
                    "name": shape, "shape": shape, "color": rng.choice(COLORS),
                    "position_m": [x, y], "size_m": size,
                    "yaw_degrees": rng.choice([0, 0, 90, 180, 270]),
                })
                break
    return {
        "schema": "semantic_3d_chat.spatial_lens.room_spec.v1",
        "name": name,
        "size_m": [width, depth, 2.8],
        "floor_color": rng.choice(["wood", "dark_wood", "gray", "cream"]),
        "wall_color": rng.choice(["cream", "white", "gray"]),
        "objects": placed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--prefix", default="room")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="rooms")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for index in range(args.count):
        name = f"{args.prefix}{index:02d}"
        spec = make_room(name, args.seed * 1000 + index)
        (out / f"{name}.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        made.append((name, len(spec["objects"])))
    for name, n in made:
        print(f"  {name}: {n} objects")
    print(f"\nwrote {len(made)} rooms to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
