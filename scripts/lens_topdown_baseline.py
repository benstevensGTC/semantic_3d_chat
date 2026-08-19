#!/usr/bin/env python3
"""Zero-training control: can a convention Gemma already knows replace the head?

The grounding head exists because a bird's-eye grid of Gemma feature vectors is
a layout the model has never seen. The obvious way to avoid training is to use a
layout it HAS seen -- an ordinary top-down picture -- so this renders the same
semantic point cloud as an image, hands it to Gemma as pixels, and asks the same
"where is X" question.

If this matched the trained head, the head would be unnecessary. Measuring it is
the honest way to find out.
"""

from __future__ import annotations

import argparse
import json
import math
import re

import numpy as np
from PIL import Image, ImageDraw

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.spatial_lens.perceive import SemanticCloud
from semantic_3d_chat.spatial_lens.scene_graph import SceneGraph


def render_topdown(cloud: SemanticCloud, size: int = 448, floor_margin_m: float = 0.09):
    """Paint the point cloud from above: real colour, real metric layout.

    Single pixels per voxel produced a sparse dust that the model read as
    nothing, so points are painted as small discs and the picture is drawn at
    high resolution before downsampling. Axis ticks are labelled in metres so a
    reading model has the same coordinate frame the trained head is given.
    """

    width, depth, height = cloud.room_size_m
    centers = np.asarray(cloud.centers_m)
    rgb = np.asarray(cloud.rgb)
    standing = (centers[:, 2] > floor_margin_m) & (centers[:, 2] < height - floor_margin_m)
    centers, rgb = centers[standing], rgb[standing]
    order = np.argsort(centers[:, 2])          # paint taller points last
    centers, rgb = centers[order], rgb[order]

    scale = 2
    canvas = Image.new("RGB", (size * scale, size * scale), (240, 240, 238))
    painter = ImageDraw.Draw(canvas)
    radius = max(2, int(size * scale / 110))
    for point, colour in zip(centers, rgb, strict=True):
        x = (point[0] + width / 2) / width * size * scale
        y = (point[1] + depth / 2) / depth * size * scale
        fill = tuple(int(v) for v in (np.clip(colour, 0, 1) * 255).astype(int))
        painter.ellipse([x - radius, y - radius, x + radius, y + radius], fill=fill)

    for fraction in range(1, 4):
        offset = size * scale * fraction / 4
        painter.line([(offset, 0), (offset, size * scale)], fill=(190, 190, 190), width=2)
        painter.line([(0, offset), (size * scale, offset)], fill=(190, 190, 190), width=2)
    image = canvas.resize((size, size), Image.LANCZOS)

    labeller = ImageDraw.Draw(image)
    for fraction in range(5):
        x_metres = -width / 2 + width * fraction / 4
        y_metres = -depth / 2 + depth * fraction / 4
        offset = size * fraction / 4
        labeller.text((min(offset + 2, size - 26), 2), f"x={x_metres:+.1f}", fill=(90, 90, 90))
        labeller.text((2, min(offset + 2, size - 12)), f"y={y_metres:+.1f}", fill=(90, 90, 90))
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rooms", nargs="+", required=True)
    parser.add_argument("--reasoner", choices=("gemma", "ollama"), default="gemma")
    parser.add_argument("--ollama-model", default="qwen3.8:27b")
    parser.add_argument("--output")
    args = parser.parse_args()

    from semantic_3d_chat.spatial_lens.gemma_client import GemmaChat, OllamaChat

    chat = (
        OllamaChat.load(model=args.ollama_model)
        if args.reasoner == "ollama"
        else GemmaChat.load()
    )
    rows, errors, gaps = [], [], []
    for room in args.rooms:
        root = PROJECT_ROOT / "data" / "spatial_lens" / room
        cloud = SemanticCloud.load(root / "point_cloud.npz")
        graph = SceneGraph.load(root / "scene_graph.json")
        image = render_topdown(cloud)
        (root / "topdown.png").parent.mkdir(parents=True, exist_ok=True)
        image.save(root / "topdown.png")
        width, depth, _ = cloud.room_size_m
        print(f"\n[{room}]")
        for item in graph.objects:
            question = (
                f"This is a top-down map of a room seen from above. The image "
                f"spans x from {-width/2:.1f} to {width/2:.1f} metres (left to "
                f"right) and y from {-depth/2:.1f} to {depth/2:.1f} metres (top "
                f"to bottom). Where is the {item.name}? Reply with only "
                '{"x": <metres>, "y": <metres>}.'
            )
            reply = chat.ask_image(image, question, max_new_tokens=40)
            match = re.search(r"\{.*?\}", reply, re.DOTALL)
            if match is None:
                print(f"  {item.name:<16} no parse"); continue
            try:
                payload = json.loads(match.group(0))
                said = (float(payload["x"]), float(payload["y"]))
            except (ValueError, KeyError, TypeError):
                print(f"  {item.name:<16} no parse"); continue
            truth = item.center_m[:2]
            error = math.dist(said, truth)
            # Same footprint metric the trained head is scored on: zero when the
            # answer lands anywhere on the object.
            gap = math.hypot(
                max(item.bbox_min_m[0] - said[0], 0.0, said[0] - item.bbox_max_m[0]),
                max(item.bbox_min_m[1] - said[1], 0.0, said[1] - item.bbox_max_m[1]),
            )
            errors.append(error)
            gaps.append(gap)
            rows.append({"room": room, "object": item.name,
                         "said_m": [round(v, 2) for v in said],
                         "true_m": [round(v, 2) for v in truth],
                         "error_m": round(error, 3),
                         "footprint_gap_m": round(gap, 3)})
            print(f"  {item.name:<16} ({said[0]:+.2f},{said[1]:+.2f}) true "
                  f"({truth[0]:+.2f},{truth[1]:+.2f})  err {error:.2f} m", flush=True)

    summary = {
        "answered": len(errors),
        "median_error_m": round(float(np.median(errors)), 3) if errors else None,
        "within_1m": round(float(np.mean([e <= 1.0 for e in errors])), 3) if errors else None,
        "lands_on_object": round(float(np.mean([g <= 0.0 for g in gaps])), 4) if gaps else None,
        "median_footprint_gap_m": round(float(np.median(gaps)), 3) if gaps else None,
        "footprint_gap_under_0p5m": round(float(np.mean([g <= 0.5 for g in gaps])), 4) if gaps else None,
    }
    print(f"\nzero-training top-down baseline: {summary}")
    if args.output:
        destination = PROJECT_ROOT / args.output
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({
            "schema": "semantic_3d_chat.spatial_lens.topdown_baseline.v1",
            "rooms": args.rooms, "training_used": False, "reasoner": args.reasoner,
            "summary": summary, "results": rows,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
