#!/usr/bin/env python3
"""Discover objects in the point cloud and ask Gemma what each one is.

Perception ends here: the output is a metric scene graph built only from the
scan.  The author's words are never loaded, and no weights are trained.
"""

from __future__ import annotations

import argparse
import json

from PIL import Image

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.spatial_lens.discover import discover_objects
from semantic_3d_chat.spatial_lens.gemma_client import GemmaChat
from semantic_3d_chat.spatial_lens.naming import (
    NAME_PROMPT,
    disambiguate,
    highlight,
    normalize_answer,
    select_views,
    tight_box,
    vote,
)
from semantic_3d_chat.spatial_lens.perceive import SemanticCloud
from semantic_3d_chat.spatial_lens.scene_graph import (
    SceneGraph,
    SceneObject,
    build_free_grid,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", required=True)
    parser.add_argument("--views-per-object", type=int, default=3)
    parser.add_argument("--grid-resolution-m", type=float, default=0.05)
    parser.add_argument("--rover-radius-m", type=float, default=0.18)
    parser.add_argument("--drivable-height-m", type=float, default=0.06)
    parser.add_argument("--save-crops", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    room_root = PROJECT_ROOT / "data" / "spatial_lens" / args.room
    graph_path = room_root / "scene_graph.json"
    if graph_path.is_file() and not args.force:
        raise SystemExit(f"{graph_path} exists; pass --force to rebuild")

    cloud = SemanticCloud.load(room_root / "point_cloud.npz")
    proposals = discover_objects(cloud)
    if not proposals:
        raise SystemExit("No object proposals were discovered")
    print(f"discovered {len(proposals)} object proposals", flush=True)

    scan_root = room_root / "scans"
    manifest = json.loads((scan_root / "manifest.json").read_text(encoding="utf-8"))
    frames = list(manifest["frames"])
    image_size = (int(manifest["resolution"][0]), int(manifest["resolution"][1]))

    chat = GemmaChat.load()
    crops_root = room_root / "naming_views"
    if args.save_crops:
        crops_root.mkdir(parents=True, exist_ok=True)

    raw_names: list[str] = []
    raw_votes: list[dict[str, int]] = []
    confidences: list[float] = []
    for proposal in proposals:
        points = cloud.centers_m[proposal.voxel_indices]
        views = select_views(
            points, frames, image_size=image_size, max_views=args.views_per_object
        )
        answers: list[str] = []
        for order, view in enumerate(views):
            frame = frames[view.frame_index]
            image = Image.open(scan_root / frame["rgb_path"]).convert("RGB")
            box = tight_box(points, frame, image_size)
            picture = highlight(image, box, view.box)
            if args.save_crops:
                picture.save(crops_root / f"{proposal.proposal_id}_{order}.png")
            answers.append(normalize_answer(chat.ask_image(picture, NAME_PROMPT)))
        name, votes = vote(answers)
        confidence = votes.get(name, 0) / max(1, len(answers))
        print(
            f"  {proposal.proposal_id}: {name!r} from {answers} "
            f"({len(views)} views)",
            flush=True,
        )
        raw_names.append(name)
        raw_votes.append(votes)
        confidences.append(confidence)

    # Two blobs Gemma both called "table" need distinct handles before anything
    # can refer to them by name.
    unique_names = disambiguate(raw_names, [p.mean_rgb for p in proposals])
    objects = [
        SceneObject(
            object_id=proposal.proposal_id,
            name=unique_names[index],
            center_m=proposal.center_m,
            bbox_min_m=proposal.bbox_min_m,
            bbox_max_m=proposal.bbox_max_m,
            mean_rgb=proposal.mean_rgb,
            voxel_count=proposal.voxel_count,
            name_confidence=confidences[index],
            name_votes=raw_votes[index],
        )
        for index, proposal in enumerate(proposals)
    ]

    grid = build_free_grid(
        objects,
        cloud.room_size_m,
        resolution_m=args.grid_resolution_m,
        rover_radius_m=args.rover_radius_m,
        ignore_height_m=args.drivable_height_m,
    )
    graph = SceneGraph(
        room=args.room,
        room_size_m=cloud.room_size_m,
        objects=tuple(objects),
        free_grid=grid,
        grid_resolution_m=args.grid_resolution_m,
        rover_radius_m=args.rover_radius_m,
    )
    graph.save(graph_path)
    print()
    print(graph.describe())
    print()
    print(
        json.dumps(
            {"room": args.room, "objects": len(objects), "graph": str(graph_path)},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
