"""Measure every downloaded asset by importing it, not by trusting its metadata.

Poly Haven publishes a ``dimensions`` field and it is wrong often enough to
matter: for a third of the catalogue it disagrees with the mesh by more than a
tenth, sometimes with x and y transposed, and in a few cases by a factor of ten
or a hundred. The builder fits a mesh inside the requested box with a single
scale factor, so a wrong ratio does not stretch anything -- it silently shrinks
the whole object, and the scorer's ground truth then records a size the room
does not contain. One asset was being built at 34% of its recorded height in
eight different rooms.

Importing each file and reading the bounds Blender computes removes the
disagreement at its source: after this, requested size and built size are the
same number.

    blender --background --python blender/measure_assets.py -- --assets data/assets
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import mathutils

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scene_utils import atomic_json, blender_cli_args, reset_scene  # type: ignore[import-not-found]


def _measure(path: Path) -> list[float] | None:
    reset_scene()
    before = set(bpy.data.objects)
    try:
        bpy.ops.import_scene.gltf(filepath=str(path))
    except Exception:  # noqa: BLE001 - a broken asset must not stop the catalogue
        return None
    meshes = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    if not meshes:
        return None
    bpy.context.view_layer.update()
    corners = []
    for obj in meshes:
        corners.extend(obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box)
    if not corners:
        return None
    low = [min(c[a] for c in corners) for a in range(3)]
    high = [max(c[a] for c in corners) for a in range(3)]
    return [round(high[a] - low[a], 5) for a in range(3)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True)
    args = parser.parse_args(blender_cli_args())

    root = Path(args.assets)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    measured, dropped, changed = [], [], 0
    for entry in manifest:
        gltf = root / entry["asset_id"] / entry["gltf"]
        size = _measure(gltf) if gltf.is_file() else None
        if size is None or min(size) <= 0.0:
            dropped.append(entry["asset_id"])
            continue
        claimed = [float(v) for v in entry["size_m"]]
        worst = max(abs(size[a] - claimed[a]) / max(claimed[a], 1e-6) for a in range(3))
        if worst > 0.1:
            changed += 1
        entry = dict(entry)
        entry["published_size_m"] = [round(v, 5) for v in claimed]
        entry["size_m"] = size
        entry["metadata_error"] = round(worst, 4)
        measured.append(entry)

    atomic_json(root / "manifest.json", measured)
    print(
        json.dumps(
            {
                "measured": len(measured),
                "dropped": dropped,
                "disagreed_with_metadata_over_10pct": changed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
