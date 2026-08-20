#!/usr/bin/env python3
"""Download the CC0 furniture the rooms are built from.

Textures are fetched at 1k. Gemma's vision encoder takes 224x224, and an object
occupies a fraction of that, so anything sharper is bytes the pipeline throws
away before it ever reaches the model.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from semantic_3d_chat.assets.catalogue import download_urls, indoor_assets
from semantic_3d_chat.config import PROJECT_ROOT

ROOT = PROJECT_ROOT / "data" / "assets"


def fetch_one(record, resolution: str, force: bool) -> tuple[str, str]:
    target = ROOT / record.asset_id
    marker = target / "complete.json"
    if marker.is_file() and not force:
        return record.asset_id, "cached"
    try:
        urls = download_urls(record.asset_id, resolution)
    except Exception as error:  # noqa: BLE001 - one bad asset must not stop the set
        return record.asset_id, f"skipped ({error})"
    target.mkdir(parents=True, exist_ok=True)
    gltf_name = None
    for relative, url in urls.items():
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and not force:
            if relative.endswith(".gltf"):
                gltf_name = relative
            continue
        request = urllib.request.Request(url, headers={"User-Agent": "semantic-3d-chat"})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                destination.write_bytes(response.read())
        except Exception as error:  # noqa: BLE001
            return record.asset_id, f"failed ({error})"
        if relative.endswith(".gltf"):
            gltf_name = relative
    if gltf_name is None:
        return record.asset_id, "failed (no gltf)"
    marker.write_text(
        json.dumps(
            {
                "asset_id": record.asset_id,
                "category": record.category,
                "size_m": [round(v, 4) for v in record.size_m],
                "gltf": gltf_name,
                "resolution": resolution,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return record.asset_id, "downloaded"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", default="1k", choices=["1k", "2k", "4k"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    records = indoor_assets()
    if args.limit:
        records = records[: args.limit]
    ROOT.mkdir(parents=True, exist_ok=True)
    print(f"{len(records)} assets to ensure at {args.resolution}")

    tally: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(fetch_one, record, args.resolution, args.force)
            for record in records
        ]
        for index, future in enumerate(futures, 1):
            asset_id, status = future.result()
            tally[status.split(" ")[0]] = tally.get(status.split(" ")[0], 0) + 1
            if not status.startswith(("cached", "downloaded")):
                print(f"  {asset_id}: {status}")
            if index % 25 == 0:
                print(f"  ... {index}/{len(records)}")

    manifest = []
    for record in records:
        marker = ROOT / record.asset_id / "complete.json"
        if marker.is_file():
            entry = json.loads(marker.read_text(encoding="utf-8"))
            entry["tags"] = list(record.tags)
            manifest.append(entry)
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\n{tally}")
    print(f"usable assets: {len(manifest)} -> {ROOT.relative_to(PROJECT_ROOT)}/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
