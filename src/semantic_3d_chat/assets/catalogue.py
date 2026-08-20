"""Pick real furniture out of Poly Haven's CC0 model library.

Hand-built primitives were a quiet confound. Flat-shaded coloured boxes make
colour almost sufficient to identify an object, which is why the colour-only
control scored two thirds of what Gemma's embeddings did; and placing one
distinct primitive per slot meant no room ever contained two chairs, so any
question of the form "which chair" had a single answer for trivial reasons.

Real scanned-and-photogrammetred assets fix both at once. They also come with
measured dimensions, so a chair is chair-sized without anyone deciding that.

The category assigned here never reaches the model. It exists so a room can be
composed with deliberate duplicates, and so the scorer knows what it placed;
every name the system actually reasons about still comes from Gemma looking at
the object.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

API = "https://api.polyhaven.com"

# Ordered: the first pattern that matches wins, so "coffee table" is a table
# rather than whatever else its tag list mentions.
CATEGORY_RULES: tuple[tuple[str, str], ...] = (
    ("bed", r"\bbed\b|mattress"),
    ("sofa", r"\bsofa\b|couch|settee"),
    ("armchair", r"armchair|\bottoman\b"),
    ("chair", r"\bchair\b|\bstool\b|\bseat\b"),
    ("desk", r"\bdesk\b"),
    ("table", r"\btable\b|console|commode|nightstand"),
    ("cabinet", r"cabinet|cupboard|wardrobe|dresser|drawer"),
    ("bookshelf", r"bookshelf|\bshelf\b|shelving|bookcase"),
    ("television", r"television|\btv\b|monitor|screen"),
    ("chandelier", r"chandelier"),
    ("lamp", r"\blamp\b|lantern|\bsconce\b|light fixture"),
    ("speaker", r"loudspeaker|boombox|speaker|stereo|radio"),
    ("barrel", r"\bbarrel\b"),
    ("crate", r"\bcrate\b|\bbox\b|\bcase\b"),
    ("basket", r"basket"),
    ("vase", r"\bvase\b|\burn\b"),
    ("pot", r"\bpot\b|\bpan\b|kettle|cauldron"),
    ("bottle", r"bottle|flask|canteen"),
    ("books", r"\bbook\b|books|encyclopedia"),
    ("plant", r"potted|houseplant|\bfern\b|monstera|succulent"),
    ("clock", r"\bclock\b"),
    ("bucket", r"bucket|\bpail\b"),
    ("toolbox", r"toolbox|tool box"),
    ("suitcase", r"suitcase|luggage|trunk"),
    ("rug", r"\brug\b|carpet"),
    ("mirror", r"mirror"),
    ("painting", r"painting|\bframe\b|artwork"),
    ("ball", r"\bball\b|football|basketball"),
    ("guitar", r"guitar|ukulele|banjo"),
    ("camera", r"\bcamera\b"),
    ("fan", r"\bfan\b"),
    ("heater", r"heater|radiator|stove"),
    ("sign", r"\bsign\b"),
)

# Anything that reads as outdoor, industrial-yard or weaponry has no business
# standing in a living room, whatever its size says.
EXCLUDE = re.compile(
    r"sword|katana|estoc|dagger|axe|gun|rifle|pistol|grenade|ammo|"
    r"rock|boulder|stone|cliff|tree|bush|grass|log|stump|"
    r"gravestone|tombstone|hydrant|bollard|traffic|manhole|pallet",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AssetRecord:
    """One downloadable object, with the size it really is."""

    asset_id: str
    category: str
    size_m: tuple[float, float, float]
    tags: tuple[str, ...]

    @property
    def footprint_m(self) -> float:
        return max(self.size_m[0], self.size_m[1])

    @property
    def height_m(self) -> float:
        return self.size_m[2]


def _fetch_json(url: str, timeout: float = 60.0) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "semantic-3d-chat"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def classify(asset_id: str, tags: list[str], categories: list[str]) -> str | None:
    """Which everyday word describes this object, if any."""

    haystack = " ".join([re.sub(r"[_\d]+", " ", asset_id), *tags, *categories]).lower()
    if EXCLUDE.search(haystack):
        return None
    for name, pattern in CATEGORY_RULES:
        if re.search(pattern, haystack):
            return name
    return None


def indoor_assets(
    *,
    min_extent_m: float = 0.12,
    max_extent_m: float = 2.6,
    catalogue: dict[str, Any] | None = None,
) -> list[AssetRecord]:
    """Everything CC0 that could plausibly stand in a room."""

    catalogue = catalogue if catalogue is not None else _fetch_json(f"{API}/assets?type=models")
    records: list[AssetRecord] = []
    for asset_id, entry in catalogue.items():
        dimensions = entry.get("dimensions")
        if not dimensions or len(dimensions) != 3:
            continue
        size = tuple(float(value) / 1000.0 for value in dimensions)
        if not (min_extent_m <= max(size) <= max_extent_m):
            continue
        category = classify(
            asset_id, list(entry.get("tags", [])), list(entry.get("categories", []))
        )
        if category is None:
            continue
        records.append(
            AssetRecord(
                asset_id=asset_id,
                category=category,
                size_m=size,  # type: ignore[arg-type]
                tags=tuple(entry.get("tags", [])),
            )
        )
    records.sort(key=lambda record: (record.category, record.asset_id))
    return records


def download_urls(asset_id: str, resolution: str = "2k") -> dict[str, str]:
    """The glTF and every texture it needs, as relative-path -> url."""

    files = _fetch_json(f"{API}/files/{asset_id}")
    if "gltf" not in files or resolution not in files["gltf"]:
        raise ValueError(f"{asset_id} has no glTF at {resolution}")
    entry = files["gltf"][resolution]["gltf"]
    urls = {Path(entry["url"]).name: entry["url"]}
    for relative, info in entry.get("include", {}).items():
        urls[relative] = info["url"]
    return urls


__all__ = [
    "API",
    "CATEGORY_RULES",
    "AssetRecord",
    "classify",
    "download_urls",
    "indoor_assets",
]
