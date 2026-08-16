"""Extract complete-image dense features for an explicit scene batch.

The frozen vision model is loaded once, then reused across scenes.  This is an
execution optimization only: every RGB frame is still encoded as one complete
image and retains its spatial patch grid.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from semantic_3d_chat.config import load_config
from semantic_3d_chat.data.scene_variants import batch_scene_plans, batch_scene_splits
from semantic_3d_chat.device import select_device
from semantic_3d_chat.vision.encoder import (
    DenseImageEncoder,
    extract_manifest_features,
    load_configured_dense_image_encoder,
)


def selected_scene_ids(
    config: dict[str, Any],
    *,
    split: str,
    include_deferred_test: bool,
) -> tuple[str, ...]:
    plans = batch_scene_plans(config)
    splits = batch_scene_splits(config, plans)
    if splits is None:
        raise ValueError("Batch feature extraction requires explicit batch.splits")
    batch = config.get("batch")
    if not isinstance(batch, Mapping):
        raise TypeError("Batch feature extraction requires batch to be a mapping")
    raw_deferred = batch.get("deferred_splits", [])
    if (
        isinstance(raw_deferred, (str, bytes))
        or not isinstance(raw_deferred, (list, tuple))
        or not all(isinstance(value, str) for value in raw_deferred)
    ):
        raise TypeError("batch.deferred_splits must be a list or tuple of split names")
    allowed_splits = {"train", "validation", "test"}
    deferred = set(raw_deferred)
    if unknown := deferred - allowed_splits:
        raise ValueError(f"batch.deferred_splits contains unknown splits: {sorted(unknown)}")
    if split not in splits:
        raise ValueError(f"Requested split {split!r} is absent from batch.splits")
    if split in deferred and not include_deferred_test:
        raise ValueError(
            f"Deferred split {split!r} requires --include-deferred-test"
        )
    scene_ids = tuple(splits[split])
    if not scene_ids or len(scene_ids) != len(set(scene_ids)):
        raise ValueError(f"Batch split {split!r} must contain unique scenes")
    return scene_ids


def extract_batch_features(
    config: dict[str, Any],
    scene_ids: Sequence[str],
    *,
    local_files_only: bool,
    device: torch.device,
    encoder_loader: Callable[..., DenseImageEncoder] = load_configured_dense_image_encoder,
    extractor: Callable[..., dict[str, Any]] = extract_manifest_features,
) -> list[dict[str, Any]]:
    if not scene_ids or len(scene_ids) != len(set(scene_ids)):
        raise ValueError("scene_ids must be nonempty and unique")
    encoder = encoder_loader(
        config,
        device=device,
        local_files_only=local_files_only,
    )
    return [
        extractor(
            config,
            scene_id,
            local_files_only=local_files_only,
            device=device,
            encoder=encoder,
        )
        for scene_id in scene_ids
    ]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--include-deferred-test", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_config(args.config)
    scene_ids = selected_scene_ids(
        config,
        split=args.split,
        include_deferred_test=args.include_deferred_test,
    )
    device = select_device() if args.device == "auto" else torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    results = extract_batch_features(
        config,
        scene_ids,
        local_files_only=args.offline,
        device=device,
    )
    print(
        json.dumps(
            {
                "phase": "vision_batch_complete",
                "scene_ids": list(scene_ids),
                "scene_count": len(scene_ids),
                "frame_count": sum(int(item["frames"]) for item in results),
                "extracted": sum(int(item["extracted"]) for item in results),
                "reused": sum(int(item["reused"]) for item in results),
                "device": str(device),
                "one_complete_image_call_per_extracted_frame": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
