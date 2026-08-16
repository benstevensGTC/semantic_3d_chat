"""Is the 3D information present in the scene tokens, or lost building them?

The V15 causal controls showed that substituting another room's prefix, or
shuffling the 256 content latents, barely changes the controller's action
choice.  There are two very different explanations:

    (a) the point cloud -> scene-token bridge never encoded room-specific
        geometry, so there is nothing to read; or
    (b) the geometry is present in the tokens and the readout throws it away.

These are distinguished by a probe that needs no Gemma forward at all.  Each of
the 256 content latents has a fixed Halton anchor in normalized room XYZ
(:func:`spatial_anchors`) and is built from a distance-weighted average of the
blocks near that anchor.  So if the bridge preserved geometry, a *linear* map
should recover a token's own anchor from the token vector, and should do so in
rooms the probe never saw.

Measured on the 41-room cache: R^2 = 0.98-0.99 per axis on held-out rooms, and
room identity is 86% decodable from a single token against 2.4% chance.  The
answer is (b).  This matters because (a) would require redesigning the scene
encoder, while (b) is a readout problem -- and the readout is the cheap part.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.torch import load_file

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.scene_encoder.perceiver import spatial_anchors

PROBE_SCHEMA = "semantic_3d_chat.v15_scene_token_probe.v1"


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _ridge(features: np.ndarray, targets: np.ndarray, penalty: float) -> np.ndarray:
    design = np.hstack([features, np.ones((len(features), 1))])
    gram = design.T @ design + penalty * np.eye(design.shape[1])
    return np.linalg.solve(gram, design.T @ targets)


def _apply(weights: np.ndarray, features: np.ndarray) -> np.ndarray:
    return np.hstack([features, np.ones((len(features), 1))]) @ weights


def probe_scene_tokens(
    prefix_cache_root: str | Path,
    *,
    holdout_room_count: int = 8,
    ridge_penalty: float = 1e2,
    content_token_count: int = 256,
) -> dict[str, Any]:
    """Linearly decode each token's room anchor, and its room identity."""

    root = _rooted(prefix_cache_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    scenes = sorted(manifest["scenes"])
    if len(scenes) <= holdout_room_count + 1:
        raise ValueError("Probe needs more rooms than the requested holdout")
    anchors = spatial_anchors(content_token_count).numpy()

    tokens: list[np.ndarray] = []
    room_index: list[np.ndarray] = []
    for index, scene_id in enumerate(scenes):
        filename = manifest["scenes"][scene_id]["filename"]
        prefix = load_file(str(root / filename))["scene_prefix"].float().numpy()[0]
        # Drop the two native Gemma boundary tokens; they carry no scene content.
        content = prefix[1:-1]
        if content.shape[0] != content_token_count:
            raise ValueError(f"{scene_id} has {content.shape[0]} content latents")
        tokens.append(content)
        room_index.append(np.full(content_token_count, index))

    features = np.concatenate(tokens)
    rooms = np.concatenate(room_index)
    targets = np.tile(anchors, (len(scenes), 1))

    # Split by ROOM: the probe must generalize to rooms it never fitted.
    train_mask = rooms < len(scenes) - holdout_room_count
    test_mask = ~train_mask
    mean = features[train_mask].mean(axis=0)
    scale = features[train_mask].std(axis=0) + 1e-6
    standardized = (features - mean) / scale

    weights = _ridge(standardized[train_mask], targets[train_mask], ridge_penalty)
    predicted = _apply(weights, standardized[test_mask])
    observed = targets[test_mask]
    residual = ((observed - predicted) ** 2).sum(axis=0)
    total = ((observed - observed.mean(axis=0)) ** 2).sum(axis=0)
    r_squared = 1.0 - residual / total

    identity = np.eye(len(scenes))[rooms]
    identity_weights = _ridge(
        standardized[train_mask], identity[train_mask], ridge_penalty
    )
    identity_accuracy = float(
        (_apply(identity_weights, standardized[train_mask]).argmax(axis=1)
         == rooms[train_mask]).mean()
    )

    return {
        "schema": PROBE_SCHEMA,
        "prefix_cache_root": root.as_posix(),
        "room_count": len(scenes),
        "holdout_room_count": holdout_room_count,
        "content_token_count": content_token_count,
        "token_count": int(features.shape[0]),
        "hidden_size": int(features.shape[1]),
        "gemma_forward_required": False,
        "anchor_regression": {
            "held_out_rooms": True,
            "r_squared_xyz": [float(value) for value in r_squared],
            "mean_absolute_error_normalized_xyz": [
                float(value) for value in np.abs(observed - predicted).mean(axis=0)
            ],
        },
        "room_identity_decoding": {
            "linear_accuracy": identity_accuracy,
            "chance_accuracy": 1.0 / len(scenes),
        },
        "conclusion": (
            "geometry_present_in_tokens"
            if float(min(r_squared)) > 0.75
            else "geometry_absent_from_tokens"
        ),
    }


def format_probe(report: dict[str, Any]) -> str:
    regression = report["anchor_regression"]
    identity = report["room_identity_decoding"]
    axes = ", ".join(f"{value:.3f}" for value in regression["r_squared_xyz"])
    return (
        f"rooms={report['room_count']} tokens={report['token_count']}\n"
        f"anchor R^2 (held-out rooms) x,y,z = {axes}\n"
        f"room identity from one token = {identity['linear_accuracy']:.1%} "
        f"(chance {identity['chance_accuracy']:.1%})\n"
        f"conclusion: {report['conclusion']}"
    )


__all__ = ["PROBE_SCHEMA", "format_probe", "probe_scene_tokens", "spatial_anchors"]


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix-cache",
        default="data_gemma4/scene_tokens/gemma_waypoint_policy_v15_rooms",
    )
    parser.add_argument("--holdout-rooms", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = probe_scene_tokens(
        args.prefix_cache, holdout_room_count=args.holdout_rooms
    )
    if args.output is not None:
        destination = _rooted(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(format_probe(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
