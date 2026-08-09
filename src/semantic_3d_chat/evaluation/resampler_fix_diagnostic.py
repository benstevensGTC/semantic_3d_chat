"""Measure v2 scene-signal retention using the trained v1 block encoder.

This diagnostic deliberately does not train.  It transfers all shape-compatible
weights from the measured collapsed checkpoint, including point/block encoders,
Perceiver layers, and LM projection.  Only the new identity/coverage/bypass paths
are freshly introduced, making the before/after comparison attributable to the
architecture change rather than another optimization run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from semantic_3d_chat.chat.runtime import construct_scene_tokenizer
from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.evaluation.scene_signal_audit import (
    PAIR_SPECS,
    _latent_diversity,
    _tensor_metrics,
)
from semantic_3d_chat.scene_encoder.map_io import load_map_tensors


def _transfer_v1_weights(model: torch.nn.Module, checkpoint: Path) -> dict[str, Any]:
    source = load_file(checkpoint / "adapter.safetensors", device="cpu")
    target_state = model.state_dict()
    transferred: dict[str, torch.Tensor] = {}
    for target_name, target in target_state.items():
        source_name = f"scene_model.{target_name}"
        if target_name == "resampler.learned_queries":
            source_name = "scene_model.resampler.latents"
        elif target_name.startswith("language_projection.trainable."):
            legacy_name = target_name.replace("language_projection.trainable.", "")
            source_name = f"scene_model.language_projection.{legacy_name}"
        if source_name in source and source[source_name].shape == target.shape:
            transferred[target_name] = source[source_name]
    missing, unexpected = model.load_state_dict(transferred, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected transferred keys: {unexpected}")
    # v2-only identity/coverage buffers are nonpersistent and have no state keys.
    allowed_missing: set[str] = set()
    unrecognized = sorted(set(missing) - allowed_missing)
    if unrecognized:
        raise RuntimeError(f"Unrecognized missing v2 parameters: {unrecognized}")
    return {
        "transferred_tensor_count": len(transferred),
        "source_tensor_count": sum(key.startswith("scene_model.") for key in source),
    }


def _encode(
    model: torch.nn.Module, config: dict[str, Any], scene_id: str
) -> dict[str, torch.Tensor]:
    map_path = PROJECT_ROOT / config["paths"]["data_root"] / "maps" / scene_id / "voxel_map.npz"
    data = load_map_tensors(
        map_path,
        config["scene"]["room_size_m"],
        device="cpu",
        input_voxel_size_m=float(config["scene_encoder"]["input_voxel_size_m"]),
    )
    with torch.inference_mode():
        points, _ = model.point_projection(
            data.semantic,
            data.xyz,
            data.rgb,
            data.normal,
            data.confidence,
            data.observation_count,
            data.room_min,
            data.room_max,
        )
        blocks, audit = model.block_encoder(points, data.xyz, data.room_min, data.room_max)
        native = model.resampler(blocks, audit["block_token_positions_normalized"])
        projected = model.language_projection(native)
    return {
        "native_latents": native.cpu(),
        "projected_scene_tokens_float32": projected.cpu(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/multiscene.yaml")
    parser.add_argument("--checkpoint", default="data/checkpoints/multiscene/best")
    parser.add_argument("--before", default="reports/metrics/scene_signal_audit.json")
    parser.add_argument("--output", default="reports/metrics/resampler_fix_diagnostic.json")
    args = parser.parse_args()

    torch.manual_seed(20250308)
    config = load_config(args.config)
    checkpoint = PROJECT_ROOT / args.checkpoint
    before_path = PROJECT_ROOT / args.before
    before = json.loads(before_path.read_text(encoding="utf-8"))
    with (checkpoint / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    model = construct_scene_tokenizer(
        config,
        semantic_dim=int(metadata["semantic_dim"]),
        language_hidden_dim=int(metadata["language_hidden_dim"]),
    ).cpu()
    transfer = _transfer_v1_weights(model, checkpoint)
    model.eval()

    relevant_specs = [
        spec
        for spec in PAIR_SPECS
        if any(item["pair_id"] == spec["pair_id"] for item in before["pairs"])
    ]
    scene_ids = sorted(
        {scene for spec in relevant_specs for scene in (spec["scene_a"], spec["scene_b"])}
    )
    representations = {scene_id: _encode(model, config, scene_id) for scene_id in scene_ids}
    before_by_pair = {item["pair_id"]: item for item in before["pairs"]}
    pairs: list[dict[str, Any]] = []
    for spec in relevant_specs:
        earlier = before_by_pair[spec["pair_id"]]
        first = representations[spec["scene_a"]]
        second = representations[spec["scene_b"]]
        native = _tensor_metrics(first["native_latents"], second["native_latents"])
        projected = _tensor_metrics(
            first["projected_scene_tokens_float32"],
            second["projected_scene_tokens_float32"],
        )
        before_native = float(earlier["native_latents"]["relative_l2"])
        before_projected = float(earlier["projected_scene_tokens_float32"]["relative_l2"])
        pairs.append(
            {
                **spec,
                "before": {
                    "block_relative_l2": earlier["block_tokens"]["common_block_tokens"][
                        "relative_l2"
                    ],
                    "native_relative_l2": before_native,
                    "projected_relative_l2": before_projected,
                    "native_mean_off_diagonal_cosine": earlier["latent_diversity"][
                        "scene_a_native"
                    ]["mean_off_diagonal_cosine"],
                    "projected_mean_off_diagonal_cosine": earlier["latent_diversity"][
                        "scene_a_projected"
                    ]["mean_off_diagonal_cosine"],
                },
                "after": {
                    "native_relative_l2": native["relative_l2"],
                    "projected_relative_l2": projected["relative_l2"],
                    "native_mean_off_diagonal_cosine": _latent_diversity(first["native_latents"])[
                        "mean_off_diagonal_cosine"
                    ],
                    "projected_mean_off_diagonal_cosine": _latent_diversity(
                        first["projected_scene_tokens_float32"]
                    )["mean_off_diagonal_cosine"],
                },
                "improvement_factor": {
                    "native_scene_change": native["relative_l2"] / before_native,
                    "projected_scene_change": projected["relative_l2"] / before_projected,
                },
            }
        )

    payload = {
        "schema_version": 1,
        "purpose": "CPU-only no-training before/after collapse-fix diagnostic",
        "architecture_version": config["scene_encoder"]["architecture_version"],
        "config": args.config,
        "legacy_checkpoint": args.checkpoint,
        "legacy_audit": args.before,
        "weight_transfer": transfer,
        "pairs": pairs,
    }
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
