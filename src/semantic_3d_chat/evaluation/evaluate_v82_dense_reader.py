"""Model-free numeric evaluation of a sealed V82 candidate and cache."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.language.v81_structured_dense_atlas_sidecar import (
    ATLAS_MEMORY_TOKENS,
    HIDDEN_SIZE,
    bind_fixed_prefix_before_question_v81,
    split_v75_v2_prefix_v81,
)
from semantic_3d_chat.training.v82_reader_artifacts import (
    CACHE_METADATA_FILENAME,
    CACHE_TENSOR_FILENAME,
    CANDIDATE_METADATA_FILENAME,
    canonical_sha256_v82,
    load_v82_cache,
    load_v82_candidate,
    sha256_file_v82,
)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    rooted = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(rooted))


def _load_config(path: str | Path) -> dict[str, Any]:
    source = _resolve(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(source)
    if not isinstance(payload, Mapping) or set(payload) != {"v82"}:
        raise ValueError("V82 evaluation config must contain exactly v82")
    config = payload["v82"]
    if not isinstance(config, Mapping) or config.get("schema_version") != 82:
        raise ValueError("V82 evaluation config changed")
    return dict(config)


def _memory_with_values(
    memory: torch.Tensor,
    *,
    atlas_values: torch.Tensor,
    base_values: torch.Tensor,
) -> torch.Tensor:
    banks = split_v75_v2_prefix_v81(memory)
    atlas = torch.cat((banks.probe_keys.unsqueeze(2), atlas_values), dim=2).reshape(
        memory.shape[0], ATLAS_MEMORY_TOKENS, HIDDEN_SIZE
    )
    return torch.cat((banks.boi, atlas, base_values, banks.eoi), dim=1)


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def evaluate(
    config_path: str | Path,
    *,
    cache_root: str | Path | None = None,
    candidate_root: str | Path | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    config = _load_config(config_path)
    cache_path = _resolve(cache_root or config["cache"]["development_output"])
    candidate_path = _resolve(candidate_root or config["fit"]["candidate_output"])
    output_path = _resolve(output or config["evaluation"]["output"])
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    cache = load_v82_cache(cache_path)
    if cache.metadata.get("split_role") != (
        "historical_pair_scene_disjoint_development_fold"
    ):
        raise ValueError("V82 evaluation requires the historical development cache")
    candidate = load_v82_candidate(candidate_path, device="cpu")
    model = candidate.model
    tensors = cache.tensors
    row_count = int(tensors["row_scene_indices"].numel())
    batch_size = 12
    cosine_values: list[torch.Tensor] = []
    normalized_mse_values: list[torch.Tensor] = []
    changed_preferences: list[torch.Tensor] = []
    wrong_scene_follow_preferences: list[torch.Tensor] = []
    shuffled_deltas: list[torch.Tensor] = []
    minimum_atlas = float("inf")
    minimum_base = float("inf")
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, row_count, batch_size):
            indices = torch.arange(start, min(start + batch_size, row_count))
            scenes = tensors["row_scene_indices"][indices]
            paired_scenes = tensors["row_paired_scene_indices"][indices]
            query = tensors["question_queries"][tensors["row_query_indices"][indices]]
            target = tensors["target_controls"][indices].float()
            paired_target = tensors["paired_target_controls"][indices].float()
            changed = tensors["row_expected_change"][indices]
            memory = tensors["scene_memories"][scenes].float()
            paired_memory = tensors["scene_memories"][paired_scenes].float()
            binding = bind_fixed_prefix_before_question_v81(memory)
            primary = model(memory, query, binding=binding)
            paired_binding = bind_fixed_prefix_before_question_v81(paired_memory)
            wrong = model(paired_memory, query, binding=paired_binding)
            banks = split_v75_v2_prefix_v81(memory)
            shuffled_memory = _memory_with_values(
                memory,
                atlas_values=banks.atlas_values.roll(shifts=1, dims=1),
                base_values=banks.base_latents,
            )
            shuffled = model(
                shuffled_memory,
                query,
                binding=bind_fixed_prefix_before_question_v81(shuffled_memory),
            )
            prediction_flat = primary.controls.flatten(1)
            target_flat = target.flatten(1)
            paired_flat = paired_target.flatten(1)
            wrong_flat = wrong.controls.flatten(1)
            cosine_values.append(
                F.cosine_similarity(prediction_flat, target_flat, dim=-1, eps=1e-8)
            )
            mse = (primary.controls - target).square().mean(dim=(1, 2))
            normalized_mse_values.append(
                mse / target.square().mean(dim=(1, 2)).clamp_min(1e-8)
            )
            if bool(changed.any()):
                own_similarity = F.cosine_similarity(
                    prediction_flat[changed], target_flat[changed], dim=-1, eps=1e-8
                )
                opposite_similarity = F.cosine_similarity(
                    prediction_flat[changed], paired_flat[changed], dim=-1, eps=1e-8
                )
                changed_preferences.append(own_similarity - opposite_similarity)
                follows_pair = F.cosine_similarity(
                    wrong_flat[changed], paired_flat[changed], dim=-1, eps=1e-8
                ) - F.cosine_similarity(
                    wrong_flat[changed], target_flat[changed], dim=-1, eps=1e-8
                )
                wrong_scene_follow_preferences.append(follows_pair)
            shuffled_deltas.append(
                (primary.controls - shuffled.controls).square().mean(dim=(1, 2)).sqrt()
            )
            minimum_atlas = min(minimum_atlas, float(primary.atlas_weights.min()))
            minimum_base = min(minimum_base, float(primary.base_weights.min()))

        zero_source = tensors["scene_memories"][:1].float()
        zero_banks = split_v75_v2_prefix_v81(zero_source)
        zero_memory = _memory_with_values(
            zero_source,
            atlas_values=torch.zeros_like(zero_banks.atlas_values),
            base_values=torch.zeros_like(zero_banks.base_latents),
        )
        zero = model(
            zero_memory,
            tensors["question_queries"][:1],
            binding=bind_fixed_prefix_before_question_v81(zero_memory),
        )
    cosine = torch.cat(cosine_values)
    normalized_mse = torch.cat(normalized_mse_values)
    changed_preference = torch.cat(changed_preferences)
    wrong_follow = torch.cat(wrong_scene_follow_preferences)
    shuffle_delta = torch.cat(shuffled_deltas)
    zero_max = float(zero.controls.abs().max())
    if zero_max != 0.0:
        raise RuntimeError("V82 evaluation zero-environment control is not exact zero")
    result: dict[str, Any] = {
        "schema_version": 82,
        "artifact": "v82_strict_dense_reader_historical_development_metrics_v1",
        "status": "historical_development_diagnostic_not_promoted",
        "cache_sha256": sha256_file_v82(cache_path / CACHE_TENSOR_FILENAME),
        "cache_metadata_sha256": sha256_file_v82(cache_path / CACHE_METADATA_FILENAME),
        "candidate_weights_sha256": candidate.metadata["weights_sha256"],
        "candidate_metadata_sha256": sha256_file_v82(
            candidate_path / CANDIDATE_METADATA_FILENAME
        ),
        "row_count": row_count,
        "scene_count": cache.metadata["scene_count"],
        "changed_row_count": int(changed_preference.numel()),
        "mean_control_cosine": float(cosine.mean()),
        "normalized_mse": float(normalized_mse.mean()),
        "changed_wrong_scene_preference_mean": float(changed_preference.mean()),
        "changed_wrong_scene_positive_sides": int((changed_preference > 0).sum()),
        "wrong_scene_follows_paired_scene_preference_mean": float(wrong_follow.mean()),
        "wrong_scene_follows_paired_scene_positive_sides": int((wrong_follow > 0).sum()),
        "shuffled_atlas_control_delta_rms_mean": float(shuffle_delta.mean()),
        "zero_environment_maximum_absolute_control": zero_max,
        "minimum_atlas_attention_weight": minimum_atlas,
        "minimum_base_attention_weight": minimum_base,
        "elapsed_seconds": time.perf_counter() - started,
        "questions_or_answers_serialized": False,
        "oracle_loaded": False,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "candidate_selection_after_scoring": False,
        "runtime_promotion_authorized": False,
    }
    result["metrics_sha256"] = canonical_sha256_v82(result)
    _atomic_create_json(output_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/gemma4_v82_strict_dense_learned_reader.yaml",
    )
    parser.add_argument("--cache")
    parser.add_argument("--candidate")
    parser.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate(
        args.config,
        cache_root=args.cache,
        candidate_root=args.candidate,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["evaluate", "main"]
