"""Bounded CPU fit of the V82 reader on the sealed historical train cache."""

from __future__ import annotations

import argparse
import json
import math
import random
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
from semantic_3d_chat.language.v82_dense_learned_reader import (
    DenseLearnedSceneReaderV82,
    wrong_scene_contrast_loss_v82,
)
from semantic_3d_chat.training.v82_reader_artifacts import (
    CACHE_METADATA_FILENAME,
    CACHE_TENSOR_FILENAME,
    load_v82_cache,
    save_v82_candidate,
    sha256_file_v82,
)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _load_config(path: str | Path) -> dict[str, Any]:
    source = _resolve(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"v82"}:
        raise ValueError("V82 fit config must contain exactly v82")
    config = payload["v82"]
    if not isinstance(config, Mapping) or config.get("schema_version") != 82:
        raise ValueError("V82 fit config changed")
    fit = config.get("fit")
    required = {
        "device": "cpu",
        "training_fold_only": True,
        "intermediate_selection": False,
        "runtime_promotion_authorized": False,
    }
    if not isinstance(fit, Mapping) or any(
        fit.get(field) != value for field, value in required.items()
    ):
        raise ValueError("V82 fit scope changed")
    return dict(config)


def _zero_payload_memory(memory: torch.Tensor) -> torch.Tensor:
    banks = split_v75_v2_prefix_v81(memory)
    atlas = torch.cat(
        (banks.probe_keys.unsqueeze(2), torch.zeros_like(banks.atlas_values)),
        dim=2,
    ).reshape(memory.shape[0], ATLAS_MEMORY_TOKENS, HIDDEN_SIZE)
    return torch.cat(
        (banks.boi, atlas, torch.zeros_like(banks.base_latents), banks.eoi), dim=1
    )


def _loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    paired_target: torch.Tensor,
    changed: torch.Tensor,
    *,
    model: DenseLearnedSceneReaderV82,
    source: Mapping[str, torch.Tensor],
    fit: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    flat_prediction = prediction.float().flatten(1)
    flat_target = target.detach().float().flatten(1)
    cosine = 1.0 - F.cosine_similarity(
        flat_prediction, flat_target, dim=-1, eps=1e-8
    ).mean()
    normalized_mse = (prediction.float() - target.detach().float()).square().mean()
    normalized_mse = normalized_mse / target.detach().float().square().mean().clamp_min(1e-8)
    if bool(changed.any()):
        contrast, preference = wrong_scene_contrast_loss_v82(
            prediction[changed],
            target[changed],
            paired_target[changed],
            margin=float(fit["wrong_scene_margin"]),
        )
    else:
        contrast = prediction.sum() * 0.0
        preference = torch.empty(0, device=prediction.device)
    anchor = torch.stack(
        [
            (parameter.float() - source[name].to(parameter)).square().mean()
            for name, parameter in model.named_parameters()
        ]
    ).mean()
    total = (
        float(fit["cosine_weight"]) * cosine
        + float(fit["normalized_mse_weight"]) * normalized_mse
        + float(fit["wrong_scene_contrast_weight"]) * contrast
        + float(fit["source_anchor_weight"]) * anchor
    )
    if total.ndim != 0 or not bool(torch.isfinite(total)):
        raise RuntimeError("V82 fit loss became nonfinite")
    return total, {
        "cosine_loss": cosine,
        "normalized_mse": normalized_mse,
        "wrong_scene_contrast": contrast,
        "wrong_scene_preference": preference.mean()
        if preference.numel()
        else torch.tensor(float("nan"), device=prediction.device),
        "source_anchor": anchor,
    }


def train(
    config_path: str | Path,
    *,
    cache_root: str | Path | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    config = _load_config(config_path)
    fit = config["fit"]
    cache_path = _resolve(cache_root or config["cache"]["training_output"])
    cache = load_v82_cache(cache_path)
    if cache.metadata.get("split_role") != "historical_optimization_fold":
        raise ValueError("V82 fit refuses any non-training cache")
    if (
        cache.metadata.get("row_count") != config["split"]["train_row_count"]
        or cache.metadata.get("scene_count") != config["split"]["train_scene_count"]
        or cache.metadata.get("source_qa_sha256")
        != config["sources"]["historical_qa_sha256"]
        or cache.metadata.get("source_v73_config_sha256")
        != config["sources"]["v73_config_sha256"]
        or cache.metadata.get("source_prefix_manifest_sha256")
        != config["sources"]["prefix_manifest_sha256"]
        or cache.metadata.get("source_controller_sha256")
        != config["sources"]["v75_controller_weights_sha256"]
    ):
        raise ValueError("V82 training cache provenance changed")
    if torch.backends.mps.is_available() and str(fit["device"]) != "cpu":
        raise ValueError("V82 fit is intentionally CPU-only")

    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    model = DenseLearnedSceneReaderV82(initialization_seed=seed).cpu().train()
    source = {
        name: parameter.detach().cpu().float().clone()
        for name, parameter in model.named_parameters()
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(fit["learning_rate"]),
        weight_decay=float(fit["weight_decay"]),
    )
    tensors = cache.tensors
    row_count = int(tensors["row_scene_indices"].numel())
    batch_size = int(fit["batch_size"])
    epochs = int(fit["epochs"])
    maximum_updates = int(fit["optimizer_updates_maximum"])
    if (
        min(batch_size, epochs, maximum_updates) < 1
        or batch_size > row_count
        or math.ceil(row_count / batch_size) * epochs > maximum_updates
    ):
        raise ValueError("V82 bounded fit schedule is invalid")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    update = 0
    for epoch in range(epochs):
        order = torch.randperm(row_count, generator=generator)
        for start in range(0, row_count, batch_size):
            indices = order[start : start + batch_size]
            scene_indices = tensors["row_scene_indices"][indices]
            memory = tensors["scene_memories"][scene_indices].float()
            query = tensors["question_queries"][tensors["row_query_indices"][indices]]
            target = tensors["target_controls"][indices].float()
            paired_target = tensors["paired_target_controls"][indices].float()
            changed = tensors["row_expected_change"][indices]
            binding = bind_fixed_prefix_before_question_v81(memory)
            optimizer.zero_grad(set_to_none=True)
            output_value = model(memory, query, binding=binding)
            total, parts = _loss(
                output_value.controls,
                target,
                paired_target,
                changed,
                model=model,
                source=source,
                fit=fit,
            )
            total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(fit["gradient_clip_norm"])
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("V82 fit gradient became nonfinite")
            optimizer.step()
            model.assert_parameter_contract()
            update += 1
            record: dict[str, float | int] = {
                "update": update,
                "epoch": epoch + 1,
                "loss": float(total.detach()),
                "cosine_loss": float(parts["cosine_loss"].detach()),
                "normalized_mse": float(parts["normalized_mse"].detach()),
                "wrong_scene_contrast": float(parts["wrong_scene_contrast"].detach()),
                "gradient_norm": float(gradient_norm.detach()),
            }
            history.append(record)
            if update == 1 or update % 8 == 0 or update == maximum_updates:
                print(json.dumps({"event": "v82_fit_update", **record}, sort_keys=True))
    if update != math.ceil(row_count / batch_size) * epochs:
        raise RuntimeError("V82 fit did not execute its complete bounded schedule")

    model.eval()
    with torch.inference_mode():
        sample_memory = tensors["scene_memories"][:1].float()
        zero_memory = _zero_payload_memory(sample_memory)
        zero_binding = bind_fixed_prefix_before_question_v81(zero_memory)
        zero_output = model(
            zero_memory,
            tensors["question_queries"][:1],
            binding=zero_binding,
        )
    zero_max = float(zero_output.controls.abs().max())
    if zero_max != 0.0:
        raise RuntimeError("V82 fitted reader violated exact-zero environment")
    final = history[-1]
    fit_summary: dict[str, int | float | bool] = {
        "seed": seed,
        "epochs": epochs,
        "optimizer_updates": update,
        "batch_size": batch_size,
        "initial_loss": history[0]["loss"],
        "final_loss": final["loss"],
        "final_cosine_loss": final["cosine_loss"],
        "final_normalized_mse": final["normalized_mse"],
        "final_wrong_scene_contrast": final["wrong_scene_contrast"],
        "zero_environment_maximum_absolute_control": zero_max,
        "elapsed_seconds": time.perf_counter() - started,
        "training_fold_only": True,
    }
    destination = _resolve(output or fit["candidate_output"])
    metadata = save_v82_candidate(
        destination,
        model,
        training_cache_sha256=sha256_file_v82(cache_path / CACHE_TENSOR_FILENAME),
        training_cache_metadata_sha256=sha256_file_v82(
            cache_path / CACHE_METADATA_FILENAME
        ),
        fit_summary=fit_summary,
    )
    return {
        "phase": "v82_bounded_cpu_fit_complete",
        "output": str(destination),
        "weights_sha256": metadata["weights_sha256"],
        "trainable_parameter_count": metadata["trainable_parameter_count"],
        "fit_summary": fit_summary,
        "runtime_promotion_authorized": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/gemma4_v82_strict_dense_learned_reader.yaml",
    )
    parser.add_argument("--cache")
    parser.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = train(args.config, cache_root=args.cache, output=args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "train"]
