"""Offline, no-step audit for the V16 zero-output scene residual.

This command intentionally reads supervised counterfactual QA metadata.  It is
an evaluation/training diagnostic and is not imported by the chat runtime.  The
environment still enters Gemma only through the fixed continuous scene prefix;
the QA records provide questions and losses for measuring gradients.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def gradient_comparison(first: torch.Tensor, second: torch.Tensor) -> dict[str, float | int]:
    """Return stable norm/cosine diagnostics for two same-shaped gradients."""

    if first.shape != second.shape or first.numel() == 0:
        raise ValueError("Gradient tensors must be nonempty and have the same shape")
    a = first.detach().float().reshape(-1)
    b = second.detach().float().reshape(-1)
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Gradient tensors must be finite")
    norm_a = a.norm()
    norm_b = b.norm()
    dot = torch.dot(a, b)
    cosine = dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else torch.tensor(float("nan"))
    return {
        "parameter_count": int(a.numel()),
        "first_l2_norm": float(norm_a),
        "second_l2_norm": float(norm_b),
        "dot_product": float(dot),
        "cosine_similarity": float(cosine),
    }


def simulate_first_adamw_update(
    gradient: torch.Tensor,
    *,
    learning_rate: float,
    gradient_clip_norm: float,
    epsilon: float = 1.0e-8,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Simulate AdamW's first update from an exact-zero, no-decay tensor.

    Bias correction cancels beta1 and beta2 on the first step, leaving
    ``-lr * g / (abs(g) + eps)`` after global-norm clipping.  This function does
    not construct an optimizer and does not mutate a parameter.
    """

    if learning_rate <= 0 or gradient_clip_norm <= 0 or epsilon <= 0:
        raise ValueError("Learning rate, clip norm, and epsilon must be positive")
    grad = gradient.detach().float()
    if grad.numel() == 0 or not torch.isfinite(grad).all():
        raise ValueError("Gradient must be nonempty and finite")
    pre_clip_norm = grad.norm()
    clip_scale = min(1.0, gradient_clip_norm / (float(pre_clip_norm) + 1.0e-6))
    clipped = grad * clip_scale
    update = -float(learning_rate) * clipped / (clipped.abs() + float(epsilon))
    if not torch.isfinite(update).all():
        raise RuntimeError("Simulated AdamW update contains NaN or infinity")
    return update, {
        "parameter_count": int(update.numel()),
        "pre_clip_gradient_l2_norm": float(pre_clip_norm),
        "clip_scale": float(clip_scale),
        "post_clip_gradient_l2_norm": float(clipped.norm()),
        "update_l2_norm": float(update.norm()),
        "update_rms": float(update.square().mean().sqrt()),
        "update_absolute_maximum": float(update.abs().max()),
        "nonzero_update_count": int(torch.count_nonzero(update)),
    }


def scene_delta_metrics(core: torch.Tensor, delta: torch.Tensor) -> dict[str, float]:
    """Decompose a residual into shared-across-slot and slot-varying energy."""

    if core.shape != delta.shape or core.ndim != 3 or core.shape[1] < 2:
        raise ValueError("Core and delta must have matching [B,L,H] shapes with L > 1")
    core32 = core.detach().float()
    delta32 = delta.detach().float()
    if not torch.isfinite(core32).all() or not torch.isfinite(delta32).all():
        raise ValueError("Core and delta tensors must be finite")
    core_rms = core32.square().mean().sqrt()
    delta_rms = delta32.square().mean().sqrt()
    slot_mean = delta32.mean(dim=1, keepdim=True)
    slot_varying = delta32 - slot_mean
    mean_rms = slot_mean.square().mean().sqrt()
    varying_rms = slot_varying.square().mean().sqrt()
    total_energy = delta32.square().mean()
    mean_fraction = (
        slot_mean.square().mean() / total_energy if total_energy > 0 else torch.tensor(0.0)
    )
    varying_fraction = (
        slot_varying.square().mean() / total_energy if total_energy > 0 else torch.tensor(0.0)
    )
    return {
        "core_rms": float(core_rms),
        "delta_rms": float(delta_rms),
        "delta_to_core_rms_ratio": float(delta_rms / core_rms),
        "across_slot_mean_rms": float(mean_rms),
        "slot_varying_rms": float(varying_rms),
        "across_slot_mean_energy_fraction": float(mean_fraction),
        "slot_varying_energy_fraction": float(varying_fraction),
        "delta_absolute_maximum": float(delta32.abs().max()),
    }


def pair_delta_metrics(
    first_core: torch.Tensor,
    second_core: torch.Tensor,
    first_delta: torch.Tensor,
    second_delta: torch.Tensor,
) -> dict[str, float]:
    """Measure whether a residual changes a scene pair rather than both alike."""

    shapes = {tuple(value.shape) for value in (first_core, second_core, first_delta, second_delta)}
    if len(shapes) != 1:
        raise ValueError("Pair tensors must all have the same shape")
    core_difference = first_core.detach().float() - second_core.detach().float()
    delta_difference = first_delta.detach().float() - second_delta.detach().float()
    core_rms = core_difference.square().mean().sqrt()
    delta_rms = delta_difference.square().mean().sqrt()
    if core_rms <= 0:
        raise ValueError("Pair core difference must be nonzero")
    cosine = F.cosine_similarity(core_difference.reshape(-1), delta_difference.reshape(-1), dim=0)
    return {
        "core_pair_difference_rms": float(core_rms),
        "residual_pair_difference_rms": float(delta_rms),
        "residual_to_core_pair_difference_ratio": float(delta_rms / core_rms),
        "residual_core_difference_cosine": float(cosine),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parameter_gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().square().sum())
    return math.sqrt(squared)


def _residual_hidden(module: torch.nn.Module, scene_tokens: torch.Tensor) -> torch.Tensor:
    """Reproduce V16's hidden state without applying its output projection."""

    normalized = module.scene_norm(scene_tokens)
    local_content = module.scene_projection(normalized)
    global_content = local_content.float().mean(dim=1, keepdim=True).to(local_content.dtype)
    positions = module.position_projection(
        module.position_features.to(device=scene_tokens.device, dtype=scene_tokens.dtype)
    ).unsqueeze(0)
    return torch.tanh(local_content + global_content + positions)


def _functional_residual_delta(
    module: torch.nn.Module, scene_tokens: torch.Tensor, output_weight: torch.Tensor
) -> torch.Tensor:
    hidden = _residual_hidden(module, scene_tokens)
    return F.linear(hidden, output_weight.to(device=hidden.device, dtype=hidden.dtype))


def run_audit(
    config_path: str | Path,
    report_path: str | Path,
    *,
    candidate_learning_rates: Sequence[float] = (1.0e-4, 3.0e-4),
) -> dict[str, Any]:
    """Run the pinned V16 gradient audit without executing an optimizer step."""

    # Heavy training/model imports stay inside the offline command so importing
    # metric helpers cannot pull supervised QA machinery into chat inference.
    from semantic_3d_chat.config import PROJECT_ROOT, artifact_root, load_config, project_path
    from semantic_3d_chat.data.dataset import SceneQADataset
    from semantic_3d_chat.language.local_lm import load_local_language_model
    from semantic_3d_chat.language.lora import install_lora_banks, lora_banks_settings
    from semantic_3d_chat.language.prefix_injection import (
        ContinuousPrefixComposer,
        scene_boundary_mode_setting,
        scene_prefix_after_bos_setting,
    )
    from semantic_3d_chat.scene_encoder.global_residual import (
        apply_global_scene_residual,
        construct_global_scene_residual,
        global_scene_residual_settings,
    )
    from semantic_3d_chat.scene_encoder.map_io import load_map_tensors
    from semantic_3d_chat.training.checkpointing import (
        load_adapter_checkpoint,
        module_collection_state_sha256,
    )
    from semantic_3d_chat.training.losses import QuestionGroundingHead
    from semantic_3d_chat.training.pair_curriculum import (
        build_exact_question_pair_units,
        cap_pair_units_per_pair,
        pair_curriculum_settings,
        select_pair_only_records,
    )
    from semantic_3d_chat.training.train_adapter import (
        combine_pair_training_losses,
        construct_scene_tokenizer,
        pair_batch_objective,
        set_seed,
    )

    config = load_config(config_path)
    set_seed(int(config["seed"]))
    training = config["training"]
    if training.get("train_global_scene_residual_only") is not True:
        raise ValueError("Audit requires the residual-only V16 training contract")
    residual_settings = global_scene_residual_settings(config)
    if not residual_settings.enabled:
        raise ValueError("Audit requires an enabled global scene residual")
    pair_settings = pair_curriculum_settings(config)
    if not pair_settings.pair_only or pair_settings.max_units_per_pair is None:
        raise ValueError("Audit requires the capped pair-only V16 curriculum")

    qa_root = artifact_root(config, "qa")
    records = SceneQADataset(qa_root / "train.jsonl").records
    selected = select_pair_only_records(records, pair_settings.pair_only_scene_ids)
    selected = cap_pair_units_per_pair(
        selected, pair_settings.max_units_per_pair, seed=int(config["seed"])
    )
    units = build_exact_question_pair_units(selected)
    by_change: dict[str, list[Any]] = defaultdict(list)
    for unit in units:
        change = unit.reference.counterfactual_change_type
        if not change or change != unit.counterfactual.counterfactual_change_type:
            raise ValueError(f"Pair {unit.pair_id} has inconsistent change metadata")
        by_change[change].append(unit)
    if set(by_change) != {"color_swap", "mirror_lr"}:
        raise ValueError(f"Expected color_swap and mirror_lr units, observed {sorted(by_change)}")

    language = load_local_language_model(
        config["language"]["model_id"],
        config["language"]["revision"],
        config["language"]["dtype"],
        freeze=True,
        local_files_only=True,
        backend=str(config["language"].get("backend", "auto")),
        decoder_gradient_checkpointing=bool(
            training.get("language_decoder_gradient_checkpointing", False)
        ),
    )
    language.model.config.use_cache = False
    lora = install_lora_banks(language.model, lora_banks_settings(config))
    if lora is None or lora.trainable_parameter_count != 0:
        raise ValueError("V16 audit requires installed, entirely frozen named LoRA banks")
    lora.eval()

    scene_ids = sorted({record.scene_id for record in selected})
    maps = {
        scene_id: load_map_tensors(
            project_path(config, "maps", scene_id, "voxel_map.npz"),
            config["scene"]["room_size_m"],
            language.device,
            input_voxel_size_m=config["scene_encoder"].get("input_voxel_size_m"),
        )
        for scene_id in scene_ids
    }
    feature_dims = {data.feature_dim for data in maps.values()}
    if len(feature_dims) != 1:
        raise ValueError(f"Inconsistent semantic dimensions: {feature_dims}")
    scene_model = construct_scene_tokenizer(config, feature_dims.pop(), language.hidden_size).to(
        language.device
    )
    residual = construct_global_scene_residual(
        config,
        scene_dim=language.hidden_size,
        latent_count=int(config["scene_encoder"]["global_latents"]),
    )
    if residual is None:
        raise RuntimeError("Residual construction unexpectedly returned None")
    residual = residual.to(language.device)
    composer = ContinuousPrefixComposer(
        language.hidden_size,
        scene_prefix_after_bos=scene_prefix_after_bos_setting(config),
        bos_token_id=language.bos_token_id,
        scene_boundary_mode=scene_boundary_mode_setting(config),
        native_boundary_embeddings=language.scene_boundary_embeddings(
            scene_boundary_mode_setting(config)
        ),
    ).to(language.device)
    grounding = QuestionGroundingHead(
        int(config["scene_encoder"]["model_dim"]),
        language.hidden_size,
        int(config["scene_encoder"]["global_latents"]),
        int(config["scene_encoder"]["model_dim"]),
    ).to(language.device)

    source_value = training.get("initialize_from")
    if not isinstance(source_value, str) or not source_value:
        raise ValueError("V16 config must pin training.initialize_from")
    source = Path(source_value)
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    source = source.resolve()
    scene_state_modules = {
        "scene_model": scene_model,
        "composer": composer,
        "grounding": grounding,
    }
    source_modules = dict(scene_state_modules)
    source_modules.update(lora.state_modules())
    source_metadata = load_adapter_checkpoint(source, source_modules, device=str(language.device))
    expected_adapter = training.get("initialize_expected_adapter_sha256")
    expected_metadata = training.get("initialize_expected_metadata_sha256")
    observed_adapter = _file_sha256(source / "adapter.safetensors")
    observed_metadata = _file_sha256(source / "metadata.json")
    if observed_adapter != expected_adapter or observed_metadata != expected_metadata:
        raise ValueError("Pinned V14 source artifact hash mismatch")
    source_scene_hash = module_collection_state_sha256(scene_state_modules)
    expected_scene_hash = config.get("experiment", {}).get("source_scene_state_sha256")
    if source_scene_hash != expected_scene_hash:
        raise ValueError(
            "Pinned V14 frozen scene-state hash mismatch: "
            f"expected={expected_scene_hash} observed={source_scene_hash}"
        )
    source_lora_hashes = lora.state_sha256()
    expected_lora_hashes = {
        "inherited_v12": config.get("experiment", {}).get("source_inherited_bank_sha256"),
        "extension_v13": config.get("experiment", {}).get("source_extension_bank_sha256"),
    }
    if source_lora_hashes != expected_lora_hashes:
        raise ValueError(
            "Pinned V14 frozen LoRA-bank hash mismatch: "
            f"expected={expected_lora_hashes} observed={source_lora_hashes}"
        )

    scene_model.requires_grad_(False).eval()
    composer.requires_grad_(False).eval()
    grounding.requires_grad_(False).eval()
    residual.requires_grad_(True).train()
    initial_residual_hash = module_collection_state_sha256({"global_scene_residual": residual})
    if initial_residual_hash != residual_settings.expected_initial_state_sha256:
        raise ValueError("Deterministic V16 residual hash mismatch")
    if torch.count_nonzero(residual.output_projection.weight).item() != 0:
        raise ValueError("V16 residual output projection is not exact zero")

    with torch.no_grad():
        core_outputs = {
            scene_id: scene_model(
                data.semantic,
                data.xyz,
                data.rgb,
                data.normal,
                data.confidence,
                data.observation_count,
                data.room_min,
                data.room_max,
            )
            for scene_id, data in maps.items()
        }

    zero = torch.zeros((), device=language.device)
    gradients: dict[str, torch.Tensor] = {}
    objective_reports: dict[str, dict[str, float | int | str]] = {}
    output_parameter = residual.output_projection.weight
    non_output_parameters = [
        parameter
        for name, parameter in residual.named_parameters()
        if name != "output_projection.weight"
    ]
    for change_type in ("color_swap", "mirror_lr"):
        change_units = by_change[change_type]
        residual.zero_grad(set_to_none=True)
        component_sums = defaultdict(float)
        for unit in change_units:
            outputs = {
                scene_id: apply_global_scene_residual(core_outputs[scene_id], residual)
                for scene_id in unit.scene_ids
            }
            base, language_loss, grounding_loss, ranking_loss, diagnostics = pair_batch_objective(
                outputs,
                [unit],
                maps,
                language,
                composer,
                grounding,
                config,
                ranking_margin=pair_settings.ranking_margin,
                ranking_mode=pair_settings.ranking_mode,
                collect_full_vocab_first_answer_token=True,
                full_vocab_ranking_margin=pair_settings.full_vocab_ranking_margin,
            )
            full_vocab_loss = diagnostics["first_answer_token_full_vocab_ranking_loss"]
            assert isinstance(full_vocab_loss, torch.Tensor)
            loss = combine_pair_training_losses(
                base,
                ranking_loss,
                full_vocab_loss,
                zero,
                zero,
                pair_ranking_weight=pair_settings.ranking_weight,
                full_vocab_ranking_weight=pair_settings.full_vocab_ranking_weight,
                diversity_weight=0.0,
                scene_separation_weight=0.0,
            )
            (loss / len(change_units)).backward()
            component_sums["total_loss"] += float(loss.detach())
            component_sums["language_loss"] += float(language_loss.detach())
            component_sums["grounding_loss"] += float(grounding_loss.detach())
            component_sums["candidate_ranking_loss"] += float(ranking_loss.detach())
            component_sums["full_vocab_ranking_loss"] += float(full_vocab_loss.detach())
            del outputs, base, loss, language_loss, grounding_loss, ranking_loss, diagnostics
        if output_parameter.grad is None:
            raise RuntimeError(f"{change_type} objective produced no output-projection gradient")
        gradients[change_type] = output_parameter.grad.detach().float().cpu().clone()
        non_output_norm = _parameter_gradient_norm(non_output_parameters)
        objective_reports[change_type] = {
            "pair_id": change_units[0].pair_id,
            "unit_count": len(change_units),
            **{
                f"mean_{name}": value / len(change_units)
                for name, value in sorted(component_sums.items())
            },
            "output_projection_gradient_l2_norm": float(gradients[change_type].norm()),
            "non_output_parameter_gradient_l2_norm": non_output_norm,
        }

    comparison = gradient_comparison(gradients["color_swap"], gradients["mirror_lr"])
    aggregate_gradient = (gradients["color_swap"] + gradients["mirror_lr"]) * 0.5
    learning_rates = {float(training["learning_rate"]), *map(float, candidate_learning_rates)}
    if any(rate <= 0 or not math.isfinite(rate) for rate in learning_rates):
        raise ValueError("Candidate learning rates must be finite and positive")
    simulations: dict[str, dict[str, Any]] = {}
    for learning_rate in sorted(learning_rates):
        simulated_weight, update_metrics = simulate_first_adamw_update(
            aggregate_gradient,
            learning_rate=learning_rate,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
        )
        simulated_weight = simulated_weight.to(language.device)

        scene_metrics: dict[str, dict[str, float]] = {}
        deltas: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for scene_id, output in core_outputs.items():
                delta = _functional_residual_delta(residual, output.scene_tokens, simulated_weight)
                deltas[scene_id] = delta
                scene_metrics[scene_id] = scene_delta_metrics(output.scene_tokens, delta)

        pair_metrics: dict[str, dict[str, float | str]] = {}
        for change_type, change_units in by_change.items():
            first_id, second_id = change_units[0].scene_ids
            pair_metrics[change_type] = {
                "pair_id": change_units[0].pair_id,
                "first_scene_id": first_id,
                "second_scene_id": second_id,
                **pair_delta_metrics(
                    core_outputs[first_id].scene_tokens,
                    core_outputs[second_id].scene_tokens,
                    deltas[first_id],
                    deltas[second_id],
                ),
            }
        simulations[f"{learning_rate:.0e}"] = {
            "simulation_only": True,
            "optimizer_step_executed": False,
            "learning_rate": learning_rate,
            "weight_decay": float(training["weight_decay"]),
            "gradient_clip_norm": float(training["gradient_clip_norm"]),
            "gradient_aggregation": "mean_of_six_color_and_six_mirror_unit_objectives",
            **update_metrics,
            "scene_delta": scene_metrics,
            "pair_delta": pair_metrics,
        }

    report = {
        "schema_version": 1,
        "audit_type": "offline_no_step_v16_zero_residual_gradient_audit",
        "runtime_eligible": False,
        "uses_supervised_qa_metadata": True,
        "optimizer_step_executed": False,
        "question_dependent_scene_processing": False,
        "config_path": str(Path(config["_config_path"]).relative_to(PROJECT_ROOT)),
        "source_checkpoint": str(source.relative_to(PROJECT_ROOT)),
        "source_adapter_sha256": observed_adapter,
        "source_metadata_sha256": observed_metadata,
        "source_checkpoint_epoch": source_metadata.get("epoch"),
        "source_scene_state_sha256": source_scene_hash,
        "source_lora_bank_state_sha256": source_lora_hashes,
        "initial_residual_state_sha256": initial_residual_hash,
        "residual_parameter_count": sum(parameter.numel() for parameter in residual.parameters()),
        "residual_output_projection_parameter_count": output_parameter.numel(),
        "scene_count": len(scene_ids),
        "scene_ids": scene_ids,
        "scene_latent_count": int(config["scene_encoder"]["global_latents"]),
        "scene_hidden_dimension": language.hidden_size,
        "objective_by_change_type": objective_reports,
        "output_projection_gradient_comparison": {
            "first": "color_swap",
            "second": "mirror_lr",
            **comparison,
        },
        "configured_learning_rate_simulation_key": f"{float(training['learning_rate']):.0e}",
        "simulated_first_adamw_updates": simulations,
    }
    destination = Path(report_path)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"phase": "v16_gradient_audit_complete", "report": str(destination)}))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/gemma4_color_mirror_global_scene_residual_v16.yaml",
    )
    parser.add_argument(
        "--report",
        default="reports/gemma4/metrics/v16_zero_residual_gradient_audit.json",
    )
    parser.add_argument(
        "--candidate-learning-rate",
        action="append",
        type=float,
        dest="candidate_learning_rates",
        help="Additional no-step first-update learning rate (repeatable).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_audit(
        args.config,
        args.report,
        candidate_learning_rates=(
            (1.0e-4, 3.0e-4)
            if args.candidate_learning_rates is None
            else tuple(args.candidate_learning_rates)
        ),
    )


if __name__ == "__main__":  # pragma: no cover - exercised by the local model command
    main()
