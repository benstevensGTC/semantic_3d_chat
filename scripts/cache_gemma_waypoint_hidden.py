#!/usr/bin/env python3
"""Cache one authenticated local-Gemma final hidden state per expert row."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping

import torch

from semantic_3d_chat.config import load_config
from semantic_3d_chat.robot.gemma_runtime_binding import (
    language_gemma_runtime_binding,
)
from semantic_3d_chat.training.gemma_waypoint_hidden_reuse import (
    assemble_hidden_with_reuse,
    load_waypoint_dataset_for_hidden_reuse,
    reusable_hidden_rows,
    revalidate_cached_hidden_forward_contract,
    validate_forward_revalidation_destination,
)
from semantic_3d_chat.training.gemma_waypoint_policy import (
    ActualGemmaWaypointForward,
    gemma_hidden_input_binding,
    load_actual_waypoint_stack,
    load_gemma_hidden_cache,
    load_gemma_hidden_cache_for_forward_revalidation,
    load_waypoint_data_from_config,
    save_gemma_hidden_cache,
    select_balanced_waypoint_samples,
    validate_waypoint_settings,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/gemma_waypoint_policy_v1.yaml")
    parser.add_argument("--dataset")
    parser.add_argument("--output")
    parser.add_argument(
        "--reuse-dataset",
        help="older trace dataset whose authenticated hidden rows may be reused",
    )
    parser.add_argument(
        "--reuse-cache",
        help="hidden cache paired with --reuse-dataset",
    )
    parser.add_argument(
        "--forward-chunk-size",
        type=int,
        default=64,
        help="number of cache-miss rows forwarded between progress records",
    )
    parser.add_argument(
        "--gemma-batch-size",
        type=int,
        default=2,
        help="same-length frozen-Gemma rows per MPS/CPU forward (default: 2)",
    )
    parser.add_argument(
        "--revalidate-forward-contract",
        action="store_true",
        help=(
            "permit migration only when an old reuse cache differs solely in its "
            "forward-source hash and stratified current forwards are bit-exact"
        ),
    )
    parser.add_argument(
        "--revalidation-sample-count",
        type=int,
        default=64,
        help="stratified old rows recomputed per split during explicit migration",
    )
    args = parser.parse_args()
    if (args.reuse_dataset is None) != (args.reuse_cache is None):
        parser.error("--reuse-dataset and --reuse-cache must be supplied together")
    if args.revalidate_forward_contract:
        if args.reuse_cache is None:
            parser.error("--revalidate-forward-contract requires a reuse source")
        if args.output is None:
            parser.error(
                "--revalidate-forward-contract requires an explicit new --output"
            )
        try:
            validate_forward_revalidation_destination(args.reuse_cache, args.output)
        except ValueError as exc:
            parser.error(str(exc))
    config = load_config(args.config)
    settings = validate_waypoint_settings(config)
    dataset, cache = load_waypoint_data_from_config(config, dataset_path=args.dataset)
    language, controller, state_encoder, _state_hash = load_actual_waypoint_stack(config)
    runner = ActualGemmaWaypointForward(
        language,
        controller,
        state_encoder,
        scene_token_count=int(settings["scene_token_count"]),
        robot_token_count=int(settings["robot_token_count"]),
        hidden_size=int(settings["hidden_size"]),
        state_dim=int(settings["state_dim"]),
        history_dim=int(settings["history_dim"]),
    )
    train = select_balanced_waypoint_samples(
        dataset.split("train"), settings.get("training_sample_limit")
    )
    validation = select_balanced_waypoint_samples(
        dataset.split("validation"), settings.get("validation_sample_limit")
    )
    hidden_input_binding = gemma_hidden_input_binding(
        language,
        controller,
        state_encoder,
        cache,
        (*train, *validation),
        history_parameterization=str(settings["history_parameterization"]),
    )
    reused_train: Mapping[str, torch.Tensor] = {}
    reused_validation: Mapping[str, torch.Tensor] = {}
    forward_revalidation: Mapping[str, object] | None = None
    if args.reuse_dataset is not None:
        previous_dataset = load_waypoint_dataset_for_hidden_reuse(
            args.reuse_dataset,
            state_dim=int(settings["state_dim"]),
            history_dim=int(settings["history_dim"]),
            history_parameterization=str(settings["history_parameterization"]),
            max_history_tokens=int(settings["max_history_tokens"]),
            max_waypoint_step_m=float(settings["max_waypoint_step_m"]),
        )
        previous_train = select_balanced_waypoint_samples(
            previous_dataset.split("train"), settings.get("training_sample_limit")
        )
        previous_validation = select_balanced_waypoint_samples(
            previous_dataset.split("validation"), settings.get("validation_sample_limit")
        )
        previous_input_binding = gemma_hidden_input_binding(
            language,
            controller,
            state_encoder,
            cache,
            (*previous_train, *previous_validation),
            history_parameterization=str(settings["history_parameterization"]),
        )
        try:
            old_train_hidden, old_validation_hidden, _old_metadata = (
                load_gemma_hidden_cache(
                    args.reuse_cache,
                    train_samples=previous_train,
                    validation_samples=previous_validation,
                    dataset_sha256=previous_dataset.sha256,
                    hidden_size=int(settings["hidden_size"]),
                    expected_gemma_runtime_binding=language_gemma_runtime_binding(language),
                    expected_hidden_input_binding=previous_input_binding,
                )
            )
        except ValueError:
            if not args.revalidate_forward_contract:
                raise
            old_train_hidden, old_validation_hidden, _old_metadata = (
                load_gemma_hidden_cache_for_forward_revalidation(
                    args.reuse_cache,
                    train_samples=previous_train,
                    validation_samples=previous_validation,
                    dataset_sha256=previous_dataset.sha256,
                    hidden_size=int(settings["hidden_size"]),
                    expected_gemma_runtime_binding=language_gemma_runtime_binding(language),
                    expected_hidden_input_binding=previous_input_binding,
                )
            )
            forward_revalidation = revalidate_cached_hidden_forward_contract(
                runner,
                cache,
                previous_train,
                previous_validation,
                old_train_hidden,
                old_validation_hidden,
                sample_count_per_split=args.revalidation_sample_count,
                gemma_batch_size=args.gemma_batch_size,
            )
        reused_train = reusable_hidden_rows(previous_train, old_train_hidden)
        reused_validation = reusable_hidden_rows(previous_validation, old_validation_hidden)

    def progress(split: str):
        def report(completed: int, total: int) -> None:
            print(
                json.dumps(
                    {
                        "phase": "gemma_waypoint_hidden_cache",
                        "split": split,
                        "cache_miss_rows_completed": completed,
                        "cache_miss_rows_total": total,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        return report

    train_hidden, train_reused, train_computed = assemble_hidden_with_reuse(
        runner,
        cache,
        train,
        reused_train,
        forward_chunk_size=args.forward_chunk_size,
        gemma_batch_size=args.gemma_batch_size,
        progress=progress("train"),
    )
    validation_hidden, validation_reused, validation_computed = assemble_hidden_with_reuse(
        runner,
        cache,
        validation,
        reused_validation,
        forward_chunk_size=args.forward_chunk_size,
        gemma_batch_size=args.gemma_batch_size,
        progress=progress("validation"),
    )
    result = save_gemma_hidden_cache(
        args.output or str(settings["hidden_cache"]),
        train_hidden=train_hidden,
        validation_hidden=validation_hidden,
        train_samples=train,
        validation_samples=validation,
        dataset_sha256=dataset.sha256,
        gemma_runtime_binding=language_gemma_runtime_binding(language),
        hidden_input_binding=hidden_input_binding,
    )
    result["reuse"] = {
        "source_dataset": args.reuse_dataset,
        "source_cache": args.reuse_cache,
        "train_rows_reused": train_reused,
        "train_rows_computed": train_computed,
        "validation_rows_reused": validation_reused,
        "validation_rows_computed": validation_computed,
        "matched_on_exact_frozen_gemma_inputs": True,
        "gemma_batch_size": args.gemma_batch_size,
        "forward_contract_revalidation": forward_revalidation,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
