#!/usr/bin/env python3
"""Assemble the V15 room-generalization release summary from real artifacts.

Unlike the hand-authored V14 summary, every field here is read out of a file
produced by the pipeline and every referenced file is hashed, so the summary
cannot drift from the evidence it describes.  Missing stages are reported as
absent rather than silently omitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "semantic_3d_chat.gemma_waypoint_release_summary.v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _evidence(paths: Mapping[str, Path]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, path in paths.items():
        rooted = path if path.is_absolute() else ROOT / path
        out[name] = (
            {"path": path.as_posix(), "sha256": _sha256(rooted), "present": True}
            if rooted.is_file()
            else {"path": path.as_posix(), "present": False}
        )
    return out


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".partial")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def build_summary(
    *,
    checkpoint: Path,
    training_metrics: Path,
    general_dataset: Path,
    sealed_dataset: Path,
    sealed_controls: Path,
    heldout_score: Path,
) -> dict[str, Any]:
    metadata = _read(ROOT / checkpoint / "runtime_metadata.json")
    training = _read(ROOT / training_metrics)
    general = _read(ROOT / general_dataset / "manifest.json")
    sealed = _read(ROOT / sealed_dataset / "manifest.json")
    controls = _read(ROOT / sealed_controls)
    heldout = _read(ROOT / heldout_score)

    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": "gemma_waypoint_v15_room_generalization",
        "navigation_checkpoint": checkpoint.as_posix(),
        "checkpoint_present": metadata is not None,
    }
    if metadata is not None:
        summary["navigation_checkpoint_sha256"] = metadata["weights_sha256"]
        summary["gemma_runtime_binding_sha256"] = metadata[
            "gemma_runtime_binding_sha256"
        ]
        summary["model_id"] = metadata["model_id"]
        summary["model_revision"] = metadata["model_revision"]
        summary["runtime_contract"] = {
            "checkpoint_schema": metadata["schema"],
            "history_dim": metadata["history_dim"],
            "history_parameterization": metadata["history_parameterization"],
            "scene_token_count": metadata["scene_token_count"],
            "robot_token_count": metadata["robot_token_count"],
            "actual_gemma_causal_forward": metadata["actual_gemma_causal_forward"],
            "complete_scene_prefix_required": metadata[
                "complete_scene_prefix_required"
            ],
            "every_scene_token_processed": metadata["every_scene_token_processed"],
            "model_selects_every_waypoint_and_heading": metadata[
                "model_selects_every_waypoint_and_heading"
            ],
            "deterministic_route_planner_allowed_at_runtime": metadata[
                "deterministic_route_planner_allowed_at_runtime"
            ],
            "oracle_inputs_at_runtime": metadata["oracle_inputs_at_runtime"],
            "environmental_text_inputs": metadata["environmental_text_inputs"],
            "training_scene_count": metadata["training_scene_count"],
            "validation_scene_count": metadata["validation_scene_count"],
            "training_sample_count": metadata["training_sample_count"],
            "validation_sample_count": metadata["validation_sample_count"],
        }
    if general is not None:
        summary["rooms"] = {
            "train_scene_ids": general["train_scene_ids"],
            "development_scene_ids": general["validation_scene_ids"],
            "sealed_scene_ids": (
                sealed["validation_scene_ids"] if sealed is not None else None
            ),
            "train_room_count": general["train_scene_count"],
            "development_room_count": general["validation_scene_count"],
            "sealed_room_count": (
                sealed["validation_scene_count"] if sealed is not None else None
            ),
            "generated_rows": general["sample_count"],
            "generated_episodes": general["episode_count"],
            "action_sample_counts": general["action_sample_counts"],
            "unroutable_lap_start_count": general["unroutable_lap_start_count"],
        }
    if training is not None:
        summary["training"] = {
            "dataset_sha256": training["dataset_sha256"],
            "best_epoch": training["best_epoch"],
            "epochs_completed": training["epochs_completed"],
            "elapsed_seconds": training["elapsed_seconds"],
            "training_rows": training["gemma_hidden_cache"]["training_rows"],
            "validation_rows": training["gemma_hidden_cache"]["validation_rows"],
            "training_action_accuracy": training["training_metrics"][
                "action_accuracy"
            ],
            "training_stop_recall": training["training_metrics"]["stop_recall"],
        }
        summary["development_controls"] = _control_block(training.get("controls"))
    if controls is not None:
        summary["sealed_controls"] = _control_block(controls)
    if heldout is not None:
        summary["sealed_closed_loop"] = {
            "goal_count": heldout["goal_count"],
            "passed_count": heldout["passed_count"],
            "pass_rate": heldout["pass_rate"],
            "model_selected_terminal_stop_rate": heldout[
                "model_selected_terminal_stop_rate"
            ],
            "per_metric": heldout["per_metric"],
            "scene_ids": heldout["scene_ids"],
            "rollout_process_read_oracle": heldout["rollout_process_read_oracle"],
        }
    summary["evidence"] = _evidence(
        {
            "checkpoint_metadata": checkpoint / "runtime_metadata.json",
            "checkpoint_weights": checkpoint / "policy.safetensors",
            "training_report": training_metrics,
            "general_trace_manifest": general_dataset / "manifest.json",
            "sealed_trace_manifest": sealed_dataset / "manifest.json",
            "sealed_controls": sealed_controls,
            "heldout_closed_loop_score": heldout_score,
        }
    )
    summary["stages_present"] = {
        name: bool(value["present"]) for name, value in summary["evidence"].items()
    }
    summary["complete"] = all(summary["stages_present"].values())
    return summary


def _control_block(controls: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(controls, Mapping):
        return None
    conditions = controls["conditions"]
    return {
        "evaluated_conditions": controls.get(
            "evaluated_conditions", sorted(conditions)
        ),
        "sample_count": conditions["primary"]["sample_count"],
        "primary": {
            "action_accuracy": conditions["primary"]["action_accuracy"],
            "action_macro_recall": conditions["primary"]["action_macro_recall"],
            "action_recall": conditions["primary"]["action_recall"],
            "stop_recall": conditions["primary"]["stop_recall"],
            "stop_precision": conditions["primary"]["stop_precision"],
            "waypoint_error_m_mean": conditions["primary"]["waypoint_error_m_mean"],
            "heading_error_degrees_mean": conditions["primary"][
                "heading_error_degrees_mean"
            ],
        },
        "accuracy_drop_from_primary": controls["accuracy_drop_from_primary"],
        "output_change_from_primary": controls["output_change_from_primary"],
        "per_condition_action_accuracy": {
            name: value["action_accuracy"] for name, value in conditions.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data_gemma4/checkpoints/gemma_waypoint_policy_v15_general"),
    )
    parser.add_argument(
        "--training-metrics",
        type=Path,
        default=Path(
            "reports/gemma4/metrics/gemma_waypoint_policy_v15_general_training.json"
        ),
    )
    parser.add_argument(
        "--general-dataset",
        type=Path,
        default=Path("data_gemma4/training/gemma_waypoint_policy_v15_general"),
    )
    parser.add_argument(
        "--sealed-dataset",
        type=Path,
        default=Path("data_gemma4/training/gemma_waypoint_policy_v15_sealed"),
    )
    parser.add_argument(
        "--sealed-controls",
        type=Path,
        default=Path("reports/gemma4/metrics/gemma_waypoint_v15_sealed_controls.json"),
    )
    parser.add_argument(
        "--heldout-score",
        type=Path,
        default=Path("reports/gemma4/metrics/gemma_waypoint_v15_heldout_score.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/gemma4/metrics/gemma_waypoint_v15_summary.json"),
    )
    args = parser.parse_args()
    summary = build_summary(
        checkpoint=args.checkpoint,
        training_metrics=args.training_metrics,
        general_dataset=args.general_dataset,
        sealed_dataset=args.sealed_dataset,
        sealed_controls=args.sealed_controls,
        heldout_score=args.heldout_score,
    )
    _atomic_json(
        args.output if args.output.is_absolute() else ROOT / args.output, summary
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
