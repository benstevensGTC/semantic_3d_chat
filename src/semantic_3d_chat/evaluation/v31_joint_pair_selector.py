"""Independently select among all V31 diverse28 development checkpoints.

The numerical and causal evidence is recomputed by the audited V30 selector;
this wrapper adds V31's expanded-dataset provenance checks and refuses a run
whose nine contiguous arms were not all trained on scenes 11--18 plus 31--38.
Passing the development gate is not chat promotion: promotion still requires
at least 6/12 complete changed validation pairs and aggregate non-regression.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from semantic_3d_chat.config import config_hash, load_config
from semantic_3d_chat.evaluation.v27_sidecar_screen import _atomic_json
from semantic_3d_chat.evaluation.v30_joint_pair_selector import (
    ArmEvaluator,
    SelectionRequirements,
    _RuntimeEvaluator,
    select_joint_pair,
)
from semantic_3d_chat.training.checkpointing import TRAINING_METADATA_FILENAME
from semantic_3d_chat.training.train_joint_pair_v31 import V31Contract, v31_contract

DEFAULT_CONFIG = Path("configs/experiments/gemma4_diverse28_joint_pair_v31.yaml")
DEFAULT_CHECKPOINT_ROOT = Path("data_gemma4/checkpoints/gemma4_v31_diverse28_joint_pair")
DEFAULT_OUTPUT = Path("reports/gemma4/metrics/v31_joint_pair_selection.json")


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def validate_v31_checkpoint_envelope(
    config: Mapping[str, Any], checkpoint_root: Path, contract: V31Contract
) -> tuple[Path, ...]:
    """Require update_000 through update_008 and exact expanded QA provenance."""

    paths = tuple(checkpoint_root / f"update_{index:03d}" for index in range(9))
    observed = sorted(path.name for path in checkpoint_root.glob("update_*") if path.is_dir())
    expected = [path.name for path in paths]
    if observed != expected:
        raise FileNotFoundError(
            f"V31 must expose every intermediate checkpoint: observed={observed} expected={expected}"
        )
    expected_config_hash = config_hash(dict(config))
    for index, path in enumerate(paths):
        metadata_path = path / TRAINING_METADATA_FILENAME
        if not metadata_path.is_file():
            raise FileNotFoundError(f"V31 checkpoint lacks metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, Mapping):
            raise TypeError(f"V31 checkpoint metadata is not a mapping: {path.name}")
        if metadata.get("optimizer_step") != index:
            raise ValueError(f"V31 checkpoint/update mismatch: {path.name}")
        if metadata.get("config_hash") != expected_config_hash:
            raise ValueError(f"V31 checkpoint config hash mismatch: {path.name}")
        engine = _mapping(metadata.get("v30_joint_pair"), "metadata.v30_joint_pair")
        if tuple(engine.get("train_scene_ids", ())) != contract.train_scene_ids:
            raise ValueError(f"V31 checkpoint train split mismatch: {path.name}")
        if tuple(engine.get("validation_scene_ids", ())) != contract.validation_scene_ids:
            raise ValueError(f"V31 checkpoint validation split mismatch: {path.name}")
        if engine.get("train_question_count") != contract.train_question_count:
            raise ValueError(f"V31 checkpoint train QA count mismatch: {path.name}")
        if engine.get("validation_question_count") != contract.validation_question_count:
            raise ValueError(f"V31 checkpoint validation QA count mismatch: {path.name}")
        if engine.get("final_test_scene_ids_loaded") != []:
            raise ValueError(f"V31 checkpoint touched deferred final scenes: {path.name}")
        qa = _mapping(engine.get("qa_dataset"), "metadata.v30_joint_pair.qa_dataset")
        if Path(str(qa.get("qa_root", ""))).resolve() != contract.qa_root:
            raise ValueError(f"V31 checkpoint QA root mismatch: {path.name}")
        if qa.get("split_fingerprint") != contract.split_fingerprint:
            raise ValueError(f"V31 checkpoint split fingerprint mismatch: {path.name}")
        if qa.get("deferred_test_scene_ids_loaded") != []:
            raise ValueError(f"V31 checkpoint loaded deferred final QA: {path.name}")
    return paths


def select_v31(
    config_path: Path,
    checkpoint_root: Path,
    *,
    evaluator_factory: Callable[
        [dict[str, Any], dict[str, Any], Path, SelectionRequirements], ArmEvaluator
    ] = _RuntimeEvaluator,
) -> dict[str, Any]:
    """Run the independent selector and add a fail-closed V31 envelope."""

    config = load_config(config_path)
    contract = v31_contract(config)
    validate_v31_checkpoint_envelope(config, checkpoint_root, contract)
    engine_report = select_joint_pair(
        config_path,
        checkpoint_root,
        evaluator_factory=evaluator_factory,
    )
    arms = engine_report.get("arms")
    if not isinstance(arms, list) or [arm.get("update") for arm in arms] != list(range(9)):
        raise ValueError("V31 selector did not inspect every intermediate checkpoint")
    if engine_report.get("validation_scene_ids") != list(contract.validation_scene_ids):
        raise ValueError("V31 selector validation scenes changed")
    requirements = _mapping(engine_report.get("requirements"), "requirements")
    if requirements.get("selected_v29_source_nll_must_improve") is not True:
        raise ValueError("V31 selector lost strict validation-NLL improvement")
    if (
        requirements.get("color_full_vocab_sides") != 12
        or requirements.get("mirror_full_vocab_sides") != 10
        or requirements.get("no_new_negative_sides") is not True
    ):
        raise ValueError("V31 selector weakened the old causal retention controls")
    if requirements.get("minimum_greedy_complete_units_correct") != 1:
        raise ValueError("V31 development progress must require at least 1/12 changed pairs")
    if requirements.get("chat_promotion_changed_complete_pairs_minimum") != 6:
        raise ValueError("V31 chat promotion must require at least 6/12 changed pairs")
    if (
        requirements.get("chat_promotion_aggregate_validation_exact_accuracy_no_regression")
        is not True
    ):
        raise ValueError("V31 chat promotion must retain aggregate validation exact accuracy")
    for arm in arms[1:]:
        checks = _mapping(arm.get("checks"), f"arm[{arm.get('update')}].checks")
        required_checks = {
            "color_retained",
            "mirror_retained",
            "no_new_negative_sides",
            "below_selected_v29_source_nll",
            "greedy_changed_units_demonstrated",
            "broad_exact_accuracy_retained",
        }
        if not required_checks <= set(checks):
            raise ValueError(f"V31 intermediate arm lacks mandatory checks: {arm.get('update')}")

    promotion = _mapping(engine_report.get("chat_promotion"), "chat_promotion")
    promotion_checks = _mapping(promotion.get("checks"), "chat_promotion.checks")
    if set(promotion_checks) != {
        "development_checkpoint_selected",
        "changed_complete_pair_threshold_met",
        "aggregate_validation_exact_accuracy_retained",
    }:
        raise ValueError("V31 chat-promotion evidence is incomplete")
    return {
        **engine_report,
        "artifact": "v31_diverse28_joint_pair_development_selection",
        "engine_artifact": engine_report["artifact"],
        "all_intermediate_checkpoints_inspected": True,
        "train_scene_ids": list(contract.train_scene_ids),
        "validation_scene_ids": list(contract.validation_scene_ids),
        "deferred_final_scene_ids": list(contract.deferred_final_scene_ids),
        "final_test_scenes_touched": False,
        "development_progress_is_not_chat_promotion": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = select_v31(args.config, args.checkpoint_root)
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["select_v31", "validate_v31_checkpoint_envelope"]
