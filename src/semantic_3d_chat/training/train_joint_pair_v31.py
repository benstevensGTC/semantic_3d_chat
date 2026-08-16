"""Train the expanded diverse28 V31 development candidate.

V31 deliberately reuses V30's audited exact-zero/frozen-state training engine,
but starts afresh from the selector-approved V29 update_004.  This wrapper is
the fail-closed experiment boundary: it locks the enlarged training split,
keeps validation scenes 19--24 byte-for-byte at the split level, and refuses
any QA or runtime artifact from deferred final scenes 25--30.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from semantic_3d_chat.config import PROJECT_ROOT, artifact_root, load_config
from semantic_3d_chat.data.dataset import QARecord
from semantic_3d_chat.training.pair_curriculum import build_exact_question_pair_units
from semantic_3d_chat.training.train_joint_pair_v30 import (
    ApprovedV29Source,
    require_approved_v29_source,
    run_v30,
    v30_contract,
    v30_settings,
)
from semantic_3d_chat.training.train_post_stack_decoder import (
    load_stage_b_qa_records,
    v29_development_contract,
)
from semantic_3d_chat.training.train_post_stack_sidecar import _file_sha256

DEFAULT_CONFIG = Path("configs/experiments/gemma4_diverse28_joint_pair_v31.yaml")
DEFAULT_OUTPUT = Path("data_gemma4/checkpoints/gemma4_v31_diverse28_joint_pair")
_SCENE_ID = re.compile(r"scene_[0-9]{6}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TRAIN_SCENES = tuple(
    [f"scene_{index:06d}" for index in range(11, 19)]
    + [f"scene_{index:06d}" for index in range(31, 39)]
)
_NEW_TRAIN_SCENES = tuple(f"scene_{index:06d}" for index in range(31, 39))
_VALIDATION_SCENES = tuple(f"scene_{index:06d}" for index in range(19, 25))
_DEFERRED_FINAL_SCENES = tuple(f"scene_{index:06d}" for index in range(25, 31))


@dataclass(frozen=True)
class V31Contract:
    diverse28_config: Path
    diverse28_config_sha256: str
    qa_root: Path
    split_fingerprint: str
    train_scene_ids: tuple[str, ...]
    validation_scene_ids: tuple[str, ...]
    deferred_final_scene_ids: tuple[str, ...]
    train_question_count: int
    validation_question_count: int
    train_changed_pair_unit_count: int
    train_changed_pair_units_by_type: tuple[tuple[str, int], ...]
    optimizer_cycles: int
    development_changed_complete_pairs_minimum: int
    chat_promotion_changed_complete_pairs_minimum: int


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _scene_ids(raw: Mapping[str, Any], field: str) -> tuple[str, ...]:
    values = raw.get(field)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"v31_expanded_pair.{field} must be a sequence")
    parsed = tuple(str(value) for value in values)
    if not parsed or len(parsed) != len(set(parsed)):
        raise ValueError(f"v31_expanded_pair.{field} must be nonempty and unique")
    if any(_SCENE_ID.fullmatch(value) is None for value in parsed):
        raise ValueError(f"v31_expanded_pair.{field} contains an invalid scene ID")
    return parsed


def _positive_int(raw: Mapping[str, Any], field: str) -> int:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"v31_expanded_pair.{field} must be a positive integer")
    return value


def v31_contract(config: Mapping[str, Any]) -> V31Contract:
    """Validate the immutable V31 experiment envelope without loading QA."""

    raw = config.get("v31_expanded_pair")
    if not isinstance(raw, Mapping):
        raise TypeError("V31 requires a v31_expanded_pair mapping")
    required = {
        "schema_version",
        "role",
        "engine",
        "diverse28_config",
        "diverse28_config_sha256",
        "qa_root",
        "split_fingerprint",
        "train_scene_ids",
        "validation_scene_ids",
        "deferred_final_scene_ids",
        "question_counts",
        "train_changed_pair_unit_count",
        "train_changed_pair_units_by_type",
        "source_selected_update",
        "optimizer_cycles",
        "checkpoint_every_cycle",
        "inspect_every_intermediate",
        "strict_validation_nll_improvement",
        "old_color_mirror_no_new_negative_retention",
        "development_changed_complete_pairs_minimum",
        "chat_promotion_changed_complete_pairs_minimum",
        "chat_promotion_aggregate_exact_no_regression",
    }
    missing = sorted(required - set(raw))
    unknown = sorted(set(raw) - required)
    if missing or unknown:
        raise ValueError(f"Invalid v31_expanded_pair fields: missing={missing} unknown={unknown}")
    if raw["schema_version"] != 1:
        raise ValueError("v31_expanded_pair.schema_version must be 1")
    if raw["role"] != "approved_v29_diverse28_joint_pair_development_v31":
        raise ValueError("v31_expanded_pair.role does not authorize this experiment")
    if raw["engine"] != "v30_exact_zero_frozen_state_joint_pair":
        raise ValueError("V31 must reuse the audited V30 exact-zero engine")

    train_scenes = _scene_ids(raw, "train_scene_ids")
    validation_scenes = _scene_ids(raw, "validation_scene_ids")
    deferred_scenes = _scene_ids(raw, "deferred_final_scene_ids")
    if train_scenes != _TRAIN_SCENES:
        raise ValueError("V31 train split must be exactly scenes 11-18 plus 31-38")
    if validation_scenes != _VALIDATION_SCENES:
        raise ValueError("V31 validation split must remain exactly scenes 19-24")
    if deferred_scenes != _DEFERRED_FINAL_SCENES:
        raise ValueError("V31 deferred final split must remain exactly scenes 25-30")
    split_sets = (set(train_scenes), set(validation_scenes), set(deferred_scenes))
    if any(split_sets[left] & split_sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("V31 train, validation, and deferred-final splits must be disjoint")

    sha = raw["diverse28_config_sha256"]
    fingerprint = raw["split_fingerprint"]
    if not isinstance(sha, str) or _SHA256.fullmatch(sha) is None:
        raise ValueError("v31_expanded_pair.diverse28_config_sha256 must be SHA-256")
    if not isinstance(fingerprint, str) or _SHA256.fullmatch(fingerprint) is None:
        raise ValueError("v31_expanded_pair.split_fingerprint must be SHA-256")
    diverse_config_path = _resolve(str(raw["diverse28_config"]))
    if not diverse_config_path.is_file() or diverse_config_path.is_symlink():
        raise FileNotFoundError(f"Pinned diverse28 config is missing: {diverse_config_path}")
    if _file_sha256(diverse_config_path) != sha:
        raise ValueError("Pinned diverse28 config SHA-256 changed")

    qa_root = _resolve(str(raw["qa_root"]))
    if qa_root != artifact_root(dict(config), "qa").resolve():
        raise ValueError("V31 contract QA root differs from paths.qa_root")
    if "oracle" in {part.casefold() for part in qa_root.parts}:
        raise ValueError("V31 QA supervision must not use an oracle path")
    counts = raw["question_counts"]
    if not isinstance(counts, Mapping) or set(counts) != {"train", "validation"}:
        raise ValueError("V31 question_counts must contain exactly train and validation")
    train_count = _positive_int(counts, "train")
    validation_count = _positive_int(counts, "validation")
    if (train_count, validation_count) != (384, 216):
        raise ValueError("V31 balanced QA counts must remain 384 train and 216 validation")
    train_pair_count = _positive_int(raw, "train_changed_pair_unit_count")
    pair_types = raw["train_changed_pair_units_by_type"]
    expected_pair_types = {
        "chair_orientation": 1,
        "object_relocation": 4,
        "color_swap": 4,
        "object_count": 1,
        "book_support": 4,
        "mirror_lr": 4,
        "picture_support": 4,
        "object_removal": 3,
    }
    if not isinstance(pair_types, Mapping) or dict(pair_types) != expected_pair_types:
        raise ValueError("V31 changed-pair type counts differ from the locked diverse28 QA")
    if train_pair_count != sum(expected_pair_types.values()) or train_pair_count != 25:
        raise ValueError("V31 must contain exactly 25 answer-changing training units")

    cycles = _positive_int(raw, "optimizer_cycles")
    development_min = _positive_int(raw, "development_changed_complete_pairs_minimum")
    promotion_min = _positive_int(raw, "chat_promotion_changed_complete_pairs_minimum")
    if int(raw["source_selected_update"]) != 4:
        raise ValueError("V31 must start afresh from approved V29 update_004")
    if cycles != 8:
        raise ValueError("V31 must run eight bounded optimizer cycles")
    for field in (
        "checkpoint_every_cycle",
        "inspect_every_intermediate",
        "strict_validation_nll_improvement",
        "old_color_mirror_no_new_negative_retention",
        "chat_promotion_aggregate_exact_no_regression",
    ):
        if raw[field] is not True:
            raise ValueError(f"v31_expanded_pair.{field} must remain true")
    if development_min != 1 or promotion_min != 6:
        raise ValueError("V31 development/chat pair gates must remain 1/12 and 6/12")

    # Cross-check the inherited engine rather than trusting duplicated prose.
    engine_contract = v30_contract(config)
    settings = v30_settings(config)
    if int(engine_contract["source_selected_update"]) != 4:
        raise ValueError("V31 engine source is not approved V29 update_004")
    if settings.max_optimizer_steps != cycles or settings.evaluation_interval_steps != 1:
        raise ValueError("V31 must checkpoint and evaluate every one of its eight cycles")
    if (settings.sidecar_learning_rate, settings.decoder_learning_rate) != (0.0001, 0.0002):
        raise ValueError("V31 stronger optimization rates differ from their audited values")
    if settings.gradient_clip_norm != 1.0 or settings.weight_decay != 0.0:
        raise ValueError("V31 safe optimizer guardrails changed")
    selection = engine_contract["selection_requires"]
    promotion = engine_contract["promotion_requires"]
    required_retention = (
        selection.get("color_full_vocab_sides") == 12
        and selection.get("mirror_full_vocab_sides") == 10
        and selection.get("no_new_negative_sides") is True
        and selection.get("source_v29_validation_nll_must_improve") is True
        and selection.get("minimum_greedy_complete_units_correct") == development_min
        and promotion.get("validation_changed_complete_pairs_minimum") == promotion_min
        and promotion.get("aggregate_validation_exact_accuracy_no_regression") is True
    )
    if not required_retention:
        raise ValueError("V31 inherited selection or promotion gates were weakened")

    development = v29_development_contract(config)
    if development is None:
        raise ValueError("V31 requires a strict scene-disjoint QA contract")
    if development.qa_root != qa_root:
        raise ValueError("V31 QA roots disagree")
    if development.split_fingerprint != fingerprint:
        raise ValueError("V31 split fingerprints disagree")
    if development.train_scene_ids != train_scenes:
        raise ValueError("V31 training scene locks disagree")
    if development.validation_scene_ids != validation_scenes:
        raise ValueError("V31 validation scene locks disagree")
    if development.deferred_test_scene_ids != deferred_scenes:
        raise ValueError("V31 deferred final scene locks disagree")
    if (development.train_question_count, development.validation_question_count) != (
        train_count,
        validation_count,
    ):
        raise ValueError("V31 QA count locks disagree")

    # Validate the source generator's split declarations without opening any
    # oracle data or generated QA.
    diverse_config = load_config(diverse_config_path)
    splits = diverse_config.get("batch", {}).get("splits", {})
    if tuple(splits.get("train", ())) != train_scenes:
        raise ValueError("Pinned diverse28 generator train split differs from V31")
    if tuple(splits.get("validation", ())) != validation_scenes:
        raise ValueError("Pinned diverse28 generator validation split differs from V31")
    if tuple(splits.get("test", ())) != deferred_scenes:
        raise ValueError("Pinned diverse28 generator final split differs from V31")

    return V31Contract(
        diverse28_config=diverse_config_path,
        diverse28_config_sha256=sha,
        qa_root=qa_root,
        split_fingerprint=fingerprint,
        train_scene_ids=train_scenes,
        validation_scene_ids=validation_scenes,
        deferred_final_scene_ids=deferred_scenes,
        train_question_count=train_count,
        validation_question_count=validation_count,
        train_changed_pair_unit_count=train_pair_count,
        train_changed_pair_units_by_type=tuple(expected_pair_types.items()),
        optimizer_cycles=cycles,
        development_changed_complete_pairs_minimum=development_min,
        chat_promotion_changed_complete_pairs_minimum=promotion_min,
    )


def load_v31_qa_records(
    config: Mapping[str, Any],
) -> tuple[list[QARecord], list[QARecord], dict[str, Any]]:
    """Load the complete development QA only after validating the V31 lock."""

    contract = v31_contract(config)
    train, validation, audit = load_stage_b_qa_records(
        config,
        max_train_questions=None,
        max_validation_questions=None,
    )
    if tuple(sorted({record.scene_id for record in train})) != contract.train_scene_ids:
        raise ValueError("Loaded V31 train QA differs from its locked scenes")
    if tuple(sorted({record.scene_id for record in validation})) != contract.validation_scene_ids:
        raise ValueError("Loaded V31 validation QA differs from its locked scenes")
    touched = {record.scene_id for record in (*train, *validation)} & set(
        contract.deferred_final_scene_ids
    )
    if touched:
        raise ValueError(f"V31 loaded deferred final QA: {sorted(touched)}")
    pair_units = build_exact_question_pair_units(train)
    observed_types = Counter(
        str(unit.reference.counterfactual_change_type) for unit in pair_units
    )
    if len(pair_units) != contract.train_changed_pair_unit_count:
        raise ValueError("V31 loaded changed-pair unit count differs from its lock")
    if dict(observed_types) != dict(contract.train_changed_pair_units_by_type):
        raise ValueError("V31 loaded changed-pair type distribution differs from its lock")
    return train, validation, {**audit, "v31_expanded_pair_contract": True}


def preflight_v31(config: Mapping[str, Any], *, require_qa: bool = True) -> ApprovedV29Source:
    """Verify the experiment, immutable V29 source, and optionally persisted QA."""

    v31_contract(config)
    source = require_approved_v29_source(config)
    if require_qa:
        load_v31_qa_records(config)
    return source


def run_v31(*, config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Run V31 through the audited V30 engine after all expanded-data checks."""

    contract = v31_contract(config)
    source = preflight_v31(config, require_qa=True)
    engine_report = run_v30(
        config=config,
        output=output,
        allow_unpinned_source_scene_ids=_NEW_TRAIN_SCENES,
    )
    if int(engine_report["optimizer_updates"]) != contract.optimizer_cycles:
        raise RuntimeError("V31 engine did not save all eight optimizer cycles")
    if engine_report.get("final_test_scene_ids_loaded") != []:
        raise RuntimeError("V31 engine touched deferred final scenes")
    return {
        **engine_report,
        "artifact": "v31_diverse28_joint_pair_training",
        "engine_artifact": engine_report["artifact"],
        "source_v29_checkpoint": str(source.checkpoint),
        "train_scene_ids": list(contract.train_scene_ids),
        "validation_scene_ids": list(contract.validation_scene_ids),
        "deferred_final_scene_ids": list(contract.deferred_final_scene_ids),
        "all_intermediate_checkpoints_required": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.preflight_only:
        source = preflight_v31(config, require_qa=True)
        report = {
            "artifact": "v31_diverse28_joint_pair_preflight",
            "passed": True,
            "source_v29_checkpoint": str(source.checkpoint),
            "final_test_scenes_touched": False,
        }
    else:
        report = run_v31(config=config, output=_resolve(args.output))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V31Contract",
    "load_v31_qa_records",
    "preflight_v31",
    "run_v31",
    "v31_contract",
]
