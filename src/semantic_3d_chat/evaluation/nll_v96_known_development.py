"""Separately authorized aggregate-only NLL evaluator for V96 development."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.evaluation.predict_v96_known_development import (
    load_predictor_stack_v96,
)
from semantic_3d_chat.evaluation.score_v96_known_development import load_references_v96
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import sha256_file_v85
from semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight import (
    CONFIG,
    load_config_v96,
)
from semantic_3d_chat.evaluation.v96_known_development_common import (
    ARMS,
    CHANGED_SIDE_COUNT,
    FULL_INTERIOR_PERMUTATION,
    NLL_ARTIFACT,
    NLL_COMPLETION_ARTIFACT,
    PAIRED_WRONG_SCENE,
    PRIMARY,
    QUESTION_COUNT,
    REFERENCE_SHA256,
    SCHEMA_VERSION,
    ZERO_PAYLOAD,
    assert_aggregate_only_v96,
    assert_bound_config_path_v96,
    assert_output_bundle_state_v96,
    assert_same_candidate_v96,
    audit_report_v96,
    authenticate_fixed_final_candidate_v96,
    authenticate_fixed_inputs_before_questions_v96,
    authenticate_nll_bundle_v96,
    canonical_sha256_v96,
    evaluation_paths_v96,
    load_known_questions_v96,
    mandatory_fixed_input_reads_v96,
    nll_forbidden_roots_v96,
    resolve_v96,
    validate_nll_metrics_v96,
    write_json_create_once_v96,
)
from semantic_3d_chat.evaluation.v96_known_development_implementation import (
    hardened_evaluation_stage_v96,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.training.train_v84_strict_bridge import _measure_nll_v84


def _mean(values: Sequence[float], label: str) -> float:
    if not values:
        raise ValueError(f"V96 NLL aggregate is empty: {label}")
    return sum(values) / len(values)


@hardened_evaluation_stage_v96
@torch.inference_mode()
def measure_known_development_nll_v96(
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    """Measure four arms while serializing no row-level label-bearing values."""

    assert_bound_config_path_v96(config_path)
    initial = load_config_v96(config_path, allow_draft=False)
    paths = evaluation_paths_v96(initial)
    outputs = (paths.nll, paths.nll_access, paths.nll_completion)
    states = [path.exists() or path.is_symlink() for path in outputs]
    if all(states):
        authenticated = authenticate_nll_bundle_v96(config_path)
        return {
            **authenticated["report"],
            "nll_sha256": authenticated["sha256"],
            "reused_authenticated_create_once_bundle": True,
        }
    assert_output_bundle_state_v96(outputs, complete=False)

    audit = FileAccessAudit(
        nll_forbidden_roots_v96(initial),
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    with audit:
        config = load_config_v96(config_path, allow_draft=False)
        fixed = authenticate_fixed_inputs_before_questions_v96(
            config, config_path=config_path, audit=audit
        )
        stack = load_predictor_stack_v96(
            config,
            expected_candidate_state_sha256=str(fixed.candidate["state_sha256"]),
        )
        questions = load_known_questions_v96()
        references = load_references_v96(config, questions)
        totals: defaultdict[str, list[float]] = defaultdict(list)
        for ordinal, reference in enumerate(references, 1):
            scene_id = str(reference["scene_id"])
            row = SimpleNamespace(
                question=str(reference["question"]), answer=str(reference["answer"])
            )
            arm_nll: dict[str, float] = {}
            for arm in ARMS:
                measured, _layout = _measure_nll_v84(
                    stack.language,
                    stack.system_prompt,
                    fixed.memories[arm][scene_id],
                    row,
                )
                arm_nll[arm] = float(measured["mean_nll"])
            primary = arm_nll[PRIMARY]
            for arm in ARMS:
                totals[arm].append(arm_nll[arm])
            totals["wrong_gap"].append(arm_nll[PAIRED_WRONG_SCENE] - primary)
            totals["zero_gap"].append(arm_nll[ZERO_PAYLOAD] - primary)
            totals["permutation_gap"].append(
                arm_nll[FULL_INTERIOR_PERMUTATION] - primary
            )
            if reference.get("counterfactual_expected_change") is True:
                totals["changed_wrong_gap"].append(
                    arm_nll[PAIRED_WRONG_SCENE] - primary
                )
            if ordinal == 1 or ordinal % 12 == 0 or ordinal == QUESTION_COUNT:
                print(
                    json.dumps(
                        {
                            "event": "v96_known_development_nll",
                            "ordinal": ordinal,
                            "total": QUESTION_COUNT,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        memory_hashes_after = {
            arm: {
                scene_id: prefix_sha256(fixed.memories[arm][scene_id])
                for scene_id in fixed.memories[arm]
            }
            for arm in ARMS
        }
        candidate_after = authenticate_fixed_final_candidate_v96(
            config, config_path=config_path, audit=audit
        )
    audit.assert_clean()
    assert_same_candidate_v96(fixed.candidate, candidate_after)
    if memory_hashes_after != fixed.memory_hashes:
        raise RuntimeError("V96 NLL evaluator mutated a fixed scene/control memory")
    metrics = {
        "primary_mean_nll": _mean(totals[PRIMARY], PRIMARY),
        "paired_wrong_scene_mean_nll": _mean(
            totals[PAIRED_WRONG_SCENE], PAIRED_WRONG_SCENE
        ),
        "zero_payload_mean_nll": _mean(totals[ZERO_PAYLOAD], ZERO_PAYLOAD),
        "full_interior_permutation_mean_nll": _mean(
            totals[FULL_INTERIOR_PERMUTATION], FULL_INTERIOR_PERMUTATION
        ),
        "mean_wrong_minus_primary_nll": _mean(totals["wrong_gap"], "wrong gap"),
        "mean_changed_wrong_minus_primary_nll": _mean(
            totals["changed_wrong_gap"], "changed wrong gap"
        ),
        "zero_payload_mean_nll_gap": _mean(totals["zero_gap"], "zero gap"),
        "full_interior_permutation_mean_nll_gap": _mean(
            totals["permutation_gap"], "permutation gap"
        ),
        "row_count_per_arm": len(totals[PRIMARY]),
        "changed_row_count": len(totals["changed_wrong_gap"]),
    }
    validate_nll_metrics_v96(metrics)
    report = {
        "artifact": NLL_ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "status": "measured_aggregate_only_not_yet_gated",
        "candidate_fingerprint_sha256": fixed.candidate["fingerprint_sha256"],
        "frozen_v95_state_sha256": fixed.candidate["frozen_v95_state_sha256"],
        "memory_manifest_sha256": fixed.memory_manifest_sha256,
        "bound_memory_inventory_sha256": canonical_sha256_v96(fixed.memory_hashes),
        "question_manifest_sha256": questions.manifest_sha256,
        "questions_sha256": questions.questions_sha256,
        "reference_sha256": REFERENCE_SHA256,
        "row_count": QUESTION_COUNT,
        "scene_count": 6,
        "arms": list(ARMS),
        "fixed_final_and_memories_authenticated_before_labels_opened": True,
        "labels_opened_only_by_separate_nll_evaluator": True,
        "row_level_content_serialized": False,
        "metrics": metrics,
        "runtime_promotion_authorized": False,
    }
    assert_aggregate_only_v96(report)
    access = audit_report_v96(audit)
    mandatory = mandatory_fixed_input_reads_v96(config, fixed, config_path=config_path)
    label_path = resolve_v96(config["known_development_gate"]["labels_path"])
    mandatory.add(str(label_path.resolve()))
    if (
        access["passed"] is not True
        or access["protected_read_count"] != 0
        or not mandatory <= set(access["loaded_files"])
    ):
        raise RuntimeError("V96 NLL file-access evidence is incomplete")

    write_json_create_once_v96(paths.nll, report)
    write_json_create_once_v96(paths.nll_access, access)
    completion = {
        "artifact": NLL_COMPLETION_ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "candidate_fingerprint_before": fixed.candidate["fingerprint_sha256"],
        "candidate_fingerprint_after": candidate_after["fingerprint_sha256"],
        "candidate_immutable": True,
        "frozen_v95_parent_immutable": (
            fixed.candidate["frozen_v95_state_sha256"]
            == candidate_after["frozen_v95_state_sha256"]
        ),
        "memory_hashes_invariant": True,
        "nll_sha256": sha256_file_v85(paths.nll),
        "nll_access_sha256": sha256_file_v85(paths.nll_access),
        "row_count_per_arm": QUESTION_COUNT,
        "changed_row_count": CHANGED_SIDE_COUNT,
        "row_level_content_serialized": False,
        "runtime_promotion_authorized": False,
    }
    assert_aggregate_only_v96(completion)
    write_json_create_once_v96(paths.nll_completion, completion)
    authenticated = authenticate_nll_bundle_v96(
        config_path, fixed=fixed, questions=questions
    )
    return {
        **report,
        "nll_sha256": authenticated["sha256"],
        "reused_authenticated_create_once_bundle": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(measure_known_development_nll_v96(args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "measure_known_development_nll_v96"]
