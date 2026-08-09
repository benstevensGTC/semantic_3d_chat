"""Strict report-only selector for the two predeclared V17 residual LR arms.

The selector reads resolved experiment configs and training reports only.  It
does not load model checkpoints, QA/oracle files, or a chat runtime.  Every arm
must satisfy the complete shared provenance contract before any epoch can be
ranked.  Color preservation is an eligibility filter.  The selector first
chooses one mirror-ranked representative epoch per arm, then compares those two
representatives with the same mirror ranking and the declared lower-LR final
tiebreaker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from semantic_3d_chat.config import PROJECT_ROOT, config_hash, load_config

COLOR_PAIR_ID = "pair_000001"
MIRROR_PAIR_ID = "pair_000003"
EXPECTED_ARM_SPECS: dict[str, dict[str, Any]] = {
    "lr1e4": {
        "learning_rate": 1.0e-4,
        "config_hash": "e98ec331d74e",
        "output_namespace": "gemma4_color_mirror_global_scene_residual_v17_lr1e4",
    },
    "lr3e4": {
        "learning_rate": 3.0e-4,
        "config_hash": "46eaa68f482d",
        "output_namespace": "gemma4_color_mirror_global_scene_residual_v17_lr3e4",
    },
}
EXPECTED_RANKING_FIELDS = (
    "mirror_full_vocab_units",
    "mirror_full_vocab_sides",
    "mirror_candidate_units",
    "mirror_candidate_sides",
    "mirror_mean_full_vocab_margin",
    "mirror_minimum_full_vocab_margin",
)
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_OBJECT_ID = re.compile(r"[0-9a-f]{40,64}")


class ResidualLRResponseViolation(ValueError):
    """A strict V17 evidence/configuration violation."""


def _fail(message: str) -> None:
    raise ResidualLRResponseViolation(message)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _fail(f"{field} must be a sequence")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{field} must be a positive integer")
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{field} must be finite")
    return result


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_finite_tree(value: Any, field: str = "report") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_tree(item, f"{field}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _assert_finite_tree(item, f"{field}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        _fail(f"{field} contains NaN or infinity")


def _validate_source_provenance(value: Any, field: str) -> dict[str, Any]:
    source = dict(_mapping(value, field))
    if source.get("schema_version") != 1:
        _fail(f"{field}.schema_version must be 1")
    if source.get("scope") != "repository_excluding_generated_artifacts_v1":
        _fail(f"{field}.scope is not the repository training scope")
    if source.get("available") is not True or source.get("is_clean") is not True:
        _fail(f"{field} must record clean, available source")
    for key in ("head_commit", "head_tree"):
        item = source.get(key)
        if not isinstance(item, str) or _GIT_OBJECT_ID.fullmatch(item) is None:
            _fail(f"{field}.{key} must be a Git object ID")
    if _sha256(source.get("tracked_diff_sha256"), f"{field}.tracked_diff_sha256") != (
        _EMPTY_SHA256
    ):
        _fail(f"{field} records a non-empty tracked diff")
    return source


def _response_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    response = dict(_mapping(config.get("lr_response"), "config.lr_response"))
    required = {
        "schema_version": 1,
        "role": "exact_restart_global_scene_residual_learning_rate_response",
        "arms": [1.0e-4, 3.0e-4],
        "updates_per_arm": 4,
        "conditional_max_updates": 12,
        "color_pair_id": COLOR_PAIR_ID,
        "mirror_pair_id": MIRROR_PAIR_ID,
        "ranking_descending": list(EXPECTED_RANKING_FIELDS),
        "final_tiebreaker": "lower_learning_rate",
        "greedy_audit_only_after_full_teacher_gate": True,
    }
    for key, expected in required.items():
        if response.get(key) != expected:
            _fail(f"config.lr_response.{key} mismatch: expected={expected!r}")
    for section in (
        "eligibility_requires",
        "continuation_requires",
        "full_teacher_gate_requires",
    ):
        _mapping(response.get(section), f"config.lr_response.{section}")
    for key in (
        "expected_selection_sha256",
        "expected_pair_membership_sha256",
        "expected_source_adapter_sha256",
        "expected_source_metadata_sha256",
        "expected_frozen_scene_state_sha256",
        "expected_frozen_inherited_bank_sha256",
        "expected_frozen_extension_bank_sha256",
        "expected_initial_residual_state_sha256",
    ):
        _sha256(response.get(key), f"config.lr_response.{key}")
    return response


def _validate_arm_config(name: str, config: Mapping[str, Any]) -> dict[str, Any]:
    if name not in EXPECTED_ARM_SPECS:
        _fail(f"Unexpected V17 arm name: {name!r}")
    spec = EXPECTED_ARM_SPECS[name]
    observed_hash = config_hash(dict(config))
    if observed_hash != spec["config_hash"]:
        _fail(
            f"{name} config hash mismatch: expected={spec['config_hash']} observed={observed_hash}"
        )
    training = _mapping(config.get("training"), f"{name}.training")
    learning_rate = _finite(training.get("learning_rate"), f"{name}.training.learning_rate")
    if learning_rate != spec["learning_rate"]:
        _fail(f"{name} learning rate mismatch: {learning_rate}")
    if training.get("output_namespace") != spec["output_namespace"]:
        _fail(f"{name} output namespace mismatch")
    if training.get("epochs") != 4 or training.get("gradient_accumulation") != 12:
        _fail(f"{name} must declare four epochs and accumulation 12")
    if training.get("pair_steps_per_epoch") != 12:
        _fail(f"{name} must declare 12 paired microsteps per epoch")
    if training.get("train_global_scene_residual_only") is not True:
        _fail(f"{name} must be residual-only training")
    if training.get("freeze_scene_adapter") is not True:
        _fail(f"{name} must freeze the core scene adapter")
    response = _response_contract(config)
    if _finite(response.get("arm_learning_rate"), f"{name}.lr_response.arm_learning_rate") != (
        learning_rate
    ):
        _fail(f"{name} response arm learning rate does not match training")
    return {
        "learning_rate": learning_rate,
        "config_hash": observed_hash,
        "output_namespace": spec["output_namespace"],
        "response_contract": response,
    }


def _count_from_accuracy(value: Any, total: int, field: str) -> int:
    accuracy = _finite(value, field)
    if not 0.0 <= accuracy <= 1.0:
        _fail(f"{field} must be in [0,1]")
    raw = accuracy * total
    count = round(raw)
    if not math.isclose(raw, count, rel_tol=0.0, abs_tol=1.0e-5):
        _fail(f"{field} does not represent an integer count over {total}")
    return int(count)


def _pair_epoch_metrics(pair: Any, pair_id: str) -> dict[str, float | int]:
    value = _mapping(pair, pair_id)
    if value.get("ranking_mode") != "candidate_logit":
        _fail(f"{pair_id} is not a candidate-logit gate")
    for key, expected in (
        ("same_next_token_distribution", True),
        ("shared_candidate_tokens_excluded", True),
        ("free_generation_evaluated", False),
        ("first_answer_token_full_vocab_evaluated", True),
    ):
        if value.get(key) is not expected:
            _fail(f"{pair_id}.{key} mismatch")
    unit_count = _positive_int(value.get("unit_count"), f"{pair_id}.unit_count")
    side_count = _positive_int(value.get("side_count"), f"{pair_id}.side_count")
    if unit_count != 6 or side_count != 12:
        _fail(f"{pair_id} must contain exactly 6 units and 12 sides")
    return {
        "full_vocab_units": _count_from_accuracy(
            value.get("first_answer_token_top1_unit_accuracy"),
            unit_count,
            f"{pair_id}.first_answer_token_top1_unit_accuracy",
        ),
        "full_vocab_sides": _count_from_accuracy(
            value.get("first_answer_token_top1_accuracy"),
            side_count,
            f"{pair_id}.first_answer_token_top1_accuracy",
        ),
        "candidate_units": _count_from_accuracy(
            value.get("changed_unit_accuracy"),
            unit_count,
            f"{pair_id}.changed_unit_accuracy",
        ),
        "candidate_sides": _count_from_accuracy(
            value.get("side_accuracy"), side_count, f"{pair_id}.side_accuracy"
        ),
        "mean_full_vocab_margin": _finite(
            value.get("mean_first_answer_token_target_vs_best_other_logit_margin"),
            f"{pair_id}.mean_first_answer_token_target_vs_best_other_logit_margin",
        ),
        "minimum_full_vocab_margin": _finite(
            value.get("minimum_first_answer_token_target_vs_best_other_logit_margin"),
            f"{pair_id}.minimum_first_answer_token_target_vs_best_other_logit_margin",
        ),
        "mean_candidate_margin": _finite(
            value.get("mean_own_vs_alternate_candidate_logit_margin"),
            f"{pair_id}.mean_own_vs_alternate_candidate_logit_margin",
        ),
        "minimum_candidate_margin": _finite(
            value.get("minimum_own_vs_alternate_candidate_logit_margin"),
            f"{pair_id}.minimum_own_vs_alternate_candidate_logit_margin",
        ),
    }


def _color_eligible(metrics: Mapping[str, float | int], contract: Mapping[str, Any]) -> bool:
    required = _mapping(contract.get("eligibility_requires"), "eligibility_requires")
    return bool(
        metrics["full_vocab_sides"] == required.get("color_full_vocab_sides")
        and metrics["full_vocab_units"] == required.get("color_full_vocab_units")
        and metrics["minimum_candidate_margin"] > 0.0
        and metrics["minimum_full_vocab_margin"] > 0.0
        and required.get("color_positive_minimum_candidate_margin") is True
        and required.get("color_positive_minimum_full_vocab_margin") is True
    )


def _continuation_passed(candidate: Mapping[str, Any], contract: Mapping[str, Any]) -> bool:
    required = _mapping(contract.get("continuation_requires"), "continuation_requires")
    color = _mapping(candidate["color"], "candidate.color")
    mirror = _mapping(candidate["mirror"], "candidate.mirror")
    return bool(
        color["full_vocab_sides"] == required.get("color_full_vocab_sides")
        and color["full_vocab_units"] == required.get("color_full_vocab_units")
        and color["minimum_candidate_margin"] > 0.0
        and color["minimum_full_vocab_margin"] > 0.0
        and required.get("color_positive_minimum_candidate_margin") is True
        and required.get("color_positive_minimum_full_vocab_margin") is True
        and mirror["full_vocab_sides"] >= required.get("mirror_minimum_full_vocab_sides")
        and mirror["full_vocab_units"] >= required.get("mirror_minimum_full_vocab_units")
    )


def _full_teacher_passed(candidate: Mapping[str, Any], contract: Mapping[str, Any]) -> bool:
    required = _mapping(contract.get("full_teacher_gate_requires"), "full_teacher_gate_requires")
    color = _mapping(candidate["color"], "candidate.color")
    mirror = _mapping(candidate["mirror"], "candidate.mirror")
    all_minimums_positive = all(
        metrics[key] > 0.0
        for metrics in (color, mirror)
        for key in ("minimum_candidate_margin", "minimum_full_vocab_margin")
    )
    return bool(
        color["full_vocab_sides"] == required.get("color_full_vocab_sides")
        and color["full_vocab_units"] == required.get("color_full_vocab_units")
        and mirror["full_vocab_sides"] == required.get("mirror_full_vocab_sides")
        and mirror["full_vocab_units"] == required.get("mirror_full_vocab_units")
        and required.get("all_candidate_and_full_vocab_minimum_margins_positive") is True
        and all_minimums_positive
    )


def _validate_report(
    name: str,
    config_evidence: Mapping[str, Any],
    report: Mapping[str, Any],
    report_sha256: str,
    report_path: str,
) -> dict[str, Any]:
    _assert_finite_tree(report)
    contract = _mapping(config_evidence["response_contract"], "response_contract")
    namespace = config_evidence["output_namespace"]
    if report.get("output_namespace") != namespace:
        _fail(f"{name} report output namespace mismatch")
    if report.get("optimizer_steps") != 4 or report.get("epochs") != 4:
        _fail(f"{name} must contain exactly four completed optimizer updates")
    if report.get("target_epochs") != 4 or report.get("steps") != 48:
        _fail(f"{name} target epoch/microstep count mismatch")
    if report.get("gradient_accumulation") != 12 or report.get("stopped_early") is not False:
        _fail(f"{name} accumulation or stopping contract mismatch")
    if report.get("freeze_scene_adapter") is not True:
        _fail(f"{name} did not freeze the scene adapter")
    if report.get("train_global_scene_residual_only") is not True:
        _fail(f"{name} is not residual-only training")
    if report.get("question_dependent_scene_processing") is not False:
        _fail(f"{name} used question-dependent scene processing")
    if report.get("global_scene_residual_parameter_count") != 400000:
        _fail(f"{name} residual parameter count mismatch")
    if (
        report.get("lora_trainable_parameter_count") != 0
        or report.get("lora_optimizer") is not None
    ):
        _fail(f"{name} unexpectedly trained a LoRA bank")

    expected_selection = contract["expected_selection_sha256"]
    expected_membership = contract["expected_pair_membership_sha256"]
    selection = _mapping(report.get("selection"), f"{name}.selection")
    selected_train = _mapping(selection.get("train"), f"{name}.selection.train")
    if selected_train.get("selected_ids_sha256") != expected_selection:
        _fail(f"{name} selection hash mismatch")
    if selected_train.get("selected_count") != 24:
        _fail(f"{name} selected question count mismatch")
    if selection.get("training_counterfactual_pair_membership_sha256") != expected_membership:
        _fail(f"{name} nested pair membership hash mismatch")
    if report.get("training_counterfactual_pair_membership_sha256") != expected_membership:
        _fail(f"{name} pair membership hash mismatch")
    if report.get("counterfactual_pair_unit_count") != 12:
        _fail(f"{name} pair unit count mismatch")

    source = _validate_source_provenance(report.get("source_provenance"), f"{name}.source")
    nested_source = _validate_source_provenance(
        selection.get("source_provenance"), f"{name}.selection.source_provenance"
    )
    if source != nested_source:
        _fail(f"{name} top-level and selection source provenance differ")

    expected_adapter = contract["expected_source_adapter_sha256"]
    expected_metadata = contract["expected_source_metadata_sha256"]
    if report.get("initialize_expected_adapter_sha256") != expected_adapter:
        _fail(f"{name} expected source adapter hash mismatch")
    if report.get("initialize_expected_metadata_sha256") != expected_metadata:
        _fail(f"{name} expected source metadata hash mismatch")
    initialization = _mapping(report.get("initialization_provenance"), f"{name}.initialization")
    required_initialization = {
        "schema_version": 3,
        "mode": "named_lora_banks_frozen_plus_zero_output_scene_residual",
        "adapter_sha256": expected_adapter,
        "metadata_sha256": expected_metadata,
        "expected_adapter_sha256": expected_adapter,
        "expected_metadata_sha256": expected_metadata,
        "checkpoint_epoch": 7,
        "optimizer_state_loaded": False,
        "history_loaded": False,
        "all_source_lora_banks_frozen": True,
        "global_scene_residual_zero_output": True,
    }
    for key, expected in required_initialization.items():
        if initialization.get(key) != expected:
            _fail(f"{name}.initialization.{key} mismatch")

    expected_scene = contract["expected_frozen_scene_state_sha256"]
    expected_banks = {
        "inherited_v12": contract["expected_frozen_inherited_bank_sha256"],
        "extension_v13": contract["expected_frozen_extension_bank_sha256"],
    }
    expected_residual = contract["expected_initial_residual_state_sha256"]
    if report.get("frozen_scene_state_sha256") != expected_scene:
        _fail(f"{name} frozen scene hash mismatch")
    for field in ("frozen_lora_bank_state_sha256", "lora_bank_state_sha256"):
        if report.get(field) != expected_banks:
            _fail(f"{name}.{field} mismatch")
    if initialization.get("source_lora_bank_state_sha256") != expected_banks:
        _fail(f"{name} initialized frozen bank hashes mismatch")
    if report.get("global_scene_residual_initial_state_sha256") != expected_residual:
        _fail(f"{name} residual initial hash mismatch")
    if initialization.get("global_scene_residual_initial_state_sha256") != expected_residual:
        _fail(f"{name} initialization residual hash mismatch")
    residual_contract = _mapping(report.get("global_scene_residual"), f"{name}.residual")
    if residual_contract.get("expected_initial_state_sha256") != expected_residual:
        _fail(f"{name} residual contract hash mismatch")
    _sha256(report.get("global_scene_residual_state_sha256"), f"{name}.residual_state")

    equivalence = _mapping(
        report.get("global_scene_residual_zero_output_equivalence"), f"{name}.zero_equivalence"
    )
    if equivalence.get("verified") is not True:
        _fail(f"{name} lacks update-0 zero-output equivalence")
    if equivalence.get("question_dependent_scene_processing") is not False:
        _fail(f"{name} zero-output equivalence is question-dependent")
    prefixes = _mapping(equivalence.get("scene_prefixes"), f"{name}.scene_prefixes")
    if len(prefixes) != 4:
        _fail(f"{name} must attest four scene prefixes")
    prefix_hashes: dict[str, str] = {}
    for scene_id, values in sorted(prefixes.items()):
        pair = _mapping(values, f"{name}.scene_prefixes.{scene_id}")
        core = _sha256(pair.get("core_prefix_sha256"), f"{scene_id}.core_prefix_sha256")
        adapted = _sha256(pair.get("adapted_prefix_sha256"), f"{scene_id}.adapted_prefix_sha256")
        if core != adapted:
            _fail(f"{name} update-0 prefix differs for {scene_id}")
        prefix_hashes[str(scene_id)] = core

    history = _sequence(report.get("history"), f"{name}.history")
    if len(history) != 4:
        _fail(f"{name} history must contain exactly four epochs")
    epochs: list[dict[str, Any]] = []
    for expected_epoch, raw_epoch in enumerate(history, start=1):
        epoch = _mapping(raw_epoch, f"{name}.history[{expected_epoch - 1}]")
        if epoch.get("epoch") != expected_epoch:
            _fail(f"{name} history epochs must be exactly 1,2,3,4")
        gate = _mapping(epoch.get("pair_candidate_gate"), f"{name}.epoch{expected_epoch}.gate")
        by_pair = _mapping(gate.get("by_pair"), f"{name}.epoch{expected_epoch}.by_pair")
        if set(by_pair) != {COLOR_PAIR_ID, MIRROR_PAIR_ID}:
            _fail(f"{name} epoch {expected_epoch} has unexpected pair gates")
        color = _pair_epoch_metrics(by_pair[COLOR_PAIR_ID], COLOR_PAIR_ID)
        mirror = _pair_epoch_metrics(by_pair[MIRROR_PAIR_ID], MIRROR_PAIR_ID)
        epochs.append(
            {
                "arm": name,
                "epoch": expected_epoch,
                "learning_rate": config_evidence["learning_rate"],
                "color_eligible": _color_eligible(color, contract),
                "color": color,
                "mirror": mirror,
            }
        )
    if report.get("pair_candidate_gate") != history[-1].get("pair_candidate_gate"):
        _fail(f"{name} top-level pair gate is not epoch 4")
    return {
        "name": name,
        "path": report_path,
        "input_report_sha256": report_sha256,
        "learning_rate": config_evidence["learning_rate"],
        "config_hash": config_evidence["config_hash"],
        "output_namespace": namespace,
        "source_provenance": source,
        "prefix_hashes": prefix_hashes,
        "epochs": epochs,
    }


def summarize_residual_lr_response(
    configs: Mapping[str, Mapping[str, Any]],
    reports: Mapping[str, Mapping[str, Any]],
    *,
    report_paths: Mapping[str, str] | None = None,
    report_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate both arms and select one eligible epoch under the V17 contract."""

    expected_names = set(EXPECTED_ARM_SPECS)
    if set(configs) != expected_names or set(reports) != expected_names:
        _fail(
            "V17 inputs must contain exactly the lr1e4 and lr3e4 arms: "
            f"configs={sorted(configs)} reports={sorted(reports)}"
        )
    paths = dict(report_paths or {name: f"<{name}>" for name in expected_names})
    hashes = dict(
        report_sha256 or {name: _canonical_sha256(reports[name]) for name in expected_names}
    )
    if set(paths) != expected_names or set(hashes) != expected_names:
        _fail("Report paths and hashes must cover both exact V17 arms")
    for name, value in hashes.items():
        _sha256(value, f"{name}.input_report_sha256")

    config_evidence = {
        name: _validate_arm_config(name, configs[name]) for name in sorted(expected_names)
    }
    contracts = [
        {
            key: value
            for key, value in evidence["response_contract"].items()
            if key != "arm_learning_rate"
        }
        for evidence in config_evidence.values()
    ]
    if contracts[0] != contracts[1]:
        _fail("V17 arm response contracts differ beyond arm_learning_rate")
    contract = contracts[0]
    arms = [
        _validate_report(
            name,
            config_evidence[name],
            reports[name],
            hashes[name],
            paths[name],
        )
        for name in sorted(expected_names)
    ]
    if arms[0]["source_provenance"] != arms[1]["source_provenance"]:
        _fail("V17 arms do not share the exact clean source commit/tree")
    if arms[0]["prefix_hashes"] != arms[1]["prefix_hashes"]:
        _fail("V17 arms do not share identical update-0 scene prefixes")

    def mirror_ranking_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
        mirror = _mapping(candidate["mirror"], "candidate.mirror")
        values = {
            "mirror_full_vocab_units": mirror["full_vocab_units"],
            "mirror_full_vocab_sides": mirror["full_vocab_sides"],
            "mirror_candidate_units": mirror["candidate_units"],
            "mirror_candidate_sides": mirror["candidate_sides"],
            "mirror_mean_full_vocab_margin": mirror["mean_full_vocab_margin"],
            "mirror_minimum_full_vocab_margin": mirror["minimum_full_vocab_margin"],
        }
        return tuple(-float(values[field]) for field in EXPECTED_RANKING_FIELDS)

    eligible_candidates: list[dict[str, Any]] = []
    arm_representatives: list[dict[str, Any]] = []
    representative_ambiguity = False
    for arm in arms:
        eligible = [deepcopy(epoch) for epoch in arm["epochs"] if epoch["color_eligible"]]
        eligible_candidates.extend(eligible)
        eligible.sort(key=mirror_ranking_key)
        arm["eligible_epoch_count"] = len(eligible)
        arm["representative_epoch"] = None
        arm["representative_ambiguous"] = False
        arm["representative"] = None
        if not eligible:
            continue
        best_key = mirror_ranking_key(eligible[0])
        tied = [candidate for candidate in eligible if mirror_ranking_key(candidate) == best_key]
        if len(tied) != 1:
            arm["representative_ambiguous"] = True
            representative_ambiguity = True
            continue
        representative = eligible[0]
        arm["representative_epoch"] = representative["epoch"]
        arm["representative"] = deepcopy(representative)
        arm_representatives.append(representative)

    def cross_arm_ranking_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
        return (*mirror_ranking_key(candidate), float(candidate["learning_rate"]))

    ranking = sorted(arm_representatives, key=cross_arm_ranking_key)
    for index, candidate in enumerate(ranking, start=1):
        candidate["rank"] = index
        candidate["continuation_gate_passed"] = _continuation_passed(candidate, contract)
        candidate["full_teacher_gate_passed"] = _full_teacher_passed(candidate, contract)
    selected: dict[str, Any] | None = None
    selection_ambiguous = representative_ambiguity
    if ranking and not selection_ambiguous:
        best_key = cross_arm_ranking_key(ranking[0])
        tied = [candidate for candidate in ranking if cross_arm_ranking_key(candidate) == best_key]
        selection_ambiguous = len(tied) != 1
        if not selection_ambiguous:
            selected = ranking[0]

    continuation_authorized = bool(selected is not None and selected["continuation_gate_passed"])
    full_teacher_gate_passed = bool(selected is not None and selected["full_teacher_gate_passed"])
    greedy_authorized = bool(
        full_teacher_gate_passed
        and contract.get("greedy_audit_only_after_full_teacher_gate") is True
    )
    if greedy_authorized and not full_teacher_gate_passed:  # pragma: no cover - defensive
        raise AssertionError("Greedy audit cannot be authorized without full teacher gate")

    arm_rows = []
    for arm in arms:
        arm_rows.append(
            {
                key: deepcopy(arm[key])
                for key in (
                    "name",
                    "path",
                    "input_report_sha256",
                    "learning_rate",
                    "config_hash",
                    "output_namespace",
                    "source_provenance",
                    "prefix_hashes",
                    "epochs",
                    "eligible_epoch_count",
                    "representative_epoch",
                    "representative_ambiguous",
                    "representative",
                )
            }
        )
    return {
        "schema_version": 1,
        "response_type": "strict_v17_global_scene_residual_lr_response",
        "report_only": True,
        "question_dependent_scene_processing": False,
        "expected_arm_specs": deepcopy(EXPECTED_ARM_SPECS),
        "response_contract": deepcopy(contract),
        "response_contract_sha256": _canonical_sha256(contract),
        "source_provenance": deepcopy(arms[0]["source_provenance"]),
        "arm_count": len(arms),
        "arms": arm_rows,
        "eligible_epoch_count": len(eligible_candidates),
        "arm_representative_count": len(arm_representatives),
        "ranking": ranking,
        "selection_ambiguous": selection_ambiguous,
        "selected_arm": None if selected is None else selected["arm"],
        "selected_epoch": None if selected is None else selected["epoch"],
        "selected_learning_rate": None if selected is None else selected["learning_rate"],
        "selected_report": (
            None
            if selected is None
            else next(arm["path"] for arm in arms if arm["name"] == selected["arm"])
        ),
        "selected_report_sha256": (
            None
            if selected is None
            else next(arm["input_report_sha256"] for arm in arms if arm["name"] == selected["arm"])
        ),
        "continuation_authorized": continuation_authorized,
        "conditional_max_optimizer_updates": contract["conditional_max_updates"],
        "full_teacher_gate_passed": full_teacher_gate_passed,
        "greedy_audit_authorized": greedy_authorized,
        "decision": (
            "ambiguous_no_action"
            if selection_ambiguous
            else "no_color_eligible_epoch"
            if selected is None
            else "full_teacher_gate_passed_greedy_audit_allowed"
            if greedy_authorized
            else "continue_selected_arm_no_greedy_audit"
            if continuation_authorized
            else "screen_failed_no_extension_no_greedy_audit"
        ),
    }


def _load_json_strict(path: Path) -> tuple[Mapping[str, Any], str]:
    raw = path.read_bytes()

    def reject_constant(value: str) -> None:
        _fail(f"JSON constant {value} is forbidden")

    value = json.loads(raw.decode("utf-8"), parse_constant=reject_constant)
    return _mapping(value, str(path)), hashlib.sha256(raw).hexdigest()


def _parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or name not in EXPECTED_ARM_SPECS or not path:
        raise argparse.ArgumentTypeError("input must be lr1e4=PATH or lr3e4=PATH")
    return name, Path(path)


def write_response(summary: Mapping[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", type=_parse_named_path, required=True)
    parser.add_argument("--report", action="append", type=_parse_named_path, required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    config_paths = dict(args.config)
    report_paths = dict(args.report)
    if len(config_paths) != len(args.config) or len(report_paths) != len(args.report):
        parser.error("duplicate V17 arm binding")
    configs = {name: load_config(path) for name, path in config_paths.items()}
    loaded = {name: _load_json_strict(path) for name, path in report_paths.items()}
    reports = {name: value for name, (value, _sha) in loaded.items()}
    hashes = {name: sha for name, (_value, sha) in loaded.items()}
    summary = summarize_residual_lr_response(
        configs,
        reports,
        report_paths={name: str(path) for name, path in report_paths.items()},
        report_sha256=hashes,
    )
    destination = write_response(summary, args.output)
    print(
        json.dumps(
            {
                "output": str(destination),
                "selected_arm": summary["selected_arm"],
                "selected_epoch": summary["selected_epoch"],
                "continuation_authorized": summary["continuation_authorized"],
                "greedy_audit_authorized": summary["greedy_audit_authorized"],
                "decision": summary["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the public CLI
    raise SystemExit(main())


__all__ = [
    "EXPECTED_ARM_SPECS",
    "EXPECTED_RANKING_FIELDS",
    "ResidualLRResponseViolation",
    "main",
    "summarize_residual_lr_response",
    "write_response",
]
