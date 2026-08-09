"""Strictly validate and rank bounded learning-rate sweep training reports.

The selector is intentionally report-only: it never loads a model checkpoint or
an oracle artifact.  An arm can be ranked only when its final, post-update
teacher-forced gate preserves the established color behavior and contains the
mirror metrics used by the predeclared selection rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

COLOR_PAIR_ID = "pair_000001"
MIRROR_PAIR_ID = "pair_000003"
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_OBJECT_ID = re.compile(r"[0-9a-f]{40,64}")
_ARM_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class ExpectedSweepContract:
    """Pinned scientific protocol shared by every bounded V14 LR arm."""

    epochs: int = 4
    steps: int = 48
    optimizer_steps: int = 4
    gradient_accumulation: int = 12
    required_pair_unit_count: int = 6
    required_pair_side_count: int = 12
    frozen_scene_state_sha256: str = (
        "690bd890bfda024dbb5c7d3c68087b8113bc3b8ee81dd6143c7eb2a884e7245b"
    )
    frozen_bank_name: str = "inherited_v12"
    frozen_bank_state_sha256: str = (
        "dec768bed654c8c4e16da0318857543ad54d8f5f68f4d24a9a87cd19ec706594"
    )
    source_adapter_sha256: str = "a4c85c14a214e4e594992e489a784cb4bacb64d3dfda519ad3da18b1595d9f22"
    source_metadata_sha256: str = "f097c6477546460440e77a3d225afb55818cb13abf9cbb4a90500f75a879b0f5"
    selection_sha256: str = "7f0714e3151c9ddb57c1da95a457820a833e490c070881a88a9fee4a9168f933"
    pair_membership_sha256: str = "99ee448c23fb71b7269a353a54b2156ac55701847af170597dcc351af15cbcbe"
    trainable_initial_state_sha256: str = (
        "b4ec0518e4759dda33fc93c9c1d4c76f52f1024fd5b8b1667ad1b4ef5da198af"
    )
    v13_candidate_hinge: float = 2.0455729961395264
    v13_full_vocab_hinge: float = 19.23177146911621
    v12_candidate_minimum_margin: float = -4.6875
    v12_full_vocab_minimum_margin: float = -25.75


EXPECTED_SWEEP_CONTRACT = ExpectedSweepContract()

RANKING_FIELDS = (
    "full_vocab_unit_accuracy",
    "full_vocab_side_accuracy",
    "candidate_changed_unit_accuracy",
    "candidate_side_accuracy",
    "full_vocab_mean_margin",
    "full_vocab_minimum_margin",
    "candidate_mean_margin",
    "candidate_minimum_margin",
    "learning_rate",
)


class ReportViolation(ValueError):
    """A deterministic report-integrity or eligibility violation."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _violation(code: str, message: str) -> None:
    raise ReportViolation(code, message)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _violation("missing_or_invalid_field", f"{field} must be an object")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _violation("missing_or_invalid_field", f"{field} must be a positive integer")
    return value


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _violation("missing_or_invalid_field", f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _violation("non_finite_value", f"{field} contains NaN or infinity")
    return result


def _accuracy(value: Any, field: str) -> float:
    result = _finite_number(value, field)
    if not 0.0 <= result <= 1.0:
        _violation("missing_or_invalid_field", f"{field} must be in [0, 1]")
    return result


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        _violation("invalid_hash", f"{field} must be a lowercase SHA-256 digest")
    return value


def _assert_finite_tree(value: Any, field: str = "report") -> None:
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            _assert_finite_tree(value[key], f"{field}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _assert_finite_tree(item, f"{field}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        _violation("non_finite_value", f"{field} contains NaN or infinity")


def _validate_source_provenance(value: Any, field: str) -> Mapping[str, Any]:
    provenance = _mapping(value, field)
    if provenance.get("schema_version") != 1:
        _violation("invalid_provenance", f"{field}.schema_version must be 1")
    if provenance.get("scope") != "repository_excluding_generated_artifacts_v1":
        _violation("invalid_provenance", f"{field}.scope is not the training source scope")
    if provenance.get("available") is not True or provenance.get("is_clean") is not True:
        _violation("invalid_provenance", f"{field} must record clean, available source")
    for key in ("head_commit", "head_tree"):
        value = provenance.get(key)
        if not isinstance(value, str) or _GIT_OBJECT_ID.fullmatch(value) is None:
            _violation("invalid_provenance", f"{field}.{key} is not a Git object ID")
    diff_hash = _sha256(provenance.get("tracked_diff_sha256"), f"{field}.tracked_diff_sha256")
    if diff_hash != _EMPTY_SHA256:
        _violation("invalid_provenance", f"{field} records a non-empty tracked diff")
    return provenance


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_frozen_state(report: Mapping[str, Any]) -> dict[str, Any]:
    if report.get("freeze_scene_adapter") is not True:
        _violation("frozen_state_violation", "freeze_scene_adapter must be true")
    scene_hash = _sha256(report.get("frozen_scene_state_sha256"), "frozen_scene_state_sha256")
    frozen_banks = _mapping(
        report.get("frozen_lora_bank_state_sha256"), "frozen_lora_bank_state_sha256"
    )
    final_banks = _mapping(report.get("lora_bank_state_sha256"), "lora_bank_state_sha256")
    if not frozen_banks:
        _violation("frozen_state_violation", "at least one frozen LoRA bank is required")
    expected_frozen = {
        EXPECTED_SWEEP_CONTRACT.frozen_bank_name: (EXPECTED_SWEEP_CONTRACT.frozen_bank_state_sha256)
    }
    if dict(frozen_banks) != expected_frozen:
        _violation("frozen_state_violation", "frozen LoRA banks do not match V14 protocol")
    if scene_hash != EXPECTED_SWEEP_CONTRACT.frozen_scene_state_sha256:
        _violation("frozen_state_violation", "frozen scene hash does not match V14 protocol")
    normalized_frozen: dict[str, str] = {}
    for name in sorted(frozen_banks):
        if not isinstance(name, str) or not name:
            _violation("frozen_state_violation", "frozen LoRA bank names must be non-empty")
        expected = _sha256(frozen_banks[name], f"frozen_lora_bank_state_sha256.{name}")
        actual = _sha256(final_banks.get(name), f"lora_bank_state_sha256.{name}")
        if actual != expected:
            _violation("frozen_state_violation", f"frozen LoRA bank {name!r} changed")
        normalized_frozen[name] = expected
    return {"scene_state_sha256": scene_hash, "lora_bank_state_sha256": normalized_frozen}


def _validate_initialization_provenance(report: Mapping[str, Any]) -> dict[str, Any]:
    initialization = _mapping(report.get("initialization_provenance"), "initialization_provenance")
    if initialization.get("schema_version") != 2:
        _violation("invalid_provenance", "initialization_provenance.schema_version must be 2")
    if initialization.get("mode") != "legacy_lora_into_frozen_named_bank":
        _violation("invalid_provenance", "sweep arms must restart from a legacy frozen bank")
    adapter_hash = _sha256(initialization.get("adapter_sha256"), "initialization.adapter_sha256")
    metadata_hash = _sha256(initialization.get("metadata_sha256"), "initialization.metadata_sha256")
    expected_adapter_hash = _sha256(
        initialization.get("expected_adapter_sha256"),
        "initialization.expected_adapter_sha256",
    )
    expected_metadata_hash = _sha256(
        initialization.get("expected_metadata_sha256"),
        "initialization.expected_metadata_sha256",
    )
    if adapter_hash != expected_adapter_hash or metadata_hash != expected_metadata_hash:
        _violation("invalid_provenance", "loaded initialization hashes do not match their pins")
    if adapter_hash != EXPECTED_SWEEP_CONTRACT.source_adapter_sha256:
        _violation("invalid_provenance", "initialization adapter is not the pinned V12 source")
    if metadata_hash != EXPECTED_SWEEP_CONTRACT.source_metadata_sha256:
        _violation("invalid_provenance", "initialization metadata is not the pinned V12 source")
    if (
        _sha256(
            report.get("initialize_expected_adapter_sha256"),
            "initialize_expected_adapter_sha256",
        )
        != adapter_hash
    ):
        _violation("invalid_provenance", "top-level initialization adapter pin disagrees")
    if (
        _sha256(
            report.get("initialize_expected_metadata_sha256"),
            "initialize_expected_metadata_sha256",
        )
        != metadata_hash
    ):
        _violation("invalid_provenance", "top-level initialization metadata pin disagrees")
    if initialization.get("optimizer_state_loaded") is not False:
        _violation("invalid_provenance", "LR sweep arms must not inherit optimizer state")
    if initialization.get("history_loaded") is not False:
        _violation("invalid_provenance", "LR sweep arms must not inherit training history")
    if initialization.get("new_trainable_banks_zero_output") is not True:
        _violation("invalid_provenance", "new trainable banks must start with zero output")
    checkpoint_source = _validate_source_provenance(
        initialization.get("checkpoint_source_provenance"),
        "initialization.checkpoint_source_provenance",
    )
    target_bank = initialization.get("target_bank")
    if not isinstance(target_bank, str) or not target_bank:
        _violation("invalid_provenance", "initialization.target_bank must be named")
    target_hash = _sha256(
        initialization.get("target_bank_state_sha256"),
        "initialization.target_bank_state_sha256",
    )
    frozen_banks = _mapping(
        report.get("frozen_lora_bank_state_sha256"), "frozen_lora_bank_state_sha256"
    )
    if frozen_banks.get(target_bank) != target_hash:
        _violation("invalid_provenance", "initialization target bank is not the frozen bank")
    checkpoint = initialization.get("checkpoint")
    namespace = initialization.get("checkpoint_output_namespace")
    epoch = initialization.get("checkpoint_epoch")
    if not isinstance(checkpoint, str) or not checkpoint:
        _violation("invalid_provenance", "initialization checkpoint must be recorded")
    if not isinstance(namespace, str) or not namespace:
        _violation("invalid_provenance", "initialization checkpoint namespace must be recorded")
    _positive_int(epoch, "initialization.checkpoint_epoch")
    return {
        "mode": initialization["mode"],
        "checkpoint": checkpoint,
        "checkpoint_epoch": epoch,
        "checkpoint_output_namespace": namespace,
        "adapter_sha256": adapter_hash,
        "metadata_sha256": metadata_hash,
        "checkpoint_source_provenance": dict(checkpoint_source),
        "target_bank": target_bank,
        "target_bank_state_sha256": target_hash,
    }


def _validate_selection_provenance(report: Mapping[str, Any]) -> dict[str, Any]:
    selection = _mapping(report.get("selection"), "selection")
    train = _mapping(selection.get("train"), "selection.train")
    selected_hash = _sha256(train.get("selected_ids_sha256"), "selection.train.selected_ids_sha256")
    membership_hash = _sha256(
        report.get("training_counterfactual_pair_membership_sha256"),
        "training_counterfactual_pair_membership_sha256",
    )
    if selection.get("training_counterfactual_pair_membership_sha256") != membership_hash:
        _violation("invalid_provenance", "selection pair-membership hash disagrees")
    if selected_hash != EXPECTED_SWEEP_CONTRACT.selection_sha256:
        _violation("invalid_provenance", "selected records do not match the V14 protocol")
    if membership_hash != EXPECTED_SWEEP_CONTRACT.pair_membership_sha256:
        _violation("invalid_provenance", "pair membership does not match the V14 protocol")
    embedded_source = _validate_source_provenance(
        selection.get("source_provenance"), "selection.source_provenance"
    )
    if dict(embedded_source) != dict(report["source_provenance"]):
        _violation("invalid_provenance", "selection source provenance disagrees")
    if selection.get("initialize_expected_adapter_sha256") != report.get(
        "initialize_expected_adapter_sha256"
    ):
        _violation("invalid_provenance", "selection adapter initialization pin disagrees")
    if selection.get("initialize_expected_metadata_sha256") != report.get(
        "initialize_expected_metadata_sha256"
    ):
        _violation("invalid_provenance", "selection metadata initialization pin disagrees")
    train_scene_ids = report.get("train_scene_ids")
    test_scene_ids = report.get("test_scene_ids")
    if selection.get("train_scene_ids") != train_scene_ids:
        _violation("invalid_provenance", "selection train scenes disagree")
    if selection.get("test_scene_ids") != test_scene_ids:
        _violation("invalid_provenance", "selection test scenes disagree")
    return {
        "strategy": selection.get("strategy"),
        "selected_ids_sha256": selected_hash,
        "pair_membership_sha256": membership_hash,
        "train_scene_ids": deepcopy(train_scene_ids),
        "test_scene_ids": deepcopy(test_scene_ids),
    }


def _validate_lora_architecture(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    lora = _mapping(report.get("lora"), "lora")
    if lora.get("schema_version") != 2 or lora.get("enabled") is not True:
        _violation("invalid_provenance", "named-bank LoRA schema 2 must be enabled")
    raw_banks = lora.get("banks")
    if not isinstance(raw_banks, list) or not raw_banks:
        _violation("invalid_provenance", "lora.banks must be a non-empty list")
    final_hashes = _mapping(report.get("lora_bank_state_sha256"), "lora_bank_state_sha256")
    architecture: list[dict[str, Any]] = []
    trainable_banks: list[tuple[str, str]] = []
    names: set[str] = set()
    for index, raw_bank in enumerate(raw_banks):
        bank = dict(_mapping(raw_bank, f"lora.banks[{index}]"))
        name = bank.get("name")
        if not isinstance(name, str) or not name or name in names:
            _violation("invalid_provenance", "LoRA bank names must be unique and non-empty")
        names.add(name)
        _sha256(final_hashes.get(name), f"lora_bank_state_sha256.{name}")
        initial_state_hash = _sha256(
            bank.get("expected_initial_state_sha256"),
            f"lora.banks[{index}].expected_initial_state_sha256",
        )
        if bank.get("trainable") is True:
            trainable_banks.append((name, initial_state_hash))
        elif bank.get("trainable") is not False:
            _violation("invalid_provenance", f"lora.banks[{index}].trainable must be boolean")
        elif (
            name != EXPECTED_SWEEP_CONTRACT.frozen_bank_name
            or initial_state_hash != EXPECTED_SWEEP_CONTRACT.frozen_bank_state_sha256
        ):
            _violation("invalid_provenance", "frozen bank architecture does not match V14 pin")
        # The sweep intentionally changes only optimizer response.  Runtime-added
        # optimizer fields are excluded from the cross-arm architecture contract.
        bank.pop("learning_rate", None)
        bank.pop("weight_decay", None)
        architecture.append(bank)
    if set(final_hashes) != names:
        _violation("invalid_provenance", "final LoRA state has missing or unconfigured banks")
    if len(trainable_banks) != 1:
        _violation("invalid_provenance", "V14 protocol requires exactly one trainable LoRA bank")
    trainable_name, initial_hash = trainable_banks[0]
    if initial_hash != EXPECTED_SWEEP_CONTRACT.trainable_initial_state_sha256:
        _violation("invalid_provenance", "trainable bank initialization is not the V14 pin")
    if final_hashes[trainable_name] == initial_hash:
        _violation("frozen_state_violation", "trainable extension state did not change")
    return architecture


def _extract_learning_rate(report: Mapping[str, Any]) -> float:
    optimizer = _mapping(report.get("lora_optimizer"), "lora_optimizer")
    learning_rate = _finite_number(optimizer.get("learning_rate"), "lora_optimizer.learning_rate")
    if learning_rate <= 0:
        _violation("missing_or_invalid_field", "learning rate must be positive")
    lora = _mapping(report.get("lora"), "lora")
    for index, raw_bank in enumerate(lora.get("banks", [])):
        bank = _mapping(raw_bank, f"lora.banks[{index}]")
        if bank.get("trainable") is True and "learning_rate" in bank:
            bank_lr = _finite_number(bank["learning_rate"], f"lora.banks[{index}].learning_rate")
            if bank_lr != learning_rate:
                _violation("invalid_provenance", "trainable bank learning rate disagrees")
    return learning_rate


def _extract_final_pair_gates(report: Mapping[str, Any]) -> Mapping[str, Any]:
    epochs = _positive_int(report.get("epochs"), "epochs")
    steps = _positive_int(report.get("steps"), "steps")
    optimizer_steps = _positive_int(report.get("optimizer_steps"), "optimizer_steps")
    gradient_accumulation = _positive_int(
        report.get("gradient_accumulation"), "gradient_accumulation"
    )
    expected = EXPECTED_SWEEP_CONTRACT
    if (
        epochs != expected.epochs
        or report.get("target_epochs") != expected.epochs
        or steps != expected.steps
        or optimizer_steps != expected.optimizer_steps
        or gradient_accumulation != expected.gradient_accumulation
    ):
        _violation("protocol_violation", "training update counts do not match V14 protocol")
    history = report.get("history")
    if not isinstance(history, list) or not history:
        _violation("missing_final_pair_gate", "history must contain a final post-update epoch")
    final = _mapping(history[-1], "history[-1]")
    if final.get("epoch") != epochs:
        _violation("missing_final_pair_gate", "history[-1] is not the reported final epoch")
    if "pair_candidate_gate" not in final:
        _violation("missing_final_pair_gate", "history[-1] lacks a post-update pair gate")
    final_gate = _mapping(final["pair_candidate_gate"], "history[-1].pair_candidate_gate")
    if report.get("pair_candidate_gate") != final_gate:
        _violation("missing_final_pair_gate", "top-level gate is not the final history gate")
    by_pair = _mapping(final_gate.get("by_pair"), "history[-1].pair_candidate_gate.by_pair")
    for pair_id in (COLOR_PAIR_ID, MIRROR_PAIR_ID):
        pair = _mapping(by_pair.get(pair_id), f"final pair gate {pair_id}")
        if pair.get("ranking_mode") != "candidate_logit":
            _violation("missing_final_pair_gate", f"{pair_id} is not a candidate-logit gate")
        if pair.get("same_next_token_distribution") is not True:
            _violation("missing_final_pair_gate", f"{pair_id} lacks same-distribution scoring")
        if pair.get("shared_candidate_tokens_excluded") is not True:
            _violation("missing_final_pair_gate", f"{pair_id} includes shared candidate tokens")
        if pair.get("free_generation_evaluated") is not False:
            _violation("missing_final_pair_gate", f"{pair_id} has an invalid gate type")
        if pair.get("first_answer_token_full_vocab_evaluated") is not True:
            _violation("missing_final_pair_gate", f"{pair_id} lacks full-vocabulary scoring")
        unit_count = _positive_int(pair.get("unit_count"), f"{pair_id}.unit_count")
        side_count = _positive_int(pair.get("side_count"), f"{pair_id}.side_count")
        if (
            unit_count != EXPECTED_SWEEP_CONTRACT.required_pair_unit_count
            or side_count != EXPECTED_SWEEP_CONTRACT.required_pair_side_count
        ):
            _violation("protocol_violation", f"{pair_id} gate counts do not match V14 protocol")
    return by_pair


def _pair_metrics(pair: Mapping[str, Any], pair_id: str) -> dict[str, float]:
    return {
        "full_vocab_unit_accuracy": _accuracy(
            pair.get("first_answer_token_top1_unit_accuracy"),
            f"{pair_id}.first_answer_token_top1_unit_accuracy",
        ),
        "full_vocab_side_accuracy": _accuracy(
            pair.get("first_answer_token_top1_accuracy"),
            f"{pair_id}.first_answer_token_top1_accuracy",
        ),
        "candidate_changed_unit_accuracy": _accuracy(
            pair.get("changed_unit_accuracy"), f"{pair_id}.changed_unit_accuracy"
        ),
        "candidate_side_accuracy": _accuracy(pair.get("side_accuracy"), f"{pair_id}.side_accuracy"),
        "full_vocab_mean_margin": _finite_number(
            pair.get("mean_first_answer_token_target_vs_best_other_logit_margin"),
            f"{pair_id}.mean_first_answer_token_target_vs_best_other_logit_margin",
        ),
        "full_vocab_minimum_margin": _finite_number(
            pair.get("minimum_first_answer_token_target_vs_best_other_logit_margin"),
            f"{pair_id}.minimum_first_answer_token_target_vs_best_other_logit_margin",
        ),
        "candidate_mean_margin": _finite_number(
            pair.get("mean_own_vs_alternate_candidate_logit_margin"),
            f"{pair_id}.mean_own_vs_alternate_candidate_logit_margin",
        ),
        "candidate_minimum_margin": _finite_number(
            pair.get("minimum_own_vs_alternate_candidate_logit_margin"),
            f"{pair_id}.minimum_own_vs_alternate_candidate_logit_margin",
        ),
        "candidate_hinge": _finite_number(
            pair.get("ranking_hinge_at_configured_margin"),
            f"{pair_id}.ranking_hinge_at_configured_margin",
        ),
        "full_vocab_hinge": _finite_number(
            pair.get("first_answer_token_target_vs_best_other_hinge"),
            f"{pair_id}.first_answer_token_target_vs_best_other_hinge",
        ),
    }


def _validate_report(name: str, path: str, report: Any) -> dict[str, Any]:
    payload = _mapping(report, "report")
    _assert_finite_tree(payload)
    source = _validate_source_provenance(payload.get("source_provenance"), "source_provenance")
    frozen = _validate_frozen_state(payload)
    initialization = _validate_initialization_provenance(payload)
    selection = _validate_selection_provenance(payload)
    lora_architecture = _validate_lora_architecture(payload)
    learning_rate = _extract_learning_rate(payload)
    pairs = _extract_final_pair_gates(payload)
    color = _pair_metrics(_mapping(pairs[COLOR_PAIR_ID], COLOR_PAIR_ID), COLOR_PAIR_ID)
    mirror = _pair_metrics(_mapping(pairs[MIRROR_PAIR_ID], MIRROR_PAIR_ID), MIRROR_PAIR_ID)
    contract = {
        "source_provenance": dict(source),
        "frozen_state": frozen,
        "initialization_provenance": initialization,
        "selection_provenance": selection,
        "lora_architecture": lora_architecture,
        "gradient_accumulation": payload.get("gradient_accumulation"),
        "target_epochs": payload.get("target_epochs"),
        "train_scene_ids": deepcopy(payload.get("train_scene_ids")),
        "test_scene_ids": deepcopy(payload.get("test_scene_ids")),
    }
    color_integrity_passed = (
        all(
            color[key] == 1.0
            for key in (
                "full_vocab_unit_accuracy",
                "full_vocab_side_accuracy",
                "candidate_changed_unit_accuracy",
                "candidate_side_accuracy",
            )
        )
        and color["full_vocab_minimum_margin"] > 0.0
        and color["candidate_minimum_margin"] > 0.0
    )
    result: dict[str, Any] = {
        "name": name,
        "path": path,
        "eligible": color_integrity_passed,
        "rejection_reasons": ([] if color_integrity_passed else ["color_integrity_failed"]),
        "learning_rate": learning_rate,
        "final_epoch": payload["epochs"],
        "optimizer_steps": payload["optimizer_steps"],
        "frozen_state": frozen,
        "provenance_contract_sha256": _canonical_sha256(contract),
        "color_integrity": {"passed": color_integrity_passed, **color},
        "mirror_metrics": mirror,
    }
    return result


def _load_json_strict(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        _violation("non_finite_value", f"JSON constant {value} is forbidden")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _rejected_arm(name: str, path: str, violation: ReportViolation) -> dict[str, Any]:
    return {
        "name": name,
        "path": path,
        "eligible": False,
        "rejection_reasons": [violation.code],
        "rejection_detail": str(violation),
        "learning_rate": None,
        "final_epoch": None,
        "optimizer_steps": None,
        "frozen_state": None,
        "provenance_contract_sha256": None,
        "color_integrity": None,
        "mirror_metrics": None,
    }


def summarize_lr_sweep(named_reports: Mapping[str, str | Path]) -> dict[str, Any]:
    """Load, validate, and deterministically rank named training reports."""

    if not named_reports:
        raise ValueError("At least one named report is required")
    arms: list[dict[str, Any]] = []
    for name in sorted(named_reports):
        if not isinstance(name, str) or _ARM_NAME.fullmatch(name) is None:
            raise ValueError(f"Invalid arm name: {name!r}")
        raw_path = Path(named_reports[name])
        path = str(raw_path)
        try:
            report = _load_json_strict(raw_path)
            arm = _validate_report(name, path, report)
        except ReportViolation as exc:
            arm = _rejected_arm(name, path, exc)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            arm = _rejected_arm(
                name,
                path,
                ReportViolation("unreadable_report", f"cannot read valid JSON: {exc}"),
            )
        arms.append(arm)

    # Cross-arm reproducibility is part of eligibility.  A unique largest
    # contract group is the deterministic reference; no arbitrary arm ordering
    # decides a tied provenance dispute.
    contract_counts = Counter(
        arm["provenance_contract_sha256"]
        for arm in arms
        if arm["provenance_contract_sha256"] is not None
    )
    canonical_contract: str | None = None
    if contract_counts:
        largest = max(contract_counts.values())
        leaders = sorted(key for key, count in contract_counts.items() if count == largest)
        if len(leaders) == 1:
            canonical_contract = leaders[0]
    for arm in arms:
        contract = arm["provenance_contract_sha256"]
        if contract is not None and contract != canonical_contract:
            arm["eligible"] = False
            if "cross_arm_provenance_mismatch" not in arm["rejection_reasons"]:
                arm["rejection_reasons"].append("cross_arm_provenance_mismatch")

    eligible = [arm for arm in arms if arm["eligible"]]

    def sort_key(arm: Mapping[str, Any]) -> tuple[Any, ...]:
        metrics = _mapping(arm["mirror_metrics"], "mirror_metrics")
        return (
            *(-float(metrics[field]) for field in RANKING_FIELDS[:-1]),
            float(arm["learning_rate"]),
            str(arm["name"]),
        )

    ranking = sorted(eligible, key=sort_key)
    ranked_rows = []
    for index, arm in enumerate(ranking, start=1):
        ranked_rows.append(
            {
                "rank": index,
                "name": arm["name"],
                "path": arm["path"],
                "learning_rate": arm["learning_rate"],
                **arm["mirror_metrics"],
            }
        )
    expected = EXPECTED_SWEEP_CONTRACT
    for row in ranked_rows:
        extension_checks = {
            "candidate_hinge_better_than_v13": (
                row["candidate_hinge"] < expected.v13_candidate_hinge
            ),
            "full_vocab_hinge_better_than_v13": (
                row["full_vocab_hinge"] < expected.v13_full_vocab_hinge
            ),
            "candidate_minimum_not_worse_than_v12": (
                row["candidate_minimum_margin"] >= expected.v12_candidate_minimum_margin
            ),
            "full_vocab_minimum_not_worse_than_v12": (
                row["full_vocab_minimum_margin"] >= expected.v12_full_vocab_minimum_margin
            ),
        }
        row["extension_checks"] = extension_checks
        row["extension_qualified"] = all(extension_checks.values())
    ranked_winner = ranked_rows[0] if ranked_rows else None
    extension_candidates = [row for row in ranked_rows if row["extension_qualified"]]
    winner = extension_candidates[0] if extension_candidates else None
    return {
        "schema_version": 1,
        "expected_sweep_contract": asdict(EXPECTED_SWEEP_CONTRACT),
        "selection_rule": {
            "required_final_pair_ids": [COLOR_PAIR_ID, MIRROR_PAIR_ID],
            "color_integrity_required": {
                "full_vocab_unit_accuracy": 1.0,
                "full_vocab_side_accuracy": 1.0,
                "candidate_changed_unit_accuracy": 1.0,
                "candidate_side_accuracy": 1.0,
                "full_vocab_minimum_margin": "> 0",
                "candidate_minimum_margin": "> 0",
            },
            "ordered_ranking": [
                {
                    "field": field,
                    "direction": "ascending" if field == "learning_rate" else "descending",
                }
                for field in RANKING_FIELDS
            ],
            "deterministic_final_tiebreak": {"field": "name", "direction": "ascending"},
            "extension_requires": {
                "candidate_hinge": f"< {expected.v13_candidate_hinge}",
                "full_vocab_hinge": f"< {expected.v13_full_vocab_hinge}",
                "candidate_minimum_margin": f">= {expected.v12_candidate_minimum_margin}",
                "full_vocab_minimum_margin": f">= {expected.v12_full_vocab_minimum_margin}",
            },
        },
        "report_count": len(arms),
        "eligible_count": len(eligible),
        "rejected_count": len(arms) - len(eligible),
        "canonical_provenance_contract_sha256": canonical_contract,
        "provenance_contract_groups": dict(sorted(contract_counts.items())),
        "arms": arms,
        "ranking": ranked_rows,
        "ranked_winner": None if ranked_winner is None else ranked_winner["name"],
        "extension_qualified_count": len(extension_candidates),
        "selected_arm": None if winner is None else winner["name"],
        "selected_report": None if winner is None else winner["path"],
    }


def write_summary(summary: Mapping[str, Any], output: str | Path) -> None:
    """Atomically write a strict JSON sweep summary."""

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def parse_named_report(value: str) -> tuple[str, Path]:
    """Parse one ``NAME=PATH`` command-line report binding."""

    name, separator, raw_path = value.partition("=")
    if not separator or _ARM_NAME.fullmatch(name) is None or not raw_path:
        raise argparse.ArgumentTypeError("reports must use NAME=PATH with a safe non-empty name")
    return name, Path(raw_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        type=parse_named_report,
        metavar="NAME=PATH",
        help="named training report; repeat once per sweep arm",
    )
    parser.add_argument("--output", required=True, type=Path, help="machine-readable JSON output")
    args = parser.parse_args(argv)
    reports: dict[str, Path] = {}
    for name, path in args.report:
        if name in reports:
            parser.error(f"duplicate report name: {name}")
        reports[name] = path
    summary = summarize_lr_sweep(reports)
    write_summary(summary, args.output)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "ranked_winner",
                    "selected_arm",
                    "extension_qualified_count",
                    "eligible_count",
                    "rejected_count",
                )
            },
            sort_keys=True,
        )
    )
    return 0 if summary["selected_arm"] is not None else 2


if __name__ == "__main__":  # pragma: no cover - exercised through the public CLI entry function
    raise SystemExit(main())


__all__ = [
    "COLOR_PAIR_ID",
    "EXPECTED_SWEEP_CONTRACT",
    "MIRROR_PAIR_ID",
    "RANKING_FIELDS",
    "ExpectedSweepContract",
    "main",
    "parse_named_report",
    "summarize_lr_sweep",
    "write_summary",
]
