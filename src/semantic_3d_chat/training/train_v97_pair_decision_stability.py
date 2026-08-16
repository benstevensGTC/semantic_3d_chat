"""Train V97's fixed V96-continuation repair under the sealed data boundary."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.chat.v94_strict_multiscene_runtime import (
    EXPECTED_BANKS as V94_BANKS,
)
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import (
    atomic_create_json_v85,
    canonical_sha256_v85,
    sha256_file_v85,
)
from semantic_3d_chat.evaluation.v95_strict_causal_successor_preflight import (
    FRESH_BANK_NAME as V95_BANK_NAME,
)
from semantic_3d_chat.evaluation.v95_strict_causal_successor_preflight import (
    load_config_v95,
)
from semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight import (
    balanced_class_weights_v96,
    family_weights_v96,
    load_scene_memories_v96,
    load_training_rows_v96,
    pair_units_v96,
)
from semantic_3d_chat.evaluation.v97_pair_decision_stability_preflight import (
    CONFIG,
    EXPECTED_CHANGED_PAIR_STEPS,
    EXPECTED_FROZEN_BANK_COUNT,
    EXPECTED_FROZEN_PARAMETER_COUNT,
    EXPECTED_INITIAL_STATE_SHA256,
    EXPECTED_INVARIANT_PAIR_STEPS,
    EXPECTED_MICRO_STEPS,
    EXPECTED_OPTIMIZER_UPDATES,
    EXPECTED_RETENTION_STEPS,
    EXPECTED_TOTAL_ADAPTER_PARAMETER_COUNT,
    EXPECTED_TOTAL_NLL_FORWARDS,
    FRESH_BANK_NAME,
    FRESH_PARAMETER_COUNT,
    TARGET_MODULES,
    TrainingStepV97,
    assert_deferred_final_absent_v97,
    authenticate_cpu_preflight_v97,
    authenticate_parent_v96_v97,
    authenticate_training_sources_v97,
    forbidden_training_roots_v97,
    invariant_subset_v97,
    load_config_v97,
    training_schedule_v97,
)
from semantic_3d_chat.language.local_lm import load_local_language_model
from semantic_3d_chat.language.lora import (
    LoRABankCollection,
    LoRABankSettings,
    LoRABanksSettings,
    LoRASettings,
    install_lora_banks,
    tensor_state_sha256,
)
from semantic_3d_chat.training.train_v84_strict_bridge import _prepared_v84
from semantic_3d_chat.training.train_v95_strict_causal_successor import (
    _load_v85_banks_v95,
    _load_v94_bank_v95,
    combined_lora_settings_v95,
    load_fixed_final_bridge_v95,
)
from semantic_3d_chat.training.train_v96_atomic_pair_repair import (
    smoothmax_v96,
    symmetric_pair_objective_v96,
)

TRAINING_ARTIFACT: Final[str] = (
    "gemma4_v97_pair_decision_stability_training_v1"
)
CANDIDATE_ARTIFACT: Final[str] = (
    "gemma4_v97_pair_decision_stability_fixed_final_v1"
)
CHECKPOINT_ARTIFACT: Final[str] = (
    "gemma4_v97_pair_decision_stability_resume_v1"
)
WEIGHTS_FILENAME: Final[str] = "bridge.safetensors"
METADATA_FILENAME: Final[str] = "runtime_metadata.json"


def _leaf_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return Path(os.path.abspath(value))


def _strict_json(path: str | Path) -> dict[str, Any]:
    source = _leaf_path(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"V97 JSON must contain one object: {source}")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def combined_lora_settings_v97(
    runtime_config: Mapping[str, Any], experiment: Mapping[str, Any]
) -> LoRABanksSettings:
    """Replace V96's trainable slot with a checkpoint-initialized V97 slot."""

    v96_config_path = experiment["sources"]["frozen_v96_config"]
    from semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight import (
        load_config_v96,
    )

    v96_config = load_config_v96(v96_config_path, allow_draft=False)
    v95_config = load_config_v95(v96_config["sources"]["frozen_v95_config"], allow_draft=False)
    parent = combined_lora_settings_v95(runtime_config, v95_config)
    if (
        len(parent.banks) != EXPECTED_FROZEN_BANK_COUNT
        or tuple(bank.name for bank in parent.banks) != V94_BANKS + (V95_BANK_NAME,)
        or sum(bank.trainable for bank in parent.banks) != 1
    ):
        raise ValueError("V97 requires the exact nine-bank pre-V96 topology")
    v95 = parent.banks[-1]
    frozen_v95 = LoRABankSettings(
        name=v95.name,
        trainable=False,
        adapter=v95.adapter,
        initialization_algorithm="checkpoint_overwrite",
        expected_initial_state_sha256=str(
            v96_config["frozen_stack"]["v95_bank_state_sha256"]
        ),
    )
    bridge = experiment["bridge"]
    fresh = LoRABankSettings(
        name=FRESH_BANK_NAME,
        trainable=True,
        adapter=LoRASettings(
            enabled=True,
            rank=int(bridge["rank"]),
            alpha=float(bridge["alpha"]),
            dropout=float(bridge["dropout"]),
            target_modules=tuple(bridge["target_modules"]),
        ),
        initialization_algorithm="checkpoint_overwrite",
        expected_initial_state_sha256=EXPECTED_INITIAL_STATE_SHA256,
    )
    result = LoRABanksSettings(parent.banks[:-1] + (frozen_v95, fresh))
    if (
        len(result.banks) != 10
        or sum(bank.trainable for bank in result.banks) != 1
        or result.banks[-1].name != FRESH_BANK_NAME
    ):
        raise RuntimeError("V97 exact ten-bank continuation topology failed")
    return result


def _fresh_state(collection: LoRABankCollection) -> dict[str, torch.Tensor]:
    fresh = collection.bank(FRESH_BANK_NAME).installation
    if len(fresh.adapters) != 1:
        raise ValueError("V97 trainable bridge must wrap exactly one module")
    return {
        name: value.detach().cpu().contiguous()
        for name, value in fresh.state_module.state_dict().items()
    }


def _load_fresh_state(
    collection: LoRABankCollection, archive: Mapping[str, torch.Tensor]
) -> None:
    fresh = collection.bank(FRESH_BANK_NAME).installation
    if set(archive) != set(fresh.state_module.state_dict()):
        raise ValueError("V97 bridge tensor inventory changed")
    fresh.state_module.load_state_dict(dict(archive), strict=True)
    fresh.validate_state()


def load_frozen_parent_v97(
    collection: LoRABankCollection, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Load exact V85/V94/V95 banks and V96's state into V97's slot."""

    if collection.bank_names != V94_BANKS + (V95_BANK_NAME, FRESH_BANK_NAME):
        raise ValueError("V97 installed bank order changed")
    from semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight import (
        load_config_v96,
    )

    v96_config = load_config_v96(config["sources"]["frozen_v96_config"], allow_draft=False)
    v95_config = load_config_v95(v96_config["sources"]["frozen_v95_config"], allow_draft=False)
    sources = v95_config["sources"]
    v85 = _load_v85_banks_v95(collection, sources["frozen_v85_checkpoint"])
    v94 = _load_v94_bank_v95(collection, sources["frozen_v94_fixed_final"])
    v95 = load_fixed_final_bridge_v95(collection, v96_config["sources"]["frozen_v95_fixed_final"])
    v96_weights = (
        _leaf_path(config["sources"]["frozen_v96_fixed_final"]) / WEIGHTS_FILENAME
    )
    _load_fresh_state(collection, load_file(str(v96_weights), device="cpu"))
    fresh = collection.bank(FRESH_BANK_NAME).installation
    if (
        fresh.target_names != TARGET_MODULES
        or fresh.parameter_count != FRESH_PARAMETER_COUNT
        or fresh.state_sha256() != EXPECTED_INITIAL_STATE_SHA256
        or collection.parameter_count != EXPECTED_TOTAL_ADAPTER_PARAMETER_COUNT
        or collection.trainable_parameter_count != FRESH_PARAMETER_COUNT
    ):
        raise ValueError("V97 did not load the exact V96 continuation state")
    collection.validate_state()
    return {
        "parent": "v96_fixed_final_failed_exactly_two_known_development_gates",
        "v85": v85,
        "v94": v94,
        "v95": {
            "weights_sha256": v95["weights_sha256"],
            "state_sha256": v95["state_sha256"],
        },
        "v96": {
            "weights_sha256": sha256_file_v85(v96_weights),
            "state_sha256": fresh.state_sha256(),
            "row_level_development_content_loaded": False,
        },
        "frozen_bank_count": EXPECTED_FROZEN_BANK_COUNT,
        "frozen_parameter_count": EXPECTED_FROZEN_PARAMETER_COUNT,
        "runtime_release_loaded": False,
    }


def _finite_scalar(value: torch.Tensor, name: str) -> None:
    if value.ndim != 0 or not torch.isfinite(value):
        raise ValueError(f"V97 {name} must be a finite scalar")


def greedy_prefix_margin_v97(
    correct_tail: Any,
    alternative_tail: Any,
    *,
    target_margin: float,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Require greedy correctness through divergence against the full vocabulary."""

    correct_targets = correct_tail.targets
    alternative_targets = alternative_tail.targets
    if correct_targets.ndim != 1 or alternative_targets.ndim != 1:
        raise ValueError("V97 answer-tail targets must be vectors")
    limit = min(int(correct_targets.numel()), int(alternative_targets.numel()))
    divergence = next(
        (
            index
            for index in range(limit)
            if int(correct_targets[index]) != int(alternative_targets[index])
        ),
        None,
    )
    if divergence is None:
        raise ValueError("V97 changed answers have no token-level divergence")
    if not torch.equal(
        correct_targets[:divergence], alternative_targets[:divergence]
    ):
        raise RuntimeError("V97 first-divergence prefix changed")
    if not math.isfinite(target_margin) or target_margin < 0.0:
        raise ValueError("V97 greedy target margin must be finite nonnegative")
    logits = correct_tail.logits[0, : divergence + 1].float()
    targets = correct_targets[: divergence + 1]
    correct_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    # topk scans the complete vocabulary.  If top-1 is the correct token, top-2
    # is the maximum rival; otherwise top-1 is already the maximum rival.
    top_values, top_indices = torch.topk(logits, k=2, dim=-1)
    rival_logits = torch.where(
        top_indices[:, 0].eq(targets), top_values[:, 1], top_values[:, 0]
    )
    penalties = torch.relu(rival_logits - correct_logits + float(target_margin))
    penalty = penalties.max()
    decisive_margin = correct_logits[-1] - rival_logits[-1]
    _finite_scalar(penalty, "greedy prefix margin penalty")
    _finite_scalar(decisive_margin, "first-divergent greedy margin")
    return penalty, divergence, decisive_margin


def changed_pair_objective_v97(
    left_correct_tail: Any,
    right_correct_tail: Any,
    left_alternative_tail: Any,
    right_alternative_tail: Any,
    *,
    left_class_weight: float,
    right_class_weight: float,
    family_weight: float,
    correct_ce_weight: float,
    answer_margin_weight: float,
    answer_target_margin: float,
    causal_margin_weight: float,
    causal_target_margin: float,
    first_token_margin_weight: float,
    first_token_target_margin: float,
    smoothmax_temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    base, components = symmetric_pair_objective_v96(
        left_correct_tail.mean_nll.float(),
        right_correct_tail.mean_nll.float(),
        left_alternative_tail.mean_nll.float(),
        right_alternative_tail.mean_nll.float(),
        left_class_weight=left_class_weight,
        right_class_weight=right_class_weight,
        family_weight=family_weight,
        correct_ce_weight=correct_ce_weight,
        answer_margin_weight=answer_margin_weight,
        answer_target_margin=answer_target_margin,
        causal_margin_weight=causal_margin_weight,
        causal_target_margin=causal_target_margin,
        smoothmax_temperature=smoothmax_temperature,
    )
    left_penalty, left_index, left_decisive_margin = greedy_prefix_margin_v97(
        left_correct_tail,
        left_alternative_tail,
        target_margin=first_token_target_margin,
    )
    right_penalty, right_index, right_decisive_margin = greedy_prefix_margin_v97(
        right_correct_tail,
        right_alternative_tail,
        target_margin=first_token_target_margin,
    )
    token_penalty = smoothmax_v96(
        left_penalty, right_penalty, temperature=smoothmax_temperature
    )
    objective = base + float(family_weight) * float(first_token_margin_weight) * token_penalty
    return objective, {
        **components,
        "left_first_token_penalty": left_penalty,
        "right_first_token_penalty": right_penalty,
        "first_token_smoothmax_penalty": token_penalty,
        "left_first_divergent_index": left_index,
        "right_first_divergent_index": right_index,
        "left_first_divergent_greedy_margin": left_decisive_margin,
        "right_first_divergent_greedy_margin": right_decisive_margin,
    }


def answer_tail_js_v97(
    left_tail: Any,
    right_tail: Any,
    *,
    first_token_weight: float = 0.5,
    mean_tail_weight: float = 0.5,
) -> torch.Tensor:
    """Exact full-vocabulary JS with first-token and whole-tail components."""

    if (
        not torch.equal(left_tail.targets, right_tail.targets)
        or left_tail.logits.shape != right_tail.logits.shape
    ):
        raise ValueError("V97 invariant tails must have identical targets and shapes")
    left_log = F.log_softmax(left_tail.logits.float(), dim=-1)
    right_log = F.log_softmax(right_tail.logits.float(), dim=-1)
    mixture_log = torch.logaddexp(left_log, right_log) - math.log(2.0)
    left_kl = torch.sum(left_log.exp() * (left_log - mixture_log), dim=-1)
    right_kl = torch.sum(right_log.exp() * (right_log - mixture_log), dim=-1)
    if (
        not math.isfinite(first_token_weight)
        or not math.isfinite(mean_tail_weight)
        or first_token_weight < 0.0
        or mean_tail_weight < 0.0
        or abs(first_token_weight + mean_tail_weight - 1.0) > 1e-12
    ):
        raise ValueError("V97 Jensen-Shannon weights must be nonnegative and sum to one")
    per_token = 0.5 * (left_kl + right_kl)
    # The mean includes the final EOS/end-of-turn decision supplied by the
    # canonical answer tokenizer, so stability covers termination as well.
    result = (
        float(first_token_weight) * per_token[0, 0]
        + float(mean_tail_weight) * per_token.mean()
    )
    _finite_scalar(result, "invariant answer-tail JS divergence")
    if float(result.detach().cpu()) < -1e-6:
        raise ValueError("V97 Jensen-Shannon divergence became negative")
    return torch.clamp_min(result, 0.0)


def invariant_pair_objective_v97(
    left_tail: Any,
    right_tail: Any,
    *,
    left_class_weight: float,
    right_class_weight: float,
    family_weight: float,
    correct_ce_weight: float,
    consistency_weight: float,
    consistency_tolerance: float,
    answer_tail_js_weight: float,
    answer_tail_js_first_token_weight: float,
    answer_tail_js_mean_tail_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    left_nll = left_tail.mean_nll.float()
    right_nll = right_tail.mean_nll.float()
    correct_ce = 0.5 * (
        float(left_class_weight) * left_nll
        + float(right_class_weight) * right_nll
    )
    gap = torch.abs(left_nll - right_nll)
    consistency = torch.relu(gap - float(consistency_tolerance))
    js = answer_tail_js_v97(
        left_tail,
        right_tail,
        first_token_weight=answer_tail_js_first_token_weight,
        mean_tail_weight=answer_tail_js_mean_tail_weight,
    )
    objective = float(family_weight) * (
        float(correct_ce_weight) * correct_ce
        + float(consistency_weight) * consistency
        + float(answer_tail_js_weight) * js
    )
    return objective, {
        "correct_ce": correct_ce,
        "absolute_nll_gap": gap,
        "consistency_penalty": consistency,
        "answer_tail_js": js,
    }


def parent_anchor_rms_v97(
    collection: LoRABankCollection,
    parent_state: Mapping[str, torch.Tensor],
    *,
    epsilon: float,
) -> torch.Tensor:
    fresh = collection.bank(FRESH_BANK_NAME).installation
    current = fresh.state_module.state_dict(keep_vars=True)
    if set(current) != set(parent_state):
        raise ValueError("V97 parent-anchor tensor inventory changed")
    count = sum(value.numel() for value in current.values())
    if count != FRESH_PARAMETER_COUNT or not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("V97 parent-anchor contract changed")
    squared_delta = sum(
        torch.sum(
            (value.float() - parent_state[name].to(value.device).float()).square()
        )
        for name, value in current.items()
    )
    squared_parent = sum(
        torch.sum(parent_state[name].to(value.device).float().square())
        for name, value in current.items()
    )
    if not torch.isfinite(squared_parent) or float(squared_parent.detach().cpu()) <= 0.0:
        raise ValueError("V97 parent-anchor reference norm is invalid")
    result = torch.sqrt(squared_delta / squared_parent + float(epsilon))
    _finite_scalar(result, "relative parent-anchor RMS")
    return result


def _answer_tail(language: Any, prepared: Any) -> Any:
    from semantic_3d_chat.language.gemma4_answer_tail import answer_tail_forward

    return answer_tail_forward(language, prepared)


def _optimizer_tensors(
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    state = optimizer.state_dict()
    tensors: dict[str, torch.Tensor] = {}
    for parameter_index, values in state["state"].items():
        for name, value in values.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError("V97 AdamW state must be tensor-only")
            tensors[f"optimizer.{parameter_index}.{name}"] = value.detach().cpu().contiguous()
    groups: list[dict[str, Any]] = []
    for group in state["param_groups"]:
        normalized = dict(group)
        normalized["params"] = [int(value) for value in normalized["params"]]
        groups.append(normalized)
    return tensors, groups


def _restore_optimizer(
    optimizer: torch.optim.Optimizer,
    archive: Mapping[str, torch.Tensor],
    groups: Sequence[Mapping[str, Any]],
) -> None:
    state: dict[int, dict[str, torch.Tensor]] = {}
    for key, value in archive.items():
        if key.startswith("optimizer."):
            _prefix, raw_index, name = key.split(".", 2)
            state.setdefault(int(raw_index), {})[name] = value.detach().cpu()
    optimizer.load_state_dict(
        {"state": state, "param_groups": [dict(group) for group in groups]}
    )


def save_resume_checkpoint_v97(
    work_root: str | Path,
    collection: LoRABankCollection,
    optimizer: torch.optim.Optimizer,
    *,
    update: int,
    row_cursor: int,
    history: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Any],
    schedule_sha256: str,
) -> Path:
    root = _leaf_path(work_root)
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ValueError("V97 resume work root must be an unlinked directory")
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"update_{update:06d}"
    if destination.exists() or destination.is_symlink():
        existing = _strict_json(destination / "state.json")
        if (
            existing.get("update") == update
            and existing.get("row_cursor") == row_cursor
            and existing.get("fresh_state_sha256")
            == collection.bank(FRESH_BANK_NAME).installation.state_sha256()
        ):
            return destination
        raise FileExistsError(f"V97 checkpoint collision: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=root))
    try:
        optimizer_tensors, groups = _optimizer_tensors(optimizer)
        weights = temporary / "state.safetensors"
        save_file(
            {**_fresh_state(collection), **optimizer_tensors},
            str(weights),
            metadata={
                "artifact": CHECKPOINT_ARTIFACT,
                "environmental_memory_serialized": "false",
                "questions_or_answers_serialized": "false",
                "oracle_serialized": "false",
            },
        )
        state = {
            "artifact": CHECKPOINT_ARTIFACT,
            "schema_version": 97,
            "status": "resumable_training_state",
            "update": update,
            "row_cursor": row_cursor,
            "schedule_sha256": schedule_sha256,
            "fresh_state_sha256": collection.bank(FRESH_BANK_NAME).installation.state_sha256(),
            "tensor_file_sha256": sha256_file_v85(weights),
            "optimizer_param_groups": groups,
            "history": list(history),
            "bindings": dict(bindings),
            "environmental_memory_serialized": False,
            "questions_or_answers_serialized": False,
            "oracle_serialized": False,
        }
        (temporary / "state.json").write_text(
            json.dumps(state, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, destination)
        return destination
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def discover_resume_checkpoint_v97(
    work_root: str | Path,
    *,
    bindings: Mapping[str, Any],
    schedule_sha256: str,
    optimizer_groups: Sequence[Mapping[str, Any]],
) -> tuple[Path, dict[str, Any]] | None:
    root = _leaf_path(work_root)
    if root.is_symlink():
        raise ValueError("V97 resume root may not be a symlink")
    if not root.exists():
        return None
    if not root.is_dir():
        raise ValueError("V97 resume root must be a directory")
    paths = set(root.glob("update_[0-9][0-9][0-9][0-9][0-9][0-9]"))
    if set(root.iterdir()) != paths:
        raise ValueError("V97 resume-root inventory changed")
    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    for path in paths:
        if path.is_symlink() or not path.is_dir():
            raise ValueError("V97 resume checkpoint must be an unlinked directory")
        if {child.name for child in path.iterdir()} != {"state.json", "state.safetensors"}:
            raise ValueError("V97 resume checkpoint inventory changed")
        metadata = _strict_json(path / "state.json")
        tensor_path = path / "state.safetensors"
        if tensor_path.is_symlink() or not tensor_path.is_file():
            raise ValueError("V97 resume tensor is absent or linked")
        archive = load_file(str(tensor_path), device="cpu")
        update = metadata.get("update")
        cursor = metadata.get("row_cursor")
        history = metadata.get("history")
        fresh_keys = {"adapters.0.lora_a", "adapters.0.lora_b"}
        optimizer_keys = {key for key in archive if key.startswith("optimizer.")}
        expected_optimizer_keys = {
            f"optimizer.{index}.{name}"
            for index in range(2)
            for name in ("step", "exp_avg", "exp_avg_sq")
        }
        if (
            metadata.get("artifact") != CHECKPOINT_ARTIFACT
            or metadata.get("schema_version") != 97
            or metadata.get("status") != "resumable_training_state"
            or type(update) is not int
            or type(cursor) is not int
            or path.name != f"update_{update:06d}"
            or not 1 <= update <= EXPECTED_OPTIMIZER_UPDATES
            or update % 12 != 0
            or cursor != update * 6
            or metadata.get("schedule_sha256") != schedule_sha256
            or metadata.get("bindings") != dict(bindings)
            or metadata.get("optimizer_param_groups")
            != [dict(group) for group in optimizer_groups]
            or metadata.get("tensor_file_sha256") != sha256_file_v85(tensor_path)
            or metadata.get("environmental_memory_serialized") is not False
            or metadata.get("questions_or_answers_serialized") is not False
            or metadata.get("oracle_serialized") is not False
            or not isinstance(history, list)
            or len(history) != update
            or set(archive) - optimizer_keys != fresh_keys
            or optimizer_keys != expected_optimizer_keys
            or tensor_state_sha256({key: archive[key] for key in fresh_keys})
            != metadata.get("fresh_state_sha256")
            or any(
                not isinstance(record, Mapping)
                or record.get("update") != index
                or record.get("row_cursor") != index * 6
                or not _is_sha256(record.get("state_sha256"))
                for index, record in enumerate(history, 1)
            )
        ):
            raise ValueError(f"V97 resume checkpoint authentication failed: {path}")
        for index, fresh_name in enumerate(
            ("adapters.0.lora_a", "adapters.0.lora_b")
        ):
            if (
                archive[f"optimizer.{index}.step"].numel() != 1
                or float(archive[f"optimizer.{index}.step"].item()) != float(update)
                or archive[f"optimizer.{index}.exp_avg"].shape != archive[fresh_name].shape
                or archive[f"optimizer.{index}.exp_avg_sq"].shape != archive[fresh_name].shape
            ):
                raise ValueError("V97 resume optimizer tensor shape changed")
        candidates.append((update, path, metadata))
    if not candidates:
        return None
    _update, path, metadata = max(candidates, key=lambda item: item[0])
    return path, metadata


def restore_resume_checkpoint_v97(
    checkpoint: Path,
    metadata: Mapping[str, Any],
    collection: LoRABankCollection,
    optimizer: torch.optim.Optimizer,
) -> None:
    archive = load_file(str(checkpoint / "state.safetensors"), device="cpu")
    fresh_keys = set(
        collection.bank(FRESH_BANK_NAME).installation.state_module.state_dict()
    )
    _load_fresh_state(
        collection, {name: archive[name] for name in fresh_keys if name in archive}
    )
    if (
        collection.bank(FRESH_BANK_NAME).installation.state_sha256()
        != metadata["fresh_state_sha256"]
    ):
        raise ValueError("V97 resumed bridge hash changed")
    groups = metadata.get("optimizer_param_groups")
    if not isinstance(groups, list):
        raise TypeError("V97 resume optimizer groups are missing")
    _restore_optimizer(optimizer, archive, groups)


def publish_fixed_final_candidate_v97(
    destination: str | Path,
    collection: LoRABankCollection,
    *,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    root = _leaf_path(destination)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"V97 create-once candidate exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        weights = temporary / WEIGHTS_FILENAME
        state = _fresh_state(collection)
        save_file(
            state,
            str(weights),
            metadata={
                "artifact": CANDIDATE_ARTIFACT,
                "environmental_memory_serialized": "false",
                "questions_or_answers_serialized": "false",
                "oracle_serialized": "false",
            },
        )
        fresh = collection.bank(FRESH_BANK_NAME).installation
        metadata = {
            "artifact": CANDIDATE_ARTIFACT,
            "schema_version": 97,
            "status": "fixed_final_awaiting_v97_known_development_gate",
            "parent": "v96_fixed_final_failed_exactly_two_known_development_gates",
            "bank_name": FRESH_BANK_NAME,
            "target_modules": list(TARGET_MODULES),
            "rank": 8,
            "alpha": 16.0,
            "dropout": 0.0,
            "parameter_count": fresh.parameter_count,
            "initial_parent_state_sha256": EXPECTED_INITIAL_STATE_SHA256,
            "state_sha256": fresh.state_sha256(),
            "weights_sha256": sha256_file_v85(weights),
            "tensor_inventory": sorted(state),
            "environmental_memory_serialized": False,
            "questions_or_answers_serialized": False,
            "oracle_serialized": False,
            "known_development_scored": False,
            "deferred_final_generated": False,
            "runtime_promotion_authorized": False,
            "bindings": dict(bindings),
        }
        (temporary / METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, root)
        return metadata
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_fixed_final_bridge_v97(
    collection: LoRABankCollection, candidate: str | Path
) -> dict[str, Any]:
    root = _leaf_path(candidate)
    if (
        root.is_symlink()
        or not root.is_dir()
        or {child.name for child in root.iterdir()}
        != {WEIGHTS_FILENAME, METADATA_FILENAME}
    ):
        raise ValueError("V97 fixed-final candidate inventory changed")
    metadata = _strict_json(root / METADATA_FILENAME)
    weights = root / WEIGHTS_FILENAME
    fresh = collection.bank(FRESH_BANK_NAME).installation
    if (
        weights.is_symlink()
        or not weights.is_file()
        or metadata.get("artifact") != CANDIDATE_ARTIFACT
        or metadata.get("schema_version") != 97
        or metadata.get("status")
        != "fixed_final_awaiting_v97_known_development_gate"
        or metadata.get("bank_name") != FRESH_BANK_NAME
        or metadata.get("target_modules") != list(TARGET_MODULES)
        or metadata.get("parameter_count") != FRESH_PARAMETER_COUNT
        or metadata.get("initial_parent_state_sha256")
        != EXPECTED_INITIAL_STATE_SHA256
        or metadata.get("weights_sha256") != sha256_file_v85(weights)
        or metadata.get("tensor_inventory")
        != sorted(fresh.state_module.state_dict())
        or metadata.get("environmental_memory_serialized") is not False
        or metadata.get("questions_or_answers_serialized") is not False
        or metadata.get("oracle_serialized") is not False
        or metadata.get("known_development_scored") is not False
        or metadata.get("deferred_final_generated") is not False
        or metadata.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V97 fixed-final candidate authentication failed")
    _load_fresh_state(collection, load_file(str(weights), device="cpu"))
    if fresh.state_sha256() != metadata["state_sha256"]:
        raise ValueError("V97 candidate state changed")
    return metadata


def finalize_fixed_final_candidate_v97(
    destination: str | Path,
    collection: LoRABankCollection,
    *,
    bindings: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    root = _leaf_path(destination)
    if not root.exists() and not root.is_symlink():
        return publish_fixed_final_candidate_v97(root, collection, bindings=bindings), False
    expected_state = collection.bank(FRESH_BANK_NAME).installation.state_sha256()
    metadata = load_fixed_final_bridge_v97(collection, root)
    if metadata.get("bindings") != dict(bindings) or metadata.get("state_sha256") != expected_state:
        raise ValueError("V97 existing candidate bindings changed")
    return metadata, True


def run_training_v97(config_path: str | Path = CONFIG) -> dict[str, Any]:
    started = time.monotonic()
    config = load_config_v97(config_path, allow_draft=False)
    audit = FileAccessAudit(
        forbidden_training_roots_v97(config),
        forbidden_component_names=frozenset(),
        block_forbidden=True,
    )
    with audit:
        result = _run_training_under_audit_v97(
            config_path=config_path, config=config, audit=audit, started=started
        )
    audit.assert_clean()
    return result


def _run_training_under_audit_v97(
    *,
    config_path: str | Path,
    config: Mapping[str, Any],
    audit: FileAccessAudit,
    started: float,
) -> dict[str, Any]:
    source_hashes = authenticate_training_sources_v97(config)
    preflight = authenticate_cpu_preflight_v97(config, config_path=config_path)
    parent_evidence = authenticate_parent_v96_v97(
        config, allow_pending_aggregate=False
    )
    assert_deferred_final_absent_v97(config)
    rows = load_training_rows_v96(config)
    changed_units, _ = pair_units_v96(rows)
    invariant_subset = invariant_subset_v97(rows)
    class_weights = balanced_class_weights_v96(config, rows)
    changed_family_weights = family_weights_v96(changed_units)
    invariant_family_weights = family_weights_v96(invariant_subset)
    training = config["training"]
    schedule = training_schedule_v97(rows, seed=int(training["schedule_seed"]))
    schedule_sha = canonical_sha256_v85([step.identity() for step in schedule])
    subset_sha = canonical_sha256_v85(
        [[unit.pair_id, unit.question_key] for unit in invariant_subset]
    )
    if (
        schedule_sha != training["schedule_sha256"]
        or subset_sha != training["invariant_subset_sha256"]
    ):
        raise RuntimeError("V97 fixed schedule hashes changed")
    counts = Counter(step.kind for step in schedule)
    if counts != Counter(
        retention=EXPECTED_RETENTION_STEPS,
        changed_pair=EXPECTED_CHANGED_PAIR_STEPS,
        invariant_pair=EXPECTED_INVARIANT_PAIR_STEPS,
    ):
        raise RuntimeError("V97 fixed schedule inventory changed")
    outputs = config["outputs"]
    report_path = _leaf_path(outputs["training_report"])
    candidate_path = _leaf_path(outputs["fixed_final_candidate"])
    if report_path.exists() or report_path.is_symlink():
        raise FileExistsError("V97 create-once training report exists")

    cpu_memories, memory_hashes_before = load_scene_memories_v96(config, rows)
    runtime = load_runtime_config(config["sources"]["runtime_config"])
    language_config = runtime["language"]
    language = load_local_language_model(
        str(language_config["model_id"]),
        str(language_config["revision"]),
        str(language_config["dtype"]),
        freeze=True,
        local_files_only=True,
        backend="gemma4",
        decoder_gradient_checkpointing=True,
    )
    with torch.enable_grad():
        if language.device.type != "mps":
            raise RuntimeError("V97 full-model training requires local MPS")
        collection = install_lora_banks(
            language.model, combined_lora_settings_v97(runtime, config)
        )
        if not isinstance(collection, LoRABankCollection):
            raise TypeError("V97 LoRA bank installation failed")
        frozen_source = load_frozen_parent_v97(collection, config)
        collection.assert_trainable_surface(language.model)
        fresh = collection.bank(FRESH_BANK_NAME).installation
        initial_state_sha256 = fresh.state_sha256()
        anchor_state = {
            name: value.detach().clone()
            for name, value in fresh.state_module.state_dict().items()
        }
        memory_by_scene = {
            scene_id: memory.to(device=language.device, dtype=torch.bfloat16)
            for scene_id, memory in cpu_memories.items()
        }
        system_prompt = str(language_config["system_prompt"])
        parameters = collection.parameters()
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        _unused_tensors, optimizer_groups = _optimizer_tensors(optimizer)
        bindings: dict[str, Any] = {
            **preflight,
            "trainer_source_sha256": source_hashes[
                str(config["sources"]["trainer_source"])
            ],
            "schedule_sha256": schedule_sha,
            "invariant_subset_sha256": subset_sha,
            "v96_final_score_sha256": parent_evidence["v96_final_score_sha256"],
            "v96_evidence_sha256": parent_evidence["v96_evidence_sha256"],
            "v96_parent_state_sha256": EXPECTED_INITIAL_STATE_SHA256,
        }
        resumed = discover_resume_checkpoint_v97(
            outputs["work_root"],
            bindings=bindings,
            schedule_sha256=schedule_sha,
            optimizer_groups=optimizer_groups,
        )
        history: list[dict[str, Any]] = []
        step_cursor = 0
        optimizer_update = 0
        resumed_from: str | None = None
        if resumed is not None:
            checkpoint, metadata = resumed
            restore_resume_checkpoint_v97(checkpoint, metadata, collection, optimizer)
            step_cursor = int(metadata["row_cursor"])
            optimizer_update = int(metadata["update"])
            history = list(metadata["history"])
            resumed_from = checkpoint.relative_to(PROJECT_ROOT).as_posix()
        if step_cursor > len(schedule) or step_cursor % 6:
            raise ValueError("V97 resume cursor is outside the fixed schedule")

        language.decoder_module.train()
        collection.train()
        optimizer.zero_grad(set_to_none=True)
        interval: list[dict[str, Any]] = []
        seen = Counter(step.kind for step in schedule[:step_cursor])
        nll_forwards = sum(
            1 if step.kind == "retention" else 4 if step.kind == "changed_pair" else 2
            for step in schedule[:step_cursor]
        )

        def tail(memory: torch.Tensor, row: Any) -> Any:
            prepared, _layout = _prepared_v84(language, system_prompt, memory, row)
            return _answer_tail(language, prepared)

        for cursor in range(step_cursor, len(schedule)):
            step: TrainingStepV97 = schedule[cursor]
            record: dict[str, Any] = {"kind": step.kind}
            if step.kind == "retention":
                if step.row is None:
                    raise RuntimeError("V97 retention step lost its row")
                correct_tail = tail(memory_by_scene[step.row.scene_id], step.row)
                objective = (
                    float(training["retention_balanced_ce_weight"])
                    * float(class_weights[step.row.answer_class])
                    * correct_tail.mean_nll.float()
                )
                correct_nll = correct_tail.mean_nll.float()
                record.update(
                    first_token_margin_penalty=0.0,
                    answer_margin_penalty=0.0,
                    causal_margin_penalty=0.0,
                    invariant_consistency_penalty=0.0,
                    invariant_answer_tail_js=0.0,
                )
                nll_forwards += 1
            elif step.kind == "changed_pair":
                if step.unit is None:
                    raise RuntimeError("V97 changed-pair step lost its unit")
                left, right = step.unit.left, step.unit.right
                left_correct = tail(memory_by_scene[left.scene_id], left)
                right_correct = tail(memory_by_scene[right.scene_id], right)
                left_alternative = tail(
                    memory_by_scene[left.scene_id],
                    replace(left, answer=right.answer, answer_class=right.answer_class),
                )
                right_alternative = tail(
                    memory_by_scene[right.scene_id],
                    replace(right, answer=left.answer, answer_class=left.answer_class),
                )
                objective, components = changed_pair_objective_v97(
                    left_correct,
                    right_correct,
                    left_alternative,
                    right_alternative,
                    left_class_weight=float(class_weights[left.answer_class]),
                    right_class_weight=float(class_weights[right.answer_class]),
                    family_weight=float(changed_family_weights[step.unit.change_type]),
                    correct_ce_weight=float(training["pair_correct_ce_weight"]),
                    answer_margin_weight=float(training["within_memory_answer_margin_weight"]),
                    answer_target_margin=float(training["within_memory_answer_target_margin_nll"]),
                    causal_margin_weight=float(training["across_memory_causal_margin_weight"]),
                    causal_target_margin=float(training["across_memory_causal_target_margin_nll"]),
                    first_token_margin_weight=float(training["first_divergent_token_margin_weight"]),
                    first_token_target_margin=float(training["first_divergent_token_target_margin_nll"]),
                    smoothmax_temperature=float(training["pair_side_smoothmax_temperature"]),
                )
                correct_nll = 0.5 * (
                    left_correct.mean_nll.float() + right_correct.mean_nll.float()
                )
                record.update(
                    first_token_margin_penalty=float(
                        components["first_token_smoothmax_penalty"].detach().cpu()  # type: ignore[union-attr]
                    ),
                    answer_margin_penalty=float(
                        components["answer_smoothmax_penalty"].detach().cpu()  # type: ignore[union-attr]
                    ),
                    causal_margin_penalty=float(
                        components["causal_smoothmax_penalty"].detach().cpu()  # type: ignore[union-attr]
                    ),
                    invariant_consistency_penalty=0.0,
                    invariant_answer_tail_js=0.0,
                    family=step.unit.change_type,
                )
                nll_forwards += 4
            elif step.kind == "invariant_pair":
                if step.unit is None:
                    raise RuntimeError("V97 invariant-pair step lost its unit")
                left, right = step.unit.left, step.unit.right
                left_tail = tail(memory_by_scene[left.scene_id], left)
                right_tail = tail(memory_by_scene[right.scene_id], right)
                objective, components = invariant_pair_objective_v97(
                    left_tail,
                    right_tail,
                    left_class_weight=float(class_weights[left.answer_class]),
                    right_class_weight=float(class_weights[right.answer_class]),
                    family_weight=float(invariant_family_weights[step.unit.change_type]),
                    correct_ce_weight=float(training["invariant_correct_ce_weight"]),
                    consistency_weight=float(training["invariant_nll_consistency_weight"]),
                    consistency_tolerance=float(training["invariant_nll_consistency_tolerance"]),
                    answer_tail_js_weight=float(training["invariant_answer_tail_js_weight"]),
                    answer_tail_js_first_token_weight=float(
                        training["invariant_answer_tail_js_first_token_weight"]
                    ),
                    answer_tail_js_mean_tail_weight=float(
                        training["invariant_answer_tail_js_mean_tail_weight"]
                    ),
                )
                correct_nll = 0.5 * (
                    left_tail.mean_nll.float() + right_tail.mean_nll.float()
                )
                record.update(
                    first_token_margin_penalty=0.0,
                    answer_margin_penalty=0.0,
                    causal_margin_penalty=0.0,
                    invariant_consistency_penalty=float(
                        components["consistency_penalty"].detach().cpu()
                    ),
                    invariant_answer_tail_js=float(
                        components["answer_tail_js"].detach().cpu()
                    ),
                    family=step.unit.change_type,
                )
                nll_forwards += 2
            else:
                raise RuntimeError(f"V97 unknown schedule kind: {step.kind}")

            anchor = parent_anchor_rms_v97(
                collection,
                anchor_state,
                epsilon=float(training["parent_anchor_epsilon"]),
            )
            objective = objective + float(training["parent_anchor_rms_weight"]) * anchor
            if not torch.isfinite(objective):
                raise RuntimeError("V97 objective is nonfinite")
            record.update(
                correct_nll=float(correct_nll.detach().cpu()),
                parent_anchor_rms=float(anchor.detach().cpu()),
                objective=float(objective.detach().cpu()),
            )
            interval.append(record)
            (objective / 6).backward()
            step_cursor = cursor + 1
            seen[step.kind] += 1
            if step_cursor % 6:
                continue
            for parameter in parameters:
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("V97 found a missing or nonfinite gradient")
            gradient_l2 = float(collection.gradient_norms()["total_l2"])
            if not math.isfinite(gradient_l2) or gradient_l2 <= 0.0:
                raise RuntimeError("V97 gradient is zero or nonfinite")
            clipped = torch.nn.utils.clip_grad_norm_(
                parameters, float(training["gradient_clip_norm"])
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_update += 1
            collection.validate_state()
            history.append(
                {
                    "update": optimizer_update,
                    "row_cursor": step_cursor,
                    "mean_correct_nll": sum(v["correct_nll"] for v in interval)
                    / len(interval),
                    "mean_objective": sum(v["objective"] for v in interval)
                    / len(interval),
                    "mean_first_token_margin_penalty": sum(
                        v["first_token_margin_penalty"] for v in interval
                    )
                    / len(interval),
                    "mean_invariant_answer_tail_js": sum(
                        v["invariant_answer_tail_js"] for v in interval
                    )
                    / len(interval),
                    "mean_parent_anchor_rms": sum(
                        v["parent_anchor_rms"] for v in interval
                    )
                    / len(interval),
                    "gradient_l2_before_clip": gradient_l2,
                    "clip_return_l2": float(clipped.detach().cpu()),
                    "state_sha256": fresh.state_sha256(),
                }
            )
            interval.clear()
            if optimizer_update % 12 == 0:
                save_resume_checkpoint_v97(
                    outputs["work_root"],
                    collection,
                    optimizer,
                    update=optimizer_update,
                    row_cursor=step_cursor,
                    history=history,
                    bindings=bindings,
                    schedule_sha256=schedule_sha,
                )
            if optimizer_update in {1, 54, 107, 160, 214}:
                print(json.dumps(history[-1], sort_keys=True), flush=True)

        if (
            step_cursor != EXPECTED_MICRO_STEPS
            or optimizer_update != EXPECTED_OPTIMIZER_UPDATES
            or interval
        ):
            raise RuntimeError("V97 fixed schedule did not complete exactly")
        memory_hashes_after = {
            scene_id: __import__(
                "semantic_3d_chat.language.prefix_injection", fromlist=["prefix_sha256"]
            ).prefix_sha256(memory.detach().cpu())
            for scene_id, memory in memory_by_scene.items()
        }
        candidate_bindings = {
            **bindings,
            "fixed_final_optimizer_updates": optimizer_update,
            "class_weight_inventory_sha256": config["training_pool"][
                "balanced_class_weight_inventory_sha256"
            ],
            "changed_family_weight_inventory_sha256": config["training_pool"][
                "changed_family_weight_inventory_sha256"
            ],
            "invariant_family_weight_inventory_sha256": config["training_pool"][
                "invariant_family_weight_inventory_sha256"
            ],
            "known_development_row_level_content_opened": False,
            "deferred_final_generated": False,
        }
        candidate_metadata, candidate_reused = finalize_fixed_final_candidate_v97(
            candidate_path, collection, bindings=candidate_bindings
        )
    audit.assert_clean()

    gates = {
        "all_960_rows_consumed_once": seen["retention"] == EXPECTED_RETENTION_STEPS,
        "all_66_changed_units_consumed_twice": seen["changed_pair"]
        == EXPECTED_CHANGED_PAIR_STEPS,
        "all_192_invariant_units_consumed_once": seen["invariant_pair"]
        == EXPECTED_INVARIANT_PAIR_STEPS,
        "exact_1872_nll_forward_schedule": nll_forwards
        == EXPECTED_TOTAL_NLL_FORWARDS,
        "fixed_final_update_214_reached": optimizer_update
        == EXPECTED_OPTIMIZER_UPDATES,
        "only_continued_45056_parameters_trainable": len(parameters) == 2
        and sum(parameter.numel() for parameter in parameters) == FRESH_PARAMETER_COUNT,
        "exact_v96_parent_state_loaded": initial_state_sha256
        == EXPECTED_INITIAL_STATE_SHA256,
        "nonzero_finite_gradient_every_update": all(
            math.isfinite(float(record["gradient_l2_before_clip"]))
            and float(record["gradient_l2_before_clip"]) > 0.0
            for record in history
        ),
        "all_scene_hashes_invariant": memory_hashes_after == memory_hashes_before,
        "protected_read_count_zero": len(audit.forbidden_accesses()) == 0,
        "deferred_final_still_absent": not assert_deferred_final_absent_v97(config)[
            "physical_artifacts_present"
        ],
    }
    if not all(gates.values()):
        raise RuntimeError(f"V97 training gate failed: {gates}")
    report = {
        "artifact": TRAINING_ARTIFACT,
        "schema_version": 97,
        "status": "fixed_final_training_complete_not_promoted",
        "config_sha256": preflight["config_sha256"],
        "preregistration_sha256": preflight["preregistration_sha256"],
        "cpu_preflight_sha256": preflight["cpu_preflight_sha256"],
        "device": "mps",
        "model_id": config["sources"]["model_id"],
        "model_revision": config["sources"]["model_revision"],
        "strict_input_contract": config["strict_input_contract"],
        "source_hashes": source_hashes,
        "frozen_source": frozen_source,
        "trainable_bridge": {
            "bank_name": FRESH_BANK_NAME,
            "target_modules": list(TARGET_MODULES),
            "parameter_count": FRESH_PARAMETER_COUNT,
            "initial_v96_state_sha256": initial_state_sha256,
            "final_state_sha256": candidate_metadata["state_sha256"],
            "parent": "v96_fixed_final_failed_exactly_two_known_development_gates",
            "unmerged": True,
        },
        "training_protocol": training,
        "schedule_sha256": schedule_sha,
        "invariant_subset_sha256": subset_sha,
        "micro_steps_consumed": step_cursor,
        "retention_steps_consumed": seen["retention"],
        "changed_pair_steps_consumed": seen["changed_pair"],
        "invariant_pair_steps_consumed": seen["invariant_pair"],
        "total_nll_forwards": nll_forwards,
        "optimizer_updates": optimizer_update,
        "resumed_from": resumed_from,
        "candidate_reused_after_interruption": candidate_reused,
        "training_history": history,
        "scene_memories": {
            "compiled_before_question_tokenization": True,
            "shape_each": [1, 738, 1536],
            "hashes_before": memory_hashes_before,
            "hashes_after": memory_hashes_after,
            "hash_invariant": True,
            "all_memory_slots_retained": True,
            "question_derived_environmental_tokens": 0,
            "question_dependent_retrieval": False,
        },
        "gates": gates,
        "candidate": {
            "path": candidate_path.relative_to(PROJECT_ROOT).as_posix(),
            "weights_sha256": candidate_metadata["weights_sha256"],
            "metadata_canonical_sha256": canonical_sha256_v85(candidate_metadata),
            "fixed_final": True,
            "known_development_scored": False,
            "runtime_promotion_authorized": False,
        },
        "loaded_file_count": len(audit.unique_paths),
        "loaded_file_inventory_sha256": canonical_sha256_v85(audit.unique_paths),
        "protected_read_count": len(audit.forbidden_accesses()),
        "known_development_row_level_content_loaded": False,
        "deferred_final_generated": False,
        "oracle_loaded": False,
        "runtime_promotion_authorized": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_create_json_v85(report_path, report)
    return report


def authenticate_training_report_v97(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    preflight = authenticate_cpu_preflight_v97(config, config_path=config_path)
    path = _leaf_path(config["outputs"]["training_report"])
    report = _strict_json(path)
    if (
        report.get("artifact") != TRAINING_ARTIFACT
        or report.get("schema_version") != 97
        or report.get("status") != "fixed_final_training_complete_not_promoted"
        or report.get("config_sha256") != preflight["config_sha256"]
        or report.get("preregistration_sha256") != preflight["preregistration_sha256"]
        or report.get("cpu_preflight_sha256") != preflight["cpu_preflight_sha256"]
        or report.get("optimizer_updates") != EXPECTED_OPTIMIZER_UPDATES
        or report.get("micro_steps_consumed") != EXPECTED_MICRO_STEPS
        or report.get("retention_steps_consumed") != EXPECTED_RETENTION_STEPS
        or report.get("changed_pair_steps_consumed") != EXPECTED_CHANGED_PAIR_STEPS
        or report.get("invariant_pair_steps_consumed") != EXPECTED_INVARIANT_PAIR_STEPS
        or report.get("total_nll_forwards") != EXPECTED_TOTAL_NLL_FORWARDS
        or report.get("protected_read_count") != 0
        or report.get("known_development_row_level_content_loaded") is not False
        or report.get("deferred_final_generated") is not False
        or report.get("oracle_loaded") is not False
        or report.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V97 fixed-final training report changed")
    return {**preflight, "training_report_sha256": sha256_file_v85(path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_training_v97(args.config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_ARTIFACT",
    "CHECKPOINT_ARTIFACT",
    "TRAINING_ARTIFACT",
    "answer_tail_js_v97",
    "authenticate_training_report_v97",
    "changed_pair_objective_v97",
    "combined_lora_settings_v97",
    "discover_resume_checkpoint_v97",
    "finalize_fixed_final_candidate_v97",
    "greedy_prefix_margin_v97",
    "invariant_pair_objective_v97",
    "load_fixed_final_bridge_v97",
    "load_frozen_parent_v97",
    "parent_anchor_rms_v97",
    "publish_fixed_final_candidate_v97",
    "restore_resume_checkpoint_v97",
    "run_training_v97",
    "save_resume_checkpoint_v97",
]
