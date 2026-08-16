"""Read-only CPU preflight for the unsealed Gemma-4 tool decoder V3 draft."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.evaluation.gemma4_tool_decoder_v3_preregistration import (
    MODEL_ID,
    MODEL_REVISION,
    build_tool_decoder_v3_preregistration,
)
from semantic_3d_chat.training.gemma4_tool_decoder_v3_design import (
    ACTION_NAMES,
    MICROBATCH_COUNT,
    OPTIMIZER_UPDATES,
    TOKEN_ROLE_WEIGHTS,
    TRAIN_ROW_COUNT,
    argument_bin_name,
    balanced_schedule_v3,
    canonical_sha256,
    canonical_tool_json_v3,
    load_training_rows_only,
    schedule_summary_v3,
    token_roles_and_weights,
)

CONFIG_PATH: Final[Path] = Path(
    "configs/experiments/gemma4_embodied_tool_decoder_v3.yaml"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_local_tokenizer_for_v3() -> Any:
    """Load only the pinned local tokenizer; never instantiate model weights."""

    from transformers import AutoTokenizer  # Imported only for this tiny CPU audit.

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
    )
    if getattr(tokenizer, "is_fast", False) is not True:
        raise TypeError("V3 role weighting requires the pinned fast tokenizer")
    return tokenizer


def _token_role_audit(rows: tuple[Any, ...], tokenizer: Any) -> dict[str, Any]:
    if getattr(tokenizer, "is_fast", False) is not True:
        raise TypeError("V3 tokenizer audit requires character offsets")
    role_counts: Counter[str] = Counter()
    weight_mass: Counter[str] = Counter()
    action_role_counts: dict[str, Counter[str]] = defaultdict(Counter)
    canonical_cache: dict[str, tuple[tuple[str, ...], tuple[float, ...]]] = {}
    token_count = 0
    for row in rows:
        answer = canonical_tool_json_v3(row)
        cached = canonical_cache.get(answer)
        if cached is None:
            encoded = tokenizer(
                answer,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            if not isinstance(encoded, Mapping):
                raise TypeError("V3 tokenizer did not return a mapping")
            token_ids = encoded.get("input_ids")
            offsets = encoded.get("offset_mapping")
            if (
                not isinstance(token_ids, list)
                or not isinstance(offsets, list)
                or len(token_ids) != len(offsets)
                or not token_ids
            ):
                raise ValueError("V3 tokenizer returned invalid answer offsets")
            cached = token_roles_and_weights(answer, offsets, append_eos=True)
            canonical_cache[answer] = cached
        roles, weights = cached
        if "action" not in roles:
            raise ValueError("V3 tokenizer produced no action-name token")
        if row.action_name not in {"stop", "scan"} and "argument_value" not in roles:
            raise ValueError("V3 tokenizer produced no numeric-value token")
        role_counts.update(roles)
        for role, weight in zip(roles, weights, strict=True):
            weight_mass[role] += weight
            action_role_counts[row.action_name][role] += 1
        token_count += len(roles)
    all_roles = set(TOKEN_ROLE_WEIGHTS)
    if set(role_counts) != all_roles:
        raise ValueError("V3 tokenizer audit did not exercise every objective role")
    semantic_roles = {"action", "argument_value"}
    semantic_tokens = sum(role_counts[role] for role in semantic_roles)
    semantic_mass = sum(weight_mass[role] for role in semantic_roles)
    total_mass = sum(weight_mass.values())
    unweighted_fraction = semantic_tokens / token_count
    weighted_fraction = semantic_mass / total_mass
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "fast_tokenizer": True,
        "training_answer_count": len(rows),
        "unique_canonical_answer_count": len(canonical_cache),
        "answer_token_count_including_eos": token_count,
        "mean_answer_tokens_including_eos": token_count / len(rows),
        "role_token_counts": dict(sorted(role_counts.items())),
        "role_weight_mass": dict(sorted(weight_mass.items())),
        "role_token_counts_by_action": {
            action: dict(sorted(action_role_counts[action].items()))
            for action in ACTION_NAMES
        },
        "semantic_action_and_value_unweighted_fraction": unweighted_fraction,
        "semantic_action_and_value_weighted_fraction": weighted_fraction,
        "semantic_weight_fraction_increase": weighted_fraction - unweighted_fraction,
        "all_answers_have_action_tokens": True,
        "all_numeric_answers_have_value_tokens": True,
        "model_weights_loaded": False,
        "mps_used": False,
    }


def run_tool_decoder_v3_preflight(
    project_root: Path,
    *,
    tokenizer: Any,
) -> dict[str, Any]:
    """Authenticate the negative and test V3's fixed CPU-only design mechanics."""

    preregistration = build_tool_decoder_v3_preregistration(project_root)
    rows = load_training_rows_only(project_root)
    action_counts = Counter(row.action_name for row in rows)
    argument_cells = Counter(
        (row.action_name, argument_bin_name(row.action_name, row.normalized_argument))
        for row in rows
    )
    schedule = balanced_schedule_v3(rows)
    schedule_summary = schedule_summary_v3(rows, schedule)
    token_audit = _token_role_audit(rows, tokenizer)
    expected_actions = {
        "move_backward": 102,
        "move_forward": 1542,
        "scan": 112,
        "stop": 876,
        "turn": 1568,
    }
    expected_cells = {
        "move_backward:center": 102,
        "move_forward:neg_extreme": 12,
        "move_forward:neg_mid": 20,
        "move_forward:center": 162,
        "move_forward:pos_mid": 304,
        "move_forward:pos_extreme": 1044,
        "scan:none": 112,
        "stop:none": 876,
        "turn:neg_extreme": 604,
        "turn:neg_mid": 148,
        "turn:center": 146,
        "turn:pos_mid": 106,
        "turn:pos_extreme": 564,
    }
    observed_cells = {
        f"{action}:{bin_name}": count
        for (action, bin_name), count in sorted(argument_cells.items())
    }
    schedule_actions = schedule_summary["action_counts"]
    schedule_cells = schedule_summary["action_argument_bin_counts"]
    schedule_bin_checks = []
    for action in ACTION_NAMES:
        relevant = [
            count
            for cell, count in schedule_cells.items()
            if cell.startswith(f"{action}:")
        ]
        schedule_bin_checks.append(max(relevant) - min(relevant) == 0)
    checks = {
        "v2_terminal_sha_exact": preregistration["v2_2_terminal_negative"]["sha256"]
        == preregistration["parent_evidence"]["sha256"],
        "v2_terminal_negative": preregistration["v2_2_terminal_negative"]["status"]
        == "rejected_before_greedy_generation_no_runtime_checkpoint",
        "v2_checkpoint_absent": preregistration["v2_2_terminal_negative"][
            "runtime_checkpoint_absent"
        ]
        is True,
        "training_row_count": len(rows) == TRAIN_ROW_COUNT,
        "training_actions_exact": dict(sorted(action_counts.items()))
        == expected_actions,
        "training_argument_cells_exact": observed_cells == expected_cells,
        "training_scenes_exact": {row.scene_id for row in rows}
        == {f"scene_{index:06d}" for index in range(11, 25)},
        "schedule_microbatches_exact": schedule_summary["microbatch_count"]
        == MICROBATCH_COUNT,
        "schedule_updates_exact": schedule_summary["optimizer_updates"]
        == OPTIMIZER_UPDATES,
        "schedule_actions_exactly_balanced": set(schedule_actions.values()) == {160},
        "schedule_occupied_bins_exactly_balanced_per_action": all(
            schedule_bin_checks
        ),
        "schedule_covers_all_train_scenes": len(schedule_summary["scene_counts"])
        == 14,
        "schedule_covers_all_train_families": len(schedule_summary["family_counts"])
        == 7,
        "fast_tokenizer_offsets": token_audit["fast_tokenizer"] is True,
        "semantic_role_weight_mass_increased": token_audit[
            "semantic_weight_fraction_increase"
        ]
        > 0.20,
        "every_answer_has_action_tokens": token_audit[
            "all_answers_have_action_tokens"
        ]
        is True,
        "every_numeric_answer_has_value_tokens": token_audit[
            "all_numeric_answers_have_value_tokens"
        ]
        is True,
        "no_model_weights_loaded": token_audit["model_weights_loaded"] is False,
        "no_mps": token_audit["mps_used"] is False,
        "training_still_unauthorized": preregistration["authorization"][
            "training_authorized"
        ]
        is False,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Gemma tool-decoder V3 CPU preflight failed: {failed}")
    return {
        "schema": "semantic_3d_chat.gemma4_tool_decoder_v3_cpu_preflight.draft",
        "artifact": "gemma4_embodied_tool_decoder_v3_cpu_preflight_draft",
        "status": "passed_unsealed_cpu_preflight_training_still_unauthorized",
        "preregistration_canonical_sha256": canonical_sha256(preregistration),
        "config_path": str(CONFIG_PATH),
        "config_sha256": _sha256(project_root / CONFIG_PATH),
        "v2_2_terminal_negative": preregistration["v2_2_terminal_negative"],
        "diagnosis": preregistration["diagnosis"],
        "training_rows": {
            "count": len(rows),
            "actions": dict(sorted(action_counts.items())),
            "action_argument_bins": observed_cells,
            "scene_count": len({row.scene_id for row in rows}),
            "family_count": len({row.family for row in rows}),
            "heldout_rows_read": 0,
            "heldout_predictions_read": 0,
        },
        "schedule": schedule_summary,
        "token_role_audit": token_audit,
        "checks": checks,
        "passed": True,
        "loaded_environmental_text": [],
        "oracle_runtime_inputs": [],
        "execution": {
            "full_model_loaded": False,
            "model_weights_loaded": False,
            "tokenizer_loaded_cpu_only": True,
            "mps_used": False,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "heldout_rows_read": 0,
            "heldout_predictions_read": 0,
            "teacher_forced_forwards": 0,
            "greedy_generations": 0,
            "checkpoint_written": False,
            "training_authorized": False,
        },
    }


__all__ = [
    "CONFIG_PATH",
    "load_local_tokenizer_for_v3",
    "run_tool_decoder_v3_preflight",
]
