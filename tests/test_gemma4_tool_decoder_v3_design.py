from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.gemma4_tool_decoder_v3_preflight import (
    run_tool_decoder_v3_preflight,
)
from semantic_3d_chat.evaluation.gemma4_tool_decoder_v3_preregistration import (
    build_tool_decoder_v3_preregistration,
)
from semantic_3d_chat.training.gemma4_tool_decoder_v3_design import (
    ACTION_NAMES,
    MICROBATCH_COUNT,
    OPTIMIZER_UPDATES,
    TOKEN_ROLE_WEIGHTS,
    V2_RUNTIME_CHECKPOINT,
    V2_TERMINAL_PATH,
    V2_TERMINAL_SHA256,
    TrainingRowV3,
    answer_character_spans,
    authenticate_v2_2_terminal_negative,
    balanced_schedule_v3,
    canonical_tool_json_v3,
    load_training_rows_only,
    schedule_summary_v3,
    token_roles_and_weights,
    weighted_loss_from_token_losses,
)


class _CharacterTokenizer:
    is_fast = True
    eos_token_id = 1

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, object]:
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def _copy_terminal(destination_root: Path) -> Path:
    destination = destination_root / V2_TERMINAL_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PROJECT_ROOT / V2_TERMINAL_PATH, destination)
    return destination


def test_v3_authenticates_exact_terminal_negative_and_checkpoint_absence() -> None:
    result = authenticate_v2_2_terminal_negative(PROJECT_ROOT)
    assert result["sha256"] == V2_TERMINAL_SHA256
    assert result["runtime_checkpoint_absent"] is True
    assert result["runtime_checkpoint_published"] is False
    assert result["greedy_generation_executed"] is False
    assert result["aggregate_metrics"]["answer_token_nll"] == pytest.approx(
        0.37775762747489017
    )


def test_v3_terminal_authentication_fails_closed_on_bytes_or_checkpoint(
    tmp_path: Path,
) -> None:
    terminal = _copy_terminal(tmp_path)
    assert authenticate_v2_2_terminal_negative(tmp_path)["sha256"] == V2_TERMINAL_SHA256
    terminal.write_bytes(terminal.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="bytes changed"):
        authenticate_v2_2_terminal_negative(tmp_path)

    terminal.unlink()
    _copy_terminal(tmp_path)
    (tmp_path / V2_RUNTIME_CHECKPOINT).mkdir(parents=True)
    with pytest.raises(FileExistsError, match="final path exists"):
        authenticate_v2_2_terminal_negative(tmp_path)


def test_v3_reads_only_exact_training_prefix_and_builds_fixed_schedule() -> None:
    rows = load_training_rows_only(PROJECT_ROOT)
    assert len(rows) == 4200
    assert rows[0].sample_id == "g_00000000"
    assert rows[-1].sample_id == "g_00004199"
    assert {row.scene_id for row in rows} == {
        f"scene_{index:06d}" for index in range(11, 25)
    }
    schedule = balanced_schedule_v3(rows)
    assert schedule == balanced_schedule_v3(rows)
    summary = schedule_summary_v3(rows, schedule)
    assert summary["microbatch_count"] == MICROBATCH_COUNT
    assert summary["optimizer_updates"] == OPTIMIZER_UPDATES
    assert summary["action_counts"] == {action: 160 for action in sorted(ACTION_NAMES)}
    assert summary["action_argument_bin_counts"]["turn:neg_extreme"] == 32
    assert summary["action_argument_bin_counts"]["turn:pos_extreme"] == 32
    assert summary["action_argument_bin_counts"]["move_forward:neg_extreme"] == 32
    assert summary["action_argument_bin_counts"]["move_forward:pos_extreme"] == 32


def test_v3_weight_roles_prioritize_action_and_numeric_value() -> None:
    row = TrainingRowV3(
        sample_id="g_test",
        scene_id="scene_000011",
        family="face",
        action_index=2,
        action_name="turn",
        normalized_argument=-1.0,
    )
    answer = canonical_tool_json_v3(row)
    assert answer == '{"arguments":{"angle_degrees":-45.0},"tool":"turn"}'
    spans = answer_character_spans(answer)
    assert answer[slice(*spans["action"][0])] == "turn"
    assert answer[slice(*spans["argument_value"][0])] == "-45.0"
    offsets = [(index, index + 1) for index in range(len(answer))]
    roles, weights = token_roles_and_weights(answer, offsets)
    assert roles[-1] == "eos"
    assert TOKEN_ROLE_WEIGHTS["action"] == 8.0
    assert TOKEN_ROLE_WEIGHTS["argument_value"] == 6.0
    assert max(weights) == TOKEN_ROLE_WEIGHTS["action"]
    assert roles.count("action") == len("turn")
    assert roles.count("argument_value") == len("-45.0")


def test_v3_weighted_loss_is_per_answer_normalized_and_fail_closed() -> None:
    assert weighted_loss_from_token_losses([1.0, 3.0], [1.0, 3.0]) == 2.5
    with pytest.raises(ValueError, match="nonempty and aligned"):
        weighted_loss_from_token_losses([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="finite and positive"):
        weighted_loss_from_token_losses([1.0], [0.0])


def test_v3_preregistration_is_one_arm_unsealed_and_keeps_all_v2_gates() -> None:
    report = build_tool_decoder_v3_preregistration(PROJECT_ROOT)
    assert report["status"] == "unsealed_cpu_design_only_training_unauthorized"
    assert report["authorization"] == {
        "sealed": False,
        "training_authorized": False,
        "full_model_load_authorized": False,
        "mps_authorized": False,
        "optimizer_construction_authorized": False,
        "heldout_rows_authorized": False,
        "greedy_generation_authorized": False,
        "checkpoint_write_authorized": False,
    }
    assert report["single_fixed_arm"]["arm_count"] == 1
    assert report["single_fixed_arm"]["target_module"].startswith(
        "model.language_model.layers.34."
    )
    assert report["surface_isolation"]["disjoint_from_static_v6"] is True
    assert report["unchanged_early_teacher_forced_gates"][
        "teacher_forced_argmax_valid_schema_rate_minimum"
    ] == 0.8
    assert report["unchanged_primary_greedy_gates"][
        "exact_json_accuracy_minimum"
    ] == 0.6
    assert report["current_execution"]["optimizer_steps"] == 0


def test_v3_cpu_preflight_passes_without_model_optimizer_or_heldout_access() -> None:
    report = run_tool_decoder_v3_preflight(
        PROJECT_ROOT,
        tokenizer=_CharacterTokenizer(),
    )
    assert report["passed"] is True
    assert report["training_rows"]["heldout_rows_read"] == 0
    assert report["execution"]["full_model_loaded"] is False
    assert report["execution"]["mps_used"] is False
    assert report["execution"]["optimizer_constructed"] is False
    assert report["execution"]["optimizer_steps"] == 0
    assert report["execution"]["checkpoint_written"] is False
    assert report["execution"]["training_authorized"] is False
    assert report["token_role_audit"]["semantic_weight_fraction_increase"] > 0.2


def test_v3_design_sources_contain_no_heavy_execution_or_artifact_writer() -> None:
    relative_paths = (
        "src/semantic_3d_chat/training/gemma4_tool_decoder_v3_design.py",
        "src/semantic_3d_chat/evaluation/gemma4_tool_decoder_v3_preregistration.py",
        "src/semantic_3d_chat/evaluation/gemma4_tool_decoder_v3_preflight.py",
    )
    combined = "\n".join(
        (PROJECT_ROOT / path).read_text(encoding="utf-8") for path in relative_paths
    )
    forbidden = (
        "AutoModelForCausalLM",
        "torch.optim",
        ".backward(",
        "optimizer.step(",
        "mps.empty_cache",
        "save_file(",
        "torch.save(",
    )
    assert all(fragment not in combined for fragment in forbidden)
    config = (PROJECT_ROOT / "configs/experiments/gemma4_embodied_tool_decoder_v3.yaml").read_text(
        encoding="utf-8"
    )
    assert "training_authorized: false" in config
    assert "checkpoint_write_authorized: false" in config
    assert json.loads(
        (PROJECT_ROOT / V2_TERMINAL_PATH).read_text(encoding="utf-8")
    )["runtime_checkpoint_published"] is False
