from __future__ import annotations

import ast
import json
from copy import deepcopy
from pathlib import Path

import pytest

from semantic_3d_chat.config import load_config
from semantic_3d_chat.evaluation.v19_epoch_selector import (
    EXPECTED_FROZEN_BANKS,
    EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256,
    EXPECTED_FROZEN_SCENE_SHA256,
    EXPECTED_GLOBAL_RESIDUAL_PARAMETER_COUNT,
    EXPECTED_INITIAL_GLOBAL_RESIDUAL_SHA256,
    EXPECTED_INITIAL_SIGNED_X_SHA256,
    EXPECTED_PAIR_MEMBERSHIP_SHA256,
    EXPECTED_PAIR_SELECTION_SHA256,
    EXPECTED_SELECTION_SHA256,
    EXPECTED_SIGNED_X_PARAMETER_COUNT,
    EXPECTED_SOURCE_ADAPTER_SHA256,
    EXPECTED_SOURCE_METADATA_SHA256,
    PINNED_CONFIG_HASH,
    V19EpochSelectorViolation,
    _load_json_strict,
    main,
    summarize_v19_epochs,
)

CONFIG_PATH = "configs/experiments/gemma4_color_mirror_signed_x_moment_v19.yaml"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
TRAIN_SCENES = ["scene_000003", "scene_000004", "scene_000007", "scene_000008"]
TEST_SCENES = ["scene_000005", "scene_000006"]
OBJECTIVE_POLICY_SHA256 = "81da45626db8e09fab95d062271a0a22d182589145b945ab51fb8c4cf4490734"
OBJECTIVE_COVERAGE_SHA256 = "6747b11e2803aefbe9622b97346e4c3e2e0e2eb7e2e9b8a3e64c6755b4d052cc"


@pytest.fixture
def config() -> dict:
    return load_config(CONFIG_PATH)


def _source(*, commit: str = "a" * 40, tree: str = "b" * 40, clean: bool = True) -> dict:
    return {
        "schema_version": 1,
        "scope": "repository_excluding_generated_artifacts_v1",
        "available": True,
        "head_commit": commit,
        "head_tree": tree,
        "is_clean": clean,
        "tracked_diff_sha256": EMPTY_SHA256,
    }


def _global_contract() -> dict:
    return {
        "schema_version": 2,
        "enabled": True,
        "width": 128,
        "fourier_bands": 4,
        "initialization_seed": 18018,
        "expected_initial_state_sha256": EXPECTED_INITIAL_GLOBAL_RESIDUAL_SHA256,
        "architecture_version": "zero_spatial_mean_content_gate_v1",
        "gate_temperature": 1.0,
        "spatial_centering": "all_slots_fp32",
        "content_gate": "bias_free_scalar_sigmoid_centered_content",
    }


def _signed_contract() -> dict:
    return {
        "schema_version": 1,
        "enabled": True,
        "architecture_version": "signed_x_moment_v1",
        "expected_initial_state_sha256": EXPECTED_INITIAL_SIGNED_X_SHA256,
        "spatial_statistic": "centered_unit_rms_signed_x_moment",
        "spatial_centering": "all_slots_fp32",
        "trainable_surface": "bias_free_output_projection_only",
    }


def _objective_policy() -> dict:
    return {
        "schema_version": 1,
        "configured": True,
        "allow_unlisted_pair_ids": False,
        "legacy_default": {
            "role": "legacy_global",
            "language_nll_weight": 1.0,
            "candidate_hinge_weight": 8.0,
            "candidate_margin": 1.0,
            "full_vocab_hinge_weight": 2.0,
            "full_vocab_margin": 1.0,
        },
        "by_pair": {
            "pair_000001": {
                "role": "retention_control",
                "language_nll_weight": 0.0,
                "candidate_hinge_weight": 8.0,
                "candidate_margin": 0.25,
                "full_vocab_hinge_weight": 2.0,
                "full_vocab_margin": 0.25,
            },
            "pair_000003": {
                "role": "signed_target",
                "language_nll_weight": 0.0,
                "candidate_hinge_weight": 8.0,
                "candidate_margin": 1.0,
                "full_vocab_hinge_weight": 2.0,
                "full_vocab_margin": 1.0,
            },
        },
        "contract_sha256": OBJECTIVE_POLICY_SHA256,
    }


def _objective_coverage() -> dict:
    return {
        "schema_version": 1,
        "selected_pair_ids": ["pair_000001", "pair_000003"],
        "configured_pair_ids": ["pair_000001", "pair_000003"],
        "unlisted_pair_ids": [],
        "allow_unlisted_pair_ids": False,
        "resolved_by_pair": deepcopy(_objective_policy()["by_pair"]),
        "complete": True,
        "coverage_sha256": OBJECTIVE_COVERAGE_SHA256,
    }


def _lora_contract() -> dict:
    return {
        "schema_version": 2,
        "enabled": True,
        "banks": [
            {
                "name": "inherited_v12",
                "trainable": False,
                "initialization_algorithm": "checkpoint_overwrite",
                "expected_initial_state_sha256": EXPECTED_FROZEN_BANKS["inherited_v12"],
            },
            {
                "name": "extension_v13",
                "trainable": False,
                "initialization_algorithm": "checkpoint_overwrite",
                "expected_initial_state_sha256": EXPECTED_FROZEN_BANKS["extension_v13"],
            },
        ],
    }


def _selection() -> dict:
    return {
        "schema_version": 1,
        "source_provenance": _source(),
        "train": {
            "available_count": 24,
            "selected_count": 24,
            "selected_ids_sha256": EXPECTED_SELECTION_SHA256,
            "expected_change_units_selected": 12,
            "expected_change_units_complete": 12,
            "expected_change_units_incomplete": 0,
        },
        "freeze_scene_adapter": True,
        "train_global_scene_residual_only": False,
        "train_signed_x_scene_residual_only": True,
        "gradient_accumulation": 12,
        "initialize_expected_adapter_sha256": EXPECTED_SOURCE_ADAPTER_SHA256,
        "initialize_expected_metadata_sha256": EXPECTED_SOURCE_METADATA_SHA256,
        "initialize_expected_global_scene_residual_state_sha256": (
            EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256
        ),
        "initialize_legacy_lora_into_bank": None,
        "initialize_named_lora_freeze_transition": False,
        "initialize_source_residual_into_frozen_base": True,
        "train_scene_ids": TRAIN_SCENES,
        "validation_scene_ids": [],
        "test_scene_ids": TEST_SCENES,
        "counterfactual_pair_unit_count": 12,
        "counterfactual_pair_unit_selection_sha256": EXPECTED_PAIR_SELECTION_SHA256,
        "training_counterfactual_pair_count": 2,
        "training_counterfactual_pair_membership_sha256": EXPECTED_PAIR_MEMBERSHIP_SHA256,
        "global_scene_residual": _global_contract(),
        "signed_x_scene_residual": _signed_contract(),
        "pair_curriculum": {
            "enabled": True,
            "pair_only": True,
            "pair_only_scene_ids": TRAIN_SCENES,
            "max_units_per_pair": 6,
            "batch_fraction": 1.0,
            "units_per_batch": 1,
            "ranking_mode": "candidate_logit",
            "ranking_margin": 1.0,
            "ranking_weight": 8.0,
            "full_vocab_ranking_margin": 1.0,
            "full_vocab_ranking_weight": 2.0,
            "gate_enabled": True,
            "gate_every_epochs": 1,
            "gate_stop_when_passed": False,
            "gate_first_answer_token_top1_accuracy": 1.0,
            "objective_policy": _objective_policy(),
            "objective_policy_coverage": _objective_coverage(),
        },
        "lora": _lora_contract(),
    }


def _pair(
    *,
    full_sides: int,
    full_units: int,
    candidate_sides: int | None = None,
    candidate_units: int | None = None,
    mean_full: float = 1.0,
    minimum_full: float = 0.5,
    mean_candidate: float = 1.0,
    minimum_candidate: float = 0.5,
) -> dict:
    candidate_sides = full_sides if candidate_sides is None else candidate_sides
    candidate_units = full_units if candidate_units is None else candidate_units
    return {
        "evaluation_type": "teacher_forced_same_distribution_candidate_logit_ranking",
        "ranking_mode": "candidate_logit",
        "same_next_token_distribution": True,
        "shared_candidate_tokens_excluded": True,
        "free_generation_evaluated": False,
        "first_answer_token_full_vocab_evaluated": True,
        "unit_count": 6,
        "side_count": 12,
        "first_answer_token_top1_unit_accuracy": full_units / 6,
        "first_answer_token_top1_accuracy": full_sides / 12,
        "changed_unit_accuracy": candidate_units / 6,
        "side_accuracy": candidate_sides / 12,
        "mean_first_answer_token_target_vs_best_other_logit_margin": mean_full,
        "minimum_first_answer_token_target_vs_best_other_logit_margin": minimum_full,
        "mean_own_vs_alternate_candidate_logit_margin": mean_candidate,
        "minimum_own_vs_alternate_candidate_logit_margin": minimum_candidate,
    }


def _gate(color: dict, mirror: dict) -> dict:
    return {
        "evaluation_type": "teacher_forced_same_distribution_candidate_logit_ranking",
        "ranking_mode": "candidate_logit",
        "same_next_token_distribution": True,
        "shared_candidate_tokens_excluded": True,
        "free_generation_evaluated": False,
        "first_answer_token_full_vocab_evaluated": True,
        "pair_count": 2,
        "unit_count": 12,
        "side_count": 24,
        "by_pair": {"pair_000001": color, "pair_000003": mirror},
    }


def _initialization() -> dict:
    return {
        "schema_version": 4,
        "mode": "frozen_v18_residual_base_plus_zero_output_signed_x_residual",
        "checkpoint": (
            "data_gemma4/checkpoints/gemma4_color_mirror_centered_content_gate_v18/epoch_004"
        ),
        "adapter_sha256": EXPECTED_SOURCE_ADAPTER_SHA256,
        "metadata_sha256": EXPECTED_SOURCE_METADATA_SHA256,
        "expected_adapter_sha256": EXPECTED_SOURCE_ADAPTER_SHA256,
        "expected_metadata_sha256": EXPECTED_SOURCE_METADATA_SHA256,
        "checkpoint_epoch": 4,
        "checkpoint_output_namespace": "gemma4_color_mirror_centered_content_gate_v18",
        "checkpoint_config_hash": "38b0fd8e679d",
        "checkpoint_source_provenance": _source(commit="c" * 40, tree="d" * 40),
        "optimizer_state_loaded": False,
        "history_loaded": False,
        "source_global_scene_residual_state_sha256": EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256,
        "expected_source_global_scene_residual_state_sha256": (
            EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256
        ),
        "global_scene_residual_frozen": True,
        "signed_x_scene_residual_initial_state_sha256": EXPECTED_INITIAL_SIGNED_X_SHA256,
        "signed_x_scene_residual_zero_output": True,
    }


def _zero_equivalence() -> dict:
    prefixes = {}
    for index, scene_id in enumerate(TRAIN_SCENES, start=5):
        digest = f"{index:x}" * 64
        prefixes[scene_id] = {
            "v18_base_prefix_sha256": digest,
            "signed_x_adapted_prefix_sha256": digest,
        }
    return {
        "verified": True,
        "base": "loaded_frozen_global_scene_residual",
        "question_dependent_scene_processing": False,
        "all_scene_slots_accounted": True,
        "scene_count": 4,
        "scene_prefixes": prefixes,
    }


def _epoch_artifacts(
    mirrors: list[dict] | None = None, colors: list[dict] | None = None
) -> dict[int, dict]:
    mirrors = mirrors or [
        _pair(
            full_sides=6,
            full_units=0,
            minimum_full=-2.0,
            minimum_candidate=-2.0,
        )
        for _ in range(4)
    ]
    colors = colors or [_pair(full_sides=12, full_units=6) for _ in range(4)]
    history: list[dict] = []
    result: dict[int, dict] = {}
    for epoch, (color, mirror) in enumerate(zip(colors, mirrors, strict=True), start=1):
        gate = _gate(color, mirror)
        train_loss = float(20 - epoch)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "pair_batch_count": 12,
                "pair_batch_fraction": 1.0,
                "pair_candidate_gate": gate,
            }
        )
        result[epoch] = {
            "schema_version": 3,
            "epoch": epoch,
            "train_loss": train_loss,
            "global_step": epoch * 12,
            "optimizer_step": epoch,
            "config_hash": PINNED_CONFIG_HASH,
            "output_namespace": "gemma4_color_mirror_signed_x_moment_v19",
            "freeze_scene_adapter": True,
            "train_global_scene_residual_only": False,
            "train_signed_x_scene_residual_only": True,
            "question_dependent_scene_processing": False,
            "global_scene_residual_parameter_count": (EXPECTED_GLOBAL_RESIDUAL_PARAMETER_COUNT),
            "global_scene_residual": _global_contract(),
            "global_scene_residual_initial_state_sha256": (EXPECTED_INITIAL_GLOBAL_RESIDUAL_SHA256),
            "global_scene_residual_state_sha256": (EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256),
            "frozen_global_scene_residual_state_sha256": (EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256),
            "global_scene_residual_zero_output_equivalence": None,
            "signed_x_scene_residual_parameter_count": EXPECTED_SIGNED_X_PARAMETER_COUNT,
            "signed_x_scene_residual": _signed_contract(),
            "signed_x_scene_residual_initial_state_sha256": EXPECTED_INITIAL_SIGNED_X_SHA256,
            "signed_x_scene_residual_state_sha256": f"{epoch:x}" * 64,
            "signed_x_scene_residual_zero_output_equivalence": _zero_equivalence(),
            "frozen_scene_state_sha256": EXPECTED_FROZEN_SCENE_SHA256,
            "frozen_lora_bank_state_sha256": EXPECTED_FROZEN_BANKS,
            "lora_bank_state_sha256": EXPECTED_FROZEN_BANKS,
            "lora_trainable_parameter_count": 0,
            "lora": _lora_contract(),
            "scene_ids": TRAIN_SCENES,
            "train_scene_ids": TRAIN_SCENES,
            "validation_scene_ids": [],
            "test_scene_ids": TEST_SCENES,
            "scene_latents": 256,
            "scene_model_dim": 384,
            "semantic_dim": 3072,
            "language_hidden_dim": 1536,
            "language_model_id": "google/gemma-4-E2B-it",
            "language_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
            "language_backend": "gemma4",
            "scene_encoder_architecture_version": "signal_preserving_resampler_v3",
            "scene_prefix_after_bos": True,
            "scene_boundary_mode": "gemma4_native_image",
            "gradient_accumulation": 12,
            "counterfactual_pair_unit_count": 12,
            "counterfactual_pair_unit_selection_sha256": EXPECTED_PAIR_SELECTION_SHA256,
            "training_counterfactual_pair_count": 2,
            "training_counterfactual_pair_membership_sha256": (EXPECTED_PAIR_MEMBERSHIP_SHA256),
            "initialize_expected_adapter_sha256": EXPECTED_SOURCE_ADAPTER_SHA256,
            "initialize_expected_metadata_sha256": EXPECTED_SOURCE_METADATA_SHA256,
            "initialize_expected_global_scene_residual_state_sha256": (
                EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256
            ),
            "initialize_source_residual_into_frozen_base": True,
            "source_provenance": _source(),
            "initialization_provenance": _initialization(),
            "pair_curriculum": {
                "enabled": True,
                "pair_only": True,
                "pair_only_scene_ids": TRAIN_SCENES,
                "max_units_per_pair": 6,
                "ranking_weight": 8.0,
                "ranking_margin": 1.0,
                "ranking_mode": "candidate_logit",
                "full_vocab_ranking_weight": 2.0,
                "full_vocab_ranking_margin": 1.0,
                "batch_fraction": 1.0,
                "units_per_batch": 1,
                "steps_per_epoch": 12,
                "gate_enabled": True,
                "objective_policy": _objective_policy(),
                "objective_policy_coverage": _objective_coverage(),
            },
            "pair_gate_policy": {
                "stop_when_passed": False,
                "first_answer_token_top1_accuracy_threshold": 1.0,
            },
            "history": deepcopy(history),
            "pair_candidate_gate": gate,
        }
    return result


def test_ranks_color_eligible_epochs_by_six_mirror_fields_then_lower_epoch(
    config: dict,
) -> None:
    mirrors = [
        _pair(
            full_sides=12,
            full_units=6,
            mean_full=3.0,
            minimum_full=2.0,
            candidate_sides=12,
            candidate_units=6,
        ),
        _pair(
            full_sides=9,
            full_units=3,
            minimum_full=-0.5,
            candidate_sides=12,
            candidate_units=6,
        ),
        _pair(
            full_sides=8,
            full_units=3,
            minimum_full=-0.2,
            candidate_sides=12,
            candidate_units=6,
        ),
        _pair(
            full_sides=11,
            full_units=5,
            minimum_full=-0.1,
            candidate_sides=12,
            candidate_units=6,
        ),
    ]
    colors = [
        _pair(full_sides=12, full_units=6, minimum_candidate=0.0),
        *[_pair(full_sides=12, full_units=6) for _ in range(3)],
    ]

    summary = summarize_v19_epochs(config, _selection(), _epoch_artifacts(mirrors, colors))

    assert summary["eligible_epoch_count"] == 3
    assert [row["epoch"] for row in summary["ranking"]] == [4, 2, 3]
    assert summary["selected_epoch"] == 4
    assert summary["continuation_authorized"] is True
    assert summary["greedy_audit_authorized"] is False


def test_lower_epoch_is_final_tiebreaker_and_threshold_controls_continuation(
    config: dict,
) -> None:
    tied = _pair(
        full_sides=8,
        full_units=2,
        candidate_sides=9,
        candidate_units=2,
        minimum_full=-0.5,
        minimum_candidate=-0.25,
    )
    weaker = _pair(
        full_sides=7,
        full_units=1,
        minimum_full=-1.0,
        minimum_candidate=-1.0,
    )
    summary = summarize_v19_epochs(
        config,
        _selection(),
        _epoch_artifacts([deepcopy(tied), deepcopy(tied), weaker, weaker]),
    )

    assert [row["epoch"] for row in summary["ranking"][:2]] == [1, 2]
    assert summary["selected_epoch"] == 1
    assert summary["continuation_authorized"] is True


def test_full_teacher_gate_is_the_only_path_to_greedy(config: dict) -> None:
    perfect = _pair(
        full_sides=12,
        full_units=6,
        candidate_sides=12,
        candidate_units=6,
        mean_full=2.0,
        minimum_full=0.25,
        mean_candidate=2.0,
        minimum_candidate=0.25,
    )
    summary = summarize_v19_epochs(
        config,
        _selection(),
        _epoch_artifacts([deepcopy(perfect) for _ in range(4)]),
    )

    assert summary["selected_epoch"] == 1
    assert summary["full_teacher_gate_passed"] is True
    assert summary["greedy_audit_authorized"] is True
    assert summary["decision"] == "full_teacher_gate_passed_greedy_audit_allowed"


@pytest.mark.parametrize("pair_name", ["color", "mirror"])
@pytest.mark.parametrize("minimum_name", ["minimum_full", "minimum_candidate"])
def test_full_gate_requires_all_four_minima_positive(
    config: dict, pair_name: str, minimum_name: str
) -> None:
    kwargs = {
        "full_sides": 12,
        "full_units": 6,
        "candidate_sides": 12,
        "candidate_units": 6,
        "minimum_full": 0.25,
        "minimum_candidate": 0.25,
    }
    color_kwargs = dict(kwargs)
    mirror_kwargs = dict(kwargs)
    (color_kwargs if pair_name == "color" else mirror_kwargs)[minimum_name] = 0.0
    summary = summarize_v19_epochs(
        config,
        _selection(),
        _epoch_artifacts(
            [_pair(**mirror_kwargs) for _ in range(4)],
            [_pair(**color_kwargs) for _ in range(4)],
        ),
    )

    assert summary["full_teacher_gate_passed"] is False
    assert summary["greedy_audit_authorized"] is False


def test_no_color_eligible_epoch_authorizes_nothing(config: dict) -> None:
    colors = [
        _pair(
            full_sides=11,
            full_units=5,
            minimum_full=-0.1,
            minimum_candidate=-0.1,
        )
        for _ in range(4)
    ]
    summary = summarize_v19_epochs(config, _selection(), _epoch_artifacts(colors=colors))

    assert summary["selected_epoch"] is None
    assert summary["continuation_authorized"] is False
    assert summary["greedy_audit_authorized"] is False
    assert summary["decision"] == "no_color_eligible_epoch_no_extension_no_greedy"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows[2].update(config_hash="0" * 12), "config_hash"),
        (
            lambda rows: rows[3].update(global_scene_residual_state_sha256="0" * 64),
            "global_scene_residual_state",
        ),
        (
            lambda rows: rows[4].update(frozen_scene_state_sha256="0" * 64),
            "frozen_scene_state",
        ),
        (
            lambda rows: rows[2].update(signed_x_scene_residual_parameter_count=1),
            "signed_x_scene_residual_parameter_count",
        ),
        (
            lambda rows: rows[3]["signed_x_scene_residual"].update(trainable_surface="anything"),
            "signed_x_scene_residual",
        ),
        (
            lambda rows: rows[2]["source_provenance"].update(is_clean=False),
            "clean",
        ),
        (
            lambda rows: rows[3]["pair_curriculum"]["objective_policy"]["by_pair"][
                "pair_000001"
            ].update(language_nll_weight=1.0),
            "pair_curriculum",
        ),
        (
            lambda rows: rows[4]["history"][-1]["pair_candidate_gate"]["by_pair"][
                "pair_000003"
            ].update(free_generation_evaluated=True),
            "free_generation_evaluated",
        ),
    ],
)
def test_rejects_architecture_frozen_state_policy_or_protocol_mutation(
    config: dict, mutation, message: str
) -> None:
    artifacts = _epoch_artifacts()
    mutation(artifacts)

    with pytest.raises(V19EpochSelectorViolation, match=message):
        summarize_v19_epochs(config, _selection(), artifacts)


def test_rejects_missing_epoch_history_drift_and_repeated_signed_state(config: dict) -> None:
    missing = _epoch_artifacts()
    del missing[4]
    with pytest.raises(V19EpochSelectorViolation, match="exactly epoch artifacts 1,2,3,4"):
        summarize_v19_epochs(config, _selection(), missing)

    drift = _epoch_artifacts()
    drift[3]["history"][0]["train_loss"] += 0.125
    with pytest.raises(V19EpochSelectorViolation, match="preserve exact cumulative history"):
        summarize_v19_epochs(config, _selection(), drift)

    repeated = _epoch_artifacts()
    repeated[3]["signed_x_scene_residual_state_sha256"] = repeated[2][
        "signed_x_scene_residual_state_sha256"
    ]
    with pytest.raises(V19EpochSelectorViolation, match="repeats or rolls back"):
        summarize_v19_epochs(config, _selection(), repeated)


def test_rejects_nonfinite_or_internally_inconsistent_teacher_metrics(config: dict) -> None:
    nonfinite = _epoch_artifacts()
    nonfinite[2]["history"][-1]["pair_candidate_gate"]["by_pair"]["pair_000003"][
        "mean_own_vs_alternate_candidate_logit_margin"
    ] = float("nan")
    with pytest.raises(V19EpochSelectorViolation, match="NaN or infinity"):
        summarize_v19_epochs(config, _selection(), nonfinite)

    inconsistent = _epoch_artifacts()
    pair = inconsistent[2]["history"][-1]["pair_candidate_gate"]["by_pair"]["pair_000003"]
    pair["minimum_first_answer_token_target_vs_best_other_logit_margin"] = 0.1
    with pytest.raises(V19EpochSelectorViolation, match="contradicts its accuracy"):
        summarize_v19_epochs(config, _selection(), inconsistent)


def test_rejects_selection_or_config_drift_before_ranking(config: dict) -> None:
    selection = _selection()
    selection["pair_curriculum"]["objective_policy_coverage"]["complete"] = False
    with pytest.raises(V19EpochSelectorViolation, match="pair_curriculum"):
        summarize_v19_epochs(config, selection, _epoch_artifacts())

    modified = deepcopy(config)
    modified["v19_screen"]["stage_1_stop_required"] = False
    with pytest.raises(V19EpochSelectorViolation, match="config hash mismatch"):
        summarize_v19_epochs(modified, _selection(), _epoch_artifacts())


def test_json_loader_rejects_oracle_path_before_open(tmp_path: Path) -> None:
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    path = oracle / "metadata.json"
    path.write_text("{}")

    with pytest.raises(V19EpochSelectorViolation, match="runtime/oracle"):
        _load_json_strict(path)


def test_cli_writes_report_using_only_selection_and_four_metadata_files(
    tmp_path: Path,
) -> None:
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(_selection()))
    epoch_args: list[str] = []
    for epoch, artifact in _epoch_artifacts().items():
        path = tmp_path / f"epoch_{epoch:03d}_metadata.json"
        path.write_text(json.dumps(artifact))
        epoch_args.extend(["--epoch", f"{epoch}={path}"])
    output = tmp_path / "report.json"

    assert (
        main(
            [
                "--config",
                CONFIG_PATH,
                "--selection",
                str(selection_path),
                *epoch_args,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    assert report["selector_type"] == "strict_v19_signed_x_moment_epoch_selector"
    assert report["checkpoint_tensor_state_loaded"] is False
    assert report["cumulative_update_evidence"]["history_prefixes_exact"] is True


def test_selector_module_has_no_runtime_or_model_imports() -> None:
    path = Path(__file__).parents[1] / ("src/semantic_3d_chat/evaluation/v19_epoch_selector.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    forbidden_fragments = (
        ".chat",
        ".language",
        ".mapping",
        ".scene_encoder",
        ".training",
        ".data",
    )
    assert not any(fragment in module for module in imported for fragment in forbidden_fragments)
