from __future__ import annotations

import ast
import json
from copy import deepcopy
from pathlib import Path

import pytest

from semantic_3d_chat.config import load_config
from semantic_3d_chat.evaluation.v18_epoch_selector import (
    EXPECTED_RESIDUAL_ARCHITECTURE,
    EXPECTED_RESIDUAL_PARAMETER_COUNT,
    EXPECTED_STAGE_EXECUTION,
    PINNED_CONFIG_HASH,
    V18EpochSelectorViolation,
    _load_json_strict,
    main,
    summarize_v18_epochs,
)

CONFIG_PATH = "configs/experiments/gemma4_color_mirror_centered_content_gate_v18.yaml"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
TRAIN_SCENES = ["scene_000003", "scene_000004", "scene_000007", "scene_000008"]
TEST_SCENES = ["scene_000005", "scene_000006"]
SOURCE_ADAPTER = "9e15e8c93da083bd23c009bf67cdf4d532d6beb01b12f17f8bf664e2374294c7"
SOURCE_METADATA = "e4cf9134f5ef931df821820c80f96f1839fd2ae9a89b4c06ce4998db330e930e"
INITIAL_RESIDUAL = "f7f6353edb6216029bd155e2baab1b5051c85f297a0e6d6b63210354fe0ff0e0"
FROZEN_SCENE = "690bd890bfda024dbb5c7d3c68087b8113bc3b8ee81dd6143c7eb2a884e7245b"
SELECTION_HASH = "7f0714e3151c9ddb57c1da95a457820a833e490c070881a88a9fee4a9168f933"
PAIR_HASH = "99ee448c23fb71b7269a353a54b2156ac55701847af170597dcc351af15cbcbe"
BANKS = {
    "extension_v13": "4eb90fb9b0bea579d14cfcb0f61ebd5b6d566fd600bd3d5e1bfe5177a39e1b34",
    "inherited_v12": "dec768bed654c8c4e16da0318857543ad54d8f5f68f4d24a9a87cd19ec706594",
}
PREFIXES = {
    "scene_000003": "27ed6afddd70f3bd82c30c9781ac1080f21d0c4f3da3216a1fa81066120c5786",
    "scene_000004": "1f682f48dcc7ac0503886f3c1359d62b9534bf25e27858c10a64b751170ebbf9",
    "scene_000007": "ca22948ab94d16ff46ab23f0ef86f47e7dd555aa152be083625391ba07bcedd8",
    "scene_000008": "ec4d180ca93ac5bcd3344373152ec699020b698d88e48f193affdd2a136875a8",
}


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


def _residual_contract() -> dict:
    return {
        "schema_version": 2,
        "enabled": True,
        "width": 128,
        "fourier_bands": 4,
        "initialization_seed": 18018,
        "expected_initial_state_sha256": INITIAL_RESIDUAL,
        "architecture_version": EXPECTED_RESIDUAL_ARCHITECTURE,
        "gate_temperature": 1.0,
        "spatial_centering": "all_slots_fp32",
        "content_gate": "bias_free_scalar_sigmoid_centered_content",
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
                "expected_initial_state_sha256": BANKS["inherited_v12"],
            },
            {
                "name": "extension_v13",
                "trainable": False,
                "initialization_algorithm": "checkpoint_overwrite",
                "expected_initial_state_sha256": BANKS["extension_v13"],
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
            "selected_ids_sha256": SELECTION_HASH,
            "expected_change_units_selected": 12,
            "expected_change_units_complete": 12,
            "expected_change_units_incomplete": 0,
        },
        "freeze_scene_adapter": True,
        "train_global_scene_residual_only": True,
        "gradient_accumulation": 12,
        "initialize_expected_adapter_sha256": SOURCE_ADAPTER,
        "initialize_expected_metadata_sha256": SOURCE_METADATA,
        "initialize_legacy_lora_into_bank": None,
        "initialize_named_lora_freeze_transition": True,
        "train_scene_ids": TRAIN_SCENES,
        "validation_scene_ids": [],
        "test_scene_ids": TEST_SCENES,
        "counterfactual_pair_unit_count": 12,
        "training_counterfactual_pair_count": 2,
        "training_counterfactual_pair_membership_sha256": PAIR_HASH,
        "global_scene_residual": _residual_contract(),
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
        "schema_version": 3,
        "mode": "named_lora_banks_frozen_plus_zero_output_scene_residual",
        "checkpoint": (
            "data_gemma4/checkpoints/gemma4_color_mirror_decoder_banks_v14_lr2e3/epoch_007"
        ),
        "adapter_sha256": SOURCE_ADAPTER,
        "metadata_sha256": SOURCE_METADATA,
        "expected_adapter_sha256": SOURCE_ADAPTER,
        "expected_metadata_sha256": SOURCE_METADATA,
        "checkpoint_epoch": 7,
        "checkpoint_output_namespace": "gemma4_color_mirror_decoder_banks_v14_lr2e3",
        "checkpoint_config_hash": "93ff12019b76",
        "checkpoint_source_provenance": _source(commit="c" * 40, tree="d" * 40),
        "optimizer_state_loaded": False,
        "history_loaded": False,
        "source_lora_bank_state_sha256": BANKS,
        "all_source_lora_banks_frozen": True,
        "global_scene_residual_initial_state_sha256": INITIAL_RESIDUAL,
        "global_scene_residual_zero_output": True,
    }


def _epoch_artifacts(
    mirrors: list[dict] | None = None, colors: list[dict] | None = None
) -> dict[int, dict]:
    mirrors = mirrors or [
        _pair(full_sides=6, full_units=0, minimum_full=-2.0, minimum_candidate=-2.0)
        for _ in range(4)
    ]
    colors = colors or [_pair(full_sides=12, full_units=6) for _ in range(4)]
    history: list[dict] = []
    result: dict[int, dict] = {}
    for epoch, (color, mirror) in enumerate(zip(colors, mirrors, strict=True), start=1):
        gate = _gate(color, mirror)
        history.append({"epoch": epoch, "pair_candidate_gate": gate})
        result[epoch] = {
            "schema_version": 3,
            "epoch": epoch,
            "global_step": epoch * 12,
            "optimizer_step": epoch,
            "config_hash": PINNED_CONFIG_HASH,
            "output_namespace": "gemma4_color_mirror_centered_content_gate_v18",
            "freeze_scene_adapter": True,
            "train_global_scene_residual_only": True,
            "question_dependent_scene_processing": False,
            "global_scene_residual_parameter_count": EXPECTED_RESIDUAL_PARAMETER_COUNT,
            "global_scene_residual": _residual_contract(),
            "global_scene_residual_initial_state_sha256": INITIAL_RESIDUAL,
            "global_scene_residual_state_sha256": str(epoch) * 64,
            "frozen_scene_state_sha256": FROZEN_SCENE,
            "frozen_lora_bank_state_sha256": BANKS,
            "lora_bank_state_sha256": BANKS,
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
            "training_counterfactual_pair_count": 2,
            "training_counterfactual_pair_membership_sha256": PAIR_HASH,
            "initialize_expected_adapter_sha256": SOURCE_ADAPTER,
            "initialize_expected_metadata_sha256": SOURCE_METADATA,
            "v18_stage_execution": deepcopy(EXPECTED_STAGE_EXECUTION),
            "source_provenance": _source(),
            "initialization_provenance": _initialization(),
            "global_scene_residual_zero_output_equivalence": {
                "verified": True,
                "question_dependent_scene_processing": False,
                "scene_count": 4,
                "scene_prefixes": {
                    scene_id: {
                        "core_prefix_sha256": digest,
                        "adapted_prefix_sha256": digest,
                    }
                    for scene_id, digest in PREFIXES.items()
                },
            },
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
            },
            "history": deepcopy(history),
            "pair_candidate_gate": gate,
        }
    return result


def test_ranks_only_color_eligible_epochs_by_six_fields_then_lower_epoch(config: dict) -> None:
    mirrors = [
        _pair(full_sides=12, full_units=6, mean_full=9.0, minimum_full=2.0),
        _pair(full_sides=9, full_units=3, mean_full=1.0, minimum_full=-0.5),
        _pair(full_sides=8, full_units=3, mean_full=5.0, minimum_full=-0.2),
        _pair(full_sides=11, full_units=5, mean_full=8.0, minimum_full=1.0),
    ]
    colors = [
        _pair(full_sides=12, full_units=6, minimum_candidate=0.0),
        _pair(full_sides=12, full_units=6),
        _pair(full_sides=12, full_units=6),
        _pair(full_sides=12, full_units=6),
    ]

    summary = summarize_v18_epochs(config, _selection(), _epoch_artifacts(mirrors, colors))

    assert summary["eligible_epoch_count"] == 3
    assert [row["epoch"] for row in summary["ranking"]] == [4, 2, 3]
    assert summary["selected_epoch"] == 4
    assert summary["continuation_authorized"] is True
    assert summary["greedy_audit_authorized"] is False


def test_lower_epoch_is_exact_final_tiebreaker(config: dict) -> None:
    tied = _pair(
        full_sides=8,
        full_units=2,
        candidate_sides=9,
        candidate_units=2,
        mean_full=0.25,
        minimum_full=-0.5,
    )
    weaker = _pair(full_sides=7, full_units=1, minimum_full=-1.0)
    artifacts = _epoch_artifacts([deepcopy(tied), deepcopy(tied), weaker, weaker])

    summary = summarize_v18_epochs(config, _selection(), artifacts)

    assert [row["epoch"] for row in summary["ranking"][:2]] == [1, 2]
    assert summary["selected_epoch"] == 1
    assert summary["continuation_authorized"] is True


def test_full_teacher_gate_is_only_path_to_greedy(config: dict) -> None:
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
    artifacts = _epoch_artifacts([deepcopy(perfect) for _ in range(4)])

    summary = summarize_v18_epochs(config, _selection(), artifacts)

    assert summary["selected_epoch"] == 1
    assert summary["full_teacher_gate_passed"] is True
    assert summary["greedy_audit_authorized"] is True
    assert summary["greedy_audit_forbidden"] is False
    assert summary["decision"] == "full_teacher_gate_passed_greedy_audit_allowed"


@pytest.mark.parametrize("pair_name", ["color", "mirror"])
@pytest.mark.parametrize("minimum_name", ["minimum_full", "minimum_candidate"])
def test_full_gate_requires_all_four_minimum_margins_positive(
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
    colors = [_pair(**color_kwargs) for _ in range(4)]
    mirrors = [_pair(**mirror_kwargs) for _ in range(4)]

    summary = summarize_v18_epochs(config, _selection(), _epoch_artifacts(mirrors, colors))

    assert summary["full_teacher_gate_passed"] is False
    assert summary["greedy_audit_authorized"] is False
    assert summary["greedy_audit_forbidden"] is True


def test_no_color_eligible_epoch_authorizes_nothing(config: dict) -> None:
    colors = [
        _pair(full_sides=11, full_units=5, minimum_full=1.0, minimum_candidate=1.0)
        for _ in range(4)
    ]

    summary = summarize_v18_epochs(config, _selection(), _epoch_artifacts(colors=colors))

    assert summary["selected_epoch"] is None
    assert summary["continuation_authorized"] is False
    assert summary["greedy_audit_authorized"] is False
    assert summary["decision"] == "no_color_eligible_epoch_no_extension_no_greedy"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda artifacts: artifacts[2].update(config_hash="0" * 12), "config_hash"),
        (
            lambda artifacts: artifacts[3].update(global_scene_residual_parameter_count=400_000),
            "parameter_count",
        ),
        (
            lambda artifacts: artifacts[1]["global_scene_residual"].update(
                architecture_version="global_mean_v1"
            ),
            "global_scene_residual",
        ),
        (
            lambda artifacts: artifacts[4].update(frozen_scene_state_sha256="0" * 64),
            "frozen_scene_state",
        ),
        (
            lambda artifacts: artifacts[2]["source_provenance"].update(is_clean=False),
            "clean",
        ),
        (
            lambda artifacts: artifacts[2]["v18_stage_execution"].update(
                stage_2_load_history=False
            ),
            "v18_stage_execution",
        ),
        (
            lambda artifacts: artifacts[3]["pair_candidate_gate"].update(
                free_generation_evaluated=True
            ),
            "top-level pair gate",
        ),
        (
            lambda artifacts: artifacts[4]["history"][-1]["pair_candidate_gate"]["by_pair"][
                "pair_000003"
            ].update(free_generation_evaluated=True),
            "free_generation_evaluated",
        ),
    ],
)
def test_rejects_checkpoint_architecture_source_or_protocol_mutation(
    config: dict, mutation, message: str
) -> None:
    artifacts = _epoch_artifacts()
    mutation(artifacts)

    with pytest.raises(V18EpochSelectorViolation, match=message):
        summarize_v18_epochs(config, _selection(), artifacts)


def test_rejects_missing_or_extra_epoch_artifact(config: dict) -> None:
    artifacts = _epoch_artifacts()
    del artifacts[4]

    with pytest.raises(V18EpochSelectorViolation, match="exactly epoch artifacts 1,2,3,4"):
        summarize_v18_epochs(config, _selection(), artifacts)


def test_rejects_selection_hash_or_source_mismatch(config: dict) -> None:
    selection = _selection()
    selection["train"]["selected_ids_sha256"] = "0" * 64

    with pytest.raises(V18EpochSelectorViolation, match="selected_ids_sha256"):
        summarize_v18_epochs(config, selection, _epoch_artifacts())

    selection = _selection()
    selection["source_provenance"]["head_commit"] = "e" * 40
    with pytest.raises(V18EpochSelectorViolation, match="exact clean source provenance"):
        summarize_v18_epochs(config, selection, _epoch_artifacts())


def test_rejects_history_drift_across_resumed_checkpoint_metadata(config: dict) -> None:
    artifacts = _epoch_artifacts()
    artifacts[3]["history"][0]["pair_candidate_gate"]["by_pair"]["pair_000003"][
        "mean_first_answer_token_target_vs_best_other_logit_margin"
    ] += 0.125

    with pytest.raises(V18EpochSelectorViolation, match="preserve exact cumulative history"):
        summarize_v18_epochs(config, _selection(), artifacts)


def test_rejects_nonfinite_metric_and_unchanged_residual_state(config: dict) -> None:
    artifacts = _epoch_artifacts()
    artifacts[2]["history"][-1]["pair_candidate_gate"]["by_pair"]["pair_000003"][
        "mean_own_vs_alternate_candidate_logit_margin"
    ] = float("nan")
    with pytest.raises(V18EpochSelectorViolation, match="NaN or infinity"):
        summarize_v18_epochs(config, _selection(), artifacts)

    artifacts = _epoch_artifacts()
    artifacts[1]["global_scene_residual_state_sha256"] = INITIAL_RESIDUAL
    with pytest.raises(V18EpochSelectorViolation, match="did not change"):
        summarize_v18_epochs(config, _selection(), artifacts)


def test_rejects_any_config_drift_before_reading_epoch_metrics(config: dict) -> None:
    modified = deepcopy(config)
    modified["v18_screen"]["epoch_tiebreaker"] = "higher_epoch"

    with pytest.raises(V18EpochSelectorViolation, match="config hash mismatch"):
        summarize_v18_epochs(modified, _selection(), _epoch_artifacts())


def test_json_loader_rejects_oracle_path_before_open(tmp_path: Path) -> None:
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    path = oracle / "metadata.json"
    path.write_text("{}")

    with pytest.raises(V18EpochSelectorViolation, match="runtime/oracle"):
        _load_json_strict(path)


def test_cli_writes_report_from_only_config_selection_and_four_metadata_files(
    tmp_path: Path,
) -> None:
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(_selection()))
    epoch_args: list[str] = []
    for epoch, artifact in _epoch_artifacts().items():
        path = tmp_path / f"epoch_{epoch:03d}.json"
        path.write_text(json.dumps(artifact))
        epoch_args.extend(["--epoch", f"{epoch}={path}"])
    output = tmp_path / "decision.json"

    result = main(
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

    assert result == 0
    report = json.loads(output.read_text())
    assert report["report_only"] is True
    assert report["model_inference_executed"] is False
    assert report["checkpoint_tensor_state_loaded"] is False
    assert report["greedy_audit_authorized"] is False


def test_selector_module_has_no_model_runtime_or_oracle_imports() -> None:
    source_path = (
        Path(__file__).parents[1]
        / "src"
        / "semantic_3d_chat"
        / "evaluation"
        / "v18_epoch_selector.py"
    )
    tree = ast.parse(source_path.read_text())
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)
    forbidden = {
        name
        for name in imports
        if name in {"torch", "transformers", "safetensors"}
        or name.startswith(
            (
                "semantic_3d_chat.chat",
                "semantic_3d_chat.data",
                "semantic_3d_chat.language",
                "semantic_3d_chat.robot",
            )
        )
    }
    assert forbidden == set()
