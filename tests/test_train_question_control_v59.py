from __future__ import annotations

from pathlib import Path

from semantic_3d_chat.training.train_question_control_v59 import (
    _derive_subset_prefix_cache,
    _parser,
    _validate_args,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_A = "a" * 64
_B = "b" * 64


def _args() -> list[str]:
    return [
        "--base-runtime-config",
        "configs/runtime/gemma4_v56_question_control.yaml",
        "--base-checkpoint",
        "data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000",
        "--source-control-checkpoint",
        "data_gemma4/checkpoints/gemma4_v58_question_control_pair31_32",
        "--anchor-teacher-artifact",
        "data_gemma4/training/v58_teachers_pair31_32",
        "--anchor-teacher-source-control-checkpoint",
        "data_gemma4/checkpoints/gemma4_v57_question_control_pair31_32_conditioning_u40",
        "--train-qa",
        "data_diverse52/qa/train.jsonl",
        *(value for scene in (31, 32, 33, 34, 37, 38) for value in ("--scene-id", f"scene_{scene:06d}")),
        "--full-prefix-cache",
        "data_gemma4/scene_tokens/v56_question_control_full_prefixes",
        "--subset-prefix-cache",
        "data_gemma4/scene_tokens/v59_locked_six_prefixes",
        "--teacher-cache",
        "data_gemma4/training/v59_expansion_teachers",
        "--output-checkpoint",
        "data_gemma4/checkpoints/v59_runtime",
        "--training-report",
        "reports/gemma4/metrics/v59_training.json",
    ]


def test_v59_cli_locks_six_scenes_and_compact_architecture() -> None:
    args = _parser().parse_args(_args())
    _validate_args(args)
    assert args.moment_count == 8
    assert args.interaction_dim == 24
    assert args.output_rank == 64
    assert args.maximum_control_rms == 0.2
    assert args.initial_control_rms == 0.075
    assert args.distill_epochs == 80


def test_v59_subset_cache_derives_exact_six_from_validated_full_cache(
    tmp_path: Path,
) -> None:
    source = PROJECT_ROOT / "data_gemma4/scene_tokens/v56_question_control_full_prefixes"
    destination = tmp_path / "six_prefixes"
    # Use the real immutable prefix cache's measured identities, not fake data.
    import json

    manifest = json.loads((source / "manifest.json").read_text())
    prefixes, subset, created = _derive_subset_prefix_cache(
        source_cache=source,
        destination_cache=destination,
        base_checkpoint_sha256=manifest["base_checkpoint_sha256"],
        base_runtime_config_sha256=manifest["base_runtime_config_sha256"],
    )
    assert created is True
    assert set(prefixes) == {
        "scene_000031",
        "scene_000032",
        "scene_000033",
        "scene_000034",
        "scene_000037",
        "scene_000038",
    }
    assert subset["scene_count"] == 6
    assert all(tuple(prefix.shape) == (1, 258, 1536) for prefix in prefixes.values())
    cached, cached_manifest, created_again = _derive_subset_prefix_cache(
        source_cache=source,
        destination_cache=destination,
        base_checkpoint_sha256=manifest["base_checkpoint_sha256"],
        base_runtime_config_sha256=manifest["base_runtime_config_sha256"],
    )
    assert created_again is False
    assert cached_manifest == subset
    assert set(cached) == set(prefixes)
