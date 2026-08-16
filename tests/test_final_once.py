from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import final_once
from semantic_3d_chat.evaluation.final_once import (
    FINAL_SCENE_IDS,
    WorkflowPaths,
    authorize_launch,
    build_launch_identity,
    build_stage_commands,
    run_final_once,
)

_REAL_VALIDATE_CHECKPOINT_RUNTIME_CONTRACT = (
    final_once._validate_checkpoint_runtime_contract
)
_REAL_ARTIFACT_ROOT = final_once.artifact_root


@pytest.fixture(autouse=True)
def _stub_large_checkpoint_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unit fixtures use a tiny sidecar; production executes the real validator."""

    isolated_maps_root = tmp_path / "sanitized_maps"
    monkeypatch.setattr(
        final_once,
        "artifact_root",
        lambda config, kind: (
            isolated_maps_root
            if kind == "maps"
            else _REAL_ARTIFACT_ROOT(config, kind)
        ),
    )
    monkeypatch.setattr(
        final_once,
        "_validate_checkpoint_runtime_contract",
        lambda _metadata, _runtime, _dataset: [],
    )
    monkeypatch.setattr(
        final_once,
        "local_model_snapshot_identity",
        lambda _config: {
            "model_id": "google/gemma-4-E2B-it",
            "revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
            "file_count": 9,
            "total_size_bytes": 1,
            "tree_sha256": "9" * 64,
            "files": [],
        },
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _workflow(tmp_path: Path, *, eligible: bool = True) -> WorkflowPaths:
    artifact_base = tmp_path / "isolated_final_artifacts"
    dataset_config = tmp_path / "isolated_diverse28.yaml"
    dataset_config.write_text(
        "\n".join(
            (
                f"_base_: {Path('configs/experiments/diverse28.yaml').resolve()}",
                "paths:",
                f"  data_root: {artifact_base / 'data'}",
                f"  oracle_root: {artifact_base / 'oracle'}",
                f"  rendered_root: {artifact_base / 'rendered'}",
                f"  features_root: {artifact_base / 'features'}",
                f"  maps_root: {artifact_base / 'maps'}",
                f"  qa_root: {artifact_base / 'qa'}",
                f"  checkpoints_root: {artifact_base / 'checkpoints'}",
                f"  reports_root: {artifact_base / 'reports'}",
                "",
            )
        ),
        encoding="utf-8",
    )
    qa_root = artifact_base / "qa"
    qa_root.mkdir(parents=True)
    (qa_root / "train.jsonl").write_text('{"scene_id":"scene_000011"}\n', encoding="utf-8")
    (qa_root / "validation.jsonl").write_text(
        '{"scene_id":"scene_000019"}\n', encoding="utf-8"
    )
    checkpoint = tmp_path / "checkpoint" / "update_008"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter.safetensors").write_bytes(b"continuous-final-adapter")
    _write_json(checkpoint / "runtime_metadata.json", {"schema_version": 3})
    selector = tmp_path / "selector.json"
    checks = {
        "development_checkpoint_selected": eligible,
        "changed_complete_pair_threshold_met": eligible,
        "aggregate_validation_exact_accuracy_retained": eligible,
    }
    _write_json(
        selector,
        {
            "passed": eligible,
            "development_selection_passed": eligible,
            "chat_promotion_eligible": eligible,
            "selected_checkpoint": str(checkpoint.resolve()) if eligible else None,
            "selected_update": 8 if eligible else None,
            "selected_optimizer_step": 8 if eligible else None,
            "final_test_scenes_touched": False,
            "chat_promotion": {
                "eligible": eligible,
                "evaluated": True,
                "checks": checks,
            },
        },
    )
    executable = Path(sys.executable).resolve()
    pointer = (
        Path("configs/runtime")
        / f".test_final_once_{os.getpid()}_{tmp_path.name}.json"
    ).resolve()
    return WorkflowPaths(
        dataset_config=dataset_config.resolve(),
        runtime_config=Path("configs/runtime/gemma4_primary.yaml").resolve(),
        selector_report=selector.resolve(),
        checkpoint=checkpoint.resolve(),
        work_root=(tmp_path / "final_once").resolve(),
        primary_pointer=pointer,
        coordinator_python=executable,
        gemma_python=executable,
        blender="/usr/bin/true",
    )


def test_final_once_preflight_is_read_only_and_binds_exact_final_split(tmp_path: Path) -> None:
    paths = _workflow(tmp_path)
    launch = authorize_launch(paths, write=False)

    assert launch["split_contract"]["final_scene_ids"] == list(FINAL_SCENE_IDS)
    assert launch["selector"]["selected_update"] == 8
    assert launch["selector"]["sha256"]
    assert launch["checkpoint"]["files"][0]["sha256"]
    assert launch["shared_maps_root"] == str((tmp_path / "sanitized_maps").resolve())
    assert not paths.work_root.exists()


def test_final_once_rejects_dataset_runtime_maps_root_mismatch_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _workflow(tmp_path)
    monkeypatch.setattr(final_once, "artifact_root", _REAL_ARTIFACT_ROOT)

    with pytest.raises(ValueError, match="must resolve the same maps_root"):
        authorize_launch(paths, write=False)

    assert not paths.launch.exists()


def test_final_once_refuses_ineligible_selector_before_runner_or_launch(
    tmp_path: Path,
) -> None:
    paths = _workflow(tmp_path, eligible=False)
    calls: list[tuple[str, ...]] = []

    with pytest.raises(ValueError, match="Selector did not pass"):
        run_final_once(paths, command_runner=lambda command: calls.append(tuple(command)))

    assert calls == []
    assert not paths.launch.exists()


def test_final_once_refuses_development_only_v32_result_before_any_write(
    tmp_path: Path,
) -> None:
    paths = _workflow(tmp_path)
    selector = json.loads(paths.selector_report.read_text(encoding="utf-8"))
    selector["chat_promotion_eligible"] = False
    selector["chat_promotion"]["eligible"] = False
    selector["chat_promotion"]["checks"][
        "changed_complete_pair_threshold_met"
    ] = False
    _write_json(paths.selector_report, selector)
    calls: list[tuple[str, ...]] = []

    with pytest.raises(ValueError, match="explicitly denied chat promotion"):
        run_final_once(paths, command_runner=lambda command: calls.append(tuple(command)))

    assert calls == []
    assert not paths.launch.exists()


def test_final_once_refuses_runtime_contract_mismatch_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _workflow(tmp_path)
    monkeypatch.setattr(
        final_once,
        "_validate_checkpoint_runtime_contract",
        lambda *_args: (_ for _ in ()).throw(
            ValueError("Checkpoint is incompatible with runtime architecture")
        ),
    )
    with pytest.raises(ValueError, match="incompatible with runtime architecture"):
        run_final_once(paths)
    assert not paths.launch.exists()


def test_final_once_launch_seal_rejects_selector_drift(tmp_path: Path) -> None:
    paths = _workflow(tmp_path)
    first = authorize_launch(paths, write=True)
    assert json.loads(paths.launch.read_text(encoding="utf-8")) == first

    selector = json.loads(paths.selector_report.read_text(encoding="utf-8"))
    selector["diagnostic_note"] = "changed after authorization"
    _write_json(paths.selector_report, selector)
    with pytest.raises(RuntimeError, match="launch identity changed"):
        authorize_launch(paths, write=True)


def test_final_once_rejects_checkpoint_symlink_before_authorization(tmp_path: Path) -> None:
    paths = _workflow(tmp_path)
    alias = tmp_path / "checkpoint-alias"
    alias.symlink_to(paths.checkpoint, target_is_directory=True)
    aliased = replace(paths, checkpoint=Path(os.path.abspath(alias)))

    with pytest.raises(ValueError, match="Checkpoint.*symbolic-link"):
        authorize_launch(aliased, write=False)
    assert not paths.launch.exists()


def test_final_once_rejects_symlinked_config_dependency(tmp_path: Path) -> None:
    paths = _workflow(tmp_path)
    base_alias = tmp_path / "diverse28-alias.yaml"
    base_alias.symlink_to(Path("configs/experiments/diverse28.yaml").resolve())
    paths.dataset_config.write_text(
        paths.dataset_config.read_text(encoding="utf-8").replace(
            str(Path("configs/experiments/diverse28.yaml").resolve()),
            str(base_alias),
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="config dependency.*symbolic-link"):
        authorize_launch(paths, write=False)
    assert not paths.launch.exists()


def test_final_once_rejects_bad_programmatic_stop_before_launch(tmp_path: Path) -> None:
    paths = _workflow(tmp_path)
    with pytest.raises(ValueError, match="Unknown final-once stop_after"):
        run_final_once(paths, stop_after="typo")
    assert not paths.launch.exists()


def test_final_once_commands_are_explicit_unpromoted_complete_runs(tmp_path: Path) -> None:
    paths = _workflow(tmp_path)
    dataset = final_once.load_config(paths.dataset_config)
    launch = authorize_launch(paths, write=True)

    generate = build_stage_commands("generate", paths, dataset, launch=launch)[0]
    features = build_stage_commands("features", paths, dataset)[0]
    qa = build_stage_commands("qa", paths, dataset)[0]
    primary = build_stage_commands("primary_predictions", paths, dataset)[0]
    empty = build_stage_commands("empty_prefix_predictions", paths, dataset)[0]

    assert ("--split", "test") == generate[generate.index("--split") :][:2]
    assert "--include-deferred-test" in generate
    assert ("--blender", "/usr/bin/true") == generate[generate.index("--blender") :][:2]
    assert generate[-1] == "--force"
    assert "semantic_3d_chat.vision.batch_encoder" in features
    assert "--include-deferred-test" in features
    assert str(paths.qa_build_config) in qa
    assert "semantic_3d_chat.evaluation.predict" in primary
    assert "require-gemma4-promoted" not in primary
    assert empty[-2:] == ("--condition", "empty_scene_prefix")
    assert "--max-questions-per-scene" not in empty


def test_final_once_refuses_force_command_without_matching_launch_seal(
    tmp_path: Path,
) -> None:
    paths = _workflow(tmp_path)
    dataset = final_once.load_config(paths.dataset_config)
    launch = build_launch_identity(paths)

    with pytest.raises(RuntimeError, match="without an existing.*launch seal"):
        build_stage_commands("generate", paths, dataset, launch=launch)

    authorize_launch(paths, write=True)
    wrong = {**launch, "identity_sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="different launch identity"):
        build_stage_commands("render", paths, dataset, launch=wrong)


def test_isolated_qa_publication_never_rewrites_development_splits(
    tmp_path: Path,
) -> None:
    paths = _workflow(tmp_path)
    launch = build_launch_identity(paths)
    dataset = final_once.load_config(paths.dataset_config)
    qa_root = final_once.artifact_root(dataset, "qa")
    train = qa_root / "train.jsonl"
    validation = qa_root / "validation.jsonl"
    baseline_bytes = {path: path.read_bytes() for path in (train, validation)}

    final_once._prepare_qa_build(paths)
    for path in (train, validation):
        (paths.qa_build_root / path.name).write_bytes(baseline_bytes[path])
    (paths.qa_build_root / "test.jsonl").write_bytes(b'{"sealed":"test"}\n')
    (paths.qa_build_root / "splits.json").write_bytes(b'{"sealed":"splits"}\n')
    final_once._publish_isolated_qa(paths, dataset, launch)

    assert {path: path.read_bytes() for path in (train, validation)} == baseline_bytes
    assert (qa_root / "test.jsonl").read_bytes() == b'{"sealed":"test"}\n'
    assert (qa_root / "splits.json").read_bytes() == b'{"sealed":"splits"}\n'


def test_final_once_receipt_resumes_and_rejects_output_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _workflow(tmp_path)
    generated = tmp_path / "fake_outputs" / "generated.bin"
    calls: list[tuple[str, ...]] = []

    def fake_roots(stage: str, *_args: object) -> tuple[Path, ...]:
        assert stage == "generate"
        return (generated,)

    def fake_runner(command: tuple[str, ...]) -> None:
        calls.append(tuple(command))
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_bytes(b"sealed deterministic output")

    monkeypatch.setattr(final_once, "_stage_output_roots", fake_roots)
    first = run_final_once(paths, stop_after="generate", command_runner=fake_runner)
    second = run_final_once(paths, stop_after="generate", command_runner=fake_runner)

    assert first["completed_stages"] == ["generate"]
    assert second["completed_stages"] == ["generate"]
    assert len(calls) == 1

    generated.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="outputs changed"):
        run_final_once(paths, stop_after="generate", command_runner=fake_runner)


def test_final_once_implementation_files_are_hash_bound(tmp_path: Path) -> None:
    launch = build_launch_identity(_workflow(tmp_path))
    names = {item["path"] for item in launch["implementation_files"]}
    assert "src/semantic_3d_chat/evaluation/final_once.py" in names
    assert "src/semantic_3d_chat/chat/promotion.py" in names
    assert all(len(item["sha256"]) == 64 for item in launch["implementation_files"])


def test_expected_dimensions_come_from_pinned_gemma_contract() -> None:
    dataset = final_once.load_config("configs/experiments/diverse28.yaml")
    runtime = final_once.load_runtime_config("configs/runtime/gemma4_primary.yaml")
    assert final_once._expected_runtime_dimensions(dataset, runtime) == (3072, 1536)


def test_final_once_counts_enabled_block_cross_residual_in_runtime_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Module:
        def __init__(self, parameter_count: int) -> None:
            self.parameter_count = parameter_count

    observed: dict[str, object] = {}

    monkeypatch.setattr(
        final_once,
        "_expected_runtime_dimensions",
        lambda *_args: (3072, 1536),
    )
    monkeypatch.setattr(
        "semantic_3d_chat.scene_encoder.dense_alignment.construct_dense_alignment",
        lambda *_args, **_kwargs: _Module(11),
    )
    monkeypatch.setattr(
        "semantic_3d_chat.scene_encoder.dense_sidecar_adapter.construct_dense_sidecar_adapter",
        lambda *_args, **_kwargs: _Module(22),
    )

    def construct_block_cross(_config, **kwargs):
        observed["block_constructor"] = kwargs
        return _Module(33)

    monkeypatch.setattr(
        "semantic_3d_chat.scene_encoder.block_cross_residual.construct_block_cross_residual",
        construct_block_cross,
    )

    def validate(_metadata, _config, **kwargs):
        observed["validator"] = kwargs
        return []

    monkeypatch.setattr(
        "semantic_3d_chat.chat.runtime.validate_checkpoint_contract",
        validate,
    )
    runtime = {
        "scene_encoder": {"model_dim": 384, "global_latents": 256},
    }
    metadata = {
        "semantic_dim": 3072,
        "language_hidden_dim": 1536,
        "lora_bank_parameter_counts": {},
    }

    assert _REAL_VALIDATE_CHECKPOINT_RUNTIME_CONTRACT(metadata, runtime, {}) == []
    assert observed["block_constructor"] == {
        "scene_dim": 1536,
        "block_dim": 384,
        "latent_count": 256,
    }
    assert observed["validator"]["block_cross_residual_parameter_count"] == 33


def _write_complete_final_qa(paths: WorkflowPaths, launch: dict) -> Path:
    dataset = final_once.load_config(paths.dataset_config)
    qa_root = final_once.artifact_root(dataset, "qa")
    records: list[dict] = []
    final_pairs = (
        ("pair_000012", "scene_000025", "scene_000026"),
        ("pair_000013", "scene_000027", "scene_000028"),
        ("pair_000014", "scene_000029", "scene_000030"),
    )
    question_index = 1
    for pair_id, first_scene, second_scene in final_pairs:
        for unit in range(36):
            changed = unit < 4
            for scene_id in (first_scene, second_scene):
                records.append(
                    {
                        "scene_id": scene_id,
                        "question_id": f"q_{question_index:06d}",
                        "counterfactual_pair_id": pair_id,
                        "counterfactual_question_key": f"unit_{unit:02d}",
                        "counterfactual_expected_change": changed,
                        "target_xyz": [0.0, 0.0, 0.0],
                    }
                )
                question_index += 1
    test_path = qa_root / "test.jsonl"
    test_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    _write_json(
        qa_root / "splits.json",
        {
            "splits": launch["split_contract"]["splits"],
            "question_counts": {"train": 384, "validation": 216, "test": 216},
        },
    )
    return test_path


def test_final_qa_validator_requires_exact_count_and_pair_coverage(tmp_path: Path) -> None:
    paths = _workflow(tmp_path)
    launch = build_launch_identity(paths)
    dataset = final_once.load_config(paths.dataset_config)
    test_path = _write_complete_final_qa(paths, launch)

    final_once._validate_qa_stage(paths, dataset, launch)

    records = test_path.read_text(encoding="utf-8").splitlines()
    test_path.write_text("\n".join(records[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="count differs"):
        final_once._validate_qa_stage(paths, dataset, launch)
