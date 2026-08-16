from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import semantic_3d_chat.training.gemma_waypoint_hidden_reuse as reuse_module
from semantic_3d_chat.training.gemma_waypoint_hidden_reuse import (
    assemble_hidden_with_reuse,
    frozen_gemma_input_sha256,
    load_legacy_waypoint_dataset_for_hidden_reuse,
    load_waypoint_dataset_for_hidden_reuse,
    reusable_hidden_rows,
    revalidate_cached_hidden_forward_contract,
    validate_forward_revalidation_destination,
)
from semantic_3d_chat.training.gemma_waypoint_policy import WaypointTraceSample


def _sample(
    sample_id: str,
    *,
    instruction: str = "Do a lap around the room.",
    state: tuple[float, ...] = (0.1, 0.2, 0.3),
    history: tuple[tuple[float, ...], ...] = ((1.0, 0.0),),
) -> WaypointTraceSample:
    return WaypointTraceSample(
        sample_id=sample_id,
        scene_id="scene_000001",
        split="train",
        instruction=instruction,
        state=torch.tensor(state, dtype=torch.float32),
        history=torch.tensor(history, dtype=torch.float32),
        action_index=0,
        waypoint_delta_robot_m=torch.zeros(2),
        heading_degrees=0.0,
    )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_legacy_dataset(
    root: Path,
    *,
    history_parameterization: str | None = None,
) -> str:
    root.mkdir(parents=True)
    rows = [
        {
            "sample_id": "old_train",
            "scene_id": "scene_000001",
            "split": "train",
            "instruction": "Do a lap around the room.",
            "state_features": [0.0] * 18,
            "history": [],
            "action": "stop",
            "waypoint_delta_robot_m": None,
            "heading_degrees": None,
        },
        {
            "sample_id": "old_validation",
            "scene_id": "scene_000031",
            "split": "validation",
            "instruction": "Stop.",
            "state_features": [0.0] * 18,
            "history": [],
            "action": "stop",
            "waypoint_delta_robot_m": None,
            "heading_degrees": None,
        },
    ]
    traces = root / "traces.jsonl"
    traces.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    body: dict[str, object] = {
        "schema": "semantic_3d_chat.gemma_waypoint_trace_dataset.v1",
        "profile": "operator",
        "sample_count": len(rows),
        "train_scene_ids": ["scene_000001"],
        "validation_scene_ids": ["scene_000031"],
        "scene_splits_disjoint": True,
        "runtime_compatible": False,
        "runtime_must_block_parent_tree": True,
        "environmental_text_training_only": True,
        "expert_planners_available_at_runtime": False,
        "oracle_inputs_at_runtime": False,
        "runtime_preprogrammed_lap_function": False,
        "action_names": ["MOVE_TO", "FACE", "STOP"],
        "state_feature_dim": 18,
        "history_feature_dim": 12,
        "history_length": 16,
        "max_waypoint_step_m": 0.5,
        "traces_sha256": hashlib.sha256(traces.read_bytes()).hexdigest(),
    }
    if history_parameterization is not None:
        body["history_parameterization"] = history_parameterization
    digest = _canonical_sha256(body)
    (root / "manifest.json").write_text(
        json.dumps({**body, "dataset_sha256": digest}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return digest


def _load_legacy(root: Path):
    return load_legacy_waypoint_dataset_for_hidden_reuse(
        root,
        state_dim=18,
        history_dim=12,
        max_history_tokens=16,
        max_waypoint_step_m=0.5,
    )


def test_legacy_reuse_loader_authenticates_missing_parameterization_only(
    tmp_path: Path,
) -> None:
    source = tmp_path / "training" / "legacy"
    declared = _write_legacy_dataset(source)

    loaded = _load_legacy(source)

    assert loaded.sha256 == declared
    assert loaded.traces_sha256 == hashlib.sha256(
        (source / "traces.jsonl").read_bytes()
    ).hexdigest()
    assert len(loaded.samples) == 2

    wrong = tmp_path / "training" / "wrong_parameterization"
    _write_legacy_dataset(wrong, history_parameterization="wrong_history_v0")
    with pytest.raises(ValueError, match="manifest authentication failed"):
        _load_legacy(wrong)

    corrupt = tmp_path / "training" / "corrupt"
    _write_legacy_dataset(corrupt)
    with (corrupt / "traces.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="manifest authentication failed"):
        _load_legacy(corrupt)


def test_v1_reuse_source_is_rejected_before_v2_cache_assembly(tmp_path: Path) -> None:
    source = tmp_path / "training" / "legacy_v1"
    _write_legacy_dataset(source)

    with pytest.raises(ValueError, match="history parameterization differs"):
        load_waypoint_dataset_for_hidden_reuse(
            source,
            state_dim=18,
            history_dim=16,
            history_parameterization="selected_action_parameters_goal_progress_v2",
            max_history_tokens=16,
            max_waypoint_step_m=0.5,
        )


def test_identical_inputs_reuse_across_changed_ids_and_target_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _sample("old_first")
    second = _sample("old_second", state=(0.4, 0.5, 0.6))
    source_hidden = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    reusable = reusable_hidden_rows((first, second), source_hidden)
    targets = (
        replace(second, sample_id="new_second"),
        replace(first, sample_id="new_first"),
        replace(second, sample_id="new_second_duplicate"),
    )

    def forbidden_forward(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("A fully reusable target must not call Gemma")

    monkeypatch.setattr(
        reuse_module,
        "cache_actual_gemma_decision_hidden",
        forbidden_forward,
    )
    assembled, reused, computed = assemble_hidden_with_reuse(object(), object(), targets, reusable)
    assert reused == 3
    assert computed == 0
    assert torch.equal(
        assembled,
        torch.tensor([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )


def test_state_history_and_instruction_changes_are_cache_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _sample("source")
    reusable = reusable_hidden_rows((source,), torch.tensor([[1.0, 2.0]]))
    changed = (
        replace(source, sample_id="state", state=source.state + 0.01),
        replace(source, sample_id="history", history=source.history + 0.01),
        replace(source, sample_id="instruction", instruction="Take a different lap."),
    )
    assert all(frozen_gemma_input_sha256(sample) not in reusable for sample in changed)
    forwarded_ids: list[str] = []

    def fake_forward(
        _runner: object,
        _cache: object,
        samples: tuple[WaypointTraceSample, ...] | list[WaypointTraceSample],
    ) -> torch.Tensor:
        forwarded_ids.extend(sample.sample_id for sample in samples)
        return torch.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])

    monkeypatch.setattr(
        reuse_module,
        "cache_actual_gemma_decision_hidden",
        fake_forward,
    )
    assembled, reused, computed = assemble_hidden_with_reuse(object(), object(), changed, reusable)
    assert forwarded_ids == ["state", "history", "instruction"]
    assert reused == 0
    assert computed == 3
    assert torch.equal(
        assembled,
        torch.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]),
    )


def test_conflicting_hidden_rows_for_duplicate_exact_inputs_fail_closed() -> None:
    first = _sample("first")
    duplicate = replace(first, sample_id="duplicate")
    with pytest.raises(
        RuntimeError,
        match="Identical Gemma inputs have different cached hidden states",
    ):
        reusable_hidden_rows(
            (first, duplicate),
            torch.tensor([[1.0, 2.0], [1.0, 3.0]]),
        )


def test_cache_misses_forward_in_bounded_chunks_and_report_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = tuple(_sample(f"row_{index}", state=(float(index), 0.0, 0.0)) for index in range(5))
    batches: list[list[str]] = []

    def fake_forward(
        _runner: object,
        _cache: object,
        values: tuple[WaypointTraceSample, ...] | list[WaypointTraceSample],
    ) -> torch.Tensor:
        batches.append([sample.sample_id for sample in values])
        return torch.tensor(
            [[float(sample.state[0]), 1.0] for sample in values],
            dtype=torch.float32,
        )

    monkeypatch.setattr(
        reuse_module,
        "cache_actual_gemma_decision_hidden",
        fake_forward,
    )
    progress: list[tuple[int, int]] = []
    assembled, reused, computed = assemble_hidden_with_reuse(
        object(),
        object(),
        samples,
        {},
        forward_chunk_size=2,
        progress=lambda completed, total: progress.append((completed, total)),
    )
    assert batches == [["row_0", "row_1"], ["row_2", "row_3"], ["row_4"]]
    assert progress == [(2, 5), (4, 5), (5, 5)]
    assert reused == 0
    assert computed == 5
    assert torch.equal(
        assembled,
        torch.tensor([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]]),
    )


def test_input_hash_ignores_labels_but_binds_scene_and_tensor_shapes() -> None:
    source = _sample("source")
    relabeled = replace(
        source,
        sample_id="relabeled",
        action_index=2,
        waypoint_delta_robot_m=torch.tensor([0.3, -0.2]),
        heading_degrees=40.0,
    )
    assert frozen_gemma_input_sha256(source) == frozen_gemma_input_sha256(relabeled)
    assert frozen_gemma_input_sha256(source) != frozen_gemma_input_sha256(
        replace(source, scene_id="scene_000002")
    )
    assert frozen_gemma_input_sha256(source) != frozen_gemma_input_sha256(
        replace(source, history=torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    )


def test_forward_contract_migration_recomputes_strata_and_requires_bit_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def split_samples(split: str) -> tuple[WaypointTraceSample, ...]:
        return tuple(
            replace(
                _sample(
                    f"{split}_{index}",
                    instruction=f"goal variant {index % 3}",
                    state=(float(index), 0.2, 0.3),
                    history=tuple((float(step), 0.0) for step in range(index % 4)),
                ),
                split=split,
                scene_id=f"scene_{1 + index % 2:06d}",
                action_index=index % 3,
            )
            for index in range(24)
        )

    train = split_samples("train")
    validation = split_samples("validation")

    def encoded(sample: WaypointTraceSample) -> torch.Tensor:
        return torch.tensor(
            [float(sample.state[0]), float(sample.action_index), float(sample.history.shape[0])]
        )

    forwarded_batches: list[tuple[str, ...]] = []

    def batch_sensitive_forward(
        _runner: object,
        _cache: object,
        samples: tuple[WaypointTraceSample, ...],
        *,
        forward_batch_size: int,
    ) -> torch.Tensor:
        assert forward_batch_size == 2
        assert len(samples) == 2
        assert len({(sample.instruction, sample.history.shape[0]) for sample in samples}) == 1
        forwarded_batches.append(tuple(sample.sample_id for sample in samples))
        batch_marker = sum(float(sample.state[0]) for sample in samples)
        return torch.stack([encoded(sample) + batch_marker for sample in samples])

    def reference_hidden(
        samples: tuple[WaypointTraceSample, ...],
    ) -> torch.Tensor:
        rows: list[torch.Tensor | None] = [None] * len(samples)
        groups: dict[tuple[str, int], list[int]] = {}
        for index, sample in enumerate(samples):
            groups.setdefault(
                (sample.instruction, int(sample.history.shape[0])), []
            ).append(index)
        for key in sorted(groups):
            group = groups[key]
            for offset in range(0, len(group), 2):
                batch = group[offset : offset + 2]
                output = batch_sensitive_forward(
                    object(), object(), tuple(samples[index] for index in batch),
                    forward_batch_size=2,
                )
                for row, index in zip(output, batch, strict=True):
                    rows[index] = row
        return torch.stack([row for row in rows if row is not None])

    train_hidden = reference_hidden(train)
    validation_hidden = reference_hidden(validation)
    forwarded_batches.clear()

    monkeypatch.setattr(
        reuse_module,
        "cache_actual_gemma_decision_hidden",
        batch_sensitive_forward,
    )
    report = revalidate_cached_hidden_forward_contract(
        object(),
        object(),
        train,
        validation,
        train_hidden,
        validation_hidden,
        sample_count_per_split=10,
        gemma_batch_size=2,
    )
    assert report["forward_contract_revalidated"] is True
    assert report["bit_exact_hidden_equality_required"] is True
    assert report["all_context_rows_bit_exact_required"] is True
    assert report["source_order_batch_context_reconstructed"] is True
    assert report["gemma_batch_size"] == 2
    assert report["train_rows_recomputed"] == 10
    assert report["train_target_rows_recomputed"] == 10
    assert report["train_context_rows_recomputed"] > 10
    assert report["train_companion_rows_recomputed"] == (
        report["train_context_rows_recomputed"] - 10
    )
    assert report["validation_target_rows_recomputed"] == 10
    assert report["validation_context_rows_recomputed"] > 10
    assert all(len(batch) == 2 for batch in forwarded_batches)
    assert report["train_action_strata"] == 3
    assert report["train_history_length_strata"] == 4
    assert report["train_instruction_strata"] == 3
    assert report["train_scene_strata"] == 2

    with pytest.raises(RuntimeError, match="changed cached hidden rows"):
        revalidate_cached_hidden_forward_contract(
            object(),
            object(),
            train,
            validation,
            train_hidden + 0.001,
            validation_hidden,
            sample_count_per_split=10,
            gemma_batch_size=2,
        )


def test_forward_revalidation_requires_a_fresh_distinct_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="must differ"):
        validate_forward_revalidation_destination(source, source)

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="must not already exist"):
        validate_forward_revalidation_destination(source, existing)

    fresh = tmp_path / "fresh"
    assert validate_forward_revalidation_destination(source, fresh) == fresh
    assert not fresh.exists()
