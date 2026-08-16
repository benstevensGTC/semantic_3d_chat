from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from semantic_3d_chat.language.local_lm import LocalLanguageModel
from semantic_3d_chat.robot.gemma_runtime_binding import (
    gemma_runtime_binding_sha256,
    raw_hf_gemma_runtime_binding,
)
from semantic_3d_chat.robot.gemma_waypoint_policy import ActualGemmaWaypointPolicy
from semantic_3d_chat.robot.waypoint_history import (
    HISTORY_PARAMETERIZATION_V1,
    HISTORY_PARAMETERIZATION_V2,
)
from semantic_3d_chat.training.gemma_waypoint_policy import (
    ACTION_TO_INDEX,
    ActualGemmaWaypointForward,
    ScenePrefixCache,
    WaypointPolicyTensors,
    WaypointTraceDataset,
    WaypointTraceSample,
    cache_actual_gemma_decision_hidden,
    evaluate_waypoint_controls,
    gemma_hidden_input_binding,
    load_gemma_hidden_cache,
    load_gemma_hidden_cache_for_forward_revalidation,
    load_waypoint_checkpoint,
    load_waypoint_retention_reference,
    load_waypoint_trace_jsonl,
    refit_waypoint_action_classifier,
    refit_waypoint_action_classifier_constrained,
    save_gemma_hidden_cache,
    save_waypoint_checkpoint,
    select_balanced_waypoint_samples,
    validate_waypoint_settings,
    waypoint_loss,
    waypoint_metrics,
    waypoint_retention_loss,
)
from semantic_3d_chat.training.gemma_waypoint_policy import (
    _waypoint_retention_gate_failure_message as retention_gate_failure_message,
)
from semantic_3d_chat.training.gemma_waypoint_policy import (
    _waypoint_retention_gate_report as retention_gate_report,
)


class _Tokenizer:
    bos_token_id = 2

    def apply_chat_template(self, *_args, **_kwargs):
        return torch.tensor([[2, 7, 8]], dtype=torch.long)


class _FakeGemma(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=hidden_size))
        self.calls: list[dict[str, object]] = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        values = kwargs["inputs_embeds"]
        # Every final position depends on every earlier scene, prompt, history,
        # and decision vector. This is a tiny causal stand-in, not a fake policy.
        hidden = torch.cumsum(values.float(), dim=1)
        return SimpleNamespace(hidden_states=(values.float(), hidden))


class _Prepared:
    def __init__(self, inputs: torch.Tensor, prefix_length: int) -> None:
        self.inputs_embeds = inputs
        self.per_layer_inputs = torch.zeros(
            (*inputs.shape[:2], 1, 1), dtype=inputs.dtype, device=inputs.device
        )
        self.attention_mask = torch.ones(inputs.shape[:2], dtype=torch.long)
        self.mm_token_type_ids = torch.zeros(inputs.shape[:2], dtype=torch.long)
        self.scene_prefix_length = prefix_length


class _FakeBackend:
    def __init__(self, model: _FakeGemma, hidden_size: int) -> None:
        self.model = model
        self.hidden_size = hidden_size
        self.prepared_prefixes: list[torch.Tensor] = []

    def prepare(
        self,
        prefix: torch.Tensor,
        prompt_ids: torch.Tensor,
        *,
        control_tokens: torch.Tensor,
        **_kwargs,
    ) -> _Prepared:
        self.prepared_prefixes.append(prefix.detach().clone())
        prompt = torch.nn.functional.one_hot(
            prompt_ids.remainder(self.hidden_size), num_classes=self.hidden_size
        ).to(prefix)
        inputs = torch.cat((prefix, prompt, control_tokens.to(prefix)), dim=1)
        return _Prepared(inputs, int(prefix.shape[1]))


class _StateEncoder(nn.Module):
    def __init__(self, state_dim: int, token_count: int, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(state_dim, token_count * hidden_size, bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(0.01)
        self.token_count = token_count
        self.hidden_size = hidden_size

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.linear(state).reshape(-1, self.token_count, self.hidden_size)


def _stack(hidden_size: int = 8, history_dim: int = 3):
    model = _FakeGemma(hidden_size)
    backend = _FakeBackend(model, hidden_size)
    language = LocalLanguageModel(
        model=model,
        tokenizer=_Tokenizer(),
        device=torch.device("cpu"),
        prefix_backend=backend,
        backend_name="gemma4",
    )
    policy = ActualGemmaWaypointPolicy(
        hidden_size=hidden_size,
        scene_token_count=5,
        robot_token_count=2,
        history_feature_dim=history_dim,
        max_history_tokens=4,
        head_hidden_dim=6,
        max_waypoint_step_m=0.5,
        freeze_context_projection=True,
    )
    state = _StateEncoder(5, 2, hidden_size)
    runner = ActualGemmaWaypointForward(
        language,
        policy,
        state,
        scene_token_count=5,
        robot_token_count=2,
        hidden_size=hidden_size,
        state_dim=5,
        history_dim=history_dim,
    )
    return model, backend, policy, state, runner


def _binding() -> dict[str, object]:
    return raw_hf_gemma_runtime_binding(
        model_id="google/gemma-4-E2B-it",
        model_revision="a" * 40,
        language_dtype="bfloat16",
    )


def _hidden_input_binding(
    *,
    history_dim: int | None = None,
    history_parameterization: str | None = None,
) -> dict[str, object]:
    binding: dict[str, object] = {
        "schema": (
            "semantic_3d_chat.gemma_waypoint_hidden_input_binding.v1"
            if history_parameterization is None
            else "semantic_3d_chat.gemma_waypoint_hidden_input_binding.v2"
        ),
        "scene_prefix_sha256": {"scene_000001": "1" * 64},
        "scene_prefix_file_sha256": {"scene_000001": "2" * 64},
        "robot_state_encoder_sha256": "3" * 64,
        "controller_context_sha256": "4" * 64,
        "prompt_token_ids_sha256": {"5" * 64: "6" * 64},
        "forward_contract_sha256": "7" * 64,
    }
    if history_parameterization is not None:
        binding.update(
            {
                "history_dim": history_dim,
                "history_parameterization": history_parameterization,
            }
        )
    return binding


def _waypoint_settings_config(
    *,
    history_dim: int,
    history_parameterization: str | None,
) -> dict[str, object]:
    settings: dict[str, object] = {
        "scene_token_count": 5,
        "robot_token_count": 2,
        "hidden_size": 8,
        "state_dim": 5,
        "history_dim": history_dim,
        "max_history_tokens": 4,
        "head_hidden_dim": 6,
        "context_token_count": 1,
        "epochs": 1,
        "gradient_accumulation_steps": 1,
        "head_batch_size": 1,
        "seed": 1,
        "action_refit_max_iter": 0,
        "heading_refit_steps": 0,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "waypoint_loss_weight": 1.0,
        "heading_loss_weight": 1.0,
        "action_class_weight_power": 0.5,
        "gradient_clip_norm": 1.0,
        "max_waypoint_step_m": 0.5,
        "max_turn_delta_degrees": 40.0,
        "action_refit_learning_rate": 0.5,
        "minimum_training_action_accuracy": 0.0,
        "heading_refit_learning_rate": 0.1,
        "minimum_training_turn_margin_degrees": 0.0,
        "checkpoint_selection": "heldout_validation",
        "batch_size": 1,
    }
    if history_parameterization is not None:
        settings["history_parameterization"] = history_parameterization
    return {"gemma_waypoint_policy": settings}


def _sample(
    sample_id: str,
    scene_id: str,
    split: str,
    action: str,
    *,
    marker: float,
) -> WaypointTraceSample:
    return WaypointTraceSample(
        sample_id=sample_id,
        scene_id=scene_id,
        split=split,
        instruction=f"goal {sample_id}",
        state=torch.tensor([marker, 0.1, -0.2, 0.0, 1.0]),
        history=torch.tensor([[marker, 0.0, 1.0]]),
        action_index=ACTION_TO_INDEX[action],
        waypoint_delta_robot_m=(
            torch.tensor([0.2, -0.1]) if action == "move_to" else torch.zeros(2)
        ),
        heading_degrees=45.0 if action == "face" else 0.0,
    )


def test_trace_loader_accepts_variable_numeric_history_and_rejects_scene_leak(
    tmp_path: Path,
) -> None:
    path = tmp_path / "traces.jsonl"
    rows = [
        {
            "sample_id": "a",
            "scene_id": "scene_000001",
            "split": "train",
            "instruction": "move beside the target",
            "state": [0.0, 0.1, 0.2, 0.0, 1.0],
            "history": [],
            "action": "MOVE_TO",
            "waypoint_delta_robot_m": [0.2, -0.1],
        },
        {
            "sample_id": "b",
            "scene_id": "scene_000002",
            "split": "validation",
            "instruction": "face it",
            "state": [0.0, 0.1, 0.2, 0.0, 1.0],
            "history": [[0.0, 1.0, 0.0]],
            "action": "FACE",
            "heading_degrees": -90.0,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    data = load_waypoint_trace_jsonl(
        path,
        state_dim=5,
        history_dim=3,
        max_history_tokens=4,
        max_waypoint_step_m=0.5,
    )
    assert data.samples[0].history.shape == (0, 3)
    assert data.samples[1].history.shape == (1, 3)
    rows[1]["scene_id"] = "scene_000001"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="occurs in both"):
        load_waypoint_trace_jsonl(
            path,
            state_dim=5,
            history_dim=3,
            max_history_tokens=4,
            max_waypoint_step_m=0.5,
        )


def test_trace_manifest_dataset_digest_is_canonical_and_tamper_evident(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    rows = [
        {
            "sample_id": "a",
            "scene_id": "scene_000001",
            "split": "train",
            "instruction": "do the task",
            "state": [0.0, 0.1, 0.2, 0.0, 1.0],
            "history": [],
            "action": "STOP",
        },
        {
            "sample_id": "b",
            "scene_id": "scene_000002",
            "split": "validation",
            "instruction": "do the task",
            "state": [0.2, 0.1, 0.2, 0.0, 1.0],
            "history": [],
            "action": "STOP",
        },
    ]
    traces = root / "traces.jsonl"
    traces.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    body = {
        "schema": "semantic_3d_chat.gemma_waypoint_trace_dataset.v1",
        "traces_sha256": hashlib.sha256(traces.read_bytes()).hexdigest(),
        "sample_count": 2,
        "train_scene_ids": ["scene_000001"],
        "validation_scene_ids": ["scene_000002"],
        "scene_splits_disjoint": True,
        "environmental_text_training_only": True,
        "expert_planners_available_at_runtime": False,
        "oracle_inputs_at_runtime": False,
        "history_parameterization": "selected_action_parameters_v1",
        "policy_selects_all_headings_and_waypoints_at_runtime": True,
    }
    manifest = {
        **body,
        "dataset_sha256": hashlib.sha256(
            json.dumps(
                body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    loaded = load_waypoint_trace_jsonl(
        root,
        state_dim=5,
        history_dim=3,
        max_history_tokens=4,
        max_waypoint_step_m=0.5,
    )
    assert loaded.sha256 == manifest["dataset_sha256"]

    rows[0]["state"][0] = 0.5
    traces.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest["traces_sha256"] = hashlib.sha256(traces.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="manifest contract differs"):
        load_waypoint_trace_jsonl(
            root,
            state_dim=5,
            history_dim=3,
            max_history_tokens=4,
            max_waypoint_step_m=0.5,
        )


def test_actual_forward_consumes_complete_scene_robot_history_and_hidden_states() -> None:
    model, backend, _policy, _state, runner = _stack()
    scene = torch.arange(40, dtype=torch.float32).reshape(1, 5, 8) / 100.0
    output = runner(
        scene,
        "do a lap and choose every waypoint",
        torch.tensor([[0.1, 0.2, 0.3, 0.0, 1.0]]),
        torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
    )
    assert output.action_logits.shape == (1, 3)
    assert backend.prepared_prefixes[0].shape == (1, 7, 8)
    # Robot tokens are inserted before the end boundary, preserving Gemma's
    # scene-memory boundary protocol.
    assert torch.equal(backend.prepared_prefixes[0][:, -1], scene[:, -1])
    assert model.calls[0]["output_hidden_states"] is True
    assert model.calls[0]["use_cache"] is False
    assert runner.last_decision_hidden is not None
    assert runner.last_audit == {
        "actual_gemma_causal_forward": True,
        "output_hidden_states_requested": True,
        "complete_scene_token_count": 5,
        "robot_token_count": 2,
        "history_token_count": 2,
        "decision_hidden_position": 12,
        "question_dependent_scene_retrieval": False,
    }


def test_batched_hidden_cache_matches_serial_complete_prefix_forwards() -> None:
    model, _backend, _policy, _state, runner = _stack()
    samples = tuple(
        replace(
            _sample(
                f"b{index}",
                "scene_000001",
                "train",
                ("move_to", "face", "stop")[index],
                marker=0.1 * (index + 1),
            ),
            instruction="same high-level goal",
        )
        for index in range(3)
    )
    cache = ScenePrefixCache(
        prefixes={"scene_000001": torch.ones(1, 5, 8)},
        file_sha256={"scene_000001": "a" * 64},
        token_count=5,
        hidden_size=8,
    )
    serial = cache_actual_gemma_decision_hidden(
        runner, cache, samples, forward_batch_size=1
    )
    serial_calls = len(model.calls)
    batched = cache_actual_gemma_decision_hidden(
        runner, cache, samples, forward_batch_size=2
    )
    assert torch.allclose(serial, batched, rtol=0.0, atol=2e-6)
    assert serial_calls == 3
    assert len(model.calls) - serial_calls == 2


def test_metrics_include_waypoint_heading_and_stop_scores() -> None:
    samples = (
        _sample("a", "scene_000001", "train", "move_to", marker=0.1),
        _sample("b", "scene_000001", "train", "face", marker=0.2),
        _sample("c", "scene_000001", "train", "stop", marker=0.3),
    )
    outputs = WaypointPolicyTensors(
        action_logits=torch.eye(3) * 10.0,
        waypoint_delta_robot_m=torch.tensor([[0.3, -0.1], [0.0, 0.0], [0.0, 0.0]]),
        turn_delta_degrees=torch.tensor([[0.0], [40.0], [0.0]]),
    )
    metrics = waypoint_metrics(outputs, samples)
    assert metrics["action_accuracy"] == 1.0
    assert metrics["waypoint_error_m_mean"] == pytest.approx(0.1)
    assert metrics["heading_error_degrees_mean"] == pytest.approx(5.0)
    assert metrics["stop_precision"] == metrics["stop_recall"] == 1.0


def test_controls_run_wrong_zero_scene_and_empty_history_through_gemma() -> None:
    model, _backend, _policy, _state, runner = _stack()
    samples = (
        _sample("a", "scene_000001", "validation", "move_to", marker=0.1),
        _sample("b", "scene_000002", "validation", "face", marker=0.2),
        _sample("c", "scene_000001", "validation", "stop", marker=0.3),
    )
    cache = ScenePrefixCache(
        prefixes={
            "scene_000001": torch.ones(1, 5, 8),
            "scene_000002": torch.full((1, 5, 8), 2.0),
        },
        file_sha256={"scene_000001": "a" * 64, "scene_000002": "b" * 64},
        token_count=5,
        hidden_size=8,
    )
    result = evaluate_waypoint_controls(runner, cache, samples)
    assert set(result["conditions"]) == {
        "primary",
        "wrong_scene_prefix",
        "zero_scene_prefix",
        "zero_history",
    }
    assert result["conditions"]["zero_scene_prefix"]["scene_content_latents_zeroed"]
    assert len(model.calls) == len(samples) * 4


def test_checkpoint_and_hidden_cache_store_no_language_weights_or_raw_text(
    tmp_path: Path,
) -> None:
    _model, _backend, policy, _state, _runner = _stack(history_dim=12)
    destination = tmp_path / "checkpoint"
    metadata = save_waypoint_checkpoint(
        destination,
        policy,
        metadata={
            "model_id": "google/gemma-4-E2B-it",
            "model_revision": "a" * 40,
            "scene_token_count": 5,
            "robot_token_count": 2,
            "history_dim": 12,
        },
    )
    assert metadata["frozen_gemma_weights_saved"] is False
    assert metadata["schema"] == "semantic_3d_chat.gemma_waypoint_checkpoint.v3"
    assert metadata["history_parameterization"] == "selected_action_parameters_v1"
    assert (destination / "policy.safetensors").stat().st_size < 1_000_000
    fresh = _stack(history_dim=12)[2]
    loaded = load_waypoint_checkpoint(destination, fresh)
    assert loaded["weights_sha256"] == metadata["weights_sha256"]

    old_metadata = dict(metadata)
    old_metadata["schema"] = "semantic_3d_chat.gemma_waypoint_checkpoint.v2"
    old_metadata.pop("history_parameterization")
    (destination / "runtime_metadata.json").write_text(
        json.dumps(old_metadata, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="runtime contract differs"):
        load_waypoint_checkpoint(destination, fresh)

    samples = (_sample("a", "scene_000001", "train", "stop", marker=0.1),)
    cache_metadata = save_gemma_hidden_cache(
        tmp_path / "hidden",
        train_hidden=torch.ones(1, 8),
        validation_hidden=torch.ones(1, 8) * 2,
        train_samples=samples,
        validation_samples=samples,
        dataset_sha256="d" * 64,
        gemma_runtime_binding=_binding(),
        hidden_input_binding=_hidden_input_binding(),
    )
    serialized = json.dumps(cache_metadata)
    assert "goal a" not in serialized
    assert cache_metadata["raw_instructions_stored"] is False


def test_v2_checkpoint_uses_v4_and_rejects_crossed_history_contracts(
    tmp_path: Path,
) -> None:
    policy = _stack(history_dim=16)[2]
    destination = tmp_path / "checkpoint_v2"
    metadata = save_waypoint_checkpoint(
        destination,
        policy,
        metadata={
            "model_id": "google/gemma-4-E2B-it",
            "model_revision": "a" * 40,
            "scene_token_count": 5,
            "robot_token_count": 2,
            "history_dim": 16,
            "history_parameterization": HISTORY_PARAMETERIZATION_V2,
        },
    )
    assert metadata["schema"] == "semantic_3d_chat.gemma_waypoint_checkpoint.v4"
    assert metadata["history_dim"] == 16
    assert metadata["history_parameterization"] == HISTORY_PARAMETERIZATION_V2
    loaded = load_waypoint_checkpoint(destination, _stack(history_dim=16)[2])
    assert loaded["weights_sha256"] == metadata["weights_sha256"]

    with pytest.raises(ValueError, match="dimension/parameterization pair differs"):
        save_waypoint_checkpoint(
            tmp_path / "crossed",
            policy,
            metadata={
                "history_dim": 16,
                "history_parameterization": HISTORY_PARAMETERIZATION_V1,
            },
        )

    tampered = dict(metadata)
    tampered["schema"] = "semantic_3d_chat.gemma_waypoint_checkpoint.v3"
    (destination / "runtime_metadata.json").write_text(
        json.dumps(tampered, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="runtime contract differs"):
        load_waypoint_checkpoint(destination, _stack(history_dim=16)[2])


def test_config_selects_only_exact_v1_or_v2_history_pairs() -> None:
    legacy = validate_waypoint_settings(
        _waypoint_settings_config(history_dim=12, history_parameterization=None)
    )
    assert legacy["history_parameterization"] == HISTORY_PARAMETERIZATION_V1
    assert legacy["action_refit_l2_weight"] == 0.0
    v2 = validate_waypoint_settings(
        _waypoint_settings_config(
            history_dim=16,
            history_parameterization=HISTORY_PARAMETERIZATION_V2,
        )
    )
    assert v2["history_dim"] == 16
    assert v2["history_parameterization"] == HISTORY_PARAMETERIZATION_V2
    for history_dim, parameterization in (
        (12, HISTORY_PARAMETERIZATION_V2),
        (16, HISTORY_PARAMETERIZATION_V1),
    ):
        with pytest.raises(ValueError, match="dimension/parameterization pair differs"):
            validate_waypoint_settings(
                _waypoint_settings_config(
                    history_dim=history_dim,
                    history_parameterization=parameterization,
                )
            )


@pytest.mark.parametrize("invalid", [-0.1, float("inf"), float("nan"), True, "0.1"])
def test_action_refit_l2_config_must_be_finite_nonnegative(invalid: object) -> None:
    config = _waypoint_settings_config(history_dim=12, history_parameterization=None)
    config["gemma_waypoint_policy"]["action_refit_l2_weight"] = invalid
    with pytest.raises((TypeError, ValueError), match="action_refit_l2_weight"):
        validate_waypoint_settings(config)


def test_v2_hidden_cache_manifest_binds_history_and_rejects_v1_binding(
    tmp_path: Path,
) -> None:
    sample = replace(
        _sample("v2", "scene_000001", "train", "stop", marker=0.1),
        history=torch.zeros((1, 16), dtype=torch.float32),
    )
    samples = (sample,)
    v2_binding = _hidden_input_binding(
        history_dim=16,
        history_parameterization=HISTORY_PARAMETERIZATION_V2,
    )
    destination = tmp_path / "hidden_v2"
    metadata = save_gemma_hidden_cache(
        destination,
        train_hidden=torch.ones(1, 8),
        validation_hidden=torch.ones(1, 8),
        train_samples=samples,
        validation_samples=samples,
        dataset_sha256="d" * 64,
        gemma_runtime_binding=_binding(),
        hidden_input_binding=v2_binding,
    )
    assert metadata["schema"] == "semantic_3d_chat.gemma_waypoint_hidden_cache.v4"
    assert metadata["history_dim"] == 16
    assert metadata["history_parameterization"] == HISTORY_PARAMETERIZATION_V2
    loaded_train, loaded_validation, _ = load_gemma_hidden_cache(
        destination,
        train_samples=samples,
        validation_samples=samples,
        dataset_sha256="d" * 64,
        hidden_size=8,
        expected_gemma_runtime_binding=_binding(),
        expected_hidden_input_binding=v2_binding,
    )
    assert loaded_train.shape == loaded_validation.shape == (1, 8)
    with pytest.raises(ValueError, match="hidden cache contract differs"):
        load_gemma_hidden_cache(
            destination,
            train_samples=samples,
            validation_samples=samples,
            dataset_sha256="d" * 64,
            hidden_size=8,
            expected_gemma_runtime_binding=_binding(),
            expected_hidden_input_binding=_hidden_input_binding(),
        )


def test_v2_hidden_input_builder_records_exact_numeric_history_contract() -> None:
    _model, _backend, policy, state, runner = _stack(history_dim=16)
    sample = replace(
        _sample("v2", "scene_000001", "train", "stop", marker=0.1),
        history=torch.zeros((1, 16), dtype=torch.float32),
    )
    cache = ScenePrefixCache(
        prefixes={"scene_000001": torch.zeros((1, 5, 8))},
        file_sha256={"scene_000001": "a" * 64},
        token_count=5,
        hidden_size=8,
    )
    binding = gemma_hidden_input_binding(
        runner.language,
        policy,
        state,
        cache,
        (sample,),
        history_parameterization=HISTORY_PARAMETERIZATION_V2,
    )
    assert binding["schema"] == (
        "semantic_3d_chat.gemma_waypoint_hidden_input_binding.v2"
    )
    assert binding["history_dim"] == 16
    assert binding["history_parameterization"] == HISTORY_PARAMETERIZATION_V2


def test_balanced_limits_match_hidden_cache_order_contract(tmp_path: Path) -> None:
    train_all = tuple(
        _sample(
            f"t{index}",
            "scene_000001",
            "train",
            ("move_to", "face", "stop")[index % 3],
            marker=index / 100.0,
        )
        for index in range(30)
    )
    validation_all = tuple(
        _sample(
            f"v{index}",
            f"scene_00000{2 + index % 2}",
            "validation",
            ("move_to", "face", "stop")[index % 3],
            marker=index / 100.0,
        )
        for index in range(24)
    )
    train = select_balanced_waypoint_samples(train_all, 12)
    validation = select_balanced_waypoint_samples(validation_all, 12)
    destination = tmp_path / "hidden"
    save_gemma_hidden_cache(
        destination,
        train_hidden=torch.arange(96, dtype=torch.float32).reshape(12, 8),
        validation_hidden=torch.arange(96, dtype=torch.float32).reshape(12, 8) + 1,
        train_samples=train,
        validation_samples=validation,
        dataset_sha256="d" * 64,
        gemma_runtime_binding=_binding(),
        hidden_input_binding=_hidden_input_binding(),
    )
    loaded_train, loaded_validation, metadata = load_gemma_hidden_cache(
        destination,
        train_samples=select_balanced_waypoint_samples(train_all, 12),
        validation_samples=select_balanced_waypoint_samples(validation_all, 12),
        dataset_sha256="d" * 64,
        hidden_size=8,
        expected_gemma_runtime_binding=_binding(),
        expected_hidden_input_binding=_hidden_input_binding(),
    )
    assert loaded_train.shape == loaded_validation.shape == (12, 8)
    assert metadata["train_sample_count"] == metadata["validation_sample_count"] == 12


def test_action_refit_reaches_exact_fit_without_changing_numeric_branches() -> None:
    _model, _backend, policy, _state, _runner = _stack()
    actions = ("move_to", "face", "stop") * 3
    samples = tuple(
        _sample(
            f"r{index}",
            "scene_000001",
            "train",
            action,
            marker=index / 10.0,
        )
        for index, action in enumerate(actions)
    )
    hidden = torch.zeros(len(samples), 8)
    for index, sample in enumerate(samples):
        hidden[index, sample.action_index] = 4.0
        hidden[index, 3 + index // 3] = 0.1
    before = {
        name: value.detach().clone()
        for name, value in policy.numeric_heads.state_dict().items()
        if not name.startswith("action.")
    }
    report = refit_waypoint_action_classifier(
        policy,
        hidden,
        samples,
        max_iter=100,
        learning_rate=0.5,
    )
    assert report["final_action_accuracy"] == 1.0
    assert report["remaining_action_errors"] == 0
    assert report["l2_weight"] == 0.0
    assert report["initial_objective"] == report["initial_loss"]
    assert report["final_objective"] == report["final_loss"]
    assert report["shared_norm_and_numeric_branches_unchanged"] is True
    for name, expected in before.items():
        assert torch.equal(policy.numeric_heads.state_dict()[name], expected)


def test_action_refit_zero_l2_matches_legacy_path_exactly() -> None:
    _model_a, _backend_a, legacy, _state_a, _runner_a = _stack()
    _model_b, _backend_b, explicit_zero, _state_b, _runner_b = _stack()
    actions = ("move_to", "face", "stop") * 3
    samples = tuple(
        _sample(
            f"z{index}",
            "scene_000001",
            "train",
            action,
            marker=index / 10.0,
        )
        for index, action in enumerate(actions)
    )
    hidden = torch.zeros(len(samples), 8)
    for index, sample in enumerate(samples):
        hidden[index, sample.action_index] = 4.0
        hidden[index, 3 + index // 3] = 0.1

    legacy_report = refit_waypoint_action_classifier(
        legacy,
        hidden,
        samples,
        max_iter=100,
        learning_rate=0.5,
    )
    explicit_report = refit_waypoint_action_classifier(
        explicit_zero,
        hidden,
        samples,
        max_iter=100,
        learning_rate=0.5,
        l2_weight=0.0,
    )
    for name, expected in legacy.numeric_heads.action.state_dict().items():
        assert torch.equal(explicit_zero.numeric_heads.action.state_dict()[name], expected)
    assert explicit_report == legacy_report


def test_action_refit_l2_bounds_separable_classifier_norm() -> None:
    _model_a, _backend_a, unregularized, _state_a, _runner_a = _stack()
    _model_b, _backend_b, regularized, _state_b, _runner_b = _stack()
    actions = ("move_to", "face", "stop") * 8
    samples = tuple(
        _sample(
            f"l2_{index}",
            "scene_000001",
            "train",
            action,
            marker=index / 100.0,
        )
        for index, action in enumerate(actions)
    )
    hidden = torch.zeros(len(samples), 8)
    for index, sample in enumerate(samples):
        hidden[index, sample.action_index] = 6.0
        hidden[index, 3 + index // 8] = 0.05

    baseline = refit_waypoint_action_classifier(
        unregularized,
        hidden,
        samples,
        max_iter=100,
        learning_rate=0.5,
    )
    bounded = refit_waypoint_action_classifier(
        regularized,
        hidden,
        samples,
        max_iter=100,
        learning_rate=0.5,
        l2_weight=0.01,
    )
    assert bounded["l2_weight"] == 0.01
    assert bounded["final_action_accuracy"] == 1.0
    assert bounded["final_weight_l2_norm"] < baseline["final_weight_l2_norm"]
    assert bounded["final_weight_l2_norm"] < 10.0
    assert bounded["final_objective"] >= bounded["final_loss"]


def test_constrained_action_refit_is_deterministic_exact_and_fail_closed() -> None:
    _model_a, _backend_a, policy_a, _state_a, _runner_a = _stack()
    _model_b, _backend_b, policy_b, _state_b, _runner_b = _stack()
    generator = torch.Generator().manual_seed(91)
    hidden = torch.randn(12, 8, generator=generator)
    samples = tuple(
        _sample(
            f"constrained_{index}",
            "scene_000001",
            "train",
            "move_to" if index < 9 else ("face", "stop", "face")[index - 9],
            marker=index / 100.0,
        )
        for index in range(12)
    )
    with torch.no_grad():
        policy_a.numeric_heads.action.weight.zero_()
        policy_a.numeric_heads.action.bias.copy_(torch.tensor([2.0, 0.0, -2.0]))
        policy_b.load_state_dict(policy_a.state_dict())
    reference_weight = policy_a.numeric_heads.action.weight.detach().clone()
    reference_bias = policy_a.numeric_heads.action.bias.detach().clone()
    with torch.no_grad():
        reference_logits = policy_a.numeric_heads.action(
            policy_a.numeric_heads.input_norm(hidden)
        )
    shared = torch.tensor([True] * 9 + [False] * 3)
    frozen = {
        name: value.detach().clone()
        for name, value in policy_a.numeric_heads.state_dict().items()
        if not name.startswith("action.")
    }

    reports = []
    for policy in (policy_a, policy_b):
        # Prove the solver starts from authenticated reference parameters, not
        # whatever a preceding joint optimizer left in the action branch.
        with torch.no_grad():
            policy.numeric_heads.action.weight.add_(3.0)
            policy.numeric_heads.action.bias.sub_(4.0)
        reports.append(
            refit_waypoint_action_classifier_constrained(
                policy,
                hidden,
                samples,
                reference_logits=reference_logits,
                reference_action_weight=reference_weight,
                reference_action_bias=reference_bias,
                retention_mask=shared,
                positive_margin=0.001,
                maximum_centered_logit_rmse=1.0,
            )
        )
    assert reports[0] == reports[1]
    assert reports[0]["shared_action_agreement"] == 1.0
    assert reports[0]["new_action_accuracy"] == 1.0
    assert reports[0]["minimum_shared_margin"] > 0.0
    assert reports[0]["minimum_new_margin"] > 0.0
    assert reports[0]["all_action_constraints_validated_after_float32_materialization"]
    assert torch.equal(
        policy_a.numeric_heads.action.weight, policy_b.numeric_heads.action.weight
    )
    assert torch.equal(
        policy_a.numeric_heads.action.bias, policy_b.numeric_heads.action.bias
    )
    for name, expected in frozen.items():
        assert torch.equal(policy_a.numeric_heads.state_dict()[name], expected)

    _model_c, _backend_c, rejected, _state_c, _runner_c = _stack()
    rejected.load_state_dict(policy_a.state_dict())
    before = {
        name: value.detach().clone()
        for name, value in rejected.numeric_heads.action.state_dict().items()
    }
    with pytest.raises(RuntimeError, match="rejected before mutation"):
        refit_waypoint_action_classifier_constrained(
            rejected,
            hidden,
            samples,
            reference_logits=reference_logits,
            reference_action_weight=reference_weight,
            reference_action_bias=reference_bias,
            retention_mask=shared,
            positive_margin=0.001,
            maximum_centered_logit_rmse=1e-12,
        )
    for name, expected in before.items():
        assert torch.equal(rejected.numeric_heads.action.state_dict()[name], expected)


def test_weighted_new_rows_and_centered_output_retention_are_training_only() -> None:
    samples = (
        _sample("shared_move", "scene_000001", "train", "move_to", marker=0.1),
        _sample("shared_face", "scene_000001", "train", "face", marker=0.2),
        _sample("new_stop", "scene_000001", "train", "stop", marker=0.3),
    )
    reference = WaypointPolicyTensors(
        action_logits=torch.tensor([[5.0, 0.0, -2.0], [0.0, 5.0, -2.0], [4.0, 0.0, -3.0]]),
        waypoint_delta_robot_m=torch.tensor([[0.2, -0.1], [0.0, 0.0], [0.1, 0.1]]),
        turn_delta_degrees=torch.tensor([[0.0], [40.0], [0.0]]),
    )
    # Centered-logit retention ignores a softmax-invariant common offset and
    # ignores the new row entirely.
    unchanged = WaypointPolicyTensors(
        action_logits=(reference.action_logits + torch.tensor([[7.0], [-3.0], [99.0]])).requires_grad_(),
        waypoint_delta_robot_m=reference.waypoint_delta_robot_m.clone().requires_grad_(),
        turn_delta_degrees=reference.turn_delta_degrees.clone().requires_grad_(),
    )
    loss, parts = waypoint_retention_loss(
        unchanged,
        reference,
        samples,
        torch.tensor([True, True, False]),
        max_waypoint_step_m=0.5,
        max_turn_delta_degrees=40.0,
        logit_weight=1.0,
        waypoint_weight=1.0,
        heading_weight=1.0,
    )
    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-12)
    assert parts["shared_rows"] == 2.0

    drifted = WaypointPolicyTensors(
        action_logits=(reference.action_logits + torch.tensor([[0.0, 2.0, 0.0], [0.0, 0.0, 1.0], [50.0, 0.0, 0.0]])).requires_grad_(),
        waypoint_delta_robot_m=(reference.waypoint_delta_robot_m + torch.tensor([[0.1, 0.0], [0.0, 0.0], [9.0, 9.0]])).requires_grad_(),
        turn_delta_degrees=(reference.turn_delta_degrees + torch.tensor([[0.0], [-5.0], [30.0]])).requires_grad_(),
    )
    retained, retained_parts = waypoint_retention_loss(
        drifted,
        reference,
        samples,
        torch.tensor([True, True, False]),
        max_waypoint_step_m=0.5,
        max_turn_delta_degrees=40.0,
        logit_weight=1.0,
        waypoint_weight=1.0,
        heading_weight=1.0,
    )
    assert retained_parts["logit"] > 0.0
    assert retained_parts["waypoint"] > 0.0
    assert retained_parts["heading"] > 0.0
    retained.backward()
    assert drifted.action_logits.grad is not None
    assert drifted.waypoint_delta_robot_m.grad is not None
    assert drifted.turn_delta_degrees.grad is not None

    supervised_outputs = WaypointPolicyTensors(
        action_logits=torch.tensor([[5.0, 0.0, -2.0], [5.0, 0.0, -2.0]], requires_grad=True),
        waypoint_delta_robot_m=torch.tensor([[0.2, -0.1], [0.0, 0.0]], requires_grad=True),
        turn_delta_degrees=torch.zeros(2, 1, requires_grad=True),
    )
    supervised_samples = (samples[0], samples[2])
    unweighted, _ = waypoint_loss(
        supervised_outputs,
        supervised_samples,
        max_waypoint_step_m=0.5,
        max_turn_delta_degrees=40.0,
    )
    weighted, _ = waypoint_loss(
        supervised_outputs,
        supervised_samples,
        max_waypoint_step_m=0.5,
        max_turn_delta_degrees=40.0,
        sample_weights=torch.tensor([1.0, 16.0]),
    )
    assert float(weighted.detach()) > float(unweighted.detach())


def test_action_refit_reference_retention_limits_shared_logit_drift() -> None:
    _model_a, _backend_a, baseline, _state_a, _runner_a = _stack()
    _model_b, _backend_b, retained, _state_b, _runner_b = _stack()
    shared = tuple(
        _sample(
            f"shared_{index}",
            "scene_000001",
            "train",
            ("move_to", "face", "stop")[index % 3],
            marker=index / 10.0,
        )
        for index in range(9)
    )
    # This added STOP is deliberately close to a prior MOVE state, creating a
    # measurable pressure that the teacher term must absorb during the test.
    new = replace(shared[0], sample_id="new_stop", action_index=ACTION_TO_INDEX["stop"])
    samples = (*shared, new)
    hidden = torch.zeros(len(samples), 8)
    for index, sample in enumerate(shared):
        hidden[index, sample.action_index] = 4.0
        hidden[index, 3 + index // 3] = 0.1
    hidden[-1] = hidden[0]
    with torch.no_grad():
        teacher_logits = retained.forward_heads_from_cached_gemma_hidden(hidden).action_logits
    row_weights = torch.tensor([1.0] * len(shared) + [16.0])
    refit_waypoint_action_classifier(
        baseline,
        hidden,
        samples,
        max_iter=100,
        learning_rate=0.5,
        sample_weights=row_weights,
    )
    report = refit_waypoint_action_classifier(
        retained,
        hidden,
        samples,
        max_iter=100,
        learning_rate=0.5,
        sample_weights=row_weights,
        reference_logits=teacher_logits,
        retention_mask=torch.tensor([True] * len(shared) + [False]),
        retention_weight=100.0,
    )

    def centered_drift(policy: nn.Module) -> float:
        logits = policy.forward_heads_from_cached_gemma_hidden(hidden).action_logits
        observed = logits[: len(shared)] - logits[: len(shared)].mean(dim=-1, keepdim=True)
        expected = teacher_logits[: len(shared)] - teacher_logits[: len(shared)].mean(
            dim=-1, keepdim=True
        )
        return float(torch.nn.functional.mse_loss(observed, expected).detach())

    assert report["retention_shared_rows"] == len(shared)
    assert report["final_retention_loss"] < report["initial_retention_loss"] + 0.01
    assert centered_drift(retained) < centered_drift(baseline)


def test_retention_settings_are_backward_compatible_and_fail_closed() -> None:
    config = _waypoint_settings_config(
        history_dim=12, history_parameterization=HISTORY_PARAMETERIZATION_V1
    )
    legacy = validate_waypoint_settings(config)
    assert legacy["retention_reference_checkpoint"] is None
    assert legacy["retention_logit_weight"] == 0.0
    assert legacy["retention_new_sample_weight"] == 1.0
    assert legacy["retention_freeze_input_norm"] is False
    assert legacy["retention_joint_training_epochs"] is None
    assert legacy["waypoint_branch_refit_enabled"] is False
    assert legacy["waypoint_branch_refit_steps"] == 300
    assert legacy[
        "waypoint_branch_refit_minimum_new_within_tolerance_fraction"
    ] == 1.0

    config["gemma_waypoint_policy"]["retention_logit_weight"] = 1.0
    with pytest.raises(ValueError, match="authenticated reference"):
        validate_waypoint_settings(config)
    config["gemma_waypoint_policy"].update(
        {
            "retention_reference_checkpoint": "prior",
            "retention_reference_trace_dataset": "prior_rows",
            "retention_minimum_shared_action_agreement": 1.1,
        }
    )
    with pytest.raises(ValueError, match="must be in"):
        validate_waypoint_settings(config)

    invalid_skip = _waypoint_settings_config(
        history_dim=12, history_parameterization=HISTORY_PARAMETERIZATION_V1
    )
    invalid_skip["gemma_waypoint_policy"]["retention_joint_training_epochs"] = -1
    with pytest.raises(ValueError, match="null or nonnegative"):
        validate_waypoint_settings(invalid_skip)

    missing_reference = _waypoint_settings_config(
        history_dim=12, history_parameterization=HISTORY_PARAMETERIZATION_V1
    )
    missing_reference["gemma_waypoint_policy"][
        "waypoint_branch_refit_enabled"
    ] = True
    with pytest.raises(ValueError, match="authenticated reference"):
        validate_waypoint_settings(missing_reference)

    invalid_fraction = _waypoint_settings_config(
        history_dim=12, history_parameterization=HISTORY_PARAMETERIZATION_V1
    )
    invalid_fraction["gemma_waypoint_policy"].update(
        {
            "waypoint_branch_refit_minimum_new_within_tolerance_fraction": 1.1,
        }
    )
    with pytest.raises(ValueError, match="must be in"):
        validate_waypoint_settings(invalid_fraction)


def test_retention_gate_failure_exposes_observed_and_required_values() -> None:
    settings = {
        "retention_minimum_shared_action_agreement": 1.0,
        "retention_minimum_new_action_accuracy": 1.0,
        "retention_maximum_shared_centered_logit_rmse": 0.2,
        "retention_maximum_shared_waypoint_drift_m": 0.025,
        "retention_maximum_shared_heading_drift_degrees": 1.0,
    }
    observed = {
        "shared_action_agreement": 1.0,
        "new_action_accuracy": 0.75,
        "shared_centered_logit_rmse": 0.35,
        "shared_move_waypoint_drift_m_max": 0.04,
        "shared_face_heading_drift_degrees_max": 0.5,
    }
    gates = retention_gate_report(observed, settings)
    assert gates["passed"] is False
    assert gates["failures"] == [
        "new_action_accuracy",
        "shared_centered_logit_rmse",
        "shared_move_waypoint_drift_m_max",
    ]
    diagnostics = {
        row["metric"]: row for row in gates["failure_details"]
    }
    assert diagnostics["new_action_accuracy"] == {
        "metric": "new_action_accuracy",
        "observed": 0.75,
        "comparison": "minimum",
        "required": 1.0,
    }
    assert diagnostics["shared_centered_logit_rmse"]["observed"] == 0.35
    assert diagnostics["shared_centered_logit_rmse"]["required"] == 0.2
    assert diagnostics["shared_move_waypoint_drift_m_max"]["observed"] == 0.04
    assert diagnostics["shared_move_waypoint_drift_m_max"]["required"] == 0.025
    message = retention_gate_failure_message(gates)
    decoded = json.loads(message.split("; diagnostics=", maxsplit=1)[1])
    assert decoded == gates["failure_details"]


def test_retention_reference_authenticates_checkpoint_and_exact_shared_rows(
    tmp_path: Path,
) -> None:
    base_samples = tuple(
        replace(
            _sample(
                f"prior_{index}",
                "scene_000001",
                "train",
                action,
                marker=0.1 + index / 10.0,
            ),
            history=torch.zeros((1, 12), dtype=torch.float32),
        )
        for index, action in enumerate(("move_to", "face", "stop"))
    )
    # The reference intentionally contains one exact duplicate occurrence with
    # a different generated ID. Retention must preserve its multiplicity while
    # remaining independent of those unstable IDs.
    source_samples = (
        *base_samples,
        replace(base_samples[0], sample_id="prior_duplicate"),
    )
    rows = []
    for sample in source_samples:
        row = {
            "sample_id": sample.sample_id,
            "scene_id": sample.scene_id,
            "split": sample.split,
            "instruction": sample.instruction,
            "state": sample.state.tolist(),
            "history": sample.history.tolist(),
            "action": sample.action_name,
        }
        if sample.action_name == "move_to":
            row["waypoint_delta_robot_m"] = sample.waypoint_delta_robot_m.tolist()
        elif sample.action_name == "face":
            row["heading_degrees"] = sample.heading_degrees
        rows.append(row)
    trace_path = tmp_path / "prior.jsonl"
    trace_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    reference_dataset = load_waypoint_trace_jsonl(
        trace_path,
        state_dim=5,
        history_dim=12,
        history_parameterization=HISTORY_PARAMETERIZATION_V1,
        max_history_tokens=4,
        max_waypoint_step_m=0.5,
    )
    checkpoint = tmp_path / "prior_checkpoint"
    prior_policy = _stack(history_dim=12)[2]
    save_waypoint_checkpoint(
        checkpoint,
        prior_policy,
        metadata={
            "model_id": "google/gemma-4-E2B-it",
            "model_revision": "a" * 40,
            "gemma_runtime_binding": _binding(),
            "gemma_runtime_binding_sha256": gemma_runtime_binding_sha256(_binding()),
            "dataset_sha256": reference_dataset.sha256,
            "training_traces_sha256": reference_dataset.traces_sha256,
            "training_sample_count": 4,
            "scene_token_count": 5,
            "robot_token_count": 2,
            "hidden_size": 8,
            "state_dim": 5,
            "history_dim": 12,
            "history_parameterization": HISTORY_PARAMETERIZATION_V1,
            "max_history_tokens": 4,
            "context_token_count": 1,
            "head_hidden_dim": 6,
            "max_waypoint_step_m": 0.5,
            "max_turn_delta_degrees": 40.0,
            "heading_parameterization": "robot_relative_bounded_scalar_tanh",
        },
    )
    config = _waypoint_settings_config(
        history_dim=12, history_parameterization=HISTORY_PARAMETERIZATION_V1
    )
    config["language"] = {
        "model_id": "google/gemma-4-E2B-it",
        "revision": "a" * 40,
    }
    config["gemma_waypoint_policy"].update(
        {
            "retention_reference_checkpoint": str(checkpoint),
            "retention_reference_trace_dataset": str(trace_path),
            "retention_logit_weight": 1.0,
            "retention_new_sample_weight": 8.0,
        }
    )
    settings = validate_waypoint_settings(config)
    shared = reference_dataset.split("train")
    shifted_shared = tuple(
        replace(sample, sample_id=f"shifted_{index}")
        for index, sample in enumerate(shared)
    )
    new_sample = replace(
        shared[-1],
        # Deliberately collide with an old sequential ID while changing the
        # actual row. An ID join would misclassify this as shared.
        sample_id=shared[0].sample_id,
        instruction="new goal",
        state=shared[-1].state + torch.tensor([0.05, 0.0, 0.0, 0.0, 0.0]),
    )
    current_samples = (new_sample, *shifted_shared)
    current_dataset = WaypointTraceDataset(
        samples=current_samples,
        sha256="c" * 64,
        traces_sha256="d" * 64,
        state_dim=5,
        history_dim=12,
        history_parameterization=HISTORY_PARAMETERIZATION_V1,
    )
    hidden = torch.randn(5, 8, generator=torch.Generator().manual_seed(7))
    loaded = load_waypoint_retention_reference(
        config,
        _stack(history_dim=12)[2],
        current_dataset,
        current_samples,
        hidden,
        settings=settings,
        gemma_runtime_binding=_binding(),
        device=torch.device("cpu"),
    )
    assert loaded is not None
    assert loaded.shared_mask.tolist() == [False, True, True, True, True]
    assert loaded.sample_weights.tolist() == [8.0, 1.0, 1.0, 1.0, 1.0]
    assert loaded.metadata["all_reference_rows_preserved_exactly"] is True
    assert loaded.metadata["identity_uses_generated_sample_id"] is False
    assert loaded.metadata["one_to_one_reference_row_occurrence_matching"] is True
    assert loaded.metadata["reference_unique_fingerprint_buckets"] == 3
    assert loaded.metadata["reference_duplicate_row_occurrences"] == 1
    assert loaded.metadata["shared_training_rows"] == 4
    assert loaded.metadata["new_training_rows"] == 1

    changed_samples = (
        current_samples[0],
        replace(current_samples[1], state=current_samples[1].state + 0.01),
        *current_samples[2:],
    )
    with pytest.raises(ValueError, match="removed or changed"):
        load_waypoint_retention_reference(
            config,
            _stack(history_dim=12)[2],
            current_dataset,
            changed_samples,
            hidden,
            settings=settings,
            gemma_runtime_binding=_binding(),
            device=torch.device("cpu"),
        )


def test_hidden_cache_rejects_a_different_gemma_runtime_stack(tmp_path: Path) -> None:
    samples = (_sample("a", "scene_000001", "train", "stop", marker=0.1),)
    destination = tmp_path / "hidden"
    save_gemma_hidden_cache(
        destination,
        train_hidden=torch.ones(1, 8),
        validation_hidden=torch.ones(1, 8),
        train_samples=samples,
        validation_samples=samples,
        dataset_sha256="d" * 64,
        gemma_runtime_binding=_binding(),
        hidden_input_binding=_hidden_input_binding(),
    )
    different = raw_hf_gemma_runtime_binding(
        model_id="google/gemma-4-E2B-it",
        model_revision="b" * 40,
        language_dtype="bfloat16",
    )
    with pytest.raises(ValueError, match="hidden cache contract differs"):
        load_gemma_hidden_cache(
            destination,
            train_samples=samples,
            validation_samples=samples,
            dataset_sha256="d" * 64,
            hidden_size=8,
            expected_gemma_runtime_binding=different,
            expected_hidden_input_binding=_hidden_input_binding(),
        )


def test_hidden_cache_rejects_a_different_continuous_input_stack(tmp_path: Path) -> None:
    samples = (_sample("a", "scene_000001", "train", "stop", marker=0.1),)
    destination = tmp_path / "hidden"
    save_gemma_hidden_cache(
        destination,
        train_hidden=torch.ones(1, 8),
        validation_hidden=torch.ones(1, 8),
        train_samples=samples,
        validation_samples=samples,
        dataset_sha256="d" * 64,
        gemma_runtime_binding=_binding(),
        hidden_input_binding=_hidden_input_binding(),
    )
    changed = _hidden_input_binding()
    changed["robot_state_encoder_sha256"] = "8" * 64
    with pytest.raises(ValueError, match="hidden cache contract differs"):
        load_gemma_hidden_cache(
            destination,
            train_samples=samples,
            validation_samples=samples,
            dataset_sha256="d" * 64,
            hidden_size=8,
            expected_gemma_runtime_binding=_binding(),
            expected_hidden_input_binding=changed,
        )


def test_hidden_cache_forward_revalidation_seam_allows_only_source_hash_change(
    tmp_path: Path,
) -> None:
    train = (_sample("a", "scene_000001", "train", "move_to", marker=0.1),)
    validation = (
        _sample("b", "scene_000001", "validation", "stop", marker=0.2),
    )
    destination = tmp_path / "hidden"
    save_gemma_hidden_cache(
        destination,
        train_hidden=torch.arange(8, dtype=torch.float32).reshape(1, 8),
        validation_hidden=torch.arange(8, dtype=torch.float32).reshape(1, 8) + 1,
        train_samples=train,
        validation_samples=validation,
        dataset_sha256="d" * 64,
        gemma_runtime_binding=_binding(),
        hidden_input_binding=_hidden_input_binding(),
    )
    current = _hidden_input_binding()
    current["forward_contract_sha256"] = "8" * 64
    with pytest.raises(ValueError, match="cache contract differs"):
        load_gemma_hidden_cache(
            destination,
            train_samples=train,
            validation_samples=validation,
            dataset_sha256="d" * 64,
            hidden_size=8,
            expected_gemma_runtime_binding=_binding(),
            expected_hidden_input_binding=current,
        )
    loaded_train, loaded_validation, metadata = (
        load_gemma_hidden_cache_for_forward_revalidation(
            destination,
            train_samples=train,
            validation_samples=validation,
            dataset_sha256="d" * 64,
            hidden_size=8,
            expected_gemma_runtime_binding=_binding(),
            expected_hidden_input_binding=current,
        )
    )
    assert loaded_train.tolist() == [list(map(float, range(8)))]
    assert loaded_validation.tolist() == [list(map(float, range(1, 9)))]
    assert metadata["hidden_input_binding"]["forward_contract_sha256"] == "7" * 64

    changed_beyond_source = dict(current)
    changed_beyond_source["robot_state_encoder_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="differs beyond"):
        load_gemma_hidden_cache_for_forward_revalidation(
            destination,
            train_samples=train,
            validation_samples=validation,
            dataset_sha256="d" * 64,
            hidden_size=8,
            expected_gemma_runtime_binding=_binding(),
            expected_hidden_input_binding=changed_beyond_source,
        )
