from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.question_control_runtime import (
    QuestionControlledChatRuntime,
    _load_control_head,
    block_question_control_training_artifacts,
)
from semantic_3d_chat.chat.runtime import ChatAnswer
from semantic_3d_chat.evaluation.question_control_leakage import (
    run_question_control_leakage,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.scene_encoder.question_control_v3 import (
    TeacherBasisFullSceneQuestionControlV3,
)
from semantic_3d_chat.scene_encoder.question_control_v7 import (
    AlwaysOnTeacherBasisFullSceneQuestionControlV7,
)
from semantic_3d_chat.training.question_control_v7_checkpoint import (
    load_unsealed_v7_checkpoint_for_training_gate,
    save_v7_control_checkpoint,
    v7_value_state_sha256,
)


def _source() -> TeacherBasisFullSceneQuestionControlV3:
    torch.manual_seed(766)
    basis = torch.linalg.qr(torch.randn(8, 4)).Q.T.contiguous()
    return TeacherBasisFullSceneQuestionControlV3(
        8,
        basis,
        control_tokens=2,
        expected_environment_latents=4,
        moment_count=2,
        interaction_dim=3,
        trunk_dim=5,
        maximum_control_rms=0.3,
        initial_control_rms=0.1,
    ).eval()


def test_v7_copies_v3_value_function_exactly_but_always_uses_control() -> None:
    source = _source()
    candidate = AlwaysOnTeacherBasisFullSceneQuestionControlV7.from_v3(source).eval()
    prefix = torch.randn(1, 6, 8)
    question = torch.randn(1, 3, 8)

    with torch.inference_mode():
        expected = source(prefix, question)
        observed = candidate(prefix, question)

    assert set(source.state_dict()) == set(candidate.state_dict())
    assert all(
        torch.equal(source.state_dict()[name], candidate.state_dict()[name])
        for name in source.state_dict()
    )
    assert torch.equal(observed.control_tokens, expected.control_tokens)
    assert torch.equal(observed.coefficient_directions, expected.coefficient_directions)
    assert torch.equal(observed.control_rms, expected.control_rms)
    assert bool(torch.all(observed.gate_probabilities > 0.999999))
    audit = candidate.audit()
    assert audit.control_used is True
    assert audit.always_on_continuous_control is True
    assert audit.gate_scene_question_conditioned is False
    assert audit.exact_no_control_route is False
    assert audit.legacy_route_parameters_ignored is True


def test_v7_uses_all_environment_latents_and_has_no_retrieval() -> None:
    candidate = AlwaysOnTeacherBasisFullSceneQuestionControlV7.from_v3(_source()).eval()
    with torch.inference_mode():
        candidate(torch.randn(1, 6, 8), torch.randn(1, 5, 8))
    audit = candidate.audit()

    assert audit.scene_token_count == 6
    assert audit.environment_latent_count == 4
    assert audit.every_environment_latent_influenced_signature is True
    assert audit.control_values_scene_question_bilinear is True
    assert audit.question_dependent_scene_retrieval is False
    assert audit.softmax_scene_attention_used is False


def test_v7_legacy_route_parameters_are_frozen_and_ignored() -> None:
    candidate = AlwaysOnTeacherBasisFullSceneQuestionControlV7.from_v3(_source())

    assert all(
        not parameter.requires_grad
        for name, parameter in candidate.named_parameters()
        if name.startswith("route_")
    )
    with torch.no_grad():
        candidate.route_bias.fill_(-1_000.0)
        candidate.route_log_scale.fill_(-1_000.0)
    candidate(torch.randn(1, 6, 8), torch.randn(1, 2, 8))

    assert candidate.audit().control_used is True


def test_v7_checkpoint_is_minimal_and_staged_state_roundtrips(tmp_path: Path) -> None:
    candidate = AlwaysOnTeacherBasisFullSceneQuestionControlV7.from_v3(_source()).eval()
    state_hash = v7_value_state_sha256(candidate)
    checkpoint = tmp_path / "candidate"

    hashes = save_v7_control_checkpoint(
        checkpoint,
        control=candidate,
        base_checkpoint_sha256="1" * 64,
        base_runtime_config_sha256="2" * 64,
        expected_training_fit_state_sha256=state_hash,
    )
    loaded = load_unsealed_v7_checkpoint_for_training_gate(
        checkpoint,
        hidden_size=8,
    )

    assert hashes["source_v66_training_fit_state_sha256"] == state_hash
    assert v7_value_state_sha256(loaded) == state_hash
    assert {path.name for path in checkpoint.iterdir()} == {
        "control.safetensors",
        "runtime_metadata.json",
    }


def test_v7_checkpoint_rejects_state_not_bound_to_training_fit(tmp_path: Path) -> None:
    candidate = AlwaysOnTeacherBasisFullSceneQuestionControlV7.from_v3(_source())

    with pytest.raises(ValueError, match="training-fit state changed"):
        save_v7_control_checkpoint(
            tmp_path / "candidate",
            control=candidate,
            base_checkpoint_sha256="1" * 64,
            base_runtime_config_sha256="2" * 64,
            expected_training_fit_state_sha256="3" * 64,
        )


def test_v7_public_runtime_loads_only_sealed_always_on_checkpoint(
    tmp_path: Path,
) -> None:
    candidate = AlwaysOnTeacherBasisFullSceneQuestionControlV7.from_v3(_source())
    checkpoint = tmp_path / "sealed"
    save_v7_control_checkpoint(
        checkpoint,
        control=candidate,
        base_checkpoint_sha256="1" * 64,
        base_runtime_config_sha256="2" * 64,
        expected_training_fit_state_sha256=v7_value_state_sha256(candidate),
        saved_runtime_training_gate_passed=True,
        saved_runtime_training_gate_attestation_sha256="4" * 64,
    )

    loaded, metadata = _load_control_head(
        checkpoint,
        hidden_size=8,
        device=torch.device("cpu"),
    )

    assert type(loaded) is AlwaysOnTeacherBasisFullSceneQuestionControlV7
    assert metadata["always_on_continuous_control"] is True
    assert metadata["training_answers_runtime_loaded"] is False
    assert metadata["answer_class_codebook_runtime_loaded"] is False
    assert v7_value_state_sha256(loaded) == v7_value_state_sha256(candidate)


def test_v7_public_runtime_rejects_unsealed_training_stage(tmp_path: Path) -> None:
    candidate = AlwaysOnTeacherBasisFullSceneQuestionControlV7.from_v3(_source())
    checkpoint = tmp_path / "unsealed"
    save_v7_control_checkpoint(
        checkpoint,
        control=candidate,
        base_checkpoint_sha256="1" * 64,
        base_runtime_config_sha256="2" * 64,
        expected_training_fit_state_sha256=v7_value_state_sha256(candidate),
    )

    with pytest.raises(ValueError, match="runtime contract mismatch"):
        _load_control_head(
            checkpoint,
            hidden_size=8,
            device=torch.device("cpu"),
        )


def test_v7_runtime_caches_scene_before_raw_question_and_injects_continuous_gemma_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    raw_questions: list[str] = []
    prompt_questions: list[str] = []
    control = AlwaysOnTeacherBasisFullSceneQuestionControlV7.from_v3(_source()).eval()
    original_encode_scene = control.encode_scene

    def observed_encode_scene(scene_prefix: torch.Tensor) -> torch.Tensor:
        events.append("encode_scene")
        assert not torch.is_grad_enabled()
        return original_encode_scene(scene_prefix)

    monkeypatch.setattr(control, "encode_scene", observed_encode_scene)

    class Tokenizer:
        def __call__(
            self,
            text: str,
            *,
            add_special_tokens: bool,
            return_tensors: str,
        ) -> dict[str, torch.Tensor]:
            events.append("tokenize_question")
            raw_questions.append(text)
            assert add_special_tokens is False
            assert return_tensors == "pt"
            token = 4 if "bowl" in text else 5
            return {"input_ids": torch.tensor([[token, 6]])}

        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            **_: object,
        ) -> torch.Tensor:
            prompt_questions.append(messages[-1]["content"])
            return torch.tensor([[2, 3, 4]])

        def decode(self, *_: object, **__: object) -> str:
            return "left"

    class GemmaBackend:
        def __init__(self) -> None:
            self.control_tokens: list[torch.Tensor] = []
            self.scene_prefixes: list[torch.Tensor] = []

        def prepare(
            self,
            scene_prefix: torch.Tensor,
            prompt_ids: torch.Tensor,
            **kwargs: object,
        ) -> object:
            continuous = kwargs.get("control_tokens")
            assert isinstance(continuous, torch.Tensor)
            assert prompt_ids.shape == (1, 3)
            self.control_tokens.append(continuous.detach().float().clone())
            self.scene_prefixes.append(scene_prefix.detach().float().clone())
            return object()

        def generate(self, *_: object, **__: object) -> torch.Tensor:
            return torch.tensor([[7]])

    torch.manual_seed(767)
    scene_prefix = torch.randn(1, 6, 8)
    scene_hash = prefix_sha256(scene_prefix)
    embedding = torch.nn.Embedding(16, 8)
    backend = GemmaBackend()
    base = SimpleNamespace(
        language=SimpleNamespace(
            tokenizer=Tokenizer(),
            model=SimpleNamespace(get_input_embeddings=lambda: embedding),
            device=torch.device("cpu"),
            prefix_backend=backend,
        ),
        scene_prefix=scene_prefix,
        scene_prefix_hash=scene_hash,
        config={
            "language": {
                "system_prompt": "Use only continuous memory.",
                "max_question_tokens": 8,
                "max_answer_tokens": 4,
                "scene_prefix_after_bos": False,
                "scene_boundary_mode": "learned",
            }
        },
        assert_prefix_unchanged=lambda: (
            None
            if prefix_sha256(scene_prefix) == scene_hash
            else (_ for _ in ()).throw(RuntimeError("base prefix changed"))
        ),
        current_prefix_hash=lambda: prefix_sha256(scene_prefix),
        startup_summary=lambda: {"prefix_shape": [1, 6, 8], "device": "cpu"},
        _eos_token_ids=lambda: 1,
        _predict_grounding=lambda _: ((1.0, 2.0, 3.0), 0.75, 0.25),
    )
    runtime = QuestionControlledChatRuntime(
        base,
        control,
        {
            "architecture": "always_on_teacher_basis_full_scene_control_v7",
            "schema_version": 7,
        },
    )

    assert events == ["encode_scene"]
    assert runtime.questions_answered == 0
    assert runtime.scene_control_signature_hash is not None
    assert runtime._scene_control_signature is not None
    assert runtime._scene_control_signature.grad_fn is None
    cached_signature = runtime._scene_control_signature.detach().clone()

    def refuse_question_time_scene_encoding(*_: object) -> torch.Tensor:
        raise AssertionError("scene was recomputed after user question arrived")

    monkeypatch.setattr(control, "encode_scene", refuse_question_time_scene_encoding)
    first = runtime.answer("Is the bowl left?")
    first_control_hash = runtime.last_control_tokens_sha256
    first_environment_input_hash = runtime.last_environment_conditioned_input_sha256
    second = runtime.answer("Where is the lamp?")
    second_control_hash = runtime.last_control_tokens_sha256
    second_environment_input_hash = runtime.last_environment_conditioned_input_sha256

    assert events[0] == "encode_scene"
    assert events.count("encode_scene") == 1
    assert raw_questions == [
        "Is the bowl left?",
        "Is the bowl left?",
        "Where is the lamp?",
        "Where is the lamp?",
    ]
    assert prompt_questions == ["Is the bowl left?", "Where is the lamp?"]
    assert runtime.questions_answered == 2
    assert first.prefix_hash == second.prefix_hash == scene_hash
    assert runtime.current_prefix_hash() == scene_hash
    assert first_control_hash is not None and second_control_hash is not None
    assert first_control_hash != second_control_hash
    assert first_environment_input_hash != second_environment_input_hash
    assert runtime.startup_summary()["strict_fixed_environment_embedding_input"] is False
    assert runtime.startup_summary()["question_conditioned_scene_readout_tokens"] is True
    assert torch.equal(runtime._scene_control_signature, cached_signature)
    assert len(backend.control_tokens) == 2
    assert all(value.shape == (1, 2, 8) for value in backend.control_tokens)
    assert all(torch.isfinite(value).all() for value in backend.control_tokens)
    assert all(float(value.square().mean().sqrt()) > 0.0 for value in backend.control_tokens)
    assert all(torch.equal(value, scene_prefix) for value in backend.scene_prefixes)
    assert runtime.last_control_audit is not None
    assert runtime.last_control_audit["always_on_continuous_control"] is True
    assert runtime.last_control_audit["control_used"] is True
    assert runtime.last_control_audit["question_dependent_scene_retrieval"] is False


def test_v7_runtime_detects_cached_signature_mutation() -> None:
    scene_prefix = torch.randn(1, 6, 8)
    base = SimpleNamespace(
        scene_prefix=scene_prefix,
        scene_prefix_hash=prefix_sha256(scene_prefix),
        assert_prefix_unchanged=lambda: None,
    )
    runtime = QuestionControlledChatRuntime(
        base,
        AlwaysOnTeacherBasisFullSceneQuestionControlV7.from_v3(_source()).eval(),
        {
            "architecture": "always_on_teacher_basis_full_scene_control_v7",
            "schema_version": 7,
        },
    )
    assert runtime._scene_control_signature is not None

    runtime._scene_control_signature[0, 0, 0] += 1.0

    with pytest.raises(RuntimeError, match="cached scene signature changed"):
        runtime.assert_prefix_unchanged()


def test_v7_checkpoint_reads_only_two_runtime_files_and_blocks_training_root(
    tmp_path: Path,
) -> None:
    derived = tmp_path / "derived"
    checkpoint = derived / "checkpoints" / "sealed"
    candidate = AlwaysOnTeacherBasisFullSceneQuestionControlV7.from_v3(_source())
    save_v7_control_checkpoint(
        checkpoint,
        control=candidate,
        base_checkpoint_sha256="1" * 64,
        base_runtime_config_sha256="2" * 64,
        expected_training_fit_state_sha256=v7_value_state_sha256(candidate),
        saved_runtime_training_gate_passed=True,
        saved_runtime_training_gate_attestation_sha256="4" * 64,
    )
    forbidden = derived / "training" / "answer_bank" / "values.safetensors"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_bytes(b"must never be read")
    audit = FileAccessAudit(block_forbidden=True)
    block_question_control_training_artifacts(
        audit,
        {
            "paths": {
                "data_root": str(tmp_path / "runtime_data"),
                "checkpoints_root": str(derived / "checkpoints"),
            }
        },
    )

    with audit:
        loaded, metadata = _load_control_head(
            checkpoint,
            hidden_size=8,
            device=torch.device("cpu"),
            audit=audit,
        )

    assert type(loaded) is AlwaysOnTeacherBasisFullSceneQuestionControlV7
    assert metadata["training_answers_runtime_loaded"] is False
    assert metadata["answer_class_codebook_runtime_loaded"] is False
    assert set(audit.unique_paths) == {
        str((checkpoint / "control.safetensors").resolve()),
        str((checkpoint / "runtime_metadata.json").resolve()),
    }
    assert audit.forbidden_accesses() == []
    assert str(forbidden.resolve()) not in audit.unique_paths


def test_v7_isolation_integration_hides_oracle_and_training_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from semantic_3d_chat.chat import runtime_config as runtime_config_module

    runtime_configs = tmp_path / "runtime_configs"
    runtime_configs.mkdir()
    config_path = runtime_configs / "v7.yaml"
    source_config = yaml.safe_load(
        Path("configs/runtime/gemma4_v54.yaml").read_text(encoding="utf-8")
    )
    runtime_data = tmp_path / "runtime_data"
    runtime_maps = tmp_path / "runtime_maps"
    derived = tmp_path / "derived"
    source_config["paths"] = {
        "data_root": str(runtime_data),
        "reports_root": str(tmp_path / "runtime_reports"),
        "maps_root": str(runtime_maps),
        "checkpoints_root": str(derived / "checkpoints"),
    }
    config_path.write_text(yaml.safe_dump(source_config), encoding="utf-8")
    monkeypatch.setattr(runtime_config_module, "RUNTIME_CONFIG_ROOT", runtime_configs)

    oracle = runtime_data / "oracle"
    oracle.mkdir(parents=True)
    (oracle / "truth.json").write_text("must stay isolated", encoding="utf-8")
    numeric_map = runtime_maps / "scene_000001" / "voxel_map.npz"
    numeric_map.parent.mkdir(parents=True)
    numeric_map.write_bytes(b"continuous numeric map")
    training_artifact = derived / "training" / "v66_numeric_teacher"
    training_artifact.mkdir(parents=True)
    (training_artifact / "values.safetensors").write_bytes(b"training only")
    control_checkpoint = derived / "checkpoints" / "sealed_v7"
    candidate = AlwaysOnTeacherBasisFullSceneQuestionControlV7.from_v3(_source())
    save_v7_control_checkpoint(
        control_checkpoint,
        control=candidate,
        base_checkpoint_sha256="1" * 64,
        base_runtime_config_sha256="2" * 64,
        expected_training_fit_state_sha256=v7_value_state_sha256(candidate),
        saved_runtime_training_gate_passed=True,
        saved_runtime_training_gate_attestation_sha256="4" * 64,
    )
    base_checkpoint = derived / "checkpoints" / "base"
    base_checkpoint.mkdir()
    observed = {"oracle_absent": False, "training_absent": False}

    class Runtime:
        scene_prefix_hash = "7" * 64

        def __init__(self, metadata: dict[str, object]) -> None:
            self.control_metadata = metadata
            self.base = SimpleNamespace(startup_summary=lambda: {"device": "cpu"})

        @classmethod
        def load(cls, *_: object, **kwargs: object) -> Runtime:
            audit = kwargs["audit"]
            assert isinstance(audit, FileAccessAudit)
            observed["oracle_absent"] = not oracle.exists()
            observed["training_absent"] = not training_artifact.exists()
            numeric_map.read_bytes()
            audit.record(numeric_map)
            loaded, metadata = _load_control_head(
                kwargs["control_checkpoint"],
                hidden_size=8,
                device=torch.device("cpu"),
                audit=audit,
            )
            assert type(loaded) is AlwaysOnTeacherBasisFullSceneQuestionControlV7
            return cls(metadata)

        def answer(self, question: str) -> ChatAnswer:
            return ChatAnswer(
                question=question,
                answer="unknown",
                grounding_xyz_m=(0.0, 0.0, 0.0),
                grounding_confidence=0.0,
                grounding_support_distance_m=0.0,
                prefix_hash=self.scene_prefix_hash,
                generated_tokens=1,
                elapsed_seconds=0.01,
            )

        def assert_prefix_unchanged(self) -> None:
            return None

    monkeypatch.setattr(
        "semantic_3d_chat.chat.question_control_runtime.QuestionControlledChatRuntime",
        Runtime,
    )
    monkeypatch.setattr(
        "semantic_3d_chat.evaluation.predict_question_control._control_checkpoint_sha256",
        lambda _path: "8" * 64,
    )
    output = tmp_path / "runtime_reports" / "leakage.json"

    report = run_question_control_leakage(
        config_path=config_path,
        scene_id="scene_000001",
        base_checkpoint=base_checkpoint,
        control_checkpoint=control_checkpoint,
        training_artifact=training_artifact,
        questions=("first?", "second?"),
        report_path=output,
    )

    assert observed == {"oracle_absent": True, "training_absent": True}
    assert report["passed"] is True
    assert report["control_schema_version"] == 7
    assert report["control_architecture"] == ("always_on_teacher_basis_full_scene_control_v7")
    assert report["oracle_was_renamed"] is True
    assert report["oracle_unavailable_during_inference"] is True
    assert report["oracle_restored"] is True
    assert report["training_artifact_was_renamed"] is True
    assert report["training_artifact_unavailable_during_inference"] is True
    assert report["training_artifact_restored"] is True
    assert report["prefix_computed_before_first_question"] is True
    assert report["prefix_invariant"] is True
    assert report["qa_or_oracle_loaded"] is False
    assert report["training_artifact_loaded_paths"] == []
    assert report["forbidden_accesses"] == []
    assert oracle.is_dir()
    assert training_artifact.is_dir()
