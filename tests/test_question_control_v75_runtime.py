from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.question_control_runtime import (
    QuestionControlledChatRuntime,
    _load_control_head,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.scene_encoder.question_control_v75 import (
    DenseFullSceneContinuousControlV75,
)
from semantic_3d_chat.training.question_control_v75_checkpoint import (
    V75_RUNTIME_ARCHITECTURE,
    V75_RUNTIME_METADATA_FIELDS,
    V75_RUNTIME_STATE_FIELDS,
    save_v75_control_checkpoint,
    v75_state_sha256,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _control() -> DenseFullSceneContinuousControlV75:
    torch.manual_seed(750075)
    return DenseFullSceneContinuousControlV75(
        8,
        torch.eye(4, 8),
        environment_latents=4,
        query_count=2,
        model_dimension=3,
        coefficient_decoder_hidden_dimension=7,
        uniform_floor_mass=0.05,
        maximum_control_rms=0.25,
    ).eval()


def _save(path: Path) -> tuple[DenseFullSceneContinuousControlV75, Path]:
    control = _control()
    save_v75_control_checkpoint(
        path,
        control=control,
        base_checkpoint_sha256="1" * 64,
        base_runtime_config_sha256="2" * 64,
        source_v75_candidate_sha256="3" * 64,
        expected_training_fit_state_sha256=v75_state_sha256(control),
        saved_runtime_training_gate_attestation_sha256="4" * 64,
    )
    return control, path


def test_v75_runtime_roundtrip_is_exact_two_file_and_reconstructs_hidden_width(
    tmp_path: Path,
) -> None:
    expected, checkpoint = _save(tmp_path / "v75")
    audit = FileAccessAudit()

    with audit:
        loaded, metadata = _load_control_head(
            checkpoint,
            hidden_size=8,
            device=torch.device("cpu"),
            audit=audit,
        )

    assert type(loaded) is DenseFullSceneContinuousControlV75
    assert loaded.coefficient_decoder_hidden_dimension == 7
    assert v75_state_sha256(loaded) == v75_state_sha256(expected)
    assert metadata["architecture"] == V75_RUNTIME_ARCHITECTURE
    assert set(metadata) == V75_RUNTIME_METADATA_FIELDS
    assert {item.name for item in checkpoint.iterdir()} == {
        "control.safetensors",
        "runtime_metadata.json",
    }
    assert set(audit.unique_paths) == {
        str((checkpoint / "control.safetensors").resolve()),
        str((checkpoint / "runtime_metadata.json").resolve()),
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("saved_runtime_training_gate_passed", False, "contract mismatch"),
        ("coefficient_decoder_hidden_dimension", 8, "tensor shapes changed"),
        ("environmental_text_inputs", ["forbidden"], "contract mismatch"),
        ("training_answers_runtime_loaded", True, "contract mismatch"),
        ("answer_text_runtime_loaded", True, "contract mismatch"),
        ("answer_class_codebook_runtime_loaded", True, "contract mismatch"),
        ("oracle_runtime_loaded", True, "contract mismatch"),
        ("question_dependent_scene_retrieval", True, "contract mismatch"),
        ("source_v75_candidate_sha256", "not-a-hash", "digest is invalid"),
    ],
)
def test_v75_runtime_rejects_contract_tampering(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    _expected, checkpoint = _save(tmp_path / field)
    metadata_path = checkpoint / "runtime_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _load_control_head(checkpoint, hidden_size=8, device=torch.device("cpu"))


def test_v75_runtime_rejects_any_extra_label_or_codebook_metadata(tmp_path: Path) -> None:
    _expected, checkpoint = _save(tmp_path / "extra")
    metadata_path = checkpoint / "runtime_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["answer_labels"] = ["forbidden"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata fields changed"):
        _load_control_head(checkpoint, hidden_size=8, device=torch.device("cpu"))


def test_v75_runtime_rejects_extra_inventory_and_state_fields(tmp_path: Path) -> None:
    _expected, checkpoint = _save(tmp_path / "inventory")
    (checkpoint / "training_answers.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory must contain only"):
        _load_control_head(checkpoint, hidden_size=8, device=torch.device("cpu"))
    (checkpoint / "training_answers.json").unlink()

    weights = checkpoint / "control.safetensors"
    state = load_file(str(weights), device="cpu")
    state["answer_codebook"] = torch.ones(1)
    save_file(state, weights)
    metadata_path = checkpoint / "runtime_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["weights_sha256"] = _sha256(weights)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="state fields changed"):
        _load_control_head(checkpoint, hidden_size=8, device=torch.device("cpu"))


def test_v75_runtime_rejects_nonfinite_and_shape_tampering(tmp_path: Path) -> None:
    for name, mutate, message in (
        (
            "nonfinite",
            lambda state: state["coefficient_hidden.weight"].fill_(float("nan")),
            "nonfinite or nonfloat",
        ),
        (
            "shape",
            lambda state: state.__setitem__(
                "coefficient_hidden.weight",
                state["coefficient_hidden.weight"][:-1].contiguous(),
            ),
            "tensor shapes changed",
        ),
    ):
        _expected, checkpoint = _save(tmp_path / name)
        weights = checkpoint / "control.safetensors"
        state = load_file(str(weights), device="cpu")
        mutate(state)
        save_file(state, weights)
        metadata_path = checkpoint / "runtime_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["weights_sha256"] = _sha256(weights)
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            _load_control_head(checkpoint, hidden_size=8, device=torch.device("cpu"))


def test_v75_runtime_caches_all_scene_kv_before_question_and_never_reencodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _control()
    scene_prefix = torch.randn(1, 6, 8)
    scene_hash = prefix_sha256(scene_prefix)
    events: list[str] = []
    original = control.encode_scene

    def observed(scene: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        events.append("scene")
        assert not torch.is_grad_enabled()
        return original(scene)

    monkeypatch.setattr(control, "encode_scene", observed)

    class Tokenizer:
        def __call__(self, *_args: object, **_kwargs: object) -> dict[str, torch.Tensor]:
            events.append("question")
            return {"input_ids": torch.tensor([[1, 2]])}

    embedding = torch.nn.Embedding(8, 8)
    base = SimpleNamespace(
        scene_prefix=scene_prefix,
        scene_prefix_hash=scene_hash,
        language=SimpleNamespace(
            tokenizer=Tokenizer(),
            model=SimpleNamespace(get_input_embeddings=lambda: embedding),
            device=torch.device("cpu"),
        ),
        config={"language": {"max_question_tokens": 8}},
        assert_prefix_unchanged=lambda: None,
        startup_summary=dict,
    )
    runtime = QuestionControlledChatRuntime(
        base,
        control,
        {"architecture": V75_RUNTIME_ARCHITECTURE, "schema_version": 75},
    )
    assert events == ["scene"]
    key_before = runtime._scene_control_key.clone()
    value_before = runtime._scene_control_value.clone()

    def refuse(_scene: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("V75 attempted question-time scene re-encoding")

    monkeypatch.setattr(control, "encode_scene", refuse)
    tokens, audit = runtime._control_tokens("different user question")

    assert events == ["scene", "question"]
    assert tokens is not None and tokens.shape == (1, 2, 8)
    assert torch.equal(runtime._scene_control_key, key_before)
    assert torch.equal(runtime._scene_control_value, value_before)
    assert runtime.current_prefix_hash() == scene_hash
    assert audit["prequestion_scene_key_value_cache"] is True
    assert audit["coefficient_decoder_hidden_dimension"] == 7
    assert audit["bias_free_nonlinear_coefficient_decoder"] is True
    assert audit["answer_text_runtime_loaded"] is False
    assert audit["answer_class_codebook_runtime_loaded"] is False


def test_optional_v78_grounding_does_not_change_v75_answer_generation() -> None:
    hidden_size = 8

    class Tokenizer:
        def apply_chat_template(self, *_: object, **__: object) -> torch.Tensor:
            return torch.tensor([[2, 3, 4]])

        def __call__(self, *_: object, **__: object) -> dict[str, torch.Tensor]:
            return {"input_ids": torch.tensor([[4, 5]])}

        def decode(self, *_: object, **__: object) -> str:
            return "right"

    class Backend:
        def __init__(self) -> None:
            self.scene_prefix: torch.Tensor | None = None
            self.control_tokens: torch.Tensor | None = None

        def prepare(
            self,
            scene_prefix: torch.Tensor,
            _prompt_ids: torch.Tensor,
            **kwargs: object,
        ) -> object:
            control = kwargs["control_tokens"]
            assert isinstance(control, torch.Tensor)
            self.scene_prefix = scene_prefix.detach().clone()
            self.control_tokens = control.detach().clone()
            return object()

        def generate(self, *_: object, **__: object) -> torch.Tensor:
            return torch.tensor([[7]])

    class GroundingSidecar:
        def __init__(self, expected_prefix: torch.Tensor) -> None:
            self.expected_prefix = expected_prefix
            self.calls = 0

        def assert_prefix_unchanged(self, scene_prefix: torch.Tensor) -> None:
            assert torch.equal(scene_prefix, self.expected_prefix)

        def predict(self, *_: object, **__: object) -> SimpleNamespace:
            self.calls += 1
            return SimpleNamespace(
                xyz_m=(0.25, -0.5, 1.0),
                confidence=0.8,
                support_distance_m=0.1,
                audit={"all_scene_tokens_scored": True},
            )

    torch.manual_seed(750078)
    scene_prefix = torch.randn(1, 6, hidden_size)
    scene_hash = prefix_sha256(scene_prefix)
    embedding = torch.nn.Embedding(16, hidden_size)

    def make_base(backend: Backend) -> SimpleNamespace:
        return SimpleNamespace(
            language=SimpleNamespace(
                tokenizer=Tokenizer(),
                model=SimpleNamespace(get_input_embeddings=lambda: embedding),
                device=torch.device("cpu"),
                prefix_backend=backend,
            ),
            scene_prefix=scene_prefix,
            scene_prefix_hash=scene_hash,
            map_data=SimpleNamespace(
                xyz=torch.zeros(1, 3),
                confidence=torch.ones(1),
            ),
            config={
                "language": {
                    "system_prompt": "Use only continuous memory.",
                    "max_question_tokens": 8,
                    "max_answer_tokens": 4,
                    "scene_prefix_after_bos": False,
                    "scene_boundary_mode": "learned",
                }
            },
            assert_prefix_unchanged=lambda: None,
            _eos_token_ids=lambda: 1,
            _predict_grounding=lambda _: ((9.0, 9.0, 9.0), 0.1, 9.0),
        )

    baseline_backend = Backend()
    sidecar_backend = Backend()
    baseline_control = _control()
    sidecar_control = _control()
    sidecar = GroundingSidecar(scene_prefix)
    baseline = QuestionControlledChatRuntime(
        make_base(baseline_backend),
        baseline_control,
        {"architecture": V75_RUNTIME_ARCHITECTURE, "schema_version": 75},
    )
    augmented = QuestionControlledChatRuntime(
        make_base(sidecar_backend),
        sidecar_control,
        {"architecture": V75_RUNTIME_ARCHITECTURE, "schema_version": 75},
        grounding_sidecar=sidecar,  # type: ignore[arg-type]
    )

    baseline_answer = baseline.answer("Where is the chair?")
    augmented_answer = augmented.answer("Where is the chair?")

    assert augmented_answer.answer == baseline_answer.answer == "right"
    assert augmented_answer.generated_tokens == baseline_answer.generated_tokens == 1
    assert augmented.last_control_tokens_sha256 == baseline.last_control_tokens_sha256
    assert torch.equal(sidecar_backend.scene_prefix, baseline_backend.scene_prefix)
    assert torch.equal(sidecar_backend.control_tokens, baseline_backend.control_tokens)
    assert augmented_answer.grounding_xyz_m == (0.25, -0.5, 1.0)
    assert baseline_answer.grounding_xyz_m == (9.0, 9.0, 9.0)
    assert sidecar.calls == 1

    robot_tokens = torch.randn(1, 2, hidden_size)
    active_prefix = torch.cat(
        (scene_prefix[:, :-1], robot_tokens, scene_prefix[:, -1:]), dim=1
    )
    active_hash = prefix_sha256(active_prefix)
    augmented.base.scene_prefix = active_prefix
    augmented.base.scene_prefix_hash = active_hash
    augmented.scene_prefix_hash = active_hash

    embodied_answer = augmented.answer("Where is the chair?")

    assert embodied_answer.answer == baseline_answer.answer
    assert embodied_answer.prefix_hash == active_hash
    assert sidecar_backend.scene_prefix is not None
    assert torch.equal(sidecar_backend.scene_prefix, active_prefix)
    assert sidecar.calls == 2

    corrupted = active_prefix.clone()
    corrupted[:, 3] += 1.0
    corrupted_hash = prefix_sha256(corrupted)
    augmented.base.scene_prefix = corrupted
    augmented.base.scene_prefix_hash = corrupted_hash
    augmented.scene_prefix_hash = corrupted_hash
    with pytest.raises(RuntimeError, match="not preserved"):
        augmented.answer("Where is the chair?")


def test_v75_metadata_and_state_allowlists_have_no_text_payload_fields() -> None:
    assert V75_RUNTIME_STATE_FIELDS == {
        "output_basis",
        "key.weight",
        "value.weight",
        "query.weight",
        "coefficient_hidden.weight",
        "coefficient_output.weight",
    }
    forbidden = ("label", "caption", "description", "scene_graph", "answer_items")
    assert not any(fragment in field for field in V75_RUNTIME_METADATA_FIELDS for fragment in forbidden)
