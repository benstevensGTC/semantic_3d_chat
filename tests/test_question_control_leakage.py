from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.question_control_runtime import _load_control_head
from semantic_3d_chat.evaluation.question_control_leakage import (
    _LeakageRuntimeAdapter,
    run_question_control_leakage,
    teacher_artifact_temporarily_unavailable,
)
from semantic_3d_chat.scene_encoder.question_control_v3 import (
    TeacherBasisFullSceneQuestionControlV3,
)
from semantic_3d_chat.scene_encoder.question_control_v4 import (
    SceneConditionedGateTeacherBasisControlV4,
)
from semantic_3d_chat.training.question_control_v4_checkpoint import (
    inherited_value_state_sha256,
    save_v4_control_checkpoint,
)


def _safe_control_metadata(
    architecture: str = "full_scene_question_control_v1",
    schema_version: int = 1,
) -> dict[str, object]:
    return {
        "architecture": architecture,
        "schema_version": schema_version,
        "environmental_text_inputs": [],
        "question_dependent_scene_retrieval": False,
        "complete_scene_prefix_required": True,
    }


def _v4_checkpoint(path: Path) -> Path:
    torch.manual_seed(619)
    basis = torch.linalg.qr(torch.randn(12, 6)).Q.T.contiguous()
    source = TeacherBasisFullSceneQuestionControlV3(
        12,
        basis,
        control_tokens=2,
        expected_environment_latents=5,
        moment_count=3,
        interaction_dim=4,
        trunk_dim=7,
        maximum_control_rms=0.4,
        initial_control_rms=0.1,
    ).eval()
    control = SceneConditionedGateTeacherBasisControlV4.from_v60(
        source, gate_hidden_dim=3
    ).eval()
    inherited = inherited_value_state_sha256(control)
    save_v4_control_checkpoint(
        path,
        control=control,
        base_checkpoint_sha256="1" * 64,
        base_runtime_config_sha256="2" * 64,
        source_v60_checkpoint_sha256="3" * 64,
        expected_inherited_state_sha256=inherited,
    )
    return path


def test_leakage_adapter_tracks_questions_without_changing_prefix() -> None:
    class Runtime:
        scene_prefix_hash = "a" * 64
        control_metadata: ClassVar[dict[str, object]] = _safe_control_metadata()
        base = SimpleNamespace(startup_summary=lambda: {"device": "cpu"})

        def answer(self, question: str) -> SimpleNamespace:
            return SimpleNamespace(question=question)

        def assert_prefix_unchanged(self) -> None:
            return None

    adapter = _LeakageRuntimeAdapter(Runtime())

    assert adapter.questions_answered == 0
    assert adapter.current_prefix_hash() == "a" * 64
    assert adapter.startup_summary()["environmental_text_inputs"] == []
    assert adapter.startup_summary()["control_schema_version"] == 1
    assert adapter.answer("Question?").question == "Question?"
    assert adapter.questions_answered == 1
    adapter.assert_prefix_unchanged()


def test_teacher_artifact_is_atomically_hidden_and_restored_on_failure(
    tmp_path: Path,
) -> None:
    teacher = tmp_path / "data_gemma4" / "training" / "v58_teachers"
    weights = teacher / "teachers.safetensors"
    teacher.mkdir(parents=True)
    weights.write_bytes(b"teacher")

    class ExpectedFailure(RuntimeError):
        pass

    with (
        pytest.raises(ExpectedFailure),
        teacher_artifact_temporarily_unavailable(teacher) as isolation,
    ):
        assert isolation.renamed is True
        assert not teacher.exists()
        assert isolation.hidden is not None and isolation.hidden.is_dir()
        assert (isolation.hidden / weights.name).read_bytes() == b"teacher"
        raise ExpectedFailure("exercise finally restoration")

    assert teacher.is_dir()
    assert weights.read_bytes() == b"teacher"
    assert not any(
        path.name.startswith(".v58_teachers-unavailable-") for path in teacher.parent.iterdir()
    )


def test_teacher_artifact_isolation_rejects_non_training_parent(tmp_path: Path) -> None:
    invalid = tmp_path / "derived" / "teachers"
    invalid.mkdir(parents=True)

    with (
        pytest.raises(ValueError, match="direct child of a training root"),
        teacher_artifact_temporarily_unavailable(invalid),
    ):
        pass


def test_question_control_leakage_blocks_training_root_and_attests_runtime_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    derived = tmp_path / "data_gemma4"
    teacher = derived / "training" / "v58_teachers"
    teacher.mkdir(parents=True)
    (teacher / "teachers.safetensors").write_bytes(b"teacher")
    control = derived / "checkpoints" / "control"
    control.mkdir(parents=True)
    (control / "control.safetensors").write_bytes(b"control")
    (control / "runtime_metadata.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "leakage.json"

    class Runtime:
        scene_prefix_hash = "a" * 64
        control_metadata: ClassVar[dict[str, object]] = _safe_control_metadata()
        base = SimpleNamespace(startup_summary=lambda: {"device": "cpu"})

        @classmethod
        def load(cls, *_args: object, **kwargs: object) -> Runtime:
            audit = kwargs["audit"]
            assert isinstance(audit, FileAccessAudit)
            assert teacher.parent in audit.forbidden_roots
            assert "training" not in audit.forbidden_component_names
            assert not teacher.exists()
            audit.record(control / "control.safetensors")
            audit.record(control / "runtime_metadata.json")
            return cls()

    def fake_generic_leakage(**kwargs: object) -> dict[str, object]:
        assert not teacher.exists()
        audit = FileAccessAudit(block_forbidden=True)
        with audit:
            loader = kwargs["runtime_loader"]
            assert callable(loader)
            loader(
                {
                    "paths": {
                        "data_root": str(tmp_path / "data"),
                        "checkpoints_root": str(derived / "checkpoints"),
                    }
                },
                "scene_000031",
                tmp_path / "base",
                audit,
            )
        return {
            "passed": True,
            "loaded_files": audit.unique_paths,
            "forbidden_accesses": [],
        }

    monkeypatch.setattr(
        "semantic_3d_chat.chat.question_control_runtime.QuestionControlledChatRuntime",
        Runtime,
    )
    monkeypatch.setattr(
        "semantic_3d_chat.evaluation.question_control_leakage.run_leakage_evaluation",
        fake_generic_leakage,
    )
    monkeypatch.setattr(
        "semantic_3d_chat.evaluation.predict_question_control._control_checkpoint_sha256",
        lambda _path: "b" * 64,
    )

    report = run_question_control_leakage(
        config_path=tmp_path / "runtime.yaml",
        scene_id="scene_000031",
        base_checkpoint=tmp_path / "base",
        control_checkpoint=control,
        teacher_artifact=teacher,
        questions=("Question?",),
        report_path=output,
    )

    assert report["passed"] is True
    assert report["teacher_artifact_was_renamed"] is True
    assert report["teacher_artifact_unavailable_during_inference"] is True
    assert report["teacher_artifact_restored"] is True
    assert report["teacher_artifact_loaded"] is False
    assert report["training_artifact_loaded_paths"] == []
    assert report["runtime_checkpoint_files_complete"] is True
    assert set(report["runtime_checkpoint_files_loaded"]) == {
        str((control / "control.safetensors").resolve()),
        str((control / "runtime_metadata.json").resolve()),
    }
    assert teacher.is_dir()
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["runtime_checkpoint_files_complete"] is True


def test_v4_leakage_loads_only_sanitized_checkpoint_without_teacher_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise schema V4 loading under the real training-root file denylist.

    This remains an offline unit test: the base runtime and generic question
    loop are replaced, so neither Gemma generation nor QA/oracle reads occur.
    """

    derived = tmp_path / "derived"
    control = _v4_checkpoint(derived / "checkpoints" / "control_v4")
    training_root = derived / "training"
    training_root.mkdir(parents=True)
    (training_root / "numeric_training_state.bin").write_bytes(b"must not load")
    output = tmp_path / "leakage.json"

    class Runtime:
        scene_prefix_hash = "4" * 64

        def __init__(self, metadata: dict[str, object]) -> None:
            self.control_metadata = metadata
            self.base = SimpleNamespace(startup_summary=lambda: {"device": "cpu"})

        @classmethod
        def load(cls, *_args: object, **kwargs: object) -> Runtime:
            audit = kwargs["audit"]
            assert isinstance(audit, FileAccessAudit)
            assert training_root.resolve() in audit.forbidden_roots
            module, metadata = _load_control_head(
                kwargs["control_checkpoint"],
                hidden_size=12,
                device=torch.device("cpu"),
                audit=audit,
            )
            assert isinstance(module, SceneConditionedGateTeacherBasisControlV4)
            return cls(metadata)

    def fake_generic_leakage(**kwargs: object) -> dict[str, object]:
        audit = FileAccessAudit(block_forbidden=True)
        with audit:
            loader = kwargs["runtime_loader"]
            assert callable(loader)
            adapter = loader(
                {
                    "paths": {
                        "data_root": str(tmp_path / "runtime_data"),
                        "checkpoints_root": str(derived / "checkpoints"),
                    }
                },
                "scene_000031",
                tmp_path / "base",
                audit,
            )
            assert adapter.startup_summary()["control_schema_version"] == 4
        assert not any(
            {"qa", "oracle"} & {part.casefold() for part in Path(path).parts}
            for path in audit.unique_paths
        )
        return {
            "passed": True,
            "loaded_files": audit.unique_paths,
            "forbidden_accesses": audit.forbidden_accesses(),
        }

    monkeypatch.setattr(
        "semantic_3d_chat.chat.question_control_runtime.QuestionControlledChatRuntime",
        Runtime,
    )
    monkeypatch.setattr(
        "semantic_3d_chat.evaluation.question_control_leakage.run_leakage_evaluation",
        fake_generic_leakage,
    )

    report = run_question_control_leakage(
        config_path=tmp_path / "runtime.yaml",
        scene_id="scene_000031",
        base_checkpoint=tmp_path / "base",
        control_checkpoint=control,
        questions=("Question?",),
        report_path=output,
    )

    expected_files = {
        str((control / "control.safetensors").resolve()),
        str((control / "runtime_metadata.json").resolve()),
    }
    assert report["passed"] is True
    assert report["control_architecture"] == (
        "scene_conditioned_gate_teacher_basis_control_v4"
    )
    assert report["control_schema_version"] == 4
    assert report["control_runtime_contract_safe"] is True
    assert report["training_artifact_isolation_requested"] is False
    assert report["training_artifact_unavailable_during_inference"] is None
    assert report["training_artifact_loaded_paths"] == []
    assert report["qa_or_oracle_loaded"] is False
    assert set(report["runtime_checkpoint_files_loaded"]) == expected_files
    assert set(report["loaded_files"]) == expected_files
