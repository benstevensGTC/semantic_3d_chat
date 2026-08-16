from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from semantic_3d_chat.chat.fixed_prefix_runtime import FixedPrefixAtlasChatRuntime
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas import (
    compile_fixed_scene_atlas,
    tensor_sha256,
    validate_probe_bank,
)
from semantic_3d_chat.scene_encoder.question_control_v7 import (
    AlwaysOnTeacherBasisFullSceneQuestionControlV7,
)
from semantic_3d_chat.training.fixed_prefix_atlas_checkpoint import (
    ATLAS_METADATA_FILENAME,
    deterministic_spherical_probe_bank,
    load_fixed_prefix_atlas_checkpoint,
    save_fixed_prefix_atlas_checkpoint,
    two_file_checkpoint_fingerprint,
    validate_fixed_prefix_atlas_metadata,
)

_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64


def _controller() -> AlwaysOnTeacherBasisFullSceneQuestionControlV7:
    generator = torch.Generator().manual_seed(1701)
    basis, _ = torch.linalg.qr(torch.randn(8, 4, generator=generator))
    return AlwaysOnTeacherBasisFullSceneQuestionControlV7(
        8,
        basis.T.contiguous(),
        control_tokens=2,
        expected_environment_latents=4,
        moment_count=2,
        interaction_dim=3,
        trunk_dim=5,
        maximum_control_rms=0.2,
        initial_control_rms=0.05,
    ).eval()


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1702)
    prefix = torch.randn(1, 6, 8, generator=generator)
    probes = torch.randn(3, 8, generator=generator)
    return prefix, probes


def _probe_audit(probes: torch.Tensor) -> dict[str, object]:
    return {
        "algorithm": "deterministic_farthest_first_spherical_kmeans_v1",
        "source_vector_count": 9,
        "source_vector_set_sha256": _D,
        "probe_bank_sha256": tensor_sha256(probes),
    }


def test_compiler_has_no_user_question_argument_and_is_deterministic() -> None:
    assert "question" not in inspect.signature(compile_fixed_scene_atlas).parameters
    prefix, probes = _inputs()
    controller = _controller()
    first = compile_fixed_scene_atlas(prefix, controller, probes)
    second = compile_fixed_scene_atlas(prefix, controller, probes)
    assert torch.equal(first.scene_prefix, second.scene_prefix)
    assert first.audit == second.audit
    assert first.audit.compiled_before_user_question is True
    assert first.audit.user_question_inputs_used_for_compilation is False
    assert first.audit.question_dependent_scene_processing is False
    assert first.audit.question_dependent_retrieval is False


def test_compiler_preserves_complete_base_and_appends_every_probe_value() -> None:
    prefix, probes = _inputs()
    result = compile_fixed_scene_atlas(prefix, _controller(), probes)
    # BOI + all four base latents are byte-identical at the front; EOI stays last.
    assert torch.equal(result.scene_prefix[:, :5], prefix[:, :-1])
    assert torch.equal(result.scene_prefix[:, -1:], prefix[:, -1:])
    assert result.scene_prefix.shape == (1, 15, 8)
    assert result.atlas_values.shape == (3, 2, 8)
    assert result.audit.atlas_memory_token_count == 9
    assert result.audit.complete_atlas_appended is True
    assert result.audit.every_probe_processed is True
    assert result.audit.semantic_or_spatial_top_k_selection is False


def test_compiled_prefix_changes_with_scene_but_not_external_text() -> None:
    prefix, probes = _inputs()
    first = compile_fixed_scene_atlas(prefix, _controller(), probes)
    changed = prefix.clone()
    changed[:, 2] += 0.4
    second = compile_fixed_scene_atlas(changed, _controller(), probes)
    assert first.audit.base_scene_prefix_sha256 != second.audit.base_scene_prefix_sha256
    assert first.audit.fixed_scene_prefix_sha256 != second.audit.fixed_scene_prefix_sha256
    # There is no post-compilation question operation capable of changing it.
    assert prefix_sha256(first.scene_prefix) == first.audit.fixed_scene_prefix_sha256


def test_probe_validation_rejects_zero_nonfinite_and_wrong_width() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        validate_probe_bank(torch.zeros(2, 8), hidden_size=8)
    bad = torch.ones(2, 8)
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_probe_bank(bad, hidden_size=8)
    with pytest.raises(ValueError, match="hidden size"):
        validate_probe_bank(torch.ones(2, 7), hidden_size=8)


def test_deterministic_spherical_probe_bank_retains_no_source_records() -> None:
    vectors = torch.randn(19, 8, generator=torch.Generator().manual_seed(17))
    first, first_audit = deterministic_spherical_probe_bank(
        vectors, probe_count=5, iterations=8
    )
    second, second_audit = deterministic_spherical_probe_bank(
        vectors, probe_count=5, iterations=8
    )
    assert torch.equal(first, second)
    assert first_audit == second_audit
    assert first_audit["source_records_retained"] is False
    assert first_audit["source_text_retained"] is False
    assert first_audit["cluster_minimum_size"] >= 1


def test_checkpoint_round_trip_contains_only_numeric_runtime_state(tmp_path: Path) -> None:
    _prefix, probes = _inputs()
    destination = tmp_path / "strict_atlas"
    saved = save_fixed_prefix_atlas_checkpoint(
        destination,
        controller=_controller(),
        probe_embeddings=probes,
        source_controller_checkpoint_sha256=_A,
        source_controller_files={
            "control.safetensors": _B,
            "runtime_metadata.json": _C,
        },
        base_checkpoint_sha256=_B,
        base_runtime_config_sha256=_C,
        probe_audit=_probe_audit(probes),
    )
    assert {item.name for item in destination.iterdir()} == {
        "atlas.safetensors",
        "runtime_metadata.json",
    }
    loaded = load_fixed_prefix_atlas_checkpoint(
        destination,
        device=torch.device("cpu"),
        expected_hidden_size=8,
        expected_base_checkpoint_sha256=_B,
        expected_base_runtime_config_sha256=_C,
    )
    assert torch.equal(loaded.probe_embeddings, probes.float())
    assert loaded.metadata["environmental_text_inputs"] == []
    assert loaded.metadata["runtime_source_records_loaded"] is False
    assert loaded.metadata["user_question_inputs_used_for_compilation"] is False
    assert saved["metadata"] == loaded.metadata


def test_checkpoint_rejects_tamper_and_extra_files(tmp_path: Path) -> None:
    _prefix, probes = _inputs()
    destination = tmp_path / "strict_atlas"
    save_fixed_prefix_atlas_checkpoint(
        destination,
        controller=_controller(),
        probe_embeddings=probes,
        source_controller_checkpoint_sha256=_A,
        source_controller_files={
            "control.safetensors": _B,
            "runtime_metadata.json": _C,
        },
        base_checkpoint_sha256=_B,
        base_runtime_config_sha256=_C,
        probe_audit=_probe_audit(probes),
    )
    (destination / "extra.txt").write_text("not allowed", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly"):
        two_file_checkpoint_fingerprint(destination)


def test_checkpoint_rejects_symlinked_root(tmp_path: Path) -> None:
    _prefix, probes = _inputs()
    destination = tmp_path / "strict_atlas"
    save_fixed_prefix_atlas_checkpoint(
        destination,
        controller=_controller(),
        probe_embeddings=probes,
        source_controller_checkpoint_sha256=_A,
        source_controller_files={
            "control.safetensors": _B,
            "runtime_metadata.json": _C,
        },
        base_checkpoint_sha256=_B,
        base_runtime_config_sha256=_C,
        probe_audit=_probe_audit(probes),
    )
    alias = tmp_path / "atlas_alias"
    alias.symlink_to(destination, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        two_file_checkpoint_fingerprint(alias)


def test_checkpoint_rejects_mismatched_probe_audit(tmp_path: Path) -> None:
    _prefix, probes = _inputs()
    bad_audit = {**_probe_audit(probes), "probe_bank_sha256": _A}
    with pytest.raises(ValueError, match="does not match"):
        save_fixed_prefix_atlas_checkpoint(
            tmp_path / "strict_atlas",
            controller=_controller(),
            probe_embeddings=probes,
            source_controller_checkpoint_sha256=_A,
            source_controller_files={
                "control.safetensors": _B,
                "runtime_metadata.json": _C,
            },
            base_checkpoint_sha256=_B,
            base_runtime_config_sha256=_C,
            probe_audit=bad_audit,
        )


def test_metadata_rejects_semantic_text_and_question_conditioning(tmp_path: Path) -> None:
    _prefix, probes = _inputs()
    destination = tmp_path / "strict_atlas"
    saved = save_fixed_prefix_atlas_checkpoint(
        destination,
        controller=_controller(),
        probe_embeddings=probes,
        source_controller_checkpoint_sha256=_A,
        source_controller_files={
            "control.safetensors": _B,
            "runtime_metadata.json": _C,
        },
        base_checkpoint_sha256=_B,
        base_runtime_config_sha256=_C,
        probe_audit=_probe_audit(probes),
    )
    conditioned = {**saved["metadata"], "question_dependent_scene_processing": True}
    with pytest.raises(ValueError, match="contract"):
        validate_fixed_prefix_atlas_metadata(conditioned)
    semantic = {**saved["metadata"], "behavioral_evaluation_status": "chair"}
    with pytest.raises(ValueError):
        validate_fixed_prefix_atlas_metadata(semantic)


class _FakeBase:
    def __init__(self, base_prefix: torch.Tensor) -> None:
        self.config = {"language": {}}
        self.scene_id = "scene_000001"
        self.scene_prefix_hash = prefix_sha256(base_prefix)
        self._base_prefix = base_prefix

    def assert_prefix_unchanged(self) -> None:
        assert prefix_sha256(self._base_prefix) == self.scene_prefix_hash

    def startup_summary(self) -> dict[str, object]:
        return {"phase": "base", "prefix_hash": self.scene_prefix_hash}


def test_runtime_retains_no_compiler_or_probe_bank_after_startup() -> None:
    base_prefix, probes = _inputs()
    compiled = compile_fixed_scene_atlas(base_prefix, _controller(), probes)
    runtime = FixedPrefixAtlasChatRuntime(
        _FakeBase(base_prefix),  # type: ignore[arg-type]
        fixed_scene_prefix=compiled.scene_prefix,
        atlas_audit=compiled.audit,
        atlas_metadata={"probe_count": 3},
        atlas_checkpoint_path=Path("opaque_checkpoint"),
    )
    assert not hasattr(runtime, "controller")
    assert not hasattr(runtime, "probe_embeddings")
    assert runtime.questions_answered == 0
    assert runtime.current_prefix_hash() == runtime.scene_prefix_hash
    summary = runtime.startup_summary()
    assert summary["compiler_retained_after_startup"] is False
    assert summary["question_dependent_scene_processing"] is False
    assert summary["prefix_compiled_before_user_question"] is True
    assert summary["scene_prefix_computed_before_question"] is True
    assert summary["strict_fixed_environment_embedding_input"] is True
    assert summary["environment_conditioned_input_sha256"] == runtime.scene_prefix_hash
    assert summary["question_conditioned_scene_readout_tokens"] is False


def test_runtime_rejects_a_question_conditioned_audit() -> None:
    base_prefix, probes = _inputs()
    compiled = compile_fixed_scene_atlas(base_prefix, _controller(), probes)
    bad = replace(compiled.audit, user_question_inputs_used_for_compilation=True)
    with pytest.raises(ValueError, match="question-conditioned"):
        FixedPrefixAtlasChatRuntime(
            _FakeBase(base_prefix),  # type: ignore[arg-type]
            fixed_scene_prefix=compiled.scene_prefix,
            atlas_audit=bad,
            atlas_metadata={},
            atlas_checkpoint_path=Path("opaque_checkpoint"),
        )


def test_workflow_config_declares_expected_fixed_token_count() -> None:
    text = Path(
        "configs/experiments/gemma4_strict_fixed_prefix_atlas_v1.yaml"
    ).read_text(encoding="utf-8")
    assert "compiled_fixed_prefix_tokens: 738" in text
    assert "question_dependent_scene_processing: false" in text
    assert "semantic_or_spatial_top_k_selection: false" in text


def test_runtime_metadata_file_has_no_source_text_after_save(tmp_path: Path) -> None:
    _prefix, probes = _inputs()
    destination = tmp_path / "strict_atlas"
    save_fixed_prefix_atlas_checkpoint(
        destination,
        controller=_controller(),
        probe_embeddings=probes,
        source_controller_checkpoint_sha256=_A,
        source_controller_files={
            "control.safetensors": _B,
            "runtime_metadata.json": _C,
        },
        base_checkpoint_sha256=_B,
        base_runtime_config_sha256=_C,
        probe_audit=_probe_audit(probes),
    )
    metadata = json.loads((destination / ATLAS_METADATA_FILENAME).read_text())
    assert "source_path" not in metadata
    assert "question_text" not in metadata
    assert "answer_text" not in metadata
    assert metadata["environmental_text_inputs"] == []
