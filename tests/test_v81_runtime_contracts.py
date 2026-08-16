from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

import semantic_3d_chat.chat.runtime_config as runtime_config_module
import semantic_3d_chat.evaluation.v81_scene_memory_leakage as leakage_module
from semantic_3d_chat.chat.v81_scene_memory_cli import _parser as chat_parser
from semantic_3d_chat.chat.v81_scene_memory_runtime import (
    V81SceneMemoryChatRuntime,
)
from semantic_3d_chat.evaluation.v75_fixed_atlas_behavior import (
    PREFIX_MANIFEST_SHA256,
)
from semantic_3d_chat.evaluation.v81_historical_behavior import (
    _fixed_from_banks,
    _source_provenance,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.language.v81_structured_dense_atlas_sidecar import (
    FIXED_PREFIX_TOKENS,
    HIDDEN_SIZE,
    reconstruct_base_v54_prefix_v81,
    split_v75_v2_prefix_v81,
)
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas import tensor_sha256
from semantic_3d_chat.scene_encoder.v81_scene_memory_artifact import (
    MEMORY_FILENAME,
    METADATA_FILENAME,
    LoadedV81SceneMemory,
    build_v81_scene_memory_metadata,
    save_v81_scene_memory,
)

_BASE_SHA = "a" * 64
_RUNTIME_SHA = "b" * 64
_CONTROL_SHA = "c" * 64
_PROBE_SHA = "d" * 64


def _memory(seed: int = 8110) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(
            1,
            FIXED_PREFIX_TOKENS,
            HIDDEN_SIZE,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.01
    ).to(torch.bfloat16)


def _save(destination: Path, memory: torch.Tensor) -> dict[str, Any]:
    return save_v81_scene_memory(
        destination,
        memory,
        scene_id="scene_000001",
        source_base_checkpoint_sha256=_BASE_SHA,
        runtime_config_sha256=_RUNTIME_SHA,
        source_control_checkpoint_sha256=_CONTROL_SHA,
        source_probe_tensor_sha256=_PROBE_SHA,
    )


def test_v81_scene_memory_is_create_once_and_preserves_existing_bytes(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "scene_000001"
    _save(destination, _memory())
    before = {
        name: (destination / name).read_bytes()
        for name in (MEMORY_FILENAME, METADATA_FILENAME)
    }

    with pytest.raises(FileExistsError):
        _save(destination, _memory(8111))

    assert {
        name: (destination / name).read_bytes()
        for name in (MEMORY_FILENAME, METADATA_FILENAME)
    } == before


@pytest.mark.parametrize("component", ["training", ".oracle-unavailable-123-deadbeef"])
def test_v81_scene_memory_rejects_forbidden_runtime_location(
    tmp_path: Path,
    component: str,
) -> None:
    with pytest.raises(ValueError, match="forbidden data"):
        _save(tmp_path / component / "scene_000001", _memory())


def test_v81_zero_and_shuffle_controls_have_exact_tensor_contracts() -> None:
    memory = _memory()
    banks = split_v75_v2_prefix_v81(memory)

    zero = _fixed_from_banks(
        memory,
        atlas_values=torch.zeros_like(banks.atlas_values),
        zero_base_latents=True,
        zero_keys=True,
    )
    zero_banks = split_v75_v2_prefix_v81(zero)
    assert torch.equal(zero_banks.boi, banks.boi)
    assert torch.equal(zero_banks.eoi, banks.eoi)
    assert torch.count_nonzero(zero_banks.probe_keys).item() == 0
    assert torch.count_nonzero(zero_banks.atlas_values).item() == 0
    assert torch.count_nonzero(zero_banks.base_latents).item() == 0

    expected_values = banks.atlas_values.roll(shifts=1, dims=1)
    shuffled = _fixed_from_banks(memory, atlas_values=expected_values)
    shuffled_banks = split_v75_v2_prefix_v81(shuffled)
    assert torch.equal(shuffled_banks.boi, banks.boi)
    assert torch.equal(shuffled_banks.eoi, banks.eoi)
    assert torch.equal(shuffled_banks.probe_keys, banks.probe_keys)
    assert torch.equal(shuffled_banks.base_latents, banks.base_latents)
    assert torch.equal(shuffled_banks.atlas_values, expected_values)
    assert not torch.equal(shuffled_banks.atlas_values, banks.atlas_values)


def test_v81_historical_provenance_does_not_mislabel_base_checkpoint_hash() -> None:
    source = _source_provenance(
        probe_metadata={"probe_tensor_sha256": "1" * 64},
        controller_metadata={
            "architecture": "controller",
            "weights_sha256": "2" * 64,
        },
        prefix_manifest={"base_checkpoint_sha256": "3" * 64},
        question_metadata={"questions_file_sha256": "4" * 64},
    )

    assert source["prefix_cache_manifest_sha256"] == PREFIX_MANIFEST_SHA256
    assert source["prefix_cache_base_checkpoint_sha256"] == "3" * 64
    assert source["prefix_cache_manifest_sha256"] != source[
        "prefix_cache_base_checkpoint_sha256"
    ]


class _BaseRuntime:
    def __init__(self, scene_prefix: torch.Tensor) -> None:
        self.config: dict[str, Any] = {}
        self.scene_id = "scene_000001"
        self.scene_prefix = scene_prefix
        self.assertion_count = 0

    def assert_prefix_unchanged(self) -> None:
        self.assertion_count += 1

    def startup_summary(self) -> dict[str, Any]:
        return {"base_runtime_ready": True}


def test_v81_runtime_binds_fixed_hash_before_questions_and_fails_on_mutation(
    tmp_path: Path,
) -> None:
    memory = _memory()
    base_prefix = reconstruct_base_v54_prefix_v81(memory)
    metadata = build_v81_scene_memory_metadata(
        memory,
        scene_id="scene_000001",
        tensor_file_sha256="e" * 64,
        source_base_checkpoint_sha256=_BASE_SHA,
        runtime_config_sha256=_RUNTIME_SHA,
        source_control_checkpoint_sha256=_CONTROL_SHA,
        source_probe_tensor_sha256=_PROBE_SHA,
    )
    loaded = LoadedV81SceneMemory(
        root=tmp_path / "scene_000001",
        memory=memory,
        metadata=metadata,
    )
    runtime = V81SceneMemoryChatRuntime(_BaseRuntime(base_prefix), loaded)  # type: ignore[arg-type]

    assert runtime.questions_answered == 0
    assert runtime.current_prefix_hash() == prefix_sha256(memory)
    assert runtime.scene_memory_tensor_sha256 == tensor_sha256(memory)
    startup = runtime.startup_summary()
    assert startup["scene_prefix_computed_before_question"] is True
    assert startup["same_fixed_memory_reused_for_every_question"] is True

    runtime.fixed_scene_memory[0, 2, 7] += 0.125
    with pytest.raises((RuntimeError, ValueError), match="changed after"):
        runtime.assert_prefix_unchanged()


def test_v81_chat_parser_requires_sealed_memory_and_preserves_question_order() -> None:
    parser = chat_parser()
    args = parser.parse_args(
        [
            "--config",
            "runtime.yaml",
            "--scene",
            "scene_000001",
            "--base-checkpoint",
            "base",
            "--scene-memory",
            "memory",
            "--question",
            "first",
            "--question",
            "second",
        ]
    )
    assert args.question == ["first", "second"]
    assert args.scene_memory == "memory"

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--config",
                "runtime.yaml",
                "--scene",
                "scene_000001",
                "--base-checkpoint",
                "base",
            ]
        )


class _FakeAnswer:
    def __init__(self, question: str) -> None:
        self.question = question

    def to_dict(self) -> dict[str, Any]:
        return {"question": self.question, "answer": "numeric-memory-answer"}


class _FakeLeakageRuntime:
    def __init__(self, config: dict[str, Any], *, change_hash: bool = False) -> None:
        self.config = config
        self.questions_answered = 0
        self.base_scene_prefix_hash = "6" * 64
        self._fixed_hash = "5" * 64
        self._change_hash = change_hash
        self.last_reader_audit: dict[str, Any] | None = None
        self.last_prepared_layout_audit: dict[str, Any] | None = None

    def startup_summary(self) -> dict[str, Any]:
        return {
            "scene_prefix_computed_before_question": True,
            "compiler_or_probe_bank_loaded_by_chat": False,
        }

    def current_prefix_hash(self) -> str:
        if self._change_hash and self.questions_answered:
            return "7" * 64
        return self._fixed_hash

    def answer(self, question: str) -> _FakeAnswer:
        self.questions_answered += 1
        self.last_reader_audit = {
            "all_96_groups_positive": True,
            "all_384_values_receive_positive_floor_weight": True,
            "question_dependent_scene_retrieval": False,
            "semantic_or_spatial_top_k_selection": False,
            "environmental_text_inputs": [],
        }
        self.last_prepared_layout_audit = {
            "base_scene_prefix_tokens": 258,
            "control_activation_tokens": 4,
            "control_pad_ple": True,
            "control_text_modality_zero": True,
        }
        return _FakeAnswer(question)

    def assert_prefix_unchanged(self) -> None:
        return None


def _leakage_roots(tmp_path: Path) -> dict[str, Path]:
    roots = {
        kind: tmp_path / "data" / kind
        for kind in ("oracle", "qa", "rendered", "features")
    }
    roots["oracle"].mkdir(parents=True)
    (roots["oracle"] / "private.bin").write_bytes(b"private")
    return roots


def _patch_leakage_config(
    monkeypatch: pytest.MonkeyPatch,
    roots: dict[str, Path],
    config: dict[str, Any],
) -> None:
    def fake_load_runtime_config(
        path: str | Path,
        *,
        record_file: Any | None = None,
    ) -> dict[str, Any]:
        if record_file is not None:
            record_file(path)
        return config

    monkeypatch.setattr(
        runtime_config_module,
        "load_runtime_config",
        fake_load_runtime_config,
    )
    monkeypatch.setattr(
        leakage_module,
        "artifact_root",
        lambda _config, kind: roots[kind],
    )


def test_v81_leakage_audit_is_model_free_and_hash_invariant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = _leakage_roots(tmp_path)
    config: dict[str, Any] = {"_runtime_safe_config": True}
    _patch_leakage_config(monkeypatch, roots, config)
    fake = _FakeLeakageRuntime(config)

    def fake_load(*_args: Any, **_kwargs: Any) -> _FakeLeakageRuntime:
        return fake

    monkeypatch.setattr(V81SceneMemoryChatRuntime, "load", fake_load)
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text("runtime: safe\n", encoding="utf-8")
    report_path = tmp_path / "reports" / "leakage.json"
    report = leakage_module.run_v81_scene_memory_leakage(
        config_path=config_path,
        scene_id="scene_000001",
        base_checkpoint=tmp_path / "base",
        scene_memory=tmp_path / "memory",
        compiler_checkpoint=tmp_path / "compiler",
        probe_bank=tmp_path / "probes",
        questions=("question one", "question two"),
        report_path=report_path,
    )

    assert report["passed"] is True
    assert report["oracle_was_renamed"] is True
    assert report["oracle_unavailable_during_inference"] is True
    assert report["oracle_restored"] is True
    assert report["fixed_738_memory_invariant"] is True
    assert report["base_258_prefix_invariant"] is True
    assert report["forbidden_accesses"] == []
    assert roots["oracle"].is_dir()


@pytest.mark.parametrize("change_hash", [True, False])
def test_v81_leakage_audit_rejects_hash_change_or_forbidden_oracle_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change_hash: bool,
) -> None:
    roots = _leakage_roots(tmp_path)
    config: dict[str, Any] = {"_runtime_safe_config": True}
    _patch_leakage_config(monkeypatch, roots, config)

    if change_hash:
        fake = _FakeLeakageRuntime(config, change_hash=True)

        def fake_load(*_args: Any, **_kwargs: Any) -> _FakeLeakageRuntime:
            return fake

    else:

        def fake_load(*_args: Any, **_kwargs: Any) -> _FakeLeakageRuntime:
            hidden = next(roots["oracle"].parent.glob(".oracle-unavailable-*"))
            (hidden / "private.bin").read_bytes()
            raise AssertionError("forbidden read should have been blocked")

    monkeypatch.setattr(V81SceneMemoryChatRuntime, "load", fake_load)
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text("runtime: safe\n", encoding="utf-8")
    report_path = tmp_path / "reports" / "failed.json"
    with pytest.raises(RuntimeError, match="V81 leakage audit"):
        leakage_module.run_v81_scene_memory_leakage(
            config_path=config_path,
            scene_id="scene_000001",
            base_checkpoint=tmp_path / "base",
            scene_memory=tmp_path / "memory",
            compiler_checkpoint=tmp_path / "compiler",
            probe_bank=tmp_path / "probes",
            questions=("one question",),
            report_path=report_path,
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["oracle_directory"] == str(roots["oracle"].resolve())
    assert report["oracle_restored"] is True
    assert roots["oracle"].is_dir()
    if change_hash:
        assert report["fixed_738_memory_invariant"] is False
    else:
        assert report["failure"].startswith("PermissionError:")
        assert report["forbidden_accesses"]
