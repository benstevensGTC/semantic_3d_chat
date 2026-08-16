from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from semantic_3d_chat.chat import fixed_prefix_ple_reader_runtime as v54_runtime
from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.runtime_config import (
    effective_runtime_config_sha256,
    load_runtime_config,
)
from semantic_3d_chat.mcp_server import server as mcp_server
from semantic_3d_chat.scene_encoder.question_control_v75 import (
    DenseFullSceneContinuousControlV75,
)
from semantic_3d_chat.training.question_control_v75_checkpoint import (
    V75_RUNTIME_ARCHITECTURE,
    save_v75_control_checkpoint,
    v75_state_sha256,
)

ROOT = Path(__file__).parents[1]
V54_MODEL_ID = "google/gemma-4-E2B-it"
V54_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_v54_fixture(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    legacy_metadata: bool,
) -> tuple[Path, Path]:
    root.mkdir()
    adapter = root / "adapter.safetensors"
    runtime_metadata = root / "runtime_metadata.json"
    offline_metadata = root / "metadata.json"
    adapter.write_bytes(b"tiny numeric adapter fixture")
    runtime_metadata.write_text(
        json.dumps(
            {
                "semantic_dim": 3072,
                "language_hidden_dim": 1536,
                "language_model_id": V54_MODEL_ID,
                "language_revision": V54_REVISION,
                "scene_latents": 256,
                "scene_model_dim": 384,
                "question_dependent_scene_processing": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if legacy_metadata:
        offline_metadata.write_text(
            '{"offline_provenance_only":true}\n', encoding="utf-8"
        )
    monkeypatch.setattr(v54_runtime, "_V54_ADAPTER_SHA256", _sha256(adapter))
    monkeypatch.setattr(
        v54_runtime,
        "_V54_RUNTIME_METADATA_SHA256",
        _sha256(runtime_metadata),
    )
    return root, offline_metadata


def _v54_config() -> dict[str, object]:
    return {
        "scene_encoder": {
            "language_aligned_tail_dim": 1536,
            "global_latents": 256,
            "model_dim": 384,
        },
        "language": {
            "model_id": V54_MODEL_ID,
            "revision": V54_REVISION,
        },
    }


@pytest.mark.parametrize("legacy_metadata", [False, True])
def test_v54_reader_accepts_minimal_and_legacy_inventory_without_metadata_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_metadata: bool,
) -> None:
    checkpoint, offline_metadata = _write_v54_fixture(
        tmp_path / ("base_legacy" if legacy_metadata else "base_minimal"),
        monkeypatch,
        legacy_metadata=legacy_metadata,
    )
    audit = FileAccessAudit(block_forbidden=True)

    with audit:
        result = v54_runtime.validate_v54_checkpoint(checkpoint, audit=audit)

    assert result.root == checkpoint.resolve()
    assert set(audit.unique_paths) == {
        str((checkpoint / "adapter.safetensors").resolve()),
        str((checkpoint / "runtime_metadata.json").resolve()),
    }
    assert audit.forbidden_accesses() == []
    if legacy_metadata:
        assert offline_metadata.resolve() in audit.forbidden_roots
    else:
        assert not offline_metadata.exists()


@pytest.mark.parametrize("legacy_metadata", [False, True])
def test_mcp_base_preflight_accepts_two_or_three_files_without_training_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_metadata: bool,
) -> None:
    checkpoint, offline_metadata = _write_v54_fixture(
        tmp_path / ("mcp_legacy" if legacy_metadata else "mcp_minimal"),
        monkeypatch,
        legacy_metadata=legacy_metadata,
    )
    audit = FileAccessAudit(block_forbidden=True)

    with audit:
        result = mcp_server._base_checkpoint_preflight(
            checkpoint,
            _v54_config(),
            semantic_dim=3072,
            audit=audit,
        )

    assert result["training_metadata_opened"] is False
    assert result["exact_v54_release_authenticated"] is True
    assert result["checkpoint_identity_sha256"] == mcp_server._V54_BASE_CHECKPOINT_ID
    assert set(audit.unique_paths) == {
        str((checkpoint / "adapter.safetensors").resolve()),
        str((checkpoint / "runtime_metadata.json").resolve()),
    }
    assert audit.forbidden_accesses() == []
    if legacy_metadata:
        assert offline_metadata.resolve() in audit.forbidden_roots


def test_v75_mcp_controller_preflight_is_numeric_two_file_and_model_free(
    tmp_path: Path,
) -> None:
    config_path = ROOT / "configs/runtime/gemma4_v56_question_control.yaml"
    runtime_hash = effective_runtime_config_sha256(load_runtime_config(config_path))
    torch.manual_seed(750075)
    control = DenseFullSceneContinuousControlV75(
        8,
        torch.eye(4, 8),
        environment_latents=4,
        query_count=2,
        model_dimension=3,
        coefficient_decoder_hidden_dimension=7,
        uniform_floor_mass=0.05,
        maximum_control_rms=0.25,
    ).eval()
    checkpoint = tmp_path / "v75_control"
    save_v75_control_checkpoint(
        checkpoint,
        control=control,
        base_checkpoint_sha256=mcp_server._V54_BASE_CHECKPOINT_ID,
        base_runtime_config_sha256=runtime_hash,
        source_v75_candidate_sha256="3" * 64,
        expected_training_fit_state_sha256=v75_state_sha256(control),
        saved_runtime_training_gate_attestation_sha256="4" * 64,
    )
    audit = FileAccessAudit(block_forbidden=True)

    with audit:
        result = mcp_server._control_checkpoint_preflight(
            checkpoint,
            config_path,
            hidden_size=8,
            expected_base_checkpoint_sha256=mcp_server._V54_BASE_CHECKPOINT_ID,
            audit=audit,
        )

    assert result["architecture"] == V75_RUNTIME_ARCHITECTURE
    assert result["complete_scene_prefix_required"] is True
    assert result["question_dependent_scene_retrieval"] is False
    assert result["environmental_text_inputs"] == []
    assert {item.name for item in checkpoint.iterdir()} == {
        "control.safetensors",
        "runtime_metadata.json",
    }
    required_runtime_inputs = {
        str(config_path.resolve()),
        str((checkpoint / "control.safetensors").resolve()),
        str((checkpoint / "runtime_metadata.json").resolve()),
    }
    assert required_runtime_inputs <= set(audit.unique_paths)
    assert not any(Path(path).name == "metadata.json" for path in audit.unique_paths)
    assert not any(
        "/data_gemma4/training/" in Path(path).as_posix()
        for path in audit.unique_paths
    )
    assert audit.forbidden_accesses() == []


def test_embodied_launchers_default_to_minimal_v54_plus_promoted_v75() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    launcher = (ROOT / "scripts/run_embodied_conversation.sh").read_text(
        encoding="utf-8"
    )

    assert (
        "GEMMA4_EMBODIED_CHECKPOINT ?= "
        "data_gemma4/runtime/checkpoints/gemma4_v54_release_v1"
    ) in makefile
    assert (
        "GEMMA4_EMBODIED_CONTROL_CHECKPOINT ?= "
        "data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1"
    ) in makefile
    assert (
        "GEMMA4_EMBODIED_CONTROL_CONFIG ?= "
        "configs/runtime/gemma4_v56_question_control.yaml"
    ) in makefile
    for target in (
        "gemma4-embodied-mcp",
        "gemma4-embodied-mcp-check",
        "gemma4-embodied-mcp-live-smoke",
    ):
        start = makefile.index(f"{target}:")
        body = makefile[start : makefile.find("\n\n", start)]
        assert "GEMMA4_EMBODIED_CONTROL_CHECKPOINT" in body
        assert '--control-checkpoint "$(GEMMA4_EMBODIED_CONTROL_CHECKPOINT)"' in body
        assert '--control-runtime-config "$(GEMMA4_EMBODIED_CONTROL_CONFIG)"' in body
    live_start = makefile.index("gemma4-embodied-mcp-live-smoke:")
    live_body = makefile[live_start : makefile.find("\n\n", live_start)]
    assert "scripts/run_semantic_mcp_live_smoke.py" in live_body
    assert '--runtime-asset "$(RUNTIME_SCENE_ASSET)"' in live_body
    assert '--robot-state-checkpoint "$(GEMMA4_ROBOT_STATE_CHECKPOINT)"' in live_body
    assert (
        "data_gemma4/runtime/checkpoints/gemma4_v54_release_v1" in launcher
    )
    assert (
        "data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1"
        in launcher
    )
    assert "configs/runtime/gemma4_v56_question_control.yaml" in launcher
    assert '--control-checkpoint "$EMBODIED_CONTROL_CHECKPOINT"' in launcher
    assert '--control-runtime-config "$EMBODIED_CONTROL_CONFIG"' in launcher
