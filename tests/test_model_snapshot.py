from __future__ import annotations

from pathlib import Path

from huggingface_hub import constants as hub_constants

from semantic_3d_chat.chat import model_snapshot
from semantic_3d_chat.chat.runtime_config import load_runtime_config


def test_local_model_snapshot_identity_hashes_exact_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    revision = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
    snapshot = (
        tmp_path
        / "models--google--gemma-4-E2B-it"
        / "snapshots"
        / revision
    )
    snapshot.mkdir(parents=True)
    model_file = snapshot / "model.safetensors"
    model_file.write_bytes(b"first-model-bytes")
    (snapshot / "config.json").write_bytes(b'{"hidden_size":1536}')
    for name in ("processor_config.json", "tokenizer.json", "tokenizer_config.json"):
        (snapshot / name).write_bytes(b"{}")
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(tmp_path))
    config = load_runtime_config("configs/runtime/gemma4_primary.yaml")

    first = model_snapshot.local_model_snapshot_identity(config)
    model_file.write_bytes(b"different-model-bytes")
    second = model_snapshot.local_model_snapshot_identity(config)

    assert first["file_count"] == 5
    assert first["tree_sha256"] != second["tree_sha256"]
    assert all(len(entry["sha256"]) == 64 for entry in first["files"])
