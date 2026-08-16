from pathlib import Path

from semantic_3d_chat.language.local_lm import local_model_snapshot_files


def test_local_model_snapshot_enumeration_covers_native_weight_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    hub = tmp_path / "hub"
    snapshot = hub / "models--local--model" / "snapshots" / "revision-1"
    snapshot.mkdir(parents=True)
    config = snapshot / "config.json"
    weight = snapshot / "model-00001-of-00002.safetensors"
    config.write_text("{}", encoding="utf-8")
    weight.write_bytes(b"weights")
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))

    assert local_model_snapshot_files("local/model", "revision-1") == (
        config.resolve(),
        weight.resolve(),
    )
