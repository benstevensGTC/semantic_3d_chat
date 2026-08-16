from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import scripts.prepare_demo_runtime as release


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "adapter.safetensors").write_bytes(b"numeric-adapter")
    runtime_metadata = Path(
        "data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000/runtime_metadata.json"
    ).read_bytes()
    (source / "runtime_metadata.json").write_bytes(runtime_metadata)
    # Representative forbidden training text exists only at the source.
    (source / "metadata.json").write_text(
        '{"answer_text":"left","object_category":"chair"}\n', encoding="utf-8"
    )
    destination = tmp_path / "runtime"
    manifest = {
        "schema_version": 1,
        "artifact": "semantic_3d_chat_local_demo_runtime_release_v1",
        "source_checkpoint": str(source),
        "runtime_checkpoint": str(destination),
        "inference_inventory": ["adapter.safetensors", "runtime_metadata.json"],
        "files": {
            name: {"sha256": _sha256(source / name), "size_bytes": (source / name).stat().st_size}
            for name in ("adapter.safetensors", "runtime_metadata.json")
        },
        "training_metadata_included": False,
        "environmental_text_inputs": [],
    }
    return manifest, source, destination


def test_prepare_demo_runtime_copies_only_two_sanitized_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, source, destination = _fixture(tmp_path, monkeypatch)

    result = release.prepare_runtime_release(manifest)

    assert result["validated"] is True
    assert {path.name for path in destination.iterdir()} == {
        "adapter.safetensors",
        "runtime_metadata.json",
    }
    assert not (destination / "metadata.json").exists()
    assert (source / "metadata.json").exists()


def test_prepare_demo_runtime_fails_closed_on_existing_extra_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, _source, destination = _fixture(tmp_path, monkeypatch)
    release.prepare_runtime_release(manifest)
    (destination / "metadata.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly the two sanitized"):
        release.prepare_runtime_release(manifest)


def test_missing_bootstrap_source_reports_expected_sizes_and_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, source, _destination = _fixture(tmp_path, monkeypatch)
    for child in source.iterdir():
        child.unlink()
    source.rmdir()

    with pytest.raises(FileNotFoundError) as failure:
        release.prepare_runtime_release(manifest)

    message = str(failure.value)
    for name, expected in manifest["files"].items():
        assert name in message
        assert str(expected["size_bytes"]) in message
        assert expected["sha256"] in message
