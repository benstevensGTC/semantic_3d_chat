from __future__ import annotations

import hashlib
from pathlib import Path

import scripts.check_demo_artifacts as readiness


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_readiness_reports_exact_missing_and_mismatched_assets(tmp_path: Path, monkeypatch) -> None:
    valid = b"valid"
    wrong = b"wrong"
    present = tmp_path / "present.bin"
    present.write_bytes(wrong)
    missing = tmp_path / "missing.bin"
    model_weight = tmp_path / "model.safetensors"
    model_weight.write_bytes(valid)
    model_config = tmp_path / "config.json"
    model_config.write_bytes(valid)
    manifest = {
        "artifact": "fixture",
        "environmental_text_inputs": [],
        "distribution_url": None,
        "artifacts": [
            {
                "path": str(present),
                "role": "numeric_fixture",
                "size_bytes": len(valid),
                "sha256": _digest(valid),
            },
            {
                "path": str(missing),
                "role": "missing_fixture",
                "size_bytes": 99,
                "sha256": "0" * 64,
            },
        ],
        "model": {
            "model_id": "fixture/model",
            "revision": "revision",
            "required_files": ["config.json", "model.safetensors"],
            "weights_size_bytes": len(valid),
            "weights_sha256": _digest(valid),
        },
    }
    cache = {"config.json": model_config, "model.safetensors": model_weight}
    monkeypatch.setattr(
        readiness,
        "try_to_load_from_cache",
        lambda _model, filename, revision: str(cache[filename]),
    )

    result = readiness.check_readiness(manifest)

    assert result["ready"] is False
    assert result["loads_model"] is False and result["runs_blender"] is False
    assert result["environmental_text_inputs"] == []
    mismatch, absent = result["artifacts"]
    assert mismatch["path"] == str(present)
    assert mismatch["required_size_bytes"] == len(valid)
    assert mismatch["observed_size_bytes"] == len(wrong)
    assert mismatch["sha256_matches"] is False
    assert absent["path"] == str(missing)
    assert absent["required_size_bytes"] == 99
    assert absent["required_sha256"] == "0" * 64
    assert absent["exists"] is False
    assert result["missing_or_invalid"] == [
        {
            "kind": "project_artifact",
            "path": str(present),
            "role": "numeric_fixture",
            "exists": True,
            "required_size_bytes": len(valid),
            "required_sha256": _digest(valid),
        },
        {
            "kind": "project_artifact",
            "path": str(missing),
            "role": "missing_fixture",
            "exists": False,
            "required_size_bytes": 99,
            "required_sha256": "0" * 64,
        },
    ]


def test_fast_readiness_hashes_artifacts_but_not_large_model(tmp_path: Path, monkeypatch) -> None:
    payload = b"artifact"
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(payload)
    model_weight = tmp_path / "model.safetensors"
    model_weight.write_bytes(payload)
    model_config = tmp_path / "config.json"
    model_config.write_bytes(b"{}")
    manifest = {
        "artifact": "fixture",
        "environmental_text_inputs": [],
        "distribution_url": None,
        "artifacts": [
            {
                "path": str(artifact),
                "role": "numeric_fixture",
                "size_bytes": len(payload),
                "sha256": "0" * 64,
            }
        ],
        "model": {
            "model_id": "fixture/model",
            "revision": "revision",
            "required_files": ["config.json", "model.safetensors"],
            "weights_size_bytes": len(payload),
            "weights_sha256": "0" * 64,
        },
    }
    cache = {"config.json": model_config, "model.safetensors": model_weight}
    monkeypatch.setattr(
        readiness,
        "try_to_load_from_cache",
        lambda _model, filename, revision: str(cache[filename]),
    )
    hashed: list[Path] = []
    original_sha256 = readiness._sha256

    def record_hash(path: Path) -> str:
        hashed.append(path)
        return original_sha256(path)

    monkeypatch.setattr(readiness, "_sha256", record_hash)

    result = readiness.check_readiness(manifest, verify_model_hash=False)

    assert result["ready"] is False
    assert result["artifact_hashes_verified"] is True
    assert result["model_weights_hash_verified"] is False
    assert hashed == [artifact]


def test_missing_model_weight_reports_exact_revision_size_and_hash(
    tmp_path: Path, monkeypatch
) -> None:
    payload = b"artifact"
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(payload)
    model_config = tmp_path / "config.json"
    model_config.write_bytes(b"{}")
    manifest = {
        "artifact": "fixture",
        "environmental_text_inputs": [],
        "distribution_url": None,
        "artifacts": [
            {
                "path": str(artifact),
                "role": "numeric_fixture",
                "size_bytes": len(payload),
                "sha256": _digest(payload),
            }
        ],
        "model": {
            "model_id": "fixture/model",
            "revision": "exact-revision",
            "required_files": ["config.json", "model.safetensors"],
            "weights_size_bytes": 123456,
            "weights_sha256": "a" * 64,
        },
    }
    monkeypatch.setattr(
        readiness,
        "try_to_load_from_cache",
        lambda _model, filename, revision: str(model_config) if filename == "config.json" else None,
    )

    result = readiness.check_readiness(manifest, verify_model_hash=False)

    assert result["ready"] is False
    assert result["missing_or_invalid"] == [
        {
            "kind": "model_file",
            "model_id": "fixture/model",
            "revision": "exact-revision",
            "filename": "model.safetensors",
            "exists": False,
            "required_size_bytes": 123456,
            "required_sha256": "a" * 64,
        }
    ]
