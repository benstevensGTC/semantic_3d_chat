from __future__ import annotations

import hashlib
import inspect
import json
import shutil
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import v75_fixed_atlas_behavior_result as result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_persisted_prepare_prediction_and_score_evidence_authenticate() -> None:
    authenticated = result.authenticate_v75_fixed_atlas_behavior_result()

    assert authenticated["measurement_authenticated"] is True
    assert authenticated["status"] == "authenticated_behavior_measured_not_promoted"
    assert authenticated["filesystem_read_only"] is True
    assert authenticated["model_loaded"] is False
    assert authenticated["inference_executed"] is False
    assert authenticated["scene_map_loaded"] is False
    assert authenticated["checkpoint_loaded"] is False
    assert authenticated["prefix_invariance_passed"] is True
    assert authenticated["predictor_reference_isolation_passed"] is True
    assert authenticated["forbidden_runtime_access_count"] == 0
    assert authenticated["fixed_v75_atlas"]["correct"] == 6
    assert authenticated["direct_exact_v75"]["correct"] == 9
    assert authenticated["frozen_v54"]["correct"] == 6
    assert authenticated["prediction_change_units"] == {
        "fixed_v75_atlas": 1,
        "direct_exact_v75": 2,
        "frozen_v54": 1,
        "total": 8,
    }
    assert authenticated["runtime_promotion_authorized"] is False


def test_every_persisted_evidence_digest_is_immutably_pinned() -> None:
    root = Path(result.PREPARED_ROOT)
    assert {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()} == set(
        result.PREPARED_FILE_SHA256
    )
    for relative, expected in result.PREPARED_FILE_SHA256.items():
        assert _sha256(root / relative) == expected
    assert _sha256(Path(result.PREDICTIONS)) == result.PREDICTIONS_SHA256
    assert _sha256(Path(result.SCORE)) == result.SCORE_SHA256


def test_authentication_recomputes_score_and_prefix_evidence() -> None:
    prediction = json.loads(Path(result.PREDICTIONS).read_text(encoding="utf-8"))
    score = json.loads(Path(result.SCORE).read_text(encoding="utf-8"))
    prefix = prediction["scene_prefix"]

    assert prefix["prefix_hashes_before"] == prefix["prefix_hashes_after"]
    assert prefix["all_scenes_compiled_before_question_manifest_opened"] is True
    assert prefix["question_inputs_used_for_compilation"] is False
    assert prefix["question_dependent_retrieval"] is False
    assert prediction["leakage"]["forbidden_access_count"] == 0
    assert prediction["leakage"]["scorer_reference_files_loaded"] is False
    assert score["prediction_artifact_sha256"] == result.PREDICTIONS_SHA256
    assert (
        score["reference_artifact_sha256"] == result.PREPARED_FILE_SHA256["scorer/references.jsonl"]
    )
    assert score["fixed_atlas_accuracy_gain_over_v54"] == 0.0
    assert score["fixed_atlas_accuracy_gap_to_direct_v75"] == -0.1875


def test_authenticator_fails_closed_on_preparation_tamper(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    shutil.copytree(result.PREPARED_ROOT, prepared)
    metadata = prepared / "probe_bank/runtime_metadata.json"
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["probe_count"] = 95
    metadata.write_text(json.dumps(payload), encoding="utf-8")

    authenticated = result.authenticate_v75_fixed_atlas_behavior_result(prepared_root=prepared)
    assert authenticated["measurement_authenticated"] is False
    assert "prepared artifact digest differs" in authenticated["error"]
    assert authenticated["inference_executed"] is False


@pytest.mark.parametrize("artifact", ["predictions", "score"])
def test_authenticator_fails_closed_on_result_tamper(artifact: str, tmp_path: Path) -> None:
    prediction = tmp_path / "prediction.json"
    score = tmp_path / "score.json"
    shutil.copyfile(result.PREDICTIONS, prediction)
    shutil.copyfile(result.SCORE, score)
    target = prediction if artifact == "predictions" else score
    target.write_bytes(target.read_bytes() + b" ")

    authenticated = result.authenticate_v75_fixed_atlas_behavior_result(
        predictions_path=prediction,
        score_path=score,
    )
    assert authenticated["measurement_authenticated"] is False
    assert f"{artifact.removesuffix('s')} artifact digest differs" in authenticated["error"]
    assert authenticated["runtime_promotion_authorized"] is False


def test_authenticator_executes_under_a_write_denying_path_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open = Path.open

    def guarded_open(path: Path, mode: str = "r", *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        assert not set(mode) & set("wax+")
        return original_open(path, mode, *args, **kwargs)

    def reject_write(*_args: object, **_kwargs: object) -> None:
        pytest.fail("read-only result authentication attempted a filesystem mutation")

    monkeypatch.setattr(Path, "open", guarded_open)
    for method in ("write_text", "write_bytes", "mkdir", "unlink", "rename", "replace"):
        monkeypatch.setattr(Path, method, reject_write)

    authenticated = result.authenticate_v75_fixed_atlas_behavior_result()
    assert authenticated["measurement_authenticated"] is True


def test_result_target_is_dependency_free_and_cannot_start_inference() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    target = "v75-fixed-atlas-behavior-result"
    recipe = makefile[makefile.index(f"{target}:") :]
    recipe = recipe[: recipe.index("\n\n")]

    assert recipe.splitlines()[0] == f"{target}:"
    assert "$(PYTHON) -m semantic_3d_chat.evaluation.v75_fixed_atlas_behavior_result" in recipe
    assert "GEMMA4_PYTHON" not in recipe
    assert "prepare" not in recipe
    assert "predict" not in recipe
    assert "score_v75" not in recipe
    source = inspect.getsource(result._authenticate)
    assert "StaticChatRuntime" not in source
    assert "load_local_language_model" not in source
    assert "predict(" not in source
    module_source = Path(inspect.getsourcefile(result) or "").read_text(encoding="utf-8")
    assert "from semantic_3d_chat.evaluation.v75_fixed_atlas_behavior import" not in (module_source)
    assert "import torch" not in module_source
