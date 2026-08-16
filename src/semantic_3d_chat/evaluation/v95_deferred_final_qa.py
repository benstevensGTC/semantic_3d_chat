"""Post-unlock exact-quota QA projection for the V95 deferred final.

The generic QA generator intentionally emits a broad candidate pool.  Its
balanced sampler does not support exact global answer-type quotas, so this
module selects complete two-sided counterfactual units with a deterministic,
predeclared contract.  It is evaluation-side and label-aware; prediction code
must never import it or open either its input or output.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import (
    sha256_file_v85,
)

SCHEMA_VERSION: Final[int] = 95
SELECTION_ARTIFACT: Final[str] = "gemma4_v95_deferred_final_exact_qa_selection_v1"
SCENE_IDS: Final[tuple[str, ...]] = tuple(
    f"scene_{index:06d}" for index in range(25, 31)
)
PAIR_SCENES: Final[dict[str, tuple[str, str]]] = {
    "pair_000013": ("scene_000025", "scene_000026"),
    "pair_000014": ("scene_000027", "scene_000028"),
    "pair_000015": ("scene_000029", "scene_000030"),
}
PAIR_UNIT_QUOTAS: Final[dict[str, int]] = {
    "attribute": 8,
    "count": 7,
    "metric": 1,
    "orientation": 1,
    "presence": 7,
    "spatial_relation": 8,
    "support": 4,
}
ANSWER_TYPE_TOTALS: Final[dict[str, int]] = {
    answer_type: count * len(PAIR_SCENES) * 2
    for answer_type, count in PAIR_UNIT_QUOTAS.items()
}
ROWS_PER_SCENE: Final[int] = sum(PAIR_UNIT_QUOTAS.values())
CHANGED_UNITS_PER_PAIR: Final[int] = 4
SELECTION_SEED: Final[int] = 950095
RAW_QA: Final[Path] = (
    PROJECT_ROOT
    / "reports/gemma4/artifacts/v95_deferred_final/qa_raw/test.jsonl"
)
FINAL_QA: Final[Path] = PROJECT_ROOT / "data_diverse52/qa/test.jsonl"
SELECTION_MANIFEST: Final[Path] = (
    PROJECT_ROOT
    / "reports/gemma4/artifacts/v95_deferred_final/qa_selection_manifest.json"
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _selection_rank(*parts: object) -> str:
    encoded = "\0".join((str(SELECTION_SEED), *(str(part) for part in parts))).encode()
    return hashlib.sha256(encoded).hexdigest()


def _strict_fixed_path(value: str | Path | None, expected: Path, label: str) -> Path:
    actual = expected.resolve() if value is None else Path(value).expanduser().resolve()
    if actual != expected.resolve():
        raise ValueError(f"V95 deferred-final {label} has one fixed path")
    if actual.is_symlink():
        raise ValueError(f"V95 deferred-final {label} may not be a symlink")
    return actual


def _unit_key(record: Mapping[str, Any]) -> tuple[str, str]:
    pair_id = record.get("counterfactual_pair_id")
    question_key = record.get("counterfactual_question_key")
    if not isinstance(pair_id, str) or not isinstance(question_key, str):
        raise TypeError("Every V95 final row must belong to a complete paired unit")
    return pair_id, question_key


def _candidate_units(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, list[tuple[str, bool, tuple[dict[str, Any], dict[str, Any]]]]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for raw in records:
        record = dict(raw)
        scene_id = record.get("scene_id")
        answer_type = record.get("answer_type")
        expected_change = record.get("counterfactual_expected_change")
        if scene_id not in SCENE_IDS:
            raise ValueError(f"V95 raw QA contains an unexpected scene: {scene_id!r}")
        if answer_type not in PAIR_UNIT_QUOTAS:
            # The broad generator intentionally produces additional families;
            # they are outside this fixed seven-type final contract.
            continue
        if not isinstance(expected_change, bool):
            raise TypeError("V95 paired QA requires a boolean expected-change flag")
        grouped[_unit_key(record)].append(record)

    result: dict[
        str,
        dict[str, list[tuple[str, bool, tuple[dict[str, Any], dict[str, Any]]]]],
    ] = {
        pair_id: {answer_type: [] for answer_type in PAIR_UNIT_QUOTAS}
        for pair_id in PAIR_SCENES
    }
    for (pair_id, question_key), members in grouped.items():
        if pair_id not in PAIR_SCENES:
            raise ValueError(f"V95 raw QA contains an unexpected pair: {pair_id!r}")
        if len(members) != 2:
            raise ValueError(f"V95 unit {pair_id}/{question_key} must have two sides")
        expected_scenes = PAIR_SCENES[pair_id]
        ordered = tuple(sorted(members, key=lambda item: str(item["scene_id"])))
        if tuple(str(item["scene_id"]) for item in ordered) != expected_scenes:
            raise ValueError(f"V95 unit {pair_id}/{question_key} has incorrect scenes")
        answer_types = {str(item["answer_type"]) for item in ordered}
        change_flags = {bool(item["counterfactual_expected_change"]) for item in ordered}
        if len(answer_types) != 1 or len(change_flags) != 1:
            raise ValueError(f"V95 unit {pair_id}/{question_key} disagrees across sides")
        answer_type = answer_types.pop()
        if answer_type not in PAIR_UNIT_QUOTAS:
            continue
        result[pair_id][answer_type].append(
            (question_key, change_flags.pop(), (ordered[0], ordered[1]))
        )
    return result


def _changed_allocation(
    pair_id: str,
    by_type: Mapping[
        str, Sequence[tuple[str, bool, tuple[dict[str, Any], dict[str, Any]]]]
    ],
) -> dict[str, int]:
    """Choose a feasible four-unit changed allocation without seeing answers."""

    answer_types = tuple(PAIR_UNIT_QUOTAS)
    ranges = []
    for answer_type in answer_types:
        candidates = by_type[answer_type]
        changed = sum(int(item[1]) for item in candidates)
        stable = len(candidates) - changed
        minimum = max(0, PAIR_UNIT_QUOTAS[answer_type] - stable)
        maximum = min(PAIR_UNIT_QUOTAS[answer_type], changed)
        ranges.append(range(minimum, maximum + 1))
    feasible = [
        values
        for values in itertools.product(*ranges)
        if sum(values) == CHANGED_UNITS_PER_PAIR
    ]
    if not feasible:
        availability = {
            answer_type: {
                "changed": sum(int(item[1]) for item in by_type[answer_type]),
                "stable": sum(int(not item[1]) for item in by_type[answer_type]),
                "required": PAIR_UNIT_QUOTAS[answer_type],
            }
            for answer_type in answer_types
        }
        raise ValueError(
            f"V95 pair {pair_id} cannot satisfy the preregistered exact quotas: "
            f"{availability}"
        )
    chosen = min(
        feasible,
        key=lambda values: _selection_rank(pair_id, "changed-allocation", *values),
    )
    return dict(zip(answer_types, chosen, strict=True))


def select_exact_final_records_v95(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select 108 complete pair units (216 sides) by a fixed hash order."""

    if not records:
        raise ValueError("V95 raw QA candidate pool is empty")
    units = _candidate_units(records)
    selected: list[dict[str, Any]] = []
    selected_unit_inventory: list[list[Any]] = []
    availability: dict[str, Any] = {}
    for pair_id in PAIR_SCENES:
        allocation = _changed_allocation(pair_id, units[pair_id])
        availability[pair_id] = {}
        pair_selected = 0
        pair_changed = 0
        for answer_type, quota in PAIR_UNIT_QUOTAS.items():
            candidates = units[pair_id][answer_type]
            changed = sorted(
                (item for item in candidates if item[1]),
                key=lambda item: _selection_rank(pair_id, answer_type, "changed", item[0]),
            )
            stable = sorted(
                (item for item in candidates if not item[1]),
                key=lambda item: _selection_rank(pair_id, answer_type, "stable", item[0]),
            )
            changed_count = allocation[answer_type]
            chosen = changed[:changed_count] + stable[: quota - changed_count]
            if len(chosen) != quota:
                raise ValueError(
                    f"V95 pair {pair_id}/{answer_type} cannot fill quota {quota}"
                )
            availability[pair_id][answer_type] = {
                "candidate_changed_units": len(changed),
                "candidate_stable_units": len(stable),
                "selected_changed_units": changed_count,
                "selected_stable_units": quota - changed_count,
            }
            for question_key, expected_change, members in chosen:
                selected.extend(dict(member) for member in members)
                selected_unit_inventory.append(
                    [pair_id, answer_type, question_key, bool(expected_change)]
                )
                pair_selected += 1
                pair_changed += int(expected_change)
        if pair_selected != ROWS_PER_SCENE or pair_changed != CHANGED_UNITS_PER_PAIR:
            raise AssertionError("V95 pair selection implementation drifted")

    selected.sort(
        key=lambda record: (
            SCENE_IDS.index(str(record["scene_id"])),
            _selection_rank(
                record["scene_id"],
                record["answer_type"],
                record["counterfactual_question_key"],
            ),
        )
    )
    scene_counts = Counter(str(record["scene_id"]) for record in selected)
    type_counts = Counter(str(record["answer_type"]) for record in selected)
    changed_sides = sum(
        int(bool(record["counterfactual_expected_change"])) for record in selected
    )
    if (
        len(selected) != len(SCENE_IDS) * ROWS_PER_SCENE
        or scene_counts != Counter({scene_id: ROWS_PER_SCENE for scene_id in SCENE_IDS})
        or dict(type_counts) != ANSWER_TYPE_TOTALS
        or changed_sides != len(PAIR_SCENES) * CHANGED_UNITS_PER_PAIR * 2
    ):
        raise AssertionError("V95 exact QA totals drifted after selection")
    manifest = {
        "artifact": SELECTION_ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "selection_seed": SELECTION_SEED,
        "scene_ids": list(SCENE_IDS),
        "pair_scenes": {key: list(value) for key, value in PAIR_SCENES.items()},
        "row_count": len(selected),
        "rows_per_scene": ROWS_PER_SCENE,
        "pair_unit_quotas": dict(PAIR_UNIT_QUOTAS),
        "answer_type_totals": dict(type_counts),
        "changed_unit_count": len(PAIR_SCENES) * CHANGED_UNITS_PER_PAIR,
        "changed_side_count": changed_sides,
        "selected_unit_inventory_sha256": _canonical_sha256(selected_unit_inventory),
        "availability": availability,
        "selection_uses_question_or_answer_text": False,
        "selection_uses_answer_values": False,
        "complete_counterfactual_units_only": True,
    }
    return selected, manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise TypeError(f"V95 QA line {line_number} is not an object")
        records.append(value)
    return records


def _serialized_jsonl(records: Sequence[Mapping[str, Any]]) -> bytes:
    return (
        "\n".join(json.dumps(dict(item), sort_keys=True, allow_nan=False) for item in records)
        + "\n"
    ).encode("utf-8")


def _create_or_authenticate_bytes(path: Path, encoded: bytes) -> bool:
    """Create once, while allowing the preregistered zero-byte placeholder."""

    if path.is_symlink():
        raise ValueError(f"V95 QA destination may not be a symlink: {path}")
    if path.exists() and path.stat().st_size:
        if path.read_bytes() != encoded:
            raise FileExistsError(f"V95 QA destination already differs: {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def select_final_qa_v95(
    *,
    config_path: str | Path,
    unlock_path: str | Path | None = None,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Authenticate the unlock, then create the exact answer-bearing final."""

    # Delayed import keeps the pure selector independently testable and makes
    # the authorization boundary visibly precede the first answer-bearing read.
    from semantic_3d_chat.evaluation.v95_deferred_final import (
        authenticate_deferred_final_unlock_v95,
    )

    unlock = authenticate_deferred_final_unlock_v95(config_path, unlock_path)
    source = _strict_fixed_path(input_path, RAW_QA, "raw QA input")
    destination = _strict_fixed_path(output_path, FINAL_QA, "final QA output")
    manifest_destination = _strict_fixed_path(
        manifest_path, SELECTION_MANIFEST, "selection manifest"
    )
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"V95 raw QA pool is unavailable: {source}")
    records = _read_jsonl(source)
    selected, selection = select_exact_final_records_v95(records)
    encoded = _serialized_jsonl(selected)
    created = _create_or_authenticate_bytes(destination, encoded)
    payload = {
        **selection,
        "status": "created_post_authenticated_unlock",
        "unlock_file_sha256": unlock["unlock_file_sha256"],
        "unlock_identity_sha256": unlock["unlock_identity_sha256"],
        "raw_qa_sha256": sha256_file_v85(source),
        "final_qa_sha256": hashlib.sha256(encoded).hexdigest(),
        "raw_qa_path": source.relative_to(PROJECT_ROOT).as_posix(),
        "final_qa_path": destination.relative_to(PROJECT_ROOT).as_posix(),
        "labels_opened_only_after_unlock_authentication": True,
        "prediction_process_imported": False,
        "model_loaded": False,
    }
    manifest_encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _create_or_authenticate_bytes(manifest_destination, manifest_encoded)
    return {
        **payload,
        "created": created,
        "selection_manifest_sha256": hashlib.sha256(manifest_encoded).hexdigest(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("select",))
    parser.add_argument(
        "--config",
        default="configs/experiments/gemma4_v95_strict_causal_successor.yaml",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = select_final_qa_v95(config_path=args.config)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANSWER_TYPE_TOTALS",
    "CHANGED_UNITS_PER_PAIR",
    "FINAL_QA",
    "PAIR_SCENES",
    "PAIR_UNIT_QUOTAS",
    "RAW_QA",
    "ROWS_PER_SCENE",
    "SCENE_IDS",
    "SELECTION_MANIFEST",
    "main",
    "select_exact_final_records_v95",
    "select_final_qa_v95",
]
