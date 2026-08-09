from __future__ import annotations

import copy

import pytest

from semantic_3d_chat.training.pair_curriculum import (
    PairObjectivePolicy,
    canonical_pair_objective_policy_sha256,
    pair_objective_policy_contract,
    pair_objective_policy_settings,
    validate_pair_objective_policy_coverage,
)


def _policy(
    role: str,
    *,
    language_nll_weight: float,
    candidate_hinge_weight: float,
    candidate_margin: float,
    full_vocab_hinge_weight: float,
    full_vocab_margin: float,
) -> dict[str, object]:
    return {
        "role": role,
        "language_nll_weight": language_nll_weight,
        "candidate_hinge_weight": candidate_hinge_weight,
        "candidate_margin": candidate_margin,
        "full_vocab_hinge_weight": full_vocab_hinge_weight,
        "full_vocab_margin": full_vocab_margin,
    }


def _configured(*, allow_unlisted: bool = False) -> dict[str, object]:
    return {
        "training": {
            "pair_ranking_weight": 8.0,
            "pair_ranking_margin": 1.0,
            "pair_full_vocab_ranking_weight": 2.0,
            "pair_full_vocab_ranking_margin": 1.0,
            "pair_objectives": {
                "schema_version": 1,
                "allow_unlisted_pair_ids": allow_unlisted,
                # Deliberately reverse insertion order. Resolution and hashing
                # must depend on opaque IDs, never YAML/dict insertion order.
                "by_pair": {
                    "pair_000003": _policy(
                        "optimize",
                        language_nll_weight=0.0,
                        candidate_hinge_weight=8.0,
                        candidate_margin=1.0,
                        full_vocab_hinge_weight=2.0,
                        full_vocab_margin=1.0,
                    ),
                    "pair_000001": _policy(
                        "retention",
                        language_nll_weight=0.0,
                        candidate_hinge_weight=8.0,
                        candidate_margin=0.25,
                        full_vocab_hinge_weight=2.0,
                        full_vocab_margin=0.25,
                    ),
                },
            },
        }
    }


def test_legacy_config_resolves_exact_existing_global_objective() -> None:
    config = {
        "training": {
            "pair_ranking_weight": 8,
            "pair_ranking_margin": 0.5,
            "pair_full_vocab_ranking_weight": 2,
            "pair_full_vocab_ranking_margin": 1,
        }
    }
    original = copy.deepcopy(config)

    settings = pair_objective_policy_settings(config)
    resolved = settings.resolve("arbitrary_opaque_pair")

    assert config == original
    assert settings.configured is False
    assert settings.allow_unlisted_pair_ids is True
    assert settings.pair_ids == ()
    assert resolved == PairObjectivePolicy(
        role="legacy_global",
        language_nll_weight=1.0,
        candidate_hinge_weight=8.0,
        candidate_margin=0.5,
        full_vocab_hinge_weight=2.0,
        full_vocab_margin=1.0,
    )
    assert settings.policy_for("arbitrary_opaque_pair") is resolved


def test_configured_policies_resolve_by_opaque_id_in_canonical_order() -> None:
    settings = pair_objective_policy_settings(_configured())

    assert settings.configured is True
    assert settings.allow_unlisted_pair_ids is False
    assert settings.pair_ids == ("pair_000001", "pair_000003")
    assert settings.resolve("pair_000001").role == "retention"
    assert settings.resolve("pair_000001").candidate_margin == 0.25
    assert settings.resolve("pair_000003").role == "optimize"
    assert settings.resolve("pair_000003").full_vocab_margin == 1.0
    with pytest.raises(KeyError, match="No pair objective policy"):
        settings.resolve("pair_999999")


def test_policy_contract_and_hash_are_canonical_and_order_independent() -> None:
    first = pair_objective_policy_settings(_configured())
    reordered_config = _configured()
    by_pair = reordered_config["training"]["pair_objectives"]["by_pair"]  # type: ignore[index]
    assert isinstance(by_pair, dict)
    reordered_config["training"]["pair_objectives"]["by_pair"] = {  # type: ignore[index]
        key: by_pair[key] for key in reversed(tuple(by_pair))
    }
    second = pair_objective_policy_settings(reordered_config)

    first_contract = pair_objective_policy_contract(first)
    second_contract = pair_objective_policy_contract(second)
    assert first_contract == second_contract
    assert len(first_contract["contract_sha256"]) == 64
    unhashed = {key: value for key, value in first_contract.items() if key != "contract_sha256"}
    assert first_contract["contract_sha256"] == canonical_pair_objective_policy_sha256(unhashed)

    changed = _configured()
    changed_policy = changed["training"]["pair_objectives"]["by_pair"][  # type: ignore[index]
        "pair_000001"
    ]
    changed_policy["candidate_margin"] = 0.5  # type: ignore[index]
    assert (
        pair_objective_policy_contract(pair_objective_policy_settings(changed))["contract_sha256"]
        != first_contract["contract_sha256"]
    )


def test_selected_pair_coverage_is_complete_and_deterministic() -> None:
    settings = pair_objective_policy_settings(_configured())
    first = validate_pair_objective_policy_coverage(settings, ["pair_000003", "pair_000001"])
    second = validate_pair_objective_policy_coverage(settings, ["pair_000001", "pair_000003"])

    assert first == second
    assert first["complete"] is True
    assert first["selected_pair_ids"] == ["pair_000001", "pair_000003"]
    assert first["configured_pair_ids"] == ["pair_000001", "pair_000003"]
    assert first["unlisted_pair_ids"] == []
    assert first["resolved_by_pair"]["pair_000001"]["role"] == "retention"
    assert len(first["coverage_sha256"]) == 64


def test_selected_pair_coverage_rejects_missing_stale_and_duplicate_ids() -> None:
    strict = pair_objective_policy_settings(_configured())
    with pytest.raises(ValueError, match="unselected pair IDs"):
        validate_pair_objective_policy_coverage(strict, ["pair_000001"])

    one_policy = _configured()
    del one_policy["training"]["pair_objectives"]["by_pair"]["pair_000003"]  # type: ignore[index]
    incomplete = pair_objective_policy_settings(one_policy)
    with pytest.raises(ValueError, match="lack pair objective policies"):
        validate_pair_objective_policy_coverage(incomplete, ["pair_000001", "pair_000003"])
    with pytest.raises(ValueError, match="duplicates"):
        validate_pair_objective_policy_coverage(
            strict, ["pair_000001", "pair_000001", "pair_000003"]
        )
    with pytest.raises(TypeError, match="must be a sequence"):
        validate_pair_objective_policy_coverage(strict, "pair_000001")


def test_allow_unlisted_pair_ids_uses_exact_legacy_fallback() -> None:
    config = _configured(allow_unlisted=True)
    del config["training"]["pair_objectives"]["by_pair"]["pair_000003"]  # type: ignore[index]
    settings = pair_objective_policy_settings(config)
    coverage = validate_pair_objective_policy_coverage(settings, ["pair_000001", "pair_000003"])

    assert coverage["unlisted_pair_ids"] == ["pair_000003"]
    assert settings.resolve("pair_000003") == settings.legacy_default
    assert coverage["resolved_by_pair"]["pair_000003"] == {
        "role": "legacy_global",
        "language_nll_weight": 1.0,
        "candidate_hinge_weight": 8.0,
        "candidate_margin": 1.0,
        "full_vocab_hinge_weight": 2.0,
        "full_vocab_margin": 1.0,
    }


@pytest.mark.parametrize(
    ("mutation", "error", "message"),
    [
        (lambda value: value.update(schema_version=2), ValueError, "schema_version"),
        (
            lambda value: value.update(allow_unlisted_pair_ids=1),
            TypeError,
            "must be a boolean",
        ),
        (lambda value: value.update(extra=True), ValueError, "keys mismatch"),
        (lambda value: value.pop("by_pair"), ValueError, "keys mismatch"),
        (lambda value: value.update(by_pair=[]), TypeError, "by_pair must be an object"),
    ],
)
def test_pair_objective_root_schema_is_strict(
    mutation: object, error: type[Exception], message: str
) -> None:
    config = _configured()
    raw = config["training"]["pair_objectives"]  # type: ignore[index]
    assert isinstance(raw, dict)
    mutation(raw)  # type: ignore[operator]
    with pytest.raises(error, match=message):
        pair_objective_policy_settings(config)


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("language_nll_weight", -0.1, ValueError, "finite and nonnegative"),
        ("candidate_hinge_weight", float("nan"), ValueError, "finite and nonnegative"),
        ("candidate_margin", True, TypeError, "must be numeric"),
        ("full_vocab_hinge_weight", float("inf"), ValueError, "finite and nonnegative"),
        ("full_vocab_margin", "1", TypeError, "must be numeric"),
        ("role", " ", ValueError, "nonempty string"),
    ],
)
def test_pair_objective_policy_values_are_fail_closed(
    field: str, value: object, error: type[Exception], message: str
) -> None:
    config = _configured()
    policy = config["training"]["pair_objectives"]["by_pair"]["pair_000001"]  # type: ignore[index]
    policy[field] = value  # type: ignore[index]
    with pytest.raises(error, match=message):
        pair_objective_policy_settings(config)


def test_policy_rejects_missing_unknown_keys_and_invalid_weight_combinations() -> None:
    missing = _configured()
    del missing["training"]["pair_objectives"]["by_pair"]["pair_000001"][  # type: ignore[index]
        "candidate_margin"
    ]
    with pytest.raises(ValueError, match="keys mismatch"):
        pair_objective_policy_settings(missing)

    unknown = _configured()
    unknown["training"]["pair_objectives"]["by_pair"]["pair_000001"][  # type: ignore[index]
        "semantic_name"
    ] = "forbidden"
    with pytest.raises(ValueError, match="keys mismatch"):
        pair_objective_policy_settings(unknown)

    no_candidate = _configured()
    no_candidate_policy = no_candidate["training"]["pair_objectives"]["by_pair"][  # type: ignore[index]
        "pair_000001"
    ]
    no_candidate_policy["candidate_hinge_weight"] = 0.0  # type: ignore[index]
    with pytest.raises(ValueError, match="requires candidate_hinge_weight"):
        pair_objective_policy_settings(no_candidate)

    no_loss = _configured()
    no_loss_policy = no_loss["training"]["pair_objectives"]["by_pair"][  # type: ignore[index]
        "pair_000001"
    ]
    no_loss_policy["language_nll_weight"] = 0.0  # type: ignore[index]
    no_loss_policy["candidate_hinge_weight"] = 0.0  # type: ignore[index]
    no_loss_policy["full_vocab_hinge_weight"] = 0.0  # type: ignore[index]
    with pytest.raises(ValueError, match="at least one loss weight"):
        pair_objective_policy_settings(no_loss)


def test_null_pair_objectives_preserves_legacy_mode() -> None:
    config = {
        "training": {
            "pair_ranking_weight": 0.0,
            "pair_objectives": None,
        }
    }
    settings = pair_objective_policy_settings(config)

    assert settings.configured is False
    assert settings.resolve("pair_any").contract() == {
        "role": "legacy_global",
        "language_nll_weight": 1.0,
        "candidate_hinge_weight": 0.0,
        "candidate_margin": 0.5,
        "full_vocab_hinge_weight": 0.0,
        "full_vocab_margin": 0.0,
    }
