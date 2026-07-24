from __future__ import annotations

import pytest

from analyze_submission_v6 import factorial_effects
from run_reference_baselines_v6 import build_plan
from run_submission_extensions_v5 import (
    VARIANTS,
    _import_row_matches_protocol,
)


def _matched_import_row(variant: str) -> dict[str, object]:
    specification = VARIANTS[variant]
    uses_ha = specification["aggregation"] == "balanced_ha"
    return {
        "dataset": "default_credit_card_clients",
        "split": "noniid_alpha=0.5",
        "epsilon_strategy": specification["schedule"],
        "aggregation_strategy": specification["aggregation"],
        "metadata_epsilon": 0.1 if specification["uses_metadata"] else 0.0,
        "ha_strength": 1.0 if uses_ha else "",
        "ha_prior_strength": 50.0 if uses_ha else "",
    }


def test_final_factorial_variant_has_ha_with_uniform_scheduler() -> None:
    specification = VARIANTS["ha_uniform"]
    assert specification["aggregation"] == "balanced_ha"
    assert specification["schedule"] == "uniform"
    assert specification["uses_metadata"] is True


def test_strict_import_rejects_protocol_mismatch() -> None:
    row = _matched_import_row("full_ha_dp")
    assert _import_row_matches_protocol(
        row,
        variant="full_ha_dp",
        dataset="default_credit",
        metadata_epsilon=0.1,
        ha_strength=1.0,
        ha_prior_strength=50.0,
    )

    wrong_dataset = dict(row, dataset="german_credit")
    wrong_schedule = dict(row, epsilon_strategy="uniform")
    wrong_ha = dict(row, ha_prior_strength=20.0)
    for candidate in (wrong_dataset, wrong_schedule, wrong_ha):
        assert not _import_row_matches_protocol(
            candidate,
            variant="full_ha_dp",
            dataset="default_credit",
            metadata_epsilon=0.1,
            ha_strength=1.0,
            ha_prior_strength=50.0,
        )


def test_reference_plan_runs_global_references_once_per_seed() -> None:
    plan = build_plan(
        ["centralized_mlp", "xgboost", "fedavg_noniid"],
        [0.5, 0.1],
        [42, 123],
    )
    assert len(plan) == 8
    assert plan.count(("centralized_mlp", None, 42)) == 1
    assert plan.count(("xgboost", None, 42)) == 1
    assert ("fedavg_noniid", 0.5, 42) in plan
    assert ("fedavg_noniid", 0.1, 42) in plan


def _factorial_row(seed: int, variant: str, auprc: float) -> dict[str, object]:
    return {
        "dataset": "default_credit_card_clients",
        "alpha": 0.5,
        "target_epsilon": 3.0,
        "seed": seed,
        "paper_method": variant,
        "auc": auprc + 0.2,
        "auprc": auprc,
        "f1": auprc,
        "balanced_accuracy": auprc,
        "brier": 1.0 - auprc,
    }


def test_factorial_analysis_reports_interaction() -> None:
    rows: list[dict[str, object]] = []
    values = {
        "dp_uniform": 0.40,
        "scheduler": 0.45,
        "ha_uniform": 0.42,
        "full_ha_dp": 0.50,
    }
    for seed in (42, 123):
        rows.extend(
            _factorial_row(seed, variant, value)
            for variant, value in values.items()
        )

    effects = factorial_effects(rows)
    interaction = next(
        row
        for row in effects
        if row["metric"] == "auprc"
        and row["contrast"] == "factorial_interaction"
    )
    assert interaction["num_pairs"] == 2
    assert interaction["mean_difference"] == pytest.approx(0.03)
    assert interaction["mean_improvement"] == pytest.approx(0.03)
