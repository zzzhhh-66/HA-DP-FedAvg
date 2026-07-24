import pytest

from run_ablation_experiments_v3 import (
    build_effect_rows,
    build_plan,
    completed_run_keys,
)
from src.ablation_configs_v3 import (
    FACTORIAL_ABLATIONS,
    validate_factorial_ablation_specs,
)


def test_factorial_ablation_has_all_four_cells():
    validate_factorial_ablation_specs()
    cells = {
        (spec.use_heterogeneity_aware, spec.epsilon_strategy)
        for spec in FACTORIAL_ABLATIONS
    }
    assert cells == {
        (False, "uniform"),
        (False, "back_loaded"),
        (True, "uniform"),
        (True, "back_loaded"),
    }


def test_default_plan_has_40_unique_runs():
    plan = build_plan(
        alphas=[0.5, 0.1],
        seeds=[42, 123, 2026, 31415, 27182],
    )
    keys = {(alpha, seed, spec.ablation_id) for alpha, seed, spec in plan}
    assert len(plan) == 40
    assert len(keys) == 40


def test_completed_run_keys_accepts_csv_values():
    rows = [
        {"alpha": "0.5", "seed": "42", "ablation_id": "A0"},
        {"alpha": 0.1, "seed": 123, "ablation_id": "A3"},
    ]
    assert completed_run_keys(rows) == {(0.5, 42, "A0"), (0.1, 123, "A3")}


def test_effect_table_computes_full_method_delta():
    summary = []
    values = {"A0": 0.60, "A1": 0.62, "A2": 0.63, "A3": 0.66}
    for ablation_id, auc in values.items():
        summary.append(
            {
                "dataset": "default_credit",
                "alpha": 0.5,
                "ablation_id": ablation_id,
                "num_runs": 5,
                "auc_mean": auc,
                "auprc_mean": auc - 0.1,
                "f1_mean": auc - 0.2,
                "balanced_accuracy_mean": auc,
                "precision_mean": auc,
                "recall_mean": auc,
                "training_time_mean": 1.0,
            }
        )

    effects = build_effect_rows(summary)
    full = next(row for row in effects if row["contrast"] == "full_method_vs_matched_dp")
    assert full["auc_delta"] == pytest.approx(0.06)
