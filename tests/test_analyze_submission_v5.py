from __future__ import annotations

from analyze_submission_v5 import (
    add_multiplicity_adjustments,
    paired_tests,
    summarize_rows,
)


def _row(
    *,
    seed: int,
    variant: str,
    auprc: float,
    brier: float,
    client_auprc_mean: float | None = None,
) -> dict[str, object]:
    return {
        "dataset": "toy",
        "alpha": 0.5,
        "target_epsilon": 3.0,
        "seed": seed,
        "v5_variant": variant,
        "auprc": auprc,
        "brier": brier,
        "client_auprc_mean": (
            "" if client_auprc_mean is None else client_auprc_mean
        ),
        "secure_rng": False,
        "privacy_claim_valid": False,
    }


def test_summary_reports_metric_specific_sample_sizes() -> None:
    rows = [
        _row(seed=1, variant="full_ha_dp", auprc=0.4, brier=0.2, client_auprc_mean=0.3),
        _row(seed=2, variant="full_ha_dp", auprc=0.5, brier=0.1),
    ]
    summary = summarize_rows(rows)[0]
    assert summary["num_runs"] == 2
    assert summary["auprc_n"] == 2
    assert summary["client_auprc_mean_n"] == 1
    assert summary["client_profile_runs"] == 1


def test_brier_wins_use_lower_is_better_direction() -> None:
    rows = [
        _row(seed=1, variant="full_ha_dp", auprc=0.5, brier=0.10),
        _row(seed=1, variant="dp_uniform", auprc=0.4, brier=0.20),
        _row(seed=2, variant="full_ha_dp", auprc=0.6, brier=0.15),
        _row(seed=2, variant="dp_uniform", auprc=0.5, brier=0.25),
    ]
    tests = paired_tests(rows, "full_ha_dp", ["dp_uniform"])
    brier = next(row for row in tests if row["metric"] == "brier")
    assert brier["metric_goal"] == "minimize"
    assert brier["mean_difference"] < 0
    assert brier["mean_improvement"] > 0
    assert brier["target_wins"] == 2
    assert brier["comparator_wins"] == 0


def test_primary_and_global_holm_are_both_reported() -> None:
    rows = [
        {
            "metric": "auprc",
            "comparator": "dp_uniform",
            "paired_t_p": 0.01,
            "wilcoxon_p": 0.02,
        },
        {
            "metric": "f1",
            "comparator": "dp_uniform",
            "paired_t_p": 0.03,
            "wilcoxon_p": 0.04,
        },
    ]
    add_multiplicity_adjustments(rows, ["auprc"], ["dp_uniform"])
    assert rows[0]["primary_hypothesis"] is True
    assert rows[1]["primary_hypothesis"] is False
    assert "paired_t_p_holm_primary" in rows[0]
    assert "paired_t_p_holm_primary" not in rows[1]
    assert "paired_t_p_holm_global" in rows[0]
    assert "paired_t_p_holm_global" in rows[1]
