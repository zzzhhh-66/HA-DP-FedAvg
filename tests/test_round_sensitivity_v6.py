from __future__ import annotations

import csv
from pathlib import Path

import pytest

from analyze_round_sensitivity_v6 import (
    build_decision_table,
    load_sensitivity_rows,
    paired_sensitivity_effects,
    summarize_sensitivity,
)


def _write_rows(
    path: Path,
    *,
    rounds: int,
    alphas: list[float],
    seeds: list[int],
) -> None:
    rows: list[dict[str, object]] = []
    for alpha in alphas:
        for seed in seeds:
            for variant, offset in (("dp_uniform", 0.0), ("full_ha_dp", 0.03)):
                rows.append(
                    {
                        "dataset": "default_credit_card_clients",
                        "alpha": alpha,
                        "seed": seed,
                        "target_epsilon": 3.0,
                        "v5_variant": variant,
                        "global_rounds": rounds,
                        "auc": 0.60 + offset + rounds / 10000,
                        "auprc": 0.40 + offset + rounds / 10000,
                        "f1": 0.35 + offset,
                        "balanced_accuracy": 0.55 + offset,
                        "brier": 0.20 - offset,
                        "epsilon": 3.0,
                        "training_time": float(rounds),
                        "best_round": rounds,
                    }
                )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_round_sensitivity_combines_complete_horizons(tmp_path: Path) -> None:
    sources: dict[int, Path] = {}
    for rounds in (50, 75, 100):
        path = tmp_path / f"raw_{rounds}.csv"
        _write_rows(path, rounds=rounds, alphas=[0.5, 0.1], seeds=[42, 123])
        sources[rounds] = path

    rows = load_sensitivity_rows(
        sources,
        dataset="default_credit",
        alphas=[0.5, 0.1],
        seeds=[42, 123],
        variants=["dp_uniform", "full_ha_dp"],
        target_epsilon=3.0,
    )
    assert len(rows) == 24

    summary = summarize_sensitivity(rows)
    assert len(summary) == 12
    full_100 = next(
        row
        for row in summary
        if row["sensitivity_rounds"] == 100
        and row["alpha"] == 0.5
        and row["paper_method"] == "full_ha_dp"
    )
    assert full_100["boundary_selection_rate"] == pytest.approx(1.0)

    effects = paired_sensitivity_effects(rows)
    full_vs_dp = next(
        row
        for row in effects
        if row["contrast_family"] == "full_vs_dp"
        and row["metric"] == "auprc"
        and row["alpha"] == 0.5
        and row["target_rounds"] == 100
    )
    assert full_vs_dp["num_pairs"] == 2
    assert full_vs_dp["mean_difference"] == pytest.approx(0.03)
    assert full_vs_dp["improved_pairs"] == 2

    decision = build_decision_table(summary, effects)
    assert len(decision) == 2
    assert decision[0]["full_auprc_100"] > decision[0]["full_auprc_50"]


def test_round_sensitivity_rejects_missing_cell(tmp_path: Path) -> None:
    sources: dict[int, Path] = {}
    for rounds in (50, 75, 100):
        path = tmp_path / f"raw_{rounds}.csv"
        _write_rows(path, rounds=rounds, alphas=[0.5], seeds=[42])
        sources[rounds] = path

    rows = list(csv.DictReader(sources[100].open(encoding="utf-8")))
    with sources[100].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows[:-1])

    with pytest.raises(ValueError, match="Missing 1 sensitivity cells"):
        load_sensitivity_rows(
            sources,
            dataset="default_credit",
            alphas=[0.5],
            seeds=[42],
            variants=["dp_uniform", "full_ha_dp"],
            target_epsilon=3.0,
        )
