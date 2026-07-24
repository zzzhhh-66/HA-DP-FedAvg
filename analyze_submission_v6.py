"""Create paper-ready summaries for the final v6 experiment suite."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t
from scipy.stats import ttest_1samp, wilcoxon

from src.paths import TABLES_DIR, ensure_result_dirs
from src.v4_experiment_utils import parse_float, read_csv_rows, write_csv_rows

PRIVACY_METRICS = (
    "auc",
    "auprc",
    "f1",
    "balanced_accuracy",
    "brier",
)
CLIENT_METRICS = (
    "client_auc_mean",
    "client_auc_min",
    "client_auprc_mean",
    "client_auprc_min",
    "client_f1_mean",
    "client_f1_min",
)
COST_METRICS = ("training_time", "communication_cost")
FACTORIAL_VARIANTS = {
    "A0": "dp_uniform",
    "A1": "scheduler",
    "A2": "ha_uniform",
    "A3": "full_ha_dp",
}
FACTORIAL_CONTRASTS = {
    "scheduler_without_ha": ("A1", "A0"),
    "ha_under_uniform": ("A2", "A0"),
    "ha_under_scheduler": ("A3", "A1"),
    "scheduler_with_ha": ("A3", "A2"),
    "full_vs_uniform_dp": ("A3", "A0"),
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Analyze final privacy, ablation, baseline, client, and cost results."
    )
    result.add_argument(
        "--privacy-raw",
        nargs="+",
        type=Path,
        required=True,
        help="Final 2x2 privacy-ablation raw CSV files.",
    )
    result.add_argument(
        "--baseline-raw",
        nargs="*",
        type=Path,
        default=[],
        help="Reference-baseline raw CSV files.",
    )
    result.add_argument("--output-prefix", default="paper_v6")
    return result


def _float_or_none(value: Any) -> float | None:
    if value in (None, "", "None", "nan"):
        return None
    try:
        result = parse_float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _string_alpha(row: dict[str, Any]) -> str:
    value = _float_or_none(row.get("alpha"))
    return "" if value is None else f"{value:g}"


def _load(paths: Iterable[Path], source_kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        for row in read_csv_rows(path):
            item: dict[str, Any] = dict(row)
            item["analysis_source"] = str(path)
            item["method_family"] = source_kind
            item["paper_method"] = str(
                item.get("v5_variant")
                or item.get("baseline_variant")
                or item.get("method")
            )
            rows.append(item)
    return rows


def _deduplicate(
    rows: Iterable[dict[str, Any]],
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    result: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in keys)
        result[key] = row
    return list(result.values())


def _summary_stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    count = int(array.size)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if count > 1 else 0.0
    ci95 = (
        float(student_t.ppf(0.975, df=count - 1) * std / math.sqrt(count))
        if count > 1
        else 0.0
    )
    return {"mean": mean, "std": std, "n": count, "ci95": ci95}


def summarize_methods(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("dataset", "")),
            str(row.get("paper_method", "")),
            str(row.get("method_family", "")),
            str(row.get("split", "")),
            _string_alpha(row),
            str(row.get("target_epsilon", "")),
        )
        groups[key].append(row)

    summary: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        item: dict[str, Any] = dict(
            zip(
                (
                    "dataset",
                    "paper_method",
                    "method_family",
                    "split",
                    "alpha",
                    "target_epsilon",
                ),
                key,
                strict=True,
            )
        )
        item["num_runs"] = len(group)
        for metric in PRIVACY_METRICS + CLIENT_METRICS + COST_METRICS:
            values = [
                value
                for row in group
                if (value := _float_or_none(row.get(metric))) is not None
            ]
            if not values:
                continue
            stats = _summary_stats(values)
            for suffix, value in stats.items():
                item[f"{metric}_{suffix}"] = value
        if "communication_cost_mean" in item:
            item["communication_mb_mean"] = (
                float(item["communication_cost_mean"]) / (1024 * 1024)
            )
        summary.append(item)
    return summary


def _paired_test(differences: list[float]) -> dict[str, Any]:
    values = np.asarray(differences, dtype=float)
    count = int(values.size)
    stats = _summary_stats(differences)
    if count > 1:
        paired_t_p = float(ttest_1samp(values, popmean=0.0).pvalue)
        try:
            wilcoxon_p = float(wilcoxon(values).pvalue)
        except ValueError:
            wilcoxon_p = 1.0
    else:
        paired_t_p = float("nan")
        wilcoxon_p = float("nan")
    return {
        "num_pairs": count,
        "mean_difference": stats["mean"],
        "difference_std": stats["std"],
        "difference_ci95": stats["ci95"],
        "difference_ci95_low": float(stats["mean"]) - float(stats["ci95"]),
        "difference_ci95_high": float(stats["mean"]) + float(stats["ci95"]),
        "paired_t_p": paired_t_p,
        "wilcoxon_p": wilcoxon_p,
        "positive_differences": int(np.sum(values > 0)),
        "negative_differences": int(np.sum(values < 0)),
        "ties": int(np.sum(values == 0)),
    }


def _holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    finite_indices = [
        index for index, value in enumerate(p_values) if math.isfinite(value)
    ]
    adjusted = np.full(len(p_values), np.nan, dtype=float)
    if not finite_indices:
        return adjusted.tolist()
    order = sorted(finite_indices, key=lambda index: p_values[index])
    running = 0.0
    total = len(finite_indices)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def factorial_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if str(row.get("paper_method", "")) in FACTORIAL_VARIANTS.values()
    ]
    by_run: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    for row in eligible:
        seed_value = _float_or_none(row.get("seed"))
        if seed_value is None:
            continue
        key = (
            str(row.get("dataset", "")),
            _string_alpha(row),
            str(row.get("target_epsilon", "")),
            int(seed_value),
            str(row["paper_method"]),
        )
        by_run[key] = row

    settings = sorted(
        {
            (dataset, alpha, epsilon)
            for dataset, alpha, epsilon, _, _ in by_run
        }
    )
    effects: list[dict[str, Any]] = []
    for dataset, alpha, epsilon in settings:
        seeds = sorted(
            {
                seed
                for row_dataset, row_alpha, row_epsilon, seed, _ in by_run
                if (row_dataset, row_alpha, row_epsilon)
                == (dataset, alpha, epsilon)
            }
        )
        for metric in PRIVACY_METRICS:
            maximize = metric != "brier"
            for contrast, (target_cell, comparator_cell) in FACTORIAL_CONTRASTS.items():
                target_variant = FACTORIAL_VARIANTS[target_cell]
                comparator_variant = FACTORIAL_VARIANTS[comparator_cell]
                differences: list[float] = []
                paired_seeds: list[int] = []
                for seed in seeds:
                    target = by_run.get(
                        (dataset, alpha, epsilon, seed, target_variant)
                    )
                    comparator = by_run.get(
                        (dataset, alpha, epsilon, seed, comparator_variant)
                    )
                    if target is None or comparator is None:
                        continue
                    target_value = _float_or_none(target.get(metric))
                    comparator_value = _float_or_none(comparator.get(metric))
                    if target_value is None or comparator_value is None:
                        continue
                    differences.append(target_value - comparator_value)
                    paired_seeds.append(seed)
                if not differences:
                    continue
                row = {
                    "dataset": dataset,
                    "alpha": alpha,
                    "target_epsilon": epsilon,
                    "metric": metric,
                    "metric_goal": "maximize" if maximize else "minimize",
                    "contrast": contrast,
                    "target_cell": target_cell,
                    "target_variant": target_variant,
                    "comparator_cell": comparator_cell,
                    "comparator_variant": comparator_variant,
                    "paired_seeds": paired_seeds,
                    **_paired_test(differences),
                }
                row["mean_improvement"] = (
                    row["mean_difference"]
                    if maximize
                    else -float(row["mean_difference"])
                )
                effects.append(row)

            interaction_differences: list[float] = []
            interaction_seeds: list[int] = []
            for seed in seeds:
                cells = {
                    cell: by_run.get(
                        (
                            dataset,
                            alpha,
                            epsilon,
                            seed,
                            FACTORIAL_VARIANTS[cell],
                        )
                    )
                    for cell in FACTORIAL_VARIANTS
                }
                if any(value is None for value in cells.values()):
                    continue
                values = {
                    cell: _float_or_none(row.get(metric))
                    for cell, row in cells.items()
                    if row is not None
                }
                if any(value is None for value in values.values()):
                    continue
                interaction_differences.append(
                    float(values["A3"])
                    - float(values["A2"])
                    - float(values["A1"])
                    + float(values["A0"])
                )
                interaction_seeds.append(seed)
            if interaction_differences:
                interaction = {
                    "dataset": dataset,
                    "alpha": alpha,
                    "target_epsilon": epsilon,
                    "metric": metric,
                    "metric_goal": "maximize" if maximize else "minimize",
                    "contrast": "factorial_interaction",
                    "target_cell": "A3-A2",
                    "target_variant": "scheduler_effect_with_ha",
                    "comparator_cell": "A1-A0",
                    "comparator_variant": "scheduler_effect_without_ha",
                    "paired_seeds": interaction_seeds,
                    **_paired_test(interaction_differences),
                }
                interaction["mean_improvement"] = (
                    interaction["mean_difference"]
                    if maximize
                    else -float(interaction["mean_difference"])
                )
                effects.append(interaction)

    primary_indices = [
        index for index, row in enumerate(effects) if row["metric"] == "auprc"
    ]
    for field in ("paired_t_p", "wilcoxon_p"):
        adjusted = _holm_adjust(
            [float(effects[index][field]) for index in primary_indices]
        )
        for index, value in zip(primary_indices, adjusted, strict=True):
            effects[index][f"{field}_holm_auprc_family"] = value
    return effects


def client_summary(method_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in method_summary
        if any(f"{metric}_mean" in row for metric in CLIENT_METRICS)
    ]


def cost_summary(method_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in method_summary
        if "training_time_mean" in row or "communication_cost_mean" in row
    ]


def main() -> None:
    args = parser().parse_args()
    ensure_result_dirs()
    privacy = _deduplicate(
        _load(args.privacy_raw, "privacy_matched"),
        ("dataset", "alpha", "seed", "target_epsilon", "v5_variant"),
    )
    baselines = _deduplicate(
        _load(args.baseline_raw, "reference"),
        ("dataset", "alpha", "seed", "baseline_variant"),
    )
    all_rows = privacy + baselines
    summary = summarize_methods(all_rows)
    effects = factorial_effects(privacy)

    outputs = {
        "method_summary": TABLES_DIR / f"{args.output_prefix}_method_summary.csv",
        "factorial_effects": TABLES_DIR / f"{args.output_prefix}_factorial_effects.csv",
        "client_profiles": TABLES_DIR
        / f"{args.output_prefix}_client_profile_summary.csv",
        "costs": TABLES_DIR / f"{args.output_prefix}_cost_summary.csv",
    }
    write_csv_rows(outputs["method_summary"], summary)
    write_csv_rows(outputs["factorial_effects"], effects)
    write_csv_rows(outputs["client_profiles"], client_summary(summary))
    write_csv_rows(outputs["costs"], cost_summary(summary))

    print(
        f"[Complete] privacy_rows={len(privacy)} baseline_rows={len(baselines)} "
        f"factorial_effects={len(effects)}",
        flush=True,
    )
    for label, path in outputs.items():
        print(f"  {label}: {path}", flush=True)


if __name__ == "__main__":
    main()
