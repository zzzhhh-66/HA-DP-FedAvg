"""Create publication-oriented summaries and paired significance tests for v5."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
from scipy.stats import t as student_t
from scipy.stats import ttest_rel, wilcoxon

TABLES_DIR = Path("results") / "tables"

METRICS = (
    "auc",
    "auprc",
    "f1",
    "balanced_accuracy",
    "brier",
    "client_auprc_mean",
    "client_auprc_min",
    "client_f1_mean",
    "client_f1_min",
)
LOWER_IS_BETTER = {"brier"}
CLIENT_METRICS = {
    "client_auprc_mean",
    "client_auprc_min",
    "client_f1_mean",
    "client_f1_min",
}


def ensure_result_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def parse_float(value: Any) -> float:
    return float(value)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Analyze v5 experiment tables.")
    result.add_argument("inputs", nargs="+", type=Path)
    result.add_argument("--target", default="full_ha_dp")
    result.add_argument(
        "--comparators",
        nargs="+",
        default=[
            "dp_uniform",
            "scheduler",
            "front_loaded",
            "dp_label_representative",
            "scheduler_label_representative",
            "noise_aware_ha_v5",
        ],
    )
    result.add_argument(
        "--primary-metrics",
        nargs="+",
        default=["auprc"],
        choices=METRICS,
        help=(
            "Metrics in the prespecified primary hypothesis family. The default "
            "uses AUPRC because credit default prediction is class imbalanced."
        ),
    )
    result.add_argument(
        "--primary-comparators",
        nargs="+",
        default=["dp_uniform", "scheduler"],
        help=(
            "Comparators in the primary hypothesis family. All other tests are "
            "still retained and receive a global Holm correction."
        ),
    )
    result.add_argument("--output-prefix", default="submission_v5_analysis")
    return result


def _variant(row: dict[str, Any]) -> str:
    return str(row.get("v5_variant") or row.get("curve_method") or "")


def _row_key(row: dict[str, Any]) -> tuple[str, float, float, int, str]:
    return (
        str(row.get("dataset", "")),
        parse_float(row["alpha"]),
        parse_float(row.get("target_epsilon", row.get("total_epsilon"))),
        int(parse_float(row["seed"])),
        _variant(row),
    )


def deduplicate(paths: list[Path]) -> list[dict[str, Any]]:
    rows: dict[tuple[str, float, float, int, str], dict[str, Any]] = {}
    for path in paths:
        for source in read_csv_rows(path):
            row: dict[str, Any] = dict(source)
            row["v5_variant"] = _variant(row)
            key = _row_key(row)
            previous = rows.get(key)
            if previous is None or row.get("v5_source") == "native_v5":
                rows[key] = row
    return list(rows.values())


def _finite_metric(row: dict[str, Any], metric: str) -> float | None:
    value = row.get(metric)
    if value in (None, ""):
        return None
    parsed = parse_float(value)
    return parsed if np.isfinite(parsed) else None


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_keys = ("dataset", "alpha", "target_epsilon", "v5_variant")
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row.get(name, "")) for name in group_keys)
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        item: dict[str, Any] = dict(zip(group_keys, key, strict=True))
        item["num_runs"] = len(group)
        item["client_profile_runs"] = sum(
            any(_finite_metric(row, metric) is not None for metric in CLIENT_METRICS)
            for row in group
        )
        item["secure_rng_true_runs"] = sum(
            str(row.get("secure_rng", "")).strip().lower() == "true" for row in group
        )
        item["privacy_claim_valid_true_runs"] = sum(
            str(row.get("privacy_claim_valid", "")).strip().lower() == "true"
            for row in group
        )
        for metric in METRICS + ("epsilon", "training_time"):
            values = [
                value
                for row in group
                if (value := _finite_metric(row, metric)) is not None
            ]
            item[f"{metric}_n"] = len(values)
            if not values:
                continue
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summary.append(item)
    return summary


def _safe_wilcoxon(target: list[float], control: list[float]) -> float:
    differences = np.asarray(target) - np.asarray(control)
    if np.allclose(differences, 0.0):
        return 1.0
    try:
        return float(wilcoxon(target, control, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def paired_tests(
    rows: list[dict[str, Any]],
    target: str,
    comparators: list[str],
) -> list[dict[str, Any]]:
    lookup = {_row_key(row): row for row in rows}
    groups = sorted({(key[0], key[1], key[2]) for key in lookup})
    results: list[dict[str, Any]] = []
    for dataset, alpha, epsilon in groups:
        for comparator in comparators:
            for metric in METRICS:
                target_values: list[float] = []
                control_values: list[float] = []
                seeds: list[int] = []
                candidate_seeds = sorted(
                    {
                        key[3]
                        for key in lookup
                        if key[:3] == (dataset, alpha, epsilon) and key[4] == target
                    }
                )
                for seed in candidate_seeds:
                    target_row = lookup.get((dataset, alpha, epsilon, seed, target))
                    control_row = lookup.get((dataset, alpha, epsilon, seed, comparator))
                    if target_row is None or control_row is None:
                        continue
                    target_value = _finite_metric(target_row, metric)
                    control_value = _finite_metric(control_row, metric)
                    if target_value is None or control_value is None:
                        continue
                    target_values.append(target_value)
                    control_values.append(control_value)
                    seeds.append(seed)
                if not target_values:
                    continue
                differences = [a - b for a, b in zip(target_values, control_values, strict=True)]
                direction = -1.0 if metric in LOWER_IS_BETTER else 1.0
                improvements = [direction * value for value in differences]
                count = len(differences)
                effect_mean = mean(differences)
                effect_std = stdev(differences) if count > 1 else 0.0
                critical = student_t.ppf(0.975, df=count - 1) if count > 1 else float("nan")
                ci95 = critical * effect_std / math.sqrt(count) if count > 1 else float("nan")
                improvement_mean = direction * effect_mean
                improvement_low = (
                    direction * (effect_mean - ci95)
                    if direction > 0
                    else direction * (effect_mean + ci95)
                )
                improvement_high = (
                    direction * (effect_mean + ci95)
                    if direction > 0
                    else direction * (effect_mean - ci95)
                )
                t_p = (
                    float(ttest_rel(target_values, control_values).pvalue)
                    if count > 1 and not np.allclose(differences, 0.0)
                    else 1.0
                )
                results.append(
                    {
                        "dataset": dataset,
                        "alpha": alpha,
                        "target_epsilon": epsilon,
                        "target": target,
                        "comparator": comparator,
                        "metric": metric,
                        "num_pairs": count,
                        "paired_seeds": ",".join(str(seed) for seed in seeds),
                        "target_mean": mean(target_values),
                        "comparator_mean": mean(control_values),
                        "metric_goal": "minimize" if direction < 0 else "maximize",
                        "mean_difference": effect_mean,
                        "difference_std": effect_std,
                        "difference_ci95": ci95,
                        "difference_ci95_low": effect_mean - ci95,
                        "difference_ci95_high": effect_mean + ci95,
                        "mean_improvement": improvement_mean,
                        "improvement_ci95_low": improvement_low,
                        "improvement_ci95_high": improvement_high,
                        "cohen_dz": effect_mean / effect_std if effect_std > 0 else float("nan"),
                        "improvement_cohen_dz": (
                            improvement_mean / effect_std if effect_std > 0 else float("nan")
                        ),
                        "target_wins": sum(value > 0 for value in improvements),
                        "comparator_wins": sum(value < 0 for value in improvements),
                        "ties": sum(value == 0 for value in improvements),
                        "paired_t_p": t_p,
                        "wilcoxon_p": _safe_wilcoxon(target_values, control_values),
                    }
                )
    return results


def add_holm_adjustment(
    rows: list[dict[str, Any]],
    p_key: str,
    output_key: str,
    selected_indices: set[int] | None = None,
) -> None:
    finite = [
        (index, float(row[p_key]))
        for index, row in enumerate(rows)
        if (selected_indices is None or index in selected_indices)
        and np.isfinite(float(row[p_key]))
    ]
    ordered = sorted(finite, key=lambda item: item[1])
    adjusted = [0.0] * len(ordered)
    running_max = 0.0
    total = len(ordered)
    for rank, (_, value) in enumerate(ordered):
        running_max = max(running_max, min(1.0, (total - rank) * value))
        adjusted[rank] = running_max
    for (index, _), value in zip(ordered, adjusted, strict=True):
        rows[index][output_key] = value


def add_multiplicity_adjustments(
    rows: list[dict[str, Any]],
    primary_metrics: list[str],
    primary_comparators: list[str],
) -> None:
    primary_indices = {
        index
        for index, row in enumerate(rows)
        if row["metric"] in primary_metrics and row["comparator"] in primary_comparators
    }
    for index, row in enumerate(rows):
        row["primary_hypothesis"] = index in primary_indices
        row["primary_family_definition"] = (
            f"metrics={','.join(primary_metrics)};"
            f"comparators={','.join(primary_comparators)}"
        )
    for p_key in ("paired_t_p", "wilcoxon_p"):
        add_holm_adjustment(rows, p_key, f"{p_key}_holm_global")
        add_holm_adjustment(
            rows,
            p_key,
            f"{p_key}_holm_primary",
            selected_indices=primary_indices,
        )


def main() -> None:
    args = parser().parse_args()
    ensure_result_dirs()
    rows = deduplicate(args.inputs)
    if not rows:
        raise ValueError("No experiment rows were found.")
    summary = summarize_rows(rows)
    tests = paired_tests(rows, args.target, args.comparators)
    if tests:
        add_multiplicity_adjustments(
            tests,
            primary_metrics=args.primary_metrics,
            primary_comparators=args.primary_comparators,
        )

    summary_path = TABLES_DIR / f"{args.output_prefix}_method_summary.csv"
    tests_path = TABLES_DIR / f"{args.output_prefix}_paired_tests.csv"
    write_csv_rows(summary_path, summary)
    if tests:
        write_csv_rows(tests_path, tests)
    print(f"[Complete] summary={summary_path} paired_tests={tests_path}")


if __name__ == "__main__":
    main()
