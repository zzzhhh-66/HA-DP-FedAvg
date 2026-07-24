"""Analyze fixed-privacy 50/75/100-round sensitivity experiments."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from analyze_submission_v6 import _holm_adjust, _paired_test, _summary_stats
from src.config import DATASET_CONFIG
from src.paths import DEFAULT_SEEDS, TABLES_DIR, ensure_result_dirs
from src.v4_experiment_utils import parse_float, read_csv_rows, write_csv_rows

SENSITIVITY_VARIANTS = ("dp_uniform", "full_ha_dp")
SENSITIVITY_METRICS = (
    "auc",
    "auprc",
    "f1",
    "balanced_accuracy",
    "brier",
    "epsilon",
    "training_time",
    "best_round",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Combine the final 50-round results with matched 75/100-round "
            "runs and compute paired sensitivity evidence."
        )
    )
    result.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIG))
    result.add_argument("--raw-50", required=True, type=Path)
    result.add_argument("--raw-75", required=True, type=Path)
    result.add_argument("--raw-100", required=True, type=Path)
    result.add_argument("--alphas", nargs="+", type=float, default=[0.5, 0.1])
    result.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    result.add_argument(
        "--variants",
        nargs="+",
        choices=SENSITIVITY_VARIANTS,
        default=list(SENSITIVITY_VARIANTS),
    )
    result.add_argument("--target-epsilon", type=float, default=3.0)
    result.add_argument("--output-prefix", default="round_sensitivity_v6")
    return result


def _optional_float(value: Any) -> float | None:
    if value in (None, "", "None", "nan"):
        return None
    try:
        parsed = parse_float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _variant(row: dict[str, Any]) -> str:
    return str(row.get("v5_variant") or row.get("curve_method") or "")


def load_sensitivity_rows(
    sources: dict[int, Path],
    *,
    dataset: str,
    alphas: list[float],
    seeds: list[int],
    variants: list[str],
    target_epsilon: float,
) -> list[dict[str, Any]]:
    expected_dataset = str(DATASET_CONFIG[dataset]["name"])
    alpha_keys = {float(value) for value in alphas}
    seed_keys = {int(value) for value in seeds}
    variant_keys = set(variants)
    selected: dict[tuple[int, float, int, str], dict[str, Any]] = {}

    for rounds, path in sorted(sources.items()):
        if not path.exists():
            raise FileNotFoundError(path)
        for source in read_csv_rows(path):
            if str(source.get("dataset", "")) != expected_dataset:
                continue
            variant = _variant(source)
            if variant not in variant_keys:
                continue
            alpha = _optional_float(source.get("alpha"))
            seed = _optional_float(source.get("seed"))
            epsilon = _optional_float(
                source.get("target_epsilon", source.get("total_epsilon"))
            )
            if alpha is None or seed is None or epsilon is None:
                continue
            seed_int = int(seed)
            if (
                alpha not in alpha_keys
                or seed_int not in seed_keys
                or not math.isclose(epsilon, target_epsilon, abs_tol=1e-9)
            ):
                continue

            recorded_rounds = _optional_float(source.get("global_rounds"))
            if recorded_rounds is not None and int(recorded_rounds) != rounds:
                raise ValueError(
                    f"{path}: recorded global_rounds={recorded_rounds:g}, "
                    f"expected {rounds}."
                )

            key = (rounds, alpha, seed_int, variant)
            if key in selected:
                raise ValueError(f"Duplicate sensitivity cell: {key}")
            row: dict[str, Any] = dict(source)
            row.update(
                {
                    "sensitivity_rounds": rounds,
                    "global_rounds": rounds,
                    "paper_method": variant,
                    "analysis_source": str(path),
                }
            )
            selected[key] = row

    expected = {
        (rounds, alpha, seed, variant)
        for rounds in sources
        for alpha in alpha_keys
        for seed in seed_keys
        for variant in variant_keys
    }
    missing = sorted(expected.difference(selected))
    if missing:
        preview = ", ".join(str(key) for key in missing[:8])
        raise ValueError(
            f"Missing {len(missing)} sensitivity cells. First missing: {preview}"
        )

    return [selected[key] for key in sorted(selected)]


def summarize_sensitivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            int(parse_float(row["sensitivity_rounds"])),
            parse_float(row["alpha"]),
            str(row["paper_method"]),
        )
        groups[key].append(row)

    summary: list[dict[str, Any]] = []
    for (rounds, alpha, variant), group in sorted(groups.items()):
        item: dict[str, Any] = {
            "dataset": str(group[0]["dataset"]),
            "sensitivity_rounds": rounds,
            "alpha": alpha,
            "paper_method": variant,
            "num_runs": len(group),
        }
        for metric in SENSITIVITY_METRICS:
            values = [
                value
                for row in group
                if (value := _optional_float(row.get(metric))) is not None
            ]
            if not values:
                continue
            stats = _summary_stats(values)
            for suffix, value in stats.items():
                item[f"{metric}_{suffix}"] = value
        best_rounds = [
            value
            for row in group
            if (value := _optional_float(row.get("best_round"))) is not None
        ]
        if best_rounds:
            item["boundary_selection_rate"] = sum(
                value >= rounds for value in best_rounds
            ) / len(best_rounds)
            item["best_round_fraction_mean"] = sum(
                value / rounds for value in best_rounds
            ) / len(best_rounds)
        summary.append(item)
    return summary


def paired_sensitivity_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (
            int(parse_float(row["sensitivity_rounds"])),
            parse_float(row["alpha"]),
            int(parse_float(row["seed"])),
            str(row["paper_method"]),
        ): row
        for row in rows
    }
    alphas = sorted({key[1] for key in by_key})
    seeds = sorted({key[2] for key in by_key})
    horizons = sorted({key[0] for key in by_key})
    variants = sorted({key[3] for key in by_key})
    effects: list[dict[str, Any]] = []

    def add_effect(
        *,
        family: str,
        alpha: float,
        metric: str,
        target_rounds: int,
        target_variant: str,
        comparator_rounds: int,
        comparator_variant: str,
    ) -> None:
        differences: list[float] = []
        paired_seeds: list[int] = []
        for seed in seeds:
            target = by_key.get((target_rounds, alpha, seed, target_variant))
            comparator = by_key.get(
                (comparator_rounds, alpha, seed, comparator_variant)
            )
            if target is None or comparator is None:
                continue
            target_value = _optional_float(target.get(metric))
            comparator_value = _optional_float(comparator.get(metric))
            if target_value is None or comparator_value is None:
                continue
            differences.append(target_value - comparator_value)
            paired_seeds.append(seed)
        if not differences:
            return
        result: dict[str, Any] = {
            "dataset": str(next(iter(by_key.values()))["dataset"]),
            "contrast_family": family,
            "alpha": alpha,
            "metric": metric,
            "metric_goal": "minimize" if metric == "brier" else "maximize",
            "target_rounds": target_rounds,
            "target_variant": target_variant,
            "comparator_rounds": comparator_rounds,
            "comparator_variant": comparator_variant,
            "paired_seeds": paired_seeds,
            **_paired_test(differences),
        }
        direction = -1.0 if metric == "brier" else 1.0
        result["mean_improvement"] = direction * float(result["mean_difference"])
        result["improved_pairs"] = sum(
            direction * difference > 0 for difference in differences
        )
        effects.append(result)

    for alpha in alphas:
        for metric in SENSITIVITY_METRICS[:5]:
            for variant in variants:
                for rounds in horizons:
                    if rounds == 50:
                        continue
                    add_effect(
                        family="horizon_vs_50",
                        alpha=alpha,
                        metric=metric,
                        target_rounds=rounds,
                        target_variant=variant,
                        comparator_rounds=50,
                        comparator_variant=variant,
                    )
            for rounds in horizons:
                add_effect(
                    family="full_vs_dp",
                    alpha=alpha,
                    metric=metric,
                    target_rounds=rounds,
                    target_variant="full_ha_dp",
                    comparator_rounds=rounds,
                    comparator_variant="dp_uniform",
                )

    for family in ("horizon_vs_50", "full_vs_dp"):
        indices = [
            index
            for index, row in enumerate(effects)
            if row["contrast_family"] == family and row["metric"] == "auprc"
        ]
        t_adjusted = _holm_adjust(
            [float(effects[index]["paired_t_p"]) for index in indices]
        )
        w_adjusted = _holm_adjust(
            [float(effects[index]["wilcoxon_p"]) for index in indices]
        )
        for position, index in enumerate(indices):
            effects[index]["paired_t_p_holm_auprc_family"] = t_adjusted[position]
            effects[index]["wilcoxon_p_holm_auprc_family"] = w_adjusted[position]
    return effects


def build_decision_table(
    summary: list[dict[str, Any]],
    effects: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_summary = {
        (
            int(row["sensitivity_rounds"]),
            float(row["alpha"]),
            str(row["paper_method"]),
        ): row
        for row in summary
    }
    full_vs_dp = {
        (int(row["target_rounds"]), float(row["alpha"])): row
        for row in effects
        if row["contrast_family"] == "full_vs_dp" and row["metric"] == "auprc"
    }
    alphas = sorted({key[1] for key in by_summary})
    horizons = sorted({key[0] for key in by_summary})
    output: list[dict[str, Any]] = []
    for alpha in alphas:
        item: dict[str, Any] = {
            "dataset": str(next(iter(by_summary.values()))["dataset"]),
            "alpha": alpha,
        }
        for rounds in horizons:
            for variant in SENSITIVITY_VARIANTS:
                row = by_summary[(rounds, alpha, variant)]
                prefix = "full" if variant == "full_ha_dp" else "dp"
                item[f"{prefix}_auprc_{rounds}"] = row.get("auprc_mean", "")
                item[f"{prefix}_best_round_mean_{rounds}"] = row.get(
                    "best_round_mean", ""
                )
                item[f"{prefix}_boundary_rate_{rounds}"] = row.get(
                    "boundary_selection_rate", ""
                )
            effect = full_vs_dp[(rounds, alpha)]
            item[f"full_minus_dp_auprc_{rounds}"] = effect["mean_difference"]
            item[f"full_vs_dp_wins_{rounds}"] = effect["improved_pairs"]
        output.append(item)
    return output


def main() -> None:
    args = parser().parse_args()
    if args.target_epsilon <= 0:
        raise ValueError("target epsilon must be positive.")
    ensure_result_dirs()
    sources = {50: args.raw_50, 75: args.raw_75, 100: args.raw_100}
    rows = load_sensitivity_rows(
        sources,
        dataset=args.dataset,
        alphas=args.alphas,
        seeds=args.seeds,
        variants=args.variants,
        target_epsilon=args.target_epsilon,
    )
    summary = summarize_sensitivity(rows)
    effects = paired_sensitivity_effects(rows)
    decision = build_decision_table(summary, effects)

    stem = f"{args.dataset}_{args.output_prefix}"
    raw_path = TABLES_DIR / f"{stem}_raw.csv"
    summary_path = TABLES_DIR / f"{stem}_mean_std.csv"
    effects_path = TABLES_DIR / f"{stem}_paired_effects.csv"
    decision_path = TABLES_DIR / f"{stem}_decision.csv"
    write_csv_rows(raw_path, rows)
    write_csv_rows(summary_path, summary)
    write_csv_rows(effects_path, effects)
    write_csv_rows(decision_path, decision)

    print(
        f"[Complete] rows={len(rows)} raw={raw_path} summary={summary_path} "
        f"effects={effects_path} decision={decision_path}",
        flush=True,
    )
    for row in decision:
        print(
            f"[Decision] dataset={row['dataset']} alpha={row['alpha']} "
            f"full_auprc=({row['full_auprc_50']:.4f}, "
            f"{row['full_auprc_75']:.4f}, {row['full_auprc_100']:.4f}) "
            f"for rounds=(50,75,100)",
            flush=True,
        )


if __name__ == "__main__":
    main()
