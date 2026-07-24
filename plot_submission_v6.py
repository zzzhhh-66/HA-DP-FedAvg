"""Generate compact paper figures for the final v6 matched protocol."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from src.paths import FIGURES_DIR, ensure_result_dirs

VARIANT_ORDER = ["dp_uniform", "scheduler", "ha_uniform", "full_ha_dp"]
VARIANT_LABELS = {
    "dp_uniform": "DP-FedAvg",
    "scheduler": "+ Scheduler",
    "ha_uniform": "+ HA",
    "full_ha_dp": "Full",
}
COLORS = {
    "dp_uniform": "#777777",
    "scheduler": "#3B73A1",
    "ha_uniform": "#2B8A78",
    "full_ha_dp": "#A23B3B",
}
DATASET_LABELS = {
    "default_credit_card_clients": "Default Credit",
    "give_me_some_credit": "GMSC",
    "german_credit": "German Credit",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Plot final v6 paper figures.")
    result.add_argument("privacy_raw", nargs="+", type=Path)
    result.add_argument("--output-prefix", default="paper_v6")
    return result


def load_rows(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True, sort=False)
    if "v5_variant" not in frame:
        raise ValueError("Expected a v5_variant column in final privacy raw files.")
    keys = ["dataset", "alpha", "target_epsilon", "seed", "v5_variant"]
    source_rank = frame.get("v5_source", pd.Series("", index=frame.index)).eq(
        "native_v5"
    )
    frame = frame.assign(_source_rank=source_rank.astype(int)).sort_values(
        "_source_rank"
    )
    frame = frame.drop_duplicates(keys, keep="last")
    return frame[
        frame["v5_variant"].isin(VARIANT_ORDER)
        & np.isclose(frame["target_epsilon"].astype(float), 3.0)
    ]


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.2,
            "figure.dpi": 180,
        }
    )


def ci95(values: pd.Series) -> float:
    clean = values.dropna().astype(float)
    if len(clean) < 2:
        return 0.0
    return float(
        student_t.ppf(0.975, df=len(clean) - 1)
        * clean.std(ddof=1)
        / math.sqrt(len(clean))
    )


def save(fig: plt.Figure, stem: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"{stem}.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def grouped_bars(
    frame: pd.DataFrame,
    *,
    metrics: list[tuple[str, str]],
    output_prefix: str,
    suffix: str,
) -> None:
    available = [(name, label) for name, label in metrics if name in frame]
    if not available:
        return
    datasets = [
        dataset
        for dataset in DATASET_LABELS
        if dataset in set(frame["dataset"].astype(str))
    ]
    alphas = sorted(frame["alpha"].dropna().astype(float).unique(), reverse=True)
    panel_columns = [(dataset, alpha) for dataset in datasets for alpha in alphas]
    fig, axes = plt.subplots(
        len(available),
        len(panel_columns),
        figsize=(2.25 * len(panel_columns), 1.75 * len(available)),
        squeeze=False,
        sharex=True,
    )
    x = np.arange(len(VARIANT_ORDER))
    for row_index, (metric, metric_label) in enumerate(available):
        for column_index, (dataset, alpha) in enumerate(panel_columns):
            axis = axes[row_index, column_index]
            subset = frame[
                (frame["dataset"] == dataset)
                & np.isclose(frame["alpha"].astype(float), alpha)
            ]
            means = []
            errors = []
            for variant in VARIANT_ORDER:
                values = subset.loc[subset["v5_variant"] == variant, metric]
                means.append(float(values.mean()) if len(values) else np.nan)
                errors.append(ci95(values))
            axis.bar(
                x,
                means,
                yerr=errors,
                width=0.72,
                capsize=2,
                color=[COLORS[variant] for variant in VARIANT_ORDER],
                edgecolor="black",
                linewidth=0.35,
            )
            axis.set_title(
                f"{DATASET_LABELS.get(dataset, dataset)}\n"
                + rf"$\alpha={alpha:g}$"
            )
            axis.set_ylabel(metric_label if column_index == 0 else "")
            axis.set_xticks(
                x,
                [VARIANT_LABELS[variant] for variant in VARIANT_ORDER],
                rotation=35,
                ha="right",
            )
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)
            axis.set_axisbelow(True)
    fig.tight_layout()
    save(fig, f"{output_prefix}_{suffix}")


def plot_costs(frame: pd.DataFrame, output_prefix: str) -> None:
    available = [
        metric
        for metric in ("training_time", "communication_cost")
        if metric in frame
    ]
    if not available:
        return
    cost_frame = frame.copy()
    if "communication_cost" in cost_frame:
        cost_frame["communication_mb"] = (
            cost_frame["communication_cost"].astype(float) / (1024 * 1024)
        )
    grouped_bars(
        cost_frame,
        metrics=[
            ("training_time", "Training time (s)"),
            ("communication_mb", "Communication (MB)"),
        ],
        output_prefix=output_prefix,
        suffix="costs",
    )


def main() -> None:
    args = parser().parse_args()
    ensure_result_dirs()
    style()
    frame = load_rows(args.privacy_raw)
    grouped_bars(
        frame,
        metrics=[("auc", "AUC"), ("auprc", "AUPRC"), ("f1", "F1")],
        output_prefix=args.output_prefix,
        suffix="global_metrics",
    )
    grouped_bars(
        frame,
        metrics=[
            ("client_auprc_mean", "Mean client AUPRC"),
            ("client_auprc_min", "Worst client AUPRC"),
            ("client_f1_min", "Worst client F1"),
        ],
        output_prefix=args.output_prefix,
        suffix="client_robustness",
    )
    plot_costs(frame, args.output_prefix)
    print(
        f"[Complete] figures={FIGURES_DIR / (args.output_prefix + '_*.pdf')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
