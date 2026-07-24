"""Generate figures from experiment CSV results."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .utils import FIGURES_DIR, TABLES_DIR, ensure_result_dirs


def plot_privacy_utility(df: pd.DataFrame, output_path: Path, metric: str = "auc") -> None:
    epsilon_col = "epsilon_mean" if "epsilon_mean" in df.columns else "epsilon"
    dp_df = df[df["method"].isin(["dp_fedavg", "ha_dp_fedavg"]) & df[epsilon_col].notna()].copy()
    if dp_df.empty:
        return

    plt.figure(figsize=(6, 4))
    for (method, split), group in dp_df.groupby(["method", "split"], dropna=False):
        group = group.sort_values(epsilon_col)
        plt.plot(
            group[epsilon_col],
            group[metric],
            marker="o",
            label=f"{method} / {split}",
        )
    plt.xlabel("Composed epsilon")
    plt.ylabel(metric.upper())
    plt.title("Privacy-Utility Trade-off")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_method_comparison(
    df: pd.DataFrame,
    output_path: Path,
    metric: str = "auc",
    split_prefix: str = "iid",
) -> None:
    split_df = df[df["split"].astype(str).str.startswith(split_prefix)].copy()
    if split_df.empty:
        return

    grouped = (
        split_df.groupby("method", as_index=False)[metric]
        .mean()
        .sort_values(metric, ascending=False)
    )

    plt.figure(figsize=(7, 4))
    plt.bar(grouped["method"], grouped[metric])
    plt.ylabel(metric.upper())
    plt.title(f"Method Comparison ({split_prefix})")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(TABLES_DIR / "final_results_mean_std.csv"),
    )
    parser.add_argument(
        "--allow-legacy-smoke",
        action="store_true",
        help="Allow plotting incomplete single-seed legacy artifacts.",
    )
    args = parser.parse_args()

    ensure_result_dirs()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Result file not found: {input_path}")

    df = pd.read_csv(input_path)
    required_current_columns = {"auprc_mean", "privacy_mode", "num_runs"}
    if not args.allow_legacy_smoke and (
        not required_current_columns.issubset(df.columns)
        or pd.to_numeric(df.get("num_runs"), errors="coerce").fillna(0).max() < 2
    ):
        raise ValueError(
            "Input is a legacy/incomplete smoke-test artifact. Regenerate current "
            "multi-seed results or pass --allow-legacy-smoke for diagnostics only."
        )
    metric_col = "auc_mean" if "auc_mean" in df.columns else "auc"

    plot_privacy_utility(
        df,
        FIGURES_DIR / "privacy_utility_tradeoff.png",
        metric=metric_col,
    )
    plot_method_comparison(
        df,
        FIGURES_DIR / "method_comparison_iid.png",
        metric=metric_col,
    )
    plot_method_comparison(
        df,
        FIGURES_DIR / "method_comparison_noniid.png",
        metric=metric_col,
        split_prefix="noniid",
    )
    print(f"Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
