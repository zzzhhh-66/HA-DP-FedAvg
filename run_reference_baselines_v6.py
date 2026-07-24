"""Run leakage-safe centralized and non-private federated reference models."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from src.paths import TABLES_DIR, ensure_result_dirs
from src.v4_experiment_utils import (
    parse_float,
    read_csv_rows,
    summarize_numeric,
    write_csv_rows,
)

FINAL_SEEDS = [42, 123, 2026, 31415, 27182, 73, 101, 909, 4096, 65537]
DEFAULT_ALPHAS = [0.5, 0.1]


@dataclass(frozen=True)
class ReferenceVariant:
    name: str
    method: str
    split: str
    uses_alpha: bool
    description: str


REFERENCE_VARIANTS: dict[str, ReferenceVariant] = {
    "centralized_mlp": ReferenceVariant(
        name="centralized_mlp",
        method="centralized",
        split="iid",
        uses_alpha=False,
        description="Centralized CreditRiskMLP upper reference",
    ),
    "logistic_regression": ReferenceVariant(
        name="logistic_regression",
        method="logistic_regression",
        split="iid",
        uses_alpha=False,
        description="Centralized weighted logistic regression",
    ),
    "hist_gbdt": ReferenceVariant(
        name="hist_gbdt",
        method="hist_gbdt",
        split="iid",
        uses_alpha=False,
        description="Centralized histogram gradient boosting",
    ),
    "xgboost": ReferenceVariant(
        name="xgboost",
        method="xgboost",
        split="iid",
        uses_alpha=False,
        description="Centralized XGBoost upper reference",
    ),
    "fedavg_iid": ReferenceVariant(
        name="fedavg_iid",
        method="fedavg",
        split="iid",
        uses_alpha=False,
        description="Non-private FedAvg under IID partitioning",
    ),
    "fedavg_noniid": ReferenceVariant(
        name="fedavg_noniid",
        method="fedavg",
        split="noniid",
        uses_alpha=True,
        description="Non-private FedAvg under matched Dirichlet partitioning",
    ),
    "fedprox_noniid": ReferenceVariant(
        name="fedprox_noniid",
        method="fedprox",
        split="noniid",
        uses_alpha=True,
        description="Non-private FedProx under matched Dirichlet partitioning",
    ),
}

DEFAULT_VARIANTS = [
    "centralized_mlp",
    "logistic_regression",
    "hist_gbdt",
    "xgboost",
    "fedavg_iid",
    "fedavg_noniid",
    "fedprox_noniid",
]

SUMMARY_METRICS = (
    "auc",
    "auprc",
    "f1",
    "balanced_accuracy",
    "precision",
    "recall",
    "brier",
    "training_time",
    "communication_cost",
    "client_auprc_mean",
    "client_auprc_min",
    "client_f1_mean",
    "client_f1_min",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run centralized and non-private federated v6 references."
    )
    result.add_argument(
        "--dataset",
        choices=["default_credit", "gmsc", "german"],
        required=True,
    )
    result.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    result.add_argument("--seeds", nargs="+", type=int, default=FINAL_SEEDS)
    result.add_argument(
        "--variants",
        nargs="+",
        choices=REFERENCE_VARIANTS,
        default=DEFAULT_VARIANTS,
    )
    result.add_argument("--rounds", type=int, default=50)
    result.add_argument("--centralized-epochs", type=int, default=30)
    result.add_argument("--prox-mu", type=float, default=0.01)
    result.add_argument("--output-prefix", default="reference_baselines_v6")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--smoke", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def validate(args: argparse.Namespace) -> None:
    if args.rounds <= 0 or args.centralized_epochs <= 0:
        raise ValueError("rounds and centralized epochs must be positive.")
    if args.prox_mu < 0:
        raise ValueError("FedProx mu cannot be negative.")
    if any(alpha <= 0 for alpha in args.alphas):
        raise ValueError("Dirichlet alphas must be positive.")
    if len(set(args.alphas)) != len(args.alphas):
        raise ValueError("--alphas contains duplicates.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds contains duplicates.")
    if args.resume and args.overwrite:
        raise ValueError("Use either --resume or --overwrite, not both.")


def build_plan(
    variants: list[str],
    alphas: list[float],
    seeds: list[int],
) -> list[tuple[str, float | None, int]]:
    plan: list[tuple[str, float | None, int]] = []
    for variant_name in variants:
        variant = REFERENCE_VARIANTS[variant_name]
        variant_alphas: list[float | None] = alphas if variant.uses_alpha else [None]
        for alpha in variant_alphas:
            for seed in seeds:
                plan.append((variant_name, alpha, seed))
    return plan


def _key(row: dict[str, Any]) -> tuple[str, float | None, int]:
    alpha_value = row.get("alpha")
    alpha = (
        None
        if alpha_value in (None, "", "None", "nan")
        else parse_float(alpha_value)
    )
    return (
        str(row["baseline_variant"]),
        alpha,
        int(parse_float(row["seed"])),
    )


def main() -> None:
    args = parser().parse_args()
    validate(args)
    ensure_result_dirs()

    seeds = args.seeds[:1] if args.smoke else args.seeds
    alphas = args.alphas[:1] if args.smoke else args.alphas
    variants = args.variants[:1] if args.smoke else args.variants
    rounds = 2 if args.smoke else args.rounds
    centralized_epochs = 2 if args.smoke else args.centralized_epochs
    suffix = "_smoke" if args.smoke else ""
    stem = f"{args.dataset}_{args.output_prefix}{suffix}"
    raw_path = TABLES_DIR / f"{stem}_raw.csv"
    summary_path = TABLES_DIR / f"{stem}_mean_std.csv"

    if args.overwrite:
        raw_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    elif raw_path.exists() and not args.resume:
        raise FileExistsError(f"{raw_path} exists; use --resume or --overwrite.")

    plan = build_plan(variants, alphas, seeds)
    print(
        f"[Reference plan] dataset={args.dataset} runs={len(plan)} "
        f"rounds={rounds} variants={variants}",
        flush=True,
    )
    if args.dry_run:
        for index, (variant_name, alpha, seed) in enumerate(plan, 1):
            print(
                f"[{index:03d}/{len(plan)}] variant={variant_name} "
                f"alpha={alpha} seed={seed}"
            )
        return

    from src.experiment_runner import run_single_experiment

    rows: list[dict[str, Any]] = read_csv_rows(raw_path) if args.resume else []
    completed = {_key(row) for row in rows}
    for index, (variant_name, alpha, seed) in enumerate(plan, 1):
        key = (variant_name, alpha, seed)
        if key in completed:
            print(f"[Skip] {index}/{len(plan)} {key}", flush=True)
            continue
        variant = REFERENCE_VARIANTS[variant_name]
        print(
            f"\n=== [{index}/{len(plan)}] variant={variant_name} "
            f"alpha={alpha} seed={seed} ===",
            flush=True,
        )
        row = run_single_experiment(
            dataset=args.dataset,
            method=variant.method,
            split=variant.split,
            alpha=alpha,
            seed=seed,
            verbose=False,
            global_rounds_override=rounds,
            centralized_epochs_override=centralized_epochs,
            class_weighting="balanced",
            prox_mu=args.prox_mu,
            evaluate_client_profiles=variant.method in {"fedavg", "fedprox"},
            ablation_id=f"V6_REFERENCE_{variant_name}",
            ablation_description=variant.description,
        )
        row.update(
            {
                "baseline_variant": variant_name,
                "reference_description": variant.description,
                "reference_protocol": "validation_threshold_matched_v6",
                "reference_scope": (
                    "centralized_upper_reference"
                    if variant.method
                    in {
                        "centralized",
                        "logistic_regression",
                        "hist_gbdt",
                        "xgboost",
                    }
                    else "non_private_federated_reference"
                ),
            }
        )
        rows.append(row)
        completed.add(key)
        write_csv_rows(raw_path, rows)
        print(f"[Checkpoint] {len(completed)}/{len(plan)} -> {raw_path}", flush=True)

    summary = summarize_numeric(
        rows,
        ("dataset", "baseline_variant", "split", "alpha", "reference_scope"),
        SUMMARY_METRICS,
    )
    write_csv_rows(summary_path, summary)
    print(f"[Complete] raw={raw_path} summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
