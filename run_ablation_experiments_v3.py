"""Run privacy-matched HA-DP-FedAvg v3 ablations with checkpoints.

The core experiment is a 2 x 2 factorial design:

    HA weighting:       off / on
    privacy schedule:   uniform / back-loaded

All cells use the same total epsilon. When HA weighting is enabled, the
metadata release is included in that total budget.
"""

from __future__ import annotations

import argparse
import csv
import inspect
from pathlib import Path
from typing import Any

from src.ablation_configs_v3 import (
    FACTORIAL_ABLATIONS,
    AblationSpec,
    validate_factorial_ablation_specs,
)
from src.experiment_runner import (
    aggregate_ablation_results,
    run_single_experiment,
    save_results,
)
from src.utils import DEFAULT_SEEDS, TABLES_DIR, ensure_result_dirs

DEFAULT_ALPHAS = [0.5, 0.1]

NUMERIC_CSV_FIELDS = {
    "accuracy",
    "alpha",
    "auc",
    "auprc",
    "balance_gamma",
    "balanced_accuracy",
    "best_round",
    "best_val_round",
    "brier",
    "communication_cost",
    "configured_noise_multiplier",
    "delta",
    "epsilon",
    "f1",
    "fixed_f1",
    "fixed_precision",
    "fixed_recall",
    "ha_prior_strength",
    "ha_strength",
    "max_grad_norm",
    "mcc",
    "metadata_epsilon",
    "noise_max",
    "noise_min",
    "noise_multiplier",
    "num_clients",
    "num_features",
    "positive_class_weight",
    "precision",
    "recall",
    "seed",
    "selected_threshold",
    "specificity",
    "total_epsilon",
    "training_time",
}

BOOLEAN_CSV_FIELDS = {
    "privacy_claim_valid",
    "secure_rng",
    "use_heterogeneity_aware",
}

EFFECT_METRICS = (
    "auc",
    "auprc",
    "f1",
    "balanced_accuracy",
    "precision",
    "recall",
    "training_time",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the privacy-matched HA-DP-FedAvg v3 factorial ablation."
    )
    parser.add_argument(
        "--dataset",
        choices=["gmsc", "german", "default_credit"],
        default="default_credit",
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--epsilon", type=float, default=3.0)
    parser.add_argument("--metadata-epsilon", type=float, default=0.1)
    parser.add_argument("--ha-strength", type=float, default=0.5)
    parser.add_argument("--ha-prior-strength", type=float, default=20.0)
    parser.add_argument("--balance-gamma", type=float, default=1.0)
    parser.add_argument(
        "--class-weighting",
        choices=["balanced", "none"],
        default="balanced",
    )
    parser.add_argument(
        "--output-prefix",
        default="ablation_v3",
        help="Result filename prefix after the dataset name.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load the raw checkpoint and skip completed runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing files for this output prefix before running.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one alpha, one seed and three rounds under a _smoke prefix.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print and validate the plan without training.",
    )
    return parser


def build_plan(
    alphas: list[float],
    seeds: list[int],
    specs: tuple[AblationSpec, ...] = FACTORIAL_ABLATIONS,
) -> list[tuple[float, int, AblationSpec]]:
    return [
        (alpha, seed, spec)
        for alpha in alphas
        for seed in seeds
        for spec in specs
    ]


def _validate_args(args: argparse.Namespace) -> None:
    validate_factorial_ablation_specs()
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive.")
    if args.epsilon <= 0:
        raise ValueError("--epsilon must be positive.")
    if args.metadata_epsilon < 0 or args.metadata_epsilon >= args.epsilon:
        raise ValueError("--metadata-epsilon must be in [0, epsilon).")
    if not 0 <= args.ha_strength <= 1:
        raise ValueError("--ha-strength must be in [0, 1].")
    if args.ha_prior_strength < 0:
        raise ValueError("--ha-prior-strength cannot be negative.")
    if args.balance_gamma < 0:
        raise ValueError("--balance-gamma cannot be negative.")
    if any(alpha <= 0 for alpha in args.alphas):
        raise ValueError("Every Dirichlet alpha must be positive.")
    if len(set(args.alphas)) != len(args.alphas):
        raise ValueError("--alphas contains duplicates.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds contains duplicates.")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together.")


def _validate_v3_core_interface() -> None:
    parameters = inspect.signature(run_single_experiment).parameters
    required = {"ha_strength", "ha_prior_strength"}
    missing = sorted(required.difference(parameters))
    if missing:
        raise RuntimeError(
            "The installed core is older than v3 and cannot reproduce the tuned "
            f"HA weighting. Missing run_single_experiment arguments: {missing}. "
            "Upload the v3 core update before running this ablation."
        )


def _coerce_csv_value(key: str, value: str) -> Any:
    if value == "":
        return ""
    if key in BOOLEAN_CSV_FIELDS:
        return value.strip().lower() == "true"
    if key in NUMERIC_CSV_FIELDS:
        number = float(value)
        if key in {"best_round", "best_val_round", "num_clients", "num_features", "seed"}:
            return int(number)
        return number
    return value


def load_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return [
            {key: _coerce_csv_value(key, value) for key, value in row.items()}
            for row in csv.DictReader(file)
        ]


def completed_run_keys(rows: list[dict[str, Any]]) -> set[tuple[float, int, str]]:
    completed = set()
    for row in rows:
        try:
            completed.add(
                (
                    float(row["alpha"]),
                    int(float(row["seed"])),
                    str(row["ablation_id"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return completed


def build_effect_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create paper-friendly component contrasts from the summary table."""
    by_cell = {
        (float(row["alpha"]), str(row["ablation_id"])): row
        for row in summary
        if row.get("alpha") not in (None, "")
    }
    contrasts = (
        ("ha_effect_uniform", "A2", "A0"),
        ("ha_effect_back_loaded", "A3", "A1"),
        ("schedule_effect_without_ha", "A1", "A0"),
        ("schedule_effect_with_ha", "A3", "A2"),
        ("full_method_vs_matched_dp", "A3", "A0"),
    )
    effect_rows: list[dict[str, Any]] = []
    for alpha in sorted({key[0] for key in by_cell}):
        for contrast, treatment_id, reference_id in contrasts:
            treatment = by_cell.get((alpha, treatment_id))
            reference = by_cell.get((alpha, reference_id))
            if treatment is None or reference is None:
                continue
            effect: dict[str, Any] = {
                "dataset": treatment.get("dataset", ""),
                "alpha": alpha,
                "contrast": contrast,
                "treatment": treatment_id,
                "reference": reference_id,
                "num_runs_treatment": treatment.get("num_runs", ""),
                "num_runs_reference": reference.get("num_runs", ""),
            }
            for metric in EFFECT_METRICS:
                key = f"{metric}_mean"
                if treatment.get(key) is not None and reference.get(key) is not None:
                    effect[f"{metric}_delta"] = float(treatment[key]) - float(reference[key])
            effect_rows.append(effect)
    return effect_rows


def _output_paths(
    dataset: str,
    output_prefix: str,
) -> tuple[Path, Path, Path]:
    stem = f"{dataset}_{output_prefix}"
    return (
        TABLES_DIR / f"{stem}_raw.csv",
        TABLES_DIR / f"{stem}_mean_std.csv",
        TABLES_DIR / f"{stem}_effects.csv",
    )


def _save_checkpoint(
    rows: list[dict[str, Any]],
    raw_path: Path,
    summary_path: Path,
    effects_path: Path,
) -> None:
    summary = aggregate_ablation_results(rows)
    save_results(rows, raw_path)
    save_results(summary, summary_path)
    save_results(build_effect_rows(summary), effects_path)


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    ensure_result_dirs()

    alphas = [args.alphas[0]] if args.smoke else args.alphas
    seeds = [args.seeds[0]] if args.smoke else args.seeds
    rounds = 3 if args.smoke else args.rounds
    output_prefix = (
        f"{args.output_prefix}_smoke" if args.smoke else args.output_prefix
    )
    plan = build_plan(alphas, seeds)
    raw_path, summary_path, effects_path = _output_paths(
        args.dataset, output_prefix
    )

    print(
        f"Ablation plan: dataset={args.dataset}, runs={len(plan)}, rounds={rounds}, "
        f"alphas={alphas}, seeds={seeds}, epsilon={args.epsilon}",
        flush=True,
    )
    for index, (alpha, seed, spec) in enumerate(plan, start=1):
        print(
            f"  [{index:02d}/{len(plan):02d}] alpha={alpha} seed={seed} "
            f"{spec.ablation_id}={spec.paper_label}",
            flush=True,
        )

    if args.dry_run:
        print("Dry-run passed. No training was executed.", flush=True)
        return

    _validate_v3_core_interface()
    if args.overwrite:
        for path in (raw_path, summary_path, effects_path):
            path.unlink(missing_ok=True)
    elif raw_path.exists() and not args.resume:
        raise FileExistsError(
            f"{raw_path} already exists. Use --resume to continue or "
            "--overwrite to start again."
        )

    rows = load_checkpoint(raw_path) if args.resume else []
    completed = completed_run_keys(rows)
    if completed:
        print(
            f"[Resume] loaded {len(completed)} completed runs from {raw_path}",
            flush=True,
        )

    for run_index, (alpha, seed, spec) in enumerate(plan, start=1):
        run_key = (float(alpha), int(seed), spec.ablation_id)
        if run_key in completed:
            print(
                f"[Skip] [{run_index}/{len(plan)}] alpha={alpha} seed={seed} "
                f"ablation={spec.ablation_id}",
                flush=True,
            )
            continue

        metadata_epsilon = (
            args.metadata_epsilon if spec.use_heterogeneity_aware else 0.0
        )
        print(
            f"\n=== [{run_index}/{len(plan)}] dataset={args.dataset} "
            f"alpha={alpha} seed={seed} ablation={spec.ablation_id} "
            f"ha={spec.use_heterogeneity_aware} "
            f"schedule={spec.epsilon_strategy} ===",
            flush=True,
        )
        row = run_single_experiment(
            dataset=args.dataset,
            method="ha_dp_fedavg",
            split="noniid",
            alpha=alpha,
            seed=seed,
            verbose=False,
            global_rounds_override=rounds,
            privacy_mode="epsilon_allocation",
            total_epsilon=args.epsilon,
            epsilon_strategy=spec.epsilon_strategy,
            use_heterogeneity_aware=spec.use_heterogeneity_aware,
            metadata_epsilon=metadata_epsilon,
            balance_gamma=args.balance_gamma,
            class_weighting=args.class_weighting,
            ha_strength=args.ha_strength,
            ha_prior_strength=args.ha_prior_strength,
            ablation_id=spec.ablation_id,
            ablation_description=spec.description,
        )
        row.update(
            {
                "ablation_id": spec.ablation_id,
                "ablation_label": spec.paper_label,
                "ablation_description": spec.description,
                "factor_ha_weighting": spec.use_heterogeneity_aware,
                "factor_privacy_schedule": spec.epsilon_strategy,
                "ha_strength": args.ha_strength
                if spec.use_heterogeneity_aware
                else 0.0,
                "ha_prior_strength": args.ha_prior_strength
                if spec.use_heterogeneity_aware
                else 0.0,
                "ablation_protocol": "privacy_matched_2x2_v3",
            }
        )
        rows.append(row)
        completed.add(run_key)
        _save_checkpoint(rows, raw_path, summary_path, effects_path)
        print(
            f"[Checkpoint] completed={len(completed)}/{len(plan)} "
            f"raw={raw_path} summary={summary_path} effects={effects_path}",
            flush=True,
        )

    print(
        f"\nCompleted {len(completed)}/{len(plan)} ablation runs. "
        f"Summary: {summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()

