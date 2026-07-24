"""Select HA strength on development seeds using validation AUPRC only."""

from __future__ import annotations

import argparse
from statistics import mean

from src.experiment_runner import run_single_experiment
from src.utils import TABLES_DIR, ensure_result_dirs
from src.v4_experiment_utils import (
    parse_float,
    read_csv_rows,
    require_v3_core,
    summarize_numeric,
    write_csv_rows,
    write_json,
)

FINAL_SEEDS = {42, 123, 2026, 31415, 27182}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Leakage-safe HA hyperparameter selection on development seeds."
    )
    result.add_argument("--dataset", default="default_credit")
    result.add_argument("--alphas", nargs="+", type=float, default=[0.5, 0.1])
    result.add_argument("--dev-seeds", nargs="+", type=int, default=[7, 11])
    result.add_argument("--rounds", type=int, default=30)
    result.add_argument("--epsilon", type=float, default=3.0)
    result.add_argument("--metadata-epsilon", type=float, default=0.1)
    result.add_argument("--ha-strengths", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    result.add_argument("--ha-priors", nargs="+", type=float, default=[5.0, 20.0, 50.0])
    result.add_argument("--output-prefix", default="ha_tuning_v4")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def validate(args: argparse.Namespace) -> None:
    if FINAL_SEEDS.intersection(args.dev_seeds):
        raise ValueError("Development seeds must not overlap the five final evaluation seeds.")
    if args.rounds <= 0 or args.epsilon <= 0:
        raise ValueError("rounds and epsilon must be positive.")
    if not 0 <= args.metadata_epsilon < args.epsilon:
        raise ValueError("metadata epsilon must be in [0, epsilon).")
    if any(not 0 < strength <= 1 for strength in args.ha_strengths):
        raise ValueError("HA strengths must be in (0, 1].")
    if any(prior < 0 for prior in args.ha_priors):
        raise ValueError("HA priors cannot be negative.")
    if args.resume and args.overwrite:
        raise ValueError("Use either --resume or --overwrite, not both.")


def main() -> None:
    args = parser().parse_args()
    validate(args)
    ensure_result_dirs()
    require_v3_core()

    stem = f"{args.dataset}_{args.output_prefix}"
    raw_path = TABLES_DIR / f"{stem}_raw.csv"
    summary_path = TABLES_DIR / f"{stem}_mean_std.csv"
    selected_path = TABLES_DIR / f"{stem}_selected.json"
    if args.overwrite:
        for path in (raw_path, summary_path, selected_path):
            path.unlink(missing_ok=True)
    elif raw_path.exists() and not args.resume:
        raise FileExistsError(f"{raw_path} exists; use --resume or --overwrite.")

    candidate_plan = [
        ("ha_candidate", alpha, seed, strength, prior)
        for alpha in args.alphas
        for seed in args.dev_seeds
        for strength in args.ha_strengths
        for prior in args.ha_priors
    ]
    baseline_plan = [
        ("scheduler_only", alpha, seed, 0.0, 0.0)
        for alpha in args.alphas
        for seed in args.dev_seeds
    ]
    plan = baseline_plan + candidate_plan
    print(
        f"[HA tuning] runs={len(plan)} dataset={args.dataset} alphas={args.alphas} "
        f"dev_seeds={args.dev_seeds} rounds={args.rounds}",
        flush=True,
    )
    if args.dry_run:
        for index, item in enumerate(plan, 1):
            print(
                f"[{index}/{len(plan)}] type={item[0]} alpha={item[1]} "
                f"seed={item[2]} strength={item[3]} prior={item[4]}"
            )
        return

    rows = read_csv_rows(raw_path) if args.resume else []
    completed = {
        (
            str(row["tuning_method"]),
            float(row["alpha"]),
            int(float(row["seed"])),
            float(row["ha_strength"]),
            float(row["ha_prior_strength"]),
        )
        for row in rows
    }
    for index, (tuning_method, alpha, seed, strength, prior) in enumerate(plan, 1):
        key = (tuning_method, alpha, seed, strength, prior)
        if key in completed:
            print(f"[Skip] {index}/{len(plan)} {key}", flush=True)
            continue
        use_ha = tuning_method == "ha_candidate"
        print(
            f"\n=== [{index}/{len(plan)}] type={tuning_method} alpha={alpha} "
            f"seed={seed} strength={strength} prior={prior} ===",
            flush=True,
        )
        row = run_single_experiment(
            dataset=args.dataset,
            method="ha_dp_fedavg",
            split="noniid",
            alpha=alpha,
            seed=seed,
            verbose=False,
            global_rounds_override=args.rounds,
            privacy_mode="epsilon_allocation",
            total_epsilon=args.epsilon,
            epsilon_strategy="back_loaded",
            use_heterogeneity_aware=use_ha,
            metadata_epsilon=args.metadata_epsilon if use_ha else 0.0,
            balance_gamma=1.0,
            class_weighting="balanced",
            ha_strength=strength,
            ha_prior_strength=prior,
            ablation_id="HA_TUNE_DEV_ONLY" if use_ha else "SCHEDULER_DEV_CONTROL",
            ablation_description="HA tuning selected using validation AUPRC on disjoint development seeds",
        )
        row.update({
            "ha_strength": strength,
            "ha_prior_strength": prior,
            "tuning_method": tuning_method,
            "selection_split": "validation_only",
            "selection_metric": "validation_auprc",
            "development_seed": True,
        })
        rows.append(row)
        completed.add(key)
        write_csv_rows(raw_path, rows)
        print(f"[Checkpoint] {len(completed)}/{len(plan)} -> {raw_path}", flush=True)

    summary = summarize_numeric(
        rows,
        ("tuning_method", "ha_strength", "ha_prior_strength"),
        ("validation_selection_score", "auc", "auprc", "f1", "training_time"),
    )
    write_csv_rows(summary_path, summary)
    candidates = [row for row in summary if row["tuning_method"] == "ha_candidate"]
    best = max(candidates, key=lambda row: parse_float(row["validation_selection_score_mean"]))
    baseline_scores = [
        parse_float(row["validation_selection_score"])
        for row in rows
        if row["tuning_method"] == "scheduler_only"
    ]
    baseline_score = mean(baseline_scores)
    selection_score = parse_float(best["validation_selection_score_mean"])
    selected = {
        "dataset": args.dataset,
        "ha_strength": parse_float(best["ha_strength"]),
        "ha_prior_strength": parse_float(best["ha_prior_strength"]),
        "selection_metric": "mean validation AUPRC",
        "selection_score": selection_score,
        "scheduler_only_validation_score": baseline_score,
        "validation_score_delta_vs_scheduler": selection_score - baseline_score,
        "ha_supported_on_development": selection_score > baseline_score,
        "development_seeds": args.dev_seeds,
        "final_evaluation_seeds": sorted(FINAL_SEEDS),
        "test_metrics_were_not_used_for_selection": True,
    }
    write_json(selected_path, selected)
    print(f"[Selected] {selected}", flush=True)
    print(f"[Output] {selected_path}", flush=True)


if __name__ == "__main__":
    main()
