"""Tune the exploratory HA-scheduler coupling on development seeds only."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.experiment_runner import run_single_experiment
from src.utils import TABLES_DIR, ensure_result_dirs
from src.v4_experiment_utils import (
    parse_float,
    read_csv_rows,
    require_v5_core,
    summarize_numeric,
    write_csv_rows,
    write_json,
)

FINAL_SEEDS = {42, 123, 2026, 31415, 27182, 73, 101, 909, 4096, 65537}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Tune v5 HA-scheduler coupling.")
    result.add_argument("--dataset", default="default_credit")
    result.add_argument("--alphas", nargs="+", type=float, default=[0.5, 0.1])
    result.add_argument("--dev-seeds", nargs="+", type=int, default=[7, 11])
    result.add_argument("--rounds", type=int, default=30)
    result.add_argument("--epsilon", type=float, default=3.0)
    result.add_argument("--metadata-epsilon", type=float, default=0.1)
    result.add_argument("--ha-strength", type=float, default=1.0)
    result.add_argument("--ha-prior-strength", type=float, default=50.0)
    result.add_argument("--ha-config", type=Path)
    result.add_argument("--schedule-powers", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    result.add_argument("--couplings", nargs="+", type=float, default=[0.5, 1.0])
    result.add_argument("--output-prefix", default="scheduler_coupling_tuning_v5")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def load_ha_config(args: argparse.Namespace) -> None:
    if args.ha_config is None:
        return
    import json

    payload = json.loads(args.ha_config.read_text(encoding="utf-8"))
    args.ha_strength = float(payload["ha_strength"])
    args.ha_prior_strength = float(payload["ha_prior_strength"])


def validate(args: argparse.Namespace) -> None:
    if FINAL_SEEDS.intersection(args.dev_seeds):
        raise ValueError("Development seeds must not overlap final evaluation seeds.")
    if args.rounds <= 0 or args.epsilon <= args.metadata_epsilon:
        raise ValueError("Invalid rounds or privacy budgets.")
    if any(power <= 0 for power in args.schedule_powers):
        raise ValueError("Schedule powers must be positive.")
    if any(coupling < 0 for coupling in args.couplings):
        raise ValueError("Couplings must be non-negative.")
    if args.resume and args.overwrite:
        raise ValueError("Use either --resume or --overwrite, not both.")


def main() -> None:
    args = parser().parse_args()
    load_ha_config(args)
    validate(args)
    require_v5_core()
    ensure_result_dirs()

    stem = f"{args.dataset}_{args.output_prefix}"
    raw_path = TABLES_DIR / f"{stem}_raw.csv"
    summary_path = TABLES_DIR / f"{stem}_mean_std.csv"
    selected_path = TABLES_DIR / f"{stem}_selected.json"
    if args.overwrite:
        for path in (raw_path, summary_path, selected_path):
            path.unlink(missing_ok=True)
    elif raw_path.exists() and not args.resume:
        raise FileExistsError(f"{raw_path} exists; use --resume or --overwrite.")

    candidates = [(1.0, 0.0)] + [
        (power, coupling) for power in args.schedule_powers for coupling in args.couplings
    ]
    plan = [
        (alpha, seed, power, coupling)
        for alpha in args.alphas
        for seed in args.dev_seeds
        for power, coupling in candidates
    ]
    print(f"[Coupling tuning] runs={len(plan)} candidates={candidates}", flush=True)
    if args.dry_run:
        for index, item in enumerate(plan, 1):
            print(
                f"[{index:03d}/{len(plan)}] alpha={item[0]} seed={item[1]} "
                f"power={item[2]} coupling={item[3]}"
            )
        return

    rows = read_csv_rows(raw_path) if args.resume else []
    completed = {
        (
            parse_float(row["alpha"]),
            int(parse_float(row["seed"])),
            parse_float(row["schedule_power"]),
            parse_float(row["schedule_ha_coupling"]),
        )
        for row in rows
    }
    for index, (alpha, seed, power, coupling) in enumerate(plan, 1):
        key = (alpha, seed, power, coupling)
        if key in completed:
            print(f"[Skip] {index}/{len(plan)} {key}", flush=True)
            continue
        print(f"\n=== [{index}/{len(plan)}] {key} ===", flush=True)
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
            use_heterogeneity_aware=True,
            aggregation_strategy="balanced_ha",
            metadata_epsilon=args.metadata_epsilon,
            class_weighting="balanced",
            ha_strength=args.ha_strength,
            ha_prior_strength=args.ha_prior_strength,
            schedule_power=power,
            schedule_ha_coupling=coupling,
            fast_epsilon_tracking=True,
            evaluate_client_profiles=False,
            ablation_id=f"V5_COUPLING_DEV_ONLY_pow{power:g}_couple{coupling:g}",
            ablation_description="Coupling selected with validation AUPRC on development seeds",
        )
        row.update(
            {
                "tuning_method": "uncoupled" if coupling == 0 else "coupled",
                "development_seed": True,
                "selection_split": "validation_only",
                "selection_metric": "validation_auprc",
            }
        )
        rows.append(row)
        completed.add(key)
        write_csv_rows(raw_path, rows)
        print(f"[Checkpoint] {len(completed)}/{len(plan)} -> {raw_path}", flush=True)

    summary = summarize_numeric(
        rows,
        ("schedule_power", "schedule_ha_coupling"),
        ("validation_selection_score", "auc", "auprc", "f1", "training_time"),
    )
    write_csv_rows(summary_path, summary)
    best = max(summary, key=lambda row: parse_float(row["validation_selection_score_mean"]))
    uncoupled = next(
        row
        for row in summary
        if parse_float(row["schedule_power"]) == 1.0
        and parse_float(row["schedule_ha_coupling"]) == 0.0
    )
    selected_score = parse_float(best["validation_selection_score_mean"])
    uncoupled_score = parse_float(uncoupled["validation_selection_score_mean"])
    payload = {
        "dataset": args.dataset,
        "schedule_power": parse_float(best["schedule_power"]),
        "schedule_ha_coupling": parse_float(best["schedule_ha_coupling"]),
        "selection_metric": "mean validation AUPRC",
        "selection_score": selected_score,
        "uncoupled_validation_score": uncoupled_score,
        "validation_score_delta_vs_uncoupled": selected_score - uncoupled_score,
        "coupling_supported_on_development": selected_score > uncoupled_score,
        "development_seeds": args.dev_seeds,
        "test_metrics_were_not_used_for_selection": True,
    }
    write_json(selected_path, payload)
    print(f"[Selected] {payload}", flush=True)
    print(f"[Output] {selected_path}", flush=True)


if __name__ == "__main__":
    main()
