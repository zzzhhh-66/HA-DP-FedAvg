"""Run a privacy-matched epsilon sweep for DP, scheduler, and full HA-DP."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from src.experiment_runner import run_single_experiment
from src.utils import DEFAULT_SEEDS, TABLES_DIR, ensure_result_dirs
from src.v4_experiment_utils import (
    read_csv_rows,
    require_v3_core,
    summarize_numeric,
    write_csv_rows,
)

CURVE_METHODS = {
    "dp_uniform": {"ha": False, "schedule": "uniform"},
    "scheduler": {"ha": False, "schedule": "back_loaded"},
    "full_ha_dp": {"ha": True, "schedule": "back_loaded"},
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Privacy-utility epsilon sweep.")
    result.add_argument("--dataset", default="default_credit")
    result.add_argument("--alphas", nargs="+", type=float, default=[0.5, 0.1])
    result.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    result.add_argument("--epsilons", nargs="+", type=float, default=[1.0, 2.0, 3.0, 5.0, 8.0])
    result.add_argument("--methods", nargs="+", choices=CURVE_METHODS, default=list(CURVE_METHODS))
    result.add_argument("--rounds", type=int, default=50)
    result.add_argument("--metadata-epsilon", type=float, default=0.1)
    result.add_argument("--ha-strength", type=float, default=0.5)
    result.add_argument("--ha-prior-strength", type=float, default=20.0)
    result.add_argument("--ha-config", type=Path)
    result.add_argument("--secure-rng", action="store_true")
    result.add_argument("--output-prefix", default="epsilon_curve_v4")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--smoke", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def load_ha_config(args: argparse.Namespace) -> None:
    if args.ha_config is None:
        return
    payload = json.loads(args.ha_config.read_text(encoding="utf-8"))
    args.ha_strength = float(payload["ha_strength"])
    args.ha_prior_strength = float(payload["ha_prior_strength"])


def validate(args: argparse.Namespace) -> None:
    if args.rounds <= 0 or any(value <= 0 for value in args.epsilons):
        raise ValueError("rounds and all epsilons must be positive.")
    if any(args.metadata_epsilon >= value for value in args.epsilons):
        raise ValueError("metadata epsilon must be smaller than every total epsilon.")
    if not 0 <= args.ha_strength <= 1 or args.ha_prior_strength < 0:
        raise ValueError("Invalid HA strength or prior.")
    if args.resume and args.overwrite:
        raise ValueError("Use either --resume or --overwrite, not both.")
    if args.secure_rng and importlib.util.find_spec("torchcsprng") is None:
        raise RuntimeError(
            "--secure-rng requires torchcsprng in HyperET. Install a build "
            "compatible with the current PyTorch/CUDA before this final run."
        )


def main() -> None:
    args = parser().parse_args()
    load_ha_config(args)
    validate(args)
    require_v3_core()
    ensure_result_dirs()

    alphas = args.alphas[:1] if args.smoke else args.alphas
    seeds = args.seeds[:1] if args.smoke else args.seeds
    epsilons = args.epsilons[:1] if args.smoke else args.epsilons
    rounds = 3 if args.smoke else args.rounds
    suffix = "_smoke" if args.smoke else ""
    stem = f"{args.dataset}_{args.output_prefix}{suffix}"
    raw_path = TABLES_DIR / f"{stem}_raw.csv"
    summary_path = TABLES_DIR / f"{stem}_mean_std.csv"
    if args.overwrite:
        raw_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    elif raw_path.exists() and not args.resume:
        raise FileExistsError(f"{raw_path} exists; use --resume or --overwrite.")

    plan = [
        (alpha, seed, epsilon, method_name)
        for alpha in alphas
        for epsilon in epsilons
        for seed in seeds
        for method_name in args.methods
    ]
    print(
        f"[Epsilon curve] runs={len(plan)} epsilons={epsilons} methods={args.methods} "
        f"secure_rng={args.secure_rng} ha_strength={args.ha_strength} "
        f"ha_prior={args.ha_prior_strength}",
        flush=True,
    )
    if args.dry_run:
        for index, item in enumerate(plan, 1):
            print(f"[{index}/{len(plan)}] alpha={item[0]} seed={item[1]} eps={item[2]} method={item[3]}")
        return

    rows = read_csv_rows(raw_path) if args.resume else []
    completed = {
        (float(row["alpha"]), int(float(row["seed"])), float(row["target_epsilon"]), str(row["curve_method"]))
        for row in rows
    }
    for index, (alpha, seed, epsilon, method_name) in enumerate(plan, 1):
        key = (alpha, seed, epsilon, method_name)
        if key in completed:
            print(f"[Skip] {index}/{len(plan)} {key}", flush=True)
            continue
        spec = CURVE_METHODS[method_name]
        metadata_epsilon = args.metadata_epsilon if spec["ha"] else 0.0
        print(f"\n=== [{index}/{len(plan)}] alpha={alpha} seed={seed} epsilon={epsilon} method={method_name} ===", flush=True)
        row = run_single_experiment(
            dataset=args.dataset,
            method="ha_dp_fedavg",
            split="noniid",
            alpha=alpha,
            seed=seed,
            verbose=False,
            global_rounds_override=rounds,
            privacy_mode="epsilon_allocation",
            total_epsilon=epsilon,
            epsilon_strategy=str(spec["schedule"]),
            use_heterogeneity_aware=bool(spec["ha"]),
            metadata_epsilon=metadata_epsilon,
            balance_gamma=1.0,
            class_weighting="balanced",
            ha_strength=args.ha_strength,
            ha_prior_strength=args.ha_prior_strength,
            secure_rng=args.secure_rng,
            ablation_id=f"EPS_{method_name}",
            ablation_description="Privacy-utility curve with matched total epsilon",
        )
        row.update({
            "curve_method": method_name,
            "target_epsilon": epsilon,
            "curve_protocol": "matched_total_epsilon_v4",
            "ha_strength": args.ha_strength if spec["ha"] else 0.0,
            "ha_prior_strength": args.ha_prior_strength if spec["ha"] else 0.0,
        })
        rows.append(row)
        completed.add(key)
        write_csv_rows(raw_path, rows)
        print(f"[Checkpoint] {len(completed)}/{len(plan)} -> {raw_path}", flush=True)

    summary = summarize_numeric(
        rows,
        ("alpha", "target_epsilon", "curve_method"),
        ("auc", "auprc", "f1", "balanced_accuracy", "precision", "recall", "brier", "epsilon", "training_time"),
    )
    write_csv_rows(summary_path, summary)
    print(f"[Complete] raw={raw_path} summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()

