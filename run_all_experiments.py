import argparse

from src.experiment_runner import (
    aggregate_results,
    run_single_experiment,
    save_results,
)
from src.utils import DEFAULT_SEEDS, TABLES_DIR, ensure_result_dirs

GMS_EXPERIMENTS = [
    ("iid", None, "centralized", 0.0),
    ("iid", None, "logistic_regression", 0.0),
    ("iid", None, "hist_gbdt", 0.0),
    ("iid", None, "xgboost", 0.0),
    ("iid", None, "lightgbm", 0.0),
    ("iid", None, "local_only", 0.0),
    ("iid", None, "fedavg", 0.0),
    ("iid", None, "fedprox", 0.0),
    ("iid", None, "dp_fedavg", 0.5),
    ("iid", None, "dp_fedavg", 1.0),
    ("iid", None, "dp_fedavg", 2.0),
    ("noniid", 0.5, "fedavg", 0.0),
    ("noniid", 0.5, "fedprox", 0.0),
    ("noniid", 0.5, "label_aware_fedavg", 0.0),
    ("noniid", 0.5, "dp_fedavg", 0.5),
    ("noniid", 0.5, "dp_fedavg", 1.0),
    ("noniid", 0.5, "dp_fedavg", 2.0),
    ("noniid", 0.5, "dp_fedavg", None),
    ("noniid", 0.5, "ha_dp_fedavg", 0.0),
    ("noniid", 0.1, "fedavg", 0.0),
    ("noniid", 0.1, "fedprox", 0.0),
    ("noniid", 0.1, "label_aware_fedavg", 0.0),
    ("noniid", 0.1, "dp_fedavg", 0.5),
    ("noniid", 0.1, "dp_fedavg", 1.0),
    ("noniid", 0.1, "dp_fedavg", 2.0),
    ("noniid", 0.1, "dp_fedavg", None),
    ("noniid", 0.1, "ha_dp_fedavg", 0.0),
]

GERMAN_EXPERIMENTS = [
    ("iid", None, "centralized", 0.0),
    ("iid", None, "logistic_regression", 0.0),
    ("iid", None, "hist_gbdt", 0.0),
    ("iid", None, "xgboost", 0.0),
    ("iid", None, "lightgbm", 0.0),
    ("iid", None, "fedavg", 0.0),
    ("iid", None, "fedprox", 0.0),
    ("iid", None, "dp_fedavg", 0.7),
    ("noniid", 0.5, "fedavg", 0.0),
    ("noniid", 0.5, "fedprox", 0.0),
    ("noniid", 0.5, "label_aware_fedavg", 0.0),
    ("noniid", 0.5, "dp_fedavg", 0.7),
    ("noniid", 0.5, "dp_fedavg", None),
    ("noniid", 0.5, "ha_dp_fedavg", 0.0),
]


DEFAULT_CREDIT_EXPERIMENTS = [
    ("iid", None, "centralized", 0.0),
    ("iid", None, "logistic_regression", 0.0),
    ("iid", None, "hist_gbdt", 0.0),
    ("iid", None, "xgboost", 0.0),
    ("iid", None, "lightgbm", 0.0),
    ("iid", None, "fedavg", 0.0),
    ("iid", None, "fedprox", 0.0),
    ("iid", None, "dp_fedavg", 1.0),
    ("noniid", 0.5, "fedavg", 0.0),
    ("noniid", 0.5, "fedprox", 0.0),
    ("noniid", 0.5, "label_aware_fedavg", 0.0),
    ("noniid", 0.5, "dp_fedavg", 0.5),
    ("noniid", 0.5, "dp_fedavg", 1.0),
    ("noniid", 0.5, "dp_fedavg", 2.0),
    ("noniid", 0.5, "dp_fedavg", None),
    ("noniid", 0.5, "ha_dp_fedavg", 0.0),
    ("noniid", 0.1, "fedavg", 0.0),
    ("noniid", 0.1, "fedprox", 0.0),
    ("noniid", 0.1, "label_aware_fedavg", 0.0),
    ("noniid", 0.1, "dp_fedavg", 0.5),
    ("noniid", 0.1, "dp_fedavg", 1.0),
    ("noniid", 0.1, "dp_fedavg", 2.0),
    ("noniid", 0.1, "dp_fedavg", None),
    ("noniid", 0.1, "ha_dp_fedavg", 0.0),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full experiment matrix.")
    parser.add_argument(
        "--dataset", choices=["gmsc", "german", "default_credit", "all"], default="all"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run the minimum viable IID experiment only.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    ensure_result_dirs()

    all_results = []
    datasets = ["gmsc", "default_credit", "german"] if args.dataset == "all" else [args.dataset]
    seeds = [42] if args.quick else args.seeds

    for dataset in datasets:
        experiments = {
            "gmsc": GMS_EXPERIMENTS,
            "default_credit": DEFAULT_CREDIT_EXPERIMENTS,
            "german": GERMAN_EXPERIMENTS,
        }[dataset]

        if args.quick:
            experiments = [
                ("iid", None, "centralized", 0.0),
                ("iid", None, "logistic_regression", 0.0),
                ("iid", None, "hist_gbdt", 0.0),
                ("iid", None, "fedavg", 0.0),
                ("noniid", 0.5, "fedprox", 0.0),
                ("noniid", 0.5, "label_aware_fedavg", 0.0),
                ("iid", None, "dp_fedavg", 1.0),
                ("noniid", 0.5, "ha_dp_fedavg", 0.0),
            ]

        dataset_results = []
        suffix = "quick_results" if args.quick else "all_results"
        raw_path = TABLES_DIR / f"{dataset}_{suffix}.csv"
        if raw_path.exists():
            raw_path.unlink()
        total_runs = len(seeds) * len(experiments)
        completed_runs = 0
        for seed in seeds:
            for split, alpha, method, noise in experiments:
                print(
                    f"\n=== [{completed_runs + 1}/{total_runs}] "
                    f"Running dataset={dataset}, split={split}, "
                    f"alpha={alpha}, method={method}, noise={noise}, seed={seed} ===",
                    flush=True,
                )
                target_epsilon = 2.0 if dataset == "german" else 3.0
                result = run_single_experiment(
                    dataset=dataset,
                    method=method,
                    split=split,
                    alpha=alpha,
                    noise_multiplier=1.0 if noise is None else noise,
                    seed=seed,
                    verbose=False,
                    global_rounds_override=5 if args.quick else None,
                    centralized_epochs_override=5 if args.quick else None,
                    total_epsilon=target_epsilon,
                    metadata_epsilon=0.1 if method == "ha_dp_fedavg" else 0.0,
                    dp_target_epsilon=(
                        target_epsilon if method == "dp_fedavg" and noise is None else None
                    ),
                )
                dataset_results.append(result)
                all_results.append(result)
                completed_runs += 1
                save_results(dataset_results, raw_path)
                partial_summary = aggregate_results(all_results)
                partial_summary_name = (
                    "quick_results_mean_std.csv" if args.quick else "final_results_mean_std.csv"
                )
                save_results(partial_summary, TABLES_DIR / partial_summary_name)
                print(
                    f"[Checkpoint] completed={completed_runs}/{total_runs} raw={raw_path}",
                    flush=True,
                )

    summary = aggregate_results(all_results)
    summary_name = "quick_results_mean_std.csv" if args.quick else "final_results_mean_std.csv"
    summary_path = TABLES_DIR / summary_name
    if summary_path.exists():
        summary_path.unlink()
    save_results(summary, summary_path)

    print(f"\nSaved raw results and summary to {TABLES_DIR}")


if __name__ == "__main__":
    main()
