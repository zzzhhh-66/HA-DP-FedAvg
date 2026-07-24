from pathlib import Path

from src.experiment_runner import build_arg_parser, run_single_experiment, save_results
from src.utils import TABLES_DIR, ensure_result_dirs


def main():
    parser = build_arg_parser("Run a single experiment on German Credit.")
    args = parser.parse_args()

    ensure_result_dirs()
    result = run_single_experiment(
        dataset="german",
        method=args.method,
        split=args.split,
        alpha=args.alpha if args.split == "noniid" else None,
        noise_multiplier=args.noise,
        seed=args.seed,
        privacy_mode=args.privacy_mode,
        total_epsilon=args.total_epsilon,
        epsilon_strategy=args.epsilon_strategy,
        noise_schedule=args.noise_schedule,
        noise_min=args.noise_min,
        noise_max=args.noise_max,
        balance_gamma=args.balance_gamma,
        ha_strength=args.ha_strength,
        ha_prior_strength=args.ha_prior_strength,
        use_heterogeneity_aware=(
            False if args.method == "dp_fedavg" else not args.no_ha_aggregation
        ),
        class_weighting=args.class_weighting,
        prox_mu=args.prox_mu,
        metadata_epsilon=args.metadata_epsilon,
        secure_rng=args.secure_rng,
        dp_target_epsilon=args.dp_target_epsilon,
    )

    output_path = (
        Path(args.output)
        if args.output
        else TABLES_DIR / f"german_{args.split}_seed{args.seed}.csv"
    )
    save_results([result], output_path)
    print(f"Saved result to {output_path}")


if __name__ == "__main__":
    main()
