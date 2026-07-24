"""Verify the v5 exact RDP optimization and report its speedup."""

from __future__ import annotations

import argparse
import time

import numpy as np

from src.privacy_accounting import (
    calibrate_noise_schedule,
    clear_noise_schedule_cache,
    compose_rdp_epsilon,
    compose_rdp_epsilon_prefixes,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--batches", type=int, default=14)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=2.9)
    parser.add_argument("--delta", type=float, default=1e-5)
    args = parser.parse_args()
    sample_rate = 1.0 / args.batches
    steps = args.batches * args.local_epochs

    for strategy in ("uniform", "front_loaded", "back_loaded"):
        common = dict(
            target_epsilon=args.epsilon,
            delta=args.delta,
            total_rounds=args.rounds,
            sample_rate=sample_rate,
            steps_per_round=steps,
            strategy=strategy,
            use_cache=False,
        )
        clear_noise_schedule_cache()
        start = time.perf_counter()
        legacy = calibrate_noise_schedule(**common, use_fast_accounting=False)
        legacy_seconds = time.perf_counter() - start

        start = time.perf_counter()
        optimized = calibrate_noise_schedule(**common, use_fast_accounting=True)
        optimized_seconds = time.perf_counter() - start
        legacy_epsilon = compose_rdp_epsilon(legacy, sample_rate, steps, args.delta)
        optimized_epsilon = compose_rdp_epsilon(optimized, sample_rate, steps, args.delta)
        if not np.allclose(legacy, optimized, rtol=0.0, atol=1e-10):
            raise AssertionError("Optimized calibration changed the noise schedule.")

        start = time.perf_counter()
        legacy_prefixes = [
            compose_rdp_epsilon(legacy[:index], sample_rate, steps, args.delta)
            for index in range(1, len(legacy) + 1)
        ]
        legacy_prefix_seconds = time.perf_counter() - start
        start = time.perf_counter()
        optimized_prefixes = compose_rdp_epsilon_prefixes(
            optimized,
            sample_rate,
            steps,
            args.delta,
        )
        optimized_prefix_seconds = time.perf_counter() - start
        if not np.allclose(legacy_prefixes, optimized_prefixes, rtol=0.0, atol=1e-10):
            raise AssertionError("Optimized prefix accounting changed epsilon history.")

        print(
            f"strategy={strategy} legacy_eps={legacy_epsilon:.9f} "
            f"optimized_eps={optimized_epsilon:.9f} "
            f"calibration_speedup={legacy_seconds / max(optimized_seconds, 1e-12):.1f}x "
            f"prefix_speedup={legacy_prefix_seconds / max(optimized_prefix_seconds, 1e-12):.1f}x",
            flush=True,
        )


if __name__ == "__main__":
    main()
