"""Reusable RDP accounting utilities for record-level private local training."""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from opacus.accountants import RDPAccountant
from opacus.accountants.analysis import rdp as rdp_analysis


def compose_rdp_epsilon(
    noise_multipliers: Sequence[float],
    sample_rate: float,
    steps_per_round: int,
    delta: float,
) -> float:
    """Compose all local DP-SGD steps performed by one client."""
    if not noise_multipliers:
        return 0.0
    if not 0 < sample_rate <= 1:
        raise ValueError("sample_rate must be in (0, 1].")
    if steps_per_round <= 0:
        raise ValueError("steps_per_round must be positive.")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1).")

    accountant = RDPAccountant()
    for noise_multiplier in noise_multipliers:
        if noise_multiplier <= 0:
            raise ValueError("noise multipliers must be positive.")
        for _ in range(steps_per_round):
            accountant.step(
                noise_multiplier=float(noise_multiplier),
                sample_rate=float(sample_rate),
            )
    return float(accountant.get_epsilon(delta=delta))


def compose_rdp_epsilon_fast(
    noise_multipliers: Sequence[float],
    sample_rate: float,
    steps_per_round: int,
    delta: float,
) -> float:
    """Compose a round-wise schedule without expanding every optimizer step.

    Opacus' RDP accountant is additive. Computing each round's RDP contribution
    and summing the vectors is mathematically equivalent to calling
    ``accountant.step`` for every local optimizer step, but it avoids creating a
    long Python history for non-uniform schedules.
    """
    if not noise_multipliers:
        return 0.0
    if not 0 < sample_rate <= 1:
        raise ValueError("sample_rate must be in (0, 1].")
    if steps_per_round <= 0:
        raise ValueError("steps_per_round must be positive.")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1).")
    if any(float(noise) <= 0 for noise in noise_multipliers):
        raise ValueError("noise multipliers must be positive.")

    orders = np.asarray(RDPAccountant.DEFAULT_ALPHAS, dtype=float)
    total_rdp = np.zeros_like(orders)

    # Group exactly equal noise values. Uniform schedules become one Opacus
    # analysis call, while non-uniform schedules remain exact round by round.
    grouped: dict[float, int] = {}
    for noise in noise_multipliers:
        value = float(noise)
        grouped[value] = grouped.get(value, 0) + 1

    for noise, round_count in grouped.items():
        total_rdp += np.asarray(
            rdp_analysis.compute_rdp(
                q=float(sample_rate),
                noise_multiplier=noise,
                steps=int(steps_per_round * round_count),
                orders=orders,
            ),
            dtype=float,
        )

    epsilon, _ = rdp_analysis.get_privacy_spent(
        orders=orders,
        rdp=total_rdp,
        delta=float(delta),
    )
    return float(epsilon)


def compose_rdp_epsilon_prefixes(
    noise_multipliers: Sequence[float],
    sample_rate: float,
    steps_per_round: int,
    delta: float,
) -> list[float]:
    """Return exact cumulative epsilon after every communication round.

    The noise schedule is known before training, so repeatedly rebuilding an
    Opacus accountant from an increasingly long heterogeneous history is
    unnecessary. Accumulating the same per-round RDP vectors once is exact and
    makes progress reporting effectively constant-time during training.
    """
    if not noise_multipliers:
        return []
    if not 0 < sample_rate <= 1:
        raise ValueError("sample_rate must be in (0, 1].")
    if steps_per_round <= 0:
        raise ValueError("steps_per_round must be positive.")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1).")
    if any(float(noise) <= 0 for noise in noise_multipliers):
        raise ValueError("noise multipliers must be positive.")

    orders = np.asarray(RDPAccountant.DEFAULT_ALPHAS, dtype=float)
    cumulative_rdp = np.zeros_like(orders)
    epsilon_history: list[float] = []
    for noise in noise_multipliers:
        cumulative_rdp += np.asarray(
            rdp_analysis.compute_rdp(
                q=float(sample_rate),
                noise_multiplier=float(noise),
                steps=int(steps_per_round),
                orders=orders,
            ),
            dtype=float,
        )
        epsilon, _ = rdp_analysis.get_privacy_spent(
            orders=orders,
            rdp=cumulative_rdp,
            delta=float(delta),
        )
        epsilon_history.append(float(epsilon))
    return epsilon_history


def privacy_allocation_weights(
    total_rounds: int,
    strategy: str,
    schedule_power: float = 1.0,
) -> list[float]:
    """Return normalized inverse-noise-squared cost-shaping weights.

    These are not literal per-round epsilon shares under subsampled RDP. The
    final schedule is calibrated jointly so only its composed epsilon has the
    requested formal meaning.
    """
    if total_rounds <= 0:
        raise ValueError("total_rounds must be positive.")
    if schedule_power <= 0:
        raise ValueError("schedule_power must be positive.")
    if strategy == "uniform":
        raw = [1.0] * total_rounds
    elif strategy == "front_loaded":
        raw = [float(total_rounds - index) ** schedule_power for index in range(total_rounds)]
    elif strategy == "back_loaded":
        raw = [float(index + 1) ** schedule_power for index in range(total_rounds)]
    else:
        raise ValueError(f"Unknown privacy allocation strategy: {strategy}")
    total = sum(raw)
    return [value / total for value in raw]


def _scaled_noise_schedule(base_noise: float, weights: Sequence[float]) -> list[float]:
    # Gaussian privacy cost is approximately proportional to 1 / sigma^2.
    # The factor T keeps the uniform schedule equal to base_noise.
    total_rounds = len(weights)
    return [base_noise / math.sqrt(weight * total_rounds) for weight in weights]


def _calibrate_noise_schedule_uncached(
    target_epsilon: float,
    delta: float,
    total_rounds: int,
    sample_rate: float,
    steps_per_round: int,
    strategy: str = "back_loaded",
    schedule_power: float = 1.0,
    tolerance: float = 1e-3,
    max_iterations: int = 60,
    use_fast_accounting: bool = False,
) -> list[float]:
    """Calibrate a scheduled set of noises to one composed target epsilon.

    The calibration uses Opacus' RDP analysis and composes every local optimizer
    step. The fast path sums equivalent RDP vectors instead of expanding a long
    Python accountant history. This is materially different from independently
    targeting epsilon in each communication round, which does not yield the
    claimed total privacy budget.
    """
    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be positive.")
    weights = privacy_allocation_weights(total_rounds, strategy, schedule_power)

    compose = compose_rdp_epsilon_fast if use_fast_accounting else compose_rdp_epsilon

    def epsilon_for_base(base_noise: float) -> float:
        return compose(
            _scaled_noise_schedule(base_noise, weights),
            sample_rate,
            steps_per_round,
            delta,
        )

    lower, upper = 1e-3, 1.0
    while True:
        epsilon = epsilon_for_base(upper)
        if epsilon <= target_epsilon:
            break
        upper *= 2.0
        if upper > 1e6:
            raise RuntimeError("Could not calibrate a finite DP noise schedule.")

    for _ in range(max_iterations):
        midpoint = (lower + upper) / 2.0
        epsilon = epsilon_for_base(midpoint)
        if 0.0 <= target_epsilon - epsilon <= tolerance:
            return _scaled_noise_schedule(midpoint, weights)
        if epsilon > target_epsilon:
            lower = midpoint
        else:
            upper = midpoint

    return _scaled_noise_schedule(upper, weights)


@lru_cache(maxsize=2048)
def _calibrate_noise_schedule_cached(
    target_epsilon: float,
    delta: float,
    total_rounds: int,
    sample_rate: float,
    steps_per_round: int,
    strategy: str,
    schedule_power: float,
    tolerance: float,
    max_iterations: int,
    use_fast_accounting: bool,
) -> tuple[float, ...]:
    return tuple(
        _calibrate_noise_schedule_uncached(
            target_epsilon=target_epsilon,
            delta=delta,
            total_rounds=total_rounds,
            sample_rate=sample_rate,
            steps_per_round=steps_per_round,
            strategy=strategy,
            schedule_power=schedule_power,
            tolerance=tolerance,
            max_iterations=max_iterations,
            use_fast_accounting=use_fast_accounting,
        )
    )


def calibrate_noise_schedule(
    target_epsilon: float,
    delta: float,
    total_rounds: int,
    sample_rate: float,
    steps_per_round: int,
    strategy: str = "back_loaded",
    schedule_power: float = 1.0,
    tolerance: float = 1e-3,
    max_iterations: int = 60,
    use_fast_accounting: bool = False,
    use_cache: bool = True,
) -> list[float]:
    """Calibrate and cache an exact RDP noise schedule.

    Cache keys contain every privacy-relevant input. Returning a fresh list
    prevents callers from mutating the cached schedule. Reusing a cached result
    changes neither the schedule nor its composed privacy guarantee.
    """
    arguments = dict(
        target_epsilon=float(target_epsilon),
        delta=float(delta),
        total_rounds=int(total_rounds),
        sample_rate=float(sample_rate),
        steps_per_round=int(steps_per_round),
        strategy=strategy,
        schedule_power=float(schedule_power),
        tolerance=float(tolerance),
        max_iterations=int(max_iterations),
        use_fast_accounting=bool(use_fast_accounting),
    )
    if not use_cache:
        return _calibrate_noise_schedule_uncached(**arguments)
    return list(_calibrate_noise_schedule_cached(**arguments))


def clear_noise_schedule_cache() -> None:
    """Clear cached schedules for profiling and isolated tests."""
    _calibrate_noise_schedule_cached.cache_clear()
