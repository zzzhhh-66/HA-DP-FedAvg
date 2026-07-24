"""Privacy budget and noise schedules for DP-FedAvg."""


def allocate_round_epsilon(
    total_epsilon: float,
    round_idx: int,
    total_rounds: int,
    strategy: str = "back_loaded",
) -> float:
    """
    Allocate a per-round epsilon budget from a fixed total privacy budget.

    Strategies:
      - uniform:      each round uses total_epsilon / T
      - front_loaded: early rounds get more budget (smaller noise)
      - back_loaded:  late rounds get more budget (smaller noise later)
    """
    if total_epsilon <= 0:
        raise ValueError("total_epsilon must be positive.")
    if total_rounds <= 0:
        raise ValueError("total_rounds must be positive.")
    if not 0 <= round_idx < total_rounds:
        raise ValueError("round_idx must be in [0, total_rounds).")

    if strategy == "uniform":
        return total_epsilon / total_rounds

    if strategy == "front_loaded":
        weights = [total_rounds - i for i in range(total_rounds)]
    elif strategy == "back_loaded":
        weights = [i + 1 for i in range(total_rounds)]
    else:
        raise ValueError(f"Unknown epsilon allocation strategy: {strategy}")

    weight_sum = float(sum(weights))
    return total_epsilon * weights[round_idx] / weight_sum


def get_round_noise_multiplier(
    round_idx: int,
    total_rounds: int,
    noise_min: float,
    noise_max: float,
    schedule: str = "increasing",
) -> float:
    """
    Direct noise-multiplier schedule (alternative to target-epsilon mode).

    Schedules:
      - uniform:    constant noise_max
      - increasing: noise grows over rounds (more privacy later)
      - decreasing: noise shrinks over rounds (stronger privacy early, more
        accurate updates later)
    """
    if noise_min <= 0 or noise_max <= 0:
        raise ValueError("noise_min and noise_max must be positive.")
    if noise_max < noise_min:
        raise ValueError("noise_max must be greater than or equal to noise_min.")
    if total_rounds <= 0:
        raise ValueError("total_rounds must be positive.")
    if not 0 <= round_idx < total_rounds:
        raise ValueError("round_idx must be in [0, total_rounds).")
    if schedule == "uniform":
        return noise_max

    if total_rounds <= 1:
        return noise_max

    progress = round_idx / (total_rounds - 1)

    if schedule == "increasing":
        return noise_min + progress * (noise_max - noise_min)

    if schedule == "decreasing":
        return noise_max - progress * (noise_max - noise_min)

    raise ValueError(f"Unknown noise schedule: {schedule}")
