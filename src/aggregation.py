"""Client aggregation strategies for federated learning."""

import math
import secrets

import numpy as np


def compute_sample_weights(clients: list[tuple[np.ndarray, np.ndarray]]) -> list[float]:
    """Standard FedAvg weights: proportional to sample count n_k."""
    return [float(len(y)) for _, y in clients]


def compute_heterogeneity_aware_weights(
    clients: list[tuple[np.ndarray, np.ndarray]],
    gamma: float = 1.0,
    min_balance: float = 0.05,
    positive_rates: list[float] | None = None,
    ha_strength: float = 0.5,
    prior_strength: float = 20.0,
) -> list[float]:
    """Compute bounded, reliability-aware aggregation weights.

    The raw client rate is shrunk toward the global rate before its balance is
    measured. This prevents tiny clients from receiving extreme weights due to
    one or two labels. The HA score is then mixed with standard FedAvg:

        alpha_k = n_k * ((1-lambda) + lambda * balance_k**gamma)

    With ``lambda < 1``, every non-empty client retains a fixed fraction of its
    FedAvg weight. This is important for imbalanced credit data: an all-positive
    client is heterogeneous, but it may also contain valuable minority-class
    signal and should not be almost removed from aggregation.
    """
    if gamma < 0:
        raise ValueError("gamma must be non-negative.")
    if not 0 < min_balance <= 1:
        raise ValueError("min_balance must be in (0, 1].")
    if not 0 <= ha_strength <= 1:
        raise ValueError("ha_strength must be in [0, 1].")
    if prior_strength < 0:
        raise ValueError("prior_strength must be non-negative.")
    if positive_rates is not None and len(positive_rates) != len(clients):
        raise ValueError("positive_rates must have one value per client.")

    total_samples = sum(len(y) for _, y in clients)
    if total_samples <= 0:
        raise ValueError("At least one client sample is required.")
    observed_rates = [
        float(np.mean(y)) if positive_rates is None else float(positive_rates[index])
        for index, (_, y) in enumerate(clients)
    ]
    global_positive_rate = float(
        sum(
            len(y) * np.clip(rate, 0.0, 1.0)
            for (_, y), rate in zip(clients, observed_rates, strict=True)
        )
        / total_samples
    )

    weights: list[float] = []

    for client_id, (_, y) in enumerate(clients):
        n_k = len(y)
        if n_k == 0:
            weights.append(0.0)
            continue

        raw_rate = float(np.clip(observed_rates[client_id], 0.0, 1.0))
        shrunk_rate = (n_k * raw_rate + prior_strength * global_positive_rate) / (
            n_k + prior_strength
        )
        balance = min(shrunk_rate, 1.0 - shrunk_rate) * 2.0
        balance = max(float(balance), min_balance)
        ha_factor = (1.0 - ha_strength) + ha_strength * (balance**gamma)
        weights.append(float(n_k * ha_factor))

    return weights


def compute_label_distribution_aware_weights(
    clients: list[tuple[np.ndarray, np.ndarray]],
    gamma: float = 1.0,
    min_similarity: float = 0.05,
    positive_rates: list[float] | None = None,
    representativeness_strength: float = 1.0,
    prior_strength: float = 0.0,
) -> list[float]:
    """Label-distribution-aware FedAvg control weights.

    This baseline rewards clients whose positive-label prevalence is close to
    the global training prevalence. It is intentionally different from the
    proposed HA rule, which rewards internally balanced clients regardless of
    the global prevalence.
    """
    if gamma < 0:
        raise ValueError("gamma must be non-negative.")
    if not 0 < min_similarity <= 1:
        raise ValueError("min_similarity must be in (0, 1].")
    if not 0 <= representativeness_strength <= 1:
        raise ValueError("representativeness_strength must be in [0, 1].")
    if prior_strength < 0:
        raise ValueError("prior_strength must be non-negative.")
    if positive_rates is not None and len(positive_rates) != len(clients):
        raise ValueError("positive_rates must have one value per client.")

    total_samples = sum(len(y) for _, y in clients)
    if total_samples <= 0:
        raise ValueError("At least one client sample is required.")
    observed_rates = [
        float(np.mean(y)) if positive_rates is None else float(positive_rates[index])
        for index, (_, y) in enumerate(clients)
    ]
    global_positive_rate = float(
        sum(
            len(y) * np.clip(rate, 0.0, 1.0)
            for (_, y), rate in zip(clients, observed_rates, strict=True)
        )
        / total_samples
    )

    weights: list[float] = []
    for client_id, (_, y) in enumerate(clients):
        n_k = len(y)
        if n_k == 0:
            weights.append(0.0)
            continue
        client_positive_rate = float(np.clip(observed_rates[client_id], 0.0, 1.0))
        shrunk_rate = (
            n_k * client_positive_rate + prior_strength * global_positive_rate
        ) / (n_k + prior_strength)
        similarity = 1.0 - abs(shrunk_rate - global_positive_rate)
        similarity = max(similarity, min_similarity)
        factor = (1.0 - representativeness_strength) + representativeness_strength * (
            similarity**gamma
        )
        weights.append(float(n_k * factor))

    return weights


def compute_noise_aware_representative_weights(
    clients: list[tuple[np.ndarray, np.ndarray]],
    noise_multipliers: list[float],
    positive_rates: list[float] | None = None,
    representativeness_strength: float = 0.5,
    discrepancy_scale: float = 2.0,
    noise_strength: float = 0.5,
    prior_strength: float = 20.0,
    min_factor: float = 0.05,
    noise_floor: float = 1e-6,
) -> list[float]:
    """Combine sample size, label representativeness, and DP reliability.

    This is an exploratory v5 candidate, not the published Robust-HDP or
    FedDisco algorithm. The convex floors keep every non-empty client present,
    while inverse-noise reliability discounts updates that contain more DP
    noise in the current round.
    """
    if len(noise_multipliers) != len(clients):
        raise ValueError("noise_multipliers must have one value per client.")
    if positive_rates is not None and len(positive_rates) != len(clients):
        raise ValueError("positive_rates must have one value per client.")
    if not 0 <= representativeness_strength <= 1:
        raise ValueError("representativeness_strength must be in [0, 1].")
    if not 0 <= noise_strength <= 1:
        raise ValueError("noise_strength must be in [0, 1].")
    if discrepancy_scale < 0 or prior_strength < 0 or noise_floor <= 0:
        raise ValueError("scales and prior_strength must be non-negative.")
    if not 0 < min_factor <= 1:
        raise ValueError("min_factor must be in (0, 1].")
    if any(float(noise) <= 0 for noise in noise_multipliers):
        raise ValueError("noise multipliers must be positive.")

    total_samples = sum(len(y) for _, y in clients)
    if total_samples <= 0:
        raise ValueError("At least one client sample is required.")
    observed_rates = [
        float(np.mean(y)) if positive_rates is None else float(positive_rates[index])
        for index, (_, y) in enumerate(clients)
    ]
    global_rate = float(
        sum(
            len(y) * np.clip(rate, 0.0, 1.0)
            for (_, y), rate in zip(clients, observed_rates, strict=True)
        )
        / total_samples
    )
    median_variance = float(np.median(np.square(noise_multipliers)))

    weights: list[float] = []
    for client_id, (_, y) in enumerate(clients):
        n_k = len(y)
        if n_k == 0:
            weights.append(0.0)
            continue
        raw_rate = float(np.clip(observed_rates[client_id], 0.0, 1.0))
        shrunk_rate = (n_k * raw_rate + prior_strength * global_rate) / (
            n_k + prior_strength
        )
        discrepancy = abs(shrunk_rate - global_rate)
        representative = max(math.exp(-discrepancy_scale * discrepancy), min_factor)
        representative_factor = (
            1.0 - representativeness_strength
        ) + representativeness_strength * representative
        relative_reliability = (median_variance + noise_floor) / (
            float(noise_multipliers[client_id]) ** 2 + noise_floor
        )
        relative_reliability = float(
            np.clip(relative_reliability, min_factor, 1.0 / min_factor)
        )
        reliability_factor = (1.0 - noise_strength) + noise_strength * relative_reliability
        weights.append(float(n_k * representative_factor * reliability_factor))
    return weights


def compute_client_label_skews(
    clients: list[tuple[np.ndarray, np.ndarray]],
    positive_rates: list[float] | None = None,
    prior_strength: float = 20.0,
) -> list[float]:
    """Return smoothed label-skew scores in [0, 1] for schedule coupling."""
    if prior_strength < 0:
        raise ValueError("prior_strength must be non-negative.")
    if positive_rates is not None and len(positive_rates) != len(clients):
        raise ValueError("positive_rates must have one value per client.")
    total_samples = sum(len(y) for _, y in clients)
    if total_samples <= 0:
        raise ValueError("At least one client sample is required.")
    rates = [
        float(np.mean(y)) if positive_rates is None else float(positive_rates[index])
        for index, (_, y) in enumerate(clients)
    ]
    global_rate = float(
        sum(len(y) * np.clip(rate, 0.0, 1.0) for (_, y), rate in zip(clients, rates, strict=True))
        / total_samples
    )
    skews = []
    for (_, y), rate in zip(clients, rates, strict=True):
        if len(y) == 0:
            skews.append(0.0)
            continue
        shrunk = (len(y) * np.clip(rate, 0.0, 1.0) + prior_strength * global_rate) / (
            len(y) + prior_strength
        )
        skews.append(float(1.0 - 2.0 * min(shrunk, 1.0 - shrunk)))
    return skews


def release_positive_rates(
    clients: list[tuple[np.ndarray, np.ndarray]],
    metadata_epsilon: float = 0.0,
    deterministic_noise_seed: int | None = None,
) -> list[float]:
    """Release exact or Laplace-private client label rates once per run.

    A zero epsilon encodes the explicit threat-model assumption that client
    label prevalence is public metadata. With epsilon > 0, positive counts are
    sanitized with sensitivity one and the cost must be composed with training.

    ``deterministic_noise_seed`` exists only for explicit non-private research
    tests. Production/private runs must leave it as ``None`` so Laplace noise is
    drawn from operating-system entropy and cannot be reconstructed from the
    public experiment seed.
    """
    if metadata_epsilon < 0:
        raise ValueError("metadata_epsilon cannot be negative.")
    deterministic_rng = (
        np.random.default_rng(deterministic_noise_seed)
        if deterministic_noise_seed is not None
        else None
    )
    secure_rng = secrets.SystemRandom()

    def laplace_noise(scale: float) -> float:
        if deterministic_rng is not None:
            return float(deterministic_rng.laplace(0.0, scale))
        # Inverse-CDF sampling using a cryptographically secure uniform source.
        centered = secure_rng.random() - 0.5
        if centered == 0.0:
            return 0.0
        return -scale * math.copysign(math.log1p(-2.0 * abs(centered)), centered)

    rates = []
    for _, labels in clients:
        if len(labels) == 0:
            rates.append(0.5)
            continue
        positive_count = float(np.sum(labels))
        if metadata_epsilon > 0:
            positive_count += laplace_noise(1.0 / metadata_epsilon)
        rates.append(float(np.clip(positive_count, 0.0, len(labels)) / len(labels)))
    return rates
