import numpy as np

from src.aggregation import (
    compute_client_label_skews,
    compute_heterogeneity_aware_weights,
    compute_label_distribution_aware_weights,
    compute_noise_aware_representative_weights,
    compute_sample_weights,
    release_positive_rates,
)


def _clients():
    return [
        (np.zeros((4, 2)), np.array([0, 0, 1, 1], dtype=np.float32)),
        (np.zeros((4, 2)), np.array([0, 0, 0, 1], dtype=np.float32)),
    ]


def test_ha_weights_discount_more_skewed_client():
    clients = _clients()
    assert compute_sample_weights(clients) == [4.0, 4.0]
    weights = compute_heterogeneity_aware_weights(clients)
    assert weights[0] > weights[1] > 0


def test_ha_weights_keep_a_fedavg_floor_for_extreme_clients():
    clients = [
        (np.zeros((10, 2)), np.zeros(10, dtype=np.float32)),
        (np.zeros((10, 2)), np.ones(10, dtype=np.float32)),
    ]
    weights = compute_heterogeneity_aware_weights(
        clients,
        ha_strength=0.5,
        prior_strength=0.0,
    )
    assert all(weight >= 5.0 for weight in weights)


def test_zero_ha_strength_recovers_sample_count_weights():
    clients = _clients()
    weights = compute_heterogeneity_aware_weights(clients, ha_strength=0.0)
    assert weights == compute_sample_weights(clients)


def test_private_rate_release_is_bounded_and_reproducible():
    # Deterministic noise is exposed only for explicitly non-private tests.
    first = release_positive_rates(_clients(), metadata_epsilon=0.5, deterministic_noise_seed=7)
    second = release_positive_rates(_clients(), metadata_epsilon=0.5, deterministic_noise_seed=7)
    assert first == second
    assert all(0.0 <= value <= 1.0 for value in first)


def test_private_rate_release_does_not_use_public_experiment_seed_by_default():
    labels = np.array([0, 1] * 500, dtype=np.float32)
    clients = [(np.zeros((1000, 2)), labels)]
    first = release_positive_rates(clients, metadata_epsilon=1.0)
    second = release_positive_rates(clients, metadata_epsilon=1.0)
    assert first != second


def test_label_distribution_aware_weights_reward_global_representativeness():
    clients = [
        (np.zeros((8, 2)), np.array([0, 0, 1, 1, 1, 1, 1, 1], dtype=np.float32)),
        (np.zeros((8, 2)), np.array([1] * 8, dtype=np.float32)),
        (np.zeros((8, 2)), np.array([0, 0, 1, 1, 1, 1, 1, 1], dtype=np.float32)),
    ]
    weights = compute_label_distribution_aware_weights(clients)
    assert weights[0] > weights[1] > 0
    assert weights[0] == weights[2]


def test_label_representative_zero_strength_recovers_fedavg():
    clients = _clients()
    weights = compute_label_distribution_aware_weights(
        clients,
        representativeness_strength=0.0,
        prior_strength=20.0,
    )
    assert weights == compute_sample_weights(clients)


def test_noise_aware_weights_discount_noisier_equal_client():
    clients = _clients()
    weights = compute_noise_aware_representative_weights(
        clients,
        noise_multipliers=[1.0, 2.0],
        representativeness_strength=0.0,
        noise_strength=1.0,
    )
    assert weights[0] > weights[1] > 0


def test_client_skew_scores_are_bounded():
    skews = compute_client_label_skews(_clients(), prior_strength=10.0)
    assert len(skews) == 2
    assert all(0.0 <= value <= 1.0 for value in skews)
