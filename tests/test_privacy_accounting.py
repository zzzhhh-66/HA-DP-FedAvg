import pytest

from src.privacy_accounting import calibrate_noise_schedule, compose_rdp_epsilon
from src.privacy_schedule import get_round_noise_multiplier


def test_calibrated_schedule_composes_to_requested_budget():
    target = 3.0
    schedule = calibrate_noise_schedule(
        target_epsilon=target,
        delta=1e-5,
        total_rounds=4,
        sample_rate=0.1,
        steps_per_round=10,
        strategy="front_loaded",
    )
    composed = compose_rdp_epsilon(schedule, 0.1, 10, 1e-5)
    assert composed <= target + 1e-6
    assert schedule[0] < schedule[-1]


def test_more_rounds_increase_composed_epsilon_for_fixed_noise():
    one = compose_rdp_epsilon([1.0], 0.1, 10, 1e-5)
    two = compose_rdp_epsilon([1.0, 1.0], 0.1, 10, 1e-5)
    assert two > one


def test_noise_schedule_validates_bounds_and_direction():
    assert get_round_noise_multiplier(0, 3, 0.5, 2.0, "decreasing") == 2.0
    assert get_round_noise_multiplier(2, 3, 0.5, 2.0, "decreasing") == 0.5
    with pytest.raises(ValueError):
        get_round_noise_multiplier(3, 3, 0.5, 2.0, "decreasing")
    with pytest.raises(ValueError):
        get_round_noise_multiplier(0, 3, 2.0, 0.5, "decreasing")
