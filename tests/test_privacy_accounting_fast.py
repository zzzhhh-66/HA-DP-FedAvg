import pytest

pytest.importorskip("opacus")

from src.privacy_accounting import (
    calibrate_noise_schedule,
    clear_noise_schedule_cache,
    compose_rdp_epsilon,
    compose_rdp_epsilon_fast,
    compose_rdp_epsilon_prefixes,
    privacy_allocation_weights,
)


@pytest.mark.parametrize("strategy", ["uniform", "back_loaded"])
def test_fast_rdp_composition_matches_original(strategy):
    schedule = calibrate_noise_schedule(
        target_epsilon=3.0,
        delta=1e-5,
        total_rounds=5,
        sample_rate=0.2,
        steps_per_round=5,
        strategy=strategy,
        use_fast_accounting=True,
    )
    original = compose_rdp_epsilon(schedule, 0.2, 5, 1e-5)
    fast = compose_rdp_epsilon_fast(schedule, 0.2, 5, 1e-5)
    assert fast == pytest.approx(original, abs=1e-10)
    assert fast <= 3.001


def test_fast_rdp_rejects_invalid_noise():
    with pytest.raises(ValueError, match="positive"):
        compose_rdp_epsilon_fast([1.0, 0.0], 0.2, 5, 1e-5)


@pytest.mark.parametrize("strategy", ["uniform", "front_loaded", "back_loaded"])
def test_prefix_epsilon_matches_exact_composition(strategy):
    schedule = calibrate_noise_schedule(
        target_epsilon=3.0,
        delta=1e-5,
        total_rounds=5,
        sample_rate=0.2,
        steps_per_round=5,
        strategy=strategy,
        use_fast_accounting=True,
    )
    prefixes = compose_rdp_epsilon_prefixes(schedule, 0.2, 5, 1e-5)
    expected = [compose_rdp_epsilon(schedule[:index], 0.2, 5, 1e-5) for index in range(1, 6)]
    assert prefixes == pytest.approx(expected, abs=1e-10)
    assert prefixes == sorted(prefixes)


def test_schedule_power_changes_shape_but_keeps_normalization():
    linear = privacy_allocation_weights(5, "back_loaded", schedule_power=1.0)
    quadratic = privacy_allocation_weights(5, "back_loaded", schedule_power=2.0)
    assert sum(linear) == pytest.approx(1.0)
    assert sum(quadratic) == pytest.approx(1.0)
    assert quadratic[-1] > linear[-1]


@pytest.mark.parametrize("strategy", ["uniform", "back_loaded"])
def test_cached_calibration_matches_cold_schedule(strategy):
    clear_noise_schedule_cache()
    common = dict(
        target_epsilon=3.0,
        delta=1e-5,
        total_rounds=5,
        sample_rate=0.2,
        steps_per_round=5,
        strategy=strategy,
        use_fast_accounting=False,
    )
    cold = calibrate_noise_schedule(**common, use_cache=False)
    cached_first = calibrate_noise_schedule(**common, use_cache=True)
    cached_second = calibrate_noise_schedule(**common, use_cache=True)
    assert cached_first == cold
    assert cached_second == cold
    assert cached_first is not cached_second
    assert compose_rdp_epsilon(cached_second, 0.2, 5, 1e-5) <= 3.001
