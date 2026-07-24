import math

from src.experiment_runner import aggregate_results


def _row(seed: int, realized_noise: float, auc: float) -> dict:
    return {
        "dataset": "demo",
        "split": "noniid_alpha=0.5",
        "alpha": 0.5,
        "method": "ha_dp_fedavg",
        "configured_noise_multiplier": "",
        "noise_min": "",
        "noise_max": "",
        "privacy_mode": "epsilon_allocation",
        "epsilon_strategy": "back_loaded",
        "total_epsilon": 3.0,
        "use_heterogeneity_aware": True,
        "class_weighting": "balanced",
        "prox_mu": "",
        "metadata_epsilon": 0.1,
        "balance_gamma": 1.0,
        "noise_multiplier": realized_noise,
        "epsilon": 3.0,
        "accuracy": auc,
        "balanced_accuracy": auc,
        "precision": auc,
        "recall": auc,
        "specificity": auc,
        "f1": auc,
        "mcc": auc,
        "auc": auc,
        "auprc": auc,
        "brier": 1.0 - auc,
        "training_time": 1.0,
        "communication_cost": 1.0,
        "seed": seed,
    }


def test_target_epsilon_runs_group_despite_different_realized_noise():
    summary = aggregate_results([_row(1, 1.2, 0.7), _row(2, 1.4, 0.8), _row(3, 1.6, 0.9)])
    assert len(summary) == 1
    assert summary[0]["num_runs"] == 3
    # t(0.975, 2)=4.30265, not the large-sample 1.96 multiplier.
    expected = 4.302652729696142 * 0.1 / math.sqrt(3)
    assert math.isclose(summary[0]["auc_ci95"], expected, rel_tol=1e-6)
