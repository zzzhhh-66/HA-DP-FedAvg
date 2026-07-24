import numpy as np

from src.evaluate import (
    compute_metrics,
    evaluate_probability_outputs,
    select_decision_threshold,
    undo_positive_class_weight,
)


def test_metrics_include_imbalance_sensitive_scores():
    y_true = np.array([0, 0, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_metrics(y_true, y_prob)
    assert metrics["auc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert "mcc" in metrics and "brier" in metrics


def test_inverse_positive_weight_recovers_probability_scale():
    true_probability = np.array([0.1, 0.5, 0.9])
    weight = 4.0
    weighted_score = (
        weight * true_probability / (1.0 - true_probability + weight * true_probability)
    )
    recovered = undo_positive_class_weight(weighted_score, weight)
    assert np.allclose(recovered, true_probability)


def test_threshold_is_selected_from_validation_probabilities():
    y_val = np.array([0, 0, 1, 1])
    val_prob = np.array([0.10, 0.30, 0.40, 0.45])
    threshold, score = select_decision_threshold(y_val, val_prob)
    assert np.isclose(threshold, 0.40)
    assert np.isclose(score, 1.0)


def test_tuned_and_fixed_threshold_metrics_are_both_reported():
    y_val = np.array([0, 0, 1, 1])
    val_prob = np.array([0.10, 0.30, 0.40, 0.45])
    y_test = np.array([0, 0, 1, 1])
    test_prob = np.array([0.05, 0.20, 0.41, 0.48])
    result = evaluate_probability_outputs(y_val, val_prob, y_test, test_prob)
    assert result["metrics"]["f1"] == 1.0
    assert result["fixed_threshold_metrics"]["f1"] == 0.0
    assert result["threshold_strategy"] == "validation_f1"
    assert len(result["validation_prediction_data"]["y_prob"]) == len(y_val)
