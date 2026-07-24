from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@torch.no_grad()
def predict_scores(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
    logits = model(X_tensor)
    probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
    return probs


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = y_true.astype(int)
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1).")
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "threshold": float(threshold),
    }

    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc"] = float("nan")

    try:
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics["auprc"] = float("nan")

    return metrics


def select_decision_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    strategy: str = "f1",
    fallback_threshold: float = 0.5,
) -> tuple[float, float]:
    """Select a classification threshold using validation data only.

    Ranking metrics do not depend on a threshold, but F1 and recall do. The
    chosen threshold must therefore come from the validation set rather than
    the test set. Ties are resolved in favor of higher precision and then the
    threshold closest to 0.5, which avoids unstable extreme thresholds.
    """
    if strategy != "f1":
        raise ValueError(f"Unknown threshold strategy: {strategy}")
    if not 0 < fallback_threshold < 1:
        raise ValueError("fallback_threshold must be in (0, 1).")

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError("y_true and y_prob must have the same length.")
    if y_true.size == 0:
        return float(fallback_threshold), 0.0
    if np.unique(y_true).size < 2:
        fallback_metrics = compute_metrics(y_true, y_prob, fallback_threshold)
        return float(fallback_threshold), float(fallback_metrics["f1"])

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if thresholds.size == 0:
        fallback_metrics = compute_metrics(y_true, y_prob, fallback_threshold)
        return float(fallback_threshold), float(fallback_metrics["f1"])

    denominator = precision[:-1] + recall[:-1]
    scores = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_score = float(np.max(scores))
    candidates = np.flatnonzero(np.isclose(scores, best_score, rtol=1e-10, atol=1e-12))
    best_precision = float(np.max(precision[candidates]))
    candidates = candidates[
        np.isclose(precision[candidates], best_precision, rtol=1e-10, atol=1e-12)
    ]
    best_idx = int(candidates[np.argmin(np.abs(thresholds[candidates] - fallback_threshold))])
    selected = float(np.clip(thresholds[best_idx], 1e-7, 1.0 - 1e-7))
    return selected, best_score


def undo_positive_class_weight(y_score: np.ndarray, pos_weight: float | None) -> np.ndarray:
    """Map weighted-BCE sigmoid scores back to the unweighted probability scale.

    For positive weight ``a``, weighted BCE targets
    ``q = a*p / (1-p+a*p)``. This inverse prevents Brier score and a 0.5
    threshold from being interpreted on the distorted cost-sensitive scale.
    It is an analytical prior correction, not a substitute for validation-set
    calibration when calibrated probabilities are a primary paper claim.
    """
    if pos_weight is None or float(pos_weight) == 1.0:
        return np.asarray(y_score, dtype=float)
    if pos_weight <= 0:
        raise ValueError("pos_weight must be positive.")
    score = np.clip(np.asarray(y_score, dtype=float), 1e-7, 1.0 - 1e-7)
    denominator = float(pos_weight) + (1.0 - float(pos_weight)) * score
    return np.clip(score / denominator, 0.0, 1.0)


def evaluate_probability_outputs(
    y_val: np.ndarray,
    val_prob: np.ndarray,
    y_test: np.ndarray,
    test_prob: np.ndarray,
    threshold_strategy: str = "f1",
) -> dict[str, Any]:
    """Evaluate test probabilities with a validation-selected threshold."""
    threshold, validation_score = select_decision_threshold(
        y_val,
        val_prob,
        strategy=threshold_strategy,
    )
    metrics = compute_metrics(y_test, test_prob, threshold=threshold)
    validation_metrics = compute_metrics(y_val, val_prob, threshold=threshold)
    fixed_threshold_metrics = compute_metrics(y_test, test_prob, threshold=0.5)
    return {
        "metrics": metrics,
        "validation_metrics": validation_metrics,
        "fixed_threshold_metrics": fixed_threshold_metrics,
        "selected_threshold": threshold,
        "threshold_strategy": f"validation_{threshold_strategy}",
        "validation_selection_score": validation_score,
        "prediction_data": {
            "y_true": np.asarray(y_test, dtype=int),
            "y_prob": np.asarray(test_prob, dtype=float),
            "y_pred": (np.asarray(test_prob) >= threshold).astype(int),
            "y_pred_fixed_0_5": (np.asarray(test_prob) >= 0.5).astype(int),
        },
        "validation_prediction_data": {
            "y_true": np.asarray(y_val, dtype=int),
            "y_prob": np.asarray(val_prob, dtype=float),
            "y_pred": (np.asarray(val_prob) >= threshold).astype(int),
            "y_pred_fixed_0_5": (np.asarray(val_prob) >= 0.5).astype(int),
        },
    }


def evaluate_model(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
    pos_weight: float | None = None,
) -> dict[str, float]:
    y_prob = undo_positive_class_weight(predict_scores(model, X_test, device), pos_weight)
    return compute_metrics(y_test, y_prob, threshold=threshold)


def evaluate_client_slices(
    model: nn.Module,
    clients: list[tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    threshold: float,
    pos_weight: float | None = None,
) -> dict[str, Any]:
    """Evaluate a global model on held-out client-profile slices."""
    per_client: list[dict[str, float]] = []
    sizes: list[int] = []
    positive_rates: list[float] = []
    for X_client, y_client in clients:
        if len(y_client) == 0:
            continue
        metrics = evaluate_model(
            model,
            X_client,
            y_client,
            device,
            threshold=threshold,
            pos_weight=pos_weight,
        )
        per_client.append(metrics)
        sizes.append(int(len(y_client)))
        positive_rates.append(float(np.mean(y_client)))

    summary: dict[str, Any] = {
        "client_eval_num_slices": len(per_client),
        "client_eval_sizes": sizes,
        "client_eval_positive_rates": positive_rates,
        "client_eval_metrics": per_client,
    }
    for metric in ("auc", "auprc", "f1", "recall", "balanced_accuracy", "brier"):
        values = np.asarray([row[metric] for row in per_client], dtype=float)
        finite = values[np.isfinite(values)]
        prefix = f"client_{metric}"
        summary[f"{prefix}_valid"] = int(finite.size)
        if finite.size == 0:
            summary[f"{prefix}_mean"] = float("nan")
            summary[f"{prefix}_std"] = float("nan")
            summary[f"{prefix}_min"] = float("nan")
            summary[f"{prefix}_p10"] = float("nan")
            continue
        summary[f"{prefix}_mean"] = float(np.mean(finite))
        summary[f"{prefix}_std"] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        summary[f"{prefix}_min"] = float(np.min(finite))
        summary[f"{prefix}_p10"] = float(np.percentile(finite, 10))
    return summary


def aggregate_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    keys = metric_dicts[0].keys()
    return {key: float(np.mean([metrics[key] for metrics in metric_dicts])) for key in keys}


def aggregate_metric_std(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    keys = metric_dicts[0].keys()
    return {key: float(np.std([metrics[key] for metrics in metric_dicts])) for key in keys}


def format_metrics(metrics: dict[str, Any]) -> str:
    return (
        f"acc={metrics.get('accuracy', 0):.4f}, "
        f"prec={metrics.get('precision', 0):.4f}, "
        f"rec={metrics.get('recall', 0):.4f}, "
        f"f1={metrics.get('f1', 0):.4f}, "
        f"auc={metrics.get('auc', 0):.4f}, "
        f"auprc={metrics.get('auprc', 0):.4f}"
    )
