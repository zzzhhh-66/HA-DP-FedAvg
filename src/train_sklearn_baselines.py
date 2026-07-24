"""Strong centralized tabular baselines used as non-federated reference points."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .evaluate import compute_metrics, format_metrics, undo_positive_class_weight
from .utils import Timer


def train_sklearn_baseline(
    X_train,
    y_train,
    X_test,
    y_test,
    method: str,
    class_weighting: str = "balanced",
    positive_class_weight: float | None = None,
    seed: int = 42,
) -> dict:
    class_weight = (
        {0: 1.0, 1: float(positive_class_weight)}
        if class_weighting == "balanced" and positive_class_weight is not None
        else None
    )
    if method == "logistic_regression":
        model = LogisticRegression(
            class_weight=class_weight,
            max_iter=2000,
            random_state=seed,
        )
    elif method == "hist_gbdt":
        model = HistGradientBoostingClassifier(
            class_weight=class_weight,
            max_iter=200,
            early_stopping=True,
            random_state=seed,
        )
    elif method == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost baseline requires the optional dependency `xgboost`."
            ) from exc

        scale_pos_weight = (
            float(positive_class_weight)
            if class_weighting == "balanced" and positive_class_weight is not None
            else 1.0
        )
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=seed,
            scale_pos_weight=scale_pos_weight,
        )
    elif method == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError(
                "lightgbm baseline requires the optional dependency `lightgbm`."
            ) from exc

        model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight=class_weight,
            random_state=seed,
            verbose=-1,
        )
    else:
        raise ValueError(f"Unknown sklearn baseline: {method}")

    timer = Timer()
    model.fit(X_train, y_train.astype(int))
    training_time = timer.stop()
    probabilities = np.asarray(model.predict_proba(X_test)[:, 1], dtype=float)
    if class_weighting == "balanced" and positive_class_weight is not None:
        probabilities = undo_positive_class_weight(
            probabilities,
            positive_class_weight,
        )
    metrics = compute_metrics(y_test, probabilities)
    print(f"[{method}] {format_metrics(metrics)} | time={training_time:.2f}s")
    return {
        "method": method,
        "metrics": metrics,
        "training_time": training_time,
        "communication_cost": 0,
        "epsilon": None,
        "delta": None,
        "noise_multiplier": 0.0,
        "max_grad_norm": None,
        "model": model,
    }
