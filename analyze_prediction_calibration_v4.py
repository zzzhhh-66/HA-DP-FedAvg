"""Audit validation-selected thresholds and optional Platt calibration.

This script never uses test labels to select a threshold or fit a calibrator.
It consumes the per-run prediction CSV files written by the v3 experiment core.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

LABEL_COLUMNS = ("y_true", "label", "target", "ground_truth")
PROBABILITY_COLUMNS = ("y_prob", "probability", "score", "positive_probability")
SPLIT_COLUMNS = ("evaluation_split", "split", "subset", "partition")


def find_column(frame: pd.DataFrame, candidates: Iterable[str], kind: str) -> str:
    lowered = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError(f"Prediction CSV has no recognized {kind} column. Columns={list(frame.columns)}")


def select_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if thresholds.size == 0:
        return 0.5
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(thresholds[int(np.nanargmax(f1))])


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(y_prob, edges[1:-1]), bins - 1)
    result = 0.0
    for bin_id in range(bins):
        mask = assignments == bin_id
        if not np.any(mask):
            continue
        result += float(np.mean(mask)) * abs(float(np.mean(y_prob[mask])) - float(np.mean(y_true[mask])))
    return result


def metric_row(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece_10": expected_calibration_error(y_true, y_prob, bins=10),
    }


def fit_platt(y_val: np.ndarray, p_val: np.ndarray, p_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if np.unique(y_val).size < 2:
        raise ValueError("Validation labels contain only one class; Platt calibration is undefined.")
    clipped_val = np.clip(p_val, 1e-7, 1.0 - 1e-7)
    clipped_test = np.clip(p_test, 1e-7, 1.0 - 1e-7)
    val_logit = np.log(clipped_val / (1.0 - clipped_val)).reshape(-1, 1)
    test_logit = np.log(clipped_test / (1.0 - clipped_test)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    calibrator.fit(val_logit, y_val)
    return calibrator.predict_proba(val_logit)[:, 1], calibrator.predict_proba(test_logit)[:, 1]


def split_predictions(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    label_col = find_column(frame, LABEL_COLUMNS, "label")
    probability_col = find_column(frame, PROBABILITY_COLUMNS, "probability")
    split_col = find_column(frame, SPLIT_COLUMNS, "split")
    normalized = frame[split_col].astype(str).str.lower()
    val_mask = normalized.isin({"val", "valid", "validation"})
    test_mask = normalized.isin({"test", "testing"})
    if not val_mask.any() or not test_mask.any():
        raise ValueError(f"Prediction CSV needs validation and test rows. Values={sorted(normalized.unique())}")
    return (
        frame.loc[val_mask, label_col].to_numpy(dtype=int),
        frame.loc[val_mask, probability_col].to_numpy(dtype=float),
        frame.loc[test_mask, label_col].to_numpy(dtype=int),
        frame.loc[test_mask, probability_col].to_numpy(dtype=float),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/tables/calibration_v4_report.csv"))
    args = parser.parse_args()
    raw = pd.read_csv(args.raw_results)
    if "prediction_path" not in raw:
        raise ValueError("The result table has no prediction_path column.")

    reports: list[dict[str, object]] = []
    for index, experiment in raw.iterrows():
        prediction_path = Path(str(experiment["prediction_path"]))
        if not prediction_path.exists():
            print(f"[Skip] missing prediction file: {prediction_path}", flush=True)
            continue
        try:
            y_val, p_val, y_test, p_test = split_predictions(pd.read_csv(prediction_path))
            raw_threshold = select_f1_threshold(y_val, p_val)
            calibrated_val, calibrated_test = fit_platt(y_val, p_val, p_test)
            calibrated_threshold = select_f1_threshold(y_val, calibrated_val)
        except (ValueError, OSError) as error:
            print(f"[Skip] {prediction_path}: {error}", flush=True)
            continue

        identity = {
            "result_row": int(index),
            "dataset": experiment.get("dataset", ""),
            "method": experiment.get("curve_method", experiment.get("method", "")),
            "alpha": experiment.get("alpha", ""),
            "seed": experiment.get("seed", ""),
            "target_epsilon": experiment.get("target_epsilon", experiment.get("total_epsilon", "")),
            "prediction_path": str(prediction_path),
            "selection_data": "validation_only",
        }
        variants = (
            ("raw_fixed_0p5", p_test, 0.5),
            ("raw_validation_f1", p_test, raw_threshold),
            ("platt_fixed_0p5", calibrated_test, 0.5),
            ("platt_validation_f1", calibrated_test, calibrated_threshold),
        )
        for variant, probability, threshold in variants:
            reports.append({**identity, "variant": variant, **metric_row(y_test, probability, threshold)})
        print(f"[Processed] {prediction_path}", flush=True)

    if not reports:
        raise RuntimeError("No prediction files were processed.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(reports).to_csv(args.output, index=False)
    print(f"[Complete] {len(reports)} rows -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
