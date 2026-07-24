import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from .config import DATASET_CONFIG
from .evaluate import (
    evaluate_client_slices,
    evaluate_probability_outputs,
    format_metrics,
    predict_scores,
    undo_positive_class_weight,
)
from .preprocess import (
    preprocess_default_credit_card_clients,
    preprocess_german_credit_numeric,
    preprocess_give_me_some_credit,
    print_dataset_summary,
)
from .split_clients import (
    create_iid_clients,
    create_noniid_clients_dirichlet,
    create_profile_matched_test_clients,
    print_client_distribution,
)
from .utils import (
    PREDICTIONS_DIR,
    ensure_result_dirs,
    flatten_result,
    get_device,
    resolve_data_path,
    runtime_metadata,
    set_seed,
)


def load_dataset(dataset: str, seed: int):
    data_path = resolve_data_path(dataset)
    if dataset == "gmsc":
        return preprocess_give_me_some_credit(str(data_path), random_state=seed)
    if dataset == "german":
        return preprocess_german_credit_numeric(str(data_path), random_state=seed)
    if dataset == "default_credit":
        return preprocess_default_credit_card_clients(str(data_path), random_state=seed)
    raise ValueError(f"Unknown dataset: {dataset}")


def build_clients(
    X_train,
    y_train,
    num_clients: int,
    split: str,
    alpha: float | None,
    seed: int,
):
    if split == "iid":
        return create_iid_clients(X_train, y_train, num_clients=num_clients, random_state=seed)

    if split == "noniid":
        if alpha is None:
            raise ValueError("Non-IID split requires alpha.")
        return create_noniid_clients_dirichlet(
            X_train,
            y_train,
            num_clients=num_clients,
            alpha=alpha,
            random_state=seed,
        )

    raise ValueError(f"Unknown split type: {split}")


def _predict_probabilities(
    model: Any,
    X: np.ndarray,
    pos_weight: float | None,
) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    else:
        probabilities = predict_scores(model, X, get_device())
    return undo_positive_class_weight(probabilities, pos_weight)


def _safe_filename_component(value: Any) -> str:
    return str(value).replace(".", "p").replace("=", "-").replace("/", "-")


def save_prediction_artifact(
    prediction_data: dict[str, np.ndarray],
    validation_prediction_data: dict[str, np.ndarray],
    *,
    dataset: str,
    method: str,
    split: str,
    seed: int,
    run_tag: str,
) -> Path:
    ensure_result_dirs()
    filename = "_".join(
        _safe_filename_component(value)
        for value in (dataset, method, split, run_tag, f"seed{seed}")
        if value not in ("", None)
    )
    output_path = PREDICTIONS_DIR / f"{filename}.csv"
    fieldnames = [
        "evaluation_split",
        "sample_index",
        "y_true",
        "y_prob",
        "y_pred",
        "y_pred_fixed_0_5",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for evaluation_split, data in (
            ("validation", validation_prediction_data),
            ("test", prediction_data),
        ):
            for index in range(len(data["y_true"])):
                writer.writerow(
                    {
                        "evaluation_split": evaluation_split,
                        "sample_index": index,
                        "y_true": int(data["y_true"][index]),
                        "y_prob": float(data["y_prob"][index]),
                        "y_pred": int(data["y_pred"][index]),
                        "y_pred_fixed_0_5": int(data["y_pred_fixed_0_5"][index]),
                    }
                )
    return output_path


def run_single_experiment(
    dataset: str,
    method: str,
    split: str = "iid",
    alpha: float | None = None,
    noise_multiplier: float = 0.0,
    seed: int = 42,
    verbose: bool = True,
    global_rounds_override: int | None = None,
    centralized_epochs_override: int | None = None,
    privacy_mode: str = "epsilon_allocation",
    total_epsilon: float = 3.0,
    epsilon_strategy: str = "back_loaded",
    noise_schedule: str = "decreasing",
    noise_min: float = 0.5,
    noise_max: float = 2.0,
    use_heterogeneity_aware: bool | None = None,
    balance_gamma: float = 1.0,
    ha_strength: float = 0.5,
    ha_prior_strength: float = 20.0,
    class_weighting: str = "balanced",
    prox_mu: float = 0.01,
    metadata_epsilon: float = 0.0,
    secure_rng: bool = False,
    dp_target_epsilon: float | None = None,
    ablation_id: str | None = None,
    ablation_description: str | None = None,
    threshold_strategy: str = "f1",
    aggregation_strategy: str | None = None,
    representativeness_strength: float = 0.5,
    discrepancy_scale: float = 2.0,
    noise_strength: float = 0.5,
    schedule_power: float = 1.0,
    schedule_ha_coupling: float = 0.0,
    fast_epsilon_tracking: bool = True,
    evaluate_client_profiles: bool = False,
) -> dict[str, Any]:
    if dataset not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset}")
    config = DATASET_CONFIG[dataset]
    if use_heterogeneity_aware is None:
        use_heterogeneity_aware = method == "ha_dp_fedavg"
    global_rounds = (
        config["global_rounds"] if global_rounds_override is None else global_rounds_override
    )
    centralized_epochs = (
        config["centralized_epochs"]
        if centralized_epochs_override is None
        else centralized_epochs_override
    )
    if global_rounds <= 0 or centralized_epochs <= 0:
        raise ValueError("Training epochs and communication rounds must be positive.")
    set_seed(seed)

    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_dataset(dataset, seed=seed)
    input_dim = X_train.shape[1]

    if verbose:
        print_dataset_summary(
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            config["name"],
        )

    pos_weight = config["positive_class_weight"] if class_weighting == "balanced" else None
    clients = None
    federated_methods = {
        "local_only",
        "fedavg",
        "fedprox",
        "label_aware_fedavg",
        "dp_fedavg",
        "ha_dp_fedavg",
    }
    if method in federated_methods:
        clients = build_clients(
            X_train,
            y_train,
            num_clients=config["num_clients"],
            split=split,
            alpha=alpha,
            seed=seed,
        )
        if verbose:
            split_name = split if split == "iid" else f"noniid_alpha={alpha}"
            print_client_distribution(clients, f"{config['name']} - {split_name}")

    common_kwargs = {
        "input_dim": input_dim,
        "batch_size": config["batch_size"],
        "learning_rate": 1e-3,
        "pos_weight": pos_weight,
        "seed": seed,
    }

    if method == "centralized":
        from .train_centralized import train_centralized

        result = train_centralized(
            X_train,
            y_train,
            X_test,
            y_test,
            epochs=centralized_epochs,
            X_val=X_val,
            y_val=y_val,
            **common_kwargs,
        )
    elif method in {"logistic_regression", "hist_gbdt", "xgboost", "lightgbm"}:
        from .train_sklearn_baselines import train_sklearn_baseline

        result = train_sklearn_baseline(
            X_train,
            y_train,
            X_test,
            y_test,
            method=method,
            class_weighting=class_weighting,
            positive_class_weight=pos_weight,
            seed=seed,
        )
    elif method == "local_only":
        from .train_local import train_local_only

        result = train_local_only(
            clients,
            X_test,
            y_test,
            epochs=centralized_epochs,
            **common_kwargs,
        )
    elif method == "fedavg":
        from .train_fedavg import train_fedavg

        result = train_fedavg(
            clients,
            X_test,
            y_test,
            global_rounds=global_rounds,
            local_epochs=config["local_epochs"],
            X_val=X_val,
            y_val=y_val,
            **common_kwargs,
        )
    elif method == "fedprox":
        from .train_fedprox import train_fedprox

        result = train_fedprox(
            clients,
            X_test,
            y_test,
            global_rounds=global_rounds,
            local_epochs=config["local_epochs"],
            prox_mu=prox_mu,
            X_val=X_val,
            y_val=y_val,
            **common_kwargs,
        )
    elif method == "label_aware_fedavg":
        from .train_label_aware_fedavg import train_label_aware_fedavg

        result = train_label_aware_fedavg(
            clients,
            X_test,
            y_test,
            global_rounds=global_rounds,
            local_epochs=config["local_epochs"],
            label_gamma=balance_gamma,
            X_val=X_val,
            y_val=y_val,
            **common_kwargs,
        )
    elif method == "dp_fedavg":
        from .train_dp_fedavg import train_dp_fedavg

        result = train_dp_fedavg(
            clients,
            X_test,
            y_test,
            global_rounds=global_rounds,
            local_epochs=config["local_epochs"],
            noise_multiplier=noise_multiplier,
            max_grad_norm=1.0,
            use_heterogeneity_aware=use_heterogeneity_aware,
            balance_gamma=balance_gamma,
            ha_strength=ha_strength,
            ha_prior_strength=ha_prior_strength,
            metadata_epsilon=metadata_epsilon,
            secure_rng=secure_rng,
            target_epsilon=dp_target_epsilon,
            X_val=X_val,
            y_val=y_val,
            fast_epsilon_tracking=fast_epsilon_tracking,
            **common_kwargs,
        )
    elif method == "ha_dp_fedavg":
        from .train_ha_dp_fedavg import train_ha_dp_fedavg

        result = train_ha_dp_fedavg(
            clients,
            X_test,
            y_test,
            global_rounds=global_rounds,
            local_epochs=config["local_epochs"],
            max_grad_norm=1.0,
            privacy_mode=privacy_mode,
            total_epsilon=total_epsilon,
            epsilon_strategy=epsilon_strategy,
            noise_min=noise_min,
            noise_max=noise_max,
            noise_schedule=noise_schedule,
            use_heterogeneity_aware=use_heterogeneity_aware,
            balance_gamma=balance_gamma,
            ha_strength=ha_strength,
            ha_prior_strength=ha_prior_strength,
            metadata_epsilon=metadata_epsilon,
            secure_rng=secure_rng,
            X_val=X_val,
            y_val=y_val,
            aggregation_strategy=aggregation_strategy,
            representativeness_strength=representativeness_strength,
            discrepancy_scale=discrepancy_scale,
            noise_strength=noise_strength,
            schedule_power=schedule_power,
            schedule_ha_coupling=schedule_ha_coupling,
            fast_epsilon_tracking=fast_epsilon_tracking,
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    fixed_threshold_metrics: dict[str, float] = {}
    client_profile_metrics: dict[str, Any] = {}
    prediction_path = ""
    if result.get("model") is not None:
        probability_weight = pos_weight

        val_prob = _predict_probabilities(result["model"], X_val, probability_weight)
        test_prob = _predict_probabilities(result["model"], X_test, probability_weight)
        selected = evaluate_probability_outputs(
            y_val,
            val_prob,
            y_test,
            test_prob,
            threshold_strategy=threshold_strategy,
        )
        result["metrics"] = selected["metrics"]
        result["threshold_validation_metrics"] = selected["validation_metrics"]
        result["selected_threshold"] = selected["selected_threshold"]
        result["threshold_strategy"] = selected["threshold_strategy"]
        result["validation_selection_score"] = selected["validation_selection_score"]
        fixed_threshold_metrics = selected["fixed_threshold_metrics"]

        if evaluate_client_profiles and clients is not None:
            test_clients = create_profile_matched_test_clients(
                X_test,
                y_test,
                clients,
                random_state=seed + 100_003,
            )
            client_profile_metrics = evaluate_client_slices(
                result["model"],
                test_clients,
                get_device(),
                threshold=float(selected["selected_threshold"]),
                pos_weight=probability_weight,
            )

        if method == "dp_fedavg":
            run_tag = (
                f"eps{dp_target_epsilon}"
                if dp_target_epsilon is not None
                else f"noise{noise_multiplier}"
            )
        elif method == "ha_dp_fedavg":
            run_tag = f"eps{total_epsilon}_{epsilon_strategy}"
        else:
            run_tag = ablation_id or "standard"
        if ablation_id:
            run_tag = f"{ablation_id}_{run_tag}"
        split_tag = split if split == "iid" else f"noniid_alpha{alpha}"
        prediction_path = str(
            save_prediction_artifact(
                selected["prediction_data"],
                selected["validation_prediction_data"],
                dataset=dataset,
                method=method,
                split=split_tag,
                seed=seed,
                run_tag=run_tag,
            )
        )
        print(
            f"[Threshold] method={method} selected={selected['selected_threshold']:.4f} "
            f"val_f1={selected['validation_selection_score']:.4f} "
            f"test_f1={selected['metrics']['f1']:.4f}",
            flush=True,
        )

    experiment_result = {
        "dataset": config["name"],
        "split": split if split == "iid" else f"noniid_alpha={alpha}",
        "alpha": alpha if split == "noniid" else None,
        "method": method,
        "ablation_id": ablation_id or "",
        "ablation_description": ablation_description or "",
        "use_heterogeneity_aware": use_heterogeneity_aware
        if method in {"dp_fedavg", "ha_dp_fedavg"}
        else "",
        "class_weighting": class_weighting,
        "positive_class_weight": pos_weight,
        "class_weight_source": "fixed_public_protocol" if pos_weight is not None else "none",
        "probability_correction": "inverse_positive_class_weight"
        if pos_weight is not None
        else "none",
        "preprocessing_scope": "public_benchmark_global",
        "prox_mu": prox_mu if method == "fedprox" else "",
        "metadata_epsilon": result.get("metadata_epsilon", 0.0),
        "privacy_mode": result.get("privacy_mode", ""),
        "epsilon_strategy": epsilon_strategy if method == "ha_dp_fedavg" else "",
        "aggregation_strategy": result.get("aggregation_strategy", ""),
        "representativeness_strength": result.get("representativeness_strength", ""),
        "discrepancy_scale": result.get("discrepancy_scale", ""),
        "noise_strength": result.get("noise_strength", ""),
        "schedule_power": result.get("schedule_power", ""),
        "schedule_ha_coupling": result.get("schedule_ha_coupling", ""),
        "client_schedule_powers": result.get("client_schedule_powers", ""),
        "epsilon_tracking": result.get("epsilon_tracking", ""),
        "total_epsilon": result.get("total_epsilon", ""),
        "noise_multiplier": (
            result.get("noise_multiplier", noise_multiplier)
            if method in {"dp_fedavg", "ha_dp_fedavg"}
            else 0.0
        ),
        "configured_noise_multiplier": (
            noise_multiplier if method == "dp_fedavg" and dp_target_epsilon is None else ""
        ),
        "noise_min": noise_min
        if method == "ha_dp_fedavg" and privacy_mode == "noise_schedule"
        else "",
        "noise_max": noise_max
        if method == "ha_dp_fedavg" and privacy_mode == "noise_schedule"
        else "",
        "balance_gamma": balance_gamma
        if (
            method == "label_aware_fedavg"
            or (method in {"dp_fedavg", "ha_dp_fedavg"} and use_heterogeneity_aware)
        )
        else "",
        "ha_strength": result.get("ha_strength", ""),
        "ha_prior_strength": result.get("ha_prior_strength", ""),
        "best_round": result.get("best_round", ""),
        "round_metrics": result.get("round_metrics", ""),
        "epsilon_history": result.get("epsilon_history", ""),
        "noise_history": result.get("noise_history", ""),
        "model_selection_metric": result.get("model_selection_metric", "final_model"),
        "threshold_strategy": result.get("threshold_strategy", "fixed_0.5"),
        "validation_selection_score": result.get("validation_selection_score", ""),
        "prediction_path": prediction_path,
        "fixed_0_5_precision": fixed_threshold_metrics.get("precision", ""),
        "fixed_0_5_recall": fixed_threshold_metrics.get("recall", ""),
        "fixed_0_5_f1": fixed_threshold_metrics.get("f1", ""),
        "fixed_0_5_balanced_accuracy": fixed_threshold_metrics.get("balanced_accuracy", ""),
        "epsilon": result.get("epsilon"),
        "delta": result.get("delta"),
        "max_grad_norm": result.get("max_grad_norm"),
        "privacy_unit": result.get("privacy_unit", ""),
        "accountant": result.get("accountant", ""),
        "secure_rng": result.get("secure_rng", ""),
        "privacy_claim_valid": result.get("privacy_claim_valid", ""),
        "adjacency": result.get("adjacency", ""),
        "client_sample_counts": result.get("client_sample_counts", ""),
        "metadata_rng": result.get("metadata_rng", ""),
        "client_sizes": result.get(
            "client_sizes",
            [int(len(client_y)) for _, client_y in clients] if clients is not None else "",
        ),
        "client_label_rates": (
            [float(np.mean(client_y)) for _, client_y in clients]
            if clients is not None
            and method in {"local_only", "fedavg", "fedprox", "label_aware_fedavg"}
            else result.get("released_positive_rates", "")
        ),
        "released_positive_rates": result.get("released_positive_rates", ""),
        "aggregation_weights": result.get("aggregation_weights", ""),
        "client_evaluation_protocol": (
            "held_out_label_profile_matched_slices"
            if client_profile_metrics
            else "not_run"
        ),
        **client_profile_metrics,
        "seed": seed,
        **result["metrics"],
        "training_time": result["training_time"],
        "communication_cost": result["communication_cost"],
        "num_features": len(feature_names),
        "num_clients": config["num_clients"] if clients is not None else None,
        **runtime_metadata(),
    }

    if verbose:
        print(
            f"[Done] dataset={experiment_result['dataset']} method={method} "
            f"split={experiment_result['split']} seed={seed} "
            f"{format_metrics(result['metrics'])}"
        )

    return experiment_result


def save_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    ensure_result_dirs()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    flattened_rows = [flatten_result(row) for row in rows]
    fieldnames = sorted({key for row in flattened_rows for key in row.keys()})

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in flattened_rows:
            writer.writerow(row)


def _numeric_metric_values(
    rows: list[dict[str, Any]],
    metric: str,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(metric)
        if value in (None, ""):
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isnan(numeric_value):
            values.append(numeric_value)
    return values


def aggregate_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []

    group_keys = [
        "dataset",
        "split",
        "alpha",
        "method",
        "configured_noise_multiplier",
        "noise_min",
        "noise_max",
        "privacy_mode",
        "epsilon_strategy",
        "total_epsilon",
        "use_heterogeneity_aware",
        "class_weighting",
        "prox_mu",
        "metadata_epsilon",
        "balance_gamma",
        "ha_strength",
        "ha_prior_strength",
    ]
    grouped: dict[tuple, list[dict[str, Any]]] = {}

    for row in rows:
        key = tuple(row.get(k, "") for k in group_keys)
        grouped.setdefault(key, []).append(row)

    summary = []
    metric_keys = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "mcc",
        "auc",
        "auprc",
        "brier",
        "training_time",
        "communication_cost",
        "threshold",
        "fixed_0_5_precision",
        "fixed_0_5_recall",
        "fixed_0_5_f1",
        "fixed_0_5_balanced_accuracy",
        "best_round",
    ]

    for key, group_rows in grouped.items():
        summary_row = {name: value for name, value in zip(group_keys, key, strict=True)}
        summary_row["num_runs"] = len(group_rows)

        for metric in metric_keys:
            values = _numeric_metric_values(group_rows, metric)
            if values:
                summary_row[f"{metric}_mean"] = float(np.mean(values))
                summary_row[f"{metric}_std"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                )
                summary_row[f"{metric}_ci95"] = (
                    float(student_t.ppf(0.975, df=len(values) - 1))
                    * summary_row[f"{metric}_std"]
                    / math.sqrt(len(values))
                    if len(values) > 1
                    else float("nan")
                )

        eps_values = [row["epsilon"] for row in group_rows if row["epsilon"] is not None]
        if eps_values:
            summary_row["epsilon_mean"] = float(np.mean(eps_values))
            summary_row["epsilon_std"] = (
                float(np.std(eps_values, ddof=1)) if len(eps_values) > 1 else 0.0
            )

        summary.append(summary_row)

    return summary


def aggregate_ablation_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []

    group_keys = [
        "dataset",
        "split",
        "alpha",
        "ablation_id",
        "method",
        "use_heterogeneity_aware",
        "privacy_mode",
        "epsilon_strategy",
        "total_epsilon",
        "configured_noise_multiplier",
        "noise_min",
        "noise_max",
        "class_weighting",
        "metadata_epsilon",
        "balance_gamma",
        "ha_strength",
        "ha_prior_strength",
    ]
    grouped: dict[tuple, list[dict[str, Any]]] = {}

    for row in rows:
        key = tuple(row.get(k, "") for k in group_keys)
        grouped.setdefault(key, []).append(row)

    summary = []
    metric_keys = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "mcc",
        "auc",
        "auprc",
        "brier",
        "training_time",
        "communication_cost",
        "threshold",
        "fixed_0_5_precision",
        "fixed_0_5_recall",
        "fixed_0_5_f1",
        "fixed_0_5_balanced_accuracy",
        "best_round",
    ]

    for key, group_rows in grouped.items():
        summary_row = {name: value for name, value in zip(group_keys, key, strict=True)}
        summary_row["ablation_description"] = group_rows[0].get("ablation_description", "")
        summary_row["num_runs"] = len(group_rows)

        for metric in metric_keys:
            values = _numeric_metric_values(group_rows, metric)
            if values:
                summary_row[f"{metric}_mean"] = float(np.mean(values))
                summary_row[f"{metric}_std"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                )
                summary_row[f"{metric}_ci95"] = (
                    float(student_t.ppf(0.975, df=len(values) - 1))
                    * summary_row[f"{metric}_std"]
                    / math.sqrt(len(values))
                    if len(values) > 1
                    else float("nan")
                )

        eps_values = [row["epsilon"] for row in group_rows if row.get("epsilon") not in (None, "")]
        if eps_values:
            eps_values = [float(v) for v in eps_values]
            summary_row["epsilon_mean"] = float(np.mean(eps_values))
            summary_row["epsilon_std"] = (
                float(np.std(eps_values, ddof=1)) if len(eps_values) > 1 else 0.0
            )

        summary.append(summary_row)

    summary.sort(key=lambda row: (str(row["dataset"]), str(row["ablation_id"])))
    return summary


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--method",
        choices=[
            "centralized",
            "logistic_regression",
            "hist_gbdt",
            "xgboost",
            "lightgbm",
            "local_only",
            "fedavg",
            "fedprox",
            "label_aware_fedavg",
            "dp_fedavg",
            "ha_dp_fedavg",
        ],
        required=True,
    )
    parser.add_argument("--split", choices=["iid", "noniid"], default="iid")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha for Non-IID.")
    parser.add_argument("--noise", type=float, default=1.0, help="DP noise multiplier.")
    parser.add_argument(
        "--dp-target-epsilon",
        type=float,
        default=None,
        help="Calibrate standard DP-FedAvg to this composed epsilon.",
    )
    parser.add_argument(
        "--class-weighting",
        choices=["none", "balanced"],
        default="balanced",
        help="Training-loss correction for imbalanced credit defaults.",
    )
    parser.add_argument("--prox-mu", type=float, default=0.01, help="FedProx proximal strength.")
    parser.add_argument(
        "--metadata-epsilon",
        type=float,
        default=0.1,
        help="One-time epsilon for private HA label-rate release; 0 assumes public metadata.",
    )
    parser.add_argument(
        "--secure-rng",
        action="store_true",
        help="Use Opacus cryptographically secure randomness (requires torchcsprng).",
    )
    parser.add_argument(
        "--privacy-mode",
        choices=["epsilon_allocation", "noise_schedule"],
        default="epsilon_allocation",
        help="Only for ha_dp_fedavg.",
    )
    parser.add_argument("--total-epsilon", type=float, default=3.0, help="Total epsilon budget.")
    parser.add_argument(
        "--epsilon-strategy",
        choices=["uniform", "front_loaded", "back_loaded"],
        default="back_loaded",
        help="Inverse-noise-squared cost shape; final epsilon is jointly RDP-calibrated.",
    )
    parser.add_argument("--noise-min", type=float, default=0.5)
    parser.add_argument("--noise-max", type=float, default=2.0)
    parser.add_argument("--balance-gamma", type=float, default=1.0)
    parser.add_argument(
        "--ha-strength",
        type=float,
        default=0.5,
        help="Mixing strength between FedAvg and HA weights; must be in [0, 1].",
    )
    parser.add_argument(
        "--ha-prior-strength",
        type=float,
        default=20.0,
        help="Pseudo-count used to stabilize client label-rate estimates.",
    )
    parser.add_argument(
        "--noise-schedule",
        choices=["uniform", "increasing", "decreasing"],
        default="decreasing",
    )
    parser.add_argument(
        "--no-ha-aggregation",
        action="store_true",
        help="Disable heterogeneity-aware weights (ablation for ha_dp_fedavg).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional CSV output path under results/tables/.",
    )
    return parser
