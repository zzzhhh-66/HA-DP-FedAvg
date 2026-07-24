"""Label-distribution-aware FedAvg baseline for Non-IID control experiments."""

from __future__ import annotations

import numpy as np

from .aggregation import compute_label_distribution_aware_weights
from .communication import total_communication_cost
from .dataset import create_dataloader
from .evaluate import evaluate_model, format_metrics
from .models import CreditRiskMLP
from .train_centralized import train_model
from .utils import (
    Timer,
    clone_model,
    count_parameters,
    fedavg_aggregate,
    get_device,
    get_model_state_dict,
    load_state_dict,
    set_seed,
)


def train_label_aware_fedavg(
    clients: list[tuple[np.ndarray, np.ndarray]],
    X_test,
    y_test,
    input_dim: int,
    global_rounds: int = 50,
    local_epochs: int = 1,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    label_gamma: float = 1.0,
    pos_weight: float | None = None,
    seed: int = 42,
    X_val=None,
    y_val=None,
) -> dict:
    """Train FedAvg with label-distribution-aware aggregation weights.

    This is a non-private label-aware control baseline. Report it as a
    heuristic control unless you replace it with an exact FedLA reproduction
    from the cited paper.
    """
    if global_rounds <= 0 or local_epochs <= 0:
        raise ValueError("global_rounds and local_epochs must be positive.")
    set_seed(seed)
    device = get_device()

    global_model = CreditRiskMLP(input_dim).to(device)
    num_parameters = count_parameters(global_model)
    num_clients = len(clients)
    client_weights = compute_label_distribution_aware_weights(
        clients,
        gamma=label_gamma,
    )

    timer = Timer()
    round_metrics = []
    has_validation = X_val is not None and y_val is not None
    best_state = None
    best_round = global_rounds
    best_validation_metrics = None
    best_validation_auprc = float("-inf")

    for round_idx in range(global_rounds):
        client_states = []

        for client_id, (client_X, client_y) in enumerate(clients):
            local_model = clone_model(global_model).to(device)
            train_loader = create_dataloader(
                client_X,
                client_y,
                batch_size=batch_size,
                shuffle=True,
                seed=seed + round_idx * num_clients + client_id,
            )
            train_model(
                local_model,
                train_loader,
                epochs=local_epochs,
                learning_rate=learning_rate,
                device=device,
                pos_weight=pos_weight,
            )
            client_states.append(
                {key: value.detach().cpu() for key, value in local_model.state_dict().items()}
            )

        aggregated = fedavg_aggregate(client_states, client_weights)
        load_state_dict(global_model, {key: value.to(device) for key, value in aggregated.items()})

        evaluation_X = X_val if X_val is not None else X_test
        evaluation_y = y_val if y_val is not None else y_test
        metrics = evaluate_model(
            global_model,
            evaluation_X,
            evaluation_y,
            device,
            pos_weight=pos_weight,
        )
        round_metrics.append(metrics)
        if has_validation and np.isfinite(metrics["auprc"]):
            if metrics["auprc"] > best_validation_auprc:
                best_validation_auprc = float(metrics["auprc"])
                best_round = round_idx + 1
                best_validation_metrics = dict(metrics)
                best_state = get_model_state_dict(global_model)

        if round_idx == 0 or (round_idx + 1) % 10 == 0:
            print(f"[Label-Aware FedAvg Val Round {round_idx + 1}] {format_metrics(metrics)}")

    training_time = timer.stop()
    if best_state is not None:
        load_state_dict(
            global_model,
            {key: value.to(device) for key, value in best_state.items()},
        )
    else:
        best_validation_metrics = round_metrics[-1]
    final_metrics = evaluate_model(global_model, X_test, y_test, device, pos_weight=pos_weight)
    _, communication_cost = total_communication_cost(num_parameters, num_clients, global_rounds)

    print(
        f"[Label-Aware FedAvg Final] {format_metrics(final_metrics)} | "
        f"best_val_round={best_round} | "
        f"time={training_time:.2f}s | comm={communication_cost / (1024 * 1024):.2f} MB"
    )

    return {
        "method": "label_aware_fedavg",
        "metrics": final_metrics,
        "round_metrics": round_metrics,
        "validation_metrics": best_validation_metrics,
        "best_round": best_round,
        "model_selection_metric": "validation_auprc" if has_validation else "final_round",
        "training_time": training_time,
        "communication_cost": communication_cost,
        "epsilon": None,
        "delta": None,
        "noise_multiplier": 0.0,
        "max_grad_norm": None,
        "label_gamma": label_gamma,
        "client_sizes": [int(len(client_y)) for _, client_y in clients],
        "client_label_rates": [float(np.mean(client_y)) for _, client_y in clients],
        "aggregation_weights": [float(weight / sum(client_weights)) for weight in client_weights],
        "model": global_model,
    }
