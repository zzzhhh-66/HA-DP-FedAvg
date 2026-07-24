"""FedProx baseline for heterogeneous federated optimization."""

from __future__ import annotations

import numpy as np
import torch

from .communication import total_communication_cost
from .dataset import create_dataloader
from .evaluate import evaluate_model, format_metrics
from .models import CreditRiskMLP
from .training import binary_criterion
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


def train_fedprox(
    clients: list[tuple[np.ndarray, np.ndarray]],
    X_test,
    y_test,
    input_dim: int,
    global_rounds: int = 50,
    local_epochs: int = 1,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    prox_mu: float = 0.01,
    pos_weight: float | None = None,
    seed: int = 42,
    X_val=None,
    y_val=None,
) -> dict:
    if prox_mu < 0:
        raise ValueError("prox_mu must be non-negative.")
    set_seed(seed)
    device = get_device()
    global_model = CreditRiskMLP(input_dim).to(device)
    num_parameters = count_parameters(global_model)
    round_metrics = []
    timer = Timer()
    has_validation = X_val is not None and y_val is not None
    best_state = None
    best_round = global_rounds
    best_validation_metrics = None
    best_validation_auprc = float("-inf")

    for round_idx in range(global_rounds):
        client_states = []
        client_weights = []
        reference = [parameter.detach().clone() for parameter in global_model.parameters()]

        for client_id, (client_X, client_y) in enumerate(clients):
            local_model = clone_model(global_model).to(device)
            loader = create_dataloader(
                client_X,
                client_y,
                batch_size=batch_size,
                shuffle=True,
                seed=seed + round_idx * len(clients) + client_id,
            )
            criterion = binary_criterion(pos_weight, device)
            optimizer = torch.optim.Adam(local_model.parameters(), lr=learning_rate)
            local_model.train()
            for _ in range(local_epochs):
                for features, labels in loader:
                    features, labels = features.to(device), labels.to(device)
                    optimizer.zero_grad()
                    empirical_loss = criterion(local_model(features), labels)
                    proximal = sum(
                        torch.sum((parameter - global_parameter) ** 2)
                        for parameter, global_parameter in zip(
                            local_model.parameters(), reference, strict=True
                        )
                    )
                    loss = empirical_loss + 0.5 * prox_mu * proximal
                    loss.backward()
                    optimizer.step()

            client_states.append(
                {key: value.detach().cpu() for key, value in local_model.state_dict().items()}
            )
            client_weights.append(len(client_y))

        aggregated = fedavg_aggregate(client_states, client_weights)
        load_state_dict(global_model, {key: value.to(device) for key, value in aggregated.items()})
        evaluation_X = X_val if X_val is not None else X_test
        evaluation_y = y_val if y_val is not None else y_test
        metrics = evaluate_model(
            global_model, evaluation_X, evaluation_y, device, pos_weight=pos_weight
        )
        round_metrics.append(metrics)
        if has_validation and np.isfinite(metrics["auprc"]):
            if metrics["auprc"] > best_validation_auprc:
                best_validation_auprc = float(metrics["auprc"])
                best_round = round_idx + 1
                best_validation_metrics = dict(metrics)
                best_state = get_model_state_dict(global_model)
        if round_idx == 0 or (round_idx + 1) % 10 == 0:
            print(f"[FedProx Val Round {round_idx + 1}] {format_metrics(metrics)}")

    training_time = timer.stop()
    if best_state is not None:
        load_state_dict(
            global_model,
            {key: value.to(device) for key, value in best_state.items()},
        )
    else:
        best_validation_metrics = round_metrics[-1]
    _, communication_cost = total_communication_cost(num_parameters, len(clients), global_rounds)
    final_metrics = evaluate_model(global_model, X_test, y_test, device, pos_weight=pos_weight)
    print(
        f"[FedProx Final] {format_metrics(final_metrics)} | "
        f"best_val_round={best_round} | "
        f"time={training_time:.2f}s | comm={communication_cost / (1024 * 1024):.2f} MB"
    )
    return {
        "method": "fedprox",
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
        "prox_mu": prox_mu,
        "model": global_model,
    }
