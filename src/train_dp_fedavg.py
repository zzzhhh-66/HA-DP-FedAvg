"""Record-level DP-FedAvg with privacy composed across communication rounds."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader

from .aggregation import (
    compute_heterogeneity_aware_weights,
    compute_sample_weights,
    release_positive_rates,
)
from .communication import total_communication_cost
from .dataset import create_dataloader
from .evaluate import evaluate_model, format_metrics
from .models import CreditRiskMLP
from .privacy_accounting import calibrate_noise_schedule, compose_rdp_epsilon_prefixes
from .training import binary_criterion
from .utils import (
    Timer,
    clone_model,
    compute_delta,
    count_parameters,
    fedavg_aggregate,
    get_device,
    get_model_state_dict,
    load_state_dict,
    set_seed,
)


def train_model_dp(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    noise_multiplier: float,
    max_grad_norm: float,
    accountant: RDPAccountant | None,
    pos_weight: float | None = None,
    secure_rng: bool = False,
) -> tuple[nn.Module, float]:
    """Train one local model and update a persistent per-client accountant."""
    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be positive for DP training.")
    model = ModuleValidator.fix(model)
    model.train()
    criterion = binary_criterion(pos_weight, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=secure_rng)
    model, optimizer, private_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    sample_rate = 1.0 / len(private_loader)
    steps = 0
    for _ in range(epochs):
        for features, labels in private_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            steps += 1

    if accountant is not None:
        for _ in range(steps):
            accountant.step(
                noise_multiplier=float(noise_multiplier),
                sample_rate=float(sample_rate),
            )
    return model, float(sample_rate)


def train_dp_fedavg(
    clients: list[tuple[np.ndarray, np.ndarray]],
    X_test,
    y_test,
    input_dim: int,
    global_rounds: int = 50,
    local_epochs: int = 1,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    noise_multiplier: float = 1.0,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    use_heterogeneity_aware: bool = False,
    balance_gamma: float = 1.0,
    ha_strength: float = 0.5,
    ha_prior_strength: float = 20.0,
    metadata_epsilon: float = 0.0,
    pos_weight: float | None = None,
    secure_rng: bool = False,
    X_val=None,
    y_val=None,
    target_epsilon: float | None = None,
    fast_epsilon_tracking: bool = True,
) -> dict:
    """Run local record-level DP-SGD and compose each client's privacy loss.

    Clients own disjoint records, so the global guarantee is the worst composed
    client epsilon (parallel composition). Repeated participation by the same
    client is composed by its persistent RDP accountant.
    """
    if global_rounds <= 0 or local_epochs <= 0:
        raise ValueError("global_rounds and local_epochs must be positive.")
    set_seed(seed)
    device = get_device()
    global_model = CreditRiskMLP(input_dim).to(device)
    num_parameters = count_parameters(global_model)
    num_clients = len(clients)
    n_train = sum(len(client_y) for _, client_y in clients)
    delta = compute_delta(n_train)
    accountants = None if fast_epsilon_tracking else [RDPAccountant() for _ in clients]
    effective_metadata_epsilon = metadata_epsilon if use_heterogeneity_aware else 0.0
    if target_epsilon is not None and target_epsilon <= effective_metadata_epsilon:
        raise ValueError("target_epsilon must exceed metadata_epsilon.")

    client_noise_schedules: list[list[float]] = []
    if target_epsilon is None:
        if noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be positive for fixed-noise DP.")
        client_noise_schedules = [[noise_multiplier] * global_rounds for _ in clients]
    else:
        training_epsilon = target_epsilon - effective_metadata_epsilon
        for _, client_y in clients:
            batches_per_epoch = max(1, int(np.ceil(len(client_y) / batch_size)))
            client_noise_schedules.append(
                calibrate_noise_schedule(
                    target_epsilon=training_epsilon,
                    delta=delta,
                    total_rounds=global_rounds,
                    sample_rate=1.0 / batches_per_epoch,
                    steps_per_round=batches_per_epoch * local_epochs,
                    strategy="uniform",
                    use_fast_accounting=True,
                )
            )

    client_epsilon_histories = None
    if fast_epsilon_tracking:
        client_epsilon_histories = []
        for (_, client_y), schedule in zip(clients, client_noise_schedules, strict=True):
            batches_per_epoch = max(1, int(np.ceil(len(client_y) / batch_size)))
            client_epsilon_histories.append(
                compose_rdp_epsilon_prefixes(
                    schedule,
                    sample_rate=1.0 / batches_per_epoch,
                    steps_per_round=batches_per_epoch * local_epochs,
                    delta=delta,
                )
            )

    positive_rates = None
    if use_heterogeneity_aware:
        positive_rates = release_positive_rates(
            clients, metadata_epsilon=effective_metadata_epsilon
        )

    timer = Timer()
    round_metrics: list[dict[str, float]] = []
    epsilon_history: list[float] = []
    noise_history: list[float] = []
    has_validation = X_val is not None and y_val is not None
    best_state = None
    best_round = global_rounds
    best_validation_metrics = None
    best_validation_auprc = float("-inf")

    for round_idx in range(global_rounds):
        client_state_dicts = []
        round_noises = []
        for client_id, (client_X, client_y) in enumerate(clients):
            local_model = clone_model(global_model).to(device)
            train_loader = create_dataloader(
                client_X,
                client_y,
                batch_size=batch_size,
                shuffle=True,
                seed=seed + round_idx * num_clients + client_id,
            )
            round_noise = client_noise_schedules[client_id][round_idx]
            local_model, _ = train_model_dp(
                local_model,
                train_loader,
                epochs=local_epochs,
                learning_rate=learning_rate,
                device=device,
                noise_multiplier=round_noise,
                max_grad_norm=max_grad_norm,
                accountant=None if accountants is None else accountants[client_id],
                pos_weight=pos_weight,
                secure_rng=secure_rng,
            )
            client_state_dicts.append(get_model_state_dict(local_model))
            round_noises.append(round_noise)

        client_weights = (
            compute_heterogeneity_aware_weights(
                clients,
                gamma=balance_gamma,
                positive_rates=positive_rates,
                ha_strength=ha_strength,
                prior_strength=ha_prior_strength,
            )
            if use_heterogeneity_aware
            else compute_sample_weights(clients)
        )
        composed_epsilons = (
            [history[round_idx] for history in client_epsilon_histories]
            if client_epsilon_histories is not None
            else [accountant.get_epsilon(delta) for accountant in accountants]
        )
        round_epsilon = float(max(composed_epsilons)) + effective_metadata_epsilon
        epsilon_history.append(round_epsilon)
        noise_history.append(float(np.mean(round_noises)))

        aggregated = fedavg_aggregate(client_state_dicts, client_weights)
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
            print(
                f"[DP-FedAvg Val Round {round_idx + 1}] {format_metrics(metrics)} | "
                f"composed_eps={round_epsilon:.3f}"
            )

    training_time = timer.stop()
    final_epsilon = epsilon_history[-1]
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
        f"[DP-FedAvg Final] {format_metrics(final_metrics)} | "
        f"best_val_round={best_round} | "
        f"epsilon={final_epsilon:.3f}, delta={delta:.2e}, "
        f"time={training_time:.2f}s | comm={communication_cost / (1024 * 1024):.2f} MB"
    )
    return {
        "method": "dp_fedavg",
        "metrics": final_metrics,
        "round_metrics": round_metrics,
        "validation_metrics": best_validation_metrics,
        "best_round": best_round,
        "model_selection_metric": "validation_auprc" if has_validation else "final_round",
        "epsilon_history": epsilon_history,
        "training_time": training_time,
        "communication_cost": communication_cost,
        "epsilon": final_epsilon,
        "delta": delta,
        "noise_multiplier": float(np.mean(noise_history)),
        "max_grad_norm": max_grad_norm,
        "privacy_unit": "record",
        "accountant": "RDP",
        "metadata_epsilon": effective_metadata_epsilon,
        "ha_strength": ha_strength if use_heterogeneity_aware else 0.0,
        "ha_prior_strength": ha_prior_strength if use_heterogeneity_aware else 0.0,
        "metadata_rng": "system_entropy" if effective_metadata_epsilon > 0 else "not_used",
        "secure_rng": secure_rng,
        "privacy_claim_valid": bool(secure_rng),
        "adjacency": "fixed_size_replace_one",
        "client_sample_counts": "public_fixed_metadata",
        "client_sizes": [int(len(client_y)) for _, client_y in clients],
        "released_positive_rates": positive_rates,
        "aggregation_weights": [float(weight / sum(client_weights)) for weight in client_weights],
        "privacy_mode": "target_epsilon" if target_epsilon is not None else "fixed_noise",
        "total_epsilon": target_epsilon,
        "epsilon_tracking": (
            "precomputed_exact_rdp" if fast_epsilon_tracking else "runtime_opacus_history"
        ),
        "model": global_model,
    }
