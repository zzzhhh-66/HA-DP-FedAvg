"""Heterogeneity-aware record-level DP-FedAvg with composed privacy schedules."""

from __future__ import annotations

import numpy as np
from opacus.accountants import RDPAccountant

from .aggregation import (
    compute_client_label_skews,
    compute_heterogeneity_aware_weights,
    compute_label_distribution_aware_weights,
    compute_noise_aware_representative_weights,
    compute_sample_weights,
    release_positive_rates,
)
from .communication import total_communication_cost
from .dataset import create_dataloader
from .evaluate import evaluate_model, format_metrics
from .models import CreditRiskMLP
from .privacy_accounting import calibrate_noise_schedule, compose_rdp_epsilon_prefixes
from .privacy_schedule import get_round_noise_multiplier
from .train_dp_fedavg import train_model_dp
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


def train_ha_dp_fedavg(
    clients: list[tuple[np.ndarray, np.ndarray]],
    X_test,
    y_test,
    input_dim: int,
    global_rounds: int = 50,
    local_epochs: int = 1,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    use_heterogeneity_aware: bool = True,
    balance_gamma: float = 1.0,
    ha_strength: float = 0.5,
    ha_prior_strength: float = 20.0,
    privacy_mode: str = "epsilon_allocation",
    total_epsilon: float = 3.0,
    epsilon_strategy: str = "back_loaded",
    noise_min: float = 0.5,
    noise_max: float = 2.0,
    noise_schedule: str = "decreasing",
    metadata_epsilon: float = 0.0,
    pos_weight: float | None = None,
    secure_rng: bool = False,
    X_val=None,
    y_val=None,
    aggregation_strategy: str | None = None,
    representativeness_strength: float = 0.5,
    discrepancy_scale: float = 2.0,
    noise_strength: float = 0.5,
    schedule_power: float = 1.0,
    schedule_ha_coupling: float = 0.0,
    fast_epsilon_tracking: bool = True,
) -> dict:
    """Train the proposed method under a closed, auditable privacy budget.

    In compatibility-named ``epsilon_allocation`` mode, each client gets an
    inverse-noise-squared cost schedule jointly calibrated so all local optimizer
    steps across all rounds compose to ``total_epsilon``. The strategy weights
    are not literal per-round epsilon shares. If HA metadata is privatized,
    ``metadata_epsilon`` is reserved from that total budget and composed once.
    """
    if privacy_mode not in {"epsilon_allocation", "noise_schedule"}:
        raise ValueError(f"Unknown privacy_mode: {privacy_mode}")
    if global_rounds <= 0 or local_epochs <= 0:
        raise ValueError("global_rounds and local_epochs must be positive.")
    if metadata_epsilon < 0:
        raise ValueError("metadata_epsilon cannot be negative.")
    if schedule_power <= 0 or schedule_ha_coupling < 0:
        raise ValueError("schedule_power must be positive and coupling non-negative.")
    if aggregation_strategy is None:
        aggregation_strategy = "balanced_ha" if use_heterogeneity_aware else "sample"
    allowed_aggregation = {
        "sample",
        "balanced_ha",
        "label_representative",
        "noise_aware_representative",
    }
    if aggregation_strategy not in allowed_aggregation:
        raise ValueError(f"Unknown aggregation_strategy: {aggregation_strategy}")
    uses_label_metadata = aggregation_strategy != "sample" or schedule_ha_coupling > 0
    effective_metadata_epsilon = metadata_epsilon if uses_label_metadata else 0.0
    if privacy_mode == "epsilon_allocation" and total_epsilon <= effective_metadata_epsilon:
        raise ValueError("total_epsilon must exceed metadata_epsilon.")

    set_seed(seed)
    device = get_device()
    global_model = CreditRiskMLP(input_dim).to(device)
    num_parameters = count_parameters(global_model)
    num_clients = len(clients)
    n_train = sum(len(client_y) for _, client_y in clients)
    delta = compute_delta(n_train)
    positive_rates = None
    if uses_label_metadata:
        positive_rates = release_positive_rates(
            clients, metadata_epsilon=effective_metadata_epsilon
        )
    if aggregation_strategy == "balanced_ha":
        client_weights = compute_heterogeneity_aware_weights(
            clients,
            gamma=balance_gamma,
            positive_rates=positive_rates,
            ha_strength=ha_strength,
            prior_strength=ha_prior_strength,
        )
    elif aggregation_strategy == "label_representative":
        client_weights = compute_label_distribution_aware_weights(
            clients,
            gamma=balance_gamma,
            positive_rates=positive_rates,
            representativeness_strength=representativeness_strength,
            prior_strength=ha_prior_strength,
        )
    else:
        client_weights = compute_sample_weights(clients)

    client_skews = (
        compute_client_label_skews(
            clients,
            positive_rates=positive_rates,
            prior_strength=ha_prior_strength,
        )
        if schedule_ha_coupling > 0
        else [0.0] * len(clients)
    )
    client_schedule_powers = [
        schedule_power * (1.0 + schedule_ha_coupling * skew) for skew in client_skews
    ]

    client_noise_schedules: list[list[float]] = []
    if privacy_mode == "epsilon_allocation":
        training_epsilon = total_epsilon - effective_metadata_epsilon
        for client_id, (_, client_y) in enumerate(clients):
            batches_per_epoch = max(1, int(np.ceil(len(client_y) / batch_size)))
            client_noise_schedules.append(
                calibrate_noise_schedule(
                    target_epsilon=training_epsilon,
                    delta=delta,
                    total_rounds=global_rounds,
                    sample_rate=1.0 / batches_per_epoch,
                    steps_per_round=batches_per_epoch * local_epochs,
                    strategy=epsilon_strategy,
                    schedule_power=client_schedule_powers[client_id],
                    use_fast_accounting=True,
                )
            )
    else:
        shared_schedule = [
            get_round_noise_multiplier(
                round_idx=round_idx,
                total_rounds=global_rounds,
                noise_min=noise_min,
                noise_max=noise_max,
                schedule=noise_schedule,
            )
            for round_idx in range(global_rounds)
        ]
        client_noise_schedules = [list(shared_schedule) for _ in clients]

    accountants = None if fast_epsilon_tracking else [RDPAccountant() for _ in clients]
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
        client_states = []
        round_noises = []
        for client_id, (client_X, client_y) in enumerate(clients):
            local_model = clone_model(global_model).to(device)
            loader = create_dataloader(
                client_X,
                client_y,
                batch_size=batch_size,
                shuffle=True,
                seed=seed + round_idx * num_clients + client_id,
            )
            round_noise = client_noise_schedules[client_id][round_idx]
            local_model, _ = train_model_dp(
                local_model,
                loader,
                epochs=local_epochs,
                learning_rate=learning_rate,
                device=device,
                noise_multiplier=round_noise,
                max_grad_norm=max_grad_norm,
                accountant=None if accountants is None else accountants[client_id],
                pos_weight=pos_weight,
                secure_rng=secure_rng,
            )
            client_states.append(get_model_state_dict(local_model))
            round_noises.append(round_noise)

        composed_epsilons = (
            [history[round_idx] for history in client_epsilon_histories]
            if client_epsilon_histories is not None
            else [accountant.get_epsilon(delta) for accountant in accountants]
        )
        round_epsilon = float(max(composed_epsilons)) + effective_metadata_epsilon
        epsilon_history.append(round_epsilon)
        noise_history.append(float(np.mean(round_noises)))

        if aggregation_strategy == "noise_aware_representative":
            client_weights = compute_noise_aware_representative_weights(
                clients,
                round_noises,
                positive_rates=positive_rates,
                representativeness_strength=representativeness_strength,
                discrepancy_scale=discrepancy_scale,
                noise_strength=noise_strength,
                prior_strength=ha_prior_strength,
            )
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
            print(
                f"[HA-DP-FedAvg Val Round {round_idx + 1}] {format_metrics(metrics)} | "
                f"noise_mean={noise_history[-1]:.3f} | composed_eps={round_epsilon:.3f}"
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
        f"[HA-DP-FedAvg Final] {format_metrics(final_metrics)} | "
        f"best_val_round={best_round} | "
        f"epsilon={final_epsilon:.3f}, delta={delta:.2e}, "
        f"time={training_time:.2f}s | comm={communication_cost / (1024 * 1024):.2f} MB"
    )
    return {
        "method": "ha_dp_fedavg",
        "metrics": final_metrics,
        "round_metrics": round_metrics,
        "validation_metrics": best_validation_metrics,
        "best_round": best_round,
        "model_selection_metric": "validation_auprc" if has_validation else "final_round",
        "epsilon_history": epsilon_history,
        "noise_history": noise_history,
        "training_time": training_time,
        "communication_cost": communication_cost,
        "epsilon": final_epsilon,
        "delta": delta,
        "noise_multiplier": float(np.mean(noise_history)),
        "max_grad_norm": max_grad_norm,
        "privacy_mode": privacy_mode,
        "epsilon_strategy": epsilon_strategy if privacy_mode == "epsilon_allocation" else None,
        "noise_schedule": noise_schedule if privacy_mode == "noise_schedule" else None,
        "total_epsilon": total_epsilon if privacy_mode == "epsilon_allocation" else None,
        "privacy_unit": "record",
        "accountant": "RDP",
        "metadata_epsilon": effective_metadata_epsilon,
        "ha_strength": ha_strength if aggregation_strategy == "balanced_ha" else 0.0,
        "ha_prior_strength": ha_prior_strength if uses_label_metadata else 0.0,
        "aggregation_strategy": aggregation_strategy,
        "representativeness_strength": representativeness_strength,
        "discrepancy_scale": discrepancy_scale,
        "noise_strength": noise_strength,
        "schedule_power": schedule_power,
        "schedule_ha_coupling": schedule_ha_coupling,
        "client_schedule_powers": client_schedule_powers,
        "epsilon_tracking": (
            "precomputed_exact_rdp" if fast_epsilon_tracking else "runtime_opacus_history"
        ),
        "metadata_rng": "system_entropy" if effective_metadata_epsilon > 0 else "not_used",
        "secure_rng": secure_rng,
        "privacy_claim_valid": bool(
            secure_rng and (not uses_label_metadata or effective_metadata_epsilon > 0)
        ),
        "adjacency": "fixed_size_replace_one",
        "client_sample_counts": "public_fixed_metadata",
        "client_sizes": [int(len(client_y)) for _, client_y in clients],
        "released_positive_rates": positive_rates,
        "aggregation_weights": [float(weight / sum(client_weights)) for weight in client_weights],
        "model": global_model,
    }
