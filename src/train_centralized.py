import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import create_dataloader
from .evaluate import evaluate_model, format_metrics
from .models import CreditRiskMLP
from .training import binary_criterion
from .utils import (
    Timer,
    get_device,
    get_model_state_dict,
    load_state_dict,
    set_seed,
)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    pos_weight: float | None = None,
) -> nn.Module:
    model.train()
    criterion = binary_criterion(pos_weight, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    return model


def train_centralized(
    X_train,
    y_train,
    X_test,
    y_test,
    input_dim: int,
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    pos_weight: float | None = None,
    seed: int = 42,
    X_val=None,
    y_val=None,
) -> dict:
    set_seed(seed)
    device = get_device()

    model = CreditRiskMLP(input_dim).to(device)
    train_loader = create_dataloader(
        X_train, y_train, batch_size=batch_size, shuffle=True, seed=seed
    )

    timer = Timer()
    criterion = binary_criterion(pos_weight, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    has_validation = X_val is not None and y_val is not None
    best_state = None
    best_epoch = epochs
    best_validation_metrics = None
    best_validation_auprc = float("-inf")
    epoch_metrics: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        if has_validation:
            validation_metrics = evaluate_model(
                model,
                X_val,
                y_val,
                device,
                pos_weight=pos_weight,
            )
            epoch_metrics.append(validation_metrics)
            if (
                np.isfinite(validation_metrics["auprc"])
                and validation_metrics["auprc"] > best_validation_auprc
            ):
                best_validation_auprc = float(validation_metrics["auprc"])
                best_validation_metrics = dict(validation_metrics)
                best_epoch = epoch + 1
                best_state = get_model_state_dict(model)

    training_time = timer.stop()

    if best_state is not None:
        load_state_dict(
            model,
            {key: value.to(device) for key, value in best_state.items()},
        )
    metrics = evaluate_model(model, X_test, y_test, device, pos_weight=pos_weight)
    print(
        f"[Centralized] {format_metrics(metrics)} | "
        f"best_val_epoch={best_epoch} | time={training_time:.2f}s"
    )

    return {
        "method": "centralized",
        "metrics": metrics,
        "round_metrics": epoch_metrics,
        "validation_metrics": best_validation_metrics,
        "best_round": best_epoch,
        "model_selection_metric": "validation_auprc" if has_validation else "final_epoch",
        "training_time": training_time,
        "model": model,
        "communication_cost": 0,
        "epsilon": None,
        "delta": None,
        "noise_multiplier": 0.0,
        "max_grad_norm": None,
    }
