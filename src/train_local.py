import numpy as np

from .dataset import create_dataloader
from .evaluate import aggregate_metric_dicts, evaluate_model, format_metrics
from .models import CreditRiskMLP
from .train_centralized import train_model
from .utils import Timer, get_device, set_seed


def train_local_only(
    clients: list[tuple[np.ndarray, np.ndarray]],
    X_test,
    y_test,
    input_dim: int,
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    pos_weight: float | None = None,
    seed: int = 42,
) -> dict:
    set_seed(seed)
    device = get_device()

    timer = Timer()
    client_metrics = []

    for client_id, (client_X, client_y) in enumerate(clients):
        model = CreditRiskMLP(input_dim).to(device)
        train_loader = create_dataloader(
            client_X,
            client_y,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + client_id,
        )
        train_model(
            model,
            train_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            pos_weight=pos_weight,
        )
        metrics = evaluate_model(model, X_test, y_test, device, pos_weight=pos_weight)
        client_metrics.append(metrics)
        print(f"[Local-only Client {client_id + 1}] {format_metrics(metrics)}")

    training_time = timer.stop()
    mean_metrics = aggregate_metric_dicts(client_metrics)

    print(f"[Local-only Mean] {format_metrics(mean_metrics)} | time={training_time:.2f}s")

    return {
        "method": "local_only",
        "metrics": mean_metrics,
        "client_metrics": client_metrics,
        "training_time": training_time,
        "communication_cost": 0,
        "epsilon": None,
        "delta": None,
        "noise_multiplier": 0.0,
        "max_grad_norm": None,
    }
