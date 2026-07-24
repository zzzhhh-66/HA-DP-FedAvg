import json
import os
import platform
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn

from . import paths as _paths

PROJECT_ROOT = _paths.PROJECT_ROOT
RESULTS_DIR = _paths.RESULTS_DIR
TABLES_DIR = _paths.TABLES_DIR
FIGURES_DIR = _paths.FIGURES_DIR
LOGS_DIR = _paths.LOGS_DIR
PREDICTIONS_DIR = _paths.PREDICTIONS_DIR
DEFAULT_SEEDS = _paths.DEFAULT_SEEDS
ensure_result_dirs = _paths.ensure_result_dirs

DATA_PATHS = {
    "gmsc": [
        PROJECT_ROOT / "data" / "give_me_some_credit" / "cs-training.csv",
        PROJECT_ROOT / "give me credits" / "cs-training.csv",
    ],
    "german": [
        PROJECT_ROOT / "data" / "german_credit" / "german.data-numeric",
        PROJECT_ROOT / "german" / "german.data-numeric",
    ],
    "default_credit": [
        PROJECT_ROOT / "data" / "default_credit_card" / "default of credit card clients.xls",
        PROJECT_ROOT / "data" / "default_credit_card" / "default_credit_card_clients.xls",
        PROJECT_ROOT / "default of credit card clients.xls",
    ],
}

def resolve_data_path(dataset: str) -> Path:
    for path in DATA_PATHS[dataset]:
        if path.exists():
            return path
    raise FileNotFoundError(f"No data file found for dataset '{dataset}'.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def clone_model(model: nn.Module) -> nn.Module:
    cloned = deepcopy(model)
    cloned.load_state_dict(model.state_dict())
    return cloned


def state_dict_to_vector(state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.detach().cpu().reshape(-1) for v in state_dict.values()])


def fedavg_aggregate(
    client_state_dicts: list[dict[str, torch.Tensor]],
    client_weights: list[int],
) -> dict[str, torch.Tensor]:
    if not client_state_dicts:
        raise ValueError("At least one client state is required.")
    if len(client_state_dicts) != len(client_weights):
        raise ValueError("Each client state must have one aggregation weight.")
    if any(weight < 0 for weight in client_weights):
        raise ValueError("Aggregation weights cannot be negative.")
    total = float(sum(client_weights))
    if total <= 0:
        raise ValueError("Aggregation weights must have a positive sum.")
    aggregated: dict[str, torch.Tensor] = {}

    for key in client_state_dicts[0]:
        aggregated[key] = sum(
            state_dict[key] * (weight / total)
            for state_dict, weight in zip(client_state_dicts, client_weights, strict=True)
        )

    return aggregated


def get_model_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    if hasattr(model, "_module"):
        return {key: value.detach().cpu() for key, value in model._module.state_dict().items()}
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def load_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    target = model._module if hasattr(model, "_module") else model
    target.load_state_dict(state_dict)


def compute_delta(n_train: int) -> float:
    return min(1e-5, 1.0 / max(n_train, 1))


def runtime_metadata() -> dict[str, str]:
    """Small, CSV-friendly provenance record for reproducible experiments."""
    try:
        import opacus

        opacus_version = opacus.__version__
    except ImportError:
        opacus_version = "not-installed"
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "opacus_version": opacus_version,
        "device": str(get_device()),
    }


class Timer:
    def __init__(self) -> None:
        self.start_time = time.time()
        self.elapsed = 0.0

    def stop(self) -> float:
        self.elapsed = time.time() - self.start_time
        return self.elapsed


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in result.items():
        if value is None:
            flattened[key] = ""
        elif isinstance(value, list | tuple | dict):
            flattened[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        else:
            flattened[key] = value
    return flattened
