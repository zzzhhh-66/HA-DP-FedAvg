"""Public project API with lazy imports.

Keeping this module lightweight lets CSV analysis and plotting run on login
nodes that do not have the full PyTorch training stack installed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_IMPORTS = {
    "CreditRiskMLP": (".models", "CreditRiskMLP"),
    "bytes_to_mb": (".communication", "bytes_to_mb"),
    "communication_per_round": (".communication", "communication_per_round"),
    "compute_metrics": (".evaluate", "compute_metrics"),
    "create_iid_clients": (".split_clients", "create_iid_clients"),
    "create_noniid_clients_dirichlet": (
        ".split_clients",
        "create_noniid_clients_dirichlet",
    ),
    "evaluate_model": (".evaluate", "evaluate_model"),
    "format_metrics": (".evaluate", "format_metrics"),
    "model_size_bytes": (".communication", "model_size_bytes"),
    "preprocess_german_credit_numeric": (
        ".preprocess",
        "preprocess_german_credit_numeric",
    ),
    "preprocess_give_me_some_credit": (
        ".preprocess",
        "preprocess_give_me_some_credit",
    ),
    "print_client_distribution": (
        ".split_clients",
        "print_client_distribution",
    ),
    "print_dataset_summary": (".preprocess", "print_dataset_summary"),
    "run_single_experiment": (".experiment_runner", "run_single_experiment"),
    "save_results": (".experiment_runner", "save_results"),
    "total_communication_cost": (
        ".communication",
        "total_communication_cost",
    ),
}

__all__ = sorted(_LAZY_IMPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_IMPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
