"""Lightweight experiment constants shared by training and analysis scripts."""

from __future__ import annotations

from typing import Any

DATASET_CONFIG: dict[str, dict[str, Any]] = {
    "gmsc": {
        "name": "give_me_some_credit",
        "num_clients": 10,
        "batch_size": 128,
        "global_rounds": 50,
        "local_epochs": 1,
        "centralized_epochs": 30,
        "dp_noise_levels": [0.5, 1.0, 2.0],
        "positive_class_weight": 13.96,
    },
    "german": {
        "name": "german_credit",
        "num_clients": 5,
        "batch_size": 64,
        "global_rounds": 50,
        "local_epochs": 1,
        "centralized_epochs": 30,
        "dp_noise_levels": [0.3, 0.7, 1.0],
        "positive_class_weight": 7.0 / 3.0,
    },
    "default_credit": {
        "name": "default_credit_card_clients",
        "num_clients": 10,
        "batch_size": 128,
        "global_rounds": 50,
        "local_epochs": 1,
        "centralized_epochs": 30,
        "dp_noise_levels": [0.5, 1.0, 2.0],
        "positive_class_weight": 23364.0 / 6636.0,
    },
}
