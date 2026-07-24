"""Filesystem paths and seed constants without training dependencies."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

DEFAULT_SEEDS = [42, 123, 2026, 31415, 27182]


def ensure_result_dirs() -> None:
    for directory in (TABLES_DIR, FIGURES_DIR, LOGS_DIR, PREDICTIONS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
