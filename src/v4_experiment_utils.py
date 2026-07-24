"""Shared checkpoint and validation helpers for v4 follow-up experiments."""

from __future__ import annotations

import csv
import inspect
import json
from collections.abc import Iterable
from pathlib import Path
from statistics import mean, stdev
from typing import Any

REQUIRED_V3_ARGUMENTS = {
    "ha_prior_strength",
    "ha_strength",
    "metadata_epsilon",
    "secure_rng",
}
REQUIRED_V5_ARGUMENTS = REQUIRED_V3_ARGUMENTS | {
    "aggregation_strategy",
    "evaluate_client_profiles",
    "fast_epsilon_tracking",
    "schedule_ha_coupling",
    "schedule_power",
}


def require_v3_core() -> None:
    from .experiment_runner import run_single_experiment

    parameters = inspect.signature(run_single_experiment).parameters
    missing = sorted(REQUIRED_V3_ARGUMENTS.difference(parameters))
    if missing:
        raise RuntimeError(
            "The HPC project does not contain the required v3 core. Missing "
            f"run_single_experiment arguments: {missing}"
        )


def require_v5_core() -> None:
    from .experiment_runner import run_single_experiment

    parameters = inspect.signature(run_single_experiment).parameters
    missing = sorted(REQUIRED_V5_ARGUMENTS.difference(parameters))
    if missing:
        raise RuntimeError(
            "The HPC project does not contain the required v5 core. Missing "
            f"run_single_experiment arguments: {missing}"
        )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in materialized for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def parse_float(value: Any) -> float:
    return float(value)


def summarize_numeric(
    rows: Iterable[dict[str, Any]],
    group_keys: tuple[str, ...],
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row.get(name, "")) for name in group_keys)
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        item: dict[str, Any] = dict(zip(group_keys, key, strict=False))
        item["num_runs"] = len(group)
        for metric in metrics:
            values = [parse_float(row[metric]) for row in group if row.get(metric) not in (None, "")]
            if not values:
                continue
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summary.append(item)
    return summary
