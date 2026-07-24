"""Validate completeness and protocol consistency of final v6 result files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.paths import TABLES_DIR, ensure_result_dirs
from src.v4_experiment_utils import parse_float, read_csv_rows

FINAL_SEEDS = {42, 123, 2026, 31415, 27182, 73, 101, 909, 4096, 65537}
ALPHAS = {0.5, 0.1}
PRIVACY_VARIANTS = {
    "dp_uniform",
    "scheduler",
    "ha_uniform",
    "full_ha_dp",
}
GLOBAL_BASELINES = {
    "centralized_mlp",
    "logistic_regression",
    "hist_gbdt",
    "xgboost",
    "fedavg_iid",
}
NONIID_BASELINES = {"fedavg_noniid", "fedprox_noniid"}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Validate final v6 result bundles.")
    result.add_argument("--privacy-raw", nargs="+", type=Path, required=True)
    result.add_argument("--baseline-raw", nargs="+", type=Path, required=True)
    result.add_argument(
        "--output",
        type=Path,
        default=TABLES_DIR / "paper_v6_integrity_report.json",
    )
    return result


def _float(value: Any) -> float | None:
    if value in ("", None, "None", "nan"):
        return None
    return parse_float(value)


def _seed(row: dict[str, Any]) -> int:
    return int(parse_float(row["seed"]))


def _dataset_groups(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        for row in read_csv_rows(path):
            groups.setdefault(str(row.get("dataset", "")), []).append(row)
    return groups


def _privacy_report(
    dataset: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    unique: dict[tuple[float, int, float, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            float(parse_float(row["alpha"])),
            _seed(row),
            float(parse_float(row.get("target_epsilon", row.get("total_epsilon")))),
            str(row["v5_variant"]),
        )
        unique[key] = row
    rows = list(unique.values())
    errors: list[str] = []
    variants = {str(row.get("v5_variant", "")) for row in rows}
    seeds = {_seed(row) for row in rows}
    alphas = {float(parse_float(row["alpha"])) for row in rows}
    epsilons = {
        float(parse_float(row.get("target_epsilon", row.get("total_epsilon"))))
        for row in rows
    }
    if len(rows) != 80:
        errors.append(f"expected 80 unique privacy rows, found {len(rows)}")
    if variants != PRIVACY_VARIANTS:
        errors.append(f"privacy variants mismatch: {sorted(variants)}")
    if seeds != FINAL_SEEDS:
        errors.append(f"privacy seeds mismatch: {sorted(seeds)}")
    if alphas != ALPHAS:
        errors.append(f"privacy alphas mismatch: {sorted(alphas)}")
    if epsilons != {3.0}:
        errors.append(f"target epsilon mismatch: {sorted(epsilons)}")

    cell_counts = Counter(
        (float(parse_float(row["alpha"])), str(row["v5_variant"])) for row in rows
    )
    for alpha in ALPHAS:
        for variant in PRIVACY_VARIANTS:
            if cell_counts[(alpha, variant)] != 10:
                errors.append(
                    f"cell alpha={alpha:g}, variant={variant} has "
                    f"{cell_counts[(alpha, variant)]} rows"
                )

    threshold_missing = sum(
        str(row.get("threshold_strategy", "")) != "validation_f1" for row in rows
    )
    prediction_missing = sum(not str(row.get("prediction_path", "")) for row in rows)
    profile_missing = sum(
        _float(row.get("client_auprc_mean")) is None for row in rows
    )
    if threshold_missing:
        errors.append(f"{threshold_missing} rows do not use validation_f1 thresholding")
    if prediction_missing:
        errors.append(f"{prediction_missing} rows have no prediction artifact path")
    if profile_missing:
        errors.append(f"{profile_missing} rows have no client-profile AUPRC")

    return {
        "dataset": dataset,
        "kind": "privacy_factorial",
        "unique_rows": len(rows),
        "cell_counts": {
            f"alpha={alpha:g}/{variant}": cell_counts[(alpha, variant)]
            for alpha in sorted(ALPHAS, reverse=True)
            for variant in sorted(PRIVACY_VARIANTS)
        },
        "errors": errors,
        "valid": not errors,
    }


def _baseline_report(
    dataset: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    unique: dict[tuple[str, float | None, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["baseline_variant"]),
            _float(row.get("alpha")),
            _seed(row),
        )
        unique[key] = row
    rows = list(unique.values())
    errors: list[str] = []
    variants = {str(row.get("baseline_variant", "")) for row in rows}
    expected_variants = GLOBAL_BASELINES | NONIID_BASELINES
    if len(rows) != 90:
        errors.append(f"expected 90 unique baseline rows, found {len(rows)}")
    if variants != expected_variants:
        errors.append(f"baseline variants mismatch: {sorted(variants)}")

    counts = Counter(str(row["baseline_variant"]) for row in rows)
    for variant in GLOBAL_BASELINES:
        if counts[variant] != 10:
            errors.append(f"{variant} has {counts[variant]} rows, expected 10")
    for variant in NONIID_BASELINES:
        if counts[variant] != 20:
            errors.append(f"{variant} has {counts[variant]} rows, expected 20")

    for variant in NONIID_BASELINES:
        variant_rows = [
            row for row in rows if str(row["baseline_variant"]) == variant
        ]
        variant_alphas = {
            float(parse_float(row["alpha"])) for row in variant_rows
        }
        if variant_alphas != ALPHAS:
            errors.append(f"{variant} alphas mismatch: {sorted(variant_alphas)}")

    threshold_missing = sum(
        str(row.get("threshold_strategy", "")) != "validation_f1" for row in rows
    )
    if threshold_missing:
        errors.append(f"{threshold_missing} baselines lack validation_f1 thresholding")
    return {
        "dataset": dataset,
        "kind": "reference_baselines",
        "unique_rows": len(rows),
        "variant_counts": dict(sorted(counts.items())),
        "errors": errors,
        "valid": not errors,
    }


def main() -> None:
    args = parser().parse_args()
    ensure_result_dirs()
    privacy_groups = _dataset_groups(args.privacy_raw)
    baseline_groups = _dataset_groups(args.baseline_raw)
    reports = [
        *[
            _privacy_report(dataset, rows)
            for dataset, rows in sorted(privacy_groups.items())
        ],
        *[
            _baseline_report(dataset, rows)
            for dataset, rows in sorted(baseline_groups.items())
        ],
    ]
    errors = [
        f"{report['dataset']}/{report['kind']}: {error}"
        for report in reports
        for error in report["errors"]
    ]
    payload = {
        "valid": not errors,
        "num_datasets_privacy": len(privacy_groups),
        "num_datasets_baseline": len(baseline_groups),
        "reports": reports,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[Integrity] valid={payload['valid']} errors={len(errors)} "
        f"report={args.output}",
        flush=True,
    )
    for error in errors:
        print(f"  [Error] {error}", flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
