"""Run the v5 submission-strengthening experiments with resumable checkpoints."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

from src.config import DATASET_CONFIG
from src.paths import DEFAULT_SEEDS, TABLES_DIR, ensure_result_dirs
from src.v4_experiment_utils import (
    parse_float,
    read_csv_rows,
    require_v5_core,
    summarize_numeric,
    write_csv_rows,
)

NEW_FINAL_SEEDS = [73, 101, 909, 4096, 65537]
ALL_FINAL_SEEDS = DEFAULT_SEEDS + NEW_FINAL_SEEDS

VARIANTS: dict[str, dict[str, Any]] = {
    "dp_uniform": {
        "aggregation": "sample",
        "schedule": "uniform",
        "uses_metadata": False,
        "description": "Matched-epsilon DP-FedAvg",
    },
    "scheduler": {
        "aggregation": "sample",
        "schedule": "back_loaded",
        "uses_metadata": False,
        "description": "Back-loaded privacy scheduler only",
    },
    "front_loaded": {
        "aggregation": "sample",
        "schedule": "front_loaded",
        "uses_metadata": False,
        "description": "Front-loaded scheduler control",
    },
    "full_ha_dp": {
        "aggregation": "balanced_ha",
        "schedule": "back_loaded",
        "uses_metadata": True,
        "description": "Selected HA weighting plus back-loaded scheduler",
    },
    "ha_uniform": {
        "aggregation": "balanced_ha",
        "schedule": "uniform",
        "uses_metadata": True,
        "description": "Selected HA weighting with a uniform privacy schedule",
    },
    "dp_label_representative": {
        "aggregation": "label_representative",
        "schedule": "uniform",
        "uses_metadata": True,
        "description": "DP label-representative Non-IID control",
    },
    "scheduler_label_representative": {
        "aggregation": "label_representative",
        "schedule": "back_loaded",
        "uses_metadata": True,
        "description": "Label-representative aggregation plus scheduler",
    },
    "noise_aware_ha_v5": {
        "aggregation": "noise_aware_representative",
        "schedule": "back_loaded",
        "uses_metadata": True,
        "description": "Exploratory noise-aware representative aggregation",
    },
    "coupled_ha_scheduler_v5": {
        "aggregation": "balanced_ha",
        "schedule": "back_loaded",
        "uses_metadata": True,
        "description": "Exploratory client-skew-coupled HA scheduler",
    },
}

IMPORTABLE_MATCHED_VARIANTS = {
    "dp_uniform",
    "scheduler",
    "ha_uniform",
    "full_ha_dp",
}
SUMMARY_METRICS = (
    "auc",
    "auprc",
    "f1",
    "balanced_accuracy",
    "precision",
    "recall",
    "brier",
    "epsilon",
    "training_time",
    "client_auc_mean",
    "client_auc_min",
    "client_auprc_mean",
    "client_auprc_min",
    "client_f1_mean",
    "client_f1_min",
    "client_recall_mean",
    "client_recall_min",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run v5 submission-strengthening controls.")
    result.add_argument("--dataset", default="default_credit")
    result.add_argument("--alphas", nargs="+", type=float, default=[0.5, 0.1])
    result.add_argument("--seeds", nargs="+", type=int, default=ALL_FINAL_SEEDS)
    result.add_argument("--epsilons", nargs="+", type=float, default=[3.0])
    result.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    result.add_argument("--rounds", type=int, default=50)
    result.add_argument("--metadata-epsilon", type=float, default=0.1)
    result.add_argument("--ha-strength", type=float, default=1.0)
    result.add_argument("--ha-prior-strength", type=float, default=50.0)
    result.add_argument("--ha-config", type=Path)
    result.add_argument("--representativeness-strength", type=float, default=0.5)
    result.add_argument("--discrepancy-scale", type=float, default=2.0)
    result.add_argument("--noise-strength", type=float, default=0.5)
    result.add_argument("--schedule-power", type=float, default=1.0)
    result.add_argument("--schedule-ha-coupling", type=float, default=1.0)
    result.add_argument("--secure-rng", action="store_true")
    result.add_argument("--no-client-profiles", action="store_true")
    result.add_argument("--legacy-epsilon-tracking", action="store_true")
    result.add_argument(
        "--import-v4-raw",
        "--import-matched-raw",
        dest="import_v4_raw",
        nargs="*",
        type=Path,
        default=[],
    )
    result.add_argument(
        "--strict-import",
        action="store_true",
        help=(
            "Require imported rows to match the dataset, HA configuration, "
            "metadata budget, scheduler, and aggregation strategy."
        ),
    )
    result.add_argument("--output-prefix", default="submission_extensions_v5")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--smoke", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def load_ha_config(args: argparse.Namespace) -> None:
    if args.ha_config is None:
        return
    payload = json.loads(args.ha_config.read_text(encoding="utf-8"))
    args.ha_strength = float(payload["ha_strength"])
    args.ha_prior_strength = float(payload["ha_prior_strength"])


def validate(args: argparse.Namespace) -> None:
    if args.rounds <= 0 or any(value <= 0 for value in args.epsilons):
        raise ValueError("rounds and all epsilons must be positive.")
    if not 0 <= args.metadata_epsilon < min(args.epsilons):
        raise ValueError("metadata epsilon must be in [0, every total epsilon).")
    if not 0 <= args.ha_strength <= 1 or args.ha_prior_strength < 0:
        raise ValueError("Invalid HA strength or prior.")
    if not 0 <= args.representativeness_strength <= 1:
        raise ValueError("representativeness strength must be in [0, 1].")
    if not 0 <= args.noise_strength <= 1 or args.discrepancy_scale < 0:
        raise ValueError("Invalid noise-aware aggregation parameters.")
    if args.schedule_power <= 0 or args.schedule_ha_coupling < 0:
        raise ValueError("Invalid scheduler power or coupling.")
    if args.resume and args.overwrite:
        raise ValueError("Use either --resume or --overwrite, not both.")
    if args.secure_rng and importlib.util.find_spec("torchcsprng") is None:
        raise RuntimeError(
            "--secure-rng requires a torchcsprng build compatible with the current "
            "PyTorch/CUDA environment."
        )


def _key(row: dict[str, Any]) -> tuple[float, int, float, str]:
    variant = str(row.get("v5_variant") or row.get("curve_method"))
    return (
        parse_float(row["alpha"]),
        int(parse_float(row["seed"])),
        parse_float(row.get("target_epsilon", row.get("total_epsilon"))),
        variant,
    )


def import_matched_rows(
    paths: list[Path],
    requested: set[tuple[float, int, float, str]],
    existing: set[tuple[float, int, float, str]],
    *,
    dataset: str | None = None,
    metadata_epsilon: float | None = None,
    ha_strength: float | None = None,
    ha_prior_strength: float | None = None,
    strict: bool = False,
) -> list[dict[str, Any]]:
    imported: list[dict[str, Any]] = []
    for path in paths:
        for source in read_csv_rows(path):
            variant = str(source.get("v5_variant") or source.get("curve_method", ""))
            if variant not in IMPORTABLE_MATCHED_VARIANTS:
                continue
            try:
                key = _key(source)
            except (KeyError, TypeError, ValueError):
                continue
            if key not in requested or key in existing:
                continue
            if strict and not _import_row_matches_protocol(
                source,
                variant=variant,
                dataset=dataset,
                metadata_epsilon=metadata_epsilon,
                ha_strength=ha_strength,
                ha_prior_strength=ha_prior_strength,
            ):
                print(
                    f"[Import reject] path={path} key={key} protocol mismatch",
                    flush=True,
                )
                continue
            row: dict[str, Any] = dict(source)
            source_is_v5 = bool(source.get("v5_source"))
            row.update(
                {
                    "v5_variant": variant,
                    "v5_source": (
                        "reused_v5_matched_protocol"
                        if source_is_v5
                        else "imported_v4_matched_protocol"
                    ),
                    "v5_protocol": "matched_record_dp_noniid_v5",
                }
            )
            if not source_is_v5:
                row["client_evaluation_protocol"] = "not_available_for_imported_v4"
            imported.append(row)
            existing.add(key)
    return imported


def _optional_float_matches(
    row: dict[str, Any],
    field: str,
    expected: float | None,
    *,
    tolerance: float = 1e-9,
) -> bool:
    if expected is None:
        return True
    value = row.get(field)
    if value in (None, ""):
        return False
    try:
        return abs(parse_float(value) - expected) <= tolerance
    except (TypeError, ValueError):
        return False


def _import_row_matches_protocol(
    row: dict[str, Any],
    *,
    variant: str,
    dataset: str | None,
    metadata_epsilon: float | None,
    ha_strength: float | None,
    ha_prior_strength: float | None,
) -> bool:
    spec = VARIANTS[variant]
    if dataset is not None:
        expected_dataset = DATASET_CONFIG[dataset]["name"]
        if str(row.get("dataset", "")) != expected_dataset:
            return False
    if not str(row.get("split", "")).startswith("noniid_alpha="):
        return False
    if str(row.get("epsilon_strategy", "")) != str(spec["schedule"]):
        return False
    if str(row.get("aggregation_strategy", "")) != str(spec["aggregation"]):
        return False

    expected_metadata = metadata_epsilon if spec["uses_metadata"] else 0.0
    if not _optional_float_matches(row, "metadata_epsilon", expected_metadata):
        return False
    if spec["aggregation"] == "balanced_ha":
        if not _optional_float_matches(row, "ha_strength", ha_strength):
            return False
        if not _optional_float_matches(
            row,
            "ha_prior_strength",
            ha_prior_strength,
        ):
            return False
    return True


def add_metric_run_counts(
    summary: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    for item in summary:
        matching = [
            row
            for row in rows
            if str(row.get("alpha")) == str(item["alpha"])
            and str(row.get("target_epsilon")) == str(item["target_epsilon"])
            and str(row.get("v5_variant")) == str(item["v5_variant"])
        ]
        item["native_v5_runs"] = sum(row.get("v5_source") == "native_v5" for row in matching)
        item["imported_v4_runs"] = sum(
            row.get("v5_source") == "imported_v4_matched_protocol" for row in matching
        )
        item["reused_v5_runs"] = sum(
            row.get("v5_source") == "reused_v5_matched_protocol" for row in matching
        )
        item["client_profile_runs"] = sum(
            row.get("client_evaluation_protocol")
            == "held_out_label_profile_matched_slices"
            for row in matching
        )


def build_paired_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {_key(row): row for row in rows}
    effect_rows: list[dict[str, Any]] = []
    metrics = ("auc", "auprc", "f1", "balanced_accuracy", "client_auprc_mean")
    baselines = ("dp_uniform", "scheduler")
    for (alpha, seed, epsilon, variant), row in sorted(by_key.items()):
        if variant in baselines:
            continue
        for baseline in baselines:
            control = by_key.get((alpha, seed, epsilon, baseline))
            if control is None:
                continue
            effect: dict[str, Any] = {
                "alpha": alpha,
                "seed": seed,
                "target_epsilon": epsilon,
                "v5_variant": variant,
                "comparator": baseline,
            }
            for metric in metrics:
                if row.get(metric) in (None, "") or control.get(metric) in (None, ""):
                    continue
                effect[f"delta_{metric}"] = parse_float(row[metric]) - parse_float(
                    control[metric]
                )
            effect_rows.append(effect)
    return effect_rows


def summarize_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = tuple(
        f"delta_{metric}"
        for metric in ("auc", "auprc", "f1", "balanced_accuracy", "client_auprc_mean")
    )
    result = summarize_numeric(
        rows,
        ("alpha", "target_epsilon", "v5_variant", "comparator"),
        metrics,
    )
    for item in result:
        for metric in metrics:
            mean_key = f"{metric}_mean"
            std_key = f"{metric}_std"
            if mean_key not in item:
                continue
            count = int(item["num_runs"])
            item[f"{metric}_ci95"] = (
                1.96 * float(item[std_key]) / math.sqrt(count) if count > 1 else float("nan")
            )
    return result


def main() -> None:
    args = parser().parse_args()
    load_ha_config(args)
    validate(args)
    ensure_result_dirs()

    alphas = args.alphas[:1] if args.smoke else args.alphas
    seeds = args.seeds[:1] if args.smoke else args.seeds
    epsilons = args.epsilons[:1] if args.smoke else args.epsilons
    variants = args.variants[:1] if args.smoke else args.variants
    rounds = 2 if args.smoke else args.rounds
    suffix = "_smoke" if args.smoke else ""
    stem = f"{args.dataset}_{args.output_prefix}{suffix}"
    raw_path = TABLES_DIR / f"{stem}_raw.csv"
    summary_path = TABLES_DIR / f"{stem}_mean_std.csv"
    effects_raw_path = TABLES_DIR / f"{stem}_paired_effects_raw.csv"
    effects_summary_path = TABLES_DIR / f"{stem}_paired_effects_mean_std.csv"
    trace_path = TABLES_DIR / f"{stem}_round_trace.csv"
    output_paths = (
        raw_path,
        summary_path,
        effects_raw_path,
        effects_summary_path,
        trace_path,
    )
    if args.overwrite:
        for path in output_paths:
            path.unlink(missing_ok=True)
    elif raw_path.exists() and not args.resume:
        raise FileExistsError(f"{raw_path} exists; use --resume or --overwrite.")

    plan = [
        (alpha, seed, epsilon, variant)
        for alpha in alphas
        for epsilon in epsilons
        for seed in seeds
        for variant in variants
    ]
    requested = set(plan)
    print(
        f"[V5 plan] runs={len(plan)} dataset={args.dataset} rounds={rounds} "
        f"variants={variants} secure_rng={args.secure_rng}",
        flush=True,
    )
    if args.dry_run:
        for index, (alpha, seed, epsilon, variant) in enumerate(plan, 1):
            print(
                f"[{index:03d}/{len(plan)}] alpha={alpha} seed={seed} "
                f"epsilon={epsilon} variant={variant}"
            )
        return

    require_v5_core()
    from src.experiment_runner import run_single_experiment

    rows: list[dict[str, Any]] = read_csv_rows(raw_path) if args.resume else []
    trace_rows: list[dict[str, Any]] = read_csv_rows(trace_path) if args.resume else []
    completed = {_key(row) for row in rows}
    imported = import_matched_rows(
        args.import_v4_raw,
        requested,
        completed,
        dataset=args.dataset,
        metadata_epsilon=args.metadata_epsilon,
        ha_strength=args.ha_strength,
        ha_prior_strength=args.ha_prior_strength,
        strict=args.strict_import,
    )
    if imported:
        rows.extend(imported)
        write_csv_rows(raw_path, rows)
        print(f"[Import] reused {len(imported)} matched prior runs", flush=True)

    for index, (alpha, seed, epsilon, variant) in enumerate(plan, 1):
        key = (alpha, seed, epsilon, variant)
        if key in completed:
            print(f"[Skip] {index}/{len(plan)} {key}", flush=True)
            continue
        spec = VARIANTS[variant]
        metadata_epsilon = args.metadata_epsilon if spec["uses_metadata"] else 0.0
        coupling = (
            args.schedule_ha_coupling if variant == "coupled_ha_scheduler_v5" else 0.0
        )
        schedule_power = (
            args.schedule_power if variant == "coupled_ha_scheduler_v5" else 1.0
        )
        print(
            f"\n=== [{index}/{len(plan)}] alpha={alpha} seed={seed} "
            f"epsilon={epsilon} variant={variant} ===",
            flush=True,
        )
        row = run_single_experiment(
            dataset=args.dataset,
            method="ha_dp_fedavg",
            split="noniid",
            alpha=alpha,
            seed=seed,
            verbose=False,
            global_rounds_override=rounds,
            privacy_mode="epsilon_allocation",
            total_epsilon=epsilon,
            epsilon_strategy=str(spec["schedule"]),
            use_heterogeneity_aware=bool(spec["uses_metadata"]),
            aggregation_strategy=str(spec["aggregation"]),
            metadata_epsilon=metadata_epsilon,
            balance_gamma=1.0,
            class_weighting="balanced",
            ha_strength=args.ha_strength,
            ha_prior_strength=args.ha_prior_strength,
            representativeness_strength=args.representativeness_strength,
            discrepancy_scale=args.discrepancy_scale,
            noise_strength=args.noise_strength,
            schedule_power=schedule_power,
            schedule_ha_coupling=coupling,
            secure_rng=args.secure_rng,
            fast_epsilon_tracking=not args.legacy_epsilon_tracking,
            evaluate_client_profiles=not args.no_client_profiles,
            ablation_id=(
                f"V5_{variant}_meta{metadata_epsilon:g}_pow{schedule_power:g}_"
                f"couple{coupling:g}_secure{int(args.secure_rng)}"
            ),
            ablation_description=str(spec["description"]),
        )
        row.update(
            {
                "v5_variant": variant,
                "target_epsilon": epsilon,
                "global_rounds": rounds,
                "v5_source": "native_v5",
                "v5_protocol": "matched_record_dp_noniid_v5",
                "curve_method": variant,
            }
        )
        round_metrics = row.get("round_metrics", [])
        epsilon_history = row.get("epsilon_history", [])
        noise_history = row.get("noise_history", [])
        for round_index, metrics in enumerate(round_metrics, 1):
            trace_rows.append(
                {
                    "dataset": row["dataset"],
                    "alpha": alpha,
                    "seed": seed,
                    "target_epsilon": epsilon,
                    "v5_variant": variant,
                    "round": round_index,
                    "validation_auc": metrics.get("auc", ""),
                    "validation_auprc": metrics.get("auprc", ""),
                    "validation_f1_fixed_0_5": metrics.get("f1", ""),
                    "composed_epsilon": (
                        epsilon_history[round_index - 1]
                        if round_index <= len(epsilon_history)
                        else ""
                    ),
                    "mean_noise_multiplier": (
                        noise_history[round_index - 1]
                        if round_index <= len(noise_history)
                        else ""
                    ),
                }
            )
        rows.append(row)
        completed.add(key)
        write_csv_rows(raw_path, rows)
        write_csv_rows(trace_path, trace_rows)
        print(f"[Checkpoint] {len(completed)}/{len(plan)} -> {raw_path}", flush=True)

    summary = summarize_numeric(
        rows,
        ("alpha", "target_epsilon", "v5_variant"),
        SUMMARY_METRICS,
    )
    add_metric_run_counts(summary, rows)
    write_csv_rows(summary_path, summary)

    paired = build_paired_effects(rows)
    if paired:
        write_csv_rows(effects_raw_path, paired)
        write_csv_rows(effects_summary_path, summarize_effects(paired))
    print(
        f"[Complete] raw={raw_path} summary={summary_path} "
        f"paired={effects_summary_path} trace={trace_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
