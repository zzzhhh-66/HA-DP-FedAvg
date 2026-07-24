# Reproducibility workflow

## 1. Validate the environment

```bash
python -m pytest -q
python run_submission_extensions_v5.py \
  --dataset default_credit \
  --variants dp_uniform scheduler ha_uniform full_ha_dp \
  --dry-run
```

## 2. Select HA hyperparameters

The paper used separate development seeds for validation-only selection:

```bash
sbatch jobs/default_credit_ha_tuning_v4.sbatch
```

The selected configuration is written to:

```text
results/tables/default_credit_ha_tuning_v4_selected.json
```

The archived paper selection is included at the expected root results path, so
the final jobs can also be run directly without repeating development tuning.

## 3. Run the final privacy-matched factorial experiment

```bash
sbatch jobs/default_credit_final_ablation_v6.sbatch
sbatch jobs/german_final_ablation_10seed_v6.sbatch
sbatch jobs/gmsc_final_ablation_10seed_v6.sbatch
```

These public-release jobs train all four variants from scratch. They do not
import early V4/V5 result rows.

Each dataset uses the same ten final seeds:

```text
42, 123, 2026, 31415, 27182, 73, 101, 909, 4096, 65537
```

The final 2x2 variants are `dp_uniform`, `scheduler`, `ha_uniform`, and
`full_ha_dp`.

## 4. Run reference baselines

```bash
sbatch jobs/default_credit_reference_baselines_v6.sbatch
sbatch jobs/german_reference_baselines_v6.sbatch
sbatch jobs/gmsc_reference_baselines_v6.sbatch
```

These jobs cover centralized MLP, logistic regression, histogram gradient
boosting, XGBoost, IID FedAvg, Non-IID FedAvg, and Non-IID FedProx.

## 5. Analyze the primary experiment

```bash
python analyze_submission_v6.py \
  --privacy-raw \
    results/tables/final/default_credit_final_ablation_10seed_v6_raw.csv \
    results/tables/final/german_final_ablation_10seed_v6_raw.csv \
    results/tables/final/gmsc_final_ablation_10seed_v6_raw.csv \
  --baseline-raw \
    results/tables/final/default_credit_reference_baselines_v6_raw.csv \
    results/tables/final/german_reference_baselines_v6_raw.csv \
    results/tables/final/gmsc_reference_baselines_v6_raw.csv \
  --output-prefix paper_v6
```

Validate protocol completeness:

```bash
python validate_submission_v6.py \
  --privacy-raw \
    results/tables/final/default_credit_final_ablation_10seed_v6_raw.csv \
    results/tables/final/german_final_ablation_10seed_v6_raw.csv \
    results/tables/final/gmsc_final_ablation_10seed_v6_raw.csv \
  --baseline-raw \
    results/tables/final/default_credit_reference_baselines_v6_raw.csv \
    results/tables/final/german_reference_baselines_v6_raw.csv \
    results/tables/final/gmsc_reference_baselines_v6_raw.csv
```

## 6. Run round sensitivity

After the 50-round raw files exist:

```bash
sbatch jobs/default_credit_round_sensitivity_v6.sbatch
sbatch jobs/gmsc_round_sensitivity_v6.sbatch
```

Each sensitivity job retrains the two privacy-matched methods at 75 and 100
rounds, recalibrating the noise schedule to the same total `epsilon=3`.

## 7. Generate figures

```bash
python plot_submission_v6.py \
  results/tables/final/default_credit_final_ablation_10seed_v6_raw.csv \
  results/tables/final/german_final_ablation_10seed_v6_raw.csv \
  results/tables/final/gmsc_final_ablation_10seed_v6_raw.csv \
  --output-prefix paper_v6
```

Generated files are written under `results/figures/`.
