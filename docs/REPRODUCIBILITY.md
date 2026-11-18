# Reproducing the paper experiments

Run the commands from the repository root in Bash. Set up the Python
environment and benchmark files using the [README](../README.md) and the
[dataset guide](../data/README.md).

## 1. Run the final factorial comparison and reference baselines locally

The archived HA configuration supplies the settings selected on validation
data from Default Credit development seeds 7 and 11. The final evaluation
uses the ten seeds listed below for every dataset and variant.

```bash
ha_config=results/tables/default_credit_ha_tuning_v4_selected.json
seeds=(42 123 2026 31415 27182 73 101 909 4096 65537)
datasets=(default_credit german gmsc)

for dataset in "${datasets[@]}"; do
  python -u run_submission_extensions_v5.py \
    --dataset "$dataset" --alphas 0.5 0.1 --seeds "${seeds[@]}" \
    --epsilons 3 --rounds 50 --metadata-epsilon 0.1 \
    --variants dp_uniform scheduler ha_uniform full_ha_dp \
    --ha-config "$ha_config" --output-prefix final_ablation_10seed_v6 \
    --resume || break
done
```

Run the reference comparisons with the same seeds. Install
`requirements-optional.txt` first for XGBoost.

```bash
for dataset in "${datasets[@]}"; do
  python -u run_reference_baselines_v6.py \
    --dataset "$dataset" --alphas 0.5 0.1 --seeds "${seeds[@]}" \
    --variants fedavg_iid fedavg_noniid fedprox_noniid centralized_mlp \
      logistic_regression hist_gbdt xgboost \
    --rounds 50 --centralized-epochs 30 --prox-mu 0.01 \
    --output-prefix reference_baselines_v6 --resume || break
done
```

Each dataset produces 80 private runs (two α values × four variants × ten
seeds) and 90 reference runs (nine configurations × ten seeds). The runners
checkpoint completed configurations in `results/tables/`. Reuse the same
arguments and `--resume` to continue an interrupted run. Pick a fresh
`--output-prefix` when changing the protocol or starting a separate run.
The archived files under `results/tables/final/` are separate from these
outputs.

## 2. Run longer communication horizons locally

Default Credit and GMSC use the first five final seeds. Each longer run
trains from scratch and recalibrates noise to the same total ε of 3. Complete
the corresponding 50-round comparison before running this block.

```bash
for dataset in default_credit gmsc; do
  for rounds in 75 100; do
    python -u run_submission_extensions_v5.py \
      --dataset "$dataset" --alphas 0.5 0.1 --seeds "${seeds[@]:0:5}" \
      --epsilons 3 --rounds "$rounds" --metadata-epsilon 0.1 \
      --variants dp_uniform full_ha_dp --no-client-profiles \
      --ha-config "$ha_config" --output-prefix "round_sensitivity_${rounds}r_v6" \
      --resume || break 2
  done
  python analyze_round_sensitivity_v6.py \
    --dataset "$dataset" --alphas 0.5 0.1 --seeds "${seeds[@]:0:5}" \
    --raw-50 "results/tables/${dataset}_final_ablation_10seed_v6_raw.csv" \
    --raw-75 "results/tables/${dataset}_round_sensitivity_75r_v6_raw.csv" \
    --raw-100 "results/tables/${dataset}_round_sensitivity_100r_v6_raw.csv" \
    --output-prefix round_sensitivity_v6 || break
done
```

## 3. Submit the equivalent jobs with Slurm

Set up the Conda environment and GPU partition using the
[cluster instructions](../jobs/README.md). Submit from the repository root.

```bash
mkdir -p results/logs
sbatch jobs/default_credit_final_ablation_v6.sbatch
sbatch jobs/german_final_ablation_10seed_v6.sbatch
sbatch jobs/gmsc_final_ablation_10seed_v6.sbatch

sbatch jobs/default_credit_reference_baselines_v6.sbatch
sbatch jobs/german_reference_baselines_v6.sbatch
sbatch jobs/gmsc_reference_baselines_v6.sbatch
```

After the corresponding 50-round CSVs have been generated in
`results/tables/`, submit these jobs for 75 and 100 rounds:

```bash
sbatch jobs/default_credit_round_sensitivity_v6.sbatch
sbatch jobs/gmsc_round_sensitivity_v6.sbatch
```

For optional development-stage tuning, run
`sbatch jobs/default_credit_ha_tuning_v4.sbatch` before launching the
factorial runs. It writes the selected validation configuration to
`results/tables/default_credit_ha_tuning_v4_selected.json`. Keep the checked-in
configuration when replicating the published settings.

## 4. Reanalyze the results without training

Use the checked-in per-seed CSVs to validate coverage, summarize the main
comparison and factorial effects, and regenerate comparison plots. To use your
own completed runs, set `tables=results/tables` instead.

```bash
tables=results/tables/final
privacy=("$tables/"{default_credit,german,gmsc}_final_ablation_10seed_v6_raw.csv)
baselines=("$tables/"{default_credit,german,gmsc}_reference_baselines_v6_raw.csv)

python validate_submission_v6.py \
  --privacy-raw "${privacy[@]}" --baseline-raw "${baselines[@]}"
python analyze_submission_v6.py \
  --privacy-raw "${privacy[@]}" --baseline-raw "${baselines[@]}" \
  --output-prefix paper_v6
python plot_submission_v6.py "${privacy[@]}" --output-prefix paper_v6
```

The validator checks the expected seed and variant coverage and accounting
settings. Summaries and the integrity report go to `results/tables/`. The plot
script writes PDF and PNG comparison plots to `results/figures/`. See the
[results index](../results/README.md) for the included paper plots and tables.

## 5. Reproduce the privacy–utility curve and calibration diagnostics

The [curve job](../jobs/default_credit_epsilon_curve_v4.sbatch) runs Default
Credit at ε values of 1, 2, 3, 5, and 8 with five seeds. The equivalent local
commands generate the curve summaries and fit the calibration models using
the validation predictions produced by the curve run.

```bash
python -u run_epsilon_curve_v4.py \
  --dataset default_credit --alphas 0.5 0.1 --seeds 42 123 2026 31415 27182 \
  --epsilons 1 2 3 5 8 --methods dp_uniform scheduler full_ha_dp \
  --rounds 50 --metadata-epsilon 0.1 \
  --ha-config results/tables/default_credit_ha_tuning_v4_selected.json \
  --output-prefix epsilon_curve_v4 --resume

python analyze_prediction_calibration_v4.py \
  --raw-results results/tables/default_credit_epsilon_curve_v4_raw.csv \
  --output results/tables/default_credit_calibration_v4_report.csv
```

Run calibration in the same checkout as the curve. The raw CSV points to the
local saved prediction files. The repository includes the
[diagnostic summaries](../results/tables/diagnostics/) for inspection.
