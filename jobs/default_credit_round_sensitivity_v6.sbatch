#!/bin/bash
#SBATCH --job-name=dc_rounds_v6
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/logs/default_credit_round_sensitivity_v6_%j.out
#SBATCH --error=results/logs/default_credit_round_sensitivity_v6_%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/setup_env.sh"

CONFIG=results/tables/default_credit_ha_tuning_v4_selected.json
RAW50=results/tables/default_credit_final_ablation_10seed_v6_raw.csv
test -f "$CONFIG" || { echo "Missing $CONFIG"; exit 1; }
test -f "$RAW50" || { echo "Missing $RAW50"; exit 1; }

echo "[Job] id=$SLURM_JOB_ID host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

for ROUNDS in 75 100; do
  echo "[Stage] dataset=default_credit rounds=$ROUNDS start=$(date -Is)"
  python -u run_submission_extensions_v5.py \
    --dataset default_credit \
    --alphas 0.5 0.1 \
    --seeds 42 123 2026 31415 27182 \
    --epsilons 3 \
    --variants dp_uniform full_ha_dp \
    --rounds "$ROUNDS" \
    --metadata-epsilon 0.1 \
    --ha-config "$CONFIG" \
    --no-client-profiles \
    --output-prefix "round_sensitivity_${ROUNDS}r_v6" \
    --resume
done

python -u analyze_round_sensitivity_v6.py \
  --dataset default_credit \
  --raw-50 "$RAW50" \
  --raw-75 results/tables/default_credit_round_sensitivity_75r_v6_raw.csv \
  --raw-100 results/tables/default_credit_round_sensitivity_100r_v6_raw.csv \
  --alphas 0.5 0.1 \
  --seeds 42 123 2026 31415 27182 \
  --output-prefix round_sensitivity_v6

echo "[Job] completed=$(date -Is)"
