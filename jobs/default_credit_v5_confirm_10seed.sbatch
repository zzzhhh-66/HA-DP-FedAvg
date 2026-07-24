#!/bin/bash
#SBATCH --job-name=dc_v5_10seed
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/logs/default_credit_v5_confirm_10seed_%j.out
#SBATCH --error=results/logs/default_credit_v5_confirm_10seed_%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/setup_env.sh"

CONFIG=results/tables/default_credit_ha_tuning_v4_selected.json
V4_RAW=results/tables/default_credit_epsilon_curve_v4_raw.csv
test -f "$CONFIG" || { echo "Missing $CONFIG"; exit 1; }
test -f "$V4_RAW" || { echo "Missing $V4_RAW"; exit 1; }
echo "[Job] id=$SLURM_JOB_ID host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -u run_submission_extensions_v5.py \
  --dataset default_credit \
  --alphas 0.5 0.1 \
  --seeds 42 123 2026 31415 27182 73 101 909 4096 65537 \
  --epsilons 3 \
  --variants dp_uniform scheduler full_ha_dp \
  --rounds 50 \
  --metadata-epsilon 0.1 \
  --ha-config "$CONFIG" \
  --import-v4-raw "$V4_RAW" \
  --output-prefix confirm_10seed_v5 \
  --resume
echo "[Job] completed=$(date -Is)"
