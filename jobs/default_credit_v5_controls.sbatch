#!/bin/bash
#SBATCH --job-name=dc_v5_ctrl
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/logs/default_credit_v5_controls_%j.out
#SBATCH --error=results/logs/default_credit_v5_controls_%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/setup_env.sh"

CONFIG=results/tables/default_credit_ha_tuning_v4_selected.json
V4_RAW=results/tables/default_credit_epsilon_curve_v4_raw.csv
V5_CONFIRM=results/tables/default_credit_confirm_10seed_v5_raw.csv
test -f "$CONFIG" || { echo "Missing $CONFIG"; exit 1; }
test -f "$V4_RAW" || { echo "Missing $V4_RAW"; exit 1; }
test -f "$V5_CONFIRM" || { echo "Missing $V5_CONFIRM; finish the 10-seed job first"; exit 1; }
echo "[Job] id=$SLURM_JOB_ID host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -u run_submission_extensions_v5.py \
  --dataset default_credit \
  --alphas 0.5 0.1 \
  --seeds 73 101 909 4096 65537 \
  --epsilons 3 \
  --variants dp_uniform scheduler full_ha_dp front_loaded \
    dp_label_representative scheduler_label_representative noise_aware_ha_v5 \
  --rounds 50 \
  --metadata-epsilon 0.1 \
  --ha-config "$CONFIG" \
  --import-v4-raw "$V4_RAW" "$V5_CONFIRM" \
  --output-prefix controls_v5 \
  --resume
echo "[Job] completed=$(date -Is)"
