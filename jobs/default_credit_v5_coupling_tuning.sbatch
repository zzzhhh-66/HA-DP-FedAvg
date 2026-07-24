#!/bin/bash
#SBATCH --job-name=dc_v5_couple
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/logs/default_credit_v5_coupling_tuning_%j.out
#SBATCH --error=results/logs/default_credit_v5_coupling_tuning_%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/setup_env.sh"

CONFIG=results/tables/default_credit_ha_tuning_v4_selected.json
test -f "$CONFIG" || { echo "Missing $CONFIG"; exit 1; }
echo "[Job] id=$SLURM_JOB_ID host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -u run_scheduler_coupling_tuning_v5.py \
  --dataset default_credit \
  --alphas 0.5 0.1 \
  --dev-seeds 7 11 \
  --rounds 30 \
  --epsilon 3 \
  --metadata-epsilon 0.1 \
  --ha-config "$CONFIG" \
  --schedule-powers 0.5 1.0 2.0 \
  --couplings 0.5 1.0 \
  --output-prefix scheduler_coupling_tuning_v5 \
  --resume
echo "[Job] completed=$(date -Is)"
