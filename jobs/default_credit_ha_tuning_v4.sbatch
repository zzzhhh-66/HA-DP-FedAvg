#!/bin/bash
#SBATCH --job-name=dc_ha_tune_v4
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/logs/default_credit_ha_tuning_v4_%j.out
#SBATCH --error=results/logs/default_credit_ha_tuning_v4_%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/setup_env.sh"

echo "[Job] id=$SLURM_JOB_ID host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -u profile_privacy_accounting_v4.py
python -u run_ha_tuning_v4.py \
  --dataset default_credit \
  --alphas 0.5 0.1 \
  --dev-seeds 7 11 \
  --rounds 30 \
  --epsilon 3.0 \
  --metadata-epsilon 0.1 \
  --ha-strengths 0.25 0.5 0.75 1.0 \
  --ha-priors 5 20 50 \
  --output-prefix ha_tuning_v4 \
  --resume
echo "[Job] completed=$(date -Is)"
