#!/bin/bash
#SBATCH --job-name=dc_refs_v6
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/logs/default_credit_reference_v6_%j.out
#SBATCH --error=results/logs/default_credit_reference_v6_%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/setup_env.sh"

echo "[Job] id=$SLURM_JOB_ID host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -u run_reference_baselines_v6.py \
  --dataset default_credit \
  --alphas 0.5 0.1 \
  --seeds 42 123 2026 31415 27182 73 101 909 4096 65537 \
  --variants fedavg_iid fedavg_noniid fedprox_noniid centralized_mlp \
             logistic_regression hist_gbdt xgboost \
  --rounds 50 \
  --centralized-epochs 30 \
  --prox-mu 0.01 \
  --resume
echo "[Job] completed=$(date -Is)"
