#!/bin/bash
#SBATCH --job-name=dc_ablate_v6
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/logs/default_credit_final_ablation_v6_%j.out
#SBATCH --error=results/logs/default_credit_final_ablation_v6_%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/setup_env.sh"

CONFIG=results/tables/default_credit_ha_tuning_v4_selected.json
test -f "$CONFIG" || { echo "Missing $CONFIG"; exit 1; }

echo "[Job] id=$SLURM_JOB_ID host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -u run_submission_extensions_v5.py \
  --dataset default_credit \
  --alphas 0.5 0.1 \
  --seeds 42 123 2026 31415 27182 73 101 909 4096 65537 \
  --epsilons 3 \
  --variants dp_uniform scheduler ha_uniform full_ha_dp \
  --rounds 50 \
  --metadata-epsilon 0.1 \
  --ha-config "$CONFIG" \
  --output-prefix final_ablation_10seed_v6 \
  --resume
echo "[Job] completed=$(date -Is)"
