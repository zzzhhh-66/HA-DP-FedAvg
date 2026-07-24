#!/bin/bash
#SBATCH --job-name=dc_v5_secure
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/logs/default_credit_v5_secure_confirm_%j.out
#SBATCH --error=results/logs/default_credit_v5_secure_confirm_%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/setup_env.sh"

CONFIG=results/tables/default_credit_ha_tuning_v4_selected.json
test -f "$CONFIG" || { echo "Missing $CONFIG"; exit 1; }
python -c "import torchcsprng; print('[Secure RNG] torchcsprng import OK')" || {
  echo "Secure confirmation stopped: install a torchcsprng build compatible with HyperET."
  exit 2
}
echo "[Job] id=$SLURM_JOB_ID host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -u run_submission_extensions_v5.py \
  --dataset default_credit \
  --alphas 0.5 0.1 \
  --seeds 42 \
  --epsilons 3 \
  --variants dp_uniform full_ha_dp \
  --rounds 50 \
  --metadata-epsilon 0.1 \
  --ha-config "$CONFIG" \
  --secure-rng \
  --output-prefix secure_confirm_v5 \
  --resume
echo "[Job] completed=$(date -Is)"
