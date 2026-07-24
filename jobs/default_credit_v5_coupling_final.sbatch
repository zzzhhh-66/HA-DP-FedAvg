#!/bin/bash
#SBATCH --job-name=dc_v5_cplfin
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/logs/default_credit_v5_coupling_final_%j.out
#SBATCH --error=results/logs/default_credit_v5_coupling_final_%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/setup_env.sh"

HA_CONFIG=results/tables/default_credit_ha_tuning_v4_selected.json
COUPLING_CONFIG=results/tables/default_credit_scheduler_coupling_tuning_v5_selected.json
V4_RAW=results/tables/default_credit_epsilon_curve_v4_raw.csv
V5_CONFIRM=results/tables/default_credit_confirm_10seed_v5_raw.csv
test -f "$HA_CONFIG" || { echo "Missing $HA_CONFIG"; exit 1; }
test -f "$COUPLING_CONFIG" || { echo "Missing $COUPLING_CONFIG"; exit 1; }
test -f "$V4_RAW" || { echo "Missing $V4_RAW"; exit 1; }
test -f "$V5_CONFIRM" || { echo "Missing $V5_CONFIRM; finish the 10-seed job first"; exit 1; }
read -r SUPPORTED POWER COUPLING < <(
  python -c "import json; d=json.load(open('$COUPLING_CONFIG')); print(d['coupling_supported_on_development'], d['schedule_power'], d['schedule_ha_coupling'])"
)
if [[ "$SUPPORTED" != "True" ]]; then
  echo "[Stop] Development validation did not support coupling; do not test-tune it."
  exit 0
fi
echo "[Selected] power=$POWER coupling=$COUPLING"
echo "[Job] id=$SLURM_JOB_ID host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -u run_submission_extensions_v5.py \
  --dataset default_credit \
  --alphas 0.5 0.1 \
  --seeds 73 101 909 4096 65537 \
  --epsilons 3 \
  --variants full_ha_dp coupled_ha_scheduler_v5 \
  --rounds 50 \
  --metadata-epsilon 0.1 \
  --ha-config "$HA_CONFIG" \
  --schedule-power "$POWER" \
  --schedule-ha-coupling "$COUPLING" \
  --import-v4-raw "$V4_RAW" "$V5_CONFIRM" \
  --output-prefix coupling_final_v5 \
  --resume
echo "[Job] completed=$(date -Is)"
