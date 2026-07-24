#!/bin/bash
#SBATCH --job-name=dc_ablation_v3
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/logs/default_credit_ablation_v3_%j.out
#SBATCH --error=results/logs/default_credit_ablation_v3_%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/setup_env.sh"

echo "[Job] id=$SLURM_JOB_ID host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -c "import torch; print('[Python] torch=', torch.__version__, 'cuda=', torch.cuda.is_available(), 'gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

python -u run_ablation_experiments_v3.py \
  --dataset default_credit \
  --alphas 0.5 0.1 \
  --seeds 42 123 2026 31415 27182 \
  --rounds 50 \
  --epsilon 3.0 \
  --metadata-epsilon 0.1 \
  --ha-strength 0.5 \
  --ha-prior-strength 20.0 \
  --balance-gamma 1.0 \
  --class-weighting balanced \
  --output-prefix ablation_v3 \
  --resume

echo "[Job] completed=$(date -Is)"
