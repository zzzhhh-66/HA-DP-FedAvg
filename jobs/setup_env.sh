#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-HyperET}"
if [[ -n "${CONDA_SH:-}" ]]; then
  source "$CONDA_SH"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "Conda initialization script not found." >&2
  echo "Set CONDA_SH=/path/to/conda.sh before submitting." >&2
  exit 1
fi

conda activate "$CONDA_ENV_NAME"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export TMPDIR="$REPO_ROOT/.tmp"

mkdir -p "$TMPDIR" results/logs results/tables results/predictions
