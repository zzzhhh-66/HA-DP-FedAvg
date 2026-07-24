# Slurm jobs

Submit jobs from the repository root. Every job sources `jobs/setup_env.sh`,
which uses `SLURM_SUBMIT_DIR` instead of a user-specific filesystem path.

## Environment variables

The default Conda environment name is `HyperET`. Override it when needed:

```bash
export CONDA_ENV_NAME=my_environment
```

If Conda is installed in a nonstandard location:

```bash
export CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh
```

These variables must be exported before `sbatch`, or added to the local shell
profile.

## Cluster-specific directives

The provided scripts request:

```text
partition: gpu4090
GPU:       1
CPU:       8
memory:    32 GB
time:      48 hours
```

Change the `#SBATCH --partition` line to a locally available GPU partition.
The Python code does not require an RTX 4090 specifically.

## Monitoring

```bash
JOB=$(sbatch --parsable jobs/default_credit_v5_controls.sbatch)
squeue -j "$JOB"
tail -F "results/logs/default_credit_v5_controls_${JOB}.out"
```

The scripts use unbuffered Python output and write checkpoints after completed
configurations, so progress is visible and resumable.
