# Release notes

## Version 1.0.0

This package is the public research-code release for HA-DP-FedAvg. It includes:

- the final model, privacy-accounting, client-partitioning, and training code;
- portable Slurm jobs for the final factorial experiments and reference baselines;
- ten-seed final result tables for all three benchmark datasets;
- 75/100-round sensitivity results for Default Credit and GMSC;
- statistical analysis, validation, plotting, and calibration scripts;
- the method overview figure and reproducibility documentation.

Raw benchmark datasets, prediction files, checkpoints, caches, and HPC logs are
excluded.

## Validation

- All 57 Python files pass syntax parsing.
- All repository-local Markdown links resolve.
- Eighteen dependency-light unit and statistical-analysis tests pass.
- The archived V6 integrity report is valid.
- Each privacy experiment has 80 unique rows: two Non-IID levels, four variants,
  and ten final seeds.
- Each reference-baseline experiment has 90 unique rows.
- Personal workstation and HPC absolute paths have been removed.

The complete training test suite requires PyTorch and Opacus and should be run
after installing `requirements-dev.txt`.

## License

This release is distributed under the MIT License.
