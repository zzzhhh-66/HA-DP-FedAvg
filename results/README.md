# Included result artifacts

## `tables/final`

This directory contains the final paper-facing artifacts:

- ten-seed 2x2 privacy ablations for all three datasets;
- ten-seed reference baselines;
- 50/75/100-round sensitivity results for Default Credit and GMSC;
- per-method, client-profile, communication-cost, and factorial summaries;
- paired effects and the final protocol-integrity report.

Files ending in `_raw.csv` contain one row per completed configuration and
seed. Files ending in `_mean_std.csv` are descriptive summaries. Statistical
claims should be based on the paired raw rows and their associated paired
effects, not on mean values alone.

## `tables/diagnostics`

This directory contains development diagnostics used in the paper:

- the privacy-utility curve summary;
- probability-calibration and threshold report;
- the earlier four-component ablation summary.

## `figures`

These figures are generated from the included CSV files. They are not a
substitute for the underlying numeric artifacts.

No raw credit records, model checkpoints, or per-record prediction files are
included.
