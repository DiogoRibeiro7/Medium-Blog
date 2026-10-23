# Reproducibility audit — Parts 16–20

Audit date: 2026-10-22.

All five committed experiment scripts and archived result schemas were inspected.

## Confirmed problem and correction: Part 19

The exact Part 19 source model was re-executed locally with seed 20261018, 400 repetitions, and the same sample sizes. The previously archived CSV did not match the present script. Its nonoverlap full-target row also labelled the target as 2 rather than 2.5, while calculating a bias inconsistent with that label.

Corrected results include:

| Scenario and estimand | Correct target | Reproduced mean |
| --- | ---: | ---: |
| Measured modifier: transported | 2 | 2.00408922 |
| Omitted modifier: trial | 1 | 0.99575874 |
| Omitted modifier: transported | 2 (stipulated) | 0.99227935 |
| Nonoverlap: full target | 2.5 | 2.50243513 |
| Nonoverlap: support-restricted | 1.75 | 1.74986085 |

The archived Part 19 CSV and manuscript numerical references have been updated accordingly. The unobserved target modifier shift in the omitted-modifier scenario remains an analytical assumption; it is not recovered from measured target covariates.

## Parts 16–18 and 20

The scripts and archived CSVs have been inspected for their structure and causal targets, but **no completed independent Monte Carlo rerun is certified here**. These parts must remain pending until the exact scripts have been executed in a controlled environment and results compared against the archive. Do not report the full-series audit as completed based on this PR.

## Next steps

Run the existing reproduction checker on Parts 16–20, check every generated CSV column, and resolve any discrepancies before publication. Then proceed with manuscript-to-result crosschecks and bibliography verification.
