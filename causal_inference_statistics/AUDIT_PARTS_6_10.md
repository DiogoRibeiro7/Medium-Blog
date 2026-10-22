# Reproducibility check: Parts 6–7

Date: 2026-10-22. The source scripts and archived CSVs were inspected from `main`. Independent local execution reconstructed their published data-generating logic and used the original NumPy RNG seed and sampling order.

## Part 6: PASS

All three Difference-in-Differences scenarios reproduce the archived Monte Carlo metrics at machine precision (mean, bias, sample SD, mean SE, coverage, placebo mean and placebo rejection rate). In particular: parallel mean -0.9951334883981987 and coverage 0.954; diverging post-trend mean -0.24685915641048692; anticipation mean -0.5977422112290203.

## Part 7: FAIL — investigate before changing results

The first (strong valid) instrument scenario exactly matches the archived mean OLS 1.3323106385483539 and IV 0.9983553815989099. Subsequent scenarios diverge despite preserving a single RNG stream and the published structural formula, suggesting an archived-results/script-history mismatch requiring a fresh run of the exact committed script.

Selected independently reproduced values:

| Design | Method | Archived mean | Independent mean | Archived RMSE | Independent RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| Valid strong | IV | 0.9983554 | 0.9983554 | 0.0434282 | 0.0434282 |
| Valid moderate | IV | 0.9862815 | 0.9910485 | 0.1628468 | 0.1525208 |
| Valid weak | IV | 0.4766103 | 1.1289835 | 8.7481763 | 3.2619814 |
| Invalid strong | IV | 1.2999088 | 1.2993055 | 0.3023362 | 0.3020675 |

Weak-IV mean/RMSE are especially sensitive to extreme sample ratios; this discrepancy is substantive for the numerical discussion. Do **not** treat the stored weak-IV results as independently reproduced until the exact script rerun and package versions have been verified.

## Remaining

Parts 8–10 have not been rerun in this PR. No publication clearance is claimed. Use `causal_inference_statistics/audit_reproducibility.py --parts 6 7 8 9 10` in a pinned environment and investigate rather than silently replacing archived results.
