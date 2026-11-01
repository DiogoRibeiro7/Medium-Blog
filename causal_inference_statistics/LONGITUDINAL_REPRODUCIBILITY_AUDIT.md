# Longitudinal simulation reproducibility audit (Parts 11 and 15)

This audit independently reran the committed experiments on 1 November 2026 using Python 3.12, NumPy 2.3.5, SciPy 1.17.0 and scikit-learn 1.8.0.

## Structural target

The two experiments share a two-decision data-generating model, with the intermediate confounder satisfying `L1 = 0.8 L0 + 1.2 A0 + noise` and the outcome `Y = 1 + 0.5 L0 + 1.3 L1 + 0.7 A0 + 1.1 A1 + noise`. Under the sustained treatment interventions `(A0,A1)=(1,1)` versus `(0,0)`, the total contrast is

`0.7 + 1.1 + 1.3 × 1.2 = 3.36`.

This includes the mediated effect of the first treatment via `L1`. A conditional outcome regression coefficient sum estimates a different quantity and therefore cannot be interpreted as the total sustained-regime effect.

## Independently reproduced summaries

| Estimator | Part 11 (300 × 2,000) | Part 15 (160 × 2,200) |
| --- | ---: | ---: |
| Correct sequential g-formula | 3.3671417288 | 3.3570932694 |
| Inverse-probability weighted sustained regime | — | 3.3539826283 |
| Naive/direct conditional coefficient sum | 1.8065828581 | 1.7973158538 |
| Deliberately misspecified g-formula (omits L1) | 4.1142409288 | — |
| Effective sample size, treated regime (1,1) | — | 503.9524824272 |
| Effective sample size, untreated regime (0,0) | — | 299.4496171163 |

Every numeric cell in both generated tables matched the committed `results.csv` files on the local run. No result was changed as part of this audit.

## How to verify

Run from the repository root:

```bash
python verify_longitudinal_results.py
python verify_longitudinal_results.py --recompute
```

The first command uses only the Python standard library and validates estimator labels, simulation dimensions, the structural truth, bias and RMSE arithmetic, reported empirical quantiles and effective-sample-size fields. The second command additionally requires NumPy, SciPy and scikit-learn; it runs the complete simulations using temporary copies of the scripts and compares the regenerated CSVs to committed results to a numerical tolerance of `1e-8`. Existing committed outputs remain untouched.

## Interpretation and limits

- The deterministic match supports computational reproducibility of the published numbers in the tested environment. It does not, by itself, establish robustness to model misspecification, extreme weights, alternate data-generating processes or other software versions.
- Part 15 estimates the sustained-regime contrast with normalized inverse-probability weights, not a fitted marginal structural model coefficient from a separate weighted regression; the article should preserve that distinction.
- Both experiments employ correctly specified parametric nuisance models for the main estimators. Their relative performance in this demonstration should not be generalized to arbitrary observational studies.
