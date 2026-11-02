# Computational reproducibility audit — Causal Inference Parts 16–20

Audit date: 2 November 2026.

All five original seeded simulation scripts were independently executed using Python 3.12, NumPy 2.3.5, SciPy 1.17.0 and scikit-learn 1.8.0. Across **40 published CSV rows**, every regenerated numeric cell matched the committed `results.csv` outputs. Each run took place in a temporary directory; the original scripts and committed results were not changed.

| Part | Experiment | Result rows | Reproduction |
| --- | --- | ---: | --- |
| 16 | Omitted-confounder bias sensitivity and oracle adjustment | 8 | Passed |
| 17 | Negative controls and placebo exposure under three confounding structures | 9 | Passed |
| 18 | Complete-case selection and outcome-missingness correction under MCAR, MAR and MNAR | 12 | Passed |
| 19 | Trial-to-target transportability, hidden effect modification and limited support | 7 | Passed |
| 20 | Capacity-constrained treatment targeting with separate train and test populations | 4 | Passed |

## Independent reproduction

From the repository root:

```bash
python verify_sensitivity_results.py
python verify_sensitivity_results.py --recompute
```

The first command checks unique experimental designs, replication counts, bias identities, meaningful probability ranges and Part 16's analytical omitted-confounder bias formula. It also checks Part 20's population-gain identity, `population gain = capacity × mean benefit`. The full mode requires NumPy, SciPy and scikit-learn; it executes each original Monte Carlo in a temporary directory and compares all output columns against the published CSVs to numerical tolerance `1e-8`.

## Scientific scope and cautions

- Part 16's algebraic bias relation is specific to its assumed linear data-generating model. Sensitivity analyses require defensible ranges for unobserved-confounder strength.
- Part 17 negative-control diagnostics can detect some confounding patterns but are not sufficient to prove exchangeability when tests do not reject their nulls.
- Part 18's missingness mechanisms determine whether selection adjustment is identified. Under MNAR, response weighting based only on measured variables is not generally sufficient.
- Part 19's transportability estimates depend on measuring relevant effect modifiers and maintaining support in the target population; limited support requires explicit restrictions or extrapolation assumptions.
- Part 20's ranking exercise is evaluated on independent simulated test subjects with a fixed treatment budget. Comparing policy benefit is different from estimating the population-average treatment effect.

## Series-wide status

The main `experiment.py` → `results.csv` outputs of Parts 1–20 have now been independently regenerated and compared via these audit batches: Parts 1–5, 6–10, 11 and 15, 12–14, and 16–20. This does **not** establish robustness under new random seeds or causal identification in real observational settings.

Part 13 also includes a separate `validation_results.csv` produced outside its main experiment workflow. That additional artifact remains unverified under these main-output audits and should be investigated separately.
