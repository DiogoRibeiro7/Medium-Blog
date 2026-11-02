# Part 13 — Independent nuisance-basis validation audit

Date: 2 November 2026.

The separate `validate_nuisance_basis.py` script was independently run with Python 3.12 and NumPy 2.3.5, using its fixed seed `20261018`. It generated 1,000 independent simulation replications per design, each with 800 observations. Every cell of the six-row `validation_results.csv` matched the committed artifact to numerical tolerance `1e-8`.

| Design | Linear basis | Correct nonlinear basis | Oracle nuisance means |
| --- | ---: | ---: | ---: |
| Linear: estimated causal effect | 1.0017357429 | 1.0018237370 | 1.0016730987 |
| Nonlinear: estimated causal effect | 1.9509332798 | 1.0001007315 | 1.0001402385 |

The structural treatment effect is 1.0. The nonlinear design demonstrates substantial bias when the nuisance regression uses an insufficient linear basis, while the correctly expanded basis and oracle conditional means recover the effect under the simulated assumptions.

## Independent check

```bash
python verify_part13_nuisance_validation.py
python verify_part13_nuisance_validation.py --recompute
```

The default command uses the Python standard library to check schema, six expected design-method combinations, sample size, replication count, finite numeric values, the known treatment effect and the identities relating bias, standard deviation and RMSE. The `--recompute` mode requires NumPy and runs the source validation script in a temporary directory, comparing every generated cell to the committed table without overwriting the repository artifact.

## Interpretation

This validation supports the numerical claim that a misspecified nuisance basis can invalidate the intended orthogonalization even with cross-fitting. It does **not** establish that the selected basis is appropriate for observational data where nuisance functions and effect modifiers are unknown. Oracle performance represents knowledge available only within this simulation.

With this separate validation reproduced, every committed primary experiment table and this additional Part 13 validation table in the 20-article series have been checked through the independent computational audit. Further scientific review should examine causal assumptions, equations, figure-to-table consistency and sensitivity beyond fixed seeds.
