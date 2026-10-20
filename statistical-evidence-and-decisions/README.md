# Statistical evidence and decisions

A reproducible notebook study of a practical question: **can structural
diagnostics, regression inference, regularisation and validation be combined
to make better decisions than selecting variables by p-value or by PCA
explained variance alone?**

This is a research experiment, not a claim that any one method is universally
superior. It is separate from the historical `DimensionReduction/PCA.ipynb`.

## Questions

1. What does PCA reveal about predictor geometry that coefficient p-values do not?
2. When does 95%-variance principal-component regression discard the direction
   carrying almost all the response signal?
3. How do marginal coefficient tests, a joint F-test, VIF and bootstrap
   stability behave under near-collinearity?
4. How do OLS, naive p-value screening, ridge, elastic net, PCR and PLS compare
   on unseen observations?
5. Does choosing a prediction model on validation-set **decision utility**
   necessarily outperform choosing it on prediction error?

## Design

We generate eight independent latent factors, each measured by a correlated
pair of variables, and consider two data-generating mechanisms:

- **Low-variance signal:** `E[Y | X] = 10 (X1 - X2)`. Predictor differences
  have low variance, but their contrast determines the response.
- **High-variance signal:** `E[Y | X] = 1.2 (X1 + X2)`. The response follows a
  major predictor direction.

Both scenarios include irreducible observation noise. The conditional mean is
known to the simulation and is **used exclusively for evaluation**.

An action is taken when predicted benefit exceeds a fixed cost `c`. Its
expected net payoff is `(E[Y|X] - c) * 1(prediction > c)`. Regret is the
difference between the oracle's expected payoff and that of the model.
Because a decision can change discontinuously at the threshold, minimum RMSE
and minimum economic regret need not select the same estimator.

The benchmark includes a transparent, exploratory p-value screening baseline.
Its post-selection p-values do **not** support valid confirmatory inference.

A decision-guided baseline selects among candidate estimators using an
**internal training/validation split** and observed net benefit; the external
test set is reserved for evaluation. Tuning, scaling, and PCA occur inside
training pipelines. This is a first comparison, not yet a fully nested
selection-and-tuning design.

## Contents

- [`evidence-review.ipynb`](evidence-review.ipynb): mathematical rationale,
  structural diagnostics, joint tests, bootstrap stability, experiments,
  figures and interpretation.
- [`analysis.py`](analysis.py): typed data generation, estimators, policy
  metrics and repeated experiments.
- [`tests/test_analysis.py`](tests/test_analysis.py): deterministic
  scientific and shape invariants.
- [`requirements.txt`](requirements.txt): project-scoped dependencies.

## Reproduce

From this directory, with Python 3.12 or later:

```bash
python -m venv .venv
# Activate .venv for your shell.
python -m pip install -r requirements.txt
python -m pytest -q tests
jupyter lab evidence-review.ipynb
```

The notebook runs offline, downloads no data and uses fixed random seeds.
By default it evaluates 12 replicates per scenario and 150 bootstrap samples.
For a quick execution check:

```bash
EVIDENCE_REPEATS=1 EVIDENCE_BOOTSTRAPS=20 \
  jupyter nbconvert --execute --to notebook \
  --output evidence-review.executed.ipynb evidence-review.ipynb
```

The command above uses POSIX environment-variable syntax; set the variables
separately when working in PowerShell. Notebook outputs are left uncommitted.

## Interpretation rules

- Predictor variance is not predictive or causal importance.
- A conditional coefficient p-value is not a measure of effect size or an
  automatic feature-removal rule.
- Multicollinearity can undermine individual coefficient precision while a
  group is jointly informative.
- Bootstrap intervals here describe empirical coefficient stability; naive
  resampling does not repair invalid selection inference.
- Validation-set model choice can overfit validation noise. Compare it on
  independent test data, and eventually use nested/repeated resampling.
- Selecting a model on observed decision utility is not causal policy
  evaluation: a real intervention requires identifying causal effects and
  justifying the cost/utility function.
- Results from these synthetic scenarios must not be generalised to real
  agricultural, economic or healthcare data without additional experiments.

## Planned extensions

Introduce MCAR/MAR/MNAR missingness, contamination, nonlinear interactions,
different `p/n` ratios, grouping constraints, covariate shift, stability
selection and nested tuning. Then study what happens when utility functions
and intervention costs are uncertain.

## Selected references

- Jolliffe, I. T. (2002). *Principal Component Analysis*, 2nd ed. Springer.
- Hoerl, A. E., & Kennard, R. W. (1970). Ridge Regression: Biased Estimation
  for Nonorthogonal Problems. *Technometrics*, 12(1), 55–67.
- Frank, I. E., & Friedman, J. H. (1993). A Statistical View of Some
  Chemometrics Regression Tools. *Technometrics*, 35(2), 109–148.
- Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso.
  *Journal of the Royal Statistical Society B*, 58(1), 267–288.
- Zou, H., & Hastie, T. (2005). Regularization and Variable Selection via
  the Elastic Net. *Journal of the Royal Statistical Society B*, 67(2), 301–320.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.
  Chapman & Hall.
