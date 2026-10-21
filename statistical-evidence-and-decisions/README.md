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

## Robustness notebook: imperfect observational data

[robustness-review.ipynb](robustness-review.ipynb) extends the initial
comparison with paired stress tests under missing predictors, training-response
contamination and different training sample sizes. The independent test set
remains untouched except for missing *observed* predictors when appropriate.

- MCAR: independent 20% masking in 15 predictors; one anchor remains observed.
- MAR: missingness depends on the always-observed third predictor.
- Outliers: 10% of training outcomes receive a symmetric +/-12-unit gross error.
- Combined MAR and training-label contamination.
- OLS, ridge, elastic net, Huber, PCR and PLS each perform fold-local median
  imputation with missingness indicators and standardisation.
- Validation-based choice by observed RMSE versus observed net payoff.
- Paired replicate differences from clean-data regret, with bootstrap intervals
  over independent simulation replicates (not test-set rows).

The synthetic conditional mean, retained to score decisions, is available only
to the test evaluator. Missing-data regret is measured relative to a
full-information oracle; it includes the irreducible loss of unobserved
predictors. Validation outcome contamination can distort both selection rules.
These comparisons do not identify causal policy effects.

Run with the existing requirements:

    STRESS_REPEATS=1 STRESS_SIZES=80 jupyter nbconvert --execute \
        --to notebook --output robustness-review.executed.ipynb robustness-review.ipynb

For the full experiment, use the notebook defaults (6 replicates and sample
sizes 80, 180, 360) or configure STRESS_REPEATS/STRESS_SIZES. The smoke values
are for verification, not statistical interpretation.

The second notebook is tracked in the same path-scoped GitHub Actions workflow.
Future work includes MNAR sensitivity, explicit missingness models, nested
selection, nonlinear effects and uncertainty in intervention costs.

## Cost uncertainty and model-selection stability

[model-selection-and-costs.ipynb](model-selection-and-costs.ipynb) extends the
first two notebooks to distinguish the *information used to fit a model*
from the *decision objective used to select it*. It compares:

- Selection using a single validation split versus repeated overlapping
  holdout splits (three by default).
- Lowest observed validation RMSE versus maximum observed validation net
  utility at a nominal cost and averaged across an equally weighted cost grid.
- Clean training data versus MAR predictors plus contaminated training labels.
- Low-variance versus high-variance response signals.

All transformations and hyperparameter tuning happen within each internal
training split. After selection, the winning candidate is refitted on the full
training data, and independent test observations are used to evaluate both
RMSE and economic regret. The synthetic full-information conditional mean
never enters selection.

Costs are assumed uncertain at model-selection time but *known when the
intervention decision is made*. The action threshold therefore changes with
the realised cost. A commitment required before the cost is known would need
a different decision rule. This is not a causal treatment-effect estimate.

Use the existing project requirements. To execute a small smoke test:

    SELECTION_REPEATS=1 SELECTION_SPLITS=2 SELECTION_SIZES=80 \
      SELECTION_SMOKE=1 jupyter nbconvert --execute --to notebook \
      --output model-selection-and-costs.executed.ipynb \
      model-selection-and-costs.ipynb

The notebook defaults to six independent simulation replicates, three
overlapping validation splits and 500 independent test observations per
scenario. Six independent replicates are exploratory; increase this
substantially before drawing conclusions from bootstrap uncertainty.
The independent Monte Carlo replicate—not the repeated validation split,
test observation or cost value—is the appropriate resampling unit.

Code and tests are in [cost_selection.py](cost_selection.py) and
[tests/test_cost_selection.py](tests/test_cost_selection.py).

## Quantifying uncertainty at decision time

[uncertainty-and-decisions.ipynb](uncertainty-and-decisions.ipynb) examines
uncertainty about *coefficients*, *predictions*, *model choice* and *actions*.
The analysis is in [uncertainty_analysis.py](uncertainty_analysis.py), with
invariant tests in [tests/test_uncertainty_analysis.py](tests/test_uncertainty_analysis.py).

- Bootstrap resamples are fitted in-bag, and candidates are compared on
  strictly out-of-bag observations. Duplicated in-bag records therefore cannot
  also serve as validation outcomes.
- Algorithms use pre-specified hyperparameters, not CV on duplicate-resampled
  datasets. Candidate pipelines keep all transformations within each fit.
- Four uncertainty policies are studied: fixed OLS, fixed ridge, OOB selection
  by MSE and OOB selection by observed cost-grid economic utility.
- The law of total variance partitions bootstrap predictive variability
  *descriptively* into within-selected-algorithm and between-algorithm-label
  terms. Neither term is a causal effect of model switching.
- Analytic versus bootstrap coefficient intervals are shown only for an
  *unselected*, complete-data OLS model.
- Conditional-mean uncertainty and observation noise are evaluated separately
  using independent synthetic test outcomes. The observation-noise parameter
  is given by the simulation and never estimated from test observations.
- Mean, majority-vote and lower-quantile intervention rules are compared using
  economic regret, overall action frequency and unprofitable actions.

These are frequentist resampling diagnostics, not posterior probabilities or
guarantees of nominal interval coverage. The known full-information conditional
mean is used only for final evaluation. The MAR scenario is compared against
the unattainable full-information oracle and is therefore not directly a pure
model-quality measure.

For a short reproducibility smoke execution:

    UNCERTAINTY_BOOTSTRAPS=6 UNCERTAINTY_N_TRAIN=80 \
      UNCERTAINTY_N_TEST=40 jupyter nbconvert --execute --to notebook \
      --output uncertainty-and-decisions.executed.ipynb \
      uncertainty-and-decisions.ipynb

The full notebook defaults to 60 bootstrap replicates and 300 test observations.
These are illustrations: for reliable coverage or selection-frequency claims,
repeat the entire experiment with many independent generated datasets.

## Calibration and independent coverage testing

[calibration-and-decisions.ipynb](calibration-and-decisions.ipynb) examines
empirical nominal coverage across independently regenerated simulation datasets.

- A held-out training/selection partition, independent calibration partition
  and independent external test set are used throughout.
- Gaussian training-residual plug-in intervals are contrasted with
  finite-sample-corrected split-conformal prediction intervals.
- The split-conformal guarantee concerns **marginal coverage of observed Y
  under exchangeability**, not individual conditional coverage, the regression
  mean, treatment effects or profits.
- Experiments vary calibration size, nominal miscoverage, signal mechanism,
  MAR predictor missingness and intentional test-noise inflation.
- Marginal and observed-anchor subgroup coverage are reported separately.
- Economic policies compare point-threshold action, conservative interval
  screening, and **oracle-assisted** review with explicit review costs.
  The final policy is an optimistic benchmark, not a practical reviewer.
- Coverage uncertainty resamples independent generated datasets, not test
  rows that share a fitted model or calibration interval.

The code and tests are in [calibration_analysis.py](calibration_analysis.py)
and [tests/test_calibration_analysis.py](tests/test_calibration_analysis.py).

Smoke run:

    CALIBRATION_SMOKE=1 CALIBRATION_REPEATS=1 CALIBRATION_N_FIT=80 \
      CALIBRATION_N_CAL=30 CALIBRATION_N_TEST=40 \
      jupyter nbconvert --execute --to notebook \
      --output calibration-and-decisions.executed.ipynb \
      calibration-and-decisions.ipynb

The full notebook defaults to 12 independent replicates and requires a
larger number of independent experiments before reliable coverage conclusions
can be made. It must not be confused with a causal policy evaluation.

## Temporal drift and prequential calibration

[temporal-drift-and-decisions.ipynb](temporal-drift-and-decisions.ipynb)
investigates independent chronological deployment streams under stable,
covariate-shift, noise-inflation, and concept-shift regimes. It compares
frozen/static conformal intervals against rolling residual recalibration,
with and without periodic model refitting. Every prediction, interval and
action is recorded *before* its observed outcome is revealed.

Source: [temporal_drift.py](temporal_drift.py). Regression tests:
[tests/test_temporal_drift.py](tests/test_temporal_drift.py).
The project-scoped CI workflow executes a small notebook smoke run.

This is not a formal time-series conformal guarantee. Only the initial
calibration under exchangeability has standard marginal coverage theory.
The rolling methods are heuristics, and the changed regimes intentionally
violate that assumption. Bootstrap coverage summaries sample independent
generated streams, never individual deployment time steps. The economic
oracle is available only to the evaluator; decisions are not causal claims.

Smoke reproduction:

    DRIFT_SMOKE=1 DRIFT_REPEATS=1 jupyter nbconvert --execute --to notebook \
      --output temporal-drift-and-decisions.executed.ipynb \
      temporal-drift-and-decisions.ipynb
