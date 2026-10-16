# Predictive comparison: pre-registered protocol

Written and committed before any held-out prediction of this comparison was computed,
Bayesian or classical. Changes made after the first held-out computation are listed in
the change log at the end, with reasons.

## 1. Questions

1. Out of sample, does the hierarchical Bayesian model of
   [BAYES_PROTOCOL.md](BAYES_PROTOCOL.md) predict quarterly changes in the unemployment
   rate better than classical country-by-country regressions with the same regressors?
2. If it does, which part is responsible: partial pooling (`gaussian_iid` against OLS,
   same mean equation and Gaussian errors) or the error model (`primary` against
   `gaussian_iid`: Student-t AR(1) errors and a common factor)?
3. Once the quarter's GDP growth is known, does the Okun relation predict better than
   assuming no change in unemployment?

This is predictive accuracy. It says nothing about causal effects. Better predictions
from a model are not evidence that its coefficients are structural.

## 2. Data, information set and evaluation sets

- **Snapshot.** The frozen snapshot `snapshots/eurostat-2026-10-15` (manifest SHA-256
  `8a8c8a10…`), the same as for the Bayesian estimates.
- **Target.** `y[i,t]`, the change in country *i*'s unemployment rate from quarter t-1
  to quarter t, in percentage points.
- **Information set: a conditional prediction.** The prediction for quarter t may use
  every permitted observation up to t-1, plus real GDP growth in quarter t and its lags.
  The pre-registered models describe unemployment changes *given* GDP growth in the same
  quarter, so this is the setting in which they can be evaluated.
  - These are not forecasts that could have been made at the end of t-1: GDP growth for
    t is treated as known.
  - All values are revised data. Release lags and data vintages are not reconstructed.
- **Evaluation sets.** Both have 16 target quarters. Each was chosen as the longest set
  that leaves at least 27 training quarters, or that lies entirely after the pandemic.

  | Set | Target quarters | Training data for target t |
  | --- | --- | --- |
  | **E1, pre-pandemic** | 2016-Q1 to 2019-Q4 | 2009-Q2 to t-1 (27 to 42 quarters) |
  | **E2, post-pandemic** | 2022-Q3 to 2026-Q2 | 2009-Q2 to 2019-Q4, plus 2022-Q3 to t-1 as a second segment |

  No value from 2020-Q1 to 2021-Q4 is used, either as an outcome or as lag history, as
  in the `pandemic_excluded` specification.
- **Expanding window.** Every model is re-estimated at every origin, using only
  observations strictly before the target quarter. Nothing is shuffled.
- **Eurostat flags.** Flags follow the masking rules of BAYES_PROTOCOL section 2.
  - A cell `(i, t)` is scored only if its own likelihood term is unmasked and the change
    in quarter t-1 is usable, since the lagged-change models below need it.
  - This leaves 418 of 432 cells in E1 and 396 of 432 in E2.
  - ES and FR cannot be scored anywhere in E2, because their unemployment series carry
    "definition differs" flags from 2021.
  - Every model is scored on the same cells.
  - Training rows follow the same masks, and lagged-change models also need a usable
    previous change. Nothing is imputed.

## 3. Models (fixed in advance)

### Bayesian

These are three of the pre-registered specifications, with priors and likelihoods
unchanged:

| Name | Specification | Why it is included |
| --- | --- | --- |
| `bayes_primary` | `primary` | The headline model. |
| `bayes_no_common_factor` | `no_common_factor` | Isolates the common factor (descriptive only). |
| `bayes_gaussian_iid` | `gaussian_iid` | Partial pooling with the classical mean equation and Gaussian i.i.d. errors. |

The predictive distribution is the posterior predictive of `y[i,t]` given the training
data and GDP growth in t. For each posterior draw:

```text
y[i,t] = alpha[i] + theta[i]*g[i,t] + sum_k b[i,k]*(g[i,t-k] - g[i,t])
         + lambda[i]*f_new + phi[i]*e[i,t-1] + scale[i]*eps
```

- `f_new` is the common factor in the target quarter. It is unknown, so it is drawn
  from its N(0, 1) prior, with one value shared by all countries.
- `e[i,t-1]` is the previous quarter's residual, computed with that draw's in-sample
  factor value.
- At the first quarter of a segment there is no AR(1) term, and
  `scale = sigma / sqrt(1 - phi^2)`. Elsewhere `scale = sigma`.
- `eps` is Student-t(nu) or standard Normal, according to the specification.

### Classical

Each classical model is estimated by OLS, country by country, so there is no pooling.
The predictive distribution is the standard OLS predictive distribution: Student-t with
n - k degrees of freedom, location `x'b`, and scale `s * sqrt(1 + x'(X'X)^{-1} x)`. It
is exact under Gaussian i.i.d. errors.

| Name | Regressors | Role |
| --- | --- | --- |
| `ols_dl` | 1, g[t], g[t-1], g[t-2] | Classical Okun distributed lag; the same mean equation as the Bayesian models. |
| `ardl` | 1, y[t-1], g[t], g[t-1], g[t-2] | Classical model with the same information as the Bayesian AR(1) models. |

### Benchmarks without target-quarter GDP

| Name | Predictive distribution |
| --- | --- |
| `zero_change` | Student-t with n degrees of freedom, location 0 and scale `s0`. `s0^2` is the mean squared training change. |
| `historical_mean` | OLS on a constant. |
| `unemployment_ar1` | OLS on 1 and y[t-1]. |

## 4. Computation

- **Sampler settings.** NUTS settings, acceptance criteria and the single retry with
  3000 warm-up iterations are those of BAYES_PROTOCOL section 6: 4 chains, 1000 warm-up
  and 1000 retained draws, target acceptance 0.95, maximum tree depth 10, seed 20261016.
- **Failed refits.** A refit that still fails after the retry is reported, and its
  predictions are scored in the primary analysis. A forecaster at that origin would
  still have issued a prediction, and convergence is known before the target quarter
  is observed. A sensitivity analysis drops every target quarter in which any Bayesian
  refit failed, for all models.
- **Predictive samples.** Each of the 4000 posterior draws is combined with 4 draws of
  the innovation (and of `f_new`), giving 16,000 samples per cell. The seed is fixed per
  origin.
- **Expected cost.** About 2 CPU-hours for 3 specifications × 32 origins. Like the
  Bayesian fits, this runs in a separate workflow and never in ordinary pull-request
  checks.

## 5. Scores

- **Primary score:** CRPS, in percentage points (lower is better).
  - It is a proper score for the whole predictive distribution, and reduces to the
    absolute error for a point forecast.
  - It is computed in closed form for the Student-t predictive distributions, and with
    the empirical-distribution estimator for Bayesian samples.
- **Secondary, descriptive scores:**
  - log predictive density. For Bayesian models it is a mixture over posterior draws,
    with the common factor integrated by 40-node Gauss-Hermite quadrature.
  - RMSE of the predictive mean.
  - MAE of the predictive median.
  - empirical coverage of the central 50% and 90% predictive intervals.
  - PIT histograms.

## 6. Tests and decision rules

**The test.** Models A and B are compared through the panel loss differential.

- For each target quarter t, `d[t]` is the mean over the countries scored in t of
  `CRPS_A - CRPS_B`.
- Averaging within the quarter first makes the test valid under any cross-country
  dependence. Serial dependence in `d[t]` is handled with a Newey-West variance
  (Bartlett kernel, 2 lags).
- The statistic gets the Harvey-Leybourne-Newbold small-sample correction for one-step
  predictions, and is compared with a t distribution with T - 1 = 15 degrees of freedom.
  The test is two-sided.
- The effect size is the ratio of the mean CRPS of the two models over the quarters.

**The pre-registered family.** Five comparisons in each evaluation set, ten tests in
total:

| # | Comparison | Question |
| --- | --- | --- |
| 1 | `bayes_primary` vs `ols_dl` | 1: hierarchical model against classical regressions |
| 2 | `bayes_gaussian_iid` vs `ols_dl` | 2: partial pooling alone |
| 3 | `bayes_primary` vs `bayes_gaussian_iid` | 2: the error model |
| 4 | `bayes_primary` vs `ardl` | 1: against the classical model with the same information |
| 5 | `bayes_primary` vs `zero_change` | 3: the Okun relation against no change |

**Verdicts.**

- The ten p-values are Holm-adjusted together, at a 5% familywise level.
- A comparison is "A better" or "B better" when the adjusted p-value is below 0.05,
  according to the sign of the mean differential. Otherwise it is "not distinguishable".
- With 16 quarters per set, power is low. "Not distinguishable" never means "equally
  accurate".

**Everything else is descriptive.** This covers other pairs and scores, coverage and
PIT, and the sensitivity without failed refits. In particular:

- Country results appear only in alphabetical tables and as counts of countries.
- There are no per-country significance statements and no country rankings.

**Expectation, stated before evaluation.** Partial pooling should help most where
training samples are short (E1) and should hurt where country relations differ
genuinely. Results that contradict this expectation are reported as contradictions.

## 7. Not covered here

- Real-time forecasting. It needs GDP nowcasts or forecasts and vintage data.
- Multivariate (joint) scores across countries.
- Time-varying coefficients and the misfits found by the posterior predictive checks
  (BAYES_PROTOCOL change log). These are separate milestones.

## Change log

None yet.
