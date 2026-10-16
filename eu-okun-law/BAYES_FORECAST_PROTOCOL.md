# Hierarchical Bayesian forecasts against the classical forecasting models: protocol

Written and committed before any Bayesian forecasting model was fitted and before any
forecast in this comparison was scored. Changes after the first fit are listed in the
change log at the end, with reasons.

## 1. Questions

1. Do hierarchical Bayesian forecasting models give better one-quarter-ahead forecasts of
   the change in the unemployment rate than the classical models of the forecasting
   framework (PR #149, `FORECAST_VALIDATION.md`)? The classical models are persistence,
   the historical mean, AR(1) and the GDP-lag regressions.
2. Are their predictive distributions calibrated? This covers interval coverage and PIT,
   not only point accuracy.
3. How much does forecasting performance differ across countries?

This is predictive accuracy. Better forecasts are not evidence of causal effects, and a
forecast improvement is kept separate from its statistical significance.

## 2. Information set and data

- **Target and timing.** The target is `y[i,t]`, the change in country *i*'s unemployment
  rate from t-1 to t, in percentage points. The forecast is issued at the end of quarter
  t-1.
- **Information set.** This is the framework's own definition. It contains unemployment
  and participation up to t-1 and real GDP growth up to t-1. **No value dated t is
  used**, so there is no contemporaneous GDP or participation.
  - The assumption that GDP growth for t-1 is known at the end of t-1 is optimistic.
- **Retrospective, not real-time.** Every forecast uses the revised values of the frozen
  snapshot `snapshots/eurostat-2026-10-15`. Release lags and historical data vintages are
  not reconstructed. This is a retrospective, pseudo-out-of-sample evaluation, not a
  real-time forecast, which would require Eurostat data as published at each origin.
- **Sample.** This is the framework's sample, unchanged:
  - training starts at 2009-Q1;
  - the common complete-case quarters across all 27 countries run from 2009-Q3 to
    2026-Q2 (68 quarters, contiguous);
  - there are 34 forecast origins, with targets from 2018-Q1 to 2026-Q2;
  - for each target, every model is trained on all common quarters strictly before it,
    which gives 34 to 67 quarters.
- **Identical across models.** Origins, training rows and targets are the same for every
  model. As in the framework, Eurostat flags are not used, and the pandemic quarters are
  included in training and in the targets.

## 3. Models (fixed in advance)

### Classical models (PR #149, unchanged)

The six framework models are `zero_change` (persistence), `historical_mean`,
`unemployment_ar1`, `gdp_lags`, `gdp_unemployment_ar1` and `gdp_ar1_participation`. Their
point forecasts are exactly those of `eu-okun forecast`.

Their predictive distributions:

- **The regressions:** the OLS Student-t predictive with n - k degrees of freedom,
  located at the framework's point forecast.
- **`zero_change`:** Student-t with n degrees of freedom, location 0 and scale
  `sqrt(mean y^2)`.

### Hierarchical Bayesian models (new)

The parameters of all 27 countries are partially pooled:

```text
y[i,t] ~ Student-t(nu, alpha[i] + x[i,t]' beta[i] [+ lambda[i] f[t]], sigma[i])
alpha[i] = mu_alpha + tau_alpha z[i];   beta[i,j] = mu_j + tau_j z[i,j];
log sigma[i] = mu_s + tau_s z_s[i]      (non-centred)
```

| Name | Predictors `x[i,t]` (all dated t-1 or earlier) | Common factor |
| --- | --- | --- |
| `bayes_ar1` | du[t-1] | no |
| `bayes_gdp_ar1` | du[t-1], g[t-1], g[t-2] | no |
| `bayes_gdp_ar1_factor` | du[t-1], g[t-1], g[t-2] | yes |

**Priors.** These are the analogous priors of BAYES_PROTOCOL.md:

| Parameter | Prior |
| --- | --- |
| `mu_alpha`, `tau_alpha` | N(0, 0.25), HalfNormal(0.25) |
| `mu_j`, `tau_j` | N(0, 0.5), HalfNormal(0.25) |
| `mu_s`, `tau_s` | N(log 0.3, 1), HalfNormal(0.5) |
| `nu` | Gamma(2, 0.1) |
| `lambda[i] >= 0` | HalfNormal(0.2) |
| `f[t]` | N(0, 1) |

**Forecasts.** The forecast for quarter t draws the unknown factor value `f[t]` from its
prior, with one value shared by all countries.

## 4. Computation

- **Refits.** Every Bayesian model is refitted at every origin: 3 models × 34 origins =
  102 refits, about one to two CPU-hours. Full expanding-window refitting is affordable,
  so no subset of cutoffs is needed.
- **Sampler.** 4 chains, 1000 warm-up and 1000 retained draws, target acceptance 0.95,
  maximum tree depth 10, seed 20261016.
- **Acceptance and retry.** The acceptance criteria and the single retry with 3000
  warm-up iterations are those of BAYES_PROTOCOL.md, section 6.
- **Failed refits.** A refit that still fails is reported and its forecasts are scored.
  A sensitivity analysis drops its target quarters for all models.
- **Predictive samples.** Each of the 4000 posterior draws is combined with 4 innovation
  draws (and, for the factor model, factor draws). The seed is fixed per origin.
- **Records.** For every refit: runtime, attempts, diagnostics, training window
  (first and last quarter and count) and the snapshot hashes.

## 5. Scores and skill

- **Primary score:** CRPS, in percentage points. It is computed in closed form for
  Student-t predictive distributions and from samples for the Bayesian models.
- **Secondary scores:**
  - MAE of the predictive median;
  - RMSE of the predictive mean;
  - bias, defined as the mean of the predictive mean minus the outcome;
  - log predictive density;
  - empirical coverage of the central 50% and 90% intervals;
  - PIT histograms.
- **Skill.** Skill is `1 - score / score_ref` for CRPS and MAE, with persistence
  (`zero_change`) and `unemployment_ar1` as references.
- **Aggregation.** Scores are averaged over countries within each target quarter, then
  over target quarters.

## 6. Uncertainty in model differences

**The resampling.** A synchronised circular block bootstrap over target quarters.

- All countries of a quarter stay together, which preserves cross-country dependence.
- The primary block length is 4; block length 8 is a sensitivity analysis.
- 999 resamples, seed 20261020.

**For each comparison of models A and B,** with `d[t]` the cross-country mean CRPS
difference A - B in quarter t:

- the mean of `d[t]`, with its 95% percentile interval;
- the ratio of mean CRPS;
- the null-centred bootstrap p-value `(1 + #{|mean* - mean| >= |mean|}) / 1000`.

**The pre-registered family.** CRPS, all 34 targets:

| # | Comparison | Question |
| --- | --- | --- |
| 1 | `bayes_gdp_ar1` vs `gdp_unemployment_ar1` | Hierarchical against classical, same predictors |
| 2 | `bayes_ar1` vs `unemployment_ar1` | Hierarchical against classical AR(1) |
| 3 | `bayes_gdp_ar1` vs `zero_change` | Against persistence |
| 4 | `bayes_gdp_ar1` vs `bayes_ar1` | Value of lagged GDP growth in the hierarchical model |
| 5 | `bayes_gdp_ar1_factor` vs `bayes_gdp_ar1` | The common factor |

**Verdicts.**

- The five p-values are Holm-adjusted at a 5% familywise level.
- A comparison is "A better" or "B better" by the sign of the mean difference when the
  adjusted p-value is below 0.05; otherwise it is "not distinguishable".

**Everything else is descriptive.** This covers:

- the other comparisons and scores;
- block length 8;
- the 24 targets outside 2020-Q1..2022-Q2;
- the sensitivity without failed refits.

With 34 target quarters, power is limited, and "not distinguishable" never means
"equally accurate".

## 7. Country heterogeneity

- **Per country:** CRPS and MAE skill relative to `unemployment_ar1` and to persistence,
  for every model, in alphabetical tables.
- **Counts:** the number of countries in which each Bayesian model beats its classical
  counterpart.
- **Bootstrap intervals:** for comparison 1, each country gets a block-bootstrap interval
  (block length 4). These are descriptive.
- **No per-country significance claims and no country rankings.**

## 8. Not covered

- Real-time forecasting with historical vintages.
- Multi-step horizons.
- Joint (multivariate) scores.
- The hierarchical Okun models of BAYES_PROTOCOL.md. They condition on GDP growth in the
  target quarter; their conditional evaluation is PREDICTIVE_PROTOCOL.md.

## Change log

None yet.
