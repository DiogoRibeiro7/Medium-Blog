# Research design and identification

This is a research plan; conclusions must follow the data, not precede them.

## Notation and units

For country `i` and quarter `t`, let `g[i,t]` denote quarterly real GDP
growth in percent, `u[i,t]` the unemployment rate in percent of the labour
force ages 15–74 and `p[i,t]` participation in percent of resident population
ages 15–74. Define `du[i,t] = u[i,t] - u[i,t-1]` and
`dp[i,t] = p[i,t] - p[i,t-1]` in percentage points.

## Stage 1 — Testable classical baselines

A descriptive distributed-lag model is:

```text
du[i,t] = alpha[i] + sum(k=0..K) beta[i,k]*g[i,t-k]
          + gamma[i]*dp[i,t] + epsilon[i,t]
```

Initially `K=2`; contrast with `K=0,1,4`. Estimate both with and
without `dp`, since participation may be a response to GDP itself.
Use HAC (initially four lags) and inspect autocorrelation, structural breaks,
heterogeneity, missingness and influential observations. Compare country
models with pooled fixed-effects and naive forecasts. Validate in chronological
holdouts rather than shuffled cross-validation.

The sum of growth coefficients describes a conditional relationship. It
is *not* automatically the response to an exogenous GDP level shock.

### Review of the classical results (implemented)

`eu-okun review` ([docs/CLASSICAL_RESULTS_REVIEW.md](docs/CLASSICAL_RESULTS_REVIEW.md))
recomputes every published number by an independent route (QR solve, kernel-matrix
HAC on calendar distances, coefficient sums read off the reparameterised regression
on `g[t]` and `g[t-k] - g[t]`) and re-estimates the same specifications on fixed
alternative samples: pandemic excluded, 2009-2019, Eurostat-flag screened and
post-2021. Its main finding shapes everything that follows: 2020-2021 supply most of
the GDP variation in any sample containing them, so pandemic treatment is an explicit
design choice, never a default.

## Stage 2 — Hierarchical Bayesian estimation (implemented)

Pre-registered in [BAYES_PROTOCOL.md](BAYES_PROTOCOL.md) before any fit to Eurostat
data; implemented in `bayes_data.py`, `bayes_model.py`, `bayes_fit.py` and
`bayes_report.py`; results in [docs/HIERARCHICAL_BAYES.md](docs/HIERARCHICAL_BAYES.md).

```text
du[i,t] = alpha[i] + theta[i]*g[i,t] + sum_k b[i,k]*(g[i,t-k] - g[i,t])
          + lambda[i]*f[t] + e[i,t]
e[i,t]  = phi[i]*e[i,t-1] + sigma[i]*eps[i,t],   eps ~ Student-t(nu)
theta[i] ~ Normal(mu_theta, tau_theta)   (non-centred; likewise alpha, b, phi, log sigma)
```

- **Estimand.** The reparameterisation makes `theta[i]` the coefficient sum itself.
  `mu_theta` is the *unweighted* mean of the country sums: a hyperparameter, not a
  population-weighted EU coefficient and not a pooled regression slope.
- **Dependence.** A common quarterly factor `f[t]` with non-negative loadings captures
  residual co-movement across countries; country AR(1) errors capture serial
  dependence; Student-t errors and country scales handle heavy tails and the
  eightfold spread of residual scales.
- **Data.** Primary sample 2009-Q2..2019-Q4 (contiguous and definition-consistent for
  all 27 countries). Likelihood terms relying on flagged Eurostat values are masked,
  never altered or imputed.
- **Computation.** NUTS (NumPyro, float64), 4 chains x 1000 retained draws. A fit is
  accepted only with R-hat <= 1.01, bulk and tail ESS >= 400, no divergences and
  E-BFMI >= 0.3; failed fits are reported and not interpreted.
- **Reporting.** Country ranks only as posterior rank distributions and pairwise
  probabilities `P(theta[i] < theta[j])`; nine fixed sensitivity specifications
  (priors, pooling, common factor, error model, participation, lag order, sample).

### Amendment 1: shared quarter effects, prior predictive checks, recovery (implemented)

Registered in [BAYES_PROTOCOL.md](BAYES_PROTOCOL.md), section 9, before any of its fits;
implemented in `bayes_model.py`, `bayes_artifacts.py` and `bayes_checks.py`; results in
[docs/BAYES_MODEL_CHECKS.md](docs/BAYES_MODEL_CHECKS.md).

In the textbook notation

```text
du[i,t] = alpha[i] + sum(k=0..K) beta[i,k]*g[i,t-k] + gamma[i]*dp[i,t] + lambda[t] + e[i,t]
```

the models above use `beta[i,0] = theta[i] - sum_k b[i,k]` and `beta[i,k] = b[i,k]`, so the
coefficient sum `theta[i]` is a parameter. Units: `du`, `dp`, `alpha`, `lambda` and the
error scales in pp of the unemployment rate per quarter; `beta`, `theta` in pp per pp
of quarterly GDP growth; `gamma` in pp per pp of participation.

- **Shared quarter effects.** `lambda[t]` enters either as a common factor with
  country loadings (`lambda[i]*f[t]`, the pre-registered specifications) or, in the
  amendment, as an additive effect `delta[t]` common to all countries with a
  sum-to-zero constraint (`ZeroSumNormal`, scale `tau_delta ~ HalfNormal(0.25)`).
- **Identifiability.** The constraint separates `delta` from the intercepts. Because
  `delta[t]` absorbs everything common to a quarter, `theta[i]` then measures the
  association with country-specific GDP growth beyond the common quarter effect, and
  `mu_theta` is identified only from cross-country differences in growth within
  quarters. Factor and additive effects are never combined.
- **Design.** With/without shared quarter effects x with/without participation
  (`time_effects`, `time_effects_participation`, `no_common_factor`,
  `participation_no_common_factor`), compared by PSIS-LOO and the posterior predictive
  cross-country correlation.
- **Prior predictive checks.** 1000 prior draws per specification propagated to
  simulated data on the real design; quantiles of the simulated standard deviations,
  largest changes, autocorrelation and cross-country correlation against the observed.
- **Recovery.** One data set per specification simulated from parameters fixed in code
  (`mu_theta = -0.30`, `tau_theta = 0.15`) on the real design, refitted with the
  protocol settings; coverage of the 50% and 90% intervals.
- **Artifacts.** Every fit writes `posterior.nc` (ArviZ NetCDF: posterior,
  sample_stats, prior, prior_predictive, observed_data, constant_data, log_likelihood)
  with seeds, sampler settings, software versions and snapshot hashes.

### Predictive comparison (implemented)

Pre-registered in [PREDICTIVE_PROTOCOL.md](PREDICTIVE_PROTOCOL.md) before any held-out
prediction was computed; implemented in `predictive.py`, `bayes_predictive.py` and
`predictive_report.py`; results in
[docs/PREDICTIVE_COMPARISON.md](docs/PREDICTIVE_COMPARISON.md).

- **Task.** Predict `du[i,t]` from data up to t-1 plus `g[i,t]` and its lags: a
  *conditional* prediction, the setting the Okun models describe. It is not a real-time
  forecast (GDP growth for t is treated as known, revised data throughout).
- **Evaluation sets.** E1 targets 2016-Q1..2019-Q4 (training from 2009-Q2); E2 targets
  2022-Q3..2026-Q2 (training 2009-Q2..2019-Q4 plus 2022-Q3 onward). Every model is
  re-estimated at every origin; flagged cells are not scored; all models are scored on
  the same cells.
- **Models.** Three Bayesian specifications (`primary`, `no_common_factor`,
  `gaussian_iid`), with the target quarter's common factor drawn from its prior;
  country OLS distributed lag and ARDL with exact Student-t predictive distributions;
  zero-change, historical-mean and AR(1) benchmarks.
- **Scores.** CRPS (primary), log predictive density, squared and absolute error,
  interval coverage, PIT. The factor is integrated out of the log score exactly, via
  the Student-t scale-mixture representation.
- **Test.** For each target quarter the cross-country mean loss differential `d[t]`;
  a Diebold-Mariano statistic on `d[t]` (Newey-West, 2 lags, Harvey-Leybourne-Newbold
  correction, t with T-1 df). Averaging within quarters makes the test valid under any
  cross-country dependence. Five pre-registered comparisons per set, Holm-adjusted
  over the ten.

### Amendment 2: robustness of country differences and ranks (implemented)

Registered in [BAYES_PROTOCOL.md](BAYES_PROTOCOL.md), section 10, before computing any
cross-specification result; implemented in `bayes_robustness.py`; results in
[docs/BAYES_ROBUSTNESS.md](docs/BAYES_ROBUSTNESS.md).

- **Coefficient sums.** Posterior mean, median and 90% interval of `theta[i]` from the
  joint draws; `theta = beta0 + sum_k b_k` is verified draw by draw.
- **Pairwise differences.** `P(theta[i] < theta[j])` from aligned joint draws. A
  difference survives when it is credible (>= 0.95 or <= 0.05) in the same direction in
  every accepted fit of set A (the nine pre-registered specifications plus K = 1);
  set B adds the quarter-effect fits.
- **Ranks.** Rank distributions per fit and the envelope of the 90% rank intervals
  across set A; no model averaging. Positions are described only through surviving
  pairs.
- **Focus countries.** The six pairs among PT, ES, NL and LU against the classical
  review's dependence-aware paired block bootstrap.

Still to do in this stage: address the misfits of the posterior predictive checks
(dynamic factor, lagged unemployment changes) and time-varying coefficients.

## Stage 3 — Identification before causal language

With reduced-form covariance `Sigma = B B'`, `B Q` has the same
covariance for any orthogonal `Q`. A multivariate dynamic model
therefore does not uniquely identify a structural GDP disturbance.

For causal claims specify the source of the shock and an acceptable
identification strategy (e.g. defensible instruments or structural
restrictions), test assumptions and include sensitivity analyses.
Otherwise call results conditional associations, not causal impulse
responses. Every response chart must state the shock units, horizon,
normalization, counterfactual and uncertainty.

## Stage 4 — Robustness and mechanisms

- Examine Eurostat quality flags and 2021 Labour Force Survey breaks.
- Compare pre-pandemic, pandemic-excluded and post-pandemic periods.
- Vary lag orders, GDP specification, age bands, pooling assumptions and
  plausible residual models.
- Evaluate cross-country dependence and common European shocks.
- Examine Luxembourg's cross-border workforce: GDP is produced partly by
  non-residents while resident unemployment uses another denominator.
- For the Netherlands, examine contract composition, hours and participation
  before attributing a delayed response to temporary contracts.
- Publish null, unstable and non-identifiable outcomes as findings.

Descriptive utility requires data checks and robust estimates;
predictive utility requires untouched holdout evidence; causal
language requires separately argued structural identification.
