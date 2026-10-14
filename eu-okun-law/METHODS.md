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

## Stage 2 — Hierarchical Bayesian estimation

Example partial pooling: `beta[i,k] ~ Normal(mu[k], tau[k])`.
Specify prior scales, shared shocks, residual dependence and pooling
assumptions. Compare static and time-varying parameters. Report convergence,
divergences, effective sample sizes, posterior predictive diagnostics and
sensitivity to priors and exclusion windows.

Country rankings require the joint posterior and pairwise probabilities;
a sorted table of posterior means is not enough. Define whether an
EU-wide coefficient means population weighting, unweighted averaging,
a pooled estimate, or a hyperparameter. Those estimands differ.

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
