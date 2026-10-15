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

Still to do in this stage: predictive comparison with the classical baselines
(expanding-window refits) and time-varying coefficients.

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
