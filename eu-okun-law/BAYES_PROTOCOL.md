# Hierarchical Bayesian analysis: pre-registered protocol

Written and committed before any Bayesian model was fitted to Eurostat data. Changes
after the first real-data fit are listed in the change log at the end, with reasons.

## 1. Question and estimands

How large is the conditional association between quarterly real GDP growth and the
change in the unemployment rate in each EU-27 country, once country estimates borrow
strength from one another and cross-country and serial dependence are modelled?

| Symbol | Estimand |
| --- | --- |
| `theta[i]` | Country *i*'s sum of GDP coefficients at lags 0..K (pp of unemployment-rate change per pp of quarterly growth). Same estimand as the classical coefficient sum. |
| `mu_theta` | Mean of the population distribution of country sums: an **unweighted** cross-country average. It is not an EU aggregate, not population-weighted and not a pooled regression slope. |
| `tau_theta` | Between-country standard deviation of the country sums. |
| `beta0[i]` | Contemporaneous coefficient; `theta[i] - beta0[i]` is the part carried by lagged growth. |

All of these are descriptive associations. GDP growth is not an identified shock, and
nothing here is a causal effect.

## 2. Data and sample

- Frozen snapshot `snapshots/eurostat-2026-10-15` (manifest SHA-256 `8a8c8a10…`).
- **Primary sample: 2009-Q2 to 2019-Q4**, all 27 countries, 43 contiguous quarters.
  Chosen because the classical review showed that 2020-2021 dominate any sample that
  contains them, that post-2021 estimates are unstable, and that Spain and France
  change their unemployment definition from 2021-Q1. It is the longest window that is
  contiguous and definition-consistent for every country.
- Eurostat flags: likelihood terms for changes that rely on a flagged observation are
  masked, using the rule of the classical review (a labour-force break invalidates the
  change into the break quarter; low reliability or "definition differs" invalidates
  every change touching that quarter; a GDP break invalidates rows using that growth
  rate). Because of the AR(1) error, the term following a masked quarter is masked too.
  No value is imputed or altered.

## 3. Model (primary specification)

For country *i* and quarter *t*, with `g` real GDP growth and `y = Δu`:

```text
y[i,t] = alpha[i] + theta[i] * g[i,t] + sum_{k=1..K} b[i,k] * (g[i,t-k] - g[i,t])
         + lambda[i] * f[t] + e[i,t]
e[i,t] = phi[i] * e[i,t-1] + sigma[i] * eps[i,t],     eps[i,t] ~ Student-t(nu)
```

The reparameterisation makes `theta[i]` the coefficient sum directly:
`beta0[i] = theta[i] - sum_k b[i,k]` and `beta_k[i] = b[i,k]`. K = 2.

- `f[t] ~ Normal(0, 1)` is a common quarterly factor shared by all countries, with
  loadings `lambda[i] >= 0`. It represents residual cross-country dependence (common
  European shocks not explained by each country's own GDP growth). The non-negative
  loadings are an identifying assumption.
- `e` is AR(1) within each contiguous segment; the first quarter of a segment has the
  stationary scale `sigma[i] / sqrt(1 - phi[i]^2)`.

Hierarchy (non-centred):

```text
theta[i]   = mu_theta + tau_theta * eta[i]
alpha[i]   = mu_alpha + tau_alpha * zeta[i]
b[i,k]     = mu_b[k]  + tau_b * xi[i,k]
phi[i]     = tanh(mu_phi + tau_phi * omega[i])
log sigma[i] = mu_sigma + tau_sigma * kappa[i]
eta, zeta, xi, omega, kappa ~ Normal(0, 1)
```

## 4. Priors (baseline)

Units: percentage points per quarter, and pp per pp of growth.

| Parameter | Prior | Reasoning |
| --- | --- | --- |
| `mu_theta` | Normal(0, 0.5) | Centred at **zero**: the prior does not assume Okun's law. |
| `tau_theta` | HalfNormal(0.25) | Allows country sums to differ by several tenths. |
| `mu_alpha`, `tau_alpha` | Normal(0, 0.25), HalfNormal(0.25) | Drift in unemployment, pp per quarter. |
| `mu_b[k]`, `tau_b` | Normal(0, 0.5), HalfNormal(0.25) | Lag weights. |
| `mu_phi`, `tau_phi` | Normal(0, 0.5), HalfNormal(0.5) | AR(1) on the tanh scale, so abs(phi) < 1. |
| `mu_sigma`, `tau_sigma` | Normal(log 0.3, 1), HalfNormal(0.5) | Residual scales differ eightfold across countries in the classical fits. |
| `lambda[i]` | HalfNormal(0.2) | Common-factor loadings. |
| `nu` | Gamma(2, 0.1) | Standard weakly informative prior on tail weight. |

## 5. Sensitivity set (fixed in advance)

| Name | Change from the primary specification |
| --- | --- |
| `primary` | none |
| `wide_priors` | every location and scale prior above widened four-fold |
| `strong_pooling` | `tau_theta ~ HalfNormal(0.05)` |
| `no_pooling` | country parameters independent, Normal(0, 0.5) priors, no hyperparameters |
| `no_common_factor` | `lambda = 0` |
| `gaussian_iid` | Normal errors, no AR(1), no common factor (closest to OLS) |
| `participation` | adds `gamma[i] * Δp[i,t]`, partially pooled like `theta` |
| `lags_4` | K = 4 |
| `pandemic_excluded` | 2009-Q2..2019-Q4 and 2022-Q3..2026-Q2 as two segments |

No specification is added, dropped or preferred because of its estimates.

## 6. Computation and acceptance criteria

NUTS (NumPyro), 4 chains, 1000 warm-up and 1000 retained draws per chain, target
acceptance 0.95, maximum tree depth 10, fixed seed. A fit is **accepted** only if:

- max R-hat <= 1.01 over all parameters;
- min bulk-ESS and min tail-ESS >= 400;
- no divergent transitions after warm-up;
- E-BFMI >= 0.3 in every chain.

A fit that fails is reported as failed, with its diagnostics, and its estimates are not
interpreted. Failures may be re-run with more warm-up or a higher target acceptance;
the model and priors are not changed to make a fit pass.

Posterior predictive checks: coverage of the observed `y` by one-step-ahead 50% and
90% predictive intervals, and replicated versus observed values of residual lag-1
autocorrelation, average cross-country correlation and country standard deviations.

## 7. Reporting rules

- Country results are posterior medians with 90% credible intervals, alphabetically
  ordered, next to the classical OLS/HAC estimate on the same sample.
- No ranking by point estimate. Rank uncertainty is shown as the posterior distribution
  of each country's rank, and differences as pairwise probabilities `P(theta[i] < theta[j])`.
- Three earlier claims are examined descriptively, without causal attribution:
  a weak association for Luxembourg, a delayed (lag-heavy) response for the
  Netherlands, and a large association for Spain. Contradictions are reported as such.

Decision thresholds (90% equal-tailed credible intervals throughout):

| Statement | Rule |
| --- | --- |
| `theta[i]` credibly negative / positive | P(theta[i] < 0) >= 0.95 / <= 0.05; otherwise "sign uncertain" |
| Country difference supported | P(theta[i] < theta[j]) >= 0.95 or <= 0.05 |
| Robust across specifications | the same sign classification in every **accepted** specification |
| Luxembourg weak | P(theta[LU] > mu_theta) >= 0.95 **and** the 90% interval of theta[LU] includes zero |
| Netherlands delayed | P(lag share > 0.5) >= 0.95, lag share = (theta - beta0) / theta, evaluated only if P(theta[NL] < 0) >= 0.95 |
| Spain large | P(theta[ES] < mu_theta) >= 0.95 |

A claim is "supported" if its rule holds in the primary and every accepted
sensitivity fit, "contradicted" if the opposite inequality holds at the same level in
the primary fit, and "not resolved" otherwise.

## 8. Deferred

Predictive comparison with the classical baselines (expanding-window refits) and
time-varying coefficients are separate milestones.

## 9. Amendment 1: additive quarter effects, prior predictive checks, recovery

Registered on 2026-10-16, after the nine fits above and before any fit of the
specifications below. Nothing in sections 1-7 changes: the robustness and claim
judgements stay based on the nine original specifications.

### 9.1 Additive shared quarter effects

The common factor above lets each country load differently on a shared quarterly
shock. The classical two-way analogue is an additive effect common to all countries:

```text
y[i,t] = alpha[i] + theta[i]*g[i,t] + sum_k b[i,k]*(g[i,t-k] - g[i,t])
         [+ gamma[i]*Δp[i,t]] + delta[t] + e[i,t]
delta ~ ZeroSumNormal(tau_delta) over the T sample quarters,   tau_delta ~ HalfNormal(0.25)
```

`delta[t]` is in pp of unemployment-rate change per quarter, and `e` is the Student-t
AR(1) error of the primary specification.

**Identification.**

- The sum-to-zero constraint separates `delta` from the intercepts. `alpha[i]` carries
  each country's average change, and `delta[t]` carries the quarter's common deviation.
- `delta[t]` also absorbs anything common to all countries in quarter t, including
  `mu_theta` times a GDP movement that every country shares. So `theta[i]` measures the
  association with the country's GDP growth *beyond* the quarter's common effect.
  `mu_theta` is identified only from cross-country differences in growth within
  quarters. If every country grew at the same rate, `mu_theta` and `delta` would be
  confounded.
- Quarter effects and the common factor are never combined in one specification.

**New specifications.** These complete a 2 x 2 design: with or without shared quarter
effects, and with or without participation.

| Name | Change from the primary specification |
| --- | --- |
| `time_effects` | the common factor is replaced by `delta[t]` |
| `time_effects_participation` | as `time_effects`, plus `gamma[i] * Δp[i,t]`, partially pooled as in `participation` |
| `participation_no_common_factor` | as `participation`, without the common factor |

Sampler settings, acceptance criteria and the single retry are those of section 6.

### 9.2 With versus without shared quarter effects

These are descriptive model comparisons, not tests of a hypothesis:

- **PSIS-LOO.** The expected log pointwise predictive density, leaving out one
  unmasked cell at a time. For AR(1) errors each term is the conditional density given
  the previous quarter. This is not temporal validation; that is the role of
  PREDICTIVE_PROTOCOL.md.
- **Posterior predictive check.** The average cross-country correlation from section 6.

| # | Comparison | Question |
| --- | --- | --- |
| 1 | `time_effects` vs `no_common_factor` | Shared quarter effects, without participation |
| 2 | `time_effects_participation` vs `participation_no_common_factor` | Shared quarter effects, with participation |
| 3 | `time_effects` vs `primary` | Additive effects against country loadings |
| 4 | `time_effects_participation` vs `time_effects` | Participation, given quarter effects |

**Decision rule.**

- One model is "better supported by LOO" when the elpd difference exceeds twice its
  standard error. Otherwise the two are "not distinguishable".
- Any comparison involving a fit with Pareto k > 0.7 in more than 1% of cells is
  labelled unreliable.
- Only fits accepted under section 6 are compared.

### 9.3 Prior predictive checks

For every specification, 1000 draws from the prior are each propagated to a simulated
data set on the real design. The following are reported against the observed values:

- 5%, 50% and 95% quantiles of the mean country standard deviation of `y`;
- the same quantiles of the largest absolute change, the lag-1 autocorrelation and the
  cross-country correlation;
- the share of simulated changes larger than 5 pp in absolute value.

The priors are fixed and are not changed in response to these checks. Implausible
prior mass is reported as such.

### 9.4 Posterior recovery on synthetic data

- **Data.** For `primary`, `time_effects` and `gaussian_iid`, one data set is simulated
  on the real design (GDP growth, masks and segments). The generating parameters are
  fixed in code before any fit, with `mu_theta = -0.30` and `tau_theta = 0.15`.
- **Fit.** Each data set is fitted with the protocol settings.
- **Reported.** The coverage of the 50% and 90% intervals for the country parameters,
  and whether the hyperparameters fall inside their 90% intervals.
- **Interpretation.** With 27 countries, 90% coverage between 21 and 27 is consistent
  with calibration. A single data set per specification is a recovery check, not
  simulation-based calibration.

### 9.5 Artifacts

Posterior draws of every fit are also written as ArviZ-compatible NetCDF. Each file
holds these groups: posterior, sample statistics when available, prior, prior
predictive, observed and constant data, and the pointwise log likelihood. It also
records the seed, sampler settings, software versions and snapshot hashes.

## Change log

- Before any estimate was inspected: added the decision thresholds of section 7.
  The only real-data computation until then was a short timing run of the primary
  specification (2 chains x 600 iterations) whose convergence diagnostics, runtime
  and predictive coverage were printed; no coefficient was looked at.
- Before any estimate was inspected: in `no_pooling` there is no `mu_theta`; the rules
  use its definition instead, the per-draw unweighted mean of `theta[i]` over the 27
  countries. "Contradicted" means, in the primary fit: Luxembourg
  P(theta[LU] > mu_theta) <= 0.05; Netherlands P(lag share > 0.5) <= 0.05 (with
  theta[NL] credibly negative); Spain P(theta[ES] < mu_theta) <= 0.05.
- First full run (1000 warm-up): seven specifications met every criterion.
  `strong_pooling` failed with max R-hat 1.012 (phi[9], Spain's AR(1) coefficient) and
  `no_pooling` with max R-hat 1.011 (phi[26], Slovakia's); ESS, divergences and E-BFMI
  passed. Both were re-run with 3000 warm-up iterations, as section 6 allows, with the
  model, priors, seed and retained draws unchanged. The first-attempt outputs are kept
  as CI artifacts of the Bayesian workflow and in `results/bayes-first-attempt/` locally.
- Re-run outcome: `strong_pooling` passed (max R-hat 1.007). `no_pooling` failed again
  (max R-hat 1.0106, phi[26]); no further attempt was made, so it is reported as failed
  and excluded from every robustness and claim judgement. The workflow now applies the
  same single retry (`--retry-warmup 3000`).
- Not pre-registered, added after fitting and labelled as such in the report: checks of
  the common factor's independence assumptions (its lag-1 autocorrelation and its
  correlation with average GDP growth and unemployment change), and a decomposition of
  the gap to OLS into pooling (`gaussian_iid` vs OLS) and error model (`primary` vs
  `gaussian_iid`). They were prompted by the failed lag-1 autocorrelation check and by
  the size of that gap. They change the interpretation of `theta` in factor
  specifications (association beyond the common European cycle), not any fit.
- Replication on Linux (Bayesian workflow run 38813919865, same lock file, snapshot,
  seed, settings and single retry): `primary` and `strong_pooling` failed at first;
  after the retry `primary` passed and `strong_pooling` failed again (R-hat 1.017),
  while `no_pooling` passed. Posterior medians agree within 0.004 wherever both runs
  accept a fit, and the claim verdicts are identical. The acceptance of
  `strong_pooling` and `no_pooling` is therefore not stable across runs; the report
  states the robustness judgements under both acceptance sets instead of choosing one.
