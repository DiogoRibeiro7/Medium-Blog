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

## 10. Amendment 2: robustness of country differences and ranks

Registered on 2026-10-16. At that point the fits of sections 5 and 9 existed. No
pairwise probability or rank distribution had been computed for any specification other
than `primary`, and `lags_1` had not been fitted.

### 10.1 One more lag order

`lags_1` is `primary` with K = 1. With `primary` (K = 2) and `lags_4` (K = 4), it gives
the lag orders 1, 2 and 4 that the classical analysis uses. K = 0 is not possible because
the model needs at least one lag. Priors, sampler, acceptance criteria and the single
retry follow section 6.

### 10.2 Specification sets

**Set A** is the registered robustness set: the nine specifications of section 5 plus
`lags_1`.

**Set B** is set A plus the three specifications of amendment 1. With quarter effects,
`theta` measures a within-quarter association (section 9.1). Set B is therefore reported
alongside set A and never replaces it.

Only fits accepted under section 6 enter a set.

### 10.3 Country summaries

For every fit, from the joint draws of `theta[i]`:

- the posterior mean and median;
- the 90% equal-tailed interval;
- P(theta[i] < 0).

`theta[i]` is the coefficient sum by construction. The report also checks, draw by draw,
that it equals `beta0[i] + sum_k b[i,k]`.

### 10.4 Which country differences survive

- For each fit, `P(theta[i] < theta[j])` is computed from aligned joint draws, i.e. the
  same draw index for both countries.
- A difference is **credible in a fit** if this probability is at least 0.95 or at most
  0.05.
- A difference **survives modelling and prior uncertainty** if it is credible, in the
  same direction, in every accepted fit of set A.

The report gives:

- the number of the 351 pairs that survive in set A and in set B;
- the pairs that are credible in `primary` but do not survive;
- for each country, the number of countries it differs from robustly.

### 10.5 Rank uncertainty

- For every fit: the rank distribution (1 = most negative sum), the median rank and the
  90% rank interval.
- Across set A, the envelope from the lowest lower bound to the highest upper bound.
  This describes rank uncertainty including model uncertainty.
- There is no model averaging, because no weights are justified.

Ranks are always reported as intervals. A country's position is described only through
the pairwise differences that survive under 10.4.

### 10.6 Portugal, Spain, the Netherlands and Luxembourg

All six pairs among these countries are compared in every fit with the classical review's
dependence-aware paired block bootstrap (`docs/review/pairs_review.csv`).

**Classical apparent difference.** The Holm-adjusted sign p-value is below 0.05 at some
block length in the pandemic-excluded sample, which is the sample closest to the Bayesian
2009-2019 window. The full-sample results are shown alongside.

**Verdict for each pair:**

| Classical | Bayesian (rule 10.4) | Verdict |
| --- | --- | --- |
| apparent | survives | **survives** |
| apparent | does not survive | **does not survive** |
| not apparent | survives | **credible only in the Bayesian fits** |
| not apparent | does not survive | **not supported by either** |

### 10.7 Diagnostics and comparison with OLS

For every fit, a table and a figure show:

- the convergence diagnostics: maximum R-hat, minimum bulk and tail ESS, divergences,
  minimum E-BFMI and tree-depth saturation;
- the posterior predictive p-values of section 6.

Failed fits stay listed. Country summaries sit next to the classical OLS/HAC estimates on
the same rows.

The decision rules apply to this run. A replication run on a second platform is compared
descriptively, as in section 9.

## 11. Amendment 3: dynamics that address the misfits

Registered on 2026-10-16, before any fit of the specifications below. Sections 1-10 are
unchanged, and the pre-registered judgements keep using their own specification sets.

### 11.1 The misfits

The posterior predictive checks found two problems (HIERARCHICAL_BAYES.md, sections 7-8):

- **Persistence.** Every specification understates the persistence of unemployment
  changes: the observed lag-1 autocorrelation is 0.27, against replicated means of 0.02
  to 0.20.
- **The common factor.** It is modelled as independent over time, yet its posterior is
  autocorrelated (median 0.75) and correlated with average GDP growth (median -0.60).

### 11.2 New specifications

Each specification changes the primary one as follows:

| Name | Change from `primary` |
| --- | --- |
| `dynamic_factor` | The common factor follows a stationary AR(1) process. |
| `ardl` | The AR(1) error is replaced by the lagged dependent variable `psi[i] * y[i,t-1]` in the mean equation, with i.i.d. Student-t errors and an i.i.d. factor. |
| `ardl_dynamic_factor` | `ardl` with the AR(1) factor. |

**The AR(1) factor.** `f[t] = rho_f f[t-1] + sqrt(1 - rho_f^2) e[t]` with `e[t] ~ N(0, 1)`.

- `f` is stationary with unit variance at the first quarter of each segment, so the
  loadings keep their scale.
- `rho_f = tanh(r)` with `r ~ N(0, 1)`.

**The lagged change.** `psi[i] = tanh(mu_psi + tau_psi * omega[i])`, with
`mu_psi ~ N(0, 0.5)` and `tau_psi ~ HalfNormal(0.5)`, so that `|psi| < 1`.

- The likelihood conditions on the first quarter of each segment: its term is not used,
  because its lagged change lies outside the window (and is flagged for most countries
  in 2009-Q1).
- Terms that follow a masked term are already masked by the rule of section 2.

All other priors, the sampler, the acceptance criteria and the single retry follow
sections 4 and 6.

### 11.3 Estimands

**In `dynamic_factor`.** `theta[i]` keeps its meaning: the sum of GDP coefficients,
beyond the common factor.

**In the ARDL specifications.**

- `theta[i]` is the *short-run* sum of the coefficients at lags 0-2.
- The comparable total association is the long-run multiplier
  `theta_LR[i] = theta[i] / (1 - psi[i])`. It is computed from the joint draws and
  compared with `theta` of `primary`.

### 11.4 Decision rules

- **Persistence misfit.** It is **fixed** in a specification when the posterior
  predictive p-value of the lag-1 autocorrelation lies in [0.05, 0.95].
- **Factor dynamics.** They are **supported** when the 90% interval of `rho_f` excludes
  zero. The correlation of the posterior factor with average GDP growth is reported
  again; the new specifications are not expected to remove it.
- **Model comparison: PSIS-LOO.**
  - It is computed on the cells common to every compared fit, which excludes each
    segment's first quarter.
  - The comparisons are:
    1. `dynamic_factor` against `primary`;
    2. `ardl` against `primary`;
    3. `ardl_dynamic_factor` against `primary`;
    4. `ardl_dynamic_factor` against `ardl`.
  - One model is "better supported" when the elpd difference exceeds twice its standard
    error, as in section 9.2.
- **Conclusions.** For each accepted new specification, the report gives:
  - the country estimates (`theta` or `theta_LR`) next to `primary`;
  - the countries that are credibly negative;
  - the three earlier claims, using `theta_LR` for Luxembourg and Spain in the ARDL
    specifications.
- **Status of the new specifications.** They are reported **alongside** the
  pre-registered results. A specification that fixes the misfits and is better supported
  is described as such. It does not retroactively replace the primary specification of
  sections 3-7.

## 12. Amendment 4: regime-specific and time-varying coefficients

Registered on 2026-10-16, before any fit of the specifications below. Sections 1-11 are
unchanged, and the earlier judgements keep using their own specification sets.

### 12.1 The question

The classical review (docs/CLASSICAL_RESULTS_REVIEW.md, section 5) found a weaker
association after 2021 in 13 countries. It could not report the size or sign of any
post-2021 coefficient, for five reasons:

- the window is short;
- GDP varied less;
- many post-2021 sums change sign when one quarter is dropped;
- Spain and France changed their unemployment definition;
- post-2021 GDP is partly provisional.

`pandemic_excluded` (section 5) imposes one coefficient on both periods. This amendment
asks four things:

- whether the association differs between 2009-2019 and 2022-2026;
- by how much;
- in which countries;
- whether the change looks abrupt or gradual.

Partial pooling is the reason to ask it in this model: each country's shift borrows
strength from the others.

### 12.2 Data

- **Windows and masks.** The windows of `pandemic_excluded`: 2009-Q2 to 2019-Q4 (43
  quarters) and 2022-Q3 to 2026-Q2 (16 quarters), with K = 2 and the masks of
  section 2.
- **Regime indicator.** `post[t] = 1` from 2022-Q3, `0` before.
- **Spain and France.** Every post-2021 term is masked (definition flag). Their shifts
  are therefore informed only through pooling.

### 12.3 New specifications

| Name | Change |
| --- | --- |
| `regime_shift` | `pandemic_excluded` plus, after 2021: a country shift in the GDP coefficient sum, `theta_post[i] = theta[i] + delta[i]`; a country shift in the intercept; and a common shift of the log error scale. |
| `regime_shift_final_gdp` | `regime_shift` with every term masked whose GDP inputs (quarters t, t-1, t-2) carry the provisional flag `p`, together with the AR(1) term that follows it (section 2). |
| `pandemic_excluded_dynamic` | `pandemic_excluded` with the dynamics of `ardl_dynamic_factor` (section 11). |
| `regime_shift_dynamic` | `regime_shift` with the dynamics of `ardl_dynamic_factor`. |
| `drifting_association` | `pandemic_excluded` plus the intercept and scale shifts of `regime_shift`, and a common random walk in the GDP coefficient sum: `theta[i,t] = theta[i] + kappa[t]`. |

**Priors.**

- **Coefficient shift.** `delta[i] = mu_delta + tau_delta * raw[i]`, with
  `mu_delta ~ N(0, 0.5)` and `tau_delta ~ HalfNormal(0.25)`. These are the scales of
  `mu_theta` and `tau_theta`, so the prior favours no particular shift size.
- **Intercept shift.** Partially pooled with the scales of `alpha`:
  `mu ~ N(0, 0.25)`, `tau ~ HalfNormal(0.25)`.
- **Scale shift.** A common shift of `log sigma`, distributed `N(0, 0.5)`, after 2021.
- **Random walk.**
  - `kappa_raw` is zero at the first sample quarter.
  - Its increments are `N(0, tau_kappa^2 * gap)`, where `gap` is the number of calendar
    quarters between consecutive sample quarters: 1 inside a segment, 11 from 2019-Q4
    to 2022-Q3.
  - `kappa = kappa_raw - mean(kappa_raw)` over the sample quarters, so `theta[i]` is the
    time average.
  - `tau_kappa ~ HalfNormal(0.05)`.
- **ARDL specifications.** The shift applies to the short-run sum. The long-run shift is
  `delta / (1 - psi)`, which has the same sign.

All other priors, the sampler, the acceptance criteria and the single retry follow
sections 4, 6 and 11.

### 12.4 Estimands

- **Regime specifications.**
  - `delta[i]`: the change of the GDP coefficient sum after 2021.
  - `theta_post[i] = theta[i] + delta[i]`: the post-2021 sum.
  - `mu_delta`: the average shift.
- **Random walk.**
  - `D1`: the mean of `kappa` over 2014-Q1 to 2019-Q4, minus its mean over 2009-Q2 to
    2013-Q4.
  - `D2`: the mean of `kappa` over 2022-Q3 to 2026-Q2, minus its mean over 2014-Q1 to
    2019-Q4.

### 12.5 Decision rules

- **Evaluable countries.** A country is evaluable in a specification when that
  specification's likelihood holds at least 12 post-2021 terms for it. The shifts of the
  other countries (Spain and France in every specification) are pooled values,
  reported as "not evaluable".
- **Country shift.** For an evaluable country:
  - "rises" when `P(delta[i] > 0) >= 0.95`;
  - "falls" when `P(delta[i] > 0) <= 0.05`;
  - "no credible change" otherwise.
  - For a country whose pre-2020 sum `theta[i]` is credibly negative, "rises" means a
    weaker association.
- **Average shift.** The same rule applied to `mu_delta`.
- **Post-2021 association.** The sign class of `theta_post[i]` (section 7) for
  evaluable countries.
- **Classical comparison.** Agreement, country by country, with the review's
  weaker-after-2021 set. The report recomputes that set from the committed
  docs/review/stability_review.csv and docs/review/grid.csv. The comparison is
  descriptive.
- **Measurement.** Each verdict of `regime_shift` is compared with
  `regime_shift_final_gdp` (the average shift, and each country evaluable in both). A
  verdict that changes "depends on provisional GDP".
- **Dynamics.** The same comparison with `regime_shift_dynamic`. A verdict that changes
  "depends on the dynamics".
- **Drift.**
  - `D1` and `D2` are judged by the same 0.95 rule; the 90% interval of `tau_kappa` is
    reported.
  - The random walk cannot place a change inside the unobserved 2020-2022 gap.
- **PSIS-LOO.** Each comparison uses the cells common to the two fits and the 2 SE rule
  of section 9.2:
  1. `regime_shift` against `pandemic_excluded`;
  2. `drifting_association` against `pandemic_excluded`;
  3. `drifting_association` against `regime_shift`;
  4. `regime_shift_dynamic` against `pandemic_excluded_dynamic`.

  `regime_shift_final_gdp` is scored on fewer cells and is not compared.
- **Recovery.**
  - One data set is simulated from `regime_shift` on the real design: the
    pandemic-excluded windows, the real masks and seed 20261021.
  - Generating values: `mu_theta = -0.30`, `tau_theta = 0.15` and `nu = 8`, as in
    section 9.4; `mu_delta = +0.15` and `tau_delta = 0.10`; intercept shifts
    `N(0, 0.05)` and no scale shift; other parameters as in section 9.4.
  - Reported: the 90% coverage of `delta[i]` for the evaluable countries, against the
    binomial range, and whether the average-shift rule detects the known shift. One
    data set is a check, not a power analysis.
- **Prior predictive check.** That of section 9.3, for the new specifications.
- **Replication.** An independent Linux run. A verdict that differs between the runs is
  reported with both values and is not treated as established.
- **Status.** The new specifications are reported alongside the earlier results. Every
  coefficient is a descriptive association. A shift could reflect any of these, and this
  amendment cannot separate them (docs/IDENTIFICATION_RESULTS.md):
  - measurement (definitions, provisional data);
  - a different mix of shocks (the energy shock, inflation);
  - a changed labour-market response.

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
- Amendment 3, before any of its fits: in the ARDL specifications `mu_theta` is the
  mean of the *short-run* sums, so it is not comparable with a long-run multiplier. The
  Luxembourg and Spain rules therefore compare `theta_LR[i]` with the per-draw
  unweighted mean of `theta_LR` over the 27 countries, the same definition as
  `no_pooling`'s substitute for `mu_theta`. The Netherlands rule keeps `theta` and
  `beta0` as section 11.4 states. The long-run delayed share `(theta_LR - beta0) /
  theta_LR` is reported as a descriptive addition, outside the rule. Because
  `1 - psi > 0`, `theta_LR` and `theta` always have the same sign, so the credibly
  negative set is the same under either estimand. The AR(1) factor is sampled in its
  centred form (the density of `f` given `rho_f`): this is the same model as
  section 11.2, written in a different parameterisation.
- Amendment 3, after its fits: locally, all three specifications met every criterion at
  the first attempt.
  - The prior predictive check of section 9.3 was applied to the new priors (same seed
    and draws); it is not a decision rule.
  - Replication on Linux (Bayesian workflow run 38975791746, specs `primary` and the
    three new ones): all four were accepted, `primary` after the retry.
  - Medians agree within 0.006, `rho_f` and the credibly negative sets are identical.
  - Three verdicts sit on their thresholds and differ between the runs. The report
    states them with both values and does not treat them as established:
    - the persistence verdict of `dynamic_factor` (p = 0.030 and 0.052);
    - "Netherlands delayed" in `dynamic_factor` (0.948 locally);
    - PSIS-LOO `ardl_dynamic_factor` against `ardl` (+11.8 with SE 5.7, and +10.7 with
      SE 5.8).
- Amendment 4, after its fits.
  - **Local run.**
    - `regime_shift`, `pandemic_excluded_dynamic` and `regime_shift_dynamic` met every
      criterion at the first attempt.
    - `drifting_association` passed after the single retry.
    - `regime_shift_final_gdp` failed after the retry (R-hat 1.012 for `phi[9]`,
      Spain's AR(1) coefficient). It is reported as failed, and the provisional-GDP
      sensitivity cannot be judged.
    - The amendment 3 report now restricts its context rows to the specifications
      registered up to amendment 3, so that later fits in the same directory do not
      change it.
  - **Replication on Linux** (Bayesian workflow run 38983863147):
    - the same acceptance for all six fits, including the failure of
      `regime_shift_final_gdp`;
    - shift medians within 0.003, and the same average, country and drift verdicts;
    - the same recovery.
    - One verdict sits on its threshold and differs: PSIS-LOO `drifting_association`
      against `regime_shift` (+4.3 with SE 1.8 locally, +3.5 with SE 1.8 on Linux). It
      is not treated as established.
  - **Earlier dispatch.** Run 38983097415 stopped at the general report step, which
    needs a `primary` fit; its results are not used.
