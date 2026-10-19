# Small-sample inference for the classical headline sums (protocol)

Registered on 2026-10-19, before any bootstrap or simulation result was computed. The
headline estimates, their Newey-West standard errors and the normal-reference count (22
of 27 countries with a 95% interval below zero; 21 after Holm) were already known: they
are committed in `docs/review/grid.csv` and quoted in the manuscript. This study changes
only the reference distribution of the published statistic.

## 1. Question

The classical headline is the two-lag sum `theta = beta_0 + beta_1 + beta_2` of OLS
regressions of the unemployment-rate change on current and lagged GDP growth, one per
country, on the pandemic-excluded common sample. Its interval uses a calendar-aware
Newey-West (Bartlett) standard error with bandwidth 4 and normal critical values, on 57
quarters per country.

The small-sample study (SMALL_SAMPLE_PROTOCOL.md) showed that the same kind of normal
reference rejected a true null 22.8% of the time at a nominal 5% in a 60-quarter design.
An external review on 2026-10-19 asked whether the headline count survives inference
that is valid in small samples. This study asks:

- How many headline sums are negative under null-imposed bootstrap inference?
- What is the size of the normal-reference test in this design?

## 2. Design (unchanged)

- **Data.** The frozen snapshot `snapshots/eurostat-2026-10-15`.
- **Sample.** The `pandemic_excluded` common sample of the classical review
  (`review.sample_designs`): 2020-2021 removed from outcomes and lag history, one
  calendar intersection for all countries.
- **Model.** For each of the 27 countries, `du_t = a + b0 g_t + b1 g_{t-1} + b2 g_{t-2}
  + e_t`, without the participation control.
- **Statistic.** `t = theta_hat / se_HAC`, with calendar-aware Bartlett weights,
  bandwidth 4 and no small-sample adjustment, exactly as published.
- **Check before any bootstrap.** The implementation must reproduce `gdp_sum` and `se`
  of the committed `docs/review/grid.csv` (sample `pandemic_excluded`, two lags, no
  participation) to within 1e-10 for every country. Otherwise the run stops.

## 3. Reference distributions

### 3.1 Null-imposed sieve bootstrap (primary)

The scheme of SMALL_SAMPLE_PROTOCOL.md, section 3.1, applied to this regression:

1. Fit the restricted model (`theta = 0`) by restricted least squares.
2. Fit an AR(1) to the restricted residuals within each run of consecutive quarters
   (the sample has a gap around 2020-2021), with `rho` clipped to [-0.95, 0.95] and
   centred innovations.
3. Build wild AR(1) errors with Rademacher weights, add them to the restricted fit, and
   recompute the HAC t-statistic.
4. `B = 9999`. The two-sided p-value is `(1 + #{|t*| >= |t|}) / (B + 1)`. With
   `B = 9999` the smallest attainable p-value, 0.0001, is below the smallest Holm
   threshold, 0.05/27 = 0.00185.

### 3.2 Null-imposed moving-block bootstrap (second scheme)

The restricted residuals are resampled in blocks of 4 consecutive quarters within each
run of consecutive quarters, with the same `B` and p-value.

### 3.3 Settings

- **Seeds.** Sieve: `[20261022, index]`; block: `[20261022, index, 100]`, where `index`
  is the country's position in the alphabetical EU-27 list.
- **Bandwidth.** The primary bandwidth is 4. As a sensitivity, the sieve bootstrap is
  repeated at bandwidths 2 and 8 (seed `[20261022, index, bandwidth]`).

## 4. Bootstrap-t intervals

For each country, a 95% bootstrap-t interval from the unrestricted sieve bootstrap: AR(1)
with Rademacher innovations, the fitted coefficients as truth, bandwidth 4, `B = 9999`
and seed `[20261023, index]`. The interval is
`[theta - q*_0.975 se, theta - q*_0.025 se]`, where `q*` are quantiles of
`(theta* - theta) / se*`.

## 5. Size study

- **Data.** For each country, `R = 200` data sets simulated from its fitted null sieve
  DGP (3.1).
- **Tests.** On each data set: the normal test (`|t| > 1.959964`) and the full sieve
  bootstrap test with `B = 199`, everything re-estimated from the simulated data,
  rejecting when `p <= 0.05`.
- **Seed.** `[20261024, index]`.
- **Reported.** Both rejection rates for each country, their pooled rates over the 5,400
  simulated tests, and the binomial 95% range for 200 and for 5,400 trials at 0.05.

## 6. Decision rules

- **Negative under small-sample inference.** `theta_hat < 0` and the two-sided p-value is
  at most 0.05 under the sieve scheme **and** under the block scheme. A sum that meets the
  rule under one scheme only is reported as scheme-dependent.
- **Counts.**
  - Unadjusted: the number of negative sums, against the normal-reference 22.
  - Holm over the 27 countries, each scheme's p-values adjusted separately; a sum counts
    when both adjusted p-values are at most 0.05. Compared with the normal-reference 21.
- **Intervals.** The bootstrap-t intervals of section 4 are descriptive. They are
  reported next to the normal intervals; their count of intervals below zero is not a
  decision rule.
- **Size.** A pooled bootstrap rate inside the binomial 95% range for 5,400 trials means
  adequate size under these data-generating processes. A country whose normal-test rate
  is above its binomial upper bound for 200 trials over-rejects.
- **Bandwidth.** For each sensitivity bandwidth, the unadjusted and Holm counts under the
  sieve scheme.
- **Status.** Every result is a descriptive statement about a conditional association,
  not a causal effect. Nothing else in the study changes: the Bayesian, predictive,
  identification and case-study analyses do not use these p-values.
- **Manuscript.** The counts are reported next to the normal-reference counts. If they
  differ, the manuscript's headline count becomes the sieve-and-block count, and the
  normal-reference count is kept as the published classical figure.

## 7. Reproducibility

- NumPy only; reads only the frozen snapshot. Command: `eu-okun headline-inference`.
- Outputs: `docs/headline-inference/` (tables) and `docs/HEADLINE_INFERENCE.md`.
- The committed tables are regenerated by a test in the ordinary CI if the computation
  stays below about three minutes; otherwise the test regenerates the main tables only.

## Change log
