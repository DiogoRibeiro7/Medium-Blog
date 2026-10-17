# Small-sample inference for the post-2021 window (protocol)

Registered on 2026-10-17, before any bootstrap or simulation result was computed. The
only real-data computation done beforehand was a check that `fit_stability` still
reproduces the published family: 14 of 54 Holm-significant changes at bandwidth 4.

## 1. Question

The classical stability tests (`eu-okun infer`, `stability.py`) compare 2009-2019
(43 quarters) with 2022 onwards (17 quarters). The model is fully interacted, with one
GDP lag. The test is a calendar-aware Newey-West (Bartlett) t-test on the change in the
GDP coefficient sum, with a normal p-value, Holm-adjusted over 54 tests (27 countries,
with and without participation).

The classical review (docs/CLASSICAL_RESULTS_REVIEW.md, section 5) found 14 rejections
but judged the test unreliable: there are 17 post-2021 quarters for 3-4 coefficients
per period, and nothing corrects the normal approximation. The hierarchical model
(docs/BAYES_REGIMES.md) found no credible shift.

This study asks two questions:

- Do the rejections survive inference that is valid in small samples?
- Can the sign of any post-2021 sum be reported?

It changes only the inference. The data, samples, design, statistic and family are
those of the published test, so any difference is due to the reference distribution.

## 2. Design (unchanged)

- **Data.** The frozen snapshot `snapshots/eurostat-2026-10-15`; flagged observations
  are kept, as in the published test.
- **Samples.** 2009-Q2 to 2019-Q4 and 2022-Q2 to 2026-Q2: the common samples of
  `stability._eligible_periods`, with disjoint input histories.
- **Model.** `y = [X, post*X] beta + e`, with `X = [1, g_t, g_{t-1}]` (plus
  `delta p_t` in the participation variant).
- **Statistic.** `t = c'beta / se_HAC`, where `c` picks the two GDP interaction
  coefficients, so that `c'beta` is the post-minus-pre change of the GDP sum.
- **Covariance.** `se_HAC` uses calendar-aware Bartlett weights with bandwidth 4 and
  no small-sample adjustment, exactly as published.

## 3. Reference distributions

### 3.1 Null-imposed sieve bootstrap (primary)

1. Fit the restricted model (`c'beta = 0`) by restricted least squares, and keep its
   coefficients `beta_r` and residuals `e_r`.
2. In each period separately (pre, post), fit an AR(1) to `e_r` by least squares on
   consecutive quarters. `rho` is clipped to [-0.95, 0.95] and the innovations
   `u_t = e_r[t] - rho e_r[t-1]` are centred.
3. Bootstrap errors are built in each period from Rademacher weights `v`:
   - `e*` at the first quarter of a period is `v e_r`;
   - afterwards `e*_t = rho e*_{t-1} + v_t u_t`.

   The wild weights keep heteroskedasticity within each period, and the periods keep
   their own AR coefficient and scale.
4. `y* = X_pooled beta_r + e*`; the unrestricted model is refitted, giving `t*`.
5. `B = 999` draws. The p-value is `(1 + #{|t*| >= |t|}) / (B + 1)`.

Bootstrapping the HAC-studentised statistic with the regressors fixed reproduces its
fixed-b behaviour: Gonçalves and Vogelsang (2011, *Econometric Theory* 27) show that
the block bootstrap of HAC-studentised statistics matches the fixed-b distribution.
The sieve is a second scheme with the same studentisation.

### 3.2 Null-imposed moving-block bootstrap (second scheme)

The restricted residuals are resampled in blocks of 4 consecutive quarters, drawn and
concatenated within each period (the pre and post periods separately), with the same
`B` and p-value as 3.1.

### 3.3 Settings

- **Seed.** 20261017. Each of the 54 tests has its own generator, seeded with
  `[20261017, index]` in the published row order.
- **Bandwidth.** The primary bandwidth is 4. The sieve bootstrap is repeated at
  bandwidths 0, 2 and 8, as in the review, as a sensitivity.

## 4. Size study

The question is whether each test keeps its nominal 5% size under a realistic null.

- **Data.** For each of the 54 specifications, `R = 200` data sets are simulated from
  its fitted null sieve DGP (3.1).
- **Tests.** On each data set:
  - the asymptotic test (`|t| > 1.959964`);
  - the full sieve bootstrap test, with `B = 199` and everything re-estimated from the
    simulated data, rejecting when `p <= 0.05`.
- **Seed.** `[20261018, index]`.
- **Reported.**
  - the rejection rates of both tests for each specification;
  - their pooled rates over all 10,800 simulated tests;
  - the binomial 95% range for 200 replications at 0.05.
- **Interpretation.** A pooled bootstrap rate inside the binomial 95% range for
  10,800 tests at 0.05 means the bootstrap test has adequate size under these DGPs. A
  specification whose asymptotic rate lies above its binomial upper bound
  over-rejects.

## 5. Post-2021 sums

The review could not report the size or sign of any post-2021 sum.

- **Estimate.** For each country, with and without participation, the post-only
  regression `[1, g_t, g_{t-1}]` (plus `delta p_t`) on the 17 post quarters, with
  sum `theta_post`.
- **Interval.** A 95% bootstrap-t interval from the unrestricted sieve bootstrap of
  that regression: AR(1) with Rademacher innovations as in 3.1, the fitted
  coefficients as truth, bandwidth 4, `B = 999` and seed `[20261019, index]`. The
  interval is
  `[theta_post - q*_0.975 se, theta_post - q*_0.025 se]`, where `q*` are quantiles of
  `(theta*_post - theta_post) / se*`.
- **Sign.** The sign is reportable when the interval excludes zero, without
  multiplicity adjustment. The count is compared with what is expected by chance
  (2.5% per tail).

## 6. Decision rules

- **Primary family.**
  - The sieve bootstrap p-values (bandwidth 4) are Holm-adjusted over the 54 tests.
  - A change survives when its Holm-adjusted p-value is at most 0.05 under the sieve
    scheme and also under the block scheme. A change that survives under one scheme
    only is reported as scheme-dependent.
- **Headline count.** The number of surviving changes, against the published 14.
- **Bandwidth.** For each bandwidth, the number of sieve-Holm survivors.
- **Classical comparison.** The survivors are compared, descriptively, with the
  review's weaker-after-2021 set, recomputed from its committed tables.
- **Status.** Every result is a descriptive statement about coefficient stability, not
  causal evidence. ES and FR keep their definition-flagged post-2021 data, as in the
  published test; the report marks them.
- **Re-test.** The roadmap asks for a re-test once more quarters are available. It
  needs a new snapshot and is not part of this study; the frozen snapshot ends in
  2026-Q2.

## 7. Reproducibility

- The study uses NumPy only and reads only the frozen snapshot.
- The committed tables are regenerated in the ordinary CI by a reproducibility test,
  provided the full computation stays below about three minutes. Otherwise the size
  study is regenerated by a separate command, and the test checks the main tables only.

## Change log

- **After the registered run: B resolution.**
  - With the registered `B = 999` no bootstrap p-value can fall below 1/1000. Holm
    over 54 tests rejects the smallest p only at or below 0.05/54 = 0.00093, so a test
    at the resolution floor could never survive. This is a design flaw that was noticed
    only after the run.
  - The registered analysis is unchanged. A resolution check was added and labelled as
    not pre-registered: both schemes are repeated at bandwidth 4 with `B = 9999` and
    fresh seeds (`[20261017, index, 200]` and `[20261017, index, 201]`).
  - It gives the same verdict: no change survives. The smallest registered sieve
    p-value (0.007) was above the floor, and the smallest block-scheme Holm value with
    `B = 9999` is 0.076.
- **Descriptive addition.** The summary also counts reportable post-2021 signs without
  the participation variants, which are not independent of them.
