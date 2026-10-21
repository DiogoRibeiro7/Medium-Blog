# A common-correlated-effects mean-group panel (protocol)

Registered on 2026-10-21 and committed and pushed before any common-correlated-effects
estimate was computed. Changes after the first computation go in the change log at the
end, with reasons.

## 1. Question, and what was already seen

The classical country sums contain the common European cycle. The Bayesian model removes
it with a latent factor, and its sums are smaller than OLS in most countries. Both
external reviews asked for the standard classical answer: Pesaran's (2006)
common-correlated-effects mean-group estimator (CCE-MG). It approximates unobserved common
factors by cross-country averages of the variables.

**Already known before this protocol was written.**

- The per-country OLS two-lag sums on the headline sample. These are the case study's
  unemployment elasticities, and their mean is close to the hours study's
  unemployment mean.
- The Bayesian factor-model sums, which are smaller in magnitude than OLS in 25 of 27
  countries.

So the direction of the comparison in P2 below is anticipated. It is registered as a
test of size, not of a surprise.

## 2. Data and sample

- **Data.** `snapshots/eurostat-2026-10-15`: the unemployment-rate change `du` and GDP
  growth `g`. No new data.
- **Sample.** The headline sample, as in every classical estimate: quarters 2009-Q3 to
  2026-Q2, with the window [t−2, t] clear of 2020-Q1..2021-Q4 (`heterogeneity.lag_rows`).
- **Cross-sectional averages.** For every quarter, `gbar[t]` and `dubar[t]` are the
  unweighted means of `g` and `du` over the countries with data. They are computed on
  every quarter of the snapshot.

## 3. Estimators

For each country i:

```text
OLS:  du[i,t] = a_i + sum_{k=0..2} b_ik g[i,t-k] + e
CCE:  du[i,t] = a_i + sum_{k=0..2} b_ik g[i,t-k]
                    + sum_{k=0..2} d_ik gbar[t-k] + f_i dubar[t] + e
```

- `theta_i = sum_k b_ik`.
- The mean-group estimates are the unweighted means over the 27 countries:
  `theta_MG` (OLS) and `theta_CCE` (CCE).
- The OLS part must reproduce the case study's unemployment elasticities to 1e-9.
  Otherwise the run stops.

## 4. Inference

**Mean-group variance (Pesaran 2006).** `var(theta_CCE) = sum_i (theta_i − theta_CCE)^2 /
(N(N−1))`. The reference is a t distribution with N − 1 degrees of freedom.

**Bootstrap.** A synchronised circular block bootstrap over quarters, as in the case,
hours and asymmetry studies.

- Whole quarters are resampled, with their averages. Each country's regressions are
  re-estimated with the resampling counts as weights.
- Block length 4, B = 1999, seed **20261028**.
- 95% percentile intervals, and the two-sided p-value `2 min(share ≤ 0, share ≥ 0)`.
- A design is singular when its condition number exceeds 1e8. Draws with a singular
  country design are dropped.
- If more than 5% of draws are dropped, the result is labelled unreliable.

## 5. Predictions and verdicts

**Predictions.**

- **P1:** `theta_CCE < 0`. The association survives removing the common factors. Tested
  with the mean-group t statistic.
- **P2:** `theta_CCE − theta_MG > 0`. Removing the common factors makes the mean sum
  less negative, as in the Bayesian factor model. Tested with the bootstrap.

**Verdicts.** Holm over P1 and P2.

- **Supported:** adjusted p < 0.05 with the predicted sign.
- **Contradicted:** adjusted p < 0.05 with the opposite sign.
- **Not determined:** otherwise.

**What a verdict would mean.** A CCE sum is the association with a country's own growth,
beyond what the cross-country averages absorb. Like the Bayesian sum, it is a different
estimand from the classical sum, and neither is causal.

## 6. Reported without a test

- **Every country:** `theta_i` by OLS and by CCE, with bootstrap intervals.
- **Agreement:** the Spearman correlations of the CCE sums with the OLS sums and with
  the Bayesian primary medians (`docs/bayes/country_estimates.csv`).
- **Counts:** the number of countries whose CCE interval lies below zero, and whose CCE
  sum is closer to zero than its OLS sum.
- **The pandemic:** CCE-MG and MG on the full sample, with the pandemic quarters. A
  common shock should be largely absorbed by the averages.

## 7. Sensitivity (fixed in advance)

Each setting reports P1 and P2, with their verdicts under the same rule and Holm within
the setting. The primary verdicts are those of section 5.

- **Lag order:** K = 1 and K = 4. The averages of g take the same lags.
- **Averages:** leave-one-out, excluding the country itself.
- **Bootstrap:** block length 8.
- **Countries:** without IE and LU.
- **Period:** before 2020 only.

## 8. Use in the manuscript (fixed in advance)

- **Classical section:** a paragraph reports P1 and P2, set against the Bayesian
  factor model.
- **Discussion:**
  - the CCE panel is removed from "standard analyses not done";
  - the result is listed by its verdicts.
- **Medium article:** one or two sentences.
- **Bibliography:** Pesaran (2006) is cited.

## 9. Outputs

`eu-okun cce` writes:

- `docs/cce/countries.csv`;
- `docs/cce/tests.csv`;
- `docs/cce/sensitivity.csv`;
- `docs/cce/summary.json`;
- the report `docs/CCE.md`.

A test regenerates them, and checks the OLS reproduction and the estimator on synthetic
data with a known common factor.

## Change log
