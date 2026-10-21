# Asymmetry between expansions and contractions (protocol)

Registered on 2026-10-21 and committed and pushed before any asymmetric estimate was
computed. Changes after the first computation go in the change log at the end, with
reasons.

## 1. Question, and what was already seen

Does the unemployment rate respond more strongly to falling output than to rising output?
Both external reviews list this among the standard analyses the study lacks. The usual
mechanism is that firing is quicker than hiring, so unemployment rises faster in a
contraction than it falls in an expansion of the same size.

**Already known before this protocol was written.**

- The symmetric classical sums are published (`docs/review/grid.csv`).
- While designing this study, the number of quarters with negative GDP growth in each
  country's headline sample was counted. This used GDP only. They run from 1 (SK) and 2
  (LT) to 25 (EL), with a median of 12 and 320 in all.

No regression with separate positive and negative growth terms was run before this
protocol was pushed.

## 2. Data and sample

- **Data.** `snapshots/eurostat-2026-10-15`: the change in the unemployment rate `du` and
  quarterly GDP growth `g`. No new data.
- **Sample.** The headline sample of every classical estimate: quarters 2009-Q3 to
  2026-Q2, with the window [t−2, t] clear of 2020-Q1..2021-Q4. The rows are the same as
  the case study's (`heterogeneity.lag_rows`).

## 3. Model

Negative growth is split off:

```text
g_neg[t] = min(g[t], 0)
du[i,t] = a_i + sum_{k=0..2} b_ik g[i,t-k] + sum_{k=0..2} c_ik g_neg[i,t-k] + e[i,t]
```

- `theta_plus_i = sum_k b_ik` is the response to growth when growth is positive.
- `theta_minus_i = sum_k (b_ik + c_ik)` is the response when growth is negative.
- The asymmetry `A_i = sum_k c_ik = theta_minus_i − theta_plus_i`. A negative `A_i` means
  unemployment responds more strongly to contractions, because the coefficients are
  negative.

**Primary: common asymmetry, pooled.**

- Every country has its own intercept and its own `b_ik`.
- The `c_k` are common to all 27 countries: `c_ik = c_k`.
- The statistic is `C = sum_k c_k`, estimated by least squares on all rows.
- Pooling uses all 320 negative-growth quarters. The country slopes absorb the
  differences in the size of each country's Okun coefficient.

**Secondary: mean group.**

- The model is estimated country by country, with countries' own `c_ik`.
- The statistic is the unweighted mean of `A_i` over the countries with at least 8
  negative-growth quarters in their rows.
- The country set is read from the data, not typed. Section 1's counts give 23
  countries (all but SK, LT, BE and MT).

## 4. Inference

**Bootstrap.** A synchronised circular block bootstrap over quarters, as in the case and
hours studies.

- Whole quarters are resampled, keeping every country together.
- Block length 4, B = 1999, seed **20261027**.
- Estimates are recomputed with the resampling counts as weights.

**Interval and p-value.**

- A 95% percentile interval.
- The two-sided bootstrap p-value `2 min(share ≤ 0, share ≥ 0)`, as in the case study.

**Degenerate draws.**

- A draw in which a country's weighted design is singular has no estimate for that
  country. This can happen when the draw contains too few of its negative-growth quarters.
- For C, a draw is dropped if any country's symmetric part is singular.
- For the mean group, a draw is dropped if any eligible country is singular.
- The report gives the share of draws dropped. If it exceeds 5%, the result is labelled
  unreliable and is not given a verdict.

## 5. Predictions and verdicts

**Predictions.**

- **P1 (primary):** `C < 0`, unemployment responds more strongly to contractions.
- **P2 (secondary):** mean-group asymmetry `< 0`.

**Verdicts.** Holm over P1 and P2.

- **Supported:** adjusted p < 0.05 with a negative sign.
- **Contradicted:** adjusted p < 0.05 with a positive sign.
- **Not determined:** otherwise.

**What a verdict would mean.**

- A supported asymmetry is a pattern in how unemployment and output moved together. It
  is not evidence of a mechanism.
- The contractions in this sample are concentrated in 2009-2013, and in the countries hit
  by the euro-area crisis. Different shocks could produce the same pattern.

## 6. Reported without a test

- `theta_plus_i`, `theta_minus_i` and `A_i` for every country, with intervals and the
  number of negative-growth quarters. Countries below the eligibility threshold are
  marked.
- The pooled `theta_plus` and `theta_minus` averaged over countries: the mean of
  `sum_k b_ik`, and that mean plus `C`.

## 7. Sensitivity (fixed in advance)

Each setting reports C and the mean-group asymmetry, with their verdicts under the same
rule and Holm within the setting. The primary verdicts are those of section 5.

- **Lag order:** K = 1 and K = 4.
- **Threshold:** each country's mean growth over its rows instead of zero, so that `g_neg`
  is `min(g − mean_i, 0)`. This compares slower and faster growth, not falling and rising
  output.
- **Bootstrap:** block length 8.
- **Countries:** without IE and LU, whose GDP is dominated by multinational accounting
  and by cross-border commuters; and without EL, which has the most negative-growth
  quarters.
- **Period:** before 2020 only (rows up to 2019-Q4).

## 8. Use in the manuscript (fixed in advance)

- **Classical section:** a paragraph reports P1 and P2 with their verdicts.
- **Discussion:**
  - asymmetry is removed from "standard analyses not done";
  - the result is listed under positive, negative or unstable by its verdict and its
    sensitivity.
- **Medium article:** one paragraph.

## 9. Outputs

`eu-okun asymmetry` writes:

- `docs/asymmetry/countries.csv`;
- `docs/asymmetry/tests.csv`;
- `docs/asymmetry/sensitivity.csv`;
- `docs/asymmetry/summary.json`;
- the report `docs/ASYMMETRY.md`.

A test regenerates them, and checks that the symmetric part reproduces the case study's
unemployment elasticities when `c` is fixed at zero.

## Change log
