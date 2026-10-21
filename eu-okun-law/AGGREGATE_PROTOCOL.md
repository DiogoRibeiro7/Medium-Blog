# The EU as a whole: a labour-force-weighted aggregate (protocol)

Registered on 2026-10-21 and committed and pushed before its data were frozen or any
estimate was computed. Changes after the first computation go in the change log at the
end, with reasons.

## 1. Question, and what was already seen

Every average in the study weights countries equally, so Malta counts as much as Germany.
Both external reviews asked for an aggregate that weights countries by size. This study
asks two questions:

- What is the Okun coefficient of the EU as a whole?
- Does weighting countries by their labour force change the average country
  coefficient?

**Already known before this protocol was written.**

- The per-country OLS sums on the headline sample, and their unweighted mean. These are
  the case study's unemployment elasticities; the mean is -0.29.
- An availability check of the series in section 2, run before this protocol. It
  confirmed that they exist for every country and for `EU27_2020` from 2009-Q1, and it
  computed nothing else.

The labour-force weights and every EU-level estimate are new. No sign is predicted for
the comparisons.

## 2. Data

A new snapshot, `snapshots/eu-aggregate-<date>/`. It holds raw responses with SHA-256
hashes, a processed table and an audit that re-derives the table.

| Series | Dataset | Filters | Use |
| --- | --- | --- | --- |
| Unemployed persons, thousands | `une_rt_q` | `Y15-74`, `T`, `THS_PER`, `SA` | labour force (below) |
| Unemployment rate, % | `une_rt_q` | `Y15-74`, `T`, `PC_ACT`, `SA` | labour force; the official EU rate |
| GDP growth, % q/q | `namq_10_gdp` | `CLV_PCH_PRE`, `SCA`, `B1GQ` | the official EU growth rate |

Each series covers the 27 countries and `EU27_2020`, for 2008-Q1 to 2026-Q2.

**Labour force.** For country i and quarter t, `LF_it = U_it / (u_it / 100)`, with both
taken from the same seasonally adjusted `une_rt_q` release.

**Vintages.** The country rates and growth of the estimates come from the frozen study
snapshot (`eurostat-2026-10-15`). The EU-level series come from the new snapshot, and
Eurostat may have revised them in between. The report records both release dates.

**Data check (reported, not a test).** An aggregate rate is built from the country data,
`sum_i LF_it u_it / sum_i LF_it`. It should match the official `EU27_2020` rate. The
report gives the largest absolute difference and the correlation of the quarterly changes.

## 3. Estimands and estimators

**Sample.** The headline sample throughout: quarters 2009-Q3 to 2026-Q2, with the window
[t−2, t] clear of 2020-Q1..2021-Q4.

**Country sums.** `theta_i` is the two-lag OLS sum of the unemployment change on GDP
growth, as in the case study (`heterogeneity.lag_rows`). The CCE sums of
`CCE_PROTOCOL.md` are also used, for a descriptive comparison.

**Weights.** `w_i` is country i's share of the EU labour force, averaged over the
quarters 2009-Q3..2026-Q2. The weights are fixed: they describe the population and are
not resampled.

**Estimates.**

- `theta_MG = mean_i theta_i`, the unweighted mean.
- `theta_LF = sum_i w_i theta_i`, the labour-force-weighted mean.
- `theta_EU` is the two-lag sum from the same regression on the official `EU27_2020`
  series: the change in the EU unemployment rate on EU GDP growth.

## 4. Inference

**Bootstrap.** A synchronised circular block bootstrap over quarters, as in the case,
asymmetry and CCE studies.

- Whole quarters are resampled, keeping every country and the EU series together.
- Block length 4, B = 1999, seed **20261029**.
- 95% percentile intervals, and the two-sided p-value `2 min(share ≤ 0, share ≥ 0)`.
- A design is singular when its condition number exceeds 1e8. Such draws are dropped.
- If more than 5% of draws are dropped, the result is labelled unreliable.

## 5. Tests and verdicts

**Tests.** Both are two-sided, with Holm over the two.

- **T1:** `theta_LF − theta_MG ≠ 0`. Does weighting by labour force change the average?
- **T2:** `theta_EU − theta_LF ≠ 0`. Does the aggregate regression give the weighted
  country average? Under heterogeneity it need not. The EU growth rate weights countries
  by GDP, not by labour force. Cross-country correlation of the shocks also enters the
  aggregate.

**Verdicts.**

- **Differs:** adjusted p < 0.05, reported with its sign.
- **Not determined:** otherwise.

**Reported with intervals, without a test.** `theta_EU`, `theta_LF` and `theta_MG`.

## 6. Reported without a test

- The weights of every country, and the five largest.
- The labour-force-weighted mean of the CCE sums, against their unweighted mean.
- The data check of section 2.

## 7. Sensitivity (fixed in advance)

Each setting reports T1 and T2, with their verdicts under the same rule and Holm within
the setting. The primary verdicts are those of section 5.

- **Lag order:** K = 1 and K = 4.
- **Bootstrap:** block length 8.
- **Period:** before 2020 only.

## 8. Use in the manuscript (fixed in advance)

- **Classical section:** a paragraph gives the EU-level coefficient, the weighted and
  unweighted means, and T1 and T2.
- **Discussion:** the aggregate is removed from "standard analyses not done".
- **Medium article:** one or two sentences.

## 9. Outputs

- `eu-okun fetch-eu-aggregate` freezes the snapshot.
- `eu-okun aggregate` writes:
  - `docs/aggregate/weights.csv`;
  - `docs/aggregate/tests.csv`;
  - `docs/aggregate/sensitivity.csv`;
  - `docs/aggregate/summary.json`;
  - the report `docs/AGGREGATE.md`.
- The publication audit checks the snapshot.
- A test regenerates the outputs, and checks the reproduction of the case study's
  country sums.

## Change log
