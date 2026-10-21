# Monthly outcomes for the identified common shocks: pre-registered protocol

An amendment to [IDENTIFICATION_PROTOCOL.md](IDENTIFICATION_PROTOCOL.md), written and
committed before any monthly estimate was computed. Changes after the first estimate are
listed in the change log at the end, with reasons.

**Data work before this protocol was written.** Only coverage and documentation were
inspected. No regression or correlation involving a shock and an outcome was computed.
The inspection covered:

- Eurostat `une_rt_m` and `sts_inpr_m`: downloaded from 1998-01 to check country
  coverage, gaps and quality flags.
- Eurostat's reference metadata for `une_rt_m` (ESMS page, read on 2026-10-18): how each
  country produces its monthly figures.
- Eurostat's vintage datasets `ei_lm_m_vtg` and `ei_na_q_vtg`: structure only.
- The frozen monetary shock file: its columns, last month, and the number of months
  without an ECB announcement (zero shock).

## 1. Why, and what is asked

The quarterly study ([docs/IDENTIFICATION_RESULTS.md](docs/IDENTIFICATION_RESULTS.md))
judged two common shocks defensibly identified: ECB monetary policy surprises (MP) and
OPEC oil supply surprises (OIL). Their first stages for GDP were weak (|t| at most 1.27
at h = 4 and 8), and every shock-conditional Okun set was unbounded. The decision document
([docs/IDENTIFICATION_STUDY.md](docs/IDENTIFICATION_STUDY.md), section 8) names monthly
outcomes as the next step: the shocks keep their native frequency, the time dimension
triples, and no aggregation assumption (A6) is needed.

Questions:

1. **Responses.** At which horizons (0-24 months) are the responses of the unemployment
   rate and of industrial production to MP and OIL distinguishable from zero?
2. **Ratio.** Is the shock-conditional ratio `kappa_IP(h)`, the unemployment response
   divided by the industrial-production response, informative at h = 12 and h = 24? If it
   is for both shocks at the same horizon, do the demand and supply sets overlap?
3. **Relevance.** Is the first stage (the cumulative industrial-production response)
   stronger than the quarterly GDP first stage? This is descriptive.

**What this cannot answer.**

- **Not Okun's coefficient.** `kappa_IP` is not the GDP-based Okun coefficient.
  - Industrial production covers NACE B-D (mining, manufacturing, energy), roughly a
    fifth of value added and far less in service economies.
  - `kappa_GDP = kappa_IP * (IP response / GDP response)` for each shock. That factor is
    the weak quarterly first stage again, so it is not identified here.
  - No official monthly GDP exists for the EU-27.
- **Unidentified shocks.** Productivity, labour-supply and country-specific GDP shocks
  remain unidentified (decision document, section 4.4).
- **Participation.** It is quarterly only and is not studied here.

## 2. Shocks and assumptions

**Series.** The same published series as the quarterly study, at their native monthly
frequency, from the frozen snapshot `snapshots/external-shocks-2026-10-16`. Nothing is
downloaded again; the raw files are checked against their SHA-256 hashes.

| Shock | Series | Coverage | Normalisation |
| --- | --- | --- | --- |
| **MP** (demand) | `MP_median`, monthly | 1999-01 to 2025-10 | +10 basis points (0.10 on the 1-year OIS scale), a tightening |
| **OIL** (supply, not productivity) | `oil_surprise`, monthly | 1975-01 to 2025-12 | +10% oil supply surprise (negative supply news) |

**Assumptions.** A1 (event-window exogeneity), A2 (purity) and A5 (linearity,
homogeneity) are unchanged from IDENTIFICATION_PROTOCOL.md. The others change as follows:

- **A3. Lead-lag exogeneity.** Partly testable, now at monthly frequency (section 5).
- **A4. Relevance.** Testable. The first stage is the cumulative industrial-production
  response.
- **A6. Aggregation.** No longer needed: no shock is summed over months. A shock is dated
  in the month of the announcement, and the outcome of that month is partly measured
  before the announcement. This attenuates the response at h = 0; it does not bias later
  horizons.
- **A7. Outcome timing (new).** The value published for month m uses no information from
  after month m, apart from standard seasonal adjustment.
  - It fails by construction for trend series, centred moving averages, and monthly
    figures interpolated between quarterly survey results.
  - In those series a "lagged" control or the base month `t-1` can already contain part
    of the response to the shock at t.
  - It is handled by an a-priori country classification (section 3) and partly tested by
    a placebo (section 5).

## 3. Data

**New frozen Eurostat snapshot** (`snapshots/eurostat-monthly-<download date>/`): raw
JSON-stat responses, SHA-256 manifest, processed country-month CSV.

| Quantity | Dataset | Filters | Units |
| --- | --- | --- | --- |
| Unemployment rate | `une_rt_m` | `SA`, age `TOTAL`, sex `T`, `PC_ACT` | % of labour force |
| Industrial production | `sts_inpr_m` | `PRD`, NACE `B-D`, `SCA`, `I21` | index 2021 = 100 |

- **Coverage.** Months from 2000-01 to the last month in which both series are available
  for all 27 countries.
- **Flags.** Eurostat flags are stored and counted in the report; as in every earlier
  stage, they are not used in estimation.
- **Units.**
  - `du[i,m]` is the monthly change of the unemployment rate (pp).
  - `dip[i,m] = 100 * (ln IP[i,m] - ln IP[i,m-1])` (%).

**Country classification (A7).** This comes from Eurostat's reference metadata for
`une_rt_m`, read on 2026-10-18, and is fixed here before estimation.

| Group | How monthly unemployment is produced | Countries | Use |
| --- | --- | --- | --- |
| **D** | Monthly estimates directly from the Labour Force Survey, seasonally adjusted | AT, CZ, EL, FI, HU, IT, RO, SE | **Primary** |
| T | Direct survey estimates, but the series published as "seasonally adjusted" is a trend | DE, NL | Sensitivity only |
| A | Listed as direct, but "the last of the three months is nowcasted" | DK | Sensitivity only (ambiguous) |
| M | Three-month moving average published under the middle month | EE, PT | Sensitivity only |
| I | Quarterly survey results interpolated with registered unemployment (Denton: BE, BG, CY, ES, FR, LV, PL, SK; AR(1): LT) | BE, BG, CY, ES, FR, LT, LV, PL, SK | Sensitivity only |
| B | Benchmarking to moving annual averages of survey results | HR, LU | Sensitivity only |
| U | Method not stated | IE, MT, SI | Sensitivity only |

Industrial production is compiled monthly from business statistics in every country, so
A7 concerns the unemployment series only.

**Primary samples.**

- **MP:** group D countries that are euro-area members at the origin month t (AT, EL, FI
  and IT throughout the sample).
- **OIL:** all eight group D countries.

**Period and pandemic.**

- Origins run from 2009-01, the start of the study period, to the last month for which
  the shock and the outcome at t+h exist.
- An observation at origin t and horizon h is used only if no month from t-12 to t+h lies
  in 2020-01..2021-12.

## 4. Estimation

**Lag-augmented local projections**, pooled over countries with country fixed effects.
For each shock, outcome and horizon h = 0..24:

```text
x[i,t+h] - x[i,t-1] = a_i^h + b^h s[t]
                      + sum_{l=1..12} ( c_l^h s[t-l] + d_l^h du[i,t-l] + e_l^h dip[i,t-l] ) + error
```

- **Outcomes.**
  - The unemployment rate (pp).
  - Cumulative industrial production, `100 * (ln IP[t+h] - ln IP[t-1])` (%).
- **Inference.** Standard errors clustered by month, which is robust to any cross-country
  correlation within a month. Pointwise 68% and 90% bands use t critical values with G - 1
  degrees of freedom, where G is the number of months.
- **Normalisation.** As in section 2.
- **Shock-conditional ratio** at h = 12 and h = 24:

  ```text
  kappa_IP(h) = response of u[t+h] - u[t-1] / response of 100 (ln IP[t+h] - ln IP[t-1])
  ```

  - Units: pp of unemployment per 1% of industrial production.
  - It is estimated by LP-IV, with the shock as instrument and the same controls.
  - Its 90% **Anderson-Rubin (Fieller)** set uses the month-clustered covariance of the
    two reduced-form responses. It is solved in closed form, as in the quarterly study,
    and may be unbounded.
- **Secondary: industrial production, all members.** Industrial production has no timing
  problem, so its response is also estimated over all members: the euro-area members for
  MP and the EU-27 for OIL.
  - The controls are the lags of the shock and of `dip` only, because the `du` lags of
    group T, A, M, I, B and U countries could contain post-shock information.
  - Its first-stage t is reported next to the primary one.
- **Countries (descriptive).** Every country is fitted separately: the euro-area members
  for MP and the EU-27 for OIL.
  - Time-series LPs at h = 12, with Newey-West standard errors (h + 1 lags).
  - Each row is labelled with its group.
  - No rankings and no per-country significance claims.

## 5. Diagnostics

- **Predictability (A3).** The monthly shock is regressed on 12 lags of the cross-country
  means of `du` and `dip`, over the shock's primary countries. The report gives the
  classical F-test of the 24 lag coefficients.
  - The test uses the months entering the primary LP at h = 0, so windows touching the
    pandemic are excluded.
  - The classical F-test is kept for the reason recorded in the quarterly change log: under
    the null, the shock is independent of the regressors. Months without an ECB
    announcement make the MP shock zero-inflated and heavy-tailed, so a unit test checks
    the test's size by simulation with such a shock.
- **Placebo (A7, and A3 again).** The change in the month before the shock is regressed on
  the shock:

  ```text
  x[i,t-1] - x[i,t-2] = a_i + b s[t] + sum_{l=1..12} c_l s[t-l]
                        + sum_{l=2..13} ( d_l du[i,t-l] + e_l dip[i,t-l] ) + error
  ```

  - Under A3 and A7, b = 0. The test is two-sided, month-clustered, with a t(G - 1)
    reference.
  - It is computed for both outcomes, in the primary sample and in the all-members
    sensitivity sample. A trend, moving average or interpolation would show up there.
- **Relevance (A4).** The first-stage t of the cumulative industrial-production response
  at h = 12 and 24.

## 6. Decision rules

- **Identified response.** A response is identified under A1-A5 and A7. It is labelled
  "distinguishable from zero" at a horizon when its 90% band excludes zero.
  - The report names the horizons.
  - It counts the bands that exclude zero out of 50 per shock (2 outcomes × 25 horizons),
    against the 5 expected by chance. The bands are correlated across horizons, so the
    count is only a rough guide.
- **Ratio.** `kappa_IP(h)` is **informative** when its 90% AR set is bounded, and
  **uninformative** otherwise.
  - Demand and supply are compared only if both sets are informative at the same horizon.
    The comparison is descriptive: do the sets overlap?
- **Relevance.** A first-stage |t| below 3.16 (F < 10) is reported as weak. The AR set
  stays valid when the instrument is weak.
- **Predictability.** p < 0.05 is evidence against A3 for that shock. Its results are then
  labelled "identification doubtful".
- **Placebo.** p < 0.05 for either outcome in the primary sample is evidence against A7 or
  A3. That shock's results are then labelled "timing doubtful".
- **Nothing else.** No GDP-based Okun coefficient is computed from these estimates, and no
  reduced-form response is labelled causal.

## 7. Robustness (fixed in advance)

| Variant | Change from the primary specification | Label |
| --- | --- | --- |
| All members as published | Euro-area members (MP) or EU-27 (OIL), every group | "A7 doubtful by construction" for groups other than D |
| `MP_pm` | Poor-man's sign restriction instead of `MP_median` | identified under A1-A7 |
| `MP_raw` | Raw policy surprise `pc1_hf` | contrast only (contains information effects) |
| `OIL_var` | VAR-based oil supply news shock, one standard deviation of the monthly series since 2009-01 | identified under A1-A7 |
| Pandemic included | Every observation | as primary |
| Six lags | 6 instead of 12 lags of every control | as primary |
| Driscoll-Kraay | Bartlett, h + 1 monthly lags, instead of month clustering | as primary |
| Long sample | Origins from 2001-02, the first month with 12 lags of `du` from data starting in 2000-01 | as primary; the report counts the pre-2009 flags |

## 8. Comparison with the quarterly study (descriptive)

- **First stages.** The monthly first-stage t at h = 12 and 24 is set next to the
  quarterly GDP first-stage t at h = 4 and 8, read from the committed
  `docs/identification/okun_sets.csv`.
- **Responses.** The unemployment responses at 12 and 24 months are set next to the
  quarterly ones at h = 4 and 8.
- **Different estimands.** The horizons and the outcome (industrial production against
  GDP) differ, so the comparison is about informativeness, not about equal numbers.

## 9. Not covered

- GDP-based shock-conditional Okun coefficients. These need monthly GDP or an identified
  IP-to-GDP mapping.
- Productivity, labour-supply and country-specific GDP shocks.
- Participation, hours and employment decompositions.
- Real-time data. Revised data are used; no vintages.

## Change log

- **During implementation, before any estimate on real data: clarifications.** None of
  these changes a registered choice; they fill in details the text above left open.
  - *Placebo window.* The placebo's controls reach back to t-13, so its pandemic rule
    uses the window [t-13, t].
  - *Autocorrelation.* The lag-1 autocorrelation in the predictability table is that of
    the shock over every month from 2009-01. It is descriptive.
  - *Country LPs.* Every country's LP uses the primary controls, including the `du` lags.
    The country table labels each country with its group.
  - *Snapshot.* The snapshot was downloaded on 2026-10-18 and ends in 2026-08. Its audit
    passes. Its one missing cell is Estonia's unemployment rate in 2000-01.
- **After the results, 2026-10-19: a post-hoc check, not registered.** An external review
  noted that the change at the longest lag is measured from the level one period before
  the start of the pandemic window, so that level can fall in 2021 while the observation
  is kept. The registered rule is unchanged and its results remain primary. A labelled
  sensitivity widens the window by one period (`eu-okun window-sensitivity`, results in
  [docs/WINDOW_SENSITIVITY.md](docs/WINDOW_SENSITIVITY.md)). No conclusion changes.
- **2026-10-21: an implementation correction to the descriptive country responses.** As
  in the quarterly study, the Newey-West covariance of the country LPs counted its
  h + 1 lags in rows, so rows across a gap in the sample were treated as adjacent.
  - The lags are now counted in calendar months.
  - Only the intervals of the descriptive country table and figure change. The point
    estimates, the panel results and every registered test are unaffected.
