# Structural identification: pre-registered protocol

Written and committed before any structural estimate was computed. Before this protocol
was written, the only data work was:

- fetching and inspecting the public shock series (file structure, coverage, licence);
- checking that the Eurostat datasets exist.

The methodological reasoning (which strategies are defensible, and why) is in
[docs/IDENTIFICATION_STUDY.md](docs/IDENTIFICATION_STUDY.md). This protocol fixes only
what is estimated, how, and how it will be judged. Changes after the first estimate are
listed in the change log, with reasons.

## 1. What is and is not estimated

| Research question | Identifiable with public data? | Estimated here |
| --- | --- | --- |
| Do demand-driven and productivity-driven changes in GDP have different employment effects? | **Demand: partly.** A euro-area monetary policy shock is identified by high-frequency surprises. **Productivity: no.** No credible quarterly instrument exists for EU countries. | Responses to a monetary policy shock (demand), and to an oil supply news shock (a supply shock, **not** a productivity shock) |
| Can labour-supply shocks be distinguished from aggregate-demand shocks? | **No.** There is no instrument for labour-supply shocks; sign restrictions would be set-identified and prior-dominated. | Not estimated. The participation response to the two identified shocks is reported descriptively. |
| Under which assumptions are country-specific GDP shocks identifiable? | **Not with public data for the EU-27.** It would need country-specific instruments. | Not estimated. Countries' responses to the *common* shocks are reported. |

No recursive (Cholesky) ordering, long-run restriction or sign restriction is used. No
reduced-form or orthogonalised impulse response is labelled causal.

## 2. Identified shocks and their assumptions

| Shock | Series (public, CC BY 4.0) | Units of the quarterly shock |
| --- | --- | --- |
| **MP**: euro-area monetary policy | Jarociński and Karadi (2020), updated; `MP_median`, the sign-restricted monetary policy component of high-frequency OIS surprises around ECB announcements (monthly, from 1999) | Sum of the monthly values, in percentage points of the 1-year OIS rate |
| **OIL**: oil supply news | Känzig (2021), updated; the oil supply surprise series, i.e. the change in Brent futures prices around OPEC announcements (monthly, from 1975) | Sum of the monthly values, in % |

**Assumptions for each shock.** Testability is in brackets.

- **A1. Exogeneity in the event window.** Within the narrow window around an
  announcement, the surprise moves only because of the announcement. *(Not testable.
  Defended by high-frequency identification.)*
- **A2. Purity.**
  - *(MP)* The sign restriction separates monetary policy from central-bank information
    effects: a policy shock moves interest rates and stock prices in opposite directions.
    *(Not testable.)*
  - *(OIL)* OPEC announcements are not responses to European demand news in the window.
    *(Not testable.)*
- **A3. Lead-lag exogeneity.** The quarterly shock is uncorrelated with other structural
  shocks at all leads and lags.
  - *Partly testable.* The shock should not be predictable from lagged EU
    macroeconomic data. Test: regress the shock on four lags of the cross-country mean
    unemployment change and GDP growth; report the joint F-test and its p-value.
- **A4. Relevance for these outcomes.** *(Testable.)* The first-stage response of
  cumulative GDP to the shock is reported with its t-statistic. Inference on the
  shock-conditional Okun coefficient is weak-instrument robust, so validity does not
  depend on relevance.
- **A5. Linearity and homogeneity.** Responses are linear in the shock. The pooled
  coefficient is an average over the countries in the sample. *(Linearity is not
  tested.)*
- **A6. Aggregation.** Monthly surprises summed to quarters are the quarterly shock.
  *(A modelling choice.)*

## 3. Data

- **Outcomes.** These come from the frozen Eurostat snapshot `snapshots/eurostat-2026-10-15`:
  - the unemployment rate `u`;
  - the participation rate `p`;
  - real GDP growth `g` (% quarter on quarter);
  - cumulative GDP: `Y[t+h] - Y[t-1] = sum_{j=0..h} g[t+j]` (%, approximately 100 times
    the log change).
- **Shocks.** Frozen in `snapshots/external-shocks-2026-10-16/`:
  - the raw files, downloaded from pinned commits;
  - a SHA-256 manifest with source URLs, commits, licences and citations;
  - the quarterly series.
- **Coverage.**
  - **MP:** 2009-Q2 to 2025-Q3. The shocks end in October 2025, so 2025-Q4 is incomplete
    and dropped. Country-quarters are included only when the country is in the euro area
    at the origin quarter t (euro-area entry dates are in the code).
  - **OIL:** 2009-Q2 to 2025-Q4, all 27 countries.
- **Pandemic.** In the primary sample, an observation at origin t and horizon h is used
  only if no quarter from t-4 to t+h lies in 2020-Q1..2021-Q4. A sensitivity analysis
  uses every observation.

## 4. Estimation

**Lag-augmented local projections** (Jordà 2005; Montiel Olea and Plagborg-Møller 2021).
For each shock, outcome and horizon h = 0..8:

```text
x[i,t+h] - x[i,t-1] = a_i^h + b^h s[t] + sum_{l=1..4} ( c_l^h s[t-l] + d_l^h du[i,t-l]
                      + e_l^h g[i,t-l] + f_l^h dp[i,t-l] ) + error
```

- **Fit.** The model is pooled over countries with country fixed effects.
- **Inference.** Standard errors are clustered by quarter, which is robust to any
  cross-country correlation within a quarter. Lag augmentation justifies
  heteroskedasticity-robust inference across horizons.
- **Bands.** 68% and 90% pointwise bands.
- **Normalisation.**
  - **MP:** a +10 basis point shock (0.10 on the 1-year OIS scale), which tightens policy.
  - **OIL:** a +10% oil supply surprise, which is negative oil supply news.
- **Shock-conditional Okun coefficient** at h = 4 and h = 8:

  ```text
  kappa_h = response of u[t+h] - u[t-1] / response of Y[t+h] - Y[t-1]
  ```

  - It is estimated as an LP-IV: the cumulative unemployment change is regressed on
    cumulative GDP, instrumented by the shock, with the same controls.
  - Its 90% **Anderson-Rubin** confidence set comes from inverting the time-clustered
    t-test over a grid of kappa values. The set may be unbounded or a union of intervals,
    and is reported as such.
- **Country responses.** Country-by-country LPs at h = 4, with Newey-West standard errors
  (h + 1 lags). These are descriptive, with no rankings and no per-country significance
  claims.

## 5. Decision rules

- **Identified response.** A response is **identified under A1-A6**. It is labelled
  "distinguishable from zero" at a horizon when its 90% band excludes zero.
  - The text names the horizons, not the shape.
- **Okun coefficient.** The shock-conditional Okun coefficient is **informative** when
  its 90% AR set is bounded, and **uninformative** otherwise.
  - Demand-driven and supply-driven Okun coefficients are compared only if both are
    informative at the same horizon. The comparison is descriptive: do the sets overlap?
- **Relevance.** A first-stage |t| below 3.16 (F < 10) is reported as weak. AR inference
  stays valid when the instrument is weak.
- **Predictability.** A predictability p-value below 0.05 is reported as evidence against
  A3 for that shock. Its responses are then labelled "identification doubtful".
- **No other model is estimated, and no reduced-form response is labelled causal.**

## 6. Robustness (fixed in advance)

- **MP decomposition.** `MP_pm` (poor-man's sign restrictions) instead of `MP_median`,
  and the raw policy surprise `pc1_hf`. The raw surprise still contains information
  effects; it is reported as a contrast, not as an identified shock.
- **OIL measure.** The VAR-based oil supply news shock instead of the raw surprise.
- **Pandemic.** Every observation included.
- **Lags.** Two lags instead of four.
- **Standard errors.** Driscoll-Kraay with h + 1 Bartlett lags instead of time-clustered
  ones.

## Change log

- **Before any estimate on real data: the covariance of the predictability F-test (A3).**
  - The protocol did not specify it.
  - Simulations with about 64 quarters and nine regressors showed that HAC, HC1 and HC3
    covariances over-reject. HC1 rejected 17% of the time at a nominal 5%.
  - The classical F-test kept its size.
  - Under the null the shock is independent of the lagged regressors, so its variance
    cannot depend on them and the classical F-test is valid. It is used; a unit test
    checks its size by simulation.
- **After the results, 2026-10-19: a post-hoc check, not registered.** An external review
  noted that the change at the longest lag is measured from the level one period before
  the start of the pandemic window, so that level can fall in 2021 while the observation
  is kept. The registered rule is unchanged and its results remain primary. A labelled
  sensitivity widens the window by one period (`eu-okun window-sensitivity`, results in
  [docs/WINDOW_SENSITIVITY.md](docs/WINDOW_SENSITIVITY.md)). No conclusion changes.
