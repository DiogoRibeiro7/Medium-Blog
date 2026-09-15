# Missing Data Is Not Empty Space: Statistical Analysis of an Irregular Blood-Pressure Tracker

A time series is defined not only by the values observed, but also by **when and how often observations were taken**.

This case study starts from a small personal blood-pressure tracker and asks what can be learned when the observation process is irregular, clustered, and dominated by missing calendar days. The private source rows are never committed. The public repository contains only privacy-safe, day-indexed aggregate statistics.

This is a statistical analysis, not medical advice or a clinical interpretation.

## Medium-facing notebook

The executable article is:

- [`blood-pressure-missingness.ipynb`](blood-pressure-missingness.ipynb)

The notebook synthesises the complete analysis stack and calls the same tested Python modules used in CI. It is a narrative layer rather than a second statistical implementation.

## 1. Observation design

The current privacy-safe snapshot, refreshed from the private Google Sheet, contains:

| Item | Count |
|---|---:|
| Valid measurements | 238 |
| Measurement sessions | 108 |
| Observed calendar days | 31 |
| Calendar days in analysis window | 72 |
| Missing calendar days | 41 |
| Longest missing run | 32 days |

Calendar-day coverage is therefore **43.1%**, and the 32-day gap accounts for **78.0% of all missing days**.

Sampling intensity remains highly uneven: observed days contain between **1 and 22 readings**. Heavily sampled days tend to have lower observed systolic means. The reading-weighted systolic mean is **114.21 mmHg**, compared with **116.97 mmHg** when each observed day receives equal weight.

That makes a flat 238-row i.i.d. analysis a poor default.

The private source also accepts two optional pulse-oximeter fields: `SpO2` for oxygen saturation and `bpm_spo2` for the pulse reported by that same device. `SpO2` is mapped internally to the lowercase `spo2` analysis field. They remain source-specific measurements rather than replacements for the blood-pressure monitor's `bpm`. Historical rows may leave both fields blank. The pulse-oximeter extension has its own aggregate diagnostics, daily relative-day snapshot, coverage-aware longitudinal analysis, and observed-window coverage diagnostic; none of those outputs silently changes the established blood-pressure estimands or article conclusions.

## 2. Data quality before modelling

The current live source still requires explicit preprocessing rules:

- thirty-seven measurements require an Excel day/month inversion repair;
- two text dates require the same day/month correction;
- the current Sheet contains no blank placeholder measurement rows;
- pulse pressure matches `systolic - diastolic` on every valid row in the current source;
- blank `Meal` and `Symptoms` fields remain unknown/not-recorded values, not automatically negative labels.

The public refresh code validates these rules before writing any aggregate output.

## 3. Global descriptive trend

The day-level model

\[
y_t = \beta_0 + \beta_1 t + \varepsilon_t
\]

is fitted only to observed calendar days with HC3 robust covariance.

For systolic pressure, the current estimate is

\[
\boxed{-4.42\ \text{mmHg per 30 days}}
\]

with 95% HC3 interval

\[
[-6.38,-2.46].
\]

This remains descriptive rather than a claim of a smooth trajectory, because the line spans a 32-day interval with no measurements.

## 4. Sensitivity to observation intensity

The global systolic slope remains negative under all five tested weighting and adjustment choices:

| Specification | Slope per 30 days | 95% HC3 CI |
|---|---:|---:|
| Equal observed day | -4.42 | [-6.38, -2.46] |
| Adjust for `log(1 + readings/day)` | -3.14 | [-5.05, -1.23] |
| Reading-count weighted | -4.49 | [-6.16, -2.82] |
| Capped reading weight | -4.51 | [-6.31, -2.72] |
| Inverse-intensity stress | -3.43 | [-6.68, -0.18] |

The inverse-intensity case is deliberately a **stress test, not inverse-probability weighting**. Observation probabilities and the MCAR/MAR/MNAR mechanism are not identified from this tracker.

## 5. The long gap changes the story

The unique longest internal missing run still separates the observed data into two episodes.

The pre-gap mean systolic level is about **122.01 mmHg**, and the post-gap mean is about **115.22 mmHg**. In the common-linear gap-aware model, the post-minus-pre episode contrast is

\[
\boxed{-6.79\ \text{mmHg}}
\]

with HC3 interval

\[
[-10.42,-3.17].
\]

An exact OLS covariance decomposition shows that about **79.1% of the negative global time-pressure covariance** comes from separation between the two observed episodes.

The common within-episode slope is about **-8.60 mmHg per 30 days**, with full-data HC3 interval **[-14.36, -2.84]**. On the refreshed snapshot its leave-one-day-out inference has strengthened: every deletion keeps the within-episode slope negative and its HC3 interval below zero.

This is **not change-point detection**. There are no measurements inside the 32-day gap, so the data cannot identify when, how, or why the level difference arose.

## 6. Single-day influence

With 31 observed days, small-sample influence still matters. Leave-one-observed-day-out refits hold the full-data episode definition fixed and remove each observed day once.

On the current snapshot:

- the global systolic slope ranges from **-4.92 to -3.99 mmHg/30d** and its HC3 interval stays below zero after every deletion;
- the episode contrast ranges from **-7.58 to -5.87 mmHg** and its HC3 interval also stays below zero after every deletion;
- the within-episode slope ranges from **-10.30 to -7.58 mmHg/30d**, and all 31 deletion-specific HC3 intervals remain below zero.

So none of the three negative findings is carried by one isolated observed day on the current snapshot.

## 7. Episode contrast under unequal sampling intensity

The gap-defined post-minus-pre contrast remains negative under all five tested choices:

| Specification | Post - pre contrast | 95% HC3 CI |
|---|---:|---:|
| Equal day | -6.79 | [-10.42, -3.17] |
| Sampling adjusted | -5.46 | [-8.84, -2.07] |
| Reading weighted | -6.67 | [-9.88, -3.47] |
| Capped weight | -6.96 | [-10.36, -3.56] |
| Inverse-intensity stress | -6.09 | [-11.65, -0.52] |

Even the deliberately aggressive inverse-intensity stress contrast is now below zero. That does not identify an observation model; it only shows that this particular contrast is more robust to the tested reweighting on the refreshed snapshot.

## 8. Episode contrast under alternative within-episode time forms

The current live snapshot remains robust to the tested time-form sensitivity analysis:

| Within-episode specification | Post - pre contrast | 95% HC3 CI |
|---|---:|---:|
| No time adjustment | -6.79 | [-10.48, -3.10] |
| Common linear slope | -6.79 | [-10.42, -3.17] |
| Separate linear slopes | -6.79 | [-11.07, -2.52] |
| Common quadratic curvature | -6.37 | [-10.66, -2.08] |
| Separate slopes + common quadratic stress | -6.38 | [-11.27, -1.50] |

All five intervals are now below zero. The five-parameter specification remains a **stress model, not a preferred trajectory**; the sample is still small and only eight observed days precede the long gap.

## 9. Temporal dependence must respect calendar distance

Residual dependence is assessed after removing the established gap-aware common-linear episode structure. The key complication is that consecutive observed rows are not consecutive calendar days: among the 30 adjacent observed-row pairs, the actual spacings are **24 one-day gaps, one two-day gap, four three-day gaps, and one 33-day gap**.

If those 30 pairs are treated mechanically as a single row-order lag, the residual correlation is about **-0.201**. Restricting the comparison to the **24 pairs exactly one calendar day apart within the same gap-defined episode** gives an essentially zero residual correlation of about **-0.049**.

The longer exact-calendar-lag correlations fluctuate with small pair counts. They are therefore reported as descriptive diagnostics, not as a formal test of serial independence. In particular, this analysis does **not** use ordinary row-order Newey-West/HAC inference, because that would treat day 8 and day 41 as if they were one time step apart.

The practical result is methodological: temporal dependence should be indexed by actual calendar distance, not by row position in the observed subset.

## 10. State-space uncertainty

A local-linear-trend Gaussian state-space model remains useful for visualising latent uncertainty across missing calendar days:

\[
y_t = \mu_t + \varepsilon_t,
\]

\[
\mu_{t+1}=\mu_t+\beta_t+\eta_t,
\]

\[
\beta_{t+1}=\beta_t+\zeta_t.
\]

Missing days remain missing observations. The Kalman smoother propagates uncertainty through the latent state; it does not manufacture replacement measurements.

## Pulse-oximeter extension

The pulse-oximeter extension is deliberately separate from the main blood-pressure estimands. The refresh publishes four privacy-safe aggregate artifacts:

- `data/pulse_oximeter_diagnostics.json` — overall coverage, same-device field completeness, descriptive SpO2 summaries, and aggregate cross-device BPM diagnostics;
- `data/pulse_oximeter_daily_snapshot.csv` — daily relative-day counts and means for SpO2, `bpm_spo2`, and the paired difference `bpm - bpm_spo2`;
- `data/pulse_oximeter_longitudinal_diagnostics.json` — coverage-aware descriptive time associations for daily mean SpO2 and daily mean cross-device BPM difference;
- `data/pulse_oximeter_observation_window.json` — pulse capture measured only from the first observed pulse day through the end of the public BP calendar.

The first real secret-backed pulse refresh contains **9 SpO2 readings and 9 paired device-BPM readings**, all on one observed pulse-oximeter day. Relative to all 238 blood-pressure measurements, pulse-oximeter row coverage is therefore **3.8%**. That is a full-history denominator and should not be read as the capture rate once pulse observations begin. The nine SpO2 values have mean **97.56**, median **98**, sample SD **0.88**, and range **96 to 99**. These are descriptive measurements only; no clinical threshold is applied.

For the same nine paired readings, the blood-pressure monitor reports mean BPM **93.56** and the pulse oximeter mean BPM **92.44**. Defining the paired difference as `bpm - bpm_spo2`, the mean difference is **1.11 BPM**, the median difference is **1 BPM**, the mean absolute difference is **1.11 BPM**, the RMSE is **1.41 BPM**, and the maximum absolute difference is **3 BPM**. These diagnostics describe agreement on the observed pairs; they do not establish device interchangeability.

The daily pulse-oximeter snapshot uses the same `day_index` origin as `data/analysis_snapshot.csv`, which permits alignment without exposing calendar dates. The current pulse data occur only on relative day index 71, so there is just **1 pulse-oximeter observed day across the 72-day calendar window**. Longitudinal diagnostics require at least four usable days; consequently both the SpO2 time association and the cross-device BPM-difference time association correctly report `estimable: false` rather than fitting a one-day trend.

A second denominator answers a different question. Starting at the **first observed pulse day** and continuing through the end of the current BP calendar, the observable pulse window contains **1 BP-observed day with 15 BP readings**. Nine of those readings contain SpO2 and `bpm_spo2`, giving an observed-window reading-level capture of **60.0%**. This does **not** identify the true date when the pulse oximeter was introduced. The first observed pulse day is only an observable lower-bound boundary, and pre-window rows are not reclassified as missing pulse measurements.

These outputs do not identify the missingness mechanism, apply clinical thresholds, establish interchangeability between the two BPM devices, or support causal interpretation.

## 11. What can actually be concluded?

The current synthesis is:

1. The observed global systolic association is negative.
2. About four-fifths of that negative global time-pressure covariance is structurally tied to separation between two observed episodes around the 32-day gap.
3. The post-gap episode is about **5.5-7.0 mmHg lower** than the pre-gap episode across the tested sampling-intensity specifications and about **6.4-6.8 mmHg lower** across the tested within-episode time forms.
4. The episode contrast survives all tested sampling-intensity adjustments, all five tested within-episode time forms, and deletion of any one observed day on the current snapshot.
5. The common within-episode slope is also negative under every single-day deletion, although the sampling-adjusted and inverse-intensity within-episode slope specifications remain much less precise than the equal-day fit.
6. Residual-dependence diagnostics change materially when actual calendar spacing is respected; row-order adjacency is not a valid daily lag for this irregular sample.
7. The data cannot identify when or why the episode difference arose inside the unobserved interval.
8. The tracker cannot identify the missingness mechanism as MCAR, MAR, or MNAR from the observed data alone.
9. The current pulse-oximeter sample is sufficient for descriptive SpO2 and paired-BPM device diagnostics, but not for a longitudinal trend because it covers only one observed pulse day. Full-history pulse coverage is **3.8%**, whereas observed-window reading capture is **60.0%**; those percentages use different denominators and answer different questions.

The broader lesson is methodological:

> **Missing data are part of the statistical process. A credible analysis should challenge the conclusions created by the observation design rather than erase the gaps and report one smooth line.**

## 12. Reproducibility and privacy

The supported implementation lives under the canonical `blood_pressure_missingness` package. Important public surfaces include:

- `blood_pressure_missingness.public_analysis` — primary validation, descriptive statistics, HC3 trends, and state-space analysis;
- `blood_pressure_missingness.analyses.observation_process` — global trend sensitivity to sampling intensity;
- `blood_pressure_missingness.analyses.gap_aware` — within/between episode decomposition;
- `blood_pressure_missingness.analyses.day_influence` — Cook's distance, DFBETA, leverage, and leave-one-day-out refits;
- `blood_pressure_missingness.analyses.episode_observation` — episode contrast sensitivity to sampling intensity;
- `blood_pressure_missingness.analyses.episode_time_form` — episode contrast sensitivity to within-episode time form;
- `blood_pressure_missingness.analyses.temporal_dependence` — exact-calendar-lag residual diagnostics that preserve irregular spacing;
- `blood_pressure_missingness.analyses.pulse_oximeter` — aggregate pulse-oximeter diagnostics and daily relative-day snapshot;
- `blood_pressure_missingness.analyses.pulse_oximeter_longitudinal` — coverage-aware SpO2 and cross-device BPM longitudinal diagnostics;
- `blood_pressure_missingness.analyses.pulse_oximeter_observation_window` — coverage after pulse observations first appear, without inferring the device-introduction date;
- `blood_pressure_missingness.validation.current_influence` — refresh-time gate for influence conclusions;
- `blood_pressure_missingness.validation.current_narrative` — consistency gate between the current snapshot and public narrative;
- `blood-pressure-missingness.ipynb` — executable Medium-facing synthesis;
- `data/analysis_snapshot.csv` — privacy-safe relative-day blood-pressure aggregate snapshot;
- `data/source_audit.json` — aggregate source-quality audit;
- `data/pulse_oximeter_diagnostics.json` — aggregate pulse-oximeter coverage and device diagnostics after refresh;
- `data/pulse_oximeter_daily_snapshot.csv` — privacy-safe daily pulse-oximeter aggregate snapshot after refresh;
- `data/pulse_oximeter_longitudinal_diagnostics.json` — coverage-aware pulse-oximeter longitudinal diagnostics after refresh;
- `data/pulse_oximeter_observation_window.json` — privacy-safe observed-window pulse coverage after refresh.

The raw workbook, private Google Sheet identifiers, credentials, calendar dates, and row-level pulse-oximeter readings are never committed.

To run the public test suite:

```bash
python -m pip install -e .
python -m unittest discover -s tests -p 'test_*.py' -v
```

See [`SOURCE_REFRESH.md`](SOURCE_REFRESH.md) for the canonical secret-backed refresh and local reproduction commands.

CI validates the current scientific gates and executes the notebook end to end. The manual secret-backed refresh workflow rebuilds only privacy-safe aggregates and derived outputs, then opens a reviewable PR when public aggregate results change.
