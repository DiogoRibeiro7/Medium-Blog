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
| Valid measurements | 171 |
| Measurement sessions | 87 |
| Observed calendar days | 26 |
| Calendar days in analysis window | 65 |
| Missing calendar days | 39 |
| Longest missing run | 32 days |

Calendar-day coverage is therefore **40.0%**, and the 32-day gap accounts for **82.1% of all missing days**.

Sampling intensity remains highly uneven: observed days contain between **1 and 21 readings**. Heavily sampled days tend to have lower observed systolic means. The reading-weighted systolic mean is **115.16 mmHg**, compared with **118.11 mmHg** when each observed day receives equal weight.

That makes a flat 171-row i.i.d. analysis a poor default.

## 2. Data quality before modelling

The current live source still requires explicit preprocessing rules:

- thirty measurements require an Excel day/month inversion repair;
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
\boxed{-4.28\ \text{mmHg per 30 days}}
\]

with 95% HC3 interval

\[
[-6.52,-2.04].
\]

This remains descriptive rather than a claim of a smooth trajectory, because the line spans a 32-day interval with no measurements.

## 4. Sensitivity to observation intensity

The global systolic slope remains negative under ordinary alternatives:

| Specification | Slope per 30 days | 95% HC3 CI |
|---|---:|---:|
| Equal observed day | -4.28 | [-6.52, -2.04] |
| Adjust for `log(1 + readings/day)` | -3.15 | [-5.48, -0.83] |
| Reading-count weighted | -4.88 | [-7.05, -2.71] |
| Capped reading weight | -4.48 | [-6.52, -2.43] |
| Inverse-intensity stress | -3.13 | [-6.62, 0.35] |

The inverse-intensity case is deliberately a **stress test, not inverse-probability weighting**. Observation probabilities and the MCAR/MAR/MNAR mechanism are not identified from this tracker.

## 5. The long gap changes the story

The unique longest internal missing run still separates the observed data into two episodes.

The pre-gap mean systolic level is about **122.53 mmHg**, and the post-gap mean is about **116.15 mmHg**. In the common-linear gap-aware model, the post-minus-pre episode contrast is

\[
\boxed{-6.38\ \text{mmHg}}
\]

with HC3 interval

\[
[-10.10,-2.66].
\]

An exact OLS covariance decomposition shows that about **86.5% of the negative global time-pressure covariance** comes from separation between the two observed episodes.

The common within-episode slope is now about **-9.16 mmHg per 30 days**, with full-data HC3 interval **[-18.19, -0.13]**. Its inferential stability is still weaker than the global and episode-level results because leave-one-day-out significance depends on which observed day is removed.

This is **not change-point detection**. There are no measurements inside the 32-day gap, so the data cannot identify when, how, or why the level difference arose.

## 6. Single-day influence

With 26 observed days, small-sample influence still matters. Leave-one-observed-day-out refits hold the full-data episode definition fixed and remove each observed day once.

On the current snapshot:

- the global systolic slope ranges from **-4.87 to -3.80 mmHg/30d** and its HC3 interval stays below zero after every deletion;
- the episode contrast ranges from **-7.24 to -5.53 mmHg** and its HC3 interval also stays below zero after every deletion;
- every within-episode point estimate remains negative, but deletion-specific significance is mixed.

So neither the global association nor the episode contrast is carried by one isolated observed day.

## 7. Episode contrast under unequal sampling intensity

The gap-defined post-minus-pre contrast remains negative under ordinary choices:

| Specification | Post - pre contrast | 95% HC3 CI |
|---|---:|---:|
| Equal day | -6.38 | [-10.10, -2.66] |
| Sampling adjusted | -4.77 | [-8.43, -1.11] |
| Reading weighted | -6.62 | [-10.22, -3.02] |
| Capped weight | -6.63 | [-10.08, -3.17] |
| Inverse-intensity stress | -5.03 | [-10.92, 0.86] |

Only the deliberately aggressive inverse-intensity stress case is inconclusive.

## 8. Episode contrast under alternative within-episode time forms

The current live snapshot is more robust to this particular sensitivity analysis than the earlier workbook snapshot:

| Within-episode specification | Post - pre contrast | 95% HC3 CI |
|---|---:|---:|
| No time adjustment | -6.38 | [-10.07, -2.69] |
| Common linear slope | -6.38 | [-10.10, -2.66] |
| Separate linear slopes | -6.38 | [-10.65, -2.11] |
| Common quadratic curvature | -5.28 | [-9.83, -0.72] |
| Separate slopes + common quadratic stress | -5.31 | [-10.42, -0.20] |

All five intervals are now below zero. The five-parameter specification remains a **stress model, not a preferred trajectory**; the sample is still small and only eight observed days precede the long gap.

## 9. Temporal dependence must respect calendar distance

Residual dependence is assessed after removing the established gap-aware common-linear episode structure. The key complication is that consecutive observed rows are not consecutive calendar days: among the 25 adjacent observed-row pairs, the actual spacings are **20 one-day gaps, one two-day gap, three three-day gaps, and one 33-day gap**.

If those 25 pairs are treated mechanically as a single row-order lag, the residual correlation is about **-0.158**. Restricting the comparison to the **20 pairs exactly one calendar day apart within the same gap-defined episode** gives an essentially zero residual correlation of about **0.005**.

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

## 11. What can actually be concluded?

The current synthesis is:

1. The observed global systolic association is negative.
2. Most of that association is structurally tied to separation between two observed episodes around the 32-day gap.
3. The post-gap episode is about **5-6 mmHg lower** than the pre-gap episode across the tested ordinary specifications.
4. The episode contrast survives ordinary sampling-intensity adjustments, all tested within-episode time forms, and deletion of any one observed day on the current snapshot.
5. The inverse-intensity sampling stress specification remains inconclusive.
6. Residual-dependence diagnostics change materially when actual calendar spacing is respected; row-order adjacency is not a valid daily lag for this irregular sample.
7. The data cannot identify when or why the episode difference arose inside the unobserved interval.
8. The tracker cannot identify the missingness mechanism as MCAR, MAR, or MNAR from the observed data alone.

The broader lesson is methodological:

> **Missing data are part of the statistical process. A credible analysis should challenge the conclusions created by the observation design rather than erase the gaps and report one smooth line.**

## 12. Reproducibility and privacy

The repository contains:

- `analysis.py` — primary validation, descriptive statistics, HC3 trends, and state-space analysis;
- `observation_process_sensitivity.py` — global trend sensitivity to sampling intensity;
- `gap_aware_trend_decomposition.py` — within/between episode decomposition;
- `day_influence_sensitivity.py` — Cook's distance, DFBETA, leverage, and leave-one-day-out refits;
- `episode_observation_sensitivity.py` — episode contrast sensitivity to sampling intensity;
- `episode_time_form_sensitivity.py` — episode contrast sensitivity to within-episode time form;
- `temporal_dependence_diagnostics.py` — exact-calendar-lag residual diagnostics that preserve irregular spacing;
- `validate_current_influence_findings.py` — refresh-time gate for influence conclusions;
- `validate_current_narrative.py` — consistency gate between the current snapshot and public narrative;
- `blood-pressure-missingness.ipynb` — executable Medium-facing synthesis;
- `data/analysis_snapshot.csv` — privacy-safe relative-day aggregate snapshot;
- `data/source_audit.json` — aggregate source-quality audit.

The raw workbook, private Google Sheet identifiers, credentials, and calendar dates are never committed.

To run the public analysis:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_*.py' -v
```

CI validates the current scientific gates and executes the notebook end to end. The manual secret-backed refresh workflow rebuilds only privacy-safe aggregates and derived outputs, then opens a reviewable PR when public aggregate results change.
