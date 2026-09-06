# Missing Data Is Not Empty Space: Statistical Analysis of an Irregular Blood-Pressure Tracker

A time series is defined not only by the values observed, but also by **when and how often observations were taken**.

This case study starts from a small personal blood-pressure tracker and asks what can be learned when the observation process is irregular, clustered, and dominated by missing calendar days. The private source rows are never committed. The public repository contains only privacy-safe, day-indexed aggregate statistics.

This is a statistical analysis, not medical advice or a clinical interpretation.

## Medium-facing notebook

The executable article is:

- [`blood-pressure-missingness.ipynb`](blood-pressure-missingness.ipynb)

The notebook now synthesises the complete analysis stack rather than reproducing only the original global trend. Its code cells call the same tested Python modules used in CI, so the notebook is a narrative layer rather than a second statistical implementation.

## 1. Observation design

The cleaned source contains:

| Item | Count |
|---|---:|
| Valid measurements | 144 |
| Measurement sessions | 79 |
| Observed calendar days | 25 |
| Calendar days in analysis window | 64 |
| Missing calendar days | 39 |
| Longest missing run | 32 days |

Calendar-day coverage is therefore only **39.1%**, and one 32-day gap accounts for **82.1% of all missing days**.

Sampling intensity is also highly uneven: observed days contain between **1 and 21 readings**. This matters because heavily sampled days tend to have lower observed systolic means. The reading-weighted systolic mean is **115.85 mmHg**, compared with **118.27 mmHg** when each observed day receives equal weight.

That alone makes a flat 144-row i.i.d. analysis a poor default.

## 2. Data quality before modelling

The source requires explicit preprocessing rules rather than silent cleaning:

- thirty early measurements require an Excel day/month inversion repair;
- two later text dates require the same day/month correction;
- six blank placeholder rows contain a derived pulse-pressure value of zero and would bias the spreadsheet pulse-pressure mean downward by about **4%** if treated as observations;
- blank `Meal` and `Symptoms` fields are unknown/not-recorded values, not automatically negative labels.

The public refresh code validates these rules before writing any aggregate output.

## 3. Original global trend

The original exploratory day-level model is

\[
y_t = \beta_0 + \beta_1 t + \varepsilon_t,
\]

fitted only to observed calendar days with HC3 robust covariance.

For systolic pressure, the global estimate is approximately

\[
\boxed{-4.27\ \text{mmHg per 30 days}}
\]

with a 95% HC3 interval of about

\[
[-6.70,-1.83].
\]

That remains a useful descriptive statistic, but it is **not the final interpretation**, because the line spans a 32-day interval with no measurements.

## 4. Sensitivity to observation intensity

The global systolic slope remains negative under ordinary alternatives:

| Specification | Slope per 30 days | 95% HC3 CI |
|---|---:|---:|
| Equal observed day | -4.27 | [-6.70, -1.83] |
| Adjust for `log(1 + readings/day)` | -3.52 | [-6.00, -1.04] |
| Reading-count weighted | -4.63 | [-6.86, -2.39] |
| Capped reading weight | -4.49 | [-6.76, -2.22] |
| Inverse-intensity stress | -3.34 | [-6.87, 0.19] |

The inverse-intensity case is deliberately a **stress test, not inverse-probability weighting**. Observation probabilities and the MCAR/MAR/MNAR mechanism are not identified from this tracker.

## 5. The long gap changes the story

A gap-aware decomposition splits observed days into the two episodes on either side of the unique longest internal missing run.

The pre-gap mean systolic level is about **122.53 mmHg**, and the post-gap mean is about **116.26 mmHg**. In the common-linear gap-aware model, the post-minus-pre episode contrast is approximately

\[
\boxed{-6.27\ \text{mmHg}}
\]

with HC3 interval about

\[
[-10.13,-2.41].
\]

An exact OLS covariance decomposition shows that roughly **87% of the negative global time-pressure covariance** comes from the separation between the two observed episodes.

So the global slope should not be read as evidence of a smooth decline through the unobserved interval.

This is **not change-point detection**. There are no observations inside the 32-day gap, so the data cannot identify when, how, or why the level difference arose.

## 6. Single-day influence

With only 25 observed days, small-sample influence matters. Leave-one-observed-day-out refits hold the full-data episode definition fixed and remove each observed day once.

On the current snapshot:

- the global systolic slope remains negative after every deletion, roughly from **-4.87 to -3.78 mmHg/30d**;
- its HC3 interval remains below zero after every deletion;
- the episode contrast remains negative, roughly **-7.13 to -5.42 mmHg**;
- its HC3 interval also remains below zero after every deletion.

The common within-episode slope is less stable: its point estimate stays negative, but whether its interval excludes zero depends on which day is removed.

Thus the global association and episode contrast are not artifacts of one isolated observed day, while inference on the within-episode slope is more fragile.

## 7. Does the episode contrast survive unequal sampling intensity?

Yes under ordinary choices:

| Specification | Post - pre contrast | 95% HC3 CI |
|---|---:|---:|
| Equal day | -6.27 | [-10.13, -2.41] |
| Sampling adjusted | -5.61 | [-9.98, -1.23] |
| Reading weighted | -6.38 | [-9.99, -2.78] |
| Capped weight | -6.52 | [-10.14, -2.91] |
| Inverse-intensity stress | -5.41 | [-11.29, 0.48] |

Again, only the deliberately aggressive inverse-intensity stress case becomes inconclusive.

## 8. Does the episode contrast depend on the within-episode time form?

The answer is similar:

| Within-episode specification | Post - pre contrast | 95% HC3 CI |
|---|---:|---:|
| No time adjustment | -6.27 | [-10.08, -2.46] |
| Common linear slope | -6.27 | [-10.13, -2.41] |
| Separate linear slopes | -6.27 | [-10.65, -1.89] |
| Common quadratic curvature | -4.86 | [-9.46, -0.27] |
| Separate slopes + common quadratic stress | -4.90 | [-10.06, 0.25] |

The ordinary alternatives preserve a negative episode contrast. Only the most flexible five-parameter model pushes the upper interval slightly above zero.

That model is explicitly a **small-sample stress specification**, not a preferred trajectory. There are only 25 observed days, including eight before the long gap.

## 9. State-space uncertainty

A local-linear-trend Gaussian state-space model is still useful for visualising a latent trajectory across missing calendar days:

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

## 10. What can actually be concluded?

The mature synthesis is narrower than a smooth-trend story and stronger than a single regression coefficient:

1. The observed global systolic association is negative.
2. Most of that association is structurally tied to the separation between two observed episodes around the 32-day gap.
3. The post-gap episode is roughly **5-6 mmHg lower** than the pre-gap episode under ordinary modelling choices.
4. That episode contrast survives ordinary sampling-intensity adjustments, ordinary within-episode functional-form alternatives, and deletion of any one observed day.
5. Deliberately aggressive stress specifications can make the contrast inconclusive.
6. The data cannot identify when or why the episode difference arose inside the unobserved interval.
7. The tracker cannot identify the missingness mechanism as MCAR, MAR, or MNAR from the observed data alone.

So the main lesson is methodological:

> **Missing data are part of the statistical process. A credible analysis should challenge the conclusions created by the observation design rather than erase the gaps and report one smooth line.**

## 11. Reproducibility and privacy

The repository contains:

- `analysis.py` — primary validation, descriptive statistics, HC3 trends, and state-space analysis;
- `observation_process_sensitivity.py` — global trend sensitivity to sampling intensity;
- `gap_aware_trend_decomposition.py` — within/between episode decomposition;
- `day_influence_sensitivity.py` — Cook's distance, DFBETA, leverage, and leave-one-day-out refits;
- `episode_observation_sensitivity.py` — episode contrast sensitivity to sampling intensity;
- `episode_time_form_sensitivity.py` — episode contrast sensitivity to within-episode time form;
- `validate_current_influence_findings.py` — refresh-time gate for snapshot-sensitive scientific claims;
- `blood-pressure-missingness.ipynb` — executable Medium-facing synthesis;
- `data/analysis_snapshot.csv` — privacy-safe relative-day aggregate snapshot;
- `data/source_audit.json` — aggregate source-quality audit.

The raw workbook and private Google Sheet identifiers are never committed.

To run the public analysis:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_*.py' -v
```

The test suite validates and executes the notebook end to end. The manual secret-backed refresh workflow rebuilds only privacy-safe aggregates and derived outputs, then opens a reviewable PR if anything changed.
