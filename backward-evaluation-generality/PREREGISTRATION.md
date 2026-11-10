# Pre-registration: do the backward-window findings travel to electricity load and web traffic?

Written and committed on 10 November 2026, **before** the data are downloaded and before any
analysis code is committed. The only contact with the data so far is the availability probe in
`probe/` (commit `8719ed8`), which recorded column names, first and last timestamps, observation
counts and canonical Wikipedia titles, never values.

## Background (known before this test)

Three earlier studies (companion folders `../backward-evaluation`, `../backward-evaluation-pooling`,
`../backward-evaluation-episode-end`) tested a backward training-window rule on **monthly
macroeconomic and financial series** with autoregressions: fit the model on the most recent block,
walk back block by block, stop at the first block where the recent model is significantly worse
than the block's own cross-validated model, and train on the blocks that passed. The findings this
study tests:

1. **Pooling over window lengths beat the backward rules.** The equal-weight pool of predictive
   densities over windows of every length (Pesaran and Pick, 2011) had a higher mean gain over the
   expanding window than the plain backward rule in every real-data analysis, and than the rule with
   an end-of-episode check on the ten international series (+0.0177 against +0.0101, Student-t).
2. **The end-of-episode check repairs the plain rule under Gaussian models.** When the walk stops,
   compare a model of the change (window minus its most recent span) with a model of the rejected
   block on the most recent span; if the past model is better, the change has ended and the rule
   trains on all the data. With Gaussian models it removed 79% of the plain rule's losses after
   simulated episodes, 93% on the ten US series and 46% on the ten international series, keeping
   nearly all gains.
3. **Heavy tails do much of the check's work.** With Student-t models the plain rule lost much less
   (total loss on the international series where it lost: 0.2236 Gaussian, 0.0523 Student-t), and
   the check removed less of what remained (31% in simulation, 38% US, 19% international).

On the international series a sensitivity variant with threshold `c = 0` removed 53% of the plain
rule's losses (`c = 1.645`: 19%). That variant was not pre-registered there; here `c = 0` is the
primary threshold and `c = 1.645` is secondary.

Nothing about electricity load or Wikipedia page views has been analysed by this project before.

## Data (fixed now)

**Electricity load (domain `elec`).** Open Power System Data, time series package version
2020-10-06, file `time_series_60min_singleindex.csv` (SHA-256 `6a7f2bc5…c704c8` in the probe log),
columns `<CC>_load_actual_entsoe_transparency` (country level, ENTSO-E Transparency Platform).
Daily load = 24 × the mean of the hours with data on that UTC day, if at least 20 hours have data;
otherwise the day is missing. Inclusion: first hour with data no later than 2015-01-31 23:00; at
most 2% of days from 2015-01-01 to 2020-09-30 missing; no run of more than 7 missing days. Missing
days are filled by linear interpolation of log daily load. The probe implies 29 countries (Cyprus and
Ukraine start too late). Transformed series: `y = log(daily load in MWh)`.

**Web traffic (domain `web`).** English Wikipedia daily page views (Wikimedia pageviews API,
`all-access`, agent `user`) from 2015-07-01 to 2026-10-31 for the canonical titles of the 193 UN
member states and the 118 chemical elements listed in `probe/probe.py`. Inclusion: the candidate
title is the canonical title (no rename; excludes East Timor → Timor-Leste) and the API returned at
least 99% of the 4,141 days (excludes Eswatini, North Macedonia, Nihonium, Moscovium, Tennessine and
Oganesson): 304 articles. Days the API does not return count as 0 views. Transformed series:
`y = log(1 + daily views)`.

## Forecasting setup

One-step-ahead daily forecasts of `y`. The design starts 28 days after the first day (the longest
lag). Block length `L = 364` days, gap 28 days, at most 8 blocks back, `z_crit = Φ⁻¹(1 − 0.05/8) =
2.498`. The first forecast origin is `2L + 28` days after the design start; origins run daily to the
last day. Window decisions and model refits happen every 91 days (re-selection origins); between
them every model is held fixed and forecasts one step ahead from the actual data.

**Model families** (all on `y`):

- `AR`: autoregression of order 7 with intercept;
- `SLM`: seasonal linear model, lags 1–7, 14, 21, 28, six day-of-week dummies, annual Fourier terms
  (two harmonics), intercept;
- `ETS`: additive Holt–Winters without trend, weekly season, smoothing parameters by least squares;
- `GBM`: LightGBM (100 trees, learning rate 0.1, 15 leaves, at least 20 rows per leaf, feature
  fraction 0.9, 63 bins, deterministic) on lags 1–7, 14, 21, 28, the 7- and 28-day means of past
  values, day of week and annual sine and cosine; predictive scale from three contiguous out-of-fold
  blocks of the training window.

**Predictive distributions.** Gaussian, and Student-t with 5 degrees of freedom. For `AR` and `SLM`
the Student-t version is a Student-t regression (maximum likelihood); for `ETS` and `GBM` the point
forecast is the least-squares one and the Student-t scale is estimated by maximum likelihood on the
residuals (one-step in-sample errors for `ETS`, out-of-fold errors for `GBM`).

**Window decisions.** The backward walk and the end-of-episode check use the `SLM` design and a
linear model with the same error distribution as the forecasts (Gaussian decisions for Gaussian
forecasts, Student-t for Student-t), with 5-fold cross-validation within blocks. The chosen window
is then used to train every family. End-of-episode check: when the walk stops at block `j`, a change
model is fitted on `[o − jL, o − 91)` and a past model on the rejected block `[o − (j+1)L, o − jL)`;
both score the last 91 days; `z_end` = mean / Newey–West standard error of (change-model log loss −
past-model log loss); if `z_end > c` the rule trains on the expanding window.

**Schemes** (per family and distribution): `E` expanding window; `B` plain backward rule; `C0`
backward rule with the check at `c = 0` (**primary checked rule**); `C1645` the same at `c = 1.645`;
`P0.25` linear pool `0.25 f_B + 0.75 f_E` whenever the walk stops; `AveW` equal-weight pool over
windows of length `L, 2L, 3L, …` and all data.

**Scores.** Log loss (negative log predictive density of `y`, nats per day). For each series, family,
distribution and scheme, the gain `G` = mean log loss of `E` minus mean log loss of the scheme over
all forecast days. Primary scoring: all days. Secondary: electricity without 2020-03-01 to
2020-06-30.

## Findings tested and verdicts

Cells: 2 domains × 4 families = 8. Unit of analysis: the series. Primary: Student-t distributions,
all days scored (F2 is defined on Gaussian models, as the finding was).

- **F1, pooling over windows beats the backward rules.** Holds in a cell if the cross-series mean of
  `G(AveW) − G(B)` and of `G(AveW) − G(C0)` are both positive.
- **F2, the check repairs the plain rule under Gaussian models.** Holds in a cell if, with Gaussian
  distributions, over the series where `G(B) < 0`, `Σ G(C0) ≥ ½ Σ G(B)`, and over the series where
  `G(B) > 0`, `Σ G(C0) ≥ ½ Σ G(B)` (a part with no series is not testable; a cell with neither part
  testable is not testable).
- **F3, heavy tails do much of the check's work.** Holds in a cell if the plain rule's mean gain over
  the expanding window is higher with Student-t than with Gaussian distributions
  (`mean G_t(B) > mean G_g(B)`) **and** the check's mean improvement on the plain rule is smaller with
  Student-t than with Gaussian distributions (`mean[G_t(C0) − G_t(B)] < mean[G_g(C0) − G_g(B)]`).

Verdict for each finding: **travels** if it holds in at least 6 of the 8 cells (at least 75% of the
testable cells); **travels partly** if it holds in at least 3 (at least 37.5%); **does not travel**
otherwise. For every comparison the cross-series mean, its standard error and a one-sided paired
t-test p-value are reported; the verdicts use the signs of the means, as in the earlier studies.

## Secondary analyses (reported, not part of the verdicts)

- F1 and F3 with `C1645` in place of `C0`; F1 with Gaussian distributions.
- Electricity without March–June 2020 in the scoring.
- Wikipedia countries and elements separately.
- Whether the plain rule, the checked rule and the fixed pool beat the expanding window at all, by
  family (mean `G` and the share of series with `G > 0`).
- Decisions: share of re-selections in which the walk stopped and in which the check switched, by
  domain and distribution; their timing around March 2020 for electricity.
- Absolute comparison of the families (mean log loss of `E` by family and distribution).
