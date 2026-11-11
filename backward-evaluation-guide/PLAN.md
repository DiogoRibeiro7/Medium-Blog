# Analysis plan: the backward-evaluation guide

Written and committed on 11 November 2026, before the full computation of `daily.py`.

This folder is a guide, not a test: its main outputs are readings of the train-period ×
test-period matrix on data used in the earlier studies of the series, made with the module
`bwdiag.py`. One part of it is a test, and that part is fixed here.

## What was seen before this plan

While the code was written, `daily.py` was run on two series: electricity load in Italy
(development data of the earlier studies) and the Wikipedia article *Formula One* (one of the
245 test articles of `../backward-evaluation-model-choice`). On both, LightGBM fitted on the
most recent year lost much more log loss on earlier years than the other families, and most of
that loss was not removed by a constant shift of its forecasts; the level-normalised variant
`GBMn` (below) lost almost nothing on earlier years, and its deployed forecasts had a lower log
loss than LightGBM's on both series. To check the combining script, the first two series of each data set in alphabetical order (Austria and Belgium; Afghanistan and Albania; Bangkok and Beijing, the first two test articles) were also run; on all six, `GBMn` had the lower deployed log loss. No other series has been looked at with this code.

## The test

`GBMn` is LightGBM with the settings of the earlier studies, fed the same features expressed
relative to the mean of the previous 28 days (the ten lags and the 7-day mean minus the 28-day
mean, plus the day of the week and the annual sine and cosine), and predicting the day's
deviation from the 28-day mean. Its predictive distribution is Student-t(5) with the scale from
out-of-fold residuals, as for LightGBM. The idea comes from reading the matrix: if a model's
fit to one year fails on other years because its forecasts are tied to that year's levels, a
model that does not see levels should transfer, and might also forecast better.

**Comparison.** Both models deployed as in the model-choice study: refitted on all data to date
every 91 days, one-day-ahead forecasts, log loss over the same evaluation days (from one year
after the first deployed forecast to the end; 2,993 days per article).

**Hypothesis T1.** On the 245 test articles, `mean[L(GBM) − L(GBMn)] > 0`. **Supported** if the
one-sided paired t-test across articles gives p < 0.05; **reversed** if the mean is negative with
p < 0.05 in the other direction; **not supported** otherwise. Robustness, as before: the sign of
the mean in each of the seven lists. Reported alongside, not part of the verdict: the same
comparison on the 304 development articles and the 25 electricity series, and `GBMn` against the
seasonal linear model.

## Readings (descriptive, no verdicts)

- The simulated checks of `sims.py` (what the matrix reads for a known level, spread, dynamics
  or temporary change).
- US CPI inflation and industrial production growth (`macro.py`, data of the first study).
- For every daily series and family: the full matrix and its decomposition; today's model
  (fitted on the last 364 days) on every earlier year; the seasonal linear model monitored every
  91 days (`daily.py`).
