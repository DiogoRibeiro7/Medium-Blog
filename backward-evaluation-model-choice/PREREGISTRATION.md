# Pre-registration: does backward evaluation choose better forecasting models than a forward backtest?

Written and committed on 10 November 2026, **before** the test data are downloaded and before any
analysis code exists. The only contact with the test series so far is the availability probe in
`probe/` (commits `a58d924` and `d545f5d`), which recorded canonical titles and day
counts, never values.

## The question

The standard way to choose between candidate forecasting models is a **forward backtest**: score
each candidate's forecasts over a recent stretch of data that came after its training data, and keep
the best. **Backward evaluation** inverts the direction: fit each candidate on the most recent data
and score it on the past. A candidate that forecasts the past well from a model of the present is one
whose structure holds across regimes; the bet is that such a candidate will also hold up in the
future, which the forward backtest, scoring a single recent stretch, may not reveal.

## Background (known before this test)

Four earlier studies in this repository used backward evaluation to choose **training windows**, not
models. The window rule lost to pooling over window lengths in every real-data analysis
(`../backward-evaluation-generality`: 8 of 8 domain × family cells), and the strength of a past
rejection did not predict whether a change persisted (`../backward-evaluation-pooling`). On the
development data below (electricity load of 25 countries; page views of 304 Wikipedia articles) the
seasonal linear model with Student-t errors had the lowest mean log loss of the four families in both
domains, and Student-t distributions beat Gaussian ones in every family. Model choice by backward
evaluation has not been tested on any data.

A conjecture stated before this registration: backward evaluation helps most when the candidates
differ in how robust they are across regimes (different model families), and least when they are
settings of one model (for example autoregressive orders).

## Data

**Development set (in-sample, reported but not part of any verdict).** The data of
`../backward-evaluation-generality`: daily electricity load of 25 European countries (2015-01 to
2020-09) and daily page views of 304 English Wikipedia articles on UN member states and chemical
elements (2015-07 to 2026-10), with the same transformations (log load; log(1 + views)).

**Test set (gives the verdicts).** Daily English Wikipedia page views (pageviews API, `all-access`,
agent `user`, 2015-07-01 to 2026-10-31) of the articles in the seven lists of `probe/probe.py`: 50 US
states, 40 large cities, 25 Solar System bodies, 30 programming languages, 30 musical instruments, 35
sports and 40 mammals (250 candidates). Inclusion, as in the earlier study: the candidate title is
the canonical title (excludes the renamed Bangalore → Bengaluru, Kotlin, Lua and Swimming (sport) →
Competitive swimming) and the API returns at least 99% of the 4,141 days (excludes C++, 4,099 days):
**245 articles**. Days the API does not return count as 0 views; `y = log(1 + views)`.

## Candidates

All models from `../backward-evaluation-generality/models.py`, one-step-ahead daily forecasts of `y`.

- **Set A, model families (primary):** `AR` (order 7), `SLM` (seasonal linear: lags 1–7, 14, 21, 28,
  day-of-week, annual Fourier terms), `ETS` (additive Holt–Winters, weekly season), `GBM` (LightGBM,
  100 trees); all with Student-t(5) predictive distributions.
- **Set B, settings of one model (for H2):** autoregressions of order 1, 7, 14 and 28, Student-t(5).
- **Set A8 (secondary):** the four families with Gaussian and with Student-t distributions.

## Deployment

Every candidate is **deployed on the expanding window**: refitted on all data to date every 91 days
and held fixed in between, forecasting one day ahead from the actual data. Every selection procedure
chooses among these deployed forecasts, so the procedures differ only in what they choose.

Selection decisions are taken every 91 days, from the first origin at which a full year of deployed
forecasts exists for the forward backtest. With the design start 28 days after the first day and the
first deployed forecast `2 × 364 + 28` days after that (as in the earlier study), the first decision
is 364 days after the first deployed forecast. The chosen candidate's deployed forecasts are used for
the next 91 days.

## Selection procedures

At decision origin `o`, for each candidate:

- **FWD, forward backtest:** mean log loss of the candidate's deployed forecasts over `[o − 364, o)`.
- **BWD, backward evaluation:** fit the candidate on the most recent year `[o − 364, o)` and score its
  one-step forecasts on every earlier day `[0, o − 392)` (a 28-day gap), equal weights; mean log loss.
  For ETS the smoothing parameters come from the recent-year fit and the states are filtered from the
  first day of the scored span (first 28 days not scored).
- **HYB:** mean of the FWD and BWD scores.

Each procedure picks the candidate with the lowest score. Comparators:

- **ENS:** equal-weight pool of all candidates' deployed predictive densities (no choice);
- **FIX:** the candidate with the lowest mean log loss on the development set over the same decision
  periods, computed by the analysis code before the test results are read;
- **ORACLE:** the best candidate in each 91-day period in hindsight (for scale only).

## Hypotheses and verdicts

Unit: the series. `L(·)` = mean log loss of a procedure over all forecast days after the first
decision. Tests: one-sided paired t-tests across the 245 test articles.

- **H1 (main), backward evaluation chooses better models than the forward backtest, set A:**
  `mean[L(FWD) − L(BWD)] > 0`. **Supported** if positive with p < 0.05; **reversed** if negative with
  p < 0.05 in the other direction; **not supported** otherwise.
- **H2 (the conjecture), the advantage is larger across families than across settings:**
  `mean[(L_A(FWD) − L_A(BWD)) − (L_B(FWD) − L_B(BWD))] > 0`, same verdict rule.
- **H3 (from the earlier studies), pooling the candidates beats choosing one, set A:**
  `mean[L(BWD) − L(ENS)] > 0` **and** `mean[L(FWD) − L(ENS)] > 0`, each with p < 0.05.

Because articles share platform-wide influences, the series-level p-values overstate precision. As a
robustness check registered now, each comparison is also reported over the seven list-level means
(a sign count out of 7).

## Secondary analyses (reported, not verdicts)

- H1 and H3 with set A8 and with set B; HYB against FWD and BWD.
- The same analyses on the development set (electricity; the 304 earlier articles), labelled
  in-sample.
- Selection quality: how often each procedure picks the period's best candidate; regret against
  ORACLE; how often each procedure changes its choice.
- Predictive value of the scores: within each decision, the Spearman correlation between a
  procedure's scores and the candidates' realised log loss over the next 91 days, averaged over
  decisions and series.
- A variant of BWD that scores only the year before the gap (`[o − 756, o − 392)`).
