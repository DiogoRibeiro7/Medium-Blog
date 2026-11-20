# Pre-registration: the success specification decides which forecast wins

Written on 20 November 2026, after the M5 archive was downloaded and its file list and calendar were read, and
before any sales figure was read or any number below was computed. Anything changed after this file was committed is
listed under "Changes after registration", with its reason and date, and marked as such in the article.

## Question

A forecasting team that starts modelling before it has written down what success means still chooses a definition
of success: the one built into its tools, usually an accuracy metric at the level of the data it was given. This
plan tests whether that choice matters, on a demand-planning problem: does the model a team would pick under a
default definition of success differ from the one it would pick under a definition written with the planners who
use the forecast, and how much of the available improvement is lost by picking under the default?

A **success specification** is written here as five clauses:

1. **Unit:** the level at which the forecast is used (item x store x week).
2. **Timing:** which part of the future the decision can still change (the week the order covers).
3. **Cost:** what an error costs in that decision (a missing unit costs more than an excess one).
4. **Tolerance:** how many large misses the users accept before they stop trusting the forecast.
5. **Stability:** how much the forecast for the same week may move between two runs, since every move has to be
   explained by the planner to someone else.

The specifications below are written for a stylized planner, not elicited from real ones; the test is whether the
answer moves with the specification, not whether these are the right clauses for every retailer.

## Data

M5 (Walmart unit sales, Makridakis, Spiliotis and Assimakopoulos, 2022): daily unit sales of 3,049 products in 10
stores in California, Texas and Wisconsin, 29 January 2011 to 22 May 2016 (`sales_train_evaluation.csv`, days
d_1 to d_1941), with the calendar (events, SNAP days) and weekly prices. The archive is the M5 data mirrored in
`Nixtla/m5-forecasts` on GitHub, SHA-256 `cc704ba15d6802f8262e6ec7d4c6041e4ad6366a94365e8c84f721e450eed774`;
`fetch.py` downloads it and checks the hash.

**Scope:** the FOODS category in all 10 stores (1,437 products x 10 stores). A series (product x store) is included
if it sold at least one unit in the 364 days up to the first origin (d_1107 to d_1470). Sales are recorded sales,
not demand: the data carry no stock-out flags, so a week with a stock-out understates demand for every model alike.

## Origins and windows

Days are M5 day numbers. Weeks run Saturday to Friday, as in the M5 calendar. An **origin** o is a Friday; the data
up to and including day o are known.

- **Test origins:** the 52 Fridays d_1568 (15 May 2015) to d_1925 (6 May 2016), in four quarters of 13 origins.
- **Warm-up origins:** the 14 Fridays d_1470 to d_1561, used only to give every test origin 13 earlier weekly
  errors (the cost clause).
- From each origin: **week A** is days o+1 to o+7, **week B** is days o+8 to o+14.

Models are refitted at the first origin of the warm-up block and of each test quarter (five fits), on data up to
that origin only.

## Candidates

Daily models, each producing forecasts for days o+1 to o+28:

- **snaive:** the value on the same weekday in the last observed week.
- **ma28:** the mean of the last 28 days (the **incumbent**: the four-week average a planner keeps in a spreadsheet).
- **dow4:** the mean of the same weekday over the last four weeks.
- **ses:** simple exponential smoothing; alpha chosen per series at each refit from {0.02, 0.05, 0.1, 0.2, 0.3} by
  one-step-ahead squared error over the 364 days before the refit origin.
- **lgb_l2, lgb_l1, lgb_tweedie:** one LightGBM model for all series, direct (the horizon h = 1 to 28 is a feature),
  with squared-error, absolute-error and Tweedie (variance power 1.1) objectives.

Weekly models, built only if the unit and timing clauses are written first, each producing week A and week B totals:

- **lgbw_tweedie:** LightGBM on weekly totals (week index 1 or 2 as a feature), Tweedie objective.
- **lgbw_q80:** the same on the 0.8 quantile objective.

LightGBM features, all computed at the origin from data up to the origin, plus calendar and price of the target
day or week: horizon; target weekday (daily) ; month; event type of the target day (daily) or number of event days
in the target week (weekly); SNAP flag of the store's state on the target day, or SNAP days in the target week; price
in the target week, its ratio to the series' mean price over the 52 weeks before the origin, and its ratio to the
price in the origin's week; the series' means over the last 7, 14, 28, 56 and 112 days; the standard deviation over
the last 28; the share of zero days in the last 28; days since the last sale (capped at 112); the mean of the target
weekday over the last 4 and 8 weeks (daily); the mean over the same 7 days a year earlier (target day minus 364,
plus or minus 3 days, daily; target week minus 52 weeks, weekly); the product's mean over all 10 stores in the last
28 days; store and department as categorical features.

Training rows: the 52 weekly origins before the refit origin whose targets end on or before it; a 10% random sample
of (series, origin) pairs for the daily models (all 28 horizons of each sampled pair) and a 50% sample for the
weekly models (both weeks). Settings: 400 rounds, learning rate 0.05, 63 leaves, at least 100 rows per leaf, 80%
feature and row subsampling, seed 7, deterministic mode, 2 threads. No tuning. Forecasts below zero are set to zero.

A daily model's week total is the sum of its daily forecasts over the week.

## Specifications

- **S0, default.** Weighted absolute percentage error (WAPE: sum of absolute errors over sum of sales) of daily
  forecasts at item x store x day, days o+1 to o+28, from the 13 test origins spaced four weeks apart (d_1575 to
  d_1911), which together cover the test year once. Candidates: the seven daily models (the weekly ones would not
  exist without the unit clause).
- **S1, unit and timing.** WAPE of week B totals at item x store, over all 52 test origins. All nine candidates.
- **S2, cost.** The order for week B is the forecast plus the 0.8 quantile of the model's own week B errors (sales
  minus forecast) for that series at the 13 most recent origins whose week B had ended by the origin (o-14 back to
  o-98); lgbw_q80 orders its forecast with no adjustment. Cost per series-week: 0.8 x max(sales - order, 0) +
  0.2 x max(order - sales, 0), the cost of a week in which a missing unit costs four times an excess unit. Summed
  over series and test origins and divided by total week B sales. All nine candidates.
- **S3, tolerance.** A week B forecast is a **bad week** if its absolute error exceeds the larger of 5 units and 50%
  of that week's sales. Metric: the share of series-quarters (13 test origins each) with more than one bad week.
  The point forecast is judged (for lgbw_q80, its forecast). All nine candidates.
- **S4, stability.** The revision of a week: the absolute difference between the week B forecast at origin o and the
  week A forecast of the same week at origin o+7. Metric: total revisions over total sales of those weeks, test
  origins d_1568 to d_1918 (51 pairs). All nine candidates.
- **S5, full specification.** The lowest S2 cost among the candidates that are no worse than the incumbent (ma28) on
  S3 and on S4 (point estimates). If none is, the incumbent stays.

Lower is better on every metric. A specification's **winner** is its best candidate on the point estimate.

## Uncertainty

Products are resampled with replacement, 1,000 times, seed 11, each with all its stores' series; every metric is
recomputed from the resampled totals. A difference counts as established if its 95% percentile interval excludes
zero in the stated direction.

## Hypotheses

- **H1, the winner depends on the specification.** The S0 winner is not the winner of at least two of S1 to S4, and
  in each of those the specification's winner beats the S0 winner on that specification by an established margin.
  Supported if at least two specifications meet both conditions.
- **H2, choosing by the default gives up most of the available gain.** Under S5, with I the S2 cost of the
  incumbent, W the S2 cost of the S5 winner and D the S2 cost of the S0 winner: the **captured share** is
  (I - D) / (I - W) if the S0 winner is admissible under S5, and 0 if it is not (the planner keeps the incumbent).
  Supported if the captured share is below 0.5, with the upper end of its 95% interval below 0.5 when the S0 winner
  is admissible. Not testable if the S5 winner is the incumbent.
- **H3, accuracy and stability pull apart.** Kendall's tau between the orderings of the nine candidates under S1 and
  under S4 is negative, with its 95% interval below zero.

## Also reported

- Every metric for every candidate, with intervals; the bias of week B totals (sum of forecast minus sales over sum
  of sales); the share of week B sales covered by the S2 order.
- The unconstrained captured share, (I - D) / (I - W2) with W2 the S2 winner, alongside H2.
- **Robustness, not deciding:** S2 at a cost ratio of 0.7 and 0.9 (quantile and cost changed together); S3 with
  thresholds (3 units, 30%) and (10 units, 70%); S0 with the root mean squared error in place of WAPE; S2 with the
  daily models' week B order taken as the forecast with no adjustment ("as delivered").
- The S1 to S4 results by test quarter.

## Changes after registration

- 20 November 2026, after the first evaluation: ma28 and dow4 give the same week totals by construction (the four
  weekday means of the last four weeks sum to seven times the 28-day mean), yet their S3 shares differed (0.477 and
  0.468). The cause was float32 rounding of errors that equal the bad-week threshold exactly, which the registered
  rule ("exceeds") counts as not bad. Forecasts are now rounded to 0.001 units before the comparison; the rule is
  unchanged. All S3 numbers in the article come from the corrected run; no verdict depended on the change.
