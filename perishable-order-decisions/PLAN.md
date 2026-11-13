# Pre-registration: algorithms against the decisions around them, for daily orders of perishable goods

Written and committed on 13 November 2026, before the data were downloaded or any number below was computed. The
code was checked only on synthetic data with the dataset's column layout (`tests/`). The
workflow downloads the data only after this file is on the branch. Anything changed after this commit is listed
under "Changes after registration", with its reason and date, and marked as such in the article.

## Question

A conjecture left open in an earlier replay of a hotel cancellation system: in operational forecasting, the
decisions around the model move value far more than the choice of algorithm, and most production failures are
calibration failures that ranking metrics miss. This plan tests it on one operation: the daily order of perishable
goods, where an unsold unit is wasted at the end of the day and a missing unit is a lost sale.

"Decisions around the model" are fixed here as two choices a team makes regardless of the algorithm:

- **What the model is trained on.** Recorded sales, or demand recovered for the hours a product was out of stock.
- **How a forecast becomes an order.** Order the forecast itself, or order a cost-based quantile of demand built
  from recent forecast errors, scaled for all series together or for each series.

## Data

FreshRetailNet-50K (Dingdong-Inc, CC BY 4.0), from the Hugging Face dataset `Dingdong-Inc/FreshRetailNet-50K`:
50,000 store-product series of a fresh-food retailer, 898 stores, 18 cities, 90 days in the `train` split and 7 in
the `eval` split, with hourly sales and an hourly out-of-stock status. Sales are rescaled by an undisclosed global
coefficient, so all costs are reported per unit of demand.

The workflow records the dataset revision it downloads (`sources/REVISION`) and the SHA-256 of both files
(`sources/FILES`); later runs use and verify that revision.

**Structural checks, before anything else.** Every series has the same 97 dates, 90 in `train` and 7 in `eval`, all
`eval` dates after all `train` dates; every hourly list has 24 values. The coding of the hourly status (which value
means out of stock) and the hours covered by `stock_hour6_22_cnt` are read from the data by matching the count of
out-of-stock hours to that column; the run stops if no coding matches on at least 99% of days. These checks are
written to `results/structure.txt`.

## Windows

Days are indexed 0 to 96 in date order.

- **Fit:** days 0 to 75. Models are fitted here; LightGBM uses training rows from day 28 on, when all its lags exist.
- **Calibration:** days 76 to 89. One-day-ahead forecasts from the fitted models give the errors that the order rules
  use.
- **Test:** days 90 to 96, the dataset's own `eval` split. One-day-ahead forecasts and orders are scored here.

Forecasts are one day ahead: the order for day t uses data up to the end of day t - 1. Models are not refitted after
the fit window. Planned covariates of day t (discount, holiday flag, activity flag) are treated as known when the
order is placed; weather is not used.

## Recovered demand

For each series, an hourly profile is computed on the fully in-stock days of the fit window (days with
`stock_hour6_22_cnt` equal to 0): the share of the series' sales falling in each hour. A series with fewer than 7
such days, or no sales on them, uses the profile pooled over all series of the same `third_category_id`, and if
that is empty the profile pooled over all series.

For a day with out-of-stock hours, the observed share is the sum of the profile over the in-stock hours. If it is at
least 0.2, recovered demand is the larger of the day's sales and the in-stock-hour sales divided by the observed
share. If it is below 0.2, recovered demand is the larger of the day's sales and the series' mean sales on fully
in-stock days of the fit window. A day with no out-of-stock hours keeps its sales.

The same rule, with the same fit-window profiles, produces the demand used to train the "recovered" models, the
lagged inputs of those models on later days, and the demand against which every order is scored.

## Algorithms

Four algorithms, each producing a one-day-ahead point forecast of the series it is given:

- **naive7:** the value on the same weekday a week earlier.
- **ma4w:** the mean of the values on the same weekday one, two, three and four weeks earlier.
- **lgbm_l2:** one LightGBM model for all series, squared-error objective.
- **lgbm_tweedie:** the same, Tweedie objective with variance power 1.2.

LightGBM features: lags 1 to 7, 14, 21 and 28; means over the previous 7, 14 and 28 days; the standard deviation
over the previous 7 days; the ma4w value; the weekday of day t; discount, holiday and activity flags of day t; and
city, store, product, management group and the three category levels as categorical features. Settings: 500 trees,
learning rate 0.05, 63 leaves, at least 100 rows per leaf, 80% feature and row subsampling, seed 7, deterministic
mode with 4 threads. No tuning. Forecasts below zero are set to zero.

## Decision layers

Two training targets:

- **sales:** the model is given recorded daily sales.
- **recovered:** the model is given recovered demand.

Three order rules, for a cost ratio tau (the cost of a missing unit over the sum of the costs of a missing and of a
wasted unit):

- **point:** order the forecast.
- **pooled:** order the forecast plus the series' level times the tau-quantile of the calibration errors divided by
  their series' level, pooled over all series. The level is the series' mean target over the fit window.
- **series:** order the forecast plus the series' error scale times the tau-quantile of the calibration errors
  divided by their series' error scale, pooled over all series. The error scale is the root mean squared
  calibration error of that series and algorithm; a series with zero scale orders its forecast.

Errors are measured against the target the model was given: a team training on sales also judges its errors
against sales. Orders below zero are set to zero. That makes six decision layers: the **default layer** is sales and
point; the **full layer** is recovered and series.

## Cost

For each test series-day, with order q and demand d (recovered demand as defined above), the cost is
tau x max(d - q, 0) + (1 - tau) x max(q - d, 0). Unsold units do not carry over to the next day. Costs are summed
over all series and test days and divided by total test demand. Three cost ratios: tau = 0.5, 0.7 and 0.9. That
gives 4 algorithms x 6 layers x 3 ratios = 72 cells.

## Uncertainty

Stores are resampled with replacement, 2,000 times, seed 11; each resample recomputes every normalized cost from
store totals, and every quantity below from those costs. A difference counts as established if its 95% percentile
interval excludes zero in the stated direction.

## Hypotheses

Asymmetric ratios are those where a forecast of the mean is the wrong order by construction: tau = 0.7 and 0.9.

- **H1.** At tau = 0.7 and 0.9, ma4w with the full layer costs less than lgbm_l2 with the default layer and less
  than lgbm_tweedie with the default layer. Supported if all four differences are established.
- **H2.** At tau = 0.7 and 0.9, the decision layer explains a larger share of the variation in cost over the 24
  combinations of algorithm and layer than the algorithm does: in the two-way additive decomposition of the 4 x 6
  table of costs, the layer sum of squares exceeds the algorithm sum of squares. Supported if the difference of the
  two shares is established at both ratios. The shares depend on the two menus, which are fixed above and not
  changed after the results.
- **H3.** At tau = 0.7 and 0.9, training on recovered demand costs less than training on sales, for every algorithm
  and every order rule. Supported if all 24 differences are established.
- **H4.** Under the full layer at tau = 0.9, the order of the four algorithms by cost differs from their order by
  the weighted absolute percentage error (WAPE) of their point forecasts against demand. Decided on point
  estimates.

## Also reported

- The 72 costs, with WAPE, bias and the share of test series-days covered by the order (demand at or below it).
- For each tau, including 0.5, how the variation of cost over the 4 x 6 table of algorithms and layers splits into
  an algorithm part, a layer part and a remainder (sums of squares of the two-way additive decomposition).
- The cost at tau = 0.5, with no hypothesis: there a symmetric cost makes the point forecast a reasonable order.
- **Robustness, not deciding:** H1 to H3 recomputed on in-stock test days only (no out-of-stock hours between 6:00
  and 22:00), where recorded sales equal demand without any recovery.
- Recovery statistics: the share of fit and test days with out-of-stock hours, the share recovered with the
  fallback, the profile source of each series, and the ratio of recovered demand to sales.

## Changes after registration

- 13 November 2026, before the data were downloaded: the first workflow run stopped in the unit tests because the
  LightGBM scikit-learn wrapper needs scikit-learn, which is not installed. LightGBM is now called through its native
  interface with the same settings (500 rounds, and the parameters listed above under their native names).

## Descriptive analyses added after the results (13 November 2026)

Not part of the registered test, and labelled as such in the article; `details.py`, from the test orders that
`evaluate.py` saves with `--orders` (no change to how anything registered is computed).

- **Strength of the recovery.** The in-stock robustness check reversed part of the result, and in-stock days are
  selected on demand staying below the stock on the shelf. The registered costs are recomputed against
  d(s) = sales + s x (recovered demand - sales) for s = 0, 0.25, 0.5, 0.75 and 1: s = 0 treats recorded sales as
  demand, a lower bound on days with out-of-stock hours; s = 1 is the registered evaluation.
- **The point rule left out.** The H2 decomposition repeated on the 4 x 4 table of algorithms and quantile layers,
  since ordering a forecast of the mean is wrong by construction when tau is not 0.5.
- **Target and rule apart.** A three-way additive decomposition of the 4 x 2 x 3 table (algorithm, target, rule).
- **Orders by cost and by WAPE under every layer**, and the coverage of the quantile rules.

