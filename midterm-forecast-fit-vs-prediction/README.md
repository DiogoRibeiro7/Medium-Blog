# The Forecast Is Not the Fit: Rebuilding an 89% Midterm Election Model

A reproducible, historical-source-based statistical reconstruction and
independent predictive audit of Steve Rattner's 30 October 2026 U.S. House
midterm forecast. The companion supports the [Medium-ready article](article.md).

**As-of date:** 31 October 2026. The 2026 election has not occurred in this
dataset. The House generic-ballot input D+8 and 220 Republican seats defended
are a **snapshot of the original forecast**, not an ongoing live poll average.

## The finding, before any interpretation

| Quantity | Primary reconstruction |
| --- | ---: |
| Historical midterms included | 18 (1950–2022, excluding 1986) |
| OLS in-sample R² | 0.8934 |
| In-sample RMSE | 6.66 seats |
| Leave-one-election-out RMSE | 7.99 seats |
| Leave-one-election-out R² | 0.8465 |
| Expanding-origin forward RMSE (10 test elections) | 8.70 seats |
| Expanding-origin generic-ballot-only RMSE (same 10) | 20.09 seats |
| Expanding-origin seats-only RMSE (same 10) | 23.00 seats |
| Expanding-origin past-mean baseline RMSE | 23.81 seats |
| Conditional forecast: Republicans, 2026 | −32.32 seats |
| Generic-ballot-only conditional 2026 forecast | −31.56 seats |
| Alternative Gallup source: forward full / generic-only RMSE | 8.97 / 20.63 seats |
| Parametric Student-t 95% predictive interval | [−48.43, −16.22] seats |
| Pairs/residual bootstrap 95% sensitivity interval | [−41.75, −14.37] seats |
| Gallup retrospective alternative forecast | −31.91 seats |

The original chart says approximately **−33** and **R²=0.89**. These numerical
matches are close, but **not exact replication**: the author's design matrix,
poll-screen decisions and 1986 input are not published in that presentation.
We cannot claim to have identified the author's exact regression.

## Reproduce independently

Use Python **3.12** and the exact versions in `requirements.txt`. From this
directory:

```bash
python -m pip install -r requirements.txt
python analysis.py
python figures.py
python -m unittest discover -s tests -v
```

These commands regenerate committed data tables, nine original PNG figures,
and run **32 scientific, data-provenance and fresh-kernel notebook tests**.
The notebook is [model_walkthrough.ipynb](model_walkthrough.ipynb).
The `.github/workflows/midterm-forecast-fit-vs-prediction.yml` job is scoped to
this folder and can be dispatched manually. It regenerates into a separate
temporary folder and checks the canonical numerical summary against that
reproduction. No API tokens, proprietary datasets or election-night data are
needed.

For a direct test of whether the second predictor adds value, the study also
fits two simpler OLS models from scratch at each strictly earlier training
origin: intercept + generic ballot and intercept + defended seats. All four
rules (including the historical mean) predict the **same ten elections** with
the **same information cutoff**. The row-level table is
[`results/forward_model_comparison.csv`](results/forward_model_comparison.csv).
The full model beats generic-ballot-only absolute error on nine of ten target
years, although its 2006 error is larger. The comparisons are exploratory and
not an evaluation against district-level or probabilistic election forecasts.
The same ranking and nine-of-ten count hold when replacing Gallup's two
disputed historical retrospective values. Exact alternative-source forward
errors are committed in [`results/source_sensitivity.csv`](results/source_sensitivity.csv).

To regenerate without changing the working tree:

```bash
python analysis.py --output /tmp/midterm-reproduction
python figures.py --output /tmp/midterm-reproduction
```

The model uses only a few dependencies: NumPy for transparent OLS and linear
algebra, SciPy for Student-t critical values, Matplotlib for original figures,
and the Jupyter kernel/client for notebook reproducibility. No scikit-learn or
polling API is required.

## Source data and methodological decisions

- [`data/historical_midterms.csv`](data/historical_midterms.csv) contains the
  **18 real, sourced observations**: year, president's party, initial House
  seats held, seat change, final pre-election generic-ballot spread, source key.
- [`data/PROVENANCE.md`](data/PROVENANCE.md) provides every primary publisher
  and the data dictionary, poll-universe qualification, and transcription
  rules. The American Presidency Project provides outcomes and seats, Gallup
  provides historical generic-ballot estimates through 2002, and RCP provides
  the final polling averages for 2006–2022.
- 1986 has no Gallup generic ballot observation and is **excluded**, not
  interpolated. Rattner's chart plots 1986, but its polling series is not
  specified; our reconstruction has 18 observations, not an invented 19th.
- [`data/gallup_alternative.csv`](data/gallup_alternative.csv) contains
  alternative retrospective Gallup figures for 1962 and 1998. We report the
  resulting sensitivity instead of silently treating inconsistent Gallup
  publications as identical.
- [`data/2026_scenario.csv`](data/2026_scenario.csv) stores only ex-ante
  predictors, **never a 2026 election result**.

### Model

For election \(t\), with the generic ballot oriented toward the president's
party, fit

\[
\Delta S_t=\beta_0+\beta_1 G_t^{(P)}+\beta_2S_t+\varepsilon_t.
\]

The fitted coefficients under the primary source series are approximately

\[
\widehat{\Delta S_t}=106.998+2.708G_t^{(P)}-0.535S_t.
\]

The 2026 input \(G^{(P)}=-8\), \(S=220\) produces approximately
\(-32.32\) House seats for Republicans. All slopes are **predictive
associations, not policy effects**; in particular, the defended-seats
coefficient is partly a mechanical capacity/context measure.

## Nine original evidence figures

| # | PNG | Exact claim or diagnostic |
| --- | --- | --- |
| 1 | [historical fit](figures/fig1_historical_fit.png) | Observed vs fitted, with R² and election-year labels |
| 2 | [historical residuals](figures/fig2_residuals_over_time.png) | Within-sample errors by year, including 2006 |
| 3 | [leave-one-out](figures/fig3_leave_one_out.png) | Held-out predictions, distinguish from fitted predictions |
| 4 | [forward testing](figures/fig4_forward_validation.png) | Model vs actual vs historical mean using earlier years only |
| 5 | [forward model comparison](figures/fig5_error_comparison.png) | Fair RMSE comparison: full model vs ballot-only, seats-only and earlier-election mean on the same 10 elections |
| 6 | [2026 intervals](figures/fig6_2026_prediction_intervals.png) | Conditional Student-t and pairs/residual-bootstrap uncertainty |
| 7 | [polling sensitivity](figures/fig7_polling_sensitivity.png) | Conditional response to D+2…D+14 and 210/220/230 seats |
| 8 | [influence](figures/fig8_influence_diagnostics.png) | Leverage, standardized residual and Cook's distance |
| 9 | [scenario heatmap](figures/fig9_scenario_heatmap.png) | Two-input sensitivity surface; not a probability distribution |

Figures 1–9 are generated **exclusively** from committed data and source code,
never digitized from the Rattner screenshot. Every generated table under
[`results/`](results/) is either a result, diagnostic, forecast or scenario—not
newly observed historical data. `results/summary.json` records the exact
numbers cited in the article, including reproducibility seed and draw count.

## Statistical and interpretation limits

1. **Small sample:** 18 elections and three fitted regression coefficients.
   Leave-one-out testing and expanding-origin backtests are useful diagnostics,
   not an independent large prospective validation set.
2. **Conditioning on final polling:** In historical folds the final
   generic-ballot average is known. This is a short-horizon conditional
   election forecast, not an early-cycle, pure structural forecast.
3. **Changing polls:** Gallup retrospective values and RCP averages have
   different survey/aggregation conventions. The series is transparently
   harmonized by **sign**, not proven commensurate by sampling design.
4. **Prediction intervals:** Student-t assumptions include stable coefficients,
   approximately normal independent errors and homogeneous residual variance.
   The pairs/residual bootstrap assumes exchangeable historical observations
   and is not a calibrated cure for structural shifts.
5. **Outcome:** The forecast is for the *president's party's seat change*, not
   a vote share or the probability of controlling the House. A 95% model-based
   interval is not a 95% guarantee.
6. **Politics and causation:** The model estimates no effects of police action,
   public sentiment, policies or campaigns. Its popularity-poll covariate is a
   predictive summary. Correlation is entirely sufficient for some forecasting
   tasks, but predictive stability still needs testing.

## Publication contract

The draft has more than 8,000 words, original derivations, nine data-grounded
figures with captions and alt text, links to all primary data and methodology,
and a balanced conclusion distinguishing fit from prediction. No claims of
post-election success or failure are made before 3 November 2026.

This repository change is meant for **a reviewable PR only**. It does not
merge itself or publish on Medium.
