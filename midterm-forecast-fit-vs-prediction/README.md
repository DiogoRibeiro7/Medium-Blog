# The Forecast Is Not the Fit: Rebuilding an 89% Midterm Election Model

A reproducible, historical-source-based statistical reconstruction and
independent predictive audit of Steve Rattner's 30 October 2026 U.S. House
midterm forecast. The companion supports the [Medium-ready article](article.md).

**As-of date:** 31 October 2026. The 2026 election has not occurred in this
dataset. The House generic-ballot input D+8 and 220 Republican seats defended
are a **snapshot of the original forecast**, not an ongoing live poll average.
The original inputs and predictions were separately **registered on 1 November
2026**, before election day, in a [content-addressed forecast lock](PRE_ELECTION_REGISTRATION.md).

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
| Fixed Student-t (ν=4) forward RMSE / 2026 conditional forecast | 8.48 / −33.76 seats |
| Fixed ridge (α=2) forward RMSE / 2026 conditional forecast | 12.18 / −31.06 seats |
| Fixed 24-year recency WLS forward RMSE / 2026 conditional forecast | 9.13 / −30.98 seats |
| Student-t minus OLS forward RMSE, primary / alternative Gallup | −0.214 / +0.048 seats |
| Student-t minus OLS paired-bootstrap percentile range, primary | [−0.98, +0.33] seats (descriptive, **not** a CI) |

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
python model_extensions.py
python comparison_uncertainty.py
python forecast_lock.py
python figures.py
python -m unittest discover -s tests -v
```

These commands regenerate committed data tables, thirteen original PNG figures,
and run **65 scientific, data-provenance, post-election guard, and fresh-kernel
notebook tests**.
The notebook is [model_walkthrough.ipynb](model_walkthrough.ipynb).
The `.github/workflows/midterm-forecast-fit-vs-prediction.yml` job is scoped to
this folder and can be dispatched manually. It regenerates into a separate
temporary folder and compares canonical numerical metrics and both benchmark
tables using tight floating-point tolerances (1e-9 absolute and relative),
while requiring exact metadata, schemas, row counts and figure counts. The
additional Student-t optimizer snapshots receive a 1e-5 numeric tolerance,
as do its paired-error diagnostics, with schemas, evaluation dates and fixed
seed/draw specifications checked exactly. The pre-election registration
manifest must match **exact source SHA-256 fingerprints**. This
allows harmless platform-level BLAS differences without accepting substantive
numerical drift. No API tokens, proprietary datasets or election-night data
are needed.

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

The second comparison holds the predictors and ten forward origins fixed while
changing the estimator: heavy-tailed Student-t likelihood (degrees of freedom
4), ridge (alpha 2 on fold-standardized slopes, unpenalized intercept), and
recency-weighted OLS (24-year age-weight half-life). Their settings are **fixed
without selecting the winner from historical backtests**. Student-t has a
slight primary-source RMSE advantage of about 0.21 seats over OLS; that
advantage reverses under the alternative Gallup figures. Neither ridge nor
recency weighting improves primary-source RMSE. The original OLS 2026 point
forecast remains the frozen reference, not a revised prediction chosen after
testing. All values and assumptions are recorded in
[`results/alternative_models_summary.json`](results/alternative_models_summary.json),
with ten common forward predictions in
[`results/alternative_model_forward.csv`](results/alternative_model_forward.csv),
the source sensitivity in
[`results/alternative_model_forward_gallup.csv`](results/alternative_model_forward_gallup.csv),
and retrospective fixed-2026-input training-path calculations in
[`results/alternative_model_path.csv`](results/alternative_model_path.csv).

The third analysis audits how stable the **model ranking** is, without changing
the forecasts. It computes candidate-minus-OLS squared loss for every common
test election, omits each test year simultaneously from both methods, and
resamples the ten **paired** forecast-error rows 20,000 times (seed 20261031).
Under the primary Gallup transcription, Student-t's apparent 0.214-seat RMSE
advantage reverses if the 2010 test year is omitted. The primary paired
bootstrap 2.5–97.5% RMSE-difference percentile range is [−0.98, +0.33]; the
alternative-Gallup range is [−0.97, +0.51]. These are **descriptive resampling
ranges, not calibrated confidence intervals**. Resampling does not create
independent midterm elections, and chronological errors can share training
history. The outputs are the [per-year squared-loss differences](results/paired_loss_differences.csv),
[paired resampling summaries](results/paired_model_comparisons.csv),
[one-year omission rows](results/paired_leave_one_year_out.csv), and
[run specification](results/paired_uncertainty_summary.json).

### Pre-election registration and subsequent evaluation

The [registration protocol](PRE_ELECTION_REGISTRATION.md) and
[`results/forecast_lock.json`](results/forecast_lock.json) preserve the exact
October OLS forecast **−32.322888**, its 95% conditional interval
**[−48.428387, −16.217390]**, all alternative point estimates, the frozen
input scenario, and twelve source/evidence SHA-256 hashes. The information
cut-off (**31 October**) is explicitly distinguished from the registration
date (**1 November**). The source commit is independently inspectable in Git;
this is not third-party notarization, nor a claim that the historical
specification was selected without hindsight.

The [post-election scoring program](score_after_election.py) is prepared, but
**no 2026 outcome is recorded or scored yet**. It refuses dates before the
3 November 2026 election, requires an explicitly confirmed final result from
the House Historian or American Presidency Project, and writes a new report
without modifying the locked forecast. Details and a future CLI example are in
the registration protocol. Synthetic unit-test results are not election data.

To regenerate without changing the working tree:

```bash
python analysis.py --output /tmp/midterm-reproduction
python model_extensions.py --output /tmp/midterm-reproduction
python comparison_uncertainty.py --output /tmp/midterm-reproduction
python forecast_lock.py --output /tmp/midterm-reproduction
python figures.py --output /tmp/midterm-reproduction
```

The model uses only a few dependencies: NumPy for transparent OLS and linear
algebra, SciPy for Student-t critical values and robust maximum likelihood,
Matplotlib for original figures,
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

## Thirteen original evidence figures

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
| 10 | [estimator RMSE comparison](figures/fig10_candidate_model_rmse.png) | OLS, Student-t, ridge, recency weighting and ballot-only on common targets under both Gallup source readings |
| 11 | [training-endpoint stability](figures/fig11_training_origin_stability.png) | Four historical coefficient paths for the fixed 2026 scenario; explicitly not contemporary historical forecasts |
| 12 | [Student-t paired losses](figures/fig12_student_t_paired_losses.png) | Held-out squared-error differences relative to OLS, primary and alternative Gallup sources |
| 13 | [paired RMSE uncertainty](figures/fig13_paired_rmse_uncertainty.png) | Paired-year bootstrap percentile ranges for candidate-minus-OLS RMSE; **not confidence intervals** |

Figures 1–13 are generated **exclusively** from committed data and source code,
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
7. **Alternative-model uncertainty:** Changing the estimator does not produce
   calibrated uncertainty for the new Student-t, ridge or recency methods.
   Comparing their point forecasts is not model averaging or an interval.
   Small differences in forward RMSE on ten elections do not select a winner.
8. **Paired-bootstrap limitation:** Resampling ten shared error pairs avoids
   treating candidate and reference losses as independent, but it does not
   establish exchangeability or independence across electoral cycles. Its
   quantile bands and resampling frequencies are diagnostic descriptions,
   not confidence intervals or probabilities of future superiority.

## Publication contract

The draft has more than 8,000 words, original derivations, thirteen data-grounded
figures with captions and alt text, links to all primary data and methodology,
and a balanced conclusion distinguishing fit from prediction. No claims of
post-election success or failure are made before 3 November 2026.

This repository change is meant for **a reviewable PR only**. It does not
merge itself or publish on Medium.
