# Pre-treatment placebo diagnostics for the tax-cut study

## Purpose and identification boundary

The earlier exploratory matching exercise found a positive top-income-share
contrast after major tax cuts. That contrast alone cannot distinguish a
causal treatment effect from a persistent difference in trends between
reforming and donor countries.

This module checks whether **the same treated countries and donors already
had different income-share trajectories before the recorded reform**.
It implements an independent matched event-time diagnostic inspired by
the [PanelMatch placebo documentation](https://search.r-project.org/CRAN/refmans/PanelMatch/html/placebo_test.html),
not a reproduction of that package's propensity-score or Mahalanobis
refinement and not a replication of Hope and Limberg's main estimator.

The project follows the caution in Jonathan Roth's
[*Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends*](https://doi.org/10.1257/aeri.20210236):
pre-trend checks can have low power, and passing them cannot establish
the required counterfactual parallel-trends assumption.

## Estimand and matching rule

For each event in country `i` at year `t`, define the average donor outcome
`Y_D(s)` from countries in the fixed donor set `D(i,t)`. For every relative
time `k` from `-L` to `H` except `-1`, calculate

\[
  \widehat{\delta}_{i,t}(k) =
  \bigl(Y_{i,t+k}-Y_{i,t-1}\bigr) -
  \bigl(\bar{Y}_{D(i,t),t+k}-\bar{Y}_{D(i,t),t-1}\bigr).
\]

The pre-reform values from `k = -L` to `k = -2` are **placebo lead contrasts**.
Their mean across matched reform events is zero if the treated-minus-donor
gap was constant in the observed preperiod. The implementation additionally
reports the OLS slope of that observed gap over all pre-reform years.

To prevent changes in the sample being mistaken for changes over event time,
all comparisons use the **same event-level donor set** throughout:

1. The treated country must have a recorded cut in `t`, no cut in `t-1`, and
   complete outcomes at every year from `t-L` through `t+H`.
2. Every donor must match the treated country's **full prior `L`-year
   treatment history** and have complete outcomes throughout that window.
3. Donors must have no major tax cut in any year `t` through `t+H`.
   The treated country may undergo another cut after `t`; this is an
   explicit remaining source of treatment contamination.
4. With no pre-treatment economic covariates, every eligible donor is equally
   weighted. If numeric covariates are supplied, the existing pre-treatment
   Mahalanobis ranking restricts the number of donors.

The balanced cohort is necessarily different from the earlier estimator,
which allowed a different donor set at each post-reform horizon. The
**post-treatment values should therefore not be described as a literal
reproduction of its frozen +0.547 or +0.795 percentage-point contrasts**.

## Outputs and interpretation

`python -m causal_tax_cuts.event_study` creates:

- `event_time_contrasts.csv`: all event-level contrasts, stable donor
  identifiers, placebo time and the event's pretrend-slope gap.
- `event_time_summary.csv`: mean, median, event count and treated-country
  count at every event time, excluding baseline year `t-1`.
- `event_study_audit.json`: eligible/matched counts, donor ranges, the
  average pretrend-slope gap, exact source path and an explicit no-inference
  status.

The GitHub Action uploads these outputs beside the original matched effects
as short-lived artifacts after downloading the *actual* WID and policy data.
No synthetic diagnostics are presented as evidence about real neoliberal
policy. The tested method has no reported confidence intervals or p-values.

The first real-data output, including pre-event values that are **not zero**,
is preserved with source-vintage checks in
[`results/PRE_TREND_RESULTS_2026-11-13.md`](results/PRE_TREND_RESULTS_2026-11-13.md).

Possible conclusions are deliberately modest:

- Nonzero pre-event contrasts weaken confidence in untreated-trend
  comparability and should be investigated before publishing a causal claim.
- Small measured pre-event contrasts **do not prove** parallel trends,
  absence of anticipation or lack of time-varying confounding.
- Selection on future donor tax policy, revised WID values, missingness,
  endogenous reform timing and staggered overlapping treatments remain
  potential biases. Country-block uncertainty and balance on additional
  economic covariates are distinct next research steps.
