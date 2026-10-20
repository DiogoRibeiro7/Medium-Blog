# The core robustness grid without the participation control (protocol)

Registered on 2026-10-19, before the re-classification below was computed. The estimates
it uses are already published in `docs/review/grid.csv`. The re-classification is
mechanical, so this protocol fixes the rule, and how the manuscript uses the result,
before the result is known.

## 1. Question

The classical review (docs/CLASSICAL_RESULTS_REVIEW.md) classifies each country from a
core grid of 24 estimates of the GDP coefficient sum:

- the full, pandemic-excluded, pre-2020 and quality-screened samples;
- 1, 2 and 4 lags of GDP growth;
- with and without the change in the participation rate as a control.

Half of these control for participation. That variable is itself a response to output,
and it is mechanically linked to the outcome, because the labour force is the denominator
of the unemployment rate. Those specifications estimate a different, partly mediated
quantity. An external review (2026-10-19) asked that the participation control be kept
out of the core grid.

## 2. Rule

- **Grid.** The 12 estimates without the participation control: four samples × three
  lag orders, read from the committed `docs/review/grid.csv`.
- **Verdict.** The review's rule (`review.verdict_for`), unchanged:
  - *robust negative*: every point estimate below zero and every 95% interval below zero;
  - *negative, significance specification-dependent*: every point estimate below zero,
    but some interval reaches zero;
  - *sign-unstable*: at least one point estimate at or above zero.
- **Second verdict.** As in the review, the same rule without the full sample (9
  estimates).
- **Check.** The implementation must reproduce the committed 24-estimate verdicts of
  `docs/review/robustness.csv` for every country before it computes the new ones.

## 3. Use in the manuscript (fixed in advance)

- The participation-free verdicts become the primary classical robustness verdicts:
  - the count of robust negative countries quoted in the text;
  - the verdict column of the country table;
  - the verdicts of the focus countries.
- The published 24-estimate count is kept as the classical review's published figure. It
  is reported next to the new count, with the countries whose verdict changes.
- Nothing else changes. The headline estimates and intervals do not use the participation
  control, and neither do the Bayesian, predictive, identification and case-study
  analyses.

## 4. Outputs

`eu-okun core-grid` writes `docs/core-grid/verdicts.csv`, `docs/core-grid/summary.json`
and `docs/CORE_GRID.md`. A test regenerates them and checks the reproduction of the
published verdicts.

## Change log
