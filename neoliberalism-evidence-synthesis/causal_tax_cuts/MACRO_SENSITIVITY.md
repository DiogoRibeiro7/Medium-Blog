# Sensitivity of macro donor matching to pre-reform history and donor count

This is a **design and balance sensitivity** study, not a treatment-effect
study. The immediately preceding [macro matching audit](MACRO_MATCHING.md)
found that selecting donors on three World Bank variables lowered measured
macroeconomic gaps but slightly worsened pre-reform income-share slopes.
That trade-off makes a robustness grid necessary before any attempt to claim
a macro-adjusted effect of tax cuts.

## Declared grid and unchanged constraints

| Dimension | Values | Original reference |
| --- | --- | --- |
| History length `L` (years) | **3, 5, 7** | 5 |
| Maximum nearest donors `K` | **3, 5, 7** | 5 |
| Post-event outcome horizon | 5 years, fixed | 5 |
| Macro sources | GDP per capita growth, CPI inflation, trade openness | Same |

The resulting **nine** rules are evaluated systematically, in sorted order.
No point estimate is chosen for its appealing statistical result. Each grid
cell calls the same tested `audit_macro_matching` implementation; it varies
only `L` and `K`. Treatment-history matching, no event at `t-1`, balanced WID
income-share outcomes from `t-L` through `t+5`, complete WDI data only
from `t-L` through `t-1`, Mahalanobis ranking and future donor-treatment
censoring all retain their original explicit definitions.

Changing `L` necessarily changes *eligibility* as well as selection profiles.
For example, seven pre-reform observed outcome years and WDI measurements
are a stricter requirement than three. The grid therefore distinguishes:

- Counts for **every rule's supported event cohort** and donor-event pairs.
- The intersection of event identifiers (`country`, reform year) supported by
  **all nine rules**. This common cohort prevents the apparent balance change
  from being driven solely by a different mix of reforms.
- The Jaccard overlap of selected **donor sets** with the original `L=5,K=5`
  rule, on those common events. Changing `K` changes donor-set size by design,
  so overlap is a descriptive stability measure, not a success criterion.

## Comparable evaluation: a fixed pre-three-year window

The older audit uses each rule's entire `L`-year selection profile to measure
balance. Those gaps cannot be compared naively: a three-year mean and a
seven-year mean describe different intervals, and their standard deviations
also differ. For this grid, every selected donor set is evaluated using
**only `t-3`, `t-2` and `t-1`**:

1. For the treated reform and selected donors, compute three-year pre-reform
   means for each WDI series. The signed mean gap is divided by the
   **population standard deviation of all WDI-observed study-country means
   over that same fixed window and reform year**, not by a denominator that
   changes with the donor cap. A zero-variance denominator is reported as
   undefined, never as a zero gap.
2. The income-share outcome's pre-reform three-year OLS slope is the
   difference between observed values at `t-1` and `t-3` divided by two,
   in percentage points per year. Compare the treated slope with the
   average selected-donor slope. Outcomes are never used to rank donors.
3. Aggregate the **absolute** gaps over the common event intersection.
   The per-event CSV also retains signed gaps and all donor identities.

This standardisation is a descriptive diagnostic; the fixed three-year
controls **overlap the original matching features**, so this is not a
genuinely independent holdout validation of covariate balance. The outcome
slope is not a matching feature, but neither pre-trend balance nor an apparently
stable donor pool demonstrates exchangeability or a causal interpretation.

The fixed-three-year mean absolute income-share slope gap for `L=5,K=5` is
**not expected** to equal the previous five-year slope audit: the outcome
window is shorter and the event sample is the intersection, not all reforms
supported by the original reference alone. Never compare those two numbers
as if they were the same diagnostic.

## Run and interpret

From `neoliberalism-evidence-synthesis/`, after generating the audited WDI
join using [the source importer](MACRO_CONTROLS.md):

```bash
python -m causal_tax_cuts.macro_sensitivity \
  --panel data/wdi-source/outcome_panel_with_macro.csv \
  --provenance data/wdi-source/world_bank_provenance.json \
  --frozen-reference causal_tax_cuts/results/macro_balance_2026-11-14.json \
  --lags 3 5 7 --donor-caps 3 5 7 --max-horizon 5 \
  --output-dir data/tax-study/macro-sensitivity
```

Outputs:

- `macro_sensitivity_grid.csv`: nine grid cells, rule-specific attrition,
  selected donor counts, reference donor overlap and common-cohort gaps.
- `macro_sensitivity_events.csv`: all original reform IDs under all nine
  rules, each eligibility status and selected donor set, and fixed-window
  pre-reform diagnostics where matched.
- `macro_sensitivity_audit.json`: the exact grid, common reform IDs, metric
  definitions and caveats, joined-panel SHA-256 and optional original WDI
  ZIP hashes.

If the supplied panel's SHA-256 matches the frozen five/five audit, the tool
re-runs that rule and verifies its event count and original slope diagnostics
before publishing. If the live WDI/WID vintage has changed, it **marks the
source as incomparable** instead of silently asserting the old numbers.

The source-probe workflow runs this grid after the macro balance audit and
uploads the outputs as a seven-day `pre-reform-macro-sensitivity` artifact.

This is not an exercise in selecting the rule with the lowest visible gap.
The next research gate is to address *actual overlap/positivity, historical
time-varying confounding and donor-exposure censoring*, then design credible
clustered inference. The empirical numbers are frozen in
[`results/MACRO_SENSITIVITY_RESULTS_2026-11-14.md`](results/MACRO_SENSITIVITY_RESULTS_2026-11-14.md).
