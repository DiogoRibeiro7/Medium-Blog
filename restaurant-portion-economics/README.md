# Restaurant portion economics: food waste, demand, and contribution margins

A reproducible R research case study for `Medium-Blog`, focused on a deliberately
narrow economic question:

> Under which conditions can a restaurant reduce default portion sizes,
> keep its menu prices unchanged, and improve contribution margins without
> creating unacceptable customer attrition?

This is a *testable business hypothesis*, not a recommendation that every
restaurant should serve less food. Adequate nutrition, value for money, fairness
and customer satisfaction are constraints, not externalities to ignore.

## Evidence and economic questions

1. **Observed food waste.** Audit real food-waste records from hotels and
   restaurants in the URBAN-WASTE dataset (Zenodo, 2019).
2. **Plate vs kitchen waste.** Separately describe measured daily food waste
   per diner in a 155-day Finnish institutional catering dataset (2024–25).
3. **Economic counterfactuals.** Model the *conditional* effect of smaller
   portions on procurement expense, demand and contribution margin.
4. **Nutritional design.** Later connect ingredient-level composition to
   authoritative nutrition data. Do not infer calories from total waste weight.

These sources do **not** contain a representative sample of restaurant
ingredient costs, menu prices, meals sold, repeated customers or a randomized
portion-size/price experiment. They cannot identify the profitability effect
of reducing portions at unchanged prices. Scenario outputs are hypothetical.

## Quick start

Requires R 4.3+ (R scripts use base R; `readxl` is needed for the
Finland Excel dataset).

```sh
cd restaurant-portion-economics
Rscript tests/test-economics.R
Rscript scripts/run-analysis.R                    # offline: economic model only
Rscript scripts/download-data.R                   # public source files
Rscript -e 'install.packages("readxl")'           # once, for Excel input
Rscript scripts/run-analysis.R                    # observed-data audits + model
```

The fetcher checks the publisher-listed MD5 for each file, does not
commit upstream data, and fails loudly on a changed file or unsuccessful
download. Files stay under `data/raw/` (ignored by Git). The analysis
writes derived tables to `outputs/` (also ignored by Git). This is a
project-local environment: no new repository-wide dependencies.

If Zenodo rate-limits downloads, use the exact record links in
[DATA_SOURCES.md](DATA_SOURCES.md) to save the named files to `data/raw/`,
then rerun the analysis.

## Economic model

Let `p` be menu price, `c` ingredient cost per meal, `v` other variable
cost, `N` monthly meals, `s` affected ingredient-cost share, `r`
portion reduction, `a` actual procurement pass-through (fraction of
the theoretical ingredient reduction that really lowers purchasing),
`d` proportional demand change, and `I` incremental monthly
implementation cost.

```text
ingredient saving per meal:  S = c * s * r * a
baseline contribution:      C0 = N * (p - c - v)
new contribution:           C1 = N * (1 + d) * (p - c - v + S) - I
change:                     ΔC = C1 - C0
```

The break-even demand change, when the new per-meal contribution is
positive, is:

```text
d* = (I/N - S) / (p - c - v + S)
```

If `d*` is negative, its absolute value is the maximum assumed
customer-volume loss compatible with nonnegative incremental contribution.
It is **not** an estimated elasticity or a forecast.

The example values in `R/economics.R` are chosen to illustrate the
mechanism, not to represent observed Portuguese restaurant accounts.
Reduced disposal or storage expenses are excluded until properly measured;
fixed costs cancel from the comparison.

## Outputs

- `outputs/scenario.csv`: the stated hypothetical baseline.
- `outputs/sensitivity.csv`: portion-size reduction vs volume-loss grid.
- `outputs/threshold.csv`: demand-change break-even point.
- `outputs/urbanwaste_schema.csv`: columns, types, missing values from a
  *real* hotel/restaurant waste-tracking dataset.
- `outputs/urbanwaste_numeric_audit.csv`: unlabelled candidate numeric
  variables for review; **not** interpreted as kilograms or euros.
- `outputs/finland_daily.csv`, `outputs/finland_summary.csv`: comparable
  institutional-food-service daily waste descriptors when the relevant
  spreadsheet columns can be validated.

The first run can be made entirely offline; the measured-data outputs
require downloading the original sources. The institutional dataset is not
silently merged with hotel/restaurant observations.

## Methodological guardrails

- Do not equate food left on plates with procurement savings. Customers may
  eat less, demand may fall, and some ingredients or preparation steps may
  be fixed.
- Separate ingredient costs from labour, rent and disposal; don't treat
  all costs as proportional to portion size.
- A before/after institutional-catering observation does not identify a
  causal effect (menus, seasons, attendance and interventions may change).
- Evaluate sensitivity to the *affected* cost share, purchase pass-through
  and customer response before presenting a business case.
- Respect source rights. The optional WASTEWISE EU database is explicitly
  non-commercial/share-alike: it is a research reference, not silently
  repurposed as a commercial data asset.

See [DATA_SOURCES.md](DATA_SOURCES.md), [METHODS.md](METHODS.md) and
[ROADMAP.md](ROADMAP.md).

Status: initial reproducible research scaffold; descriptive findings
should not be reported before actual source-file validation and execution.
