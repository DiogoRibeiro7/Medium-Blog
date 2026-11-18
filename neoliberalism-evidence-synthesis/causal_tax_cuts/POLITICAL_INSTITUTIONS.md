# Pre-reform political and institutional controls: DPI2023

This is a **source and coverage audit**, not political adjustment, a new
matching specification, causal identification, or an effect estimate. It
extends the existing tax-cut evidence pipeline with three observed annual
variables and preserves the original WID/WDI reform and donor eligibility.

## Source, coding and permitted use

The [Inter-American Development Bank's Database of Political Institutions
(DPI2023)](https://data.iadb.org/dataset/the-database-of-political-institutions-dpi-2023)
is published under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
DOI [10.60966/8tx1xs4i](https://doi.org/10.60966/8tx1xs4i). Its
[downloadable country-year CSV](https://data.iadb.org/files/download/30f45628-8608-4c0c-ad55-bc4c870dd62d)
contains 1975–2023 observations. All classification decisions below derive
from the [official codebook](https://data.iadb.org/files/download/7d23d94b-dfec-4c84-b50e-174320432521).

| Source column | Audit name | Interpretation and handling |
| --- | --- | --- |
| `execrlc` | `executive_orientation` | Executive economic-party category. `Right`, `Center`, and `Left` remain **nominal**, never converted into a quantitative ideological score. Code `0` = unknown, `-999` / `NA` = not applicable, blank = missing. |
| `checks` | `checks` | Published index of political checks/veto players. Store observed numeric values, but do not infer causal institutional quality from the index alone. |
| `eiec` | `electoral_competitiveness` | Published executive electoral competitiveness code. Store as an observed numeric code; fractional values such as `6.5` are preserved. |

The original `countryname` and `year` fields are validated; `year` must
be a January 1 ISO-date snapshot. The 18-country mapping is explicit,
including `FRG/Germany → DEU`, `UK → GBR`, `USA → USA`. The source's `GDR`
is a separate country and **must never be merged** into the West Germany/
unified Germany series. Switzerland's collegial executive has no single
classified economic-party orientation; its `-999` values are not imputed.

The program downloads the official CSV at execution time, records the
**source bytes and SHA-256**, and emits only the 18-country derived panel,
the eligibility diagnostics and a source/provenance JSON. The full raw IDB
file is not checked in. Treat CC BY attribution as applying to derived
CSV artifacts as well as the raw data; the repository's code licence does
not relicense IDB data.

## Time ordering and inherited cohort

For a tax-cut reform in year \(t\), audit the **five calendar years before**
the reform, \(t-5,\ldots,t-1\), for each of the three DPI variables.
Never read year \(t\) or later in deciding political-covariate coverage.
The snapshot format also avoids claiming a precise within-year election or
tax-law ordering which these annual data cannot establish.

All treatment events come from Hope–Limberg's original binary `taxcut_2SD`
panel; WID outcomes and WDI macro histories are unchanged. The prior
`donor_support.py` decision is authoritative: pre-censor donor–event pairs,
future tax-cut exclusion by horizon, complete `t-5..t+5` WID outcome
window and five-year WDI eligibility remain **identical**. Political
variables are **not** new eligibility filters or new distance weights.

The output counts history availability separately for each treatment
episode and each already-eligible donor. A history is complete for a
variable if all five pre-treatment annual entries are observed. Joint
prehistory is complete only when all **three** variables are complete.
The source starts in 1975, making full prehistories structurally impossible
for 1965–1979 reforms. Output rows for the 180 pre-1975 sample country-years
have `orientation_status=out_of_source_period` and no fabricated values.

## Reproduce and inspect

From `neoliberalism-evidence-synthesis/`:

```bash
python -m causal_tax_cuts.political_institutions \
  --panel data/wdi-source/outcome_panel_with_macro.csv \
  --frozen-reference causal_tax_cuts/results/donor_support_2026-11-15.json \
  --output-dir data/tax-study/political-institutions
```

For offline inspection, pass `--source-csv` pointing to the unmodified
official download. The command produces four files:

- `dpi_political_controls.csv`: 918 rows (18 × 51), explicit missingness,
  annual national source values and no interpolation.
- `political_reform_coverage.csv`: five-year availability and donor
  completeness for each previously eligible reform.
- `political_donor_coverage.csv`: one row per previously eligible donor–event
  pair, including its unchanged future-censoring horizon.
- `political_coverage_audit.json`: counts, official source and joined-panel
  SHA-256 hashes, coding/missingness rules and reconciliation checks.

If the joined panel hash matches the frozen earlier `donor_support` audit,
the number of baseline-eligible reforms, initial donors and year-five donors
**must** reproduce that reference exactly or this command fails. If the
live WID/WDI vintage has changed, the result explicitly records
`frozen_reference_status=source_vintage_changed`; it does not falsely
assert equality to that archived sample. Output JSON is deterministic for
the same source bytes, joined panel and algorithm version.

The separate source-probe GitHub Action regenerates these outputs using
official live downloads and uploads a seven-day artifact called
`tax-cut-political-institutions`. Synthetic tests cover field validation,
unknown versus inapplicable orientation, exact country mappings,
future-policy censoring and no post-treatment political-data leakage.

## Interpretation limits and next research gate

The initial source-verified counts are in
[`results/POLITICAL_INSTITUTIONS_RESULTS_2026-11-18.md`](results/POLITICAL_INSTITUTIONS_RESULTS_2026-11-18.md).
Coverage does not establish that political institutions were balanced
between treated and donor countries. Executive-party orientation is not
implemented tax policy, legislative voting or a time-varying measure of
every institutional confounder. National annual party classifications
cannot capture an individual law's expectations or exact timing.

The previous convex-hull audit found severe support limitations even in
three WDI macro means; adding observed political histories cannot solve
that by itself. Before claiming causality, a new preregistered design
would need a defensible treatment strategy, sensitivity to political
confounding and country/serial dependence, and support evaluated on its
**full** covariate set. No further effect is estimated in this audit.
