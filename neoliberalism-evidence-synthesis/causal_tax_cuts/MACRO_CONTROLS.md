# Official macroeconomic controls: ingestion and coverage audit

This stage obtains the observed macroeconomic series required to develop a
covariate-adjusted matching analysis. **It does not recalculate, reweight or
reinterpret the existing exploratory tax-cut estimates.** The previous
outcome/event panel is preserved row-for-row, and missing macro observations
are left blank. This isolates a data-quality question before selecting a
statistical specification.

## Published primary data and variable definitions

The [World Bank World Development Indicators API](https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation)
provides public annual observations. The importer requests the official `source=2`
database through its [documented API query format](https://datahelpdesk.worldbank.org/knowledgebase/articles/898581-api-basic-call-structures)
for the paper's exact **18 countries, 1965–2015**. No API key is needed.

| Output column | World Bank series | Unit | Intended use |
| --- | --- | --- | --- |
| `gdp_pc_growth_pct` | [`NY.GDP.PCAP.KD.ZG`](https://data.worldbank.org/indicator/NY.GDP.PCAP.KD.ZG) | Annual growth, % | Candidate pre-reform matching control |
| `inflation_cpi_pct` | [`FP.CPI.TOTL.ZG`](https://data.worldbank.org/indicator/FP.CPI.TOTL.ZG) | Annual inflation, % | Candidate pre-reform matching control |
| `trade_open_pct_gdp` | [`NE.TRD.GNFS.ZS`](https://data.worldbank.org/indicator/NE.TRD.GNFS.ZS) | Imports + exports / GDP, % | Candidate pre-reform matching control |
| `unemployment_pct_labor` | [`SL.UEM.TOTL.ZS`](https://data.worldbank.org/indicator/SL.UEM.TOTL.ZS) | % of labour force (modelled ILO estimate) | **Coverage audit only**; early history may be unavailable |

These are transparent candidate controls, **not an exact reconstruction** of
Hope and Limberg's covariate specification. In particular, GDP-per-capita
*growth* is not the paper's Penn World Table level in constant dollars; a
World Bank trade proxy is not a certified IMF vintage, and unemployment may
have shorter coverage than the tax-cut event history. The political left-vote
share and capital openness controls remain outstanding.

## Reproduce the data and coverage audit

From `neoliberalism-evidence-synthesis/`, after building the existing canonical
real-data panel with `event_source.py`, `wid.py` and `prepare.py`:

```bash
python -m causal_tax_cuts.world_bank \
  --panel data/tax-study/panel.csv \
  --output-dir data/wdi-source
```

`--panel` can be omitted when auditing WDI alone. The command generates:

1. `world_bank_controls.csv`: **918 country-years**, four measured macro
   columns; absent values are empty cells, not zeros or interpolations.
2. `world_bank_provenance.json`: requested exact API URLs, retrieval timestamp,
   SHA-256 of every original JSON response, its size, observed counts, and
   country-level missingness. This is a *live* extraction, not a frozen vintage.
3. `outcome_panel_with_macro.csv` (with `--panel`): the entire original
   outcome/policy panel, joined by exact ISO3/year, with blank macro gaps.
4. `macro_overlap_audit.json` (with `--panel`): original outcome and event
   counts, complete-case coverage for three core indicators versus all four,
   losses at the treatment year, country-level overlap, and the count of
   treated reforms with all five **pre-treatment** macro years available.

The GitHub Actions `Historic tax-cut source availability` workflow performs
this live audit on every relevant pull request and publishes the four files as
a seven-day `world-bank-macro-coverage` artifact. It fails when a source changes
schema, unexpectedly returns a subset/pagination, or contains duplicate,
nonfinite or misidentified observations.

## Interpretation and next research gate

- There is no automatic complete-case restriction on the outcome panel here.
  The existing exploratory estimates remain exactly as before.
- Missing data are identified separately for each indicator; a measured zero
  is retained as zero. The optional unemployment series is not included in
  the three-control complete-case benchmark.
- Lagged control availability is a **necessary but not sufficient** matching
  criterion. Eligible donors, their treatment histories and their outcome
  windows must still be re-evaluated after specifying an adjusted design.
- Only lagged **pre-treatment** macro values should enter donor selection;
  conditioning on post-treatment GDP, inflation, trade or unemployment could
  create bias. We must explicitly review confounding and selection before
  interpreting changes in estimates causally.

The next PR should evaluate pre-treatment balance and sample attrition under
declared lag/missingness rules before reporting any adjusted point estimates.
