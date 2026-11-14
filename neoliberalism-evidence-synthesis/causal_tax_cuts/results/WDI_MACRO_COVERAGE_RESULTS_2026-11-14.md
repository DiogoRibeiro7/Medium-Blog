# Official World Bank macroeconomic coverage — 14 November 2026

This is the first independently downloaded World Development Indicators (WDI)
coverage audit for the 18-country tax-cut and top-income-share study. It is
**not** a causal estimate, and the existing 0.795 percentage-point five-year
exploratory contrast has **not** been re-estimated or adjusted here.

## Verified source vintage and requested sample

- Sample: **18 OECD countries × 51 years (1965–2015) = 918 country-years**.
- Publisher: [World Bank WDI](https://data.worldbank.org/); source-2 annual
  series from four official indicator CSV ZIP exports.
- The CSV archives state **last updated 11 November 2026**. The source archives
  were downloaded and independently checked on 14 November 2026.
- Source data are never interpolated or filled with zero. Each downloaded ZIP
  has its own SHA-256, preserved in the companion
  [`wdi_macro_coverage_2026-11-14.json`](wdi_macro_coverage_2026-11-14.json).

| WDI series | Variable | Observed / 918 | Missing |
| --- | --- | ---: | ---: |
| `NY.GDP.PCAP.KD.ZG` | Real GDP per capita growth, annual % | 918 | 0 |
| `FP.CPI.TOTL.ZG` | Consumer price inflation, annual % | 918 | 0 |
| `NE.TRD.GNFS.ZS` | Trade as % of GDP | 853 | 65 |
| `SL.UEM.TOTL.ZS` | Modelled ILO unemployment, % of labour force | 450 | 468 |

The unemployment series covers 1991–2015 in all 18 countries; its 468 missing
values are exactly **26 earlier years × 18 countries**. Treat it as optional
coverage information rather than requiring it in the initial matching model.
Trade openness is missing in some early country-years. The first three series
are candidate pre-treatment controls, **not** an exact recreation of the
authors' Penn World Table, IMF and OECD covariates.

## Overlap with the previously audited income-share panel

The existing WID/treatment panel has **853** country-year observations and
**29** observed major tax-cut event-years; both counts are unchanged. Exact
ISO3/year matching gives:

| Availability criterion | Observations |
| --- | ---: |
| Original outcome/treatment country-years | 853 |
| With GDP per capita growth observed | 853 |
| With CPI inflation observed | 853 |
| With trade openness observed | 810 |
| With all three core controls available | **810** |
| With all four controls, including unemployment | 450 |
| Original outcome rows lacking one or more core controls | **43** |
| Tax-cut event-years missing core controls at the event year | 0 of 29 |
| Tax-cut events with all five **pre-reform** years of core controls | **27 of 29** |

Thus an uncritical complete-case panel for three core controls would discard
**43 of the 853 outcome rows (5.0%)**. However, the adjusted estimator must
not blindly discard entire event cohorts based on controls after treatment.
The 27/29 figure audits only lagged macro availability for treated countries;
it is **not** a count of reform events eligible for a valid matched
counterfactual with observed donors and outcome windows.

## Country-level joint core-control coverage

| Country | Joint outcome + three controls | Country | Joint outcome + three controls |
| --- | ---: | --- | ---: |
| AUS | 51 | AUT | 36 |
| BEL | 36 | CAN | 51 |
| CHE | 41 | DEU | 40 |
| DNK | 49 | FIN | 46 |
| FRA | 51 | GBR | 46 |
| IRL | 45 | ITA | 43 |
| JPN | 46 | NLD | 40 |
| NOR | 46 | NZL | 46 |
| SWE | 51 | USA | 46 |

The country-level sums reconcile to **810**, and the underlying four source
hashes, indicator totals and panel-overlap counts are frozen in JSON. The
unmodified live ZIP downloads, derived country-year values and the joined
panel are reproducible with:

```bash
python -m causal_tax_cuts.world_bank \
  --panel data/tax-study/panel.csv \
  --output-dir data/wdi-source
```

The GitHub Actions source-probe job also publishes the four derived files in
the `world-bank-macro-coverage` artifact for seven days. Future runs may
differ as the World Bank revises its source data; compare archive checksums.

## Research interpretation

The good GDP/inflation coverage and the 810-row three-control overlap make a
controlled sensitivity analysis feasible. They do **not** make it identified:
the donor sets, pre-treatment balance, missingness selection, temporally
varying confounding and exposure timing remain to be evaluated. The next
study step is to quantify the exact eligible matched cohort and pre-reform
balance under declared lagged-control rules before estimating an adjusted
contrast or any uncertainty interval.
