# GDP per Capita, Life Expectancy and Basic Services: What Cross-Country Correlations Can and Cannot Identify

This directory is the **public computational companion** to the long-form
Medium manuscript [GDP per Capita, Life Expectancy and Basic Services: What Cross-Country Correlations Can and Cannot Identify](https://github.com/DiogoRibeiro7/research-articles-medium/blob/main/themes/economics-and-society/gdp-life-outcomes/article.md), kept in the author's private
`research-articles-medium` repository. It examines a popular interpretation of Max Roser's
[29 October 2026 Our World in Data chart](https://ourworldindata.org/data-insights/how-is-life-at-different-levels-of-gdp-per-capita).

**Scope:** an independent, reproducible World Bank WDI analysis of four
living-condition outcomes, a cross-sectional snapshot, endpoint changes,
explicit missingness, matched-sample covariate sensitivity, and country-level
dispersion. **Not** a direct reproduction of all twelve OWID graphs, **not** a
causal identification design, and **not** an estimate of the effect of any
particular public policy.

## Primary evidence

- World Bank [Indicators API, v2](https://datahelpdesk.worldbank.org/knowledgebase/articles/889392).
- GDP per capita: [`NY.GDP.PCAP.PP.KD`](https://data.worldbank.org/indicator/NY.GDP.PCAP.PP.KD), constant 2021 purchasing-power-parity (PPP) international dollars.
- Life expectancy: [`SP.DYN.LE00.IN`](https://data.worldbank.org/indicator/SP.DYN.LE00.IN), years.
- Under-five mortality: [`SH.DYN.MORT`](https://data.worldbank.org/indicator/SH.DYN.MORT), deaths per 1,000 live births.
- Electricity access: [`EG.ELC.ACCS.ZS`](https://data.worldbank.org/indicator/EG.ELC.ACCS.ZS), percentage of population.
- Safely managed drinking water: [`SH.H2O.SMDW.ZS`](https://data.worldbank.org/indicator/SH.H2O.SMDW.ZS), percentage of population.
- Survey Gini: [`SI.POV.GINI`](https://data.worldbank.org/indicator/SI.POV.GINI).
- Public health expenditure: [`SH.XPD.GHED.GD.ZS`](https://data.worldbank.org/indicator/SH.XPD.GHED.GD.ZS), percentage of GDP.
- Population: [`SP.POP.TOTL`](https://data.worldbank.org/indicator/SP.POP.TOTL).

The fixed API snapshot was captured on **6 November 2026**, using World Bank
country metadata to remove aggregates. The source reports an update date of
**8 October 2026**. `data/provenance.json` gives the retrieval time, exact URLs,
raw response hashes, row counts, source update, and SHA-256 of the archived
`data/wdi_panel.csv`. The 3,021 retained rows cover only the years used in the
prespecified comparisons; there are no imputed values. **Do not silently
refresh the canonical dataset**: source revisions must be a separate PR.

## Reproduce the published numerical analysis

Python 3.11 or 3.12 is recommended.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python analysis.py --panel data/wdi_panel.csv --results-dir /tmp/gdp-results --figures-dir /tmp/gdp-figures
python audit.py --reference results --candidate /tmp/gdp-results --panel data/wdi_panel.csv --figures /tmp/gdp-figures
```

If you deliberately need a **new data vintage** (which will change the
canonical hash and potentially the estimates), use the separate explicit
capture command below, then rerun analysis and audit your numbers and prose:

```bash
python fetch_data.py --start-year 2000 --end-year 2023 --output-dir /tmp/new-wdi-capture
```

The endpoint performs only public read-only requests; no API credentials are
required. Avoid including temporary outputs in the commit.

## Research design and estimands

The main comparison holds calendar year **2023** fixed and fits one OLS
regression for each of four country outcomes, using a log2 GDP regressor.
For under-five mortality only, the outcome also uses log2; other outcomes
remain in years or percentage points. HC3 heteroskedasticity-robust intervals
are conditional regression intervals, not uncertainty about causal effects.
The country is the sampling unit; unweighted estimates give equal country
weight, while a separate population-weighted slope changes the descriptive
estimand (and is explicitly labelled as lacking an inferential interval).

An endpoint change analysis compares **2000 against 2023** for countries with
both observations; it does not pretend to exploit all annual data or to
control for every evolving confounder. A separate matched-country comparison
uses the latest Gini survey (2018–2023) and public health expenditure
(2020–2023) alongside exact-year 2023 GDP and life expectancy. These
**non-contemporaneous controls** are a sensitivity check, not an identified
model of mediation, public policy or redistribution.

These designs cannot establish the effect of *doing* anything to GDP,
spending or inequality. They expose how much the descriptive relationship
changes with time dimension, missingness, estimand and conditioning.

## Findings as captured (do not extrapolate)

- In 2023, **197** countries have GDP and life expectancy; the unweighted
  log2-GDP slope is **3.651** life-years per GDP doubling (HC3 95% interval
  **3.363–3.939**; two-variable R² **0.742**).
- For **191** countries with both endpoint observations, the 2000–2023
  *change* regression slope is **0.878** years per GDP doubling in the
  change (HC3 95% interval **−0.111 to 1.868**; R² **0.015**).
- The 2023 safely managed water comparison has only **137** paired countries,
  compared with 197 for life expectancy: coverage is not uniform.
- In the GDP band of **10,000–25,000 constant 2021 PPP international dollars**,
  life expectancy varies by **10.15 years** between the 10th and 90th
  country percentiles (n=54). This spread is descriptive, not a within-country
  inequality estimate.
- Adding survey Gini and public health expenditure to a matched-country
  specification changes the GDP coefficient from **3.877** to **3.454**;
  the matched sample contains only **132** countries. This coefficient shift
  does not establish that either covariate is a confounder or mediator.

See all tables in [`results/`](results/) and source-specific caveats in
[`SCIENTIFIC_AUDIT.md`](SCIENTIFIC_AUDIT.md).

## Publication assets

1. `figures/01_2023_country_scatter.png` — four 2023 country scatter plots.
2. `figures/02_changes_not_causality.png` — paired-country changes, 2000–2023.
3. `figures/03_life_residuals.png` — cross-sectional unexplained variation.
4. `figures/04_model_sensitivity.png` — country selection and adjustment.
5. `figures/05_outcome_coverage.png` — percentage of countries with outcomes.

The private manuscript embeds the image URLs at a **fixed public commit SHA**
to prevent later code or data updates from silently rewriting its evidence.
Nothing in this directory is the privately edited manuscript.

## Limitations and interpretation

1. Cross-country ecological associations do not identify individual effects.
2. GDP per capita is a national output average, not a median household income.
3. PPP improves international price comparability but is not a local budget.
4. Survey Gini has infrequent, methodologically heterogeneous observations.
5. Health spending may be a downstream consequence of income, not a valid
   pre-treatment adjustment variable; contemporaneous controls cannot solve
   dynamic confounding or reverse causation.
6. Cross-sectional and change regressions answer different descriptive
   questions. Their different R² values are **not** a formal test of causality.
7. Under-five mortality and other outcomes are affected by historical public
   health innovation, epidemiology, demography and institutions.
8. Observational units are countries, but their residuals may be correlated
   regionally and historically. HC3 intervals do not fix this dependence.
9. Linear percentage-point regression can extrapolate outside [0, 100]; the
   fit is used descriptively only, within the observed country range.
10. Source data can be revised. The public snapshot is versioned and hashed.

For a source-by-source provenance and estimation audit, read
[`SCIENTIFIC_AUDIT.md`](SCIENTIFIC_AUDIT.md).
