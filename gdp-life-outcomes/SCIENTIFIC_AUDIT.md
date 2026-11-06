# Scientific and numerical audit

## Claim boundary

The original [Our World in Data chart (Roser, 2026)](https://ourworldindata.org/data-insights/how-is-life-at-different-levels-of-gdp-per-capita)
contains 12 comparisons from several agencies and potentially different
underlying data years. This companion is an **independent, narrower World Bank
WDI analysis** of four outcomes and related sensitivity checks. It must not
be represented as a pixel-level, value-level, or full-replication audit of
all 12 graphs.

## Dataset contract

`fetch_data.py` requests each source series directly from World Bank API v2
over 2000–2023 and retains only the year(s) used in this analysis. It excludes
regional and income aggregates using the `region.id == "NA"` country metadata
marker, checks indicator IDs, uniqueness and finite values, and archives
3021 nonmissing observations. World Bank revision metadata is 2026-10-08;
capture time and all API payload canonical hashes are recorded in
`data/provenance.json`. Country identities use World Bank ISO3 codes; a unit
is a country or reporting territory in the World Bank country endpoint, and
some separate territories are included. There is **no imputation**.

The published code and archived source are mechanically linked by SHA-256.
`audit.py` checks the panel hash and independent numeric re-computation. CI
tests do not call external APIs and cannot silently absorb a WDI revision.

## Statistical protocol

- Predictor: `log2(GDP per capita PPP, constant 2021 international dollars)`.
- Levels: 2023 country cross-section, OLS with an intercept and HC3 intervals.
- Outcomes: life expectancy in years, log2 under-five mortality, electricity
  percentage points, safely managed water percentage points.
- Secondary weighting: population-weighted descriptive OLS slope, no interval.
- Changes: 2000–2023 endpoint differences; fit `delta_y ~ 1 + delta_log2_GDP`.
- Adjustment sensitivity: same matched set first with and then without Gini
  (latest survey 2018–2023) and government health expenditure share of GDP
  (latest 2020–2023). No causal meaning is assigned to adjusted coefficients.
- Regional analysis: Spearman correlations within World Bank regions when
  at least eight countries report both GDP and life expectancy.
- Dispersion: 10th–90th percentile spread of life expectancy within fixed
  prespecified PPP GDP ranges.

## Independently verified numerical checkpoints

The following figures are archived in machine-readable CSV in `results/`.

| Comparison | N | Estimate | 95% HC3 interval | R² |
| --- | ---: | ---: | ---: | ---: |
| 2023 life expectancy per GDP doubling | 197 | 3.651 years | [3.363, 3.939] | 0.742 |
| 2023 log2 under-five mortality per GDP doubling | 187 | -0.819 | [-0.883, -0.756] | 0.741 |
| 2023 electricity access per GDP doubling | 196 | 9.784 pp | [8.082, 11.486] | 0.543 |
| 2023 safely managed water per GDP doubling | 137 | 15.925 pp | [14.616, 17.234] | 0.760 |
| 2000–2023 change in life-years per change in GDP log2 | 191 | 0.878 years | [-0.111, 1.868] | 0.015 |
| 2000–2023 change in log2 under-five mortality per change in GDP log2 | 184 | -0.469 | [-0.621, -0.317] | 0.187 |

The original post's rhetoric about a $1,000 to $10,000 move being equivalent
to a $10,000 to $100,000 move mixes **equal multiplicative log-axis distance**
with unspecified welfare effects. This study reports associations *per
doubling*, not a causal effect of tenfold GDP growth.

For life expectancy in 2023, GDP explains about 74% of the variation in this
one-regressor fit across the included countries. In the endpoint-change
regression, the corresponding R² is about 1.5%; the interval for the GDP
change slope spans zero. **This does not prove that economic growth fails to
improve health.** It shows that the two uncontrolled descriptive designs
answer different questions and that the cross-sectional fit alone cannot
identify a dynamic causal process.

## Missingness, sensitivity and ecological fallacy

The 2023 safely managed water series contains 137 country pairs versus 197
GDP–life pairs. For survey Gini and health-spending matched comparisons,
only 132 countries remain. Relative GDP slopes, on the 2023 life-years scale:

- All country pairs: 3.651 (n=197).
- Matched GDP-only: 3.877 (n=132).
- Matched with reported Gini and public health-spending share: 3.454 (n=132).

Matching changes the sample; conditioning changes the model. These are
different contrasts. The coefficient change after conditioning is not an
identified estimate of the effect of inequality or public spending. Poverty,
income distribution, administrative capacity, epidemiological environment,
demography and history remain partially or entirely unmeasured.

The country-level life expectancy difference between the 90th and 10th
percentiles is 10.15 years within the PPP GDP range 10,000–25,000 (n=54).
This is a difference **between countries with broadly comparable national
averages**, not an estimate of the within-country rich-poor mortality gap.

## What was and was not established

**Established:** a strong contemporaneous association with log GDP, notable
outcome-specific missingness, residual country differences, and instability
of descriptive slopes across samples and temporal designs.

**Not established:** GDP's exogenous effect on health; the effect of taxation,
public insurance, redistribution, deregulation or private production; an
adequate causal adjustment set; any policy ranking; or an intervention
counterfactual. The absence of these results is intentional, not a tacit
argument that growth is irrelevant.

## Source provenance and caveats

- World Bank, [WDI API documentation](https://datahelpdesk.worldbank.org/knowledgebase/articles/889392): programmatic source and query semantics.
- World Bank, [GDP per capita PPP](https://data.worldbank.org/indicator/NY.GDP.PCAP.PP.KD): volume of domestic value added, not a direct household income series.
- World Bank, [Gini index methodology](https://databank.worldbank.org/metadataglossary/jobs/series/SI.POV.GINI): survey instruments, reference years, welfare concepts and comparability differ.
- Our World in Data, [international dollars](https://ourworldindata.org/international-dollars): why purchasing-power parity matters and what it does not fix.
- Our World in Data, [healthcare financing](https://ourworldindata.org/financing-healthcare): income, public revenue, public health spending and causal qualifications.
- Our World in Data, [disease burden](https://ourworldindata.org/burden-of-disease): institutional and epidemiological determinants beyond aggregate income.
- Pritchett and Summers (1993/1996), [Wealthier Is Healthier](https://documents.worldbank.org/en/publication/documents-reports/documentdetail/913361468739485204/wealthier-is-healthier): a distinct instrumental-variable attempt with additional assumptions, not evidence supplied by the 12 charts.
- Acemoglu and Johnson (2007), [Disease and Development](https://www.nber.org/papers/w12269): uses historical epidemiological transitions to address the reverse direction.

References support the contextual discussion, not the validity of the present
observational design. Report reproducibility, identification and external
validity as separate scientific claims.
