# Primary sources and input status (13 November 2026)

| Dataset or method | Primary source | Status and caveat |
| --- | --- | --- |
| Published causal model | [Hope & Limberg (2022), *Socio-Economic Review*, 20(2), 539–559](https://doi.org/10.1093/ser/mwab061) | Author-estimated 18-country, 1965–2015 panel with 30 major tax-cut observations. |
| Authors' event-coded data | [Taxrich.uk data catalogue](https://www.taxrich.uk/data), [`hope_limberg_reforms_11_2020.csv` direct preview](https://drive.google.com/file/d/1FgV9wfFl7FG5cl7PZ7UMzQQ24Wye0VT3/view?usp=sharing) | Link verified; CSV content **not retrieved**. Do not guess headings or coding. |
| Authors' latent tax indicator | [Taxrich.uk data catalogue](https://www.taxrich.uk/data), [`hope_limberg_tax_rich_11_2020.csv` direct preview](https://drive.google.com/file/d/1r2k0Rex1gMTFmRkbzzuJ93PWIe5gePeK/view?usp=sharing) | Link verified; CSV **not retrieved**; 7-indicator Bayesian latent construct not independently recomputed. |
| Analytical appendix | [LSE Research Online — original working paper](https://eprints.lse.ac.uk/107919/1/Hope_economic_consequences_of_major_tax_cuts_published.pdf) | Authors' matching design, covariate origins in Appendix Table A2, model results. |
| Top 1% pre-tax income share | [WID bulk data](https://wid.world/data/), [official WID codes dictionary](https://wid.world/codes-dictionary/), [official R download tool](https://github.com/world-inequality-database/wid-r-tool) | Live downloader is implemented for `sptinc992j`, percentile `p99p100`, equal-split adults, official 0–1 fraction converted to percentage points; source archive not yet frozen. |
| Harmonised unemployment | [OECD Data Explorer](https://data-explorer.oecd.org/) | Employment outcome; must harmonise historic definition. |
| Real GDP per capita | [Penn World Table / University of Groningen](https://www.rug.nl/ggdc/productivity/pwt/) | Authors use 2011 USD; price base, series and log transform must be aligned. |
| Capital-account openness | [Chinn–Ito Index — authors' site](https://web.pdx.edu/~ito/Chinn-Ito_website.htm) | Financial openness covariate; release not yet fixed. |
| Trade openness | [IMF Data](https://data.imf.org/) | Exports plus imports as share of GDP; vintage and GDP concept not yet reconciled. |
| Government expenditure | [OECD Data Explorer](https://data-explorer.oecd.org/) | Government expenditure / GDP; series definition not yet reconciled. |
| Left vote share | Brady et al. (2020), cited in Appendix Table A2 of the [working paper](https://eprints.lse.ac.uk/107919/1/Hope_economic_consequences_of_major_tax_cuts_published.pdf) | No exact compatible archive export verified; **unavailable**. |

## Paper-sample country codes

The paper's 18-country sample is: `AUS`, `AUT`, `BEL`, `CAN`, `DNK`, `FIN`,
`FRA`, `DEU`, `IRL`, `ITA`, `JPN`, `NLD`, `NZL`, `NOR`, `SWE`, `CHE`, `GBR`,
`USA`. The authors' current source catalogue says **19 countries**, so the
additional country's inclusion must **not** be assumed.

## Rules before presenting real-data estimates

1. Download the real author files, record download date, source, release and
   SHA-256. Inspect the CSV shape and assess redistribution permissions.
2. Reconcile the paper's **18 countries / 30 events** with the downloaded
   version. A mismatch blocks replication until explained.
3. Preserve original WID units (0–1 fractions vs 0–100 percentages) and the
   exact measure of *pre-tax national income*, not household money-income Gini.
4. Record all country-year exclusions and imputations, including pre-trend
   periods, post-reform windows and covariate missingness.
5. Distinguish (a) the authors' published effect, (b) tested synthetic
   arithmetic and (c) effects independently estimated on fully audited data.

**No real-data causal estimates have been produced by this PR.**
