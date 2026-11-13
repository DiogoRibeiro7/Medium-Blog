# Data sources, definitions and provenance

Frozen research inputs reviewed on **13 November 2026**. `data/manifest.json` pins the originating public `DiogoRibeiro7/Medium-Blog` revision (`d054aeb97a3dba4e93eb489157c3c2cd11d48dd2`) and each local CSV's SHA-256. The underlying source projects keep their own acquisition logs, detailed transformations and source notices.

| Local snapshot | Original project | Primary publisher and series | Measurement caveat |
| --- | --- | --- | --- |
| `housing_indices_2015_2025.csv` | [`housing-supply-counterfactual`](../housing-supply-counterfactual/data/source_manifest.json) | Eurostat `prc_hpi_q` and `tipsho60`/`tipsho30` | Arithmetic mean of four quarterly *nominal* HPI observations; official standardised house-price/disposable-income indicator is national, with country-specific historical scaling. |
| `fte_salary_2015_2025.csv` | [`minimum-wage-in-work-poverty`](../minimum-wage-in-work-poverty/data/fetch_log.txt) | Eurostat `nama_10_fte`, annual EUR | Salary per full-time-equivalent employee; nominal gross *average*, not median wages or disposable income. |
| `portugal_labour_shares_1995_2024.csv` | [`portugal-labour-share`](../portugal-labour-share/README.md) | Eurostat `nama_10_gdp`, `nama_10_a10_e`, `gov_10a_main`, `nasa_10_nf_tr` | 18 alternative labour-share definitions; GDP market prices and factor-cost denominators are different. |
| `us_top_income_tax_rate_1964_2018.csv` | [`tax-cuts-rich-historical-cycles`](../tax-cuts-rich-historical-cycles/data/historical_rates_sources.md) | IRS Statistics of Income Table 23 via FRED `IITTRHB` | Top *statutory marginal* rate, not effective taxes paid or all top-income taxation. |
| `us_household_gini_1967_2018.csv` | [`tax-cuts-rich-historical-cycles`](../tax-cuts-rich-historical-cycles/data/inequality_sources.md) | U.S. Census Bureau historical household income tables | Money-income Gini before taxes and transfers; **1993 Current Population Survey redesign** changes comparability. |
| `us_growth_investment_1964_2018.csv` | [`tax-cuts-rich-historical-cycles`](../tax-cuts-rich-historical-cycles/data/macro_sources.md) | BEA real GDP growth and gross private domestic investment share | Annual growth and investment ratios, not a tax-law counterfactual or firm-level incentive response. |

Primary dataset pages:

- Eurostat [quarterly house-price index](https://ec.europa.eu/eurostat/databrowser/view/prc_hpi_q/default/table), [housing-price-to-income indicator](https://ec.europa.eu/eurostat/databrowser/view/tipsho60/default/table), [salary per full-time equivalent](https://ec.europa.eu/eurostat/databrowser/view/nama_10_fte/default/table) and [national accounts](https://ec.europa.eu/eurostat/databrowser/view/nama_10_gdp/default/table).
- U.S. Federal Reserve Bank of St. Louis, [highest individual statutory rate `IITTRHB`](https://fred.stlouisfed.org/series/IITTRHB).
- U.S. Census Bureau, [historical income tables](https://www.census.gov/data/tables/time-series/demo/income-poverty/historical-income-households.html).
- U.S. Bureau of Economic Analysis, [national income and product accounts](https://www.bea.gov/products/national-income-and-product-accounts).
- The [Eurostat November 2026 release](https://ec.europa.eu/eurostat/web/products-eurostat-news/w/ddn-20261109-1?language=en) confirms the published 2015–2025 standardised price/income changes, including the ≈60.8% figure for Portugal (small differences arise from calculating directly on the frozen rounded index values).

## Data-processing contract

1. The housing input is the unmodified derived annual snapshot from the European housing companion, with original coverage and status flags retained.
2. The salary input is **only** a geography/year filter of the `nama_10_fte`, annual, EUR observations for Portugal, Spain, France, Germany and EU27 from 2015 through 2025; all values and quality flags are unchanged.
3. The 18-definition Portuguese labour-share table is copied unchanged, covering 1995–2024.
4. U.S. historical tax, Gini and GDP/investment CSVs are copied unchanged.
5. All snapshots have declared row counts and SHA-256. `analysis.py` refuses changed or incomplete inputs instead of filling missing years.

## Analytical definitions

For geography \(c\), in year \(t\), with the 2015 reference year \(0\):

\[
H_{ct}=100\frac{\mathrm{HPI}_{ct}}{\mathrm{HPI}_{c0}}, \quad
W_{ct}=100\frac{\mathrm{FTEsalary}_{ct}}{\mathrm{FTEsalary}_{c0}}, \quad
R_{ct}=100\frac{H_{ct}}{W_{ct}}.
\]

The relative gap shown in the summary is \(R_{c,2025}-100\) percent, with ratio index normalised to 100 in 2015. The official price/income line is independently normalised from Eurostat's *standardised* indicator. It is **not** derived from the FTE salary series. The common price level cancels in a ratio of two nominal indices; this does not make the result a measure of rental burden, mortgage interest, deposit affordability, or purchasing power by wealth percentile.

The reported labour-share endpoint difference is \(s_{2024}-s_{1995}\) **percentage points**, not relative percent growth. It does not replicate or replace the trend and break inference in the upstream module.

## Licences, vintages and limitations

These are frozen extracts of separately documented official data, not live queries. Eurostat, Census, IRS/FRED and BEA observations may later be revised; this publication captures an explicit vintage. Source agencies' reuse conditions continue to apply. The repository licence for *code* does not claim ownership of the statistical source material.

No observational co-movement is treated as proof that an institutional reform caused the observed outcome. No standard errors are attached to the descriptive ratios, because these are deterministic transformations of official published aggregates, not independently sampled policy-effect estimators.
