# Inflation Fell; Prices Did Not — Article 26

Research companion to [the long-form manuscript](article.md) on the pandemic recession, inflation, and the unequal U.S. adjustment, 2020–2025. It contains approximately 13,000 words of original prose; eight original graphics (seven from historical sources and one explicitly hypothetical model); seven frozen raw files, two derived data panels, three results tables, Python scripts, a reproducible notebook, and unit tests.

This is descriptive historical analysis, not a causal attribution of inflation or recovery to an individual law, leader, company, or central-bank action. The data are retrospective 2026-10/2026-11 vintages, not the estimates available in real time in each historical year.

## Offline reproducibility

With Python 3.12 in this folder:

1. Install pinned dependencies using **python -m pip install -r requirements.txt**.
2. Build prepared datasets using **python prepare_data.py**.
3. Build graphics and audit tables using **python analysis.py**.
4. Run both verification modes: **python prepare_data.py --check** and **python analysis.py --check**.
5. Run **python -m unittest discover -s tests -v**.
6. Open **model_walkthrough.ipynb** with the local Python kernel and execute all cells.

Once dependencies are installed, no network access is required. The original source bytes are included in the repository, and [the manifest](data/sources.json) gives original publishers, series identifiers, data scope, and SHA-256 checksums. Source exports with additional unused FRED columns were retained in their original multi-series form.

## Eight figures

| Figure | Measurement | Boundary |
| --- | --- | --- |
| [01: Pandemic unemployment](figures/figure_01_pandemic_unemployment.png) | BLS CPS national, Black and White rates; NBER dates | April 2020 survey misclassification; October 2025 unavailable |
| [02: Output and jobs](figures/figure_02_gdp_and_jobs.png) | BEA real GDP and BLS nonfarm payroll employment | Each baseline is separate; not a no-pandemic counterfactual |
| [03: Inflation and price level](figures/figure_03_inflation_vs_price_level.png) | BLS CPI-U SA level and NSA 12-month rate | Disinflation is not a decline in the general price level |
| [04: Earnings and purchasing power](figures/figure_04_nominal_and_real_earnings.png) | BLS private hourly pay, nominal and CPI-deflated | Changing employee composition; not individual wage paths |
| [05: Median income and Gini](figures/figure_05_median_income_and_gini.png) | Census real household median and pretax inequality | Sampling uncertainty, method changes, no significance test |
| [06: Federal Reserve policy](figures/figure_06_fed_policy_and_balance_sheet.png) | Effective funds rate and Fed balance sheet | Asset holdings are not household fiscal transfers |
| [07: Housing](figures/figure_07_housing_prices_and_shelter.png) | FHFA all-transactions house prices and CPI shelter | Different objects, not a direct affordability comparison |
| [08: Wage selection model](figures/figure_08_hypothetical_wage_selection.png) | Invented two-wage workforce under selective layoffs | **HYPOTHETICAL**, not estimated pandemic outcomes |

## Source files

- **data/raw/bls_monthly_2018_2025.json**: one BLS API response covering seven monthly series: LNS14000000, LNS14000003, LNS14000006, CUSR0000SA0, CUUR0000SA0, CUSR0000SAH1, CES0500000003. Extra 2018 months allow 2019 year-over-year CPI calculations.
- **data/raw/fred_PAYEMS.csv**: BLS monthly payroll employment, thousands of jobs, SA.
- **data/raw/fred_GDPC1.csv**: BEA quarterly real GDP, billions of chained dollars at SAAR.
- **data/raw/fred_FEDFUNDS.csv**: effective federal funds rate, monthly percentage.
- **data/raw/fred_WALCL.csv**: weekly Fed assets, millions of dollars.
- **data/raw/fred_debt_housing.csv**: FHFA quarterly all-transactions U.S. house-price index USSTHPI (NSA), in an original composite CSV.
- **data/raw/fred_income_gini.csv**: annual Census household money-income Gini and real median household income in 2025 dollars, in an original composite CSV.

The two prepared datasets are **data/monthly_panel_2019_2025.csv** (84 calendar months) and **data/annual_panel_2019_2025.csv** (seven annual descriptive records). Outputs also include **results/recovery_milestones.csv**, **results/endpoint_comparisons.csv**, and **results/composition_scenarios.csv**.

## Formulas and baselines

| Indicator | Computation |
| --- | --- |
| Monthly price level | 100 × headline CPI SA for month / headline CPI SA in December 2019 |
| Twelve-month inflation | 100 × (headline CPI NSA for month / headline CPI NSA twelve months earlier − 1) |
| Nominal hourly earnings | 100 × total-private average hourly earnings / December 2019 hourly earnings |
| Real hourly earnings | 100 × (wage / headline CPI SA) / its December 2019 value |
| GDP index | 100 × real quarterly GDP / real GDP in 2019Q4 |
| Payroll jobs index | 100 × monthly nonfarm payroll jobs / February 2020 payroll jobs |
| FHFA home prices | 100 × quarterly USSTHPI / 2019Q4 USSTHPI |
| Real median household income | 100 × annual income in 2025 dollars / 2019 income |
| Fed assets in trillion USD | WALCL in millions divided by 1,000,000 |

The seven-year annual data panel reports means of available monthly or quarterly observations, but it does not claim to reproduce official annual publications. **2025 CPS and CPI-derived annual means contain eleven observed months**, because official October 2025 data are missing; the observed December-over-December endpoint comparisons still use published points.

Figure 8 is a pure selection identity. If the low-pay worker share is *s*, low pay is *L*, high pay is *H*, and proportion *p* of low-paid jobs are eliminated, the mean pay among remaining workers is **[s(1−p)L + (1−s)H] / [s(1−p) + (1−s)]**. With a 50/50 initial split, fixed hourly wages of $15/$35, and half the low-paid jobs removed, average observed pay rises from $25 to $28.33 despite zero individual raises and 25% total job losses. The scenario is illustrative only.

## Source caveats and integrity

**October 2025 shutdown:** the BLS API uses a dash for missing household-survey and most CPI values. The parser accepts missing values only for those six series in October 2025 and rejects any other unexpected missing month. It does not interpolate. The nominal CES hourly wage remains available, but CPI-deflated earnings for October 2025 are blank. See the [CPS notice](https://www.bls.gov/cps/methods/2025-federal-government-shutdown-impact-cps.htm) and [CPI notice](https://www.bls.gov/cpi/additional-resources/2025-federal-government-shutdown-impact-cpi-faq.htm).

**2020 CPS survey problems:** some workers absent because of the pandemic were misclassified as employed, and BLS documented the potential understatement of unemployment. We plot the official data, with a caveat, instead of inventing a corrected series.

**2026 poverty revision:** Census and BLS corrected historical Supplemental Poverty Measure thresholds for 2019–2024. Rather than freezing superseded 2021–2022 SPM point estimates, the article discusses the change and directs readers to the corrected Census publications.

**Different measures:** GDP level recovery is not a counterfactual growth-path recovery; payroll jobs differ from work hours; a shifting-workforce wage average differs from individual wage gains; household income is not disposable income; Census repeated cross-sections are not family panels; FHFA purchase/refinance valuations are not CPI shelter services or affordability; Fed security holdings are not household grants. These distinctions are material to interpretation.

## CI

The accompanying path-scoped GitHub Actions workflow verifies source integrity, runs both preparation and plotting, checks the generated CSV files, confirms all eight PNG files exist, and runs a comprehensive unit test suite. The generated figures and tables are committed so nontechnical readers can read the article as published without running code. Small CSV datasets are exempted from Git LFS in the repository's attributes file.
