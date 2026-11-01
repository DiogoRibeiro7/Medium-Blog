# The Boom Was Not a Balance Sheet

Reproducible companion to **[The Boom Was Not a Balance Sheet: Globalization, Household Credit, and the Fragile Prosperity Before 2008](article.md)**, the next installment in the [political-economic history series](../ARTICLES.md). It follows [the stagflation and market turn essay](../stagflation-market-turn/article.md).

The period is **1990–2007**. Seven figures summarize original government economic time series. A separate eighth figure illustrates a **hypothetical** collateral and household-equity identity. No regression, graph, or scenario identifies the effect of a historical policy on any economic outcome.

## Reproduce offline

Python 3.10 or newer, from this directory:

```bash
python -m pip install -r requirements.txt
python prepare_data.py --check
python analysis.py
python -m unittest discover -s tests -v
```

The source snapshots are committed in `data/raw/`. No downloads, API keys, credentials, R, or paid services are needed for these commands. To regenerate the annual panel from the frozen source files, run `python prepare_data.py`; `--check` performs a byte-for-byte audit without changing files.

To keep generated images and results out of the checked-in directories while reviewing:

```bash
python analysis.py --output /tmp/pre-2008-reproduction
```

The script creates exactly eight PNGs in `figures/` and three CSVs in `results/`. The [notebook](model_walkthrough.ipynb) walks through the same measurements and scenario without introducing alternative numerical assumptions.

## Files

| Path | Purpose |
| --- | --- |
| [`article.md`](article.md) | Long-form Medium-ready essay with source notes and figure placement |
| [`prepare_data.py`](prepare_data.py) | Typed, validated annualization from original source snapshots |
| [`analysis.py`](analysis.py) | Seven historical figures, one hypothetical figure, and three diagnostic tables |
| [`data/sources.json`](data/sources.json) | Exact series IDs, origin institutions, measurement definitions, retrieval vintage and transformations |
| [`data/raw/`](data/raw/) | Original FRED CSV graph exports, including distinct end-of-period debt observations |
| [`data/bls_union_1990_2007.csv`](data/bls_union_1990_2007.csv) | BLS membership table transcription, preserving annual total and sector breakdowns |
| [`data/annual_panel_1990_2007.csv`](data/annual_panel_1990_2007.csv) | Frozen 18-row, 21-column annual research panel |
| [`results/endpoint_comparisons.csv`](results/endpoint_comparisons.csv) | Same-unit 1990 and 2007 endpoints and transformations |
| [`results/lead_lag_sensitivity.csv`](results/lead_lag_sensitivity.csv) | **Descriptive-only** small-sample correlations after first differencing |
| [`results/hypothetical_leverage.csv`](results/hypothetical_leverage.csv) | Explicitly invented mortgage loan-to-value stress scenarios |
| [`tests/test_analysis.py`](tests/test_analysis.py) | Data completeness, measurement, reproduction and publication checks |

## Eight figures

| # | Image | What it does and does not measure |
| --- | --- | --- |
| 1 | [`figure_01_manufacturing_employment.png`](figures/figure_01_manufacturing_employment.png) | Manufacturing payroll jobs and their share of all nonfarm payroll; no attribution of the decline to trade |
| 2 | [`figure_02_trade_balance.png`](figures/figure_02_trade_balance.png) | BEA nominal exports, imports and net exports relative to nominal GDP; not a bilateral China or NAFTA estimate |
| 3 | [`figure_03_income_growth_and_inequality.png`](figures/figure_03_income_growth_and_inequality.png) | Normalized real GDP per person and real median household income plus Census household Gini; different populations and methods |
| 4 | [`figure_04_union_membership.png`](figures/figure_04_union_membership.png) | BLS unionization among wage/salary workers and private/public member counts; not a causal effect on wages |
| 5 | [`figure_05_house_price_income.png`](figures/figure_05_house_price_income.png) | FHFA all-transactions index against nominal household median income, separately normalized to 1990; not mortgage affordability |
| 6 | [`figure_06_household_debt.png`](figures/figure_06_household_debt.png) | Q4 households-and-nonprofits debt stock relative to Q4 GDP annual rate, and the mortgage share of that debt |
| 7 | [`figure_07_interest_rates.png`](figures/figure_07_interest_rates.png) | Effective federal funds and ten-year Treasury rates; neither series is a borrower mortgage contract rate |
| 8 | [`figure_08_hypothetical_leverage.png`](figures/figure_08_hypothetical_leverage.png) | Fictional LTV scenarios with unchanged loan principal and price shocks; **not** a default probability or a historical estimate |

## Measurement and data dictionary

The annual panel covers exactly eighteen calendar years, **1990–2007**. The original CSV export was retrieved on **31 October 2026**, so observations reflect revisions available by that date. It is not a contemporaneous 1990 or 2007 vintage. The original FRED exporter returned a ZIP when combining quarter-end balance-sheet stocks with quarterly flow variables. We separated and preserved the two CSV streams: `quarterly.csv` for flows and indices and `quarterly,_end_of_period.csv` for debt. The exported README provides additional raw-file notes.

The [series inventory](data/sources.json) records exact FRED identifiers and original publishers. The major transformations are:

1. **Payroll employment:** Average all twelve seasonally adjusted BLS monthly levels for each year. Convert thousands of employees to millions. The manufacturing payroll share is the ratio of those annual means, not an industry share of GDP.
2. **Trade:** Average four quarterly BEA seasonally adjusted annual-rate dollar flows. Divide nominal imports, exports and net exports by the corresponding mean nominal GDP. Their common units make this a valid dollar-flow ratio. It does not distinguish price and physical-volume changes.
3. **Output and household income:** Average quarterly BEA real GDP per capita for each year; use annual Census median household income in the published 2025-price basis. Index **each** series to its own 1990 observation. This normalizes different scales but does not convert people to households or turn GDP into earnings.
4. **Census income inequality:** Use the all-household Gini `GINIALLRH`, not family-only Gini. These estimates refer to money income before tax and most noncash benefits, not disposable income or wealth. The CPS income instrument changed in 1993. The graph marks that change.
5. **Unions:** Transcribe the BLS 2016 historical Spotlight table for 1990–2007. Sector counts are rounded and may not exactly add to the total. The membership rate includes all employed wage and salary workers, not only private-sector workers.
6. **House prices:** Use the FHFA all-transactions U.S. index, averaged over four quarterly observations, against nominal annual Census household median income. Separate rebasing to 1990=100 does not produce a median house-price-to-income ratio.
7. **Debt:** Use *fourth-quarter end-of-period* `CMDEBT` (debt securities and loans) and `HHMSDODNS` (one-to-four-family mortgage debt) from the Federal Reserve's household-and-nonprofit sector. Divide debt measured in millions by 1,000 to express billions, then divide by the **same fourth quarter** nominal GDP at a seasonally adjusted annual rate. This stock/flow convention is a scale comparison, not a forecast of household debt service or the balance sheet of the median household.
8. **Interest rates:** Average twelve monthly federal funds rates and Treasury constant-maturity yields. Neither measure is the typical effective mortgage rate, and rates alone cannot identify the cause of house-price changes.

### Missingness is part of the evidence

The October 2026 FRED graph export of `TDSP` (household debt service payment ratio) contained historical observations only from 2005 onward. We therefore **omit** it from a 1990–2007 comparison rather than fill nineteen years of nonexistent observations or conflate it with the debt-to-GDP stock ratio. The raw export still includes the column for auditability; the annual panel intentionally does not use it.

`prepare_data.py` validates 12 monthly, four quarterly and one annual measurement for every required series-year, rejects missing values and duplicates, checks the BEA trade accounting identity, and refuses to extrapolate. Data accuracy is distinct from identification: a correct aggregate chart still cannot tell us *why* households borrowed or factories employed fewer people.

## Mathematical companion

An exploratory lag-correlation table pairs annual *log changes* in the FHFA price index with annual *percentage-point changes* in the balance-sheet debt/GDP ratio, under lags −2 through +2. At most seventeen overlapping annual differences are available. These coefficients are not hypothesis tests, forecasts, or evidence that one series caused the other. Trends were differenced, but reverse causation, common shocks, serial dependence and contemporaneous measurement remain.

The sole explicitly hypothetical model fixes the initial home price at 100, the loan balance at `100 × LTV`, and imposes a percentage price decline `d`, without repayments or transaction costs:

\[
E(d,L) = 1-d-L, \qquad \mathrm{loss\ of\ original\ equity} = \frac{d}{1-L}.
\]

`E` is expressed as a fraction of the original home price and `L` is the initial loan-to-value ratio, between 0 and 1. At 80% LTV, a 10% fall consumes half the initial 20% equity. At 95% LTV, the same decline makes equity negative. **Negative equity is not synonymous with default**; household liquidity, debt service, contract terms, recourse rules and employment are absent from this identity. Its pedagogical purpose is to demonstrate a balance-sheet sensitivity, not to estimate mortgage crisis losses.

## Interpretation boundaries

- Trade data are national totals, not import exposure by region or industry. For a causal local-market design, read Autor, Dorn and Hanson's [original research](https://www.nber.org/papers/w18054).
- The GDP/median comparison changes units of observation, household composition, price indices and distribution. The graph explicitly avoids treating their gap as a labor-share statistic.
- The Census CPS ASEC [methodology history](https://www.census.gov/topics/income-poverty/income/guidance/cps-methodology-changes.html) documents breaks during the period.
- Federal Reserve Z.1 periodically revises sector balance sheets. The series includes nonprofits and is not median household indebtedness.
- Mortgage securitization and underwriting can be studied with the Federal Reserve's [2006 HMDA analysis](https://www.federalreserve.gov/pubs/bulletin/2007/articles/hmda/) and Mian and Sufi's [mortgage-credit evidence](https://www.nber.org/papers/w13936), not inferred from the national co-movement plot.
- The [Financial Crisis Inquiry Commission](https://www.govinfo.gov/app/details/GPO-FCIC) found substantial institutional and regulatory failures; its published report also contains dissenting interpretations. A single interest-rate or trade chart cannot adjudicate those disputes.

The manuscript is for Medium publication, but inclusion in this repository does not imply it has already appeared on Medium.
