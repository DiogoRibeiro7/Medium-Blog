# The Price of Stability Was Not Shared Equally

Reproducible companion for **[The Price of Stability Was Not Shared Equally: Stagflation, Volcker, Reagan, and America's Market Turn, 1970–1990](article.md)**. This long-form Medium-ready essay follows [the civil-rights and Great Society article](../civil-rights-great-society/article.md) in the political-economic history series.

The question is not merely whether inflation fell. It is how inflation control interacted with unemployment, racial labor-market disparities, tax and fiscal policy, collective bargaining, and the redistribution of economic risk. Seven figures use carefully delimited historical series from original public institutions, and one uses **invented** mathematical parameters. None claims a causal effect from chronology alone.

## Reproduce

Python 3.10 or newer, from this directory:

```bash
python -m pip install -r requirements.txt
python analysis.py
python -m unittest discover -s tests -v
```

To direct all regenerated images and output CSVs to another path:

```bash
python analysis.py --output /tmp/stagflation-reproduction
```

The script recreates **eight original PNGs** in [figures/](figures/) and **three machine-readable result CSVs** in [results/](results/). The [Jupyter notebook](model_walkthrough.ipynb) imports the same functions from [analysis.py](analysis.py), rather than keeping an independent implementation. A scoped [GitHub Actions workflow](../.github/workflows/stagflation-market-article.yml) checks the data, model assumptions, publication content, and generation contracts.

No external services, API credentials, random generator, undocumented correction, or network request are needed at reproduction time.

## Figure evidence classification

| No. | Class | Original-source record or declared mechanism | Interpretation limit |
| --- | --- | --- | --- |
| 1. CPI-U annual inflation | Historical, derived | BLS `CPIAUCNS` through FRED; percent change of consecutive annual mean price indexes | Not December-to-December inflation; not causal |
| 2. National unemployment | Historical, derived | BLS `UNRATE` through FRED; mean of 12 monthly seasonally adjusted rates | Annual average is not unemployment spell length or cumulative job loss |
| 3. Effective federal funds | Historical, derived | Federal Reserve `FEDFUNDS` through FRED; annual mean of monthly rates | Not a policy target, expected real rate, or mortgage rate |
| 4. Black/White unemployment | Historical, derived | BLS `LNS14000006` and `LNS14000003`; annual means | Source Black data begin 1972; the gap is not a policy effect |
| 5. Household income Gini | Historical descriptive | Census `GINIALLRH` via FRED, 1967–1990 | Money income, not wealth or after-tax resources |
| 6. Union affiliation | Historical descriptive | BLS CPS Table 35, 1983–1990 | Comparable series starts 1983; membership differs from representation |
| 7. Federal deficit/GDP | Historical descriptive | OMB `FYFSDFYGDP` via FRED, 1970–1990 | **Fiscal years**, not calendar years; negative means deficit |
| 8. Real-wage indexation | **Hypothetical** | Invented 8% inflation, 3% nominal base raise, 0/50/100% indexation | Not historical wage, wage-equilibrium, or causal evidence |

## Primary historical data sources and transformations

### BLS/Federal Reserve annual macro series

[fred_annual_macro.csv](data/fred_annual_macro.csv) contains the **1969–1990** source values of:

- `cpi_u_annual_average_index`: BLS CPI-U all-items, U.S. city average, not seasonally adjusted, [CPIAUCNS](https://fred.stlouisfed.org/series/CPIAUCNS). Units: monthly index values averaged across the twelve calendar months.
- `unemployment_annual_average_percent`: BLS civilian unemployment rate, seasonally adjusted, [UNRATE](https://fred.stlouisfed.org/series/UNRATE). Units: percent; arithmetic mean of monthly percent rates.
- `fed_funds_annual_average_percent`: Federal Reserve monthly effective federal funds rate, [FEDFUNDS](https://fred.stlouisfed.org/series/FEDFUNDS). Units: percent; arithmetic mean of monthly rates.
- `black_unemployment_annual_average_percent`: BLS rate [LNS14000006](https://fred.stlouisfed.org/series/LNS14000006), seasonally adjusted, mean of twelve monthly observations. Unavailable in 1969–1971.
- `white_unemployment_annual_average_percent`: BLS rate [LNS14000003](https://fred.stlouisfed.org/series/LNS14000003), seasonally adjusted, mean of twelve monthly observations.

The monthly FRED data were retrieved on **October 31, 2026** from [FRED's original-source graph CSV download endpoint](https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE%2CCPIAUCNS%2CFEDFUNDS%2CLNS14000006%2CLNS14000003%2CGINIALLRH), then grouped by calendar year. This source endpoint returns monthly series and the annual Census series as different CSV members of a ZIP archive when frequencies differ. The checked-in values are rounded to six decimals after the annual means are calculated. Original source values and historical revisions are available through each series page.

For Figure 1 the code calculates:

\[
\pi_y=100\left(\frac{\overline{CPI}_y}{\overline{CPI}_{y-1}}-1\right).
\]

The **1969** annual CPI mean is retained to calculate `1970` inflation. No inflation observation is plotted for 1969, since the downloaded article CSV does not contain its preceding 1968 annual mean. A December-over-December method would give different numbers, even though both can be legitimate when labeled.

| Source-backed check | Value |
| --- | ---: |
| 1970 annual CPI-U inflation | approx. 5.84% |
| 1980 annual CPI-U inflation | approx. 13.55% |
| 1983 annual CPI-U inflation | approx. 3.21% |
| 1982 annual mean civilian unemployment | approx. 9.71% |
| 1981 annual mean effective federal funds | approx. 16.38% |
| 1983 annual mean Black unemployment | 19.50% |
| 1983 annual mean White unemployment | 8.375% |

The 1983 Black–White gap is **11.125 percentage points**. The gap uses rate subtraction, not a percentage ratio. The Black monthly source has no comparable entries for 1969–1971, so these local CSV cells remain empty, `np.nan` in the reader, and are excluded from the racial figure. The earliest displayed year is 1972. This is not interpolation or an asserted zero rate.

### Census household income inequality

[census_household_gini.csv](data/census_household_gini.csv) transcribes **1967–1990** annual values from the U.S. Census Bureau's [Income Gini Ratio for Households by Race of Householder, All Races](https://fred.stlouisfed.org/data/GINIALLRH), series `GINIALLRH` on FRED. The source gives **0.394** in 1970, **0.403** in 1980, and **0.428** in 1990.

This is inequality in *household money income*. It does not measure net worth, individual earnings, wealth inherited over generations, or final after-tax and after-transfer disposable income. Since it is not an after-tax resource measure, it cannot mechanically isolate who benefited from federal marginal income-tax changes. Changes in household size and composition can also influence the aggregate index.

The data source can be revised or subject to historical survey-method caveats; combining it with other inequality concepts requires documented adjustments. No unannounced smoothing or interpolation is applied.

### BLS union membership and representation

[bls_union_membership.csv](data/bls_union_membership.csv) reproduces the **1983–1990** *Total, both sexes* rows of [BLS CPS Table 35, Union affiliation of employed wage and salary workers](https://www.bls.gov/cps/wlftable35-2010.htm). Fields:

- `union_membership_rate_percent`: union **members** as share of employed wage and salary workers.
- `union_members_thousands`: count of union members, **thousands**.
- `employed_wage_salary_workers_thousands`: denominator employment count, **thousands**.
- `represented_by_unions_rate_percent`: workers represented by unions, including nonmembers, share of the same employment category.

In 1983 there were 17,717 thousand members among 88,290 thousand employed wage and salary workers, a rounded union membership share of 20.1%. In 1990 there were 16,776 thousand members among 104,876 thousand employed wage and salary workers, or 16.0%. Representation shares were 23.3% and 18.2%, respectively. The tests confirm that the member-count ratios reproduce the published rates within their rounding precision.

The BLS identifies **1983 as the earliest strictly comparable annual membership rate** for the modern series. It would be methodologically unsafe to append a 1970 union-density number from another definition without explaining how populations, survey questions, and coverage compare. The article discusses earlier labor history in words while keeping the chart bounded to the comparable observations.

### OMB fiscal balance

[omb_fiscal_balance.csv](data/omb_fiscal_balance.csv) reproduces annual **1970–1990** observations of the U.S. Office of Management and Budget's [Federal Surplus or Deficit as Percent of Gross Domestic Product, FYFSDFYGDP](https://fred.stlouisfed.org/series/FYFSDFYGDP), distributed through FRED. Negative percentages indicate **deficits**. Observations are fiscal years:

| Fiscal year | Unified balance / fiscal-year GDP |
| --- | ---: |
| FY 1980 | −2.64444% |
| FY 1982 | −3.86247% |
| FY 1983 | −5.87675% |
| FY 1990 | −3.74717% |

The federal fiscal-year end moved from June to September after 1976, and the historical OMB series is **not** contemporaneously aligned with each calendar-year point in the BLS figures. A deficit is a flow over a period, whereas public debt is an accumulated stock. The original source series is unified surplus/deficit relative to fiscal-year GDP, not real GDP growth, inflation-adjusted borrowing, or a measure of the independent tax-law effect.

## Model contracts

**Annual CPI inflation.** `annual_average_inflation` validates positive, finite annual-average CPI index levels and calculates the percentage change. Negative values are allowed in the result if the measured annual average price level falls; no assumption of positive inflation is imposed.

**Racial unemployment gap.** `racial_unemployment_gap` accepts two same-shaped one-dimensional rate arrays, preserves missing values, validates supplied finite rates as percentages, and subtracts White from Black. Its output units are **percentage points**, not percentages of another rate.

**Hypothetical wage indexation.** The sole fictional figure holds `g=0.03` (base annual nominal raise), `π=0.08` (annual inflation), and lets indexation share `a` range over 0, 0.5, 1. Nominal wage growth is stipulated to be:

\[
g_{nominal}(a)=(1-a)g+a\pi, \quad a\in[0,1].
\]

For whole-year horizon `t`, the illustrative real-wage index is:

\[
W_t(a)=100\left(\frac{1+g_{nominal}(a)}{1+\pi}\right)^t.
\]

This formula produces exactly 100 in every year under full indexation by construction. Its other paths depend entirely on invented parameters. It **does not** represent a historical wage series, causal labor-policy impact, macroeconomic equilibrium, inflation expectations, sectoral productivity, taxation, or employment selection. The analysis validates the parameter ranges and whole-year horizons and exports 63 explicitly `hypothetical` results.

## Generated result tables

- [source_based_summary.csv](results/source_based_summary.csv): **187** historical source cells, of which **three** are deliberately marked `missing` (pre-1972 Black unemployment). Source data use `historical_descriptive`, never `hypothetical`.
- [descriptive_derived_metrics.csv](results/descriptive_derived_metrics.csv): **40** derived observations: 21 annual CPI inflation rates, plus 19 Black–White unemployment rate differences for 1972–1990. Each is labeled `derived_descriptive`.
- [hypothetical_real_wage_paths.csv](results/hypothetical_real_wage_paths.csv): **63** invented index points, exactly three 21-year indexation scenarios. Every row is `hypothetical`.

## Historical boundaries

1. Statistical association over a presidential term does not establish the causal effect of that president's policies.
2. Inflation is a **rate of change**, not the price level; falling inflation is disinflation, not necessarily falling prices.
3. Monthly-to-annual averaging hides short peaks, particularly in the funds rate; annual mean is not the maximum monthly observation.
4. Effective nominal federal funds rates cannot be interpreted as expected real borrowing costs without inflation expectations and contract details.
5. Black/White rate differences describe disparities but do not separate sector, occupation, geography, discrimination, or labor-market institutions.
6. Household money-income inequality and post-tax tax incidence are distinct estimands.
7. The 1983–1990 union table uses **comparable** definitions; a broader historical union series would require a separate source audit.
8. FY fiscal balances and calendar-year labor data are not synchronized to the same dates; negative balances are deficits, not debt stocks.
9. Real-wage Figure 8 is hypothetical in title, caption, code, source CSV and output table; it must not be interpreted as observed earnings data.
10. The documented policy facts are drawn from original [Federal Reserve](https://www.federalreservehistory.org/essays/great-inflation), [FAA](https://www.faa.gov/about/history/brief_history), [DOT](https://www.transportation.gov/policy/aviation-policy/airline-rules-fares), [IRS](https://apps.irs.gov/app/understandingTaxes/teacher/whys_thm02_les06.jsp), [Congress](https://www.congress.gov/97/statute/STATUTE-95/STATUTE-95-Pg172.pdf), and [State Department](https://history.state.gov/milestones/1969-1976/oil-embargo) material. They are not inputs to an estimated national causal model.

Reproducibility means the arithmetic, source definitions, and missing observations are inspectable. It does not convert descriptive trends into identified policy effects.
