# The Promise Returns — research companion

This folder develops a cross-era political-economy essay about top-income and corporate tax changes, the mechanisms through which private incentives may affect growth, and the fiscal and distributional consequences.

## Current state

- [Manuscript](article.md): over 8,000 words, historically comparative and written for Medium.
- Literature: CRS, CBO, GAO, peer-reviewed cross-national tax-reform study, and firm-level TCJA research; direct source links are in the manuscript.
- Identification boundary: no new empirical causal effects are estimated or implied.
- **Reproducible hypothetical supplement now included:** typed standard-library Python model, six unit/integration tests, a nine-cell sensitivity CSV, an SVG figure, and path-scoped CI. These are not historical estimates.
- First official historical evidence slice added: IRS/FRED highest statutory individual rate, 1964–2018, a source-noted 55-year CSV and deterministic descriptive SVG; structural and landmark tests are included. A second official OMB/FRED 1964–2018 total federal receipts/GDP series and two-panel chart are now added, with fiscal/calendar measurement caveats. BEA real growth and gross private domestic investment share are now also frozen for 55 years with a descriptive figure and tests. Census household money-income Gini for 1967–2018 is also frozen and plotted with a visible 1993 survey break. A selected CBO 2018 before/after transfers-and-federal-taxes comparison is now included; broader after-tax time series, further disaggregation within nonresidential investment, and Medium-compatible PNGs are **not yet created**.

## Hypothetical model reproduction

Run `python -m unittest discover -s tax-cuts-rich-historical-cycles -p 'test_*.py' -v` from repository root and `python tax-cuts-rich-historical-cycles/model.py --out tax-cuts-rich-historical-cycles/results`. The workflow regenerates and compares the committed results byte-for-byte. No external packages or network calls are required.

The sensitivity model holds labor and baseline productivity fixed, varies taxable-base elasticity (0, 0.5, 1) and public-capital productivity (0, 0.3, 0.6), and uses a hypothetical rate cut from 35% to 28%. It assumes a fixed response of private capital to net-of-tax incentives and a specified share of revenue loss financed by lower public capital. These parameters are deliberately invented; the results illustrate competing channels, not a historical estimate or optimal tax recommendation.

[Figure: hypothetical output sensitivity](results/sensitivity.svg) · [Machine-readable nine-cell results](results/sensitivity.csv) · [Model implementation](model.py) · [Unit tests](test_model.py)

## Official IRS historical-rate evidence

[Annual top-bracket rates, 1964–2018](data/top_marginal_rate_1964_2018.csv) · [Provenance and interpretation](data/historical_rates_sources.md) · [Historical rate SVG](results/historical_top_rate.svg) · [Typed renderer](historical_rates.py) · [Tests](test_historical_rates.py).

The original source is IRS Statistics of Income Table 23, viewed through [FRED IITTRHB](https://fred.stlouisfed.org/data/IITTRHB). This is the top regular-income-tax *statutory marginal* rate, not an effective rate or a causal determinant of growth. The FRED series ends in 2018; no recent rates are imputed. Regenerate with `python tax-cuts-rich-historical-cycles/historical_rates.py` and run all 10 tests using the command above. The same path-scoped workflow now regenerates and compares both evidence charts.

## Federal receipts comparison

[OMB/FRED FYFRGDA188S 55-year CSV](data/federal_receipts_gdp_1964_2018.csv) · [source and alignment caveats](data/federal_receipts_sources.md) · [two-panel historical chart](results/tax_rate_and_receipts.svg) · [typed renderer](fiscal_context.py) · [structural tests](test_fiscal_context.py).

This is total federal receipts from all revenue sources divided by nominal annual GDP, **not** a measure of taxes paid by wealthy households. FRED pairs federal fiscal-year receipts with annual GDP, and the statutory tax-rate series has a different economic denominator altogether. There is no regression or causal interpretation. Regenerate with `python tax-cuts-rich-historical-cycles/fiscal_context.py`; the path-scoped workflow tests and verifies every committed CSV/SVG result. All 14 local tests passed.

## BEA growth and investment context

[FRED/BEA annual macro CSV](data/real_growth_private_investment_1964_2018.csv) · [source and measurement notes](data/macro_sources.md) · [descriptive two-panel figure](results/growth_and_investment.svg) · [typed renderer](macro_context.py) · [tests](test_macro_context.py).

The BEA data are annual real GDP percentage growth and gross private domestic investment as a share of nominal GDP, not an estimate of tax-induced business fixed investment. The latter includes housing and inventories. Both are retrospective revised accounts. The deterministic figure is regenerated with `python tax-cuts-rich-historical-cycles/macro_context.py`, and GitHub Actions checks it against the committed file. All 18 local tests passed across the companion.

## BEA nonresidential fixed investment

[BEA/FRED A008RE1A156NBEA annual 1964–2018 dataset](data/nonresidential_fixed_investment_share_1964_2018.csv) · [source and conceptual boundary](data/nonresidential_fixed_investment_sources.md) · [paired investment figure](results/nonresidential_investment.svg) · [typed renderer](nonresidential_investment.py) · [tests](test_nonresidential_investment.py).

Nonresidential fixed investment excludes housing and inventories, unlike gross private domestic investment. The comparison isolates a more relevant aggregate for evaluating proposed business-investment channels, but it does **not** estimate a causal tax effect. Generate it with `python tax-cuts-rich-historical-cycles/nonresidential_investment.py`. The full companion now passes **32 local tests**.

## Census household inequality

[Census/FRED GINIALLRH CSV](data/household_gini_1967_2018.csv) · [provenance and 1993 comparability note](data/inequality_sources.md) · [descriptive Gini chart](results/household_gini.svg) · [typed renderer](inequality_context.py) · [tests](test_inequality_context.py).

These are 52 years of *pretax household money-income* Gini, not the wealth distribution and not disposable income after federal taxation and benefits. The 1993 CPS survey redesign is visibly separated instead of joined by a continuous line, and values before 1967 are intentionally not invented. Generate with `python tax-cuts-rich-historical-cycles/inequality_context.py`. The aggregate local suite now passes **22 tests** and the workflow regenerates every evidence figure.

## CBO household distribution in 2018

[Selected quintile estimates](data/cbo_2018_distribution.csv) · [CBO provenance and incidence caveats](data/cbo_2018_distribution_sources.md) · [before/after household resources SVG](results/cbo_2018_distribution.svg) · [typed renderer](cbo_distribution.py) · [tests](test_cbo_distribution.py).

The three selected quintiles compare average household income before and after **means-tested transfers and all modeled federal taxes** in the same year. They are not a measure of the 2017 tax law's effect. Separately, CBO's modeled prior-law comparison estimates that the 2017 act reduced average federal rates by about 1.3 percentage points in the lowest two quintiles and 2.7 points in the highest quintile. That counterfactual estimate is discussed in the essay, not reverse-engineered from the income bars. The expanded suite passes **27 local tests**. Regenerate the chart with `python tax-cuts-rich-historical-cycles/cbo_distribution.py`.

## OMB fiscal balance and publicly held debt

[OMB/FRED 1964–2018 balance and public-debt CSV](data/federal_balance_public_debt_1964_2018.csv) · [provenance and denominator notes](data/fiscal_balance_debt_sources.md) · [two-panel fiscal figure](results/fiscal_balance_public_debt.svg) · [typed renderer](fiscal_balance_debt.py) · [tests](test_fiscal_balance_debt.py).

The annual budget surplus or deficit is a flow; debt held by the public is a stock. Both FRED ratios use nominal annual GDP, and their OMB federal fiscal-year numerators are not aligned perfectly with the calendar-year denominators. They cannot measure the stand-alone causal consequences of tax cuts. Regenerate with `python tax-cuts-rich-historical-cycles/fiscal_balance_debt.py`. **37 local tests pass**.

## Receipts versus net outlays

[OMB/FRED annual net outlays, 1964–2018](data/federal_net_outlays_gdp_1964_2018.csv) · [accounting and vintage notes](data/revenue_outlays_sources.md) · [paired receipts/outlays SVG](results/revenue_outlays.svg) · [typed reconciliation and renderer](revenue_outlays.py) · [tests](test_revenue_outlays.py).

The independently frozen receipts, net outlays, and deficit series reconcile within a documented 0.00015 percentage-point tolerance; the maximum residual is 0.00013 points. This is an **accounting** check and no tax-law counterfactual is estimated. The local suite now passes **42 tests**. Regenerate with `python tax-cuts-rich-historical-cycles/revenue_outlays.py`.

## Research estimate interpretation audit\n\n[Primary research and agency-estimate cross-check](data/research_estimate_audit.md) records the top-1% income-share estimand in Hope–Limberg, the *loose consensus* 11% tangible corporate investment synthesis in Chodorow-Reich–Zidar–Zwick, CRS's 4.5% alternative tax-term measurement calculation, and the GAO report's actual 2022 publication date. This is an interpretation audit, not a new causal analysis.\n\n## Verified statute chronology\n\n[Enacted-law timeline and interpretive caveats](data/legislative_chronology_sources.md) now verifies the principal 1964, 1981, 1986, 2001, 2003 and 2017 episodes against congressional primary sources. The 1964 Act explicitly predicted eventual revenue feedback, but the essay treats that as a contemporaneous political prediction rather than a demonstrated outcome. Pre-1913 case-law details and 1986 surcharge/phase-out mechanics remain open.\n\n## Claim audit and release readiness

[Claim-to-source ledger](CLAIM_AUDIT.md) identifies which numerical observations are reproduced and which historical, causal and bibliographic assertions still need independent validation. Successful local tests and earlier CI checks establish reproducibility of the frozen artifacts, not causal identification or publication readiness.

## OMB receipt composition

[OMB FY2027 Table 2.3, 1964–2018](data/omb_receipts_by_source_1964_2018.csv) · [source and vintage notes](data/omb_receipts_by_source_sources.md) · [category comparison SVG](results/receipt_composition.svg) · [typed renderer](receipt_composition.py) · [tests](test_receipt_composition.py).

This is a five-category decomposition of federal receipts expressed in percent of GDP, with each component published to one decimal. It is kept **separate from the previously frozen FRED total-receipts vintage**. Category shares are not high-income effective tax rates or causal estimates. The new module passes five tests locally; all repository tests remain checked by CI.

## OMB federal spending by function

[OMB FY2027 Table 3.1 annual fiscal categories](data/omb_outlays_by_function_1964_2018.csv) · [official definitions and vintage caveats](data/omb_outlays_by_function_sources.md) · [functional-spending SVG](results/outlay_composition.svg) · [typed renderer](outlay_composition.py) · [tests](test_outlay_composition.py).

The six OMB superfunctions (including negative undistributed offsetting receipts) reconcile with the official total to within source rounding for each fiscal year from 1964 through 2018. The figures use the same FY2027 source vintage as the new OMB receipts-by-source chart; they are **not** an estimate of reform-induced spending changes. The new module passes five local tests; the complete project runs in CI.

## Medium PNG export\n\n[Figure catalogue, captions and alt text](MEDIUM_FIGURES.md) and [typed publication exporter](export_medium_png.py) prepare all eleven tracked SVG charts as 2,200-pixel-wide PNGs. The path-scoped GitHub Actions workflow uploads a 30-day publication artifact named `tax-cuts-rich-medium-png` on successful runs. The PNG files are generated, not tracked; download and visually inspect them before pasting into Medium.\n\n## Publication gates

1. Independently verify the statute timelines and the precise scope of each historical tax reform against IRS, Congress, Treasury, CRS, and original law text.
2. ~~Freeze the first official top statutory rate series~~ — IRS 1964–2018 slice completed with caveats; add earlier historical period and current-year statutes only when independently verified, and keep effective tax burdens separate.
3. ~~Start source-verified descriptive historical figures~~ — statutory rate, receipts/GDP, BEA growth/investment, and Census household Gini now exist as SVG; add revenue composition, a full after-tax distribution time series, investment components (equipment, structures, and IP), subfunction and investment breakdowns, wealth, and Medium-compatible PNGs.
4. Implement a typed, documented Python 3.12 companion with source manifest and SHA-256 checks, deterministic plots, notebook, tests, and path-scoped CI.
5. ~~Add an initial hypothetical sensitivity analysis~~ — completed as an explicitly illustrative model; extend robustness and validate notation before publication, without presenting invented coefficients as historical causal effects.
6. Audit the tax research literature, including the measurement critique of the TCJA investment estimates and the identifying assumptions in the cross-country study.
7. Review presentation for Medium: figures as image files, no LaTeX-only graphics or large Markdown tables.
8. Add the essay to the appropriate **historical-series index** once the reproducibility and editorial gates pass. The numbered `ARTICLES.md` catalogue also contains a separate statistical series; do not renumber or overwrite it.

## Suggested primary-source acquisition

- IRS historical federal income-tax rates: official tax schedules and thresholds.
- BEA and BLS: output, investment, labor, and prices.
- CBO: projected tax-act effects and household income after taxes and transfers.
- GAO: realized effective corporate liabilities and measurement qualifications.
- Congressional Research Service: tax law, revenue, and historical evidence.
- Original research: Hope & Limberg (2022); Chodorow-Reich, Zidar & Zwick (2024).

The draft distinguishes policy mechanisms and causal hypotheses from observed results; no chart, estimate, or validation result should be claimed before it exists.