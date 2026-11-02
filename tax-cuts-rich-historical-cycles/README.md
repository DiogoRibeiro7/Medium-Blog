# The Promise Returns — research companion

This folder develops a cross-era political-economy essay about top-income and corporate tax changes, the mechanisms through which private incentives may affect growth, and the fiscal and distributional consequences.

## Current state

- [Manuscript](article.md): over 8,000 words, historically comparative and written for Medium.
- Literature: CRS, CBO, GAO, peer-reviewed cross-national tax-reform study, and firm-level TCJA research; direct source links are in the manuscript.
- Identification boundary: no new empirical causal effects are estimated or implied.
- **Reproducible hypothetical supplement now included:** typed standard-library Python model, six unit/integration tests, a nine-cell sensitivity CSV, an SVG figure, and path-scoped CI. These are not historical estimates.
- First official historical evidence slice added: IRS/FRED highest statutory individual rate, 1964–2018, a source-noted 55-year CSV and deterministic descriptive SVG; structural and landmark tests are included. A second official OMB/FRED 1964–2018 total federal receipts/GDP series and two-panel chart are now added, with fiscal/calendar measurement caveats. BEA real growth and gross private domestic investment share are now also frozen for 55 years with a descriptive figure and tests. Census household money-income Gini for 1967–2018 is also frozen and plotted with a visible 1993 survey break. After-tax distribution, disaggregated fixed investment, and Medium-compatible PNGs are **not yet created**.

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

## Census household inequality

[Census/FRED GINIALLRH CSV](data/household_gini_1967_2018.csv) · [provenance and 1993 comparability note](data/inequality_sources.md) · [descriptive Gini chart](results/household_gini.svg) · [typed renderer](inequality_context.py) · [tests](test_inequality_context.py).

These are 52 years of *pretax household money-income* Gini, not the wealth distribution and not disposable income after federal taxation and benefits. The 1993 CPS survey redesign is visibly separated instead of joined by a continuous line, and values before 1967 are intentionally not invented. Generate with `python tax-cuts-rich-historical-cycles/inequality_context.py`. The aggregate local suite now passes **22 tests** and the workflow regenerates every evidence figure.

## Publication gates

1. Independently verify the statute timelines and the precise scope of each historical tax reform against IRS, Congress, Treasury, CRS, and original law text.
2. ~~Freeze the first official top statutory rate series~~ — IRS 1964–2018 slice completed with caveats; add earlier historical period and current-year statutes only when independently verified, and keep effective tax burdens separate.
3. ~~Start source-verified descriptive historical figures~~ — statutory rate, receipts/GDP, BEA growth/investment, and Census household Gini now exist as SVG; add revenue composition, after-tax distribution, fixed business investment, debt, wealth, and Medium-compatible PNGs.
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
