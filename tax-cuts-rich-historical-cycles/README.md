# The Promise Returns — research companion

This folder develops a cross-era political-economy essay about top-income and corporate tax changes, the mechanisms through which private incentives may affect growth, and the fiscal and distributional consequences.

## Current state

- [Manuscript](article.md): over 8,000 words, historically comparative and written for Medium.
- Literature: CRS, CBO, GAO, peer-reviewed cross-national tax-reform study, and firm-level TCJA research; direct source links are in the manuscript.
- Identification boundary: no new empirical causal effects are estimated or implied.
- **Reproducible hypothetical supplement now included:** typed standard-library Python model, six unit/integration tests, a nine-cell sensitivity CSV, an SVG figure, and path-scoped CI. These are not historical estimates.
- Official historical datasets and source-verified descriptive figures: **not yet created**. This remains an editorial draft, not a complete historical research release.

## Hypothetical model reproduction

Run `python -m unittest discover -s tax-cuts-rich-historical-cycles -p 'test_*.py' -v` from repository root and `python tax-cuts-rich-historical-cycles/model.py --out tax-cuts-rich-historical-cycles/results`. The workflow regenerates and compares the committed results byte-for-byte. No external packages or network calls are required.

The sensitivity model holds labor and baseline productivity fixed, varies taxable-base elasticity (0, 0.5, 1) and public-capital productivity (0, 0.3, 0.6), and uses a hypothetical rate cut from 35% to 28%. It assumes a fixed response of private capital to net-of-tax incentives and a specified share of revenue loss financed by lower public capital. These parameters are deliberately invented; the results illustrate competing channels, not a historical estimate or optimal tax recommendation.

[Figure: hypothetical output sensitivity](results/sensitivity.svg) · [Machine-readable nine-cell results](results/sensitivity.csv) · [Model implementation](model.py) · [Unit tests](test_model.py)

## Publication gates

1. Independently verify the statute timelines and the precise scope of each historical tax reform against IRS, Congress, Treasury, CRS, and original law text.
2. Freeze the official top statutory rate series and explain the marginal-versus-effective-rate distinction, changes in tax base, and appropriate comparison periods.
3. Add original source-verified figures for rate schedules, revenue composition, household distribution, investment, debt, and inequality; label historical associations descriptive.
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
