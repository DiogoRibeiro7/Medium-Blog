# The Promise Returns — research companion

This folder develops a cross-era political-economy essay about top-income and corporate tax changes, the mechanisms through which private incentives may affect growth, and the fiscal and distributional consequences.

## Current state

- [Manuscript](article.md): over 8,000 words, historically comparative and written for Medium.
- Literature: CRS, CBO, GAO, peer-reviewed cross-national tax-reform study, and firm-level TCJA research; direct source links are in the manuscript.
- Identification boundary: no new empirical causal effects are estimated or implied.
- Figures and frozen official datasets: **not yet created**. This is an editorial draft, not a reproducible research release.

## Publication gates

1. Independently verify the statute timelines and the precise scope of each historical tax reform against IRS, Congress, Treasury, CRS, and original law text.
2. Freeze the official top statutory rate series and explain the marginal-versus-effective-rate distinction, changes in tax base, and appropriate comparison periods.
3. Add original source-verified figures for rate schedules, revenue composition, household distribution, investment, debt, and inequality; label historical associations descriptive.
4. Implement a typed, documented Python 3.12 companion with source manifest and SHA-256 checks, deterministic plots, notebook, tests, and path-scoped CI.
5. Add a transparent sensitivity analysis of the hypothetical investment/public-capital model; avoid calibrating invented coefficients as historical causal effects.
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
