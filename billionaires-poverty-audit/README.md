# A Billionaire Is Not a Poverty Policy — research companion

This project supports [the Medium manuscript](article.md) on billionaires per capita, poverty below US$30/day in 2021 PPP, economic development, and causal inference.

## Status and evidence classification

**Research draft — not publication-ready.** The original figure reports a 64-country slope of −43.3 and R² of 0.581. A preceding independent reconstruction reported a 63-country residence-based R² of 0.567, and R² 0.784 for GDP-only versus 0.791 for GDP plus billionaire density. Its original 2026 country-level snapshots, provenance ledger, model code and figures have **not** been recovered during this retry. Accordingly these reconstruction statistics remain **reported but unverified**, not claims produced by the program in this PR.

The earlier research notes also recorded significant sensitivity to Forbes residence-versus-citizenship counts and to regional controls. Reproducing that sensitivity with frozen country-level input is a gate before publishing. No fabricated data or fitted coefficients are substituted for missing observations.

## Reproduce using verified country-year inputs

The [analysis.py](analysis.py) script performs a correctly specified country-level descriptive regression using an **externally supplied** CSV; it never silently downloads a changing source or manufactures country data:

```bash
cd billionaires-poverty-audit
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python analysis.py --input data/countries.csv --output results/models.json
```

The script fails with a clear message when the required input file is absent or invalid. It uses ordinary least squares (and HC3 robust errors when estimable). Both the raw and GDP-adjusted models are fitted to the same complete-case positive-density sample, so the R² difference has a defined denominator. It reports counts and transformation, but **does not identify a causal effect**.

The required CSV schema is documented in [DATA_CONTRACT.md](DATA_CONTRACT.md), and the source retrieval/archival gates are in [CLAIM_AUDIT.md](CLAIM_AUDIT.md).

## Research design

1. Recover and archive the Forbes March 2026 list; verify the attribution rules for residence and citizenship, including revisions.
2. Archive the World Bank PIP / OWID 2021-PPP $30/day series with country and survey years, not merely current values.
3. Archive matching WDI constant-PPP GDP per capita, year, and population with source date.
4. Match country codes manually using a documented ledger; do not silently merge countries and territories.
5. Report a balanced sample comparison, survey-type splits, log-zero selection, alternative welfare lines, and leave-one-country-out checks.
6. Test sensitivity to region, population weights, wealth shares (not income Gini as a surrogate), and historical starting points.
7. Build a matched year-country panel to distinguish between-country and within-country relationships; no automatic claim of causality.
8. Commit eight original publication PNGs and SVGs only after their numerical provenance and image-generation code are verified.

## Conceptual limitations

Billionaire density is not wealth concentration; GDP per capita is not median disposable income; residence is not company birthplace; the US Census poverty measure is not the World Bank's $30 PPP measure; p-values and R² do not identify policy effects.

## Primary sources

- [Original 2026 Forbes release](https://tollbit.forbes.com/sites/pr/2026/03/10/forbes-40th-annual-worlds-billionaires-list-elon-musk-is-worlds-richest-person-ever-recorded/)
- [Forbes 2026 person-list API](https://www.forbes.com/forbesapi/person/billionaires/2026/position/true.json?limit=5000)
- [World Bank PIP](https://pip.worldbank.org/)
- [OWID $30/day poverty](https://ourworldindata.org/grapher/poverty-share-on-less-than-30-per-day)
- [WDI PPP GDP per capita](https://data.worldbank.org/indicator/NY.GDP.PCAP.PP.KD)
- [WDI income Gini](https://data.worldbank.org/indicator/SI.POV.GINI)
- [World Inequality Database](https://wid.world/data/)
- [US Census Bureau poverty report 2025](https://www.census.gov/library/publications/2026/demo/p60-290.html)
