# GDP Is Not Economic Strength

Status: research protocol; no empirical findings yet.

## Question
Does aggregate GDP conceal differences in sectoral composition, durable productive assets and exposure to shocks? Distinguish economic output, welfare, wealth and resilience. Do not assume manufacturing is intrinsically superior to services.

## Scope
1995 onward, latest comparable common year; US, EU-27 (fixed membership), UK, Switzerland, Japan, China and India, with country-level European comparisons. Reproduce the motivating nominal comparison only if its exact country list and data vintage are available; otherwise label the independently defined sample.

## Data dependency
Use https://github.com/DiogoRibeiro7/data for reusable data and provenance. Audit existing records before registering new sources. Pin immutable revisions, hashes, source dates and licenses. Do not mirror restricted data. No fabricated observations.

Candidate sources:
- World Bank WDI: https://data.worldbank.org/ (NY.GDP.MKTP.CD, NY.GDP.MKTP.PP.CD, NY.GDP.MKTP.KD, SP.POP.TOTL, NV.AGR.TOTL.ZS, NV.IND.TOTL.ZS, NV.IND.MANF.ZS, NV.SRV.TOTL.ZS).
- IMF WEO: https://www.imf.org/en/Publications/WEO/weo-database
- Eurostat: https://ec.europa.eu/eurostat/ (nama_10_a64).
- OECD STAN and TiVA: https://www.oecd.org/ (verify edition and coverage).
- World Bank CWON 2024: https://www.worldbank.org/en/publication/the-changing-wealth-of-nations
- ILOSTAT: https://ilostat.ilo.org/
- UN Comtrade: https://comtradeplus.un.org/

## Analysis
1. Nominal, PPP and real GDP comparisons; aggregate only consistent units and years. PPP is not a measure of international purchasing power.
2. Sectoral gross value added shares, concentration HHI and effective sector count. Manufacturing is nested within industry; never double count. GDP includes net product taxes, so normalize sector composition by total GVA rather than silently forcing GDP shares to sum to one.
3. Decompose nominal changes into volume, deflator and currency effects where matching series exist.
4. Historical shock windows: financial crisis, pandemic and energy shock. Compare cumulative real output losses against transparent pre-shock trend baselines and recovery duration; vary baseline and window. Descriptive associations, not causal estimates.
5. Compare real GDP per capita and real comprehensive wealth per capita using matched observations. Explicitly address valuation uncertainty.
6. Model hypothetical sector shocks with transparent assumptions and sensitivity bounds. Separate direct exposure from input-output propagation; use domestic technical coefficients only when compatible data exist. AI occupational exposure is not equivalent to job loss or output loss.
7. Compare household real disposable income and productivity where harmonized measures exist.

## Implementation and acceptance
Typed Python, documented functions, schema checks, deterministic outputs, pytest tests for missingness, non-overlapping sectors, units, HHI and shocks; notebook and figures generated from pinned inputs. Report coverage and revisions. No zero-imputation for missing values. Use existing CI conventions. No causal claim without an identification strategy.

## Publication
Manuscript belongs in https://github.com/DiogoRibeiro7/research-articles-medium. Target 5,000+ words, references, figures and publication handoff. Merge data and empirical changes before pinning the manuscript evidence. Do not present hypotheses as results. No publication or merge without review.
