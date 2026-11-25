# Portugal at the tail of the EU: schooling under the Estado Novo and the income gap, 1900–2025

Companion code for a Medium article that asks how much of Portugal's place near the bottom of the European Union still comes from the dictatorship of 1926–1974. It follows income relative to Western Europe from 1900, adult schooling from 1900 in 22 European countries, the age profile of education today, a development-accounting split of the output gap, and convergence since 1995 across the 27 member states, with Spain, Greece and the post-communist members as comparisons.

## What it finds

All numbers are from `output.txt`.

- **Where Portugal stands (Eurostat, 2025).**
  - GDP per head in purchasing power standards: 81.4 (EU27 = 100), 18th of 27. Czechia, Lithuania, Malta and Slovenia were below Portugal in 1995 and are above it now; Poland (81.1) is level and Estonia (79.7) close.
  - Labour productivity per hour worked: 66.7, 24th of 27 (only Greece, Bulgaria and Latvia are lower). Per person employed: 79.5, 22nd.
- **Income over the long run (Maddison Project Database 2023, against the aggregate of twelve Western European countries).**
  - 41% in 1900, 37% in 1925, 33% at the lowest (1940), 42% in 1950, 40% in 1960, 58% in 1973, 68% in 2000, 66% in 2022.
  - The regime's first 34 years were stagnation relative to the reference; the gain came in 1960–1973, when real GDP per head grew 6.8% a year.
- **Schooling under the regime (Barro and Lee 2015, adults aged 25–64).**
  - Portugal had 0.66 years in 1900, 1.12 in 1925 and 3.30 in 1975.
  - Across 22 European countries, the gain of 1925–1975 regressed on the 1925 level predicts 3.74 years for Portugal; it gained 2.18, a residual of −1.56, third lowest after Turkey and Norway. Spain's residual is −1.42.
  - Countries ruled by communist parties after 1945 have a mean residual of +0.73; Russia went from 1.20 to 6.50 years.
  - The seven countries that started between 2.0 and 3.5 years in 1925 led Portugal by 1.76 years then and by 3.52 in 1975.
  - Had Portugal gained what the regression predicts, it would have had 4.86 years in 1975; with Russia's gain, 6.42. In PWT human capital, those paths close 48% and 89% of the 1975 gap to the seven.
- **Persistence (Barro and Lee v3, EU members, 1950–2015).**
  - Portugal has the lowest adult schooling of the 27 in every five-year observation from 1950 to 2015.
  - Czechia and Slovakia had Portugal's 2015 level (9.41 years) in 1970; Spain, Greece and Italy in 2000.
  - Rank correlation of the members' levels between 1975 and 2015: 0.75.
- **The age profile today (Eurostat LFS, 2025).**
  - At most lower secondary education: 59.3% of Portuguese aged 55–64 (highest in the EU) and 14.8% of those aged 25–34 (EU 13.5%).
  - In 2011, 49.0% of Portuguese managers had at most lower secondary education (EU 10.6%); in 2025, 16.2%.
  - In 2015 the bands aged 45–64 (born 1951–1970, most of whom started school before 1974) carried 67% of the gap to the EU27 in that share; in 2025, 83%.
  - If every cohort keeps its attainment as it ages and new 25–34s match today's, the gap to the EU27 in the 25–64 share falls from 15.9 points to 7.6 in 2035 and 3.4 in 2045 (equal weights for the four age bands).
- **Development accounting (Penn World Table 10.01, 2019).**
  - Output per worker is 62.9% of the reference. The log gap of −0.464 splits into capital +0.233, human capital −0.322 and productivity −0.375.
  - The human-capital term is 69% of the gap; with the reference's human capital Portugal would be at 86.8%. With half the PWT returns to schooling the term is 35% of the gap.
  - Portugal's human-capital index is the lowest of the 16 comparison countries.
  - PWT's index covers adults of all ages. With Barro-Lee schooling of the 25–64s instead (2015), the human-capital term is −0.190, 42% of the gap, against −0.335 (74%) with PWT's index in the same year.
- **Convergence 1995–2025 (25 members, Luxembourg and Ireland excluded).** Growth of relative GDP per head regressed on its 1995 level and 1995 schooling: 0.0375 per year of schooling (90% bootstrap interval 0.0084 to 0.0700). With the median member's schooling, Portugal's fitted 2025 level is 10.8 points higher (90% interval +2.5 to +19.1).

## Data

Two kinds of input, both pinned and checked by SHA-256, none committed here:

- **Eurostat.** A canonical snapshot in the data registry, [DiogoRibeiro7/data](https://github.com/DiogoRibeiro7/data), `datasets/eurostat-eu-convergence-education`: `nama_10_pc`, `edat_lfse_03`, `lfsa_egised` and `nama_10_lp_ulc`, retrieved on 25 November 2026. Contract: [`../data_sources/eurostat-eu-convergence-education.json`](../data_sources/eurostat-eu-convergence-education.json).
- **External references**, recorded in the registry and read from a fixed upstream commit:
  - Barro-Lee educational attainment, version 3 (1950–2015) and the long-run series of *Education Matters* (1870–2010): [`../data_sources/barro-lee-educational-attainment.json`](../data_sources/barro-lee-educational-attainment.json).
  - Penn World Table 10.01, from the CRAN package `pwt10`: [`../data_sources/penn-world-table-10-01.json`](../data_sources/penn-world-table-10-01.json).
  - Our World in Data's `owid-co2-data.csv`, for Maddison Project GDP and population: [`../data_sources/owid-co2-data.json`](../data_sources/owid-co2-data.json).

`fetch.py` downloads the eight files into `data/` and stops if any byte differs. `data/` and `results/` are rebuilt on every run and ignored by git.

## Reproduce

```
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v    # the functions against answers known by construction
python fetch.py                            # the eight files, from the pinned commits
python legacy_numbers.py > output.txt      # every number in the article; also writes results/
python figures.py                          # figures/
```

The workflow `.github/workflows/estado-novo-legacy.yml` runs all of this from a clean checkout and fails if `output.txt` changes.

## Method notes

- **Reference group.** Twelve Western European countries: Austria, Belgium, Denmark, Finland, France, Germany, Italy, the Netherlands, Norway, Sweden, Switzerland and the United Kingdom. Relative income is a country's GDP per head over the group's total GDP over total population, for years in which all twelve are present.
- **Catch-up regression.** For each country, gain = *a* + *b* · start + residual, with start the 1925 level and gain the change to 1975, ordinary least squares over 22 countries. Luxembourg is left out because its series starts at 0.3 years in 1925.
- **Human capital.** PWT's piecewise Mincer function of years of schooling *s*: returns of 0.134 a year for the first four years, 0.101 for the next four and 0.068 after eight; the index is exp(*φ*(*s*)).
- **Development accounting.** With output per worker *y*, capital–output ratio *K*/*Y* and human capital *h*, ln *y* = *α*/(1 − *α*) · ln(*K*/*Y*) + ln *h* + ln *A*, with *α* = 1/3. Each term is the log difference from the geometric mean of the reference countries; productivity *A* is the residual. Output and capital are PWT's output-side series at current PPPs (`cgdpo`, `cn`) over persons engaged.
- **Convergence regression.** ln(*y*₂₀₂₅/*y*₁₉₉₅) = *a* + *b* · ln *y*₁₉₉₅ + *c* · *s*₁₉₉₅, where *y* is GDP per head in PPS relative to the EU27 and *s* adult schooling in 1995 (Barro-Lee v3). Intervals from 5,000 pairs-bootstrap draws (seed 20261125).
- **Which human capital.** PWT's index is built from schooling of the adult population of all ages, including the over-65s, who in Portugal have very little schooling. The 2015 check replaces it with Barro-Lee v3 schooling of the 25–64s, mapped through the same Mincer function, for Portugal and each reference country.
- **What the numbers cannot settle.** The counterfactuals hold everything else fixed; they measure how large the schooling channel is, not that schooling alone would have closed the gap.

## Files

- **`fetch.py`**: download and verify the eight pinned files.
- **`jsonstat.py`**: read a JSON-stat 2.0 response into a table.
- **`legacy_core.py`**: relative income, compound growth, catch-up residuals, the PWT human-capital function, development accounting, the convergence regression and its bootstrap, the cohort projection.
- **`legacy_numbers.py`**: all numbers, printed in order, and the tables in `results/`.
- **`figures.py`**: the six figures.
- **`tests/test_core.py`**: 9 checks.

## Licence

Code: Apache-2.0. Data: Eurostat, reused under the Commission's reuse policy (Decision 2011/833/EU); Penn World Table 10.01 and the Maddison Project Database 2023, CC BY 4.0; Barro-Lee data, cited as the authors request.
