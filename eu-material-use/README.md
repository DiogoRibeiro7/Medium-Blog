# Material use in the EU: decoupling, footprint and the recycling loop, 2000–2024

Companion code for the Medium article arguing that three common claims about European material use are each partly true: that the EU has decoupled material use from growth, that the decoupling is an illusion of where the extraction happens, and that recycling cannot close the loop. The article measures how much of each holds.

## What it finds

All numbers are from `output.txt`.

- **Decoupling (EU27, 2000–2024).**
  - Domestic material consumption (DMC) went from 6.61 to 5.99 billion tonnes (−9.4%) while GDP grew 39.3%.
  - Fossil energy materials account for the whole fall: they dropped from 1.62 to 1.00 Gt, and DMC without them was 4.99 Gt in both 2000 and 2024.
  - Non-metallic minerals rose to 4.08 Gt in 2007, fell to 2.80 Gt in 2013 and stood at 3.13 Gt in 2024.
- **Drift and growth.**
  - Year to year, EU material use moves 1.31 times as much as GDP, around a drift of −2.22% a year, so it stays flat at 1.70% growth.
  - Without fossil fuels it is flat at 1.38%, which is the EU's average growth over the period.
  - Across 27 member states with country intercepts, the response is 1.33 (s.e. 0.15), the mean drift is −2.74% a year, and use is flat at 2.06% growth.
- **Footprint.**
  - The raw material consumption (RMC) footprint fell 11.3 log points, against 9.9 for DMC; RMC/DMC went from 1.045 to 1.030.
  - For metal ores the footprint is 1.87 times DMC.
  - Across member states, 2008–2024, the footprint fell less than DMC in 8 of 27 countries.
  - Gross capital formation accounts for 46.1% of the EU footprint (2023), and construction products for 38.1%.
- **The loop.**
  - Of 7.70 Gt of processed material in 2024, 40.0% was not material use (exported or used for energy), and 2.92 Gt (37.9%) was added to stocks.
  - The circular material use rate rose from 10.7% (2010) to 12.2% (2024). Recycled tonnage grew 8.0 log points, while DMC fell.
- **Targets.**
  - Reaching 22.4% (double 2020) at 2024 material use needs 1.53 Gt of recycled material, 1.84 times the present amount. Alternatively, material use would have to fall 46% at present recycling.
  - Recycling every tonne of non-mineral waste would bring the rate to 16.0%, all construction and demolition waste as well to 16.7%, and soils and dredging spoils as well to 22.2%.
  - Only Malta ever added 10.2 points in six years.
- **Portugal.**
  - DMC fell from 242 Mt (2008) to 143 Mt (2013); it was 154 Mt in 2024.
  - The circular material use rate was 3.0% in 2024.

## Data

A canonical snapshot in the data registry, [DiogoRibeiro7/data](https://github.com/DiogoRibeiro7/data), `datasets/eurostat-material-use`. It holds ten Eurostat JSON-stat responses retrieved on 22 November 2026: `env_ac_mfa`, `env_ac_rme`, `env_ac_rmefd`, `env_ac_cur`, `env_ac_curm`, `env_ac_sd`, `env_wastrt`, `nama_10_gdp`, `nama_10_a10` and `nama_10_pe`. Eurostat's API serves only the current vintage, so the registry keeps the bytes.

The registry commit, URLs, sizes and SHA-256 are in [`../data_sources/eurostat-material-use.json`](../data_sources/eurostat-material-use.json). `fetch.py` downloads the files into `data/` and stops if any byte differs. No data are committed here: `data/` and `results/` are rebuilt on every run and ignored by git.

## Reproduce

```
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v    # the functions against answers known by construction
python fetch.py                            # the ten files, from the pinned registry commit
python materials_numbers.py > output.txt   # every number in the article; also writes results/
python figures.py                          # figures/
```

The workflow `.github/workflows/eu-material-use.yml` runs all of this from a clean checkout and fails if `output.txt` changes.

## Method notes

- **Material use** is DMC: domestic extraction plus imports minus exports, in tonnes as traded.
- **The footprint (RMC)** counts imports and exports in raw material equivalents, the materials extracted along the supply chain. It is Eurostat's model-based estimate, flagged as imputed.
- **Drift and growth.**
  - Yearly log changes of DMC are regressed on yearly log changes of GDP in volume.
  - For the member states, each country has its own intercept, the slope is common, and errors are clustered by country.
  - The intercept is the change at zero growth; the growth at which use is flat is minus the intercept over the slope.
- **Circularity.**
  - Eurostat's rate is recycled material *U* over DMC plus *U*.
  - The recycled tonnage is recovered from the published rate and DMC.
  - The ceilings add 2024's treated but unrecycled waste, category by category, and assume each recycled tonne replaces a tonne of primary material, so that material use stays at its 2024 level.
- **Processed material** splits into material use and the rest (exported or used for energy); material use splits into accumulation and the remainder, as in Eurostat's Sankey data.

## Files

- **`fetch.py`**: download and verify the registry snapshot.
- **`jsonstat.py`**: read a JSON-stat 2.0 response into a table.
- **`materials_core.py`**: log changes, intensity split, contributions, OLS and within-country regressions, circularity arithmetic.
- **`materials_numbers.py`**: all numbers, printed in order, and the tables in `results/`.
- **`figures.py`**: the five figures.
- **`tests/test_core.py`**: 9 checks.

## Licence

Code: Apache-2.0. Data: Eurostat, reused under the Commission's reuse policy (Decision 2011/833/EU), with acknowledgement of the source.
