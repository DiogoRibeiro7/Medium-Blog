# Why services get dearer: productivity, wages, margins and measurement in the EU, 1995–2025

Companion code for the Medium article arguing that the rising relative price of services has more than one cause. The article measures how much each cause contributes instead of naming one.

## What it finds

All numbers are from `output.txt`. A log point is 100 times a log change; log points add up, which is why the decomposition uses them.

- **What households pay (HICP, EU27, 2001–2025).**
  - Services +77.5%, non-energy industrial goods +28.4%, so services got 1.38 times dearer relative to manufactured goods.
  - Against all goods the gap is only 2.7 log points, because energy (+149%) and food (+113%) rose faster than services.
  - Services went from 37.8% to 43.2% of the index.
  - At the extremes: water supply +136%, home repair services +131% and restaurants +106%, against computers −85% and television sets −79%.
- **The identity.** The price of value added equals compensation per hour, divided by output per hour and by the labour share. In log points, a relative price is a wage gap plus a productivity contribution plus a margins contribution, exactly.
- **Services against manufacturing (EU27, 1995–2024).**
  - Relative price +22.9 log points (+25.8%).
  - Productivity contribution +45.8, wage gap −8.9, margins −13.9, so wages and margins offset half of what productivity alone implies.
  - Almost all of the rise came before 2010 (+21.1). Since then it was +1.8, as manufacturing productivity growth slowed from 3.02 to 1.69 log points a year.
- **Across industries.**
  - In 19 EU27 sections, the productivity term carries 0.79 of the cross-section variance of relative price changes (0.99 weighted by value added). Wages carry 0.05, margins 0.17.
  - In 450 country-industry cells (25 member states, 18 sections) the shares are 0.84, −0.00 and 0.17. The slope of price on the productivity contribution is 0.61 (s.e. 0.04).
- **Wages** vary across industries about half as much as productivity (standard deviation 30.0 against 57.1 log points) and lean slightly against it (slope −0.19).
- **Margins.** The adjusted labour share of goods fell 23.5 log points and that of services only 1.3, so margins worked against services' relative price.
- **Measurement.**
  - Education's measured output per hour fell 0.28 log points a year, and health and social work's rose 0.23.
  - If the output per hour of public administration, education and health had grown like market services' (1.03 a year), the services gap would be 16.5 log points instead of 22.9. Measurement can account for at most about a quarter of the gap.
- **Across countries (2024).**
  - The services price level ranges 3.90-fold across member states and the goods price level 1.55-fold.
  - Elasticity to GDP per head: services 1.00, goods 0.30.
  - The services price level follows services hourly pay with an elasticity of 0.67 (R² 0.90).
- **Portugal.**
  - HICP services against non-energy industrial goods ×1.74, 2001–2025.
  - Value added, services against manufacturing, +19.0 log points, with a wage gap of −21.6.
- **Forecasting.**
  - Across the 450 cells, relative price drift in 1995–2010 correlates 0.15 with drift in 2010–2024.
  - A forecast of no drift (RMSE 2.05 log points a year) beats carrying the old drift forward (2.90).

## Data

The data are a canonical snapshot in the data registry, [DiogoRibeiro7/data](https://github.com/DiogoRibeiro7/data), `datasets/eurostat-services-relative-prices`. It holds five Eurostat JSON-stat responses retrieved on 22 November 2026: `nama_10_a64`, `nama_10_a64_e`, `prc_hicp_aind`, `prc_hicp_inw` and `prc_ppp_ind`. Eurostat's API serves only the current vintage, so the registry keeps the bytes.

The registry commit, URLs, sizes and SHA-256 are in [`../data_sources/eurostat-services-relative-prices.json`](../data_sources/eurostat-services-relative-prices.json). `fetch.py` downloads the files into `data/` and stops if any byte differs. No data are committed here: `data/` and `results/` are rebuilt on every run and ignored by git.

## Reproduce

```
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v    # the functions against answers known by construction
python fetch.py                            # the five files, from the pinned registry commit
python prices_numbers.py > output.txt      # every number in the article; also writes results/
python figures.py                          # figures/
```

The workflow `.github/workflows/services-relative-prices.yml` runs all of this from a clean checkout and fails if `output.txt` changes.

## Method notes

- **The identity.** Write *w* for compensation per hour worked by employees, *a* for real value added per hour worked by everyone, and *s* for the labour share adjusted for the self-employed (*w* times all hours, over value added at current prices). The implicit price of value added is *P* = *w* / (*a* *s*). For two groups, the log price gap is the wage gap, minus the productivity gap, minus the labour-share gap. The script checks that the three add up to the price in every comparison.
- **Groups.** Services are NACE sections G to T without real estate (L), which is mostly imputed rent. Goods are A to F; the economy is A to T without L. Each group's volume is a Törnqvist index of chain-linked volumes, weighted by nominal shares.
- **Covariance shares.** For each term, cov(term, price) / var(price) across industries. The shares add up to one because the terms add up to the price.
- **Measurement bound.** The volumes of O, P and Q are replaced by their hours times the output-per-hour path of market services. Nominal value added, compensation and hours stay as published, so only the price moves.
- **Panel.** Estonia and Malta are left out because Eurostat has no hours or value added by section for 1995. Households as employers (T) is left out of the panel because several countries do not report it.

## Files

- **`fetch.py`**: download and verify the registry snapshot.
- **`jsonstat.py`**: read a JSON-stat 2.0 response into a table.
- **`prices_core.py`**: Törnqvist aggregation, the price identity, the productivity substitution, covariance shares, OLS with HC1 errors.
- **`prices_numbers.py`**: all numbers, printed in order, and the tables in `results/`.
- **`figures.py`**: the six figures.
- **`tests/test_core.py`**: 11 checks.

## Licence

Code: Apache-2.0. Data: Eurostat, reused under the Commission's reuse policy (Decision 2011/833/EU), with acknowledgement of the source.
