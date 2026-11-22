# The energy and the hours behind Europe's service economy, 1995–2024

Companion code for the Medium article arguing that services are physical in two ways: they use energy, and they use time, both the hours of the people who provide them and the hours of the people who consume them.

## What it finds

All numbers are from `output.txt`.

- **Shares (EU27).**
  - From 1995 to 2024, services (NACE G to U) went from 67.7% to 73.3% of value added at current prices.
  - Their share of hours worked went from 59.6% to 71.9%, and of final energy from 10.7% to 13.5%.
- **Services grew by adding inputs.**
  - Real value added of services ×1.74, hours ×1.39, final energy of commercial and public services ×1.24.
  - Industry (B to E) ×1.51, hours ×0.83, energy ×0.79.
  - Of the log growth of services output, 58% came from more hours and 39% from more energy. Services excluding real estate, 1995–2024.
- **Energy.**
  - Services' final energy rose 24.5% while total final energy fell 1.6%. It peaked in 2010.
  - Electricity's share of it rose from 39.6% to 52.0%, and services' electricity use rose 63.5%.
- **Hours.**
  - Real value added per hour grew 0.81% a year in services excluding real estate, against 2.41% in manufacturing.
  - It grew 0.26% in public administration, education and health, and 0.06% in arts, recreation and other services.
  - With flat hours, services output per hour would have had to grow 2.4 times faster.
- **Member states, 2000–2023.**
  - Services hours rose in 26 of 27 countries; only Latvia grew services output with fewer hours.
  - Nine countries grew it with less energy. None did both.
- **Consumers' time (HETUS 2000 and 2010, 30 country-waves).**
  - Free time varies 1.44-fold across countries while consumption per person varies 7.45-fold.
  - The elasticity of free time to consumption is 0.11, so consumption per hour of free time rises almost one for one with income (0.89).

## Data

A canonical snapshot in the data registry, [DiogoRibeiro7/data](https://github.com/DiogoRibeiro7/data), `datasets/eurostat-services-energy-time`. It holds seven Eurostat JSON-stat responses retrieved on 22 November 2026: `nama_10_a10`, `nama_10_a10_e`, `nrg_bal_c`, `tus_00age`, `nama_10_pc`, `nama_10_pe` and `prc_hicp_aind`. Eurostat's API serves only the current vintage, so the registry keeps the bytes.

The registry commit, URLs, sizes and SHA-256 are in [`../data_sources/eurostat-services-energy-time.json`](../data_sources/eurostat-services-energy-time.json). `fetch.py` downloads the files into `data/` and stops if any byte differs. No data are committed here: `data/` and `results/` are rebuilt on every run and ignored by git.

## Reproduce

```
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v    # the functions against answers known by construction
python fetch.py                            # the seven files, from the pinned registry commit
python services_numbers.py > output.txt    # every number in the article; also writes results/
python figures.py                          # figures/
```

The workflow `.github/workflows/services-energy-time.yml` runs all of this from a clean checkout and fails if `output.txt` changes.

## Method notes

- **Services aggregate.** Value added of services is a Törnqvist volume index over sections G-I, J, K, L, M_N, O-Q and R-U, weighted by nominal shares, because chain-linked volumes cannot be summed. Real estate (L) is mostly imputed rent with almost no hours, so the hours and productivity results use the aggregate without it.
- **Energy.** Final energy of services is the energy balance's "commercial and public services" (energy use). It excludes the fuel that service firms use for transport, which the balance counts under transport.
- **Free time.** HETUS "leisure, social and associative life" (AC4-8), in minutes per day for all persons, against actual individual consumption per person in PPS in the survey year.

## Files

- **`fetch.py`**: download and verify the registry snapshot.
- **`jsonstat.py`**: read a JSON-stat 2.0 response into a table.
- **`services_core.py`**: Törnqvist aggregation, growth decomposition, decoupling classes, OLS with HC1 errors.
- **`services_numbers.py`**: all numbers, printed in order, and the tables in `results/`.
- **`figures.py`**: the four figures.
- **`tests/test_core.py`**: 9 checks.

## Licence

Code: Apache-2.0. Data: Eurostat, reused under the Commission's reuse policy (Decision 2011/833/EU), with acknowledgement of the source.
