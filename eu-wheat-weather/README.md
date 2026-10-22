# Heat, drought and wheat yields in EU regions

A reproducible study of how common-wheat yields in the NUTS-2 regions of the European
Union respond to extreme heat and drought during the growing season, 2000-2024.

**Status: foundation.** This stage has the data pipeline and the research plan
([METHODS.md](METHODS.md)), committed before any yield was related to any weather
variable. No estimate exists yet. [ROADMAP.md](ROADMAP.md) lists the stages.

## Question and estimands

The question is how much a hot or dry growing season lowers wheat yields in a region,
and whether that has changed since 2000.

- **E1, extreme heat.** The change in log yield for 10 more degree days above 30 °C
  between 1 March and 31 July.
- **E2, drought.** The effect of a dry season, the region's 10th-percentile
  precipitation against its median.
- **E3, change.** The change in the heat effect between 2000-2011 and 2012-2024.
- **E4, prediction (descriptive).** How well the model predicts the 2003, 2018 and 2022
  harvests when each is left out.
- **E5, placebo.** Weather after harvest should not matter.

**Identification.** The primary model uses region and country-by-year effects, so it
compares regions of one country in the same year, each against its own normal. Weather
is close to random given those, so the coefficients are effects of weather shocks. They
are not effects of a permanent change in climate, to which farmers adapt.

## Data

| Source | What | Licence and attribution |
| --- | --- | --- |
| Eurostat `apro_cpshr` | Common wheat (`C1110`): area and harvested production at EU standard humidity, NUTS-2, annual | Free reuse with attribution: © European Union, Eurostat |
| E-OBS v33.0e (ECA&D, Copernicus) | Daily maximum and minimum temperature, precipitation and elevation, 0.25° grid, 1950-2025 | Non-commercial research and education only (below) |
| GISCO NUTS boundaries | NUTS-2 polygons of the 2006, 2010, 2013, 2016, 2021 and 2024 versions, 1:3 million | © EuroGeographics for the administrative boundaries |

**E-OBS terms.** "These observational data are strictly for use in non-commercial
research and non-commercial education projects only." This study and its articles are
not commercial. Any publication must carry:

> We acknowledge the E-OBS dataset from the EU-FP6 project UERRA (https://www.uerra.eu)
> and the Copernicus Climate Change Service, and the data providers in the ECA&D project
> (https://www.ecad.eu).
>
> Cornes, R., G. van der Schrier, E.J.M. van den Besselaar, and P.D. Jones. 2018: An
> Ensemble Version of the E-OBS Temperature and Precipitation Datasets, J. Geophys. Res.
> Atmos., 123. doi:10.1029/2017JD028200

**What is committed.**

- The Eurostat snapshot (`snapshots/eurostat-crops-<date>/`) is committed
  byte-identical, with SHA-256 hashes in its manifest.
- The E-OBS files (about 2.5 GB) and the boundary files (about 37 MB) are not committed.
  Their URLs and SHA-256 hashes are, and `eu-wheat audit` checks local copies against
  them.

## Run

Python 3.12 and Poetry 2.x, from `eu-wheat-weather/`. `poetry.toml` keeps the environment
project-local (`.venv`) and `poetry.lock` pins every dependency.

```bash
poetry install --extras dev
poetry run eu-wheat fetch-crops --output-dir snapshots/eurostat-crops-2026-10-22
poetry run eu-wheat fetch-boundaries --output-dir data/boundaries --manifest snapshots/gisco-nuts-2026-10-22.json
poetry run eu-wheat fetch-weather --output-dir data/eobs --manifest snapshots/eobs-v33.0e.json   # about 2.5 GB
poetry run eu-wheat audit --crops snapshots/eurostat-crops-2026-10-22 --weather-manifest snapshots/eobs-v33.0e.json --boundaries-manifest snapshots/gisco-nuts-2026-10-22.json
poetry run pytest
poetry run ruff check src tests
poetry run mypy src
```

If your shell sets `VIRTUAL_ENV` (for example to a conda environment), unset it first
(`env -u VIRTUAL_ENV poetry ...`).
