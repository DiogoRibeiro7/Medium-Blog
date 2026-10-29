# Heat, drought and wheat yields in EU regions

A reproducible study of how common-wheat yields in the NUTS-2 regions of the European
Union respond to extreme heat and drought during the growing season, 2000-2024.

**Status: article drafted and corrected after a critical review (phase 3). Amendment A1
([METHODS.md](METHODS.md), section 8) is registered and computed
([docs/AMENDMENT.md](docs/AMENDMENT.md)). A second crop, grain maize, is registered
([METHODS.md](METHODS.md), section 9) and estimated ([docs/MAIZE.md](docs/MAIZE.md)), and its
Amendment A2 (section 10) is computed ([docs/MAIZE_AMENDMENT.md](docs/MAIZE_AMENDMENT.md)).**

- **Plan.** [METHODS.md](METHODS.md) was committed before any yield was related to any
  weather variable. Its change log records the decisions made before any estimate, and
  the review's findings after the results.
- **Panel.** [docs/DATA.md](docs/DATA.md) describes it.
- **Results.** They are in [docs/RESULTS.md](docs/RESULTS.md):
  - **Placebo (E5):** passes. There is no detectable link between post-harvest weather
    and yields.
  - **Heat (E1):** 10 more degree days above 30 °C go with 2.6% lower yields. This is
    not determined after Holm: two-way p = 0.26, wild bootstrap p = 0.52. Without Spain
    the estimate is -1.6%.
  - **Drought (E2):** "established negative" by the registered rule, but the rule is
    flawed. It takes its significance from the joint test of the precipitation terms,
    which rejects mostly through the curvature and the wet end. Tested directly (post
    hoc), the dry-season effect (-2.3%) gives two-way p = 0.13 and wild bootstrap
    p = 0.16, so **the drought effect is not established**. Yields peak at about
    350 mm over March to July, with a 95% interval of 274-419 mm.
  - **Change since 2012 (E3):** not determined. The design can only detect a change of
    0.061 log points (about 6%) per 10 degree days above 30 °C, with 80% power at 5%,
    or 0.071 at the Holm level.
- **Amendment A1** ([docs/AMENDMENT.md](docs/AMENDMENT.md)). It was registered after the
  review and before it was computed, and is reported beside the registered results.
  - **What changes:** a window that ends at each region's wheat maturity date, a rule for
    yield series that are not regional measurements, E2 tested directly, and a
    bootstrap of the two-way statistic by country and by year.
  - **Verdicts:** every one is not determined.
    - Heat: -2.7% per 10 degree days, two-way p = 0.37.
    - Drought: -3.2%, p = 0.10.
  - **Placebo:** it passes.
  - **The 36 °C band:** it no longer differs from zero once heat after the harvest is
    left out.
- **Grain maize** ([docs/MAIZE.md](docs/MAIZE.md)). It was registered before any maize
  yield was looked at, and has the same design and inference as wheat under Amendment A1.
  - **Heat:** 10 more degree days above 30 °C go with 4.9% lower yields. This is
    established by the registered rule, but at the margin (p = 0.048 two-way and 0.036
    bootstrap, after Holm), and it is not robust. Leaving out any one of 9 of the 19
    countries, or Romania in 2024 alone, makes it not determined, and Romania supplies
    about half of the estimate (post hoc).
  - **Drought:** a dry season goes with 3.3% lower yields. This is established (p = 0.007
    and 0.027 after Holm) and far more robust: under the full rule it fails only without
    Spain (post hoc).
  - **Placebo:** it passes by the registered rule: the bootstrap does not reject
    (p = 0.158). The two-way rejection (p = 0.043) comes from rain after maturity, with a
    positive sign, mostly in France (post hoc).
  - **Amendment A2** ([docs/MAIZE_AMENDMENT.md](docs/MAIZE_AMENDMENT.md)), registered after
    the maize review. With weather over where maize grows and a wider data rule, heat
    (-4.4%) and drought (-3.5%) are both still established, beside the registered results.
    Heat's p stays below 0.05 with one-way country clustering (0.029) and with a
    randomization test (0.040). Heat costs rainfed regions more than irrigated ones
    (descriptive).
- **Review.** [docs/REVIEW.md](docs/REVIEW.md) is a critical review written after the
  results and before publication. Its checks are in the "After review (post hoc)"
  section of [docs/RESULTS.md](docs/RESULTS.md), beside the registered results and
  never in place of them. Maize has its own review
  ([docs/MAIZE_REVIEW.md](docs/MAIZE_REVIEW.md)), with its checks in the same section of
  [docs/MAIZE.md](docs/MAIZE.md).
- **Article.** The Medium article ([publication/article.md](publication/article.md))
  and its five figures are generated from the committed results. It carries the E-OBS
  acknowledgement and citation, and the Eurostat and GISCO notices under its subtitle.
  It is not published: the author reviews and publishes it. A separate maize article
  ([publication/maize/article.md](publication/maize/article.md)) is generated the same
  way.
- [ROADMAP.md](ROADMAP.md) lists the stages.

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
| Eurostat `apro_cpshr` | Common wheat (`C1110`) and, for section 9, grain maize (`C1500`): area and harvested production at EU standard humidity, NUTS-2, annual | Free reuse with attribution: © European Union, Eurostat |
| E-OBS v33.0e (ECA&D, Copernicus) | Daily maximum and minimum temperature, precipitation and elevation, 0.25° grid, 1950-2025 | Non-commercial research and education only (below) |
| GISCO NUTS boundaries | NUTS-2 polygons of the 2006, 2010, 2013, 2016, 2021 and 2024 versions, 1:3 million | Non-commercial use only, with the notice © EuroGeographics for the administrative boundaries (below) |
| MIRCA-OS v0.1 (Amendment A2) | Harvested area of irrigated and rainfed maize in 2015, 5-arcminute grid (Kebede et al. 2025, Scientific Data 12, 208; data doi:10.4211/hs.60a890eb841c460192c03bb590687145) | CC BY 4.0, with attribution; summed onto the E-OBS grid and used as weights |
| Eurostat `ef_lus_irri`, `ef_lus_allcrops` (Amendment A2) | Irrigated and total grain maize area, 2023, Spain, Portugal, Italy and Greece, for a check of the map | Free reuse with attribution: © European Union, Eurostat |
| GGCMI Phase 3 crop calendar v1.01 (Amendment A1 and section 9) | Rainfed winter wheat maturity day, and rainfed maize planting and maturity days, 0.5° grid (Jägermeyr et al. 2021, Nature Food 2, 873-885; doi:10.5281/zenodo.5062513) | CC BY 4.0, with attribution; mapped to the E-OBS grid, averaged per region and rounded to whole days |

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

**GISCO terms.** The boundaries may be used "on condition that the data will not be
used for commercial purposes", and the copyright notice "© EuroGeographics for the
administrative boundaries" must appear on the publication's introductory page. The
article carries it under its subtitle.

**What is committed.**

- The Eurostat crop snapshots (`snapshots/eurostat-crops-<date>/` for wheat,
  `snapshots/eurostat-maize-2026-10-26/` for maize), Eurostat's official NUTS change
  tables (`snapshots/nuts-changes-2026-10-23/`) and the crop calendar
  (`snapshots/ggcmi-crop-calendar-v1.01/`, 240 kB, wheat and maize) are committed
  byte-identical, with SHA-256 hashes in their manifests.
- The derived tables are committed in `docs/panel/`: the region map, the regional
  weather, each region's centroid latitude and maturity date, and the panel. The panel
  regenerates in the offline tests.
- Amendment A2's Eurostat irrigation figures (`snapshots/eurostat-irrigation-2026-10-29/`)
  are committed. The MIRCA-OS archive (172 MB) and its two maize layers stay in
  `data/mirca/`, with their hashes in `snapshots/mirca-os-v0.1.json`.
- The E-OBS files (about 2.5 GB) and the boundary files (about 37 MB) are not committed.
  Their URLs and SHA-256 hashes are, and `eu-wheat audit` checks local copies against
  them.

## Run

Python 3.12 and Poetry 2.x, from `eu-wheat-weather/`. `poetry.toml` keeps the environment
project-local (`.venv`) and `poetry.lock` pins every dependency.

```bash
poetry install --extras dev
poetry run eu-wheat fetch-crops --output-dir snapshots/eurostat-crops-2026-10-23
poetry run eu-wheat fetch-boundaries --output-dir data/boundaries --manifest snapshots/gisco-nuts-2026-10-22.json
poetry run eu-wheat fetch-weather --output-dir data/eobs --manifest snapshots/eobs-v33.0e.json   # about 2.5 GB
poetry run eu-wheat fetch-nuts-changes --output-dir snapshots/nuts-changes-2026-10-23
poetry run eu-wheat fetch-calendar --output-dir snapshots/ggcmi-crop-calendar-v1.01
poetry run eu-wheat audit --crops snapshots/eurostat-crops-2026-10-23 --weather-manifest snapshots/eobs-v33.0e.json --boundaries-manifest snapshots/gisco-nuts-2026-10-22.json
poetry run eu-wheat regions --crops snapshots/eurostat-crops-2026-10-23 --nuts-changes snapshots/nuts-changes-2026-10-23 --boundaries-dir data/boundaries --output docs/panel/region_map.csv
poetry run eu-wheat weather --eobs-dir data/eobs --boundaries-dir data/boundaries --output docs/panel/weather.csv --latitudes docs/panel/region_latitude.csv --calendar snapshots/ggcmi-crop-calendar-v1.01/wwh_rf_ggcmi_crop_calendar_phase3_v1.01.nc4 --maturity docs/panel/maturity.csv
poetry run eu-wheat panel --crops snapshots/eurostat-crops-2026-10-23 --region-map docs/panel/region_map.csv --weather docs/panel/weather.csv --output-dir docs/panel --report docs/DATA.md
poetry run eu-wheat estimate --panel docs/panel/panel.csv --weather docs/panel/weather.csv --latitudes docs/panel/region_latitude.csv --output-dir docs/results --report docs/RESULTS.md
poetry run eu-wheat amend --crops snapshots/eurostat-crops-2026-10-23   # Amendment A1, beside the registered results
# Grain maize (METHODS.md, section 9)
poetry run eu-wheat fetch-crops --crop maize --output-dir snapshots/eurostat-maize-2026-10-26
poetry run eu-wheat regions --crops snapshots/eurostat-maize-2026-10-26 --nuts-changes snapshots/nuts-changes-2026-10-23 --boundaries-dir data/boundaries --output docs/maize/region_map.csv
poetry run eu-wheat maize-weather --eobs-dir data/eobs --boundaries-dir data/boundaries --output docs/maize/weather.csv --seasons docs/maize/season.csv   # about an hour
poetry run eu-wheat maize-panel --crops snapshots/eurostat-maize-2026-10-26
poetry run eu-wheat maize-estimate   # beside wheat under Amendment A1
# Maize, Amendment A2 (METHODS.md, section 10)
poetry run eu-wheat fetch-mirca        # the MIRCA-OS map of maize, 172 MB, into data/mirca (needs bsdtar)
poetry run eu-wheat fetch-irrigation   # Eurostat's irrigated maize in 2023, for the map check
poetry run eu-wheat maize-a2-weather   # the season weather under three weightings, about 2 minutes
poetry run eu-wheat maize-amend --crops snapshots/eurostat-maize-2026-10-26   # beside the registered results
poetry run eu-wheat publication            # values, article and figures; --check compares
poetry run pytest
poetry run ruff check src tests
poetry run mypy src
```

If your shell sets `VIRTUAL_ENV` (for example to a conda environment), unset it first
(`env -u VIRTUAL_ENV poetry ...`).
