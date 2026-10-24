# Roadmap

## Phase 0: foundation

- [x] Question, estimands and data sources, with their licences ([README.md](README.md)).
- [x] Research plan committed before any estimate ([METHODS.md](METHODS.md)), with a
      change log.
- [x] Eurostat crop statistics for common wheat at NUTS-2: download, JSON-stat decoding
      that keeps flags and gaps, a frozen snapshot with SHA-256 hashes, and an audit
      that re-derives the processed table.
- [x] GISCO NUTS boundaries (2006-2024) and E-OBS v33.0e: download with recorded
      hashes.
- [x] Offline tests and CI.

## Phase 1: data

- [x] Harmonise region codes to NUTS 2021.
  - **Rules.** A code joins a region if it is the region's own code, a NUTS-1 region
    with a single NUTS-2 region, or a chain of pure recodes in Eurostat's official
    change tables.
  - **Codes.** 238 are NUTS-2021 codes, 27 are NUTS-1 and 23 are recodes; 102 are
    dropped with their reason.
  - **Change log.** It records the NUTS-1 rule and the replacement of the geometric
    test by official recodes. Both were decided before any estimate.
- [x] Aggregate E-OBS to the 242 EU-27 NUTS-2021 regions, for 2000-2024 and 3 windows,
      with all cells and with cells below 800 m.
  - **Methods.** Degree days and temperature bins use single-sine interpolation per
    cell; potential evapotranspiration is Hargreaves.
  - **Checks.** The formulas are tested against numerical integration and FAO-56.
- [x] Build the region-year panel and a coverage report ([docs/DATA.md](docs/DATA.md)).
      Still no estimate.
  - 3,771 region-years in 167 regions of 25 countries.
  - Estonia, Latvia and Luxembourg have a single region each, so they contribute nothing
    to M1.

## Phase 2: estimates

Results are in [docs/RESULTS.md](docs/RESULTS.md). Every rule was fixed before any estimate
(change log, 2026-10-24).

- [x] M1 with both inference methods.
  - **E5 placebo:** passes.
  - **E1, heat:** -2.6% per 10 degree days above 30 °C; not determined.
  - **E2, drought:** established negative by the rule. The dry-season effect's own
    interval includes zero.
  - **E3, change since 2012:** not determined; the minimum detectable change is 0.061.
- [x] M3, the shape of the temperature response. Only days at 36 °C and above have an
      interval that excludes zero.
- [x] M2 and the out-of-sample check (E4).
  - 2018 is predicted well: -5.2% observed and predicted.
  - 2003 is under-predicted: -11.9% against -4.5%.
  - 2022 is over-predicted: -3.0% against -6.3%.
- [x] The nine sensitivity analyses. None reverses a sign.
- [x] An independent recomputation of M1 by explicit dummy variables; it agrees to 2e-13.
- [x] Reproducibility: the whole estimation, bootstraps included, regenerates the
      committed results exactly in the tests.

## Phase 3: publication

- [ ] A report with every number generated from the results, and a Medium article
      carrying the E-OBS acknowledgement.

No artificial observations, no estimate before the plan, and no causal claim about
climate change from short-run weather responses.
