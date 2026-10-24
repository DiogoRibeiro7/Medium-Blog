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

- [ ] M1 with both inference methods: E1, E2 and E3 with Holm, and the E5 placebo.
- [ ] M3, the shape of the temperature response.
- [ ] M2 and the out-of-sample check of 2003, 2018 and 2022 (E4).
- [ ] The sensitivity analyses of METHODS.md, section 6.
- [ ] An independent recomputation of the main estimates by a second route.

## Phase 3: publication

- [ ] A report with every number generated from the results, and a Medium article
      carrying the E-OBS acknowledgement.

No artificial observations, no estimate before the plan, and no causal claim about
climate change from short-run weather responses.
