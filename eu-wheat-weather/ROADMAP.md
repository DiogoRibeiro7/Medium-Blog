# Roadmap

## Phase 0: foundation (this stage)

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

- [ ] Harmonise region codes to NUTS 2021 by boundary geometry (METHODS.md, section
      3.1).
- [ ] Aggregate E-OBS to NUTS-2 regions: cell weights, single-sine degree days, bin
      exposure, precipitation, and the water balance for the sensitivity analysis.
- [ ] Build the region-year panel and a coverage report. Still no estimate.

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
