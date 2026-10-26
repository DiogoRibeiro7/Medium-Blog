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
  - **E5 placebo:** passes; no detectable link.
  - **E1, heat:** -2.6% per 10 degree days above 30 °C; not determined.
  - **E2, drought:** established negative by the registered rule, but the rule is
    flawed (see phase 3). Tested directly (post hoc), the dry-season effect is not
    significant, so the drought effect is not established.
  - **E3, change since 2012:** not determined. The minimum detectable change is
    0.061 log points per 10 degree days above 30 °C (80% power at 5%), or 0.071 at the
    Holm level.
- [x] M3, the shape of the temperature response. Only time at 36 °C and above has a
      two-way interval that excludes zero. The review found that this does not hold up:
      Spain supplies most of it, the two-way error is smaller than every one-way error,
      and much of that time is July heat after the southern harvest.
- [x] M2 and the out-of-sample check (E4).
  - 2018 is predicted well: -5.2% observed and predicted.
  - 2003 is under-predicted: -11.9% against -4.5%.
  - 2022 is over-predicted: -3.0% against -6.3%.
- [x] The nine sensitivity analyses. None reverses a sign.
- [x] An independent recomputation of M1 by explicit dummy variables, which agrees within the
      registered 1e-8.
- [x] Reproducibility: the whole estimation, bootstraps included, regenerates the
      committed results in the tests: text and integers exactly, floating-point values
      within a relative tolerance of 1e-7.

## Phase 3: publication

- [x] **Report.** [docs/RESULTS.md](docs/RESULTS.md) generates every number from the
      results.
- [x] **Article.** The Medium article is
      [publication/article.md](publication/article.md), generated from
      `publication/article.template.md` with every number filled in from the results, and
      it has four figures.
  - **Tests:**
    - the template contains no hand-typed result;
    - 28 claims in words are checked against the results;
    - the E-OBS acknowledgement, the citation, the dataset version, the repository link
      and the Eurostat and GISCO notices are present, the GISCO notice before the first
      section.
  - **Check.** `eu-wheat publication --check` compares the committed article and values
    with the code's output.
- [x] **Critical review** ([docs/REVIEW.md](docs/REVIEW.md)), after the results and
      before publication, with every finding marked verified or plausible.
- [x] **Corrections without new analysis** (the review's list A).
  - **Records.** A change-log entry records the review and the E2 rule's flaw.
    RESULTS.md has an "After review (post hoc)" section, beside the registered results.
  - **Article.** Drought is not established; the heat section is retitled, with time at
    36 °C and above given per hour and its caveats; the 2003 paragraph is rewritten.
  - **Figures.** Figure 1 is per hour, Figure 2 shows the peak's interval, Figure 3 shows
    every region, and Figure 4's heat title covers the 27 °C and 33 °C rows.
  - **Licences.** The GISCO non-commercial condition and notice, and the E-OBS version.
- [ ] **Registered amendments (optional, the review's list B).** A window that ends at
      harvest, a data-quality rule for series that are not regional measurements, and
      inference for few clusters with dependence across countries. Each would need an
      amendment before it is computed, and would be reported beside the registered
      results.
- [ ] **Author review and publication.** Nothing is published automatically. Under the
      E-OBS and GISCO terms, the article must stay non-commercial.

No artificial observations, no estimate before the plan, and no causal claim about
climate change from short-run weather responses.
