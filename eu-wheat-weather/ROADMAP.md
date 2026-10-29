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
      it has five figures.
  - **Tests:**
    - the template contains no hand-typed result;
    - 28 claims in words are checked against the registered results, and 10 against the
      amendment's;
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
- [x] **Amendment A1 registered** ([METHODS.md](METHODS.md), section 8), before any of
      it is computed: the review's list B.
  - A window that ends at harvest, from the GGCMI crop calendar.
  - A rule for yield series that are not regional measurements.
  - E2 tested directly.
  - A wild bootstrap of the two-way statistic, over countries and over years.
- [x] **Amendment A1 computed**, reported in [docs/AMENDMENT.md](docs/AMENDMENT.md),
      beside the registered results.
  - **Verdicts:** E1, E2 and E3 are all not determined; the amended placebo passes.
  - **The data rule:** it flags 249 region-years. It catches four of the review's five
    cases, and also Czechia's pro-rated years.
  - **Which change does what:** heat stays near -2.7% under every combination of the
    changes, and the harvest window widens its interval.
  - **The 36 °C band:** over the harvest window, it no longer differs from zero.
- [x] **Article:** reports Amendment A1 beside the registered results.
  - A new section gives its results, and Figure 5 shows each change and both together.
  - Its words are checked against the amendment's results by 10 tests.
  - It credits the crop calendar under CC BY 4.0.
- [ ] **Author review and publication.** Nothing is published automatically. Under the
      E-OBS and GISCO terms, the article must stay non-commercial.

## Phase 4: a second crop, grain maize

- [x] **Plan registered** ([METHODS.md](METHODS.md), section 9), before any maize
      yield was looked at.
  - **Data:** Eurostat `C1500`, 2010-2024, and the GGCMI maize calendar, so each
    region's season runs from planting to maturity.
  - **Estimands:** E1 (heat) and E2 (drought, tested directly), with Holm over the two.
  - **Placebo:** the 61 days after maturity.
  - **Descriptive:** M3 and E4 (2012, 2015, 2022).
  - **Sensitivity analyses:** five, fixed in advance.
- [x] Data, with no estimate ([docs/MAIZE_DATA.md](docs/MAIZE_DATA.md)).
  - **Sources:** the Eurostat snapshot (`snapshots/eurostat-maize-2026-10-26/`) and the
    maize calendar, frozen beside the wheat one.
  - **Seasons:** each region's season, planted 29 March to 12 May and mature 31 August
    to 7 October across the panel's regions.
  - **Panel:** 1,898 region-years in 132 regions of 19 countries, 2010-2024. The data
    rule flags 39 region-years: Danish, Austrian and Czech pro-rating, Italian values
    carried forward, and Dutch and Swedish sums.
- [x] Estimates and report ([docs/MAIZE.md](docs/MAIZE.md)), beside wheat.
  - **Heat (E1):** -4.9% per 10 degree days above 30 °C. Established by the registered
    rule, at the margin: Holm p = 0.048 two-way and 0.036 bootstrap.
  - **Drought (E2):** -3.3% in a dry season. Established: Holm p = 0.007 and 0.027.
  - **Placebo:** it passes by the registered rule, but the two-way test rejects
    (p = 0.043).
  - **Sensitivity analyses:** no sign reverses in any of the five.
  - **Out of sample:** the 2015 total is predicted well; 2012 and 2022 are
    under-predicted (-16.5% against -6.8%, and -21.7% against -11.8%).
- [x] Article: a separate maize article
      ([publication/maize/article.md](publication/maize/article.md)), with four figures.
  - **Tests:** every number is generated from the results, its claims in words are
    checked, and the notices and citations are tested, as for wheat.
  - **The wheat article** gets a short paragraph pointing to it.
- [x] **Critical review** ([docs/MAIZE_REVIEW.md](docs/MAIZE_REVIEW.md)), before
      publication.
  - **Drought** holds and is robust: under the full rule it fails only without Spain.
  - **Heat** holds by the rule, at the margin, and is not robust. It turns on the
    variance estimator, on Romania (about half of the estimate) and on Romania 2024, and
    it averages about zero in the irrigated south with about -11% elsewhere.
  - **The placebo's** two-way rejection is positive rain after maturity, mostly in
    France, unrelated to the season's weather.
- [x] **Corrections after the review** (its section A, no new analysis).
  - **Records:** an "After review (post hoc)" section in [docs/MAIZE.md](docs/MAIZE.md)
    and a change-log entry in [METHODS.md](METHODS.md).
  - **Article:** heat re-hedged, with the country deletions and Romania named. The
    swing, robustness, placebo, comparison and prediction passages are rewritten, and a
    "Not every country equally" bullet is added. Each new claim is tested.
  - **Figures:** Figure 4 relabelled and captioned, and each colour has one meaning.
- [x] **Amendment A2 registered** ([METHODS.md](METHODS.md), section 10), before any of
      it is computed: the review's section B.
  - **Weather over maize:** cells weighted by the maize on the MIRCA-OS map, with the
    cells below 800 m as a minimum check.
  - **Data rule:** repeated yields, breaks where two codes join, and figures that belong
    to another code (Q6-Q8).
  - **Irrigated and rainfed maize:** heat and drought for each group (descriptive).
  - **Few clusters:** one-way country clustering and a randomization test (descriptive).
  - **Decisions:** E1-A2 and E2-A2 by the registered rule, beside the registered results.
- [ ] Amendment A2 computed and reported ([docs/MAIZE_AMENDMENT.md](docs/MAIZE_AMENDMENT.md)).
- [ ] Article: Amendment A2 beside the registered results.
- [ ] **Author review and publication** of the maize article, under the same
      non-commercial terms.

No artificial observations, no estimate before the plan, and no causal claim about
climate change from short-run weather responses.
