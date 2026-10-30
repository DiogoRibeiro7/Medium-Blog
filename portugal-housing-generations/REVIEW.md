# Scientific and technical review for v1.0.0

This review was carried out on 30 October 2026 before the first research release. It covered the five merged
stages of the project:
- national series and the mortgage-cost index (#354);
- the regional panel (#368);
- regional affordability (#371);
- the rental market (#374);
- the integrated assessment (#376).

It is an internal review, not an independent peer review. Each section below states what was checked, what was
found and what changed. Open items are listed at the end.

## 1. Code, data, methods and results

**Checked:**
- every analysis script, helper module and estimator;
- every pinned data file, and the code path that reads it.

**Data.** All 63 files in `data/` are read by at least one script, and none is redundant. One snapshot,
`eurostat_ilc_lvho02_PT_2026-10-29.json`, duplicates the total slice of `eurostat_ilc_lvho02_PT_by_poverty_*`.
It is kept because `analysis.py` reads it, and the duplicate is harmless.

**Estimators.**
- `panel_stats.py` was re-derived against textbook formulas: OLS by normal equations, CR1 with the
  G/(G−1)·(N−1)/(N−K) correction, the two-way within transform, the Wooldridge first-difference test, and
  Fisher and Bartlett intervals.
- The unit tests compare these with hand-computed values and with simulations with a known answer.

**Change: restricted wild bootstrap added.** The wild cluster bootstrap previously reported only the
unrestricted percentile-t interval (WCU). With 26 clusters, the restricted bootstrap (WCR), which imposes the
null when resampling, gives better-sized tests (Cameron, Gelbach and Miller, 2008; MacKinnon and Webb, 2017).
- `wild_cluster_bootstrap` now also returns `p_value_restricted`.
- The verdict rule requires both the WCU interval to exclude zero and the WCR p-value to be below 0.05.
- Tests check the WCR p-value under a true effect, under the null, and with two regressors.
- No verdict changed. Across the 32 tests the two p-values differ by at most 0.024, and none falls on the other
  side of 0.05.

**Documented:** the influence diagnostic ranks municipalities by the DFBETA of the first regressor only.

## 2. Reproduction of numerical claims

- **One command.** `reproduce.py` checks the data checksums, runs the tests, re-runs the five analyses and
  compares every output byte for byte. All ten result files rebuild identically.
- **Report claims.** `test_report.py` holds 128 numerical claims quoted in `REPORT.md`. For each, the test
  checks:
  - that the sentence is in the report;
  - that the number in the sentence equals the value read or computed from the result files, at the precision
    quoted.
- **README.** The national sections of the README were checked by hand against `results.json`; they are covered
  by the existing pinned-result tests.

## 3. References, attribution, geography and time coverage

**Bibliographic details checked against the publishers or repositories:**
- Franco and Santos (2021), *Regional Science and Urban Economics* 88;
- Lourenço and Rodrigues (2017), *Banco de Portugal Economic Studies*;
- MacKinnon and Webb (2017), *Journal of Applied Econometrics* 32(2);
- Garcia-López et al. (2020), *Journal of Urban Economics* 119;
- Lei 19/2022, Lei 56/2023 and Lei 82/2023.

**Cited from standard references and not re-checked:** the remaining methodological references and laws.

**Not reproducible from the pinned data:** two figures are quoted from publications.
- Banco de Portugal (2023): cohort ownership shares of 61% and 42%.
- OECD (2026): 12% of dwellings vacant and 19% secondary in 2021.

Both are labelled as external in the report.

**Geography.**
- Municipalities link every NUTS version through their DICO codes.
- Sums regroup exactly to each source's own regional and national totals.
- Medians are never aggregated.
- The NUTS 2013 to NUTS 2024 changes (Lisbon split; Médio Tejo and Beira Baixa) are documented.

**Time alignment.**
- EU-SILC income is assigned to the year before the survey, and both alignments are reported.
- Eurostat's 1 January population is assigned to 31 December of the year before.
- Breaks are never crossed: population 2020/2021, price methods 2018/2022, rent methods 2021/2026.

**Cross-checks recorded in the results:**
- The transcribed PORDATA completions series equals INE's municipal series summed to Portugal in every year
  2011–2025.
- INE's administrative-based population exceeds Eurostat's by 1.7–5.9% in 2021–2024. This is documented as
  uncertainty about recent growth.
- The 2025 overburden total is 6.3% in `ilc_lvho07a` and 6.2% in `ilc_lvho07c`. Each table is used only for its
  own breakdown.
- The early Eurostat snapshots carry dataset update timestamps but no per-request retrieval record. From this
  release they are covered by `data/SHA256SUMS`.

## 4. Causal language, ecological inference, selection and comparisons

The README and report were read for any wording that implies causation, describes individuals from area
averages, or compares measures of different populations.

**Wording changed:**
- Population and construction "respond to" employment and migration → "can respond to".
- Tenure composition "raises" overburden → "arithmetically raises", with a note that the shift-share
  decomposition is an accounting identity.
- Prices and rents "respond to" local conditions → "are likely to respond to".
- The completions association passes the mechanical "robust" rule, but it is now labelled in the README and
  report as a sale-mix artefact.

**Already handled correctly:**
- Ecological inference: municipal correlations are never read as household behaviour.
- Selection: published medians favour larger municipalities, and balanced samples drop gaps.
- Comparisons that are not like-for-like: report Section 8.4 lists the comparisons the release avoids.

## 5. Statistical assumptions, uncertainty and sensitivity

**Reviewed:**
- clustering;
- the wild bootstrap with 26 clusters;
- serial correlation (fixed effects are preferred for sale prices, first differences for appraisals);
- within-variation shares (attenuation from measurement error in medians);
- leave-one-region-out, metropolitan and influence checks;
- the cross-specification sign rule;
- effective sample sizes for national series.

**Gaps that remain** (in Limitations, not fixed):
- spatial correlation between NUTS 3 regions is not modelled;
- 26 clusters make inference approximate;
- the national series are too short for dynamic models.

## 6. Consistency between figures, tables, documentation and results

**Errors found and fixed:**
1. **Rent bands.** The share of tenancies "below €100 a month" in `rental.py` included the €100–149.99 band,
   because bands were selected by position. It now selects bands by their published bounds, and a new test
   covers this.
   - Corrected values: pre-1987 contracts 49.3% (previously reported as 67.8%); all tenancies 16.0% (22.4%).
   - The share of €500 or more and the median bands were already correct.
   - The README sentence and the census-contracts figure were updated.
2. **Rent methods.** The README said the 2026 rent method is "3–6%" above the 2021 method for Portugal; the
   computed range is 3.2–6.3%.
3. **Report drafting errors** caught while checking each quoted number against the results (these checks now
   run as tests):
   - the regions with the largest rent–income gaps (Alentejo Litoral, Madeira and Península de Setúbal, not
     Lisbon and Porto);
   - the NUTS 2 regions with the widest price–income gaps (Madeira is second);
   - the share of non-resident tenants (3–5%, not "about 3%");
   - the agreement of the two bootstrap p-values (within 0.024, not 0.02).

**Figures.** Every figure script was re-run, and each of the 21 figures is referenced in the README or the
report.

## 7–9. Report, figures and reproducibility instructions

**New files:**
- `REPORT.md`: about 7,400 words with Abstract, Introduction, Research questions, Literature and institutional
  context, Data, Methodology, Results, Statistical analysis, Discussion, Limitations, Conclusions, References
  and two appendices;
- `report_figures.py`: the national overview figure (`fig_report_national_overview.png`). An earlier version
  plotted the mortgage rate on a second y-axis; it now has its own panel.
- `article.md` and `article_figures.py`: a Medium article based on the release, with two figures of its own
  (the payment decomposition and the completions composition check). Its numbers come from the same results.

**Reproduction instructions** are in the README, in the report's Appendix A and in the `reproduce.py`
docstring.

## 10. Metadata and archival

**New files:** `CITATION.cff` (CFF 1.2.0) and `.zenodo.json`. Their title, version, licence, date and ORCID
are checked against each other by `test_release.py`.

**Archive.** Zenodo's GitHub integration archives a whole repository, and this project lives in a monorepo.
The release is therefore prepared for a manual upload:
- the workflow's `release-archive` task reproduces the release;
- it builds a zip of the committed folder with `git archive`;
- it uploads the zip and its SHA-256 as an artifact.

Nothing is published automatically.

## 11. Tests and CI

**Tests:**
- `test_report.py` checks the report's claims;
- `test_release.py` checks the manifest, metadata and release documents;
- `test_rental.py` gained a rent-band test;
- `test_panel_stats.py` gained the WCR tests.

**CI changes:**
- The test job now runs on Python 3.10 and 3.13, because the README promises 3.10.
- It verifies `data/SHA256SUMS` with `sha256sum --check`.
- It also runs `report_figures.py`.

The full suite (107 tests) and `mypy` (default settings) pass locally on Python 3.10 and 3.13, and results
rebuilt on 3.10 are byte-identical to those built on 3.13.

## 12. Artefacts removed

**From the repository:** nothing. Every tracked data file, result file, figure and script is used and
referenced.

**Locally:** two untracked caches left over from development (`discovery/`, which held only bytecode, and
`__pycache__/`) were deleted. They were never committed.

## Open items before publishing

1. **Affiliation.** Confirm the affiliation in `CITATION.cff` and `.zenodo.json`. It is currently "ESMAD –
   Escola Superior de Media Artes e Design, Politécnico do Porto".
2. **Release date.** Confirm `date-released` and `publication_date` (2026-10-30) or update them on the day of
   the tag.
3. **External figures.** Decide whether to keep the two figures that cannot be recomputed from the pinned data
   (Banco de Portugal cohort shares; OECD vacancy shares), or re-verify them against the publications.
4. **Data terms.** Check INE's and PORDATA's current terms of use before depositing the data on Zenodo.
5. **DOI.** After the Zenodo deposit, add the DOI to `CITATION.cff`, the README and the report.
