# Release notes: v1.0.0

**Prices, incomes, construction and rents in Portugal's housing market, 2008–2025: a reproducible descriptive
assessment**

First research release, prepared on 30 October 2026 from data snapshots of 29 October 2026.

## What the release contains

- **Report.** [`REPORT.md`](REPORT.md) is a consolidated research report: research questions, institutional
  context, data, methods, results, statistical analysis, discussion, limitations, conclusions and references,
  with 14 figures.
- **Data.** 55 pinned data files: 35 from INE, 18 from Eurostat, 1 from the ECB and 1 transcription from
  PORDATA. There are also 8 retrieval records, and all 63 files carry SHA-256 checksums in
  [`data/SHA256SUMS`](data/SHA256SUMS). [`DATA_SOURCES.md`](DATA_SOURCES.md) and
  [`DATA_SOURCES.csv`](DATA_SOURCES.csv) give each file's provider, source, retrieval time, terms, attribution
  and redistribution status.
- **Code.** Five standard-library Python pipelines (national, regional, affordability, rental, integrated) and
  their helper modules, plus six figure scripts producing 21 figures.
- **Results.** Ten generated result files, committed and rebuilt byte for byte in CI.
- **Tests.**
  - unit tests with known answers and simulations;
  - snapshot-integrity tests;
  - pinned-result tests;
  - 136 report-claim checks that tie each quoted number to the results;
  - data-inventory and licensing checks;
  - wording guards for the two statements corrected after an independent review.
- **Reproduction.** `python reproduce.py`.
- **Review log.** [`REVIEW.md`](REVIEW.md).
- **Metadata.** [`CITATION.cff`](CITATION.cff) and [`.zenodo.json`](.zenodo.json). No release date or DOI
  is recorded yet: both are set when the tag and the Zenodo deposit exist.
- **Release checklist.** [`RELEASE_CHECKLIST_v1.0.0.md`](RELEASE_CHECKLIST_v1.0.0.md).

## Licences

Original code and documentation are under the Apache License 2.0. Third-party data are **not** relicensed. INE
data are reused under CC BY 4.0, and Eurostat and ECB data under their providers' reuse terms, each with the
attribution listed in `DATA_SOURCES.md`.

Redistribution of the PORDATA transcription (`ine_pordata_completed_new_dwellings_PT_2026-10-29.csv`) has not
been established: PORDATA's terms grant no reuse licence. The archive task stops until the file is replaced with
INE data or permission is obtained. **This blocks the Zenodo deposit.**

## Main findings

Each finding is assigned to an evidence layer in the report.

- **Established.**
  - Since 2015, house prices have risen far faster than incomes and consumer prices: +164% nominal and about
    +111% real by 2025, against +65% for median income.
  - Mortgage entry costs relative to income peaked in 2023.
  - Regional sale prices outpaced declared income in all nine NUTS 2 regions in 2019–2024.
  - New-lease rents rose about 10% a year in 2020–2024, faster than income in 24 of 26 NUTS 3 regions.
  - The share of the population renting at market prices rose to 16.3%.
- **Descriptive associations.**
  - Prices and rents move together, the only within-municipality association that passes every check.
  - Prices rose more where population grew, not where income grew.
  - The positive association between prices and completions disappears for existing dwellings. This is
    consistent with the mix of dwellings sold, but the mechanism is not identified.
  - The HICP rent index rose more slowly than median income. It is a price index for a sample of rented
    dwellings (new and existing contracts), not a measure of what any fixed group of tenants pays.
- **Unresolved.** No causal claim about tourism, foreign demand, immigration, interest rates, regulation or
  supply is supported. Report Section 8.3 lists the data a causal design would need.

## Changes in this release

Relative to the merged stages (#354, #368, #371, #374, #376):

**Added:**
- the research report;
- the national overview figure;
- a Medium article (`article.md`) and two figures drawn for it;
- one-command reproduction and the data checksum manifest;
- citation and archive metadata, the review log and these notes.

**Statistics.** The wild cluster bootstrap now also reports restricted (WCR) p-values, and the verdict rule
uses them. No verdict changed.

**Corrected:**
- *Rent bands.* The census share of tenancies with rents below €100 a month included the €100–149.99 band.
  Corrected values: 49.3% (was 67.8%) for pre-1987 contracts and 16.0% (was 22.4%) for all tenancies. The
  README and one figure were updated.
- *Rent methods.* The README range for the difference between rent methods is now 3.2–6.3% (was "3–6%").

**Wording.** Three README passages now state associations and accounting identities without causal language.

**Corrected after an independent scientific review** (no estimate changed):
- *HICP actual rentals.* The release had described the index as tracking "sitting tenants". Its target coverage is
  new and existing contracts for private, public and social rental units (HICP Methodological Manual 2024,
  section 12.4), and it follows a panel of dwellings, not households. The report now explains what the index
  measures, and how it differs from new-lease medians in population, weighting, level versus index and coverage.
  The inference that incumbent tenants' rent burden fell was removed.
- *Completions.* The association was described as "a sale-mix artefact". It is now described as consistent with
  sale-mix composition but not identified, with the alternative explanations and the data needed to separate
  them.
- *Figures.* Labels and titles in five figures were updated to match; no plotted value changed.

**CI:**
- tests on Python 3.10 and 3.13;
- data checksum verification;
- the report figure;
- a manual `release-archive` task that builds the archive without publishing it.

**Release-readiness review (30 October 2026):**
- *Data licensing.* Added a file-by-file inventory with provider terms checked on the providers' own pages. The
  PORDATA file is flagged and blocked from the archive.
- *External evidence.*
  - The cohort ownership figures now follow the authors' own rounded values (about 70%, 60% and 40%), in place
    of the press-reported 61% and 42%.
  - The vacancy and secondary-home shares now come from INE's 2021 census counts (12.1% and 18.5%), in place of
    the OECD's 12% and 19%.
  - `results.json` records the corrected values.
- *Affiliation.* Changed to Faculdade de Media Artes e Design, Universidade Técnica do Porto (Portaria n.º
  332-A/2026/1). The planned release date was removed from the metadata until the tag exists.
- *Figures.* All 21 figures were inspected, and four were fixed for legend, label and colour-bar defects:
  - `fig_integrated_coverage`: the colour bar is now labelled in %;
  - `fig_integrated_specifications`: the legend no longer covers intervals;
  - `fig_regional_growth_and_construction`: a label no longer covers a point;
  - `fig_regional_permits_and_completions`: the legend no longer covers a line.

  No plotted value changed. `reproduce.py` now says whether a figure whose bytes differ also differs in pixels.

## Known limitations

These are listed in full in report Section 9:
- short panels;
- medians that move with composition;
- selection towards larger municipalities;
- methodological breaks that shorten every window;
- approximate inference with 26 clusters;
- two sets of figures quoted, not recomputed (Banco de Portugal economists' cohort shares; INE census occupancy
  counts);
- preliminary construction data for 2021–2025.

## Before publishing

See [`RELEASE_CHECKLIST_v1.0.0.md`](RELEASE_CHECKLIST_v1.0.0.md) for the status of every check and the manual
steps for the tag, archive checksum, Zenodo deposit and DOI update. Nothing is published automatically.
