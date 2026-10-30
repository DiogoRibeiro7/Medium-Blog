# Release notes: v1.0.0

**Prices, incomes, construction and rents in Portugal's housing market, 2008–2025: a reproducible descriptive
assessment**

First research release, prepared on 30 October 2026 from data snapshots of 29 October 2026.

## What the release contains

- **Report.** [`REPORT.md`](REPORT.md) is a consolidated research report: research questions, institutional
  context, data, methods, results, statistical analysis, discussion, limitations, conclusions and references,
  with 14 figures.
- **Data.** 63 pinned source files from INE, Eurostat and the ECB (and one PORDATA transcription), with
  retrieval records and SHA-256 checksums in [`data/SHA256SUMS`](data/SHA256SUMS).
- **Code.** Five standard-library Python pipelines (national, regional, affordability, rental, integrated) and
  their helper modules, plus six figure scripts producing 21 figures.
- **Results.** Ten generated result files, committed and rebuilt byte for byte in CI.
- **Tests.**
  - unit tests with known answers and simulations;
  - snapshot-integrity tests;
  - pinned-result tests;
  - 132 report-claim checks that tie each quoted number to the results;
  - wording guards for the two statements corrected after an independent review.
- **Reproduction.** `python reproduce.py`.
- **Review log.** [`REVIEW.md`](REVIEW.md).
- **Metadata.** [`CITATION.cff`](CITATION.cff) and [`.zenodo.json`](.zenodo.json).

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

## Known limitations

These are listed in full in report Section 9:
- short panels;
- medians that move with composition;
- selection towards larger municipalities;
- methodological breaks that shorten every window;
- approximate inference with 26 clusters;
- two figures quoted from external publications (Banco de Portugal; OECD);
- preliminary construction data for 2021–2025.

## Before publishing (manual steps, nothing is automated)

1. Merge the release pull request after review.
2. Confirm the affiliation and release date in `CITATION.cff` and `.zenodo.json`.
3. Tag the merge commit on `main` as `portugal-housing-generations-v1.0.0`.
4. In GitHub Actions, run the `portugal-housing-generations` workflow on that tag with task `release-archive`.
   Download the artifact and check the zip against its `.sha256` file.
5. Create a new upload on Zenodo:
   - attach the zip;
   - fill in the metadata from `.zenodo.json` (upload type report, licence Apache-2.0, ORCID, keywords, related
     identifier);
   - reserve the DOI, then publish.
6. Add the DOI to `CITATION.cff`, the README citation block and the report header in a follow-up commit.
   Optionally create a GitHub release for the tag and paste these notes.

Zenodo's GitHub integration is not used, because it would archive the whole repository rather than this
project's folder.
