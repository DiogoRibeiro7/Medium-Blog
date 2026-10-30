# Release-readiness checklist: v1.0.0

Status as of 30 October 2026. Each entry is **PASS** (done and verified), **BLOCKED** (cannot proceed until
something outside this repository changes) or **NOT YET DONE** (a step for the maintainer, which nothing here
does automatically).

**Verdict: not ready for a Zenodo deposit.** One data file blocks a public archive (A5). Everything else needed
for a defensible v1.0.0 is in place, apart from the release steps themselves (section E).

## A. Data and licensing

| # | Check | Status | Evidence |
|---|---|---|---|
| A1 | Every data file listed with provider, source URL, retrieval time, provider update, terms, attribution and redistribution status | PASS | `DATA_SOURCES.csv` (55 data files and 8 retrieval records), built by `data_sources.py`; `test_data_sources.py` |
| A2 | INE terms verified | PASS | INE Portal terms: CC BY 4.0, cite "Fonte: Instituto Nacional de Estatística, IP – Portugal" (`DATA_SOURCES.md`) |
| A3 | Eurostat terms verified | PASS | Eurostat copyright notice: reuse authorised with acknowledgement, dataset DOIs `10.2908/<DATASET>` |
| A4 | ECB terms verified | PASS | ECB statistics usage policy: free reuse with the source quoted, no modification |
| A5 | PORDATA transcription redistributable | **BLOCKED** | PORDATA legal terms grant no reuse licence. Routes: replace with INE data, obtain permission, or shorten the series to 2011–2025 (`DATA_SOURCES.md`) |
| A6 | Code and documentation licence kept separate from third-party data; no relicensing of data under Apache-2.0 | PASS | README "Citing and licences", `DATA_SOURCES.md`, `.zenodo.json` description; `test_release.py` |
| A7 | Archive cannot silently include a file that is not cleared | PASS | `release-archive` runs `python data_sources.py --check-redistribution` before `git archive` and stops on A5 |
| A8 | Decide whether the PORDATA file stays in the public GitHub repository | NOT YET DONE | Maintainer decision; the file has been public since an earlier pull request |

## B. External evidence

| # | Check | Status | Evidence |
|---|---|---|---|
| B1 | Banco de Portugal cohort ownership figures | PASS (with qualification) | Costa, Farinha, Lourenço and Mesquita (2023): households with reference person aged 25–34, censuses 1981–2021, about 70%, 60% and 40% (slide 12). Press-reported 61% and 42% replaced; the bulletin text itself was not retrieved |
| B2 | Vacant and secondary dwellings, 2021 | PASS | INE Censos 2021, indicator 0011495: 12.1% vacant, 18.5% secondary; OECD no longer cited |

## C. Reproducibility and figures

| # | Check | Status | Evidence |
|---|---|---|---|
| C1 | Full test suite | PASS | 121 tests pass on Python 3.10 and 3.13 (locally; CI result in the pull request) |
| C2 | Manifest validation | PASS | `sha256sum --check data/SHA256SUMS` (63 files) and `reproduce.py` step 1 |
| C3 | `reproduce.py` | PASS | Prints `REPRODUCED`; all result files and figures rebuild byte-identically in this environment |
| C4 | Generated results match committed values | PASS | Only intended changes against `main`: five external descriptives in `results.json` and the new `DATA_SOURCES.csv` |
| C5 | All 21 figures inspected visually | PASS | Four fixed (coverage colour bar, specifications legend, one overlapping label, one overlapping legend); no plotted value changed |
| C6 | PNG byte differences between environments explained | PASS | PNGs carry only a matplotlib-version text. Bytes can differ with matplotlib, FreeType or zlib builds. `reproduce.py` reports byte-only and pixel differences as warnings (`REVIEW.md`) |
| C7 | Optional: replace the secondary y-axes in `fig_integrated_national.png` with separate panels | NOT YET DONE | Design improvement, not a defect; values are correct |

## D. Publication metadata

| # | Check | Status | Evidence |
|---|---|---|---|
| D1 | Author name, ORCID and contact e-mail consistent | PASS | `CITATION.cff`, `.zenodo.json`, report header; `test_release.py` |
| D2 | Institutional affiliation | PASS | "Faculdade de Media Artes e Design, Universidade Técnica do Porto" (Portaria n.º 332-A/2026/1, 11 August 2026) |
| D3 | Version | PASS | 1.0.0 in `CITATION.cff`, `.zenodo.json`, report, release notes |
| D4 | Release date | NOT YET DONE | Removed until the tag exists; set `date-released` in `CITATION.cff` on the tag day |
| D5 | DOI | NOT YET DONE | None recorded or invented; added after the deposit |
| D6 | Citations and references | PASS | Report references updated (Costa et al. 2023; INE 0011495; Eurostat HICP manual and rent recommendations; INE rent-survey note) |
| D7 | Reproducibility instructions | PASS | README "Reproducing the release", report Appendix A, `reproduce.py` docstring |

## E. Release steps (nothing here is automated or done)

| # | Step | Status |
|---|---|---|
| E1 | Merge the readiness pull request | NOT YET DONE |
| E2 | Clear A5 (replace or license the PORDATA file), rebuild `DATA_SOURCES.csv`, rerun `reproduce.py` | **BLOCKED** |
| E3 | Set the release date in `CITATION.cff` | NOT YET DONE |
| E4 | Create the tag `portugal-housing-generations-v1.0.0` on the merge commit | NOT YET DONE |
| E5 | Build the archive and its checksum | NOT YET DONE |
| E6 | Zenodo deposit | NOT YET DONE |
| E7 | DOI update | NOT YET DONE |

### Manual steps, in order

1. **Tag (E4).** After E1–E3 are merged:
   ```bash
   git tag -a portugal-housing-generations-v1.0.0 -m "Portugal housing research release v1.0.0" <merge-sha>
   git push origin portugal-housing-generations-v1.0.0
   ```
   The tag can also be created in the GitHub UI.
2. **Archive and checksum (E5).** In GitHub Actions, run the `portugal-housing-generations` workflow on the tag
   with task `release-archive`.
   - The job reproduces the release and checks redistribution.
   - It builds `portugal-housing-generations-v1.0.0.zip` with `git archive` and writes its SHA-256 to a
     `.sha256` file.
   - It uploads both as a workflow artifact.

   Download them and run `sha256sum --check portugal-housing-generations-v1.0.0.zip.sha256`.
3. **Zenodo deposit (E6).** Create a new upload on Zenodo, not through the GitHub integration, which would archive
   the whole repository.
   - Attach the zip.
   - Enter the metadata from `.zenodo.json`: upload type publication/report, licence Apache-2.0 for code and
     documentation, creator with ORCID and affiliation, keywords, and the related identifier for the tag.
   - Paste the archive's SHA-256 in the notes.
   - Reserve the DOI, review, then publish.
4. **DOI update (E7).** In a follow-up pull request:
   - add `doi:` to `CITATION.cff`;
   - add the DOI to the README citation block and the report header;
   - optionally create a GitHub release for the tag with `RELEASE_NOTES_v1.0.0.md`.
