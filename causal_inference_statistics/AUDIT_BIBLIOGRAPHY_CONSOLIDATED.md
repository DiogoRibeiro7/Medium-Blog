# Causal inference bibliography — consolidated inventory and verification status

Audit date: 2026-10-28.

## Coverage

The reference sections of all twenty articles were inspected to establish the remaining review scope. Entries include journal papers, books, conference proceedings, working papers and repeated sources. A missing DOI is **not** equivalent to incorrect metadata: historical books, conference proceedings and author-hosted editions often have alternative permanent identifiers.

Earlier batches 1–9 contain source-specific verification logs and corrections. Those checks do not establish that all unique bibliographic entries have been individually authenticated.

## New verified correction

Part 10: the Newey and Robins entry used “J. R.” and identified only the 2018 arXiv version. The reference is now recorded as **Newey, W. K. and Robins, J. M. (2017). Cross-Fitting and Fast Remainder Rates for Semiparametric Estimation. CeMMAP Working Paper CWP41/17**, DOI **10.1920/wp.cem.2017.4117**; the revised arXiv preprint 1801.09138 is retained. Working-paper dates and revised-preprint dates should not be conflated.

## Priority queue for further metadata verification

- Books and editions without a DOI: Fisher (1935), Kish (1965), Bernardo and Smith (1994), Pearl (2009), Gelman et al. (2013), Carpenter and Kenward (2013), Little and Rubin (2019).
- Proceedings without standard DOIs: Dudík et al. (2011), Kallus (2018), Shpitser et al. (2010), and historical edited-book contributions.
- Journal citations without DOIs: Nelson and Startz (1990), Rosenbaum and Rubin (1983) sensitivity analysis, Cornfield et al. (1959), Zivich et al. (2022).
- Repeat citations should be normalized once the underlying edition and publisher metadata are established; use a stable identifier, not a fabricated DOI.

## Acceptance conditions before publication

1. Build the inventory with `python causal_inference_statistics/audit_references.py`; manually or programmatically compare to authoritative bibliographic records.
2. De-duplicate references by normalized title, year and author set while preserving article-level occurrence.
3. Verify authors, issue/online-first year, source title, edition, pages and DOI for each unique record.
4. Record individual verification status and provenance, not an unsupported blanket claim.
5. Keep all unchecked records explicitly pending and proceed to the math/figure audit independently.

This PR makes a targeted source-version correction and a consolidated worklist. **The entire bibliography is not yet verified.**
