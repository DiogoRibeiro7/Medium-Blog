# Bibliography audit — first verification pass

Date: 2026-10-26. Scope: all 20 articles inventoried at the reference-section level; publisher-level verification has **not** been completed for every entry.

## Verified correction

Part 4 cited Matsouaka and Zhou (2024), *Causal Inference in the Absence of Positivity: The Role of Overlap Weights*, as *Biostatistics* 25(1), 1–18. The authoritative bibliographic entry is:

Matsouaka, R. A. and Zhou, Y. (2024). Causal Inference in the Absence of Positivity: The Role of Overlap Weights. *Biometrical Journal*, 66(4), e2300156. https://doi.org/10.1002/bimj.202300156

This was checked against Wiley, PubMed and Duke University author metadata.

## Other checked sources

- Part 5: Cinelli, Forney and Pearl, *A Crash Course in Good and Bad Controls*, DOI 10.1177/00491241221099552, was first available online in 2022 and subsequently appeared in *Sociological Methods & Research* 53(3), 1071–1104 (2024 issue). The existing citation uses the online-publication year and final issue information; it should be made stylistically explicit in the final bibliography pass rather than silently classified as an invalid DOI.
- Robins (1986), *Mathematical Modelling* 7(9–12), 1393–1512, DOI 10.1016/0270-0255(86)90088-6, was confirmed from publisher records. Parentheses inside that DOI must survive automated extraction.

## Reproducibility tool

Run `python causal_inference_statistics/audit_references.py` from the repository root to generate `bibliographic_inventory.csv` from the 20 reference sections. It preserves DOI-internal parentheses, records entries with no DOI, and marks **every row as not_verified**. It does not query Crossref, verify a URL, resolve title ambiguities, or assert metadata correctness.

## Remaining review

The full bibliography must be checked against publisher and DOI metadata (author spelling and order, year, title, journal/book, volume, issue, page range, DOI), with explicit handling of online-first versus issue years, books, conference proceedings, and working papers. Only individually checked records should be marked verified. This initial PR makes one verified correction and provides extraction infrastructure; it is not certification of all references.
