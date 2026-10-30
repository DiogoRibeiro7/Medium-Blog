# Bibliography DOI-resolution audit — October 29, 2026

## Confirmed correction

Part 10, Kennedy (2016), *Semiparametric Theory and Empirical Processes in Causal Inference*: the archived DOI pointed to chapter 7 of the Springer volume, while the cited Kennedy chapter is chapter 8. The manuscript and the matching ledger entry now use **10.1007/978-3-319-41259-7_8**. The record remains marked `doi_linked` pending a complete check of the chapter's other metadata.

## DOI-linked cohort status

The 25 references classified `doi_linked` in PR #369 should not be counted as verified solely because their DOI URLs exist. One concrete mismatch was identified and corrected in this batch. Other records require field-by-field checking against their publisher records, including author order, title, year, source, volume/issue, and page range.

The ledger's citation ID is based on the reference with URLs stripped, so the DOI-only correction does not change Kennedy's occurrence mapping.

## Follow-up

Complete the publisher metadata comparison of the other DOI-linked records before promoting them to `verified`. This PR makes a confirmed correction, not a claim that all 25 records have passed metadata validation.
