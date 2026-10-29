# Bibliography verification ledger — initial inventory

The 20-article series contains **300 citation occurrences** and approximately **210 distinct citation strings** after stripping URL identifiers and normalizing whitespace and case. The distinction matters: repeated books and foundational articles should be verified once, then traced to all manuscript occurrences.

Run:

```bash
python causal_inference_statistics/build_reference_ledger.py
```

This writes `reference_ledger.csv` and `reference_occurrences.csv`. Entries default to **pending**. This generator does not consult Crossref or publisher records and must not be interpreted as evidence of verification. Its output is a reproducible inventory, not a completed audit.

The previous bibliography audit batches document selected externally checked records, but they are not automatically marked verified by this generator. A follow-up should populate verification status only from explicit evidence, retain provenance and handle citation variants representing the same work before final certification.

The initial inventory count is based on the repository content inspected for this PR. Counts can change as manuscript citations are edited.
