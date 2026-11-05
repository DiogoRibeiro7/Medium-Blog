# Forbes 2026 primary–secondary discrepancy triage

Status: **source-conflict review; not an approved regression input**.

This note interprets the frozen, aggregate-only evidence from [Actions run 42099896845](evidence/forbes-2026-run-42099896845/README.md). It does not replace the recorded CSVs, modify model inputs, or establish that the live 5 November 2026 Forbes endpoint is identical to the March annual release.

## What the primary audit actually establishes

| Measure | Recorded result |
| --- | ---: |
| Forbes API people and distinct person URIs | 3,428 |
| Assigned `countryOfCitizenship` entries | 3,428 (80 labels) |
| Assigned ambiguous API `country` entries | 3,407 (79 labels) |
| Missing ambiguous API `country` entries | 21 |
| Secondary table total | 3,359 (76 rows) |
| Exact-name primary/secondary comparisons | 75 |
| Nonzero differences among exact-name comparisons | 35 |
| Net primary–secondary total difference | +69 |

These are counts of **people in the live endpoint**, grouped by citizenship label. The separate `country` field has no verified residence semantics. The 69-person net difference is not an estimate of unobserved billionaires; it is a difference between two non-equivalent source snapshots/transcriptions.

## Largest observed exact-label differences

These rows are transcribed directly from `evidence/forbes-2026-run-42099896845/secondary_discrepancies.csv`. Difference = live primary minus secondary.

| Country label | Primary | Secondary | Difference |
| --- | ---: | ---: | ---: |
| Spain | 44 | 34 | +10 |
| Australia | 56 | 47 | +9 |
| Turkey | 40 | 32 | +8 |
| Belgium | 18 | 11 | +7 |
| Malaysia | 25 | 19 | +6 |
| Greece | 21 | 16 | +5 |
| Saudi Arabia | 12 | 15 | −3 |
| Thailand | 22 | 25 | −3 |

These eight largest absolute differences are **triage priorities**, not evidence that either data source is correct for the original chart's intended vintage or attribution.

## Unmatched labels must not become zeros

- Five primary labels are unmatched in the secondary transcription: Afghanistan (1), Pakistan (1), Peru (1), St. Kitts and Nevis (2), and `Eswatini (Swaziland)` (1).
- The secondary label `Eswatini` (1) is unmatched by exact spelling. This is a plausible alias for `Eswatini (Swaziland)`, but it must be an explicit, reviewed crosswalk, not an automatic name join.
- The primary labels `Hong Kong`, `Taiwan`, and `Guernsey` require explicit territorial treatment before population and poverty joins. Never silently assign them to sovereign-state totals.

## Review gates and required evidence

1. **Vintage:** Locate the archived March 2026 Forbes release or independently dated country aggregates. Record the retrieval date, publication date, source hash, and whether the API is revision-stable. A matching overall 3,428 count is insufficient proof of identical membership.
2. **Attribution:** Obtain Forbes documentation defining `countryOfCitizenship` and the separate `country` field. Do not rename the latter to residence based on intuition.
3. **Country identity:** Review all 80 primary labels against ISO3 codes, political/territorial scope, and aliases. Keep an explicit review ledger with source URL and reviewer decision for every nontrivial mapping.
4. **Discrepancies:** Resolve the eight highest-absolute-difference labels first, then all other 27 nonzero exact-label differences; preserve both original counts and documented adjudications. Do not overwrite raw aggregates.
5. **Selection:** Recover the original 64-country inclusion criteria, poverty definition and survey years. An analysis of 54 citizenship-matched countries is a distinct sensitivity analysis.
6. **Statistical sign-off:** Only after the preceding gates, regenerate same-sample unadjusted and development-adjusted regressions, with HC3 inference, year sensitivity, population weighting, and leave-one-out checks.

## Publication rule

The existing 54-country work may be described as an exploratory **citizenship-based** association. It must not be called a reproduction of the original 64-country residence chart, a test of the causal effect of billionaires on poverty, or a verified March 2026 country-count panel.

Frozen inputs: [audit JSON](evidence/forbes-2026-run-42099896845/audit.json), [count discrepancy CSV](evidence/forbes-2026-run-42099896845/secondary_discrepancies.csv), and [source provenance](evidence/forbes-2026-run-42099896845/README.md).
