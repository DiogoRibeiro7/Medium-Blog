# Manuscript-to-results consistency audit: Parts 6–10

Date: 2026-10-24.

Compared the main numerical Markdown tables in each article with its committed `results.csv`, accounting for rounding and percentage formatting.

| Part | Finding |
| --- | --- |
| 6 — Difference-in-Differences | Primary DID and placebo tables consistent |
| 7 — Instrumental Variables | Five residual stale table entries corrected after prior IV rerun |
| 8 — Regression Discontinuity | Three bandwidth comparison tables consistent after rounding |
| 9 — Synthetic Controls | Three scenario estimator comparison tables consistent |
| 10 — Doubly Robust Estimation | Four specification tables consistent |

Part 7 retained old ordinary least squares means, a median IV entry, first-stage F and percentile values despite the earlier correction to the IV estimates. They have been aligned with the archived current simulation results:

- Moderate valid OLS: 1.4776 (not 1.4782); IV 95th percentile: 1.2030 (not 1.2227)
- Weak valid OLS: 1.4962 (not 1.4971)
- Strong invalid OLS: 1.4332 (not 1.4326); median IV: 1.3027 (not 1.3018); F: 498.6 (not 499.9); 5th/95th percentiles: 1.2299/1.3639 (not 1.2387/1.3537); IV RMSE: 0.3021 (not 0.3023)

These corrections change tables only and do not alter causal assumptions or simulation code. The historical note recording the old weak-instrument estimates is intentionally retained.

## Scope

This confirms published primary Markdown tables against their matching CSV metrics and checks selected surrounding numerical prose. It is not a proof that every mathematical claim or every number elsewhere in prose is correct. Further prose-wide semantic checks, bibliography validation and figure generation remain.
