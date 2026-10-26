# Manuscript-to-results audit: Parts 16–20

Audit date: 2026-10-26.

This review checks the primary numerical manuscript tables, key numerical prose, and archived `results.csv` values, with decimal rounding and percentage conversion respected.

| Part | Evidence | Assessment |
| --- | --- | --- |
| 16 — Sensitivity analysis | Four hidden-confounding strengths, observed-only/oracle estimates, model SE and coverage | Consistent |
| 17 — Negative controls and falsification | Target, negative-control outcome, and placebo-exposure means/RMSE/rejection rates | Consistent |
| 18 — Missing outcomes | MCAR/MAR/MNAR effect estimates and RMSE for all approaches | Consistent |
| 19 — External validity and transportability | No Markdown numerical tables; verified corrected key prose numbers against archived CSV | Consistent for inspected claims |
| 20 — Causal decisions | Policy selected benefit, SD, population gain and regret | Consistent |

## Part 19: post-correction figures

The current archive reports transported estimate 2.0040892 for the measured modifier (target 2); 0.9922793 for the omitted modifier (stipulated target 2); 2.5024351 for limited-overlap full target (target 2.5); and 1.7498609 for supported target (target 1.75). Their corresponding rounded prose values 2.0041, 0.9923, 2.5024 and 1.7499 are consistent. The hidden target-modifier shift remains a stipulated causal assumption, not measured target data.

## Part 20: primary benchmark versus pending stress test

The primary policy table reports high-risk selected benefit −1.6003, random 0.5011, estimated benefit 2.5989 and oracle 2.5989, matching the archived results at four decimals. Population gains and regrets agree after rounding. The appended audit note describes a **separate planned nonlinear policy stress test**, but its `validation_results.csv` has not been verified in this review. Therefore the primary benchmark is consistent, while the additional stress test is **not publication-ready evidence** until executed and checked.

## Scope and limitations

This review does not certify all numerals embedded in every prose sentence, formula, bibliographic citation or source-code comment. It does not independently verify causal identification assumptions. No corrections to the primary manuscript tables or archived numerical outputs were required in this batch.

With the preceding audits, all twenty articles have now had their main reported numerical results compared with their stored simulation outputs. Remaining work includes a publication-wide bibliography audit, consistent mathematical notation and generation of figures from executable sources. A clean checkout and CI reproduction test should still be performed before publication.
