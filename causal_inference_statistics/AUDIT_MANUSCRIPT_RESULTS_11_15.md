# Manuscript-to-results audit: Parts 11–15

Audit date: 2026-10-24.

This review compares the displayed manuscript tables and key numerical prose with the committed experiment outputs. Numeric values are assessed at the printed number of decimal places; all results belong to synthetic Monte Carlo studies rather than external datasets.

| Part | Archived evidence | Finding |
| --- | --- | --- |
| 11 — Sequential G-formula | `part11/results.csv` | Mean, bias, SD and RMSE for all three methods agree |
| 12 — Heterogeneous treatment effects | `part12/results.csv` | CATE MSE/bias, selected-policy benefits, SD and subgroup effects agree |
| 13 — Double machine learning | `part13/results.csv` and `part13/validation_results.csv` | Original benchmark prose agrees with 20-replication archive; 1,000-replication addendum table agrees with validation archive |
| 14 — Bayesian causal inference | `part14/results.csv` | Posterior means/SD and interval coverage for the **weak-prior (precision 0.1)** settings displayed in tables agree |
| 15 — Marginal structural models | `part15/results.csv`; MSM audit addendum | Main IPW, G-computation, conventional adjustment, bias, SD, RMSE and ESS values agree |

## Special considerations

Part 13 contains **two distinct numerical investigations**. The original `experiment.py` produces `results.csv` with 20 replications per linear/nonlinear design. The separate `validate_nuisance_basis.py` produces `validation_results.csv` with 1,000 replications per design and different nuisance specifications. The later addendum must not be mistaken for a rerun of boosting. Its tabulated values (linear 1.0017/0.0349; nonlinear correct-basis 1.0001/0.0369; nonlinear oracle 1.0001/0.0359) agree with the validation CSV.

Part 14's displayed posterior tables refer to prior precision 0.1. In particular, at n=5000 unadjusted mean 2.0008 and SD 0.0173, and adjusted mean 1.0011 and SD 0.0141. The archived results for other prior strengths are not omitted errors; they are discussed outside those tables.

Part 15's explicit saturated-MSM audit is an algebraic equivalence demonstration rather than new independent evidence. The main manuscript's effect table agrees with the original simulation, while the addendum documents the separate validation output.

## Scope and limits

No stale numerical cells were found in the **primary Markdown tables inspected**. This is not a mechanical proof that every numerical statement throughout all manuscript prose, mathematical equations and references has been cross-checked. It is also not a full external bibliographic audit, causal-identification review or figure validation. No experiment outputs or scientific claims were changed in this PR.

Next: perform the same structured manuscript-to-CSV check on Parts 16–20; then independently verify citations and generate reproducible figures.
