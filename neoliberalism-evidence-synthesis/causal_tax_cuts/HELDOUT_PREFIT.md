# Chronological pre-reform prediction

Following PR #771, fit outcome-only simplex weights on t-5..t-4,
t-5..t-3 and t-5..t-2, predicting respectively t-3, t-2 and t-1.
Compare equal-weight donor outcomes and last observed treated outcome
(persistence). Scales and weights use only training outcomes; the
held-out treated outcome is used only for evaluation. Report signed
errors, RMSE and MAE in percentage points, and effective donor counts.

The original h=0 and h=5 donor pools are fixed for comparison with prior
audits. Their construction uses future outcome availability, policy
censoring and pre-onset macro completeness, so this is conditional
chronological prediction, not a prospective forecast or independent
external validation. Contemporaneous donor outcomes are available at
prediction time; persistence is a simple benchmark, not a causal model.
Macro-adjusted prediction is deferred because the original five-year
macro means include held-out years. Three holdouts per case are too few
for meaningful inference or a data-driven acceptance threshold.

No post-onset effects, causal estimates or policy assignments are made.
The live workflow produces 36 predictions and 12 summary rows.
