# Manuscript-to-results audit: Parts 1–5

Audit date: 2026-10-24.

Inspected each `article.md` and corresponding archived `results.csv` in Parts 1–5. The published primary estimate and RMSE tables agree with the CSV values at their stated precision.

| Part | Main article tables | Assessment |
| --- | --- | --- |
| 1 | Sample-size estimates; estimator RMSE, SE and coverage | Consistent |
| 2 | Observational and randomized estimates; RMSE, SE and coverage | Consistent |
| 3 | Matched and unadjusted estimates; retained fraction and balance | Consistent |
| 4 | IPW/Hájek/clipped means; RMSE and effective sample size | Consistent |
| 5 | Adjusted and unadjusted slopes across causal structures and sample sizes; RMSE | Consistent |

Not every archived diagnostic is printed in the manuscript. For example, some CSVs include simulation SD or bias alongside primary results. Absence of a CSV value from the article is not a numerical disagreement.

The `check_manuscript_tables.py` command provides a *conservative screening check* for four-decimal Markdown table values. It does not understand the mapping between a row and a statistical estimand, does not validate prose values, and cannot replace the manual audit. Formula constants and theoretically defined true effects may not be present among simulated metrics. Do not interpret a screening pass as independent proof of every causal claim.

Next: audit the numerical assertions of Parts 6–20, including prose and derived values, with explicit row/metric matching. After that, check references and figures.
