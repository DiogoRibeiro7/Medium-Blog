# Reproduction audit: Parts 8–10

Date: 2026-10-22. Archived CSVs and experiment scripts were inspected from the connected GitHub repository. The computations were independently executed locally with Python, NumPy 2.3.5, SciPy 1.17.0 and scikit-learn 1.8.0, with `OPENBLAS_NUM_THREADS=1`.

| Part | Rows | Status | Maximum observed difference in numeric fields |
| --- | ---: | --- | ---: |
| 8 — Regression discontinuity | 12 | PASS | 0 |
| 9 — Synthetic controls | 9 | PASS within tolerance | 2.08e-9 |
| 10 — Doubly robust estimation | 16 | PASS | 0 |

## Verification method

For Part 8, the original script in the previously assembled Part 8 package was executed in an isolated directory. For Parts 9 and 10, the current GitHub source was examined and its computations reproduced locally in standalone scripts with the same data-generating steps, estimators, random seeds and replication counts. The resulting CSV columns and all numeric values were compared with the repository's archived CSVs.

Part 9 uses SLSQP numerical optimization; differences at the 1e-9 level are below the series checker tolerance of 1e-8. All nine rows matched within that tolerance. Part 10 matched all 16 rows exactly at the recorded numeric precision.

## Scope

This is a numerical reproduction check of archived tables, not yet a full verification of all prose claims, scientific assumptions, citation metadata or publication figures. Part 9 and 10 checks use faithful reconstructed scripts rather than a byte-for-byte download of the GitHub source into the local runner; clean-environment execution of the exact repository checkout remains useful before final publication.

No experiment definitions or archived outcomes require revision on the basis of these checks. Parts 11–20 remain on the reproduction schedule.
