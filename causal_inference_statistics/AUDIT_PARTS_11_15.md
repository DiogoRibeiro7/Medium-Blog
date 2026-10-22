# Numerical reproducibility audit — Parts 11–15

Audit date: 2026-10-22.

The Monte Carlo calculations of Parts 11–15 were independently reproduced locally from the GitHub-connected source definitions, using the same simulation parameters, sampling sequence and random seeds. Calculations were compared with the committed `results.csv` files.

## Results

| Part | Method | Archived mean | Reproduced mean | Status |
| --- | --- | ---: | ---: | --- |
| 11 | Sequential G-formula | 3.3671417288 | 3.3671417288 | PASS |
| 11 | Omit intermediate confounder | 4.1142409288 | 4.1142409288 | PASS |
| 11 | Conventional adjustment | 1.8065828581 | 1.8065828581 | PASS |
| 12 | Random policy | 0.4987221442 | 0.4987221442 | PASS |
| 12 | Baseline-risk policy | -0.7076956626 | -0.7076956626 | PASS |
| 12 | Linear CATE policy | 2.6439433491 | 2.6439433491 | PASS |
| 12 | Quadratic CATE policy | 2.6439911936 | 2.6439911936 | PASS |
| 12 | Oracle policy | 2.6440316114 | 2.6440316114 | PASS |
| 13 | Linear design, linear DML | 0.9953911629 | 0.9953911629 | PASS |
| 13 | Nonlinear design, boosted DML | 1.3601245806 | 1.3601245806 | PASS |
| 13 | Nonlinear design, no cross-fitting | 1.3042915561 | 1.3042915561 | PASS |
| 14 | n=5000, unadjusted, weak prior | 2.0007879807 | 2.0007879807 | PASS |
| 14 | n=5000, adjusted, weak prior | 1.0011447419 | 1.0011447419 | PASS |
| 15 | Sequential IPW | 3.3539826283 | 3.3539826283 | PASS |
| 15 | G-formula | 3.3570932694 | 3.3570932694 | PASS |
| 15 | Conventional adjustment | 1.7973158538 | 1.7973158538 | PASS |

Additional checks: all Part 12 CATE prediction errors and subgroup effects reproduced; Part 14 Gaussian posterior output for all sample sizes and prior strengths reproduced under the factorial simulation; Part 15 effective sample sizes reproduced (503.9524824 and 299.4496171, to floating-point tolerance). All Part 13 methods in both designs were reproduced.

## Scope and limitations

These are independent local reproductions of the statistical calculations represented in the repository scripts; GitHub source was inspected through the connected repository, then the calculations were reconstructed in local standalone scripts. The container could not fetch the repository directly through `raw.githubusercontent.com`. The check is therefore **not** a byte-for-byte execution of the entire checked-out repository or a CI pass.

Numeric matches were checked at the displayed precision; no changes to `experiment.py`, `results.csv`, or manuscript numerical statements are proposed.

This check does not establish validity of causal identification assumptions, verify all citations, audit figures, or certify manuscript prose. The series audit should next reproduce Parts 16–20 and then independently check manuscript tables, references and publication figures.
