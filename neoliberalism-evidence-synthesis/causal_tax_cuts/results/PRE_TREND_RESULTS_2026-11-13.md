# First real-data pretrend diagnostics — 13 November 2026

These values were computed from the 18-country Hope–Limberg reform panel
joined to the **current** WID top-1% share of pre-tax national income.
See [the source audit](README.md) for the original downloads, 18 country
SHA-256 hashes, sample exclusions and remaining causal limitations.

The analysis ran successfully in
[GitHub Actions](https://github.com/DiogoRibeiro7/Medium-Blog/actions/runs/43514800284),
using a balanced matched cohort, five years of prior policy history and a
five-year post-treatment donor-exposure window. The original author-file hash
and all 18 WID bulk-file hashes **matched the earlier frozen snapshot**.

## Sample

| Audit | Observed result |
| --- | ---: |
| Reform observations in the joined outcome panel | 29 |
| Eligible with a complete before-and-after window | 22 |
| Events with at least one matching control country | **21** |
| Treated countries contributing matched events | **14** |
| Number of eligible donor countries per matched event | **7–16** |
| Eligible events lacking a matching donor | 1 |
| Average pre-reform treated-minus-donor slope gap | **−0.0265 pp/year** |

The observed year-before-reform baseline is `k = −1`; its contrast is zero
by construction and is therefore omitted. All other values are in **income
share percentage points**, relative to that common baseline. The event
cohort and donors are identical across every row.

| Year relative to reform | Mean matched contrast | Median matched contrast |
| ---: | ---: | ---: |
| −5 | +0.162 | −0.003 |
| −4 | +0.060 | +0.083 |
| −3 | +0.131 | +0.097 |
| −2 | +0.119 | +0.180 |
| 0 | +0.197 | +0.127 |
| 1 | +0.276 | +0.169 |
| 2 | +0.367 | +0.658 |
| 3 | +0.512 | +0.431 |
| 4 | +0.642 | +0.563 |
| 5 | +0.795 | +0.611 |

The exact 10-point series is in
[`pretrend_event_time_2026-11-13.csv`](pretrend_event_time_2026-11-13.csv)
and the 210 event-level observations and donor identifiers in
[`pretrend_event_contrasts_2026-11-13.csv`](pretrend_event_contrasts_2026-11-13.csv).
The latter are *derived* contrasts, not redistributions of the authors'
underlying country-year CSV or WID's original bulk files.

## What this evidence can—and cannot—show

1. The **pre-reform contrasts are not all zero**: the four mean leads
   range from **+0.060 to +0.162 percentage points**, while the mean
   treated-minus-donor pre-period slope difference is −0.0265 pp/year.
   Such observed differences must be disclosed rather than silently
   described as successful pretrend validation.
2. The matched post-reform trajectory remains positive and generally
   increases through the fifth year. With fixed donors, the year-three
   estimate is +0.512 pp; it is **not directly interchangeable** with the
   previously reported +0.547 pp from horizon-specific donors.
3. There are **no confidence intervals or p-values**. The signs and sizes
   of these descriptive averages do not quantify sampling uncertainty.
4. The observed pre-trend differences do not identify their cause, and
   small leads cannot establish the unobserved counterfactual trend.
   There is still selection on future donor reforms, potential policy
   overlap, endogenous reform timing, country dependence and unmeasured
   macroeconomic confounding.
5. The outcome is a narrow inequality indicator after major tax cuts.
   These results do **not** establish a causal effect of neoliberalism
   as a political-economic regime on wages, housing or household welfare.

The next scientifically meaningful tests are covariate balance and
country-level uncertainty, followed by explicit sensitivity to missingness,
donor definitions, and future-tax-cut censoring. This document does **not**
label any of those as already completed.
