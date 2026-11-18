# Joint pre-reform macroeconomic support — 18 November 2026

The first *joint* World Bank WDI donor audit shows that the earlier
variable-by-variable marginal-range checks considerably overstate
available three-dimensional interpolation support. These results come
from the **same joined source vintage** as the frozen
[`DONOR_SUPPORT_RESULTS_2026-11-15.md`](DONOR_SUPPORT_RESULTS_2026-11-15.md).
They are finite-sample feasibility diagnostics, **not causal effects**.

## Frozen input and identical eligibility

- Joined WID/WDI country–year panel SHA-256:
  `8dfcdd2bb7ea8c378436d5415a70ae0a83f8f1b431f4e46c70039bf578072321`.
- Original WDI four archive source checksums and last-updated dates are
  preserved verbatim in [`joint_support_2026-11-18.json`](joint_support_2026-11-18.json).
  All four World Bank series report `2026-11-11` as the archive update.
- Original observed reform episodes: **29**. Baseline eligible with
  observed full WID `t-5..t+5`, matched previous cut history and complete
  pre-reform WDI controls: **20**.
- Eligible donor–event pairs *before* subsequent cut exclusions: **239**.
  At year five, **174** remain; **65** have been excluded. These match
  the prior donor audit exactly, including all horizon-level marginal
  support counts and the frozen source hash.

## Horizon-by-horizon empirical support

The three variables are five-year pre-cut country means of GDP-per-capita
growth, CPI inflation, and trade openness. A joint pass requires the
treated vector to lie within a nonnegative convex combination of the
surviving donor vectors (strict standardized residual at most `1e-9`).

| Horizon since cut | Eligible donor–event pairs | All three marginal ranges | Strict joint hull | Marginal passes failing joint hull |
| ---: | ---: | ---: | ---: | ---: |
| Pre-censor `−1` | 239 | 9/20 | 3/20 | 6 |
| 0 | 227 | 9/20 | 3/20 | 6 |
| 1 | 212 | 9/20 | 3/20 | 6 |
| 2 | 197 | 8/20 | 3/20 | 5 |
| 3 | 184 | 8/20 | 3/20 | 5 |
| 4 | 179 | 8/20 | 2/20 | 6 |
| 5 | 174 | 8/20 | **2/20** | **6** |

The relaxed `1e-6` threshold produces the **same** joint-support counts
at all horizons; there are no tolerance-sensitive classifications in
this source vintage. At the pre-censor baseline, the joint-supported
episodes are Finland 1986, United Kingdom 1988, and Germany 2008.
United Kingdom 1988 loses joint support at horizon four, when additional
donor cuts reduce its risk set. Finland 1986 and Germany 2008 remain
joint-supported at horizon five. The generated event CSV retains the
exact witness donors and barycentric interpolation weights for audit.

Thus the previously reported **8/20** year-five marginal-range passes
should **not** be interpreted as eight reforms with multivariate donor
support: only **2/20** have a certificate using the same three macro
means and horizon-five donor restrictions. There is no new matched ATT,
causal effect, propensity score, confidence interval or causal positivity
estimate in this analysis.

## Interpretation and next research gate

- The sample is small, repeated donor–event pairs are dependent, and the
  current WID vintage and pre-existing complete-outcome-window restriction
  influence which reforms are eligible.
- Hull support among three means is insufficient for joint support of
  annual histories, political processes or omitted institutional/fiscal
  covariates. It does not establish exchangeability.
- The existing exploratory matched contrast retains repeat cuts in
  initiators but excludes comparator cuts during follow-up. The
  [dynamic treatment-path audit](DYNAMIC_EXPOSURE_RESULTS_2026-11-16.md)
  documents the distinct policy-strategy problem.
- Extrapolating outside the donor hull, or relabeling the interpolation
  certificates as causal weights, would add assumptions rather than resolve
  the identification gaps.

The next step is to strengthen the **treatment and confounder data
definition**—including legislation timing, political/fiscal context and
choice of treatment strategy—before any covariate-adjusted causal claim.
See [`JOINT_SUPPORT.md`](../JOINT_SUPPORT.md) for the algorithm, scope and
reproduction command. The contemporaneous workflow publishes a seven-day
`tax-cut-joint-macro-support` CSV/JSON artifact for event-level verification.
