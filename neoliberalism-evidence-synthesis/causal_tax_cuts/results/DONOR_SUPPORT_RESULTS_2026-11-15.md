# Source-backed donor-exposure and macro-support audit — 15 November 2026

This report describes observed donor availability for the existing
five-year-history, five-year-follow-up exploratory tax-cut comparison.
**No covariate-adjusted ATT, p-value, confidence interval or causal
identification claim was produced.** The procedure and file formats are in
[`DONOR_SUPPORT.md`](../DONOR_SUPPORT.md). Exact machine-readable figures
are frozen in [`donor_support_2026-11-15.json`](donor_support_2026-11-15.json).

## Fixed source vintage and denominator

- WID top-1% pre-tax income share and Hope–Limberg binary major-tax-cut
  indicators: **853** observed study country-years in the existing joined
  WID outcome panel; 29 observed reform-year rows after one outcome-missing
  reform is lost. No WID imputation.
- WDI GDP-per-capita growth, inflation, trade openness: official source
  archives last updated **11 November 2026**, read only from `t-5..t-1`.
  Unemployment was downloaded for coverage but not used in these profiles.
- Joined WID/WDI panel SHA-256:
  `8dfcdd2bb7ea8c378436d5415a70ae0a83f8f1b431f4e46c70039bf578072321`.
- Of 29 observed reform-year rows: **20** have baseline macro-complete
  eligible donor pools; **seven** fail the treatment-history rule, **one**
  lacks complete treated-country pre-reform macro data, and **one** has no
  eligible donor with complete pre-reform macro data.
- The 20 horizon-five supported reforms **exactly reconcile** to the
  matchable reform identifiers from the preceding five/five macro audit.

Every candidate, before future-reform exclusions, must have the same five
pre-reform cut indicators, all **11** outcome years `t-5..t+5` observed,
and all five pre-reform years of the three required WDI indicators observed.
This deliberately fixes future outcome coverage when comparing risk sets.

## Donor risk sets and descriptive support

| Horizon `h` | Donor–reform pairs remaining | Censored cumulatively | Reforms with ≥3 donors | Reforms inside all three separate WDI ranges | Mean nearest fixed-scale distance |
| ---: | ---: | ---: | ---: | ---: | ---: |
| −1: before censoring | 239 | 0 | 19 | 9 | 1.088 |
| 0 | 227 | 12 | 19 | 9 | 1.088 |
| 1 | 212 | 27 | 18 | 9 | 1.161 |
| 2 | 197 | 42 | 18 | 8 | 1.208 |
| 3 | 184 | 55 | 18 | 8 | 1.231 |
| 4 | 179 | 60 | 18 | 8 | 1.246 |
| 5 | 174 | 65 | 18 | 8 | 1.287 |

All **20** eligible reforms retain *some* donor at every horizon. But
requiring no new major cut in a donor from `t` through `t+5` removes
**65/239 (27.2%)** of the initially eligible donor–reform pairs. Mean
nearest-donor standardised macro distance rises from **1.088** to **1.287**,
an approximately **18.3%** increase. This increase is calculated with a
*fixed, pre-censor* scale, so it is not a standardisation artifact.

Even before future-reform censoring, only **9/20** reforms fall within all
three **marginal** donor ranges. After five years the count falls to **8/20**.
Marginal range checks are insufficient to establish joint multivariate
support, so these fractions are *upper bounds on the number of reforms
passing that necessary screen*, not estimates of a positivity probability.

## What this result does and does not establish

This is a direct empirical demonstration that the original *no future
donor reform* design imposes material follow-up selection, while overlap
on three recorded macro variables is already limited. That limits the
credibility of interpreting the earlier matched contrasts as causal.

It is **not** legitimate to solve this by leaving post-reform donors in an
untreated control pool after those donors reform. Such donors cease to
represent the specified no-new-cut comparison. The alternative needs an
explicit dynamic treatment design and measured time-varying confounders.

Further limitations:

1. Donor–reform pairs share countries and years; the 239 pairs are not
   independent observations.
2. The outcome window `t-5..t+5` is fixed for all horizons. That itself
   conditions on future outcome availability; we have isolated only the
   **extra** attrition due to future tax-cut censoring.
3. Distances are descriptive Euclidean scores based on three WDI pre-period
   means, not the earlier ridge-Mahalanobis ranking, nor independent balance
   validation.
4. Marginal ranges need not enclose the treated observation in the joint
   three-dimensional donor convex hull, and no propensity/positivity model
   has been fitted.
5. Political/institutional changes, other tax reforms, anticipation and
   capital-mobility conditions remain unmeasured or uncontrolled.

## Original official WDI archive hashes

| WDI series | Archive SHA-256 |
| --- | --- |
| `NY.GDP.PCAP.KD.ZG` | `e81500776bbc1edd0bc2a39d968ea75d2a7edab4d28eaed76e5c72a85245e78d` |
| `FP.CPI.TOTL.ZG` | `09f784d70cf4aa2cf10988a554cdb030629f77b9e8b1a9abebfe7ed2150e6cab` |
| `NE.TRD.GNFS.ZS` | `ce9e1d06619f8973fd68ebc0d947a00fbf1038afe4e00ad5a9ca13a7f9107dcc` |
| `SL.UEM.TOTL.ZS` (coverage only) | `dab4a9cd5934514a8f22ffbc02ffe66294c585d3e35b0a4c98d12b30ccab723f` |

The source-probe GitHub Action publishes the reform-level and donor-level
CSV files in its seven-day `tax-cut-donor-support` artifact. Archived
checksums are the criterion for verifying comparable source vintages;
live third-party source data can change.
