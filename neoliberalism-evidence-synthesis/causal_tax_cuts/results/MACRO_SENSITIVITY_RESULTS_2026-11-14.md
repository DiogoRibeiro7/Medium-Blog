# Pre-reform macro matching sensitivity — 14 November 2026

This report tests the stability of a donor-selection **diagnostic**, not an
adjusted tax-cut effect. It uses the same audited WID/treatment and World Bank
WDI observations as the [earlier macro balance study](MACRO_MATCHING_RESULTS_2026-11-14.md)
and keeps the original five-year outcome horizon unchanged.

## Cohort and source identity

- Original country-year sample: **853 rows, 29 tax-cut event-years**.
- Official WDI data: same four source-2 CSV ZIP archives, last updated
  **11 November 2026**. Their SHA-256s and the joined WID/WDI panel SHA-256
  are preserved in the [machine-readable results](macro_sensitivity_2026-11-14.json).
- Grid: `L=3,5,7` years of pre-reform history crossed with `K=3,5,7`
  nearest donors, leaving the main five/five specification unchanged.
- Original five/five event count and slope summaries were rechecked against
  the frozen prior audit because the source-panel hashes agree.
- Events supported by **all nine** rules: **19**, forming the like-for-like
  comparison cohort. All 19 are among the reference's original 20 matches.

| Pre-reform years `L` | Selected donors `K` | Supported tax-cut events | Donor-set overlap with 5/5 (Jaccard, common cohort) | Mean absolute fixed 3-year income-share slope gap (pp/year) |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 3 | 21 | 0.527 | 0.287 |
| 3 | 5 | 21 | 0.690 | 0.289 |
| 3 | 7 | 21 | 0.644 | 0.260 |
| 5 | 3 | 20 | 0.621 | 0.302 |
| **5** | **5** | **20** | **1.000** | **0.293** |
| 5 | 7 | 20 | 0.729 | 0.290 |
| 7 | 3 | 19 | 0.561 | 0.302 |
| 7 | 5 | 19 | 0.704 | 0.274 |
| 7 | 7 | 19 | 0.618 | 0.256 |

Longer histories exclude more events: **21 → 20 → 19**. Increasing the donor
cap does not recover missing treatment histories or lagged macro data, so
the supported event counts stay constant across `K` within the same `L`.

## An important balance trade-off

Every slope number above is evaluated on the *same 19 reforms* and the same
three pre-reform years (`t-3..t-1`), regardless of the matching rule.
Donor composition does change: the mean Jaccard overlap with the five/five
reference ranges from approximately **0.527 to 0.729** across alternatives.
This is partly mechanical when comparing three versus seven donors, but it
also shows that results are contingent on the chosen matching specification.

The original five/five rule has a **0.293 pp/year** absolute outcome slope
gap on the common cohort under this fixed three-year diagnostic. That is not
the older **0.218 pp/year** figure: the older figure averages five-year
pre-reform slopes over **20 events**, whereas this analysis evaluates three
years over **19 events**. Comparing them directly would be misleading.

The alternative with the smallest displayed outcome slope gap is **not**
automatically preferable. For example, changing the donor count can improve
income-share pretrends while worsening the gaps on one or more macro controls.
Using outcomes to select the most favourable grid cell after seeing results
would introduce a new analysis-selection problem. The five/five rule remains
the declared reference; the nine cells are a diagnostic envelope.

## Fixed-window World Bank balance

The machine-readable grid additionally includes the mean absolute
standardised GDP-growth, CPI-inflation and trade-openness gaps, evaluated
using identical event dates, covariate years and study-country denominator
for all nine rules. No individual rule dominates all observed dimensions.
These macro controls are candidate proxies, not a replication of the full
original Hope–Limberg covariates or a guarantee of overlap/positivity.

## Scientific conclusion

The pre-reform selection is somewhat stable in *event coverage* but materially
sensitive in *donor composition* and measured outcome-pretrend balance.
The grid strengthens the evidence that covariate matching alone cannot
resolve the study's identification risks. Donors are still excluded based on
future reforms, and countries/events are statistically dependent.

No adjusted ATT, causal interpretation, standard error or significance test
is reported. The next task should study common support and the impact of
future-treatment-based donor exclusion before implementing an adjusted
contrast with credible inference.
