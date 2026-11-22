# Pre-reform political comparability — 21 November 2026

This is a **descriptive audit of the existing donor pool**, not a new
tax-cut treatment effect, matching algorithm, or causal identification
claim. It extends the official [DPI2023 source and coverage
audit](POLITICAL_INSTITUTIONS.md) merged in PR #716. All political features
are drawn from five calendar years preceding each major cut.

## Data and inherited risk set

- Exposure: the 18-country, 1965–2015 Hope–Limberg 2-SD major tax-cut events.
- Outcomes/macro selection: unchanged from the verified WID/WDI panel.
- Political data: Cruz et al., *Database of Political Institutions 2023*,
  [IDB dataset](https://data.iadb.org/dataset/the-database-of-political-institutions-dpi-2023),
  DOI [10.60966/8tx1xs4i](https://doi.org/10.60966/8tx1xs4i), CC BY 4.0.
- The cohort remains **20 reform episodes**, **239** donor–reform pairs
  before future donor cuts are excluded, and **174** pairs at year five.
- Only **17 of 20** reforms have at least one donor comparison with complete
  political data; the other three early reforms lack a full source history.

Both the initiator and donor must have a complete five-year history
(t−5,...,t−1) for all three political fields. Missingness is never turned
into a category or a zero. Later donor reform exclusions are inherited
from the prior audit and shown at h = −1, 0, ..., 5, not newly justified.

## Measures and unit of analysis

For each donor–reform pair with complete prehistory, report:

1. Whether the executive parties' **economic orientation categories**
   (right/center/left) agree at t−1.
2. Whether their five annual orientations are identical.
3. Whether **EIEC**, the executive electoral-competitiveness code, agrees
   at t−1. This code is categorical/ordinal, not a continuous economic
   variable; identical values do not prove comparable politics.
4. The absolute gap in the mean **CHECKS index** over the preceding five
   years. This is a descriptive index distance, not a causal weight.

We report pair-weighted measures alongside an event-weighted orientation
agreement rate, because reforms have different numbers of donors. Neither
solves dependence: the same countries recur across events.

## Source-verified results

| Metric | Pre-censor (h = −1) | Year five (h = 5) |
| --- | ---: | ---: |
| Donor–reform pairs | 239 | 174 |
| Both countries complete | 206 | 143 |
| Same orientation at t−1 | 80/206 (38.8%) | 54/143 (37.8%) |
| Same orientation path for all five years | 25/206 (12.1%) | 20/143 (14.0%) |
| Same EIEC at t−1 | 205/206 (99.5%) | 142/143 (99.3%) |
| Mean absolute CHECKS index gap, pair-weighted | 1.297 | 1.397 |
| Mean orientation agreement, event-weighted | 38.0% | 36.1% |

EIEC offers virtually no separation in this sample of predominantly
competitive electoral systems. Executive-party orientation and CHECKS
have greater observed differences. **These are descriptive comparisons**,
not evidence of causal political confounding or sufficient balance.

The 143 complete year-five pairs represent only 17 eligible reform
episodes, not 143 independent experiments. The preceding
[joint macro support audit](JOINT_SUPPORT.md) found joint WDI donor-hull
support for only **2 of 20** reforms at year five. Introducing more
political regressors cannot resolve that by itself.

## Reproduce

From neoliberalism-evidence-synthesis/:

~~~bash
python -m causal_tax_cuts.political_comparability \
  --controls data/tax-study/political-institutions/dpi_political_controls.csv \
  --donors data/tax-study/political-institutions/political_donor_coverage.csv \
  --source-audit data/tax-study/political-institutions/political_coverage_audit.json \
  --output-dir data/tax-study/political-comparability
~~~

The source-probe GitHub workflow produces the three required inputs from
PR #716's original live IDB download and inherited WID/WDI risk sets. The
new step runs afterward and publishes a seven-day artifact containing:

- **political_comparability_pairs.csv:** 239 rows; missing comparisons
  are explicitly blank.
- **political_comparability_event_horizons.csv:** 140 event–horizon rows,
  preserving event-specific donor and complete-pair counts.
- **political_comparability_audit.json:** the seven horizon summaries,
  input SHA-256 hashes, source lineage and limitations.

The frozen [machine-readable snapshot](results/political_comparability_2026-11-21.json)
provides the original source vintage. When all input SHA-256 hashes agree,
CI compares the entire generated JSON against it. A changed vintage is
reported instead of passing off revised source data as the original sample.

## Next identification gate

Annual DPI categories cannot identify the enactment, announcement,
implementation, anticipation or intensity of a specific tax law. The next
research stage should audit legislative event timing and reconsider whether
the 18-country, five-year donor comparison is identifiable given its
very limited macro support.

**No new ATT, causal weights, standard errors or p-values are estimated.**