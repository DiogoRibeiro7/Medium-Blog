# Commercial records: selection before interpretation

Upstream data: Eriksson and Lindgren (2019), [Zenodo 3355300](https://zenodo.org/records/3355300).
Run [39564584621](https://github.com/DiogoRibeiro7/Medium-Blog/actions/runs/39564584621)
successfully validated the corrected CSV parser.

## Verified ingestion

- 3,321 source rows, 34 kitchens; the earlier 1,586/29 result was a
  truncation from interpreting a non-comma CSV with an unsuitable separator.
- Source contains per-meal portion counts with heavily unequal coverage.
- 380 observations have a positive whole-day count, an observed weight,
  and usable kitchen/timestamp labels; these cover 25 kitchens.
- Meal-specific candidates: 388 breakfast, 246 midday, 108 lunch,
  five brunch. These are not combined or treated as whole-day totals.

## Protocol

The R selection audit explicitly tabulates every inclusion condition,
the resulting sample size, representation by pseudonymised kitchen and
the share of recorded waste weights represented in selected rows.
Both overall and kitchen-specific weight sums remain in *source units*.

For complete observations with positive whole-day portion counts, the
audit calculates the ratio of summed recorded waste to summed portions.
It additionally reports the median row ratio and sensitivity to
omitting rows above the empirical 99th and 95th percentile. These are
descriptive diagnostics, not winsorized estimators, uncertainty intervals,
or predictions.

Crucial limitations:
1. The dataset's waste-weight unit has not been independently confirmed
   in a publisher codebook.
2. It is not established that the waste observation window aligns with
   the whole-day portion count or that the weight belongs to plate waste.
3. Zero, missing and incomplete denominator rows are not silently treated
   as zero waste; they are audited as unselected.
4. Kitchen-specific missingness may induce selection bias.
5. Without prices, ingredient costs, or demand measurements, observed
   ratios cannot be converted to margins, calories or euro savings.
6. A ratio on measured waste is not a portion-reduction causal effect.

The workflow writes \`commercial_source_overview.csv\`,
\`commercial_kitchen_coverage.csv\`, \`commercial_ratio_sensitivity.csv\`
and \`commercial_denominator_availability.csv\`. All reported source
measurements must retain these limitations until validated.
