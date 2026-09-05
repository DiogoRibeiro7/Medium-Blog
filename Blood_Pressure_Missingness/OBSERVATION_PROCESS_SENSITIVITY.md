# Observation-process sensitivity

The primary analysis gives each observed calendar day equal weight. That is a defensible default, but it does not make the observation process ignorable. The number of readings per observed day varies from 1 to 21 and is associated with the observed daily systolic mean.

This sensitivity analysis asks a narrower question:

> How much does the estimated systolic time trend change when unequal observation intensity is handled differently?

The analysis uses the same 25 observed calendar days as the primary model and reports HC3-robust uncertainty throughout.

## Specifications

| Specification | Meaning |
|---|---|
| `equal_day` | Every observed calendar day receives equal weight. This reproduces the primary analysis. |
| `sampling_adjusted` | Adds `log(1 + readings/day)` as a covariate. This asks whether the time trend remains after conditioning on observed sampling intensity. |
| `reading_weighted` | Weights each daily mean by its number of readings. This moves toward the implicit estimand of a flat reading-level analysis without pretending the readings are independent. |
| `capped_weight` | Uses `min(readings/day, 3)` to limit the leverage of heavily sampled days. |
| `inverse_intensity_stress` | Weights days by `1 / readings/day`, deliberately emphasizing sparsely sampled days. This is a stress test, **not** inverse-probability weighting. |

## Current snapshot

For the committed 25 observed days, the estimated systolic change per 30 days is:

| Specification | Slope (mmHg / 30 days) | 95% CI |
|---|---:|---:|
| Equal day | -4.27 | [-6.70, -1.83] |
| Sampling adjusted | -3.52 | [-6.00, -1.04] |
| Reading weighted | -4.63 | [-6.86, -2.39] |
| Capped weight | -4.49 | [-6.76, -2.22] |
| Inverse-intensity stress | -3.34 | [-6.87, 0.19] |

The sampling-adjusted model also estimates a coefficient of approximately **-2.52 mmHg** for `log(1 + readings/day)` with a 95% confidence interval of approximately **[-4.88, -0.15]**.

## Interpretation

The negative systolic trend is not an artifact of one particular weighting choice: it remains negative and its 95% interval stays below zero under equal-day weighting, direct adjustment for sampling intensity, reading-count weighting, and capped weighting.

However, the inverse-intensity stress test weakens the estimate enough that its 95% interval crosses zero. That is useful information. It means the apparent decline is **reasonably robust to ordinary weighting choices but not completely insensitive to strong assumptions about the observation process**.

This should not be interpreted as evidence that inverse-intensity weighting is correct. The probability that a day is observed, or that it receives a particular number of readings, is not identified from this tracker. Therefore:

- no inverse-probability weighting claim is made;
- no MCAR, MAR, or MNAR mechanism is declared identified;
- the sensitivity specifications are used to expose dependence on observation assumptions, not to manufacture a corrected causal estimate.

The broader conclusion is stronger than a single p-value and weaker than a causal claim:

\[
\boxed{\text{the negative observed trend is fairly stable, but observation-process uncertainty remains material}}
\]

## Reproduction

```bash
python observation_process_sensitivity.py \
  --data data/analysis_snapshot.csv \
  --output-json figures/observation_process_sensitivity.json \
  --output-figure figures/observation_process_sensitivity.svg
```

The secret-backed Google Sheet refresh workflow runs this analysis automatically after rebuilding the privacy-safe aggregate snapshot.
