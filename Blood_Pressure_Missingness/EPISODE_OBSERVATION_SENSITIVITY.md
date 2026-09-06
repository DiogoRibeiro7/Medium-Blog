# Episode contrast sensitivity to observation intensity

The gap-aware analysis shows that the global systolic time association is dominated by a level contrast between two observed episodes separated by the longest missing-data gap. A remaining question is whether that episode contrast itself is an artifact of unequal measurement intensity.

This sensitivity analysis keeps the same gap-defined episode split and the same within-episode time term, then changes only how measurement intensity enters the model.

## Specifications

The post-minus-pre episode contrast is estimated under five specifications:

1. equal observed-day weighting;
2. adjustment for `log(1 + readings/day)`;
3. weighting daily means by reading count;
4. capping the reading-count weight at three;
5. inverse-intensity weighting as a deliberate stress test.

All models use HC3 heteroskedasticity-robust covariance.

The inverse-intensity specification is **not inverse-probability weighting**. The probability that a calendar day is observed is not identified from this tracker, and the MCAR/MAR/MNAR mechanism remains unidentified.

## Current snapshot

On the current privacy-safe snapshot, the estimated post-minus-pre systolic contrast is approximately:

| Specification | Episode difference (mmHg) | 95% HC3 CI |
|---|---:|---:|
| Equal day | -6.27 | [-10.13, -2.41] |
| Sampling adjusted | -5.61 | [-9.98, -1.23] |
| Reading weighted | -6.38 | [-9.99, -2.78] |
| Capped weight | -6.52 | [-10.14, -2.91] |
| Inverse-intensity stress | -5.41 | [-11.29, 0.48] |

The ordinary weighting and adjustment choices therefore preserve a negative episode contrast with intervals below zero. Only the deliberately aggressive inverse-intensity stress test widens the interval enough to cross zero.

This does not prove a causal episode effect. The two episodes are separated by a long interval with no observations, so the analysis cannot identify when or why the level difference arose. The result is narrower:

> the observed pre/post episode contrast is not explained away by ordinary handling of unequal readings per observed day, but it is not completely insensitive to strong assumptions about the observation process.

The sampling pattern itself differs between episodes: the later episode has more readings per observed day on average. That is exactly why this sensitivity analysis matters.

## Reproducibility

Run from `Blood_Pressure_Missingness/`:

```bash
python episode_observation_sensitivity.py \
  --data data/analysis_snapshot.csv \
  --output-json figures/episode_observation_sensitivity.json \
  --output-figure figures/episode_observation_sensitivity.svg
```

The secret-backed Google Sheet refresh workflow regenerates these outputs whenever the private source changes.
