# Episode contrast sensitivity to within-episode time form

The gap-aware analysis estimates a post-minus-pre systolic level contrast while
adjusting for a common linear time trend inside the two observation episodes.
Because the dataset contains only 25 observed calendar days, and only eight before
the long gap, it is important to ask whether the contrast is an artifact of that
specific linear functional form.

## Specifications

All models use HC3 robust covariance and the same fixed episode split defined by
the unique longest internal missing run.

1. **Episode only**: no within-episode time adjustment.
2. **Common linear**: the established gap-aware specification.
3. **Separate linear slopes**: allows the two episodes to have different linear
   within-episode slopes.
4. **Common quadratic**: adds one shared quadratic curvature term.
5. **Separate linear + common quadratic stress**: combines separate linear slopes
   with shared curvature. With five regression parameters on 25 observed days,
   this is deliberately treated as a stress specification, not a preferred model.

## Current snapshot

The post-minus-pre systolic episode contrast is:

| Specification | Contrast (mmHg) | 95% HC3 CI |
|---|---:|---:|
| Episode only | -6.27 | [-10.08, -2.46] |
| Common linear | -6.27 | [-10.13, -2.41] |
| Separate linear slopes | -6.27 | [-10.65, -1.89] |
| Common quadratic | -4.86 | [-9.46, -0.27] |
| Separate linear + common quadratic stress | -4.90 | [-10.06, 0.25] |

The level contrast therefore remains negative and its interval remains below zero
under the ordinary alternatives: no time adjustment, the established common
linear adjustment, separate linear slopes, and a modest common quadratic term.
Only the most flexible small-sample stress specification moves the upper interval
slightly above zero.

## Interpretation

This analysis does **not** select a preferred trajectory by AIC or BIC, and it does
not claim that blood pressure follows a quadratic curve. AIC and BIC are reported
only as descriptive model diagnostics.

The appropriate conclusion is:

> the observed episode-level systolic difference is reasonably robust to ordinary
> within-episode functional-form choices, but inference is not immune to a highly
> flexible five-parameter stress model in this small dataset.

This remains an observational robustness analysis. The 32-day gap contains no
measurements, so the analysis still cannot identify when or why the level
difference arose.

## Reproducibility

Run from `Blood_Pressure_Missingness/`:

```bash
python episode_time_form_sensitivity.py \
  --data data/analysis_snapshot.csv \
  --output-json figures/episode_time_form_sensitivity.json \
  --output-figure figures/episode_time_form_sensitivity.svg
```

The private Google Sheet refresh workflow regenerates these outputs whenever the
privacy-safe snapshot changes.
