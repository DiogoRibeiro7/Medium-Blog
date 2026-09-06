# Leave-one-observed-day-out influence sensitivity

The blood-pressure case study contains only a small number of observed calendar
days. HC3 robust covariance helps with heteroskedasticity, but it does not answer a
different question:

> Is one unusual observed day carrying the longitudinal conclusion?

This analysis answers that question by deleting each observed day once and
refitting the established estimands.

## What is refitted

For every observed calendar day, the analysis recomputes:

1. the global equal-day systolic HC3 trend;
2. the gap-aware post-minus-pre episode level contrast;
3. the common within-episode HC3 slope.

The dominant missing gap and the corresponding pre/post episode definition are
first determined from the **full snapshot and then frozen**. They are not
recomputed after each deletion. This is essential: a leave-one-day-out analysis
should perturb the data while keeping the estimand fixed, rather than allowing a
boundary deletion to redefine which episodes are being compared.

Because the gap-aware baseline model requires at least four observed days per
episode, the jackknife requires at least **five observed days in each full-data
episode**. Otherwise removing one day could make the refit undefined.

The analysis also reports ordinary OLS Cook's distance, leverage, and slope DFBETA
as classical influence diagnostics. Those diagnostics identify influential
points; the leave-one-day-out refits quantify how much the scientific estimates
actually move when those points are removed.

## Current snapshot

On the current privacy-safe snapshot:

- the baseline global systolic slope is about **-4.27 mmHg per 30 days**;
- deleting any one observed day keeps that slope negative, between about
  **-4.87 and -3.78 mmHg per 30 days**;
- the HC3 95% interval for the global slope remains below zero for every
  single-day deletion;
- the post-minus-pre episode level contrast remains negative under every
  deletion, ranging from roughly **-7.13 to -5.42 mmHg**;
- its HC3 interval also remains below zero for every deletion.

Those two conclusions are therefore not artifacts of one isolated observed day.

The common within-episode slope behaves differently. Its point estimate remains
negative under the deletion checks, but inference is unstable: the baseline
interval crosses zero, whereas removing particular observed days can make the
upper confidence limit negative. That means the data do not support a robust
binary claim that the within-episode slope is either conclusively negative or
conclusively null.

## Interpretation

The appropriate interpretation is not "no influential observations." Some days
have noticeably larger Cook's distance or slope DFBETA than others. The stronger
and more useful conclusion is:

> no single observed day is sufficient to overturn the global negative trend or
> the pre/post episode level contrast on the current snapshot.

By contrast, inferential significance for the common within-episode slope is
small-sample sensitive.

This is a robustness analysis, not a causal procedure. Deleting a day does not
model the missing-data mechanism, and it does not tell us what would have been
observed on unmeasured days.

## Reproducibility and refresh-sensitive claims

Run from `Blood_Pressure_Missingness/`:

```bash
python day_influence_sensitivity.py \
  --data data/analysis_snapshot.csv \
  --output-json figures/day_influence_sensitivity.json \
  --output-figure figures/day_influence_sensitivity.svg
```

Ordinary CI tests structural properties of the influence analysis, such as one
refit per observed day, baseline estimand agreement, a fixed episode definition,
and valid episode-size preconditions. It deliberately does **not** bake the
current snapshot's scientific conclusions into general code-regression tests.

The secret-backed Google Sheet refresh workflow regenerates the influence outputs
whenever the private source changes and then runs
`validate_current_influence_findings.py`. If new measurements invalidate one of
the current statements above, the refresh stops with an explicit scientific-drift
error so the article can be reviewed before updated outputs are published.
