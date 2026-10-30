# A 90% Interval Is Not a Promise

Reproducible synthetic companion to the long-form Medium manuscript,
[A 90% Interval Is Not a Promise: Conformal Prediction Under Distribution Shift](article.md).
This is article six in the chronology of [ARTICLES.md](../ARTICLES.md).

## What the experiment actually tests

The code distinguishes an unconditional, finite-sample marginal statement
under exchangeability from subgroup coverage, dependence and changing
target distributions. It compares:

1. The correct finite-sample order statistic for split conformal, including
   the infinity edge case.
2. Pooled intervals, discrete-group Mondrian intervals, and normalized
   intervals using scales learned from separate training data.
3. A covariate-mixture change (group B prevalence 20% → 60%, unchanged
   conditional distribution), with a known-density-ratio *oracle* weighting.
4. A change in the conditional response distribution (group B noise standard
   deviation multiplied by 1.65, unchanged group frequencies).
5. A dependent AR(1) stream with a variance shift and an explicitly
   unguaranteed past-only rolling calibration heuristic.
6. Three response variables with individually calibrated 90% intervals versus
   simultaneous Bonferroni-adjusted intervals.

All data are synthetic. The predictive mean is correctly specified with four
basis terms. Model fitting uses 4,000 independent training observations;
thresholds use 800 separate calibration points; three scenario evaluations
each use 24,000 newly drawn observations. A rare, high-noise group makes the
meaning of marginal validity visible. The 400 independent calibration/test
replications use a smaller calibration size of 240 to show finite-sample
randomness of observed holdout coverage.

## Run it

From this folder:

```bash
python -m pip install -r requirements.txt
python analysis.py
python -m unittest discover -s tests -v
```

For a faster smoke test:

```bash
python analysis.py --quick
```

Quick artifacts get an `_quick` suffix, so they never replace the eight full
figures or ten baseline CSV tables. The [notebook](conformal_prediction.ipynb)
is deliberately unexecuted in git; the scoped CI workflow runs it with a
fresh kernel and executes unit tests and the quick study.

## Eight original figures

| # | Figure | Purpose |
| --- | --- | --- |
| 1 | [Heteroscedasticity](figures/fig1_heteroscedasticity.png) | Show why an accurate mean model does not imply uniform residual uncertainty |
| 2 | [Finite-sample rank](figures/fig2_finite_sample_quantile.png) | Examine correct calibration rank and distribution of coverage across fresh splits |
| 3 | [Marginal vs group](figures/fig3_marginal_vs_group_coverage.png) | Expose undercoverage concealed by acceptable pooled coverage |
| 4 | [Interval geometry](figures/fig4_interval_geometry.png) | Compare average width policies in the two synthetic groups |
| 5 | [Covariate shift](figures/fig5_covariate_mixture_shift.png) | Change group prevalence while retaining conditional outcomes |
| 6 | [Conditional shift](figures/fig6_conditional_shift.png) | Change group B noise and observe loss of coverage |
| 7 | [Temporal adaptation](figures/fig7_temporal_adaptation.png) | Compare fixed calibration and past-only rolling thresholds under drift |
| 8 | [Simultaneous targets](figures/fig8_simultaneous_targets.png) | Distinguish marginal intervals from joint coverage of three outputs |

Every figure is generated from original Python code. No public or
copyright-protected figure is copied or traced.

## Ten machine-readable CSV outputs

| CSV | Content |
| --- | --- |
| `summary.csv` | Main study parameters and reproducible diagnostics |
| `training_groups.csv` | Training counts and fitted group RMS residual scales |
| `calibration_scores.csv` | Group labels, absolute and normalized nonconformity scores |
| `replicate_coverage.csv` | Per-replicate pooled and group-specific holdout coverage |
| `baseline_coverage.csv` | Pooled, Mondrian and normalized interval coverage and widths |
| `interval_geometry.csv` | Intervals along the feature grid for each group |
| `covariate_shift.csv` | Shifted-population coverage, including oracle weighting |
| `concept_shift.csv` | Coverage with larger group B response noise |
| `temporal_stream.csv` | Scores, thresholds, hits and trailing coverage by time |
| `multitarget.csv` | Coordinate thresholds, marginal coverage and joint coverage |

## Numerical signposts from the full seeded run

| Question | Observed fraction |
| --- | ---: |
| Pooled global coverage with exchangeable test distribution | 0.9064 |
| Global coverage for noisy group B, same distribution | 0.6024 |
| Mondrian coverage for group B, same distribution | 0.9104 |
| Pooled global coverage when group B share rises to 60% | 0.7513 |
| Oracle density-ratio weighted coverage under covariate shift | 0.9093 |
| Mondrian group B coverage after its conditional noise rises 65% | 0.7078 |
| Fixed threshold post-change stream coverage | 0.6217 |
| Past-only rolling heuristic post-change stream coverage | 0.8750 |
| Simultaneous coverage of three individually 90% intervals | 0.7168 |
| Simultaneous coverage with Bonferroni-adjusted intervals | 0.8961 |

These are realized percentages of test observations for **one** seeded
realization of each scenario. A finite-sample marginal coverage theorem
averages over random calibration sets and fresh exchangeable test
observations. It does **not** require each fixed calibration set to achieve
exactly 90% observed coverage. A Bonferroni-adjusted realized test fraction
of 0.8961 is therefore not a counterexample to its unconditional theorem.

## Limitations and interpretation

- The independent training, calibration and test streams are essential to
  the split-conformal reasoning. A changing feature mixture breaks the
  original pooled exchangeability condition.
- Mondrian group-specific coverage is conditional on each **discrete group**
  under within-group exchangeability; it is not conditional coverage at every
  exact continuous feature value. Finite small-group calibration increases
  randomness and interval instability.
- Normalized scores use training-estimated RMS group scales, which creates
  adaptive widths but alone proves only marginal coverage.
- The oracle weighted method knows the true, positive, synthetic mixture
  probabilities. Density-ratio estimation and instability under severe
  weights are not studied.
- Under concept shift, the conditional response distribution changes. Merely
  reweighting the calibration covariate distribution is not sufficient.
- The AR(1) series is dependent even before its variance jump. Rolling-window
  historical recalibration is a heuristic; no exact online, dynamic, or
  anytime error guarantee is claimed.
- The three-target experiment has independent Gaussian outputs by design.
  The Bonferroni union bound does not **require** that independence, but
  the realized test numbers here are properties of this simulation.
- Symmetric absolute-residual intervals are used to make the coverage
  question visible; the study does not implement conformalized quantile
  regression, classification prediction sets, learned density ratios or
  guarantees of distribution-free exact conditional coverage.

## Research background

The manuscript links primary research by Vovk, Gammerman and Shafer;
Lei et al.; Romano et al.; Tibshirani et al.; Gibbs and Candès;
Barber et al.; and Angelopoulos and Bates. Full citations and scope
qualifications are in its [reference section](article.md#references).
