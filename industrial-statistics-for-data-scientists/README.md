# A Factory Is Not a Histogram: Industrial Statistics for Data Scientists

Reproducible research companion to the Medium-ready
[`article.md`](article.md). This is a **synthetic production line with latent
truth known by construction**, not a real factory or supplier data set.
The article connects statistical process control, process capability,
measurement-system analysis, specification decisions, and designed experiments.

## Questions

1. Why must stability and rational time subgrouping precede capability analysis?
2. How does a process-mean shift differ from a change in within-subgroup spread?
3. How can a known capable physical baseline look less capable when measured?
4. How does a crossed part × operator × repetition experiment estimate gauge
   repeatability, reproducibility, and their uncertainty?
5. Why can measurement bias change both false acceptance and false rejection?
6. What does a guard band change, and whose error costs determine its choice?
7. Why does a factorial interaction matter more than separate one-factor trials?

## Controlled design and assumptions

| Component | Full reproducible experiment |
| --- | --- |
| Physical specification | Target 10.0; LSL 9.2, USL 10.8, synthetic dimension units |
| Time-ordered process | 120 subgroups of five contemporaneous parts; known stable normal baseline `N(10.04, 0.20²)` for 70 subgroups; upward mean shift +0.36 for the remaining 50 |
| Monitoring measurements | One fixed gauge bias +0.10 and iid repeatability noise SD 0.09; no operator switch in the monitoring stream |
| Control limits | Frozen phase-I five-part X̄/R limits estimated from the first 70 measured subgroups; constants `A2=0.577`, `D3=0`, `D4=2.115`, `d2=2.326` |
| Gauge R&R study | 20 parts spanning 9.3–10.7, crossed with 3 operators, 3 repetitions per cell; variance components for repeatability, operator, part × operator, and selected-part spread |
| Repeated study uncertainty | 300 independent balanced crossed studies (70 in quick mode), using separate random-number namespaces |
| Reference master | 180 independent, operator-balanced master readings of a known dimension 10.0 to estimate deployment-wide additive bias |
| Inspection | 200,000 new parts from `N(10.20, 0.25²)`; explicit true/observed confusion outcomes before and after bias correction |
| Acceptance guard bands | 11 inward guard bands from 0.000 to 0.250; illustrative cost 50 per false accept, 1 per false reject |
| Experiment | Randomized 2² factorial (speed × temperature), six replications per corner; four interspersed centerpoints; five new physical parts per run; fitted main effects and interaction |
| Randomness | Independent `SeedSequence` namespaces with master seed `20261030` |

The inspection population is **separate from the baseline production
process**: its deliberately less capable mean and spread provide enough
near-limit observations to assess false-acceptance risk. It must not be
combined with the baseline capability sample.

The gauge study deliberately spans a wide set of parts; its part variance
is the spread of selected reference artifacts, not the factory's short-term
process variance. The nominal total gauge SD includes repeatability,
operator and part×operator effects. A separate stable monitoring gauge has
only its fixed bias and repeatability component; these are distinct designs.

## Reproduce

Run from this directory using Python 3.12:

    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -v
    python analysis.py
    python analysis.py --quick

The full run generates eight PNG figures and ten CSV result tables.
The quick run uses smaller independent repetitions and inspections,
saving outputs with `_quick` suffixes without overwriting publication
artifacts. [`industrial_statistics.ipynb`](industrial_statistics.ipynb)
is an annotated executable companion. CI installs notebook dependencies,
executes it in a fresh kernel, runs scientific unit tests, and regenerates
the eight quick-mode figures.

## Figures

| Figure | Publication image | Interpretation |
| --- | --- | --- |
| 1 | `figures/fig1_process_time_order.png` | Physical subgroup means and biased readings in time order |
| 2 | `figures/fig2_xbar_r_control_charts.png` | Frozen phase-I X̄/R limits and out-of-control postshift subgroups |
| 3 | `figures/fig3_capability_vs_mixture.png` | True/measured baseline and misleading pooled Ppk-style index |
| 4 | `figures/fig4_gauge_rr_variance_components.png` | Crossed ANOVA variance components and independent study uncertainty |
| 5 | `figures/fig5_operator_repeatability_profiles.png` | Repeated measurements of common parts by three operators |
| 6 | `figures/fig6_inspection_misclassification.png` | False acceptance and rejection, before and after master-based bias correction |
| 7 | `figures/fig7_guardband_tradeoff.png` | Guard-band error tradeoff and illustrative weighted cost |
| 8 | `figures/fig8_factorial_interaction_and_risk.png` | Randomized 2² interaction and conditional Gaussian specification-risk projection |

## Machine-readable evidence

| File | Contents |
| --- | --- |
| `results/summary.csv` | Seed and key stable/shifted, gauge, inspection and factorial outcomes |
| `results/process_subgroups.csv` | 120 true/measured subgroup means and ranges, phase and alarm flags |
| `results/capability.csv` | Known baseline and postshift indices, biased measured estimates, pooled descriptive index |
| `results/gage_rr_study.csv` | All 180 crossed part/operator/repetition measurements and latent part truth |
| `results/gage_rr_components.csv` | Nominal variances and one-study nonnegative ANOVA estimates |
| `results/inspection.csv` | Two global confusion summaries for raw and bias-corrected measurements |
| `results/inspection_guardbands.csv` | Eleven corrected-data error tradeoffs and assumed weighted loss |
| `results/factorial_runs.csv` | Randomized run order, factor settings, five-part observed and true means |
| `results/factorial_effects.csv` | Known and OLS-estimated coefficients and standard errors |
| `results/factorial_settings.csv` | Four fitted mean values and model-implied Gaussian nonconformance fractions |

No raw 200,000-part inspection array is committed: the fixed seed, typed
implementation and requirements regenerate it. Only full-run CSVs and
figures are versioned; `_quick` outputs and caches are disposable.

## Interpretive limits

The process baseline is stationary by construction; an individual
three-sigma outlier during that baseline is not proof of a genuine special
cause. In a real phase-I analysis such observations require investigation.
The pooled nonstationary distribution is **not** assigned a valid stable
capability claim. Normal-theory indices and defect probabilities are valid
only under the explicitly stated Gaussian and stability assumptions.

The ANOVA method-of-moments components are truncated at zero when negative
and have substantial finite-study uncertainty. Gauge R&R does not identify
absolute additive bias without a traceable reference. Master-based bias
correction is itself estimated. False acceptance and false rejection are
measurable here because latent part dimensions are generated and retained
internally; a real inspection system generally lacks that truth.

The weighted guard-band objective uses explicitly **invented loss weights**,
not financial outcomes. The factorial model assumes a constant known
physical sigma at all settings; its normal-tail projections are conditional
on that unverified real-world assumption and ignore parameter uncertainty
in sigma. A follow-up confirmation experiment would be essential in
production.
