# Causal Transportability and External Validity

Reproducible companion to article #12, [*A Randomized Trial Is Not a Universal Policy: Transporting Causal Effects, External Validity, and the Price of an Unrepresentative Experiment*](article.md).

The complete study is **synthetic**. It evaluates the distinction between a randomized-trial average treatment effect and the average treatment effect in a disjoint target nonparticipant population. All data and numerical findings are generated from the specified statistical mechanisms; none are claims about a real treatment, institution, patient cohort, or deployment.

## Reproduction

Use Python 3.12 or a compatible recent scientific Python environment. From this directory:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python analysis.py
python analysis.py --quick --output /tmp/causal-transport-smoke
```

The full run writes eight publication figures to [`figures/`](figures/) and twelve CSV tables to [`results/`](results/). The quick run uses smaller simulation counts, appends `_quick` to output filenames, and supports a separate output directory so it cannot accidentally overwrite publication figures. The dedicated GitHub Actions job installs the dependencies, runs all tests (including the notebook in a new kernel), and regenerates an independent quick study.

The [notebook](transportability.ipynb) computes the exact target and trial population effects, inspects one trial, demonstrates unmeasured-modifier bias, and creates a temporary independent quick study. Its executable cells have no precomputed outputs.

## Study design and assumptions

Let `S=1` denote entry into a randomized trial and `S=0` denote eligible nonparticipants in a separate target sample. We observe `(X,Z,A,Y)` in the trial and only `(X,Z)` in the target. The target is **specifically `S=0`**, not everybody eligible.

The superpopulation draws independent variables `X ~ Normal(0,1)`, `Z ~ Bernoulli(.45)` and `U ~ Bernoulli(.5)`. Trial participation obeys

\[
P(S=1\mid X,Z,U) = \operatorname{expit}(-.85 -1.1Z -\lambda X +\gamma U).
\]

The baseline has `lambda=0.8`, `gamma=0` and an additive potential-outcome effect

\[
\tau(X,Z,U)=.4+.9Z+.5X+.35ZX+\delta_UU,\quad \delta_U=0.
\]

The response satisfies

\[
Y(0)=1+.5Z+.4X+.15X^2+\varepsilon,\quad
Y(1)=Y(0)+\tau(X,Z,U),\quad \varepsilon\sim N(0,.8^2).
\]

Treatment `A` is randomized only among trial participants at probability `0.5`; consistency and no interference are built into this simulation. Observed trial outcomes are generated with independent Gaussian noise. To make the selection problem explicit, source and target covariates are independently rejection-sampled from their respective conditional populations.

All target-reference effects are calculated from the known structural model using 80-point Gaussian–Hermite quadrature, integrating exactly over the binary covariates. `oracle_effects()` returns the trial and target ATE, trial participation probability, observed-covariate-only transport estimand under hidden modifier selection, and its identification bias.

The target ATE is identified from measured data when treatment is consistent and randomized in the trial, outcome response functions or the conditional additive treatment effect are stable across `S` conditional on `(X,Z)`, the target has trial support for every relevant measured stratum, and relevant measurement and sampling assumptions hold.

## Nine estimators

The main experiment performs **300 independent replicates**, each with `n_trial=500` and `n_target=1400`, using the same exact target ATE. It evaluates:

1. Unadjusted randomized-trial treatment contrast.
2. Outcome standardization with correct features `(1,Z,X,ZX,X²)` in both arms.
3. Outcome standardization with reduced features `(1,Z)`.
4. Inverse selection-odds weighting with correct logistic predictors `(1,Z,X)`.
5. Inverse selection-odds weighting with reduced predictors `(1,Z)`.
6. Augmented weighting with correct outcome and selection models.
7. Augmented weighting with correct outcome but reduced selection model.
8. Augmented weighting with reduced outcome but correct selection model.
9. Augmented weighting with both nuisance models reduced.

The three families are implemented in `outcome_predictions()`, `trial_target_weights()`, `weighted_ate()`, and `augmented_ate()`.

The inverse odds `(1 - P(S=1|W))/P(S=1|W)` are appropriate for the *disjoint target-nonparticipant* design. We fit a logistic model for trial versus target **sample membership** in pooled records; the resulting slopes estimate the relevant conditional density ratio, while the changed intercept produces a multiplicative constant that cancels in the self-normalized estimates. The code never treats the arbitrary sample-size ratio as the superpopulation trial participation fraction.

The augmented estimator is the target mean modeled contrast plus an inverse-odds-weighted and treatment-probability-corrected trial residual term. It is doubly robust to *one of the two nuisance-model specifications* under the causal identification assumptions; it is not generally robust to omitted effect modification, nonpositivity, noncomparability of interventions, or simultaneous misspecification of both nuisance families.

## Additional experiments

### Selection-overlap stress test

We vary the strength of selection on `X` across `lambda in {0.3, 0.8, 1.4, 2.0}`. Each scenario has **180 independently generated replicates**, with `n_trial=450`, `n_target=1350` and a new exact oracle target ATE. We record bias, RMSE, Kish's weight effective sample size, and high-weight concentration. We do not truncate weights. The declining ESS does not imply a particular estimator's exact sample size or efficiency.

### Unobserved effect modifier

The latent variable `U` changes both selection and treatment effect. We evaluate a 5-by-5 grid of the latent selection coefficient `gamma` and latent treatment-effect modifier `delta_U`, each in `{-2,-1,0,1,2}` or `{-1.2,-.6,0,.6,1.2}`, respectively. The exact infinite-data bias is

\[
\delta_U\,E\{P(U=1\mid W,S=1)-P(U=1\mid W,S=0)\mid S=0\}.
\]

This is an **oracle sensitivity calculation**, not an estimate of hidden parameters from the observed `(X,Z)` data. Its purpose is to demonstrate a limitation of causal identification, independent of estimator sampling error.

### Bootstrap

For one fixed seeded source/target cohort (`550`/`1800`), `400` bootstrap iterations independently resample trial and target records and refit models. Percentile intervals illustrate two-sample sampling uncertainty. They are not claimed to have proven universal 95% coverage and cannot repair unobserved-modifier bias.

## Figures

| Figure | Interpretation |
| --- | --- |
| [1](figures/fig1_trial_target_shift.png) | The trial and target have different `X` distributions |
| [2](figures/fig2_effect_heterogeneity.png) | True conditional effect heterogeneity and two population average effects |
| [3](figures/fig3_estimators.png) | Monte Carlo distributions of nine estimators relative to target truth |
| [4](figures/fig4_double_robustness.png) | Correct/reduced nuisance-model matrix for augmented-estimator bias |
| [5](figures/fig5_overlap_ess.png) | Weight ESS declines when source-to-target overlap deteriorates |
| [6](figures/fig6_overlap_rmse.png) | RMSE versus selection intensity, using scenario-specific target truths |
| [7](figures/fig7_hidden_effect_modifier.png) | Oracle transport bias caused by unobserved modifier and selection |
| [8](figures/fig8_bootstrap_intervals.png) | Two-sample percentile intervals for a single source/target realization |

## Machine-readable evidence

| CSV in `results/` | Contents |
| --- | --- |
| `population_truth.csv` | Exact trial/target effects, transport oracle and group mass |
| `cohort_diagnostics.csv` | Realized trial/target means, counts and modifiers |
| `covariate_density.csv` | Histogram bin edges and densities for both groups |
| `conditional_effect_curves.csv` | Exact conditional effects by `X` and `Z` |
| `estimator_replicates.csv` | All nine methods for every main Monte Carlo replicate |
| `estimator_summary.csv` | Exact target truth, bias, SD and RMSE per method |
| `overlap_replicates.csv` | Raw estimates and weight diagnostics across overlap conditions |
| `overlap_summary.csv` | Scenario-specific truth, bias, RMSE, ESS and tail concentration |
| `hidden_modifier_sensitivity.csv` | Exact target, observed-only oracle and bias over a 25-point grid |
| `bootstrap_replicates.csv` | Independent two-sample bootstrap estimates |
| `bootstrap_intervals.csv` | Percentile endpoints and bootstrap standard deviations |
| `study_summary.csv` | Design-wide oracle and selected realized diagnostics |

## Important limits

- All results concern artificial populations with explicit structural equations, not external data.
- The target is `S=0` nonparticipants; extrapolating weights to a full eligible population requires changing the estimand and design formula.
- The outcome and selection models labeled *correct* are correctly specified by construction; this knowledge would not be available in a real analysis.
- The baseline assumes complete follow-up, independent observations, consistency, no interference, comparable covariates and one randomized binary treatment.
- Positivity holds in the mathematical logistic model but becomes practically weak as selection grows; no extreme-weight clipping is applied.
- The hidden-modifier experiment shows population identification bias even with infinite observed-covariate information. A narrower interval cannot remove it.
- Simulated bootstrap intervals describe a single-study two-sample resampling procedure, not verified repeated-sampling coverage.
- The script is an educational research experiment, not a validated clinical, economic or policy decision support package.

For scientific interpretation and the foundational references, read the [full article](article.md).
