# Rare-Event Detection: Prevalence, Calibration, and Decisions

Companion study for *An Alarm Score Is Not a Probability: Rare Events, Base Rates, and Decision Thresholds*, article #9 in the [series index](../ARTICLES.md).

This directory contains a self-contained **synthetic** study. Its specified data-generating process separates score discrimination, probability calibration, class prevalence, detection cost, alert capacity, and outcome verification. No corporate, patient, or industrial sensor data are used.

## Reproduction

Python 3.12 is tested in CI. From this directory, run:

    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -v
    python analysis.py
    python analysis.py --quick

The full run regenerates eight original numbered PNG figures in [figures](figures/) and ten CSV tables in [results](results/). The quick run writes separate files with the suffix **_quick**, preserving the publication evidence. Fresh-kernel tests execute the [notebook](rare_event_detection.ipynb).

## Mathematical setup

The synthetic detector score is conditionally Gaussian:

\[
Z\mid Y=0 \sim N(0,1), \qquad Z\mid Y=1\sim N(\mu,1), \qquad \mu=2.
\]

Development prevalence is \(\pi_s=0.20\), deployment prevalence \(\pi_t=0.01\). The log likelihood ratio equals \(\mu z-\mu^2/2\), so the calibrated event probability is

\[
p_\pi(z)=\operatorname{logistic}\!\left(\operatorname{logit}(\pi)+\mu z-\mu^2/2\right).
\]

With stable conditional scores, prevalence shift requires an adjustment of prior odds, but does not change the ranking or population ROC. The experiment contrasts this *label shift* with drift in the positive-class score distribution, which violates the correction's assumptions.

The decision exercise assigns eight arbitrary cost units to a false alarm and 250 to a missed event, while true alarms have zero incremental loss by construction. The Bayes probability threshold is \(8/(8+250)\). Costs, prevalence, and review limits are illustrative modelling choices, not operational recommendations.

## Figures

| Figure | Evidence |
| --- | --- |
| [1](figures/fig1_scores_and_roc.png) | Class-conditional scores and population ROC |
| [2](figures/fig2_precision_and_prevalence.png) | Precision's sensitivity to event prevalence |
| [3](figures/fig3_roc_vs_precision_recall.png) | Stable ROC, different precision–recall curves |
| [4](figures/fig4_calibration_after_shift.png) | Reliability under prior adjustment |
| [5](figures/fig5_estimating_target_prevalence.png) | Unlabeled-target MLE and black-box estimates |
| [6](figures/fig6_cost_of_the_wrong_prior.png) | Cost sensitivity and the Bayes threshold |
| [7](figures/fig7_capacity_constrained_alerts.png) | Limited review capacity and diminishing gains |
| [8](figures/fig8_when_assumptions_fail.png) | Conditional drift and selective outcome labels |

## Machine-readable evidence

| File | Contents |
| --- | --- |
| results/operating_curves.csv | Conditional densities, ROC and precision at both prevalences |
| results/prevalence_sweep.csv | Bayes precision over changing base rates |
| results/calibration_bins.csv | Observed and predicted frequency with bin counts |
| results/prior_estimates.csv | Repeated MLE and BBSE estimates across settings |
| results/cost_curves.csv | Population expected loss for each alarm threshold |
| results/policy_comparison.csv | Frozen, adjusted and near-never-alarm policies |
| results/budget_frontier.csv | Constrained policies at different alert capacities |
| results/conditional_shift.csv | Violated class-conditionals and model misspecification |
| results/verification_bias.csv | Selected-only versus weighted event estimates |
| results/summary.csv | Main exact and sampled numerical results |

Every random stream uses an explicit deterministic seed. Some results are exact analytical Gaussian integrals; others are finite-sample estimates. The manuscript keeps these separate.

## Scope and assumptions

- Gaussian class-conditionals simplify the problem. Real score distributions can be multimodal, correlated and nonstationary.
- Label-shift adjustment requires stable \(P(Z\mid Y)\), calibrated source posteriors, and enough discrimination to identify prevalence changes.
- Brier and logarithmic loss are proper probability scores. Binned expected calibration error depends on bin selection and is not a proper scoring rule.
- The black-box prevalence estimator is deliberately **unclipped** to expose finite-sample instability and out-of-range estimates.
- Alert capacity assumes cases are rankable and have equal handling costs. Sequential dynamics, heterogeneous harm and delayed actions require extensions.
- Horvitz–Thompson weighting needs correct, positive outcome-verification propensities conditional on observed information. Unknown dependence on unobserved outcomes defeats the simple correction.
- Precision for a virtually never-triggered alarm is mathematically computable here but operationally misleading.

The [manuscript](article.md) supplies derivations, scenario diagnostics, and primary research references.
