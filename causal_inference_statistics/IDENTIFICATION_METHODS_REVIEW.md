# Identification methods (Parts 6–10): scientific and implementation review

Review date: 2026-11-02. Status: **findings and follow-up actions; no statistical results changed in this PR**.

The numerical outputs of these articles were independently reproduced in the [identification reproducibility audit](IDENTIFICATION_REPRODUCIBILITY_AUDIT.md). Numerical agreement is necessary but does not independently validate identification assumptions, inferential procedures, or manuscript rendering.

## Estimands and design conditions

| Part | Estimand and synthetic design | Identification mechanism | Primary limitation |
| --- | --- | --- | --- |
| [6 — Difference-in-Differences](part6/article.md) | Post-policy ATT of −1; two groups over four periods | In the valid scenario, untreated changes are parallel; the alternative scenarios add a differential post-policy untreated shock (+0.75) or a last-pre-period anticipation effect (−0.4). | A placebo comparison of periods 0→1 cannot test future parallel trends or anticipation in period 2. The placebo is intentionally non-diagnostic for these violations. |
| [7 — Instrumental variables](part7/article.md) | Constant structural effect τ=1 for continuous X; Wald/just-identified IV slope | Z affects X and is independent of unobserved U; valid designs impose no direct Z→Y pathway (δ=0). | Relevance and exclusion are distinct. The weak-instrument ratio has unstable tails, and the F statistic measures relevance, not exogeneity or exclusion. |
| [8 — Regression discontinuity](part8/article.md) | Local cutoff effect of −1 under sharp treatment assignment | Untreated conditional mean must be continuous at the running-variable threshold. Separate-sided local linear regressions approximate the limits. | Bandwidth/curvature trade-off and a constructed untreated-outcome jump at the cutoff; neither shrinking bandwidth nor a precise discontinuity verifies continuity. |
| [9 — Synthetic controls](part9/article.md) | Mean post-treatment effect −2 for one treated unit | The treated unit's untreated path is a stable convex mixture of donors plus noise; weights fitted only on pre-treatment periods. | Stable pre-period fit does not identify a post-period idiosyncratic shock, and anticipation contaminates the training period. One treated unit does not provide conventional independent-unit replication. |
| [10 — Doubly robust AIPW](part10/article.md) | ATE=1 in a nonlinear observed-confounder DGP | Conditional exchangeability and positivity are imposed by the DGP; with adequate nuisance convergence, correct propensity *or* outcome regression supports consistent AIPW. | Double robustness is asymptotic and conditional on identification; cross-fitting alone does not address residual confounding, practically inadequate overlap, or simultaneous misspecification. |

## Specific findings

### P1 — Part 7 has a manuscript math-rendering problem

The retrieved [Part 7 article](part7/article.md) contains displayed equations written as bare bracket lines (`[` and `]`) rather than the escaped math delimiters used elsewhere in the series. Several mathematical commands also appear with missing backslashes, for example `operatorname{Cov}` and `mathrm{OLS}`, and the displayed outcome equation contains an invalid control character before `arepsilon`. This is a rendering/encoding issue independent of the simulation.

**Required follow-up:** inspect the *entire* article for damaged TeX sequences, repair them consistently, validate rendering on the target publication platform (including image alternatives where Medium cannot render LaTeX), and ensure mathematical symbols remain semantically correct. A simple numerical reproduction cannot catch this issue.

### P1 — Part 7: report distributional diagnostics for weak IV

The [simulation](part7/experiment.py) computes the just-identified IV ratio as `Cov(Z,Y)/Cov(Z,X)`. It reports means, medians, quantiles, RMSE and first-stage F, an appropriate starting point. A weak first stage makes the ratio highly non-Gaussian; sample means and RMSE are sensitive to rare near-zero denominators.

**Follow-up:** add first-stage denominator diagnostics, explicitly report the share of replications with small/negative first-stage estimates, and compare weak-IV-robust inference such as Anderson–Rubin with conventional Wald intervals if confidence claims are added. Retain the distinction between instrument relevance and exclusion validity.

### P1 — Part 8: bandwidth selection and inference are a separate question

Part 8 uses predetermined bandwidths. The findings demonstrate the bias–variance trade-off but should not be interpreted as an evaluation of data-driven bandwidth selection. Local-linear confidence intervals under curvature may require bias correction and bandwidth-sensitive inference. The sorting scenario is implemented as an unobserved baseline discontinuity, rather than an observed manipulation process in the running-variable density.

**Follow-up:** distinguish a latent discontinuity from demonstrated manipulation; add density and predetermined-covariate continuity checks as diagnostics only if those data-generating features are explicitly modelled. Consider a bias-corrected interval comparison under curvature.

### P2 — Part 6: placebo timing and anticipation

The [code](part6/experiment.py) checks a placebo change from periods 0→1 while estimation uses the change from periods 2→3. Anticipation is injected at period 2; differential untreated drift is injected at period 3. The placebo's failure to detect either is true by construction. This is a clean *counterexample* to over-interpreting pre-trend tests, not a general power study.

**Follow-up:** add a second diagnostic comparing periods 1→2 to illustrate detection of anticipation, and contrast that with the untestability of a shock first appearing after treatment. Label the second comparison as potentially contaminated, not a clean untreated pre-trend test.

### P2 — Part 9: validation outside the training window

The [synthetic-control implementation](part9/experiment.py) fits simplex-constrained weights on all 24 pre-period observations and computes in-sample pre-treatment RMSPE. It does not report validation-window error or placebo-in-space distributions. The post-treatment shock case demonstrates that even excellent pre-fit cannot exclude a treated-only shock.

**Follow-up:** hold out a pre-treatment validation window for descriptive forecast diagnostics, report pre-fit and holdout errors separately, and consider in-space or in-time placebo distributions. Such checks can challenge extrapolation but cannot prove the stability of the unobserved post-treatment counterfactual.

### P2 — Part 10: robust AIPW claims require careful qualifiers

The [implementation](part10/experiment.py) uses two-fold cross-fitting for both nuisance models and clips estimated propensity to [0.01,0.99]. The “correct” models are correctly specified in functional form in the synthetic DGP. However, finite-sample estimates are not guaranteed unbiased; clipping also modifies the weights where it binds. Simultaneously misspecified models illustrate failure of double robustness, while a near-correct naïve contrast arises from accidental bias cancellation.

**Follow-up:** report clipping frequency and prediction error by nuisance-specification regime, add overlap/small-sample sensitivity, and distinguish orthogonal-score asymptotics from a finite-sample guarantee. If inferential coverage is claimed later, derive appropriate cross-fitted influence-function variance rather than relying on Monte Carlo SD alone.

## Implementation order

1. Correct the Part 7 mathematical source and check Medium-safe rendering.
2. Extend Part 7 with first-stage denominator and weak-IV uncertainty diagnostics.
3. Extend Part 8 with bandwidth and continuity/inference sensitivity, taking care to distinguish an imposed jump from actual sorting.
4. Add timing-aware DID placebos for Part 6.
5. Add synthetic-control pre-treatment holdout diagnostics for Part 9.
6. Add propensity overlap and nuisance-prediction diagnostics for Part 10.

## Scope boundary

This is a code-to-identification review, not a claim of empirical causal validity in employment policy or another real-world setting. The model equations define what can be known in the simulations. The corresponding assumptions remain substantive, generally untestable conditions when working with observational data.
