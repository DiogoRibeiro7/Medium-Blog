# Research plan — Article #35

## Working title

**A Correlation Is Not a Mechanism: Confounding, Collider Bias, and the Limits of Observational Evidence**

This article concerns causal identification from observational data. It is distinct from the repository's articles on randomized-trial transport, missingness, feature importance and sample representativeness. Its central question is whether a statistical association estimates the causal quantity relevant to an intervention.

## Publication standard

- One self-contained, **8,000+ word** English manuscript (`article.md`) aimed at scientifically literate readers of Medium.
- Nine publication-quality PNG figures with readable legends, axes, units and source captions, and no Markdown tables requiring manual conversion.
- Reproducible `analysis.py`, `figures.py`, a standalone notebook, a pinned/ranged requirements file, 18 CSV tables (replicates and summaries for nine studies), and tests.
- Clearly label every observation as **synthetic**, every design-based identification assumption, and every nonidentified conclusion; no simulated effect may be presented as empirical evidence for a real intervention.
- Deterministic random streams and a preserved scientific audit that re-derives numerical claims directly from archival files. Never silently overwrite canonical results.
- A scoped CI workflow with independent code/tests; update `ARTICLES.md` only after the verified manuscript and assets exist.

## Causal questions and estimands

Let $T\in\{0,1\}$ be treatment, $Y(t)$ potential outcomes, $X$ measured baseline common causes, $U$ unobserved common causes and $S$ selection. The principal estimand is the finite or superpopulation average treatment effect:

$
\tau=\mathbb E[Y(1)-Y(0)].
$

Contrast it explicitly with the observational difference $\mathbb E[Y\mid T=1]-\mathbb E[Y\mid T=0]$, the covariate-standardized contrast, conditional average effects and associations conditioned on post-treatment variables. Consistency, conditional exchangeability, positivity and suitable selection conditions must be stated before any point identification claim.

## Nine reproducible synthetic studies

| ID | Study | Controlled contrast | Required diagnostic |
| --- | --- | --- | --- |
| 1 | Common-cause confounding | Naive difference vs standardized/adjusted ATE | Bias across confounder strength, empirical MC intervals |
| 2 | Simpson reversal | Aggregate vs within-stratum treatment association | Group compositions and exact algebraic decomposition |
| 3 | Collider conditioning | Marginal independence vs association induced by conditioning on common effect | Conditional covariance/ATE bias over collider strength |
| 4 | Mediator conditioning | Total effect vs direct-effect estimand when adjusting for mediator | DAG, structural equations, and estimand mismatch |
| 5 | Instrument temptation | Valid vs exclusion-violating or confounded instrument | IV ratio instability and identification caveats |
| 6 | Backdoor adjustment sets | Valid pre-treatment adjustment vs adding collider or omitting confounder | Bias and variance across adjustment choices |
| 7 | Propensity overlap | Good overlap vs near-violated positivity | Weight tails, effective sample size and CI coverage |
| 8 | Unmeasured confounding | Identical observed-law fits with different causal effects | Sensitivity curves and explicit nonidentification |
| 9 | Prediction versus intervention | Predictive feature importance vs counterfactual policy response | Predictive accuracy and decision regret separated |

At least 300 independent replications per stochastic comparison where meaningful. Treat exact identities analytically rather than disguising them as Monte Carlo estimates. Use common random numbers where comparisons benefit from paired designs; record method and seed semantics.

## Mathematical checkpoints

1. Derive the omitted-variable bias expression for the linear common-cause model and compare it against empirical Monte Carlo bias.
2. Show a genuine Simpson reversal with both marginal and within-stratum counts/probabilities; do not attribute the reversal to causation alone.
3. Derive the conditional association created by a linear-Gaussian collider and test its sign against simulations.
4. Explain why controlling for a mediator changes the target from total to direct effect even without confounding.
5. For IV, specify relevance, exclusion, independence, and monotonicity only where a LATE interpretation is intended; avoid claiming generic IV identifies an ATE.
6. Contrast a correct backdoor adjustment set against one conditioning on descendants or opening noncausal paths.
7. Display overlap and effective sample size alongside sampling uncertainty; discuss positivity as an assumption, not a tuning parameter.
8. Give two observationally equivalent data-generating systems whose intervention distributions differ, with a transparent construction.
9. Show a case where a high-predictive-power variable is not a manipulable causal lever.

## Analysis protocol

Use standard-library Python where feasible and NumPy, pandas, scipy and matplotlib where necessary. Type-annotate public APIs, provide docstrings, validate input schemas, and avoid global PRNG state. `python analysis.py --output-dir <path>` must regenerate all machine-readable outputs and figures without changing committed outputs by default. The notebook should execute from a fresh kernel using the same reusable implementations and a separate temporary output location.

For each study archive `data/<id>_replicates.csv` and `data/<id>_summary.csv`. Include the design parameters, sample size, repetitions, estimand, estimator, random seed, Monte Carlo mean/bias/SE, and relevant theoretical comparator. The documentation must distinguish MCSE of an experimental average from the sampling SE of a single estimator.

## Manuscript structure

1. Association versus intervention: the precise inferential error.
2. Potential outcomes and graphical causal models.
3. Confounding as an open backdoor path.
4. Simpson's paradox, stratification, and composition.
5. Colliders, selection and the cost of conditioning.
6. Mediators, direct effects and estimand choice.
7. Instrumental variables and fragile assumptions.
8. Adjustment sets and the temptation to control for everything.
9. Positivity, overlap and unstable weights.
10. Nonidentification and sensitivity to latent common causes.
11. Prediction, deployment and policy intervention.
12. A practical identification audit; limitations and references.

Each technical claim needs a direct citation to a relevant primary paper/book or a derivation. Preferred foundational sources for verification include Pearl (2009), Hernán & Robins (2020), Rosenbaum & Rubin (1983), Greenland, Pearl & Robins (1999), Angrist, Imbens & Rubin (1996), and VanderWeele & Ding (2017). Verify bibliographic metadata and applicability before completing the manuscript.

## Acceptance gates

- All nine experiments can be run independently and produce stable, documented results.
- Unit/property tests and fresh-notebook execution pass locally and on GitHub Actions.
- `SCIENTIFIC_AUDIT.md` checks theoretical identities, numerical outputs, and qualitative interpretations.
- No broken relative links; every figure referenced by the article exists.
- At least 8,000 actual manuscript words excluding bibliography, YAML/code and table data.
- No merge or ready-for-review claim until these gates are verified.
