# Foundations (Parts 1–5): causal identification and scientific review

Review date: 2026-11-02. Status: **methodological review; recommendations are not yet implemented**.

This complements [the numerical reproducibility audit](REPRODUCIBILITY_AUDIT.md). The underlying CSVs have been reproduced, but agreeing with a fixed-seed simulation is not evidence that the estimand, assumptions, or diagnostics are appropriate for an applied claim.

## Scope and identification ledger

| Part | Claimed estimand / question | What makes the synthetic result identified? | Generalization boundary |
| --- | --- | --- | --- |
| [1 — Omitted-variable bias](part1/article.md) | Effect of a one-unit intervention on continuous treatment X | Structural equations set Y = X + 2U + ε; the effect is exactly 1. Adjusted regression controls the generated common cause U; separately randomized X is independent of U. | U is completely measured for adjustment, disturbances are independent, effect is constant, and the outcome model is linear. The unadjusted slope 2 is an association, not an intervention effect. |
| [2 — Randomization](part2/article.md) | Average binary-treatment effect, −1 | Exactly n/2 assignments are selected without replacement using a random permutation. Assignment is independent of generated baseline prognosis U and errors, establishing exchangeability in the generated study. | No treatment noncompliance, attrition, interference, measurement error, or treatment heterogeneity is simulated. HC2 inference is model-assisted and is not a substitute for a randomization-inference check. |
| [3 — Propensity matching](part3/article.md) | ATT, numerically equal to ATE because effect is constantly 1 | Under observed-only assignment, the model uses measured X for treatment selection and outcome adjustment by matching; under hidden assignment, unmeasured U drives treatment and outcome, violating conditional exchangeability given X. | Measured-variable balance cannot establish absence of hidden confounding; calipers and control reuse change retained population/inference. |
| [4 — IPW and overlap](part4/article.md) | ATE = 1 | In the simulation, treatment follows a logistic function of X and all outcome confounding is observed in X. Both arms have positive theoretical probabilities for finite X. | Near-zero empirical propensity and estimation error create finite-sample instability even with formal positivity. Clipped weights trade variance for bias and may change the effective weighting target. |
| [5 — DAG adjustment](part5/article.md) | Total intervention effect (2 for the mediator scenario; 1 for the other three) | Distinct structural graphs specify confounder, collider, mediator and irrelevant-variable cases. Under those graph assumptions, adjusting for a confounder is appropriate; adjusting for a collider opens a path; adjusting for a mediator excludes an indirect path. | These are constructed DAGs, not graphs learned or validated from observational data. A direct-effect regression coefficient after controlling for a mediator needs separate assumptions to have a general causal interpretation. |

## Concrete findings and follow-up checks

### P1 — Add a balance diagnostic that cannot cancel across replications (Part 3)

**Evidence:** `part3/experiment.py` stores `x_difference`, the *signed* matched-treated-minus-matched-control mean for each replication, and `main()` reports the mean of those signed differences. Positive imbalance in some repetitions can cancel negative imbalance in others. The manuscript describes the post-matching observed-covariate mean difference as approximately zero, which is literally supported by the summary but insufficient as a measure of **typical absolute imbalance**.

**Follow-up:** report mean and upper quantiles of `abs(x_difference)`, preferably standardized by a clearly specified pre-match standard deviation. Add the pre-match standardized difference and the matched treated fraction, and identify whether a caliper changes the ATT target. Compare diagnostics by hidden-confounding scenario. This does not invalidate the unobserved-confounding demonstration, but it makes its measured-balance premise more rigorous.

### P1 — Quantify matching reuse and uncertainty (Part 3)

**Evidence:** nearest-neighbour control matching explicitly uses replacement (`matched_control = sorted_controls[neighbor[keep]]`). The code correctly notes that naïve independent-pair standard errors would be inappropriate, and reports between-replication SD, not a confidence interval for an individual matched estimate.

**Follow-up:** archive the number of distinct controls reused, the maximum multiplicity, treated retention, and empirical replication-level spread; if interval coverage is discussed in future, use an inference procedure appropriate to matching with replacement. Do not label the Monte Carlo between-replication SD as an estimated within-dataset standard error.

### P1 — Separate poor overlap from clipping-induced bias (Part 4)

**Evidence:** `part4/experiment.py` estimates a logistic propensity score from the same data, computes Horvitz–Thompson and Hájek estimates, then clips estimated probabilities to [0.05, 0.95] for a third estimator. The strong-selection scenario exhibits a large clipped-estimator departure from the true ATE.

**Follow-up:** compare estimated versus oracle propensity-weight estimators; report realized tail probabilities and effective sample sizes for both arms; vary clipping levels rather than relying on 0.05 alone. A claim that clipping causes a particular magnitude or direction of bias is specific to this DGP and clipping rule. Distinguish finite-sample performance from a change of target population.

### P2 — Validate uncertainty under the randomization design (Part 2)

**Evidence:** `part2/experiment.py` assigns exactly half the sample via permutation but computes HC2 robust regression standard errors and coverage using a normal 1.96 multiplier.

**Follow-up:** add randomization-based or Neyman variance benchmarks for the unadjusted randomized difference; distinguish conditional finite-population randomization inference from the repeated superpopulation sampling currently simulated. Check that the reported coverage refers to the true constant effect, not a heterogeneous-effect finite-population estimand.

### P2 — Preserve distinctions between total and direct effects (Part 5)

**Evidence:** `part5/experiment.py` sets `y_m = a + m + e_y` and `m = a + e_z`. Under intervention, the total effect is 2. Regression adjustment for m approximately recovers the controlled direct coefficient 1 in this specific additive linear SEM.

**Follow-up:** explicitly state that mediator adjustment estimates the controlled direct coefficient *in this structural model*, not a universally identified natural direct effect. For the collider scenario, include the analytic adjusted coefficient 0.5 as a population benchmark. Mark the DAG arrows as assumptions requiring substantive justification in applications.

### P2 — Interpret precision and coverage for the correct target (Part 1)

**Evidence:** `part1/experiment.py` computes nominal 95% interval coverage around the known causal effect 1. In the confounded unadjusted regression, the estimator is consistent for associational slope 2, not causal effect 1.

**Follow-up:** maintain this distinction whenever discussing “coverage”: causal-effect coverage for the naïve fit is intentionally poor, while an interval could cover its *associational* target accurately. A dual-target table would prevent a reader from interpreting failure to cover 1 as a numerical standard-error defect.

## Prioritized implementation sequence

1. Part 3: absolute standardized mean difference and control-reuse diagnostics; regenerate and independently verify its result table, then adjust the relevant manuscript statements.
2. Part 4: oracle-score versus fitted-score and trimming-sensitivity experiment with separate output artefacts; preserve the original benchmark.
3. Parts 2 and 1: uncertainty/coverage target clarifications and analytic design checks.
4. Part 5: explicit population targets and mediation-identification caveat.

## Review boundary

The audit verifies **logical fit between each script's generated structural model and its causal interpretation** and identifies limitations. It is not a peer review of all cited literature, not an independent validation of a real policy intervention, and not a certification of the full 20-part series. A later review should extend this assumption ledger to Parts 6–20 and check every numerical statement against its cited result row.
