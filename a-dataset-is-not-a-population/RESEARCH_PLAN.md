# Article #34 — A Dataset Is Not a Population: Selection, Coverage, Survey Weights, and Transportability

> Research and implementation specification. The manuscript, publication evidence and notebook are not yet complete. Do not publish this document as the article.

## Central question

A dataset is a product of a measurement and selection process, not an automatic miniature of the target population. Even a very large sample can provide a precise estimate of the wrong population parameter. The article will separate *sampling uncertainty*, *selection bias*, *measurement error*, *support failures*, and *target-population drift* through explicitly defined estimands and reproducible experiments.

This article should complement, without repeating:
- Article #12 on transporting **causal treatment effects** from trials;
- Article #30 on missing outcomes, MAR/MNAR, and imputation;
- Article #32 on label quality and selective labels;
- Article #33 on evaluation sets and leakage.

The new contribution is **descriptive and predictive population inference** from coverage-limited and nonprobability datasets, with emphasis on design-based uncertainty, calibration, overlap and scope of claims.

## Notation and estimands

Define a finite target population \\(U=\\{1,\\ldots,N\\}\\) and fixed values \\(y_i\\), with descriptive mean
\\[
\\mu_U=\\frac1N\\sum_{i\\in U}y_i.
\\]
Let \\(S_i\\) indicate sample inclusion and \\(\\pi_i=P(S_i=1)\\) be the first-order inclusion probability under the sampling design. When every \\(\\pi_i>0\\) is known, the Horvitz–Thompson total
\\[
\\widehat T_{HT}=\\sum_{i:S_i=1}\\frac{y_i}{\\pi_i}
\\]
is design-unbiased for the finite population total. With known \\(N\\), dividing by \\(N\\) gives a design-unbiased mean. The ratio/Hájek estimator
\\[
\\widehat\\mu_H=
\\frac{\\sum_{i:S_i=1}y_i/\\pi_i}{\\sum_{i:S_i=1}1/\\pi_i}
\\]
is not exactly design-unbiased in general; compare its bias and variance at finite samples rather than calling it unbiased.

Known inclusion probabilities are strong information. Voluntary participation scores estimated from respondents alone do **not** magically recover them. Under ignorable selection conditional on observed covariates \\(X\\), positivity, and a correct response/selection model, inverse propensity methods can identify certain targets; without a probability reference sample or known target margins, the required information may be unavailable.

For binary \\(Y\\), distinguish sample prevalence \\(P(Y=1\\mid S=1)\\) from the target \\(P_U(Y=1)\\). For prediction, distinguish the sampled conditional law \\(P(Y\\mid X,S=1)\\), the deployment law \\(P_T(Y\\mid X)\\), and the distribution of \\(X\\) in each population. Reweighting marginal \\(X\\) alone requires an invariance assumption about the conditional law; it cannot repair arbitrary concept shift.

## Nine experiment contracts

Every experiment uses synthetic data only. Each has an independently seeded stream keyed by a stable study identifier; replicate-level CSVs, summary CSVs and one publication-quality PNG should be produced from code, not hand-edited.

| Study | Controlled mechanism | Main comparison | Essential caveat |
| --- | --- | --- | --- |
| 1. srs | Finite population with a simple random sample without replacement | Sample mean, design SE, finite-population correction, across-sample coverage | Finite target values remain fixed within each Monte Carlo population |
| 2. strata | Disproportionate allocation across strata with heterogeneous outcome means | Unweighted, poststratified, HT and Hájek estimates | Known stratum sizes are external design information |
| 3. nonresponse | Outcome-dependent participation with observed and unobserved drivers | Complete responders, correctly specified inverse response weights, misspecified weights | MNAR component remains unidentified from respondents alone |
| 4. coverage | A frame missing a subpopulation entirely | Frame-based interval versus full-population target | A zero inclusion probability cannot be fixed by arbitrarily large weights |
| 5. volunteer | Large opt-in convenience panel with tilted recruitment | Naive panel mean versus smaller probability sample | Comparing precision to truth separates variance from bias |
| 6. calibration | Known target auxiliary margins with interacting covariates | Marginal raking, joint-cell weighting, exact standardization | Marginal balance does not ensure joint or outcome balance |
| 7. weighting | Highly variable but known inclusion propensities | Raw IPW, trimmed IPW and uncertainty across repeated samples | Trimming trades variance for estimand-dependent bias |
| 8. predictive | Training sample concentrated in one region of covariate space | Random sampled-data test versus target-population test; density-ratio weighting | Requires support and conditional invariance |
| 9. drift | Target composition shifts over time, with stable and changing outcome mechanisms separately | Historic weighted estimates, updated margins, uncorrectable mechanism drift | A corrected composition cannot correct a changed conditional process |

### Output schema

For each experiment:
- `data/<study>_replicates.csv`: `study,scenario,replicate,method,estimate,target,error` and study-specific diagnostics with documented units.
- `data/<study>_summary.csv`: number of replications, mean estimate, empirical bias, RMSE, SD, Monte Carlo SE, and confidence intervals **for simulation summaries**.
- `figures/figN_<slug>.png`: clear plotted estimand, nontruncated uncertainty, legend and sample/design conditions.
- Published numerical claims in `article.md` must match the canonical summaries within rounding tolerance.

For coverage experiments, additionally record each interval's lower and upper bounds and whether it contains the finite-population target. Do not confuse Monte Carlo uncertainty in *estimated coverage probability* with a nominal interval's intended probability of containing a target.

## Mathematical and scientific checks

1. Independent reproducibility: deterministic full run at a pinned seed, isolated streams per study and repetition; tests for exact schemas, unique rows, finite metrics and seed stability.
2. Design identity: verify empirical HT total unbiasedness under sampling designs with positive known inclusion probabilities, allowing finite Monte Carlo tolerance.
3. Finite population correction: check \\(\\operatorname{Var}(\\bar y)= (1-n/N) S^2/n\\) for SRSWOR with \\(S^2=\\sum_i(y_i-\\mu)^2/(N-1)\\).
4. Weight normalization: distinguish design weights, estimated response weights and calibrated weights. Record effective sample size \\((\\sum w)^2/\\sum w^2\\), labeled as a descriptive weight-concentration diagnostic rather than a universal design effect.
5. Calibration invariants: confirm target auxiliary totals only for the margins actually constrained. Never imply marginal matching identifies unobserved joint associations.
6. Positivity: force and verify a no-overlap case where weighting cannot identify the omitted segment's mean without additional assumptions.
7. Model invariance: test covariate shift and concept shift separately, identifying where importance weighting is valid and where it fails.
8. Sample-size stress test: demonstrate that variance may shrink while selection bias remains nonzero.
9. Sensitivity: report how conclusions change under plausible but unverifiable response/outcome associations; do not present these as identified corrections.
10. Notebook: regenerate from checked-in source; execute from the repository root and from an unrelated working directory in a fresh kernel; quick mode may not overwrite canonical figures or CSVs.

## Article outline

1. A million records can be less representative than a thousand.
2. The population, the sampling frame, the selected sample and the observed respondents.
3. What probability sampling buys: design-based estimators, uncertainty and finite-population correction.
4. Why unequal selection changes the estimand.
5. Stratification, poststratification, raking and the limits of known margins.
6. Nonresponse versus undercoverage: when the data contain no evidence.
7. Opt-in panels, self-selection and the illusion of tiny error bars.
8. Effective sample size, unstable weights and trimming.
9. External validation of predictive models against the deployment population.
10. Covariate shift, concept shift and failed transport.
11. Updating a population target when composition changes.
12. Practical provenance ledger: who could have entered, who did, who was measured and who is the target?
13. Reporting checklist and sensitivity analyses.
14. Limits and future research.

Target: at least 8,000 words of original narrative, nine publication figures, eighteen canonical CSVs and a fully executing notebook. No pseudo-empirical claims, invented dataset provenance or unsupported real-world percentages. Medium-compatible Markdown: figures as PNGs and no tables in the eventual article itself.

## Primary methodological references to verify and cite in manuscript

- Horvitz, D. G., and Thompson, D. J. (1952). A generalization of sampling without replacement from a finite universe. *Journal of the American Statistical Association*.
- Hájek, J. (1971). Comment on an essay on the logical foundations of survey sampling. In *Foundations of Statistical Inference*.
- Deville, J.-C., and Särndal, C.-E. (1992). Calibration estimators in survey sampling. *Journal of the American Statistical Association*.
- Kish, L. (1965). *Survey Sampling*. Wiley.
- Groves, R. M., et al. (2009). *Survey Methodology*, second edition. Wiley.
- Bethlehem, J. (2010). Selection bias in web surveys. *International Statistical Review*.
- Elliott, M. R., and Valliant, R. (2017). Inference for nonprobability samples. *Statistical Science*.
- Sugiyama, M., Krauledat, M., and Müller, K.-R. (2007). Covariate shift adaptation by importance weighted cross validation. *Journal of Machine Learning Research*.

The bibliography is a verification checklist at this stage. Verify titles, editions, page ranges, DOI and exact support before writing the source audit.
