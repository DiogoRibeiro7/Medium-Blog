# Causal inference series: claims-to-evidence audit, Parts 1–5

Date: 2026-10-31.

This is the first **substantive claim audit**, following the 210-reference bibliographic identity check. A reference existing does **not** establish that the article's claim follows from it. The checks below compare a precise methodological proposition in the manuscript with a primary publication's abstract or substantive result. This audit does not claim that the Monte Carlo datasets were regenerated.

## Claim-level evidence matrix

| Part | Manuscript claim / location | Primary source and check | Decision |
| --- | --- | --- | --- |
| 1 | **"Three estimators, three routes through the problem":** random assignment removes systematic confounding by the baseline cause in expectation, conditional on correct implementation. | [Imbens and Rubin (2015)](https://doi.org/10.1017/CBO9781139025751), randomized assignment and potential-outcomes framework. The statement is about the assignment mechanism, not a guarantee of finite-sample balance. | Consistent, with qualification |
| 1 | **"What randomization establishes, and what it does not":** conventional OLS covariate adjustment can worsen precision and conventional standard errors need not provide valid randomization-based uncertainty. | [Freedman (2008)](https://www.sciencedirect.com/science/article/pii/S019688580700005X), original article's findings on bias, precision and standard errors. | Supported |
| 1 | **"What randomization establishes, and what it does not":** fully interacted adjustment can avoid asymptotic precision loss and sandwich uncertainty can be valid under appropriate conditions. | [Lin (2013)](https://doi.org/10.1214/12-AOAS583), original published abstract; specifically asymptotic results. | Supported; asymptotic, not finite-sample guarantee |
| 2 | **"What randomization changes":** random assignment allows intention-to-treat comparisons without assuming a correctly specified outcome regression. | [Imbens and Rubin (2015)](https://doi.org/10.1017/CBO9781139025751), design-based causal inference. This is a design claim conditional on assignment adherence and appropriate measurement. | Consistent, with qualification |
| 2 | **"Adjustment after randomization":** automatic precision gains from un-interacted regression adjustment are not guaranteed. | [Freedman (2008)](https://www.sciencedirect.com/science/article/pii/S019688580700005X). | Supported |
| 2 | **"Adjustment after randomization":** full treatment–covariate interactions and robust uncertainty address key parts of Freedman's critique asymptotically. | [Lin (2013)](https://doi.org/10.1214/12-AOAS583). | Supported; asymptotic limitation essential |
| 3 | **"What a propensity score represents":** true propensity-score adjustment balances observed baseline covariates under the relevant assumptions, not hidden confounders. | [Rosenbaum and Rubin (1983)](https://doi.org/10.1093/biomet/70.1.41), original balancing result. | Supported |
| 3 | **"Why balance must be checked after matching":** propensity-score matching itself is not an automatic guarantee of improved balance or reduced bias. | [King and Nielsen (2019)](https://doi.org/10.1017/pan.2019.11), original critique of commonly used PSM procedures. | Supported; not a universal condemnation of all matching |
| 3 | **"What propensity-score methods can legitimately accomplish":** the ordinary nonparametric bootstrap is not generally valid for nearest-neighbour matching estimators. | [Abadie and Imbens (2008)](https://doi.org/10.3982/ECTA6474), original matching-bootstrap counterexample. | Supported, with estimator-specific scope |
| 4 | **"The causal quantity and the identifying assumptions":** identification with inverse-probability weights requires consistency, conditional exchangeability, positivity and a suitable weight model. | [Cole and Hernán (2008)](https://doi.org/10.1093/aje/kwn164), original overview and assumptions. | Supported |
| 4 | **"Clipping makes the result quieter, not necessarily truer":** weight truncation involves a bias–variance trade-off rather than uniformly improving causal identification. | [Cole and Hernán (2008)](https://doi.org/10.1093/aje/kwn164), discussion of estimated weights and truncation. | Supported; magnitude is simulation-specific |
| 4 | **"Overlap is a feature of the design":** limited overlap can make population-average effects imprecise and motivate analysis in better-supported subpopulations. | [Crump et al. (2009)](https://doi.org/10.1093/biomet/asn055), optimal subsample / overlap treatment. | Supported; target estimand changes on trimming |
| 5 | **"Case two: conditioning on a collider":** conditioning on a common effect can create an otherwise absent association and bias a causal comparison. | [Elwert and Winship (2014)](https://doi.org/10.1146/annurev-soc-071913-043455), original systematic account of endogenous selection bias. | Supported |
| 5 | **"Backdoor paths and adjustment criteria":** admissible controls must be chosen using causal structure; prediction relevance alone does not justify adjustment. | [Pearl (1995)](https://doi.org/10.1093/biomet/82.4.669) and [Cinelli, Forney and Pearl (2024)](https://doi.org/10.1177/00491241221099552), original graphical-control criteria. | Supported |
| 5 | **"Mediators, direct effects and changing the question":** controlling for an intermediate variable may change an estimand from a total to a direct effect, subject to its own assumptions. | [Cinelli, Forney and Pearl (2024)](https://doi.org/10.1177/00491241221099552), distinction among valid and invalid controls. | Consistent; direct-effect identification is not automatic |

**Decision meanings:**

- **Supported:** the cited original publication directly covers the methodological assertion; this does not independently establish numerical results quoted elsewhere in the article.
- **Consistent, with qualification:** the assertion is defensible only with the stated assumptions/scope; not a proof that all readers will interpret those qualifications correctly.
- No record here is labelled "simulation reproduced" or "claim independently derived" without having run the code or worked through a proof.

## Reproducibility claims still open

These quantitative statements are **not closed by the reference audit**. They are synthetic results that need to be compared against the archived experiment scripts, generated data and seeds:

| Part | Article's numerical headline | Verification required |
| --- | --- | --- |
| 1 | At n=10,000: unadjusted coefficient about 2.0005 versus true causal effect 1; adjusted 1.0001, randomized 0.9996. | Re-run the 1,000-replication design for all three sample sizes; reconcile standard errors and tables. |
| 2 | Observational estimate about +1.12 versus true causal effect −1, and randomized estimate near −1. | Re-run all 3,000 simulations; check the estimand and treatment-assignment process. |
| 3 | Matched means 1.0128 (measured-confounding case) versus 2.4294 (hidden confounding), 300 datasets of size 1,500. | Re-run matching, verify caliper/tie handling/replacement policy, and compare balance diagnostics. |
| 4 | Strongest-selection RMSE 0.9013 (ordinary IPW), 0.5892 (Hájek), clipped mean 1.8243. | Re-run weight generation and clipping; confirm normalization and effective-sample-size definitions. |
| 5 | Collider adjustment changes estimate near 1 to near 0.5; mediator adjustment changes total effect near 2 to a direct coefficient near 1. | Re-run four DGPs across three sample sizes, then derive population coefficients algebraically. |

## Editorial follow-up

1. Run the five experiments with the committed code and compare each numerical figure, table and headline to regenerated outputs; record package versions and RNG behavior.
2. Check that all qualifying language (especially *asymptotic*, *given exchangeability*, *under positivity* and *total versus direct effect*) appears next to the relevant prose, not only in a distant limitations section.
3. Extend this claim-level matrix to Parts 6–10 next; do not equate a validated DOI with validated scientific reasoning.
4. After all 20 parts are audited, use the same matrix to prioritize article edits, mathematical derivations and independent computational replications.

## Boundaries

The 15 checks above are an evidence-supported **sample** of the methodological claims in Parts 1–5, not an exhaustive line-by-line review. No source is used to certify the invented-data simulation outputs, and no source is treated as authoritative about a different paper's technical result.
