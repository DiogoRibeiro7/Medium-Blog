# Causal inference series: claims-to-evidence audit, Parts 6–10

Date: 2026-10-31.

This is the second batch of the claim audit, extending [the Parts 1–5 review](CLAIM_EVIDENCE_AUDIT_PARTS_01_05.md). It examines 15 **methodological** assertions against the original paper or publisher record. It is not a substitute for checking proofs or independently rerunning the simulations.

## Claim-level evidence matrix

| Part | Claim and manuscript section | Original-source check | Status |
| --- | --- | --- | --- |
| 6 | **The causal estimand:** two-group DID requires parallel evolution of *untreated potential outcomes*, not equal pre-treatment outcome levels. | [Callaway and Sant'Anna (2021)](https://doi.org/10.1016/j.jeconom.2020.12.001) formalizes DID identification for multiple periods, including conditional parallel trends. | Supported under identifying assumptions |
| 6 | **What pre-trend plots can and cannot do:** passing a pre-trend significance test cannot establish future parallel trends. | [Roth (2022)](https://doi.org/10.1257/aeri.20210236) analyzes selection and inference after testing parallel trends. | Supported; an insignificant pre-test is not evidence of equivalence |
| 6 | **Multiple groups and staggered adoption:** a simple two-way-fixed-effects coefficient need not average treatment effects with benign weights under heterogeneous timing/effects. | [Goodman-Bacon (2021)](https://doi.org/10.1016/j.jeconom.2021.03.014), decomposition of timing comparisons; [Callaway and Sant'Anna (2021)](https://doi.org/10.1016/j.jeconom.2020.12.001), group-time estimands. | Supported; not a universal failure of all DID estimators |
| 7 | **Instrument relevance and exclusion are different:** first-stage predictive strength does not establish the exclusion restriction. | [Imbens and Angrist (1994)](https://doi.org/10.2307/2951620) identifies causal effects only under additional instrumental assumptions. | Supported |
| 7 | **Local average treatment effects:** under the IV conditions (including monotonicity), the interpretable effect can apply to instrument-induced compliers rather than the whole population. | [Imbens and Angrist (1994)](https://doi.org/10.2307/2951620), original LATE result. | Supported with monotonicity and exclusion qualifications |
| 7 | **Why the weak instrument produces extreme estimates:** weak first-stage association can make conventional IV coefficient and normal approximation unreliable. | [Staiger and Stock (1997)](https://doi.org/10.2307/2171753), weak-IV asymptotics. | Supported; the particular reported Monte Carlo distribution remains unreplicated |
| 8 | **Potential outcomes at a threshold:** a discontinuity identifies a cutoff-local effect only if the untreated potential-outcome limits satisfy continuity and the design assumptions. | [Hahn, Todd and van der Klaauw (2001)](https://doi.org/10.1111/1468-0262.00183), formal RD identification. | Supported |
| 8 | **Density tests and manipulation:** testing the running-variable density at the cutoff is a diagnostic for sorting, not a proof of continuity or identification. | [McCrary (2008)](https://doi.org/10.1016/j.jeconom.2007.05.005), manipulation density test. | Supported with diagnostic-only scope |
| 8 | **Robust bias correction:** local-polynomial RD inference needs attention to smoothing bias; robust bias-corrected intervals address a well-defined part of the problem. | [Calonico, Cattaneo and Titiunik (2014)](https://doi.org/10.3982/ECTA11757), original robust interval theory. | Supported; no blanket guarantee for misspecified designs |
| 9 | **Why convex combinations are attractive:** synthetic control constructs a treated counterfactual from a weighted donor pool. | [Abadie, Diamond and Hainmueller (2010)](https://doi.org/10.1198/jasa.2009.ap08746), original case-study construction. | Supported |
| 9 | **Why pre-treatment RMSPE is not a validity test:** fit in observed pre-policy periods cannot rule out a treated-specific post-policy shock. | [Abadie (2021)](https://doi.org/10.1257/jel.20191450), requirements, failure modes and feasibility. This is a design-dependent warning, not a mechanical theorem that any low RMSPE is invalid. | Consistent with original framework |
| 9 | **Beyond the classical estimator:** augmenting synthetic controls can address bias from imperfect pre-treatment fit but adds modelling assumptions. | [Ben-Michael, Feller and Rothstein (2021)](https://doi.org/10.1080/01621459.2021.1929245), augmented synthetic control. | Supported with estimator-specific scope |
| 10 | **Augmentation and the doubly robust score:** AIPW-style estimation may remain consistent if either the outcome model or treatment model is correctly specified, under identification and regularity conditions. | [Bang and Robins (2005)](https://doi.org/10.1111/j.1541-0420.2005.00377.x), original doubly robust property. | Supported; not protection against hidden confounding |
| 10 | **AIPW as an orthogonal score:** Neyman orthogonality reduces first-order nuisance-estimation sensitivity. | [Chernozhukov et al. (2018)](https://doi.org/10.1111/ectj.12097), original DML construction. | Supported; nuisance-rate conditions still matter |
| 10 | **Why cross-fitting is used:** sample splitting helps reduce overfitting-related bias when estimating nuisance functions and evaluating scores. | [Chernozhukov et al. (2018)](https://doi.org/10.1111/ectj.12097), orthogonal moments with cross-fitting. | Supported as methodology; code implementation not yet inspected |

## Concrete editorial correction

In Part 8 the section heading previously claimed **"A study with 4,800 datasets"**, while the paragraph immediately underneath correctly specifies **400 independent datasets × 3 scenarios = 1,200 datasets**, with **4 bandwidth fits per dataset = 4,800 fits**. The section heading is corrected to **"A study with 1,200 datasets and 4,800 regression fits"**. No numerical result was changed. This is a terminology/counting correction, not independent reproduction.

## Numerical checks still open

| Part | Numerical result quoted in the manuscript | Required independent check |
| --- | --- | --- |
| 6 | DID means −0.9951, −0.2469 and −0.5977; nominal coverage 95.4%, 0% and 0.2%. | Run 500 replications per DGP and inspect placebo rejection rates; check CI construction and dependence assumptions. |
| 7 | Four IV designs over 2,000 datasets, including a weak first stage and exclusion violation. | Re-run generated structural equations, instrument F statistics, median, RMSE and weak-IV tail behavior. |
| 8 | 1,200 datasets, four bandwidths, 4,800 fits and RMSE around 0.2513 at the narrowest bandwidth in the smooth design. | Re-run 400 datasets for each of three DGPs and reconcile every bandwidth estimate and interval. |
| 9 | One treated unit, eight donors, 24 pre-treatment and 12 post-treatment periods; 900 datasets. | Re-run donor-weight optimization, check constraints, donor composition and each scenario's pre/post RMSPE. |
| 10 | AIPW mean 0.9964 when both models are correct, 1.0182 with only propensity correct, 0.9968 with only outcome correct, and 0.3942 when both are wrong. | Re-run 200 datasets per regime; verify cross-fitting folds are disjoint, nuisance specifications and true ATE. |

## Scope and next steps

- The evidence matrix covers 15 selected theoretical assertions, not every factual statement in the five articles.
- A primary publication supporting a general principle does not establish that a simulated estimator implemented that principle correctly.
- No independent code execution or numerical regeneration was performed in this batch. All five computational checks remain **open**.
- Continue with Parts 11–15, then 16–20, followed by a reproducibility pass across the series and corrections of any demonstrated discrepancies.
