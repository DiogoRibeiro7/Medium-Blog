# Causal inference series: claims-to-evidence audit, Parts 16–20

Date: 2026-11-01.

Fourth and final batch of the **selected methodological-claims review**, after [Parts 1–5](CLAIM_EVIDENCE_AUDIT_PARTS_01_05.md), [6–10](CLAIM_EVIDENCE_AUDIT_PARTS_06_10.md), and [11–15](CLAIM_EVIDENCE_AUDIT_PARTS_11_15.md).

## Claim-level source checks

| Part | Manuscript claim / location | Original publication or authoritative primary source | Result and limitation |
| --- | --- | --- | --- |
| 16 | **"Calibrating against observed covariates":** omitted-variable bias can be parameterized by the strength of unmeasured confounding relative to observed associations. | [Cinelli and Hazlett (2020)](https://doi.org/10.1111/rssb.12348), sensitivity framework. | Supported; bounds depend on substantive calibration assumptions. |
| 16 | **"Sensitivity parameters in practice":** a sensitivity parameter quantifies how strong hidden confounding would need to be to alter an observational result. | [VanderWeele and Ding (2017)](https://doi.org/10.7326/M16-2607), E-value methodology, which is a distinct parameterization from partial-R² sensitivity. | Supported as a general concept; do not interchange E-values and regression robustness values. |
| 16 | **"Why larger samples do not remove hidden bias":** precision is not a substitute for exchangeability when treatment and outcome share an omitted cause. | [Cinelli and Hazlett (2020)](https://doi.org/10.1111/rssb.12348), omitted-variable-bias algebra. | Supported, conditional on the assumed structural problem. |
| 17 | **"Negative-control outcomes and negative-control exposures":** these two categories are diagnostic tools whose validity depends on specific causal relationships. | [Lipsitch, Tchetgen Tchetgen and Cohen (2010)](https://doi.org/10.1097/EDE.0b013e3181d61eeb), original paper. | Supported. |
| 17 | **"What falsification can establish":** a passing negative-control test does not prove absence of all unmeasured confounding. | [Lipsitch et al. (2010)](https://doi.org/10.1097/EDE.0b013e3181d61eeb), conditions and scope of negative controls. | Consistent; absence of detectable control association is not identification. |
| 17 | **"A failed test is not a calibrated estimate of bias":** negative-control associations can flag residual problems without supplying a universal magnitude correction. | [Flanders et al. (2011)](https://doi.org/10.1097/EDE.0b013e3181fdcabe), residual-confounding diagnostic. | Supported as a diagnostic, not as a proof of zero residual bias. |
| 18 | **"What MAR does and does not say":** whether missingness is ignorable depends on the model and observed information; complete-case analysis is not invariably biased. | [Rubin (1976)](https://doi.org/10.1093/biomet/63.3.581), formal missing-data framework. | Consistent under correctly specified conditional model; marginal effects need additional care. |
| 18 | **"MNAR and the identification gap":** an MNAR mechanism cannot generally be identified from the observed-data distribution alone. | [Molenberghs et al. (2008)](https://doi.org/10.1111/j.1467-9868.2007.00640.x), observationally equivalent MAR and MNAR models. | Supported. |
| 18 | **"Missing data and the role of causal diagrams":** selecting complete cases can induce selection bias depending on the observation process. | [Hernán, Hernández-Díaz and Robins (2004)](https://doi.org/10.1097/01.ede.0000135174.63482.43), structural selection-bias account. | Supported; selection mechanism matters. |
| 19 | **"The experimental and target questions":** a randomized-trial ATE need not equal an effect in a population with a different effect-modifier distribution. | [Dahabreh et al. (2020)](https://doi.org/10.1002/sim.8426), extending randomized-trial inference to new populations. | Supported. |
| 19 | **"Why conditioning can support transport":** standardization/weighting can transport treatment effects only under appropriate trial-to-target exchangeability, positivity and measurement assumptions. | [Westreich et al. (2017)](https://doi.org/10.1093/aje/kwx164), inverse odds of sampling weights. | Supported with assumptions. |
| 19 | **"The third simulation: only partial common support":** lack of covariate support limits which target-population effects can be learned from the trial without extrapolation. | [Dahabreh et al. (2020)](https://doi.org/10.1002/sim.8426), transport assumptions and identification. | Supported; numerical scenario unreplicated. |
| 20 | **"A causal estimand and a decision estimand":** maximizing expected policy welfare is a distinct decision problem from estimating a population ATE. | [Kitagawa and Tetenov (2018)](https://doi.org/10.3982/ECTA13288), empirical welfare maximization. | Supported. |
| 20 | **"Off-policy evaluation and identification":** evaluation from historical actions requires assumptions about assignment and outcomes; doubly robust methods can combine outcome and propensity information. | [Dudík, Langford and Li (2011)](https://icml.cc/2011/papers/554_icmlpaper.pdf), original off-policy evaluation and learning. | Supported for the stated estimator class and assumptions. |
| 20 | **"Why a correct risk model can make the wrong decision":** targeting by expected outcome risk is not equivalent to maximizing heterogeneous treatment benefit. | [Athey and Wager (2021)](https://doi.org/10.3982/ECTA15732), observational policy learning; [Kitagawa and Tetenov (2018)](https://doi.org/10.3982/ECTA13288), welfare objective. | Consistent; this article's quantitative policy ranking remains unreplicated. |

## Independent computational verification remains open

| Part | Quantitative or implementation claim to reproduce |
| --- | --- |
| 16 | Four hidden-confounding strengths; 400 datasets each, 3,200 fitted regressions in total; observed-only mean increasing toward 1.749 while oracle remains near one. Check analytical OVB coefficient against simulation. |
| 17 | Three assignment mechanisms and two falsification checks across 1,500 datasets; verify negative-control and placebo rejection frequencies under each DGP. |
| 18 | Three missingness mechanisms, 900 datasets, four estimators; verify MAR/MNAR DGPs, observation weights, conditional versus marginal estimands and error measures. |
| 19 | Re-run the updated reproducibility-correction design and validate population-standardized effects, hidden-modification bias and partial-support labels. |
| 20 | Re-run 500 randomized trials with held-out evaluation; verify the treatment budget, oracle policy value, targeting value, regret and stress-test addendum. |

## Series-wide status and risk boundaries

- **20/20 manuscripts sampled** for claim-to-source support.
- **60 selected methodological propositions** checked (15 in each five-part batch). These are **not all** factual or causal claims in the manuscripts.
- **20 distinct article-level computational replication gates** remain open. The count is an audit worklist, not a claim that none of the repo's existing CI/tests have ever run.
- Bibliography ledger after the Part 12 correction contains **209 distinct source-linked entries and 300 occurrences**. Source links establish publication identity, not passage-level support or implementation correctness.
- This first pass found and corrected a misleading numerical heading in Part 8 and erroneous *Econometrica* metadata in Part 12. It does not rule out further issues.

## Reproduction priorities

1. **Priority 1 — identification/estimand correctness:** Parts 11, 15, 18 and 19 (feedback, missingness and transport), including the already present audit addenda.
2. **Priority 2 — numerical contradictions and fit/variance behavior:** Parts 6–10, 13 and 16–17; verify the script outputs, sample counts and interval construction.
3. **Priority 3 — decision value and external holdouts:** Parts 12 and 20; inspect policy optimization, oracle comparisons and test-set isolation.
4. **Priority 4 — finish all remaining examples:** Parts 1–5 and 14; verify claims in text against regenerated figures/CSV.

Each computational audit should record command, environment, code commit, random seed, output hashes, observed-versus-reported values, and pass/fail thresholds. Do not convert a source-backed literature check into a simulated-result verification by inference.
