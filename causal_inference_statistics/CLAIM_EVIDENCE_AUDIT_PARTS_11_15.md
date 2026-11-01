# Causal inference series: claims-to-evidence audit, Parts 11–15

Date: 2026-11-01.

This is the third batch of the claim review. Fifteen selected methodological propositions were checked against primary research sources. **Source support is distinct from reproducing each article's computations.**

## Methodological claims

| Part | Manuscript proposition | Original source and scope | Assessment |
| --- | --- | --- | --- |
| 11 | **"The causal structure":** an intermediate state can be both affected by earlier treatment and a confounder of later treatment–outcome association. | [Naimi, Cole and Kennedy (2017)](https://doi.org/10.1093/ije/dyw323), treatment–confounder feedback. | Supported |
| 11 | **"The two-treatment g-formula":** sequential g-computation integrates over the intervention-dependent intermediate-state distribution. | [Keil et al. (2014)](https://doi.org/10.1097/EDE.0000000000000160), parametric g-formula and worked example. | Supported under sequential exchangeability/positivity |
| 11 | **"Why a conventional adjusted regression answers another question":** conventional adjustment can bias the total effect when a time-dependent confounder is affected by prior treatment. | [Daniel et al. (2013)](https://doi.org/10.1002/sim.5686), time-dependent confounding; [Naimi et al. (2017)](https://doi.org/10.1093/ije/dyw323). | Supported, design-dependent |
| 12 | **"What an average treatment effect leaves out":** heterogeneous effects may call for a different allocation decision from using average effects alone. | [Manski (2004)](https://doi.org/10.1111/j.1468-0262.2004.00530.x), statistical treatment rules. | Supported |
| 12 | **"Evaluating the allocation decision":** a treatment rule should be evaluated by policy value under constraints, not merely by outcome-risk discrimination. | [Athey and Wager (2021)](https://doi.org/10.3982/ECTA15732), policy learning with application constraints. | Supported; observational policy evaluation needs identification |
| 12 | **"Model complexity and targeting":** honest estimation and attention to heterogeneity affect treatment-effect partitioning. | [Athey and Imbens (2016)](https://doi.org/10.1073/pnas.1510489113), original causal-tree method. | Supported; not a validation of this article's simulated rankings |
| 13 | **"Orthogonality and second-order error":** orthogonal scores mitigate first-order sensitivity to nuisance-estimation errors. | [Chernozhukov et al. (2018)](https://doi.org/10.1111/ectj.12097), DML theory. | Supported; rate/regularity conditions apply |
| 13 | **"Why cross-fitting is used":** cross-fitting separates nuisance estimation from score evaluation to mitigate overfitting-related bias. | [Chernozhukov et al. (2018)](https://doi.org/10.1111/ectj.12097), explicit DML ingredient. | Supported as methodology; implementation not rerun |
| 13 | **"Prediction performance is a separate metric":** predictive accuracy of nuisance functions alone does not establish accuracy of the target causal parameter. | [Chernozhukov et al. (2018)](https://doi.org/10.1111/ectj.12097), orthogonal moment and target-parameter inference. | Consistent; quantitative comparisons remain open |
| 14 | **"Why a Bayesian likelihood does not change that difference":** posterior updating alone does not identify an intervention effect when observational assignment assumptions are invalid. | [Rubin (1978)](https://doi.org/10.1214/aos/1176344064), causal effects and the role of assignment; [Dawid (2000)](https://doi.org/10.1080/01621459.2000.10474210), causal identification. | Supported, contingent on causal assumptions |
| 14 | **"Posterior credible intervals and repeated-sampling coverage":** posterior probability and repeated-sampling coverage are different notions. | [Bayarri and Berger (2004)](https://doi.org/10.1214/088342304000000116), Bayesian–frequentist comparison. | Supported |
| 14 | **"Priors over parameters and priors over causal structures":** feedback in a jointly estimated Bayesian propensity model can distort causal estimation. | [Zigler et al. (2013)](https://doi.org/10.1111/j.1541-0420.2012.01830.x), original model-feedback investigation. | Supported for the specific feedback setting |
| 15 | **"Why ordinary adjustment is not enough":** treatment–confounder feedback can invalidate an ordinary conditional-regression interpretation of sustained total effects. | [Mansournia et al. (2017)](https://doi.org/10.1136/bmj.j4587), original practical analysis; [Naimi et al. (2017)](https://doi.org/10.1093/ije/dyw323). | Supported |
| 15 | **"Sequential inverse probability weighting":** properly constructed treatment-history weights can be used for marginal structural models under identifying assumptions. | [Cole and Hernán (2008)](https://doi.org/10.1093/aje/kwn164), weight construction and practical positivity. | Supported, conditional on correct models |
| 15 | **"Parametric G-computation":** g-formula and inverse-probability methods target regime-specific contrasts but can differ in finite samples or under model misspecification. | [Naimi et al. (2017)](https://doi.org/10.1093/ije/dyw323), overview of g-methods. | Supported as distinct estimation routes |

## Concrete correction: Part 12

The Part 12 bibliography contained a purported **Chernozhukov, Demirer, Duflo and Fernández-Val (2023)** article in *Econometrica* **91(1), 1–31**, DOI **10.3982/ECTA19837**. Its ledger entry had been marked verified using an NBER working-paper link, which does not substantiate those journal metadata. The actual final journal article is the **2025 Fisher–Schultz Lecture**, *Econometrica* **93(4), 1121–1164**, DOI [10.3982/ECTA19303](https://onlinelibrary.wiley.com/doi/full/10.3982/ECTA19303), already correctly cited by Part 20.

This PR replaces the false bibliographic entry in Part 12 with the existing 2025 version of record, removes the unsupported duplicate ledger entry, and changes the Part 12 occurrence ID from `dff681c610007233` to the canonical `dd8da7113f2d5e6d`. Consequently, the ledger contains **209 distinct references** (all with a source-backed record) and **300 occurrence mappings**. A working-paper date and a final published journal citation must not be conflated.

## Numerical and implementation checks not yet closed

| Part | Independent reproduction required |
| --- | --- |
| 11 | Re-run 300 datasets; verify the two-treatment g-formula integration against the DGP and contrast with the naive regression. |
| 12 | Re-run 400 train–test replications; verify treatment-rule capacity, oracle value, and estimated policy values without test-set leakage. |
| 13 | Inspect the nuisance-feature basis and existing addendum, then independently rerun three-fold cross-fitting and both DGPs; check the new claims against archived outputs. |
| 14 | Re-run 5,400 conjugate posterior fits, explicitly verifying the prior parameterization, interval coverage and structural truth. |
| 15 | Re-run sustained-regime weighting, parametric g-computation and the saturated MSM addendum; confirm weight denominators and the identified estimand. |

This is a **sampled claim audit**, not proof of every assertion and not a computational reproduction. The next source audit should cover Parts 16–20; afterwards, prioritize full numerical validation and manuscript corrections.
