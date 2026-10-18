# The G-Formula: Why Adjusting for a Time-Varying Confounder Can Miss the Total Effect

*Part 11 of Causal Inference in Statistics. Sequential g-computation, treatment-confounder feedback, and a controlled experiment in which the true effect of two interventions is known. Three hundred independent datasets, three estimation approaches, and a comparison between conditioning on an intermediate variable and allowing it to evolve under intervention. Python code provided.*

## The decision

An evaluation of a programme delivered once is already difficult when participation depends on prognosis. Many interventions are not delivered once. A person receives an initial treatment, their condition changes, and a second treatment decision is made using information that has itself been affected by the first. In health care this can describe treatment intensification after a clinical measurement; in education, additional instruction after an interim assessment; and in employment services, further support after a period of unsuccessful job search. The outcome at the end depends not just on the two treatment decisions but also on how the intermediate state evolved between them.

This feedback poses a problem for the instinctive instruction to control for all confounders. The intermediate measurement may confound the relationship between later treatment and outcome, because it affects both. But the same measurement may be a consequence of earlier treatment. Conditioning on it in a conventional outcome regression may therefore block a pathway through which the earlier treatment works. Leaving it out, however, can confound the estimated effect of the second intervention. Both choices have a plausible statistical rationale; neither, by itself, solves the longitudinal causal question.

The g-formula approaches the problem differently. Rather than asking for one treatment coefficient after holding every measured variable constant, it computes the outcome that would be expected under a specified treatment history. It does this by combining a model for the evolution of intermediate variables with a model for the final outcome, then averaging the predicted outcomes under the intervention of interest. The method can be interpreted as standardization extended to a longitudinal system.

I investigate this with a synthetic two-decision process in which the total intervention effect is exactly 3.36. The first treatment changes an intermediate prognostic variable; that variable affects the second treatment and the outcome. The benchmark compares correctly specified sequential g-computation, a misspecified outcome model that omits the intermediate variable, and a conventional regression contrast that conditions on the intermediate variable and adds the two treatment coefficients. The three methods estimate markedly different quantities. The point is not that regression is inherently unreliable; it is that longitudinal causal effects cannot always be read from coefficients in a conditional regression.

## The causal structure

Let L0 denote baseline prognosis, A0 the first treatment decision, L1 the intermediate measurement, A1 the second treatment decision and Y the final outcome. The chronological order is L0, A0, L1, A1, Y. The assumed causal structure contains arrows from L0 into the first treatment and subsequent prognosis; from A0 into L1 and Y; from L1 into A1 and Y; and from A1 into Y. Other arrows can be present without changing the central feedback problem, but these are sufficient for the controlled example.

The intermediate variable L1 is both a mediator of the earlier treatment effect and a confounder of the later treatment-outcome relationship. The two descriptions concern different causal comparisons and are simultaneously correct. For the later treatment A1, L1 is measured before assignment and predicts the final outcome. For the earlier treatment A0, L1 is downstream and participates in the total effect. A single rule that classifies a variable as either a good or bad control without referring to treatment timing is inadequate.

Under a strategy that sets A0 to a0 and A1 to a1, the target is the mean potential outcome E[Y(a0,a1)]. I compare the static strategies (1,1) and (0,0). The corresponding total regime effect is E[Y(1,1)]−E[Y(0,0)]. It is not generally identical to the coefficient of A0 plus the coefficient of A1 in an outcome regression conditional on L1, because such a regression holds fixed a variable that the first intervention would change.

The estimand itself is an important restriction. We are not asking which individuals received treatment twice in the historical data and how their outcomes differed from those who received neither treatment. Selection into both decisions depends on prognosis. Nor are we asking about an adaptive rule that changes treatment when a measured threshold is exceeded. Those dynamic regimes are possible targets for g-methods, but this first benchmark uses two static strategies so the structural calculation can be inspected directly.

## Defining the data-generating system

For each individual, draw baseline L0 and independent disturbances from standard normal distributions. The first treatment has probability expit(0.8L0), where expit(z)=1/(1+exp(−z)). The intermediate variable is then

\[
L_1=0.8L_0+1.2A_0+\varepsilon_L.
\]

The second treatment occurs with probability expit(0.8L1). Finally, the outcome is

\[
Y=1+0.5L_0+1.3L_1+0.7A_0+1.1A_1+\varepsilon_Y.
\]

The disturbances are independent of one another and of baseline prognosis. Treatment assignments are generated using independent random draws conditional on the measured history. There is no unmeasured confounding in this constructed system. Treatment positivity holds at each finite value of the relevant prognosis, although the logistic probabilities can approach zero or one in extreme tails. Every individual has a fully observed history and outcome.

The coefficients were chosen to make all three causal channels visible. The first intervention directly raises Y by 0.7. It also raises L1 by 1.2, which raises Y by 1.3 per unit. That mediated component is 1.56. The second intervention directly raises Y by 1.1. Because the outcome is linear and the intervention effect is constant, the total contrast between setting both treatments to one and setting both to zero is

\[
\tau=0.7+(1.2)(1.3)+1.1=3.36.
\]

The observational allocation mechanisms do not change this structural truth. The first assignment depends on L0 and the second on L1, but intervening sets A0 and A1 rather than conditioning on their naturally occurring values. Under an intervention at time zero, the intermediate measurement follows a new distribution; that is the detail conventional adjusted treatment coefficients can miss.

## What the observational history can identify

The g-formula relies on causal assumptions that go beyond fitting conditional models. First, the intervention must be sufficiently well defined that observed outcomes under the actual treatment history correspond to the appropriate potential outcomes: a form of consistency. Second, treatment decisions must be sequentially exchangeable given the measured history. In this example, the first treatment is independent of all relevant future potential outcomes conditional on L0, and the second is independent of the potential outcomes under alternative second-stage interventions conditional on L0, A0 and L1. Third, each treatment choice in the proposed strategy must have positive probability for histories to which the formula applies.

These assumptions are not validated by a low prediction error. Sequential exchangeability says that there is no additional, unmeasured factor which changes a treatment decision and relevant potential outcomes once the measured history is accounted for. In the simulation this holds because we created the assignment mechanisms. In a real observational study it requires substantive justification, attention to temporal ordering, and often sensitivity analysis.

Positivity is equally consequential. If a high-risk individual could never receive one of the required interventions, outcomes under that treatment history would not be identified from the observed data without further structural assumptions. The mathematical expression for the g-formula might still be written down, but applying a fitted model in an unsupported region would be extrapolation rather than empirical identification.

The goal is not to estimate a complete scientific theory of every variable. Identification says that, under the assumptions, the intervention mean can be expressed as a functional of the observed longitudinal distribution. Estimation then approximates that functional from finite data. A flexible or well-fitting nuisance model does not replace the first step.

## The two-treatment g-formula

For the static regime (a0,a1), the longitudinal g-formula can be written in this discrete-time structure as

\[
E[Y(a_0,a_1)] = \int\!\int E[Y\mid L_0=l_0,A_0=a_0,L_1=l_1,A_1=a_1] \, f(l_1\mid l_0,a_0) f(l_0)\,dl_1\,dl_0.
\]

The formula contains two distinct components: the conditional outcome model and the distribution of the intermediate variable under the first treatment. A causal prediction under the regime must allow L1 to take the values it would take if A0 were set to a0. Merely entering a0 and a1 into an outcome model while retaining each person's observed L1 does not generally implement that operation.

In the synthetic process, the conditional outcome mean is linear in L0, L1 and the two treatments. The conditional mean of L1 given L0 and A0 is also linear. Consequently, its integration can be performed analytically. Substituting E[L1|L0,A0=a0]=0.8L0+1.2a0 into the conditional outcome expectation gives

\[
E[Y(a_0,a_1)\mid L_0]=1+0.5L_0+1.3(0.8L_0+1.2a_0)+0.7a_0+1.1a_1.
\]

After averaging over the baseline population, the L0 terms have mean zero. The intervention contrast is therefore 2.26a0+1.1a1 relative to the zero-zero regime. In particular the contrast for treating at both times is 3.36. This computation gives an independent mathematical check on the numerical experiment.

The formula can also be evaluated by simulation rather than analytic integration when the intermediate process is more complicated. One generates or predicts intermediate histories under each proposed strategy, predicts outcomes conditional on those histories and treatment assignments, and averages the resulting potential outcomes over the target baseline distribution. Such Monte Carlo g-computation is not equivalent to simulating arbitrary outcomes from a predictive model; the sequence of interventions and covariate evolution must follow the specified causal ordering.

## Why a conventional adjusted regression answers another question

Consider fitting an outcome regression of Y on L0, A0, L1 and A1. In this experiment, that model is correctly specified for the observed conditional mean. It should estimate the coefficient of A0 near 0.7 and that of A1 near 1.1. Adding those two coefficients gives about 1.8. That number describes a conditional contrast holding L1 fixed, not the total outcome change when A0 causes L1 to evolve differently.

The missing 1.56 is not random model error. It is the contribution of the pathway A0→L1→Y. The regression includes L1, so the coefficient of A0 excludes the part of the earlier treatment's effect carried through L1. If the research question concerns a controlled direct effect with the intermediate measurement fixed under a plausible intervention, a coefficient can have a relevant interpretation under additional assumptions. But using the same coefficient sum for the total effect changes the target without explaining the change.

There is another reason why simply removing L1 is problematic. The second treatment depends on L1, and L1 also affects Y. If L1 is omitted from the outcome model, the estimated effect of A1 can absorb some of the prognostic difference between individuals who did and did not receive the later treatment. Removing a mediator for A0 simultaneously removes a confounder for A1. This is the treatment-confounder feedback problem associated with longitudinal causal inference.

The two common regression choices therefore pull in opposite directions. Including L1 blocks part of the total A0 effect if one naively sums conditional coefficients. Excluding L1 leaves an important determinant of A1 assignment and Y unaccounted for. G-computation uses L1 in the outcome and evolution models while allowing its distribution to change under a0. That distinction is the source of its success in the correctly specified benchmark.

## Fitting the sequential models

I use ordinary least squares to estimate the conditional evolution of L1 from L0 and A0. I then fit a separate linear outcome regression using L0, A0, L1 and A1. Both models correspond to the actual structural conditional means. For each baseline observation in the dataset, the procedure predicts L1 under A0=1 and under A0=0, then passes those predicted intermediate values to the outcome model while setting A1 to the required strategy value. The resulting predictions are averaged across the same baseline observations, yielding regime-specific standardized outcome means.

The comparison is the estimated mean under (1,1) minus the estimated mean under (0,0). The procedure is repeated separately for every independently simulated dataset. Its error therefore includes finite-sample variation from estimating both the intermediate and outcome regressions, not simply outcome noise in a single fitted model.

Because the conditional mean of Y is linear in L1, replacing the intermediate distribution with its predicted conditional mean gives the correct integrated mean in this particular system. In a nonlinear outcome model, plugging in an average L1 could be wrong: generally E[m(L1)] differs from m(E[L1]). A more general implementation would simulate L1 repeatedly under each treatment regime and average the outcome predictions, or perform suitable numerical integration over the estimated conditional distribution.

This limitation is part of the method, not an implementation detail to be ignored. Longitudinal g-computation requires evaluating the distribution of evolving covariates under each intervention. The simple code here exploits a property of the chosen linear structural system, and should not be transferred unchanged to nonlinear data without checking whether the integration step remains valid.

## The two comparison procedures

The first comparison deliberately misspecifies the outcome model by excluding L1. It retains the same model for the evolution of L1, but the predicted intermediate history no longer enters the final outcome prediction. Treatment at A0 and A1 is still set under each regime, and the fitted conditional outcome predictions are averaged. The exercise is not intended as a different valid g-formula; it is a test of what happens when the outcome model fails to condition on a time-varying prognostic variable needed for the sequential calculation.

The second comparison fits the correctly specified outcome regression including L1, but reports only the sum of the two fitted treatment coefficients. This procedure corresponds to a familiar interpretation of a regression with several exposure variables: add the coefficients to obtain the effect of receiving both treatments. Because L1 is held fixed by the conditional regression, this sum does not incorporate the effect of the first treatment on the intermediate state.

The comparisons isolate different mistakes. Omitting L1 changes the fitted outcome relationship and creates confounding for the later exposure. Summing conditional coefficients mistakes a controlled conditional contrast for a total longitudinal regime effect. Their biases are not expected to have the same sign, and the simulation should not be reduced to the claim that one always overestimates while the other always underestimates.

All three procedures rely on the same observed histories and the same sample size. Their differences do not come from differing dataset sizes, outcome definitions or tuning budgets. Each is an ordinary linear-regression calculation, so the important difference is the counterfactual quantity each procedure reconstructs—or fails to reconstruct.

## Monte Carlo design

I generate 300 independent datasets, each containing 2,000 individuals. The pseudorandom generator uses seed 20261017. Within each dataset, I first generate baseline prognosis and treatment A0, then the intermediate variable L1, treatment A1 and the final outcome. The two conditional model-fitting steps use only the simulated observational data; the true structural coefficients are not supplied to the estimators.

The true regime contrast is derived from the structural equations before comparing estimates. For each estimator, the program records the difference between its estimated regime or coefficient contrast and 3.36. The reported summaries are the average estimate, bias relative to 3.36, standard deviation across datasets, root mean squared error, and fifth and ninety-fifth percentiles of the simulated estimates.

These percentiles describe the empirical distribution across simulated datasets. They are not confidence interval endpoints for a particular fitted study. The experiment does not implement an influence-function variance estimator, longitudinal bootstrap confidence interval or weak-overlap sensitivity analysis. Those would be appropriate extensions, but they should not be claimed based on the Monte Carlo summaries alone.

The full script is provided rather than a few illustrative lines of modelling code. It generates both treatment decisions and the intermediate history from the stated equations, estimates the nuisance regressions, constructs the intervention comparisons and writes the aggregate results. Reproduction does not require an external dataset or a dedicated causal-inference library, although such libraries can be useful for more general longitudinal problems.

## The estimated total effects

The results are:

| Estimator | Mean estimate | Bias | Standard deviation | RMSE |
| --- | ---: | ---: | ---: | ---: |
| Correct sequential g-computation | 3.3671 | +0.0071 | 0.0895 | 0.0896 |
| G-computation with L1 omitted from outcome model | 4.1142 | +0.7542 | 0.1025 | 0.7611 |
| Sum of conditional treatment coefficients | 1.8066 | −1.5534 | 0.0725 | 1.5551 |

The correctly specified sequential procedure recovers the true effect of 3.36 with a mean error of about seven thousandths. Its between-dataset variation is modest relative to the effect itself, and its RMSE is approximately 0.09. This is what the population g-formula and the fitted models predict under the constructed identifying assumptions.

The misspecified outcome model produces a mean estimate of 4.1142. Its error is roughly three quarters of an outcome unit, despite being fitted to the same sample histories. The procedure does not recover the counterfactual consequence of the first intervention correctly because the outcome model omits a variable needed to represent how treatment and prognosis jointly determine the observed outcome.

The conditional-coefficient contrast averages 1.8066, close to the direct coefficients' structural sum of 0.7+1.1=1.8. This result is not a numerical malfunction. The conditional regression is fitting its own target correctly. The scientific mistake would be claiming that the conditional sum represents the full longitudinal treatment effect of 3.36.

The contrast between these two failures matters more than choosing the estimator with the smallest numerical error. One procedure attempts g-computation with an insufficient outcome model, while the other uses a valid conditional model but extracts an estimand that excludes the mediated pathway. Statistical model validity and causal estimand validity are distinct questions.

## Why the first intervention has a larger total effect

The first treatment changes the immediate outcome contribution by 0.7, but it also changes the next-period prognosis L1 by 1.2. Since L1 influences final outcome with coefficient 1.3, the pathway through the intermediate state contributes an additional 1.56. The total effect of A0 holding the second treatment strategy fixed is therefore 2.26. The direct effect of A1 adds 1.1, producing the total 3.36.

An outcome regression holding L1 fixed cannot include that 1.56 component in the coefficient of A0, because comparing A0 values conditional on the same L1 removes the difference in intermediate prognosis induced by A0. This is not a subtle Monte Carlo phenomenon. It follows from the structural equations and is visible even with infinite data.

The result illustrates why time-varying covariates have to be interpreted relative to the intervention strategy. For the second treatment, L1 is pre-treatment history and a plausible confounder. For the first treatment, it is a downstream variable through which effects can propagate. A statistical method that merely distinguishes baseline covariates from outcomes is not sufficient when exposures are repeatedly assigned over time.

The g-formula does not ignore the intermediate variable or treat it as an ordinary baseline control. It uses the measured history to predict outcomes and then changes the distribution of that history according to the specified intervention. That is the substantive operation behind its mathematical expression.

## The cost of omitting the intermediate measurement

Why does the misspecified g-computation estimate overshoot 3.36 rather than simply return the direct-effect sum 1.8? The answer is that omitting L1 is not equivalent to conditioning on it and removing a mediated path. It changes the fitted conditional outcome relationship. The omitted L1 is associated with A0 because A0 affects it, and associated with A1 because A1 assignment depends on it. It also directly contributes to Y. Regression coefficients for treatment can therefore absorb associations arising from the unmodelled intermediate process.

The program fits separate outcome predictions for the A1 treatment groups under the misspecified model, but those groups differ in their observed L1 distribution. A1 then acts partly as a proxy for prognosis. This induces a different sort of error from simply extracting conditional direct coefficients in the correctly specified model.

This distinction is important in applied longitudinal evaluations. One might conclude that intermediate biomarkers, adherence measurements or updated risk scores should be omitted because they are affected by past treatment. That could be appropriate for certain simple total-effect regressions under restrictive designs, but it is not a general solution when the same variables also influence subsequent treatment assignment. A longitudinal method must deal with both causal roles.

The direction of the misspecification bias depends on the treatment assignment mechanism, outcome relation and interactions. Our numerical mean of 4.1142 is a result of the chosen equations, not an expected universal penalty for omitting a time-varying confounder. More extreme selection, weaker overlap, nonlinear prognosis or treatment effect heterogeneity could change both sign and magnitude.

## Sequential exchangeability as an identifying claim

The parametric calculation would remain mathematically executable if an important time-varying common cause were omitted from the observed dataset. The regression software would still return coefficients and a g-computation program would still generate counterfactual predictions. Yet the causal interpretation would no longer follow simply from fitting those models.

Sequential exchangeability requires the observed history at each treatment time to contain enough information to account for how the actual treatment decision relates to future potential outcomes. For A0, that history is the baseline information; for A1, it includes prior treatment and the intermediate measurement. The assumption is stronger than saying the fitted propensity model has acceptable discrimination or the outcome model predicts well.

Suppose an unobserved severity indicator influences both A1 and Y after L1 is recorded. Even a perfectly specified model for E[Y|L0,A0,L1,A1] as a statistical conditional expectation might not recover the intervention contrast if A1 remains associated with potential outcomes within those observed histories. The g-formula is an identification result conditional on assumptions, not an automatic causal interpretation of a sequence of fitted models.

In a real employment intervention, decisions about renewed support might depend on informal caseworker judgements, the number of unsuccessful applications, household constraints or employer contacts that are not recorded. Those missing causes could undermine the interpretation. The simulation avoids them by construction, allowing us to examine estimation separately from identification.

## Positivity along treatment histories

A longitudinal strategy requires adequate support for every relevant treatment decision conditional on the evolving history. In the example, the first and second treatment probabilities follow logistic rules, so each option has nonzero probability for every finite value of the conditioning variable. Formal positivity therefore holds.

But practical support can still become sparse. If L1 is extreme, almost everyone with that prognosis may receive the same second treatment. Estimating what would happen under the alternative then requires extrapolating the conditional outcome relationship into regions with limited data. This issue becomes more severe as the number of treatment periods grows, because the number of possible histories increases quickly.

The simulation uses 2,000 individuals and two decisions, with moderate assignment coefficients and correct linear outcome models. It is a favourable setting. If the assignment probabilities were more extreme or there were many treatment stages, even a correctly specified g-formula could be numerically fragile or strongly model-dependent in poorly supported regions.

A longitudinal evaluation should therefore inspect support for the strategies under consideration, not just marginal counts of who received each treatment. Having substantial numbers treated at time zero and time one does not necessarily mean that the specific history (a0,a1) is observed frequently across the prognostic states required by the target population.

## Static regimes and adaptive treatment rules

The benchmark compares two static treatment histories: always treat at both decision times, or never treat. In many applied settings, a static regime is neither operationally natural nor scientifically optimal. Treatment may be assigned according to the patient's or customer's evolving state. For example, a second intervention might be triggered only when an intermediate risk score exceeds a threshold.

Such a rule can be written as a dynamic strategy in which A1=d1(L0,A0,L1), rather than a constant chosen in advance. The g-formula can evaluate that strategy by simulating intermediate histories under the first intervention and applying the rule to each simulated history. The resulting mean outcome can be compared with that under another dynamic rule.

This is one of the practical motivations for longitudinal causal inference: a decision-maker often needs the consequences of a policy that adapts to observed information, not merely the effect of one exposure measured at baseline. A correct dynamic evaluation must respect the information available at each decision time and avoid using future outcomes or post-decision variables to set earlier treatments.

The present script is not a general dynamic-regime engine. It implements two static histories under a linear process. Extending it to adaptive decisions would require specifying the treatment rule and, for nonlinear evolution or outcomes, integrating over the relevant conditional distributions rather than plugging in expected intermediate values.

## G-computation versus inverse probability weighting

Part 4 of this series investigated inverse probability weighting for a single treatment. Longitudinal inverse probability weights can extend the same logic to repeated decisions: each observed treatment assignment is weighted according to its conditional probability given previous history. Marginal structural models use this principle to estimate regime effects while handling measured time-varying confounding affected by prior treatment.

Sequential g-computation takes an outcome-modelling route instead. It estimates the conditional evolution of covariates and outcomes under the relevant histories, then standardizes over counterfactual trajectories. Both approaches depend on appropriate causal assumptions. Their numerical vulnerabilities differ: outcome g-computation is sensitive to model misspecification and extrapolation, while longitudinal weighting can accumulate large weights across stages, particularly when treatment histories are uncommon.

The methods should not be portrayed as unrelated mathematical tricks. They are alternative estimators of causal functionals identified under related sequential assumptions. Doubly robust longitudinal procedures can combine aspects of outcome and treatment modelling to reduce vulnerability to certain nuisance-model errors, although their guarantees depend on the exact estimator and assumptions.

This article does not benchmark longitudinal weighting against g-computation. It concentrates on the difference between standard conditional regression and evaluating a specified intervention history. A fair numerical comparison with weighting would need appropriate treatment-history probabilities, uncertainty calculations and diagnostics for extreme longitudinal weights.

## Model checking and sensitivity analysis

Correctly specifying the outcome evolution model is central to the favourable result. Here, the conditional mean of L1 is linear and the conditional mean of Y contains exactly L0, A0, L1 and A1. The fitted regressions therefore recover their targets as the sample grows. A real system might contain nonlinear relationships, interactions between treatment and prognosis, thresholds or time-dependent measurement error.

Prediction diagnostics can reveal some forms of misspecification, but they do not validate the causal assumptions. A model can predict outcomes well under the observational treatment distribution and extrapolate poorly when a proposed intervention changes the distribution of intermediate states. Out-of-sample error is useful evidence about prediction, not sufficient evidence about intervention means.

A practical analysis should check fit at each transition, inspect residual patterns, evaluate overlap across histories and compare reasonable alternative specifications. Flexible methods may help approximate nuisance functions, but their use should be accompanied by procedures appropriate to the target causal functional and data dependence. Estimating nuisance models using highly adaptive learning does not turn an unidentifiable quantity into an identifiable one.

Sensitivity analysis should also examine unmeasured time-varying confounding and deviations from consistency or positivity. The simulation cannot adjudicate those assumptions because it constructs a world where they are true. Its role is to show that even in that favourable world, extracting the wrong regression coefficient or omitting an intermediate prognostic variable can produce systematically misleading answers.

## Interpretation of the Monte Carlo uncertainty

The reported standard deviations are empirical variation across independently simulated datasets. For the correct method this value is approximately 0.0895; for the misspecified g-computation estimator, 0.1025; for the conditional-coefficient contrast, 0.0725. The last estimator is numerically the least variable, but it is also the most inaccurate for the specified total-effect estimand.

Its smaller standard deviation is not surprising: it estimates a relatively simple conditional coefficient sum rather than allowing the intermediate prognosis to carry an additional pathway under the intervention. The problem is not that the estimate is noisy. It is that it is stably centred near 1.8 when the total regime effect is 3.36.

This provides another example of a theme established earlier in the series. Precision is a property of an estimator for a particular statistical quantity. Causal validity requires that the quantity agree with the intervention estimand. An estimate may have narrow variability while missing the target by more than one unit.

The empirical fifth and ninety-fifth percentiles, also reported by the script, describe the central part of the Monte Carlo distribution. They should not be presented as confidence intervals with guaranteed frequentist coverage for a single observational study. Deriving valid single-study uncertainty for g-computation requires accounting for estimation of the nuisance models and the standardization step, commonly through appropriate bootstrap or analytic influence-function procedures.

## What this experiment cannot establish

The benchmark uses one linear additive longitudinal system with two binary treatment decisions and one continuous intermediate variable. All relevant confounders are observed, the treatment decisions have known logistic mechanisms, all outcomes are measured, and individuals do not interfere with one another. The true intervention effects are constant rather than heterogeneous.

The correctly specified method benefits from simple linear conditional means and an analytic property that makes predicted intermediate means sufficient for outcome averaging. It does not test Monte Carlo integration over nonlinear intermediate distributions. Nor does it test highly adaptive outcome models, dynamic treatment strategies, longer histories, informative dropout or competing events.

The misspecified outcome model removes L1 entirely rather than making a subtle functional-form error. Its numerical bias is therefore a clear example of model failure, not a realistic distribution of misspecification magnitudes. Other omitted variables or nonlinear functions can produce weaker, stronger or differently signed bias.

The conventional regression comparison is intentionally aligned with the direct coefficients of the structural outcome equation. It is not intrinsically an invalid estimate of every possible direct-effect quantity. It becomes misleading when used as the estimate of the total static-regime contrast defined at the beginning of the experiment.

Finally, the simulation does not establish effectiveness of any actual repeated medical, educational or employment intervention. It demonstrates the mathematical and statistical consequences of specified treatment-confounder feedback. Applied conclusions require separate evidence about the assignment mechanism, treatment definition, target population and counterfactual assumptions.

## What the experiment established

The true total effect of treating at both periods rather than neither period is 3.36. Correct sequential g-computation returned a Monte Carlo mean of 3.3671 and RMSE 0.0896. Omitting the intermediate prognostic variable from the outcome model produced a mean of 4.1142 and RMSE 0.7611. Summing conditional treatment coefficients while holding that variable fixed produced a mean of 1.8066 and RMSE 1.5551.

Both biased procedures are ordinary statistical calculations that can appear reasonable when divorced from the intervention being estimated. The first fails to represent outcome dependence on time-varying prognosis; the second excludes the pathway through which the first treatment changes that prognosis. Sequential g-computation succeeds in this controlled setting because it evaluates how the intermediate state would evolve under the intervention and then averages the resulting outcomes.

The point is not that a sequence of regression models always identifies causal effects. It does so here because consistency, sequential exchangeability, positivity and the specified conditional models all hold by construction. The identification assumptions must be defended before the fitted intervention means can be given a causal interpretation.

For longitudinal decisions, controlling for a variable and allowing it to change under treatment are different operations. Confusing them can produce an estimate that is precise, reproducible and scientifically wrong for the question the analysis was meant to answer.

## Code

All reported numerical results are reproduced by `experiment.py` in `causal_inference_statistics/part11/` of the Medium-Blog repository. The script uses Python 3.12+, NumPy and scikit-learn, fixes seed 20261017, regenerates the 300 datasets and writes `results.csv`. No external dataset is required. It reports mean effects, bias, standard deviation, RMSE and fifth and ninety-fifth percentiles.

## The series

1. When Regression Gets the Causal Effect Wrong
2. Randomization, Confounding and the Value of Experimental Design
3. Matching and Propensity Scores
4. Inverse Probability Weighting
5. Directed Acyclic Graphs and the Cost of Controlling for Everything
6. Difference-in-Differences: When Parallel Trends Fail
7. Instrumental Variables: When a Strong First Stage Is Not Enough
8. Regression Discontinuity: Why a Sharp Cutoff Is Not Enough
9. Synthetic Controls and Counterfactual Time Series
10. Doubly Robust Estimation
11. The G-Formula: Why Adjusting for a Time-Varying Confounder Can Miss the Total Effect (this article)

## References

Bang, H. and Robins, J. M. (2005). Doubly Robust Estimation in Missing Data and Causal Inference Models. *Biometrics*, 61(4), 962–973. https://doi.org/10.1111/j.1541-0420.2005.00377.x

Daniel, R. M., Cousens, S. N., De Stavola, B. L., Kenward, M. G. and Sterne, J. A. C. (2013). Methods for Dealing with Time-Dependent Confounding. *Statistics in Medicine*, 32(9), 1584–1618. https://doi.org/10.1002/sim.5686

Hernán, M. A. and Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC. https://miguelhernan.org/whatifbook

Hernán, M. A., Brumback, B. and Robins, J. M. (2000). Marginal Structural Models to Estimate the Causal Effect of Zidovudine on the Survival of HIV-Positive Men. *Epidemiology*, 11(5), 561–570. https://doi.org/10.1097/00001648-200009000-00012

Keil, A. P., Edwards, J. K., Richardson, D. R., Naimi, A. I. and Cole, S. R. (2014). The Parametric G-Formula for Time-to-Event Data: Towards Intuition with a Worked Example. *Epidemiology*, 25(6), 889–897. https://doi.org/10.1097/EDE.0000000000000160

McGrath, S., Lin, V., Zhang, Z., Petito, L. C., Logan, R. W., Hernán, M. A. and Young, J. G. (2020). gfoRmula: An R Package for Estimating the Effects of Sustained Treatment Strategies via the Parametric G-Formula. *Patterns*, 1(3), 100008. https://doi.org/10.1016/j.patter.2020.100008

Naimi, A. I., Cole, S. R. and Kennedy, E. H. (2017). An Introduction to G Methods. *International Journal of Epidemiology*, 46(2), 756–762. https://doi.org/10.1093/ije/dyw323

Robins, J. M. (1986). A New Approach to Causal Inference in Mortality Studies with a Sustained Exposure Period—Application to Control of the Healthy Worker Survivor Effect. *Mathematical Modelling*, 7(9–12), 1393–1512. https://doi.org/10.1016/0270-0255(86)90088-6

Robins, J. M., Hernán, M. A. and Brumback, B. (2000). Marginal Structural Models and Causal Inference in Epidemiology. *Epidemiology*, 11(5), 550–560. https://doi.org/10.1097/00001648-200009000-00011

Taubman, S. L., Robins, J. M., Mittleman, M. A. and Hernán, M. A. (2009). Intervening on Risk Factors for Coronary Heart Disease: An Application of the Parametric G-Formula. *International Journal of Epidemiology*, 38(6), 1599–1611. https://doi.org/10.1093/ije/dyp192

van der Laan, M. J. and Rose, S. (2011). *Targeted Learning: Causal Inference for Observational and Experimental Data*. Springer. https://doi.org/10.1007/978-1-4419-9782-1

Young, J. G., Cain, L. E., Robins, J. M., O'Reilly, E. J. and Hernán, M. A. (2011). Comparative Effectiveness of Dynamic Treatment Regimes: An Application of the Parametric G-Formula. *Statistics in Biosciences*, 3, 119–143. https://doi.org/10.1007/s12561-011-9040-7