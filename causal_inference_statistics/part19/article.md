# External Validity and Transportability: When a Valid Trial Answers the Wrong Population Question

*Part 19 of Causal Inference in Statistics. A known-effect experiment comparing randomized trial averages, standardization to a target population, hidden effect modification, and loss of common support. Python code provided.*

## The population behind the estimate

A randomized experiment can identify the effect of offering a treatment to the people enrolled in the experiment and still provide an incomplete answer to the decision a policymaker faces. The distinction is not necessarily about the quality of randomization. It arises because the population in which the effect was identified is not always the population in which the intervention is intended to operate.

Suppose a programme is evaluated among relatively young participants, but the eventual policy covers an older population. If the programme benefits people of different ages by different amounts, an internally valid estimate from the trial need not be the effect relevant to that policy. The trial answers a causal question correctly for its participants. The policy asks a different causal question about a different distribution of people.

Transportability concerns whether and how information from one population can identify an intervention effect in another. It is tempting to interpret this as a matter of reweighting demographic averages. Reweighting can be valuable, but only when the relevant variables explaining effect differences have been measured, suitable treatment comparisons are available, and the causal mechanisms are stable enough for transport.

The experiment in this article separates three problems that are often conflated. In the first, the effect modifier is observed and the outcome model is correctly specified. Standardizing conditional effects from the randomized trial to the target population recovers the target average. In the second, the decisive effect modifier differs across populations but is missing from the analysis. Standardization on a different observed variable fails despite impeccable randomization inside the trial. In the third, the target population includes values of the effect modifier absent from the trial. A correctly specified linear model extrapolates successfully in this controlled simulation, but the evidence for the unsupported region rests on its structural functional-form assumption rather than local trial comparisons.

Across 400 simulated datasets in each scenario, the true intervention effects are known. The transported estimate averages 2.0041 for a target effect of 2 when the modifier is observed. It averages only 1.0015 when the relevant modifier is omitted, again for a target effect of 2. In the limited-overlap scenario the full target effect is 2.5; the model-based extrapolation averages 2.4927. Restricting that last target to the region represented in the trial estimates a different effect, 1.75, and recovers approximately 1.7473. These comparisons are numerical illustrations of identification and estimand choice, not empirical estimates of any particular programme.

## The experimental and target questions

For individual i, let Y_i(1) and Y_i(0) represent outcomes under treatment and no treatment. The individual effect is Y_i(1)-Y_i(0). An average over trial participants identifies a trial-specific causal estimand, while an average over a target population defines a target-specific estimand.

Write S=1 for trial membership and S=0 for membership in the intended target population. The average trial effect is

\[
\tau_S=E\{Y(1)-Y(0)\mid S=1\},
\]

whereas the target average effect is

\[
\tau_T=E\{Y(1)-Y(0)\mid S=0\}.
\]

There is no mathematical reason for these averages to coincide when effects vary and the two populations have different covariate distributions. Randomization helps identify the first quantity. It does not automatically establish equality between the first and second.

A useful example is a treatment with a greater effect among people with a particular baseline characteristic. If the trial contains fewer people with that characteristic than the target population, the unweighted trial mean will underrepresent the subgroup that benefits most. The reverse is possible as well. The direction of transport bias is determined by the interaction between effect modification and the population shift, not by randomization failure.

This distinction can arise in clinical studies, education, labour policy, agricultural interventions and online experiments. The trial may be internally credible while still being drawn from a convenience or highly selective eligible population. The decision may concern the population of future patients, customers or programme applicants. An analyst must decide which of those populations the intervention claim is supposed to cover before reporting one average coefficient as *the effect*.

It is also important to distinguish the effect of assignment from the effect of treatment receipt. A randomized offer may identify intention to treat, whereas a policy that mandates treatment might have a different estimand. Transporting an effect without carefully defining what treatment means risks combining population differences with changes in the intervention itself.

## Why conditioning can support transport

Let X denote baseline characteristics relevant to treatment effects. If treatment assignment is randomized within the trial, the conditional mean treatment difference identifies a conditional trial effect

\[
\tau(x)=E[Y\mid A=1,X=x,S=1]-E[Y\mid A=0,X=x,S=1].
\]

To transport this contrast, assume that the conditional treatment effect for target individuals with X=x is the same as for trial individuals with X=x. This is a form of conditional transportability or conditional effect exchangeability between populations. It is not a consequence of randomization; it is an additional assumption relating potential outcomes across populations.

Under that assumption, and under support conditions permitting trial information at the target values of X, the target effect can be identified by standardization:

\[
\tau_T=\int \tau(x)\,dF_T(x).
\]

The causal quantity remains the target average difference between two interventions. The calculation estimates conditional effects from the trial and averages them over the covariate distribution of the target. This is closely related to the g-formula already discussed earlier in the series, now used to change the population distribution rather than the treatment strategy over time.

The support condition matters. If the trial excludes some values of X that are common in the target, the conditional effect at those values is not identified nonparametrically from trial randomization. A parametric regression may still assign them predictions, but those predictions extrapolate a model outside the observed treatment-effect support. Such extrapolation can be correct under strong functional-form assumptions, yet it should not be described as information directly established by randomized comparisons at those values.

The approach also depends on capturing the modifiers of the treatment effect that differ between trial and target. Adjusting for variables that predict outcomes without modifying effects may improve estimation precision, but those variables are not necessarily sufficient to transport the effect. The distinction between prognostic variables and effect modifiers is central to the analysis.

## The first simulation: observed effect modification

The first data-generating mechanism assigns trial participants an observed baseline characteristic X distributed normally with mean zero and variance one. The target population has the same variance but mean one. The trial and target therefore differ in their distribution of a potentially important variable, while retaining substantial overlap.

Treatment A is randomized independently of X with probability one half. The trial outcome follows

\[
Y=0.5X+A(1+X)+\varepsilon,
\]

where the disturbance is standard normal and independent of X and A. The individual causal effect is

\[
Y(1)-Y(0)=1+X.
\]

The trial average effect is consequently 1+E[X|S=1]=1. The target average effect is 1+E[X|S=0]=2. This gives an explicit example in which a randomized trial correctly identifies its own average effect while that same average is not the quantity the decision-maker needs.

The fitted outcome model includes an intercept, X, A, and their interaction AX. The treatment coefficient and treatment-by-X coefficient together estimate the conditional effect function. The analyst can calculate an average effect for the trial by averaging predicted effects over trial X, or for the target by averaging the same conditional-effect model over target X. Both averages use the same estimated response to treatment but different covariate distributions.

The experiment uses 1,200 trial observations and 3,000 target covariate observations per replication. The target outcomes are not observed and are not used to fit the outcome regression. The target data provide the distribution over which the conditional trial effects are standardized. This matches an important practical setting in which an experiment has randomized outcomes but the population to which a policy will apply is known primarily through baseline characteristics.

The distinction between evaluating an estimated function at the target covariate distribution and refitting an outcome model in the target population should remain explicit. The latter would require target outcomes, and perhaps a different identification argument. The former uses the experimental outcomes plus the target distribution and the assumption that the conditional effect is stable.

## What the first comparison establishes

Across 400 independent datasets, the mean trial-average estimated effect is 1.0052, with RMSE approximately 0.0657 relative to the true trial effect of one. The transported estimate is 2.0041, with RMSE approximately 0.0816 relative to the target truth of two.

The result is close to the mathematical expectation because the interaction regression matches the generating equation. The target distribution shifts by one unit of X, and the effect changes by one unit per unit of X. The transported estimate therefore adds almost exactly one outcome unit to the trial-average estimate. It is estimating a different parameter, not correcting a bias in an otherwise defective trial.

This difference is important for scientific reporting. A randomized experiment producing an estimate around one can be perfectly valid for its trial participants. Reporting that estimate as a target-population effect of one would be the problem. The transport calculation yields a target effect around two because the target has more individuals with positive effect-modifier values.

The simulation does not prove that linear interaction models are generally sufficient. It demonstrates what standardization achieves when the relevant modifier is known, the outcome specification is correct, and overlap is adequate. A nonlinear or heterogeneous effect function would require a model capable of representing that structure, together with appropriate checks of its performance.

One should also distinguish estimating the conditional treatment effect from predicting the outcome level. A model with excellent baseline outcome prediction may still estimate the treatment interaction poorly, particularly when the trial has little information about treatment differences in some subgroup. Transport depends on effect differences, so predictive metrics evaluated only on observed outcomes need not be the most informative measure of its performance.

## The second simulation: an unmeasured effect modifier

The second simulation deliberately makes the relevant effect modifier unavailable to the analyst. Let U be an independent standard normal variable in the trial, and stipulate a target population where U follows a normal distribution with mean one and variance one. This target distribution is part of the known simulation design, not a quantity estimated from the recorded target sample: the code generates only the observed target characteristic X. As a result, the true target effect of two is defined analytically, not verified from generated target U observations. The treatment effect is

\[
Y(1)-Y(0)=1+U.
\]

A separate observed characteristic X still differs in distribution between trial and target, but X does not modify the treatment effect in this mechanism. The trial outcome is generated as

\[
Y=0.5X+A(1+U)+\varepsilon.
\]

Treatment remains randomized within the trial. Consequently the trial's average treatment effect is still identified, and the mean effect in the trial is one because E[U|S=1]=0. In the target, E[U|S=0]=1, so the target effect is two.

The analyst does not observe U and continues to fit an outcome regression involving only X, A and their interaction. The trial comparison remains internally randomized, so the average effect in the trial does not become confounded by omitting U. The problem is that the observed X does not explain the difference in effects between populations.

This is a distinct failure from the unmeasured confounding studied in earlier parts of the series. Here there is no treatment-assignment confounding within the trial. Randomization is sound and the trial effect is estimated correctly. What fails is the assumption that conditioning on the available variables makes the effect transportable from the trial to the target.

Because the true effect depends on U rather than X, the regression has no systematic treatment-by-X interaction to learn. Standardizing its fitted effect over a target whose X mean differs from the trial cannot reproduce the increase caused by the target's different distribution of U. A population shift can be fully visible in the data and still be causally irrelevant to transport if the variables that actually modify treatment effects are not observed.

## The failure hidden behind trial validity

Across 400 datasets in the omitted-modifier scenario, the trial effect estimate averages 0.9980, close to its true value of one. The transported estimate averages 1.0015, with RMSE approximately 1.0035 relative to the true target effect of two.

This is a striking failure because it does not arise from an unstable experimental contrast, inadequate treatment randomization or a poorly estimated target X distribution. The trial sample is large enough for the randomized average to be estimated precisely. The target sample is large enough to characterize the observed population shift. The necessary information is not available because the different distribution of U across populations was not measured.

The resulting transported estimate becomes stable near the trial effect rather than approaching the target effect. Collecting more observations with only X measured would not make the relevant population difference appear. The target effect can remain unidentified from the observed information, even though both trial randomization and the measurement of X are excellent.

This is why external validity cannot be summarized as adding weights for demographic variables. The analyst must ask which characteristics cause effects to differ and whether the relation between those characteristics and potential outcomes is stable across settings. If the variables being adjusted for are only proxies, or if they do not capture relevant effect modification, reweighting may produce a numerically precise answer to the wrong target question.

It is also possible for an omitted effect modifier to have the same distribution in trial and target, in which case it would not necessarily prevent transport of the average effect. The issue is not simply that any effect modifier is unmeasured. It is that the distribution of an unmeasured effect modifier changes in a way that changes the average effect across populations. That distinction is important when specifying what must be measured for transport.

## Selection into the trial

In practice, trial participation is rarely a simple random draw from the target population. Eligibility criteria, clinical screening, geographical access, consent, availability and institutional selection can all affect who participates. Some mechanisms depend on measured baseline characteristics; others may be related to variables the dataset does not record.

The transportability problem should therefore be represented in terms of the source and target selection mechanisms, not merely two datasets with different means. In the experiments, the distributions are specified directly. In an empirical study, the analyst should describe why the source participants differ and which of those differences are plausible causes of treatment-effect heterogeneity.

One cannot infer conditional transportability simply from similarity in a baseline table. The comparison could be balanced on age, sex, region or historical outcomes while missing a modifier related to prior treatment history, biological response, household circumstances or service implementation. Even when the target and source distributions match on measured covariates, the latent distributions affecting treatment response may differ.

An observed covariate difference is not automatically an obstacle either. If a covariate changes baseline outcome but not the treatment effect, its different distribution does not by itself change the target average causal effect. Distinguishing prognostic factors from effect modifiers is therefore more important than making every measured population feature appear identical.

A transport analysis requires assumptions about the outcome mechanisms under treatment and control as well as the population distributions. Where those assumptions are doubtful, the defensible conclusion may be narrower than a single target average. It may require restricting the target to a common-support region or presenting sensitivity ranges for plausible effect modification not captured in the data.

## The third simulation: only partial common support

The third mechanism returns to an observed effect modifier with causal effect 1+X, but changes its distribution more sharply. Trial X is uniform on [-1,1], while target X is uniform on [0.5,2.5]. Only a quarter of the target distribution lies in the trial's support, from 0.5 to 1. The remainder has effect-modifier values at which the trial contains no observations.

The trial-average effect is one, because trial X has mean zero. The target mean of X is 1.5, so the full target average effect is 2.5. For the subset of the target lying in the trial-supported interval [0.5,1], the mean X is 0.75, giving a restricted target effect of 1.75.

These are three different causal quantities. They must not be evaluated against the same numerical truth. The trial identifies its own effect directly through randomization. The target's supported portion can be approached through comparisons in a region represented in both datasets. The full target effect includes a region in which the experimental data provide no direct within-X treatment comparison.

A linear interaction model can extrapolate into that unsupported region because the model assumes the effect function remains 1+X beyond the observed trial values. The simulation is deliberately generated from precisely that function. Therefore extrapolation can work numerically; this is not a refutation of the overlap problem. It illustrates how strong and correct structural assumptions can supply what the observed support alone does not establish.

Were the effect function to bend sharply beyond X=1, the trial could fit perfectly within [-1,1] and still extrapolate badly to [1,2.5]. A model's success in a simulated world with the correct extrapolation assumption should not be generalized into confidence that such assumptions hold in real settings.

## What the overlap results show

In 400 simulations, the unweighted trial effect averages 1.0020, close to its trial target of one. The full-target extrapolated estimate averages 2.4927, close to the true full-target effect of 2.5, with RMSE approximately 0.1590. The supported-subpopulation estimate averages 1.7473, close to its different truth of 1.75, with RMSE approximately 0.0933.

The apparent success of the full-target prediction is conditional on the correctly specified linear interaction. The experiment does not contain evidence about whether linearity would continue outside the trial in another system. The extra uncertainty due to plausible model departures is not represented by the reported Monte Carlo RMSE, because every simulated dataset follows the same linear relationship.

The retained target fraction averages about 0.2495 in this scenario. Only roughly one quarter of the target population lies within the empirical support of trial X. Restricting the estimand to that group changes the scientific question and produces a lower average effect of 1.75. This is not a failed transport estimate of 2.5; it is an estimate of a deliberately narrower population effect.

The restriction may be scientifically preferable when the full-population extrapolation is not credible. If policy must cover the entire target, however, the restricted effect alone is not enough to forecast the policy's average consequence. An analyst must then explicitly state what assumptions or outside evidence are needed to speak about the unsupported portion.

The simulation uses the sample minimum and maximum of trial X to define empirical common support. Those bounds differ slightly across replications, and this produces small sampling variation in the retained fraction. The target subset's theoretical benchmark remains 1.75 for the population overlap interval [0.5,1]. The approximation is adequate at this sample size, but the distinction between population support and empirical min–max support should be made explicit.

## Why transport weights can become unstable

An alternative to outcome-model standardization is to reweight the trial population so that its covariate distribution resembles the target. In suitable settings, density-ratio or sampling-odds weights assign greater influence to trial individuals who are relatively common in the target population and less influence to those overrepresented in the trial.

This is conceptually related to inverse probability weighting for treatment assignment, but the weights here represent selection into the source rather than treatment. Transport weights need information about the distributions of covariates in trial and target populations. Under strong overlap and appropriate modelling, weighting and standardization can identify the same target effect.

Under weak overlap, transport weights can become extreme. A handful of trial observations may represent large parts of the target distribution, just as rare treatment assignments produced extreme weights in Part 4 of the series. Effective sample size falls, uncertainty grows and the estimator becomes sensitive to individual observations and weight truncation.

When target X values are entirely absent from the trial, simple density-ratio weighting cannot generate outcomes at those unsupported values. Reweighting existing observations does not create new treatment comparisons in the missing region. A different model, additional data source, or a narrower estimand is needed.

The present experiment implements outcome-model standardization rather than density-ratio weighting. The conclusions about transport weights are methodological context, not an additional numerical finding from these simulations. A separate benchmark could compare outcome standardization, weighting and augmented transport estimators under both correctly specified and misspecified sampling models.

## What trial randomization contributes

Randomization is still valuable when external validity is uncertain. It identifies causal treatment differences within the source population under the relevant trial assumptions, preventing the treatment comparison from being confounded by how treatment is assigned within the experiment.

But randomization of A does not randomize S, the indicator of participation in the trial rather than the target population. The potential outcomes' distribution can differ across S because of selection into participation, even when A is perfectly randomized given S=1.

This distinction prevents an unhelpful opposition between internal and external validity. A well-run trial may provide strong internal evidence and yet require an explicit transport analysis for the policy target. The correct response is not to discount the trial, but to use it for the quantity it identifies and separately justify the assumptions connecting that quantity to the target.

The treatment intervention itself can also differ between settings. Implementation fidelity, staff training, population expectations, complementary services and capacity constraints may alter the response to the same nominal treatment label. Conditional effect exchangeability may therefore fail for reasons not summarized by patient-level covariates alone.

A transport analysis should describe the intervention in both settings carefully. If the target policy changes the treatment, the potential-outcome definition has changed as well, and adjusting the population distribution cannot by itself correct the mismatch.

## The limits of a good predictive model

A model that predicts outcomes accurately inside the trial need not predict treatment effects correctly in the target. It may capture the relationship between baseline covariates and outcomes while failing to represent treatment-covariate interactions. It may also extrapolate outcome levels reasonably while extrapolating the treatment contrast poorly.

The second simulation demonstrates this limitation when U is omitted. The measured X remains useful for describing baseline outcomes, but the treatment effect actually varies with U. The trial's interaction model cannot learn that variation, regardless of how precisely it estimates the average treatment difference in the source.

The third simulation adds a different problem. The effect modifier is measured and the model class is correct, so it extrapolates successfully. But its success depends on knowing, effectively, that the treatment effect continues linearly outside the region observed experimentally. A model may have excellent validation scores inside the trial while behaving differently outside its observed support.

Cross-validation can test predictions within the source distribution. It cannot by itself validate predictions in a target domain with covariate values absent from that source. Validating transport requires target-relevant information, independent evidence about effect mechanisms, or assumptions strong enough to support extrapolation.

This is not a criticism of machine learning or parametric modelling specifically. Flexible and simple models alike are restricted by the information available and the assumptions that connect that information to unseen counterfactuals. Increased predictive sophistication may help estimate complicated effect modification where data have support, but it does not eliminate the causal and population assumptions behind transport.

## Statistical uncertainty versus transport uncertainty

The simulations quantify how fitted conditional-effect estimates vary over independently generated source and target datasets. Their RMSE is calculated against the known population target. The results therefore incorporate the sampling variability of the simple fitted interaction and of the observed target covariate distribution under the specified mechanism.

They do not incorporate uncertainty about whether the true effect modifier has been measured, whether the conditional response mechanism is stable across populations, or whether the linear effect model remains correct beyond the trial's support. Those are assumptions about the scientific system rather than parameters estimated from the available sample alone.

A narrow confidence interval for an effect standardized to the target distribution could therefore be misleading if the transport assumptions are doubtful. The interval would describe sampling uncertainty conditional on a model, while omitting possible differences in treatment response that the model cannot observe.

Sensitivity analysis can express how much the target effect would change under plausible unmeasured effect modification or departures from conditional invariance. Such a calculation will depend on substantive bounds or alternative assumptions. It cannot automatically prove that the target effect is identified.

The separation is the same as in earlier parts of this series. Sampling uncertainty describes uncertainty about an estimable statistical quantity within a specified model and design. Causal and transport uncertainty also includes uncertainty about why that statistical quantity corresponds to the desired intervention effect in another population.

## Selecting an appropriate target population

Before transporting an effect, an analyst should identify the population relevant to the decision. That may be all eligible people, future recipients, people who would actually take up a treatment, or a subset for whom both treatment options are feasible. Different targets can have different covariate distributions and different average effects.

The full target is not necessarily the only useful estimand. Under limited overlap, an effect for the support-restricted target may be more credible because it requires fewer extrapolation assumptions. It is scientifically legitimate to report the restricted effect if its population is defined explicitly and its relationship to the decision is made clear.

However, narrowing the population after seeing results can create a form of selective reporting. Overlap restrictions should be motivated by pre-treatment information, the available design and the desired policy scope, not chosen simply because one restriction produces a stronger or more significant effect.

Our third scenario shows why these distinctions matter numerically. The full target effect is 2.5. The effect in the overlapping target region is 1.75. Neither is a correction of the other. They are answers to different population-level questions, and a report must state which one it estimates.

The problem becomes more consequential with heterogeneous effects that vary in sign. Excluding groups with little overlap might eliminate individuals harmed by treatment or those who would benefit most. The remaining average could give a misleading impression of the programme's consequences if presented as representative of everyone.

## How I would design a transport analysis

The first task is to write down the intervention, outcome, follow-up period and target population. The trial's effect should be distinguished explicitly from the target effect. If treatment versions differ between settings, those differences should be described before any statistical transport procedure is selected.

Next, identify candidate effect modifiers that may differ between source and target. Their selection should be grounded in causal knowledge, prior research and the intervention's mechanisms. Variables predicting trial participation but not effect variation may matter for estimating selection probabilities, while variables modifying the effect are essential to explaining why the average changes across populations.

Examine the distributions of those variables in both datasets. Overlap should be evaluated rather than presumed, using density summaries, ranges and multivariate diagnostics where appropriate. Extreme sampling weights or empty target regions can reveal how heavily an analysis relies on extrapolation.

Choose the estimand before the estimator. Outcome-model standardization, weighting and doubly robust transport estimators can be useful under different modelling and efficiency conditions. A sensitivity analysis should consider misspecification of effect modification and population selection mechanisms, not merely different tuning parameters.

Report the unweighted trial effect, the transported estimate, how much the target distribution differs, what percentage of the target is supported by the trial, and what assumptions are required to interpret the resulting comparison causally. When the evidence supports only a restricted population, state that instead of silently extending the estimate to everyone.

Finally, assess whether the transported estimate answers the decision. Even a valid target average may conceal treatment heterogeneity relevant to individual allocation, cost constraints or distributional policy goals. Average effects and optimal treatment decisions are related but distinct questions, as Part 12 demonstrated.

## What the numerical benchmark does not show

The study uses a single continuous effect modifier, a randomized binary treatment, Gaussian outcome noise and a correctly specified linear interaction model. The source and target samples are generated independently. The trial contains 1,200 individuals, and the target covariate sample contains 3,000. Each scenario is repeated 400 times.

There is no treatment noncompliance, missing outcome, interference, clustered assignment or time-varying treatment. The experiment therefore isolates population transport and support rather than combining them with all possible implementation failures. The results are not estimates of a real clinical or policy intervention.

In the first scenario the relevant effect modifier is observed and the model class matches the truth. In the second, a hidden effect modifier differs across populations, so the conditional invariance assumption is not satisfied given the recorded X. In the third, a correct linear model extrapolates beyond source support. That result is conditional on its functional-form correctness and should not be interpreted as data-driven validation of extrapolation.

The restricted-target estimate uses the empirical minimum and maximum of trial X. That is a simple demonstration of common-support restriction rather than an optimized overlap procedure. In practice, support is multivariate and may be difficult to assess reliably in high-dimensional data.

The simulation reports point-estimation error rather than nominal confidence-interval coverage. A more extensive study could compare density-ratio weighting, inverse odds of sampling, calibration weighting, outcome standardization and doubly robust estimators, under both model misspecification and varying support.

None of the simulations establishes which covariates are the correct effect modifiers in a real study. Those depend on subject-matter knowledge and evidence outside the randomized trial data themselves. The numerical experiments explain consequences of known assumptions, not how to verify them automatically.

## What the investigation established

A randomized trial correctly estimated a source-population average treatment effect near one in all three scenarios. In the measured-modifier scenario, the target effect was two and standardization recovered it with mean estimate 2.0041 and RMSE 0.0816.

When the relevant effect modifier differed across populations but was omitted from the analyst's model, standardization instead produced a mean target estimate of 1.0015 for a true target effect of two. Randomization remained valid within the source. The failure was external: the observed covariates were insufficient to relate the source conditional effects to the target population.

In the limited-overlap scenario, the trial effect was one, the full target effect was 2.5 and the effect in the supported target region was 1.75. A correctly specified linear interaction model extrapolated to a mean full-target estimate of 2.4927, while restricting to observed support produced 1.7473. The result illustrates both the possibility of correct extrapolation and the assumptions needed to justify it.

The central lesson is that internal validity and external validity address different questions. Randomization can identify what happened causally among those studied. Transportability asks whether the effect remains the same, or can be reconstructed, in a different population. The answer depends on effect modification, selection, support and the stability of the intervention mechanism.

A statistically valid trial result need not be a population-wide policy effect. Nor does good prediction or demographic reweighting guarantee that it becomes one. A useful transport estimate must specify the population, identify the variables on which effects can be transported, and acknowledge where the data do not support the requested counterfactual comparison.

## Code

The complete numerical experiment is implemented in `causal_inference_statistics/part19/experiment.py`. It uses NumPy with seed 20261018 and writes `results.csv`. The script reports trial and transported estimates under all three scenarios, along with a restricted-population estimate for the limited-overlap case, with means, bias, RMSE and common-support fractions. No external data are required.

## The series

1. When Regression Gets the Causal Effect Wrong
2. Randomization, Confounding and the Value of Experimental Design
3. Matching and Propensity Scores: When Balanced Data Still Give the Wrong Causal Effect
4. Inverse Probability Weighting: When a Few Observations Control the Result
5. Directed Acyclic Graphs and the Cost of Controlling for Everything
6. Difference-in-Differences: When Parallel Trends Fail
7. Instrumental Variables: When a Strong First Stage Is Not Enough
8. Regression Discontinuity: Why a Sharp Cutoff Is Not Enough
9. Synthetic Controls and Counterfactual Time Series
10. Doubly Robust Estimation
11. The G-Formula and Longitudinal Intervention Strategies
12. Heterogeneous Treatment Effects
13. Causal Machine Learning and Double Machine Learning
14. Bayesian Causal Inference
15. Time-Varying Treatments and Marginal Structural Models
16. Sensitivity Analysis for Unmeasured Confounding
17. Negative Controls, Placebo Tests and Falsification
18. Missing Outcomes and Selection Bias
19. External Validity and Transportability (this article)
20. From Statistical Evidence to Causal Decisions

## References

Bareinboim, E. and Pearl, J. (2016). Causal Inference and the Data-Fusion Problem. *Proceedings of the National Academy of Sciences*, 113(27), 7345–7352. https://doi.org/10.1073/pnas.1510507113

Cole, S. R. and Stuart, E. A. (2010). Generalizing Evidence from Randomized Clinical Trials to Target Populations. *American Journal of Epidemiology*, 172(1), 107–115. https://doi.org/10.1093/aje/kwq084

Dahabreh, I. J., Robertson, S. E., Steingrimsson, J. A., Stuart, E. A. and Hernán, M. A. (2020). Extending Inferences from a Randomized Trial to a New Target Population. *Statistics in Medicine*, 39(14), 1999–2014. https://doi.org/10.1002/sim.8423

Hernán, M. A. and Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC. https://miguelhernan.org/whatifbook

Imbens, G. W. and Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press.

Lesko, C. R., Buchanan, A. L., Westreich, D., Edwards, J. K., Hudgens, M. G. and Cole, S. R. (2017). Generalizing Study Results: A Potential Outcomes Perspective. *Epidemiology*, 28(4), 553–561. https://doi.org/10.1097/EDE.0000000000000664

Pearl, J. and Bareinboim, E. (2014). External Validity: From Do-Calculus to Transportability Across Populations. *Statistical Science*, 29(4), 579–595. https://doi.org/10.1214/14-STS486

Rudolph, K. E. and van der Laan, M. J. (2017). Robust Estimation of Encouragement Design Intervention Effects Transported Across Sites. *Journal of the Royal Statistical Society: Series B*, 79(5), 1509–1525.

Stuart, E. A., Cole, S. R., Bradshaw, C. P. and Leaf, P. J. (2011). The Use of Propensity Scores to Assess the Generalizability of Results from Randomized Trials. *Journal of the Royal Statistical Society: Series A*, 174(2), 369–386. https://doi.org/10.1111/j.1467-985X.2010.00673.x

Tipton, E. (2013). Improving Generalizations from Experiments Using Propensity Score Subclassification: Assumptions, Properties, and Contexts. *Journal of Educational and Behavioral Statistics*, 38(3), 239–266. https://doi.org/10.3102/1076998612441947

Tipton, E. (2014). How Generalizable Is Your Experiment? An Index for Comparing Experimental Samples and Populations. *Journal of Educational and Behavioral Statistics*, 39(6), 478–501. https://doi.org/10.3102/1076998614558486

Westreich, D., Edwards, J. K., Lesko, C. R., Stuart, E. A. and Cole, S. R. (2017). Transportability of Trial Results Using Inverse Odds of Sampling Weights. *American Journal of Epidemiology*, 186(8), 1010–1014. https://doi.org/10.1093/aje/kwx164

Zivich, P. N., Cole, S. R. and Westreich, D. (2022). Transportability and Generalizability of Causal Effects: A Practical Guide. *American Journal of Epidemiology*, 191(4), 722–728.