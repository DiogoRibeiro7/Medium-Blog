# Inverse Probability Weighting: When a Few Observations Control the Result

*Part 4 of Causal Inference in Statistics. A reproducible Monte Carlo study of inverse probability weights, normalized estimators, extreme propensity scores and effective sample size. Why a large dataset can contain surprisingly little information about a causal contrast. Python code provided.*

Inverse probability weighting has a compelling interpretation. People who were unlikely to receive the treatment they actually received represent many similar people in a reweighted population. If treatment assignment is explainable by measured pre-treatment characteristics, weighting can reconstruct a comparison in which treatment no longer depends on those characteristics. It is a statistical method with a causal objective: estimate what would happen under an intervention rather than describe what happened among people who selected or were selected into treatment.

The difficulty appears when assignment probabilities become extreme. If an individual received treatment despite a predicted treatment probability of 0.01, the ordinary inverse probability weight is 100. In a dataset containing thousands of people, one outcome can then have more influence than many others combined. Standard errors grow, estimates fluctuate dramatically between samples, and seemingly minor modelling decisions begin to matter.

This is not necessarily a defect in the algorithm. When hardly anyone with a particular prognosis receives a treatment, the data contain little direct information about the outcome those people would have under treatment. Weighting attempts to recover a population comparison despite that scarcity. Large weights are one numerical sign of a deeper support problem.

In this article I compare the conventional Horvitz–Thompson form of inverse probability weighting, its normalized or Hájek form, and a clipped version of the normalized estimator. I also report the unadjusted treatment-control difference as a baseline. Each method is tested on 300 independently generated datasets containing 1,500 observations, under three increasingly selective treatment assignment mechanisms. The treatment effect is known exactly: one outcome unit for every individual.

Under the weakest selection mechanism, both weighting estimators recover the effect accurately and have root mean squared error around 0.063. Under the strongest, ordinary IPW has RMSE 0.9013 and normalized IPW 0.5892. Clipping reduces the variability but produces a mean estimate of 1.8243 instead of the true effect of one. Meanwhile, the average effective sample size among treated observations falls from about 660 to 152, even though the nominal sample size remains 1,500.

The experiment does not establish that one of these estimators always dominates. It shows how treatment assignment, overlap, weighting, normalization and clipping interact under a specific causal model. The important conclusion is that a precisely specified statistical estimator still needs information that the treatment mechanism may rarely produce.

## The decision behind the weights

Imagine an employment agency evaluating additional support for jobseekers. Staff tend to assign the programme to people with characteristics associated with greater unemployment risk. An ordinary comparison of programme participants with nonparticipants would combine the programme's effect with differences in baseline prognosis. Inverse probability weighting tries to adjust for those differences without fitting a single outcome regression whose treatment coefficient is interpreted as the causal effect.

The agency can estimate each person's probability of receiving support conditional on recorded baseline information. Participants who were relatively unlikely to receive support receive greater weight. Nonparticipants who were relatively unlikely to remain untreated also receive greater weight. The aim is to construct an artificial population in which treatment assignment is less associated with those measured characteristics.

That sounds like a practical solution until some combinations of characteristics nearly determine who receives support. If people with very high unemployment risk almost always enter the programme, there may be only a few untreated people in that group. The counterfactual outcome under no programme is then supported by very little observed information. A weighting method can give those few untreated people very large weights; it cannot turn them into thousands of independent observations.

The causal question is what the outcome would be if programme assignment were changed across the population. The observational question is how outcomes differ between people who were and were not selected. Weighting connects the two under assumptions about the allocation process. It does not remove the need for those assumptions.

The numerical experiment below uses a continuous baseline characteristic and a continuous outcome, so it does not analyse an actual unemployment dataset. The setting is an illustration of a familiar programme evaluation problem, not evidence about whether employment support works. The advantage of synthetic data is that the counterfactual outcomes and treatment assignment mechanism can be specified before estimation, making it possible to separate algorithmic error from uncertainty about the true effect.

## The causal quantity and the identifying assumptions

Let A be a binary treatment indicator, X a baseline covariate and Y the observed outcome. Each individual has potential outcomes Y(1) and Y(0), corresponding to the outcomes under treatment and no treatment. The average treatment effect is

\[
\tau=E[Y(1)-Y(0)].
\]

The causal estimand is defined by those potential outcomes, not by the fitted treatment probability or weighting algorithm. An estimator is useful only if its assumptions justify connecting observable outcomes to this contrast.

Under consistency, the observed outcome equals the potential outcome corresponding to the treatment actually received. Conditional exchangeability requires

\[
(Y(0),Y(1))\perp A\mid X.
\]

This says that, after conditioning on X, treatment assignment contains no additional information about potential outcomes. In observational data it is an assumption about the causal structure of assignment and outcome, not a consequence of having fitted a propensity-score model.

The positivity condition requires both treatment options to have positive conditional probability across the covariate values relevant to the target population:

\[
0<P(A=1\mid X=x)<1.
\]

If treatment is impossible for a particular subgroup, observations alone cannot reveal its untreated-versus-treated contrast there without extrapolation or additional assumptions. Even when the mathematical probabilities are positive, extremely small probabilities can create severe finite-sample instability.

With these conditions, the mean potential outcomes can be identified by weighting the observed outcomes according to treatment probabilities. This is the bridge between the causal question and an observable statistical functional. Its validity is not established by the smallness of a p-value, the discrimination of the propensity model or the precision of a particular weighted regression.

The assumptions also make clear why inverse probability weighting and propensity-score matching have related vulnerabilities. Both exploit treatment assignment information conditional on measured variables. Neither can correct an unmeasured common cause simply by improving the fitted probabilities. Part 3 demonstrated that near-perfect observed balance can coexist with severe hidden confounding. Here I deliberately remove hidden confounding to isolate a different difficulty: the instability caused by weak overlap.

## How inverse probability weighting reconstructs a population

The propensity score is

\[
e(x)=P(A=1\mid X=x).
\]

For an individual who received treatment, the inverse probability of that assignment is 1/e(X). For a control individual it is 1/[1−e(X)]. People whose observed assignment is unusual conditional on X receive larger weights.

The Horvitz–Thompson-style estimator of the population average treatment effect is

\[
\widehat{\tau}_{HT}
=
\frac{1}{n}\sum_{i=1}^{n}
\left[
\frac{A_iY_i}{\widehat e(X_i)}
-
\frac{(1-A_i)Y_i}{1-\widehat e(X_i)}
\right].
\]

Its treated component is the average of observed treated outcomes weighted by the inverse probability of treatment. The control component is the analogous average of control outcomes weighted by inverse probability of non-treatment.

The identification calculation follows from iterated expectation. For the treated term, conditional on X and both potential outcomes, treatment assignment has expectation e(X) under conditional exchangeability. Therefore,

\[
E\left[\frac{AY}{e(X)}\mid X\right]
=
E[Y(1)\mid X].
\]

Averaging over X yields E[Y(1)]. The same reasoning applies to the control term and gives E[Y(0)]. Subtracting identifies the ATE. This is a population identity involving the true propensity function, not a guarantee that finite-sample estimates using estimated probabilities are accurate.

The expression also clarifies why positivity matters. If a required treatment probability is zero, the corresponding inverse probability is undefined. When it is merely very small, the calculation is formally defined but a rare observation can dominate the finite-sample sum.

An alternative, often called the normalized or Hájek estimator, divides each weighted total by the corresponding sum of weights:

\[
\widehat{\tau}_{H}
=
\frac{\sum_i A_iY_i/\widehat e_i}
{\sum_i A_i/\widehat e_i}
-
\frac{\sum_i(1-A_i)Y_i/(1-\widehat e_i)}
{\sum_i(1-A_i)/(1-\widehat e_i)}.
\]

Both estimators target the same population contrast under appropriate conditions. Their finite-sample behaviour can differ substantially because the normalized version prevents random fluctuations in the total estimated population size represented by each group from directly shifting the component means.

Normalization should not be confused with a proof of good overlap. The same extreme observations can still influence the normalized mean, but the denominator changes how their influence is expressed. It is also important not to confuse normalization of the estimator with stabilized inverse probability weights used in some applications. Stabilization multiplies weights by a numerator probability and has its own interpretation and use, particularly in marginal structural models. The normalized estimator here is simply a ratio of weighted totals.

## A model with known treatment effects

The data-generating model contains one measured baseline characteristic X and an independent noise term ε:

\[
X\sim N(0,1),\qquad \varepsilon\sim N(0,1).
\]

The potential outcomes are

\[
Y(0)=2X+\varepsilon,
\]

\[
Y(1)=1+2X+\varepsilon.
\]

Consequently,

\[
Y(1)-Y(0)=1
\]

for every individual, and the ATE equals 1. The baseline characteristic strongly predicts outcome, while the effect of treatment is constant. This design means that changes in estimated effects cannot be attributed to genuine variation in who benefits from treatment.

Treatment assignment is observational, with probability

\[
P(A=1\mid X)=\operatorname{expit}(\lambda X),
\]

where

\[
\operatorname{expit}(z)=\frac{1}{1+e^{-z}}.
\]

The parameter λ controls how strongly treatment assignment depends on the prognostic covariate. I examine λ=0.7, λ=1.5 and λ=3.0. When λ is small, people with different X values still have substantial chances of entering both groups. When λ is large, those with positive X are increasingly likely to be treated while those with negative X are increasingly likely to remain untreated.

The observed outcome is

\[
Y=A+2X+\varepsilon.
\]

The treatment effect stays fixed while the assignment mechanism changes. This is critical. If an estimator deteriorates as λ increases, it is not because the causal effect became harder to describe mathematically. It is because the observed comparisons become increasingly concentrated in different regions of X.

There is no unmeasured confounder in this experiment. X is fully observed and is the only common cause of assignment and outcome. The true treatment probability follows a logistic function of X, and the analyst estimates it using logistic regression with X as predictor. The fitted model is therefore correctly specified apart from ordinary finite-sample estimation uncertainty and a negligible practical regularization setting.

That favourable setup is deliberate. If weighting is unstable even when exchangeability holds and the propensity model has the correct functional form, the problem cannot be blamed entirely on missing confounders or poor predictive modelling.

## Three levels of overlap

Under λ=0.7, treatment assignment responds moderately to X. The fitted probabilities observed in the simulated samples typically cover a range extending toward approximately 0.09 and 0.91 at their extreme ends. Across 300 datasets, the average minimum fitted probability is 0.0910 and the average maximum is 0.9086.

At λ=1.5, the assignment is considerably more selective. The average fitted minimum falls to 0.0078 and the maximum rises to 0.9925. People at the tails of X have little chance of receiving one of the two assignments. This makes the weights larger for the rare treated or untreated individuals encountered there.

At λ=3.0, the fitted probabilities in typical samples come extremely close to zero and one. The average minimum and maximum are approximately 0.0001 and 0.9999, rounded to four decimal places. Positivity still holds mathematically for every finite X, but the probability of the less likely assignment becomes so small that some population comparisons depend on rare observations.

The distinction between formal and practical positivity is essential. The structural model never sets treatment probability to exactly zero or one. Yet in finite samples, an intervention contrast may be supported by only a handful of people in some covariate regions. A weighting formula defined in the population can then be an unstable estimator in the sample.

No observations are removed in the primary IPW or normalized IPW calculations. The dataset always contains 1,500 individuals. The observed collapse in effective information is created by the distribution of weights, not by discarding records from the dataset.

## The estimators compared

The first baseline is the raw difference in average outcome between treated and untreated people. It ignores X and is confounded because treatment selection depends on X. This is deliberately simple, just as in earlier articles: it shows the direction and scale of the associational distortion before weighting.

The second is the Horvitz–Thompson-style IPW estimator, which uses the inverse propensity for each observed treatment and averages the weighted treatment and control outcomes with the original total sample size as denominator. When the propensity estimates are close to the true assignment probabilities and overlap is adequate, it should recover the ATE.

The third is normalized IPW. It uses exactly the same estimated propensities but separately divides each group's weighted outcome total by that group's total weight. Its mean estimate and variability can therefore differ even without any difference in the fitted assignment model.

The fourth estimator first clips every estimated propensity to the interval [0.05, 0.95], then applies the normalized estimator. Clipping caps the contribution from extraordinarily small fitted treatment or non-treatment probabilities. An observed treatment with a propensity originally estimated at 0.001 is treated as though that probability were 0.05 for weighting purposes.

This fourth procedure is intentionally a bias–variance trade-off. It can prevent rare observations from dominating an estimate, but the adjusted probabilities no longer represent the original treatment assignment process. When extreme observations correspond to systematically different baseline prognoses, clipping can change the estimand represented by the calculation or introduce bias for the original ATE.

The comparison does not include an outcome-adjusted or doubly robust estimator. Those methods are important and will be treated separately. Keeping the present benchmark focused makes it possible to see how much of the behaviour comes from weighting itself before adding another model that may stabilize estimation by exploiting the outcome relationship.

## Why the propensity model is not the main difficulty

In each simulated dataset, logistic regression is fitted using only X and observed A. The resulting estimated probabilities are used by all three weighting procedures. The true assignment model is logistic in X, so the analyst is not trying to approximate an arbitrarily complicated nonlinear function with the wrong model family.

The difference between the results is therefore driven largely by selection strength and the way each weighting estimator uses the fitted probabilities. At extreme λ, a numerically plausible logistic fit can still produce weights that vary by orders of magnitude. Having a correct model class is not enough to ensure the sample contains useful counterfactual comparisons.

The fitted propensity score is a nuisance function in the causal analysis: it is needed to compute the effect estimator but is not itself the scientific target. It is tempting to evaluate it only as a classifier, using discrimination or probability calibration. Those properties matter, but a highly accurate estimate near zero can still generate a very large inverse weight. Better propensity estimation does not eliminate the need for treatment overlap.

Conversely, reducing propensity discrimination by forcing estimates toward the middle may make weights less variable, but it changes the assignment model used for causal identification. That is what clipping does deliberately. The stability gained by modifying probabilities must be considered alongside the bias it creates for the target estimand.

The right model-selection objective therefore depends on what the estimated propensity will be used for. If the goal is a causal effect, predictive classification accuracy alone does not measure the quality of the resulting causal estimate. The estimator's finite-sample behaviour, overlap and sensitivity to extreme observations must be examined directly.

## The simulation protocol

For each λ I generate 300 independent datasets of 1,500 observations using random seed 20261016. The same pseudorandom generator continues across the three assignment strengths, so the experiments use independent sequences of generated datasets rather than identical realized individuals at different λ.

Each dataset contains X, noise, observed treatment and observed outcome. I fit logistic regression to estimate treatment propensity, calculate the unadjusted difference, ordinary IPW, normalized IPW and clipped normalized IPW, then record weight diagnostics.

Across simulations, the code prints the mean estimator value, its between-replication standard deviation and its RMSE relative to the known ATE. RMSE combines squared bias and sampling variance, so it can reveal when a procedure has become stable but systematically wrong.

I also record the maximum inverse assignment weight in each dataset and the fraction of all propensity estimates outside [0.05, 0.95]. The latter is the fraction whose values are modified by the clipping rule, not the fraction of observations removed. No dataset is trimmed: clipping alters weights while retaining all records.

Finally, I calculate effective sample size separately in the treated and control groups. This reports concentration of weighting influence within each group. The diagnostic is useful, but it is not an exact number of independent observations for every inferential purpose and does not, by itself, validate a variance formula.

The reported summaries are averages over the simulated datasets. They are not a collection of independently published treatment effect studies and cannot establish how a specific real programme would behave. Their value is that every quantity is computed under a known causal mechanism and a fully specified estimator.

## What the estimates say

The Monte Carlo mean treatment effects are:

| Treatment assignment strength | Unadjusted | Ordinary IPW | Normalized IPW | Clipped normalized IPW |
| ---: | ---: | ---: | ---: | ---: |
| 0.7 | 2.2534 | 0.9973 | 0.9976 | 0.9976 |
| 1.5 | 3.1240 | 1.0047 | 1.0111 | 1.1128 |
| 3.0 | 3.7649 | 1.0078 | 1.1236 | 1.8243 |

The true causal effect is 1 in every row. The unadjusted outcome difference becomes increasingly misleading as assignment depends more strongly on X. That behaviour is expected because X raises the outcome and increasingly determines who receives treatment.

At λ=0.7, all weighting approaches recover the effect closely. The ordinary estimate is 0.9973, normalized IPW is 0.9976 and clipping makes almost no difference. This is the favourable region for weighting: treatment groups overlap reasonably well and no individual observation needs disproportionate influence.

At λ=1.5, ordinary and normalized IPW retain average estimates close to one, while clipped normalized IPW averages 1.1128. Clipping now changes a meaningful fraction of scores and makes the treatment contrast biased in this particular direction. The reduced volatility of the clipped estimator partly compensates for that bias in RMSE, but the mean no longer describes the true effect as accurately.

At λ=3.0, the ordinary IPW mean remains close to one, 1.0078, although that reassuring average hides severe variability between datasets. Normalized IPW averages 1.1236, reflecting finite-sample bias under this difficult weighting distribution. Clipped normalized IPW averages 1.8243, a substantial departure from the causal truth despite retaining all observations.

Averages alone would therefore produce a misleading ranking. Ordinary IPW might seem the clear winner at λ=3.0 based on mean bias, but its between-dataset variation is very large. Evaluating both systematic bias and variability is essential before drawing a conclusion about which estimator behaves better at a feasible sample size.

## Precision, RMSE and the cost of rare assignments

The corresponding Monte Carlo RMSE values are:

| Assignment strength | Unadjusted | Ordinary IPW | Normalized IPW | Clipped normalized IPW |
| ---: | ---: | ---: | ---: | ---: |
| 0.7 | 1.2585 | 0.0630 | 0.0634 | 0.0633 |
| 1.5 | 2.1264 | 0.1754 | 0.1826 | 0.1611 |
| 3.0 | 2.7664 | 0.9013 | 0.5892 | 0.8349 |

At λ=0.7, ordinary and normalized weighting are nearly indistinguishable in this measure. At λ=1.5, clipping performs slightly better on RMSE despite introducing some positive bias. It does so because reducing variability has more than compensated for the bias in this particular simulated setting.

At λ=3.0, ordinary IPW has RMSE 0.9013, much higher than normalized IPW's 0.5892. The ordinary estimator's mean effect remains near the truth, but a small number of extreme estimates contribute strongly to mean squared error. The normalized estimator is biased upward on average, yet its lower variability gives a better overall RMSE.

Clipping at this strongest selection level produces RMSE 0.8349. Its estimates vary much less, but the systematic error is sufficiently large that it does not outperform normalized IPW on RMSE. This is an example of why a more stable estimator can still be worse for the scientific target.

The result should not be generalized into a universal estimator ranking. A different outcome model, propensity distribution, clipping threshold or sample size could reverse the comparison. The correct interpretation is that each method expresses a different compromise between reliance on rare observations, normalization and bias.

Importantly, the strong-selection case is not a demonstration that ordinary IPW is inconsistent under the published model. The population propensity remains strictly between zero and one, the outcome model has finite moments and the score model is correctly specified. The simulation instead illustrates how poor practical overlap can make convergence extremely unattractive at finite sample sizes.

## Why an effective sample size can collapse

A common effective sample size diagnostic for a nonnegative set of weights w_i is

\[
n_{\mathrm{eff}}
=
\frac{(\sum_i w_i)^2}{\sum_i w_i^2}.
\]

If all weights are equal, this equals the number of observations being weighted. If a few observations carry most of the total weight, the effective size is much smaller. The measure is often called Kish's effective sample size and reflects the concentration of influence under a weighted mean.

For the experiment I calculate this separately among treated and untreated individuals using their applicable inverse assignment weights. The reported mean effective sizes are:

| Assignment strength | Treated effective n | Control effective n |
| ---: | ---: | ---: |
| 0.7 | 659.95 | 657.78 |
| 1.5 | 399.05 | 402.63 |
| 3.0 | 152.39 | 149.77 |

Because the assignment model is symmetric around zero, approximately half of the 1,500 observations are treated on average. With moderate assignment strength, the effective size near 660 indicates that weights remain relatively evenly distributed among the roughly 750 treated or control observations.

Under strong assignment, effective size falls to roughly 150 per group. This is a serious reduction in how evenly information contributes to the weighted comparison. It helps explain why the effect estimate becomes unstable despite the dataset still containing 1,500 rows.

Effective sample size is not a literal count of unique counterfactual examples. The weights depend on estimated propensities, outcomes may vary with covariates, and the exact sampling variance depends on the full estimator. Nevertheless, the diagnostic captures a practical danger: the apparent amount of data can be very different from the amount of balanced information available for estimating the causal contrast.

It is tempting to address a low effective sample size by collecting more data without changing the design. That may help, but if assignment remains nearly deterministic in important covariate regions, the additional sample must be very large before enough rare treatment alternatives appear. In some applications, redesigning the data collection or narrowing the target population may be more useful than increasing sample size under the same selection policy.

## Maximum weights and sensitivity to individual observations

The maximum inverse assignment weight provides a complementary diagnostic. Across simulated datasets, its mean is about 8.75 at λ=0.7, 39.83 at λ=1.5 and 192.36 at λ=3.0.

The distribution of these maxima matters as much as their averages. At the strongest assignment setting, the standard deviation of the maximum weight across datasets is approximately 251.94, larger than its mean. That reflects the occurrence of occasional datasets with extremely influential rare assignments.

These weights arise because treatment assignment is probabilistic even when one option is very unlikely. A person with large negative X may occasionally be treated, despite a small propensity to treatment. The observation carries strong information about a region where treated outcomes are rare, but ordinary IPW multiplies its observed outcome by a correspondingly large inverse probability.

The statistical logic is coherent. If a rare treated observation represents many similar people who were usually untreated, it should receive weight under the population identity. The numerical problem is that the rare observation's outcome includes random noise, and the weight amplifies that noise as well as its representative information.

A large maximum weight does not automatically prove that the effect estimate is wrong. Some weighted estimators remain stable because positive and negative contributions compensate, or because the outcomes in highly weighted regions have particular structure. It is, however, a warning that the result may depend on a small number of observations. A leave-one-out influence analysis or comparison with a normalized estimator can reveal how much the conclusion changes when those contributions are perturbed.

An applied analysis should therefore inspect weight distributions rather than reporting only the fitted propensity-model coefficients. Histograms, quantiles, maxima, effective sample sizes and estimates under defensible alternative weighting rules can expose problems hidden behind a single overall treatment effect number.

## Clipping makes the result quieter, not necessarily truer

Clipping changes fitted treatment probabilities outside [0.05, 0.95] to the nearest endpoint. For treated observations, the adjusted weight is the inverse of the clipped propensity. For untreated observations, it is the inverse of one minus the clipped propensity.

The resulting maximum applicable inverse weight is 20. This prevents the enormous values seen under the strongest selection mechanism. It also changes what the estimator does with the data. An observation with an estimated assignment probability of 0.001 is no longer allowed to represent roughly 1,000 similar observations; it receives a weight based on 0.05 instead.

The average proportion of propensities changed by clipping is 0.0001 at λ=0.7, 0.0508 at λ=1.5 and 0.3305 at λ=3.0. In the strongest case, about one third of all fitted treatment probabilities are modified. Clipping is therefore not a minor numerical safeguard at that setting. It substantially changes the weights assigned across the dataset.

The results illustrate the cost. At λ=3.0, clipping produces a mean estimate of 1.8243, even though the true effect is 1. The between-dataset standard deviation falls to approximately 0.1325, far below the 0.5761 of normalized IPW and the 0.9013 of ordinary IPW. The estimate looks much more stable, but that stability is centred around a biased answer.

A narrow spread in estimates across repeated samples is useful only if the estimator targets the quantity the analysis claims to estimate. A stable estimate of a modified or confounded contrast cannot be described as an accurate estimate of the original ATE merely because its variance is small.

The distinction between clipping, trimming and overlap weights also matters. Clipping modifies the probabilities used in the weight calculation. Trimming typically excludes observations outside a selected propensity region and therefore changes the analysis sample. Overlap weighting uses a different weighting function designed to target an overlap population. Those approaches can be scientifically defensible, but they do not necessarily estimate the same population effect.

Choosing a threshold by optimizing observed estimator stability can lead to an attractive-looking estimate while obscuring the price paid in bias or target-population change. A defensible analysis should report the threshold, the share of observations affected and the sensitivity of the estimate to plausible alternatives.

## Why normalized weighting behaves differently

The difference between the ordinary and normalized estimators can be seen by inspecting their denominators. Ordinary IPW scales weighted totals by the original sample size n. In finite samples, the sums of inverse propensity weights among treated and untreated people need not equal n. When rare assignments occur, a few weights can move those totals substantially.

Normalized IPW divides each weighted outcome total by the observed sum of weights for the corresponding group. This creates a ratio estimator. Its finite-sample bias need not be zero even if the underlying propensity model is correct, but it can reduce sensitivity to fluctuations in the estimated population totals.

Under λ=3.0, the ordinary IPW estimator's mean is approximately 1.0078 with a between-replication standard deviation of 0.9013. The normalized estimator's mean is 1.1236 with standard deviation 0.5761. The mean moves away from the causal truth, but the variation drops enough that RMSE improves.

This should be interpreted as a finite-sample bias–variance comparison, not an assertion that normalization resolves practical positivity. Both estimators still need observations receiving unlikely treatment assignments. Both can be unstable under inadequate overlap. Their different formulas determine how that instability appears numerically.

Stabilized weights, which include numerator probabilities, are often introduced in longitudinal marginal structural models to moderate variability while preserving the intended estimand under suitable conditions. Their construction is not the same as simply normalizing the two weighted component means. Confusing those two operations can make applied implementations difficult to audit.

The wider statistical question is not whether a particular formula has the more familiar name. It is what population quantity the estimator targets, what assumptions identify that quantity, how it behaves with estimated propensities and what uncertainty method is appropriate. The present comparison keeps those questions visible by listing the estimators separately and reporting their empirical errors rather than presenting a single preferred number.

## Overlap is a feature of the design

When the assignment strength grows, treatment and baseline prognosis become tightly linked. Treated people increasingly come from one region of X, controls from another. The formal propensity function remains valid and correctly specified, but the counterfactual contrast requires information from assignment combinations that are uncommon in the observed data.

This is why weighting problems should not be reduced to a choice of machine-learning algorithm. A more flexible model may represent treatment probabilities better, but it cannot make uncommon treatments frequent. An excellent propensity classifier can even accentuate extreme fitted probabilities and reveal just how little overlap the study has.

A randomized experiment with assignment probabilities bounded away from zero and one for all eligible units would address the allocation problem at its source. When randomization is infeasible, the analyst may be able to restrict the target to an overlap region where comparisons are better supported, or combine observational information with an alternative identification design.

Such a restriction has consequences. If treatment effects vary with X, an effect among people with substantial overlap need not equal the effect in the whole population. A weighted analysis that quietly shifts attention to the overlap population can therefore change the policy question. The target estimand should be defined explicitly, and any exclusions or reweighting that change the population should be justified.

In this experiment the effect is constant, so every population-specific average effect is one. This removes effect heterogeneity as a source of differences between numerical estimates. In realistic data, treatment effects may vary and overlap weighting may be a sensible way to answer a different, more supportable question. It should still be described as that question, not as an exact reconstruction of the original full-population ATE.

There is a fundamental limit to what statistical estimation can achieve under near-deterministic treatment selection. Rare counterfactual support cannot be manufactured by algebra. Weighting reallocates existing information; it does not generate new outcomes under treatment assignments that scarcely occur.

## Why effective sample size is not enough

A low effective sample size is a warning, but it does not determine the direction or magnitude of bias. It is calculated from weights alone and does not encode the relationship between outcome residuals, potential outcomes and treatment assignment. Two datasets with identical weights can have different estimation errors if their outcomes differ.

The diagnostic also depends on which observations and weights are included. Here it is calculated separately within each actually treated or untreated group, using the inverse probability applicable to that group. A pooled calculation would answer a different question. Reporting a single unlabelled effective sample size can obscure such distinctions.

Estimated propensity scores add another layer of dependence. The weights are fitted from the same treatment assignments used in the effect estimator. Analytical variance formulas that treat them as fixed known probabilities may not reflect the true uncertainty of the estimator. Resampling or influence-function methods may be appropriate, but their implementation must respect the complete estimation procedure and study design.

For the present article, Monte Carlo variability is calculated directly across independently regenerated datasets, so it includes the effect of re-estimating the propensity score in each replication. This provides an empirical benchmark of the methods under the known data-generating process. It does not supply a confidence interval for an arbitrary single observational dataset.

An applied evaluation should therefore report effect estimates and uncertainty, but also the structure of the weights, overlap diagnostics, balance after weighting and sensitivity to estimator choices. No one diagnostic certifies the causal conclusion. Their combined role is to show whether the calculation relies on defensible and sufficiently supported comparisons.

## Weighting and covariate balance

Inverse probability weighting aims to construct a reweighted distribution in which the measured baseline covariates are less associated with treatment. The relevant check is whether the treatment groups become comparable on those variables after weighting, not simply whether the propensity score predicts assignment.

In the constructed model there is one observed covariate and the treatment model is correctly specified. As sample size grows and overlap permits reliable weighting, the weighted treated and control covariate distributions should approach their common target. In finite samples, extreme weights can themselves produce unstable balance diagnostics when a few rare observations receive substantial influence.

A simple standardized mean difference after weighting can summarize residual imbalance in X. Distributional checks can provide information about differences in tails or nonlinear transformations that a mean difference misses. Both are useful when X is part of a sufficient adjustment set. As Part 3 showed, measured balance cannot diagnose an unmeasured confounder that also determines assignment.

The propensity-score model and the outcome estimator therefore have distinct roles. The score estimates treatment assignment probabilities. The weighting formula uses those probabilities to build a comparison. The balance check evaluates an observable implication of the design. The causal interpretation still depends on exchangeability, consistency and overlap.

The simulation deliberately avoids hidden confounding so that extreme weights can be studied without mixing two failures. In a real analysis, these difficulties can coexist: even perfectly behaved weights cannot address an unmeasured common cause, and even a sufficient confounder set may give unstable estimates if important treatment alternatives are nearly absent.

That dual vulnerability is one reason for examining alternative estimators and designs rather than selecting a single inverse probability weighting specification and treating its output as conclusive.

## Why predictive accuracy does not settle estimator quality

The fitted assignment model predicts whether treatment is likely given X. At higher λ, treatment becomes increasingly predictable. From a conventional classification perspective, strong separation between treated and untreated people may appear to simplify the prediction problem.

For causal estimation it creates a different situation. Precisely because treatment is predictable, observations receiving unlikely assignments become rare and disproportionately influential. A treatment model can be statistically successful as a classifier while the IPW treatment-effect estimate becomes unstable.

This is not a contradiction. The prediction task and causal estimation task optimize different quantities. Predicting A from X asks how well the model reproduces observed assignment. Estimating the ATE asks what the outcome would be if the assignment were changed. The second task needs information about outcomes under both treatment options across the target population.

This distinction also cautions against tuning propensity models solely by out-of-sample classification metrics. A model that improves treatment discrimination may change the weight distribution, the resulting balance, and the finite-sample error of the causal estimator. A model with slightly worse predictive metrics can sometimes produce a more useful nuisance estimate for an explicitly defined causal procedure.

None of this means that propensity predictions should be deliberately inaccurate. It means their quality must be assessed in relation to the causal estimator and its assumptions. The estimator does not become valid because the propensity classifier achieved a high AUC, nor invalid simply because the classification task was difficult.

The experiment's assignment models are simple enough that predictive fitting is not the substantive obstacle. The result shows how much estimator behaviour can change solely because the policy assignment mechanism becomes more selective.

## What a more complete estimator could change

Inverse probability weighting is only one family of estimators for causal effects under conditional exchangeability. Outcome regression can estimate conditional potential-outcome means and average them over the target covariate distribution. Doubly robust procedures combine an outcome model with a treatment model in ways that can preserve consistency when one nuisance component is correctly specified, under the appropriate conditions.

In the present data-generating process, the outcome model is very simple: the conditional mean is linear in treatment and X. A correctly specified outcome regression would exploit that structure and could estimate the effect much more stably than ordinary IPW under severe practical overlap. That would not make propensity weighting mathematically wrong. It would show the value of additional outcome-model assumptions when the assignment mechanism provides limited comparison support.

However, outcome regression can depend heavily on extrapolation when treated and untreated observations occupy different covariate regions. A low-variance coefficient obtained by projecting a fitted function into unsupported regions may be sensitive to misspecification. The apparent efficiency advantage therefore brings a different vulnerability. A model can interpolate or extrapolate smoothly without those predictions being established by observed counterfactual comparisons.

Doubly robust estimators offer useful protection against misspecification of one nuisance model, but the phrase *doubly robust* should not be interpreted as immunity to limited overlap, unmeasured confounding or finite-sample instability. Their identifying assumptions remain, and their influence functions can involve inverse propensities that become large.

The next stages of this series will examine these trade-offs more directly. First it is important to understand why an estimator based on a valid identification identity can be numerically fragile even under a correctly specified assignment model.

## What clipping estimates

Clipping estimated propensities is often described as a variance-reduction technique, but its effect on the target deserves closer attention. The original Horvitz–Thompson identity uses the true conditional assignment probabilities. Replacing them with clipped values changes the expectation of the weighted terms.

For example, if e(X)=0.01 for a particular covariate value but clipping replaces it with 0.05, the treated outcomes in that region are weighted at one fifth of their original inverse propensity value. The weighted distribution no longer exactly reconstructs the original population distribution under the treatment assignment mechanism. Whether the resulting bias is substantial depends on the outcome relationship and on how much population mass is affected.

In the present simulation, 33.05% of estimated propensities are clipped on average at λ=3.0. The change is large enough that the normalized effect estimate is substantially biased upward. The outcome model's strong dependence on X makes the reweighting distortion consequential.

An alternative would be to exclude covariate regions with inadequate overlap and explicitly target a restricted population. Another is to use overlap weights that intentionally emphasize people who might plausibly receive either treatment. Those approaches do not necessarily estimate the original ATE, but their target can be clearly defined and scientifically useful.

By contrast, simply clipping and continuing to label the result as an estimate of the full-population ATE can conceal a change in what the calculation represents. It may be defensible as a pragmatic sensitivity analysis, provided the modification and resulting target uncertainty are acknowledged. It is not a free numerical correction.

The bias–variance comparison in the table should therefore be viewed as a diagnostic of estimator behaviour, not a recommendation to adopt the clipping setting with the smallest observed RMSE. In real data the true effect is unknown, and the threshold cannot be chosen by directly minimizing error against it.

## A practical evaluation plan for weighting analyses

A weighting analysis should begin with a target trial or intervention question. Specify who is eligible, what treatment options mean, which outcome and time horizon are relevant, and which population's effect is desired. These details determine whether the ATE, ATT or an overlap-population effect is the appropriate estimand.

Next, describe the assignment mechanism and adjustment covariates. Variables measured before treatment that jointly influence assignment and potential outcomes are candidates for adjustment. Variables affected by treatment may play different causal roles and should not be added merely because they improve assignment prediction. The temporal structure matters.

Before interpreting an effect, examine the distribution of the estimated propensities by treatment group. Look for extreme probabilities and regions with few comparable observations. Report summary weight diagnostics, including maxima, quantiles and effective sample sizes. State whether weights were normalized, stabilized, clipped or used to restrict the population.

Check covariate balance after weighting. The relevant question is whether the weighting scheme constructed the measured comparison it claims to construct. Poor balance suggests a problem with nuisance estimation or available support; good balance is encouraging but does not establish absence of hidden confounding.

Compare appropriate estimators and report both their effect estimates and uncertainty. When alternatives disagree substantially, investigate whether the differences arise from weight concentration, changes in the target population, outcome-model assumptions or sensitivity to a handful of influential observations. Do not select an estimator solely because it produces the narrowest interval.

Finally, report the causal assumptions that remain unverified. Conditional exchangeability is not testable from the observed treatment-outcome table alone. Positivity can be inspected empirically but finite samples cannot prove population support at every relevant covariate value. Consistency depends on how treatment is defined and implemented. A transparent analysis separates evidence about these conditions from numerical diagnostics.

None of these steps guarantees a correct causal estimate. They help identify whether the proposed effect is being supported by data, assumptions, extrapolation, or a mixture of all three.

## Limits of the experiment

The simulation covers three logistic assignment strengths, one observed baseline covariate, a constant treatment effect and a linear outcome equation. All observations are independent. X and the outcome disturbance are Gaussian, treatment is binary, and the fitted propensity model has the correct functional form. There are no missing outcomes, post-treatment covariates, interference or unmeasured confounders.

The estimated propensity scores are fitted afresh for each replicate, but only one logistic model and one fixed clipping threshold are studied. A systematic sensitivity analysis would vary sample size, outcome noise, the propensity estimation method, clipping thresholds and treatment-effect heterogeneity. Those choices could alter the relative RMSE of the weighting estimators.

The experiment's strong-selection setting still satisfies positivity at every finite X. Consequently, the numerical instability is finite-sample and practical rather than a demonstration of structural nonidentification. Other settings could involve exact zero treatment probabilities, in which case the full-population effect would require a different identification strategy or assumptions about unsupported interventions.

The comparison does not provide analytical confidence intervals for individual IPW estimates. The reported between-replicate standard deviations and RMSE values are available because the experiment repeatedly regenerates the entire dataset and the causal truth is known. They are not substitutes for a justified uncertainty procedure in an observational study.

Effective sample size is reported using Kish's weight-concentration formula, not a design-specific variance estimator. Its relationship to inferential precision depends on the estimator and outcome structure. The simulation also does not compare weighted covariate balance summaries systematically, although the full observed covariate and propensity values are generated and can support such an extension.

Most importantly, the experiment cannot tell us whether an actual employment programme reduces unemployment, how strongly treatment assignment depends on risk, or whether an observational dataset contains sufficient measured confounders. It establishes how the specified statistical procedures behave under a model in which the causal target and assignment mechanism are controlled.

## What the investigation established

Across 300 simulated datasets at each assignment strength, the true treatment effect was one. With moderate selection, ordinary and normalized IPW recovered the effect closely and had RMSE around 0.063. As treatment became increasingly dependent on baseline prognosis, weights became concentrated and weighted estimates became more variable.

At the strongest assignment level, ordinary IPW had RMSE 0.9013, compared with 0.5892 for normalized IPW. The average effective sample size was only about 152 treated and 150 control observations, despite a nominal dataset of 1,500. The average maximum inverse assignment weight exceeded 192, with substantial variation across samples.

Clipping probabilities to [0.05, 0.95] dramatically reduced variation but introduced large positive bias. Its mean estimate was 1.8243, and approximately one third of propensity predictions were modified by the threshold. It produced a quieter estimator, not a more accurate one for the original causal quantity.

The central lesson is about information rather than formula choice. Inverse probability weighting is grounded in a valid causal identification identity under exchangeability, consistency and positivity. Yet a valid identity does not ensure that a finite dataset contains stable support for the counterfactual comparisons it requires. When treatment assignment nearly determines treatment status, the estimator may rely on a few rare observations to represent much of the population.

The statistical problem is not solved by adding records, fitting a more accurate treatment classifier or clipping inconvenient weights without checking the consequences. The analysis must explain which population's effect is being estimated, how much overlap the data provide, and what compromises are introduced to make the estimate usable. A large observational dataset can still contain little information about a question its assignment mechanism rarely allows the analyst to observe.

## Code

The complete experiment is implemented in \`causal_inference_statistics/part4/experiment.py\` in the Medium-Blog repository. It uses Python 3.12 or later, NumPy, SciPy and scikit-learn. It generates 300 datasets at each of three assignment strengths, re-estimates logistic propensity models and reports all effect estimates and diagnostics. The seed is fixed at 20261016; no external dataset is required.

## The series

1. When Regression Gets the Causal Effect Wrong
2. Randomization, Confounding and the Value of Experimental Design
3. Matching and Propensity Scores: When Balanced Data Still Give the Wrong Causal Effect
4. Inverse Probability Weighting: When a Few Observations Control the Result (this article)
5. Directed Acyclic Graphs and the Cost of Controlling for Everything
6. Difference-in-Differences and the Parallel Trends Assumption

## References

Austin, P. C. (2011). An Introduction to Propensity Score Methods for Reducing the Effects of Confounding in Observational Studies. *Multivariate Behavioral Research*, 46(3), 399–424. https://doi.org/10.1080/00273171.2011.568786

Cole, S. R. and Hernán, M. A. (2008). Constructing Inverse Probability Weights for Marginal Structural Models. *American Journal of Epidemiology*, 168(6), 656–664. https://doi.org/10.1093/aje/kwn164

Crump, R. K., Hotz, V. J., Imbens, G. W. and Mitnik, O. A. (2009). Dealing with Limited Overlap in Estimation of Average Treatment Effects. *Biometrika*, 96(1), 187–199. https://doi.org/10.1093/biomet/asn055

Hajek, J. (1971). Comment on 'An Essay on the Logical Foundations of Survey Sampling'. In V. P. Godambe and D. A. Sprott (eds.), *Foundations of Statistical Inference*. Holt, Rinehart and Winston.

Hernán, M. A. and Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC. https://miguelhernan.org/whatifbook

Horvitz, D. G. and Thompson, D. J. (1952). A Generalization of Sampling Without Replacement from a Finite Universe. *Journal of the American Statistical Association*, 47(260), 663–685. https://doi.org/10.1080/01621459.1952.10483446

Imbens, G. W. (2004). Nonparametric Estimation of Average Treatment Effects Under Exogeneity: A Review. *Review of Economics and Statistics*, 86(1), 4–29. https://doi.org/10.1162/003465304323023651

Kang, J. D. Y. and Schafer, J. L. (2007). Demystifying Double Robustness: A Comparison of Alternative Strategies for Estimating a Population Mean from Incomplete Data. *Statistical Science*, 22(4), 523–539. https://doi.org/10.1214/07-STS227

Kish, L. (1965). *Survey Sampling*. John Wiley & Sons.

Li, F., Morgan, K. L. and Zaslavsky, A. M. (2018). Balancing Covariates via Propensity Score Weighting. *Journal of the American Statistical Association*, 113(521), 390–400. https://doi.org/10.1080/01621459.2016.1260466

Lunceford, J. K. and Davidian, M. (2004). Stratification and Weighting via the Propensity Score in Estimation of Causal Treatment Effects: A Comparative Study. *Statistics in Medicine*, 23(19), 2937–2960. https://doi.org/10.1002/sim.1903

Matsouaka, R. A. and Zhou, Y. (2024). Causal Inference in the Absence of Positivity: The Role of Overlap Weights. *Biometrical Journal*, 66(4), e2300156. https://doi.org/10.1002/bimj.202300156

Robins, J. M., Hernán, M. A. and Brumback, B. (2000). Marginal Structural Models and Causal Inference in Epidemiology. *Epidemiology*, 11(5), 550–560. https://doi.org/10.1097/00001648-200009000-00011

Rosenbaum, P. R. and Rubin, D. B. (1983). The Central Role of the Propensity Score in Observational Studies for Causal Effects. *Biometrika*, 70(1), 41–55. https://doi.org/10.1093/biomet/70.1.41

Rubin, D. B. (1974). Estimating Causal Effects of Treatments in Randomized and Nonrandomized Studies. *Journal of Educational Psychology*, 66(5), 688–701. https://doi.org/10.1037/h0037350

Stürmer, T., Rothman, K. J., Avorn, J. and Glynn, R. J. (2010). Treatment Effects in the Presence of Unmeasured Confounding: Dealing with Observations in the Tails of the Propensity Score Distribution—A Simulation Study. *American Journal of Epidemiology*, 172(7), 843–854. https://doi.org/10.1093/aje/kwq198

Westreich, D. and Cole, S. R. (2010). Invited Commentary: Positivity in Practice. *American Journal of Epidemiology*, 171(6), 674–677. https://doi.org/10.1093/aje/kwp436

Zhou, Y., Matsouaka, R. A. and Thomas, L. (2020). Propensity Score Weighting Under Limited Overlap and Model Misspecification. *Statistical Methods in Medical Research*, 29(12), 3721–3756. https://doi.org/10.1177/0962280220940334
