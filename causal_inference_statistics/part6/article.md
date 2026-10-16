# Difference-in-Differences: When Parallel Trends Fail

*Part 6 of Causal Inference in Statistics. Three identification regimes, 1,500 simulated datasets and a pre-trend placebo that cannot detect what happens after intervention. Python code provided.*

## Introduction

Difference-in-Differences (DID) is attractive because it seems to replace an unavailable randomized experiment with an elementary calculation. Observe a treated group before and after a policy, observe a comparison group over the same periods, and subtract the comparison group's change from the treated group's change. If the groups differ in permanent characteristics, differencing removes those differences. If both groups also experience common shocks, taking a second difference removes those too.

Neither observation is enough to establish a causal effect. The treated group's outcome in the absence of the policy is missing after implementation. DID replaces that missing counterfactual trend with the change observed in the comparison group. The crucial assumption is therefore not that the two groups have the same average outcome before treatment, but that their *untreated changes* would have been the same across the period used for identification.

This article examines how much that distinction matters. I constructed three data-generating processes with an intervention that genuinely lowers the outcome by one unit. In the first, the untreated trends are parallel. In the second, the treated group's untreated outcome experiences a new post-policy change of 0.75 units. In the third, treatment-related anticipation changes the treated group's outcome by −0.4 units at the last pre-policy observation. Each scenario is investigated using 500 independent datasets, with 500 treated and 500 comparison individuals observed over four periods.

The mean DID estimate is −0.9951 in the valid design, −0.2469 when the untreated post-period trend diverges, and −0.5977 under anticipation. The nominal 95% intervals cover the true effect in 95.4%, 0% and 0.2% of simulations, respectively. Yet a placebo comparison using two earlier periods rejects a null of equal earlier trends only about 5% to 6% of the time in all three scenarios. Passing that placebo tells us about the periods it covers, not what the unobserved future counterfactual would have been.

The objective is not to dismiss DID. Under a defensible design it can recover policy effects that a simple before-after comparison cannot. The objective is to separate the algebra of the estimator from the assumptions that give it a causal interpretation, and to see why a statistically convincing estimate may be uninformative about the desired intervention.

## The policy question

Suppose a regional employment agency introduces a programme intended to reduce unemployment duration. Some regions receive it and others do not. A regression comparing average unemployment duration after implementation could be confounded by permanent regional differences: one region may have older workers, different industries or consistently fewer vacancies. An ordinary before-after comparison within treated regions has another problem: the entire labour market may be improving or deteriorating during the programme.

DID combines the two comparisons. Permanent differences between regions can disappear when changes are calculated, while a common time shock can disappear when the comparison-region change is subtracted. This is a substantial advantage, but it does not remove changes specific to treated regions that would have happened without the programme.

If treated regions were already on a different trajectory, if an industry-specific shock affected only those regions after implementation, or if workers anticipated the programme and changed behaviour before its nominal start, the second difference may not isolate the policy. The estimation procedure will still return a coefficient and a standard error. The scientific problem is whether the coefficient answers the intended counterfactual question.

Our simulation uses generic continuous outcomes rather than historical unemployment records. The labour-market example clarifies the decision, but the numerical evidence concerns the specified synthetic structural model only. That separation matters: simulated identification failures should not be presented as estimates of any particular real-world employment programme.

## The causal estimand

Let G=1 indicate membership in the group that will receive treatment at the final observation and G=0 membership in the untreated comparison group. Let Y_t(1) and Y_t(0) denote outcomes under treatment and no treatment at time t. For a policy introduced at the final period, the average treatment effect among treated units is

[
ATT=E[Y_3(1)-Y_3(0)mid G=1].
]

The observed post-treatment mean among treated units reveals E[Y_3(1)|G=1], subject to consistency and correct implementation. The missing quantity is E[Y_3(0)|G=1]: the average outcome those same individuals would have experienced at time 3 without the policy.

The observed comparison group's evolution may supply a counterfactual change, but it cannot tell us the treated group's missing outcome without an assumption. Consider the two-period DID contrast using periods 2 and 3,

[
D={E[Y_3mid G=1]-E[Y_2mid G=1]}
-{E[Y_3mid G=0]-E[Y_2mid G=0]}.
]

Under no anticipation at the baseline period, Y_2 is an untreated outcome for both groups. The identifying parallel-trends condition then states

[
E[Y_3(0)-Y_2(0)mid G=1]
=
E[Y_3(0)-Y_2(0)mid G=0].
]

This is about the two *potential untreated trends*. It is not the claim that observed treated and comparison outcomes have identical levels, or that treatment is randomly assigned. DID permits baseline differences between groups; the assumption concerns how they would have changed without intervention.

When parallel trends, no anticipation, consistency and the relevant comparison-group treatment conditions hold, the observed DID contrast identifies the ATT. That final qualification is important: a change in the comparison group's outcome caused by spillovers from the policy can spoil the interpretation even if the groups would otherwise have parallel trends.

## Algebra of a failed comparison

The bias can be written without specifying a linear regression. Let B be the untreated trend difference:

[
B=
E[Y_3(0)-Y_2(0)mid G=1]
-E[Y_3(0)-Y_2(0)mid G=0].
]

If there is no anticipation and the comparison group remains untreated, the expected DID contrast equals ATT+B. The policy effect and the untreated differential trend are combined in the observed estimate. Under parallel trends B=0; otherwise the error is exactly B in this simple design.

Anticipation produces a related but distinct failure. Let the treated group experience a treatment-related response a at baseline period 2, before official implementation, while the comparison group does not. When the effect at time 3 is τ, the observed treated change includes τ−a, so DID targets τ−a even if the underlying untreated trends are parallel.

For τ=−1 and a=−0.4, the post-minus-baseline treatment-related change is −1−(−0.4)=−0.6. The estimator looks as though the intervention reduced the outcome by only 0.6, although the treatment's final-period effect relative to no programme is still −1. What fails is the assumption that baseline period 2 represents an untreated state.

These decompositions expose why the estimator itself cannot distinguish policy from counterfactual drift. Both components enter the same observed change. A flexible regression or a more powerful computer can estimate D precisely, but cannot separate ATT from B or a without further design information.

## The data-generating experiment

There are four observations per individual, indexed t=0,1,2,3. The groups contain 500 individuals each. A unit-specific baseline term, a common time sequence and independent errors generate untreated outcomes:

[
Y_{it}(0)=alpha_i+lambda_t+1{G_i=1}+arepsilon_{it},
]

where α_i and ε_it are independent standard normal draws and λ=(0,0.2,0.4,0.6). The additional unit for treated-group membership is a permanent difference in levels, not a causal treatment effect. It is included to demonstrate that DID can succeed despite baseline group differences.

In the parallel-trends scenario, the final-period outcome of treated individuals is lowered by τ=−1, and nothing else changes. The common time trend is exactly shared. This is the favourable identification design.

In the differential-post-trend scenario, treated-group untreated outcomes receive an additional +0.75 at t=3, after the baseline period. They then receive the treatment effect −1. Consequently the observed post-treatment change incorporates both the true effect and a +0.75 counterfactual shock that the comparison group does not experience.

In the anticipation scenario, treated individuals experience −0.4 at t=2, the period used as the baseline for the DID comparison. The intervention effect at t=3 remains −1. Earlier periods t=0 and t=1 are unaffected, so a placebo trend comparison using those two periods cannot see anticipation that begins only later.

The individual fixed components, group level offset and common time effects are present in all three scenarios. Only the additional post-period untreated drift or baseline anticipation distinguishes the failure cases. This controlled construction makes the source of bias identifiable by inspection and permits mathematical predictions for the estimator's mean.

## Estimation and inference

The estimator is computed as the difference between the group-specific sample means of individual changes from t=2 to t=3. With two groups and two periods, that is algebraically equivalent to the coefficient on a treated-group-by-post-period interaction in a saturated DID regression with group and period indicators.

For inference, each person's change is treated as the unit of analysis. Within each group the sample variance of those changes is divided by the number of individuals in the group; the two terms are added, and the square root is the estimated standard error. This allows unequal change variances between the groups and retains the correlation among repeated measurements of the same individual by differencing first.

The simulation forms an approximate 95% interval using 1.96 times this standard error. In this independent-individual synthetic design, that calculation behaves as expected. It is not intended as an inferential recipe for studies where treatment is assigned to a small number of regions, schools, firms or other clusters.

This difference matters enormously in applied DID. If an intervention occurs at regional level, hundreds of thousands of individual records do not create hundreds of thousands of independently assigned interventions. Outcomes within regions can have serially correlated shocks, and conventional individual-level standard errors can greatly overstate precision. Appropriate clustering and design-specific inference are not optional details.

The placebo comparison uses the group difference in changes from t=0 to t=1, two periods that are unaffected by policy or anticipation in all three mechanisms. It applies the same two-sample standard error calculation. The fraction of replications in which the placebo t-statistic exceeds 1.96 in absolute value is recorded as the rejection rate.

## Monte Carlo results

The experiment contains 500 independent replications of each scenario, each with 1,000 individuals. The results are:

| Design | Mean estimate | Bias relative to −1 | SD across datasets | Mean standard error | 95% causal coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Parallel untreated trends | −0.9951 | +0.0049 | 0.0903 | 0.0894 | 95.4% |
| Differential post-policy trend | −0.2469 | +0.7531 | 0.0907 | 0.0894 | 0.0% |
| Anticipation at baseline | −0.5977 | +0.4023 | 0.0896 | 0.0894 | 0.2% |

The parallel-trends estimate is close to the structural effect −1, with coverage near the nominal level. The permanent group offset of one outcome unit is removed by differencing. The common time sequence is removed by subtracting the comparison group's change. This is exactly what DID is designed to do.

Under differential post-policy trends, the expected DID effect is −1+0.75=−0.25. The simulated mean −0.2469 agrees with that derivation. Under anticipation the expected contrast is −1−(−0.4)=−0.6. The simulated mean −0.5977 likewise agrees with the theoretical value.

The failure cases do not display unusually large fitted standard errors. Their mean standard errors remain approximately 0.089, essentially the same as for the correctly identified design. The sampling calculation describes the variability of the DID contrast across similarly generated datasets; it does not account for the missing identification assumptions.

That is why 95% causal coverage falls to almost zero under the two violations. The intervals become precise around an estimand that differs from the true intervention effect. A confidence interval can accurately quantify sampling uncertainty and still give a misleading account of causal uncertainty.

The two biases are not produced by the random seed or a handful of anomalous samples. The data-generating equations predict them exactly in the population. Monte Carlo replication confirms their magnitude and reveals how little the ordinary sampling standard error warns us about the problem.

## What the pre-period placebo measures

The placebo comparison from t=0 to t=1 has the following results:

| Scenario | Mean placebo estimate | Two-sided 5% rejection rate |
| --- | ---: | ---: |
| Parallel trends | −0.0003 | 4.8% |
| Differential post-treatment trend | +0.0003 | 6.4% |
| Anticipation | +0.0039 | 5.4% |

Every scenario shares the same expected change between those early periods. A properly calibrated test should reject a true null approximately 5% of the time, subject to finite Monte Carlo variation. That is what the simulation shows. The placebo does not distinguish the design that supports DID from the two designs that do not.

It is tempting to interpret the absence of a statistically significant pre-trend difference as confirmation of parallel trends. The experiment shows why that inference is not justified. A test of earlier observed differences does not test the unobserved post-treatment counterfactual difference directly. More pre-periods can provide more opportunities to detect pre-existing divergence or complicated dynamics, but they cannot logically rule out a new treated-specific shock after intervention.

In the anticipation scenario, the placebo periods were deliberately chosen before the anticipatory response begins. If an analyst instead compared a pre-period trend including the affected t=2 observation, the anticipation might become visible. That is a useful diagnostic distinction, not a contradiction. A placebo only tests the interval it actually covers; study design must determine which intervals could plausibly have been affected.

There is also a statistical issue with low-powered pre-trend tests. If a real differential trend is modest relative to sampling noise, failing to reject can be common even when it is substantively important for the later estimate. Conditioning analysis choices on whether a pretest happened to reject can create additional inferential problems. The absence of statistical evidence for divergence should never be presented as evidence that future untreated trends must coincide.

## Common levels are not the assumption

The treated group's outcome in this experiment is one unit higher than the control group's even before treatment, because of the fixed group offset. That difference does not make DID fail under parallel trends. Differencing removes it by construction.

In an applied study, insisting that treated and control units have the same average baseline outcome would discard many potentially credible DID designs. Conversely, finding equal baseline outcome levels would not establish parallel trends. Groups can happen to meet at one time while following different trajectories.

What matters is the counterfactual change, and that change may depend on observed baseline characteristics, group composition, timing, or sector-specific exposure to shocks. Conditional parallel trends is a possible identifying assumption when untreated changes would be comparable after conditioning on appropriate pre-treatment variables. Weighting, matching or regression adjustment can help implement that comparison, but they add their own modelling and overlap requirements.

The central challenge remains identifying why the comparison group's untreated change is informative about the treated group's unobserved untreated change. A credible institutional argument, appropriate comparator selection and sensitivity analyses are more important than a visually similar starting level.

## Why a new post-period shock is dangerous

The differential-trend scenario introduces a +0.75 shift in the treated group's untreated outcome at the same time the programme is introduced. A conventional DID regression has no direct way to separate that shift from the intervention effect. Both occur for the treated group in the post period, so both enter the interaction coefficient.

This is the basic threat from coincident policies, changing economic conditions or selection into implementation based on expected future outcomes. Suppose a government introduces employment support in regions anticipating a sectoral downturn. The downturn could make unemployment worse in those regions relative to controls even without the programme. DID might estimate a much smaller benefit, no benefit or harm, depending on how those shocks compare with the programme effect.

An appropriate comparison group should therefore be selected using information about factors expected to shape untreated trends, not merely by choosing whichever units resemble the treated group on a few static characteristics. Multiple control groups, alternative time windows and complementary outcome variables may help assess whether the result is robust to plausible differential shocks.

None of those exercises proves the counterfactual assumption. They can, however, limit the range of explanations and make the analysis more transparent. The goal is a defensible causal design, not an interaction coefficient that happens to be statistically significant.

## Anticipation and policy timing

Anticipation is conceptually different from a newly divergent untreated trend. In the simulated anticipation scenario, the treated group changes behaviour at t=2 in response to the expected programme, before the formal treatment period t=3. The outcome at t=2 is therefore not an untreated baseline for the desired causal contrast.

Using the t=2 value subtracts part of the programme-related response from the later change. The estimate −0.6 represents the additional observed change from the anticipatory state to the final-period treated state. It is not the total contrast between final treatment and no programme.

In real settings, anticipation may arise when legislation is announced before enforcement, employers prepare for a tax change, households change purchases before a subsidy begins, or programme participants adjust behaviour when told that treatment is coming. The correct intervention time may be the announcement, the start of implementation, the first exposure or some combination, depending on the question.

Researchers should document the policy timeline and examine which baseline periods plausibly precede any response. Excluding periods immediately before rollout, studying announcement effects separately, or redefining the treatment can sometimes improve the design. Each choice should be justified by the intervention mechanism rather than made solely to produce a stronger coefficient.

The simulation's early placebo does not detect anticipation because the affected period is deliberately outside the tested interval. This emphasizes that a well-calibrated placebo is not necessarily a relevant placebo. Its scientific value depends on whether it covers the mechanism that threatens identification.

## What pre-trend plots can and cannot do

Plotting average outcomes over several periods is a useful part of a DID analysis. The figure can reveal abrupt shifts, gradual drift, unusual volatility, changing group composition or evidence that treatment-related behaviour began before the assumed policy date. An event-study plot can display differences relative to a designated reference period and show how the observed treated-control gap evolves over time.

But a flat sequence of pre-treatment coefficients does not establish the unobserved post-treatment counterfactual. A new shock can arrive with the policy. Anticipation can begin after the last placebo period. A short or noisy pre-treatment series can hide economically important drift. Inference about counterfactual trends needs substantive assumptions beyond the line drawn through earlier observations.

It is also important not to interpret every event-study coefficient as a separate dynamic causal effect without examining the estimator and treatment timing. In staggered-adoption designs, conventional two-way fixed effects can produce weighted combinations of different group-time effects, including comparisons that use already treated units as controls. When treatment effects are heterogeneous, those coefficients need not have the intended interpretation.

A well-designed figure should therefore identify which groups contribute to each comparison, what period is the reference, whether treatment is absorbing, and which assumptions identify the estimands being plotted. Figures are evidence about observed trajectories. They are not a substitute for a description of the counterfactual construction.

## Multiple groups and staggered adoption

The present experiment has one treated group, one never-treated comparison group and a single implementation date. It therefore avoids several complications that arise when regions or firms adopt a policy at different times.

In staggered-adoption data, a two-way fixed-effects regression is often written as

[
Y_{it}=alpha_i+lambda_t+eta D_{it}+arepsilon_{it},
]

where D_it indicates whether unit i has been treated by time t. With heterogeneous treatment effects, the single coefficient β can mix different comparisons and weight them in ways that are difficult to interpret causally. Some comparisons may use groups already exposed to treatment as controls for later adopters, which is problematic when effects evolve over time.

Recent methodological work has clarified these issues and developed estimators of group-time average treatment effects using appropriate untreated comparison groups. The approaches differ in their assumptions, weighting, outcome modelling and aggregation choices, but the shared principle is that an estimand should be defined before a regression coefficient is treated as a universal policy effect.

Callaway and Sant'Anna (2021), Sun and Abraham (2021), and Goodman-Bacon (2021) provide important frameworks for understanding the estimands and the decomposition of conventional estimators. Their results do not imply that every two-way fixed-effects coefficient is misleading; rather, the conditions under which it has a desired interpretation must be examined.

Our simple simulation does not test staggered-adoption estimators. It establishes the foundational counterfactual trend problem that remains relevant even before heterogeneous adoption timing complicates the analysis.

## Inference with grouped interventions

An applied DID study may observe thousands of people but assign treatment at the school, hospital, firm, municipality or state level. Outcomes can share shocks within those units across time. Treating every individual-period observation as independent can lead to confidence intervals that are much too narrow.

Bertrand, Duflo and Mullainathan (2004) showed how serial correlation and the structure of policy variation can produce severely misleading conventional DID standard errors. The relevant independent variation often comes from the number and allocation of treated and comparison clusters, not the number of rows in a panel.

Clustering standard errors at an appropriate treatment-assignment level is a common response, but small numbers of clusters can make conventional large-sample cluster-robust approximations unreliable. Alternatives such as design-based inference, wild cluster bootstrap procedures and carefully specified randomization or permutation methods depend on the actual assignment mechanism.

In the synthetic experiment, treatment is a fixed group label for independent individuals. Independent individual-level disturbances are drawn, and no shared region-time shocks are present. Differencing each person's outcomes followed by a two-sample standard error is appropriate for this narrow design. One should not carry that calculation over unchanged to a policy applied to a handful of regions.

This limitation is central to reporting responsible uncertainty: a credible DID identification assumption and a valid standard-error calculation are two separate requirements.

## Composition and missing outcomes

DID also requires attention to who is observed in each period. If the treated group's composition changes after implementation, its observed average outcome may change because different people enter or leave the sample. That change can be mistaken for a policy effect when the comparison group does not experience the same composition shift.

Balanced panels help track the same units over time, but they do not automatically solve selective attrition. If remaining observable depends on treatment and potential outcomes, restricting analysis to complete records can create selection bias. Repeated cross-sections require their own comparability assumptions and sometimes adjustments for changing population composition.

The present simulation observes every person in every period. That avoids the issue but should not be read as a claim that standard DID automatically handles missing data. In a real evaluation, the source of missingness, treatment timing, outcome definition and population membership must be addressed in the design.

A programme may also displace outcomes from controls to treated individuals rather than create net population benefits. If treated workers obtain jobs that would otherwise have gone to comparison workers, control outcomes can be affected by the intervention. Such spillovers violate the clean untreated-comparison interpretation. The causal target may then need to include equilibrium or interference effects rather than an isolated individual response.

## Alternative diagnostics and sensitivity analysis

No single pre-trend significance test can validate parallel trends. A more informative analysis combines policy history, comparator choice, long enough pre-treatment trajectories where available, negative-control outcomes, alternative reference periods and sensitivity to plausible departures from the counterfactual trend assumption.

One useful strategy is to quantify how much untreated divergence would be required to change a substantive conclusion. In the differential-trend simulation, the answer is explicit: a +0.75 untreated change shifts a true effect of −1 to a DID contrast of −0.25. In real data that divergence is unknown, but external evidence or restrictions on plausible trend changes can be used to express the resulting range of treatment effects.

Rambachan and Roth (2023) develop an approach to inference under restrictions on violations of parallel trends, moving beyond the assumption that no deviation is possible. The choice of restrictions remains substantive: confidence intervals conditional on an implausibly tight restriction may give a misleading impression of robustness.

Placebo interventions, alternative control groups and falsification outcomes can uncover particular weaknesses. Their failure to reject does not prove that every possible source of bias is absent. The purpose is to put the causal interpretation under pressure from plausible alternatives, not to accumulate non-significant tests as certificates of validity.

The simulation's two failure mechanisms illustrate exactly why that distinction matters. Different assumptions can generate the same early observed trend pattern yet imply different post-treatment counterfactuals and therefore different causal interpretations of the same DID regression.

## What this experiment does not establish

This study has two groups, four periods, a constant treatment effect, independent Gaussian individual-level shocks and fixed group composition. Treatment occurs for the designated group only at the final period; in the anticipation scenario the baseline deviation is explicitly modelled as a response to the expected programme. No outcome censoring, treatment noncompliance, spillovers or heterogeneous effects are included.

The post-policy divergence is imposed as a +0.75 shift in the untreated potential outcome of the treated group. This is an intentionally severe and precisely specified violation, not a claim about the size of unobserved trends in real policy data. Likewise, the anticipation effect −0.4 is chosen to make its direction and algebra visible.

The placebo test examines t=0 to t=1, whereas the DID baseline is t=2. This is essential to the design: the anticipation effect at t=2 will not be visible to a test ending at t=1. An analysis that examines the full available pre-treatment trajectory may uncover more evidence, although no historical comparison can directly observe the untreated post-treatment counterfactual.

The confidence intervals use independent individual changes and a large-sample normal critical value. They do not illustrate cluster-robust inference, event-study uncertainty, multiple testing or finite-cluster procedures. The simulation should not be used as a benchmark of those methods.

Most importantly, the study does not estimate any historical policy's effect. It demonstrates what happens to a familiar estimator when the identifying assumptions are satisfied or violated under a controlled mechanism. A credible empirical DID analysis must argue why its counterfactual design is appropriate for its particular intervention, population, timing and outcome.

## Conclusion

With valid parallel trends, a two-period DID comparison recovered the true policy effect of −1, producing a mean estimate of −0.9951 and nominal 95% causal coverage of 95.4%. A new untreated trend affecting only the treated group shifted the mean estimate to −0.2469. An anticipatory response in the baseline period shifted it to −0.5977. In both failure cases, the ordinary standard errors remained close to those of the valid design and causal interval coverage collapsed.

The pre-period placebo test rejected at approximately its nominal rate in all three scenarios because those earlier periods were constructed to have the same trend. Its reassuring result was correct for the hypothesis it tested and insufficient for the causal question that mattered.

Difference-in-Differences is not made causal by subtracting two differences. Its identifying content comes from the assertion that the comparison group's untreated change represents the treated group's missing untreated change. The assertion may be credible for a particular policy and comparison population, but it has to be defended through design, timing, institutional context and sensitivity analysis.

The next methods in this series will investigate alternative sources of identification, including instruments and discontinuities, and what happens when those designs' own assumptions are challenged. They face the same broad scientific requirement: estimation can quantify an identified contrast, but it cannot make an unsupported counterfactual observable.

## Code and reproducibility

The experiment is implemented in `causal_inference_statistics/part6/experiment.py`. Running it with NumPy under Python 3.12+ writes `results.csv` alongside the script and prints Monte Carlo summaries for each scenario. The seed is fixed at 20261016. All quantities are obtained by directly generating the stated equations; no external dataset is used. The numerical results distinguish the true effect from the sampling distribution of the DID contrast, and the pre-period placebo uses periods distinct from the DID baseline.

## The series

1. When Regression Gets the Causal Effect Wrong
2. Randomization, Confounding and the Value of Experimental Design
3. Matching and Propensity Scores: When Balanced Data Still Give the Wrong Causal Effect
4. Inverse Probability Weighting: When a Few Observations Control the Result
5. Directed Acyclic Graphs and the Cost of Controlling for Everything
6. Difference-in-Differences: When Parallel Trends Fail (this article)

## References

Abadie, A. (2005). Semiparametric Difference-in-Differences Estimators. *Review of Economic Studies*, 72(1), 1–19. https://doi.org/10.1111/0034-6527.00321

Angrist, J. D. and Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.

Bertrand, M., Duflo, E. and Mullainathan, S. (2004). How Much Should We Trust Differences-in-Differences Estimates? *Quarterly Journal of Economics*, 119(1), 249–275. https://doi.org/10.1162/003355304772839588

Callaway, B. and Sant'Anna, P. H. C. (2021). Difference-in-Differences with Multiple Time Periods. *Journal of Econometrics*, 225(2), 200–230. https://doi.org/10.1016/j.jeconom.2020.12.001

de Chaisemartin, C. and D'Haultfœuille, X. (2020). Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects. *American Economic Review*, 110(9), 2964–2996. https://doi.org/10.1257/aer.20181169

Goodman-Bacon, A. (2021). Difference-in-Differences with Variation in Treatment Timing. *Journal of Econometrics*, 225(2), 254–277. https://doi.org/10.1016/j.jeconom.2021.03.014

Hernán, M. A. and Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC. https://miguelhernan.org/whatifbook

Imbens, G. W. and Wooldridge, J. M. (2009). Recent Developments in the Econometrics of Program Evaluation. *Journal of Economic Literature*, 47(1), 5–86. https://doi.org/10.1257/jel.47.1.5

Lechner, M. (2011). The Estimation of Causal Effects by Difference-in-Difference Methods. *Foundations and Trends in Econometrics*, 4(3), 165–224. https://doi.org/10.1561/0800000014

Rambachan, A. and Roth, J. (2023). A More Credible Approach to Parallel Trends. *Review of Economic Studies*, 90(5), 2555–2591. https://doi.org/10.1093/restud/rdad018

Roth, J. (2022). Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends. *American Economic Review: Insights*, 4(3), 305–322. https://doi.org/10.1257/aeri.20210236

Roth, J., Sant'Anna, P. H. C., Bilinski, A. and Poe, J. (2023). What's Trending in Difference-in-Differences? A Synthesis of the Recent Econometrics Literature. *Journal of Econometrics*, 235(2), 2218–2244. https://doi.org/10.1016/j.jeconom.2023.03.008

Sant'Anna, P. H. C. and Zhao, J. (2020). Doubly Robust Difference-in-Differences Estimators. *Journal of Econometrics*, 219(1), 101–122. https://doi.org/10.1016/j.jeconom.2020.06.003

Sun, L. and Abraham, S. (2021). Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects. *Journal of Econometrics*, 225(2), 175–199. https://doi.org/10.1016/j.jeconom.2020.09.006

Wing, C., Simon, K. and Bello-Gomez, R. A. (2018). Designing Difference in Difference Studies: Best Practices for Public Health Policy Research. *Annual Review of Public Health*, 39, 453–469. https://doi.org/10.1146/annurev-publhealth-040617-013507

Wooldridge, J. M. (2021). Two-Way Fixed Effects, the Two-Way Mundlak Regression, and Difference-in-Differences Estimators. Working paper.