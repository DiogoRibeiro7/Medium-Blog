# From Statistical Evidence to Causal Decisions: When the Best Prediction Makes the Worst Policy

*Part 20 of Causal Inference in Statistics. A final experiment on treatment allocation, counterfactual benefit, policy value and the difference between an accurately estimated causal effect and a useful decision. Five hundred independent randomized trials, held-out populations, and reproducible Python code.*

The first article in this series showed that a regression can estimate the wrong causal quantity with increasing precision. The subsequent articles examined design, confounding, propensity scores, weights, causal diagrams, longitudinal interventions, natural experiments, heterogeneity, Bayesian inference, missing outcomes and transportability. Across those investigations the recurring difficulty was never simply calculating a coefficient. It was identifying the question that coefficient could answer, then determining whether the data, assumptions and estimation procedure justified that interpretation.

There is one further step. Even an internally valid estimate of a causal effect is not automatically a decision. A policymaker may need to allocate an intervention to a limited number of people, compare the consequences of alternatives, account for costs and uncertainty, or decide whether gathering more evidence is worth delaying action. Those questions require a decision rule and a value function, not merely a statistically significant average treatment effect.

I close the series with a deliberately simple allocation experiment. A treatment produces a positive average benefit of 0.5 outcome units. Some people benefit substantially, while others are harmed. The decision-maker can treat only one person in five. The same baseline characteristic that predicts poor untreated outcomes predicts *less* benefit from treatment. This is a setting in which an excellent risk model can support a poor allocation policy.

Across 500 independent simulated randomized trials and held-out target populations, targeting those with the highest baseline risk produces a mean treatment benefit of −1.6003 per selected individual. Random allocation produces 0.5011. Ranking people by a correctly specified model of their treatment benefit produces 2.5989, virtually identical to the oracle that knows the true individual benefits. The difference is not a marginal improvement in the third decimal place of an estimator. It changes the direction of the expected consequence of the policy.

The result does not establish that high-risk targeting is generally harmful, or that estimating heterogeneous effects is easy. Both conclusions would exceed the evidence. The simulation is constructed to make baseline risk and treatment benefit move in opposing directions, and its treatment-effect function is simple enough for a linear interaction model to recover. What it establishes is that the target of prediction and the target of allocation can differ enough for an apparently reasonable decision rule to be systematically wrong.

## The policy decision

Consider a public employment service that can fund intensive support for 20% of newly unemployed people. Historical records describe their characteristics and their likelihood of remaining unemployed without intervention. A conventional targeting strategy is to prioritize people at greatest risk of prolonged unemployment. This makes intuitive sense if the goal is to assist those facing the largest difficulties, and it may be defensible on equity grounds. But if the explicit objective is to maximize the reduction in unemployment caused by limited programme places, baseline risk is not sufficient.

A person can have a high probability of long-term unemployment and receive little benefit from a particular service. Another person might have a moderate baseline risk and respond strongly to the intervention. If the service is intended to maximize additional job placements, choosing by untreated risk alone can allocate places away from those who would benefit most. The relationship between need and responsiveness is an empirical and substantive question, not a statistical identity.

This distinction also appeared in my survival-analysis series. A model that predicts who is likely to remain unemployed can perform well on time-dependent discrimination and calibration while failing to identify who should receive an intervention. Predictive ranking concerns potential outcomes under the observed or untreated regime. Intervention allocation concerns differences between outcomes under competing treatments. The two rankings need not agree.

The present simulation uses a continuous outcome in abstract units. Higher values indicate an undesirable outcome, so a treatment benefit is the reduction in that outcome. The employment example motivates the allocation problem; it is not the dataset underlying the numerical estimates. The benefit function is known because the data are synthetic. In an actual employment programme, identifying that function would require an appropriate experiment or observational design and would introduce substantial additional uncertainty.

The capacity constraint is fixed at 20%. Each policy is evaluated by applying its ranking to an independent held-out population and selecting the top fifth. I measure the *true* mean causal benefit among those selected and the corresponding benefit averaged over the entire population. Because the outcome is synthetic, these values can be calculated directly from potential outcomes; in real data they would need to be estimated.

## A causal estimand and a decision estimand

Let A be a binary treatment indicator and X the baseline characteristic available at allocation. Denote the potential outcomes by Y(1) and Y(0). Because smaller outcomes are preferable, define individual treatment benefit as

\[
B=Y(0)-Y(1).
\]

The average treatment benefit is \(E[B]\). It answers what the mean reduction in outcome would be if everyone received treatment rather than nobody, under the specified intervention and population. In the experiment, this average is 0.5.

A decision rule is a function \(\pi(X)\) that assigns treatment using the observed characteristic. For binary treatment, \(\pi(X)\) takes values zero and one. If the intervention has limited capacity \(q=0.2\), the rule must satisfy a population constraint such as \(E[\pi(X)]\leq q\). With no treatment cost beyond the capacity constraint, the incremental population value relative to treating nobody is

\[
V(\pi)=E[\pi(X)B].
\]

The conditional average treatment benefit is

\[
b(x)=E[B\mid X=x].
\]

If the rule may use only X, the objective is to choose people with the highest positive values of \(b(X)\), subject to capacity. When capacity is an upper bound rather than an obligation, a rational benefit-maximizing rule need not fill places with people predicted to be harmed. In this experiment, the top 20% have positive effects, so the capacity can be filled without that complication.

The mean benefit among selected people is \(E[B\mid\pi(X)=1]\), provided treatment is assigned to a positive fraction of the population. If exactly 20% are selected, population value equals 0.2 times that conditional mean. Both measures are reported because the first describes the people receiving treatment and the second describes what the programme contributes across everyone eligible.

This formulation clarifies why an average treatment effect does not determine an allocation rule. Knowing \(E[B]=0.5\) is not enough to identify which people have \(b(x)>0\), which ones benefit most, or how much value is achievable under a capacity constraint. An average conceals the distribution of benefit and its association with the available allocation variables.

## The experiment's structural model

Generate a baseline characteristic X from a standard normal distribution. Assign treatment A independently with probability one half in the training sample. The untreated outcome is

\[
Y(0)=3-1.2X+\varepsilon,
\]

where \(\varepsilon\sim N(0,1)\) independently of X and assignment. Treatment changes the outcome by

\[
Y(1)=Y(0)-B(X),
\]

with benefit

\[
B(X)=0.5+1.5X.
\]

The expected untreated outcome is \(3-1.2X\). Therefore people with smaller X have higher baseline risk, interpreted as a worse expected outcome without treatment. Their treatment benefit is also smaller because \(B(X)\) increases with X. This deliberate negative relationship creates the possibility that a strategy aimed at high untreated risk selects those least responsive to the intervention.

Because X has mean zero, the population average benefit is

\[
E[B(X)]=0.5+1.5E[X]=0.5.
\]

The benefit varies considerably around that mean. Anyone with \(X<-1/3\) has negative benefit under the structural model. The fact that a population-wide treatment would help on average does not imply every eligible person benefits. Some treatments can impose harms or opportunity costs on individuals even when the mean effect is favourable.

The randomized training sample provides a transparent route to learning benefit differences. Treatment is independent of X, the outcome disturbance and both potential outcomes. Thus treatment-by-X interactions in a correctly specified model can identify the conditional average benefit. There is no unmeasured confounding, selective attrition or treatment noncompliance in the simulated trial.

Those conveniences are deliberate. The final experiment is about the *decision after identification*, not another demonstration of omitted-variable bias. Having made causal identification favourable, we can examine how different allocation targets translate the same evidence into very different consequences.

## Which decision rules are compared?

I compare four allocation strategies. Random allocation gives every held-out individual the same chance of being selected, subject to filling exactly 20% of the available places. It is the elementary baseline: if nobody knows who benefits more, random allocation is preferable to a ranking that systematically chooses harmful cases under this objective.

Baseline-risk targeting selects those with the largest expected untreated outcome \(3-1.2X\). This policy is a strong benchmark rather than a deliberately inaccurate predictor. It ranks untreated risk perfectly in the constructed model. Its failure, if it occurs, cannot be blamed on poor risk prediction; it comes from using risk to answer a treatment-benefit question.

Estimated-benefit targeting fits the randomized trial using a linear outcome regression with an intercept, baseline characteristic, treatment indicator and treatment-by-characteristic interaction. The model estimates how the treatment contrast changes with X. It then ranks held-out individuals by estimated benefit, with larger values receiving priority.

Finally, the oracle ranks individuals by their true structural benefit \(0.5+1.5X\). It is an unattainable benchmark in real applications, but valuable in simulation because it shows the best capacity-constrained decision possible using the model's known benefit function. The difference between oracle and learned policy is policy regret.

All four policies are applied to the *same* held-out population within each replication. This makes their differences about ranking and policy choice rather than changes in population composition. The experiment repeats both trial generation and evaluation population sampling 500 times.

The constant-effect average estimated by randomized treated-minus-control means is also calculated as a diagnostic. It is close to 0.5, showing that the training trial correctly estimates the population average, but it cannot rank who benefits without an effect-modification model.

## How treatment benefit is estimated

The observed outcome model can be written

\[
Y=\beta_0+\beta_1X+\beta_2A+\beta_3AX+\varepsilon.
\]

In the generating mechanism, the true coefficients are \(\beta_0=3\), \(\beta_1=-1.2\), \(\beta_2=-0.5\), and \(\beta_3=-1.5\). The treatment benefit conditional on X is the negative of the difference between fitted outcomes under A=1 and A=0:

\[
\widehat b(x)=-(\widehat\beta_2+\widehat\beta_3x).
\]

This is not a sophisticated causal machine-learning estimator. It is a correctly specified linear interaction model fitted to randomized data. The purpose is to distinguish an accurate causal allocation target from a highly accurate but inappropriate risk target without adding unnecessary approximation error.

The estimated treatment-benefit slope should be positive because the structural benefit increases with X. In these simulations, the randomized trial of 1,200 people identifies its sign sufficiently reliably that the learned rule selects the same top fifth of X as the oracle. This is why their policy values agree to the reported precision. They do not need identical estimates of each person's treatment benefit; they need the same ordering at the allocation boundary.

A flexible nonlinear estimator would not offer an obvious advantage when the true conditional effect is exactly linear and the allocation ranking depends on one variable. This reflects a broader lesson from earlier articles: model sophistication should follow the complexity of the estimand and the data-generating relationship, not precede it as a default requirement.

In real settings the conditional effect may be nonlinear, high-dimensional, poorly supported in subgroups or contaminated by nuisance-model errors. Estimating the ranking can then be difficult. A learned rule should be tested on data not used to choose its form or tune its hyperparameters, and uncertainty about near-ties at the selection threshold may matter substantially.

## The Monte Carlo protocol

Each replication draws 1,200 training observations from the randomized assignment mechanism. Independently, it draws 3,000 individuals for evaluation. The evaluation population contains potential-outcome benefits known to the simulation but hidden from the fitted model. This avoids measuring the decision only on the same people whose outcomes were used to fit the treatment-effect regression.

The random, risk-based, learned-benefit and oracle rankings each select exactly 600 of the 3,000 evaluation observations, corresponding to 20%. For each policy I calculate the mean true treatment benefit among selected individuals. I also multiply this by 0.2 to obtain incremental benefit per eligible person across the complete population, relative to no treatment.

The simulation records the gap between each policy and the oracle, called regret. Because every policy is evaluated against the same held-out potential outcomes in a replication, regret isolates the value lost through an inferior selection rule. An oracle gap of zero means that a policy made the same choices as the oracle in that replication, not that its individual treatment-effect predictions were numerically exact.

The experiment uses 500 independent repetitions with fixed seed 20261018. The script uses NumPy ordinary least squares, and all simulation settings are specified directly in source code. No real-world dataset, pretrained model or external service is required. The reported standard deviations summarize variation over independently drawn trials and evaluation populations.

This is a policy-value benchmark under a known structural model, not a formal uncertainty analysis for off-policy evaluation in observational data. In practice, estimating policy value requires additional consideration of how treatment was assigned and which counterfactual outcomes were observed.

## The results

The primary outcome is the mean *true* benefit among people selected for treatment. Averaged over 500 replications, the results are:

| Allocation policy | Benefit among selected | Across-replication SD | Population benefit | Regret per selected |
| --- | ---: | ---: | ---: | ---: |
| Highest baseline risk | −1.6003 | 0.0422 | −0.3201 | 4.1992 |
| Random allocation | 0.5011 | 0.0618 | 0.1002 | 2.0978 |
| Estimated treatment benefit | 2.5989 | 0.0419 | 0.5198 | 0.0000 |
| Oracle benefit ranking | 2.5989 | 0.0419 | 0.5198 | 0.0000 |

The population average benefit is 0.5. Random allocation is consequently expected to give approximately 0.5 per treated individual, as observed. Random selection has no systematic relationship with X, so the selected group's mean benefit fluctuates around the population average.

Risk targeting does much worse. Because untreated risk is highest when X is smallest, this rule selects people near the lower tail of X, precisely where benefit is lowest. The mean selected benefit is about −1.6. Under the generating equation, delivering treatment to those selected people actually increases the undesirable outcome on average.

Learned-benefit targeting reverses the ranking. It selects people with high X and therefore high benefit, producing about 2.599 benefit units per treated person. The oracle obtains the same value because both rankings depend monotonically on X in the same direction in this experiment.

These are large differences in the chosen units. The mean outcome improvement across the entire eligible population is about −0.320 under risk targeting, +0.100 under random allocation and +0.520 under the learned-benefit policy. The fact that all policies treat the same number of people makes the comparison a consequence of whom they select, not how much treatment they provide.

## The distribution beneath the average

The population average benefit of 0.5 hides a simple but consequential gradient. Because \(B(X)=0.5+1.5X\), differences in X directly translate into differences in causal effects. High X gives large positive benefit; low X gives negative benefit. The average is not wrong. It is simply insufficient for an allocation problem whose value depends on the selected subset.

The highest-risk policy uses a risk score of \(3-1.2X\), so its top 20% corresponds to the bottom 20% of X. For a standard normal variable, the mean of the lowest fifth is approximately −1.4. Substituting into the benefit equation gives roughly \(0.5+1.5(-1.4)=-1.6\), matching the simulation.

The highest-benefit policy selects the upper fifth of X, whose mean is approximately +1.4. The corresponding average benefit is about \(0.5+1.5(1.4)=2.6\), again matching the Monte Carlo output. These population calculations explain why the policy differences persist even with large evaluation samples.

There is no hidden optimization mystery. The simulation establishes the consequence algebraically and checks it numerically. The rankings disagree because untreated prognosis and responsiveness to treatment are designed to move in opposite directions. This is a genuine possibility in allocation problems, but not a claim that it is universally true.

An important distinction follows: estimating the population ATE perfectly would not solve the targeting problem. Every policy in the table exists in the same population with the same average benefit. What changes is the relationship between the ranking variable and the individual's counterfactual difference.

## Why a correct risk model can make the wrong decision

The baseline-risk model is not inaccurate in this experiment. It knows the expected untreated outcome exactly. If the decision were to identify people facing the greatest expected difficulties, or to estimate how many bad outcomes would occur without intervention, its ranking would be ideal for that purpose.

But a treatment policy intended to maximize *improvement caused by treatment* requires a different score. This is the same distinction as prediction versus intervention, now expressed as a capacity-constrained decision. A risk score estimates a level of outcome under one regime. A benefit score estimates a contrast between two regimes.

Conflating them can be especially consequential when interventions have heterogeneous effects. High risk may coincide with high responsiveness, low responsiveness or even harm. Without evidence about that relationship, assigning treatment on risk alone is not a causal optimization strategy. It may still be an explicit ethical or distributive strategy, but then it should be evaluated against its actual objective.

In healthcare, prioritizing the sickest patients can be justified by urgency even when another group would receive a larger incremental benefit. In public services, prioritizing disadvantaged people may satisfy equity goals independently of maximizing aggregate outcome improvements. A decision framework should make those objectives explicit instead of disguising them as treatment-effect estimates.

The numerical result says only that risk targeting performs badly under the chosen *benefit-maximizing* objective and generating equations. It cannot decide how a real society should trade off need, fairness, benefit, cost or access.

## The oracle and the meaning of regret

Policy regret is the difference between the benefit obtained by a benchmark best policy and the benefit obtained by the evaluated policy. With a fixed treatment capacity, define the oracle as the rule selecting the largest true conditional benefits. Then an evaluated policy's regret is the difference in value under the two allocations.

The table reports regret as benefit per selected individual, making the gap easy to read. Multiplying by the treated fraction gives regret on the whole eligible population scale. The high-risk rule loses about 4.199 benefit units per treated person compared with the oracle, equivalent to approximately 0.840 per eligible person at 20% coverage.

Random allocation loses about 2.098 per selected person, or about 0.420 per eligible person. Those numbers are not extra realized observational outcomes. They are contrasts between potential-outcome values of alternative decision policies, known only because the simulation specifies the true treatment effects.

The learned rule has zero observed regret across these replications. That needs careful interpretation. Its effect model is not guaranteed to estimate the intercept and slope without error, and the true individual benefits are not observed in a real trial. Here the ordering is determined by the sign of a single estimated slope. Provided that sign is correct, the ranking is identical to the oracle's ranking regardless of small coefficient errors.

In a nonlinear or multivariable system, two models can have similar mean squared effect prediction error but select different people near the allocation threshold. Conversely, models can have quite different individual-effect predictions while ordering the top fifth identically. Policy regret depends on the decisions produced, not solely on the numerical error of every individual prediction.

This is a reason to evaluate the final policy directly rather than infer its value from model-fitting metrics.

## Statistical uncertainty versus decision uncertainty

A confidence interval for an average treatment effect describes uncertainty about an estimand under a specified causal design and statistical procedure. It does not by itself describe uncertainty about the allocation rule's value. A decision may depend on heterogeneous benefits, unknown costs, future population composition, and error in the estimated ranking.

In the present experiment the randomization and linear effect function make identification and estimation favourable. An average treatment effect of roughly 0.5 is recovered from each training trial with sampling variation. Yet even knowing this average exactly would not identify the top-fifth allocation. Information about the variation in effect with X is essential.

Suppose an effect model is uncertain about whether two individuals have benefits 0.8 and 0.9. If both are well within the selected fifth, their relative ordering may not change the policy. If one lies at the selection boundary, a small error could change who receives treatment. The practical cost of uncertainty depends on the effect difference between those exchanged, not just on the model's global mean squared error.

This suggests evaluating decision stability under resampling, sensitivity to plausible nuisance-model choices, and uncertainty around the capacity threshold. A policy selected after extensive hyperparameter tuning should not be evaluated solely on the data used to choose it. Honest evaluation requires independent or appropriately protected outcome information.

The Monte Carlo replication standard deviations reported here describe changes across independently generated training and target samples. They are not posterior distributions over policy value or externally validated confidence intervals for a real deployment population. Those would require a separate inferential design.

## Off-policy evaluation and identification

In real data, the true benefit of every individual cannot be observed. Treatment reveals one potential outcome, not both. Consequently, evaluating a new allocation policy requires assumptions or an experimental assignment mechanism capable of identifying what would happen under that policy.

Randomized trials with known treatment probabilities can support policy-value estimation using inverse propensity weighting or augmented estimators. If the target policy depends only on measured pre-treatment features and the randomized assignment has adequate support, observed outcomes can be reweighted or modelled to estimate the counterfactual value of deploying that policy.

Observational records require stronger assumptions: treatment assignment must be conditionally exchangeable given the adjustment information, the relevant treatment options must have adequate support, and the intervention must be well defined. Unmeasured confounding can bias policy-value estimation even when the policy ranking is computed using a sophisticated machine-learning model.

This connects directly to Parts 3, 4 and 10. Propensity-score methods and doubly robust estimators can be useful tools for policy evaluation, but their performance depends on the identifying assumptions and the quality of nuisance estimation. They do not turn arbitrary observational treatment histories into a randomized trial.

A policy can also change the environment in which it is deployed. If treating one person affects another person's outcome, the potential-outcome model must account for interference. If assigning services at scale changes provider capacity or market conditions, a rule evaluated under small experimental coverage may not have the same value under wider implementation.

The clean policy comparison in this article assumes away those difficulties. That makes its distinction between prediction and decision clearer, not irrelevant to the difficulties of real-world deployment.

## Average-effect tests and policy decisions

A statistically significant positive average treatment benefit is sometimes taken as sufficient justification to deploy a treatment broadly. That inference skips several questions. Is the treatment beneficial for every subgroup? What is the cost of treatment relative to the expected outcome improvement? Does its benefit exceed the opportunity cost of using scarce capacity elsewhere? Which people would be helped or harmed?

The example gives one answer to the first question: some individuals have negative treatment benefit despite a positive ATE. Treating everyone would improve the population mean outcome by 0.5 units, but it would also impose harm on a subset. Under the fixed 20% capacity, selecting people with the highest benefits produces more aggregate improvement with fewer treatments.

If treatment costs were included, we could define net benefit as expected outcome improvement minus a cost expressed in comparable utility units. The optimal policy would then prioritize positive net effects. If costs differed by person, both benefit and cost would matter for allocation. A linear capacity fraction alone would no longer capture the resource constraint.

Similarly, risk reduction measured in statistical outcome units may not coincide with social value. Some outcomes affect people differently; some policies have distributional goals; and some harms may be unacceptable even if an average gain is positive. A formally optimal policy is optimal only with respect to the objective and constraints supplied to it.

The purpose of causal inference is not to replace those values with a coefficient. It is to describe intervention consequences credibly enough that decisions can acknowledge the trade-offs rather than conceal them.

## Why the experiment is deliberately simple

The model uses a single standard normal covariate, randomized treatment and a linear effect modifier. That is enough to construct a dramatic difference between baseline-risk and treatment-benefit rankings. It also means that the estimated-benefit policy can recover the oracle ranking relatively easily.

This is a favourable demonstration of policy learning, not a benchmark establishing the performance of arbitrary causal machine-learning methods. In a realistic setting, the effect modifier may be only partially measured, the effect function may contain interactions and nonlinearities, and some subgroups may have little experimental or observational support. Policy learning can then overfit estimated effects and select people whose predicted benefits are largely noise.

The intervention itself may also be difficult to standardize. Different versions of treatment may work through different mechanisms and produce heterogeneous effects that are not captured by a single A indicator. Uptake and adherence can depend on prognosis and preferences, changing whether an assignment policy estimates the effect of offering treatment or of receiving it.

The held-out target population in this simulation is drawn from the same distribution as the training population. That removes the transportability difficulty examined in Part 19. A rule built in one population may behave differently where the distribution of effect modifiers, treatment costs or implementation conditions changes.

Likewise, there are no missing outcomes, time-varying treatments, interference or selection into the trial. Earlier articles showed how those complications affect identification. The current article holds them fixed to isolate the decision problem after a treatment-effect model has been identified.

These simplifications should be stated explicitly because the oracle-level result might otherwise encourage an exaggerated claim about the general ease of treatment allocation.

## What the experimental results do not establish

The simulation does not show that real employment interventions harm high-risk workers. The relationship between risk and benefit was deliberately chosen to make a counterexample. Other plausible mechanisms would make risk and benefit positively correlated or unrelated. Empirical work is needed to determine which mechanism is credible for any particular programme.

It also does not show that a linear CATE model is generally sufficient. The correct linear function is part of the generating model and the training assignment is randomized. In high-dimensional observational data, flexible estimators can introduce substantial nuisance-model errors, as Part 13 illustrated. Even when cross-fitting and orthogonal scores reduce sensitivity to those errors, identifying treatment effects remains a separate requirement.

The policy evaluation uses known simulated individual benefits, which are unavailable in a real dataset. Estimates of realized deployment value would need randomized evaluation, valid off-policy estimation or other justified causal designs. Reporting the exact oracle gap is possible here only because the counterfactual function is constructed rather than inferred.

The capacity constraint is a simple number of treatments. It excludes differing costs, fixed programme expenses, risk aversion, fairness constraints and spillovers between participants. Adding those components can change which policy is optimal even if the estimated treatment effects remain unchanged.

Finally, the Monte Carlo experiment uses 500 repetitions of one known data-generating process, not multiple real populations or policies. Repetition quantifies estimator and decision variability conditional on that process. It does not establish that the model's structural assumptions apply to the world beyond the experiment.

## What the entire series established

Part 1 began with a regression that confidently estimated the wrong causal quantity. Parts 2 to 5 examined how assignment, observed confounders, propensity scores and inappropriate adjustment can change identification. Parts 6 to 9 considered quasi-experimental designs and showed how apparently favourable diagnostics can coexist with failed assumptions. Parts 10 to 15 investigated estimators, model flexibility, Bayesian uncertainty and longitudinal treatment feedback. Parts 16 to 19 examined hidden confounding, falsification, missing outcomes and the limits of transportability.

Across those studies, one principle recurred: a statistical method cannot make an unobserved counterfactual identifiable merely by calculating a precise estimate. Identification is a property of the question, assumptions and design. Estimation concerns what can be learned about the identified quantity from finite data. Evaluation concerns whether those estimates behave as intended. Decision-making adds objectives, constraints and consequences.

The final simulation demonstrates why that last stage should not be omitted. A randomized experiment can estimate a positive average treatment effect and a risk model can predict untreated outcomes perfectly, yet a decision that selects by risk can still produce negative expected benefit. The same evidence supports a better policy only when the treatment-benefit estimand is explicitly connected to the allocation decision.

This does not reduce causal inference to choosing the largest estimated effect. Scientific evidence remains incomplete, intervention costs and distributional considerations matter, and a policy can change the conditions under which it was originally evaluated. A defensible decision records those uncertainties and trade-offs rather than hiding them behind a single number.

## Conclusion

With a true average treatment benefit of 0.5 and treatment capacity for 20% of the population, risk-based targeting selected people whose treatment caused an average harm of approximately 1.60 units. Random allocation delivered an average benefit of 0.50 per treated person. A simple causal treatment-effect model fitted on independent randomized trial data selected the same people as the oracle and delivered approximately 2.60 benefit units per treated person.

These findings do not rank prediction and causal modelling by intrinsic sophistication. The risk model was perfectly aligned with its prediction target. The causal model was aligned with the intervention decision. The discrepancy arose because the two targets were different.

The series began by asking whether the coefficient produced by a statistical model could be interpreted as a causal effect. It ends by asking whether a correctly estimated causal effect is sufficient to justify a decision. The answer to both questions depends on defining what is being estimated and why. Evidence becomes useful when its assumptions, uncertainty and target are connected explicitly to the action being considered.

## Code and reproducibility

Every number in the primary results table is printed by `experiment.py` in `causal_inference_statistics/part20/`. The script uses Python 3.12 or later and NumPy, generates 500 independent randomized training trials of 1,200 individuals, and evaluates four policies on 3,000-person held-out populations. It writes `results.csv` with mean benefits, across-replication standard deviations, regret and population-level gains. The code fixes its seed to 20261018 and does not download external data. The manuscript uses these actual generated results, not illustrative values selected after writing the conclusions.

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
10. Doubly Robust Estimation: When One Correct Model Is Enough—and When Neither Is
11. The G-Formula and Longitudinal Causal Effects
12. Heterogeneous Treatment Effects
13. Causal Machine Learning and Double Machine Learning
14. Bayesian Causal Inference
15. Time-Varying Treatments and Marginal Structural Models
16. Sensitivity Analysis for Unmeasured Confounding
17. Negative Controls, Placebo Tests and Falsification
18. Missing Outcomes and Selection Bias
19. External Validity and Transportability
20. From Statistical Evidence to Causal Decisions (this article)

## References

Athey, S. and Wager, S. (2021). Policy Learning With Observational Data. *Econometrica*, 89(1), 133–161. https://doi.org/10.3982/ECTA15732

Athey, S. and Imbens, G. W. (2016). Recursive Partitioning for Heterogeneous Causal Effects. *Proceedings of the National Academy of Sciences*, 113(27), 7353–7360. https://doi.org/10.1073/pnas.1510489113

Bansak, K., Hainmueller, J., Hopkins, D. J. and Yamamoto, T. (2018). The Number of Choice Tasks and Survey Satisficing in Conjoint Experiments. *Political Analysis*, 26(1), 112–119. https://doi.org/10.1017/pan.2017.40

Chernozhukov, V., Demirer, M., Duflo, E. and Fernández-Val, I. (2025). Fisher–Schultz Lecture: Generic Machine Learning Inference on Heterogeneous Treatment Effects in Randomized Experiments, With an Application to Immunization in India. *Econometrica*, 93(4), 1121–1164. https://doi.org/10.3982/ECTA19303

Dudík, M., Langford, J. and Li, L. (2011). Doubly Robust Policy Evaluation and Learning. *Proceedings of the 28th International Conference on Machine Learning*, 1097–1104. https://icml.cc/2011/papers/554_icmlpaper.pdf

Hernán, M. A. and Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC. https://miguelhernan.org/whatifbook

Imbens, G. W. and Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press. https://doi.org/10.1017/CBO9781139025751

Kitagawa, T. and Tetenov, A. (2018). Who Should Be Treated? Empirical Welfare Maximization Methods for Treatment Choice. *Econometrica*, 86(2), 591–616. https://doi.org/10.3982/ECTA13288

Manski, C. F. (2004). Statistical Treatment Rules for Heterogeneous Populations. *Econometrica*, 72(4), 1221–1246. https://doi.org/10.1111/j.1468-0262.2004.00530.x

Manski, C. F. (2007). *Identification for Prediction and Decision*. Harvard University Press.

Murphy, S. A. (2003). Optimal Dynamic Treatment Regimes. *Journal of the Royal Statistical Society: Series B*, 65(2), 331–355. https://doi.org/10.1111/1467-9868.00389

Nie, X. and Wager, S. (2021). Quasi-Oracle Estimation of Heterogeneous Treatment Effects. *Biometrika*, 108(2), 299–319. https://doi.org/10.1093/biomet/asaa076

Robins, J. M. (2004). Optimal Structural Nested Models for Optimal Sequential Decisions. In D. Y. Lin and P. J. Heagerty (eds.), *Proceedings of the Second Seattle Symposium in Biostatistics: Analysis of Correlated Data*, pp. 189–326. Springer. https://doi.org/10.1007/978-1-4419-9076-1_11

Rosenbaum, P. R. and Rubin, D. B. (1983). The Central Role of the Propensity Score in Observational Studies for Causal Effects. *Biometrika*, 70(1), 41–55. https://doi.org/10.1093/biomet/70.1.41

Wager, S. and Athey, S. (2018). Estimation and Inference of Heterogeneous Treatment Effects Using Random Forests. *Journal of the American Statistical Association*, 113(523), 1228–1242. https://doi.org/10.1080/01621459.2017.1319839

Zhao, Y., Zeng, D., Rush, A. J. and Kosorok, M. R. (2012). Estimating Individualized Treatment Rules Using Outcome Weighted Learning. *Journal of the American Statistical Association*, 107(499), 1106–1118. https://doi.org/10.1080/01621459.2012.695674

## Scientific audit note: stress-testing the allocation rule

The original experiment deliberately uses a monotonic linear treatment benefit and randomized training assignments. That structure lets a correctly signed linear effect model rank individuals exactly as an oracle, even when its predicted benefit levels differ. The near-zero policy regret therefore establishes behaviour for this favourable design, not general optimality of estimated benefit targeting.

A separate reproducibility script, `validate_policy.py`, extends the experiment to a nonlinear treatment benefit, `0.5 + 1.5 X - 0.8(X² - 1)`, comparing linear and quadratic CATE rankings with risk targeting, random assignment and the oracle under a 20% capacity constraint. It includes an otherwise similar target-population shift and a smaller, noisier randomized training sample. The polynomial baseline-outcome features are held fixed while the treatment-effect specification varies, so the comparison isolates effect-model misspecification.

The stress-test has been committed as a planned validation, but its numerical results have **not yet been reproduced in the execution environment** and are deliberately not reported here. Do not interpret the existing near-oracle outcome as evidence about these harder regimes until `validation_results.csv` has been generated, checked, and incorporated. No original findings or tables have been replaced.
