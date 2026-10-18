# Heterogeneous Treatment Effects: The Best Predictor Is Not Always the Best Treatment Rule

*Part 12 of Causal Inference in Statistics. A controlled study of average effects, conditional treatment effects, treatment-effect prediction, and the value of allocating a limited intervention. Four hundred independent train–test replications, three treatment-effect models and a known counterfactual for every simulated individual. Python code provided.*

## The decision

A programme can have a positive average treatment effect and still harm some of the people who receive it. The average says what would happen if the intervention were applied across the target population under the stipulated assumptions. It does not say whether a particular person benefits, whether an intervention should be targeted, or how much of the available benefit can be obtained under a limited budget. Those are related but different questions, and the statistical estimators that answer them need not be evaluated with the same criterion.

Consider a public employment service able to offer intensive support to one in five eligible jobseekers. A risk prediction model identifies those most likely to remain unemployed without additional support. That model may be accurate about unemployment risk and still choose the wrong people for an intervention. The decision is about how much support changes an individual's outcome, not simply what their outcome would be without support. Risk and benefit need not rank people in the same order.

The distinction also applies to medicine, education and customer retention. A patient with the worst expected outcome does not necessarily gain most from a particular treatment; a customer with the highest predicted churn risk need not be the one whose churn is most preventable. Targeting risk can be appropriate when the objective is to find those with the greatest unmet need, but it is not automatically the optimal strategy when the objective is to maximize the causal improvement from a fixed number of interventions.

The experiment here is constructed to make that difference measurable. Treatment is randomized in the training data, so the causal effect is identifiable without adjustment for treatment assignment. The outcome depends on a continuous characteristic whose value modifies the response to treatment. The effect is deliberately nonlinear. Three models are trained to estimate different degrees of heterogeneity: a constant effect, a linear treatment interaction, and a quadratic interaction. Their predictions are evaluated on new individuals that were not used for model fitting.

There are 400 independent replications, each with 2,400 training observations and 1,200 held-out observations. The known population average effect is 0.5. Yet selecting the highest fifth of the population by estimated conditional treatment effect yields a mean true effect of about 2.644 among those selected. Selecting the fifth with the highest baseline prognosis score instead yields a mean effect of about −0.708. Those are synthetic results for a particular outcome model, not empirical estimates of an actual employment programme. Their purpose is to show how the decision changes when the target shifts from outcome risk to treatment benefit.

## What an average treatment effect leaves out

Let A denote a binary treatment and Y(1), Y(0) the potential outcomes with and without treatment. The individual treatment effect is

\[
\tau_i=Y_i(1)-Y_i(0).
\]

Only one of these potential outcomes can ordinarily be observed for each person. A randomized experiment can identify a population mean without revealing both individual outcomes. The average treatment effect is

\[
ATE=E[Y(1)-Y(0)].
\]

This is the expected result of applying treatment rather than control to a randomly selected member of the target population, with the usual assumptions about treatment definition and observation. If the effect differs among people, the average can conceal important variation. A positive ATE does not imply a positive effect for everyone; similarly, a zero average does not establish that the intervention has no meaningful effects.

A conditional average treatment effect, or CATE, uses a set of baseline characteristics X:

\[
\tau(x)=E[Y(1)-Y(0)\mid X=x].
\]

The CATE describes an average within a specified covariate profile. It is not necessarily the individual treatment effect. People with the same observed X may have different responses because of unmeasured characteristics, chance, treatment implementation, or other variables not used to define the subgroup. The distinction between conditional and individual effects is important when discussing so-called individualized treatment rules.

In our synthetic example, the effect really is a deterministic function of the single observed X. This deliberately favourable construction means that an appropriately specified model can recover the effect for an individual, up to sampling uncertainty in its fitted coefficients. A real treatment-effect model generally estimates conditional expectations, not an individual's unobservable pair of counterfactual outcomes.

A policy question adds another layer. If a decision rule d(x) takes values zero or one, representing whom we treat, its value can be written as

\[
V(d)=E[Y(0)+d(X)\{Y(1)-Y(0)\}].
\]

Relative to treating nobody, the policy improvement is E[d(X)τ(X)]. If only a fixed proportion q may be treated, the constraint is E[d(X)]≤q. For a beneficial intervention with a known conditional effect and no other fairness or operational constraints, the optimal rule allocates treatment to those with the highest positive conditional effects, up to capacity. If some of the largest effects are still negative, an unconstrained "at most q" rule may leave capacity unused rather than treat people who are harmed.

That distinction matters here. I evaluate a fixed capacity exactly equal to 20%, so every competing rule selects the same number of people. This isolates the ranking question from the separate decision of whether some individuals should be left untreated because their effect is negative.

## A model with known heterogeneity

Generate an independent standard normal baseline characteristic X and a binary treatment A assigned with probability one half, independently of X and outcome noise. The untreated prognosis is

\[
\mu_0(X)=2-1.5X+0.3X^2.
\]

The treatment effect varies with X according to

\[
\tau(X)=0.5+1.2X+0.4(X^2-1).
\]

Finally, generate the observed outcome as

\[
Y=\mu_0(X)+A\tau(X)+\varepsilon,
\qquad \varepsilon\sim N(0,1),
\]

with ε independent of X and A. Because E[X]=0 and E[X²]=1, the population ATE is exactly 0.5. The quadratic term is centred so that it introduces heterogeneity without changing the population mean.

The linear component 1.2X produces a large gradient in treatment effects, while the quadratic component changes the curvature of that relationship. Since the untreated prognosis falls as X increases over an important region, higher baseline outcome risk is not aligned with higher intervention benefit. In fact, selecting on the largest values of μ₀(X) tends to prioritize people whose treatment effect is negative in this model.

The effect distribution includes people who would be harmed by treatment as well as those who would benefit. For X below zero, the mean true effect in the simulated holdout samples is approximately −0.458. For X at or above zero, it is approximately +1.454. These are not two intrinsic biological classes; the split at zero is a convenient descriptive summary of a continuous relationship. The true effect varies within each half.

The treatment assignment mechanism is intentionally simple: each person's treatment is an independent Bernoulli draw with probability 0.5. There is no confounding of A with potential outcomes and no practical overlap problem. This design isolates heterogeneity estimation and policy choice from identification difficulties discussed in earlier parts of the series.

The observed training sample still does not reveal both potential outcomes for anyone. The model must infer how the difference between potential outcomes changes with X from randomized comparisons across different individuals. Knowing the data-generating equations as the experimenter allows the evaluation to use the actual simulated conditional effects in held-out individuals, but an empirical analyst would not have that information.

## Three models of the treatment effect

Each training dataset is analysed using ordinary least squares with an intercept, X and the centred quadratic baseline term X²−1. These three terms describe the prognostic outcome pattern common to treatment and control. The models differ only in how much treatment-effect variation they permit.

The constant-effect model adds A as a single treatment term. It estimates one coefficient representing the average effect across the training population, conditional on the specified baseline regression. Its treatment-effect prediction is the same for everybody. It can estimate the overall average accurately while being unable to rank individuals by benefit.

The linear interaction model adds A and A×X. Its predicted conditional treatment effect has the form

\[
\widehat\tau_1(x)=\widehat\beta_A+\widehat\beta_{AX}x.
\]

This can reproduce the main directional gradient in benefits but not the quadratic curvature. It is misspecified for the true treatment-effect function, though it may still rank the highest-benefit individuals sufficiently well for the particular capacity decision.

The quadratic interaction model adds A, A×X and A×(X²−1), giving

\[
\widehat\tau_2(x)=\widehat\beta_A+\widehat\beta_{AX}x+
\widehat\beta_{AX^2}(x^2-1).
\]

This model contains the correct treatment-effect functional form. With randomized treatment, correctly specified baseline terms and independent errors, its estimates should converge to the structural coefficients as sample size grows.

Using simple regressions is deliberate. The important contrast is between a model appropriate for estimating an average, one that approximates the heterogeneity, and one that specifies it correctly. A causal forest or neural-network treatment-effect estimator would introduce additional tuning, regularization and model-comparison questions before the estimand and decision have been clarified. The next article can investigate flexible estimators against these transparent baselines.

All three models share the same baseline prognostic specification, so the differences are not driven by unequal access to information about untreated outcome risk. The richer models use interactions to represent response heterogeneity, not additional raw predictor variables.

## Independent train and test data

Every replication generates 2,400 independent training individuals and 1,200 new test individuals. Treatment outcomes in the training set are used to fit the models. The test set is generated independently and is never used to fit coefficients or choose the functional forms.

The distinction matters because flexible treatment-effect models can look impressive on the data used to estimate them. A method may fit idiosyncratic outcome noise and produce apparent subgroup differences that vanish in a new population. Evaluating on fresh observations is necessary to distinguish genuine predictive heterogeneity from chance fit.

In this controlled experiment, the effect function is known for every held-out person. The test therefore evaluates mean squared error of the predicted CATE against τ(X), not against the individual's noisy observed outcome. In an ordinary observational or randomized dataset this oracle criterion is not directly available; only one treatment outcome is observed for each person. The ability to report true CATE error is a feature of simulation, not a general evaluation procedure.

I also construct treatment allocation rules using each model's predicted effects. All rules choose exactly the top 20% of held-out individuals according to their ranking scores. The resulting score is the *mean true treatment effect among selected individuals*. Larger values imply more total causal improvement for the same number treated, under the stipulated outcome scale and no intervention cost differences.

For the constant-effect model, all individuals receive the same predicted CATE. Ties are broken using random scores so that its allocation is equivalent to selecting an arbitrary fifth rather than giving the algorithm an artificial ability to exploit ordering in the test data.

Two additional rules serve as baselines: random allocation, and ranking by the true untreated prognosis μ₀(X). Finally, an oracle ranks individuals by the true τ(X), providing a benchmark that no data-based method can be expected to surpass systematically under this design. The oracle is not an implementable policy, because it uses counterfactual information unavailable in practice.

## How well are treatment effects estimated?

Across 400 independent replications, the mean test-set CATE mean squared errors are:

| Treatment-effect model | Test CATE MSE | SD across replications | Mean prediction bias |
| --- | ---: | ---: | ---: |
| Constant effect | 1.75678 | 0.11868 | +0.00001 |
| Linear interaction | 0.32471 | 0.03554 | −0.00282 |
| Quadratic interaction | 0.00482 | 0.00382 | −0.00143 |

The constant-effect model has almost no average prediction bias when averaged over all test individuals. This is consistent with estimating the population mean effect of 0.5. But its mean squared error is high because it cannot represent variation around that mean. A model can be well calibrated for an overall average and still be poor at describing how effects vary with covariates.

The linear interaction dramatically reduces CATE error by learning that benefits rise with X over much of the population. Yet it still misses curvature and therefore leaves substantial systematic variation in the effect prediction. Its MSE is about 0.325, much lower than the constant model's 1.757 but far above the correctly specified quadratic model's 0.00482.

The quadratic interaction model nearly recovers the known treatment-effect surface. Its remaining error reflects finite-sample variation from estimating coefficients using noisy outcomes. In this experiment, all terms required by the structural model are represented in the fit, the treatment is randomized, and the training sample is large enough for the estimated interaction coefficients to be stable.

Those results confirm that the three models have substantially different abilities to estimate CATEs. They do not yet tell us which model produces the best policy. A ranking decision may depend on a limited part of the effect distribution, and a model can be inaccurate in parts of the distribution irrelevant to that decision while still selecting almost all the right individuals.

## Evaluating the allocation decision

The mean true effect among the fifth selected for treatment is:

| Allocation rule | Mean true benefit among selected | SD across replications |
| --- | ---: | ---: |
| Highest baseline outcome risk | −0.70770 | 0.01078 |
| Random allocation | 0.49872 | 0.08218 |
| Constant-effect regression | 0.49655 | 0.07947 |
| Linear CATE model | 2.64394 | 0.09884 |
| Quadratic CATE model | 2.64399 | 0.09880 |
| Oracle treatment-effect ranking | 2.64403 | 0.09881 |

Random allocation gives a mean effect close to the population ATE of 0.5, as expected. The constant-effect model cannot rank people by benefit and produces almost the same result. Its success at estimating the population average is not a reason to prefer its treatment allocation over random assignment under a binding capacity constraint.

Selecting by baseline outcome risk gives a negative mean true effect, about −0.708. This rule gives priority to people with high untreated outcomes under the specified score. Because the causal effect is negatively aligned with that risk ranking in the selected region, the intervention on average makes those selected worse off on the experiment's outcome scale. The example deliberately separates need, risk and treatment responsiveness. It should not be generalized into a claim that serving high-risk individuals is usually harmful.

The linear and quadratic CATE models both select subsets with a mean true benefit of approximately 2.644, essentially identical to the oracle at the reported precision. Their rankings agree closely near the top fifth of the true effect distribution, even though their mean squared errors across all individuals are very different.

This is the central empirical result. Improving CATE prediction error by two orders of magnitude—from about 0.325 to 0.00482—does not meaningfully improve this particular capacity-constrained decision. The simpler linear interaction already identifies almost exactly the same high-benefit fifth of the population as the correctly specified quadratic model.

A different capacity level, outcome function or benefit distribution could reverse the comparison. Here, the linear model's ranking happens to preserve the relevant order near the selection threshold. The result is an example of decision-specific adequacy, not a proof that linear treatment-effect models are generally sufficient.

## Why outcome risk is a poor targeting rule here

The untreated outcome function is μ₀(x)=2−1.5x+0.3x². The CATE is τ(x)=0.5+1.2x+0.4(x²−1). These are different functions of the same baseline characteristic. The fact that they both depend on X does not make their rankings equivalent.

A prediction model of untreated outcome risk might be excellent: in the simulation its ranking is based on the true μ₀ function rather than on an estimated approximation. But the policy is evaluated on incremental outcome change from treatment. A person expected to do badly without treatment can have a small, zero or negative effect from this particular intervention. Another person whose untreated prognosis is better can gain considerably more.

The negative result for risk targeting therefore cannot be dismissed as poor risk prediction. Risk was supplied by the known structural function. The mistake is using risk for a decision whose objective is treatment benefit. More accurate prediction of the wrong quantity would not correct it.

This is closely related to the final article of the survival-analysis series. There, a ranking method that predicts who is likely to remain unemployed is useful for deciding whom a service wants to identify as being at high risk. It does not, by itself, show who gains most from job-search assistance. Establishing incremental benefits requires outcome information under different treatment assignments, ideally provided by a credible experimental or quasi-experimental design.

There are legitimate allocation objectives other than maximizing total average treatment benefit. A service may prioritize people with the greatest unmet need even when expected gains are smaller, or protect disadvantaged groups, minimize worst-case outcomes, or satisfy statutory obligations. The appropriate policy value must reflect those goals. The current simulation studies one explicitly stated utilitarian objective under equal treatment costs and fixed capacity; it does not determine which objective a real institution ought to choose.

## What the decision metric measures

With capacity q=0.2, every model selects the 240 highest-ranked individuals from the 1,200-person holdout. For each selected person, the experiment knows τ(X). The reported value is the mean of those effects over the selected subset. Since all policies treat the same number, comparing these means is equivalent to comparing the total causal gain across the selected people, up to the constant number treated.

The policy contrast can be expressed as

\[
\Delta V(d)=E[d(X)\tau(X)].
\]

For a policy that treats exactly 20%, this is 0.2 times the mean treatment effect among selected people in the target population. Thus the reported selected-group benefit of about 2.644 corresponds to an improvement of approximately 0.529 per member of the entire eligible population, relative to treating none, under the simulated model. Random selection corresponds to approximately 0.100 per eligible person.

That difference is not necessarily the net welfare gain of the intervention. The calculation excludes programme costs, spillovers, implementation constraints, treatment adherence, heterogeneous resource requirements and any equity weights. It measures one narrowly defined outcome improvement.

Reporting mean benefit among selected individuals is useful because it matches the limited-capacity decision. But it is not a universally sufficient performance metric. A policy that changes the proportion treated needs its value evaluated at the new capacity. A policy that gives different treatment doses needs a different intervention model. A policy whose costs vary with X should consider benefit net of those costs rather than raw effect alone.

The capacity constraint also makes the distinction between ranking and estimating effect magnitude important. If a service must select exactly a fifth of people, the ordering of the predicted benefits may matter more than perfect calibration of their numerical size. If it must decide whether a programme is worth funding at all, accurate effect magnitudes and uncertainty become central again.

## Why the linear model can match the oracle

The correctly specified effect is quadratic, so the linear interaction model is globally misspecified. Its poorer MSE is real. It over- or underestimates effects for some individuals because it cannot represent the curvature of τ(x). Yet, across the simulated X distribution and the chosen upper-fifth capacity, the true highest-benefit observations are also assigned high scores by the linear model.

The consequence is that most of the model's errors occur in predicting exact benefit magnitudes or in ordering individuals who would not cross the allocation threshold. Those errors matter for CATE prediction MSE but have little effect on which people receive treatment.

This distinction mirrors other decision settings. A probability model can be miscalibrated yet rank individuals well for a fixed-capacity screening decision. Conversely, a model with low average error can fail near the decision boundary and produce a poor allocation. The criterion must be tied to what happens when predictions determine actions.

The oracle benchmark helps reveal this directly. The quadratic model's selected-group effect is about 2.64399, the linear model's 2.64394 and the oracle's 2.64403. These differences are extremely small relative to the between-replication standard deviation of approximately 0.099. The simulation does not establish meaningful superiority of one model's policy over the other at this capacity.

A responsible interpretation is therefore not that the two prediction models are equivalent. They plainly differ in global CATE MSE. It is that the more accurate model does not yield a practically distinguishable targeting advantage for the specified decision in this data-generating process. Changing the decision could expose their differences.

## Why a zero average effect would not settle the question

The ATE in this experiment is positive but modest relative to the variation in τ(X). A different intercept in the treatment-effect function could make the population ATE zero without eliminating the differences among people. Positive and negative individual effects might cancel in the population average.

A study focused only on testing the null hypothesis of zero average effect might then report little evidence for a population-wide intervention. That could be a correct conclusion for universal treatment while providing incomplete evidence for targeted treatment, provided heterogeneity can be estimated credibly and a targeting rule evaluated without bias from model selection.

This does not justify searching through many subgroups until a statistically significant treatment effect appears. Data-driven subgroup discovery can generate impressive apparent differences from random noise, especially when the same outcomes are used both to find and to test the subgroup. Valid inference requires attention to multiplicity, sample splitting, honest estimation or other procedures appropriate to the adaptive analysis.

The experiment's heterogeneous effects are genuine because they are defined in the generating equations. The train/test split prevents the treatment-effect models from being evaluated on the data used to fit them. But the structural truth remains available only because this is a simulation. In a real randomized trial, one observes treatment outcomes under assigned conditions and must evaluate treatment-effect models without assuming the individual counterfactuals are known.

The difference between proving that an average effect exists and constructing a reliable allocation rule becomes important precisely when the programme must be deployed rather than merely described.

## Conditional means are not individual counterfactuals

The term individualized treatment effect is often used casually for a model's predicted CATE. A CATE model estimates E[Y(1)−Y(0)|X=x], an average over individuals sharing the conditioning characteristics. It does not generally reveal Y_i(1)−Y_i(0) for a particular person.

The distinction matters for uncertainty. We may estimate the mean treatment response among people with a particular age, baseline risk and observed history while knowing little about differences among those people not captured by the model. A narrow confidence interval for a subgroup average is not the same as certainty about every individual's response.

The current simulation makes the effect deterministic in X, which collapses that distinction within its synthetic population. This simplification is useful for scoring predictions but should not be carried into empirical interpretation. In more realistic settings one could add latent effect modifiers that remain unobserved. A correctly specified model in X would then recover the CATE but not the individual effect.

It is also possible to estimate subgroup average effects much more reliably than highly granular effects. As conditioning sets become more specific, fewer observations are available in each region and nuisance-model uncertainty grows. Flexible methods can borrow strength across covariate profiles, but the assumptions determining how they share information matter.

In applied reports, it is usually more defensible to describe predicted conditional effects, subgroup benefits and policy value than to assert that the treatment's effect on an individual person has been identified.

## Randomization identifies averages, not every effect

Treatment is randomized in this experiment, and both treatment arms are observed across the range of X. This supports identifying conditional means E[Y|A=1,X=x] and E[Y|A=0,X=x] under the usual consistency and exchangeability conditions. Their difference identifies τ(x).

Random assignment does not create both potential outcomes for the same person. It allows comparisons across exchangeable groups and, with sufficient data and modelling assumptions, across strata of X. The quality of CATE estimation still depends on sample size, covariate support, functional form and the estimation procedure.

The existence of randomized treatment therefore does not imply that any fitted heterogeneity is genuine. One could fit dozens of interaction terms and find apparently large subgroup differences by chance. The stronger the model's flexibility relative to the data, the more important honest evaluation becomes.

Our three models are intentionally modest. They estimate two nuisance components: an outcome prognosis and a treatment-effect function. Both are represented by low-dimensional polynomials, and the train/test split is repeated independently. This makes their performance interpretable without introducing tuning decisions that might obscure the principal comparison.

Observational CATE estimation has additional identification demands. When treatment assignment depends on X, conditional exchangeability and overlap must be justified before potential-outcome contrasts can be estimated. Double robustness and modern machine learning can help with estimation under appropriate assumptions, but cannot substitute for a defensible adjustment set.

## A decision-focused evaluation is not a new causal identification design

The policy results in this article use the true τ(X) for held-out observations. That makes the decision score transparent and unambiguous. In an actual programme, the individual effect is not observed, so evaluating a treatment allocation rule requires another source of evidence.

One approach is a randomized test of competing policies. Assign eligible people to a rule or comparator and estimate the difference in average outcomes under the corresponding allocation regimes, accounting for the implementation and assignment design. Another is off-policy evaluation from randomized or observational treatment data with known or estimated treatment probabilities, under the required exchangeability and overlap assumptions.

Inverse probability weighting and augmented estimators can estimate the value of policies from suitable logged decisions, but can become unstable when the proposed rule frequently selects treatment options rarely observed in the data. The positivity and nuisance-model concerns examined earlier in this series remain relevant.

It would be a methodological error to score a CATE model in real data by simply comparing observed outcomes of the people it predicts would benefit most. Those individuals may differ in baseline prognosis, treatment receipt and other causes of outcome. An observed group difference is not automatically a causal measure of the allocation rule's value.

The oracle outcome used here is a simulation-only evaluation privilege. We can use it to determine which modelling choices matter under a known mechanism. A practical study must replace the oracle calculation with a justified experimental or off-policy estimator and report its uncertainty.

## Model complexity and targeting

The prediction results favour the quadratic treatment-effect model by a very large margin on global CATE error. The decision results do not. That disagreement suggests a concrete principle for model selection: optimize an estimator for the question the system will use it to answer, while retaining diagnostics that reveal whether the output is sufficiently accurate for other purposes.

If the primary product is a report of average treatment effects, the constant model may be adequate under valid randomization and correct baseline treatment specification. If the product is a map of how benefits vary across X, the constant model is inadequate and the quadratic model is far better in the present simulation. If the product is a fixed top-fifth allocation, the linear model may already have nearly all the value the more complex model can provide.

The relevant complexity penalty is not only computational. A more flexible model needs sufficient data, careful uncertainty estimation, and monitoring when deployed in a changing population. Its larger apparent accuracy may not improve the decision and could be more sensitive to distribution changes. A simpler rule can be preferable when it achieves essentially the same value and is easier to explain or maintain.

That is not an argument for linear models by default. It is an argument for empirical comparison with a simple baseline, a clearly specified decision metric and an independent evaluation sample. The result should determine whether additional model complexity pays for itself.

The same argument applies to using a complicated estimator without comparing it against straightforward effect modification terms. If the main heterogeneity follows a transparent relationship with a few baseline covariates, a simple interaction regression may already give a useful policy rule.

## Treatment-effect uncertainty matters for decisions

Point predictions alone are not enough when the cost of a wrong treatment allocation is substantial. A model may rank people differently because its estimated interaction coefficients are uncertain. If two individuals have nearly equal predicted effects, selecting one rather than the other may not be supported strongly by the data.

In the current simulation, the capacity threshold falls in a region where the linear and quadratic rankings almost agree. This helps explain their nearly identical selected-group effects. A different intervention budget could place the threshold where the models disagree more, especially in the tails of X where quadratic curvature becomes influential.

One might evaluate policy regret relative to an oracle, confidence intervals for policy value, or the probability that the estimated rule beats a simpler alternative under repeated sampling. Those quantities address the practical consequences of model uncertainty more directly than a single global prediction metric.

A rigorous empirical evaluation would also account for the adaptive nature of choosing the rule. If the same data are used to select the best policy among many candidates and then to evaluate that selected policy, performance estimates can be optimistic. Honest holdout evaluation or suitable cross-fitting can reduce that optimism, though final prospective validation remains valuable.

Here, the models are fixed in advance and evaluated on independent holdout individuals in every replication. The oracle information is used only to score policies. The experiment therefore isolates statistical ranking performance rather than measuring a fully operational deployment process.

## Outcome scales and the meaning of benefit

A positive treatment effect is beneficial only if increasing the outcome is desirable. In this synthetic study it is, by definition. In an unemployment-duration application, a reduction in weeks unemployed would normally be desirable, so benefit might be defined as Y(0)−Y(1), reversing the sign relative to the usual outcome difference.

This matters because targeting algorithms mechanically rank according to their score. A sign error can turn a sensible estimated response function into a rule that prioritizes those predicted to be harmed. The estimand, outcome units and direction of benefit should be written down before selecting the top-ranked people.

Some interventions have multiple outcomes: employment, income, health, programme cost and distributional fairness may all matter. Optimizing one effect can worsen another. An explicit utility or social welfare function is needed to combine them; it cannot be derived from the CATE of one outcome alone.

The experiment deliberately keeps one outcome and equal treatment costs. This avoids turning the mathematical identification question into a contested policy decision. It should also make the limitation clear: maximizing average incremental outcome among the selected group is one objective, not a general theory of public resource allocation.

## Transporting a targeting rule

Even a rule that nearly matches the oracle in one distribution may not remain optimal when deployed elsewhere. The effect function could change with the population, intervention implementation, time or other conditions. The distribution of X also matters because the top-fifth threshold is determined by who is eligible.

The simulated test individuals are drawn from the same distribution as the training individuals. That allows an unbiased assessment of out-of-sample performance under a stable data-generating process. It does not test transport to another population or policy regime.

If the eligible population shifts toward larger X values, a fixed score threshold may treat a different fraction. If the rule is recalibrated to keep capacity at 20%, the selected group can have a different average benefit. If the intervention effect itself changes, even a perfectly estimated historical τ(x) might be misleading.

External validation should therefore examine effect transportability, covariate shift, eligibility changes and changes in treatment delivery. The value of a targeting rule is not a timeless attribute of a fitted model; it is a property of the model, intervention, population and decision constraints together.

This is a natural connection to the later article on external validity. The present simulation holds those elements fixed to isolate the distinction between average effect, conditional effect and allocation value.

## Why the subgroup split is descriptive rather than a decision rule

The simple split X<0 versus X≥0 gives mean effects of approximately −0.458 and +1.454. It shows the breadth of effect heterogeneity. But using that threshold directly as an allocation policy would treat about half the population, not the allowed fifth, and would include individuals whose effects vary substantially within the positive-X group.

A two-group effect summary is useful for communication but not necessarily a sufficient targeting instrument. When resources are scarce, finer ranking may produce a large additional benefit. Conversely, when effects are nearly constant within relevant groups, elaborate fine-grained modelling may add little value.

Subgroup analysis also raises questions about what groups were chosen and why. A pre-specified subgroup based on scientific understanding has a different inferential status from a group selected after searching many possible cut points for the largest apparent effect. Reusing a discovered subgroup as though it had been prespecified can exaggerate evidence for heterogeneity.

This study avoids cut-point optimization. The split at zero is fixed and only used descriptively. The policy models instead use continuous effect predictions evaluated honestly on held-out individuals. The resulting decision illustrates what is gained by ranking rather than merely naming broad subgroups.

## What the Monte Carlo design cannot establish

The simulation uses one normally distributed baseline covariate, independent treatment assignment, Gaussian outcome noise, constant treatment probability one half, a quadratic untreated outcome and a quadratic heterogeneous treatment effect. Every feature is observed without error, and there is no interference, noncompliance or missing outcome data.

Those assumptions are deliberately favourable for identifying heterogeneous effects. The experiment does not investigate what happens when treatment is confounded, overlap deteriorates, covariates are high-dimensional, baseline features are measured after treatment, or the true effect depends on latent variables. Those settings can make both effect estimation and policy evaluation substantially harder.

The quadratic model is correctly specified by construction. This does not establish that quadratic interaction regression generally outperforms forests or other flexible approaches. It simply provides a mathematical benchmark whose treatment effects can be derived and scored. The next article should compare more flexible methods under richer mechanisms where their adaptivity may be useful.

The 400 replications use new independent training and test samples each time, but all are drawn from the same known data-generating process. The between-replication standard deviations quantify variation under that process, not uncertainty about whether the mechanism resembles a real policy setting.

The decision score uses the true effect for every simulated test individual. Such oracle information is unavailable in real-world data. Estimates of policy value would need suitable randomized or observational identification assumptions and their own standard errors. No causal conclusion about an actual programme follows from these synthetic numbers.

The capacity is fixed at exactly 20%, treatment costs are identical, and the decision aims to maximize one outcome. Fairness, geographic capacity, treatment preferences, risk aversion and administrative restrictions are omitted. They may justify a different allocation even if the true CATE were known precisely.

Finally, the linear model's near-oracle performance is specific to the shape of τ(x), the distribution of X and the upper-fifth capacity. Changing any of those elements may make the quadratic model's better CATE predictions crucial to the policy decision. The benchmark is evidence for choosing metrics according to a specified decision, not a general ranking of modelling methods.

## What the investigation established

The population average treatment effect in the simulated system was 0.5. Conditional effects varied greatly, with a mean near −0.458 for negative X and +1.454 for nonnegative X. A constant-effect regression estimated the overall mean but had test CATE MSE about 1.757 because it could not reproduce that variation. A linear interaction reduced the error to about 0.325; the correctly specified quadratic model reduced it to about 0.00482.

For a policy treating one fifth of eligible individuals, random allocation and constant-effect modelling produced a mean benefit among selected people close to 0.5. Ranking by untreated outcome risk instead selected people whose average treatment effect was about −0.708. The linear and quadratic effect models selected groups with mean true benefit about 2.644, effectively matching the oracle ranking under this simulation.

The most accurate effect model was not meaningfully better for the particular decision. Conversely, an accurate prognosis ranking was poor for treatment targeting. Neither result is paradoxical once the target is explicit: one model predicts untreated outcomes, another estimates average effects, another predicts conditional differences, and the allocation rule uses only the ordering of benefits near a capacity threshold.

Estimating heterogeneity is valuable when it changes what a decision-maker can do. Its statistical evaluation should therefore connect identification, conditional effect accuracy, policy value and uncertainty rather than assuming that improvements in one metric must translate into improvements in all the others. A causal effect is not a risk score, and an accurate treatment-effect prediction is not automatically the best policy for every objective.

## Code

All reported results are generated by `causal_inference_statistics/part12/experiment.py`, which requires Python 3.12 or later and NumPy. The script fixes the random seed to 20261018, generates fresh train and test samples for each of 400 replications, fits three ordinary least squares interaction models, scores CATE predictions against known effects and evaluates six fixed-capacity allocation rules. It writes `results.csv` alongside the script. No external data or specialized causal library is required.

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
11. The G-Formula: Why Adjusting for a Time-Varying Confounder Can Miss the Total Effect
12. Heterogeneous Treatment Effects: The Best Predictor Is Not Always the Best Treatment Rule (this article)

## References

Athey, S. and Imbens, G. W. (2016). Recursive Partitioning for Heterogeneous Causal Effects. *Proceedings of the National Academy of Sciences*, 113(27), 7353–7360. https://doi.org/10.1073/pnas.1510489113

Athey, S. and Wager, S. (2021). Policy Learning with Observational Data. *Econometrica*, 89(1), 133–161. https://doi.org/10.3982/ECTA15732

Chernozhukov, V., Demirer, M., Duflo, E. and Fernández-Val, I. (2023). Generic Machine Learning Inference on Heterogeneous Treatment Effects in Randomized Experiments, with an Application to Immunization in India. *Econometrica*, 91(1), 1–31. https://doi.org/10.3982/ECTA19837

Dudík, M., Langford, J. and Li, L. (2011). Doubly Robust Policy Evaluation and Learning. *Proceedings of the 28th International Conference on Machine Learning*, 1097–1104.

Hill, J. L. (2011). Bayesian Nonparametric Modeling for Causal Inference. *Journal of Computational and Graphical Statistics*, 20(1), 217–240. https://doi.org/10.1198/jcgs.2010.08162

Hirano, K. and Porter, J. R. (2009). Asymptotics for Statistical Treatment Rules. *Econometrica*, 77(5), 1683–1701. https://doi.org/10.3982/ECTA6688

Imai, K. and Ratkovic, M. (2013). Estimating Treatment Effect Heterogeneity in Randomized Program Evaluation. *Annals of Applied Statistics*, 7(1), 443–470. https://doi.org/10.1214/12-AOAS593

Kallus, N. (2018). Balanced Policy Evaluation and Learning. *Advances in Neural Information Processing Systems*, 31.

Kent, D. M., Steyerberg, E. and van Klaveren, D. (2018). Personalized Evidence Based Medicine: Predictive Approaches to Heterogeneous Treatment Effects. *BMJ*, 363, k4245. https://doi.org/10.1136/bmj.k4245

Manski, C. F. (2004). Statistical Treatment Rules for Heterogeneous Populations. *Econometrica*, 72(4), 1221–1246. https://doi.org/10.1111/j.1468-0262.2004.00530.x

Murphy, S. A. (2003). Optimal Dynamic Treatment Regimes. *Journal of the Royal Statistical Society: Series B*, 65(2), 331–355. https://doi.org/10.1111/1467-9868.00389

Nie, X. and Wager, S. (2021). Quasi-Oracle Estimation of Heterogeneous Treatment Effects. *Biometrika*, 108(2), 299–319. https://doi.org/10.1093/biomet/asaa076

Rosenbaum, P. R. and Rubin, D. B. (1983). The Central Role of the Propensity Score in Observational Studies for Causal Effects. *Biometrika*, 70(1), 41–55. https://doi.org/10.1093/biomet/70.1.41

Rubin, D. B. (1974). Estimating Causal Effects of Treatments in Randomized and Nonrandomized Studies. *Journal of Educational Psychology*, 66(5), 688–701. https://doi.org/10.1037/h0037350

Wager, S. and Athey, S. (2018). Estimation and Inference of Heterogeneous Treatment Effects Using Random Forests. *Journal of the American Statistical Association*, 113(523), 1228–1242. https://doi.org/10.1080/01621459.2017.1319839

Zhao, Y., Zeng, D., Rush, A. J. and Kosorok, M. R. (2012). Estimating Individualized Treatment Rules Using Outcome Weighted Learning. *Journal of the American Statistical Association*, 107(499), 1106–1118. https://doi.org/10.1080/01621459.2012.695674