# Time-Varying Treatments and Marginal Structural Models: Why Ordinary Adjustment Can Remove the Effect

*Part 15 of Causal Inference in Statistics. A longitudinal experiment with treatment–confounder feedback, two sustained treatment strategies, and a comparison of sequential inverse probability weighting, parametric G-computation, and conventional regression adjustment. Python code provided.*

A treatment can change the variables used to decide whether that treatment continues. This creates a problem for ordinary adjustment that is easy to miss when a longitudinal dataset is treated as an ordinary regression table. A characteristic recorded after an initial intervention may predict the later treatment and the outcome, making it look like a confounder that should be controlled. But it may also carry part of the earlier intervention's causal effect. Conditioning on it changes what the coefficient of the first treatment means.

The problem is particularly important in healthcare, public services, customer interventions and adaptive programmes. A patient receives a medication, their condition changes, and that change influences the subsequent dose. A worker receives employment assistance, gains confidence or improves job-search skills, and that new state influences the decision to provide further support. A customer receives a retention offer, changes usage behaviour, and the changed behaviour influences the next offer. In each example, the information used to make the later decision is partly a consequence of the earlier decision.

This article investigates the problem with a completely specified system. There are two decisions, one intermediate variable, an observed baseline characteristic and an outcome. The total effect of sustained treatment at both time points rather than no treatment at either point is exactly 3.36. I simulate 160 independent datasets of 2,200 individuals and compare three estimators. Sequential inverse probability weighting estimates 3.3540, parametric G-computation estimates 3.3571, and an ordinary regression contrast that controls for the intermediate variable estimates 1.7973.

The difference is not caused by the failure of the ordinary outcome regression to predict the observed data. Its conditional mean model is correctly specified. It answers a different question: the coefficient sum holds the intermediate variable fixed and therefore does not include the entire pathway through which the initial intervention changes the later outcome. The sequential methods are constructed to estimate the two complete treatment strategies.

The investigation follows Part 11, where I introduced the longitudinal G-formula. Here the emphasis shifts to marginal structural models and the weighting logic that recovers an appropriate marginal contrast without conditioning away treatment-mediated pathways. The two approaches use different nuisance models and expose different sources of sampling uncertainty. Their agreement in this experiment is informative because the causal truth is known, not because agreement between estimators automatically proves an observational assumption.

## The programme decision

Suppose a public employment agency can offer intensive support in two consecutive stages. At the beginning, caseworkers consider workers' prior circumstances and assign an initial programme. After an interval, they observe an updated measure of job-search progress and decide whether to continue support. Their second decision depends on the progress measure, and that measure is itself affected by initial support. The agency wants to compare a policy of offering support at both stages with a policy of offering it at neither.

If the agency simply regresses eventual outcomes on the two programme indicators, baseline characteristics and updated progress, the resulting treatment coefficients measure conditional associations holding progress fixed. For the first stage, holding progress fixed may block part of the pathway that the agency wants included in the overall policy effect. Omitting progress is no simple solution either, because progress helps determine the second treatment and predicts the final outcome. The later treatment can be confounded by a variable already influenced by earlier treatment.

That combination is called treatment–confounder feedback. It is not an exotic mathematical exception. It arises naturally whenever a system adapts treatment according to information produced by earlier treatment. It is one of the reasons longitudinal causal inference requires a careful definition of treatment strategies, time order and the variables permitted in each assignment model.

The empirical example is only motivation. The data below are simulated continuous outcomes, not unemployment durations from a historical study. That choice lets the causal effect be known before estimation. Real programmes would add issues of attrition, administrative rules, competing risks, implementation variation, interference and changing populations. Those are important, but they are not needed to expose the adjustment failure.

The policy question is not whether treatment indicators are associated with outcomes at observed values of progress. It is what the average outcome would be if everyone followed the sustained-treatment strategy, compared with what it would be if everyone followed the sustained-no-treatment strategy, allowing intermediate characteristics to evolve as each intervention causes them to evolve.

## Potential outcomes for treatment histories

In a one-time intervention, an average treatment effect can be defined with two potential outcomes, Y(1) and Y(0). With two decisions, the intervention is a history rather than a single indicator. Let A₀ denote the initial treatment and A₁ the later treatment. Potential outcomes are indexed by treatment strategies, so Y(a₀,a₁) represents the outcome that would occur if both assignments were set to the specified values.

The contrast investigated here is

\[
\psi=E\{Y(1,1)-Y(0,0)\}.
\]

This is a marginal effect across the target population. It allows the intermediate state to differ between treatment strategies. It is not a comparison that fixes the same intermediate value in both worlds. That distinction separates the total strategy effect from a conditional direct contrast.

The intermediate variable L₁ also has counterfactual versions L₁(a₀), because its value depends on the earlier treatment. The final outcome under a strategy should therefore be understood as the outcome obtained when A₀ is set, L₁ evolves under that setting, and A₁ is subsequently set. The outcome responds to both treatment decisions and the intermediate state resulting from the first.

The strategies here are static: they assign the same treatment pattern regardless of an individual's measured history. Many practical policies are dynamic. For example, a clinic might continue medication only for patients above a specified risk threshold. Such a strategy is a rule mapping observed treatment history and current characteristics to the next decision. The distinction matters because static and dynamic strategies generally answer different questions and require different implementations.

By fixing the contrast at (1,1) against (0,0), the experiment makes the target explicit. A regression coefficient from a generic longitudinal model is not automatically equal to that target. One must show how the estimator reconstructs the potential-outcome means under the two interventions.

## The data-generating mechanism

For individual i, generate a baseline variable L₀ from a standard normal distribution. Initial treatment A₀ is binary, with probability

\[
P(A_0=1\mid L_0)=\operatorname{expit}(0.8L_0),
\]

where expit(z)=1/(1+e⁻ᶻ). Thus individuals with larger baseline values are more likely to receive initial treatment. The intermediate variable is then generated according to

\[
L_1=0.8L_0+1.2A_0+\varepsilon_L,
\]

with independent standard normal disturbance ε_L. The earlier treatment increases the expected intermediate state by 1.2 units. This is the pathway that an ordinary conditional regression can inadvertently remove when estimating the total effect.

Later treatment follows

\[
P(A_1=1\mid L_1)=\operatorname{expit}(0.8L_1).
\]

Because L₁ is affected by A₀ and affects the probability of A₁, later assignment depends on a treatment-induced variable. L₁ also affects the outcome, so it is a confounder of the relationship between the later treatment and outcome. At the same time, it mediates part of the first treatment's effect. These are two causal roles of the same observed variable.

The outcome equation is

\[
Y=1+0.5L_0+1.3L_1+0.7A_0+1.1A_1+\varepsilon_Y,
\]

where ε_Y is an independent standard normal disturbance. All disturbances and individuals are independent. All variables are observed; no missing data or hidden confounders are introduced. The assignment and outcome models are deliberately simple and correctly specified, so disagreement among estimators can be attributed to their causal targets and finite-sample weighting behaviour rather than unmeasured confounding.

Under the static intervention (a₀,a₁), the intermediate variable evolves as L₁(a₀)=0.8L₀+1.2a₀+ε_L. The resulting outcome is obtained by substituting this value and the assigned treatment decisions into the structural outcome equation. That operation, rather than conditioning on observed treatment values, defines the causal strategy.

## Deriving the true total effect

The treatment effect can be calculated without fitting a model. Taking expectations of the intermediate equation under a₀ gives

\[
E[L_1(a_0)]=0.8E[L_0]+1.2a_0+E[\varepsilon_L]=1.2a_0.
\]

Since the baseline variable and disturbance have mean zero, sustained treatment at the first time produces an expected intermediate state 1.2 units higher than sustained no treatment. The second treatment does not affect L₁ in this two-decision model because it is assigned afterward.

The outcome mean under a fixed strategy is therefore

\[
E[Y(a_0,a_1)]
=1+1.3(1.2a_0)+0.7a_0+1.1a_1.
\]

The effect of switching from (0,0) to (1,1) becomes

\[
\psi=1.3(1.2)+0.7+1.1=3.36.
\]

The first component, 1.56, is the pathway through the intermediate variable. The other two contributions, 0.7 and 1.1, arise from the direct structural terms of the two treatments. The total strategy effect includes all three. The individual effect is also constant in this constructed model, so the population mean does not mask treatment-effect heterogeneity.

A regression of Y on L₀, A₀, L₁ and A₁ has a correct conditional-mean specification. Its treatment coefficients converge to approximately 0.7 and 1.1. Adding them gives 1.8, which differs from 3.36 by exactly the mediated contribution 1.56. The regression is not statistically defective. The coefficient sum is a conditional contrast at fixed L₁, whereas the target allows L₁ to respond to the first intervention.

This is the central identification lesson. Estimating an outcome model correctly is not equivalent to evaluating the outcome under a complete intervention strategy. A model can fit the observed conditional distribution and yet be used to calculate the wrong causal functional.

## Why ordinary adjustment is not enough

The usual recommendation for observational treatment analysis is to control for confounders. In the present sequence, L₁ predicts both A₁ and Y. Omitting it from a simple regression of Y on A₀ and A₁ could leave confounding of the later treatment. Including it may therefore seem necessary and reasonable.

But L₁ also lies on a causal pathway from A₀ to Y. Conditioning on L₁ when reading the coefficient of A₀ removes that portion of the total effect. The two observations do not contradict one another. They show why an ordinary treatment-coefficient interpretation is not designed for a longitudinal strategy with feedback.

A causal graph would include arrows from L₀ to A₀ and L₁, from A₀ to L₁ and Y, from L₁ to A₁ and Y, and from A₁ to Y. It would also include the baseline effect on Y. This temporal graph makes the competing roles visible. L₁ is a confounder for the later treatment and simultaneously a descendant of the earlier treatment.

The correct response is not simply to omit L₁ from every model. Sequential G-computation explicitly models its conditional evolution. Inverse probability weighting uses L₁ to model the assignment mechanism for the later treatment. The methods handle this information in ways compatible with the marginal strategy estimand rather than conditioning it away in the final contrast.

In other words, the important distinction is where a variable enters the calculation and which causal functional is evaluated. Variables can appropriately enter nuisance models while being absent from a marginal structural outcome contrast. Calling one variable a confounder without reference to time and treatment history is too imprecise for longitudinal inference.

## Sequential exchangeability

The longitudinal G-formula and treatment weighting both require assumptions connecting observed outcomes to intervention outcomes. One is consistency: observed outcomes and intermediate measurements correspond to the potential outcomes for the treatment history that an individual actually experienced. Treatment versions and measurement times must therefore be meaningfully defined.

Another is sequential exchangeability. At the first decision, treatment must be independent of the relevant future potential outcomes conditional on the recorded baseline history. At the second decision, later treatment must be independent of future potential outcomes conditional on the observed history up to that decision, including baseline values, earlier treatment and intermediate measurements.

These are not consequences of fitting a propensity model. They are causal assumptions about the decision process. In the simulation they hold because assignment probabilities depend only on observed histories and independent random draws. In applied work they may fail when clinicians, caseworkers or algorithms use prognostic information that is not retained in the analysis dataset.

Positivity is also required. Each treatment option included in the target regime must have positive probability among the relevant histories. The logistic assignment mechanisms used here satisfy this condition at every finite value of the covariates, although very small probabilities can still produce unstable weights in finite samples.

Finally, an analysis should consider interference. If one individual's treatment changes another person's outcome, the simple individual-level potential-outcome indexing is incomplete. This may matter for employment programmes that redistribute scarce jobs, vaccination policies with spillovers or marketing interventions where users interact.

These assumptions identify a marginal strategy effect under the specified model. Neither weighting nor G-computation can manufacture exchangeability when relevant common causes of assignment and outcomes are missing.

## Sequential inverse probability weighting

Marginal structural models offer a way to describe outcomes under treatment strategies without conditioning the marginal effect itself on variables affected by previous treatment. Their estimation commonly uses inverse probability weights built from the observed sequence of assignments.

For two binary decisions, the conditional probability of the observed treatment history is the product of the two appropriate assignment probabilities. An individual following a particular strategy receives an inverse weight proportional to the reciprocal of that history probability. Rare observed histories receive larger weights because they stand in for other people with similar histories who usually followed a different assignment path.

For an individual with assignments a₀ and a₁, let p₀(a₀|L₀) denote the first treatment probability and p₁(a₁|L₀,A₀,L₁) the second. The joint inverse assignment weight is

\[
w_i=\frac{1}{p_0(A_{0i}\mid L_{0i})\,p_1(A_{1i}\mid L_{0i},A_{0i},L_{1i})}.
\]

The implemented analysis estimates each probability with logistic regression. The first model uses baseline L₀; the second uses the measured history L₀, A₀ and L₁. Estimated probabilities are clipped to [0.01,0.99] for numerical protection, although in this design the main objective is to recover the correctly specified treatment process rather than compare clipping rules.

For the regime (1,1), I calculate the inverse-weighted mean outcome among individuals who actually received A₀=A₁=1. I do the same for the (0,0) regime. Each regime-specific mean is normalized by the sum of its own weights, and the difference forms the sequential weighting estimate. This is a normalized, or Hájek-type, inverse-probability estimator of the two strategy means. It is not a regression on individual treatment-history indicators with an unqualified interpretation of every coefficient.

Under the identifying assumptions and consistent nuisance estimation, the weighted comparisons recover the potential-outcome means for the specified regimes. With weak overlap or extreme weights, the finite-sample variance can be substantial even when the underlying identification identity is correct.

## Why weighting uses the intermediate variable

It might seem contradictory to say that ordinary adjustment for L₁ misses a causal pathway while also recommending that L₁ enter the second treatment propensity model. The difference is the role played by the variable.

In a conventional conditional outcome regression, fixing L₁ while changing A₀ defines a comparison at the same intermediate state. That comparison does not include the effect of A₀ on L₁. In the weighting approach, L₁ helps characterize how likely each observed later assignment was, given the history the individual experienced. Weights rebalance treatment history so that the final contrast can be marginal over the intermediate response.

No intervention is made to force L₁ to be equal across the two target regimes. The outcome comparison is instead constructed to represent what would happen if the treatment decisions were set while intermediate states remained governed by their structural response to earlier treatment.

The causal interpretation depends on the whole procedure, not on whether L₁ appears somewhere in the code. A variable can be necessary in one nuisance model while inappropriate as a fixed condition for reading another model's coefficient as a total effect.

This distinction is a general feature of longitudinal causal methods. A time-varying confounder affected by earlier treatment cannot always be handled by inserting every observed time-dependent covariate into an ordinary regression and summing its treatment coefficients. The g-methods were developed to address precisely this kind of feedback.

## Parametric G-computation

The G-formula reconstructs intervention outcomes by modelling the conditional distribution of intermediate variables and outcomes and then averaging the predicted outcomes under a specified treatment strategy. In this simulation, linear models suffice because the structural intermediate and outcome equations are linear.

First fit L₁ as a function of L₀ and A₀. For a chosen static treatment a₀, predict the intermediate value each individual would have under that assignment. Second fit Y as a function of L₀, A₀, L₁ and A₁. To evaluate strategy (a₀,a₁), predict outcomes after replacing both treatment decisions and inserting the predicted intervention-specific intermediate value.

The strategy-specific predicted outcomes are averaged across the observed baseline population. Comparing the two averages gives the G-computation estimator for the sustained-regime effect. Crucially, L₁ is not held at its observed value under the intervention; its predicted value is updated according to the specified initial treatment.

The procedure uses assumptions different from those needed for propensity-only weighting. It depends on correct specification of the conditional models used in the sequential calculation. In this experiment that specification is deliberately favourable. The fitted outcome relationship includes all variables in the structural equation, and the intermediate model reproduces the correct conditional mean.

Consequently G-computation can take advantage of the simple outcome structure and achieve lower sampling variability than inverse weighting. That efficiency advantage does not establish that outcome modelling is always preferable. If the models for intermediate evolution or outcomes were misspecified, its counterfactual predictions could be systematically wrong even when treatment assignment models remained valid.

## The benchmark and its estimators

For each of 160 independent simulated datasets, I generate 2,200 complete observational histories according to the published equations. Three estimates are then calculated: the normalized sequential IPW contrast between observed (1,1) and (0,0) histories; the parametric G-computation contrast between the same strategies; and the sum of A₀ and A₁ coefficients from a conventional outcome regression adjusting for L₀ and L₁.

The benchmark records each effect estimate. For the weighted analysis, it also records a Kish-type effective sample size for each regime-specific weighted mean, calculated as the squared sum of weights divided by the sum of squared weights. These quantities show how concentrated the information is within each observed history group.

Across independent datasets I report the mean estimate, standard deviation and root mean squared error relative to the known total effect of 3.36. RMSE combines systematic bias and Monte Carlo variability. The experiment does not present confidence intervals for individual datasets, so Monte Carlo standard deviations should not be confused with a validated longitudinal sandwich or bootstrap uncertainty estimator.

The random generator seed is fixed at 20261018, and the implementation uses NumPy and scikit-learn. All nuisance models are refitted for every simulated dataset. The data are complete, so the results are not affected by attrition or censoring. No tuning procedure selects the model that looks most favourable in each replication.

The total of 160 replications is enough to show the large contrast between the methods in this constructed setting, although it is not a comprehensive simulation study of every variance estimator or implementation detail. More repetitions and additional data-generating mechanisms would be required before making general claims about small differences in performance.

## Estimates against the causal truth

The Monte Carlo summaries are:

| Method | Mean estimate | Bias | SD | RMSE |
| --- | ---: | ---: | ---: | ---: |
| Sequential IPW | 3.3540 | −0.0060 | 0.2044 | 0.2039 |
| G-computation | 3.3571 | −0.0029 | 0.0901 | 0.0899 |
| Conventional adjusted regression | 1.7973 | −1.5627 | 0.0723 | 1.5643 |

The two sequential estimators recover the known total effect closely on average. Their small mean differences from 3.36 are compatible with simulation variation. The conventional adjusted regression, by contrast, remains near 1.8, exactly where the sum of the direct structural treatment coefficients should lie.

It is important not to describe the conventional regression as an invalid model of the observed outcome distribution. The regression conditions on the intermediate variable and estimates the conditional structural coefficients as specified. The error arises when the direct conditional coefficient sum is treated as though it were the total marginal effect of sustained intervention.

G-computation achieves RMSE 0.0899 compared with IPW's 0.2039. The difference is substantial in this simulation, but it reflects favourable outcome-model specification and the noise introduced by weighting rare treatment histories. The result does not imply that G-computation has uniformly smaller variance or bias in applied observational studies.

The weighted estimator's mean is very close to the truth without directly fitting the outcome structural equation. That is one attraction of treatment-weighting approaches: they can identify the strategy contrast from the treatment mechanism and observed outcome histories, under sequential exchangeability, consistency and positivity. The price is sensitivity to the distribution of fitted treatment probabilities.

## How much information remains after weighting?

Of the 2,200 individuals in each dataset, only some follow the sustained treatment history (1,1) and others follow sustained no treatment (0,0). Their inverse weights vary because the histories are more likely for some baseline and intermediate characteristics than others. The weight concentration reduces the effective information used for each strategy mean.

The average Kish effective sample size among always-treated histories is approximately 503.95, with a between-replication standard deviation of 62.73. For the never-treated histories it is approximately 299.45, with standard deviation 61.47.

These figures should not be interpreted as if the entire causal study literally contains only 504 or 299 independent individuals. They are diagnostics for the concentration of weights within the corresponding observed regime subsets. Nor are they inferential degrees of freedom for a fully fitted sequential IPW estimator. The uncertainty must also account for estimation of treatment probabilities and the actual assignment structure.

Nevertheless, the numbers explain part of the contrast in RMSE. The G-computation estimate uses conditional models fitted across the available histories and then predicts intervention outcomes throughout the baseline sample. The IPW approach depends directly on observed histories matching the target strategies and gives more weight to histories that were unlikely under the observed assignment process.

A large longitudinal dataset can therefore contain relatively little well-supported information about particular sustained interventions, especially when treatment is almost deterministically adapted to prognosis. Positivity is more than a mathematical denominator condition; practical overlap determines how fragile the estimate may become.

## Marginal structural models and stabilized weights

A marginal structural model specifies how the mean potential outcome depends on treatment history or strategy, marginalizing over appropriately evolving time-dependent covariates. For two binary assignments, one might describe intervention means with an intercept, two treatment terms and an interaction. The interpretation of its coefficients depends on that parameterization and whether all relevant treatment histories are represented.

Weights can be constructed as unstabilized inverse probabilities or stabilized ratios. Stabilized weights commonly place probabilities conditional on reduced treatment history in the numerator and probabilities conditional on full measured confounder history in the denominator. Their purpose may include improving finite-sample stability while preserving an appropriate target under the specified model.

The numerical estimator in this article is more narrowly defined: it uses products of inverse assignment probabilities and computes separately normalized weighted means for the two sustained regimes. That is sufficient for the contrast being studied, but it is not an implementation of every possible marginal structural model or a general demonstration that stabilization always reduces variance.

The distinction matters in reproducible work. A formula labelled simply 'MSM weights' is insufficient to reconstruct an analysis. The analyst should specify treatment histories, numerator and denominator models, whether probabilities are clipped, which regimes are included, how weighted means or regression coefficients are calculated, and how uncertainty is estimated.

In practice, competing weight specifications can change both effective sample size and the implicit population comparison. Those changes should be reported as part of the scientific evidence rather than hidden as numerical preprocessing.

## Positivity and sustained interventions

A treatment history can be improbable even when each individual treatment decision has a nonzero conditional probability. The probability of receiving an entire sequence multiplies the probabilities at its component decision times. As the number of treatment decisions grows, a sustained strategy can become rare in important parts of the population.

This means that longitudinal weighting may become more variable over time. A regime requiring treatment at every visit might be uncommon among people whose evolving prognosis often causes clinicians or caseworkers to discontinue treatment. A regime forbidding treatment at every visit might be uncommon among people whose prognosis repeatedly triggers escalation. Estimating both potential-outcome means then requires rare observed paths with large inverse weights.

The two-decision simulation is intentionally moderate: the assignment probabilities are smooth logistic functions and histories are not impossibly rare. Even so, the effective sample size for each sustained regime is much smaller than the nominal sample size. Longer treatment sequences could amplify the problem.

Clipping and truncation can control extremes but may introduce bias for the original causal target. Restricting the study to people with adequate support changes the target population. Dynamic policies that match feasible clinical or operational practice may sometimes be more meaningful than hypothetical sustained static regimes. These are choices about the estimand and design, not merely the numeric stability of a weighted calculation.

A strong analysis should therefore assess the feasibility and support of the intervention before interpreting a treatment-history effect. Statistical modelling cannot generate counterfactual evidence for histories the observed assignment system essentially never produces.

## Why the two g-methods agree here

Agreement between sequential IPW and G-computation is expected under the present construction because both nuisance-model families match the true data-generating mechanism, all required covariates are observed, and the target strategy is explicitly defined. They reconstruct the same potential-outcome contrast through different observable functionals.

But agreement is not independent verification of sequential exchangeability. If both approaches operate on a dataset missing an important common cause of later assignment and outcome, both can be biased. Their numerical agreement could reflect a shared identification failure rather than strong causal evidence.

They also differ in model dependence. G-computation relies on the conditional evolution and outcome relationships used to simulate interventions. IPW relies on the assignment mechanisms and empirical outcome comparisons under reweighting. If an outcome model is badly misspecified, the weighting estimator may retain an advantage. If treatment probabilities are extreme or poorly estimated, a correctly specified outcome model may be more stable.

Combining the two nuisance-model families leads to longitudinal doubly robust methods, including targeted and augmented approaches, whose properties require their own assumptions and influence-function calculations. Double robustness does not eliminate the need for positivity or sequential exchangeability, nor does it make every finite-sample implementation reliable.

The present benchmark deliberately avoids claiming that the near-identical means validate the causal design. The design is known because it was specified, and the estimator comparison measures behaviour under those assumptions.

## Inference beyond a point estimate

The benchmark's standard deviations are empirical variations across independently simulated datasets. They are useful for comparing estimator stability under the chosen mechanism. They are not confidence intervals for an actual policy evaluation, where the causal truth is unknown and only one realized dataset may be available.

For sequential IPW, uncertainty depends on outcome variation, the assignment models, the estimated weights and any clustering or dependence in observations. A procedure that treats fitted weights as nonrandom can misrepresent variability. Sandwich estimators, bootstrapping entire independent units and influence-function methods are possible tools when their conditions hold.

For G-computation, uncertainty must account for estimating the intermediate and outcome models and applying their predictions to intervention-specific histories. A bootstrap that refits the complete procedure can be a practical option, although its validity depends on the design and estimator. Longitudinal data with clustered or dependent units require additional care.

The effective sample size diagnostic is not a replacement for these calculations. It summarizes how unequal weights are, but does not incorporate the entire estimator's stochastic dependence. A low effective size signals an instability risk; a large one does not prove the treatment assignment and counterfactual assumptions.

An eventual empirical article should report intervals and uncertainty diagnostics for each estimator alongside weight distributions and intervention-specific outcome predictions. Narrow intervals are useful only after the causal estimand and statistical procedure are clearly established.

## When the sequential models are wrong

The simulation grants the estimators favourable conditions. The logistic treatment models match the true assignment functions, and the linear intermediate and outcome models match the structural conditional means. Real longitudinal systems can contain nonlinear interactions, delayed effects, history-dependent disturbances and measurement error.

If the second treatment decision depends on an unrecorded variable that also predicts the outcome, sequential exchangeability fails. No amount of propensity tuning or outcome-model flexibility can repair an omitted common cause without additional identifying information. If the intermediate state is measured inaccurately, both assignment modelling and counterfactual simulation can be distorted.

Misspecified outcome dynamics can also affect G-computation even when treatment assignment models are correct. For instance, if the effect of the first treatment on L₁ varies with L₀ but the intermediate model assumes a constant additive coefficient, intervention-specific predictions may be systematically wrong. If later outcomes depend nonlinearly on L₁ and the fitted outcome model is linear, averaging predicted trajectories can produce biased regime means.

Weighting suffers differently. A misspecified second-stage propensity model can fail to balance treatment history and prognosis, and extremely small estimated probabilities can concentrate information on a few observations. More flexible machine learning may improve nuisance estimation but can make weight distributions harder to control and does not guarantee appropriate convergence without suitable regularity and sample-splitting strategies.

These are important reasons to treat g-methods as identification-and-estimation frameworks, not automatic causal correction procedures.

## Static versus dynamic regimes

Sustained treatment at both decision times is easy to describe and evaluate in the simulation, but may be an unrealistic policy in practice. A clinician may wish to continue treatment only when a biomarker exceeds a threshold. An employment agency may provide a second support session only when job-search progress remains low. Such rules respond to intermediate histories rather than assigning one fixed treatment path to everyone.

A dynamic treatment regime defines the treatment at each time as a function of the information available at that time. The resulting potential outcomes are indexed by the decision rule. G-computation can evaluate such rules by simulating histories sequentially and applying the rule at each decision point. Weighting approaches can also evaluate regimes, subject to support and appropriate definitions of compliance with the rule.

Dynamic strategies bring additional questions. Which state variables are available before each decision? Are the measurements timely and reliable? Is the proposed rule feasible? How many individuals in the observational data follow histories consistent with it? Do implementation costs and capacity limits change the policy objective?

The contrast of (1,1) and (0,0) in this article should therefore be read as a controlled benchmark of the underlying methods rather than a claim that fixed regimes are always the relevant operational choice. The scientific estimand must follow the actual intervention being considered.

## Relation to earlier parts of the series

Part 1 showed that a precise regression coefficient can converge to a noncausal population association. Part 3 showed why balancing observed variables does not establish the absence of hidden confounding. Part 4 examined how inverse weights become unstable when treatment probabilities are extreme. Part 5 demonstrated that conditioning on a mediator can change a total-effect estimand. Part 10 compared augmented inverse probability weighting with its component nuisance models, and Part 11 introduced counterfactual outcome modelling through the G-formula.

The present longitudinal experiment combines several of those lessons. L₁ is both prognostic and affected by earlier treatment. Adjusting for it in a final regression changes the interpretation of A₀. Sequential weighting uses its treatment-prediction role without defining the total effect at a fixed intermediate state. G-computation models its evolution under each intervention. Both sequential estimators target the complete history contrast under assumptions that are explicit in the simulation.

The exercise also shows that a valid causal strategy and a low-variance estimator are separate achievements. In this favourable model, G-computation has smaller Monte Carlo error than weighting because its parametric assumptions are correct. In another design, that advantage could vanish or reverse. A credible comparison reports where the information comes from and which assumptions each estimator uses.

## Practical evaluation plan

An analysis of time-varying treatment should begin with a timeline. Identify every decision point, the baseline and intermediate information measured before each decision, the outcome window and any censoring or selection. Treatment assigned at time one cannot legitimately depend on measurements that occur later, and variables affected by earlier treatment cannot be interpreted as ordinary baseline controls.

Next define the policy strategies. State whether they are static or dynamic and specify the population and intervention versions. An estimand described only as 'the coefficient of treatment' is too vague when treatment is a sequence of decisions. The policy must say what decisions are being contrasted and how the intermediate states are allowed to respond.

Then state the identifying assumptions. Explain why treatment assignment would be conditionally exchangeable at each decision time given the recorded history; examine whether each proposed regime has adequate support; consider consistency, interference and measurement quality. The assignment process may be governed by clinical or institutional rules that demand careful substantive investigation.

Only after that should the estimator be selected. For weighting, document the treatment probability models, any stabilization or clipping, weight quantiles and effective sample size by regime. For G-computation, document intermediate and outcome models and how predictions are propagated under the intervention. If several methods are compared, identify what each targets and why their results might differ.

Finally report uncertainty, diagnostics and sensitivity analyses. The relevant test is not whether all available columns were included in a regression, but whether the longitudinal counterfactual comparison is sufficiently supported to guide the decision being considered.

## Limits

This benchmark has two treatment times, one baseline covariate, one intermediate confounder and a continuous outcome. The structural relationships are linear except for the logistic assignment probabilities. Every confounder affecting assignment and outcome is recorded without error. Outcomes are complete, observations are independent, and treatment effects are constant across individuals.

The sequential IPW estimator uses normalized regime-specific weighted means with probabilities clipped to [0.01,0.99]. It is not a general-purpose implementation of all marginal structural models, nor does it compare stabilized and unstabilized regression estimates under many specifications. The G-computation estimator uses correctly specified linear conditional models, giving it an advantage in the present outcome mechanism.

Only 160 independent Monte Carlo datasets were generated. That supports the large qualitative differences in the results but is not enough to establish subtle efficiency rankings across a broad set of designs. The reported SD and RMSE are Monte Carlo summaries rather than validated confidence intervals for a single empirical longitudinal study.

The model does not investigate hidden confounding, positivity violations, informative loss to follow-up, competing events, interference, misspecified nuisance functions or dynamic policy optimization. Any of these can change the inferential conclusions. In particular, a successful simulation under known sequential exchangeability does not tell us whether the same assumption holds in actual treatment records.

The exercise is therefore a controlled demonstration of treatment–confounder feedback and estimator targets, not evidence about a real employment or medical intervention.

## What the investigation established

The true effect of sustained treatment rather than sustained no treatment was 3.36. Across 160 datasets with 2,200 individuals each, the sequential weighting estimator averaged 3.3540 and parametric G-computation averaged 3.3571. The conventional regression contrast averaged 1.7973, close to the direct coefficient sum of 1.8, because it omitted the mediated pathway of 1.56 from its contrast.

Both sequential estimators recovered the total effect under the specified assumptions. G-computation had lower RMSE, 0.0899 rather than 0.2039, because the simple structural models were correctly specified. Sequential weighting used the observed treatment histories but had much smaller effective information within the sustained regimes, averaging roughly 504 and 299 effective observations for always-treated and never-treated histories.

The broader lesson is that a time-varying confounder affected by earlier treatment cannot always be handled by ordinary regression adjustment when the estimand is a total longitudinal intervention effect. The relevant question is how outcomes would evolve under complete treatment strategies, not how a conditional coefficient changes while an intermediate consequence is held fixed. The G-formula and marginal structural weighting provide two different ways to connect that question to observed longitudinal data, provided the assignment, support and consistency assumptions are defensible.

## Code

The experiment is in `causal_inference_statistics/part15/experiment.py`. It uses Python 3.12+, NumPy and scikit-learn, with fixed random seed 20261018. Running the script recreates `results.csv`, containing effect-estimator Monte Carlo summaries and effective sample sizes for the two sustained regimes. All data are generated by the published equations, and no external dataset is required.

## The series

1. When Regression Gets the Causal Effect Wrong
2. Randomization, Confounding and the Value of Experimental Design
3. Matching and Propensity Scores
4. Inverse Probability Weighting
5. Directed Acyclic Graphs and the Cost of Controlling for Everything
6. Difference-in-Differences
7. Instrumental Variables
8. Regression Discontinuity
9. Synthetic Controls
10. Doubly Robust Estimation
11. The G-Formula
12. Heterogeneous Treatment Effects
13. Double Machine Learning
14. Bayesian Causal Inference
15. Time-Varying Treatments and Marginal Structural Models (this article)

## References

Bang, H. and Robins, J. M. (2005). Doubly Robust Estimation in Missing Data and Causal Inference Models. *Biometrics*, 61(4), 962–973. https://doi.org/10.1111/j.1541-0420.2005.00377.x

Cole, S. R. and Hernán, M. A. (2008). Constructing Inverse Probability Weights for Marginal Structural Models. *American Journal of Epidemiology*, 168(6), 656–664. https://doi.org/10.1093/aje/kwn164

Daniel, R. M., Cousens, S. N., De Stavola, B. L., Kenward, M. G. and Sterne, J. A. C. (2013). Methods for Dealing with Time-Dependent Confounding. *Statistics in Medicine*, 32(9), 1584–1618. https://doi.org/10.1002/sim.5686

Hernán, M. A., Brumback, B. and Robins, J. M. (2000). Marginal Structural Models to Estimate the Causal Effect of Zidovudine on the Survival of HIV-Positive Men. *Epidemiology*, 11(5), 561–570. https://doi.org/10.1097/00001648-200009000-00012

Hernán, M. A. and Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC. https://miguelhernan.org/whatifbook

Kreif, N., Tran, L., Grieve, R., De Stavola, B., Tasker, R. C. and Petersen, M. (2017). Estimating the Comparative Effectiveness of Feeding Interventions in the Pediatric Intensive Care Unit: A Demonstration of Longitudinal Targeted Maximum Likelihood Estimation. *American Journal of Epidemiology*, 186(12), 1370–1379. https://doi.org/10.1093/aje/kwx213

Laan, M. J. van der and Robins, J. M. (2003). *Unified Methods for Censored Longitudinal Data and Causality*. Springer. https://doi.org/10.1007/978-0-387-21700-0

Lipsitch, M., Tchetgen Tchetgen, E. and Cohen, T. (2010). Negative Controls: A Tool for Detecting Confounding and Bias in Observational Studies. *Epidemiology*, 21(3), 383–388. https://doi.org/10.1097/EDE.0b013e3181d61eeb

Mansournia, M. A., Etminan, M., Danaei, G., Kaufman, J. S. and Collins, G. (2017). Handling Time Varying Confounding in Observational Research. *BMJ*, 359, j4587. https://doi.org/10.1136/bmj.j4587

Naimi, A. I., Cole, S. R. and Kennedy, E. H. (2017). An Introduction to G Methods. *International Journal of Epidemiology*, 46(2), 756–762. https://doi.org/10.1093/ije/dyw323

Petersen, M. L., Porter, K. E., Gruber, S., Wang, Y. and van der Laan, M. J. (2012). Diagnosing and Responding to Violations in the Positivity Assumption. *Statistical Methods in Medical Research*, 21(1), 31–54. https://doi.org/10.1177/0962280210386207

Robins, J. M. (1986). A New Approach to Causal Inference in Mortality Studies with a Sustained Exposure Period—Application to Control of the Healthy Worker Survivor Effect. *Mathematical Modelling*, 7(9–12), 1393–1512. https://doi.org/10.1016/0270-0255(86)90088-6

Robins, J. M., Hernán, M. A. and Brumback, B. (2000). Marginal Structural Models and Causal Inference in Epidemiology. *Epidemiology*, 11(5), 550–560. https://doi.org/10.1097/00001648-200009000-00011

Robins, J. M., Rotnitzky, A. and Zhao, L. P. (1994). Estimation of Regression Coefficients When Some Regressors Are Not Always Observed. *Journal of the American Statistical Association*, 89(427), 846–866. https://doi.org/10.1080/01621459.1994.10476818

VanderWeele, T. J. (2015). *Explanation in Causal Inference: Methods for Mediation and Interaction*. Oxford University Press.

## Scientific audit addendum: an explicit saturated marginal structural model

The original sequential-IPW contrast can be expressed as a saturated marginal structural model for the four possible treatment histories. Consider

\[
E[Y^{a_0,a_1}]=\beta_0+\beta_1 a_0+\beta_2 a_1+\beta_3 a_0a_1.
\]

The causal contrast between sustained treatment and sustained non-treatment is \(\beta_1+\beta_2+\beta_3\). Fitting this model by weighted least squares with inverse *joint* treatment-history probabilities assigns a fitted mean to each of the four observed histories. In particular, the fitted values for histories (1,1) and (0,0) are precisely the corresponding weighted group means. Their difference is therefore **algebraically equivalent** to the normalized regime-IPW difference already reported in this article. It is not a new statistically independent estimator.

I tested that equivalence with the original data-generating process, random seed, 160 Monte Carlo repetitions and 2,200 individuals per replicate. Mean estimates were 3.3539826283 for both approaches, with RMSE 0.2038519483 against the true effect 3.36. The maximum absolute discrepancy between the two estimates over all replications was approximately \(4.0\times10^{-15}\), attributable to floating-point arithmetic.

The additional files `validate_explicit_msm.py` and `msm_validation_results.csv` reproduce and check this comparison. This validation clarifies the model implied by the original analysis but does **not** demonstrate superior accuracy or independent confirmation of the assumptions. Because the MSM is saturated for a two-decision binary treatment history, the method is equivalent to estimating all four weighted history-specific means. A parsimonious MSM, a more complex dynamic treatment regime, and robust inference for its fitted parameters would require further study.
