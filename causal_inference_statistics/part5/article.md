# Directed Acyclic Graphs and the Cost of Controlling for Everything

*Part 5 of Causal Inference in Statistics. Four data-generating mechanisms, 12,000 simulated datasets and a comparison of regression adjustment when the controlled variable is a confounder, collider, mediator or irrelevant covariate. Why adding a predictor can change the question—or introduce bias. Python code provided.*

A recurring recommendation in applied statistical modelling is to include every variable that helps explain an outcome. More information should improve prediction, and a regression with more relevant predictors should leave less variation unexplained. The temptation is to extend that reasoning to causal inference: if adjustment for a confounder removes bias, perhaps controlling for every available variable makes causal conclusions safer.

That conclusion is false. A variable can predict an outcome yet be inappropriate to condition on when estimating a particular causal effect. It can be a common cause of treatment and outcome, a consequence of both treatment and an unmeasured outcome cause, or an intermediate variable on the path through which treatment operates. The same mathematical operation—adding a column to an ordinary least squares design matrix—has different causal consequences in each case.

This article makes the distinction empirical. I generate four causal systems with known intervention effects. The first contains a confounder that should be adjusted for. The second contains a collider: a variable affected by treatment and by another cause of the outcome. The third contains a mediator on the causal path from treatment to outcome. The fourth adds a variable unrelated to treatment and outcome. In each system I compare an unadjusted regression with a regression controlling for the additional variable.

Each comparison is repeated on 1,000 independently generated datasets at sample sizes 200, 1,000 and 5,000. The outcome models are linear and deliberately simple enough for their population coefficients to be derived exactly. At n=5,000, adjustment changes the mean estimate from 1.5002 to 1.0003 for the confounder, removing almost all causal bias. For the collider, adjustment moves a previously correct estimate of 1.0009 to 0.5001. For the mediator, adjustment changes a correct estimate of the total effect 2.0004 to a coefficient near 1, corresponding to the direct effect instead. An irrelevant covariate makes little difference.

These are not contradictions in regression algebra. They are differences in what conditioning does to the causal relationships among variables. The result is a direct reason to define the causal estimand and adjustment set before searching for the model that best predicts the observed outcomes.

## Why the adjustment question comes first

Consider a public programme intended to reduce unemployment. A dataset records programme participation, baseline circumstances, follow-up contact with employment advisers and eventual employment outcomes. An analyst might put all available variables into one regression, reasoning that including more information makes the coefficient of programme participation more credible.

Some baseline characteristics may indeed be common causes of programme participation and outcome. They are candidates for confounding adjustment. But the number of follow-up contacts might be affected by programme participation. Conditioning on it can remove part of the programme's effect or create other biases. An indicator of whether a worker is referred for additional assessment might be determined jointly by programme participation and an unrecorded measure of difficulty. Conditioning on that indicator can create an association between treatment and prognosis even if the original assignment were randomized.

The purpose of a causal diagram is not to make the analysis look more scientific. It is to represent assumptions about which variables cause which other variables, and to work out what those assumptions imply for adjustment. A directed acyclic graph, or DAG, encodes a collection of directional relationships without directed cycles. The arrows are causal hypotheses, not estimated regression coefficients.

A model can have excellent out-of-sample prediction while being inappropriate for estimating the total effect of a policy. A post-treatment mediator may predict outcomes strongly precisely because it transmits the intervention's consequences. Adjusting for it may improve prediction while removing the part of the treatment effect the analyst intended to measure.

The central practical question is therefore not which predictors are statistically significant. It is which variables must be conditioned on to block noncausal paths without blocking the causal pathways included in the target estimand or opening previously closed paths.

## A structural definition of the effect

Let A denote treatment and Y outcome. Under an intervention that sets A to a value a, the potential outcome is Y(a). The effect of increasing treatment by one unit is the contrast Y(a+1)−Y(a), and its population average defines the causal effect of interest.

In the linear examples below, the intervention has a particularly transparent interpretation: replace the equation determining A with a fixed value, leave the other structural equations unchanged, and calculate the resulting outcome. This is the intervention interpretation associated with structural causal models.

Conditioning on A=a in observational data is not necessarily the same operation. Conditioning restricts attention to units whose treatment happened to have a particular value under the original assignment mechanism. Intervening changes that mechanism. If A shares a cause with Y, the conditional distribution of those units can differ in baseline prognosis. If conditioning additionally involves a collider, it can create associations that were absent before the restriction.

The data-generating equations allow us to compare these operations against exact causal effects. This is a methodological experiment rather than an attempt to estimate an effect from an actual social programme. It does not prove that any particular real-world DAG is correct. It shows why the arrows matter once a structure is assumed.

All disturbances in the simulation are independent standard normal variables. Observations are independent across units. This provides clean population benchmarks and avoids mixing causal identification with dependence, missing data, measurement error or numerical optimization problems.

## What a DAG says

A DAG consists of nodes representing variables and directed edges representing causal relationships. In a structural model, an arrow from X to Y indicates that X appears as a parent in the mechanism generating Y, though a fully developed causal model also specifies how these mechanisms respond to intervention.

Three local patterns are especially important. A chain X → M → Y represents a pathway through an intermediate variable M. A fork X ← U → Y represents a common cause U of two variables. A collider X → C ← U represents a common effect C of X and U.

The patterns look similar in a regression table: X, U, M and C are columns available for adjustment. Their conditional-independence implications are very different. In a fork, conditioning on the common cause can remove an association between its children. In a chain, conditioning on the middle variable can block transmission through that pathway. In a collider, conditioning on the common effect can create association between causes that were otherwise independent.

These rules are often expressed using d-separation. A path is blocked or open according to its collider and noncollider nodes and which variables, including descendants of colliders, are conditioned on. In an appropriate causal DAG, d-separation implies conditional independences under the Markov assumptions represented by the graph.

The graph itself is not discovered uniquely from a regression output. Multiple causal structures can be compatible with the same observational conditional independences. Temporal information, institutional knowledge, experiments and assumptions about the data-generating mechanisms are needed to decide which arrows are plausible.

This is one reason causal diagrams are useful: they force the modeller to state assumptions that a table of fitted coefficients otherwise leaves implicit. Their usefulness is conditional on the credibility of those assumptions, not a magical property of drawing arrows.

## Case one: adjustment for a confounder

The first mechanism has U as a common cause of treatment and outcome:

\[
U\longrightarrow A,\qquad U\longrightarrow Y,\qquad A\longrightarrow Y.
\]

Let

\[
A=U+\varepsilon_A,\qquad
Y=A+U+\varepsilon_Y,
\]

with U, ε_A and ε_Y mutually independent and each having unit variance. Setting A by intervention changes Y with coefficient one, so the causal effect of a unit increase in treatment is exactly 1.

But the unadjusted observational slope is

\[
\frac{\operatorname{Cov}(A,Y)}{\operatorname{Var}(A)}
=1+\frac{\operatorname{Cov}(A,U)}{\operatorname{Var}(A)}
=1+\frac12=1.5.
\]

The backdoor path A ← U → Y creates an associational component in addition to the direct causal path A → Y. Conditioning on U blocks the noncausal path, and the correctly specified regression of Y on A and U has treatment coefficient one.

This is the familiar success of covariate adjustment. It is important because many observational designs depend on some version of the same reasoning. If the required common causes are measured before treatment, if there is sufficient variation in treatment conditional on them, and if the estimator appropriately captures the conditional outcome relationship, adjustment can identify the target.

The success does not justify adjusting for every variable. It justifies conditioning on U in this particular graph and model. The same regression command applied to a variable with another causal role can produce a different result.

## Case two: conditioning on a collider

The second mechanism is

\[
A\longrightarrow Y,\qquad
A\longrightarrow C\longleftarrow U\longrightarrow Y.
\]

Treatment A and baseline cause U are independent. They both contribute to the collider C, while U also affects the outcome. The structural equations are

\[
C=A+U+\varepsilon_C,
\]

\[
Y=A+U+\varepsilon_Y.
\]

Every independent disturbance has unit variance. Setting A to a value a changes Y by one unit per unit change, so the causal effect is 1. Because A and U are independent, unadjusted regression of Y on A also has population slope 1.

Now include C in the regression. Conditional on a given value of C, a larger value of A makes a smaller U more plausible. C is the sum of A, U and noise, so holding the sum fixed creates an inverse association between two contributors to it.

Since U increases Y, the induced negative association between A and U changes the conditional treatment coefficient. The regression may then attribute some of the reduced U contribution to a weaker apparent effect of A. No original confounding between A and U was present; controlling for the common effect created the relevant dependence.

The coefficient can be derived exactly. With Var(A)=Var(U)=Var(ε_C)=1,

\[
\operatorname{Var}(C)=3,\quad
\operatorname{Cov}(A,C)=1,\quad
\operatorname{Cov}(Y,C)=2.
\]

Also Cov(A,Y)=1. The population partial regression coefficient of A conditional on C is

\[
\beta_{A\mid C}
=
\frac{\operatorname{Cov}(A,Y)\operatorname{Var}(C)
-\operatorname{Cov}(Y,C)\operatorname{Cov}(A,C)}
{\operatorname{Var}(A)\operatorname{Var}(C)
-\operatorname{Cov}(A,C)^2}
=\frac{3-2}{3-1}=\frac12.
\]

Thus the adjusted regression converges to 0.5 although the causal effect is 1. The adjustment-induced bias is negative one half.

This result is sometimes called collider stratification bias, and it belongs to a broader family of selection-bias mechanisms. A dataset conditioned on inclusion in a group can display the same pattern when inclusion is a common effect of treatment-related and outcome-related causes.

A treatment coefficient of 0.5 is not an alternative direct effect of treatment in this graph. C does not mediate the effect of A on Y; its causal arrow points out of A into C, and C does not cause Y. The regression coefficient is a conditional association distorted by conditioning on the collider.

The distinction is central. A variable does not need to be a confounder to be associated with both treatment and outcome in the data. In fact, conditioning on it can create the association that makes the treatment coefficient misleading.

## Why the collider predicts the outcome

The collider C contains U, and U influences Y. Consequently C can carry useful predictive information about the outcome. An analyst selecting regressors by predictive association may therefore have a clear numerical reason to include C. This is exactly what makes the example important.

Predicting Y conditional on A and C is a different task from estimating how Y changes when A is intervened on. In the observed conditional distribution, holding C fixed can imply a change in the distribution of U as A changes. Under an intervention that sets A, the independent baseline U does not change simply because a chosen value of C is held fixed in a regression.

The fitted conditional slope reflects the restriction imposed by conditioning. It is not guaranteed to represent a feasible intervention that changes A while holding the structural consequences of that change fixed.

In social and medical data, such conditioning can arise without an explicit decision to put a collider into a regression. Analysing only hospitalized patients, only employed workers, or only customers retained long enough to appear in a follow-up dataset may condition on a selection variable jointly influenced by different causes. Even when no collider column is included explicitly, selection into the observed sample can generate the same causal problem.

That does not mean every selected dataset is biased. The causal structure determines whether selection opens a relevant path. The graph is a way to state and analyse that question rather than to presume bias whenever a sample is restricted.

## Case three: adjusting for a mediator

The third mechanism has treatment affecting outcome both directly and through an intermediate variable:

\[
A\longrightarrow Y,\qquad A\longrightarrow M\longrightarrow Y.
\]

Define

\[
M=A+\varepsilon_M,\qquad
Y=A+M+\varepsilon_Y.
\]

Under an intervention setting A=a, the mediator changes with a, and the outcome becomes

\[
Y(a)=a+(a+\varepsilon_M)+\varepsilon_Y
=2a+\varepsilon_M+\varepsilon_Y.
\]

The total effect of increasing treatment by one unit is therefore 2. One unit comes through the direct A → Y path and another travels through A → M → Y.

Unadjusted regression of Y on A has population slope 2, correctly reflecting the total effect because A is independent of the disturbances.

Including M in the regression changes the meaning of the treatment coefficient. The structural outcome equation is Y=A+M+ε_Y, so the adjusted coefficient of A is 1. It corresponds to the direct effect in this simplified additive model, not to the total effect.

Calling this regression wrong without specifying the causal question would be too strong. If the research question is about a controlled direct effect under a well-defined intervention fixing M, the coefficient may be relevant under suitable identification conditions. If the research question is the total consequence of setting A, conditioning on M removes the mediated component and no longer answers it.

The distinction becomes harder in realistic mediation analyses. Mediator-outcome confounding, treatment-induced confounders, interactions and ambiguous interventions on the mediator can make direct and indirect effects more difficult to identify. A coefficient change after adjusting for a mediator is not by itself proof of a biological or behavioural mediation pathway.

The present simulation makes the interpretation exact because the structural equations are linear, additive and free of mediator-outcome confounding beyond the specified independent disturbances.

## Case four: adjusting for an irrelevant variable

The fourth mechanism serves as a negative comparison. Let A, ε_Y and V be mutually independent, with

\[
Y=A+\varepsilon_Y.
\]

The variable V affects neither treatment nor outcome. The true causal effect is 1, and unadjusted regression is correctly centred on one.

Adding V to the regression should not systematically change the population treatment coefficient. In the simulated samples it introduces a small additional estimation task but does not remove a confounding path, block a causal path or open a collider path.

Its effect on precision is not guaranteed to have a particular sign in every finite sample. An irrelevant regressor can slightly change the fitted treatment coefficient and residual variance by chance, and consumes a degree of freedom. Here, mean estimates and RMSE are almost indistinguishable at larger sample sizes.

This comparison matters because *not useful for causal identification* is not the same as *necessarily biases the effect*. A variable may be safe but unnecessary, useful for efficiency, required for identification, or harmful to the causal estimand. Treating adjustment as a simple good-versus-bad switch misses those distinctions.

It also explains why identifying an admissible adjustment set is separate from maximizing the number of explanatory variables. A defensible set contains variables appropriate to the causal structure and target, not every column available in the dataset.

## The experiment

All four systems are simulated at n=200, 1,000 and 5,000. Each sample-size condition uses 1,000 independent datasets from a fixed pseudorandom generator seed of 20261016. For every replicate, I generate four independent standard normal vectors and construct the appropriate treatment, outcome and adjustment variables.

The same random terms are reused within a replicate across the four mechanisms. This is a computational convenience that makes the scenarios comparable while preserving their stipulated structural relationships; each scenario is analysed according to its own equations. The Monte Carlo replications are independent across datasets.

Each dataset produces an ordinary least squares treatment coefficient from a model with intercept and treatment only, and another from a model with the designated additional variable. There is no automatic covariate selection, hyperparameter search or outcome-based choice of adjustment sets. The roles of U, C, M and V are fixed before simulation.

For each estimator and sample size, I report the mean coefficient, its standard deviation across simulated datasets, the bias relative to the *total* causal effect and root mean squared error relative to that effect.

The mediation case has a total effect of 2; the other three cases have a total effect of 1. Calculating the mediator-adjusted regression's error against 2 is intentional, because the article asks whether the fitted coefficient estimates the total effect. It does not imply that the coefficient near 1 is an incorrect estimate of the direct effect under this special structural model.

The experiment uses no fitted DAG discovery procedure. The graph is known because it is specified by the generating equations. The analysis tests the consequences of conditioning under known causal structures, not whether a graph could be learned from observations without prior assumptions.

## The estimates across sample sizes

The simulation outputs are:

| Structure | n | Unadjusted coefficient | Adjusted coefficient | True total effect |
| --- | ---: | ---: | ---: | ---: |
| Confounder | 200 | 1.4967 | 1.0000 | 1 |
| Confounder | 1,000 | 1.5000 | 1.0012 | 1 |
| Confounder | 5,000 | 1.5002 | 1.0003 | 1 |
| Collider | 200 | 0.9999 | 0.5031 | 1 |
| Collider | 1,000 | 1.0018 | 0.5024 | 1 |
| Collider | 5,000 | 1.0009 | 0.5001 | 1 |
| Mediator | 200 | 1.9961 | 0.9975 | 2 |
| Mediator | 1,000 | 2.0008 | 1.0023 | 2 |
| Mediator | 5,000 | 2.0004 | 1.0000 | 2 |
| Irrelevant | 200 | 1.0000 | 0.9999 | 1 |
| Irrelevant | 1,000 | 1.0012 | 1.0012 | 1 |
| Irrelevant | 5,000 | 1.0003 | 1.0003 | 1 |

Each entry is the mean of 1,000 independently fitted regressions. The closeness to the theoretical coefficients at n=5,000 confirms that the numerical experiments are displaying population relationships rather than isolated small-sample anomalies.

For the confounder, adjustment removes the extra half-unit association generated by U. For the collider, adjustment *introduces* an approximately half-unit downward bias. For the mediator, adjustment changes the estimand from the total effect of two to the direct effect of one. The irrelevant covariate leaves the estimate essentially unchanged.

A useful feature of this comparison is that the outcome-model fitting method is identical throughout. The difference cannot be attributed to one scenario using machine learning and another using classical regression, or to one algorithm receiving more tuning. It comes from the causal role of the variable conditioned on.

The fitted coefficient is a property of a specified conditional regression. Whether it estimates the desired intervention effect depends on how that conditioning relates to the generating system.

## Error does not necessarily vanish with more data

The root mean squared errors relative to the total effect show the same pattern:

| Structure | n | Unadjusted RMSE | Adjusted RMSE |
| --- | ---: | ---: | ---: |
| Confounder | 200 | 0.5005 | 0.0716 |
| Confounder | 1,000 | 0.5007 | 0.0306 |
| Confounder | 5,000 | 0.5004 | 0.0142 |
| Collider | 200 | 0.1023 | 0.5097 |
| Collider | 1,000 | 0.0439 | 0.4996 |
| Collider | 5,000 | 0.0198 | 0.5004 |
| Mediator | 200 | 0.0971 | 1.0076 |
| Mediator | 1,000 | 0.0442 | 0.9986 |
| Mediator | 5,000 | 0.0198 | 1.0002 |
| Irrelevant | 200 | 0.0713 | 0.0715 |
| Irrelevant | 1,000 | 0.0305 | 0.0305 |
| Irrelevant | 5,000 | 0.0142 | 0.0142 |

The collider comparison is particularly revealing. Without adjustment, RMSE decreases from about 0.102 at n=200 to 0.020 at n=5,000. With adjustment, RMSE remains near 0.5. The adjusted estimator becomes more concentrated around the wrong coefficient.

The confounder shows the opposite pattern. Its unadjusted RMSE remains near 0.5 because systematic bias persists, while the adjusted RMSE falls as sample size increases. The same numerical value of asymptotic bias appears in the two scenarios, but caused by opposite adjustment decisions: failing to condition in the confounder case and conditioning in the collider case.

The mediator's adjusted RMSE remains near one when judged against the total effect because the coefficient estimates the direct component. It would be misleading to present that error as evidence that the regression cannot estimate any causal parameter. It is evidence that the coefficient differs from the total effect the analysis was intended to estimate.

Finally, adjusting for V adds almost no substantive error in this chosen system. This acts as a control against an overly dramatic interpretation that any additional covariate is harmful. The point is not to minimize adjustment automatically, but to justify it.

## Why a p-value cannot choose the adjustment set

A common workflow selects covariates according to their p-values, predictive contribution, or effect on the treatment coefficient. None of those rules is sufficient for deciding whether a variable should be conditioned on in a causal analysis.

In the collider system, C predicts Y because it contains U, an outcome cause. A coefficient for C can therefore be statistically distinguishable from zero while controlling for C creates bias in the treatment effect. A significance test answers whether a statistical association is compatible with a null coefficient under a model. It does not identify the variable's causal position in the graph.

In the mediator system, M also predicts Y strongly. Adding it can improve conditional prediction, but it blocks an important causal pathway if the estimand is the total effect. A procedure that automatically adds strong outcome predictors may therefore change the scientific question while appearing to improve the regression's statistical fit.

The confounder U is an appropriate adjustment variable here because it is a common cause, not merely because it predicts Y. A variable can predict the outcome but be unnecessary for causal identification if it does not also participate in an open backdoor path. Conversely, a variable with weak marginal predictive association may be important in a more complex graph where confounding operates through interactions or conditional relationships.

Changes in the treatment coefficient after adjustment are informative but not a complete causal diagnostic. In the confounder example, a change toward one indicates removal of bias. In the collider example, the change away from one indicates introduced bias. In the mediator example, it indicates a change in estimand. The numerical fact that a coefficient changed does not reveal which explanation is correct.

The required information is about causes and timing, not only fitted regression statistics.

## Backdoor paths and adjustment criteria

The backdoor criterion provides a formal way to reason about adjustment in a causal DAG. In a simple setting, a suitable set of observed pre-treatment covariates blocks every path into treatment that carries noncausal association between treatment and outcome, without conditioning on descendants of treatment in ways that defeat the criterion.

For the confounder graph, A ← U → Y is a backdoor path, and conditioning on U blocks it. For the collider graph, A → C ← U → Y is initially blocked at the collider C. Conditioning on C opens that path, creating noncausal association between A and U. For the mediator graph, A → M → Y is a causal path, so conditioning on M blocks part of the total effect.

The full graphical adjustment theory is more general than these simple patterns. Valid adjustment sets can sometimes include variables that are not individually confounders, and restrictions involving descendants and colliders require care. Modern generalized adjustment criteria provide formal conditions beyond a simplistic instruction to control for every common cause.

The usefulness of a DAG is that it makes a candidate adjustment set testable against a stated structure in a logical sense. Given the graph, one can check whether the proposed conditioning set blocks the relevant noncausal paths. The graph itself remains an assumption. If important arrows are absent or wrong, the resulting recommendation may also be wrong.

An empirical analysis should therefore document the substantive basis for the diagram. Which variables precede treatment? Which are consequences of treatment? Which are jointly influenced by causes of selection and outcomes? These questions can be difficult in longitudinal observational data, where feedback between treatment, health or behaviour and subsequent treatment creates time-varying confounding.

No adjustment recipe can substitute for an accurate description of the process producing the data.

## The danger of adjusting for post-treatment variables

Many problematic variables become available only after treatment begins. A treatment can affect later measurements, attendance, adherence, intermediate clinical markers or engagement with a programme. Adding those measurements to a regression can be attractive because they are predictive and may explain much of the observed variation.

But a post-treatment variable can be a mediator, a collider, or part of a more complex longitudinal process. The timing alone does not determine its role, yet timing is a critical warning that ordinary baseline confounder adjustment may no longer apply.

Suppose a job-support programme increases the number of interviews secured, and interviews affect employment. Conditioning on interview count when estimating the total effect of programme participation can remove the pathway through which the programme works. If the programme also affects participation in follow-up surveys, analysing only those who respond can select on a downstream consequence of treatment.

In clinical studies, restricting analysis to hospitalized patients may open selection paths if hospitalization is affected by both exposure and underlying disease. In product analytics, restricting to users who remain active can introduce a similar issue when activity depends jointly on treatment and latent preferences related to outcomes.

Those examples are not claims that particular published datasets contain collider bias. They illustrate how the graphical structure determines whether selection or adjustment is dangerous. The simulation provides an exact mechanism: conditioning on C creates an association between A and U because both contribute to C.

## Selection bias without a visible collider column

Collider bias can occur through how observations enter the analysis, even if the dataset does not contain a column named C. Suppose the observed sample is restricted to S=1, where S is influenced by treatment and an outcome cause. Analysing only S=1 conditions on the common effect.

The mechanism is analogous to A → S ← U → Y. Before selection, A may be independent of U. Among selected observations, A and U can become associated. The resulting comparison of outcomes between treatment groups can be distorted even when the original assignment was randomized.

This is one reason complete-case analysis can be problematic when outcome observation depends on factors related to treatment and prognosis. The selection mechanism can alter the comparison population after treatment assignment. A randomized study does not automatically preserve its identifying properties in a nonrandom subset of participants.

The direction of bias is not universally downward, as in the synthetic example. It depends on how treatment and the other cause affect selection, the outcome relationship and the distributions involved. The simulation chooses positive additive coefficients so that the sign can be derived analytically.

The practical lesson is to treat eligibility, attrition, conditioning events and exclusion rules as part of the causal design. They can determine which paths are open in the analysed population.

## Mediators, direct effects and changing the question

The mediation result deserves special attention because the adjusted coefficient near one is interpretable in a way the collider-adjusted coefficient is not. Under the structural equations, treatment has a direct effect of one and an indirect effect through M of one, producing total effect two.

If the target is the overall consequence of setting A, the unadjusted estimate is appropriate in this unconfounded model. If the target is a controlled direct effect under interventions fixing M, the adjusted outcome equation can provide that component under suitable assumptions.

The important issue is that those are not interchangeable targets. An employment service might want to know the full effect of offering a programme, including its effect through more job interviews. Alternatively, researchers studying mechanisms might ask how much effect remains under an intervention that fixes interview access. These are distinct policy questions.

Mediation analysis often involves subtler estimands such as natural direct and indirect effects. Those can require cross-world assumptions or other strong identifying restrictions, depending on the formulation, and are not automatically recovered by including a mediator in a regression. Treatment-induced mediator-outcome confounding creates particular difficulties.

The simple additive example is therefore an illustration of estimand change, not a complete mediation analysis. It shows why the analyst must identify the treatment pathways that belong to the question before conditioning on intermediate variables.

## What a DAG does not establish

An attractive DAG can give the impression that a causal problem has been solved by documentation. The opposite is closer to the truth: drawing the graph exposes the assumptions the analysis needs.

Two different causal DAGs can imply the same set of observational conditional independences. This is one reason observational data generally cannot orient every causal arrow uniquely. Graphical algorithms can exploit conditional independence relations under assumptions such as causal sufficiency and faithfulness, but their output remains conditional on those assumptions and the statistical quality of the tests.

Measurement error, latent variables, feedback, temporal aggregation and selection complicate the graphical picture. A static acyclic diagram may be appropriate for a single time point while failing to represent a dynamic process with repeated treatment and outcome measurements. Such problems often require time-indexed graphs, longitudinal potential outcomes or structural models designed for time-varying confounding.

The simulations here avoid those issues by defining all relationships explicitly and using independent disturbances. The graphs are correct because they are part of the experimental design, not because an algorithm discovered them.

In a real study, subject-matter expertise, timing information, prior evidence and the intervention protocol are needed to justify the graph. A graph drawn after seeing the result can become a rationalization rather than an independent basis for choosing the analysis.

The correct use of graphical reasoning is therefore transparent and critical. State the causal structure, explain the evidence for it, identify the implications for adjustment, and examine how alternative plausible structures might change the conclusion.

## Prediction and causal effects are different targets

An outcome prediction model seeks to estimate how Y varies with information available at prediction time. Adding a variable associated with Y may improve accuracy even when conditioning on that variable is harmful for a causal treatment effect.

The collider experiment illustrates the conflict. C contains information about U, a determinant of Y. A predictive model can use C to improve prediction of Y conditional on the observed A and C values. But the coefficient of A in that conditional model is not the intervention effect because fixing C changes the implied distribution of U as A varies.

The mediation experiment has the same conflict for another reason. M is predictive because it transmits treatment effects. Conditioning on it removes part of the very pathway the total-effect estimand includes.

This distinction is increasingly important when feature selection or automated modelling precedes a causal analysis. Statistical packages can select predictors according to penalized prediction error, variable importance or cross-validation. None of those criteria encodes whether a variable is a collider, mediator or pre-treatment confounder.

Machine learning remains useful for nuisance-function estimation in causal methods. It can estimate outcome regressions, propensity scores or conditional expectations within an identification framework. The framework must still specify what is appropriate to condition on. More predictive flexibility does not independently establish the required causal assumptions.

A well-predicted outcome is not the same as a well-identified intervention effect.

## The design behind the adjustment set

The adjustment decision should begin by stating the intervention, target population, timing and estimand. Is the objective a total effect, direct effect, effect among treated individuals or another contrast? A variable that is inappropriate for one estimand may be central to another.

Next, the analyst should reconstruct the data-generating and assignment mechanisms as well as available evidence allows. Baseline characteristics that affect treatment and outcome should be distinguished from consequences of treatment, selection mechanisms and measurements taken during follow-up. Unmeasured common causes should be acknowledged rather than assumed absent by default.

A DAG can then make candidate pathways explicit. The adjustment set should be assessed against the graph using an appropriate criterion. If several defensible adjustment sets exist, their implications for overlap, measurement quality and estimation precision can be compared.

This order is different from fitting a maximal regression and deleting insignificant terms. It requires the estimand and identification argument to precede fitting. The result may be a relatively simple model, but simplicity is not the objective in itself. The objective is a model whose conditioning operations correspond to the causal question.

After identification, model form, diagnostics, robustness checks and uncertainty estimation still matter. A valid adjustment set does not guarantee that a misspecified linear regression correctly estimates a nonlinear or heterogeneous causal contrast. Graphical reasoning and statistical estimation solve different pieces of the problem.

## Sensitivity to alternative causal structures

Because the graph is not observed directly, an empirical analysis should examine plausible alternatives. What if a variable believed to be a confounder is partly affected by treatment? What if a selection variable is jointly caused by treatment and prognosis? What if the measured baseline variable is only an imperfect proxy for a true common cause?

Those alternatives can change whether adjustment is beneficial or harmful. A robust coefficient across several regressions using different variables may be reassuring, but it cannot prove that the underlying graphical assumptions are correct. Robustness to arbitrary specification changes is not the same as sensitivity to scientifically plausible causal structures.

A useful exercise is to construct alternative diagrams consistent with available institutional knowledge, identify which adjustment sets each would permit, and examine whether the causal conclusion depends on contested arrows. This is not a universal algorithmic solution, but it makes disagreements about assumptions explicit.

Negative controls, experimental validation, temporal ordering and external evidence can constrain the plausible structures. They may falsify particular explanations without verifying every alternative. The quality of causal inference depends partly on how the remaining uncertainty is acknowledged.

The four simulated mechanisms are extreme in their clarity. They demonstrate distinct adjustment consequences without ambiguity. Real data will often support several plausible explanations, making explicit sensitivity analysis more important rather than less.

## Why sample size cannot rescue collider adjustment

The most important numerical comparison may be the collider at n=5,000. The unadjusted mean is 1.0009, very close to the true effect. The adjusted mean is 0.5001, extremely close to its analytically derived population coefficient of 0.5.

As sample size increases, the collider-adjusted estimate becomes more precise about 0.5. Its sampling variance falls, but the bias relative to the causal effect remains approximately −0.5. The RMSE therefore approaches 0.5 rather than zero.

This is not a failure of numerical stability or optimization. Ordinary least squares is correctly estimating the partial regression coefficient implied by the observed joint distribution. The mistake is using that coefficient as an estimator of the total intervention effect.

The same distinction appeared with unadjusted confounding in Part 1 and with propensity matching under hidden confounding in Part 3. A statistical procedure can reliably estimate a well-defined associational quantity that differs from the causal target. Causal assumptions determine whether the statistical and causal quantities coincide.

Adding more observations is useful for reducing random error. It cannot close a path opened by conditioning on a collider or restore a pathway removed by adjusting for a mediator.

A larger dataset can therefore make a causally inappropriate regression look more persuasive by narrowing its interval around the wrong quantity.

## What the benchmark cannot show

The experiment covers four linear structural systems with independent Gaussian disturbances, one treatment variable and one candidate adjustment variable per scenario. There are no treatment interactions, nonlinear effects, imperfect measurements, multiple confounders, or longitudinal feedback. All variables required for each simulated analysis are available, even when the variable's causal role makes adjusting for it inappropriate.

The collider coefficient of 0.5 is specific to the chosen variances and coefficients. Other structural parameters could produce stronger, weaker or differently signed bias. The mediator's equal direct and indirect effects are likewise deliberate simplifications, not a claim about typical mediation proportions.

The experiment does not attempt causal discovery from observational data. It assumes the generating graph and asks how different conditioning decisions perform under that graph. Real analysts face the harder problem of justifying the graph with incomplete evidence.

The program reports point estimates and Monte Carlo RMSE, not nominal confidence interval coverage or graphical learning performance. Adding standard errors would demonstrate that the biased estimators can produce narrow intervals, but the population derivations already establish why their causal bias persists.

The simulations are independently generated datasets conditional on one specified mechanism for each case. They cannot establish the frequency of collider bias, inappropriate mediation adjustment or omitted confounding in real published research. Their conclusions are methodological: adjustment has causal consequences determined by structure and estimand.

Finally, the experiments use ordinary regression on observed variables. More general causal questions can require methods such as inverse probability weighting, g-computation, sequential estimation, mediation identification formulas, instrumental variables or other designs. The point is not that one regression specification solves all DAGs, but that the appropriate method must follow from a defined causal question and identified estimand.

## What the investigation established

At n=5,000, adjusting for the confounder moved the average coefficient from 1.5002 to 1.0003, recovering a total causal effect of one. Adjusting for the collider moved the coefficient from a correct 1.0009 to 0.5001, creating bias where there had been no baseline confounding. Adjusting for the mediator moved the coefficient from 2.0004 to 1.0000, changing the interpretation from the total effect to the direct effect under the specified model. An irrelevant variable left the estimate essentially unchanged.

The RMSE comparisons confirm that these are population phenomena rather than small-sample accidents. Confounder adjustment reduced error as sample size grew; collider adjustment left causal RMSE near 0.5; mediator adjustment left total-effect RMSE near one; irrelevant adjustment changed almost nothing.

The same regression operation—conditioning on an additional variable—can remove an unwanted association, introduce one, or block a causal pathway. Whether the variable predicts outcomes or has a small p-value does not determine which consequence applies.

Causal diagrams provide a language for representing the assumptions that matter to that decision. They do not prove those assumptions. Their value is to make the proposed intervention, causal pathways, selection mechanisms and adjustment rationale explicit before an estimator is run.

The conclusion is not to avoid adjustment. It is to adjust for a reason. The relevant question is not how much of the outcome a model can explain, but whether the coefficient or contrast it estimates answers the intervention question the analysis set out to investigate.

## Code

The complete simulation is implemented in \`causal_inference_statistics/part5/experiment.py\` in the Medium-Blog repository. It uses Python 3.12 or later and NumPy, with a fixed random seed. Running it creates \`results.csv\`, containing mean estimates, between-replicate standard deviations, bias and RMSE for every structure, sample size and adjustment choice. No external data are required.

## The series

1. When Regression Gets the Causal Effect Wrong
2. Randomization, Confounding and the Value of Experimental Design
3. Matching and Propensity Scores: When Balanced Data Still Give the Wrong Causal Effect
4. Inverse Probability Weighting: When a Few Observations Control the Result
5. Directed Acyclic Graphs and the Cost of Controlling for Everything (this article)
6. Difference-in-Differences and the Parallel Trends Assumption

## References

Cinelli, C., Forney, A. and Pearl, J. (2024). A Crash Course in Good and Bad Controls. *Sociological Methods & Research*, 53(3), 1071–1104. https://doi.org/10.1177/00491241221099552

Elwert, F. and Winship, C. (2014). Endogenous Selection Bias: The Problem of Conditioning on a Collider Variable. *Annual Review of Sociology*, 40, 31–53. https://doi.org/10.1146/annurev-soc-071913-043455

Greenland, S., Pearl, J. and Robins, J. M. (1999). Causal Diagrams for Epidemiologic Research. *Epidemiology*, 10(1), 37–48. https://www.jstor.org/stable/3702180

Hernán, M. A. and Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC. https://miguelhernan.org/whatifbook

Hernán, M. A., Hernández-Díaz, S. and Robins, J. M. (2004). A Structural Approach to Selection Bias. *Epidemiology*, 15(5), 615–625. https://doi.org/10.1097/01.ede.0000135174.63482.43

Imbens, G. W. and Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press. https://doi.org/10.1017/CBO9781139025751

Morgan, S. L. and Winship, C. (2015). *Counterfactuals and Causal Inference: Methods and Principles for Social Research*, 2nd edition. Cambridge University Press. https://doi.org/10.1017/CBO9781107587991

Pearl, J. (1995). Causal Diagrams for Empirical Research. *Biometrika*, 82(4), 669–688. https://doi.org/10.1093/biomet/82.4.669

Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*, 2nd edition. Cambridge University Press. https://doi.org/10.1017/CBO9780511803161

Pearl, J. (2014). Interpretation and Identification of Causal Mediation. *Psychological Methods*, 19(4), 459–481. https://doi.org/10.1037/a0036434

Robins, J. M. (1986). A New Approach to Causal Inference in Mortality Studies with a Sustained Exposure Period—Application to Control of the Healthy Worker Survivor Effect. *Mathematical Modelling*, 7(9–12), 1393–1512. https://doi.org/10.1016/0270-0255(86)90088-6

Rubin, D. B. (1974). Estimating Causal Effects of Treatments in Randomized and Nonrandomized Studies. *Journal of Educational Psychology*, 66(5), 688–701. https://doi.org/10.1037/h0037350

Shpitser, I., VanderWeele, T. and Robins, J. M. (2010). On the Validity of Covariate Adjustment for Estimating Causal Effects. In *Proceedings of the Twenty-Sixth Conference on Uncertainty in Artificial Intelligence*, 527–536. AUAI Press. https://mlanthology.org/uai/2010/shpitser2010uai-validity/

Spirtes, P., Glymour, C. and Scheines, R. (2000). *Causation, Prediction, and Search*, 2nd edition. MIT Press.

VanderWeele, T. J. (2015). *Explanation in Causal Inference: Methods for Mediation and Interaction*. Oxford University Press.

VanderWeele, T. J. and Shpitser, I. (2013). On the Definition of a Confounder. *Annals of Statistics*, 41(1), 196–220. https://doi.org/10.1214/12-AOS1058

Westreich, D. and Greenland, S. (2013). The Table 2 Fallacy: Presenting and Interpreting Confounder and Modifier Coefficients. *American Journal of Epidemiology*, 177(4), 292–298. https://doi.org/10.1093/aje/kws412
