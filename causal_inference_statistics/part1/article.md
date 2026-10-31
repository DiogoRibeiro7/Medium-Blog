# When Regression Gets the Causal Effect Wrong

*Part 1 of Causal Inference in Statistics: omitted-variable bias, identification, randomization, and the difference between an accurate regression and a defensible causal estimate. An experiment with 1,000 independent datasets at each of three sample sizes. Python code provided.*

A regression model can estimate its population coefficient accurately and still answer the wrong scientific question. That distinction matters when the coefficient is used to predict what would happen if a policy changed, a treatment were administered, or a system were deliberately modified. Observing that people who receive more of something have different outcomes from those who receive less is not the same as observing the effect of giving them more.

This article starts with a controlled experiment. I generate data from a structural model in which the effect of an intervention is known exactly, fit three ordinary least squares regressions, and compare what the estimators recover. The first regression uses treatment alone. The second adjusts for the variable responsible for confounding. The third uses a randomly assigned treatment and does not need to adjust for that variable. Each comparison is repeated on 1,000 independent datasets at sample sizes of 100, 1,000, and 10,000.

The large-sample result is the one that motivates the investigation. The causal effect is 1. With 10,000 observations, unadjusted regression returns a mean coefficient of 2.0005 and a mean standard error of 0.0123. The adjusted regression returns 1.0001, and the regression on randomized treatment returns 0.9996. Increasing the sample size narrows the uncertainty around the unadjusted coefficient without bringing it any closer to the causal effect. The estimate becomes more precise as evidence for the causal claim becomes no stronger.

The mathematical reason is straightforward, but the consequences for applied statistics are not. Statistical inference can provide convincing evidence that a coefficient differs from zero while providing no justification for interpreting that coefficient as the consequence of an intervention. The problem cannot be solved by choosing a more elaborate optimizer, a more flexible predictive model, or a smaller p-value. It has to be solved, if it can be solved at all, by understanding how the treatment was assigned and how the observed variables were generated.

## The decision behind the coefficient

Suppose a public employment service is considering an intervention intended to reduce unemployment duration. Historical data contain information about who received additional job-search support, their characteristics, and how quickly they returned to work. A natural first analysis is to regress unemployment duration on a measure of support, controlling for some demographic variables. If the coefficient is negative, the service might be tempted to conclude that additional support shortens unemployment.

There are at least two interpretations of the result. People who received more support may have returned to work sooner *because of* the intervention. Alternatively, the service may have assigned support to people who were expected to remain unemployed for longer, in which case a positive association between support and duration can coexist with a genuinely beneficial treatment. Selection into treatment may produce the opposite pattern as well. The direction of the association depends on how treatment and outcome are generated, not merely on the direction of the intervention's effect.

This example is a motivating decision, not the empirical dataset analysed below. The experiment uses synthetic variables to expose the statistical mechanism without complications from censoring, changing economic conditions, or competing interventions. Those complications deserve investigation, but they would make it harder to distinguish the elementary identification problem from the difficulties of a particular application.

The policy question is what would happen to unemployment duration if the service deliberately changed a person's support. The observational question is how unemployment duration differs among people who happened to receive different amounts of support. Both questions can be meaningful. Only the first is causal. A regression coefficient is not automatically an answer to it, even if the regression is a well-specified description of the conditional mean.

A similar distinction appears in medicine, education, economics, and product experimentation. Patients receiving stronger medication may have worse outcomes because they were initially sicker. Students attending extra classes may perform worse because the programme targets those already struggling. Customers exposed to a retention campaign may churn more often because the campaign was deployed to customers considered at risk. These examples do not establish the direction of any particular effect. They explain why a comparison between observed treatment levels and observed outcomes is not enough.

We therefore need to specify the quantity we want before selecting the estimator. In the experiment, the objective is the expected change in outcome when treatment is increased by one unit. The estimator's task is not simply to provide the best fitting slope, but to estimate that particular intervention effect.

## A model in which the truth is known

Let U be a characteristic that affects both treatment X and outcome Y. For each observation, generate three independent standard normal variables, U, ε_X, and ε_Y, and define

\[
X=U+\varepsilon_X,
\]

\[
Y=X+2U+\varepsilon_Y.
\]

The variables are independent across observations. All three underlying random terms have mean zero and variance one. The coefficient of X in the structural outcome equation is 1. The coefficient of U in the outcome equation is 2, and the coefficient of U in the treatment equation is 1. Consequently, an individual with a larger value of U tends to receive more treatment and also has a larger outcome, even after holding the treatment itself fixed.

The important detail is that these are structural equations defining how observations are generated, rather than fitted regressions offered after the data have been observed. Under an intervention that sets X to x, the outcome equation becomes

\[
Y(x)=x+2U+\varepsilon_Y.
\]

Setting treatment instead to x+1 gives

\[
Y(x+1)=x+1+2U+\varepsilon_Y.
\]

Subtracting the two potential outcomes gives

\[
Y(x+1)-Y(x)=1.
\]

The effect is constant across individuals and independent of the original treatment level. The average causal effect of increasing X by one unit is therefore exactly 1. In this experiment, there is no ambiguity about the answer against which the estimators will be judged.

There is nevertheless a distinction between the effect built into the model and the slope visible in the observed data. Treatment is not independent of U. People with larger U values tend to have larger X values, and their outcomes are elevated both by X and by U directly. Fitting the outcome against X without recognising U assigns some of the direct contribution of U to the apparent relationship between X and Y.

The choice of a continuous treatment is deliberate. It allows the same mathematical model to illustrate the distinction between a change in a conditional expectation and the response to an intervention without adding complications from binary treatment probabilities or separate treated and control groups. Later articles can introduce those distinctions when they are central to the investigation.

There is also no machine learning model in this first benchmark. Ordinary least squares is enough to establish the problem. Replacing the linear regression with an elaborate prediction algorithm would change the fitting procedure but would not supply the missing causal assumptions. Starting with a model that can be analysed exactly makes it possible to compare mathematical expectations with simulated estimates instead of relying only on graphical intuition.

## What unadjusted ordinary least squares estimates

When Y is regressed on X with an intercept, the population regression slope is

\[
\beta_{\mathrm{OLS}}=
\frac{\operatorname{Cov}(X,Y)}{\operatorname{Var}(X)}.
\]

Substituting the structural outcome equation,

\[
\operatorname{Cov}(X,Y)
=\operatorname{Cov}(X,X+2U+\varepsilon_Y).
\]

Because ε_Y is independent of X,

\[
\operatorname{Cov}(X,Y)
=\operatorname{Var}(X)+2\operatorname{Cov}(X,U).
\]

The population slope is consequently

\[
\beta_{\mathrm{OLS}}
=1+2\frac{\operatorname{Cov}(X,U)}{\operatorname{Var}(X)}.
\]

This is the familiar omitted-variable-bias decomposition. In the present model, X=U+ε_X, so Cov(X,U)=Var(U)=1 and Var(X)=Var(U)+Var(ε_X)=2. The second term is 2(1/2)=1, giving

\[
\beta_{\mathrm{OLS}}=2.
\]

The unadjusted coefficient equals the true causal effect plus an additional unit generated by confounding. The two components are conceptually different even though the regression combines them into a single estimated number.

The direction and magnitude of the bias depend on the signs and sizes of the relationships. Here, U increases treatment and also increases outcome, so the bias is positive. If U increased treatment but reduced outcome, it could make the unadjusted slope smaller than the causal effect or even reverse its sign. A causal intervention could be beneficial while the observed association appeared harmful, or vice versa. The sign of a regression coefficient is not protected against these changes by the sample size.

The central implication is asymptotic. Under the regularity conditions satisfied by this data-generating process, the sample regression slope converges in probability to 2. It does not converge to 1. The estimator is consistent for its own associational target, but inconsistent for the causal quantity we asked it to estimate. Saying that ordinary least squares is inconsistent without naming the target would be imprecise: it behaves exactly as expected for the statistical parameter it estimates.

This distinction is useful beyond the simplified equations. Applied researchers often describe a coefficient as becoming more reliable because the standard error is decreasing or the confidence interval is narrowing. Those statements are defensible when reliability means precision for a specified statistical parameter. They are not sufficient when reliability is taken to mean that the coefficient describes an intervention. The latter interpretation depends on causal identification, and no amount of repeated sampling from the same confounded distribution supplies it.

There is another way to see the problem. Taking conditional expectations in the structural equation gives

\[
E[Y\mid X=x]=x+2E[U\mid X=x].
\]

Because U and X are jointly Gaussian and Cov(U,X)/Var(X)=1/2,

\[
E[U\mid X=x]=x/2.
\]

Therefore,

\[
E[Y\mid X=x]=2x.
\]

But intervening to set X=x leaves the marginal distribution of U unchanged, so

\[
E[Y(x)]=x.
\]

The derivative of the observed conditional mean is 2, while the derivative of the interventional mean is 1. Both equations are correct. They answer different questions. The regression is not malfunctioning; the interpretation that turns its associational slope into a causal effect is.

## Three estimators, three routes through the problem

The first estimator fits a regression of Y on X, with an intercept. It does not include U. Its population target is the associational slope derived above, which equals 2. The fact that this estimator is deliberately misspecified for the causal question does not make it an unrealistic baseline. Analysts often use unadjusted comparisons when they have not recognised the treatment assignment mechanism, cannot measure the common causes, or assume that a sufficiently flexible model will compensate for the missing information.

The second estimator adds U to the regression:

\[
Y=\beta_0+\tau X+\gamma U+\varepsilon.
\]

In this experiment, the equation matches the structural outcome relationship. Because ε_Y is independent of X and U, the population coefficient of X is 1. There is enough residual variation in X after conditioning on U to estimate that coefficient: ε_X gives treatment variability that is not determined by the confounder. Under the model's assumptions, adjustment separates the variation in outcome attributable to treatment from the variation attributable directly to U.

The third approach changes the data-generating process. Generate X_R independently of U and ε_Y, again from a standard normal distribution, and define

\[
Y_R=X_R+2U+\varepsilon_Y.
\]

The intervention effect is still 1. What has changed is the treatment assignment mechanism: X_R and U are independent. An unadjusted regression of Y_R on X_R therefore has population slope 1. This is a randomized analogue of an experiment in which the investigator controls treatment rather than observing the assignment decisions of a system influenced by confounding characteristics.

This randomized comparison is not the same as adjusting for a confounder in observational data. Adjustment relies on the relevant variable being observed and the conditional outcome relationship being correctly represented. Random assignment breaks the dependence between treatment and the confounder through the design. The first strategy uses assumptions about measured variables and their relationships; the second uses an assignment mechanism that makes some of those assumptions unnecessary for identification.

Both have limitations in practice. Random assignment can be undermined by noncompliance, differential attrition, interference, or incorrectly defined interventions. Adjustment fails when important common causes are unobserved or measured poorly, and a fitted linear regression can be wrong even when every relevant variable is included. The present experiment deliberately removes those difficulties to isolate the elementary comparison. It shows what the estimators do when their identifying conditions hold, not a ranking that applies to all possible observational and randomized studies.

For all three estimators I use ordinary least squares with an intercept and calculate the conventional homoskedastic standard error from the residual mean square and the inverse of the design cross-product. The simulated errors satisfy the conditions needed for this calculation in each fitted model. A nominal 95% interval is formed using 1.96 times that standard error. A Student t critical value would be more exact in small samples, but the normal approximation is useful here because the same calculation can be applied consistently across the three sample sizes and its empirical coverage can be checked directly.

## How the experiment was run

The simulation uses 1,000 independent datasets for each sample size: 100, 1,000 and 10,000 observations. The random number generator is NumPy's default generator with seed 20261016. For each replicate, the observational model generates U, ε_X and ε_Y. It constructs X and Y, fits the unadjusted regression, and then fits the regression including U. A separate randomized treatment X_R is drawn independently and used to construct Y_R with the same structural coefficient and the same U and ε_Y terms.

The same underlying outcome disturbances are reused within each replicate's observational and randomized cases. This helps keep the comparisons tied to a common set of potential outcomes rather than accidentally making one scenario intrinsically easier because it received a different distribution of disturbances. It does not make the methods use identical treatment observations, nor does it imply that the three estimates are independent within a replicate. Independence is assumed across the independently generated datasets, not between methods applied to the same dataset.

The program records each estimate and its model-based standard error. Across replicates, it then reports the mean estimate, mean error relative to the true effect, root mean squared error, mean reported standard error and fraction of nominal 95% intervals containing 1. These quantities answer distinct questions. Mean error measures systematic departure from the causal truth; root mean squared error combines bias and variation; standard errors describe the uncertainty each fitted model reports; and coverage asks how often the reported interval actually contains the causal effect.

Coverage needs particular care. A conventional regression interval is constructed for the regression coefficient under the fitted statistical model. In the confounded case, that regression coefficient is 2. Our coverage calculation instead asks whether the interval contains the causal effect 1. A zero result therefore does not show that least squares confidence intervals are intrinsically invalid. It shows that intervals for an associational parameter need not cover the causal parameter an analyst might mistakenly attach to them.

There is no external dataset to download, no hyperparameter search and no fitted classifier. All numbers are generated directly from the published equations and can be reproduced with the accompanying script. The seed makes the run repeatable; it does not transform a single chosen simulation into proof about other data-generating mechanisms. The mathematical derivations specify which behaviour should persist beyond this particular Monte Carlo run.

## The estimates

The mean treatment coefficient across the 1,000 replicated datasets is as follows.

| Observations | Unadjusted regression | Adjusted regression | Randomized treatment |
| ---: | ---: | ---: | ---: |
| 100 | 1.9948 | 0.9966 | 1.0017 |
| 1,000 | 2.0015 | 1.0004 | 0.9984 |
| 10,000 | 2.0005 | 1.0001 | 0.9996 |

The exact intervention effect is 1 in every scenario. At n=100, the naive estimate averages approximately 1.995. By n=10,000 it averages approximately 2.001, almost exactly the population regression coefficient derived above. Its failure to approach the causal target is not a numerical accident: 2 is where it should converge.

The adjusted regression remains near 1 at all sample sizes. The randomized estimator does likewise. The numerical differences between 0.9966, 1.0004 and 1.0001 are small relative to the sampling variation at their respective sample sizes and are not evidence of a systematic change in the effect. The treatment effect does not change when n changes; only the precision with which we estimate it changes.

The separation between the columns is easiest to appreciate at the largest sample size. The unadjusted coefficient differs from the causal effect by almost exactly one full outcome unit, despite using 10,000 observations. The adjusted coefficient differs by about one ten-thousandth of a unit, and the randomized coefficient by less than half a thousandth. These are Monte Carlo mean estimates under a specified model, not estimates of real-world treatment benefits. Their role is to expose the distinction between the estimand and the estimator.

A small sample can make the naive regression appear less problematic than it is. Its estimates vary more from replicate to replicate when n=100, so a particular dataset could produce a value substantially below or above 2. Sampling noise can temporarily obscure the bias. Increasing the sample size removes some of that noise and makes the estimator's wrong target more visible. In this sense, a larger dataset can expose the failure more clearly even while giving a misleading impression of certainty to anyone who assumes the regression coefficient is the effect of an intervention.

It would also be misleading to call adjusted regression universally superior simply because its mean is closer to 1 here. It is superior for this constructed causal estimand because the confounder is fully observed and the specified adjustment model is correct. If U were measured with error, if the causal relationship were nonlinear, or if a variable controlled for in the regression were a collider rather than a common cause, adjustment could behave very differently. Those variations are not reflected in the three columns above.

## Precision, error and confidence intervals

The comparison becomes more informative when the mean estimates are considered alongside their uncertainty. The following quantities are means or Monte Carlo summaries across the 1,000 replicates at each sample size.

| Observations | Method | RMSE for causal effect | Mean fitted SE | 95% causal coverage |
| ---: | --- | ---: | ---: | ---: |
| 100 | Unadjusted | 1.0031 | 0.1239 | 0.000 |
| 100 | Adjusted | 0.1039 | 0.1012 | 0.943 |
| 100 | Randomized | 0.2278 | 0.2261 | 0.946 |
| 1,000 | Unadjusted | 1.0023 | 0.0388 | 0.000 |
| 1,000 | Adjusted | 0.0321 | 0.0317 | 0.942 |
| 1,000 | Randomized | 0.0685 | 0.0709 | 0.960 |
| 10,000 | Unadjusted | 1.0006 | 0.0123 | 0.000 |
| 10,000 | Adjusted | 0.0098 | 0.0100 | 0.951 |
| 10,000 | Randomized | 0.0229 | 0.0224 | 0.953 |

RMSE is calculated relative to the true causal effect, not relative to whichever coefficient a regression happens to estimate. For the unadjusted model, RMSE approaches 1 because the squared bias does not vanish. Its standard error, however, shrinks from approximately 0.124 to 0.012 as the sample size rises from 100 to 10,000. The model increasingly concentrates around an answer that is wrong for the intervention question.

Coverage makes the same point in a different form. Across 1,000 datasets at each sample size, none of the nominal 95% intervals from unadjusted regression contained 1. The adjusted and randomized estimates have coverage between approximately 94% and 96%, consistent with the intended level given the approximations and Monte Carlo variability. The contrast does not require judging individual p-values or deciding which model has the most favourable significance result.

At n=10,000, consider the approximate interval obtained by centring on the mean unadjusted coefficient and extending two mean standard errors in each direction: roughly [1.976, 2.025]. This is an illustrative interval using across-replicate summaries, not an interval fitted to one specific dataset. Its width conveys why the true effect of 1 is far outside the typical unadjusted intervals. A result can be highly precise and causally incorrect.

The adjusted regression has a smaller standard error than the unadjusted regression here. The reason is not that adding regressors always improves precision for the coefficient of interest. Conditional on X, U explains much of the variation in Y, and the remaining structural disturbance has variance 1. The resulting reduction in residual variation outweighs the cost of estimating an extra parameter and the collinearity between X and U. Other structures can produce the opposite trade-off.

The randomized unadjusted regression has a larger standard error than the adjusted observational regression. Its residual contains 2U+ε_Y, with variance 5, whereas the adjusted observational regression removes the U component and retains residual variance 1. Since randomized X_R has variance 1, the large-sample standard error of its unadjusted slope is approximately sqrt(5/n). The correctly adjusted observational regression has a leading standard error of approximately 1/sqrt(n), since the outcome disturbance and the variation in X remaining after conditioning on U both have variance 1. These approximations agree with the Monte Carlo results. The distinction highlights that identifying a causal effect and estimating it efficiently are separate problems.

For this particular design the adjusted estimator is more precise. That is a statement about the chosen data-generating process and estimator, not an argument that observational adjustment is generally preferable to experiments. Randomization supplies protection against confounding through the assignment mechanism. Statistical efficiency can often be improved further by incorporating appropriate baseline covariates into the analysis of randomized data. How that adjustment is performed matters, as work by Freedman (2008) and Lin (2013) makes clear.

## Bias does not shrink like standard error

The expression for mean squared error separates squared bias and variance:

\[
\operatorname{MSE}(\widehat{\tau})
=\operatorname{Bias}(\widehat{\tau})^2+
\operatorname{Var}(\widehat{\tau}).
\]

For the naive regression, the asymptotic bias relative to the causal effect is 1. Its variance decreases as n increases, but squared bias remains approximately 1. The RMSE therefore approaches 1 instead of zero. The Monte Carlo values, 1.0031, 1.0023 and 1.0006, illustrate exactly that behaviour.

For the adjusted and randomized estimators, the asymptotic bias is zero under the constructed assumptions. Their RMSE decreases as the sample size grows because the remaining error is principally sampling variation. The fall from approximately 0.104 to 0.0098 for adjusted regression corresponds to the familiar square-root-of-sample-size improvement. The unadjusted regression also gains that improvement in its estimation of the *wrong* slope, but the causal RMSE barely moves.

This offers a practical reason not to treat sample size as a generic defence against model assumptions. Sampling error and systematic bias do not have the same response to adding more observations. Repeating a biased measurement many times improves our estimate of the biased target. Collecting a million observations from a confounded assignment process need not reproduce the causal information that a much smaller randomized experiment could provide.

That comparison should not be exaggerated. A small randomized trial may be imprecise, unrepresentative, contaminated by noncompliance, or unable to address a long-term outcome. Large observational studies can be essential when randomization is unethical or impossible, and designs that exploit measured confounders, natural experiments, or longitudinal structure can sometimes yield defensible effects. The point is narrower: sample size addresses precision conditional on a design. It does not replace the argument that identifies the causal estimand.

A related mistake is to interpret a highly significant test of the null regression slope as evidence for a causal effect. In the confounded model, there is a genuine and strong statistical association. A test of whether the unadjusted population slope equals zero should reject with high probability in a large sample because the slope equals 2. That rejection supplies information about association. It does not distinguish whether the association is generated by intervention effects, common causes, selection mechanisms, or some combination.

## Identification comes before estimation

The word *identification* is often used as though it were another estimation method, but the distinction is more fundamental. A causal quantity is identified when it can be expressed uniquely in terms of the distribution of observed variables under a stated set of assumptions. Estimation is the subsequent attempt to calculate that identified quantity from a finite sample.

In the present structural model, the intervention effect is known by construction. If an analyst saw only the joint distribution of X and Y, however, the regression slope alone would not reveal whether the population slope of 2 arose from a treatment effect of 2 without confounding or from a treatment effect of 1 plus confounding. Different causal structures can imply the same observed conditional relationship while producing different responses to an intervention.

The adjusted model adds information because U is observed and the assumptions establish that conditioning on it removes the relevant common cause. Under consistency, suitable conditional exchangeability, and adequate support for treatment variation at the levels of U considered, the causal mean can be recovered by averaging the conditional outcome relationship over the distribution of U. In the continuous model with the correct structural equation, this gives the intervention mean x and hence the effect 1.

The randomized design obtains identification from a different assumption: treatment assignment is independent of the potential outcomes. In that case, the observed outcome conditional on assignment represents what would happen under that intervention, subject to appropriate consistency and treatment implementation assumptions. The causal difference can be estimated by a corresponding comparison of observed outcomes.

Neither strategy is completely assumption-free. Even randomization does not define what a treatment actually is, guarantee that it is delivered as intended, or prevent one person's treatment from changing another person's outcome. In the current simulation all those concerns are deliberately absent. The value of making assumptions explicit is that it allows the conclusion to be stated at the correct level: under the defined model, the intervention effect is identified and the adjusted or randomized estimator recovers it.

This is why identification should be discussed before model selection. A data scientist might try random forests, gradient boosting, neural networks, generalized additive models, or Bayesian regression, then choose the model with the lowest prediction error. Those methods may estimate E[Y|X] more accurately than a simple linear regression. They do not by themselves transform E[Y|X] into E[Y(x)]. If the problem is that treatment and unobserved causes are dependent, a better estimate of the observed conditional mean may reproduce the confounded relationship even more precisely.

The correct statistical target can be an excellent estimate of the wrong causal quantity. That is the failure illustrated by the unadjusted regression, and it does not depend on the simplicity of ordinary least squares.

## Prediction is not intervention

Prediction asks what outcome to expect for an individual with a specified set of observed characteristics, under the observed environment. Causal inference asks what the outcome would be under a different intervention. A model used for prediction can legitimately exploit variables that do not have causal effects on the outcome, as long as the relationships remain stable in the deployment population and the information is available at prediction time.

For instance, a variable describing whether a worker filed a benefit claim might help predict unemployment duration even if the act of filing did not itself prolong unemployment. That variable could encode information about circumstances and expectations that are correlated with the eventual duration. Its predictive value would not establish the effect of encouraging or discouraging claims.

This is also why the correct predictive model need not be the correct treatment allocation rule. If an employment service wants to support the workers most likely to remain unemployed, it needs a risk estimate. If it wants to allocate support where the intervention will produce the largest improvement, it needs to estimate how outcomes differ under alternative support levels. Those are different estimands. A person at high baseline risk does not necessarily have the largest treatment effect.

The simulated regression sharpens the distinction. Because the observational conditional mean is E[Y|X=x]=2x, a slope of 2 correctly describes the expected outcome among observations with treatment x. If future observations are generated by the same confounded process, the fitted relationship may predict well. But if a policy forcibly changes treatment without changing U, the expected outcome changes at slope 1. The environment under an intervention is not the same as the environment described by conditioning on a naturally occurring treatment value.

Causal inference therefore does not make prediction obsolete. It places prediction within a different decision framework. Prediction can be sufficient when the question is what is likely to happen without deliberate changes to the system. Once the model is used to justify a change, the consequences of that change require a causal argument.

## Why controlling for U works here

Regression adjustment may look mechanical: add U to the design matrix and read the coefficient of X. The mathematical justification is more specific. The structural error ε_Y has mean zero conditional on X and U because it was generated independently of them. Consequently,

\[
E[Y\mid X=x,U=u]=x+2u.
\]

Taking the difference between two treatment levels while holding U fixed cancels the direct contribution 2u, leaving the structural effect. That calculation works because U is a pre-treatment common cause, the conditional relationship is correctly modelled, and X varies within levels of U. The last property is essential. If X were completely determined by U, we could not distinguish their contributions within the observational data.

The usual omitted-variable-bias formula also shows why adjustment is more than increasing predictive power. Controlling for U does not merely reduce the unexplained variance. It changes which coefficient of X is estimated: from the marginal association, which combines two pathways, to the conditional association that corresponds to the intervention effect under the assumptions. Statistical precision may improve at the same time, but the identification argument is doing the causal work.

It is tempting to generalize this success into a rule that more adjustment variables must be better. That rule is false. A variable affected by treatment, a common effect of treatment and outcome causes, or a variable measured after the intervention may have a very different role from U. Conditioning on such a variable can create a noncausal path, remove part of the effect, or select an unrepresentative subset of observations. Even among genuine pre-treatment covariates, unnecessary adjustment can affect variance and make a fitted model less stable.

The next articles will investigate those failures rather than treating the adjustment set as something that can be determined by a statistical feature-selection routine. Variables belong in causal models because of their role in the data-generating process, not simply because they improve cross-validated prediction. A causal diagram can make some of those roles explicit, but it must itself be justified by substantive knowledge. Drawing arrows does not turn assumptions into data.

This first benchmark is deliberately favourable to regression adjustment. The single confounder is observed without error, the outcome equation is linear with additive effects, and the investigator knows its functional form. The fact that adjustment works under those conditions is reassuring but not surprising. The harder problems concern which assumptions can be defended when the causal system is not known in advance.

## What randomization establishes, and what it does not

The randomized comparison breaks the relation between treatment and U by generating treatment independently. The population covariance between randomized treatment and the direct contribution of U to the outcome is zero. In the population regression of Y_R on X_R, the omitted term 2U therefore does not bias the slope. It still contributes to its sampling variability, as reflected in the larger standard errors in the table.

Randomization gives a design-based explanation for why an observed treatment comparison can estimate a causal effect. We do not need the outcome to be exactly linear in the treatment or to include every prognostic variable in the regression in order for an appropriate randomized comparison to be causally interpretable. The precise estimator and inferential procedure still matter, especially for heterogeneous treatment effects, finite samples, or unequal assignment probabilities.

In this experiment, treatment is continuous and generated independently for each observation rather than allocated to two discrete groups. This is enough for the elementary structural demonstration, but it is not a complete model of a clinical trial or policy evaluation. In a real randomized study, the assignment procedure, intervention protocol, treatment received, analysis population and measurement process all have to be specified. Randomization at assignment does not imply that every analysis of the resulting dataset inherits the same protection.

Suppose some participants refuse treatment, or leave the study for reasons related to their potential outcomes. An analysis restricted to those who complied or remained observed may reintroduce selection. Similarly, when treatment given to one person affects another person's outcome, the simple independent-potential-outcomes formulation is insufficient. A valid design has to address those possibilities rather than relying on the word *randomized* as a guarantee.

Adjustment within randomized studies can improve precision, but it requires care. Freedman (2008) showed that conventional regression adjustment can have undesirable properties under randomization-based inference, including effects on precision and the validity of familiar variance estimates. Lin (2013) established conditions under which fully interacted adjustment and robust standard errors provide favourable large-sample properties. These contributions reinforce rather than contradict the importance of design: randomization identifies a contrast, while the analysis determines how it is estimated and how uncertainty is reported.

The simulated randomized regression here remains deliberately simple. It demonstrates that independent treatment assignment removes the particular confounding path in the structural equation. It does not attempt to benchmark modern experimental estimators or provide a general efficiency theorem.

## The dependence on assumptions

The strongest limitation of the experiment is also its greatest advantage: I know the data-generating process. The exact treatment effect, the confounder, the error distribution, and the assignment mechanism are specified before analysis. An observational researcher rarely has that information. They have measurements, institutional knowledge, and assumptions about the process, but not direct access to alternative outcomes for the same person under every possible intervention.

In real data, there may be several unmeasured common causes of treatment and outcome. A measured variable may be only an imperfect proxy for a confounder. The effect of treatment may vary by individual or treatment level. The assignment process may change over time. Outcome measurement may depend on treatment. These departures can require different estimands, adjustment strategies, uncertainty calculations and sensitivity analyses.

Linearity is not required for causal inference, but it is doing useful analytical work here. It gives a constant causal effect and a closed-form expression for omitted-variable bias. Under nonlinear or interactive effects, there may be no single constant effect corresponding to the slope of a regression. Even an otherwise correctly adjusted linear model can then average heterogeneous effects in a way that depends on the treatment distribution and covariates.

Normality also simplifies the calculation of the conditional expectation E[U|X=x] and makes the variance assumptions easy to inspect. The conceptual distinction between conditioning and intervention does not depend on normally distributed variables. Many causal problems involve binary outcomes, counts, time-to-event outcomes, non-Gaussian disturbances and nonlinear assignment mechanisms. The same identification question remains, although its mathematical expression changes.

The three sample sizes investigate what happens when observations are added under the *same* data-generating process. They do not investigate a growing population whose composition changes with n or a data source that becomes more representative over time. In applied work, larger datasets may contain additional groups and different measurement mechanisms, and the target population itself may be uncertain. Nothing in this experiment justifies assuming that increasing the number of records always preserves the same estimand.

Repeated simulation is not a substitute for empirical validation either. The 1,000 replicates quantify sampling behaviour conditional on the equations, the chosen parameters and the pseudorandom generator. They cannot tell us which causal assumptions hold in an actual employment, education or medical dataset. A carefully designed simulation tells us how a method behaves when we know the truth; external evidence and subject-matter knowledge are needed to assess whether that mechanism resembles a real problem.

## How I would evaluate a causal analysis built on observational data

The practical question is what to do when no randomized experiment is available and the analyst cannot verify the full causal structure. The answer is not to abandon statistical analysis. It is to distinguish what is assumed from what is measured, and to make the consequences of those assumptions visible.

Start by writing the intervention and the population-level estimand explicitly. What does it mean to increase treatment? Is it a decision a policymaker could actually implement? Does the treatment have multiple versions with different consequences? Which outcome, time horizon and population does the claim concern? These questions are often more consequential than which regression library is used.

Next, describe how treatment is assigned. Identify variables measured before treatment that plausibly affect both assignment and outcome. Separate those from variables that could be affected by treatment or jointly influenced by treatment and other outcome causes. Decide which variables should be adjusted for before examining which fitted specification produces the most favourable effect estimate.

Then ask whether the proposed estimand is identifiable under the design. If treatment assignment is assumed conditionally independent of potential outcomes, what supports that assumption? If overlap is required, do comparable treated and untreated observations exist across the relevant covariate distribution? If an instrumental-variable or quasi-experimental design is being used, which exclusion or continuity assumptions are necessary? Those conditions should not be hidden behind model coefficients.

Only after that should estimation dominate the discussion. Explain why the estimator targets the identified quantity, calculate appropriate uncertainty, and check the stability of the result to defensible alternatives. A regression coefficient may be one component of this analysis, but it should not carry the entire causal argument simply because its standard error is small.

Finally, distinguish robustness from proof. Alternative covariate sets, placebo outcomes, negative controls, sensitivity analyses and alternative designs may reveal weaknesses or show that a result persists under several plausible specifications. They cannot verify every identifying assumption. The result should be described according to the strength of the evidence actually obtained, not according to the confidence suggested by its numerical precision.

These steps are not a new estimator or a universal checklist that guarantees causal validity. They are a way of organising scientific reasoning so that design, identification and estimation cannot be silently treated as the same operation.

## Reproducibility and interpretation

The accompanying Python program implements the experiment using NumPy and no dedicated causal-inference package. For each generated dataset it builds the appropriate regression design matrix, solves least squares, computes residual variance and derives the coefficient's conventional standard error. The program then aggregates the results across replicates and prints one row for each model and sample size.

The output includes mean estimate, bias, root mean squared error, average standard error and coverage. The manuscript reports the observed output from the specified seed rather than values chosen because they illustrate the argument conveniently. Running a different random seed would change individual Monte Carlo summaries slightly. It would not change the population results: a causal effect of 1, an unadjusted observational slope of 2, and causal slopes of 1 for the correctly adjusted and randomized comparisons.

There are further figures worth producing for publication. One should compare the empirical distributions of all three estimates at each n, with vertical lines at the causal effect and the associational population slope. A second should show confidence-interval coverage as sample size increases. A third could plot the observed regression conditional mean against the interventional mean. Each figure serves a distinct purpose: variability, inferential coverage, and the interpretation of the quantity being estimated.

Those figures should be generated from the simulation outputs, not drawn as decorative approximations. The code currently provides the numerical benchmark and tables; the graphical presentation is a separate reproducibility improvement. An article built around experiments should not ask the reader to accept a diagram as evidence when the underlying numbers can be computed.

Reproducibility also includes reporting what was not tuned. There was no search over data-generating parameters, no selection of the sample size that made the result most dramatic, and no attempt to optimise regression settings. The chosen parameters make confounding large enough to be visible and simple enough to derive analytically. A systematic follow-up can vary the strength and direction of the confounder, treatment noise, effect heterogeneity, and measurement error. Those experiments would assess the boundaries of this article's conclusions rather than merely restating them.

## Limits

The benchmark covers one structural model, one confounder and three estimation strategies. It assumes independently sampled observations, a constant additive effect, correctly measured variables and independent Gaussian disturbances. The treatment and outcome equations are selected to make the causal and associational slopes exactly different by one unit. The experiment therefore demonstrates a particular mechanism of bias rather than estimating how common or severe omitted-variable bias is in real applications.

The adjusted regression is given an advantage that observational researchers generally lack: direct access to the complete confounder and a correct functional form. Its good performance should not be interpreted as evidence that adjustment is sufficient whenever covariates are available. The randomized case likewise abstracts from compliance, attrition, treatment interference, implementation variation and the design choices that determine what a real trial identifies.

The use of conventional normal-approximation intervals is reasonable for these models, but it does not address uncertainty in the causal assumptions. Reported standard errors quantify variation attributable to sampling under the chosen fitted statistical model. They do not account for the possibility that an omitted variable exists, that treatment is incorrectly defined, or that the assumed adjustment set is wrong. A narrow interval should never be taken to summarize all uncertainty surrounding a causal claim.

The primary quantities reported are averages over simulated datasets, not single-study estimates. The coverage calculations use the known causal truth, a luxury unavailable in most observational applications. Their role is diagnostic: they show what inferential claims would mean if the unadjusted regression coefficient were presented as an estimate of the intervention effect.

Most importantly, the result does not imply that randomized evidence is automatically decisive or observational evidence automatically invalid. It establishes a narrower logical point. A regression coefficient can be estimated accurately without identifying a causal effect, and increasing sample size cannot repair that missing identification. Evaluating actual evidence requires a design-specific analysis of assumptions and alternatives.

## What the experiment established

Across 1,000 independent datasets at each sample size, the observational regression that omitted the confounder estimated a coefficient close to 2, despite a true causal effect of 1. Its causal RMSE remained approximately 1 as sample size increased from 100 to 10,000. Its mean reported standard error fell by a factor of about ten, and none of its nominal 95% confidence intervals contained the causal effect.

Correct adjustment and independent treatment assignment recovered a mean effect close to 1. Their interval coverage remained approximately 95%, and their causal RMSE decreased as the sample size increased. The adjusted regression was more precise than the unadjusted randomized regression in this particular model, illustrating that identification and efficiency are related but distinct concerns.

The central result is not that regression is unreliable. The unadjusted regression reliably estimates the observational relationship it is designed to estimate. The failure occurs when that relationship is interpreted as the effect of changing treatment without justifying the connection. The model is statistically accurate about one quantity and scientifically unsuitable for answering another.

A larger sample can make that mistake more convincing because it reduces the uncertainty that is easiest to report while leaving the uncertainty about causal identification untouched. A precise coefficient is not a substitute for a defensible intervention, an appropriate design and explicit assumptions. The next article will examine the identification advantage and statistical limitations of random assignment in greater detail, before turning to adjustment strategies that can also introduce bias.

## Code

All numerical tables are reproduced by \`experiment.py\` in \`causal_inference_statistics/part1/\` of the Medium-Blog repository. It requires NumPy and Python 3.12 or later. It uses a fixed seed and prints mean estimates, bias, RMSE, mean standard errors and coverage for all nine method-by-sample-size combinations. The structural equations, randomization mechanism and estimation routine are explicit in the source.

## The series

1. When Regression Gets the Causal Effect Wrong (this article)
2. Randomization, Confounding and the Value of Experimental Design
3. Matching and Propensity Scores
4. Inverse Probability Weighting
5. Directed Acyclic Graphs and the Cost of Controlling for Everything
6. Difference-in-Differences and the Parallel Trends Assumption

## References

Angrist, J. D. and Pischke, J.-S. (2009). *Mostly Harmless Econometrics: An Empiricist's Companion*. Princeton University Press. https://press.princeton.edu/books/paperback/9780691120355/mostly-harmless-econometrics

Cochran, W. G. (1965). The Planning of Observational Studies of Human Populations. *Journal of the Royal Statistical Society: Series A*, 128(2), 234–255. https://doi.org/10.2307/2344179

Freedman, D. A. (2008). On Regression Adjustments to Experimental Data. *Advances in Applied Mathematics*, 40(2), 180–193. https://doi.org/10.1016/j.aam.2006.12.003

Hernán, M. A. and Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC. https://miguelhernan.org/whatifbook

Holland, P. W. (1986). Statistics and Causal Inference. *Journal of the American Statistical Association*, 81(396), 945–960. https://doi.org/10.1080/01621459.1986.10478354

Imbens, G. W. and Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction*. Cambridge University Press. https://doi.org/10.1017/CBO9781139025751

Lin, W. (2013). Agnostic Notes on Regression Adjustments to Experimental Data: Reexamining Freedman's Critique. *The Annals of Applied Statistics*, 7(1), 295–318. https://doi.org/10.1214/12-AOAS583

Neyman, J. (1923/1990). On the Application of Probability Theory to Agricultural Experiments: Essay on Principles, Section 9. *Statistical Science*, 5(4), 465–472. https://doi.org/10.1214/ss/1177012031

Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*, 2nd edition. Cambridge University Press. https://doi.org/10.1017/CBO9780511803161

Rosenbaum, P. R. (2002). *Observational Studies*, 2nd edition. Springer. https://doi.org/10.1007/978-1-4757-3692-2

Rosenbaum, P. R. and Rubin, D. B. (1983). The Central Role of the Propensity Score in Observational Studies for Causal Effects. *Biometrika*, 70(1), 41–55. https://doi.org/10.1093/biomet/70.1.41

Rubin, D. B. (1974). Estimating Causal Effects of Treatments in Randomized and Nonrandomized Studies. *Journal of Educational Psychology*, 66(5), 688–701. https://doi.org/10.1037/h0037350

Rubin, D. B. (2005). Causal Inference Using Potential Outcomes: Design, Modeling, Decisions. *Journal of the American Statistical Association*, 100(469), 322–331. https://doi.org/10.1198/016214504000001880

White, H. (1980). A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity. *Econometrica*, 48(4), 817–838. https://doi.org/10.2307/1912934
