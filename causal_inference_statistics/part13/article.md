# Causal Machine Learning and Double Machine Learning: When Flexibility Helps—and When It Does Not

*Part 13 of Causal Inference in Statistics. A partially linear causal experiment, three-fold cross-fitting, simple versus boosted nuisance models, and a comparison of causal error rather than prediction performance alone. Python code provided.*

The appeal of double machine learning is not that a predictive model suddenly becomes causal. It is that a causal estimator can use flexible prediction models to remove systematic variation due to measured confounders, while controlling how estimation errors enter the final target. The strategy has important theoretical advantages. Its finite-sample success nevertheless depends on the nuisance functions being estimated well enough.

This article compares two controlled models with the same true causal effect, one with linear covariate relationships and another with nonlinear ones. Under correct linear specification, simple regression is clearly better than boosting. Under nonlinear misspecification, boosting reduces bias but leaves a substantial gap from the truth. The comparison is based on actual simulation output, not an assumption that more flexible software should perform better.
## The question behind double machine learning

Double machine learning (DML) is often described as a way to combine flexible prediction models with statistically principled causal estimation. That description is useful but incomplete. Machine learning models are good at approximating complicated conditional relationships; causal identification asks whether such relationships contain information about a specified intervention. The distinction remains even after orthogonal scores and cross-fitting enter the calculation. An estimator may be protected against certain small errors in its nuisance functions without being protected against a consistently wrong functional approximation.

The decision motivating this article is whether a continuous intervention would change an outcome if deliberately increased by one unit. Treatment intensity is not assigned randomly. Baseline characteristics affect both treatment and the expected outcome, so an unadjusted regression confounds treatment with prognosis. The goal is a structural coefficient, not a forecast of the observed outcome given naturally occurring treatment.

The benchmark compares ordinary regression without adjustment, partialling out with linear nuisance models and cross-fitting, partialling out with gradient boosting and cross-fitting, and the same boosted residualization fitted on the entire evaluation sample. I use a setting in which linear models are correctly specified and another in which both treatment assignment and outcome relationships contain nonlinear terms. The treatment effect stays fixed at one. The contrast is designed to challenge the assumption that the more flexible algorithm must automatically win.

## What is being estimated

In a partially linear causal model, treatment D is generated from covariates X and a residual V, while outcome Y depends on treatment through a coefficient tau and on covariates through another nuisance function. Write D=m(X)+V and Y=tau D+g(X)+epsilon. The causal target is tau under an intervention that changes D while leaving the other structural mechanisms and disturbances unchanged. In this experiment tau equals one for every unit.

Identification requires more than writing these equations. The treatment residual must represent variation independent of the outcome disturbance conditional on X, and the intervention must be well defined. In this controlled experiment both residuals are independent standard normal variables generated independently of the covariates. In observational applications that independence or conditional mean restriction is a substantive assumption rather than an observable fact.

The nuisance functions m and E[Y|X] are not themselves the scientific objective. They help isolate the residual variation in treatment from systematic treatment selection. The role of flexible learning is to estimate them sufficiently well without allowing the analysis to mistake their predictive performance for identification. If important common causes are absent from X, DML cannot recover them by applying a better algorithm to the remaining features.

## Why residualizing the outcome works

Define q(X)=E[Y|X]. Under the partially linear model and conditional mean assumptions, q(X)=tau m(X)+g(X). Therefore Y-q(X)=tau[D-m(X)]+epsilon. The true treatment coefficient can be recovered from the population relationship between residualized treatment and residualized outcome. This is the Frisch–Waugh–Lovell logic expressed using conditional expectations rather than requiring a fully linear relationship in X.

One natural estimator first fits qhat(X) and mhat(X). It then calculates residualized outcomes and treatments and fits a regression of the first residual on the second, without a new intercept when residual centring is appropriate. The coefficient is the sum of products of the two residuals divided by the sum of squared treatment residuals. This is the partialling-out estimator studied here.

The algebra matters because it explains the scope of the method. It separates confounding adjustment from the final effect calculation, making nonlinear nuisance fitting possible. It does not say that every pair of estimated nuisance functions produces an accurate effect. The estimated residuals must be sufficiently close to the population residuals and must retain enough variation in treatment to identify the coefficient.

## Orthogonality and second-order error

The score can be written as psi(W;tau,q,m)=[D-m(X)]{Y-q(X)-tau[D-m(X)]}, where W collects the observed variables. The target satisfies a population moment condition E[psi]=0. Its key property is orthogonality: at the true nuisance functions, the first derivative of the population moment with respect to small perturbations of q and m vanishes along the relevant directions.

This makes the effect estimate relatively insensitive to sufficiently small first-stage approximation errors. Instead of a first-order nuisance error appearing directly in the target estimate, leading errors can enter through products or second-order terms. This is the basis for combining flexible nuisance estimators with conventional large-sample causal inference under suitable rate conditions.

Orthogonality does not eliminate arbitrary misspecification. A nuisance estimator converging to a badly wrong function need not satisfy the small-error conditions. Nor does orthogonality provide positivity, no unmeasured confounding or a meaningful intervention. It is a property of an estimating equation conditional on an identification framework. Its practical value depends on the quality of the nuisance estimates and the quantity of residual treatment variation.

## Why cross-fitting is used

Flexible learners can overfit the same observations on which their predicted nuisance functions are evaluated. Residuals measured on the training sample may consequently be too small or otherwise distorted. Cross-fitting separates the observations used to train nuisance functions from those used to calculate their residual contributions to the causal score.

The experiment uses three folds. Each fold is held out once while models are fitted on the remaining two folds. Out-of-fold predictions for both the treatment and outcome are collected for every observation. The causal coefficient is computed only after the held-out predictions have been assembled. The same fold partition is used for both nuisance functions within a replication.

The comparison also includes boosted nuisance fits and residuals from the same data. It illustrates what changes when this separation is omitted under a fixed learner configuration. However, the experiment does not claim that cross-fitting must improve every finite-sample accuracy measure. Cross-fitting generally reduces training leakage and supports asymptotic theory; at small samples it also reduces the training data available to each nuisance fit and may increase approximation errors. Those two forces can work in different directions.

## Two deliberately different data-generating systems

The covariate vector contains three independent standard normal components, X1, X2 and X3. In the linear design the treatment function is m(X)=1.4X1+0.9X2+0.5X3, while the baseline outcome function is g(X)=1.2X1+1.3X2+0.8X3. Treatment is D=m(X)+V and outcome is Y=D+g(X)+epsilon. The true effect of a one-unit increase in treatment is one.

In the nonlinear design, treatment assignment becomes m(X)=1.4sin(X1)+0.9(X2 squared minus one)+0.5X1X3. The outcome baseline becomes g(X)=1.2sin(X1)+1.3(X2 squared minus one)+0.8X1X3. Treatment and outcome disturbances remain independent standard normal, and the structural coefficient on treatment remains exactly one.

The nonlinear terms are chosen to produce confounding that a simple additive linear nuisance regression cannot represent. This is an intentionally adverse specification for linear adjustment. The flexible learner must learn nonlinear relationships from the observed sample, but it is not given the exact structural basis functions. Thus it faces a genuine estimation problem rather than simply being handed the correct formula.

## Estimators and common evaluation

The baseline unadjusted estimator is the sample covariance between D and Y divided by the variance of D. It answers the observational slope question, not necessarily the intervention question. It is included because causal estimators should be compared with transparent baselines that reveal how much confounding adjustment is actually achieving.

The linear DML estimator uses ordinary least squares to predict treatment and outcome from X within each training fold. The boosted DML estimator instead uses histogram gradient boosting regressors with fixed hyperparameters and the same three-fold splits. Finally, boosted nuisance models are fitted on the entire sample and evaluated on that same sample, with no cross-fitting.

No tuning search is performed; the objective is a reproducible comparison of clearly specified procedures, not a claim to have found the optimal learner. Fixed hyperparameters can put one procedure at a disadvantage in particular circumstances. This limitation should be understood before interpreting the estimator ranking. The code records point estimates and their between-replication mean, bias, standard deviation and RMSE relative to the known effect.

## Why the number of replications is limited

The experiment uses 20 independent datasets for each of two causal designs, with 800 individuals in each dataset. This is intentionally a modest Monte Carlo investigation. Flexible learners are refitted repeatedly for the outcome and treatment on different folds, and the computational cost of a larger simulation is material.

The first configuration was too expensive to execute reliably, so I fixed a smaller replication count and a lighter boosting specification before reporting results. A 20-replication sample is sufficient to expose large differences of the magnitude observed here, but it cannot support precise claims about small differences in estimator efficiency. Reported means and RMSE are descriptive for the chosen seed and specification.

A responsible interpretation should distinguish systematic mathematical expectations from Monte Carlo observations. For the linear design, the population result of correctly specified residualization can be shown algebraically. For boosted finite-sample performance, the tables are empirical observations from this configuration. Additional replications, alternative seeds and hyperparameter sensitivity are needed before treating narrow differences as stable general properties of the algorithms.

## The linear-design results

The true treatment effect equals one. In the linear design the unadjusted regression has mean estimate 1.7985 and RMSE 0.7989 across the 20 datasets. It is confounded as expected. The correctly specified linear cross-fitted estimator has mean estimate 0.9954 and RMSE 0.0259, much closer to the known effect.

The boosted cross-fitted estimator averages 1.1004 with RMSE 0.1062. The boosted in-sample residualization averages 1.0420 with RMSE 0.0561. With this particular data-generating model, sample size and learner configuration, the flexible models are not an improvement over correctly specified linear adjustment.

That result is not a criticism of boosting. The linear regression is supplied with exactly the covariates needed to represent the nuisance relationships, whereas boosting must learn their shapes from data and is affected by smoothing and regularization. When the relationships are genuinely linear and low-dimensional, a less flexible estimator can have a substantial finite-sample advantage.

The result also warns against interpreting cross-fitting as a mechanical accuracy guarantee. In these simulations the boosted in-sample residualization has lower RMSE than its cross-fitted counterpart. That empirical comparison does not overturn the theoretical motivation for cross-fitting; it reflects the separate influence of reduced training size, regularization and finite-sample nuisance approximation.

## The nonlinear-design results

In the nonlinear design, the unadjusted regression averages 1.9322 with RMSE 0.9331, strongly overstating the true effect. The linear cross-fitted estimator averages 1.9480 and has RMSE 0.9490. Adding linear covariate adjustment does not solve the nonlinear confounding structure; here it slightly worsens the point-estimation error in the Monte Carlo summary.

The boosted cross-fitted estimator reduces the mean to 1.3601 and RMSE to 0.3655. This is a large improvement over the linear nuisance model, but still leaves substantial positive bias. The boosted in-sample residualization has mean 1.3043 and RMSE 0.3105 under the chosen configuration.

The important distinction is between improvement and causal accuracy. Boosting captures more of the nuisance structure than a linear model, yet the resulting coefficient remains about 0.36 above the true effect on average when cross-fitted. One cannot describe this result as a successful recovery of the effect merely because it outperforms the simple model.

The model may be underfitting interactions, making regularized approximations that shrink relevant structure, or encountering finite-sample limitations. The benchmark does not isolate which source dominates. The justified observation is that flexible nuisance learning reduced bias here without eliminating it.

## A compact comparison

The complete point-estimator summary has eight method-by-design combinations. In the linear setting: naive 1.7985 (RMSE 0.7989), linear cross-fit 0.9954 (0.0259), boosted cross-fit 1.1004 (0.1062), and boosted same-sample 1.0420 (0.0561). In the nonlinear setting: naive 1.9322 (0.9331), linear cross-fit 1.9480 (0.9490), boosted cross-fit 1.3601 (0.3655), and boosted same-sample 1.3043 (0.3105).

Every estimate is compared against the same structural coefficient of one. No subgroup change, changing target population or treatment-effect heterogeneity accounts for the differences. The comparison is about how well residualization isolates treatment variation when nuisance relationships are represented accurately or poorly.

The table supports three conclusions of different strength. First, correctly specified linear adjustment works extremely well in the linear design, as mathematical expectations suggest. Second, linear adjustment is inadequate for the intentionally nonlinear design. Third, boosting is helpful relative to linear adjustment there, but this configuration is not accurate enough to justify treating the estimates as unbiased in a practical experiment.

It does not support a general winner between cross-fitted and same-sample boosting. Twenty replications, one seed and one learner setting are too narrow for that claim.

## Where the confounding bias comes from

In both systems, the treatment function m(X) and outcome baseline g(X) are correlated because they are functions of the same covariates. The unadjusted slope contains the effect tau and an association component Cov(D,g(X))/Var(D). Since D=m(X)+V and V is independent of X, this contribution is driven by Cov(m(X),g(X)).

In the linear design both functions share positive components of X1, X2 and X3. The untreated outcome is therefore higher, on average, among people with larger naturally occurring treatment. The regression cannot distinguish that dependence from the intervention effect without accounting for X.

The nonlinear design creates a similar association through sine, quadratic and interaction terms. A linearly adjusted model observes X but cannot remove all dependence created by those functions because its conditional predictions are misspecified. The estimator's residualized treatment can remain correlated with unexplained baseline outcome structure.

Machine learning is relevant because it can approximate complicated m and q functions without manually specifying every basis expansion. But the approximation must be sufficiently accurate for the orthogonal moment to behave as desired. In the experiment boosting is an improvement, not a complete solution.

## What cross-fitting removes

Cross-fitting primarily addresses dependence between nuisance fitting errors and the individual observations used in the final score. By constructing predictions for held-out units, it avoids directly scoring each observation using a flexible nuisance model trained on that observation's own outcome and treatment.

This is valuable when nuisance learners are adaptive, regularized or highly complex. Overfitted predictions can produce residual relationships that do not approximate the desired population score well, especially if the same data determine both the nuisance functions and the treatment coefficient.

The reduction in leakage does not force predictions to be accurate. If each fold's training sample is too small for the relevant nonlinear structure, the held-out residuals can still contain confounding information. Repeating the procedure with more folds may alter the trade-off, but it will not automatically supply missing covariates or correct an unsuitable learner family.

In the current results, boosted in-sample residualization happens to have lower RMSE than boosted cross-fitting in both designs. The observation should be reported, not concealed. It demonstrates why theoretical safeguards, finite-sample learner performance and the specific empirical benchmark should be evaluated separately.

## Orthogonality is local, not magic

The technical term Neyman orthogonality can sound like an unconditional guarantee of robustness. It is not. Orthogonality says that at the correct nuisance functions, certain first-order perturbations do not change the population score's expectation to first order. It does not state that large deviations from the correct functions have no effect.

Suppose a nuisance estimate converges toward the wrong linear projection of a genuinely nonlinear conditional mean. The approximation error may remain nonzero even with infinite data. The score can then be evaluated away from the neighbourhood where the orthogonality argument is most useful. The resulting coefficient may converge to a biased quantity.

The nonlinear linear-adjustment result illustrates that danger. It is not repaired by cross-fitting because the misspecification is structural rather than caused only by reusing training observations. Similarly, boosted nuisance learners with substantial approximation or regularization bias can leave residual confounding even when trained and scored on disjoint folds.

Successful double machine learning therefore depends on an appropriate identification strategy, adequate residual treatment variation, nuisance convergence conditions and an inferential procedure suited to the setting. It is not simply a machine-learning pipeline with a causal label.

## A natural benchmark for richer functions

A useful next experiment would supply the true structural basis functions to ordinary regression: sin(X1), X2 squared minus one, and X1 times X3. Since the data-generating mechanism is known, that oracle nuisance model would reveal how much of the boosted estimator's error comes from learning the nonlinear shape rather than from unavoidable sampling variation.

The current benchmark deliberately omits that oracle competitor to keep the primary comparison close to a realistic analyst choosing between simple linear features and generic flexible learning. Nevertheless, the absence limits any claim about optimal achievable accuracy. Boosting should not be judged against a claim that no other method could do better.

Another extension would compare generalized additive models, splines, random forests and boosting with independent tuning data, while holding the same causal estimand and cross-fitting protocol fixed. Hyperparameter optimization should be nested correctly to avoid using evaluation outcomes to choose the apparent best causal procedure.

Such extensions would also allow a more informative investigation of nuisance prediction error, the product of residual error rates and the corresponding treatment-effect bias. That would connect the abstract orthogonality theory to directly observed model diagnostics.

## The importance of an honest baseline

An empirical comparison should contain at least one simple method whose behaviour is easy to interpret. In this article ordinary linear adjustment is not a straw man. It is exactly right for one design and demonstrably wrong for the other, making the choice of nuisance model part of the scientific question.

The unadjusted regression provides another baseline. Its estimates near 1.80 and 1.93 show the strength of the observational association bias. Any proposed causal method should be evaluated against that distortion, but a reduction in distortion is not enough to validate a new estimate.

The final contrast should remain grounded in the true effect, not in whether one estimator has the most elaborate architecture. A method can improve prediction of treatment and outcome while still leave enough nuisance error to bias the causal coefficient. Conversely, a simple correct model can outperform a flexible method that learns more slowly or imposes unnecessary regularization.

This emphasis is consistent with the earlier articles in the series: design and identification determine what can be learned, while estimator choice determines how accurately the identified quantity is computed from available data.

## Prediction performance is a separate metric

A nuisance model predicts treatment from covariates or predicts outcomes conditional on covariates. Its usual evaluation metrics quantify errors for those prediction tasks. They are relevant to DML, but they do not by themselves establish that the final causal estimate is accurate.

The causal estimator combines two nuisance residuals in an orthogonal moment. Its bias depends on how errors in those nuisance functions interact with the treatment residual and with one another. It is therefore possible for a model with better prediction scores to produce only a small improvement in causal estimation, or even a worse finite-sample coefficient.

The current experiment does not report nuisance out-of-fold R-squared or mean squared error. That is an explicit limitation: it demonstrates final-effect performance rather than explaining every reason for the ranking of learners. A complete methodological benchmark would report nuisance errors alongside treatment-effect bias, RMSE and interval coverage.

Choosing the model with the smallest predictive error and then assuming the causal estimate is valid repeats the confusion this series has criticized from its beginning. Good prediction is useful evidence about approximation, not proof of the identifying assumptions or the target parameter.

## Cross-fitting and inference

The results are Monte Carlo point estimates. The script calculates between-replication dispersion and RMSE, not confidence intervals for individual datasets. It therefore does not establish nominal coverage of a DML variance estimator or compare normal confidence intervals with bootstrap procedures.

Under suitable orthogonality, nuisance convergence and regularity conditions, influence-function-based inference can justify asymptotically normal treatment-effect estimates. Implementing those calculations correctly would require estimating the variance of the orthogonal score and accounting for the fold structure and target parameter.

The finite-sample distributions here are based on only twenty independent replications per design, so even standard deviation comparisons are noisy. A stronger inference study would increase repetitions, calculate empirical coverage, evaluate weak residual treatment variation and consider violations of homoskedasticity or independence.

An article should not imply that cross-fitting alone makes ordinary confidence intervals valid. The causal assumptions, nuisance approximation and uncertainty method must all be considered. This experiment establishes specific point-estimation behaviour and leaves coverage as a defined extension.

## What if an important confounder is omitted?

The simulation supplies every confounding covariate to each nuisance learner. This is an intentionally favourable assumption. If an unobserved variable affected treatment and outcome after conditioning on X, neither linear nor nonlinear residualization could generally remove that source of dependence.

A more powerful learner can capture complex relationships among measured features, but it cannot recover the intervention effect from observational correlations when the adjustment set fails to identify the causal quantity. Nuisance flexibility and exchangeability answer different questions. The first concerns approximation; the second concerns whether the statistical functional corresponds to an intervention.

The same caution applies to post-treatment variables. Including a mediator or collider in the nuisance features can change the estimand or introduce selection bias, as Part 5 showed. A learner trained on every available column does not know which columns are causally admissible. Feature availability and predictive relevance are not adjustment criteria.

Causal machine learning is therefore not an alternative to a causal design. It is a collection of estimation tools that can improve particular designs when their assumptions are appropriate and their nuisance functions can be estimated well.

## Treatment overlap and residual variation

The partially linear model depends on residual treatment variation after conditioning on X. If treatment were nearly determined by measured covariates, the residualized denominator could become small, making the final estimator unstable even with excellent nuisance predictions.

This is conceptually related to the weak-overlap problem studied for inverse probability weighting and to the weak first stage in instrumental-variable estimation. A method requires enough independent variation to distinguish the treatment effect from the covariate contribution. The mathematical way that requirement appears differs across estimators, but insufficient relevant variation cannot be solved by an increasingly complex algorithm.

In this experiment D=m(X)+V with V having unit variance, so residual treatment variation is generous. The finite-sample failures are not driven by near-deterministic treatment assignment. A follow-up could reduce the variance of V to examine how DML behaves under weak residual identification while holding the structural effect fixed.

That investigation would be important before deploying DML in settings where interventions are allocated almost deterministically by risk scores or administrative rules.

## When does flexibility pay for itself?

Flexibility pays for itself when the improvement in nuisance approximation is larger than the additional estimation error introduced by the learner. In the linear design it does not pay here: the cross-fitted linear estimator has RMSE around 0.026 while cross-fitted boosting has RMSE around 0.106.

In the nonlinear design, flexibility is beneficial relative to the inappropriate linear model: boosting reduces RMSE from about 0.949 to 0.366. That is a considerable improvement, even though the remaining bias is too large to describe the result as accurate recovery of the true coefficient.

These observations point to a model-selection principle rather than a specific algorithm. Start from plausible substantive functional forms and strong simple baselines. Use flexible learners where the data and question justify them. Evaluate final causal estimates rather than relying only on nuisance prediction metrics. Report uncertainty and sensitivity to specifications that make different approximation assumptions.

It would be methodologically inappropriate to declare boosting the best estimator solely because it wins in the nonlinear scenario, or linear regression the best solely because it wins in the linear scenario. The examples are constructed to show where their strengths and weaknesses arise.

## Practical workflow for an observational study

A real analysis should begin by specifying the intervention, population, outcome and causal estimand. Next, identify the measured pre-treatment variables sufficient to justify the needed conditional independence assumptions. Only then define the nuisance functions and choose plausible estimation methods.

Partition the data before fitting nuisance learners. Preserve the independence between training folds and the observations used for the orthogonal score. Record the features and transformations included in every nuisance model, along with tuning rules. Ensure any preprocessing or variable selection is learned inside the training folds rather than leaked from the full dataset.

Evaluate nuisance predictions out of sample and inspect residual treatment variation. Estimate the target effect with an appropriate orthogonal score. Report an uncertainty method justified for the data structure and sample size, particularly when observations are clustered, longitudinal or otherwise dependent.

Finally, compare results with simpler estimators, examine alternative nuisance specifications and state which causal assumptions cannot be tested from the observed data. A robust empirical report makes these decisions auditable rather than leaving them inside a single software call.

## Limits of this experiment

The benchmark contains only two structural designs, 800 observations per dataset and 20 Monte Carlo replications per design. Covariates and disturbances are independent Gaussian variables, treatment is continuous, and the causal effect is constant. Every variable causing treatment-outcome dependence is observed, so there is no hidden confounding.

The boosted learner uses fixed settings and no tuning. A different configuration or an oracle nonlinear basis could materially change the results. The in-sample comparison is not a causal inference recommendation; it is included to inspect a common implementation decision. The lower RMSE of that version in this small simulation does not establish that same-sample fitting generally outperforms cross-fitting.

The study does not evaluate interval coverage, nuisance prediction diagnostics, treatment-effect heterogeneity or overlap violations. Its Monte Carlo means, especially the relative ordering of similarly performing methods, have appreciable simulation uncertainty. Results should be presented as a controlled computational observation rather than a population-wide ranking of estimators.

The experiment also cannot establish whether any real-world observational study has adequately measured its confounders. Its identifying assumptions are true because they were built into the synthetic mechanism. In empirical work those assumptions require separate substantive justification.

## What the benchmark established

The true intervention effect is one in both data-generating systems. Under linear nuisance functions, unadjusted regression averages 1.7985, correctly specified linear cross-fitted DML averages 0.9954, and cross-fitted boosting averages 1.1004. Under nonlinear nuisance functions, unadjusted regression averages 1.9322, linear cross-fitted DML averages 1.9480, and cross-fitted boosting averages 1.3601.

The flexible estimator reduces error substantially in the nonlinear design without fully removing bias. In the linear design it is worse than a simpler correctly specified procedure. The in-sample boosted variant has lower RMSE than boosted cross-fitting in both finite-sample comparisons, reminding us that an asymptotic safeguard does not guarantee the best point estimate in one small benchmark.

The useful conclusion is not that machine learning fails at causality. It is that causal machine learning remains an estimation problem governed by assumptions about interventions, confounding, residual variation and nuisance quality. Orthogonal scores can reduce sensitivity to small nuisance errors; cross-fitting can protect against training leakage. Neither turns a badly approximated conditional model or an unidentified observational comparison into a reliable causal estimate.

A technically sophisticated estimator should be judged by the quantity it identifies and the evidence supporting its numerical performance. The next article will investigate Bayesian causal inference, where prior assumptions and posterior uncertainty raise a related challenge: distinguishing information supplied by observations from information supplied by the model.

## Code and reproducibility

The complete experiment is in `causal_inference_statistics/part13/experiment.py`, with simulation output in `results.csv`. It uses Python 3.12+, NumPy and scikit-learn. The seed is 20261018. Twenty independent replications are generated per design, each with 800 observations. Results are summaries of point estimates, not estimates from real-world data or confidence intervals for individual studies.

## The series

This is Part 13 of *Causal Inference in Statistics*. The preceding parts covered regression, randomized assignment, propensity scores, inverse probability weighting, graphical adjustment, difference-in-differences, instrumental variables, regression discontinuity, synthetic controls, doubly robust estimation, longitudinal g-computation and heterogeneous treatment effects.

## References

Belloni, A., Chernozhukov, V. and Hansen, C. (2014). Inference on Treatment Effects after Selection among High-Dimensional Controls. *Review of Economic Studies*, 81(2), 608–650. https://doi.org/10.1093/restud/rdt044

Chernozhukov, V. et al. (2018). Double/debiased Machine Learning for Treatment and Structural Parameters. *Econometrics Journal*, 21(1), C1–C68. https://doi.org/10.1111/ectj.12097

Chernozhukov, V., Newey, W. K. and Singh, R. (2022). Automatic Debiased Machine Learning of Causal and Structural Effects. *Econometrica*, 90(3), 967–1027. https://doi.org/10.3982/ECTA18515

Crump, R. K., Hotz, V. J., Imbens, G. W. and Mitnik, O. A. (2009). Dealing with Limited Overlap in Estimation of Average Treatment Effects. *Biometrika*, 96(1), 187–199. https://doi.org/10.1093/biomet/asn055

Farrell, M. H. (2015). Robust Inference on Average Treatment Effects with Possibly More Covariates than Observations. *Journal of Econometrics*, 189(1), 1–23. https://doi.org/10.1016/j.jeconom.2015.06.017

Frisch, R. and Waugh, F. V. (1933). Partial Time Regressions as Compared with Individual Trends. *Econometrica*, 1(4), 387–401. https://doi.org/10.2307/1907330

Hernán, M. A. and Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC. https://miguelhernan.org/whatifbook

Newey, W. K. (1994). The Asymptotic Variance of Semiparametric Estimators. *Econometrica*, 62(6), 1349–1382. https://doi.org/10.2307/2951752

Neyman, J. (1959). Optimal Asymptotic Tests of Composite Statistical Hypotheses. In *Probability and Statistics: The Harald Cramér Volume*, 213–234.

Robinson, P. M. (1988). Root-N-Consistent Semiparametric Regression. *Econometrica*, 56(4), 931–954. https://doi.org/10.2307/1912705

Robins, J. M., Rotnitzky, A. and Zhao, L. P. (1994). Estimation of Regression Coefficients When Some Regressors Are Not Always Observed. *Journal of the American Statistical Association*, 89(427), 846–866. https://doi.org/10.1080/01621459.1994.10476818

Rosenbaum, P. R. and Rubin, D. B. (1983). The Central Role of the Propensity Score in Observational Studies for Causal Effects. *Biometrika*, 70(1), 41–55. https://doi.org/10.1093/biomet/70.1.41

Van der Laan, M. J. and Rose, S. (2011). *Targeted Learning: Causal Inference for Observational and Experimental Data*. Springer. https://doi.org/10.1007/978-1-4419-9782-1

Wager, S. and Athey, S. (2018). Estimation and Inference of Heterogeneous Treatment Effects using Random Forests. *Journal of the American Statistical Association*, 113(523), 1228–1242. https://doi.org/10.1080/01621459.2017.1319839