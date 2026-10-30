# Sensitivity Analysis for Unmeasured Confounding: How Much Hidden Bias Changes the Result

*Part 16 of Causal Inference in Statistics. A known omitted common cause, four treatment-assignment strengths, 3,200 regression fits and a comparison between conventional precision and causal robustness. Reproducible Python code provided.*

## The question behind a precise estimate

An observational treatment effect may appear stable across regressions, large samples and several uncertainty calculations. None of those results tells us what would happen if an unmeasured cause affected both treatment assignment and outcome. The familiar reassurance that a coefficient was adjusted for all available variables is a statement about the dataset, not necessarily about the causal system. Sensitivity analysis addresses the difference by asking how much unmeasured confounding would be needed to change the conclusion.

This article examines a synthetic system whose causal effect is known, in which the strength of an omitted confounder is controlled by a parameter. I generate 400 independent datasets of 2,000 observations for each of four values of that parameter. A regression using the observed baseline variable is compared with an oracle regression that also includes the normally hidden cause. As the hidden assignment effect rises, the observed-only estimate changes from about 1.001 to 1.749, despite a true causal effect of one. Its model-based standard error actually becomes smaller at the largest confounding strength. The correctly adjusted estimator stays near one.

The experiment is an illustration of confounding sensitivity, not a claim that a particular unobserved variable exists in a real dataset. It is also not a complete sensitivity-analysis procedure: the scientist confronting real data does not know the hidden variable or its coefficients. The simulation gives a controlled way to understand what a sensitivity parameter means and why a credible analysis needs to report plausible ranges rather than simply adding a warning about possible unobserved variables.

## The policy decision

Suppose an employment service wants to learn whether additional job-search support improves subsequent employment. Caseworkers record age, previous employment history and other observable information. They may also use knowledge about personal circumstances, motivation or informal labour-market connections that never enters the statistical dataset. Those unrecorded attributes may influence both the decision to provide support and the eventual outcome.

A regression adjusting for the recorded characteristics might estimate a positive programme effect with a narrow confidence interval. That coefficient can be a good estimate of the corresponding conditional association while remaining causally biased. If an unmeasured characteristic makes people more likely to receive support and independently improves their employment prospects, some of the observed benefit may be attributed to the intervention when it belongs to pre-existing prognosis.

A sensitivity analysis does not manufacture missing records or prove the true value of the effect. Its purpose is to evaluate how much the causal conclusion relies on assuming away a class of plausible confounders. Under one assumption the estimated effect may be robust; under another it may be compatible with no effect, or a smaller benefit than the policy requires to be worthwhile.

This distinction is especially important when a decision turns on a threshold: whether the programme is worth its cost, whether an intervention should be expanded or whether a clinical treatment's benefit exceeds its harms. The most informative question may be how much bias would move the treatment effect across that threshold, rather than whether a conventional regression p-value is below 0.05.

## The causal target and observed data

Let X denote a continuous treatment, W an observed baseline characteristic, U an unobserved baseline characteristic and Y the outcome. The causal target is the change in expected outcome under an intervention increasing X by one unit. For potential outcomes Y(x), this is the contrast E[Y(x+1)-Y(x)].

Under a structural model with a constant additive effect, that quantity may be represented by a coefficient τ. But a conditional regression of Y on X and W estimates the change in the observed conditional mean E[Y|X,W]. It equals the causal effect only under assumptions that remove common causes of X and Y after conditioning on W, together with consistency and an appropriate intervention definition.

Unmeasured confounding is therefore not an additional standard-error term automatically captured by the regression software. It is a difference between the statistical parameter estimated from the observed distribution and the causal quantity the researcher wants. A conventional confidence interval is built around the former. Reporting it as uncertainty about the latter requires a separate identification argument.

In this article, we can compare the two quantities because U is generated by the experiment and retained by the simulation. A real dataset would normally contain only W, X and Y. The inability to verify conditional exchangeability from those variables is precisely why sensitivity analysis is needed.

## Structural equations

The experiment generates independent standard normal variables W, U, ε_X and ε_Y and defines

\[
X=0.7W+\lambda U+\varepsilon_X,
\]

\[
Y=\tau X+0.8W+1.5U+\varepsilon_Y,
\qquad \tau=1.
\]

The parameter λ controls the extent to which the unmeasured characteristic U influences treatment assignment. Its coefficient in the outcome model remains 1.5. We examine λ=0, 0.25, 0.5 and 1.0.

At λ=0, U predicts outcome but does not influence treatment. It is a prognostic factor but not a confounder of the treatment-outcome comparison after adjustment for W. At positive λ, it is a common cause. Because U is independent of W and the disturbance in X, the magnitude of residual confounding can be derived explicitly.

An intervention setting X to x changes the outcome equation to Y(x)=x+0.8W+1.5U+ε_Y. Changing x by one increases the potential outcome by one for everyone. The causal effect is thus exactly one for all four assignment mechanisms, independent of λ.

This model deliberately makes the causal truth constant while varying how well the observed data identify it. Treatment is continuous and disturbances are Gaussian. No hidden post-treatment feedback, treatment-effect heterogeneity, selection into the observed sample or positivity failure is introduced. Such mechanisms deserve separate treatment, but would obscure the elementary sensitivity calculation.

## Deriving omitted-variable bias

Consider a regression of Y on an intercept, treatment X and observed W. By the Frisch–Waugh–Lovell theorem, the treatment coefficient can be obtained by residualizing both treatment and outcome with respect to W and regressing one residual on the other. Because W is independent of U and ε_X, residual treatment variation is

\[
\widetilde X=\lambda U+\varepsilon_X.
\]

After removing the observed W component, the outcome contains τ\widetilde X+1.5U+ε_Y. Hence the population treatment coefficient of the observed-only regression is

\[
\beta_{X\mid W}
=\tau+1.5\frac{\operatorname{Cov}(\widetilde X,U)}{\operatorname{Var}(\widetilde X)}
=1+\frac{1.5\lambda}{1+\lambda^2}.
\]

The result separates the causal treatment effect from the noncausal association created by hidden U. At λ=0 the bias is zero. At λ=0.25 it is approximately 0.3529; at λ=0.5 it is 0.6; and at λ=1 it is 0.75. These values predict the centres of the Monte Carlo distributions before any data are generated.

The expression is not necessarily monotonic over all possible strengths. As λ becomes extremely large, changes in the denominator alter its behaviour. The experiment studies only the specified range. For those settings the bias grows substantially as U increasingly determines treatment. The point is not that a larger confounder always creates a larger regression bias under every structural model, but that bias depends on both its relationship with assignment and its relationship with outcome.

## What an oracle regression represents

The second estimator regresses Y on treatment, observed W and the otherwise hidden U. Because the structural outcome equation is exactly linear and the outcome disturbance is independent of its predictors, the population coefficient on X is the causal effect of one. This is an oracle analysis: it uses information that the researcher in the intended observational problem does not have.

The oracle is not presented as a realistic competitor that can be selected from ordinary datasets. It is a reference against which we can assess how much error arises purely from leaving out U. Holding the sample and structural equations fixed makes the comparison particularly informative. Where the observed-only regression remains accurate, the simulation can explain why; where it fails, we can measure the difference exactly.

The term unmeasured needs careful treatment. Researchers sometimes have proxies for latent common causes, but a proxy is not equivalent to the variable itself. Measurement error can leave residual confounding after adjustment. More sophisticated latent-variable or negative-control designs may exploit proxies under additional assumptions; they cannot simply be substituted for an oracle variable.

The oracle's good performance in this benchmark depends on knowing the correct linear outcome relationship and observing the relevant confounder without error. The experiment makes these assumptions true on purpose. Its lessons about sensitivity should not be read as reassurance that an ordinary regression becomes causally valid whenever a few additional covariates are entered.

## Simulation procedure

The experiment contains four confounding strengths and two fitted models at each strength. For each configuration, 400 independent samples of 2,000 observations are generated. Ordinary least squares is fitted with an intercept. Conventional homoskedastic residual-variance standard errors are computed from the inverse design cross-product.

The experiment reports the Monte Carlo mean coefficient, bias relative to the true effect, root mean squared error, average reported standard error and empirical coverage of an approximate 95% confidence interval for the causal target. The interval is the fitted coefficient plus or minus 1.96 times its estimated standard error. The simulation records all data-generating coefficients, uses a fixed pseudorandom seed and requires no external dataset.

These measures answer distinct questions. Bias captures systematic displacement from the causal effect. RMSE combines the displacement and sampling variation. The conventional standard error describes uncertainty in the regression's statistical coefficient, while coverage asks how often its interval happens to include the intervention effect. A model can have a small standard error and zero causal coverage if the target is wrong.

There are 3,200 fitted regressions in total. The reported means are empirical summaries of the realized Monte Carlo samples, and the exact population biases are derived independently from the equations. Agreement between the two is a stronger check than numerical output alone, because it tests whether the implemented process reflects the intended model.

## The estimates

The resulting mean coefficients are:

| Hidden assignment strength λ | Observed-only estimate | Oracle-adjusted estimate | True effect |
| ---: | ---: | ---: | ---: |
| 0.00 | 1.0013 | 1.0009 | 1.0000 |
| 0.25 | 1.3513 | 0.9992 | 1.0000 |
| 0.50 | 1.6001 | 1.0013 | 1.0000 |
| 1.00 | 1.7493 | 1.0009 | 1.0000 |

At λ=0 the observed-only model is causally adequate under the specified mechanism because the hidden variable does not affect treatment. It can still explain outcome variation, but no residual common-cause pathway remains after adjusting for W. Both regressions estimate approximately one.

At λ=0.25 the observed-only estimate rises to 1.3513, closely matching the theoretical value 1.3529. At λ=0.5 it rises to 1.6001, effectively identical to the expected 1.6. At λ=1 it approaches 1.75. These differences are not changes in treatment effect. They are changes in the selection mechanism operating before outcome measurement.

The oracle estimate stays near one across every mechanism. This consistency is evidence that the simulation is isolating residual confounding, not introducing a separate causal response that the simple model cannot represent. The comparison is deliberately stark: a single hidden cause is enough to generate a large systematic error even though the observed W adjustment is always correctly specified.

## The uncertainty comparison

A point estimate can look increasingly persuasive as its reported uncertainty shrinks. The coverage results reveal why that appearance can be misleading:

| λ | Observed-only mean SE | Observed-only causal coverage | Oracle mean SE | Oracle causal coverage |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.0403 | 95.25% | 0.0224 | 94.25% |
| 0.25 | 0.0384 | 0.00% | 0.0224 | 95.00% |
| 0.50 | 0.0335 | 0.00% | 0.0224 | 95.25% |
| 1.00 | 0.0231 | 0.00% | 0.0224 | 96.25% |

At positive hidden confounding strengths, none of the 400 observed-only intervals contains the true effect. This is not because ordinary least squares failed to estimate its own population partial coefficient. It estimated that coefficient well; the causal interpretation attached to it was wrong.

The decrease in the observed-only standard error with increasing λ is especially informative. As treatment variation increasingly reflects U, treatment becomes a more informative predictor of the outcome component associated with U. The conditional regression can fit the observed data with considerable precision while attributing the shared influence to treatment. Precision is not a defense against that kind of misspecification of the causal adjustment set.

In the oracle model, coverage remains close to 95% and standard errors near 0.0224 across the scenarios. Finite Monte Carlo variation explains small departures. The oracle is stable because controlling for U isolates the structural treatment variation supplied by ε_X.

## RMSE and effective uncertainty

For the observed-only estimator, the RMSE rises from about 0.0384 at λ=0 to 0.3532, 0.6011 and 0.7496 as hidden confounding becomes stronger. The corresponding oracle RMSE values remain near 0.021–0.023. These are errors relative to the true effect, not relative to the observational regression's population coefficient.

Bias dominates the observed-only RMSE after λ becomes positive. The standard error continues to measure dispersion around the associational target, which becomes increasingly separated from the causal truth. Consequently, standard errors can shrink while total causal error grows.

This illustrates why it is useful to distinguish sampling uncertainty, model uncertainty and identification uncertainty. A regression variance calculation mainly addresses how the coefficient varies across samples under specified model conditions. It does not represent every plausible causal structure compatible with the observed data. A sensitivity analysis evaluates alternative assumptions about an unobserved cause rather than merely adding another variance component.

In an empirical study, the researcher cannot compute RMSE against a known treatment effect. What they can do is examine a sensitivity function or uncertainty region indexed by assumed confounding strengths. The simulation demonstrates the meaning of that exercise: a reported coefficient may move across a range much larger than its familiar confidence interval when plausible causal assumptions are varied.

## Sensitivity parameters in practice

The simulation varies λ while holding the outcome influence of U at 1.5. In real research, neither number would generally be known. A usable sensitivity analysis therefore has to express assumptions in terms that can be understood, bounded or compared with measured information. Possible parameterizations include associations of an unobserved covariate with treatment and outcome, an odds-ratio departure in a matched study, or partial coefficients of determination describing residual confounding after adjustment.

Each parameterization answers a somewhat different question. A linear sensitivity model may translate the combined strength of hidden common causes into a possible omitted-variable bias. Rosenbaum bounds study how much two people with the same measured covariates might differ in treatment odds because of hidden selection. E-values are designed for particular effect scales and quantify the strength of confounder associations needed to explain away a risk-ratio association. They should not be imported mechanically into a continuous-treatment linear setting without checking their definitions.

A good sensitivity analysis chooses a parameter tied to the model and effect measure, explains its units and indicates what numerical values would be considered plausible. It also reports whether those conclusions depend on the form of the sensitivity model. Parameters are not facts about an unmeasured variable merely because the analyst can vary them numerically.

Here the sensitivity curve is analytically known: bias(λ)=1.5λ/(1+λ²). Its value is that the effect of assumptions on the regression coefficient is visible rather than hidden. It cannot tell a real analyst which λ should be believed.

## Calibrating against observed covariates

One useful empirical strategy is to compare the hypothesized unmeasured confounder with known covariates. Suppose the strongest measured baseline characteristic explains a limited amount of residual treatment variation and outcome variation. A sensitivity analysis may ask what would happen if an omitted factor were as influential as that characteristic, twice as influential, or several times as influential.

This form of benchmarking makes sensitivity magnitudes more interpretable. It turns an arbitrary number into a comparison with actual data features. But the analogy is not proof. An important omitted cause can be stronger than any measured predictor, and a measured variable can be an imperfect proxy for several latent mechanisms. Benchmarking is a way to communicate assumptions, not a guarantee that their chosen range contains the truth.

There is also a direction-of-bias question. Here U raises both treatment and outcome, so the confounding contribution is positive. An omitted variable affecting treatment and outcome in opposite directions could bias the coefficient downward. Several confounders with competing directions might even cancel in the net regression coefficient while changing other intervention contrasts.

A sensitivity analysis that explores only upward bias because it supports an expected narrative would be incomplete. The plausible signs of hidden common-cause pathways should follow the substantive problem and the proposed assignment mechanisms. Where both are plausible, report both directions.

## What does it mean to overturn a result?

Researchers often ask how strong a confounder must be to make an estimated effect zero. That is one tipping-point question, but zero is not always the decision boundary. A programme may need to generate a benefit exceeding its cost; a medical decision may require a minimum clinically relevant effect; a public policy may involve distributional harms even when the average effect remains positive.

For the structural model in this article, the true effect is one throughout and the observed-only regression overestimates it as λ rises. If we started from an observed estimate near 1.75 and entertained the calibrated sensitivity mechanism λ=1, the corrected effect would return to one. This would not reverse the sign, but could materially change a cost-benefit calculation. A different direct outcome effect of U, confounding direction or observed coefficient could produce a sign reversal.

This is why sensitivity results should be reported in the units relevant to the decision. A statement that the estimate is robust to some range of unmeasured confounding has little meaning until the range and decision threshold are described. A narrow confidence interval does not make a policy valuable if its causal magnitude is uncertain by more than the benefit needed to justify adoption.

The appropriate question is not always whether an effect survives every imaginable confounder. It is whether the conclusion persists over a range of scientifically credible assignment mechanisms, and how much one would have to depart from the assumed model to change the decision.

## Negative controls and falsification

Sensitivity analysis can be strengthened by additional measurements that constrain possible causal explanations. A negative-control outcome should not be affected by the treatment under the proposed mechanism but may share some confounding pathways. A treatment association with that outcome can indicate residual bias. A negative-control exposure is related to possible confounding but is not expected to affect the target outcome directly.

Such controls are not automatically valid. If the supposedly unaffected outcome can in fact respond to treatment, or if it does not share relevant confounding sources, its interpretation changes. Some negative-control approaches use explicit assumptions to identify effects or bias terms; others are primarily falsification diagnostics.

Our simulation has no negative-control variables. The hidden U is available to the experimenter but not to the imagined analyst, so its direct inclusion would not represent a real negative-control test. A useful extension would generate proxy variables with specified causal relationships and assess what could be learned about U from those proxies.

The essential distinction is between testing an observable implication of a causal model and validating every assumption required by that model. A successful placebo does not prove that no hidden confounding exists. Evidence from several designs can make a causal explanation more credible without producing certainty about every unobserved cause.

## Why robustness across regressions is limited

An empirical paper may report that its treatment coefficient hardly changes across several regressions using different measured covariates. That can be encouraging, particularly when the adjustments target known confounding pathways. But stability among models using the same incomplete information need not imply absence of hidden bias.

Imagine fitting four regressions that vary whether age, employment history or region is included while all omit an unrecorded prognosis U. If each specification leaves a similar residual relationship between U and treatment, all four can converge toward essentially the same biased coefficient. Agreement among them tests sensitivity to the included changes, not sensitivity to the omitted factor.

Conversely, a coefficient that moves when a measured variable is added is not always evidence of confounding removed. Part 5 of this series showed that conditioning on a collider can create bias, while conditioning on a mediator can change a total-effect estimand into a direct-effect comparison. A sensitivity analysis must therefore distinguish alternative causal assumptions from arbitrary adjustment-set expansions.

The experiment fixes W as a genuine pre-treatment common cause. Every observed-only regression controls W, and the oracle additionally controls U. This isolates the omitted-confounder question instead of blending it with changes in the causal meaning of the adjustment variables.

## Model specification and interactions

The current benchmark uses linear additive equations and a constant treatment effect. It permits a closed-form bias function, but real systems can be nonlinear and heterogeneous. An unobserved characteristic may affect assignment differently across observed subgroups, interact with treatment or influence outcome only above a threshold. A one-parameter linear sensitivity curve may not capture those possibilities.

If the causal effect varies across people, unmeasured selection can bias not only the estimated average effect but also the population for which it is implicitly calculated. An observational comparison might emphasize participants whose treatment choice is especially sensitive to U. Even if the sample contains many observations, the comparison may have limited support for interventions in other parts of the population.

Sensitivity methods therefore depend on the estimand and identification strategy. For matching, hidden bias can be represented by treatment odds differences within matched sets. For instrumental variables, sensitivity may quantify departures from exclusion. For difference-in-differences, one can bound violations of counterfactual parallel trends. For longitudinal effects, departures from sequential exchangeability involve a history of unmeasured confounding rather than a single baseline factor.

There is no universal sensitivity parameter that can be applied without modification to every causal design. The choice should follow the causal problem, not merely the convenience of a software package.

## What a confidence interval really covers

The observed-only regression intervals in this study have zero empirical coverage of the causal effect at positive λ. This does not show that the regression software computed its standard errors incorrectly. The intervals describe the uncertainty of an observed partial regression coefficient under the specified sampling model. Once the parameter differs from the causal target, the resulting intervals are not necessarily valid causal confidence intervals.

A more informative causal uncertainty report could combine sampling variation with sensitivity bounds over a clearly stated range of confounding strengths. The resulting region would be wider than a conventional confidence interval when meaningful unmeasured confounding is allowed. Its interpretation must specify whether the sensitivity assumptions are fixed, estimated or varied, and whether the region has a frequentist confidence guarantee or is a descriptive identification region.

Bayesian analyses face a related distinction. A posterior may concentrate sharply conditional on a causal model and a prior for unknown bias parameters. If the assumptions assigning that prior or excluding important confounding are not justified, posterior concentration alone does not establish causal truth. Part 14 investigated that problem using an observational Bayesian regression.

The point is not to avoid uncertainty intervals. It is to label their target accurately and distinguish sampling precision from uncertainty about the causal structure.

## Why larger samples do not remove hidden bias

Each replication uses 2,000 observations, enough to make the numerical bias easy to see. We could increase the sample size substantially and the observed-only estimate would concentrate more tightly around its population coefficient. Unless the assignment mechanism changed or U became measured, the asymptotic bias function would remain the same.

At λ=0.5, for example, the observed regression coefficient targets 1.6 even though the intervention effect is one. A much larger sample would estimate 1.6 more precisely; it would not recover the missing information about U. This is the same identification-versus-estimation distinction running through the series.

More data may still help identify a causal effect if they include new variables, new treatment variation, alternative policy regimes or other information that changes what assumptions are supportable. A larger number of otherwise identical confounded observations is a different proposition from a richer research design.

Researchers should therefore be cautious about claiming that observational studies overcome confounding primarily by scale. A million records can support detailed diagnostics, heterogeneous analyses and precise estimation of observable associations. It cannot alone show that the unrecorded assignment causes no longer matter.

## A practical sensitivity-analysis plan

A responsible sensitivity investigation begins with an explicit target estimand, population, intervention and identification strategy. It should name the assumptions most consequential for interpreting the observed estimator causally. In a regression-adjusted observational study, conditional exchangeability and measurement of the relevant pre-treatment common causes may be central.

The next step is to select a sensitivity parameter that maps to a plausible failure of those assumptions. The parameter should have a clear interpretation in treatment and outcome relationships, residual explanatory power or treatment odds. Where possible, calibrate its range to observed covariates and independent substantive evidence. Explore both signs when appropriate.

For each assumed confounding strength, calculate the implied causal estimate or identification region, not merely a new p-value. Report tipping points for scientifically relevant decisions and describe which conclusions remain stable across the pre-specified range. Sampling uncertainty should be represented separately or incorporated transparently into the sensitivity procedure.

Negative controls, external validation and alternate designs can be used to constrain interpretations. But their own assumptions must be assessed. A sensitivity figure should make uncertainty more visible, not replace one unjustified assertion with another.

The value of the exercise is clarity: readers can see what assumptions support the claim, what would overturn it, and whether those departures are plausible in the substantive setting.

## Limits of the simulation

The benchmark is deliberately narrow. It uses one measured covariate W, one hidden covariate U, continuous treatment, Gaussian independent disturbances and a constant additive causal effect. The relationship between treatment and outcome is linear, and W and U are independent. The hidden outcome coefficient is fixed at 1.5 while the hidden assignment coefficient is varied over four nonnegative values.

The oracle estimator observes U without measurement error and fits the exact structural outcome model. Real observational analyses rarely enjoy that privilege. The experiment does not compare formal Rosenbaum sensitivity bounds, E-values, partial-R² robustness values or Bayesian priors on hidden confounding. It illustrates the underlying mechanism those methods address, rather than claiming to implement them.

Only one sample size is studied, with 400 independent datasets per confounding strength. The Monte Carlo standard errors and coverage summaries are appropriate to this known Gaussian design, but they do not establish uncertainty calibration in longitudinal, clustered or nonlinear settings.

The bias is positive by design because U raises both treatment and outcome. Negative or interacting confounding mechanisms could give different bias curves. The experiment therefore cannot tell researchers what strength, sign or form of unmeasured confounding is plausible in a real dataset.

Most importantly, sensitivity analysis does not prove the original causal effect is correct or incorrect. It quantifies what follows from alternative assumptions and can show that a conclusion is fragile. Its scientific value depends on the credibility of the alternatives considered.

## What the experiment established

When U did not affect treatment assignment, the regression adjusting for observed W estimated the true effect closely: mean 1.0013 with approximately 95% causal coverage. When U began to affect assignment, the observed-only coefficient moved away from the causal effect in precisely the direction and magnitude predicted by the omitted-variable-bias formula. At λ=0.25 it averaged 1.3513, at λ=0.5 it averaged 1.6001, and at λ=1 it averaged 1.7493.

Across all three nonzero confounding scenarios, none of the nominal 95% observed-only confidence intervals contained the true treatment effect in 400 simulated datasets per scenario. Yet the reported standard errors were small and fell to about 0.023 at the strongest assignment coefficient. The oracle model, which included U, stayed near the true effect of one and retained roughly nominal coverage.

These results do not imply that every observed estimate has a hidden bias of this magnitude. They show that a reasonable-looking regression table does not reveal how sensitive its causal interpretation may be to omitted common causes. Sensitivity analysis supplies a language for asking how much those assumptions matter, rather than pretending they have been verified.

A causal conclusion deserves more than a narrow interval for an observational coefficient. It needs a clear account of the assignment process, the likely unobserved pathways and the strength of departures that would change the decision. Statistical precision is useful; knowing what it is precise about is indispensable.

## Code and reproducibility

The experiment in `causal_inference_statistics/part16/experiment.py` generates 400 independent datasets for each of four assignment mechanisms, fits the observed-only and oracle regressions, and writes `results.csv`. It uses Python 3.12+ and NumPy with fixed seed 20261018. No external data or causal package is needed. The population bias function is derived independently and included in the CSV as an additional verification check.

## The series

1. When Regression Gets the Causal Effect Wrong
2. Randomization, Confounding and the Value of Experimental Design
3. Matching and Propensity Scores
4. Inverse Probability Weighting
5. Directed Acyclic Graphs and the Cost of Controlling for Everything
6. Difference-in-Differences: When Parallel Trends Fail
7. Instrumental Variables
8. Regression Discontinuity
9. Synthetic Controls
10. Doubly Robust Estimation
11. The G-Formula
12. Heterogeneous Treatment Effects
13. Double Machine Learning
14. Bayesian Causal Inference
15. Time-Varying Treatments and Marginal Structural Models
16. Sensitivity Analysis for Unmeasured Confounding (this article)

## References

Cinelli, C. and Hazlett, C. (2020). Making Sense of Sensitivity: Extending Omitted Variable Bias. *Journal of the Royal Statistical Society: Series B*, 82(1), 39–67. https://doi.org/10.1111/rssb.12348

Cornfield, J., Haenszel, W., Hammond, E. C., Lilienfeld, A. M., Shimkin, M. B. and Wynder, E. L. (1959). Smoking and Lung Cancer: Recent Evidence and a Discussion of Some Questions. *Journal of the National Cancer Institute*, 22(1), 173–203.

Ding, P. and VanderWeele, T. J. (2016). Sensitivity Analysis Without Assumptions. *Epidemiology*, 27(3), 368–377. https://doi.org/10.1097/EDE.0000000000000457

Greenland, S. (1996). Basic Methods for Sensitivity Analysis of Biases. *International Journal of Epidemiology*, 25(6), 1107–1116. https://doi.org/10.1093/ije/25.6.1107-a

Hernán, M. A. and Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC. https://miguelhernan.org/whatifbook

Imbens, G. W. (2003). Sensitivity to Exogeneity Assumptions in Program Evaluation. *American Economic Review*, 93(2), 126–132. https://doi.org/10.1257/000282803321946921

Oster, E. (2019). Unobservable Selection and Coefficient Stability: Theory and Evidence. *Journal of Business & Economic Statistics*, 37(2), 187–204. https://doi.org/10.1080/07350015.2016.1227711

Rambachan, A. and Roth, J. (2023). A More Credible Approach to Parallel Trends. *Review of Economic Studies*, 90(5), 2555–2591. https://doi.org/10.1093/restud/rdad018

Rosenbaum, P. R. (2002). *Observational Studies*, 2nd edition. Springer. https://doi.org/10.1007/978-1-4757-3692-2

Rosenbaum, P. R. and Rubin, D. B. (1983). Assessing Sensitivity to an Unobserved Binary Covariate in an Observational Study with Binary Outcome. *Journal of the Royal Statistical Society: Series B*, 45(2), 212–218. https://doi.org/10.1111/j.2517-6161.1983.tb01242.x

VanderWeele, T. J. and Ding, P. (2017). Sensitivity Analysis in Observational Research: Introducing the E-Value. *Annals of Internal Medicine*, 167(4), 268–274. https://doi.org/10.7326/M16-2607

VanderWeele, T. J., Mathur, M. B. and Chen, Y. (2020). Outcome-Wide Longitudinal Designs for Causal Inference: A New Template for Empirical Studies. *Statistical Science*, 35(3), 437–466. https://doi.org/10.1214/19-STS728

Lipsitch, M., Tchetgen Tchetgen, E. and Cohen, T. (2010). Negative Controls: A Tool for Detecting Confounding and Bias in Observational Studies. *Epidemiology*, 21(3), 383–388. https://doi.org/10.1097/EDE.0b013e3181d61eeb

Tchetgen Tchetgen, E. J. (2014). The Control Outcome Calibration Approach for Causal Inference with Unobserved Confounding. *American Journal of Epidemiology*, 179(5), 633–640. https://doi.org/10.1093/aje/kwt303

Schneeweiss, S. (2006). Sensitivity Analysis and External Adjustment for Unmeasured Confounders in Epidemiologic Database Studies of Therapeutics. *Pharmacoepidemiology and Drug Safety*, 15(5), 291–303. https://doi.org/10.1002/pds.1200