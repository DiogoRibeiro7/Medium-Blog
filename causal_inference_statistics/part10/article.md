# Doubly Robust Estimation: When One Correct Model Is Enough—and When Neither Is

*Part 10 of Causal Inference in Statistics. An empirical investigation of outcome regression, inverse probability weighting and augmented inverse probability weighting under four combinations of correct and misspecified nuisance models. A known causal effect, 800 independently simulated datasets and reproducible Python code.*

The phrase *doubly robust* sounds more reassuring than it is. It describes an important mathematical property of a family of causal estimators: under suitable identification and regularity conditions, consistency can survive misspecification of either the treatment-assignment model or the outcome model, provided the other is correctly specified. It does not mean the effect estimate is immune to confounding, poor overlap, finite-sample error, or every kind of model failure. Nor does it mean that using two incorrect models is safer than using one.

This article investigates the property directly. I simulate an observational study in which treatment depends on a baseline characteristic, outcomes depend nonlinearly on that same characteristic, and the average treatment effect is exactly one. I fit outcome regression, normalized inverse probability weighting and augmented inverse probability weighting (AIPW) under four combinations: both nuisance models correctly specified, only the propensity model correct, only the outcome model correct, and both models misspecified. Each combination is evaluated on 200 independent datasets of 1,200 observations. Cross-fitting prevents each observation from being used both to train nuisance functions and to evaluate its own AIPW score.

The results separate robustness from efficiency. When both nuisance models are correct, AIPW estimates 0.9964 on average. When only the propensity model is correct, it averages 1.0182; when only the outcome model is correct, 0.9968. When both are misspecified, the AIPW mean collapses to 0.3942. The true effect is one in all four cases. These are finite-sample Monte Carlo results, not a universal ranking of estimators.

An additional result is worth reporting rather than hiding. The naive treated-minus-control contrast happens to be close to one in this simulation despite nonrandom treatment assignment. The assignment and nonlinear outcome relationships generate opposing contributions to selection bias that nearly cancel in the marginal mean. A correct-looking unadjusted estimate does not establish exchangeability, just as a biased-looking one does not identify which assumption failed. This cancellation is part of the investigation and a reminder that agreement with the truth in one simulated design must be interpreted through the data-generating process.

## The decision and the estimand

Suppose an employment service offers job-search support according to a rule using information about a person's initial circumstances. It wants to know the effect of providing the support, not merely whether those who received it performed differently from those who did not. Since support is not randomized, a comparison of outcomes by treatment status can be confounded by characteristics used in the allocation rule.

One approach models outcomes conditional on treatment and those characteristics. Another models the probability of treatment and reweights observed outcomes to construct a comparison population. Both approaches can work under appropriate assumptions, and both can fail when their statistical models do not represent the relevant conditional relationships. The question is whether combining them provides some protection against misspecification without creating a new, unexamined claim of validity.

Let A denote binary treatment, X measured baseline information, Y the observed outcome, and Y(1), Y(0) the potential outcomes under treatment and no treatment. The target is the population average treatment effect,

\[
\tau=E[Y(1)-Y(0)].
\]

Under consistency, positivity and conditional exchangeability, the observed data identify this contrast. Conditional exchangeability requires the potential outcomes to be independent of treatment after conditioning on the sufficient baseline covariates. Positivity requires both treatment options to be possible at the relevant covariate values. The statistical estimators examined here do not prove either assumption. They provide different ways to calculate the identified contrast *if* those assumptions are appropriate.

AIPW combines a model of treatment assignment with models of the conditional potential-outcome means. These are often called nuisance models because their parameters are not the causal quantity being reported. But nuisance does not mean unimportant: their accuracy influences bias, variation and the practical behaviour of the effect estimator.

## The outcome regression route

Define

\[
\mu_a(x)=E[Y\mid A=a,X=x], \qquad a\in\{0,1\}.
\]

Under the identification assumptions, the g-computation formula expresses the ATE as

\[
\tau=E[\mu_1(X)-\mu_0(X)].
\]

A practical outcome regression estimates the two conditional mean functions and then averages their predicted difference over the empirical distribution of X. This is straightforward when the regression models are appropriate. It also exposes the method's vulnerability: the population formula may be valid while the fitted conditional mean functions are incorrect.

Suppose treatment assignment is modelled adequately but the outcome is nonlinear in X and the analyst fits a linear model. The resulting conditional predictions may systematically miss important patterns, and the averaged contrast can be biased. Good performance at predicting the marginal outcome does not necessarily mean the two treatment-specific conditional functions are adequate where the causal comparison needs them.

The approach can nevertheless be efficient when conditional outcome relationships are well estimated. In many settings it makes better use of prognostic information than a purely weighted estimator. Its strengths and weaknesses therefore differ from those of inverse probability weighting.

## The weighting route

Let

\[
e(x)=P(A=1\mid X=x)
\]

be the propensity score. The familiar inverse probability identification equation is

\[
\tau=E\left[\frac{AY}{e(X)}-\frac{(1-A)Y}{1-e(X)}\right].
\]

The identity follows because weighting each observed treatment outcome by the inverse of the probability of its assignment reconstructs the conditional potential-outcome mean. The equation requires a suitable treatment model and sufficient overlap. The previous article on inverse probability weighting examined how nearly deterministic assignment can make the finite-sample estimator unstable even when the propensity score is correctly specified.

Here I use a normalized, Hájek-style estimate rather than the unnormalized population sum. Each treatment arm's weighted outcome total is divided by its total weight. This can stabilize finite-sample behaviour, although it does not eliminate the identification assumptions or all practical overlap problems. When the fitted propensity model is misspecified, weighting may fail to balance the confounding characteristic in the way required by the identification formula.

The distinction is important: the true propensity function enters the population identity, whereas an estimated probability is a statistical approximation. Cross-validation or a strong treatment-classification score does not make the approximation sufficient for the causal target automatically.

## Augmentation and the doubly robust score

AIPW begins with an outcome-model contrast and adds inverse-probability-weighted corrections for residuals in the observed treatment arms. For one observation the score is

\[
\psi(O;\eta)=
\mu_1(X)-\mu_0(X)
+\frac{A\{Y-\mu_1(X)\}}{e(X)}
-\frac{(1-A)\{Y-\mu_0(X)\}}{1-e(X)},
\]

where \(\eta=(e,\mu_0,\mu_1)\). The AIPW estimator averages the score after replacing nuisance functions with their fitted estimates.

If the outcome models are correct, the residual terms have conditional mean zero within treatment groups. The estimated propensity can then be wrong without necessarily changing the population target, provided weighting denominators remain well behaved. Conversely, if the propensity model is correct, the weighted residual corrections cancel the population errors in the outcome models. The score then recovers the causal contrast even if the initial outcome predictions are wrong.

That is the double robustness property. It is a statement about the expectation of the score at population nuisance limits, not an assurance that any finite dataset will give a reliable answer. When both nuisance functions are wrong, neither cancellation is guaranteed.

There is a useful algebraic way to see the issue. For a fixed X, let the true conditional means be \(m_1,m_0\) and the true treatment probability be \(p\), while the fitted limits are \(\widehat m_1,\widehat m_0,\widehat p\). The conditional expected score minus the true conditional effect is

\[
\left(1-\frac{p}{\widehat p}\right)(\widehat m_1-m_1)
-\left(1-\frac{1-p}{1-\widehat p}\right)(\widehat m_0-m_0).
\]

The expression vanishes when the outcome errors vanish. It also vanishes when \(\widehat p=p\). With both nuisance limits wrong, however, the product-like errors need not cancel. The practical error also includes finite-sample fluctuations and nuisance estimation uncertainty.

## Why cross-fitting is used

When flexible models estimate nuisance functions on the same observations used to evaluate the causal score, overfitting can distort the correction terms. Cross-fitting separates these operations. The data are split into folds, the nuisance models are fitted without one fold, and the held-out observations receive predictions from those fitted models. The procedure repeats across folds so that every observation contributes to the final effect estimate using out-of-fold nuisance predictions.

I use two folds in each simulated dataset. Both treatment and outcome nuisance models are fitted on one half and evaluated on the other, then the roles are exchanged. Propensity estimates are clipped to the numerical interval [0.01,0.99] for stability. This clipping is rarely consequential in the central region but remains a finite-sample design choice that should be disclosed.

Cross-fitting does not fix an inadequate covariate set, an incorrect intervention definition or structural violation of positivity. It changes how nuisance estimation noise interacts with the treatment-effect estimator. With sufficiently accurate nuisance learners and regularity conditions, cross-fitting can help deliver asymptotic linearity and valid inference. This simulation focuses on finite-sample point estimation rather than claiming to establish asymptotic coverage.

## Constructing the experiment

The measured confounder is a standard normal covariate X. Treatment is generated according to a nonlinear logistic assignment rule,

\[
e(X)=\operatorname{expit}(-0.3+0.8X-0.45X^2).
\]

The potential outcomes are

\[
Y(0)=1+0.7X+0.8X^2+\varepsilon,
\]

\[
Y(1)=Y(0)+1,
\]

with independent standard normal disturbance \(\varepsilon\). Every individual has causal effect one. The observed outcome is the potential outcome corresponding to the treatment actually assigned.

The same X enters the treatment assignment and outcome equations, so adjustment for X is required by the chosen identifying strategy. The functional forms differ: the propensity is logistic with a quadratic index, and the conditional outcome mean is quadratic. Both correct specifications are within standard parametric model classes.

For the propensity model, the correct specification includes X and its square; the misspecified model includes X alone. For each outcome regression, the correct specification also includes X and its square, while the misspecified alternative includes X alone. The treatment-specific outcome models are fitted separately using ordinary least squares. Propensity models are fitted using logistic regression with negligible practical regularization.

There are four regimes: both correct, propensity correct but outcome wrong, propensity wrong but outcome correct, and both wrong. For each regime I generate 200 independent datasets of size 1,200. The random generator uses documented deterministic seeds, and each model is re-estimated in every replication.

The design deliberately creates a clear failure of a linear-only outcome model because it omits a positive quadratic term. The propensity misspecification similarly omits a quadratic assignment term. The amount of bias will depend on the coefficients and distributions; these are not universal misspecification penalties.

## A baseline that looks unexpectedly good

The naive treated-control difference averages between 1.0065 and 1.0327 across the four regimes, despite treatment depending on X. This is not a design-based guarantee of an unbiased comparison. It reflects the chosen coefficients, where different components of the distribution of X among treated and untreated people contribute selection differences that nearly offset in the mean outcome.

The naive estimator's RMSE is approximately 0.10 across regimes, even though its selection mechanism is not randomized. This apparent success is an excellent warning against evaluating causal assumptions by whether an estimate seems plausible. Without knowing the true effect, the analyst would have no reason to regard a naive coefficient close to one as evidence that omitted nonlinear covariate relationships had cancelled.

Nor should a more complex adjusted method automatically be declared inferior whenever it has greater RMSE in a particular simulation. The comparison is conditional on the chosen data-generating mechanism. A different outcome coefficient, treatment-assignment strength or covariate distribution can remove the cancellation and produce substantial naive bias.

I retain the naive estimator in the tables because a serious benchmark should report simple baselines even when they complicate the narrative. The experiment is intended to test identification and estimation mechanisms, not to manufacture a ranking favourable to a method's name.

## Results when both models are correct

The mean estimates and RMSE across 200 independent datasets are:

| Method | Mean estimate | RMSE |
| --- | ---: | ---: |
| Naive difference | 1.0065 | 0.1026 |
| Outcome regression | 0.9980 | 0.0783 |
| Normalized IPW | 0.9918 | 0.1556 |
| Cross-fitted AIPW | 0.9964 | 0.1081 |

The true effect is one. Every adjusted method has small Monte Carlo mean bias in this regime. Outcome regression has the lowest RMSE, because the correctly specified quadratic conditional mean takes advantage of a strong prognostic relationship. Normalized IPW has the largest RMSE of the four despite its correct propensity model. This does not contradict consistency; the estimators use different information and have different finite-sample variance.

AIPW is not automatically the most efficient implementation in every small sample. Its residual correction terms introduce inverse-propensity contributions, and cross-fitting means each nuisance regression uses only part of the data. Under the specified outcome model, ordinary outcome regression is particularly well matched to the structural equation.

The result supports the double robustness principle but does not show why it is valuable until one nuisance model is made wrong. With both functions correct, several estimators already work well.

## When only the propensity model is correct

The outcome regressions now exclude the quadratic term. The propensity regression still uses the correct nonlinear assignment specification.

| Method | Mean estimate | RMSE |
| --- | ---: | ---: |
| Naive difference | 1.0076 | 0.1068 |
| Outcome regression | 0.4567 | 0.5521 |
| Normalized IPW | 0.9974 | 0.1417 |
| Cross-fitted AIPW | 1.0182 | 0.2251 |

Outcome regression becomes severely biased, estimating less than half of the true treatment effect on average. The quadratic term is relevant to the conditional outcome means and its omission changes the fitted counterfactual predictions.

Normalized IPW remains close to one because the propensity model represents treatment assignment correctly. AIPW also remains close to one: the weighted residual correction repairs the systematic outcome-regression error in the population identity.

But AIPW's RMSE is 0.2251, larger than normalized IPW's 0.1417 and much larger than its RMSE when both nuisance models were correct. Double robustness preserves a consistency property, not the variance that would be obtained from a correctly specified and efficiently estimated outcome model. AIPW's correction can become noisy when it compensates for large outcome-model errors.

The result also highlights why selection between estimators should depend on the analysis objective and anticipated nuisance-estimation performance. An estimator may be robust to one type of misspecification yet give a wider sampling distribution in a finite dataset.

## When only the outcome model is correct

Now the outcome regressions include both X and its square, while the propensity model is fitted using X alone.

| Method | Mean estimate | RMSE |
| --- | ---: | ---: |
| Naive difference | 1.0135 | 0.0980 |
| Outcome regression | 0.9969 | 0.0740 |
| Normalized IPW | 0.5196 | 0.4893 |
| Cross-fitted AIPW | 0.9968 | 0.0747 |

The misspecified propensity model produces a substantial error for the weighting-only estimator. Fitted inverse assignment weights no longer reproduce the population contrast adequately. By contrast, correct treatment-specific outcome regressions recover the effect, and AIPW remains extremely close to their estimate.

The near-identical RMSE of outcome regression and AIPW in this regime is informative. When outcome residuals are appropriately centered, the augmentation terms do not introduce a large systematic correction even though the weighting function is wrong. Their finite-sample contribution remains relatively modest in this design.

This is the other half of the double robustness property. The estimator does not need both nuisance limits to be correct for consistency. It requires at least one appropriate route through the identification formula and the regularity conditions under which the correction behaves well.

The strength of the example is that the misspecified propensity-only estimate is visibly wrong while AIPW is not. It demonstrates a robustness property numerically rather than asserting it because of a familiar label.

## When both models are wrong

Finally, omit the quadratic term from both assignment and outcome regressions.

| Method | Mean estimate | RMSE |
| --- | ---: | ---: |
| Naive difference | 1.0327 | 0.1080 |
| Outcome regression | 0.4808 | 0.5264 |
| Normalized IPW | 0.5297 | 0.4793 |
| Cross-fitted AIPW | 0.3942 | 0.6151 |

Neither route used by the augmented estimator is reliable. Its mean estimate is farther from one than the individual misspecified estimators' means, and its RMSE is 0.6151. The augmentation terms cannot be expected to cancel the outcome-model error when the inverse assignment probabilities are also wrong.

This result makes the limitation of double robustness concrete: the procedure is not 'twice as safe' in any general sense. Its mathematical protection is conditional on one nuisance component being consistent and on the identifying assumptions being valid. When both are wrong, the estimator can amplify rather than repair systematic errors.

The naive estimator again looks relatively good. Its accidental cancellation should not be confused with the structural reliability of a randomized comparison, nor should it lead us to claim that adjustment caused the true identification problem. The parametric restrictions deliberately misspecify two important conditional functions. An alternative correctly specified model or defensible nonparametric procedure would produce different adjusted results.

An applied researcher does not know which regime generated the data. That is exactly why sensitivity analysis, model diagnostics, substantive covariate selection and careful evaluation of nuisance functions matter.

## Comparing the four regimes

A summary of AIPW across regimes is:

| Propensity model | Outcome model | Mean AIPW | RMSE |
| --- | --- | ---: | ---: |
| Correct | Correct | 0.9964 | 0.1081 |
| Correct | Wrong | 1.0182 | 0.2251 |
| Wrong | Correct | 0.9968 | 0.0747 |
| Wrong | Wrong | 0.3942 | 0.6151 |

There is no contradiction between double robustness and the much larger RMSE when only the propensity model is correct. The property concerns consistency under suitable asymptotic conditions; RMSE is a finite-sample measure combining bias and variability. A correctly specified propensity model does not guarantee that its estimated values provide stable corrections in a particular sample.

Likewise, the result with a correct outcome model is especially favourable because its functional form matches the data-generating equation. In more complicated applications outcome regressions can be hard to estimate, treatment effects can vary and covariate dimension can be large. AIPW's practical behaviour may depend substantially on nuisance learners, tuning and the distribution of treatment probabilities.

The difference between *model correctness* and *estimator quality* deserves emphasis. A method may be consistent under a theoretical model and have a poor finite-sample distribution. A misspecified estimator may look numerically good in one simulation because biases cancel. Neither result should be generalized without examining the generating process and the causal estimand.

## AIPW as an orthogonal score

The score used here has a deeper property than the elementary either-model-correct identity. Near the correct nuisance functions, its population expectation is locally insensitive to small first-order perturbations in those functions. This is the Neyman-orthogonality principle used in modern semiparametric estimation and double machine learning.

In practical terms, nuisance-estimation errors can enter the effect estimator through higher-order product terms rather than through unrestricted first-order shifts. If both nuisance estimators converge sufficiently quickly in appropriate norms and regularity conditions hold, the resulting causal estimator can have a root-n limiting distribution even when the nuisance functions are estimated flexibly.

The mathematical rates matter. Statements that 'machine learning plus AIPW gives valid inference' are incomplete. Some learners can converge slowly, some function classes are too complex for naive empirical-process arguments, and weak overlap can magnify errors through small propensity denominators. Cross-fitting helps manage overfitting and dependence, but does not remove every condition.

Our experiment deliberately uses parametric linear and logistic regressions rather than a complex learning library. The objective is to make the doubly robust correction identifiable in numerical results. A later article will examine flexible learners and orthogonalization more directly.

## Why estimated propensities matter

The propensity-score model is fitted, not supplied with its known true coefficients. This is important because a result based on oracle probabilities would exaggerate how much information the analyst normally has. In every replication the logistic regression must learn the assignment relationship from a finite dataset.

A correctly specified propensity model may still produce uncertain estimates in sparsely represented regions. Inverse-probability terms become more variable as treatment probabilities move toward zero or one. An estimated propensity near a boundary can amplify outcome residuals, and clipping changes the mathematical weighting rule.

The present assignment mechanism avoids structural positivity violations; both treatment options remain possible for every finite X. Still, practical support in tails must be evaluated rather than assumed. The AIPW score contains the same inverse propensity terms that made Part 4's weighting estimators sensitive to overlap.

Doubly robust does not mean positivity robust. Neither a correct outcome model nor the augmentation identity licenses an intervention contrast that is completely unsupported in relevant regions without some form of extrapolation or a revised estimand.

## The role of covariate selection

The experiment provides the analyst with exactly the confounding variable X. The challenge is therefore functional-form misspecification, not an omitted common cause. In real observational data, the more fundamental problem may be that important determinants of treatment and outcome were never measured.

If an unobserved U affects assignment and potential outcomes, the conditional exchangeability assumption may fail even when the fitted propensity and outcome models are statistically perfect conditional on measured X. Double robustness does not manufacture the missing information. It offers two estimation routes for a causal quantity already identified under the stated assumptions.

This connects directly with the matching experiment in Part 3. Excellent balance on measured variables did not remove hidden confounding. The same limitation applies here. The propensity and outcome nuisance models should be built on covariates justified by the causal design, not selected solely for cross-validated prediction.

Variables measured after treatment can also be inappropriate. Including mediators or colliders in nuisance models may change what is conditioned on or create new selection paths. The graphical reasoning of Part 5 remains relevant even when the final estimator has a doubly robust score.

## The difference between accuracy and identification

The result is a reminder to distinguish mathematical identification, statistical estimation, and empirical validation. Identification asks whether the causal effect can be expressed as a functional of the observed distribution under assumptions such as conditional exchangeability. Estimation asks how to calculate that functional from finite data. Validation asks what the available evidence can reveal about model performance and assumptions.

AIPW addresses an estimation vulnerability: errors in one nuisance model need not destroy consistency when the other model is correct. It cannot itself determine whether the specified intervention is meaningful, whether all confounders are measured, or whether the target population has sufficient support for both treatments.

In the simulation the identification assumptions hold by construction. That permits a clean test of statistical robustness. In a real study the assumptions would have to be argued from treatment assignment, timing, subject-matter knowledge and observational design. The statistical procedure does not replace that scientific argument.

A small standard error or narrow confidence interval would quantify one aspect of estimator uncertainty conditional on the design. It would not measure uncertainty about omitted confounders or an incorrectly defined treatment. Treating statistical precision as a certificate of causal validity would repeat the error identified in the first article of the series.

## Why a single benchmark is not enough

The model has one continuous measured confounder, a constant additive treatment effect, independent errors and a quadratic outcome relationship. The treatment assignment mechanism is logistic with a quadratic index. These choices are deliberately simple, but the numerical values depend on them.

Changing the coefficient of X squared in either equation can alter the magnitude and direction of misspecification bias. Changing the treatment-assignment strength can alter overlap and effective information. Introducing heterogeneous treatment effects can make the estimand and choice of conditional outcome specification more consequential.

The Monte Carlo results also depend on sample size. Here there are 1,200 individuals per dataset and two-fold cross-fitting. Nuisance regressions are trained on roughly half the data. With larger samples some finite-sample errors may shrink; with smaller samples the variance of weighted residual corrections may become more pronounced.

Only one random seed sequence and 200 replications per regime are used. These provide an empirical comparison for this design, not a proof of universal finite-sample ranking. The population identities supply the theoretical double robustness result, while the simulation shows how that result manifests with estimated nuisance models and finite samples.

No confidence interval procedure is benchmarked here. I report distributions of point estimates through mean, standard deviation, RMSE and quantiles. A fuller analysis would compare estimated influence-function standard errors and empirical coverage under each nuisance regime, while accounting appropriately for cross-fitting and the regularity conditions that justify asymptotic inference.

## What a serious applied analysis should report

First, define the treatment and potential outcomes, the population, and the causal contrast. State why measured covariates are sufficient for exchangeability and inspect whether the proposed interventions have overlap throughout the relevant population.

Second, describe both nuisance models. Report which variables and transformations enter the propensity and outcome functions, how hyperparameters are chosen if flexible learners are used, and whether predictions are out of fold. A statement that the final method is 'doubly robust' should not replace those details.

Third, investigate propensity distributions and residual behaviour. Extreme scores can dominate augmented corrections. Outcome model diagnostics can reveal functional patterns missed by the fit, although passing a diagnostic does not prove causal assumptions.

Fourth, compare meaningful alternatives. Outcome regression, weighting and AIPW need not agree in finite samples. Disagreement can signal nuisance misspecification, poor overlap, sample variability or a change in estimand. It should trigger investigation rather than an automatic choice of whichever result appears most persuasive.

Finally, report sensitivity to causal assumptions as well as statistical models. Hidden confounding, interference, selection and treatment-definition ambiguity remain outside the algebraic guarantee of double robustness. A scientifically careful analysis makes the boundary of the guarantee explicit.

## Limits

The experiment uses known, independent data-generating equations and no external empirical sample. All confounders are measured, treatment effects are constant and potential outcomes are additive. These are favourable conditions for identifying the ATE.

The misspecification regimes omit one quadratic feature from parametric nuisance models. They do not cover arbitrary misspecification, high-dimensional learners, severe lack of overlap, time-varying treatment, measurement error or clustered observations. The logistic regression uses very weak regularization; the outcome regressions are fitted separately by treatment arm.

The cross-fitted AIPW estimator uses two folds and propensity clipping at 0.01 and 0.99. Its Monte Carlo RMSE includes nuisance-estimation variability but does not quantify the uncertainty of a single dataset's reported effect through an estimated confidence interval. The naive result benefits from numerical cancellation in this chosen design and is not evidence of valid unadjusted identification.

The benchmark therefore illustrates the double robustness property and its limitations, but not all the challenges of contemporary semiparametric causal estimation. A later part will investigate flexible nuisance learners and inference more thoroughly.

## What the experiment established

Across 800 independently generated datasets, AIPW remained near the true effect of one when both nuisance models were correct and when either the propensity or outcome model was correctly specified. Its mean estimates were 0.9964, 1.0182 and 0.9968 in those three regimes. When both nuisance models were misspecified, its mean fell to 0.3942 and its RMSE rose to 0.6151.

The individual nuisance-based estimators failed in the expected complementary regimes: outcome regression was badly biased when its conditional mean model omitted nonlinear structure, and normalized IPW was badly biased when its treatment model was misspecified. AIPW could correct either problem in the population sense, but not both simultaneously.

The most useful conclusion is therefore conditional. Doubly robust estimation offers a genuine mathematical protection against one source of nuisance-model misspecification when the causal effect is identified and the required regularity conditions hold. It does not establish the identifying assumptions, guarantee efficient finite-sample performance or rescue two systematically wrong nuisance models.

That is a stronger and more useful statement than promising that two models are better than one. The next article will examine g-computation in greater detail, including how its identification formula is connected to longitudinal interventions and how outcome models can fail when extrapolation is mistaken for observed counterfactual information.

## Code

The companion `experiment.py` generates all 800 datasets, applies two-fold cross-fitting, fits treatment-specific outcome regressions and logistic propensity models, and writes `results.csv`. It uses Python 3.12+, NumPy, SciPy and scikit-learn. The data are synthetic; no external data source is required. The published results are the program's output, not manually selected estimates.

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
10. Doubly Robust Estimation: When One Correct Model Is Enough—and When Neither Is (this article)

## References

Bang, H. and Robins, J. M. (2005). Doubly Robust Estimation in Missing Data and Causal Inference Models. *Biometrics*, 61(4), 962–973. https://doi.org/10.1111/j.1541-0420.2005.00377.x

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W. and Robins, J. (2018). Double/Debiased Machine Learning for Treatment and Structural Parameters. *The Econometrics Journal*, 21(1), C1–C68. https://doi.org/10.1111/ectj.12097

Funk, M. J., Westreich, D., Wiesen, C., Stürmer, T., Brookhart, M. A. and Davidian, M. (2011). Doubly Robust Estimation of Causal Effects. *American Journal of Epidemiology*, 173(7), 761–767. https://doi.org/10.1093/aje/kwq439

Hernán, M. A. and Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC. https://miguelhernan.org/whatifbook

Kang, J. D. Y. and Schafer, J. L. (2007). Demystifying Double Robustness: A Comparison of Alternative Strategies for Estimating a Population Mean from Incomplete Data. *Statistical Science*, 22(4), 523–539. https://doi.org/10.1214/07-STS227

Kennedy, E. H. (2016). Semiparametric Theory and Empirical Processes in Causal Inference. In *Statistical Causal Inferences and Their Applications in Public Health Research*, 141–167. https://doi.org/10.1007/978-3-319-41259-7_8

Lunceford, J. K. and Davidian, M. (2004). Stratification and Weighting via the Propensity Score in Estimation of Causal Treatment Effects: A Comparative Study. *Statistics in Medicine*, 23(19), 2937–2960. https://doi.org/10.1002/sim.1903

Newey, W. K. and Robins, J. M. (2017). Cross-Fitting and Fast Remainder Rates for Semiparametric Estimation. CeMMAP Working Paper CWP41/17, Institute for Fiscal Studies. https://doi.org/10.1920/wp.cem.2017.4117 (revised preprint: https://arxiv.org/abs/1801.09138).

Robins, J. M. and Rotnitzky, A. (2001). Comment on the Bickel and Kwon Article, 'Inference for Semiparametric Models: Some Questions and an Answer'. *Statistica Sinica*, 11, 920–936.

Robins, J. M., Rotnitzky, A. and Zhao, L. P. (1994). Estimation of Regression Coefficients When Some Regressors Are Not Always Observed. *Journal of the American Statistical Association*, 89(427), 846–866. https://doi.org/10.1080/01621459.1994.10476818

Rosenbaum, P. R. and Rubin, D. B. (1983). The Central Role of the Propensity Score in Observational Studies for Causal Effects. *Biometrika*, 70(1), 41–55. https://doi.org/10.1093/biomet/70.1.41

Scharfstein, D. O., Rotnitzky, A. and Robins, J. M. (1999). Adjusting for Nonignorable Drop-Out Using Semiparametric Nonresponse Models. *Journal of the American Statistical Association*, 94(448), 1096–1120. https://doi.org/10.1080/01621459.1999.10473862

Tsiatis, A. A. (2006). *Semiparametric Theory and Missing Data*. Springer.

van der Laan, M. J. and Rose, S. (2011). *Targeted Learning: Causal Inference for Observational and Experimental Data*. Springer. https://doi.org/10.1007/978-1-4419-9782-1

van der Vaart, A. W. (1998). *Asymptotic Statistics*. Cambridge University Press.