# Bayesian Causal Inference: When a Precise Posterior Answers the Wrong Question

*Part 14 of Causal Inference in Statistics. A conjugate Bayesian investigation of omitted-variable bias, regularization, posterior uncertainty and the difference between estimating an observational slope and identifying an intervention effect. Five thousand four hundred fitted posteriors, with reproducible Python code.*

A Bayesian analysis can attach a probability distribution to an unknown parameter. That distribution often seems more informative than a coefficient accompanied by a standard error: it makes assumptions about the parameter explicit, provides posterior probabilities for hypotheses and permits decisions to be framed through expected loss. Yet none of those advantages automatically changes the relationship between observational data and causal interventions. A posterior can become arbitrarily concentrated around a parameter that does not represent the effect of an intervention.

The distinction is important because causal questions are commonly presented as estimation questions. An analyst asks whether additional treatment changes an outcome, chooses a likelihood, assigns priors and reports a posterior mean and credible interval. If the likelihood describes an observational association created partly by confounding, Bayesian updating estimates that association. The posterior may be mathematically coherent, computationally exact and scientifically unsuitable for the proposed causal interpretation.

This article investigates the problem with a deliberately transparent experiment. I generate data from a structural model in which the treatment effect is exactly one, while an unobserved common cause makes the unadjusted observational slope equal two. I fit conjugate Gaussian Bayesian regressions with and without adjustment for the common cause, using three degrees of prior regularization and three sample sizes. I repeat each configuration on 300 independent datasets. The resulting 5,400 posteriors expose both the advantage and the limitation of Bayesian estimation.

At 5,000 observations, the weak-prior unadjusted posterior has mean approximately 2.0008 and average posterior standard deviation 0.0173. None of its nominal 95% credible intervals contains the intervention effect of one in the 300 replications. The correctly adjusted posterior has mean 1.0011 and average posterior standard deviation 0.0141; 96.7% of its 95% credible intervals contain the true effect. The difference does not come from using a Bayesian estimator for one model and a frequentist estimator for the other. Both use the same conjugate framework. The difference is whether the conditioning information suffices to identify the causal quantity.

The experiment also produces a warning about prior sensitivity. Strong regularization can pull an unadjusted slope towards one in a small sample, but as the sample grows the same prior loses influence and the posterior moves towards the confounded observational coefficient. In the correctly adjusted analysis a strong prior can itself introduce finite-sample bias. Prior information is scientifically valuable when justified; accidental numerical proximity to a known causal effect is not evidence that the prior repaired identification.

## The decision behind the posterior

Suppose a service evaluates whether increasing the intensity of an intervention improves outcomes. Historical data contain the amount of intervention each person received and the subsequent outcome. Some people receive more support because caseworkers expect them to have worse prospects. An observational regression might therefore find that people receiving more support have worse outcomes even if additional support helps everyone.

A Bayesian analyst can use that same regression as a likelihood and choose a prior distribution for the coefficient of support. The posterior will then express uncertainty about the conditional or marginal relationship described by that likelihood. Without an argument connecting that relationship to the consequence of assigning more support, it will not reveal the intervention effect merely because uncertainty is expressed as a probability distribution.

This is not a criticism of Bayesian probability. It is a distinction between the question posed to a model and the scientific question that motivated the study. The likelihood specifies what the data are assumed to tell us. The causal model specifies what an intervention changes and which parameters or functionals answer the counterfactual question. Posterior inference operates conditional on those choices.

The distinction remains if an analyst uses a hierarchical model, Gaussian process or Bayesian neural network instead of linear regression. Greater flexibility can improve the representation of observed relationships. Whether those relationships identify the consequences of intervention is a separate requirement. An elaborate posterior over a misspecified causal structure can be more persuasive in appearance without being closer to the intended causal quantity.

The example here is synthetic rather than a real service evaluation. That decision makes the intervention effect observable to the investigator through the generating equations. It lets us calculate the population regression slope, fit models repeatedly and compare their posterior uncertainty with the known counterfactual truth. It does not quantify the efficacy of any particular intervention.

## Potential outcomes and the structural parameter

Let X be a continuous treatment and U an unobserved baseline common cause. For each individual, generate independent standard normal U, ε_X and ε_Y. Define treatment and outcome through

\[
X=U+\varepsilon_X,
\qquad
Y=X+2U+\varepsilon_Y.
\]

The structural intervention is well defined: setting X to a chosen value x replaces the treatment equation while leaving U and the outcome disturbance unchanged. The potential outcome is

\[
Y(x)=x+2U+\varepsilon_Y.
\]

Consequently, every individual has the same effect of increasing treatment by one unit,

\[
Y(x+1)-Y(x)=1.
\]

The average intervention effect is also one. That quantity is the target of the causal investigation. It does not require computing a fitted coefficient, because the experiment was constructed with full knowledge of the structural mechanism.

The observed joint distribution, however, contains information about a different relationship. Since X=U+ε_X, its variance is two and its covariance with U is one. The marginal regression coefficient of Y on X is

\[
\beta_{\mathrm{obs}}=
\frac{\operatorname{Cov}(X,Y)}{\operatorname{Var}(X)}
=1+2\frac{\operatorname{Cov}(X,U)}{\operatorname{Var}(X)}=2.
\]

This is the observational slope, not the causal effect. The equations make clear why: X carries information about U, and U affects Y independently of X. Conditioning on an observed treatment value changes the distribution of the common cause, whereas intervening to set treatment does not.

Indeed, the jointly Gaussian model implies E[U|X=x]=x/2, giving E[Y|X=x]=2x. Under intervention, E[Y(x)]=x. The derivative of the observational conditional mean is two; the derivative of the interventional mean is one. Both are mathematically correct. The confusion begins when a parameter of the first relationship is labelled as the effect described by the second.

## Why a Bayesian likelihood does not change that difference

For the unadjusted regression, the exact conditional mean is 2X. The conditional residual combines 2(U−E[U|X]) and ε_Y. Because Var(U|X)=1/2, the conditional variance is 4(1/2)+1=3. Thus the observed data satisfy a Gaussian regression of Y on X with population slope two and residual variance three.

The adjusted conditional regression of Y on X and U has treatment coefficient one, coefficient two for U and residual variance one. In both cases the Gaussian likelihood is correctly specified for the particular *statistical* conditional distribution being fitted. Only the adjusted likelihood's treatment coefficient corresponds to the desired intervention effect under the structural assumptions.

This is an especially useful benchmark because it avoids blaming causal failure on a poorly fitted statistical likelihood. There is no nonlinear relationship that the linear model cannot capture. No optimization error is needed, and no unknown heteroskedasticity is concealed. The unadjusted model fits the observational conditional distribution correctly while identifying the wrong parameter for the intervention question.

Bayes' theorem combines prior and likelihood. It cannot infer an intervention relationship from observational association without assumptions that make the relevant causal functional identifiable. If two possible causal structures generate the same observed distribution but imply different interventions, observational likelihood information alone cannot distinguish their causal effects. A prior may assign different probabilities to the structures, but then the resulting causal conclusion depends on those prior assumptions.

This dependence is legitimate when priors encode defensible external evidence. It must remain visible. A causal conclusion driven by structural restrictions in a prior is different from one learned from treatment variation supported by the observed data. Posterior concentration is not independent confirmation of its own identifying assumptions.

## The conjugate posterior

Let D denote the design matrix, with a column of ones for the intercept, a treatment column X and, in the adjusted specification, a column for U. For a Gaussian outcome likelihood with known residual variance σ²,

\[
Y\mid\theta,D\sim N(D\theta,\sigma^2 I).
\]

Assign independent zero-centred Gaussian priors to the non-intercept coefficients, each with prior variance σ²/λ, where λ is a nonnegative precision penalty. The intercept receives a flat prior. The experiment uses strictly positive values of λ, and the finite-sample design matrices have sufficient rank for the posterior to be proper.

Writing P for a diagonal matrix whose first entry is zero and whose remaining entries equal λ, the posterior covariance and mean are

\[
V_\theta=\sigma^2(D^\top D+P)^{-1},
\qquad
m_\theta=(D^\top D+P)^{-1}D^\top Y.
\]

The treatment coefficient's marginal posterior is Gaussian, so its posterior standard deviation is the square root of the relevant diagonal entry of V_θ. Its central 95% credible interval is obtained by taking the posterior mean plus or minus approximately 1.96 posterior standard deviations.

Under the model and prior, that interval contains 95% posterior probability for the regression parameter. A causal interpretation requires the additional identification argument. In the simulation I can also compute *causal coverage*: the fraction of repeated posterior intervals that contain the structural effect of one. That is a frequentist repeated-sampling diagnostic applied to Bayesian intervals, not their Bayesian definition.

The choice of known σ² isolates the question of identification and prior shrinkage. In ordinary applied data, σ² would usually be estimated or assigned its own prior. A fully Bayesian analysis could integrate it out and obtain a Student-type marginal posterior under a suitable conjugate specification. Those modifications affect finite-sample uncertainty but do not equate a confounded slope with an intervention effect.

## What the prior actually says

The parameter λ measures how strongly coefficients are pulled toward zero. A smaller λ means a more diffuse coefficient prior; a larger λ means more concentrated prior belief around zero relative to the likelihood's noise scale. The three values are 0.1, 10 and 100.

Because the prior variance is σ²/λ, the same numerical precision penalty is not the same absolute prior variance in the two regressions. For the unadjusted model σ²=3; for the adjusted model σ²=1. This is deliberate: the penalty is expressed relative to the likelihood variance and the posterior solution has the transparent form of ridge regularization.

An intercept is not penalized. The treatment and, when included, U coefficients are penalized symmetrically. This matters: a strong prior shrinking the confounder's coefficient towards zero can shift the adjusted treatment coefficient, because treatment and confounder are correlated. Thus even a correctly identified model may show counterintuitive shrinkage effects when priors are imposed jointly on correlated coefficients.

The prior mean is zero, not the true effect one. This provides a meaningful stress test: prior information is not secretly chosen to encode the simulation's answer. At small sample sizes a strong prior can dominate enough of the posterior to reduce or enlarge apparent bias. At large sample sizes its fixed penalty becomes weaker relative to accumulated data, so the likelihood increasingly determines the result.

The experiment does not search for the λ that makes the result nearest to the true effect. An applied researcher cannot observe that truth, and choosing priors after inspecting which estimate looks most desirable would make the posterior's evidential meaning difficult to defend. Here the grid is fixed before analysing the simulations.

## The Monte Carlo design

I use sample sizes n=100, n=1,000 and n=5,000. At each size, 300 independent datasets are generated from the structural model. Both unadjusted and adjusted Bayesian regressions are fitted to each dataset, at each of the three prior precision values. This gives 3×300×2×3=5,400 fitted posteriors.

The pseudorandom generator is NumPy's default generator, fixed to seed 20261018. The code constructs the observed treatment and outcome once per replication and reuses those observations for all competing posterior fits. This makes their comparisons paired within a dataset, although the observations themselves are regenerated for the next replication.

For each configuration I record the posterior treatment mean, posterior standard deviation, and whether a central 95% credible interval contains the known causal effect. Across replications I report the mean treatment estimate, bias relative to one, root mean squared error, average posterior standard deviation and causal coverage.

I also record the fraction of replications whose marginal posterior probability that the treatment coefficient is positive exceeds 95%. This is a deliberately limited sign decision; it is not a general measure of causal accuracy. A confounded model can confidently assert a positive coefficient and still overstate the causal effect by a factor of two.

All matrices are small, and the posterior is computed analytically using linear algebra rather than a Markov chain Monte Carlo algorithm. That eliminates convergence diagnostics and sampling error from an approximate posterior as alternative explanations of the results. The analysis asks what the *exact posterior under the specified models* implies.

## The main comparison

With the weak prior λ=0.1, the Monte Carlo mean estimates are:

| Observations | Unadjusted posterior mean | Adjusted posterior mean | True causal effect |
| ---: | ---: | ---: | ---: |
| 100 | 2.0076 | 1.0156 | 1 |
| 1,000 | 2.0002 | 1.0023 | 1 |
| 5,000 | 2.0008 | 1.0011 | 1 |

The unadjusted posterior concentrates around the population observational slope, two. The adjusted posterior approaches the structural effect, one. Both show increasing stability as the sample grows, so consistency for the relevant statistical parameters is not in question.

The numerical difference is not simply that adjustment improves prediction. The unadjusted model has the correct marginal conditional expectation for Y given X. Its causal defect is that a change in observed X is associated with a change in U, whereas an intervention setting X leaves U's distribution unchanged. Adjustment conditions on the common cause to separate those two pathways.

The adjusted posterior is not automatically correct for all observational studies. It works here because U is fully observed in that specification, measured without error, sufficient to address confounding, and included with the correct functional relationship. The experiment explicitly supplies the information that many real observational studies lack.

The unadjusted model's posterior mean is highly reproducible and wrong for the intended target. A reader who sees only its mean and uncertainty interval could easily mistake statistical certainty for causal certainty. A rigorous report must name the estimand supported by the likelihood rather than rely on the familiar words 'effect' and 'probability'.

## Posterior concentration and causal coverage

The weak-prior posterior standard deviations and causal coverage rates are:

| Observations | Unadjusted SD | Unadjusted coverage | Adjusted SD | Adjusted coverage |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.1235 | 0.0% | 0.1016 | 95.3% |
| 1,000 | 0.0386 | 0.0% | 0.0316 | 96.7% |
| 5,000 | 0.0173 | 0.0% | 0.0141 | 96.7% |

The unadjusted posterior standard deviation falls by more than a factor of seven between n=100 and n=5,000, but the resulting credible intervals never include the known intervention effect. This is not evidence that Bayesian credible intervals are intrinsically invalid. They describe uncertainty about the regression coefficient defined by the unadjusted model, whose population value is two.

The adjusted intervals contain the causal effect at rates close to 95%, consistent with the likelihood, prior and data-generating process. Small deviations reflect finite Monte Carlo sampling and the influence of the weak but nonzero prior. These rates demonstrate repeated-sampling behaviour under one specified mechanism; they do not establish a universal Bayesian coverage theorem.

If we looked only at the marginal posterior probability of a positive regression slope, the distinction would disappear. All configurations in this experiment put more than 95% posterior probability above zero in every replication. The confounded model correctly detects a positive association but greatly exaggerates its causal magnitude.

For some decisions the difference between effects of one and two is decisive: an intervention might be profitable at an effect of two but not at one. A posterior sign probability therefore cannot replace a causal effect estimate or policy-value analysis. Posterior probabilities must refer to the quantity the decision actually needs.

## Why a strong prior can look helpful for the wrong reason

At n=100, the unadjusted posterior means are 2.0076, 1.9109 and 1.3319 for λ=0.1, 10 and 100. The strongest prior pulls the observational slope substantially toward zero, bringing it numerically closer to the causal truth of one. Its RMSE relative to one decreases from 1.0152 to 0.3478.

This could tempt a careless analysis to claim that Bayesian regularization solved confounding. But the prior was centred at zero, and the improvement arises from numerical shrinkage of the wrong coefficient. It is not based on information about the causal pathway or a justified external estimate of the treatment effect.

At n=1,000 the strongest-prior unadjusted mean is 1.9056; at n=5,000 it is 1.9810. The same prior no longer places the estimate near one. The data increasingly dominate, and the posterior moves back towards the observational population slope two. The apparent success at n=100 was a finite-sample coincidence of the prior direction and the size of omitted-variable bias.

This also demonstrates why evaluating priors solely by how close one fitted estimate lies to an expected result is dangerous. A prior can reduce mean squared error relative to an unknown causal parameter by accident. That does not make the identifying assumptions credible or guarantee the improvement persists in a different population, assignment mechanism or sample size.

There are legitimate settings in which strong prior evidence about a causal effect contributes to identification. For example, an earlier randomized experiment could inform a transport model, or scientific restrictions could define relationships among parameters. Such evidence should be documented as external information and sensitivity to its assumptions should be examined. It should not be disguised as information learned from an unidentifying likelihood.

## Strong priors can also harm a correctly adjusted model

The adjusted posterior is correctly specified for the causal coefficient under the constructed model, but it is not immune to prior-induced bias. At n=1,000 with λ=100, its mean is 1.0708 and its causal coverage is only 30.7%, despite an average posterior standard deviation of 0.0289. At n=5,000 the mean is 1.0196 and coverage increases to 72.7%, still far below the nominal 95%.

The direction of the shift may initially seem surprising because the coefficient priors are centred at zero. But X and U are correlated, and the posterior penalizes both their coefficients. Shrinking the large U coefficient of two can cause the treatment coefficient to absorb part of U's contribution to the outcome. The resulting treatment coefficient may move upward even though its own prior is centred at zero.

In the n=100 adjusted design, the strongest prior happens to yield mean 1.0030 and RMSE 0.0685, with coverage 94.7%. That favourable result is not monotonic with sample size or sufficient to claim the prior is well calibrated. Its behaviour depends on the joint parameter geometry and the data-generating process.

This is a useful reminder that 'informative prior' is not synonymous with 'more accurate posterior'. When parameters are correlated, restrictions on one parameter can shift the marginal distribution of another. A causal interpretation of a posterior coefficient requires understanding both the underlying structural identification and the dependencies introduced by the prior.

A complete applied analysis should therefore vary plausible priors, report the estimand, and distinguish posterior precision conditional on assumptions from sensitivity to those assumptions. One posterior interval cannot summarize uncertainty about every modelling decision used to construct it.

## Priors over parameters and priors over causal structures

The experiment assigns priors to regression coefficients while holding the causal graph fixed. This is a narrower setting than Bayesian model uncertainty. An analyst might instead place prior mass on different causal structures, such as a model with a confounder and another without, or on a range of strengths for an unmeasured common cause.

That approach can represent uncertainty about the causal mechanism, but it introduces its own identification issues. If two causal structures imply the same observed distribution, accumulating more of the same observational data need not distinguish them. The posterior odds can remain sensitive to structural priors, even as parameter uncertainty within each model becomes very small.

The phenomenon is not uniquely Bayesian. In any inferential framework, observationally equivalent causal structures cannot be separated without additional assumptions or evidence. A Bayesian analysis provides a coherent formal language for assigning probabilities to those possibilities, provided the priors and likelihoods are specified. It cannot guarantee that the probability distribution is empirically identified.

Sensitivity to latent confounding can be represented by parameterizing the relationship between treatment assignment and the unobserved common cause, then examining posterior treatment effects across plausible values. Such analysis can be valuable when assumptions are transparent and informed by outside evidence. But a concentrated posterior produced by fixing an unverified confounding parameter at zero should not be mistaken for empirical proof that confounding is absent.

The central scientific distinction is between uncertainty *within* a chosen causal model and uncertainty *about* that model. The Monte Carlo credible intervals quantify the first. The omitted-confounder comparison exposes the consequence of getting the second wrong.

## Identification cannot be outsourced to computation

Modern Bayesian workflows commonly use Hamiltonian Monte Carlo, variational inference or other approximate algorithms to estimate complex posteriors. Posterior predictive checks, effective sample sizes and convergence diagnostics are essential for ensuring that the algorithm adequately represents the specified statistical model.

None of those checks establishes that the model's parameters answer an intervention question. A well-mixed chain can sample the wrong causal target perfectly. A posterior predictive distribution can reproduce observed outcomes under the observational assignment mechanism while failing to predict outcomes under an intervention that changes that mechanism.

Here the posterior is calculated analytically, so there is no sampling convergence issue. That removes a common distraction. The wrong causal estimate arises despite exact inference for a correctly specified observational likelihood. Better computation would not alter it.

The appropriate sequence is to define the causal estimand, defend the identifying assumptions, choose a likelihood that represents the required observed quantities, specify priors and only then perform posterior computation. Each stage has separate failure modes. Successful numerical inference supports the later stage; it does not retrospectively validate the earlier causal assumptions.

A Bayesian model may be particularly useful once the causal estimand is identified, especially for hierarchical pooling, uncertainty propagation, partial pooling across subgroups, and decisions involving asymmetric costs. The critique is therefore not of Bayesian modelling, but of using posterior certainty as evidence for a causal interpretation that the design does not warrant.

## Bayesian g-computation and potential outcomes

One principled application of Bayesian modelling to causal inference is posterior g-computation. If appropriate conditional outcome models and exchangeability assumptions identify a causal mean, posterior draws of their parameters can be transformed into draws of intervention-specific outcome means.

For a single treatment and sufficient measured pre-treatment covariates L, identification under conditional exchangeability and positivity gives

\[
E[Y(a)]=\int E[Y\mid A=a,L=l]p(l)\,dl.
\]

A Bayesian analysis can fit the conditional outcome regression, integrate predictions over the target covariate distribution and propagate parameter uncertainty through the integral. Posterior distributions of the intervention means can then be compared, rather than interpreting a regression coefficient directly when its meaning differs from the desired causal contrast.

This approach remains conditional on the causal assumptions. If an important common cause is unobserved, the conditional outcome model may not identify the counterfactual mean. Priors can encode external information about hidden causes, but their contribution should remain explicit.

The previous article in the series examined longitudinal g-computation and time-varying treatment-confounder feedback. Bayesian versions of that analysis can propagate uncertainty through sequential models of covariates and outcomes. They do not remove the need for sequential exchangeability, consistency and appropriate support under the interventions being evaluated.

Bayesian inference is thus compatible with the full causal identification framework. It supplies a way to represent and combine uncertainty, not a replacement for identifying the target.

## Hierarchical models and treatment heterogeneity

Another strength of Bayesian causal analysis is the treatment of heterogeneous effects. Suppose effects vary across regions, hospitals or demographic groups. A hierarchical model can partially pool group-specific effects, reducing extreme estimates in small groups while retaining differences supported by data.

Such shrinkage can be valuable, especially when policy decisions require estimates for groups with limited observations. Yet the same principle seen in the present experiment applies: pooling depends on assumptions about the distribution of effects, and shrinking a causally confounded group comparison does not automatically identify the group's treatment effect.

The target may also change with the decision. An average treatment effect is not necessarily the quantity needed to select recipients under limited capacity. Part 12 of the series demonstrated that targeting according to baseline risk can be inferior to targeting according to treatment benefit, and that a model with better effect-prediction error need not always yield a better treatment-allocation policy.

Bayesian decision theory can express the relevant allocation problem through expected utility, accounting for costs, uncertainty and constraints. But the utility must be evaluated using causal consequences of treatment, not observational correlations treated as though they were treatment responses.

A posterior distribution is most useful when it is a posterior for the right scientific quantity. The extra modelling flexibility makes the identification requirement more important, not less.

## Posterior credible intervals and repeated-sampling coverage

The experiment reports repeated-sampling coverage because the intervention effect is known. This provides a diagnostic that is unavailable in most applied observational work. It should not be confused with the definition of a Bayesian credible interval.

A 95% credible interval assigns 95% posterior probability to a parameter region conditional on the stated likelihood and prior. A frequentist coverage probability considers the proportion of repeated datasets under a specified generating process for which the interval contains a fixed parameter. These are different probability statements. Under suitable conditions and diffuse priors, Bayesian intervals may have approximately nominal frequentist coverage, but that is not an identity across all models and priors.

The adjusted weak-prior intervals have approximately nominal coverage in this simulation because the likelihood and data-generating model match and the prior exerts little influence. The strong-prior adjusted intervals can have poor causal coverage because shrinkage displaces the posterior centre. The unadjusted intervals have zero causal coverage because they are intervals for a different regression coefficient.

A Bayesian analyst need not demand nominal repeated-sampling coverage for every posterior statement. It is nonetheless useful to investigate the frequentist operating characteristics of a proposed decision procedure under plausible data-generating mechanisms, particularly when coverage failures would affect safety or public policy decisions.

This distinction is especially important when model uncertainty is ignored. A narrow credible interval conditional on one specified graph can omit uncertainty about whether the graph describes the causal system at all. Numerical interval width alone does not measure that missing uncertainty.

## Why the prior-posterior update can be misleading in causal language

Consider a report stating that the posterior probability of a positive treatment effect exceeds 99%. The claim appears stronger than a statement about a positive regression coefficient because it refers directly to a probability of an effect. But if the posterior is based on observational Y|X without valid adjustment, the probabilistic object is a coefficient of association.

In the experiment, even the most severely biased unadjusted configurations strongly support a positive slope. The positive sign happens to agree with the causal effect, but the estimated magnitude is often close to twice as large. A different confounding mechanism could also reverse the sign, so agreement here should not be generalized.

Bayesian reporting should therefore name the parameter or causal functional being summarized. It should state which intervention is contemplated, what population the estimate concerns, and which assumptions connect observed data to potential outcomes. When the analysis relies on external prior information for identification, that dependence should be distinguished from learning through the observational likelihood.

This is not a request to replace posterior probabilities with p-values. A p-value calculated for an associational coefficient does not solve the causal problem either. Both Bayesian and frequentist uncertainty measures can be correct for a statistical parameter that is irrelevant to the intervention question.

The scientific gain comes from identifying the target and making the assumptions visible before choosing a language of uncertainty.

## Why more observations strengthen the wrong conclusion

The prior precision grid gives a clear illustration of asymptotic behaviour. For fixed λ, the contribution of D' D to the posterior precision grows with n, while the prior penalty remains fixed. Thus the likelihood becomes increasingly influential and the posterior approaches the population regression coefficient implied by the model.

In the unadjusted model, that population coefficient is two. With n=5,000, the mean posterior estimates range from approximately 1.9810 to 2.0008 across the three priors. The difference between weak and strong priors has become small relative to the one-unit causal discrepancy.

The result is uncomfortable because it resembles scientific progress: estimates stabilize, posterior intervals narrow and prior sensitivity decreases. Those properties are usually desirable. Here they strengthen inference about an observational quantity that is not the causal effect.

A robust analysis cannot infer causal validity from posterior stability alone. Stability shows that the likelihood is informative for the parameter it identifies. Whether that parameter represents the intervention contrast must be justified independently.

The same argument applies to more complicated models with flexible priors. A richly parameterized likelihood can converge toward a better approximation of the observed conditional distribution while remaining fundamentally unable to distinguish observationally equivalent causal structures. The limiting obstruction is identification, not computational accuracy or sample size.

## What a practical Bayesian causal analysis should report

A causal analysis should begin by specifying the intervention and estimand, including the population and outcome horizon. It should define whether the target is a total effect, controlled direct effect, average effect, subgroup effect or decision-specific value. That choice determines which posterior functional is relevant.

Next, the analysis should explain the causal structure that makes the target identifiable from available observations. For an observational treatment comparison, this may require measured confounders sufficient for conditional exchangeability, overlap and consistency. For instrumental-variable, discontinuity or difference-in-differences designs, it requires the assumptions specific to those strategies.

The likelihood should then be chosen to represent the necessary observed quantities. Priors should be justified according to substantive information, scale and the role of parameters in the causal model. Prior predictive checks can reveal implausible implications before observing outcomes, while posterior predictive checks can examine how well the model describes the observed data.

Sensitivity analysis should address both numerical priors and causal assumptions. It is not enough to show that the posterior treatment coefficient is stable across weak and moderately informative coefficient priors if a plausible unmeasured common cause could move the estimated effect substantially. Similarly, a good observed-data fit does not establish the credibility of intervention predictions.

The report should propagate posterior uncertainty through the causal functional, rather than necessarily attaching a causal label to one convenient regression coefficient. For complex longitudinal or heterogeneous settings, this can involve counterfactual simulation, posterior g-computation, or explicit decision models.

Finally, the analysis should distinguish posterior uncertainty conditional on the model from uncertainty about the model's identifying restrictions and external validity. Some structural assumptions cannot be learned from the observed data. Transparency about that limit is a feature of rigorous Bayesian science, not a weakness.

## Limits of the experiment

The benchmark is deliberately narrow. It contains one measured treatment, one potential confounder and independent Gaussian disturbances. The true causal effect is constant and linear. No outcome is missing, there is no measurement error, and units do not interfere with one another. The joint Gaussian structure permits exact calculations of both the observational likelihood and the structural causal effect.

The adjusted specification observes U perfectly. Real observational studies commonly lack such a variable or measure it with error. The apparent success of adjusted Bayesian regression here should therefore not be interpreted as proof that including available covariates is generally enough to identify a causal effect.

Observation variances are treated as known, and the posterior is analytically conjugate. The experiment does not compare Markov chain Monte Carlo implementations, hierarchical pooling, model selection or uncertainty over causal graphs. It isolates the effect of likelihood specification, confounder adjustment and Gaussian coefficient priors on posterior treatment inference.

The prior grid contains three fixed regularization strengths and a common zero prior mean. Other priors could produce different finite-sample behaviour, including priors specifically constructed from genuine external causal evidence. The results show what happens under the published choices, not a universal ranking of Bayesian priors.

The Monte Carlo evaluation uses 300 independent datasets per sample size. Small deviations from nominal coverage should not be overinterpreted as precise estimates of systematic failure. The zero coverage of the unadjusted intervals and large biases, however, are explained by the known population structural equations and do not depend on random simulation fluctuations.

The reported positive-posterior-fraction diagnostic measures whether the marginal posterior assigns more than 95% probability to a positive regression coefficient. It is not a posterior probability that the identifying assumptions are correct, nor is it a measure of causal decision quality. Its apparent success in all configurations illustrates that a coarse sign classification can overlook a serious magnitude error.

Most importantly, no empirical evidence about a real treatment is produced here. The scientific conclusion concerns the logic and numerical behaviour of posterior inference under known confounding and adjustment mechanisms.

## What the investigation established

Across 5,400 fitted posteriors, the correctly adjusted Bayesian model recovered a structural treatment effect of one under weak regularization, while the unadjusted model concentrated on the observational slope of two. Increasing the sample size narrowed both posterior distributions. It did not make the unadjusted posterior causally valid.

At n=5,000 with λ=0.1, the unadjusted posterior mean was 2.0008 with average standard deviation 0.0173 and zero causal coverage across 300 intervals. The adjusted mean was 1.0011 with standard deviation 0.0141 and 96.7% causal coverage. Both posteriors were exact under their respective likelihoods. Only one treatment coefficient represented the intended intervention effect.

Strong priors changed finite-sample behaviour but did not replace causal assumptions. A strong zero-centred prior made the unadjusted estimate closer to one when n=100, yet this apparent improvement disappeared as n increased. In the adjusted model, the same kind of prior could create substantial bias because treatment and confounder coefficients were correlated.

The result is not that Bayesian causal inference is weaker than classical causal inference. It is that Bayesian inference operates within the same identification constraints. Priors and posterior calculations offer a rigorous language for uncertainty and decisions; they cannot make the consequences of an intervention observable from an observational distribution that does not identify them.

The next article will investigate longitudinal treatment strategies and marginal structural models, where the difficulty is not only which confounders are measured, but how treatment changes subsequent confounders and therefore changes which adjustment strategies remain valid.

## Code and reproducibility

The experiment is implemented in `causal_inference_statistics/part14/experiment.py`, using Python 3.12 or later and NumPy. It generates the structural model, computes analytic Gaussian posterior moments, and writes `results.csv` containing all 18 design-by-prior-by-sample-size summaries. No external dataset, network request or posterior sampling library is required. The seed is fixed at 20261018.

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
12. Heterogeneous Treatment Effects: The Best Predictor Is Not Always the Best Treatment Rule
13. Causal Machine Learning and Double Machine Learning
14. Bayesian Causal Inference: When a Precise Posterior Answers the Wrong Question (this article)

## References

Bayarri, M. J. and Berger, J. O. (2004). The Interplay of Bayesian and Frequentist Analysis. *Statistical Science*, 19(1), 58–80. https://doi.org/10.1214/088342304000000116

Bernardo, J. M. and Smith, A. F. M. (1994). *Bayesian Theory*. Wiley.

Dawid, A. P. (2000). Causal Inference without Counterfactuals. *Journal of the American Statistical Association*, 95(450), 407–424. https://doi.org/10.1080/01621459.2000.10474210

Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A. and Rubin, D. B. (2013). *Bayesian Data Analysis*, 3rd edition. Chapman & Hall/CRC.

Gelman, A. and Hill, J. (2007). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press. https://doi.org/10.1017/CBO9780511790942

Gustafson, P. (2015). *Bayesian Inference for Partially Identified Models: Exploring the Limits of Limited Data*. Chapman & Hall/CRC.

Hernán, M. A. and Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC. https://miguelhernan.org/whatifbook

Hill, J. L. (2011). Bayesian Nonparametric Modeling for Causal Inference. *Journal of Computational and Graphical Statistics*, 20(1), 217–240. https://doi.org/10.1198/jcgs.2010.08162

Imbens, G. W. and Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press. https://doi.org/10.1017/CBO9781139025751

Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*, 2nd edition. Cambridge University Press. https://doi.org/10.1017/CBO9780511803161

Robins, J. M. (1986). A New Approach to Causal Inference in Mortality Studies with a Sustained Exposure Period. *Mathematical Modelling*, 7(9–12), 1393–1512. https://doi.org/10.1016/0270-0255(86)90088-6

Rubin, D. B. (1978). Bayesian Inference for Causal Effects: The Role of Randomization. *Annals of Statistics*, 6(1), 34–58. https://doi.org/10.1214/aos/1176344064

Spiegelhalter, D. J., Abrams, K. R. and Myles, J. P. (2004). *Bayesian Approaches to Clinical Trials and Health-Care Evaluation*. Wiley.

Zigler, C. M., Watts, K., Yeh, R. W., Wang, Y., Coull, B. A. and Dominici, F. (2013). Model Feedback in Bayesian Propensity Score Estimation. *Biometrics*, 69(1), 263–273. https://doi.org/10.1111/j.1541-0420.2012.01830.x