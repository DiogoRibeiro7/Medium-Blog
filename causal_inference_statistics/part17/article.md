# Negative Controls, Placebo Tests and Falsification: What a Passing Test Cannot Prove

*Part 17 of Causal Inference in Statistics. A reproducible experiment with three assignment mechanisms, two falsification checks, 1,500 simulated datasets and a known causal effect. Why a negative control can reveal one kind of hidden confounding while remaining silent about another. Python code provided.*

A statistical analysis can pass every diagnostic that its author thought to run and still estimate the wrong causal effect. That is not a reason to abandon diagnostics. It is a reason to ask what each diagnostic is capable of detecting, which counterfactual assumption it interrogates, and what alternative explanations remain after a test produces an unremarkable result.

Negative controls and placebo tests are especially attractive in observational causal inference because some identification assumptions cannot be verified directly. The analyst does not observe the same person's outcome under two incompatible interventions, and generally cannot establish from the outcome table alone that all common causes of treatment and outcome were measured. A variable or period that should not be causally affected by treatment can therefore offer a testable implication of the proposed model. If a treatment appears to affect an outcome it cannot plausibly influence, something about the statistical comparison deserves investigation.

But the reverse inference is invalid. No observed association with a negative control does not imply no hidden confounding of the target outcome. The negative control might be unaffected by the particular latent variable that confounds the treatment-outcome relationship. A placebo exposure chosen to be unrelated to everything can pass in a severely confounded study without assessing the relevant failure mechanism at all. Passing tests can be reassuring under additional assumptions about how biases would manifest, but those assumptions are part of the scientific argument and must not be hidden.

I investigate this distinction in three synthetic observational studies. Each contains 1,200 individuals and a true causal treatment effect of one. In the first, treatment is independent of baseline prognosis. In the second, treatment and both the target outcome and negative-control outcome share an unobserved cause. In the third, treatment and the target outcome share an unobserved cause that does not affect the chosen negative-control outcome. I also introduce a deliberately independent placebo exposure in each design. Every study is generated 500 times, giving 1,500 independently simulated datasets and nine sets of regression summaries.

The result is not that falsification tests are useless. In the shared-confounder design, the negative-control outcome detects the problem decisively, with its estimated association near 0.5 even though its causal effect is zero. In the outcome-specific confounder design, the target treatment coefficient remains biased upward by approximately 0.75, yet the negative-control outcome's mean coefficient is essentially zero and its rejection frequency is near the nominal 5%. The placebo exposure similarly passes in both biased designs. A test only has power against the violations that generate a detectable pattern in its own observable variables.

## The decision being evaluated

Imagine an employment service evaluating an intervention intended to improve job-search outcomes. Analysts have records showing who received support and the subsequent outcomes. They also have an earlier administrative measure that treatment cannot plausibly affect, perhaps a baseline outcome collected before programme participation. An association between programme treatment and that prior outcome might indicate that participants differed before treatment in ways relevant to their later prospects.

A negative-control outcome can serve a similar purpose even without using a prior period. It could be an outcome that shares some sources of confounding with the primary outcome but is not plausibly influenced by treatment. If the treated group differs from the untreated group on that outcome after appropriate adjustment, the proposed exchangeability assumptions may be false, the study may have selection problems, or the negative-control exclusion assumption may be incorrect.

The key phrase is *shares some sources of confounding*. The negative-control outcome must be chosen because of a defensible relationship to the biases that threaten the primary estimate, not merely because it is available in the database. An unrelated measure that cannot be affected by treatment but does not respond to any relevant latent factor may consistently show no association, whether the main causal estimate is correct or badly biased.

A placebo exposure operates in the other direction. It is a variable or assignment that should not cause the outcome. Examples include a future treatment that cannot causally determine an already completed past outcome, a pseudo-policy date in a period before implementation, or a fabricated treatment assignment that preserves some features of the original design. The useful versions of such tests preserve the dependence structure that could generate bias under the design while setting the hypothesized treatment effect to zero.

Simply drawing random noise as a fake treatment almost guarantees a null association. That can verify that code does not manufacture an effect from an unrelated regressor, but it tests little about the validity of the original treatment comparison. The simulations below deliberately include that weak placebo to show the difference between computational reassurance and causal falsification.

The substantive question remains whether the intervention changes the outcome, not whether every auxiliary regression has a non-significant coefficient. Negative controls can interrogate pieces of the identification argument. They cannot substitute for defining the intervention, target population, assignment mechanism and causal estimand.

## The causal target

Let A indicate treatment and Y the observed outcome. The individual potential outcomes Y(1) and Y(0) represent the outcome under intervention and control. The average treatment effect is

\[
\tau=E\{Y(1)-Y(0)\}.
\]

In the constructed studies, the structural effect is constant and equals one for every individual. This removes effect heterogeneity from the diagnosis. A regression estimate near one is appropriate for the target, while a value around 1.75 reflects a difference between the associational slope and the known intervention effect.

To identify the effect from an observational comparison, an analyst might assume that treatment is independent of the potential outcomes conditional on measured baseline information. Our experiment omits all baseline causes from the fitted regression to make the hidden-confounding mechanism visible. The relevant distinction is whether the omitted variables influence treatment and the outcome. Some omitted prognostic variables need not be confounders; others generate bias because they influence both.

A negative-control outcome N is assumed to have no causal response to A. This can be written as N(1)=N(0), or equivalently as a zero causal treatment effect on N under the specified intervention. A test of association between A and N is therefore an attempt to find evidence incompatible with some combination of no treatment effect on N and the asserted comparability of the groups.

The no-effect assumption is not a statistical property that can be confirmed from the same observations. It is a substantive statement about how the negative-control outcome is generated and when it is measured. If treatment actually affects N, a detected association cannot be attributed uniquely to confounding. This is why the choice of controls requires domain knowledge and causal timing, not a search for variables with convenient p-values.

The negative-control test is logically asymmetric. A clear association may be evidence against an assumed causal structure, provided the negative-control exclusion is credible. A null result remains compatible with several distinct structures, including ones where treatment is confounded by variables unrelated to N.

## Three observational assignment mechanisms

For each observation generate independent standard normal variables U, V, ε_X, ε_Y, ε_N and ε_P. The two baseline variables U and V affect the outcome, while ε_X provides treatment variation. The treatment effect is always one, and the negative-control outcome is constructed as

\[
N=U+\varepsilon_N.
\]

The placebo exposure is P=ε_P, independent of the causal system by construction. It has no direct or indirect effect on Y. Because P shares no relevant causes with treatment or outcome, it is intentionally an easy but scientifically weak falsification test.

In the unconfounded scenario, treatment is A=ε_X and the outcome is

\[
Y=A+U+V+\varepsilon_Y.
\]

Both latent variables influence outcome but neither causes treatment. Ordinary regression of Y on A therefore estimates the causal effect in the population. Likewise, there is no expected association between A and N. The placebo P is unrelated to Y.

In the shared-confounder scenario, treatment and outcome are generated as

\[
A=U+\varepsilon_X,\qquad
Y=A+1.5U+V+\varepsilon_Y.
\]

Now U affects treatment, the primary outcome and the negative-control outcome. N does not respond to intervention on A, but the common cause U makes it associated with A in observational data. This is precisely the structure under which the negative control should detect that the treated and untreated comparisons differ in an unobserved prognostic characteristic.

In the outcome-specific-confounder scenario,

\[
A=V+\varepsilon_X,\qquad
Y=A+U+1.5V+\varepsilon_Y.
\]

Treatment is confounded by V. The negative-control outcome is still N=U+ε_N, and U remains independent of treatment. N is a valid no-effect outcome but is insensitive to the hidden cause that biases the target comparison. The design is intentionally informative about a central limitation of negative-control screening.

These equations define mechanisms, not fitted approximations. Under an intervention that fixes A, U and V retain their distributions and Y changes by exactly one unit for each unit increase in treatment. Every difference between that intervention coefficient and an observed regression slope is generated by selection, not by uncertainty about the true effect.

## Why the target regression becomes biased

The population slope from regressing Y on A is Cov(A,Y)/Var(A). In the unconfounded design, A is independent of both U and V, so the slope equals one.

In the shared-confounder design, Var(A)=2 and Cov(A,U)=1. Consequently,

\[
\beta_{AY}=1+1.5\frac{\operatorname{Cov}(A,U)}{\operatorname{Var}(A)}
=1+\frac{1.5}{2}=1.75.
\]

In the outcome-specific-confounder design, substitute V for U in the same calculation. Var(A)=2, Cov(A,V)=1, and the structural coefficient of V in the outcome is 1.5. The population observational slope is again 1.75. The two biased designs therefore produce the same target regression coefficient despite different causes of its bias.

For the negative control in the shared-confounder design,

\[
\beta_{AN}=\frac{\operatorname{Cov}(A,N)}{\operatorname{Var}(A)}
=\frac{\operatorname{Cov}(U+\varepsilon_X,U+\varepsilon_N)}{2}
=\frac12.
\]

In the outcome-specific design, A depends on V while N depends on U, and U and V are independent. The population slope of N on A is exactly zero. Thus the two studies can have the same biased target effect and opposite falsification results.

For the independent placebo exposure P, Cov(P,Y)=0 in every scenario. The placebo regression should therefore be approximately zero even in the two studies where treatment is confounded. A non-rejection in this placebo test does not favor one causal structure over another because its distribution is essentially the same under all three.

The algebra explains the eventual Monte Carlo result. It also shows why these cases cannot be distinguished by making the primary regression more precise. Both confounded designs consistently estimate the same wrong slope, but the selected negative control responds to only one of them.

## The Monte Carlo design

I generated 500 independent datasets of 1,200 observations per scenario with NumPy and fixed seed 20261018. Every dataset contains a target treatment and outcome, the negative-control outcome, and the independent placebo exposure. Three regressions are fitted per dataset: primary outcome on actual treatment, negative-control outcome on actual treatment, and primary outcome on placebo exposure.

All regressions contain an intercept. Their coefficient standard errors use the familiar homoskedastic ordinary least squares formula. The simulated structural disturbances satisfy the relevant finite-variance conditions, making this a reasonable fixed-model approximation for the simple benchmark. The inference is not intended to cover complex cluster assignment, serial correlation or treatment-selection-dependent outcome observation.

The program records the mean fitted coefficient, error relative to the known target, root mean squared error, average standard error and nominal 95% interval coverage. For the negative-control and placebo coefficients, it also records how frequently a conventional two-sided test rejects zero using a 1.96 standard-error threshold.

The coverage calculations deliberately use the causal target. In the confounded primary regressions, the ordinary standard error estimates sampling uncertainty for an associational coefficient centred near 1.75. Its failure to cover the true intervention effect of one should not be misinterpreted as proof that the standard-error formula fails to describe the observed regression parameter.

There is no model selection after inspecting outcomes, no threshold chosen to maximize falsification success and no estimated treatment propensity. The three structures and testing rules are specified before running the replications. This design isolates the logical capabilities of the controls rather than confusing their performance with an elaborate data-adaptive procedure.

The single simulated outcome has no censoring and no treatment-effect heterogeneity. Those simplifications are strengths for this demonstration but limitations when translating its implications into a real employment, healthcare or education study.

## The primary estimates

The mean treatment coefficients across the 500 datasets are:

| Assignment structure | True effect | Mean regression coefficient | RMSE | 95% causal coverage |
| --- | ---: | ---: | ---: | ---: |
| No treatment confounding | 1.0000 | 0.9989 | 0.0503 | 94.6% |
| Confounder shared with negative control | 1.0000 | 1.7542 | 0.7551 | 0.0% |
| Confounder specific to primary outcome | 1.0000 | 1.7499 | 0.7507 | 0.0% |

The unconfounded regression recovers the structural effect within Monte Carlo uncertainty. In the other two scenarios the mean coefficient is close to the analytically predicted 1.75, and the nominal confidence intervals never contain the causal effect in these 500 replications.

Nothing in the primary outcome regression distinguishes the two forms of confounding. The near equality of 1.7542 and 1.7499 is expected because both have the same population slope. An analyst evaluating only the primary estimates might reasonably observe that treatment is strongly associated with outcomes, but could not infer the causal mechanism generating the association.

The mean standard errors of the biased primary regression coefficients are approximately 0.036. That is small relative to the bias of about 0.75. A large dataset therefore offers a precise estimate of a parameter that is not the causal quantity in question.

This result is consistent with the first article in the series: increasing precision cannot replace identification. What Part 17 adds is a demonstration of how auxiliary variables can provide evidence about a source of bias—and how their diagnostic value depends on whether they are connected to that source.

A primary coefficient estimated precisely, a conventional significant test and an apparently well-behaved regression do not establish a valid treatment effect. The problem must be investigated using the assignment mechanism and carefully chosen implications of the causal model.

## What the negative-control outcome detects

The mean coefficients from the negative-control regression are:

| Assignment structure | Mean coefficient on A | Rejection rate for zero |
| --- | ---: | ---: |
| No treatment confounding | −0.0015 | 4.2% |
| Confounder shared with negative control | 0.5011 | 100.0% |
| Confounder specific to primary outcome | −0.0002 | 4.8% |

The negative-control outcome has no causal response to treatment under any design. In the shared-confounder scenario, however, its observed association with A is approximately 0.5, matching the theoretical covariance calculation. The usual significance test detects that association in every simulated dataset. This is a successful falsification of the assumption that the observed treatment groups are exchangeable for all the relevant baseline causes linked with N.

In the outcome-specific scenario, the negative-control coefficient is effectively zero, and the rejection rate of 4.8% is near the nominal level. This reassuring result coexists with severe confounding of the target effect. The negative control is valid in the narrow sense that treatment cannot cause it, but it is poorly connected to the latent variable that actually threatens the analysis.

The fact that both scenarios produce a primary slope near 1.75 while their negative controls behave differently is the decisive comparison. The observed diagnostic does not merely reflect the magnitude of the main coefficient. It reflects the particular causal pathways between the selected control, assignment and the unmeasured factors.

The failure to reject a zero negative-control association cannot justify the conclusion that treatment is conditionally independent of its potential outcomes. The simulated counterexample satisfies the null for the chosen negative-control outcome exactly while violating exchangeability for the primary outcome.

This does not imply that negative controls should be discarded. It implies that their interpretation must be linked to an explicit bias model or a credible argument about shared confounding mechanisms.

## Why the placebo exposure passes everywhere

The placebo exposure P is generated independently of all other variables. The corresponding regressions of Y on P give:

| Assignment structure | Mean placebo coefficient | Rejection rate for zero |
| --- | ---: | ---: |
| No treatment confounding | −0.0013 | 3.6% |
| Confounder shared with negative control | 0.0028 | 5.6% |
| Confounder specific to primary outcome | 0.0038 | 4.4% |

All three results are close to zero. None provides useful evidence about whether the primary treatment effect is identified. The placebo was deliberately constructed to be independent of both the real treatment and every cause of the outcome, so it has no pathway through which the hidden confounder could generate a placebo association.

A placebo exposure that is unrelated to the assignment mechanism is a weak diagnostic. It can detect obvious implementation errors, accidental leakage or gross defects in a fitting routine. It does not test the causal structure responsible for bias in the original comparison.

More useful placebos preserve some aspect of the original assignment process while removing the plausible treatment mechanism. For a policy study, a pseudo-treatment date may preserve treated-group membership and its latent characteristics while testing whether an apparent programme effect existed before implementation. For a longitudinal analysis, a future treatment indicator may reveal associations with outcomes measured before that treatment could act, provided temporal ordering and feedback are considered carefully.

Even such placebos have limits. A differential shock can arise only after a policy begins, an announcement can create anticipation before the formal start, and a placebo interval may not contain the period during which the threatening bias operates. The Difference-in-Differences article in this series showed precisely that phenomenon.

The lesson is to ask what would make the placebo fail if the proposed causal model were wrong. If the answer is that nothing about the actual confounding mechanism would affect the placebo, passing the test has little diagnostic value.

## Negative-control outcomes and negative-control exposures

The two broad forms of negative control are related but not interchangeable. A negative-control outcome is an outcome that the primary treatment should not causally affect but that may share relevant confounders with the primary outcome. A negative-control exposure is an exposure that should not causally affect the primary outcome but may share the confounding structure of the primary treatment.

Both rely on substantive exclusion assumptions. A purported negative-control outcome may actually be downstream of treatment through an overlooked mechanism. A purported negative-control exposure may affect outcomes through another pathway. In those cases a detected association does not have a unique interpretation as residual confounding.

A well-chosen negative-control outcome can be particularly informative when its determinants overlap with the latent causes of the target outcome. For example, a pre-treatment measure collected before the intervention could expose systematic baseline differences between treated and untreated people. But that measure need not respond to every prognostic cause of the future outcome; the outcome-specific simulation illustrates the resulting blind spot.

A negative-control exposure is more convincing when it is generated by something similar to the original treatment assignment mechanism without having the treatment's causal effect. A randomly generated dummy variable is not automatically such an exposure. Its defining property of independence can make it pass precisely when the actual exposure is heavily selected.

These are diagnostic roles, not interchangeable operational labels. Each negative control should be accompanied by an explicit statement of why it cannot have the proposed causal effect and why the confounding that threatens the primary estimate would be expected to induce an observable relationship with it.

Without those two arguments, the control can be statistically valid as an unrelated variable while scientifically irrelevant to the main identification problem.

## A failed test is not a calibrated estimate of bias

In the shared-confounder scenario, the negative-control coefficient is about 0.5 while the primary treatment coefficient's causal bias is about 0.75. Those numbers come from the known coefficients and covariances. It would be a mistake to subtract 0.5 mechanically from the primary estimate and declare the result causally corrected.

The magnitude of a negative-control association depends on how strongly the control responds to the hidden causes, how those causes affect treatment, and what measurement error the control contains. The primary bias depends additionally on how strongly those causes affect the primary outcome. Without a model connecting these relationships, the size of the negative-control coefficient is not automatically the size of the primary bias.

More elaborate negative-control methods use proxy variables, bridge functions and assumptions about completeness or latent-confounder structures to identify causal effects in settings where ordinary exchangeability given observed covariates fails. These methods go beyond simple falsification. They use the negative controls as information about the hidden variable rather than merely checking whether an auxiliary coefficient differs from zero.

The assumptions involved can be demanding. A proxy that weakly represents the relevant latent variable may provide little information; a misspecified bridge relation may produce misleading estimates; and some causal effects may remain unidentified even with several auxiliary variables. No general theorem says that two measured negative controls always repair hidden confounding.

This article deliberately does not fit such an identification model. Its goal is narrower and logically prior: show why the diagnostic interpretation of a negative control depends on its relationship to the actual confounding mechanism.

The same distinction applies to placebo outcomes. Finding a nonzero placebo effect can undermine a design, but the result does not itself supply a corrected causal coefficient or a valid uncertainty interval for the corrected effect.

## Statistical power and the interpretation of non-rejection

Even a negative control capable of detecting a bias mechanism may fail to reject because a sample is small, its outcome is noisy, the control is measured imprecisely, or the relevant association is weak. A non-significant result is then compatible with both no detectable problem and a problem that the test lacked power to detect.

Our shared-confounder design avoids that ambiguity by choosing a large and clearly observable negative-control association. With 1,200 observations the test rejects every time. That provides a clean illustration of successful detection, but it is not representative of the sensitivity of negative controls in all applied studies.

The outcome-specific design demonstrates a different limitation. There the population placebo and negative-control effects are exactly zero despite strong bias in the target regression. Increasing the sample size would narrow their intervals around zero, not reveal the omitted V pathway. The missing signal is structural rather than a problem of statistical power.

These two limitations require different responses. If the problem is low power, more observations or a better-measured control may help. If the control does not respond to the relevant hidden confounder, more data on that same variable do not solve the causal identification question.

Researchers should therefore report effect sizes and intervals for placebo and negative-control tests, not only whether their p-values cross a conventional threshold. A small interval near zero can restrict the magnitude of a particular detectable violation under an assumed relationship; it does not rule out violations operating through unrelated mechanisms.

The correct interpretation is always conditional: under the hypothesis that this negative control would respond to the bias of concern, the data provide a particular degree of evidence about that bias. Removing the conditional clause changes the scientific claim.

## Multiple testing and choosing reassuring controls

A study might have dozens of plausible control outcomes, policy dates, subgroups or alternative exposures. Running enough placebo tests creates opportunities for false positives, especially when the analyst selects and reports only the most unusual results. It also creates opportunities for misleading reassurance when only tests that happened to pass are discussed.

The testing programme should therefore be specified and justified according to plausible failure mechanisms. If the objective is falsification, controls should be selected because they are connected to those mechanisms, not because they are most likely to yield null coefficients. Reporting the full set of prespecified controls gives readers a clearer picture than exhibiting one convenient p-value.

Multiplicity matters for the statistical interpretation of multiple significant placebo associations. Correlated controls can make familywise calculations complicated, and simple independent-test adjustments may be unnecessarily conservative. The details depend on whether the controls test one composite implication, separate assumptions, or exploratory alternatives.

More importantly, no multiple-testing correction solves the choice-of-control problem. A perfectly adjusted family of independent placebo regressions can still be uninformative about a hidden common cause that none of the placebos measures. Statistical calibration and substantive relevance are distinct requirements.

In this article there are only two prespecified auxiliary regressions in each scenario, chosen before outcome analysis. I report their rejection frequencies over repeated datasets rather than selecting a single convenient realization. The resulting contrast exposes structural detectability without turning the testing exercise into a search for desirable outcomes.

## Placebo dates and longitudinal studies

Placebo dates are common in policy evaluation, particularly Difference-in-Differences and synthetic control analyses. The investigator assigns a hypothetical intervention time before the real policy and checks whether the estimated effect appears when no treatment was supposed to occur.

A pseudo-date is useful when the confounding mechanism would also generate systematic differences during the pre-policy interval. If treated regions were already diverging from controls, a placebo policy effect may make that divergence visible. However, if a treated-specific shock begins simultaneously with the actual policy, earlier periods may be parallel and the placebo can pass while post-policy identification fails.

Anticipation complicates the choice of placebo window. If individuals react to a programme announcement before formal implementation, an apparently pre-treatment outcome may already be affected by policy expectations. A placebo that excludes the anticipatory interval can miss that effect; one that includes it may reveal a deviation but still require careful interpretation of the correct treatment timing.

A synthetic control's good fit to pre-intervention outcomes is also a form of diagnostic evidence, but it cannot guarantee that the relationship between donors and treated unit will remain stable after implementation. The Part 9 simulations showed a treated-specific post-intervention shock that left pre-treatment fit reassuring while moving the estimated intervention effect away from the truth.

Negative controls and placebo tests inherit these causal timing problems. They are not generic certifications to be attached to any observational design. Their value comes from matching the test window and auxiliary outcome to the pathway that could otherwise explain the estimated treatment effect.

The present experiment uses cross-sectional variables rather than a time series, so it does not calculate placebo event studies. Its contribution is the underlying logic shared by those applications: a falsification test only constrains mechanisms that would change the observable object being tested.

## What falsification can establish

A falsification test can demonstrate that a maintained collection of assumptions is incompatible with the observed data, subject to sampling uncertainty. If treatment has no possible causal effect on an outcome but appears strongly associated with it, at least one explanation must be reconsidered. Perhaps treatment was confounded, the control was incorrectly assumed unaffected, selection changed the study population, or a measurement problem generated the association.

It is often impossible to identify which assumption failed from the auxiliary regression alone. Falsification is about excluding a joint explanation, not automatically estimating a replacement causal effect.

Conversely, a non-rejection cannot establish that the entire model is true. Different causal structures can generate the same tested moment condition. The outcome-specific hidden-confounder mechanism is a constructive counterexample: it yields a valid zero association with N while producing a target coefficient about 75% larger than the true effect.

This is a broader lesson from the philosophy of scientific testing as well as from causal statistics. A model can imply observable restrictions, and violations of those restrictions can undermine it. Passing a finite set of restrictions leaves many alternative models observationally compatible with the available evidence.

The useful question is therefore how discriminating a proposed falsification test is among the competing causal explanations that matter for the decision. A test that produces the same expected outcome under a valid design and a serious invalid alternative supplies little information for distinguishing them.

An investigation can still combine several controls, alternative identification designs and sensitivity analyses to rule out more explanations than any individual test. But the final conclusion remains conditional on what those exercises genuinely address.

## Connecting negative controls to sensitivity analysis

The preceding article investigated sensitivity to unmeasured confounding by specifying how an omitted variable affects treatment assignment and outcome. It calculated a bias curve showing how estimates move when the hidden assignment coefficient increases. That exercise is useful when the unmeasured variable is not observed but its plausible effect sizes can be bounded or calibrated.

Negative controls can supplement sensitivity analysis by supplying empirical information about a hidden pathway. If an auxiliary outcome is credibly connected to the same latent causes, its association with treatment can restrict plausible values of certain sensitivity parameters. A strongly nonzero negative-control coefficient provides evidence that at least some residual dependence remains.

Our two confounded scenarios show why calibration cannot be mechanical. In the shared case the control is informative about U. In the outcome-specific case it says nothing about V. Treating a null N association as a bound on *all* hidden confounding would rule out the very data-generating mechanism that produced the target bias.

A stronger approach would specify which latent causes are shared, how the proxy responds to them, and what restrictions can be derived under those assumptions. Those relationships could themselves be assessed with multiple measures or external knowledge. Without such structure, the negative-control null is a statement about N, not a universal confounding bound.

Sensitivity analysis and falsification address related but different questions. Sensitivity analysis asks how conclusions vary over stated departures from assumptions. Falsification asks whether observed implications of the assumptions are compatible with the data. Both become more useful when assumptions are explicit and the scientific question is clearly defined.

## The role of negative controls in a research plan

A causal analysis should start with the treatment, outcome, target population and estimand. The next step is to describe treatment assignment, identify plausible common causes and explain what makes the desired counterfactual comparison credible. Only then should the analyst choose negative controls and placebos.

For each proposed control, state why the primary treatment cannot cause it and why it would be expected to respond to at least one plausible source of bias. The second justification is often omitted, leaving only the fact that a variable is 'unaffected' by treatment. The simulations demonstrate why that is insufficient.

The timing of controls matters. Baseline measurements taken before treatment may still be selected or measured differently across groups. A future exposure can sometimes be related to earlier outcomes through reverse causality or persistent common causes; that relationship is precisely why it might be diagnostically useful, but it also complicates interpretation. Outcomes recorded after intervention may be affected through indirect pathways even when no direct effect is expected.

Report estimates, uncertainty and sample sizes for every prespecified auxiliary test. Discuss what a nonzero association would mean, what a null result can exclude under the maintained model, and which plausible failure mechanisms would leave the control unchanged. A good analysis should include some consideration of a counterexample in which the control passes but the primary estimate is biased.

Integrate the tests with balance diagnostics, sensitivity calculations and substantive knowledge rather than replacing those elements. If controls disagree, investigate their causal roles instead of averaging their p-values. If every control passes, the appropriate conclusion is that no violation detectable by those selected controls was found, not that the causal effect has been proven.

The clearest benefit of a negative-control programme is often methodological discipline: it forces analysts to explain what would have to be true about treatment assignment and the data-generating process for the causal estimate to be defensible.

## What the simulation does not show

The experiment uses continuous Gaussian treatment, outcomes and disturbances with linear additive effects. The causal treatment effect is constant, the regressions are intentionally unadjusted, and the complete data-generating process is known. This is an unusually favourable setting for distinguishing the three types of observed association, but it does not represent all practical causal evaluation problems.

The negative-control outcome has no causal response to treatment by construction. In real data, an apparently safe negative-control variable may respond through an overlooked mechanism. A significant auxiliary coefficient then has multiple explanations that require further investigation.

Only one negative-control outcome and one placebo exposure are evaluated. There is no use of multiple proxies for hidden confounders, proximal identification, negative-control bridge equations, high-dimensional covariates, or flexible machine learning. The results should not be interpreted as a benchmark of those methods.

The placebo exposure is deliberately independent of the assignment mechanism. It illustrates a weak design rather than a carefully chosen negative-control exposure. An empirically useful placebo might retain associations with observed and hidden causes of the actual treatment, making it more sensitive to particular invalid causal structures.

The test rejection frequencies are calculated from independent Monte Carlo datasets under conventional OLS standard errors. Applied settings with clustering, serial dependence, selection, heteroskedasticity or very few treated groups require different uncertainty procedures. Correctly accounting for dependence is distinct from selecting a control that addresses the relevant causal problem.

No claim is made about the prevalence of negative-control failures in published research or the magnitude of bias in any real employment programme. The study demonstrates logical possibilities under explicit structural assumptions. It shows what the chosen tests can and cannot discriminate, not how every negative-control analysis will behave.

Finally, a test that fails to detect the hidden cause of bias is not necessarily defective. It may be operating exactly as defined. The defect lies in interpreting a narrow diagnostic as universal evidence for causal identification.

## What the investigation established

Across 500 datasets per assignment mechanism, the primary regression recovered the true effect of one when treatment was independent of baseline outcome causes. In two confounded mechanisms it instead estimated approximately 1.75, with nominal 95% causal coverage of zero despite small conventional standard errors.

The negative-control outcome detected the shared confounder: its estimated treatment association was approximately 0.501, and every replication rejected a null of zero. But in the outcome-specific confounding mechanism the same negative control averaged essentially zero and rejected only 4.8% of the time, while the primary causal bias remained approximately 0.75. The independent placebo exposure similarly rejected at close to its nominal rate under all three mechanisms.

The observed results agree with the structural equations. They are not anomalies produced by finite samples. One hidden variable caused both treatment and the control outcome; another affected treatment and the primary outcome but not the control. The falsification checks distinguished those structures only when their own measurements carried information about the relevant hidden pathway.

This is the central conclusion: a failed negative-control test can undermine a causal analysis, but a passed test establishes only that the particular detectable implication was not contradicted. Identification remains a claim about interventions, potential outcomes and assignment mechanisms. Falsification makes that claim more accountable when tests are chosen to challenge plausible alternatives; it cannot turn the absence of an observed placebo association into proof that no important alternative exists.

## Code

All numerical results are reproduced by `experiment.py` in `causal_inference_statistics/part17/` of the Medium-Blog repository. The script requires Python 3.12 or later and NumPy. It generates 1,500 independent datasets, fits three regressions to each one, prints the Monte Carlo summaries and writes `results.csv`. The random seed is 20261018. No external dataset or specialized causal-inference library is required.

## The series

1. When Regression Gets the Causal Effect Wrong
2. Randomization, Confounding and the Value of Experimental Design
3. Matching and Propensity Scores
4. Inverse Probability Weighting
5. Directed Acyclic Graphs and the Cost of Controlling for Everything
6. Difference-in-Differences: When Parallel Trends Fail
7. Instrumental Variables: When a Strong First Stage Is Not Enough
8. Regression Discontinuity: Why a Sharp Cutoff Is Not Enough
9. Synthetic Controls and Counterfactual Time Series
10. Doubly Robust Estimation
11. The G-Formula
12. Heterogeneous Treatment Effects
13. Causal Machine Learning and Double Machine Learning
14. Bayesian Causal Inference
15. Time-Varying Treatments and Marginal Structural Models
16. Sensitivity Analysis for Unmeasured Confounding
17. Negative Controls, Placebo Tests and Falsification (this article)

## References

Arnold, B. F., Ercumen, A., Benjamin-Chung, J. and Colford, J. M. Jr. (2016). Brief Report: Negative Controls to Detect Selection Bias and Measurement Bias in Epidemiologic Studies. *Epidemiology*, 27(5), 637–641. https://doi.org/10.1097/EDE.0000000000000504

Bertrand, M., Duflo, E. and Mullainathan, S. (2004). How Much Should We Trust Differences-in-Differences Estimates? *Quarterly Journal of Economics*, 119(1), 249–275. https://doi.org/10.1162/003355304772839588

Callaway, B. and Sant'Anna, P. H. C. (2021). Difference-in-Differences with Multiple Time Periods. *Journal of Econometrics*, 225(2), 200–230. https://doi.org/10.1016/j.jeconom.2020.12.001

Daniel, R. M., De Stavola, B. L., Cousens, S. N. and Vansteelandt, S. (2015). Causal Mediation Analysis with Multiple Mediators. *Biometrics*, 71(1), 1–14. https://doi.org/10.1111/biom.12248

Flanders, W. D., Klein, M., Darrow, L. A., Strickland, M. J., Sarnat, S. E., Sarnat, J. A. and Waller, L. A. (2011). A Method for Detection of Residual Confounding in Time-Series and Other Observational Studies. *Epidemiology*, 22(1), 59–67. https://doi.org/10.1097/EDE.0b013e3181fdcabe

Hernán, M. A. and Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC. https://miguelhernan.org/whatifbook

Lipsitch, M., Tchetgen Tchetgen, E. and Cohen, T. (2010). Negative Controls: A Tool for Detecting Confounding and Bias in Observational Studies. *Epidemiology*, 21(3), 383–388. https://doi.org/10.1097/EDE.0b013e3181d61eeb

Miao, W., Geng, Z. and Tchetgen Tchetgen, E. J. (2018). Identifying Causal Effects With Proxy Variables of an Unmeasured Confounder. *Biometrika*, 105(4), 987–993. https://doi.org/10.1093/biomet/asy038

Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*, 2nd edition. Cambridge University Press.

Rambachan, A. and Roth, J. (2023). A More Credible Approach to Parallel Trends. *Review of Economic Studies*, 90(5), 2555–2591. https://doi.org/10.1093/restud/rdad018

Rosenbaum, P. R. (1987). The Role of a Second Control Group in an Observational Study. *Statistical Science*, 2(3), 292–306. https://doi.org/10.1214/ss/1177013224

Roth, J. (2022). Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends. *American Economic Review: Insights*, 4(3), 305–322. https://doi.org/10.1257/aeri.20210236

Shi, X., Miao, W. and Tchetgen Tchetgen, E. J. (2020). A Selective Review of Negative Control Methods in Epidemiology. *Current Epidemiology Reports*, 7, 190–202. https://doi.org/10.1007/s40471-020-00243-4

Tchetgen Tchetgen, E. J., Ying, A., Cui, Y., Shi, X. and Miao, W. (2024). An Introduction to Proximal Causal Inference. *Statistical Science*, 39(3), 375–390. https://doi.org/10.1214/23-STS911

VanderWeele, T. J. and Ding, P. (2017). Sensitivity Analysis in Observational Research: Introducing the E-Value. *Annals of Internal Medicine*, 167(4), 268–274. https://doi.org/10.7326/M16-2607