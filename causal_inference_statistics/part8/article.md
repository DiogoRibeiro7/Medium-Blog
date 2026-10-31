# Regression Discontinuity: Why a Sharp Cutoff Is Not Enough

*Part 8 of Causal Inference in Statistics. A controlled investigation of bandwidth choice, unequal curvature and selection at the cutoff, using 1,200 independently simulated datasets and 4,800 local linear regression fits. Python code and figures provided.*

A threshold is an attractive source of quasi-experimental variation. A scholarship may be awarded when an examination score passes a minimum; a business may qualify for a subsidy when employment falls below a specified number; a public programme may begin when an eligibility index reaches a fixed value. Observations immediately above and below the threshold appear similar in the running variable, yet their treatment assignments differ. If the corresponding potential outcomes evolve continuously at the threshold, their outcome discontinuity can identify a local causal effect.

That statement contains the identification assumption, not merely the estimator. A sharp jump in the observed conditional mean is not necessarily an intervention effect. The counterfactual outcomes might also jump because individuals sort around the boundary or because some other rule changes at precisely the same value. Even when the counterfactual means are continuous, estimating their limits requires a choice about how many observations away from the threshold to use and what function to fit to them. A broad window can deliver precise but biased estimates when the outcome regression is curved. A very narrow window can be comparatively unbiased but too noisy for the desired decision.

I investigate these difficulties with three fully specified data-generating processes. The policy effect is fixed at −1 in all three. In the first, untreated outcomes are linear and continuous at the cutoff. In the second, untreated outcomes are continuous but have additional curvature on the treated side. In the third, the baseline prognosis changes discontinuously at the cutoff for reasons unrelated to treatment; it is a stylized violation of the continuity requirement. I fit the same separate-sided, triangular-kernel local linear estimator at four predetermined bandwidths, from 0.15 to 1.00, in 400 independent samples of 2,000 observations per scenario.

The results distinguish the problems clearly. In the correctly specified linear design, increasing bandwidth reduces RMSE from 0.2513 to 0.0960 while leaving the mean effect close to −1. Under asymmetric curvature, a full-range bandwidth gives a mean estimate of −1.1965 and confidence-interval coverage of only 50.75%; at bandwidth 0.30 the mean is approximately −1.000, though sampling variability is larger. Under the continuity violation, all bandwidths estimate an observed jump near −0.3 instead of the true intervention effect −1. A smaller window cannot recover a counterfactual discontinuity that the design has made indistinguishable from treatment.

The purpose is not to demonstrate that one bandwidth should always be preferred or that every apparent discontinuity is suspicious. It is to separate three questions that empirical papers sometimes combine: what local effect the design identifies, how the estimator approximates the relevant limit, and which diagnostics can challenge the assumptions that connect the limit to an intervention.

## What is the intervention?

Imagine a public employment service assigning intensive assistance to people whose eligibility score exceeds a particular value. The running variable might summarize unemployment duration, prior employment and income. Above the administrative cutoff, eligible people receive a programme. Below it, people do not. A conventional regression of later unemployment outcomes on participation could confound the programme effect with the characteristics used to assign participation.

The threshold supplies a different comparison. Individuals very close to the cutoff may differ little in underlying risk while facing different treatment rules. If an applicant scoring 49.99 is untreated and an applicant scoring 50.01 is treated, it is plausible in some contexts that their outcomes without the programme would have been similar. One need not assert that all treated and untreated people have comparable potential outcomes. The identifying comparison is local to the threshold.

Yet the credibility of that reasoning depends on the administrative system. Can people strategically change the recorded score? Is the score rounded or reviewed near the boundary? Does a second programme also begin at the same cutoff? Do administrators override the eligibility rule? Do the applicants barely above and below represent the same population, or does a selection mechanism alter who appears in the dataset?

Those questions are not embellishments to a regression specification. They determine whether the jump in treatment assignment isolates a policy intervention. A beautifully fitted curve cannot resolve them if the design does not provide the counterfactual comparison it claims to provide.

The employment example motivates the analysis but does not describe the simulation's dataset. The numerical experiment concerns an abstract running variable and synthetic outcome. It is deliberately constructed so the intervention effect is known and the sources of estimator error can be separated.

## Potential outcomes at a threshold

Let X be a continuously distributed running variable with cutoff c, and let D=1{X≥c} indicate treatment. Define Y(1) and Y(0) as the potential outcomes under treatment and control. In a sharp regression discontinuity design, treatment is determined exactly by the threshold, so the observed outcome satisfies

\[
Y=D Y(1)+(1-D)Y(0).
\]

The local causal estimand is the treatment effect at the cutoff, under an appropriate conditional expectation interpretation:

\[
\tau_{RD}=E[Y(1)-Y(0)\mid X=c].
\]

Because a continuous X ordinarily satisfies P(X=c)=0, this notation is interpreted through conditional mean functions and their limits. Define m_1(x)=E[Y(1)|X=x] and m_0(x)=E[Y(0)|X=x]. If both are continuous at c, then

\[
\tau_{RD}=\lim_{x\downarrow c}E[Y\mid X=x]
-\lim_{x\uparrow c}E[Y\mid X=x].
\]

The left limit describes untreated units approaching the boundary, and the right limit describes treated units approaching it. Continuity of both potential-outcome conditional means allows the unobserved counterfactual values to be extrapolated to the same point from their observed sides.

This is not continuity of the observed regression function. The observed function is allowed—and expected—to jump because treatment changes discontinuously. The assumption concerns the potential-outcome functions that would prevail under each treatment condition. If an untreated-outcome jump occurs for a reason other than treatment, the observed jump combines that discontinuity with the effect of intervention.

Regression discontinuity is therefore an identification design rather than a formula for fitting two smooth curves. The estimator measures an observed boundary contrast. The causal interpretation of that contrast requires conditions that the fitted curves alone cannot verify.

## Why the effect is local

In the employment example, the intervention might help people with borderline eligibility more or less than those far from the boundary. A regression discontinuity estimates the effect for units at or very near that eligibility threshold, not automatically the average effect for everyone eligible or for everyone who might receive the programme after expansion.

In the simulation the treatment effect is constant, equal to −1 for every observation. This means the local effect, treated-population effect and full-population effect happen to coincide. That is an intentional simplification, not a property of RD generally. With heterogeneous effects the population being identified becomes consequential.

The problem resembles the local average treatment effect in instrumental-variable analysis, but there are important differences. In sharp RD, crossing the cutoff deterministically changes treatment status, and the primary contrast is a local treatment discontinuity. In fuzzy RD, the treatment probability jumps rather than moving exactly from zero to one. A ratio of the outcome discontinuity to the treatment-probability discontinuity can identify a local effect under additional assumptions analogous to instrumental-variable conditions.

This article investigates sharp RD only. Its simulated intervention is delivered to every unit with X≥0. There are no administrative exceptions, imperfect compliance or spillovers. This lets the investigation focus on continuity and estimation without mixing in the extra assumptions required for fuzzy assignment.

## How the synthetic data are generated

The cutoff is c=0 and the running variable is drawn independently for each person from a uniform distribution on [−1,1]. Each dataset contains 2,000 individuals. Let U be an independent standard normal outcome disturbance. The baseline untreated outcome in the smooth scenario is

\[
Y(0)=1+0.6X+U.
\]

Treatment changes outcome by exactly one unit downward:

\[
Y(1)=Y(0)-1.
\]

Treatment is D=1{X≥0}, and the observed outcome is therefore Y=Y(0)−D. Both potential-outcome conditional means are continuous at zero. A correctly estimated discontinuity has value −1.

The first failure mode changes the curvature of the untreated regression function without changing its value at the cutoff:

\[
Y(0)=1+0.6X+2X^2\mathbf 1\{X\geq0\}+U.
\]

The added term is zero at the cutoff and approaches zero continuously as X approaches it from above. Thus this scenario preserves local causal identification: the genuine RD estimand remains −1. Nevertheless, local linear fits over wide ranges attempt to summarize the curved right-side function with a straight line. Their intercept extrapolation to zero can become biased.

The third scenario adds a 0.7 jump to baseline prognosis on the treated side:

\[
Y(0)=1+0.6X+0.7\mathbf 1\{X\geq0\}+U.
\]

The treatment still reduces the outcome by one. However, the untreated conditional mean jumps by +0.7 at the boundary, so the observed conditional mean jumps by −1+0.7=−0.3. The observed discontinuity is not the intervention effect.

This third model is labelled *sorting* as a stylized discontinuity in latent prognosis associated with threshold crossing. It does not literally manipulate X or produce a density jump in the running variable. That distinction is crucial: a running-variable density test need not flag this construction, because X is drawn uniformly. The experiment challenges the continuity of untreated potential outcomes, not the observable distribution of the running variable.

## A local linear estimator

The estimator uses independent intercepts and slopes on the two sides of the threshold, with triangular weights that place more emphasis on observations near zero. For a fixed bandwidth h, observations satisfy |X|<h and receive weight

\[
w_i(h)=\left(1-\frac{|X_i|}{h}\right)\mathbf 1\{|X_i|<h\}.
\]

Within the selected observations, weighted least squares fits

\[
Y_i=\alpha+\beta X_i+\tau D_i+\gamma(D_iX_i)+\varepsilon_i.
\]

The coefficient τ is the fitted intercept difference at X=0. The interaction permits the treatment-side slope to differ from the comparison-side slope. If the outcome regression is locally linear and continuous in potential outcomes, that intercept difference approximates the local treatment effect.

The kernel matters because it changes how strongly observations at each distance influence the fitted limits. The triangular kernel downweights observations approaching the edges of the window. Other kernels, including rectangular and Epanechnikov weights, can be used. The choice is secondary to the broader problem of bandwidth and polynomial specification, although it does affect finite-sample efficiency.

I deliberately use a local polynomial of order one, not a high-degree global polynomial. The latter can behave unpredictably near boundaries and is not a generally reliable way to exploit RD designs. Local linear regression is a standard starting point because it provides a relatively transparent boundary approximation and avoids imposing a single global functional form.

The implementation fits weighted least squares directly using NumPy. Every simulated dataset is analysed at four fixed bandwidths: 0.15, 0.30, 0.60 and 1.00. This makes it possible to compare bias and sampling variation under the same estimator while changing the size of the region used for the local comparison.

## The uncertainty calculation

For each fitted local linear model, I calculate a heteroskedasticity-consistent HC1 sandwich standard error using the weighted design matrix. The interval reported in the simulation is the coefficient plus or minus 1.96 times that standard error. The bandwidth is treated as fixed. This calculation addresses conditional sampling variability of the fitted coefficient under a working local model.

It is not robust-bias-corrected RD inference. That distinction matters because local polynomial estimators can have smoothing bias even when the identifying continuity assumptions are valid. Conventional intervals centred on biased local polynomial estimates may undercover the true cutoff effect. Calonico, Cattaneo and Titiunik developed robust bias-corrected procedures that explicitly address such issues and allow data-driven bandwidth selection.

The present purpose is diagnostic: use a simple fixed-bandwidth procedure and examine its behaviour against a known effect. If a large bandwidth produces poor coverage because it fits unequal curvature with straight lines, we should see that failure rather than hide it behind a more complicated inference procedure. A publication-level empirical RD application should generally examine appropriate robust-bias-corrected confidence intervals and principled bandwidth selectors rather than reproduce this demonstration's HC1 intervals without qualification.

The simulation records, for each scenario and bandwidth, the mean estimated discontinuity, bias relative to −1, root mean squared error, mean fitted standard error, empirical coverage of the nominal 95% interval, and mean number of observations in the estimation window. Monte Carlo variability comes from independently regenerating the entire dataset for each replication.

## A study with 1,200 datasets and 4,800 regression fits

The experiment consists of 400 independently generated datasets for each of the three scenarios, or 1,200 datasets in total. Each dataset is analysed at four bandwidths, producing 4,800 separate RD regression fits. The independent replication count per scenario is 400.

The random number generator uses seed 20261017. The same sequence of simulated datasets is not reused across scenarios; each scenario receives independent draws from the continuing generator. All equations, bandwidths and estimation rules were fixed before reporting the output.

The expected number of observations with |X|<h is 2,000h because X is uniform on [−1,1]. Thus we expect about 300 observations at bandwidth 0.15, 600 at 0.30, 1,200 at 0.60 and 2,000 at the full bandwidth of one. The realized simulation means follow those expectations closely.

A narrower window focuses on observations more directly comparable at the threshold, but it discards information. A wider window reduces variance by using more data but can expose the estimator to curvature and differences far from the boundary. The best compromise depends on the unknown smoothness of the conditional mean, noise, sample size and inferential objective.

That last point deserves emphasis. The experiment can evaluate RMSE because the causal truth is known. A real researcher cannot directly select a bandwidth by measuring error against the unobserved causal effect. Bandwidth selection must rely on assumptions and data-driven criteria designed for estimation or inference, ideally accompanied by sensitivity analysis over a defensible range.

## Results: the smooth design

When both potential-outcome conditional means are linear and continuous, the estimator remains close to the true treatment effect at every bandwidth.

| Bandwidth | Mean estimate | RMSE | Mean SE | 95% coverage | Mean observations |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.15 | −1.0020 | 0.2513 | 0.2521 | 95.25% | 301 |
| 0.30 | −1.0095 | 0.1737 | 0.1787 | 95.75% | 601 |
| 0.60 | −1.0058 | 0.1246 | 0.1265 | 96.00% | 1,200 |
| 1.00 | −1.0028 | 0.0960 | 0.0980 | 95.75% | 2,000 |

The full-range fit is the most accurate by RMSE in this particular design. That is not a contradiction of local estimation principles. The regression functions on both sides really are linear throughout the observed interval, so incorporating distant observations does not introduce approximation bias. The larger window supplies more information about the intercepts.

At h=0.15, the mean fitted effect is already close to −1, but the standard error is around 0.252. At h=1.00, the mean fitted standard error falls below 0.10. Empirical coverage remains near its nominal value because the working regression is correctly specified for these potential-outcome functions.

One could be tempted to infer that broad bandwidths should always be preferred. The next scenario shows why that conclusion depends entirely on the structural shape of untreated outcomes.

## Results: continuous outcomes with unequal curvature

The curvature scenario preserves the RD identification assumption: the untreated conditional mean is continuous at zero. Only its shape away from the cutoff changes. The treatment effect is still −1 and is still the jump in the conditional mean at the threshold.

| Bandwidth | Mean estimate | Bias | RMSE | 95% coverage |
| ---: | ---: | ---: | ---: | ---: |
| 0.15 | −0.9775 | +0.0225 | 0.2477 | 95.75% |
| 0.30 | −0.9999 | +0.0001 | 0.1770 | 93.75% |
| 0.60 | −1.0652 | −0.0652 | 0.1397 | 91.75% |
| 1.00 | −1.1965 | −0.1965 | 0.2185 | 50.75% |

A large bandwidth now hurts identification *in estimation*, not in the logical causal design. The population jump at zero remains −1. However, the right-side quadratic term bends the conditional mean away from the boundary. Fitting a straight line across the entire positive range extrapolates an intercept that differs from its actual boundary value.

The sign is understandable. The right-side curve rises increasingly above the linear component as X becomes positive. A linear approximation across the positive range acquires a larger slope and a lower extrapolated intercept at X=0. The estimated right-minus-left intercept is therefore more negative than the true jump.

The conventional standard error continues to decrease as the bandwidth expands. At h=1.00 it is about 0.098, close to the smooth design. But the bias is approximately −0.196, around twice that standard error. This is why nominal coverage collapses to just over 50% across the replications.

The smallest bandwidth has substantially less approximation bias but more sampling variation. Interestingly, h=0.60 has the lowest observed RMSE, approximately 0.1397, even though its interval coverage is below nominal. Minimizing point-estimation RMSE and constructing a valid confidence interval are not the same objective. Bias that is tolerable for a point estimate may be unacceptable for conventional inference.

The h=0.30 mean lies extremely close to −1, but the near-zero Monte Carlo bias is not a universal feature of that threshold. It reflects the particular curvature, finite sample and random seed. It would be inappropriate to recommend h=0.30 as an optimal general bandwidth on the basis of this single experiment.

## Results: a discontinuity in untreated prognosis

The third scenario changes the identification conditions rather than merely the approximation problem. Its untreated outcome includes a +0.7 jump at the cutoff. The intervention effect remains −1, but the observed outcome jump is −0.3.

| Bandwidth | Mean estimate | Bias relative to effect −1 | RMSE | 95% coverage |
| ---: | ---: | ---: | ---: | ---: |
| 0.15 | −0.3099 | +0.6901 | 0.7355 | 23.00% |
| 0.30 | −0.3051 | +0.6949 | 0.7161 | 2.25% |
| 0.60 | −0.2991 | +0.7009 | 0.7122 | 0.00% |
| 1.00 | −0.3042 | +0.6958 | 0.7028 | 0.00% |

Every bandwidth returns approximately the same observational discontinuity. Narrowing the window does not move the estimate toward −1 because the counterfactual jump occurs at the boundary itself. Local polynomial fitting is behaving sensibly for the observed outcome function: it recovers the actual discontinuity near −0.3. The causal interpretation is wrong because the treatment jump and untreated prognosis jump are confounded.

As the bandwidth increases, the estimate becomes more precise around −0.3. The nominal confidence intervals increasingly exclude the true treatment effect. At h=0.60 and h=1.00, none of the 400 intervals covers −1. This resembles the earlier articles' central finding: repeated observations can concentrate a statistical estimate around a quantity that does not answer the intervention question.

This scenario is deliberately labelled a violation of untreated potential-outcome continuity. It is not evidence that manipulation of the running-variable density necessarily occurred. The running variable remains uniform on the original support, and an empirical density plot would not detect the unobserved prognosis jump directly. A study can fail continuity without displaying a dramatic density anomaly.

## The bandwidth trade-off is not only a numerical nuisance

Bandwidth choice determines which observations contribute to the boundary approximation. Under a continuously differentiable conditional mean, a sufficiently narrow local window can reduce approximation error as sample size increases, although it also reduces the effective sample size. For a local linear estimator, the leading bias can depend on curvature and bandwidth squared under common smoothness conditions, while variance scales roughly with the inverse of sample size times bandwidth.

These are asymptotic statements under regularity conditions, not exact descriptions of every dataset. They explain why point-estimation bandwidth selectors often balance squared bias against variance. The optimal rate and constants depend on smoothness, noise, density near the cutoff, kernel and polynomial order.

The synthetic curvature case makes the trade-off concrete. At h=0.15, only about 302 observations fall inside the window, and the estimator is noisy. At h=1.00, all 2,000 observations contribute, but the model is too rigid to represent the right-side curvature. The middle bandwidths balance those effects in different ways depending on whether the criterion is RMSE or confidence-interval coverage.

The usual empirical practice of showing one bandwidth and one coefficient can hide the dependence of conclusions on this choice. A better report shows a defensible primary bandwidth, explains how it was selected, and examines how estimates and inference change over nearby windows. If a claimed effect reverses sign when the bandwidth changes modestly, the analyst needs to understand whether this reflects sampling noise, nonlinear trends, influential observations or a genuine change in the local estimand.

Sensitivity alone is not proof. A biased estimator may remain nearly constant over a range of bandwidths, as the continuity-violation scenario demonstrates. Stability around a wrong causal quantity is still stability, not identification.

## Why local linear fits have two intercepts

An RD regression differs from fitting one line through every observation and adding a treatment indicator without allowing slopes to differ. The interaction D×X lets the treated and untreated sides have distinct local trends. The fitted coefficient on D is then the difference in intercepts at the cutoff rather than an average level difference across the full range.

This is useful because the running variable can influence outcomes differently on the two sides even without changing the local limits. For a sharp RD design, the objective is the discontinuity in the conditional mean exactly at c, not the difference in average outcomes among all treated and untreated individuals.

A single global polynomial imposes a functional form far from the threshold. High-order polynomials may fit the full sample while generating unstable boundary behaviour. Local methods instead give priority to the near-cutoff region, with weights that decrease with distance. The mathematical rationale is to estimate limits from nearby observations under smoothness assumptions, rather than trust a global model to extrapolate to the boundary.

The simulation explicitly permits separate slopes through an interaction. Its curved-right-side experiment shows that separate local lines are not enough when the window is too broad. Flexibility across sides is necessary for this benchmark but does not eliminate bandwidth-induced approximation bias.

## Density tests and manipulation

A familiar diagnostic in RD is to examine whether the running variable's density has a discontinuity at the cutoff. If individuals can precisely manipulate which side they land on, one may observe bunching just above a benefit threshold or missing observations just below it. McCrary (2008) formalized a density-discontinuity test relevant to this concern.

A density anomaly can be a useful warning, but it is not an automatic proof of invalidity. Heaping, measurement conventions or administrative rounding may also create patterns requiring investigation. Conversely, a continuous density is not proof of potential-outcome continuity. People might differ across the threshold in relevant unmeasured characteristics even when the marginal running-variable density appears smooth.

Our third simulation illustrates the latter limitation exactly. X is drawn uniformly, with no strategic reassignment and no density jump. Yet the untreated prognosis has a discontinuity of 0.7 at zero. If the analyst observed only X, D and Y, the jump could be mistaken for a treatment effect. The density diagnostic would not establish that the missing counterfactual mean is continuous.

An actual manipulation experiment would model agents' ability to select or report X, perhaps allowing latent prognosis to influence sorting across the cutoff. That mechanism could produce both density changes and outcome-selection effects, but the relationship is not guaranteed. The present simpler violation isolates the identification problem without attributing it to a density shift that the code does not generate.

## Covariate continuity tests

Another common check compares predetermined baseline covariates just above and below the cutoff. If a variable determined before eligibility exhibits a large discontinuity, that may suggest that the groups differ in ways unrelated to the treatment or that the assignment process changes composition at the boundary.

Covariate continuity is a meaningful falsification exercise when the variables are measured consistently, are genuinely pre-treatment and are not mechanically part of a different threshold policy. Yet the absence of a detected covariate jump cannot establish that all potential-outcome determinants are continuous. An important unmeasured variable remains untested. Even for measured variables, small near-cutoff samples can make discontinuities difficult to detect.

Multiple such tests also create interpretation problems: with many covariates, some rejection can arise by chance, while selecting or redesigning the study until all p-values exceed a threshold can bias the reported evidence. A joint test or a careful descriptive account may be more informative than treating every baseline p-value as an independent validity certificate.

The correct question is why the threshold could plausibly be as good as locally random for the potential outcomes being studied. Administrative details, timing, incentives and knowledge of score construction matter. Diagnostics are evidence about observable implications of that explanation, not substitutes for it.

## The role of institutional knowledge

One reason regression discontinuity can be credible is that administrative thresholds are often documented precisely. A rule might have been fixed before outcomes occurred, and people near the threshold may not have been able to control their position accurately. In such settings a local comparison can have a persuasive design-based interpretation.

But institutions can also undermine the comparison. Officials may grant exceptions, applicants may reapply until their score changes, score components may be rounded, multiple policies may share the same cutoff, or the eligibility measure may be altered in response to anticipated outcomes. A published rule does not necessarily describe treatment actually received.

An empirical RD paper should document how X was measured, whether the threshold was known, whether actors could precisely manipulate it, how many observations cluster at particular values, and whether treatment changes sharply or only probabilistically. It should also identify policies or behaviours that may affect outcomes directly at the same boundary.

This evidence matters more than the visual impression of two smooth curves with a vertical gap. A plot makes the discontinuity visible, but the causal interpretation rests on how the data came to have that structure.

## Fuzzy discontinuities and imperfect compliance

The simulated treatment assignment is sharp: all units at or above zero are treated, all below are untreated. Real programmes may use cutoffs only to change the probability of treatment. Some eligible individuals decline treatment; some ineligible people find another route to the intervention.

In a fuzzy RD design, one can examine the discontinuity in the treatment probability as well as the discontinuity in the outcome. The Wald-style ratio of those jumps can identify a local causal effect under suitable exclusion, monotonicity and continuity assumptions. Its interpretation is linked to the instrumental-variable framework developed in Part 7.

If the treatment probability changes only slightly at the boundary, the denominator of that ratio is small and local identification may be weak. The finite-sample instability then resembles weak instrumental-variable estimation. A dramatic-looking outcome jump is not enough; the policy's actual effect on treatment and its other pathways must be understood.

This article does not simulate fuzzy treatment, so none of the numerical tables should be used to make claims about compliance-adjusted estimates. The point is to locate the sharp RD experiment within the broader family of threshold designs and to distinguish which assumptions would be added by imperfect compliance.

## Why the result cannot automatically be generalized

Even an ideally implemented and precisely estimated sharp RD design identifies a local contrast near the cutoff. If treatment effects vary with score, risk, income or other baseline characteristics, people far from the boundary may benefit differently. A subsidy estimated to help marginally eligible firms may have another effect on much smaller or larger firms.

A policy decision might ask whether to move the eligibility threshold, abolish it, or offer treatment universally. The cutoff estimate alone does not identify the effects of all those alternatives. Moving the threshold changes which people receive treatment and may change behaviour around the boundary. Expanding a programme can alter delivery capacity, equilibrium conditions or other outcomes.

Multiple cutoffs or strong substantive assumptions about the treatment-effect function may allow further learning, but that involves additional identification arguments. It should not be described as a direct consequence of the original RD estimate.

Our simulations use a constant effect of −1, making this limitation invisible in their numerical targets. Their role is methodological: they show local estimation and continuity issues without complicating the result with heterogeneous responses. In an applied paper, the local population must be reported explicitly.

## Robust bias correction and choosing the primary analysis

The curved-outcome results provide a reason to distinguish bandwidth selection for estimation from bandwidth choice for inference. An estimator with low RMSE can have a non-negligible smoothing bias relative to its conventional standard error. Simply using the normal interval around that estimate can undercover the true effect.

Robust bias correction addresses this by estimating leading smoothing bias, often using a higher-order local polynomial, and adjusting the variance calculation to reflect the correction. Such methods require assumptions about smoothness and suitable data-driven bandwidth procedures. They do not remove violations of potential-outcome continuity or manipulation-driven sorting.

Calonico, Cattaneo and Titiunik's work provides practical procedures for these questions; Cattaneo, Idrobo and Titiunik's *A Practical Introduction to Regression Discontinuity Designs* develops the design and implementation in detail. These references are useful because they distinguish the identification assumptions from modern nonparametric inference, a separation that this simulation also emphasizes.

The NumPy script here intentionally uses fixed bandwidths and a conventional sandwich standard error to make the observed bias and coverage failures interpretable. It is a teaching and research benchmark, not a substitute for production-ready RD estimation tools or design-specific inference.

## A decision-oriented evaluation plan

An RD evaluation should begin with the policy and the cutoff. State exactly what the running variable measures, how it is computed, whether it can be manipulated, which units are eligible and what changes in treatment actually occur at the boundary. Define whether the target is a sharp treatment effect at the cutoff or a fuzzy local effect among compliers.

Next, describe the population near the threshold. Which observations contribute to the estimate, and why are they substantively comparable? What other institutional rules or incentives change at the same score? What information supports continuity of potential-outcome conditional means rather than merely continuity of the observed running-variable density?

Then plan the statistical comparison. Choose a principled primary bandwidth, local polynomial order, kernel and robust inference procedure. Report the number of observations on each side, not only a total sample size. Examine whether estimates change over sensible bandwidth ranges and whether influential observations or mass points affect results.

Finally, present identification diagnostics separately from estimation diagnostics. Density and predetermined-covariate checks can challenge the design. Bandwidth sensitivity and bias correction address approximation and inference. Neither category can replace the other. A convincing article should explain what each test can establish and where uncertainty remains.

The scientific question is not simply whether a coefficient near a threshold is statistically significant. It is whether the discontinuity represents an intervention and whether the statistical procedure estimates that discontinuity with defensible uncertainty.

## What this benchmark does not establish

The experiment contains one continuously distributed running variable, one zero cutoff, an exactly assigned treatment and a constant treatment effect. There are no treatment exceptions, covariate measurement errors, other policy thresholds, mass points, clusters, spillovers or heterogeneous treatment responses. The errors are independent Gaussian draws with constant variance.

The curved scenario has an explicitly chosen right-side quadratic term and otherwise linear structure. A different curvature, noise level or distribution of X could change which bandwidth has the lowest RMSE and how severe the coverage loss becomes. The reported bandwidth comparison is therefore not a universal recommendation.

The scenario called sorting is a discontinuity in latent untreated prognosis, not a behavioural model of agents shifting their observed scores. It demonstrates failure of potential-outcome continuity while leaving the density of X continuous. It should not be used as an empirical performance test of a McCrary-style density estimator.

The uncertainty estimates are conditional HC1 approximations with fixed bandwidths and no bias correction. Monte Carlo coverage is judged against the known intervention effect, a quantity generally unavailable in applied observational work. The experiment also uses 400 replications per scenario, so reported coverage percentages have some simulation variability.

Most importantly, no synthetic design can establish that a particular actual eligibility threshold satisfies local continuity. That credibility depends on institutional context, timing, implementation and the broader evidence. The experiment establishes how an estimator behaves when the assumptions are satisfied, how bandwidth affects it under curvature and how a discontinuity in untreated potential outcomes can defeat even a stable local fit.

## What the investigation established

Under a linear and continuous outcome model, widening the bandwidth improved precision without introducing important bias. The full-range local linear estimate averaged −1.0028 and had RMSE 0.0960, compared with RMSE 0.2513 in the narrowest window. The result was driven by a correctly specified functional form over the full range, not by a general property that larger RD windows are safer.

When the untreated outcome had unequal curvature but remained continuous at the cutoff, a full-range linear fit produced a mean estimate of −1.1965. Conventional 95% coverage fell to 50.75%. Narrower windows reduced approximation bias but increased variation, demonstrating the distinction between an identified local estimand and a finite-sample procedure that approximates it imperfectly.

When the untreated prognosis jumped by +0.7 at the cutoff, the observed discontinuity averaged near −0.3 at every bandwidth despite the true causal effect of −1. The wider windows made the wrong causal interpretation more precise. The running-variable distribution remained smooth, so a density diagnostic alone would not reveal this failure.

Regression discontinuity is powerful because it can identify a local causal comparison from an administrative assignment rule. Its weakness is not the simplicity of the estimator, but the possibility that the assumed continuity is false or that the estimator approximates the boundary poorly. A sharp observed jump is evidence of a discontinuity. To call it a treatment effect requires a separately defensible causal argument.

## Code

Every numerical summary is printed by `experiment.py` in `causal_inference_statistics/part8/`, which also writes `results.csv`. The program uses NumPy and Python 3.12 or later. It implements triangular-kernel local linear fitting and HC1 standard errors explicitly, checks input shapes and sample support, and fixes the pseudorandom seed. The three scenarios and four bandwidths are enumerated in the source; no external dataset is downloaded.

## The series

1. When Regression Gets the Causal Effect Wrong
2. Randomization, Confounding and the Value of Experimental Design
3. Matching and Propensity Scores: When Balanced Data Still Give the Wrong Causal Effect
4. Inverse Probability Weighting: When a Few Observations Control the Result
5. Directed Acyclic Graphs and the Cost of Controlling for Everything
6. Difference-in-Differences: When Parallel Trends Fail
7. Instrumental Variables: When a Strong First Stage Is Not Enough
8. Regression Discontinuity: Why a Sharp Cutoff Is Not Enough (this article)

## References

Calonico, S., Cattaneo, M. D. and Titiunik, R. (2014). Robust Nonparametric Confidence Intervals for Regression-Discontinuity Designs. *Econometrica*, 82(6), 2295–2326. https://doi.org/10.3982/ECTA11757

Calonico, S., Cattaneo, M. D. and Titiunik, R. (2014). Robust Data-Driven Inference in the Regression-Discontinuity Design. *Stata Journal*, 14(4), 909–946. https://doi.org/10.1177/1536867X1401400413

Cattaneo, M. D., Idrobo, N. and Titiunik, R. (2019). *A Practical Introduction to Regression Discontinuity Designs: Foundations*. Cambridge University Press. https://doi.org/10.1017/9781108684606

Cattaneo, M. D., Jansson, M. and Ma, X. (2020). Simple Local Polynomial Density Estimators. *Journal of the American Statistical Association*, 115(531), 1449–1455. https://doi.org/10.1080/01621459.2019.1635480

Cattaneo, M. D., Frandsen, B. R. and Titiunik, R. (2015). Randomization Inference in the Regression Discontinuity Design: An Application to Party Advantages in the U.S. Senate. *Journal of Causal Inference*, 3(1), 1–24. https://doi.org/10.1515/jci-2013-0010

Gelman, A. and Imbens, G. (2019). Why High-Order Polynomials Should Not Be Used in Regression Discontinuity Designs. *Journal of Business & Economic Statistics*, 37(3), 447–456. https://doi.org/10.1080/07350015.2017.1366909

Hahn, J., Todd, P. and van der Klaauw, W. (2001). Identification and Estimation of Treatment Effects with a Regression-Discontinuity Design. *Econometrica*, 69(1), 201–209. https://doi.org/10.1111/1468-0262.00183

Imbens, G. W. and Kalyanaraman, K. (2012). Optimal Bandwidth Choice for the Regression Discontinuity Estimator. *Review of Economic Studies*, 79(3), 933–959. https://doi.org/10.1093/restud/rdr043

Imbens, G. W. and Lemieux, T. (2008). Regression Discontinuity Designs: A Guide to Practice. *Journal of Econometrics*, 142(2), 615–635. https://doi.org/10.1016/j.jeconom.2007.05.001

Lee, D. S. (2008). Randomized Experiments from Non-Random Selection in U.S. House Elections. *Journal of Econometrics*, 142(2), 675–697. https://doi.org/10.1016/j.jeconom.2007.05.004

Lee, D. S. and Lemieux, T. (2010). Regression Discontinuity Designs in Economics. *Journal of Economic Literature*, 48(2), 281–355. https://doi.org/10.1257/jel.48.2.281

McCrary, J. (2008). Manipulation of the Running Variable in the Regression Discontinuity Design: A Density Test. *Journal of Econometrics*, 142(2), 698–714. https://doi.org/10.1016/j.jeconom.2007.05.005

Thistlethwaite, D. L. and Campbell, D. T. (1960). Regression-Discontinuity Analysis: An Alternative to the Ex Post Facto Experiment. *Journal of Educational Psychology*, 51(6), 309–317. https://doi.org/10.1037/h0044319