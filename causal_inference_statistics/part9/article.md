# Synthetic Controls and Counterfactual Time Series: When a Good Pre-Treatment Fit Is Not Enough

*Part 9 of Causal Inference in Statistics. A numerical investigation of synthetic controls, donor weights, pre-treatment fit, latent post-treatment shocks and anticipation, using 900 independent synthetic datasets and three counterfactual estimators. Python code provided.*

A policy is introduced in one region, one hospital or one company. The outcome changes after implementation. To estimate what the intervention accomplished, we need to know how the treated unit would have evolved without it. That trajectory does not exist in the observed data. Synthetic control methods try to reconstruct it from a weighted combination of units not exposed to the intervention.

The visual appeal is considerable. If a synthetic combination of untreated units tracks the treated unit closely for several years before the policy, the resulting counterfactual can look more credible than a simple average of other regions. The method also makes the comparison explicit: readers can inspect the donor pool, the weights, and how well the synthetic and treated outcomes agree before implementation.

But pre-treatment agreement is evidence about an earlier period, not a guarantee about the future. A shock affecting only the treated unit at the time the policy starts can produce an apparent treatment effect that no choice of pre-treatment donor weights can separate from the policy. Anticipation can contaminate the period used to construct the synthetic control. An excellent in-sample fit can therefore coexist with a substantially biased intervention estimate.

I investigate these issues with a controlled simulation. Each dataset has one treated unit, eight potential donors, 24 pre-treatment periods and 12 post-treatment periods. The untreated outcome of the treated unit is generated from a genuine convex mixture of donor trajectories, with measurement noise. Treatment lowers the outcome by exactly two units after implementation. Three scenarios vary whether the relationship to the donor pool remains stable, whether an untreated post-period shock affects only the treated unit, and whether the treated unit changes before formal intervention.

Three methods are compared: nonnegative synthetic control weights summing to one, a simple before–after change, and Difference-in-Differences using equally weighted donor outcomes. Each scenario is repeated on 300 independently generated datasets, giving 900 datasets and 2,700 counterfactual estimates. With a stable relationship, the synthetic control mean is −1.9909, close to the true −2. With a post-treatment untreated shock of +1.5, its mean becomes −0.4925, while mean pre-treatment root mean squared prediction error remains only about 0.183. In the anticipation scenario, the synthetic control mean is −1.7727.

The strongest finding is not that synthetic controls fail. Under their intended conditions they work very well in this experiment. The finding is that fitting the observed past and identifying an unobserved future are different operations.

## The intervention and the missing time series

Suppose a municipality implements an employment policy, while several comparable municipalities do not. We observe employment outcomes before and after implementation. A before–after comparison attributes any change in the treated municipality to the policy, but the labour market may have changed everywhere. A Difference-in-Differences comparison subtracts the average change observed elsewhere, relying on an assumption about parallel untreated trends.

Synthetic control offers a different construction. Rather than choosing one comparison municipality or assigning all comparisons equal weight, it selects nonnegative donor weights to reproduce the treated unit's pre-intervention trajectory. A combination of donors can capture persistent differences in baseline levels and responses to common factors that a simple donor average misses.

For example, a coastal municipality may respond differently to national economic conditions than an industrial municipality. Neither may be a satisfactory comparison by itself. A weighted combination could have a history that resembles the treated municipality more closely. The weights are learned using outcomes or predictors observed before intervention, with no use of post-treatment outcomes in the construction.

The scientific question is still causal. The goal is not to predict the treated unit's observed post-intervention outcome. We already observe that. The goal is to predict its *untreated potential outcome* after the policy, the trajectory that would have occurred had the intervention not taken place. Prediction tools are useful only insofar as their extrapolation has a credible counterfactual interpretation.

This example is not an analysis of a real municipal policy. The outcomes, factor process and donor relationships below are generated from stated equations. That controlled setting makes it possible to measure the accuracy of counterfactual estimates against a known intervention effect rather than inferring correctness from attractive fitted trajectories.

## Defining the causal effect

Let unit 1 receive treatment after time T₀, while units 2 through J+1 remain untreated. Write Y₁t(1) for the treated unit's potential outcome under intervention and Y₁t(0) for its outcome without intervention. The time-specific causal effect after implementation is

\[
\tau_{1t}=Y_{1t}(1)-Y_{1t}(0),\qquad t>T_0.
\]

The observed post-treatment outcome reveals Y₁t(1), subject to consistency and an appropriately defined intervention. The missing outcome Y₁t(0) must be constructed or bounded using assumptions and untreated observations. A synthetic control estimate is

\[
\widehat{Y}_{1t}(0)=\sum_{j=2}^{J+1}w_jY_{jt},
\]

where donor weights wⱼ are nonnegative and sum to one. The estimated time-specific effect is Y₁t−Ŷ₁t(0). The post-period average estimated effect is the mean of those differences over the designated horizon.

The weights are chosen using the pre-treatment periods, typically to minimize an outcome-fit criterion or a weighted combination of predictor differences. The constraints keep the estimate within the convex hull of donor combinations. This can reduce implausible extrapolation, but it may also prevent a good fit when the treated unit is unlike every possible donor mixture.

A low pre-treatment fitting error is useful because it shows that a feasible donor combination reproduces the historical treated trajectory. It does not logically imply that the same combination reproduces the untreated post-treatment trajectory. That requires a stability assumption about the untreated relationships and any factor structure underpinning them.

The causal estimand in this article is the average treatment effect for the single treated unit across the 12 post-treatment periods. Because the treatment effect is constant at −2, every post-treatment effect and their average equal −2. In real applications the time profile may be dynamic, so the chosen horizon and aggregation matter.

## Why convex combinations are attractive

Ordinary regression can extrapolate beyond the observed donor outcomes. Synthetic control with nonnegative, sum-to-one weights stays within combinations of the donor trajectories at each period. Its interpretation is straightforward: the synthetic unit is built from observed untreated units whose historical outcomes jointly resemble the treated unit.

This constraint has practical and statistical consequences. It limits unstable negative extrapolation in some designs, and it makes the contribution of each donor inspectable. However, one donor can still receive most of the weight, so the nominal size of the donor pool does not necessarily indicate how many independent trajectories materially support the counterfactual.

An optimization criterion that fits pre-treatment outcomes is meaningful only in relation to the intended causal design. Including every candidate donor is not automatically appropriate. Donors affected by the treated unit's policy, subject to spillovers, or undergoing their own interventions can contaminate the comparison. A donor that fits especially well because it was exposed to a similar policy is not a valid untreated control for that effect.

The constraint also creates an overlap-like concern. If the treated unit lies far outside the range of donor histories or baseline characteristics, no convex combination may represent it well. An apparently adequate fit produced through extensive tuning or very short pre-treatment windows deserves scrutiny, particularly if it depends on an unstable choice of predictors.

In the simulation, the construction is deliberately favourable. The untreated treated-unit outcome is itself based on a true convex combination of three donors. This means the donor pool contains a valid synthetic representation before measurement noise and post-treatment violations are added. Poor performance in a failure scenario is not caused by a fundamentally unsuitable donor pool.

## A factor-model interpretation

Synthetic control is frequently motivated using latent-factor models of untreated outcomes. A stylized model is

\[
Y_{it}(0)=\delta_t+\lambda_t^{\mathsf T}\mu_i+\theta_t^{\mathsf T}Z_i+\varepsilon_{it},
\]

where δₜ represents common time effects, λₜ unobserved time-varying factors, μᵢ unit-specific factor loadings, Zᵢ observed unit characteristics, and εᵢₜ idiosyncratic disturbances. Units may respond differently to the same changing factor because their loadings differ.

A simple Difference-in-Differences model assumes that, absent treatment, group differences do not change in a way that contaminates the effect contrast. A latent-factor structure allows richer differences in untreated trajectories. Weighted donor combinations can potentially approximate the treated unit's factor loadings, so common factor movements after intervention are reproduced by the synthetic trajectory.

That promise depends on the factor structure remaining relevant over time and on the donor weights adequately representing the treated unit. A new factor that affects only the treated unit after the intervention cannot be reconstructed from donors that do not share it. The same is true of a suddenly changed factor loading or a local shock with no untreated analogue.

Good pre-treatment fit is suggestive evidence for a stable factor representation. It is not direct evidence that unobserved factor loadings will remain unchanged or that new shocks will affect treated and donor units similarly. The simulation will isolate exactly this difference between fitting a latent relationship before intervention and maintaining it afterward.

## The data-generating model

Each simulated dataset contains eight donor units and one treated unit observed for 36 periods. The first 24 periods are pre-treatment and the last 12 are post-treatment. A deterministic common component combines a gentle linear trend and a sinusoidal pattern. A stochastic latent factor follows a random walk with normally distributed innovations.

The donor outcomes are generated from the common component, donor-specific intercepts, heterogeneous loadings on the random-walk factor, and independent Gaussian observation noise. Consequently, donors move together in response to common changes but differ in levels and sensitivity to the latent factor.

To construct the treated unit's untreated potential outcome, the program randomly selects three of the eight donors and draws nonnegative mixture weights summing to one. It takes the weighted average of their observed trajectories and adds independent measurement noise. Thus a true donor-weight representation exists in every generated dataset, with imperfect fit due to the additional noise.

The structural treatment effect is exactly −2 at every post-treatment period. In the stable scenario the observed treated outcome is its untreated potential outcome minus two after treatment begins. In the post-shift scenario an additional +1.5 affects the treated unit in the post period. In the anticipation scenario an adjustment of −0.8 affects the final four pre-treatment outcomes before the −2 post-treatment effect begins.

These disturbances have different causal interpretations. The post-shift is a treated-only change in the untreated potential outcome, coinciding with the intervention. It violates stability of the treated-donor relationship and creates confounding of the estimated post-period gap. The anticipation mechanism changes the pre-treatment outcomes used for fitting, contaminating the training period with treatment-related behaviour.

No spillovers affect donors. There are no missing observations and no multiple treated units. The treatment effect is constant. These simplifying choices isolate counterfactual construction from other problems common to longitudinal policy evaluation.

## The weight optimization

For each dataset, donor weights are estimated using the 24 pre-treatment observations. The algorithm minimizes mean squared error between the treated trajectory and the weighted donor trajectory, subject to nonnegativity and a unit-sum constraint:

\[
\widehat{w}=\arg\min_{w\ge 0,\;\mathbf 1^{\mathsf T}w=1}
\frac{1}{T_0}\sum_{t=1}^{T_0}
\left(Y_{1t}-\sum_{j=2}^{J+1}w_jY_{jt}\right)^2.
\]

The optimization uses the sequential least-squares quadratic programming implementation in SciPy. It starts with equal donor weights and uses strict numerical tolerances. If optimization does not converge successfully, the implementation raises an error rather than silently returning an arbitrary weight vector.

The fitted weights are applied unchanged to all 12 post-treatment observations. The post-treatment average gap is the synthetic control estimate. The program also reports pre-treatment root mean squared prediction error, which summarizes how closely the synthetic unit reproduces the observed treated outcomes before implementation.

This is a transparent version of the classical synthetic control idea, not a benchmark of all contemporary synthetic control estimators. It does not optimize a nested predictor-weighting matrix, use augmented synthetic control, cross-fitting, penalization, matrix completion, or interactive fixed-effects estimation. Those alternatives could improve performance in particular designs; they cannot dispense with assumptions about the untreated post-intervention outcome.

Keeping the optimization elementary lets the article focus on a central issue: under what data-generating conditions does a good pre-treatment fit support a credible future counterfactual?

## Two simple baselines

A useful benchmark needs alternatives. The first is an eight-period before–after comparison: the mean observed treated outcome during the 12 post-treatment periods minus its mean over the final eight pre-treatment periods. This ignores all donor outcomes and therefore attributes common time changes to the intervention.

The second is an equal-donor Difference-in-Differences estimate. It calculates the treated unit's change over the same windows and subtracts the corresponding change in the unweighted average of eight donor trajectories. It uses the donor pool but does not optimize the weights to represent the treated unit's baseline dynamics.

Neither baseline is designed to be universally optimal. The before–after comparison is intentionally vulnerable to shared movements over time. The equal-donor DID contrast may perform reasonably when all donor units experience an average change similar to the treated unit's untreated change, but it can be inefficient or misleading when heterogeneous factor loadings produce differential movements.

The comparison is informative because the same simulated data are passed to all three estimators. Differences in their performance cannot be attributed to one method receiving a different sample or a different treatment effect. They reflect the counterfactual construction and the data-generating mechanisms.

There is also a useful interpretive limit. If the treated-only post-period shock is completely absent from all donors, neither synthetic control nor ordinary DID can identify the intervention effect from these observations alone. A good fit criterion may favor one estimator, but no reweighting of unaffected donors can produce a counterfactual that includes the new treated-specific shock.

## How the experiment was run

The program generates 300 independent datasets per scenario with a fixed NumPy random generator seed of 20261017. Each dataset draws new donor intercepts, factor loadings, latent factor innovations, mixture weights, observation disturbances, and the treated unit's additional measurement noise.

For every dataset the weights are refitted from scratch. The three estimators are then applied, and the program records their estimated effects. Across the 300 replications it calculates the mean, median, standard deviation, bias relative to −2, and root mean squared error. The mean pre-treatment RMSPE of the synthetic fit is reported alongside all methods as a diagnostic of the underlying synthetic comparison.

The numerical experiment therefore contains 900 independently generated datasets and 2,700 method-specific treatment-effect estimates. The same sample size, pre-period length, post-period length and donor count are used in every scenario. There is no hyperparameter search or selection of the simulation that makes a preferred method look most successful.

The seed makes the table exactly reproducible with the specified environment. It does not make the finite-simulation means population identities. Differences of a few thousandths could vary under another seed. The large effects of a +1.5 post-period shock, by contrast, are directly implied by the structural modification and should not be attributed to random Monte Carlo variation.

The exact script and generated CSV are supplied with the article. The output has not been rounded during generation; displayed results below are rounded for readability. No external dataset is downloaded, and none of the numerical findings is presented as a measured effect of a real policy.

## Results under stable counterfactual dynamics

In the first scenario, the relationship between the treated unit's untreated potential outcome and the donor pool remains stable after treatment begins. The estimates are:

| Method | Mean estimate | Standard deviation | RMSE |
| --- | ---: | ---: | ---: |
| Synthetic control | −1.9909 | 0.1061 | 0.1064 |
| Before–after | −1.0209 | 0.3300 | 1.0331 |
| Equal-donor DID | −1.9974 | 0.3015 | 0.3010 |

The true treatment effect is −2. Synthetic control recovers it with small bias and considerably lower RMSE than either baseline. Equal-donor DID also has negligible mean bias in this constructed design, but higher variance because the equal donor weights do not necessarily reproduce the treated unit's latent factor loadings as effectively as the optimized combination.

The before–after comparison is biased upward by nearly one unit. The common time trend continues between the final eight pre-treatment periods and the post-treatment period, so the observed change combines the −2 intervention with the general evolution of the outcome. Without a comparison trajectory, the method cannot separate them.

Mean pre-treatment RMSPE for the synthetic fit is about 0.1795. That is modest relative to the two-unit treatment effect, and the post-treatment synthetic estimate performs well. In this scenario good historical fit is accompanied by the stability needed to interpret the synthetic post-period trajectory as the untreated counterfactual.

The point is not that optimized weights are automatically better than DID. In this simulation the outcome was created from a convex donor combination, which directly favors the synthetic control model class. The result illustrates performance under a data-generating mechanism aligned with the estimator, not a universal ranking for applied policy evaluation.

## The post-treatment shock

In the second scenario, the treated unit receives a new +1.5 untreated-outcome shift throughout the post-treatment period. This shift is separate from the intervention's −2 effect. The observed treated-versus-synthetic gap therefore contains both:

\[
\text{Observed gap}\approx -2+1.5=-0.5.
\]

The results are:

| Method | Mean estimate | Standard deviation | RMSE |
| --- | ---: | ---: | ---: |
| Synthetic control | −0.4925 | 0.0945 | 1.5104 |
| Before–after | +0.4769 | 0.3903 | 2.5073 |
| Equal-donor DID | −0.4934 | 0.3250 | 1.5411 |

Synthetic control estimates a mean effect almost exactly equal to −0.5, but the true intervention effect remains −2. The approximately +1.5 difference is not an optimization failure. It is the untreated post-period shock that no unaffected donor can reveal. Equal-donor DID exhibits almost exactly the same mean bias.

The important diagnostic is the pre-treatment fit. Mean RMSPE is 0.1830, barely different from the stable scenario's 0.1795. The synthetic unit reproduces the treated unit's earlier outcomes almost as well as before. The counterfactual nevertheless fails after treatment because a new treated-only process starts precisely when the policy is introduced.

A published evaluation could show an impressively close historical synthetic trajectory and a sharp post-policy divergence. In this scenario that divergence is real, but it is not exclusively caused by the treatment. The post-treatment gap alone cannot separate the two simultaneous changes.

The numerical example is deliberately transparent. A different post-only shock could make the effect look stronger, change its sign, or produce a near-zero estimate. Pre-treatment fitting quality cannot determine the magnitude or direction of a shock that has not yet occurred.

## Why the failure is not repaired by better weights

Suppose the optimization recovered the true convex donor weights perfectly. The treated unit's untreated outcome after intervention would still contain the additional +1.5 post-period shift. The synthetic combination would reproduce the original donor-based component but not the new shift.

The problem can be written as

\[
Y_{1t}(0)=\sum_j w_j^*Y_{jt}+u_{1t}+s_t,
\]

where u₁ₜ is idiosyncratic noise and sₜ is zero before treatment but +1.5 after treatment. The fitted synthetic outcome omits sₜ because every donor lacks it. The estimated treatment effect becomes τₜ+sₜ plus residual fitting error.

This is an identification problem. Optimizing the historical weights more precisely, using a different numerical solver or increasing the precision of pre-treatment outcome measurements does not reveal a future treated-only shock. A more sophisticated time-series model might forecast such a shock only if additional predictors contain information about it. The simulated donor histories do not.

The distinction is important when interpreting algorithmic progress. Lower pre-treatment squared error is an observable prediction criterion. The missing untreated post-treatment outcome is not observable in the treated unit. Improving the criterion may reduce variance under stability while giving no protection against a violation of that stability.

A credible applied study should therefore explain why no treated-specific contemporaneous event is likely to contaminate the intervention period, or examine such events through institutional records, negative-control outcomes, additional comparison units and sensitivity analysis. The fitted historical curve is useful evidence; it is not the entire causal design.

## Anticipation and contaminated training periods

In the third scenario the treated unit responds during the final four pre-treatment periods, before the intervention officially begins. Its observed outcome is reduced by 0.8 during that interval. The synthetic control algorithm nevertheless treats all 24 pre-treatment observations as untreated training data.

The results are:

| Method | Mean estimate | Standard deviation | RMSE |
| --- | ---: | ---: | ---: |
| Synthetic control | −1.7727 | 0.1965 | 0.3003 |
| Before–after | −0.6690 | 0.3357 | 1.3726 |
| Equal-donor DID | −1.6234 | 0.2884 | 0.4740 |

The treatment effect after implementation remains −2. The synthetic estimate is biased upward by about 0.2273, while equal-donor DID is biased upward by about 0.3766. These magnitudes are specific to the fitting window, intervention timing and generated factor structure. They should not be interpreted as general correction factors for anticipation.

Mean pre-treatment RMSPE rises from roughly 0.18 in the stable scenario to 0.3143. That change is informative: the recent treated trajectory no longer follows the same donor mixture as the earlier untreated history. The synthetic control optimization compromises between the two regimes, affecting both the fitted weights and the apparent post-treatment gap.

In applied research, anticipation can begin when a policy is announced, eligibility is communicated, or implementation becomes likely. Outcomes recorded before an administrative start date may already contain intervention-related responses. Calling them pre-treatment does not make them valid untreated observations.

A possible response is to distinguish the announcement date from the implementation date and reconsider which periods are admissible for constructing the counterfactual. Removing suspect periods may help, but the choice must be justified by institutional evidence rather than tuned until the estimated effect is large or the pre-treatment error is attractive.

## Why pre-treatment RMSPE is not a validity test

Pre-treatment root mean squared prediction error is

\[
\operatorname{RMSPE}_{pre}=
\sqrt{\frac{1}{T_0}\sum_{t\le T_0}
\left(Y_{1t}-\widehat Y_{1t}(0)\right)^2}.
\]

It summarizes observable fit, and it can reveal that the chosen donors cannot approximate the treated unit's historical trajectory. This is valuable. If a synthetic control fails to reproduce the treated series before treatment, the analyst should be reluctant to treat its post-treatment predictions as a credible counterfactual without further justification.

The converse does not follow. A low pre-treatment RMSPE is not a test of whether the treated-donor relationship will remain stable afterward. In the stable and post-shift simulations, the pre-treatment RMSPE distributions are very similar even though post-treatment causal RMSE differs by more than an order of magnitude.

An analyst who selected the method solely on pre-treatment fit would therefore prefer it in both scenarios, without observing the key difference. A placebo exercise that assigns artificial treatment dates before the actual intervention may detect failures that also occurred in the past, but cannot guarantee that no unique shock will occur later.

Pre-treatment fit should be interpreted as one component of design credibility. Donor suitability, spillovers, prior interventions, policy timing, external events and alternative assumptions all matter. The graph connecting historical outcomes to the counterfactual cannot be validated merely by fitting the historical outcomes well.

## The donor pool is part of the design

The choice of donor units is not a neutral preprocessing step. Donors should be unexposed to the intervention and sufficiently comparable in the relevant untreated outcome-generating processes. Including a donor influenced by spillovers can make the synthetic counterfactual move in response to the same policy, potentially attenuating or otherwise distorting the estimated treatment effect.

A donor can also experience another intervention during the post-period. If an important donor's outcome changes because of its own contemporaneous policy, the synthetic control inherits that change. The resulting gap may reflect both policies rather than the treated intervention alone.

The simulation excludes these complications. Donor outcomes are generated without any treatment effects or contamination, and the treated unit's baseline relationship is a genuine convex donor mixture. That design isolates stability and anticipation problems, but real-world donor selection often requires more substantive investigation.

It can be useful to compare estimates under defensible donor restrictions, leave-one-donor-out analyses and alternative predictors. A result that changes drastically when one high-weight donor is removed deserves scrutiny. But mechanical stability across arbitrary donor lists is not proof of causal validity; several contaminated donors may agree with one another.

The relevant question is not whether a donor produces a pleasing fit. It is whether it provides information about the untreated outcome the treated unit would have experienced after intervention. The distinction becomes visible when we look beyond the optimization objective to the institutional context.

## Time-series dependence and uncertainty

Repeated observations over time are not independent replications of an intervention. Thirty-six monthly outcomes from one treated municipality do not constitute thirty-six independently assigned policies. Trends, seasonal patterns and latent factors create dependence that standard regression formulas may not capture.

The Monte Carlo benchmark estimates variability by regenerating whole panel datasets independently. Its reported standard deviations describe the distribution of estimates across simulated worlds under each fixed mechanism. They are not standard errors calculated from one observed policy time series.

In a real single-unit intervention study, inference can be difficult because the number of treated units is small and the post-treatment outcome is autocorrelated. Placebo interventions across donor units, in-time placebos, conformal approaches, permutation-based procedures and model-based uncertainty frameworks provide possible tools under their respective assumptions.

An in-space placebo assigns treatment status hypothetically to donor units and compares their post-period discrepancies with the treated unit's discrepancy. Such comparisons can be informative, but treated and donor units may differ in fit quality, volatility and suitability for synthetic reconstruction. Normalizing post-treatment discrepancies by pre-treatment error is one possible diagnostic, not a universal significance test.

Inferential uncertainty should also distinguish sampling or stochastic fluctuations from uncertainty about the entire counterfactual assumption. A narrow distribution under a model with stable factors does not account for the possibility of a unique untreated shock in the policy period. The post-shift experiment makes that limitation explicit.

## Falsification and placebo interventions

An informative synthetic control analysis should examine whether effects appear before the intervention, whether alternative outcomes known not to be affected show similar discontinuities, and whether assigning placebo dates or units produces gaps resembling the main estimate.

These exercises can expose some implausible causal interpretations. If a synthetic control shows an apparent policy effect years before announcement, the estimate may be tracking an unstable relationship rather than an intervention. If a negative-control outcome changes at the intervention date despite no plausible causal pathway, a common shock or measurement problem may be involved.

But a placebo does not observe the missing untreated post-treatment trajectory. A unique new shock can arrive only in the actual intervention period, leaving all earlier placebo tests looking reassuring. Nor does passing a placebo establish that donors are immune to spillovers or that the intervention had a unique version.

Falsification tests should be selected to challenge concrete failure mechanisms, not accumulated as a checklist of nonsignificant results. A negative-control outcome is useful only when there is a reason it should not respond to treatment while still sharing the bias mechanism of concern.

In the experiment, the post-shift scenario is a particularly difficult failure for historical placebos because the deviation begins after intervention. The anticipation scenario is more likely to leave detectable abnormalities within the pre-period, although whether a diagnostic detects them depends on its timing and power.

## Bandwidths, training periods and the time horizon

A synthetic control analysis must decide how much history to use for fitting, whether to emphasize recent periods, which predictors enter the optimization and how long after treatment to measure the effect. These choices influence both the counterfactual and the estimand.

A longer pre-treatment window may help identify persistent latent factor relationships, particularly when factors vary over time. It may also include obsolete regimes or structural breaks that reduce the relevance of older periods. A shorter window can fit recent conditions closely but risks overfitting transitory noise or missing slow-moving factors.

The 24-period window in this experiment is fixed rather than selected from results. The anticipation shock contaminates its final four observations, making the choice consequential. Removing those observations would avoid direct training on anticipatory outcomes if the analyst knew the response began there, but such knowledge is not guaranteed in practice.

Similarly, a constant post-treatment effect is an intentional simplification. Real effects may build gradually, dissipate or change sign. Reporting only one average can hide that dynamic pattern. A full evaluation should display time-specific treated-versus-synthetic gaps and state which horizon matters for the policy decision.

The purpose of such choices is not to obtain the most visually convincing trajectory. It is to construct a comparison that corresponds to the intervention and the period over which the causal question is defined.

## Connection with Difference-in-Differences

Part 6 examined Difference-in-Differences under parallel trends, post-policy divergence and anticipation. Synthetic control relaxes one part of that comparison by allowing donor weights to approximate more complex historical dynamics instead of assuming that the untreated treated-group change equals a simple comparison-group average change.

That extra flexibility is valuable when stable factor loadings can be reconstructed from the donor pool. In the stable simulation, equal-donor DID is nearly unbiased on average but more variable than synthetic control. The optimized weights better represent each randomly generated treated unit's mixture of factors.

However, the shared vulnerability is clear under the post-shift design. Both methods estimate roughly −0.49 when the true effect is −2. The untreated treated-unit shock is absent from the donor trajectories, so both counterfactual constructions omit it. A different estimator of an unsupported counterfactual remains unsupported.

Anticipation also affects both methods because the pre-treatment baseline no longer represents the intended untreated state. The estimators respond differently to the contamination because synthetic control uses the full pre-history for weight fitting, whereas the equal-donor DID baseline is limited to the final eight pre-period observations.

These comparisons argue against treating synthetic control as a replacement for causal design. It provides a potentially stronger way to construct a comparison under particular factor and stability assumptions. It does not eliminate the need to explain why that comparison represents the treated unit's missing outcome.

## Beyond the classical estimator

A substantial methodological literature extends synthetic control beyond the original nonnegative-weight estimator. Augmented synthetic control methods use outcome modelling to correct imperfect pre-treatment fit, trading a pure convex donor representation for additional modelling flexibility. Generalized synthetic control and interactive fixed-effects methods exploit latent factor structure across units and time. Matrix completion approaches formulate counterfactual construction as a missing-data problem in a structured outcome panel.

These methods can address genuine shortcomings of classical synthetic control. When the treated unit is not inside the donor convex hull, augmentation or regularization may reduce extrapolation bias under suitable model assumptions. When there are several treated units and adoption dates, panel estimators can use more of the observed untreated information than a one-treated-unit construction.

None is guaranteed to correct a treated-only post-period shock that is unanticipated by all measured covariates and donor outcomes. The information missing from the counterfactual remains missing unless the design supplies a new source for identifying it.

That is not an argument against methodological development. It is an argument for evaluating methods against the specific violations they are designed to address. A procedure can improve prediction under one model class while remaining vulnerable to another. Robustness must be demonstrated for named departures from assumptions, not inferred from sophistication.

The accompanying code stays with the classical construction so the first principles are visible before these extensions are investigated in later work.

## What a policy evaluation should report

The intervention, target population and outcome horizon should be defined before fitting a synthetic unit. The analyst should document when treatment was announced and implemented, which units could have been affected, and whether the intended estimand is a level effect at a given date or an average effect across a post-treatment interval.

The donor pool needs substantive justification. For each potential donor, explain whether it was untreated, whether spillovers are plausible, and whether important contemporaneous events could change its outcome independently of the treated intervention. Report the fitted weights, not merely the aggregate synthetic trajectory.

Pre-treatment fit should be shown using the actual historical series, with quantitative measures of error and enough context to distinguish a persistent pattern from noise. When fit is poor, explain whether alternative predictors or donor sets produce a more defensible comparison. When fit is excellent, avoid treating it as proof of future validity.

Results should include the treated and synthetic outcome paths, time-specific gaps and the post-period aggregate tied to the decision. Placebo results, sensitivity to donors and pre-period windows, and diagnostics for influential units can show whether the estimate relies on brittle choices.

Finally, distinguish uncertainty about sampling and stochastic variability from uncertainty about identifying the untreated future. A good pre-treatment optimization score cannot quantify the consequences of an unknown treated-only post-period shock without additional assumptions or evidence.

## What the experiment cannot show

This benchmark uses eight donors, one treated unit, 24 pre-treatment periods and 12 post-treatment periods. The untreated treated-unit trajectory is generated from a known convex donor combination, which is favourable to classical synthetic control. All units have complete outcome histories, and donors remain untreated and unaffected by the policy.

The factor model includes a common deterministic component, a stochastic random-walk factor, heterogeneous donor loadings and Gaussian observation disturbances. More complex latent dynamics, changing factor loadings, seasonality, measurement error and heterogeneous noise could alter the relative performance of estimators.

Only three scenarios are considered. The post-treatment shift is fixed at +1.5 and the anticipatory reduction at −0.8 for four periods. These are controlled demonstrations, not estimates of typical bias sizes in policy evaluations. Different shock timing or magnitude would produce different numerical results.

The code uses a fixed equal donor pool and does not optimize predictors, tune the training period, estimate uncertainty intervals or conduct formal placebo-inference procedures. The before–after and equal-donor DID baselines are intentionally simple. Other counterfactual methods could outperform them in different designs.

The repeated simulation is valuable because the true effect is known, but it cannot validate the historical assumptions of any actual case study. In real data the post-treatment untreated outcome is unavailable by definition. Causal credibility depends on whether donor relationships, institutional history and external evidence make the counterfactual construction plausible.

Most importantly, the post-shift scenario violates a stability condition that the observed pre-period cannot verify. There is no algorithmic correction for that violation in the supplied information. The bias is a limitation of identification in the constructed design, not simply a poor choice of weights.

## What the investigation established

With stable untreated dynamics, synthetic control estimated a true effect of −2 with mean −1.9909 and RMSE 0.1064. Equal-donor DID was also nearly unbiased, but had RMSE 0.3010; a before–after comparison was substantially biased by common time movement.

When a new +1.5 untreated shock affected only the treated unit after intervention, synthetic control estimated −0.4925 instead of −2. Its mean pre-treatment RMSPE, 0.1830, was almost the same as in the valid design. The historical fit remained reassuring while the post-treatment counterfactual failed. Equal-donor DID suffered essentially the same systematic error.

Anticipation altered the observations used for fitting and shifted the synthetic estimate to −1.7727. Its higher pre-treatment RMSPE supplied a possible warning, but the magnitude of bias depended on how the anticipatory response interacted with the fitting window and donor representation.

Synthetic control is effective when historical donor relationships genuinely describe the treated unit's untreated future. Its advantage is constructing that relationship more flexibly than a simple comparison average. Its limitation is that historical fit cannot establish what would have happened after intervention under shocks the historical data do not contain.

The broader lesson is continuous with the previous articles: an estimator can behave extremely well for its statistical objective while the causal conclusion depends on assumptions outside that objective. A convincing counterfactual requires more than a convincing pre-treatment curve.

## Code

The numerical benchmark is reproduced by `experiment.py` in `causal_inference_statistics/part9/`. It uses NumPy and SciPy, fixes seed 20261017 and writes the scenario-by-estimator results to `results.csv`. The method refits donor weights independently for each generated panel. No external data are required. All numerical summaries quoted in this article come from that script.

## The series

1. When Regression Gets the Causal Effect Wrong
2. Randomization, Confounding and the Value of Experimental Design
3. Matching and Propensity Scores: When Balanced Data Still Give the Wrong Causal Effect
4. Inverse Probability Weighting: When a Few Observations Control the Result
5. Directed Acyclic Graphs and the Cost of Controlling for Everything
6. Difference-in-Differences: When Parallel Trends Fail
7. Instrumental Variables: When a Strong First Stage Is Not Enough
8. Regression Discontinuity: Why a Sharp Cutoff Is Not Enough
9. Synthetic Controls and Counterfactual Time Series: When a Good Pre-Treatment Fit Is Not Enough (this article)

## References

Abadie, A. (2021). Using Synthetic Controls: Feasibility, Data Requirements, and Methodological Aspects. *Journal of Economic Literature*, 59(2), 391–425. https://doi.org/10.1257/jel.20191450

Abadie, A., Diamond, A. and Hainmueller, J. (2010). Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California's Tobacco Control Program. *Journal of the American Statistical Association*, 105(490), 493–505. https://doi.org/10.1198/jasa.2009.ap08746

Abadie, A., Diamond, A. and Hainmueller, J. (2015). Comparative Politics and the Synthetic Control Method. *American Journal of Political Science*, 59(2), 495–510. https://doi.org/10.1111/ajps.12116

Abadie, A. and Gardeazabal, J. (2003). The Economic Costs of Conflict: A Case Study of the Basque Country. *American Economic Review*, 93(1), 113–132. https://doi.org/10.1257/000282803321455188

Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W. and Wager, S. (2021). Synthetic Difference-in-Differences. *American Economic Review*, 111(12), 4088–4118. https://doi.org/10.1257/aer.20190159

Athey, S., Bayati, M., Doudchenko, N., Imbens, G. and Khosravi, K. (2021). Matrix Completion Methods for Causal Panel Data Models. *Journal of the American Statistical Association*, 116(536), 1716–1730. https://doi.org/10.1080/01621459.2021.1891924

Ben-Michael, E., Feller, A. and Rothstein, J. (2021). The Augmented Synthetic Control Method. *Journal of the American Statistical Association*, 116(536), 1789–1803. https://doi.org/10.1080/01621459.2021.1929245

Cattaneo, M. D., Feng, Y. and Titiunik, R. (2021). Prediction Intervals for Synthetic Control Methods. *Journal of the American Statistical Association*, 116(536), 1865–1880. https://doi.org/10.1080/01621459.2021.1979561

Chernozhukov, V., Wüthrich, K. and Zhu, Y. (2021). An Exact and Robust Conformal Inference Method for Counterfactual and Synthetic Controls. *Journal of the American Statistical Association*, 116(536), 1849–1864. https://doi.org/10.1080/01621459.2021.1920957

Doudchenko, N. and Imbens, G. W. (2016). Balancing, Regression, Difference-in-Differences and Synthetic Control Methods: A Synthesis. NBER Working Paper 22791. https://doi.org/10.3386/w22791

Firpo, S. and Possebom, V. (2018). Synthetic Control Method: Inference, Sensitivity Analysis and Confidence Sets. *Journal of Causal Inference*, 6(2). https://doi.org/10.1515/jci-2016-0026

Gobillon, L. and Magnac, T. (2016). Regional Policy Evaluation: Interactive Fixed Effects and Synthetic Controls. *Review of Economics and Statistics*, 98(3), 535–551. https://doi.org/10.1162/REST_a_00537

Xu, Y. (2017). Generalized Synthetic Control Method: Causal Inference with Interactive Fixed Effects Models. *Political Analysis*, 25(1), 57–76. https://doi.org/10.1017/pan.2016.2