# Matching and Propensity Scores: When Balanced Data Still Give the Wrong Causal Effect

*Part 3 of Causal Inference in Statistics. A reproducible experiment with propensity-score matching, two assignment mechanisms and a deliberately unobserved confounder. Why excellent balance on measured variables cannot certify causal identification. Python code provided.*

An observational comparison is often criticized because the treated and untreated people are different. Matching offers an appealing response: find untreated people who resemble the treated, compare their outcomes, and interpret the resulting difference as the treatment effect. The appeal is understandable. Unlike an unadjusted regression coefficient, a matched comparison appears to compare similar individuals. It can also make the empirical support of a causal claim easier to inspect.

But similarity is not the same as exchangeability. A matching procedure can produce almost indistinguishable distributions of the variables available to the analyst while leaving the treatment and comparison groups different on the causes the dataset did not measure. This is not a rare numerical pathology or a problem that necessarily disappears with more sophisticated distance functions. It is an identification problem.

I investigate it using two otherwise similar data-generating mechanisms. In both, the treatment effect is exactly 1, one baseline covariate is observed, and a second baseline characteristic affects the outcome. In the first mechanism, treatment assignment depends only on the observed covariate. In the second, assignment depends on the observed covariate *and* the unobserved characteristic. The analyst estimates the propensity score using only the observed variable, then matches treated individuals to controls whose estimated log-odds of treatment are closest.

The numerical result is revealing. Across 300 independently generated datasets of 1,500 observations each, the matched estimate averages 1.0128 when the assignment process depends only on the measured covariate. When the hidden variable also drives assignment, the matched estimate averages 2.4294. In both experiments, the mean difference in the observed covariate after matching is approximately zero. The observed balance diagnostic therefore looks successful in both cases, although only one of the resulting comparisons is close to the true causal effect.

The lesson is not that matching is useless. Matching can be a transparent and effective design tool under defensible assumptions, and it can expose areas where treated and untreated observations have little in common. The narrower claim is that balancing measured variables does not show whether the variables required for identification have been measured in the first place.

## The decision that motivates matching

Suppose a public programme offers additional support to people facing barriers to employment. The programme is not assigned randomly. Staff examine each person's circumstances, decide who needs support and allocate resources accordingly. A subsequent evaluation compares people who received support with those who did not, using unemployment outcomes measured later.

The first difficulty is selection. People receiving support may be older, have less previous employment experience, or live in places with fewer opportunities. Those factors can affect both programme participation and unemployment outcomes. A difference in average outcomes between treated and untreated people can therefore reflect the programme, the initial circumstances of the participants, or both.

One strategy is to model the outcome conditional on treatment and the measured characteristics. Another is to construct a comparison group that resembles the treated participants in characteristics determined before treatment. Propensity-score matching takes the second route by using a statistical model to compress the observed treatment assignment information into a probability of receiving treatment.

The intended decision is still causal: would the programme change outcomes if offered to the people under consideration? Matching is not inherently the estimand. It is one possible way to approximate the counterfactual comparison required for that question. The estimand has to be defined independently of the algorithm used to assemble a sample.

The practical advantage of matching is that it can turn abstract modelling assumptions into concrete comparisons. We can examine who was matched, whether the covariates overlap, how much data were discarded and whether one control observation was reused many times. Those details are useful because an apparently precise effect estimate can conceal an implausible comparison population.

The problem is that an unrecorded reason for programme assignment may remain active after all the observed characteristics have been balanced. In the employment example, a caseworker might know something about motivation, health, informal support or immediate job prospects that was not entered into the evaluation dataset. If that information also affects outcomes, a matched sample can still compare people who had different counterfactual prospects.

The simulation below uses abstract normal variables rather than a real programme. That choice is deliberate: the causal truth is known, so the remaining bias can be measured rather than merely suspected.

## Potential outcomes and the treatment effect

Let A be a binary treatment indicator, X an observed baseline covariate and U a second baseline characteristic. The potential outcomes of individual i are Y_i(1) and Y_i(0). Under a well-defined intervention, the observed outcome is

\[
Y_i=A_iY_i(1)+(1-A_i)Y_i(0).
\]

The average treatment effect is

\[
\operatorname{ATE}=E[Y(1)-Y(0)].
\]

Matching treated individuals to untreated ones often targets the average treatment effect among the treated, the ATT,

\[
\operatorname{ATT}=E[Y(1)-Y(0)\mid A=1].
\]

Those two quantities are not generally equal. If effects vary across individuals, the people selected for treatment may have a different mean effect from the wider population. A method that discards treated people who have no acceptable match may target an even narrower matched subset. Reporting an estimate as *the treatment effect* without specifying its population is therefore incomplete.

In this experiment the individual treatment effect is constant and equal to one. I generate

\[
Y(0)=1.5X+1.5U+\varepsilon,
\]

\[
Y(1)=1+1.5X+1.5U+\varepsilon,
\]

with X, U and ε independently standard normal. Thus

\[
Y(1)-Y(0)=1
\]

for every person. The ATE, ATT and effect within any retained matched subset all equal one. This convenience removes differences between effect targets as a possible explanation for the observed bias.

Both X and U are prognostic because they affect outcomes under either treatment. Whether they are also confounders depends on the treatment assignment mechanism. That causal role cannot be inferred from the outcome equation alone.

The first article in this series showed how an omitted common cause changes the population regression coefficient. The second showed how randomization can eliminate selective assignment. This article instead asks whether a method designed to make observed treatment groups look comparable succeeds when only part of the selection mechanism is visible.

## What a propensity score represents

The propensity score is the conditional probability of receiving treatment given measured pre-treatment covariates:

\[
e(X)=P(A=1\mid X).
\]

Rosenbaum and Rubin (1983) established its balancing role: under their conditions, conditioning on the true propensity score balances the distribution of the measured covariates used to define it between treated and untreated groups. The dimensionality reduction is important. Instead of exact matching simultaneously on several covariates, one can form comparisons using a scalar assignment score.

This result is sometimes presented as though the propensity score itself creates the conditions for causal inference. It does not. Balancing a collection of observed covariates is a statistical property of the score. Identifying a causal effect requires assumptions connecting treatment assignment to both potential outcomes.

The central assumption is conditional exchangeability, often written

\[
(Y(0),Y(1))\perp A\mid X.
\]

It states that, conditional on the measured covariates X, treatment assignment does not carry additional information about either potential outcome. In observational work this is an assumption about the assignment process, not a property verified by fitting a logistic regression.

Overlap, or positivity, is also required for the desired comparison. In a binary-treatment setting, a common version is

\[
0<P(A=1\mid X=x)<1
\]

for the covariate values relevant to the target population. Without sufficient treated and untreated observations in comparable regions, an effect estimate must rely on extrapolation or move to a narrower population.

Consistency is another requirement: the observed outcome under the treatment actually received must correspond to the potential outcome defined by the causal question. Ambiguous treatment versions, interference or post-treatment selection can complicate that connection. Matching does not remove those issues.

Under exchangeability, overlap and consistency, suitable comparisons conditional on X identify a causal contrast. The propensity score can facilitate such comparisons because it is a balancing score. But when exchangeability fails given X, excellent score matching cannot repair the omitted information.

## Two assignment mechanisms

The outcome-generating process is identical in both scenarios. What changes is treatment selection. First, generate treatment according to

\[
P(A=1\mid X,U)
=\operatorname{expit}(0.9X),
\]

where

\[
\operatorname{expit}(z)=\frac{1}{1+\exp(-z)}.
\]

Treatment depends on X but not on U. Because U is independent of X and of the random draw used for treatment, conditioning on X blocks the association between treatment assignment and the potential outcomes in this constructed model. U predicts outcomes, but is not a confounder in this first mechanism.

For the second scenario, generate treatment from

\[
P(A=1\mid X,U)
=\operatorname{expit}(0.9X+1.2U).
\]

Now both X and U affect assignment. Because both also affect outcomes, each is a common cause of treatment and outcome. The analyst sees X but not U. The treatment model fitted by the analyst therefore omits an assignment variable that is also prognostic.

The two mechanisms share the same causal effect, marginal distributions of baseline variables and structural outcome equation. The distinction is entirely in which variables influence treatment. This makes the experiment an explicit test of whether balancing observed variables can conceal a failure of conditional exchangeability.

The simulation makes no claim that real analysts know which variables are hidden. Indeed, the essential observational difficulty is that they often do not. Here, U is retained internally so that the experiment can measure the residual imbalance that a real analyst would not be able to inspect.

Treatment probabilities are strictly between zero and one in both mechanisms, so the experiment does not create structural positivity violations. Nevertheless, finite samples can contain treatment-propensity regions with few suitable controls. The matching procedure therefore includes a caliper and reports how much of the treated group remains matched.

## How the propensity scores are estimated

For each simulated dataset, fit a logistic regression of treatment on the observed X alone. The model estimates

\[
\operatorname{logit}P(A=1\mid X)=\alpha_0+\alpha_1X.
\]

In the first mechanism, the specified logistic model corresponds to the true conditional treatment assignment relationship. In the second, the assignment depends on U as well. The fitted logistic model of A on X is then a reduced model; it provides an estimate of the observed conditional treatment probability but cannot capture the individual variation in assignment caused by U.

I use a logistic model with negligible practical regularization. The implementation uses scikit-learn's logistic regression with a large inverse regularization parameter. The propensity model is fitted anew in every replicated dataset. This matters because propensity-score estimation error, as well as sampling variation in outcomes, contributes to the observed Monte Carlo results.

I match on the fitted *log-odds* rather than on the raw probability. The log-odds scale avoids compressing distance near probabilities close to zero or one in the same way as the probability scale. It does not solve limited overlap, but it provides a commonly used metric for caliper matching.

For each treated person, the procedure finds the untreated observation with the nearest fitted log-odds. Control observations may be reused, which is matching with replacement. A match is retained only when its log-odds distance does not exceed 0.2 times the sample standard deviation of the fitted log-odds. The caliper convention is inspired by common practice, but there is no universal threshold that guarantees causal validity.

The effect estimate is the mean of the treated-minus-matched-control outcome differences among retained treated people. Because matching is with replacement, some controls receive considerable weight through repeated reuse. An uncertainty calculation that treated all matched differences as independent would generally be inappropriate. For that reason, this first experiment reports the Monte Carlo variability of the estimator across independently generated datasets rather than a naive within-dataset standard error.

The method is simple enough for the entire comparison to be reproducible without a dedicated matching library. It is not intended to be the best matching procedure for every problem. Its purpose is to show what a highly successful observed-covariate match does and does not establish.

## The experiment

For each of the two scenarios, I generated 300 independent datasets containing 1,500 observations each. The random generator seed was 20261016. Every dataset contains newly generated X, U and ε values, an assignment draw, and an observed outcome.

The code records five quantities: the unadjusted treated-control outcome difference, the propensity-matched outcome difference, the proportion of treated people retained after the caliper, the mean matched difference in X and the mean matched difference in U. The last quantity is available because this is a simulation. It would be unavailable to the analyst in the imagined observational study.

The use of only one observed covariate makes the balance problem intentionally easy. A score built from one covariate can often match that covariate extremely closely when there is reasonable overlap. This avoids conflating the identification question with the optimization problem of balancing a high-dimensional table.

A more complex example might use several correlated covariates, nonlinear treatment assignment and heterogeneous treatment effects. Those extensions matter for applied use, but none is necessary to expose the fundamental issue. If matching fails to recover a causal effect when the sole observed covariate is almost perfectly balanced, adding numerical complexity will not change the underlying reason for the failure.

All outcomes are observed, there is no measurement error in X, and treatment effects do not vary across people. These are favourable conditions. The second scenario remains biased not because of missing outcomes, heterogeneity or a poor outcome model, but because the available adjustment set does not satisfy conditional exchangeability.

## The numerical results

The Monte Carlo means are as follows.

| Quantity | Assignment uses X only | Assignment uses X and hidden U |
| --- | ---: | ---: |
| True treatment effect | 1.0000 | 1.0000 |
| Unadjusted outcome difference | 2.1504 | 3.2178 |
| Matched outcome difference | 1.0128 | 2.4294 |
| Fraction of treated retained | 0.9898 | 0.9928 |
| Matched mean difference in X | 0.0007 | 0.0006 |
| Matched mean difference in U | 0.0035 | 0.9514 |

Each numerical estimate averages 300 datasets. The results are not fitted to an external observational sample and should not be interpreted as empirical programme effects. Their role is to compare the two selection mechanisms against the known causal quantity.

In the first scenario, the unadjusted group difference is about 2.15, substantially larger than the true effect of one. Matching reduces that difference to about 1.013. The matched mean difference in X is less than one thousandth, and the matched difference in U is also close to zero on average. Because U does not affect assignment in this scenario and is independent of X, this is consistent with the causal design: balancing X addresses the systematic treated-control difference.

In the second scenario, the unadjusted difference is even larger, about 3.22. Matching reduces it to approximately 2.43, so the observed-covariate matching procedure removes part of the initial difference. But the remaining estimate is more than one unit above the true causal effect. The method appears effective according to the X-balance diagnostic, while remaining severely biased for the causal estimand.

The decisive number is the matched difference in U. In the second mechanism it averages approximately 0.951, despite the near-zero difference in X. The treated people remain substantially different from their matched controls in the baseline characteristic the analyst cannot observe. Since U enters the outcome with coefficient 1.5, this imbalance can generate a large residual outcome difference unrelated to treatment.

Both experiments retain almost all treated observations. The matched fractions average about 99%. This eliminates another possible explanation: the discrepancy is not caused by discarding most of the treated group and redefining the target population around an unusual subset. The treatment effect is constant anyway, but the high retention rate makes the comparison especially direct.

## A successful balance diagnostic can be insufficient

Balance diagnostics answer specific empirical questions. A standardized mean difference describes the difference between groups on a measured variable relative to a chosen scale. A distributional comparison may reveal differences in spread, tails or shape that are hidden by a mean. Neither directly establishes that the potential outcomes would be exchangeable conditional on the variables the analyst recorded.

In the second simulation, the matched difference in X is 0.0006 on average. If X were the only baseline variable in the dataset, this would look excellent. A balance table would suggest that the matching procedure accomplished what it was asked to accomplish. It would not reveal the difference of approximately 0.95 in U, because U would not be part of that table.

This distinction should not be dismissed as merely semantic. The mathematical property of a propensity score concerns balance on the covariates in the score, while the causal conclusion requires balance in potential outcomes conditional on an adequate covariate set. The latter does not follow automatically from the former.

If the outcome depends on an unobserved baseline variable and that variable affects treatment assignment, the difference in its distribution may persist after matching. That is true even when the observed propensity model has strong discrimination, the sample is large, the nearest-neighbour distances are tiny and the measured covariates overlap well.

A small observed standardized mean difference is therefore evidence about a matching algorithm's performance on observed data, not a test of no unmeasured confounding. The distinction becomes especially important when a study presents attractive balance plots as the principal justification for a causal effect estimate.

The appropriate scientific question is not simply whether balance improved, but whether the variables whose balance improved were sufficient to support identification. That question requires knowledge of the assignment mechanism, timing of measurements and substantive causes of outcomes. It cannot be answered by the propensity-score fit alone.

## How much bias remains after matching?

The hidden-confounding scenario permits a useful decomposition because the outcome equation is linear with a constant treatment effect:

\[
Y=A+1.5X+1.5U+\varepsilon.
\]

For a matched treated-control pair, the outcome difference is

\[
Y_T-Y_C
=1+1.5(X_T-X_C)+1.5(U_T-U_C)
+(\varepsilon_T-\varepsilon_C).
\]

Averaging across matched pairs gives the corresponding decomposition of the matched estimator. Because matching nearly eliminates the X difference, its contribution is small. The U difference remains large and contributes approximately 1.5 times 0.9514, or about 1.427, to the mean difference. Adding the true treatment effect of one gives approximately 2.427, close to the observed Monte Carlo matched estimate of 2.4294. The small remaining discrepancy reflects finite-simulation variation and residual disturbances.

This calculation is important because it shows the mechanism rather than merely reporting that a biased estimate exists. The matching algorithm accomplished a narrowly defined task: selecting controls close to the treated group in the estimated assignment score based on X. The causal comparison fails because the unmatched U component is still associated with treatment.

In the first scenario, where U does not influence assignment, the matched U difference averages close to zero, and the resulting matched estimate approaches the true effect. The same algorithm, matching metric and outcome equation succeed in one causal setting and fail in another. That comparison isolates the role of the identifying assumption more convincingly than comparing a sophisticated method with a deliberately poor prediction model.

The observed-covariate difference is also so small that tightening the caliper would not obviously solve the problem. A stricter match on X still does not match on information that was never recorded. It might reduce sample size and alter which treated individuals are represented, but it does not create conditional exchangeability when U remains an omitted common cause.

Similarly, replacing logistic regression with a flexible propensity model does not make U available. Such a model might estimate P(A=1|X) more accurately when the dependence on X is complex. But accurate estimation of that conditional probability cannot guarantee that treatment is independent of potential outcomes given X. The distinction is between improving a nuisance-function estimate and supplying missing identification information.

## The matching estimand is not always the ATE

In the simulation, the constant treatment effect simplifies the interpretation of the matched mean. In realistic settings, matching changes not only the method of comparison but potentially the population about which the effect is learned.

Nearest-neighbour matching often begins with treated units and selects controls for them. If the analysis retains all treated people and uses appropriate controls, the natural target can be the average treatment effect among treated individuals. When a caliper excludes treated individuals who have no suitable controls, the estimand may instead become the effect among the treated subset that is matchable under that rule.

That shift is not always a defect. If the original treated group has little overlap with available controls, estimating its counterfactual outcome may require unreliable extrapolation. A restricted overlap population can be a more credible target, provided the restriction is explicit and the resulting population is scientifically or politically relevant.

The effect for matchable treated people need not equal the effect for all treated people if treatment benefits vary with characteristics related to overlap. A report that begins by asking about the entire programme population and finishes by estimating the effect among people with convenient matches has changed its question. The change should be visible in the description of the estimand, not concealed in the software settings.

Matching with replacement also changes the weights implicit in the analysis. A control used for many treated units contributes repeatedly to the estimated counterfactual outcome. This may be appropriate for estimating a treated-population contrast, but it affects dependence, effective sample size and variance estimation. Counting every matched row as an independent person would exaggerate the information available.

The current experiment avoids those conceptual complications when interpreting the numerical bias because the effect is identical for everybody. The matched fraction is nevertheless reported because real analyses must account for exclusions and changes in target population.

## Overlap and the evidence in the tails

The propensity score is often introduced as a way to balance covariates, but examining the distribution of treatment probabilities can be just as valuable as computing a matched estimate. Treated people with estimated probabilities near one may have very few untreated counterparts. Untreated people with probabilities near zero may be similarly difficult to compare with treated individuals.

In such regions, nearest-neighbour matching may accept distant controls, reuse a small number of people repeatedly or discard difficult-to-match observations. Each response changes the information used by the analysis. Accepting poor matches threatens comparability. Discarding observations reduces the target population. Reusing controls can increase dependence and the influence of a few observations.

A caliper imposes a maximum acceptable distance. Here, 0.2 times the standard deviation of the fitted log-odds is used as a transparent, fixed rule. It yields very high retention in this experiment, but those figures should not be generalized to other datasets. A different assignment mechanism could create much more severe scarcity of controls even when the same fitting and matching algorithm is applied.

A useful empirical analysis would display the propensity distributions by treatment group, the distribution of matching distances and the number of times each control is reused. These diagnostics do not prove causal identification, but they reveal whether the proposed comparison is supported by observed data or sustained by a handful of unusual matches.

The positivity assumption is often presented as the statement that every treatment probability lies strictly between zero and one. In applied data, the practical issue is frequently weaker overlap: treatment probabilities may be formally nonzero yet so extreme that the available sample cannot support a stable comparison. Large inverse-probability weights and poor matching options are two manifestations of the same scarcity of comparable observations.

This motivates the next article on inverse probability weighting. Matching and weighting use estimated treatment assignment information differently, but both require a plausible adjustment set and sufficient support for the desired causal contrast.

## Why balance must be checked after matching

A propensity-score model is not a causal model simply because it predicts treatment. Its classification performance can be high while the matched groups remain poorly balanced on important covariates. Conversely, a treatment model with modest discrimination may be entirely adequate for constructing a balanced comparison in a setting with substantial overlap.

The relevant diagnostic for a matching analysis is therefore the balance achieved *after* constructing the matched sample, not just the fit of the propensity regression. That includes checking the distributions of the original covariates, not only the fitted score. Two groups can have similar distributions of an estimated scalar score while differing in combinations of variables that matter for the outcome.

The distinction has encouraged substantial methodological work on diagnostics and alternatives. Austin (2009) emphasizes standardized differences in observational comparisons, while Stuart (2010) surveys matching methods and the importance of designing analyses before outcome comparisons. King and Nielsen (2019) argue that conventional propensity-score matching can in some settings increase imbalance and model dependence, despite its popularity.

Their criticism should not be flattened into the statement that propensity scores must never be used. Methods differ in whether they match on scores, weight by scores, stratify or use scores as part of broader estimators. The conditions under which a score-matching algorithm improves covariate balance are not identical to those under which propensity information is useful for weighting or doubly robust estimation.

The practical response is to evaluate the design that was actually constructed. Does matching improve balance across important measured variables? Are there adequate controls for treated individuals? Are discarded observations reported? Is the chosen matching method appropriate for the target estimand? And, separately, what substantive argument supports the assumption that the adjustment set addresses confounding?

In this experiment, the first set of diagnostics is reassuring while the second causal condition fails in the hidden-confounding scenario. That separation is the main finding.

## Matching is not randomization

Matched treated and control groups are sometimes called a pseudo-randomized or quasi-randomized sample. Such terminology can be misleading if it suggests that the matching algorithm recreated an actual randomized assignment mechanism.

Random assignment breaks the dependence between treatment and both measured and unmeasured baseline potential-outcome causes in expectation under the specified allocation mechanism. Matching can balance only the information used to construct or check the comparison, except to the extent that substantive assumptions justify broader conditional exchangeability.

In the first simulated scenario, conditioning on X is sufficient, so matching works well. In the second, treatment depends on U and matching fails. The observed diagnostic difference between the two scenarios is nearly nonexistent. An investigator looking only at matched X values could not infer which assignment mechanism was operating.

The comparison also highlights the difference between design and estimation. A randomized experiment begins with an assignment rule that supports identification before outcomes are analysed. An observational matching study begins with an assignment rule that already occurred, often incompletely recorded, and attempts to construct a comparison under assumptions about that rule.

This does not imply that matching is unscientific or that randomized trials are free of implementation difficulties. It means the source of the identifying assumptions differs. A careful observational design can produce valuable causal evidence, especially when randomization is unavailable. But the evidence must be defended through the treatment assignment process, timing, covariate selection, overlap, sensitivity analysis and relation to the target question.

Simply making two observed tables look alike is not equivalent to assigning people randomly.

## Hidden confounding and sensitivity analysis

Because the simulation records U, it can measure the amount of hidden imbalance directly. An observational study cannot usually do so. That is precisely why sensitivity analysis is important. Instead of claiming that absence of observed imbalance demonstrates absence of confounding, a sensitivity analysis asks how strong an unmeasured common cause would have to be to materially change the result.

One family of methods characterizes departures from treatment assignment probabilities implied by the measured covariates. Rosenbaum's sensitivity framework, for example, can examine how conclusions from matched comparisons change when two people with identical measured characteristics have different odds of treatment because of an unobserved factor.

Other approaches quantify associations an unmeasured confounder would need with treatment and outcome to explain away an estimated effect. The form of the calculation depends on the estimand, effect measure, design and assumed structure. Such analyses do not identify the unobserved variable or prove that hidden bias exists. They state how robust a conclusion is to specified departures from exchangeability.

In this synthetic experiment, the residual matched U difference is approximately 0.95 and its structural coefficient in the outcome is 1.5. That produces a substantial residual bias. If U had a smaller outcome coefficient, the bias would be smaller; if assignment depended on U more strongly, the matched groups could be more separated. A systematic sensitivity study could vary those two dimensions and measure the resulting bias surface.

The benchmark here uses one hidden-confounding strength. It therefore demonstrates the possibility and magnitude of failure under a stated mechanism, not a universal bias formula for real data. A responsible empirical application would report sensitivity to plausible departures from its identifying assumptions rather than relying only on a nearly perfect balance table.

The same logic applies to negative controls and alternative identification designs. Evidence that a measured treatment is associated with an outcome it could not plausibly cause may reveal residual confounding or selection. A second study design with different assumptions may strengthen an argument when the results agree. None of those exercises turns an untestable assumption into a verified fact, but each can limit the set of plausible explanations.

## Why more data need not help

It might appear that matching would recover the effect if the sample were much larger. With enough controls, every treated person could have an extremely close match on X, reducing discrepancies caused by the matching algorithm itself. That would improve one aspect of the analysis.

The hidden U difference, however, does not disappear merely because X is matched more precisely. In the second assignment mechanism, even at fixed X, treatment status carries information about U. That dependence is part of the population assignment process. As the sample grows, matching can converge toward comparisons at identical observed X values while still comparing different conditional distributions of U.

The resulting estimator may therefore become more stable around a biased limit. Its sampling variability decreases without the causal discrepancy disappearing. This is the same basic separation between precision and identification encountered in the first two articles, now within an algorithm specifically designed for observational comparisons.

A more flexible propensity model could reduce misspecification in the estimated relationship between X and assignment. It cannot undo the fact that the propensity score based on observed X is not enough to make treatment independent of the potential outcomes. Better classification can improve estimation of an insufficient score.

This is an important distinction for machine-learning applications of causal inference. Predictive performance of a treatment assignment model, precision of an outcome estimator and credibility of the identifying assumptions are three different properties. Optimizing the first two does not establish the third.

Larger datasets still have substantial value. They can improve overlap, permit finer adjustment, support subgroup analysis and make estimates more precise when an appropriate identifying strategy is available. But collecting more records under the same unmeasured selection mechanism is not a substitute for recording the variables needed for a causal comparison or finding a different source of identification.

## What propensity-score methods can legitimately accomplish

It would be a mistake to conclude from the hidden-confounding example that matching has no useful role. Under conditional exchangeability, consistency and overlap, matching can create an understandable approximation to a counterfactual comparison. It can be particularly helpful when analysts want to inspect the population to which conclusions apply and avoid extrapolating across poorly supported covariate regions.

The simulation's first scenario demonstrates that potential. The naive difference averages 2.1504, while matching recovers approximately 1.0128. That is a large reduction in bias obtained with a simple score and a one-dimensional nearest-neighbour algorithm. Under the assumptions of that scenario, the matched comparison has a reasonable causal interpretation.

Matching can also serve as a design stage before examining outcomes. Decisions about matching variables, calipers and overlap can be made using treatment assignment and baseline covariates rather than being tuned to produce a preferred outcome difference. Keeping this separation helps avoid selecting the design after seeing which specification generates a desirable estimate.

Still, the estimator and uncertainty method require attention. Matching with replacement creates reused controls; matching without replacement creates dependence on ordering and potentially worse matches; estimated propensity scores introduce their own sampling behaviour. Standard bootstrap procedures are not always valid for nearest-neighbour matching, as Abadie and Imbens (2008) demonstrate for important cases.

A more credible analysis may use matching followed by outcome adjustment, inverse probability weighting or an estimator combining outcome and propensity models. These choices should be judged according to their estimands, assumptions and finite-sample behaviour rather than treated as interchangeable corrections for confounding.

The scientific objective remains to estimate a defined intervention contrast under a defensible design. Propensity scores are useful statistical tools in that process. They are not independent evidence that the design assumptions hold.

## What the benchmark does not show

The experiment uses a continuous observed X, one normally distributed hidden variable U, a binary treatment and a constant additive treatment effect. The outcome disturbances are independent and Gaussian, and every outcome is observed. There is no measurement error, interference, attrition or treatment noncompliance. Those choices isolate the hidden-confounding mechanism but are not representative of all observational studies.

The propensity model uses a single observed covariate. This makes score-based matching simple and produces extremely small mean differences in X. In high-dimensional datasets, propensity-score matching may struggle to balance particular prognostic variables or combinations even when all confounders have been measured. Poor observed balance would create an additional difficulty beyond the hidden imbalance studied here.

The caliper and matching rule are fixed. The study does not optimize the caliper, compare multiple matching algorithms or calculate confidence intervals for a single matched dataset. The reported standard deviations arise from repeated independently generated datasets. They should not be confused with valid analytic standard errors for matching with replacement.

The high retention rate is specific to these assignment mechanisms and sample sizes. More extreme propensities could substantially reduce matchability, alter the target population and increase reliance on a few control observations. Treatment-effect heterogeneity would then make the shift in estimand consequential.

The hidden-confounding scenario is constructed so that U affects assignment and outcomes strongly. Real-world hidden bias can be weaker, stronger or operate through nonlinear and interacting pathways. The numerical matched estimate of 2.4294 is not a general prediction about propensity-score matching. It is a result for the published mechanism, parameter values and random seed.

Most importantly, the simulation can inspect U because the author generated it. An observational analyst cannot usually know whether an omitted cause has this role or magnitude. The example demonstrates why a clean balance table is insufficient, not a practical diagnostic that reveals hidden confounders automatically.

## What the investigation established

The two scenarios used the same potential-outcome model and the same true effect of one. The matching algorithm fitted propensity scores from observed X, selected nearest controls using fitted log-odds, imposed a fixed caliper and retained about 99% of treated individuals.

When assignment depended only on X, matching reduced the mean unadjusted estimate from 2.1504 to 1.0128. When assignment also depended on hidden U, matching reduced the naive estimate from 3.2178 to 2.4294 but did not approach the causal effect. In both cases the observed mean difference in X after matching was about six or seven ten-thousandths. In the hidden-confounding scenario the matched difference in U was approximately 0.9514.

The matching procedure did not fail to do its stated computational job. It balanced the variable it was given, found close controls and retained almost all treated observations. The causal analysis failed because its measured adjustment set omitted an important common cause. A successful algorithmic diagnostic and a defensible causal identification claim are not the same result.

The next article studies inverse probability weighting. Rather than pairing people with similar estimated treatment propensities, weighting uses those probabilities to construct a reweighted comparison population. It raises related questions about exchangeability and overlap, with an additional challenge: small estimated assignment probabilities can give a few observations disproportionate influence over the final result.

## Code

The complete numerical experiment is implemented in \`causal_inference_statistics/part3/experiment.py\`. It uses NumPy, SciPy and scikit-learn, fixes the random seed and prints means and between-replicate standard deviations for the unadjusted and matched estimates, retention, observed-covariate difference and simulated hidden-covariate difference. It performs nearest-neighbour matching with replacement on the fitted log-odds and applies a fixed caliper. Results are generated from known potential outcomes, not estimated from an external dataset.

## The series

1. When Regression Gets the Causal Effect Wrong
2. Randomization, Confounding and the Value of Experimental Design
3. Matching and Propensity Scores: When Balanced Data Still Give the Wrong Causal Effect (this article)
4. Inverse Probability Weighting
5. Directed Acyclic Graphs and the Cost of Controlling for Everything
6. Difference-in-Differences and the Parallel Trends Assumption

## References

Abadie, A. and Imbens, G. W. (2006). Large Sample Properties of Matching Estimators for Average Treatment Effects. *Econometrica*, 74(1), 235–267. https://doi.org/10.1111/j.1468-0262.2006.00655.x

Abadie, A. and Imbens, G. W. (2008). On the Failure of the Bootstrap for Matching Estimators. *Econometrica*, 76(6), 1537–1557. https://doi.org/10.3982/ECTA6474

Abadie, A. and Imbens, G. W. (2011). Bias-Corrected Matching Estimators for Average Treatment Effects. *Journal of Business & Economic Statistics*, 29(1), 1–11. https://doi.org/10.1198/jbes.2009.07333

Austin, P. C. (2009). Balance Diagnostics for Comparing the Distribution of Baseline Covariates Between Treatment Groups in Propensity-Score Matched Samples. *Statistics in Medicine*, 28(25), 3083–3107. https://doi.org/10.1002/sim.3697

Austin, P. C. (2011). An Introduction to Propensity Score Methods for Reducing the Effects of Confounding in Observational Studies. *Multivariate Behavioral Research*, 46(3), 399–424. https://doi.org/10.1080/00273171.2011.568786

Caliendo, M. and Kopeinig, S. (2008). Some Practical Guidance for the Implementation of Propensity Score Matching. *Journal of Economic Surveys*, 22(1), 31–72. https://doi.org/10.1111/j.1467-6419.2007.00527.x

Hernán, M. A. and Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC. https://miguelhernan.org/whatifbook

Ho, D. E., Imai, K., King, G. and Stuart, E. A. (2007). Matching as Nonparametric Preprocessing for Reducing Model Dependence in Parametric Causal Inference. *Political Analysis*, 15(3), 199–236. https://doi.org/10.1093/pan/mpl013

Imbens, G. W. (2004). Nonparametric Estimation of Average Treatment Effects Under Exogeneity: A Review. *Review of Economics and Statistics*, 86(1), 4–29. https://doi.org/10.1162/003465304323023651

King, G. and Nielsen, R. (2019). Why Propensity Scores Should Not Be Used for Matching. *Political Analysis*, 27(4), 435–454. https://doi.org/10.1017/pan.2019.11

Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*, 2nd edition. Cambridge University Press.

Rosenbaum, P. R. (2002). *Observational Studies*, 2nd edition. Springer.

Rosenbaum, P. R. and Rubin, D. B. (1983). The Central Role of the Propensity Score in Observational Studies for Causal Effects. *Biometrika*, 70(1), 41–55. https://doi.org/10.1093/biomet/70.1.41

Rosenbaum, P. R. and Rubin, D. B. (1985). Constructing a Control Group Using Multivariate Matched Sampling Methods That Incorporate the Propensity Score. *The American Statistician*, 39(1), 33–38. https://doi.org/10.1080/00031305.1985.10479383

Rubin, D. B. (1973). Matching to Remove Bias in Observational Studies. *Biometrics*, 29(1), 159–183. https://doi.org/10.2307/2529684

Rubin, D. B. (1974). Estimating Causal Effects of Treatments in Randomized and Nonrandomized Studies. *Journal of Educational Psychology*, 66(5), 688–701. https://doi.org/10.1037/h0037350

Stuart, E. A. (2010). Matching Methods for Causal Inference: A Review and a Look Forward. *Statistical Science*, 25(1), 1–21. https://doi.org/10.1214/09-STS313

VanderWeele, T. J. and Ding, P. (2017). Sensitivity Analysis in Observational Research: Introducing the E-Value. *Annals of Internal Medicine*, 167(4), 268–274. https://doi.org/10.7326/M16-2607
