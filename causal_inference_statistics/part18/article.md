# Missing Outcomes and Selection Bias: When Complete Cases Mislead

*Part 18 of Causal Inference in Statistics. Three observation mechanisms, 900 simulated datasets, and four analyses of a randomized treatment with known effect. Why a complete-case regression can be valid under missing at random, why weighting can lose precision, and why neither repairs outcome-dependent selection without further assumptions. Python code provided.*

A dataset may describe the treatment assigned to every participant and yet contain outcomes for only some of them. The missing rows can look like an inconvenience that a statistical package will handle once the analyst chooses whether to delete incomplete records, impute outcomes, or reweight people who remained under observation. The causal difficulty starts earlier. Observing an outcome is itself a selection event, and that selection can alter the relationship between treatment, prognosis and the observed response.

It is tempting to describe complete-case analysis as biased and inverse probability weighting as the solution. That rule is too simple. Complete-case regression may consistently estimate a conditional treatment coefficient when the outcome is missing at random conditional on the variables in a correctly specified regression model. Weighting may then achieve little or introduce extra sampling variance. On the other hand, if the probability of observing an outcome depends on the outcome itself after conditioning on measured information, a response model fitted to the observed covariates generally cannot recover the population contrast. The distinction is about identification, not the elegance of a missing-data algorithm.

I investigate these issues with a randomized binary treatment, one prognostic baseline variable, and a constant treatment effect of one outcome unit. Every participant has a generated outcome, including those hidden from the analyst; the hidden values make it possible to evaluate the estimation error against the causal truth. Three observation mechanisms are considered: missing completely at random, missing at random conditional on treatment and baseline characteristics, and missing not at random because response additionally depends on the outcome.

For each mechanism I generate 300 independent datasets of 2,000 participants. I compare the treatment coefficient from all generated outcomes, the corresponding complete-case regression, a regression weighted by the inverse estimated probability of outcome observation, and an outcome-model estimate fitted to the observed outcomes. The last procedure intentionally uses the same linear conditional model as complete-case regression, so its numerical equality is expected rather than evidence of independent robustness.

The primary finding is nuanced. Under missing completely at random, the complete-case mean effect is 1.0014 and the response-weighted estimate is 1.0014. Under missing at random, those values are 0.9957 and 0.9938. Both are close to the true effect of one, and weighting has larger RMSE in the second case. Under outcome-dependent observation, the complete-case mean falls to 0.9724 and weighting to 0.9685. Neither method identifies the intended population effect under the assumptions it actually uses. The simulation therefore illustrates both a successful conditional complete-case comparison and a failure of observed-data corrections.

## The decision and the observation process

Suppose a public employment service randomly allocates an employment-support programme among eligible jobseekers. A subsequent evaluation measures employment outcomes several months later. Some participants cannot be contacted, decline the follow-up interview, or do not have a complete outcome record. An analyst may still have each participant's assignment and baseline characteristics, but not the outcome for everyone.

The decision is whether providing the programme improves outcomes in the eligible population. If people whose outcomes are missing would have responded differently to treatment from the people observed, an effect calculated only among respondents need not represent that population decision. Randomized treatment allocation protects assignment at baseline; it does not guarantee that the observed-outcome subset remains a representative randomized comparison.

In some settings, nonresponse depends primarily on recorded information. Participants with particular baseline characteristics or assignments may be harder to contact, but among people with the same recorded information, observation does not depend on the eventual outcome. Under suitable conditions the observed data can be used to identify the population mean outcomes. In other settings, unsuccessful participants may be less willing to respond even after baseline characteristics and assignment are held fixed. That is an outcome-dependent observation mechanism and requires additional information or assumptions.

The experiment is generic rather than an actual employment-service evaluation. The outcome is continuous, and the treatment has a fixed additive effect. That choice removes complications from censoring, competing risks, outcome misclassification and treatment-effect heterogeneity. The study isolates the effect of observing or not observing outcomes, while the substantive example explains why it matters to a decision.

A single fraction of missing observations does not establish whether selection creates bias. Two studies can each lose forty percent of outcomes yet have entirely different identification properties. The missingness mechanism, the prognostic information available to the analyst, and the conditional outcome model determine what can be learned. Reporting only the proportion missing gives readers almost none of that structure.

## The estimand and what is observed

Let A be a randomized binary treatment, X a baseline covariate and Y(a) the potential outcome under intervention a. The average treatment effect is

\[
\tau=E\{Y(1)-Y(0)\}.
\]

We observe treatment A and baseline X for every participant. Let R be the indicator that the outcome is recorded: R=1 if Y is observed and R=0 otherwise. The observed outcome information can be represented by (R,RY) together with A and X. For R=0 the product RY is merely a bookkeeping convention; it is not a substantive outcome equal to zero.

Under randomized assignment, A is independent of potential outcomes in the original eligible population. If outcomes were observed for all participants and consistency held, a difference in group means could estimate the target effect. But the analysis that retains R=1 additionally conditions on the observation event. Whether that conditional comparison is causally interpretable depends on how R is generated.

The selected-population difference between treated and untreated respondents can differ from the original ATE, especially when treatment affects response and outcome prognosis also affects response. Conditioning on R=1 may act like conditioning on a collider. However, the precise effect on a regression coefficient depends on the structure and functional form. It is not valid to declare every complete-case estimate biased simply because selection occurred.

To make the distinction clear, I use a model with an exactly constant conditional treatment effect. Under correctly specified linear outcome regression, some selection mechanisms preserve that coefficient even when the distribution of X among respondents differs substantially from the original population. The simulation is constructed to exhibit that important exception.

## The outcome-generating model

Every individual has a standard normal baseline covariate X and an independent standard normal disturbance ε. Treatment is assigned independently with probability one half. The outcome is

\[
Y=A+1.5X+\varepsilon.
\]

The associated potential outcomes are Y(0)=1.5X+ε and Y(1)=1+1.5X+ε. Therefore Y(1)−Y(0)=1 for every participant, and the population ATE is exactly one.

The coefficient 1.5 makes X strongly prognostic. An observed respondent sample may therefore have a very different mean outcome if the observation mechanism favours particular X values. Crucially, the treatment coefficient in the correctly specified conditional regression Y on A and X is still one when selection depends on A and X but not on the residual outcome variation given them.

This setup is intentionally favourable to conditional regression. The effect is constant and additive; no treatment-covariate interaction is needed. All assigned treatments and baseline covariates are recorded, so the problem concerns missing outcomes rather than missing covariates. Independent individuals avoid clustering complications. These assumptions make the interpretation of the numerical results transparent, but limit how far the outcome-regression findings can be generalized.

The analysis also includes an all-outcomes regression calculated using the generated values, including those marked as unobserved. It is an oracle benchmark: no real analyst would have access to those outcomes. Its purpose is to show how much sampling error is present without nonresponse and to distinguish selection-related error from ordinary variation in estimation.

## Three observation mechanisms

The first mechanism is missing completely at random, or MCAR. Every participant's outcome is observed independently with probability 0.65. The response indicator is unrelated to treatment, baseline prognosis, potential outcomes or the realized outcome. Complete cases consequently form a random subset of the original study, losing information but not systematically changing its population relationships.

The second mechanism is missing at random, or MAR, conditional on recorded treatment and baseline X. The observation probability is

\[
P(R=1\mid A,X,Y)=\operatorname{expit}(0.8-X+0.6A),
\]

where expit(z)=1/(1+exp(−z)). Once A and X are held fixed, the outcome Y provides no additional information about whether it is observed. Response still depends on factors related to prognosis and on assigned treatment, so the respondent population is not a simple random sample. Nevertheless, the observation mechanism is explainable using variables recorded for everyone.

The third mechanism is missing not at random, or MNAR, relative to the available information. The probability becomes

\[
P(R=1\mid A,X,Y)=\operatorname{expit}(0.8-X+0.6A-0.9Y).
\]

Now the outcome itself affects the probability of being observed even after A and X are fixed. Because the analyst lacks Y for nonrespondents, the response mechanism cannot in general be estimated from a logistic regression of R on A and X alone. The distribution of observed Y conditional on A and X is selected on the residual outcome variation.

The first two mechanisms satisfy the appropriate form of outcome-observation ignorability conditional on the variables available for modelling. The third violates it. All three use positive observation probabilities for finite covariates, but positivity alone does not resolve the MNAR case. An estimator can be numerically well behaved and still rely on an insufficient missingness model.

These mechanisms are statistical definitions relative to the information and assumptions of the study. The same nonresponse phenomenon might be MAR with respect to an enriched dataset that records its causes and MNAR with respect to a limited dataset that does not. The labels are not permanent properties of people; they describe conditional relationships in a specified information system.

## What complete-case regression estimates

The complete-case estimator fits Y on an intercept, A and X using only observations with R=1. In a general observational dataset, this procedure can be biased for many reasons, including treatment confounding and misspecified outcome models. Here treatment was randomized originally, and the conditional structural outcome mean is exactly

\[
E[Y\mid A,X]=A+1.5X.
\]

Under MCAR, restricting to respondents does not change this conditional relationship. Under MAR, R is conditionally independent of Y given A and X, so

\[
E[Y\mid A,X,R=1]=E[Y\mid A,X].
\]

The correctly specified complete-case regression therefore remains centred on treatment coefficient one. The distribution of A and X among respondents may differ, but conditioning on the same A and X in the correctly specified outcome equation preserves the slope. This is an important qualification to the familiar warning that conditioning on response creates selection bias.

For more general outcomes or heterogeneous treatment effects, a complete-case treatment coefficient can target a respondent-weighted contrast rather than the full-population ATE, even under a form of MAR. Correct specification of the conditional regression is doing substantial work in our experiment. If the treatment response varied with X, simply reporting the coefficient of A from a regression without interactions would not necessarily equal the population average effect.

The benchmark therefore does not recommend complete-case analysis by default. It illustrates conditions under which complete-case regression is valid, and asks what is gained or lost by applying more elaborate missingness corrections when those conditions already hold.

## Inverse probability of observation weighting

When the outcome is missing at random conditional on recorded covariates, the population mean of a suitable complete-data quantity can be identified by reweighting respondents according to the inverse probability of observation. Let

\[
\pi(A,X)=P(R=1\mid A,X).
\]

If R is conditionally independent of Y given A and X, and observation probabilities are positive, then

\[
E\left[\frac{RY}{\pi(A,X)}\right]=E[Y].
\]

The calculation follows by conditioning on A, X and Y: the expected response indicator equals the observation probability, so inverse weighting removes the change in frequency with which different recorded strata appear. Related identities can be used for regression scores and causal estimators, with attention to which variables and target moments are required.

In the experiment, I estimate the response probability by logistic regression of R on A and X. This is correctly specified for MCAR as an intercept-only special case and for the specified MAR mechanism in the sense that the correct logistic family includes the relevant conditional relationship. In the MNAR scenario the same fitted model cannot represent the dependence on unobserved Y among nonrespondents.

I clip fitted response probabilities to the interval [0.01,0.99] to avoid numerical extremes. Then I fit the outcome regression among respondents using inverse estimated response probabilities as weights. This is a weighted least squares estimator for the same conditional regression coefficient, not a generalized estimator of every possible marginal causal effect.

Under the correctly specified linear outcome model, adding inverse response weights need not reduce the sampling variance of the treatment coefficient. It can increase it by giving more influence to respondents with small estimated observation probabilities. A correct identification identity does not imply that its most direct weighted implementation is the most efficient estimator under every correctly specified outcome model.

## Outcome regression and the apparent duplication

The fourth method fits the outcome model among respondents and predicts the treatment contrast from that model. Because the outcome equation includes an additive constant treatment effect and the fitted regression includes A and X, the predicted contrast is exactly the fitted coefficient of A. It therefore has the same numerical result as the complete-case regression in this experiment.

This equality is a useful transparency check. Presenting the two methods as independent successes would exaggerate the evidence. They are two descriptions of the same fitted conditional outcome relationship in this particular linear constant-effect system.

In more general settings, g-computation averages predicted counterfactual outcomes over the target population's baseline covariates. The population averaging step matters when treatment effects vary with X or when the outcome response is nonlinear. Here every individual's conditional contrast is the same number, so averaging changes nothing.

The experiment does not include multiple imputation, augmented inverse probability weighting or targeted minimum loss estimation. Such approaches can be valuable, particularly when researchers want to combine observation and outcome models or account for more complex longitudinal selection. The present comparison is intentionally narrower: it isolates the basic observation mechanism and shows how correct conditional outcome specification affects both bias and precision.

## Monte Carlo protocol

Each dataset contains 2,000 independently generated individuals, with X, random treatment A and a complete underlying outcome Y. The response mechanism then hides a subset of those outcomes. The simulation repeats the process 300 times for MCAR, MAR and MNAR, with a fixed generator seed of 20261018.

Within each replicate, the four estimators are fitted anew. The all-outcomes regression acts as the inaccessible full-data benchmark. Complete-case regression and outcome regression use only R=1 observations. The response-weighted regression additionally fits the observation-probability model using R, A and X recorded for all individuals, then applies inverse probabilities among observed outcomes.

The simulation reports mean estimated treatment effect, bias relative to the true effect one, between-replication standard deviation, root mean squared error, and average fraction with observed outcomes. RMSE combines squared bias and variability; it is a more useful measure of overall estimation accuracy than mean bias alone in a repeated-simulation benchmark.

There is no hyperparameter search and no outcome-based selection of the preferred estimator. Missingness parameters, model formulas and thresholds are fixed before examining the results. This separation is important because a real observational study cannot minimize error against a known causal truth; the Monte Carlo setup can do so only because the investigator generated the potential outcomes.

The program uses ordinary NumPy least squares and scikit-learn logistic regression. The results can be regenerated without downloading an external dataset. The random seed supports replication of the published numbers, but the inferential interpretation rests on the structural model and missingness assumptions, not on that particular seed.

## How much data remain observed?

The observation mechanisms produce different response fractions. MCAR retains on average 64.99% of the 2,000 participants. MAR retains 71.14%. The outcome-dependent MNAR design retains 58.42%. The fractions differ because the logistic mechanisms have different average probabilities under the simulated treatment, covariate and outcome distributions.

The number retained is not a measure of identification strength by itself. The MAR mechanism leaves more observations available than MCAR yet changes their baseline composition. The MNAR mechanism leaves fewer and selects on outcome residuals. A fraction near sixty percent could be adequate for a particular correctly identified model but inadequate for another, depending on overlap, prognostic variables and the scientific question.

In all three scenarios, the underlying full-data estimate uses 2,000 outcomes. The respondent analyses use smaller and differently composed sets. Their increased sampling error relative to the full-data benchmark is expected even when they remain unbiased.

An applied report should distinguish the number originally eligible, the number assigned treatment, the number with measured baseline variables and the number with an outcome at each follow-up. When outcomes are measured repeatedly, the observation process can be time-dependent, and the set of observed people may change from one horizon to another. A single overall retention percentage hides that structure.

## The numerical estimates

The Monte Carlo means are:

| Mechanism | All outcomes | Complete cases | Response weighting | Outcome regression |
| --- | ---: | ---: | ---: | ---: |
| MCAR | 0.9996 | 1.0014 | 1.0014 | 1.0014 |
| MAR | 0.9947 | 0.9957 | 0.9938 | 0.9957 |
| MNAR | 1.0002 | 0.9724 | 0.9685 | 0.9724 |

The true effect is one throughout. Under MCAR the observed-outcome methods are centred close to one, as expected from a random subset of the full sample. The all-outcomes estimate is slightly closer to the structural effect in the simulation mean, but the difference is not substantively meaningful.

Under MAR, all three respondent-based methods remain near one. This may seem surprising to anyone expecting complete-case analysis to fail whenever response depends on treatment or prognosis. Here the conditional outcome model is correctly specified, so the treatment slope remains intact in the selected population. The observed-data regression does not estimate the outcome distribution's marginal mean directly; it estimates a constant conditional treatment coefficient whose value is unchanged by this form of selection.

Under MNAR, the complete-case and outcome-regression means fall to approximately 0.972, and response weighting falls to approximately 0.969. These departures are much smaller than some of the dramatic biases in earlier articles, but they are systematic under the chosen model. Their interpretation should reflect their actual scale: a few hundredths of an outcome unit, not a reversal of the effect.

The simulation therefore supports a more precise conclusion than 'missing data cause bias'. Outcome-dependent selection makes both observed-data approaches biased here, while MCAR and the specified MAR process preserve the conditional coefficient under the model. The magnitude of bias depends on the selection strength and data-generating parameters. Different mechanisms could produce far greater distortions or even changes of sign.

## Error and precision

The corresponding RMSE values are:

| Mechanism | All outcomes | Complete cases | Response weighting |
| --- | ---: | ---: | ---: |
| MCAR | 0.0440 | 0.0539 | 0.0539 |
| MAR | 0.0426 | 0.0489 | 0.0524 |
| MNAR | 0.0479 | 0.0622 | 0.0871 |

With all outcomes recorded, RMSE is between about 0.043 and 0.048 across the independently generated scenarios. Complete-case regression has larger RMSE because it uses fewer outcomes. Under MCAR, weighting makes virtually no difference, since observation probabilities are close to constant and weights carry little useful additional information.

Under MAR, the unweighted conditional regression outperforms response weighting on RMSE: 0.0489 compared with 0.0524. This is not a failure of the weighting identification argument. The outcome regression is correctly specified, and weighting needlessly increases the influence of some respondents. The comparison illustrates that consistency and finite-sample efficiency are distinct questions.

Under MNAR, response weighting performs worst in this benchmark, with RMSE 0.0871. The response model cannot explain outcome-driven missingness conditional on recorded A and X. Its weights therefore fail to remove the selection effect and add variability. A more complicated propensity classifier using the same observed variables would not necessarily solve the identification failure, because the missing outcomes needed to model response conditional on Y are unavailable.

The all-outcomes benchmark also reminds us that no method can recover information without a cost. Even when identification is valid, analysing fewer recorded outcomes usually reduces precision. Extra assumptions can sometimes improve efficiency, but the uncertainty calculation must reflect them rather than treating imputed or weighted rows as newly observed independent evidence.

## What MAR does and does not say

Missing at random is frequently misunderstood as saying that missingness is unrelated to the outcome or randomly distributed across the sample. The condition is subtler. It says that after conditioning on the recorded variables used to explain the observation process, the missingness indicator contains no further information about the unobserved outcome.

In this simulation, treatment and X are available for every participant. Under MAR the observation probability varies with both. The respondent sample is therefore selected, but its conditional outcome distribution given A and X remains the same as the original conditional distribution. That permits correct conditional regression and inverse probability methods to identify the effect under suitable model and support conditions.

MAR is not established by a low p-value in a missingness regression, nor by failing to detect a relationship between response and observed outcomes. Outcomes are not observed for the missing cases, so the essential independence condition is generally not testable from observed data alone. It requires knowledge of the measurement process or assumptions defended by other evidence.

The definition also depends on what was actually recorded. If an important determinant of response and prognosis is omitted, the data may not satisfy MAR with respect to the available covariates. Collecting baseline information related to both response and outcomes can therefore matter more than choosing a more elaborate imputation algorithm after the study has ended.

The central distinction is between conditional independence as an identifying assumption and predictive ability as a statistical property. A logistic response model can predict R reasonably well while leaving important unmeasured determinants of the missing outcomes unaccounted for. Its classification accuracy does not certify recoverability of the full-data causal estimand.

## MNAR and the identification gap

The third scenario makes observation depend directly on realized Y, conditional on A and X. The response model fitted to R using A and X averages over outcomes that are themselves selectively missing. It estimates an observed conditional response relationship, but does not recover the underlying mechanism involving the unobserved outcomes.

For respondents, the residual ε is no longer distributed as it is in the full eligible population given A and X. High outcomes are less likely to be observed under the selected parameterization. Because A also changes Y, this selection affects the fitted treatment comparison in ways that a correctly specified baseline outcome regression would not encounter under MAR.

A naïve response model based on available predictors may assign apparently sensible weights while failing to reproduce the outcome distribution of those not observed. The problem is not merely that the response probabilities were estimated imprecisely. Identification requires additional assumptions about how missing outcomes influence response, an external sample measuring outcomes for some nonrespondents, a suitable instrument for missingness, or some other information beyond the observed outcome records.

Sensitivity analyses can state how conclusions change under assumed strengths of outcome-dependent selection. Pattern-mixture and selection models can make those assumptions explicit, but they cannot turn the observed-data likelihood into unique knowledge about the missing values without restrictions. Different unobserved-outcome distributions can be compatible with the same measured data.

The simulation specifies the MNAR coefficient −0.9 and the full outcomes, so it knows the truth. An analyst of real data does not possess that knowledge. It would be misleading to recommend correcting a dataset by inserting the simulated coefficient as though it were empirically determined. Its value is to provide a controlled departure from MAR and assess the resulting estimator errors.

## Selection can change the target population

A complete-case analysis automatically uses the subset with measured outcomes. If treatment effects vary, that subset can have a different average effect from the original eligible population. Even if an estimator accurately learns the conditional effect among respondents, the effect averaged over the respondents' covariates may not answer the policy's target-population question.

Our simulation avoids this complication by imposing a constant effect of one. The treatment coefficient therefore represents the same structural difference across X. That is why the conditional complete-case regression works under MAR without explicitly reconstructing the original X distribution.

In a heterogeneous-effect study, an analyst may need to estimate conditional potential-outcome means and average predictions across the original baseline covariate distribution, including people whose outcomes were not observed. If the covariate distribution of respondents differs, averaging only over respondents can change the estimand.

The distinction matters for programme targeting. People who are hardest to follow up may also be those who benefit most from an intervention. An effect estimated only among compliant respondents could be accurate for that subgroup and poor guidance for the population a service actually intends to reach. Outcome completeness becomes a design and external-validity question, not just a data-cleaning concern.

It also matters when eligibility or follow-up depends on consequences of treatment. If the intervention itself changes the likelihood of being observed, a selected-sample analysis can be conditioned on a post-treatment event. The causal implications then depend on the full selection graph, including causes of observation and outcome. A simple label of MAR or MNAR should not replace that graphical reasoning.

## Missing data and the role of causal diagrams

The causal structure can be represented with X affecting outcome, A affecting outcome, and a response indicator R determining which Y values enter the observed dataset. Under MCAR, R has no relevant parents. Under MAR, A and X affect R, but the remaining outcome disturbance does not. Under MNAR, Y itself affects R.

These graphical differences explain why the same statistical operation has different consequences. Restricting to R=1 is harmless for the specified conditional treatment coefficient under MCAR and MAR, given the correct outcome equation. Under MNAR, conditioning on response selects partly on the outcome residual and changes the conditional response distribution.

A DAG also helps identify other pathways that this simplified model excludes. An unmeasured health condition could affect both dropout and outcomes. Treatment might alter engagement, which then affects observation. Eligibility requirements might select people jointly on treatment propensity and prognosis. These structures can invalidate comparisons even when the fitted response probability seems stable.

Causal diagrams do not verify missingness assumptions. They organize the assumptions and expose which variables need to be observed, when they must be measured, and which selection events the analysis conditions on. The aim is not to choose a graph from the outcome p-values, but to describe the measurement process before deciding how to estimate the intervention contrast.

If longitudinal treatments and repeated outcomes are involved, the observation process may feed back into future care or behaviour. Missingness then becomes time-varying and can interact with treatment-confounder feedback. Marginal structural weighting and sequential G-computation can sometimes handle suitable observed-history-dependent processes, but require the same careful attention to positivity and unmeasured selection.

## Why imputation is not independent evidence

Imputation can make an incomplete table look complete, but filled values are predictions under an assumed model. They are not newly measured outcomes. Valid multiple-imputation procedures incorporate uncertainty about the missing values and model parameters, and their justification depends on assumptions about the missingness process.

For the linear constant-effect MAR setting, an appropriate imputation model could recover the treatment effect. It may offer advantages when additional variables are partially missing or when a more complex analysis requires a completed data structure. Yet its performance should be judged against estimators that operate directly on observed data and against the relevant causal assumptions.

If missingness depends on the unobserved outcome, imputing from a model learned only among respondents can propagate selection bias. A sophisticated predictive algorithm may fit observed outcomes extremely well while still misrepresenting the unobserved conditional distribution. Better validation error among respondents does not establish correct predictions for those whose responses are missing for outcome-related reasons.

The correct evaluation therefore distinguishes imputation-model fit, inferential uncertainty, and identification of the desired population causal effect. A dataset with plausible-looking imputed values can support an unjustified causal conclusion if the observation mechanism has been overlooked.

This experiment deliberately omits imputation to keep the comparison focused. A later methodological extension could add multiple imputation and doubly robust missingness estimators, reporting both effect error and interval coverage across the three mechanisms. That would be a separate empirical claim requiring actual computation rather than an assumed ranking of methods.

## Testing and sensitivity analysis

MCAR imposes strong independence conditions that may have testable implications in certain observed variables. Researchers can compare treatment and baseline characteristics between respondents and nonrespondents, inspect patterns across follow-up periods and evaluate whether observation rates differ systematically. These checks can reject some simple missingness explanations.

But passing such checks does not establish MAR. The outcome is missing precisely where its relationship with response is at issue. A negative-control outcome or auxiliary measurement collected for some nonrespondents may help constrain explanations, but its usefulness depends on whether it shares the relevant observation mechanisms.

The MNAR example gives a natural sensitivity dimension: the coefficient of Y in the response logit. In the simulation it is −0.9. An applied sensitivity analysis could vary a comparable sensitivity parameter over values justified by external knowledge, compute the implied treatment contrasts, and report how large a selection effect would be needed to overturn a substantive decision.

Such a calculation should not be advertised as a test of the unknown parameter. It is an analysis of conclusions conditional on assumptions. A tipping-point threshold can be useful even if the threshold itself cannot be assigned a data-derived probability without extra modelling or prior information.

Another useful strategy is to invest in measuring outcomes for a randomly selected subset of initial nonrespondents. That can provide direct information about the response process that ordinary complete-case data cannot supply. Study design can therefore be more important than selecting among weighting and imputation packages after follow-up has finished.

## A more useful evaluation plan

The first step in a study with missing outcomes is to define the original target population and estimand. An effect among people with observed follow-up is not automatically the policy effect among all eligible individuals. The analysis must state whether the goal is a full-population average, a respondent-subpopulation effect or another well-defined contrast.

Next, document which variables are available for everyone and which are missing, and describe the timeline of treatment, outcomes and observation. Variables recorded only after treatment may themselves be affected by treatment. Using them mechanically in a response model can introduce additional causal complications depending on their role.

Compare outcome-observation rates and baseline distributions. Model the response process only with an explicit account of what the predictors can justify, and check for practical positivity. Very low response probabilities imply large inverse weights and an effective information loss even if MAR is credible.

Then compare estimators with distinct modelling assumptions. An outcome regression, response weighting and a properly constructed doubly robust estimator may provide informative sensitivity to nuisance-model choice. Agreement is encouraging but does not establish that the missingness mechanism is correctly described. Disagreement is a prompt to investigate model fit, selection and support rather than choose the estimate with the most convenient result.

Finally, report what remains unidentified. If outcome-dependent nonresponse is plausible, describe sensitivity analyses, follow-up validation samples, or assumptions restricting the missing-outcome distribution. A single confidence interval cannot represent uncertainty about a missingness mechanism that the available observations do not identify.

## Limits

This experiment uses a randomized binary treatment, a normally distributed baseline covariate, an additive constant treatment effect and Gaussian outcome errors. The outcome regression is linear and correctly specified when the observation mechanism is ignorable conditional on A and X. This is an important favourable condition for complete-case regression and explains its good performance under MAR.

The MNAR bias is relatively small under the chosen parameters. It should not be interpreted as an upper bound on the consequences of nonignorable selection. Stronger outcome-dependent response, treatment-dependent selection, nonlinearity or effect heterogeneity could produce larger or differently signed effects. A near-correct coefficient under one MNAR simulation also does not imply identification.

The response-weighted estimator fits a logistic model using A and X, clips estimated probabilities, and fits weighted linear outcome regression. It is not the same as a general Horvitz–Thompson mean estimator or an efficient doubly robust influence-function estimator. Its finite-sample comparison with ordinary regression should not be generalized to all weighting procedures.

Outcome regression is numerically identical to the complete-case coefficient because the causal contrast is constant and the same conditional linear model is fitted. Counting them as independent evidence for the conclusion would be inappropriate. The experiment also does not examine confidence-interval coverage, variable-specific missingness, missing treatment assignment, censoring or clustered outcomes.

Three hundred replications provide useful Monte Carlo summaries but not proof of asymptotic behaviour in every setting. The generated complete outcomes make error measurement possible; real analysts cannot observe the corresponding truth. The results are therefore evidence about these estimators under the published structural mechanisms, not evidence about any actual employment programme or data collection system.

## What the experiment established

Under completely random observation, the respondent analyses remain close to the true causal effect of one while losing precision compared with using all outcomes. Under missingness conditional on measured treatment and baseline X, correctly specified complete-case regression also remains close to one; response weighting does not improve RMSE in this particular setting. The result is consistent with the conditional regression preserving its treatment coefficient under MAR.

Under outcome-dependent missingness, the respondent-based models become biased for the original effect. Complete-case regression averages 0.9724 and response weighting 0.9685, compared with a true effect of one. Weighting using only A and X cannot correct a mechanism that additionally depends on the unobserved outcome. Its greater RMSE illustrates the separate cost of estimator variability.

The broader result is not that complete-case regression is safe or that weighting is unreliable. It is that neither method can be evaluated without naming the target and the observation assumptions. A method may be statistically consistent under one selection mechanism, inefficient under another, and biased under a third. The missingness mechanism is part of the causal design rather than an administrative detail to address after fitting the final model.

The next article examines a related but different problem: external validity and transportability. Even when an effect is identified within the observed study population, the people who enter the study may not represent the population to which a decision will be applied. Missing outcomes and missing population coverage are not identical problems, but both remind us that valid estimation of one contrast does not guarantee validity for another population.

## Code

Every numerical result is generated by `causal_inference_statistics/part18/experiment.py` in the Medium-Blog repository. The script uses Python 3.12 or later, NumPy, SciPy and scikit-learn. It fixes the random seed, generates three observation mechanisms, fits the four specified analyses and writes `results.csv` with Monte Carlo summaries. No external dataset is required. The underlying full outcome is retained inside the simulation only to evaluate estimator error.

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
10. Doubly Robust Estimation
11. The G-Formula
12. Heterogeneous Treatment Effects
13. Causal Machine Learning and Double Machine Learning
14. Bayesian Causal Inference
15. Time-Varying Treatments and Marginal Structural Models
16. Sensitivity Analysis for Unmeasured Confounding
17. Negative Controls, Placebo Tests and Falsification
18. Missing Outcomes and Selection Bias (this article)
19. External Validity and Transportability
20. From Statistical Evidence to Causal Decisions

## References

Bang, H. and Robins, J. M. (2005). Doubly Robust Estimation in Missing Data and Causal Inference Models. *Biometrics*, 61(4), 962–973. https://doi.org/10.1111/j.1541-0420.2005.00377.x

Carpenter, J. R. and Kenward, M. G. (2013). *Multiple Imputation and Its Application*. Wiley.

Daniel, R. M., Kenward, M. G., Cousens, S. N. and De Stavola, B. L. (2012). Using Causal Diagrams to Guide Analysis in Missing Data Problems. *Statistical Methods in Medical Research*, 21(3), 243–256. https://doi.org/10.1177/0962280210394469

Hernán, M. A. and Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC. https://miguelhernan.org/whatifbook

Hernán, M. A., Hernández-Díaz, S. and Robins, J. M. (2004). A Structural Approach to Selection Bias. *Epidemiology*, 15(5), 615–625. https://doi.org/10.1097/01.ede.0000135174.63482.43

Little, R. J. A. (1988). A Test of Missing Completely at Random for Multivariate Data with Missing Values. *Journal of the American Statistical Association*, 83(404), 1198–1202. https://doi.org/10.1080/01621459.1988.10478722

Little, R. J. A. and Rubin, D. B. (2019). *Statistical Analysis with Missing Data*, 3rd edition. Wiley.

Manski, C. F. (1989). Anatomy of the Selection Problem. *Journal of Human Resources*, 24(3), 343–360. https://doi.org/10.2307/145818

Molenberghs, G., Beunckens, C., Sotto, C. and Kenward, M. G. (2008). Every Missingness Not at Random Model Has a Missingness at Random Counterpart with Equal Fit. *Journal of the Royal Statistical Society: Series B*, 70(2), 371–388. https://doi.org/10.1111/j.1467-9868.2007.00640.x

Robins, J. M., Rotnitzky, A. and Zhao, L. P. (1994). Estimation of Regression Coefficients When Some Regressors Are Not Always Observed. *Journal of the American Statistical Association*, 89(427), 846–866. https://doi.org/10.1080/01621459.1994.10476818

Rubin, D. B. (1976). Inference and Missing Data. *Biometrika*, 63(3), 581–592. https://doi.org/10.1093/biomet/63.3.581

Schafer, J. L. (1997). *Analysis of Incomplete Multivariate Data*. Chapman & Hall.

Seaman, S. R., White, I. R., Copas, A. J. and Li, L. (2012). Combining Multiple Imputation and Inverse-Probability Weighting. *Biometrics*, 68(1), 129–137. https://doi.org/10.1111/j.1541-0420.2011.01666.x

Sterne, J. A. C., White, I. R., Carlin, J. B., Spratt, M., Royston, P., Kenward, M. G., Wood, A. M. and Carpenter, J. R. (2009). Multiple Imputation for Missing Data in Epidemiological and Clinical Research: Potential and Pitfalls. *BMJ*, 338, b2393. https://doi.org/10.1136/bmj.b2393

White, I. R., Horton, N. J., Carpenter, J. and Pocock, S. J. (2011). Strategy for Intention to Treat Analysis in Randomised Trials with Missing Outcome Data. *BMJ*, 342, d40. https://doi.org/10.1136/bmj.d40