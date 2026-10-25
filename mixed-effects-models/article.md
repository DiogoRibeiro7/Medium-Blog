# Mixed-Effects Models for Grouped Data: Partial Pooling, Within–Between Confounding and Honest Standard Errors

## A guide built on three public data sets, a coverage simulation and a from-scratch implementation

Most regression tutorials quietly assume that every row of a data set is a separate draw from one population. Real data rarely arrive that way. Houses sit inside counties, students inside schools, patients come back for a fourth and fifth visit, stores belong to regions, a customer generates sixty sessions before anyone notices, and a sensor writes a reading every minute. Rows that share a group share something nobody measured: the soil under the county, the head teacher, the patient's physiology, the local market. That shared component breaks three things at once, and it is worth naming them before any model appears.

The first casualty is the estimate for a small group. A county measured in two houses has an average that is mostly noise, and any method that treats that average at face value will rank it among the best or worst counties in the state for reasons that have nothing to do with the county. The second casualty is uncertainty. A standard error computed as if 7,185 students were 7,185 independent observations is far too small for anything that varies at the school level, because there are only 160 schools. The third casualty is subtler and is the one most often missed: a slope estimated from grouped data mixes two different relationships, the one inside groups and the one between groups, and the mixture answers neither question.

Mixed-effects models (also called multilevel, hierarchical or random-effects models) are the standard statistical response to all three problems. This guide covers what they estimate, how they are fitted, where they beat the tools a data scientist usually reaches for (one-hot dummies, target encoding, gradient boosting with a group column, cluster-robust standard errors, generalised estimating equations), and where they fail without warning. Every claim that can be checked against data is checked against data. The three data sets are public: Minnesota indoor radon (919 homes in 85 counties), High School and Beyond (7,185 students in 160 schools) and the 1988 Bangladesh Fertility Survey (1,934 women in 60 districts). On top of them sits a simulation of 10,000 data sets calibrated on the schools. All linear mixed models in the article are fitted with a short implementation of profiled REML written for it and cross-checked against statsmodels; the logistic model uses adaptive Gauss–Hermite quadrature, also written from scratch. The code is linked at the end.

## The model in one line

The general linear mixed model, in the form Laird and Ware (1982) gave it for longitudinal data, writes the outcome of observation i in group j as

```
y_ij = x_ij'β + z_ij'b_j + ε_ij
b_j ~ N(0, D),   ε_ij ~ N(0, σ²),   independent of each other
```

The vector β holds the fixed effects, parameters shared by every group: the effect of measuring radon on the first floor instead of the basement, the effect of a student's socio-economic status. The vector b_j holds the random effects, the deviations of group j from the shared parameters. The simplest case has a single random intercept, so z_ij = 1 and b_j is a scalar shift for county j. A random slope adds a second component, so that the effect of a covariate is allowed to differ from school to school.

The vocabulary is a well-known source of confusion. "Random" does not mean the counties were sampled at random, and "fixed" does not mean the parameter is a constant of nature. Gelman (2005) counted five incompatible definitions of fixed and random effects in the statistical literature. The operational meaning is the only one that matters for computation: a fixed effect is estimated without any distributional assumption about it, while random effects are assumed to come from a common distribution whose variance is itself estimated. That variance is what makes the model learn how much the groups differ, and therefore how much to trust each group's own data.

Integrating the random effects out gives the marginal model, which is what the likelihood is built from:

```
y_j ~ N(X_j β, V_j),   V_j = Z_j D Z_j' + σ² I
```

For a random intercept with variance τ², every pair of observations in the same group has covariance τ², every observation has variance τ² + σ², and the correlation between two observations in the same group is the intraclass correlation

```
ρ = τ² / (τ² + σ²)
```

The intraclass correlation is the first number to compute on any grouped data set, before a single covariate enters the model, because it says how much of the variation lives at the group level. In High School and Beyond the null model gives τ = 2.93 and σ = 6.26 maths points, so ρ = 0.180: eighteen per cent of the variance in a student's maths score is between schools. That number does not sound alarming until it is combined with the average school size through Kish's (1965) design effect,

```
deff = 1 + (m − 1) ρ
```

With an average of 44.9 students per school, the design effect is 8.9. For anything that varies only between schools, such as whether a school is Catholic or public, the 7,185 students carry about as much information as 806 independent students would. A method that pretends otherwise reports standard errors roughly three times too small, since the square root of 8.9 is about 3. That figure reappears below when the actual standard errors are compared.

## Three ways to estimate a group's level

Suppose the goal is the typical radon level in each Minnesota county. Radon is a radioactive gas that seeps from the soil into houses; the data set, collected by the US Environmental Protection Agency and analysed at length by Price, Nero and Gelman (1996) and Gelman and Hill (2007), records the log concentration in 919 homes, whether the measurement was taken in the basement or on the first floor, and the county's soil uranium level. County sizes are very uneven. The median county has 5 measured houses, the largest has 116, three counties have exactly one, and 46 of the 85 counties have five or fewer.

There are three options. Complete pooling ignores counties and fits one regression for the whole state; every county gets the same prediction. No pooling fits a separate intercept for each county, which is what one-hot encoding of the county column does in any linear model; every county's estimate is its own average, however few houses it rests on. Partial pooling, the mixed model, compromises between the two, and the compromise is not a tuning knob but a quantity estimated from the data. In the simplest random-intercept model, the estimated level of county j is

```
α̂_j = λ_j · ȳ_j + (1 − λ_j) · μ̂
λ_j = n_j τ² / (n_j τ² + σ²)
```

where ȳ_j is the county's own mean (adjusted for the floor of measurement, once covariates enter), μ̂ is the overall level, and λ_j is a weight that goes to 1 as the county's sample grows and to 0 as it shrinks. The ratio σ²/τ² has a clean interpretation: it is the number of observations at which a group's own mean and the overall mean receive equal weight. For radon, with the floor of measurement in the model, τ = 0.328 and σ = 0.756 on the log scale, so σ²/τ² = 5.3. A county needs about five measured houses before its own data count for as much as everything the model knows about Minnesota. The median county sits almost exactly at that point, with λ = 0.49. A county measured once gets λ = 0.16. The county with 116 houses gets λ = 0.96.

This estimator has a long history under several names. Animal breeders call it the best linear unbiased predictor, or BLUP, after Henderson's work on sire evaluation (Henderson, 1975); Robinson (1991) wrote the standard review under the memorable title "That BLUP is a good thing." Statisticians recognise it as a version of the James–Stein estimator, which Efron and Morris (1975) made famous by showing that shrinking early-season batting averages toward the league mean predicted the rest of the season better than the raw averages did. Because τ² and σ² are estimated from the same data, the procedure is empirical Bayes: the "prior" for each county is learned from the other 84.

![Figure 1](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/mixed-effects-models/figures/fig1_radon_shrinkage.png)

*Figure 1. County intercepts for basement log radon under no pooling (orange, one dummy per county) and partial pooling (blue, random-intercept BLUP), against the number of houses measured in the county. Grey segments connect the two estimates for the same county. The dashed line is the estimated grand mean. Data: Minnesota radon survey via the HLMdiag R package.*

Figure 1 shows the shrinkage directly. On the left, counties with one to four houses have no-pooling estimates scattered from about 0.6 to 3.0; partial pooling pulls most of them a long way toward the grand mean. On the right, counties with fifty or a hundred houses barely move. The figure also exposes a fact that trips up many analyses: neither set of estimates has the right spread. The standard deviation of the no-pooling intercepts is 0.480, which overstates the true between-county variation because sampling noise adds variance. The standard deviation of the BLUPs is 0.236, which understates it, because shrinkage removes variance. The model's own estimate of the between-county standard deviation, τ = 0.328, lies between them. Anyone who estimates group-level variation by taking the variance of fitted group effects, from either model, gets a wrong number.

Shrinkage matters most when the estimates are used to rank. Among the ten counties with the highest no-pooling intercepts, four have three or fewer measured houses; the median size of those ten is 4. Under partial pooling, two of the top ten have three or fewer houses and the median size is 6. This is the pattern Wainer (2007) called the most dangerous equation, after de Moivre's result that the standard error of a mean falls with the square root of the sample size: the extremes of any league table of small units are dominated by the smallest units, at both ends. Gelman and Price (1999) made the related point about maps of disease rates and went a step further: maps of raw estimates and maps of shrunken estimates are both misleading, in opposite directions, which is the spread problem described above.

## How the model is fitted

There are two standard estimation criteria. Maximum likelihood (ML) maximises the marginal likelihood of y over β, D and σ². Restricted maximum likelihood (REML), introduced by Patterson and Thompson (1971), maximises the likelihood of a set of error contrasts that do not depend on β, which corrects the downward bias of ML variance estimates in the same way that dividing by n − p instead of n corrects the residual variance in ordinary regression. REML is the default in most software and is what every linear mixed model in this article uses. One caveat follows: likelihood-ratio tests between models with different fixed effects must use ML, because REML likelihoods of models with different X are not comparable.

The computational trick that makes modern software fast is profiling. Bates, Mächler, Bolker and Walker (2015) describe the formulation used by the R package lme4: write the random-effects covariance as D = σ²ΛΛ', where Λ is a small lower-triangular matrix. Given Λ, both β and σ² have closed-form solutions (β is a generalised least squares estimate), so the optimiser only has to search over the entries of Λ: one number for a random intercept, three for a random intercept and slope. With one grouping factor the whole likelihood can be assembled from per-group cross-products, so each evaluation costs the same whether groups have 5 or 5,000 members.

The implementation written for this article follows that recipe. The heart of it is a function of the relative covariance parameters θ that returns the profiled REML criterion:

```python
def negll(theta):
    L = unpack(theta)                                   # q x q lower triangular
    M = np.eye(q) + np.einsum("ai,jab,bk->jik", L, ZtZ, L)
    K = np.einsum("ia,jab,kb->jik", L, np.linalg.inv(M), L)
    A = (XtX - np.einsum("jpa,jab,jbr->jpr", XtZ, K, ZtX)).sum(0)  # X'V^-1 X · σ²
    b = (Xty - np.einsum("jpa,jab,jb->jp",  XtZ, K, Zty)).sum(0)
    c = (yty - np.einsum("ja,jab,jb->j",    Zty, K, Zty)).sum()
    beta = np.linalg.solve(A, b)
    s2 = (c - b @ beta) / (N - p)                       # profiled σ²
    return 0.5 * ((N - p) * np.log(s2)
                  + np.linalg.slogdet(M)[1].sum()
                  + np.linalg.slogdet(A)[1])
```

The matrices M_j come from the Woodbury identity, which turns the inverse of each group's covariance matrix into the inverse of a q × q matrix, where q is the number of random effects per group. After the optimum is found, the random effects are predicted as b̂_j = ΛM_j⁻¹Λ'Z_j'(y_j − X_jβ̂), which reduces to the λ_j formula above for a random intercept. On the radon and school models the implementation reproduces the fixed effects and variance components of statsmodels' `MixedLM` to the third or fourth decimal, and fits the High School and Beyond random-slope model in under a tenth of a second.

The cross-check also surfaced a practical point. When county uranium is added to the radon model, statsmodels raised a series of convergence warnings, cycled through three optimisers, and stopped at a between-county standard deviation of 0.161; the profiled REML optimum is 0.156. The fixed effects barely differ, but convergence warnings in mixed-model software deserve to be read rather than suppressed, and when a fit is in doubt it is worth refitting with a different optimiser or a different package.

The same algebra gives a connection that is useful for anyone coming from machine learning. Henderson's mixed-model equations show that, for known variance components, the random-intercept BLUPs are exactly the solution of a penalised least-squares problem: fit one coefficient per group with an L2 penalty of σ²/τ² on the group coefficients and no penalty on the fixed effects. A random-intercept model is ridge regression on the one-hot group dummies. The difference from the way ridge is usually used is that the penalty is estimated by REML in a single fit rather than chosen by cross-validation, and that it comes with an interpretation (the ratio of two variances) and a standard error.

## Does partial pooling actually predict better?

Theory says partial pooling should win on prediction for small groups and tie on large ones. The radon data allow a direct test. In each of 200 random splits, about 30 per cent of the houses in every county with at least two houses were held out, always leaving at least one house of each county in training. That produces, per split, 8 test houses whose county had a single training house, 42 whose county had two or three, 75 with four to nine and 157 with ten or more. Seven models were fitted on each training set and scored on the held-out houses. Complete pooling and no pooling are ordinary least squares, without and with one dummy per county. Partial pooling is the random-intercept model, with and without county uranium as a county-level predictor. The other three are the options a data scientist is likely to try: scikit-learn's `TargetEncoder` on the county column feeding a linear model, once with its default smoothing and once with a smoothing constant set from the mixed model (explained in the next section), and `HistGradientBoostingRegressor` with county, floor and uranium as features and county declared categorical, so that the trees can split on it natively.

![Figure 2](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/mixed-effects-models/figures/fig2_radon_holdout.png)

*Figure 2. Hold-out RMSE of log radon relative to the random-intercept model (dashed line at zero), by the number of training houses in the test house's county. Pooled squared errors over 200 random splits. Lighter dots are thinner counties.*

Over all held-out houses, the root mean squared errors on the log scale were 0.822 for complete pooling, 0.812 for no pooling, 0.793 for the default target encoder, 0.786 for the target encoder with mixed-model smoothing, 0.793 for gradient boosting, 0.781 for partial pooling and 0.771 for partial pooling with uranium. The differences look small in aggregate, but they are not noise. Partial pooling beat no pooling in all 200 splits, with an average RMSE gain of 0.031, and beat complete pooling in all 200, by 0.041. Partial pooling with uranium beat gradient boosting in 93 per cent of splits.

The aggregate hides where the action is, which is what Figure 2 shows. For a house whose county had a single training observation, the one-hot model's error is about 20 per cent larger than partial pooling's (0.928 against 0.775); with two or three training houses, about 13 per cent larger (0.882 against 0.783); with ten or more, the difference is a tenth of a per cent (0.728 against 0.727). Complete pooling shows the opposite pattern: competitive where counties are thin, clearly worse where they are well measured (0.783 against 0.727), because it throws away real between-county differences. Partial pooling is never worse than either extreme, which is exactly the behaviour the λ formula predicts. Gradient boosting with a native categorical split behaves like a softened version of no pooling: close to the mixed model on large counties (0.732), 7 per cent worse on one-house counties (0.829), because a tree has no principled way to decide how much a two-house category deserves its own leaf.

County uranium improves partial pooling most where data are thin, from 0.775 to 0.750 for one-house counties, and hardly at all for counties with ten or more houses (0.727 to 0.726). The mechanism is visible in the variance components. Uranium has a coefficient of 0.72 (standard error 0.09) and lowers the between-county standard deviation from 0.328 to 0.156: it explains 77 per cent of the between-county variance. Each county is now shrunk not toward the state mean but toward the level its soil predicts, and for a county with one house that target is far better informed. Group-level predictors are the cheapest improvement available to any multilevel model, and the place where data scientists most often leave value on the table, because the usual feature pipeline has nowhere to put a variable that lives at the level of the group.

A different question is what happens for a county that never appeared in training, the cold-start problem of any recommender or pricing model. Repeating the exercise with five-fold cross-validation by county (40 repetitions) gives a clear answer. The random intercept contributes nothing to the point prediction of a new group: the mixed model without uranium scores 0.837 and complete pooling 0.832. With uranium, the mixed model and an ordinary regression on floor and uranium both score 0.777, and gradient boosting scores 0.824. For a new group the random effect's value lies elsewhere. Its predictive distribution has variance τ² + σ² rather than σ², so the prediction interval for a house in an unknown county is honestly wider than for a house in a known one. If those intervals feed a decision, that is the part that matters.

## Target encoding is partial pooling with a different prior

High-cardinality categorical features are where most data scientists meet partial pooling without the name. Target encoding replaces each category by a smoothed mean of the target, and the canonical reference, Micci-Barreca (2001), derives the smoothing as an empirical Bayes estimate, a weighted average of the category mean and the global mean with a weight that grows with the category's sample size. That is the λ_j formula above. The question is which variances go into the weight.

scikit-learn's `TargetEncoder` with `smooth="auto"` implements Micci-Barreca's estimator, and its documentation describes the automatic smoothing as an empirical Bayes estimate. Reading the Cython source and reproducing its output to within 10⁻¹⁵ gives the weight it actually uses:

```
λ_j = n_j · Var(y) / (n_j · Var(y) + s_j²)
```

where Var(y) is the variance of the whole target and s_j² is the variance of the target within category j. The random-intercept model uses n_j τ² / (n_j τ² + σ²). The two formulas differ in two places, and both matter.

The first difference is the prior variance. Where the mixed model uses τ², the variance between categories, the encoder uses the total variance of y, which is τ² + σ². For radon without covariates, τ² = 0.096 and Var(y) = 0.727, so the encoder behaves as if the variance between counties were seven to eight times larger than it is. The second difference is the noise variance. Where the mixed model uses the pooled within-county variance σ², estimated from all 919 houses, the encoder uses each county's own within-county variance. For a county measured once, that variance is zero, so λ = 1 and the encoding is the single observation, with no shrinkage at all. For a county measured twice with similar readings, it is close to zero, and the result is the same.

![Figure 3](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/mixed-effects-models/figures/fig3_targetencoder_lambda.png)

*Figure 3. Weight placed on a county's own mean when encoding it. Blue: the random-intercept (empirical Bayes) weight n_jτ²/(n_jτ² + σ²) with REML variance components. Orange: the weight used by scikit-learn's TargetEncoder(smooth="auto"), computed per county and confirmed against the encoder's output. Minnesota radon, no covariates.*

The numbers in Figure 3 are stark. For the three single-house counties, the encoder's weight is 1.0 and the mixed model's is 0.13. For counties with two houses, the encoder's median weight is 0.96 against 0.23; for three houses 0.90 against 0.31; for four, 0.91 against 0.38. Across all counties with five or fewer houses, the median weight is 0.92 for the encoder and 0.38 for the mixed model. The encoder, in short, barely shrinks where shrinkage is needed most.

A downstream model notices. When the cross-fitted encoding is fed to a linear regression together with the floor indicator, the coefficient on the encoded county is 0.44: the regression learns that the encodings are too extreme and halves them all. That is a single global correction applied to a problem that depends on group size. It still under-shrinks a one-house county, and it now over-shrinks a county with a hundred houses whose mean was perfectly reliable. The hold-out results show both sides: the default encoder is worse than partial pooling for thin counties (0.815 against 0.775) and for thick ones (0.745 against 0.727).

The fix is a one-liner once the problem is understood. Fit a null random-intercept model on the training data, compute σ²/τ² (6.64 for radon without covariates), and pass it as a fixed `smooth` parameter, which turns the encoder's weight into n_j / (n_j + σ²/τ²), precisely the mixed-model weight. The downstream coefficient on the encoding rises to 0.93, the hold-out RMSE drops from 0.793 to 0.786, and the corrected encoder beats the default in 84 per cent of splits. It still trails the mixed model slightly (0.786 against 0.781), because the encoding is computed on raw radon without adjusting for floor and because cross-fitting adds its own noise; but the gap that remains is small, and the remedy costs one extra model fit.

The general point is stated here as a conjecture, because a single data set cannot establish it: pipelines that rely on `smooth="auto"` probably under-shrink rare categories whenever the between-category variance is a small share of the total, which is the common case for high-cardinality features such as merchant, product, postcode or user identifiers. The test is cheap and worth running on any production encoder: compute the intraclass correlation of the target across categories, and compare the encoder's per-category weights with n_j τ² / (n_j τ² + σ²).

## Shrinkage is not conservative

It is tempting to read partial pooling as a cautious method that pulls extreme claims toward safety. That reading is wrong, and the radon data show why with a decision that the data were collected to inform. The US Environmental Protection Agency recommends fixing a home when indoor radon reaches 4 picocuries per litre (US EPA, 2016). Suppose a regulator wants to know which counties have a typical basement above that action level (ln 4 = 1.386 on the log scale).

Under no pooling, 47 counties have a basement intercept above the threshold. Under partial pooling, 51 do. No county is flagged by no pooling alone, and four are flagged only by partial pooling, with 1, 2, 2 and 4 measured houses. The reason is that the estimated grand mean, 1.462, is itself above the threshold: the typical Minnesota basement exceeds the action level. Shrinkage pulls small counties toward that mean, which moved four of them up across the line. Partial pooling does not lean toward "nothing to see here"; it leans toward the population, wherever the population is.

The better tool for this decision is not either point estimate but the probability of exceeding the threshold. With the random-intercept model, the conditional distribution of a county's effect given its data has standard deviation τ√(1 − λ_j), and the probability that the county's typical basement exceeds 4 pCi/L follows directly (this approximation ignores the uncertainty in the fixed effects, which is small here). Sixteen counties exceed the threshold with probability above 0.9, and their median size is 8 houses. That is the list to start from with a limited mitigation budget, and it is a different list from the one produced by thresholding a point estimate, which treats a county measured once and a county measured fifty times as equally certain.

## Within and between: what a random-intercept slope estimates

The second data set changes the question from levels to slopes. High School and Beyond is a 1982 survey of American high-school students that Raudenbush and Bryk (2002) used throughout their textbook. The outcome is a maths achievement score; the key covariate is a standardised measure of family socio-economic status (SES); 70 of the 160 schools are Catholic, the rest public.

There are at least four regression slopes of maths on SES in these data, and they differ by a factor of almost three. Ordinary least squares on all students, ignoring schools, gives 3.18 points per unit of SES. Regressing each school's mean maths score on its mean SES gives the between-school slope, 5.91 (standard error 0.37). Adding school fixed effects, equivalently regressing maths on SES after subtracting each school's means, gives the within-school slope, 2.19 (cluster-robust standard error 0.13). A random-intercept model with raw SES as the covariate gives 2.39.

![Figure 4](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/mixed-effects-models/figures/fig4_hsb_within_between.png)

*Figure 4. School mean maths achievement against school mean SES for 160 schools (orange public, green Catholic). The blue line is the between-school regression; each grey segment has the within-school slope and passes through one school's centroid; the dashed line shows the within-school slope at the overall centroid. Data: High School and Beyond 1982 via the mlmRev R package.*

Figure 4 makes the two relationships visible. Across schools, a school whose students are one unit richer on average scores about 5.9 points higher. Inside any school, a student one unit richer than a schoolmate scores about 2.2 points higher. The gap between the two is the contextual effect: the additional achievement associated with attending a school whose students are, on average, better off, holding the student's own SES fixed.

The random-intercept slope of 2.39 is neither. It is a precision-weighted average of the within and between estimates, a result that goes back to Maddala (1971) in the econometric literature on pooling cross-section and time-series data. In this data set the weight on the within estimate is 0.946 and on the between estimate 0.054. The weights depend on how reliable the school means are: with about 45 students per school, the reliability of a school mean, λ_j, averages 0.90, and the random-intercept model leans heavily on within-school variation. With smaller groups it would lean further toward the between-school slope, and the number reported as "the effect of SES" would drift with nothing more than the sizes of the groups. That is an uncomfortable property for a coefficient someone might quote.

The random-intercept model assumes that the school effects b_j are independent of the covariates. Here they are not: richer schools have higher intercepts, which is what a large contextual effect means. In econometrics this is the classic fixed-versus-random-effects question, usually settled with Hausman's (1978) specification test, which compares the two estimates and rejects the random-effects model when they differ by more than sampling noise. There is a more useful resolution, due to Mundlak (1978): keep the random-intercept model but add the group mean of the covariate as an extra regressor. The coefficient on the student's own SES then equals the within estimate exactly, and the coefficient on the school mean measures the contextual effect, with a standard error from the same model. In High School and Beyond, the Mundlak model gives 2.191 (standard error 0.109) for student SES, identical to the fixed-effects estimate, and 3.675 (standard error 0.378) for school mean SES, a contextual effect nearly ten standard errors from zero. Adding the school mean also lowers the between-school standard deviation from 2.18 to 1.64, since part of what looked like school "quality" was composition. Bell and Jones (2015) argue that this within–between formulation should be the default in social-science panel analysis, because it keeps everything the fixed-effects model offers while still estimating group-level effects and group-level variance.

The deeper lesson is that the three slopes answer three different questions. The within slope compares students in the same school. The between slope compares schools. The contextual effect compares two students with the same family background in schools of different composition. A data scientist asked "what is the effect of SES on achievement" has to find out which of these the person asking means, and none of them is causal by default: there are confounders at the student level and at the school level, and the mixed model adjusts for neither.

## Standard errors when the observations are not independent

Turn now to a covariate that lives only at the group level: whether a school is Catholic. A regression of maths achievement on the Catholic indicator and within-school-centred SES gives a Catholic coefficient of 2.81 points under ordinary least squares. The estimate is not the problem. The standard error is. Ordinary least squares reports 0.155. The cluster-robust (CR1) standard error, which allows arbitrary correlation within schools, is 0.422, larger by a factor of 2.72; the square root of the design effect predicted about three. The random-intercept model reports 0.439. Two methods built on different assumptions agree with each other and disagree with ordinary least squares by almost a factor of three. Moulton (1990) documented this pitfall in economics a generation ago, for regressions of individual outcomes on aggregate variables, and it remains common in applied work, especially in A/B tests randomised by store, region or account while outcomes are recorded per user or per transaction.

The covariate that varies within schools tells a different and less familiar story. For centred SES, the ordinary least-squares standard error is 0.117, the cluster-robust one 0.130, and the random-intercept model's 0.109, smaller than either. Adding a random slope for SES raises it to 0.128, close to the cluster-robust value. The random-intercept model removes school-level variation in the intercept from the residuals, which genuinely sharpens the estimate, but it assumes that every school shares the same SES slope. In these data the slope varies: the random-slope model estimates its standard deviation across schools at 0.83 points, with a correlation of 0.58 between school intercepts and slopes, and a likelihood-ratio test rejects the constant-slope model decisively. The random-intercept-only model treats the average slope as if it were estimated from thousands of independent within-school contrasts, when the effective sample for an average of 160 school-specific slopes is much smaller.

The likelihood-ratio test deserves a note of its own, because the textbook chi-squared reference distribution is wrong here. Under the null hypothesis the slope variance is zero, which sits on the boundary of its parameter space, and Self and Liang (1987) and Stram and Lee (1994) showed that the statistic then follows a mixture of chi-squared distributions. For a test that adds a slope variance and its covariance with the intercept, the reference is an equal mixture of χ² with one and two degrees of freedom. The statistic is 20.1, with a p-value of 2.6 × 10⁻⁵ under the mixture against 4.4 × 10⁻⁵ under the naive χ²₂. The naive test is conservative; the conclusion holds either way in this case, but in borderline cases the boundary matters.

### A simulation of interval coverage

Point comparisons on one data set cannot say which standard error is right. A simulation can. Data were generated from a random-intercept, random-slope model whose parameters were estimated on High School and Beyond: intercept standard deviation 2.62, slope standard deviation 0.83, intercept–slope correlation 0.58, residual standard deviation 6.06, average SES slope 2.21, and school sizes resampled from the real ones. In each simulated data set half the schools received a binary "treatment" with a true effect of zero, and the student-level covariate had a slope that varied by school. For each number of schools J in 10, 20, 40, 80 and 160, 2,000 data sets were generated; the Monte Carlo standard error of each coverage estimate is about 0.005. Six interval methods were compared: ordinary least squares with independent errors; ordinary least squares with CR1 cluster-robust errors and a t reference with J − 1 degrees of freedom; a random-intercept model with normal (z) intervals and with t intervals on J − 2 degrees of freedom; and a random-intercept-plus-slope model, again with z and t intervals.

![Figure 5](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/mixed-effects-models/figures/fig5_coverage.png)

*Figure 5. Coverage of nominal 95 per cent intervals, 2,000 simulated data sets per number of schools, calibrated on High School and Beyond. Left: a school-level treatment effect. Right: the average slope of a student-level covariate whose slope varies by school. The grey band is ±2 Monte Carlo standard errors around 0.95.*

Four findings come out of Figure 5, and each corresponds to a mistake that shows up in practice.

Intervals from ordinary least squares with independent errors cover the school-level effect about half the time (between 0.498 and 0.511) at every number of schools, from 10 to 160. Their standard errors are about 0.35 times the actual sampling standard deviation of the estimate. This is not a small-sample problem that more data will fix; adding schools of the same size leaves the ratio unchanged, because the ratio is set by the design effect.

With few groups, normal-reference intervals from the mixed model undercover. With ten schools, the random-intercept z interval covers 0.911 of the time and the random-slope z interval 0.895. Using a t reference with J − 2 degrees of freedom, the between-within approximation for a group-level effect, brings them to 0.947 and 0.934. The cluster-robust interval with a t on J − 1 degrees of freedom reaches 0.924. Better small-sample corrections exist and are routine in R: the Kenward–Roger (1997) and Satterthwaite approximations for mixed models, available through the pbkrtest and lmerTest packages, and the bias-reduced CR2 sandwich of Bell and McCaffrey (2002) with the degrees-of-freedom correction of Pustejovsky and Tipton (2018), available in clubSandwich. They were not run here; the point of the simulation is that the default z interval of a mixed model, which is what statsmodels prints, is not good enough below about twenty groups. McNeish and Stapleton (2016) reach the same conclusion in their review of small-sample multilevel studies.

The random-intercept-only model undercovers a slope that varies by group, at every number of schools. Its coverage for the student-level slope sits around 0.90 with z intervals (0.900 at J = 10, 0.918 at J = 160), and its standard errors are 0.83 to 0.88 times the true sampling standard deviation. Here, too, more schools do not help, because the missing variance component is a feature of the model, not of the sample size. Strikingly, ordinary least squares with independent errors does better on this quantity (0.92 to 0.94) than the random-intercept model does. This is the overconfidence that Schielzeth and Forstmeier (2009) and Barr, Levy, Scheepers and Tily (2013) warned about in ecology and psycholinguistics, and the reason the latter recommended including random slopes for every within-group predictor the analysis means to test.

The random-intercept-plus-slope model covers the student-level slope correctly at every J with z intervals (0.942 to 0.959). With t intervals it is conservative for small J (0.972 at J = 10), which is the safer direction to err.

None of this says when clustering should be accounted for in the first place. Abadie, Athey, Imbens and Wooldridge (2023) give the design-based answer: adjust for clustering when the treatment was assigned in clusters or when the sample was drawn in clusters, not merely because the data can be grouped. For a school-level policy with schools sampled from a larger population, both conditions hold. For a feature computed at the user level in a data set that happens to contain several sessions per user, it is the sampling, not the feature, that requires attention.

## Random slopes, cross-level interactions and a structure that is not neutral

A random slope invites a question: can the variation in slopes be explained? Raudenbush and Bryk's (2002) chapter-four model for High School and Beyond, called "intercepts and slopes as outcomes", lets both the school intercept and the school SES slope depend on school characteristics. In mixed-model notation it has student SES centred within school, school mean SES, the Catholic indicator, and the interactions of centred SES with both school variables, plus a random intercept and a random slope.

The REML estimates from the implementation used here are 12.128 for the intercept (standard error 0.199), 5.333 for school mean SES (0.369), 1.227 for Catholic (0.306), 2.945 for the within-school SES slope in a public school of average SES (0.156), 1.039 for the interaction of SES with school mean SES (0.299), and −1.643 for the interaction of SES with Catholic (0.240). An HLM 5 run of the same model, whose output is reproduced in course material from the University of Coimbra (FPCE, n.d.), gives 12.095, 5.333, 1.226, 2.938, 1.034 and −1.641, so the two agree to within a few hundredths; the small differences are most plausibly due to the version of the data file and the estimation algorithm. The variance component that differs most is the slope standard deviation, 0.32 here against 0.39 in HLM, which is typical: slope variances are the least stable quantities in any mixed model.

The substantive finding is in the interaction with Catholic. At average school SES, the within-school SES gradient is 2.95 points per unit in public schools and 1.30 in Catholic schools: inside a Catholic school, achievement depends much less on family background. Raudenbush and Bryk discuss this result at length, together with its obvious caveat that Catholic schools differ from public ones in admission and composition. Statistically, the two school characteristics explain 85 per cent of the variance in school SES slopes, from 0.696 without school predictors of the slope to 0.101 with them.

There is a less comfortable lesson in the same data. Fit a random-intercept, random-slope model with the Catholic indicator and centred SES, but without the interaction between them, and the Catholic coefficient moves from 2.81 to 3.46. The raw difference between Catholic and public school means is 2.81, and the random-intercept-only model reproduces it (2.805). Add the cross-level interaction back, and the coefficient returns to 2.808. The mechanism is the intercept–slope covariance. With the interaction omitted, Catholic schools have systematically flatter slopes, which the model can only express as negative slope random effects; because intercepts and slopes are estimated to be correlated (0.58), generalised least squares lets that misfit leak into the intercepts, and through them into the Catholic coefficient. The random-effects structure is not neutral bookkeeping for the standard errors. It can move a fixed effect by more than one standard error.

The model with the interaction restored shows the opposite risk: its intercept–slope correlation is estimated at exactly 1.00, with a slope standard deviation of 0.41. A correlation on the boundary is a singular fit, the model's way of saying that the data cannot support the covariance structure it was given. lme4 reports this with a warning; the implementation used here, like many others, simply returns the boundary value. The literature disagrees, productively, about what to do next. Barr et al. (2013) recommend starting from the maximal random-effects structure justified by the design. Matuschek, Kliegl, Vasishth, Baayen and Bates (2017) and Bates, Kliegl, Vasishth and Baayen (2015) show that maximal models can lose power without any gain in error control and recommend simplifying until the fit is supported. A third option is to keep the structure but put a weak prior on the variance parameters, as in the penalised likelihood of Chung, Rabe-Hesketh, Dorie, Gelman and Liu (2013) or a full Bayesian fit with the priors discussed by Gelman (2006). Whatever the choice, a singular fit should be reported, not hidden.

## Beyond the Gaussian case: logistic mixed models

The third data set has a binary outcome: whether each of 1,934 women in 60 districts of Bangladesh was using contraception at the time of the 1988 survey. Covariates are urban or rural residence, age (centred), and number of living children (none, one, two, three or more). Districts range from 2 to 118 women, and 39.2 per cent of the women report use.

A generalised linear mixed model (GLMM) puts the random effect inside the link function:

```
logit P(use_ij = 1 | b_j) = x_ij'β + b_j,     b_j ~ N(0, s²)
```

Unlike the linear case, the random effects cannot be integrated out in closed form. The two standard approximations are the Laplace approximation, which replaces each district's integrand with a Gaussian centred at its mode, and adaptive Gauss–Hermite quadrature, which evaluates the integrand at several points scaled to each district's mode and curvature (Pinheiro and Bates, 1995). With one quadrature point, adaptive quadrature is the Laplace approximation, which is the default in lme4's `glmer`. The implementation written for this article finds each district's mode by Newton's method, applies 25-point adaptive quadrature, and maximises the result over the fixed effects and the log of s.

For these data, the Laplace and 25-point fits are nearly indistinguishable: an urban coefficient of 0.7330 against 0.7324, and a district standard deviation of 0.461 against 0.464. That is the expected outcome with districts of moderate size and modest heterogeneity. With binary data in very small clusters, such as twins, matched pairs or a few observations per user, the Laplace approximation becomes less accurate and tends to underestimate the random-effect variance, and more quadrature points or a Bayesian fit are worth the extra computation.

### Conditional and marginal effects are different quantities

The GLMM estimates the urban coefficient at 0.732 on the log-odds scale, with standard error 0.119. A generalised estimating equations (GEE) model with an exchangeable working correlation, the approach of Liang and Zeger (1986), estimates it at 0.680 (standard error 0.157). An ordinary logistic regression that ignores districts gives 0.797, with a naive standard error of 0.105 that becomes 0.189 once made cluster-robust. The temptation is to treat these as three estimates of one parameter and ask which is best. They are not estimates of one parameter.

The GLMM coefficient is conditional: it compares an urban and a rural woman in the same district, or in districts with the same random effect. The GEE coefficient is marginal, or population-averaged: it compares the average use rate among urban women with the average among rural women, other covariates held fixed, across the whole population of districts. Because the logistic curve is non-linear, averaging S-shaped curves over the distribution of district effects produces a flatter curve, and a flatter curve has a smaller slope on the log-odds scale. Zeger, Liang and Albert (1988) derived a convenient approximation for the ratio,

```
β_marginal ≈ β_conditional / sqrt(1 + c² s²),   c² = (16√3 / 15π)² ≈ 0.346
```

which for s = 0.464 gives 0.965. The observed ratios of GEE to GLMM coefficients are 0.95 to 0.96 for age and the number-of-children indicators, close to that approximation.

![Figure 6](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/mixed-effects-models/figures/fig6_conditional_vs_marginal.png)

*Figure 6. Grey: conditional probability curves for 25 districts drawn from the estimated distribution of district effects. Blue: the curve for a district with b = 0. Orange: the average of the district curves, which is the population-averaged curve. Left: the district standard deviation estimated from the Bangladesh data. Right: a standard deviation of 2.0, typical of strongly clustered binary outcomes such as repeated measurements on the same patient.*

Figure 6 shows why the distinction is easy to miss in these data and impossible to miss elsewhere. With a district standard deviation of 0.46, the population-averaged curve barely departs from the curve of a typical district, and its log-odds slope at the centre is 0.95 times the conditional slope. With a standard deviation of 2, a value common for repeated binary measurements on the same person, the average curve is visibly flatter and its slope is 0.61 times the conditional one. Near the centre of the curve, a conditional odds ratio of 3 then corresponds to a population-averaged odds ratio of roughly 1.95. Neuhaus, Kalbfleisch and Hauck (1991) give a careful comparison of the two families of models for correlated binary data.

For urban residence the ratio of GEE to GLMM is 0.929, further from 1 than the attenuation formula predicts, and the ordinary logistic coefficient (0.797) is larger than both. The explanation is the within–between problem of the previous section reappearing in a binary model. Forty-two of the 60 districts contain both urban and rural women. A Mundlak-style GLMM that adds each district's share of urban women gives a within-district urban effect of 0.649 (standard error 0.126) and a contextual effect of 0.722 (standard error 0.366, z = 1.97), so the between-district effect is 1.371. Different estimators weight these two sources differently, which is why their urban coefficients disagree more than their other coefficients do.

Which quantity to report depends on the decision. For a national programme asking how much contraceptive use would change if a rural population had the urban rate, the marginal effect is the relevant one. On the probability scale, averaged over the women in the sample, the GLMM integrated over the distribution of district effects gives an urban–rural difference of 0.160 (0.497 against 0.337), GEE gives 0.156, and plugging in a district effect of zero gives 0.167. The "typical district" and the "average district" are not the same thing, and the gap grows with the heterogeneity. For a district health officer, the conditional model is the relevant one, and the spread is often more informative than the mean: a rural woman of average age with three or more children has a predicted probability of use of 0.415 in a typical district and 0.419 averaged over districts, but the middle 80 per cent of districts span 0.281 to 0.562. That range is what decides whether a district-level programme should target some districts and not others.

## The wider family of mixed models

The three examples cover the core, but the family is larger, and knowing its shape helps in recognising which member a problem needs.

Nested and crossed random effects. Students nested within classes nested within schools give nested effects, one variance per level. Crossed effects arise when two grouping factors do not nest, as when every student rates several lecturers and every lecturer is rated by many students; lme4's own InstEval data set, with 73,421 course ratings, is the standard example. Crossed designs are common in recommender data (users × items), in pricing (customers × products) and in evaluation of language models (prompts × raters). lme4, glmmTMB and Julia's MixedModels.jl handle them directly; statsmodels' `MixedLM` supports a single grouping factor with additional variance components inside it, which makes crossed designs awkward.

Longitudinal and growth-curve models. Repeated measurements on the same unit over time are the original application of Laird and Ware (1982). A random intercept and a random slope on time give each unit its own trajectory; residual correlation that decays with the gap between measurements (autoregressive, exponential) and residual variances that change over time can be added, which the R package nlme supports through its `correlation` and `weights` arguments (Pinheiro and Bates, 2000).

Other outcome distributions. Counts call for Poisson or negative-binomial GLMMs, and counts with excess zeros for zero-inflated or hurdle models; glmmTMB (Brooks et al., 2017) fits all of these with the same formula syntax as lme4, using the Laplace approximation through automatic differentiation.

Non-linear mixed models. In pharmacokinetics, the concentration of a drug follows a compartmental model whose parameters (absorption rate, clearance, volume) vary between patients. The random effects then enter a non-linear function, and dedicated software such as nlme's `nlme`, NONMEM or Monolix is the norm.

Bayesian hierarchical models. The same model with prior distributions on the variance components, fitted by Markov chain Monte Carlo through Stan (brms in R, Bürkner 2017), PyMC or bambi in Python (Capretto et al., 2022). The Bayesian fit is slower but behaves better exactly where the frequentist one struggles: few groups, variances near zero, correlations near ±1, and any situation where the uncertainty in the variance components should propagate into predictions.

Machine-learning hybrids. Mixed-effects random forests (Hajjem, Bellavance and Larocque, 2014) and Gaussian process boosting (Sigrist, 2022, implemented in the GPBoost library) replace the linear fixed part with a tree ensemble while keeping a random-effects structure for the grouping, alternating between fitting the trees on outcomes with the random effects removed and re-estimating the random effects on the residuals. They are the natural answer when the fixed part is genuinely non-linear and the grouping is high-cardinality, which describes a large share of applied tabular problems.

## How mixed models compare with the alternatives

Each alternative a data scientist might use instead of a mixed model solves part of the problem, and it is worth being precise about which part.

Group fixed effects, one dummy per group, give an unbiased within-group slope even when group effects are correlated with the covariates, which is their great strength in econometrics. They cannot estimate the effect of anything that is constant within a group, such as sector or soil uranium, because the dummies absorb it. They produce the worst predictions for small groups, as the radon hold-out showed, and no prediction at all for a new group. When every group is large, they and the mixed model converge.

Ordinary least squares with cluster-robust standard errors fixes inference without fixing estimation. It is robust to any within-cluster correlation structure, which a mixed model is not, but it needs a reasonably large number of clusters, it provides no shrinkage and no group-level predictions, and its slope still mixes within and between variation. Cameron and Miller (2015) is the practical reference, and Cameron, Gelbach and Miller (2008) cover the wild cluster bootstrap for the few-clusters case.

Generalised estimating equations target population-averaged effects with robust standard errors and are the right choice when the question is marginal, as in the contraception example. They do not predict group-specific quantities and, with a misspecified working correlation, can be less efficient.

Target encoding and gradient boosting with a group column are, as shown above, approximations to partial pooling whose amount of shrinkage is set by heuristics rather than by estimated variances. They can be made to agree closely with a mixed model by setting the smoothing constant from estimated variance components. Tree models with a high-cardinality categorical split behave like no pooling with mild regularisation.

Bayesian hierarchical models are the same model with priors. For many groups and a well-behaved likelihood the two give nearly the same answer; for few groups or boundary estimates the Bayesian version is usually the more honest.

## Implementation in R, Python and Julia

R remains the most complete environment for mixed models. A radon model with a county-level predictor, the High School and Beyond model and the contraception GLMM look like this in lme4:

```r
library(lme4)
m1 <- lmer(log_radon ~ floor + log_uranium + (1 | county), data = radon)

m2 <- lmer(mAch ~ meanses * cses + sector * cses + (1 + cses | school),
           data = Hsb82)

m3 <- glmer(use ~ urban + age + livch + (1 | district),
            data = Contraception, family = binomial, nAGQ = 25)

ranef(m1)        # conditional modes (BLUPs)
VarCorr(m2)      # variance components
isSingular(m2)   # boundary check
```

For small-sample inference on fixed effects, loading lmerTest before fitting adds Satterthwaite and Kenward–Roger degrees of freedom (`summary(m2, ddf = "Kenward-Roger")`, which also needs pbkrtest), and clubSandwich provides CR2 cluster-robust variances for ordinary and mixed models. glmmTMB uses the same formula syntax for negative-binomial, zero-inflated and other families; nlme handles residual correlation structures; brms translates the same formulas into Stan for a Bayesian fit.

In Python, statsmodels has a linear mixed model and GEE:

```python
import statsmodels.formula.api as smf
import statsmodels.api as sm

m1 = smf.mixedlm("y ~ floor + u", radon, groups="county").fit(reml=True)
m2 = smf.mixedlm("mAch ~ meanses*cses + catholic*cses", hsb,
                 groups="school", re_formula="~cses").fit(reml=True)
print(m1.cov_re, m1.random_effects)

gee = sm.GEE.from_formula("use ~ urban + age + C(livch)", groups="district",
                          data=contra, family=sm.families.Binomial(),
                          cov_struct=sm.cov_struct.Exchangeable()).fit()
```

statsmodels prints normal-reference p-values with no small-sample correction, and its optimiser warnings, as noted above, should be read. Its binomial and Poisson mixed models (`BinomialBayesMixedGLM`, `PoissonBayesMixedGLM`) are Bayesian, fitted by a variational approximation or a posterior-mode Laplace fit, not by maximum likelihood. For lme4's full feature set from Python, pymer4 (Jolly, 2018) wraps it through rpy2; for Bayesian models, bambi offers lme4-style formulas on top of PyMC; for tree-based fixed effects with random effects, GPBoost and the MERF package implement the hybrids described above.

Julia's MixedModels.jl, written by the lead author of lme4, uses the same profiled formulation and is the fastest option for very large crossed designs.

For scikit-learn pipelines, the practical adaptation from the target-encoding section is short:

```python
from sklearn.preprocessing import TargetEncoder

null = smf.mixedlm("y ~ 1", train, groups="county").fit(reml=True)
m = null.scale / float(null.cov_re.iloc[0, 0])          # sigma^2 / tau^2
enc = TargetEncoder(smooth=m, target_type="continuous")  # instead of "auto"
```

## A working order for grouped data

The experiments above suggest an order of work that avoids most of the problems described.

Start with the intraclass correlation and the distribution of group sizes, before any covariate enters the model. Together they say how much information lives at the group level, how far the smallest groups will be shrunk, and how badly an independence-based standard error will be wrong for a group-level quantity.

Decide which question is being asked before choosing the estimator: the level of each group, a within-group effect, a between-group effect, a contextual effect, a conditional effect or a population-averaged one. Most disagreements between methods in this article were disagreements about the question, not about the data.

Centre within-group predictors at their group means and add the group means as predictors. The Mundlak formulation costs nothing, recovers the within estimate exactly, and turns the fixed-versus-random-effects debate into an estimate with a standard error.

Put group-level predictors into the model whenever they exist. County uranium explained 77 per cent of the between-county variance in radon and gave the largest improvement for the counties with the least data.

Include a random slope for any within-group predictor whose average effect will be reported or tested, and check whether the fit is singular. If it is, simplify, regularise, or move to a Bayesian fit, and say which.

Use small-sample inference when there are fewer than about twenty to forty groups: Kenward–Roger or Satterthwaite degrees of freedom for mixed models, CR2 with corrected degrees of freedom for cluster-robust regression, at the very least a t reference on J minus the number of group-level parameters.

Validate predictive models by group. Report hold-out error by group size, as in Figure 3, because the aggregate RMSE hides the regime in which methods differ; and use group-wise cross-validation whenever predictions will be needed for groups that are not in the training data.

## What mixed models do and do not solve

Mixed-effects models address a specific data challenge: rows that are not independent because they share an unobserved group component. They do three things well. They estimate each group's level with an amount of pooling that is learned from the data and that adapts to each group's sample size, which on the radon data beat one-hot encoding in every one of 200 hold-out splits and beat gradient boosting with a native categorical split in 93 per cent of them. They produce standard errors that respect the number of groups rather than the number of rows, which in the simulation was the difference between 50 per cent and 95 per cent coverage for a group-level effect. And through the Mundlak formulation they separate within-group from between-group relationships, which in High School and Beyond differ by a factor of nearly three.

They do not adjust for confounding, at either level. They do not make a group-level effect causal because it has a standard error. Their default intervals are too narrow when groups are few. A random-intercept-only model can be more overconfident than ordinary least squares about a slope that varies by group. The random-effects structure can move fixed effects, not only standard errors. And in a non-linear model, the coefficient they report is conditional on the group, which is a different quantity from the population-averaged effect that many decisions need.

Two conjectures follow from the experiments and are offered as conjectures, not as findings. The first, stated earlier, is that target encoders with automatic smoothing under-shrink rare categories in many production systems, because the total variance of the target is a poor stand-in for the variance between categories when the intraclass correlation is low. The second is that the gains from partial pooling reported in aggregate benchmarks understate its value for decisions, because those decisions (which counties to mitigate, which stores to restock, which users to target) concentrate on the extremes of a ranking, where small groups dominate and where, in the radon data, the error of the unpooled estimate was about 20 per cent higher. Both are testable with the code below on any data set with a grouping column.

## Code and data

All code is in the `mixed-effects-models` folder of the Medium-Blog repository: https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/mixed-effects-models. It contains the profiled-REML linear mixed model and the adaptive-quadrature logistic mixed model used throughout, one script per experiment, and the text output from which every number in this article was taken. The radon, High School and Beyond and Bangladesh contraception data are read from the Rdatasets mirror of the HLMdiag and mlmRev R packages.

## References

Abadie, A., Athey, S., Imbens, G. W. and Wooldridge, J. M. (2023). When should you adjust standard errors for clustering? *Quarterly Journal of Economics*, 138(1), 1–35.

Barr, D. J., Levy, R., Scheepers, C. and Tily, H. J. (2013). Random effects structure for confirmatory hypothesis testing: Keep it maximal. *Journal of Memory and Language*, 68(3), 255–278.

Bates, D., Kliegl, R., Vasishth, S. and Baayen, H. (2015). Parsimonious mixed models. arXiv:1506.04967.

Bates, D., Mächler, M., Bolker, B. and Walker, S. (2015). Fitting linear mixed-effects models using lme4. *Journal of Statistical Software*, 67(1), 1–48.

Bell, A. and Jones, K. (2015). Explaining fixed effects: Random effects modeling of time-series cross-sectional and panel data. *Political Science Research and Methods*, 3(1), 133–153.

Bell, R. M. and McCaffrey, D. F. (2002). Bias reduction in standard errors for linear regression with multi-stage samples. *Survey Methodology*, 28(2), 169–181.

Brooks, M. E., Kristensen, K., van Benthem, K. J., Magnusson, A., Berg, C. W., Nielsen, A., Skaug, H. J., Mächler, M. and Bolker, B. M. (2017). glmmTMB balances speed and flexibility among packages for zero-inflated generalized linear mixed modeling. *The R Journal*, 9(2), 378–400.

Bürkner, P.-C. (2017). brms: An R package for Bayesian multilevel models using Stan. *Journal of Statistical Software*, 80(1), 1–28.

Cameron, A. C., Gelbach, J. B. and Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *Review of Economics and Statistics*, 90(3), 414–427.

Cameron, A. C. and Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *Journal of Human Resources*, 50(2), 317–372.

Capretto, T., Piho, C., Kumar, R., Westfall, J., Yarkoni, T. and Martin, O. A. (2022). Bambi: A simple interface for fitting Bayesian linear models in Python. *Journal of Statistical Software*, 103(15), 1–29.

Chung, Y., Rabe-Hesketh, S., Dorie, V., Gelman, A. and Liu, J. (2013). A nondegenerate penalized likelihood estimator for variance parameters in multilevel models. *Psychometrika*, 78(4), 685–709.

Efron, B. and Morris, C. (1975). Data analysis using Stein's estimator and its generalizations. *Journal of the American Statistical Association*, 70(350), 311–319.

FPCE, Universidade de Coimbra (n.d.). Doc.Net 7: Ordenadas- e inclinações-como-resultados, High School and Beyond (HLM 5 output). https://www.fpce.uc.pt/niips/amn/docsnet/docnet_07/Doc.Net_07.htm

Gelman, A. (2005). Analysis of variance: Why it is more important than ever. *Annals of Statistics*, 33(1), 1–53.

Gelman, A. (2006). Prior distributions for variance parameters in hierarchical models (comment on article by Browne and Draper). *Bayesian Analysis*, 1(3), 515–534.

Gelman, A. and Hill, J. (2007). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.

Gelman, A. and Price, P. N. (1999). All maps of parameter estimates are misleading. *Statistics in Medicine*, 18(23), 3221–3234.

Hajjem, A., Bellavance, F. and Larocque, D. (2014). Mixed-effects random forest for clustered data. *Journal of Statistical Computation and Simulation*, 84(6), 1313–1328.

Hausman, J. A. (1978). Specification tests in econometrics. *Econometrica*, 46(6), 1251–1271.

Henderson, C. R. (1975). Best linear unbiased estimation and prediction under a selection model. *Biometrics*, 31(2), 423–447.

Huq, N. M. and Cleland, J. (1990). *Bangladesh Fertility Survey 1989 (Main Report)*. National Institute of Population Research and Training, Dhaka.

Jolly, E. (2018). Pymer4: Connecting R and Python for linear mixed modeling. *Journal of Open Source Software*, 3(31), 862.

Kenward, M. G. and Roger, J. H. (1997). Small sample inference for fixed effects from restricted maximum likelihood. *Biometrics*, 53(3), 983–997.

Kish, L. (1965). *Survey Sampling*. Wiley.

Laird, N. M. and Ware, J. H. (1982). Random-effects models for longitudinal data. *Biometrics*, 38(4), 963–974.

Liang, K.-Y. and Zeger, S. L. (1986). Longitudinal data analysis using generalized linear models. *Biometrika*, 73(1), 13–22.

Maddala, G. S. (1971). The use of variance components models in pooling cross section and time series data. *Econometrica*, 39(2), 341–358.

Matuschek, H., Kliegl, R., Vasishth, S., Baayen, H. and Bates, D. (2017). Balancing Type I error and power in linear mixed models. *Journal of Memory and Language*, 94, 305–315.

McNeish, D. M. and Stapleton, L. M. (2016). The effect of small sample size on two-level model estimates: A review and illustration. *Educational Psychology Review*, 28(2), 295–314.

Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems. *ACM SIGKDD Explorations Newsletter*, 3(1), 27–32.

Moulton, B. R. (1990). An illustration of a pitfall in estimating the effects of aggregate variables on micro units. *Review of Economics and Statistics*, 72(2), 334–338.

Mundlak, Y. (1978). On the pooling of time series and cross section data. *Econometrica*, 46(1), 69–85.

Neuhaus, J. M., Kalbfleisch, J. D. and Hauck, W. W. (1991). A comparison of cluster-specific and population-averaged approaches for analyzing correlated binary data. *International Statistical Review*, 59, 25–35.

Patterson, H. D. and Thompson, R. (1971). Recovery of inter-block information when block sizes are unequal. *Biometrika*, 58(3), 545–554.

Pinheiro, J. C. and Bates, D. M. (1995). Approximations to the log-likelihood function in the nonlinear mixed-effects model. *Journal of Computational and Graphical Statistics*, 4(1), 12–35.

Pinheiro, J. C. and Bates, D. M. (2000). *Mixed-Effects Models in S and S-PLUS*. Springer.

Price, P. N., Nero, A. V. and Gelman, A. (1996). Bayesian prediction of mean indoor radon concentrations for Minnesota counties. *Health Physics*, 71(6), 922–936.

Pustejovsky, J. E. and Tipton, E. (2018). Small-sample methods for cluster-robust variance estimation and hypothesis testing in fixed effects models. *Journal of Business & Economic Statistics*, 36(4), 672–683.

Raudenbush, S. W. and Bryk, A. S. (2002). *Hierarchical Linear Models: Applications and Data Analysis Methods* (2nd ed.). Sage.

Robinson, G. K. (1991). That BLUP is a good thing: The estimation of random effects. *Statistical Science*, 6(1), 15–32.

Schielzeth, H. and Forstmeier, W. (2009). Conclusions beyond support: Overconfident estimates in mixed models. *Behavioral Ecology*, 20(2), 416–420.

Self, S. G. and Liang, K.-Y. (1987). Asymptotic properties of maximum likelihood estimators and likelihood ratio tests under nonstandard conditions. *Journal of the American Statistical Association*, 82(398), 605–610.

Sigrist, F. (2022). Gaussian process boosting. *Journal of Machine Learning Research*, 23(232), 1–46.

Stram, D. O. and Lee, J. W. (1994). Variance components testing in the longitudinal mixed effects model. *Biometrics*, 50(4), 1171–1177.

US Environmental Protection Agency (2016). *A Citizen's Guide to Radon* (EPA 402/K-12/002). https://www.epa.gov/sites/default/files/2016-12/documents/2016_a_citizens_guide_to_radon.pdf

Wainer, H. (2007). The most dangerous equation. *American Scientist*, 95(3), 249–256.

Zeger, S. L., Liang, K.-Y. and Albert, P. S. (1988). Models for longitudinal data: A generalized estimating equation approach. *Biometrics*, 44(4), 1049–1060.
