# Missing Data and Multiple Imputation: Thirteen Methods Compared on Census Wages and NHANES Blood Pressure

## Missingness mechanisms, Rubin's rules, the accuracy trap, the number of imputations, prediction under shifting missingness, and a sensitivity analysis on real survey data

Every data set with more than a handful of columns has holes in it, and every tool a data scientist uses has an opinion about what to do with them. The formula interface of statsmodels drops incomplete rows without saying so: a regression on six rows, two of them with a missing predictor, quietly reports four observations. R's `lm` does the same through its default `na.action`. scikit-learn's linear models refuse to fit and raise an error, and the most common way to make the error go away is `SimpleImputer`, whose default strategy fills every hole with the column mean. Gradient-boosting libraries accept the missing values and learn where to send them at each split. Four tools, four different assumptions about the missing data, and in most code the choice between them is made by whichever error message appears first.

The choice matters more than it looks. Deleting incomplete rows can bias an estimate or leave it untouched, depending on why the values are missing and on what is being estimated. Filling holes with a mean shrinks variances and distorts relationships. Filling them with a careful prediction can be worse still, for reasons that surprise people the first time they see them. And all single-value fixes share a defect that no amount of cleverness repairs: they treat invented numbers as if they had been measured, so every standard error computed afterwards is too small.

Multiple imputation is the standard statistical answer to this data challenge. This guide explains what it does, why it works when it works, and where it fails, and checks every claim against data. The core experiment draws a thousand samples from the 28,155 men in the March 1988 Current Population Survey, deletes 40 per cent of their years of education under four different missingness mechanisms, and compares thirteen ways of handling the gap against the answer known from the full data. Further experiments measure how many imputations are needed, what changes when the goal is prediction rather than inference and the missingness shifts at deployment, and what multiple imputation does to a real analysis of blood pressure in the NHANES 2009–2012 survey, where nobody knows the missing values and a sensitivity analysis is the only honest way to report the result. The imputation code was written from scratch for the article, following the R package mice, and checked against statsmodels; it is linked at the end.

## Three mechanisms, and why the estimand matters as much as the mechanism

Rubin (1976) gave the vocabulary that every later treatment uses. Split the data into an observed part and a missing part, and let R be the indicator of which values are missing. The data are missing completely at random (MCAR) when the probability of R does not depend on any data, observed or missing: a blood sample dropped on the floor, a page lost in the post. They are missing at random (MAR) when the probability depends only on observed values: older respondents skip a question more often, and age is recorded for everyone. They are missing not at random (MNAR) when the probability depends on the missing values themselves, even after conditioning on everything observed: people with little schooling decline to report it, people with high blood pressure skip the examination.

The names are unfortunate. "Missing at random" does not mean random in any everyday sense; it means random once the observed data are taken into account, which can describe a highly systematic pattern. And the labels are not properties of a data set in isolation. They describe the data together with a model for them. A variable can be MNAR with respect to a model that omits a predictor of missingness and MAR with respect to one that includes it, which is why the choice of auxiliary variables in an imputation model is one of the few levers an analyst has.

Two facts govern the practical consequences. The first is that MAR cannot be tested against MNAR with the data at hand. The data can show that missingness is related to observed variables, which rules out MCAR (Little, 1988b, gives a formal test), but no amount of observed data can show what the unobserved values would have been. Every analysis of incomplete data rests on an assumption about the missing part, and the choice of method is a choice of assumption, whether or not it is made deliberately.

The second fact is less familiar, and it decides whether imputation is needed at all: whether an analysis is biased depends on the estimand as much as on the mechanism. For a regression of y on x with missing values in x, complete-case analysis is unbiased whenever the probability of missingness depends on x (including the missing x itself) but not on y given x, provided the regression model is correctly specified (Little, 1992; White and Carlin, 2010). That condition holds under some MNAR mechanisms and fails under some MAR ones. The same complete cases give a biased estimate of the mean of x under almost any mechanism other than MCAR. One data set can therefore support an unbiased complete-case regression slope and a badly biased complete-case mean, and the experiments below show both, on the same samples.

## What multiple imputation does

The idea, due to Rubin (1987), is to replace each missing value not by a single best guess but by m draws from its posterior predictive distribution given the observed data, producing m completed data sets. The analysis of interest is run on each, and the m results are combined. The combination, known as Rubin's rules, takes the average of the m point estimates and adds two sources of uncertainty:

```
Q̄ = (1/m) Σ Q_i                   pooled estimate
Ū = (1/m) Σ U_i                   within-imputation variance
B = (1/(m − 1)) Σ (Q_i − Q̄)²      between-imputation variance
T = Ū + (1 + 1/m) B               total variance
```

The within-imputation variance is what each completed-data analysis reports. The between-imputation variance measures how much the answer changes when the missing values are redrawn, which is exactly the information the missing data would have carried had they been observed. The factor 1 + 1/m corrects for using a finite number of imputations. Two derived quantities are worth reporting with every analysis. The relative increase in variance is r = (1 + 1/m)B / Ū. The fraction of missing information (FMI) is approximately (1 + 1/m)B / T, the share of the total uncertainty that is due to the missing values; mice reports a small-sample refinement of it, and so does the code used here. The FMI is not the share of missing cells. With good predictors of the missing values it can be far smaller, and for a parameter that leans heavily on a poorly observed variable it can be larger.

Intervals use a t reference whose degrees of freedom reflect both the number of imputations and the sample size. Rubin's original formula, (m − 1)(1 + 1/r)², can exceed the complete-data degrees of freedom in small samples; Barnard and Rubin (1999) proposed the adjustment that mice and most current software use, which combines Rubin's value with an observed-data value based on the complete-data degrees of freedom.

For the rules to give valid inference, the imputations must be proper: they must reflect uncertainty about the parameters of the imputation model as well as the residual variation around it. Before each imputation a fresh set of imputation-model parameters is drawn from their posterior distribution (or the model is refitted to a bootstrap sample), and the missing values are drawn given those parameters. Imputing from the fitted model without redrawing its parameters, which is what a single stochastic regression imputation does, understates the between-imputation variance even when it gets the point estimate right.

The imputation model must also be at least as rich as the analysis model. Meng (1994) called this congeniality: if the analysis model contains a relationship, the imputation model must contain it too, or the imputed values will dilute it. The most common violation is leaving the outcome out of the model used to impute a covariate. Using y to fill in x and then regressing y on x feels like cheating, but it is the opposite. Omitting y imputes values that are unrelated to y by construction, and the coefficient on x is pulled toward zero (Moons, Donders, Stijnen and Harrell, 2006). The experiment below puts a number on how far.

### Chained equations and predictive mean matching

With several incomplete variables, a joint model for all of them is often hard to write down. Multiple imputation by chained equations, also called fully conditional specification, sidesteps the problem by specifying one conditional model per incomplete variable (a linear regression for a continuous variable, a logistic regression for a binary one) and cycling through them: impute each variable from its model given the current values of all the others, move to the next, and repeat for a number of iterations. The procedure is a Gibbs sampler when the conditional models are compatible with some joint distribution, and in practice it behaves well even when they are not exactly (van Buuren and Groothuis-Oudshoorn, 2011; White, Royston and Wood, 2011). The alternative is joint modelling, usually under a multivariate normal distribution, as in Amelia (Honaker, King and Blackwell, 2011); Lee and Carlin (2010) found the two approaches give similar answers in typical epidemiological applications.

Predictive mean matching (PMM) is the default method for continuous variables in mice. Instead of drawing an imputed value from a normal distribution around the regression prediction, PMM finds the observed cases whose predicted values are closest to the predicted value of the missing case and borrows the observed value of one of them, chosen at random among a small number of donors (five in mice and here). The idea goes back to Rubin (1986) and Little (1988a), and van Buuren (2018) discusses its variants and tuning at length. Because every imputed value is a value that actually occurs in the data, PMM never imputes impossible values such as negative years of schooling, it preserves skewness and heaping, and it is less sensitive than the normal model to a misspecified regression, at the cost of some bias when missing cases fall where observed donors are scarce (Morris, White and Royston, 2014).

The implementation used throughout follows mice closely. The Bayesian linear regression step draws σ² from its scaled inverse-χ² posterior and β from a normal distribution around the least-squares estimate; PMM uses the least-squares predictions for the observed rows and the posterior-draw predictions for the missing rows, which is mice's default "type 1" matching; binary variables use a logistic regression whose coefficients are drawn from the normal approximation to their posterior. Each imputation runs its own chain for ten iterations from random starting values.

## An experiment with a known answer

Many demonstrations of missing-data methods use simulated multivariate normal data, which flatters every method that assumes normality. The experiment here uses a real population instead. The AER package (Kleiber and Zeileis, 2008) distributes 28,155 records from the March 1988 Current Population Survey, collected by the US Census Bureau and analysed by Bierens and Ginther (2001): men aged 18 to 70 with positive earnings, not self-employed, with their weekly wage, years of education, years of potential experience, ethnicity (the file records only African-American or not), residence in a metropolitan area, region and part-time status. The file has no missing values, which is what makes it useful: it can serve as the population, and every method can be scored against the population answer.

The analysis model is the Mincer (1974) earnings equation, one of the most estimated regressions in labour economics: the log of the weekly wage regressed on years of education, experience, experience squared, an African-American indicator and a metropolitan-area indicator. Its coefficient on education, the return to schooling, is the main target; in the population it is 0.0832, a little over 8 per cent higher weekly earnings per additional year of school (0.0840 in the model-true version described below). Two secondary targets are the coefficient on the African-American indicator (a fully observed covariate, −0.261 in the population) and the population mean of education itself (13.07 years), which is a different kind of estimand and behaves differently.

Each of 1,000 replications draws 1,000 men without replacement and deletes 40 per cent of their education values under one of four mechanisms, each a logistic function of a standardised driver with slope 1.5:

- MCAR, the same deletion probability for everyone;
- MAR on experience, more deletions among older men, whose experience is observed;
- MAR on wage, more deletions among low earners, whose wage is observed;
- MNAR on education, more deletions among men with little schooling, the missing value itself.

Thirteen methods are compared. Complete data, before any deletion, is the benchmark. Complete-case analysis drops the rows with missing education. Six single imputations fill each hole once: the observed mean; the mean plus a missing-data indicator in the regression; a deterministic regression prediction; a stochastic regression draw; the average of the five nearest neighbours (scikit-learn's `KNNImputer`); and a random-forest prediction in the style of missForest (Stekhoven and Bühlmann, 2012). Five multiple imputations use m = 20: Bayesian linear regression, predictive mean matching, PMM with the outcome deliberately left out of the imputation model, PMM with an imputation model that adds the square of log wage and its interactions with the other covariates, and a random-forest multiple imputation modelled on mice's `rf` method (Doove, van Buuren and Dusseldorp, 2014). Every imputation model except the deliberately crippled one uses log wage, the analysis covariates and four auxiliary variables (part-time status and three region indicators). All standard errors are heteroscedasticity-robust (HC1), so that the complete-data benchmark itself has the right coverage.

There is one more design choice, and it turns out to matter. The Mincer equation is a good approximation to the relationship between schooling, experience and wages, but it is an approximation. In this population the return to schooling falls steeply with experience: adding an education-by-experience interaction to the population regression gives a t statistic of −18.6, and the implied return at twelve years of schooling is 0.124 for a man with five years of experience and 0.065 for one with thirty. To separate what the missingness mechanism does from what model misspecification does, every scenario is run twice. In the first version the population keeps its real covariates, but log wages are regenerated from the fitted Mincer equation with normal errors, so the analysis model is exactly right by construction. In the second, the wages are the real ones. With 1,000 replications per cell, the Monte Carlo standard error of a coverage estimate near 95 per cent is about 0.7 percentage points.

## What the grid shows

![Figure 1](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/multiple-imputation/figures/fig1_coverage_grid.png)

*Figure 1. Coverage of nominal 95 per cent intervals for the return to schooling, by method (rows) and missingness mechanism (columns), with the relative bias of the point estimate in small type. Left: wages regenerated from the Mincer equation, so the analysis model is correct. Right: real wages. Darker cells fall further below 95 per cent. 1,000 samples of 1,000 men per cell, 40 per cent of education deleted. Data: March 1988 CPS via the AER R package.*

The left half of Figure 1 is the textbook world, and it reproduces the textbook almost line by line.

Complete-case analysis is unbiased under MCAR (bias −0.3 per cent, coverage 94.4 per cent), under MAR on experience (+0.4 per cent, 95.6 per cent) and, more strikingly, under MNAR on education itself (−0.7 per cent, 94.6 per cent). In the last case the missingness depends on the very values that are missing, the textbook definition of the hardest case, and complete cases do not care, because the deletion depends on x and not on y given x. Complete-case analysis fails in one column only: when missingness depends on the outcome. With low earners more likely to have missing education, the complete-case return to schooling is 16.8 per cent too low and its nominal 95 per cent interval covers the truth 56.4 per cent of the time. Selecting a sample on the outcome flattens a regression slope, the same mechanism that makes regressions on truncated samples misleading. The coefficient on the fully observed African-American indicator is biased too under that mechanism, by 19 per cent, which is a reminder that deletion distorts every coefficient in the model, not only the one on the incomplete variable.

The two standard multiple imputations, Bayesian regression and PMM, get every column right. MI with Bayesian regression has biases between −1.2 and +1.9 per cent and coverage between 93.6 and 95.7 per cent; PMM has biases between −4.2 and +0.9 per cent and coverage between 92.5 and 93.6 per cent, slightly below nominal, a small cost of matching on a discrete variable with heavy heaping (37.5 per cent of the men have exactly twelve years). Under MAR on wage, the case that defeats complete cases, MI with Bayesian regression is 1.0 per cent off with 93.6 per cent coverage. Under MNAR on education, the case that should defeat MI, it is off by only 1.9 per cent. The reason is instructive. Under that mechanism the imputation model is fitted on men who reported their schooling, who have more of it, and it imputes too much schooling to the men who did not; but the error is close to a constant shift across the missing cases, and a constant shift in x moves a regression intercept, not its slope. The mean of education is where the shift shows up: complete cases estimate it at 14.23 years against a population value of 13.07, MI with Bayesian regression at 14.04, and every method that had to deal with the deleted values covered the true mean in none of its thousand intervals. No method that assumes MAR can recover an MNAR mean, and none did.

The mean of education also shows the other half of the estimand argument. Under MAR on experience, complete cases overstate mean schooling by 0.37 years and cover the truth in 6.5 per cent of samples, while the same complete cases estimate the return to schooling without bias. Multiple imputation corrects the mean (bias 0.04 years), though its intervals cover only 91.2 per cent of the time with Bayesian regression and 88.4 per cent with PMM, because older men are scarce among the observed cases and the imputation model has to extrapolate. Whether deleting incomplete rows is acceptable is a question about the parameter, not about the data set.

Leaving the outcome out of the imputation model is the most damaging mistake in the grid, and it is easy to make, because it is what tends to happen when imputation is treated as a preprocessing step separate from the analysis. The return to schooling is attenuated by 40 to 49 per cent in every column, and coverage is between 0.1 and 2.2 per cent. The imputed values are drawn from the right distribution of education given everything except wages, so they carry no information about the relationship between education and wages, and the 40 per cent of imputed rows act as noise that drags the slope toward zero. Under MCAR this method gets the mean of education exactly right (bias 0.00 years) while ruining the regression, a combination that would pass most data-quality checks.

Mean imputation attenuates the slope too, but less: 5.3 per cent under MCAR and 21.4 per cent under MAR on wage. Its damage spreads to the covariates that are correlated with the imputed one, a point made long ago by Jones (1996): the coefficient on the African-American indicator is 10.6 per cent too large in magnitude even under MCAR. In the rows where schooling has been replaced by a constant, part of its effect is picked up by the covariates correlated with it, and African-American men in these data have less schooling on average (12.3 years against 13.1). Adding a missing-data indicator to the regression, a common fix, changes nothing for the education slope here and is known to be biased in general for observational data (Groenwold, White, Donders, Carpenter, Altman and Moons, 2012); its legitimate use is in randomised trials with missing baseline covariates (White and Thompson, 2005) and, as shown below, in prediction. For the mean of education, mean imputation leaves the point estimate unchanged and simply lies about its precision: the standard error is 61 per cent of the true sampling standard deviation, and the interval covers 77.3 per cent of the time under MCAR.

The single stochastic regression imputation is the instructive near miss. Its point estimates are as good as those of multiple imputation (biases between −1.4 and +1.7 per cent), because a single random draw from the right conditional distribution preserves the relationship between education and wages. Its standard errors are not: they are 62 to 69 per cent of the true sampling variability, and coverage is between 78.3 and 82.7 per cent. This is the cleanest demonstration in the study of what Rubin's between-imputation variance is for. The single draw contains the right amount of noise and no acknowledgement of it.

That leaves the three deterministic imputations, regression, k-nearest neighbours and random forest, and they deserve a section of their own.

## Accurate imputations, wrong coefficients

![Figure 2](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/multiple-imputation/figures/fig2_accuracy_vs_validity.png)

*Figure 2. Each point is a method: the root mean squared error of its imputed education values (horizontal) against the bias of the return to schooling computed from the completed data (vertical). Left: MCAR. Right: MAR on wage. Orange: single imputations and MI without the outcome; blue: multiple imputation with Bayesian regression and with PMM. The single stochastic-regression draw (open circle) sits on top of the MI points: the same bias, with intervals that are too narrow. Wages regenerated from the Mincer equation; 1,000 samples per point.*

Deterministic regression imputation fills each missing education value with its predicted value given wages and the other covariates. It is the most accurate imputation in the study, in the sense that machine-learning benchmarks of imputation methods usually measure: under MCAR its imputed values are off by 2.55 years (root mean squared error), against about 3.6 years for the multiple imputations. It is also the most biased method in the study by a wide margin. The return to schooling computed from the completed data is 51.5 per cent too high under MCAR, 54.7 to 59.5 per cent too high under the other mechanisms, and coverage never reaches 3.5 per cent. The k-nearest-neighbour imputation has an error of 2.79 years and inflates the slope by 22.8 per cent; the random forest, at 2.71 years, inflates it by 33.7 per cent. Figure 2 shows the pattern directly. Ranked by imputation error, the methods line up in almost exactly the reverse order of their validity for the regression.

The mechanism is not subtle once it is seen. A prediction of education from wages is a function of wages with the residual variation removed. Put those predictions back into a regression of wages on education, and the imputed rows sit on a line that was fitted using wages, with none of the scatter that real education values would have around it. The relationship between education and wages in the imputed rows is stronger than in reality, and the slope is pulled up toward the reciprocal of the reverse regression. Little's (1992) review of regression with missing covariates discusses why conditional-mean imputation of regressors fails in this way. Random forests and nearest neighbours shrink less toward a line than linear regression does, which is why their bias is smaller, but they are still conditional means, and conditional means are the wrong thing to impute when the next step is to estimate a relationship.

The general point applies well beyond education and wages. The quality of an imputation for inference is not measured by how close the imputed values are to the truth; it is measured by whether the completed data reproduce the joint distribution of the variables, including their noise. A method that imputes the conditional mean minimises imputation error by construction and distorts every quantity that depends on variances and covariances. Benchmarks that rank imputation methods by RMSE on masked cells answer a question that matters for prediction pipelines (where the conditional mean is often exactly what is wanted, as the prediction experiment below shows) and is close to irrelevant for estimation. Stochastic methods look worse on that benchmark and are right for the job. This is why Shah, Bartlett, Carpenter, Nicholas and Hemingway (2014), comparing random-forest and parametric imputation, used the forests inside MICE, as an engine for proper multiple imputation, rather than as a single-imputation shortcut.

## When the analysis model is only an approximation

The right half of Figure 1 is the same experiment with real wages, and the textbook pattern bends.

Under MCAR nothing changes: complete cases (−0.1 per cent bias, 94.9 per cent coverage), MI with Bayesian regression (−0.7 per cent, 92.6 per cent) and PMM (0.0 per cent, 92.6 per cent) are all fine, and the deterministic imputations are as biased as before. Under MAR on experience, the mechanism where complete cases were unbiased in the model-true world, complete cases now overstate the return to schooling by 18.5 per cent, with 60.8 per cent coverage. Under MAR on wage, multiple imputation, which was unbiased in the model-true world, now overshoots: +22.9 per cent with Bayesian regression and +14.3 per cent with PMM, while complete cases are biased by −8.5 per cent, in the opposite direction and by less.

Neither reversal is a failure of the theory. Both come from asking a linear model to describe a population that is not linear. When the Mincer equation is only an approximation, its population coefficients are the coefficients of the best linear approximation over the whole population, and a sample selected on experience has a different best linear approximation, because the real return to schooling differs between young and old men. Here the selected sample is younger, and younger men have the higher return to schooling, which is the direction of the complete-case bias. Complete-case analysis estimates the right model on the wrong population. The imputation model has the mirror-image problem. A linear regression of education on log wage and covariates, fitted where education is observed, is extrapolated to the rows where it is not, and if the relationship between wages and education bends or differs by experience, the imputed values carry the curvature of the wrong part of the population into the completed data.

The fix is the one the congeniality argument implies: make the imputation model as rich as the structure of the data, not merely as rich as the analysis model. The PMM variant whose imputation model adds the square of log wage and its interactions with experience, experience squared, ethnicity, metropolitan residence and part-time status reduces the bias under MAR on wage from +14.3 to +4.1 per cent and raises coverage from 70.3 to 90.9 per cent; under MAR on experience the bias falls from +10.3 to +6.4 per cent and coverage rises from 81.3 to 87.0 per cent. In the model-true population, where the extra terms have nothing to capture, the richer model costs little: its biases stay between −5.8 and 0.0 per cent and its coverage between 92.5 and 94.7 per cent.

The random-forest multiple imputation, which is supposed to learn such interactions without being told about them, does not solve the problem; it replaces it with another. With the mice defaults of ten trees and leaves of at least five observations, its estimates are attenuated in every column of both populations: by 10.4 per cent under MCAR with real wages, by 13.1 per cent under MAR on wage, and by 12.5 to 23.1 per cent in the model-true population. A small forest reproduces the steep, nearly linear relationship between schooling and wages only in steps, and the donors it draws from a leaf are too loosely tied to the wage of the case being imputed, which pulls the slope toward zero in the same way that leaving the outcome out does, only less. Its intervals are wide enough to cover the truth in some cells (96.4 per cent under MAR on experience with real wages) and far from it in others (42.3 per cent under MAR on wage in the model-true population). Flexible engines have a place inside chained equations, and Shah et al. (2014) reported good results for them in simulations built on electronic health records, but flexibility in the imputation model is not automatically fidelity to the relationship the analysis is about; a forest is worth tuning, and checking against a parametric model, before it is trusted.

Two lessons follow for practice. The first is that MAR is not a property that makes multiple imputation work on its own; it makes it possible to work, given an imputation model that is right in the region of the data where values are missing. The second is that complete-case analysis is not a safe default when missingness depends only on covariates, unless the analysis model is close to correctly specified, which with real data it rarely is exactly. The choice between the two is an empirical question, and the most reliable way to answer it is to run both with a deliberately flexible imputation model and to treat a large disagreement as a finding rather than an inconvenience. Hughes, Heron, Sterne and Tilling (2019) argue the same from the epidemiological side: multiple imputation is not always the answer, and the mechanism, the estimand and the models together decide.

## How many imputations?

The early advice, going back to Rubin (1987), was that three to five imputations are enough, and it was right for the question it answered. The efficiency of a point estimate based on m imputations, relative to an infinite number, is (1 + FMI/m)⁻¹, so with 40 per cent missing information five imputations already reach 93 per cent efficiency. Point estimates, however, are not the only output of an analysis. Standard errors, confidence limits and p-values are estimated from the spread between imputations, and the spread is estimated from m − 1 degrees of freedom. Graham, Olchowski and Gilreath (2007) and Bodner (2008) showed that p-values and interval widths need far more imputations than point estimates, and von Hippel (2020) turned the argument into a rule: to keep the coefficient of variation of the pooled standard error below a chosen value, use

```
m = 1 + ½ (FMI / CV)²
```

The rule follows from the sampling variability of the between-imputation variance. If B is estimated from m − 1 degrees of freedom, the standard error √T has a coefficient of variation of about FMI / √(2(m − 1)).

![Figure 3](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/multiple-imputation/figures/fig3_number_of_imputations.png)

*Figure 3. Left: coefficient of variation of the pooled standard error of the return to schooling across 200 independent MI runs on the same data set, against the number of imputations, with the approximation FMI/√(2(m − 1)). Right: upper limit of the 95 per cent interval across the 200 runs, at m = 5 and m = 100. One sample of 1,000 men from the real CPS population with 40 per cent of education missing at random given wage; PMM with the richer imputation model.*

The experiment in Figure 3 holds the data fixed and repeats the entire multiple imputation 200 times with different random seeds, for m from 2 to 100. On this sample the fraction of missing information for the return to schooling is 0.65. The observed coefficient of variation of the pooled standard error falls from 0.44 at m = 2 to 0.23 at m = 5, 0.10 at m = 20 and 0.045 at m = 100, and the approximation tracks it at every m (the formula gives 0.46, 0.23, 0.11 and 0.046). The right panel shows what that means for a reported result. At m = 5 the upper limit of the 95 per cent interval for the return to schooling ranges from 0.086 to 0.133 across seeds; at m = 100 it ranges from 0.095 to 0.104. The point estimate moves too, by 0.35 standard errors (one standard deviation across seeds) at m = 5 and 0.08 at m = 100.

Two analysts with the same data, the same software and the same model, using m = 5 and different seeds, can therefore report upper confidence limits that differ by almost the full width of the interval, and for a coefficient near a significance threshold one of them can report a significant result and the other not. With an FMI of 0.65, von Hippel's rule gives m = 22 for a coefficient of variation of 10 per cent and m = 85 for 5 per cent. A common rule of thumb, at least as many imputations as the percentage of incomplete cases (White, Royston and Wood, 2011), gives 40 here, and setting m to 100 times the FMI gives 65. The FMI is high here partly because the richer imputation model has many parameters to redraw, each adding to the between-imputation variance. The cost of imputations is computation, which is cheap; the cost of too few is irreproducibility, which is not. Since the FMI is itself estimated noisily at small m (its standard deviation across seeds is 0.17 at m = 5), von Hippel recommends a two-stage procedure: a pilot run with a modest m to estimate the FMI, then the final run with the m that the rule gives for an upper bound on it.

## Prediction is a different problem

Everything so far concerns estimating parameters. A large share of data-science work instead trains a model to predict, and the missing-data problem changes character. A predictive model is evaluated on its predictions, not its coefficients, so the distortions that ruin inference can be harmless. And missingness itself can be informative: if the people who leave a field blank differ systematically from those who fill it in, the fact that the field is blank is a feature. Josse, Chen, Prost, Varoquaux and Scornet (2024) showed that for prediction, imputing a constant such as the mean before learning is consistent when the missing values are not informative, a sharp contrast with inference, where the same imputation distorts the distribution of the data; for informative missingness they recommend trees that treat missing values directly. Le Morvan, Josse, Scornet and Varoquaux (2021) studied which imputations make the downstream learner's job easy. Trees treat missingness directly by learning at every split which side the missing values go to, an approach Twala, Jones and Hand (2008) called "missingness incorporated in attributes" and the one scikit-learn's histogram gradient boosting uses natively. Benchmarks on health databases have found simple imputation plus a missingness indicator, or native handling in trees, competitive with far more elaborate approaches for prediction (Perez-Lebel, Varoquaux, Le Morvan, Josse and Poline, 2022).

There is a catch, and it is an operational one. A model that has learned what missingness means has made the missingness process part of its input distribution, and the missingness process is exactly the kind of thing that changes when a form is redesigned, a data source is replaced or a field becomes mandatory.

The experiment uses the real CPS wages. In each of 30 repetitions, training and test sets of 5,000 men are drawn without overlap, and 40 per cent of education is deleted in training under the MNAR mechanism, so that a missing value signals less schooling. Six models are trained: gradient boosting (`HistGradientBoostingRegressor`) with native missing values, with mean imputation, with mean imputation plus an indicator, and with regression imputation from the other features; and a linear model with experience squared, with mean imputation plus an indicator or with regression imputation. Wages cannot be used to impute education at prediction time, because they are what is being predicted, so every imputation here uses only the features. Each model is then tested under three deployment scenarios: the same missingness mechanism as in training; 40 per cent missing completely at random, as might happen after a change in data collection; and no missing values at all.

![Figure 4](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/multiple-imputation/figures/fig4_prediction_shift.png)

*Figure 4. Mean prediction error (predicted minus observed log weekly wage) on test rows whose education is missing, for six models trained with education missing not at random. Blue: test missingness follows the training mechanism. Orange: test missingness is completely at random. Real CPS 1988 wages; 30 repetitions of 5,000 training and 5,000 test men.*

Under the training mechanism, every model that can see the missingness, whether through native handling, a mean value that acts as a marker, or an explicit indicator, predicts the rows with missing education without bias (mean errors between +0.001 and +0.004 log points) and with nearly identical accuracy: test RMSE 0.551 for all three of these boosting variants and 0.540 for the linear model with an indicator, the best of the six, because the Mincer form fits these data well. Boosting on regression-imputed values behaves the same way (+0.007), presumably because imputed values, being fitted values rather than whole years, are themselves recognisable to the trees. The linear model with regression imputation is the exception. It imputes education from the observed relationship, which under MNAR overstates the schooling of the men who left it blank, and it overpredicts their wages by 0.132 log points, about 14 per cent.

Under MCAR at deployment the picture inverts. The models that learned "missing means less schooling" now apply that lesson to men whose education is missing for no reason, and underpredict their wages by 0.135 to 0.141 log points, about 13 per cent, in all five models. Their RMSE on the affected rows rises from 0.56–0.58 to 0.59–0.61, and overall test RMSE rises by 2.6 to 2.9 per cent. The linear model with regression imputation, which was the worst under the training mechanism, is now unbiased (−0.003) and has the lowest overall RMSE of the six (0.553). When education is fully observed at deployment, all models improve, by 2.0 to 2.9 per cent in RMSE, and the differences between them shrink.

Overall RMSE is a poor guide to what has happened, because the damage is concentrated in a subgroup. A 2.7 per cent rise in RMSE could easily go unnoticed on a monitoring dashboard. A 13 per cent systematic underprediction of the wages of a specific group of people, defined by nothing more than a blank field, is the kind of error that matters for pricing, credit or eligibility decisions and the kind a fairness audit looks for. The practical rules are short. For prediction, missingness indicators and native handling are usually the right default, because they exploit information that imputation throws away. The missingness rate and pattern of each feature then become inputs to be monitored like any other, and when the data-collection process changes, the model has to be retrained or at least re-validated on data with the new pattern. And the two goals should not share a pipeline: an imputation designed for a predictive model is the wrong input for an inferential analysis, and the reverse.

## Real missing data: NHANES 2009–2012

The experiments so far had the luxury of knowing the deleted values. In real work the missing values are missing, and the analysis has to rely on what the observed data say about the missingness, on assumptions, and on a sensitivity analysis. The National Health and Nutrition Examination Survey combines a household interview with a physical examination and laboratory tests in a mobile examination centre, and its public files have the kinds of gaps that real surveys have. The data used here are the 2009–2010 and 2011–2012 samples as distributed in the NHANES R package (Pruim, 2026; the `NHANESraw` table, the actual survey sample rather than the package's resampled teaching version), restricted to the 10,990 participants aged 20 to 79. The analysis ignores the survey weights and design, so its estimates describe this sample and not the US population; the point is the missing data, not the epidemiology.

The analysis model regresses systolic blood pressure (the average of up to three readings) on age, age squared, sex, body mass index, the ratio of family income to the poverty line, total cholesterol, race and ethnicity, and whether the participant has ever smoked 100 cigarettes. Two descriptive quantities are estimated as well: the mean systolic pressure and the share of participants with a reading of 140 mmHg or more.

### Who is missing

Systolic pressure is missing for 7.5 per cent of participants, BMI for 4.1 per cent, the poverty ratio for 9.3 per cent and cholesterol for 9.5 per cent; only 78.8 per cent of participants (8,659 of 10,990) are complete on all the analysis variables, so a complete-case analysis discards more than a fifth of the sample. The patterns follow the structure of the survey. The most common incomplete pattern is the poverty ratio alone (7.3 per cent of participants), which is item nonresponse to the income questions in the interview; in 1,002 of the 1,022 cases where the poverty ratio is missing, household income is missing too, so income cannot serve as a proxy. Next come cholesterol alone (5.0 per cent), presumably a blood sample not taken or not analysed, and blood pressure alone (3.0 per cent). Then there is a block of 2.5 per cent in which blood pressure, BMI and cholesterol are all missing together, most plausibly participants who were interviewed but did not attend the examination. Overall, 41 per cent of the participants without a blood-pressure reading also lack BMI and cholesterol.

The observed data clearly rule out MCAR for the poverty ratio. It is missing for 5.3 per cent of White participants, 10.4 per cent of Black participants, 13.8 per cent of Hispanic participants and 11.2 per cent of those of other race, and for 15.9 per cent of those with less than ninth-grade education against 6.4 per cent of college graduates. A logistic regression of the missingness indicator on the complete variables confirms the gradient by ethnicity and a modest age effect. Whether the missingness also depends on income itself, which is the classic concern with income questions, the data cannot say.

### Complete cases against multiple imputation

The imputation model includes every analysis variable and several auxiliary ones that the analysis model does not use but that predict the missing values or their missingness: diastolic pressure, pulse, HDL cholesterol, weight, household income, home ownership, education and survey cycle. Continuous variables are imputed by PMM, binary ones by logistic regression. A pilot run with m = 20 gives a largest fraction of missing information of 0.20, for the cholesterol coefficient; von Hippel's rule for a 5 per cent coefficient of variation then asks for 10 imputations, and the analysis uses 20.

For most coefficients the two approaches agree closely, which is the most common outcome with moderate missingness and good auxiliary variables. Age, BMI, poverty and cholesterol coefficients change by less than one standard error. The poverty coefficient, for instance, is −0.676 mmHg per unit of the income-to-poverty ratio under complete cases and −0.661 under MI, with standard errors of 0.105 and 0.103. Multiple imputation recovers some precision, with standard errors up to 8 per cent smaller than under complete cases (BMI's is unchanged), the return on using the 2,331 participants that complete-case analysis discards.

One coefficient tells a different story. The difference in systolic pressure between Hispanic and White participants, other variables held fixed, is 1.16 mmHg (standard error 0.42) under complete cases and 0.72 mmHg (standard error 0.40) under multiple imputation, and the interval moves from 0.33–1.98 to −0.05–1.50. A conventional reading changes from "significantly higher" to "not distinguishable from zero". The explanation lies in the missingness pattern above. Hispanic participants are the group most likely to have a missing poverty ratio, so complete-case analysis drops them disproportionately, and if those who did not report differ from those who did, the Hispanic participants who remain are a selected subset of their group. Multiple imputation keeps them, under the assumption that their missing poverty ratios resemble those of similar participants who reported. That assumption may or may not hold, but it is more defensible than the one implicit in complete cases, that dropping them leaves the comparison unchanged.

The descriptive quantities move little. Mean systolic pressure is 121.46 mmHg among complete cases and 121.68 under MI, and the share with a reading of 140 or more is 13.8 per cent among complete cases and 14.3 per cent under MI, the latter matching the 14.3 per cent among all participants who have a reading. Here the complete cases are a slightly healthier subset of those with a reading, partly because the complete-case filter also removes people with a missing poverty ratio, who are on average older (49.1 years against 46.9).

## Sensitivity to missingness not at random

Everything in the previous section assumed MAR: that the missing blood-pressure readings and poverty ratios resemble those of observed participants with the same values of everything else. That is an assumption, not a finding, and the honest way to report an analysis that depends on it is to show what happens if it is wrong. The simplest tool is delta adjustment (Carpenter and Kenward, 2013; Cro, Morris, Kenward and Carpenter, 2020): impute under MAR, then shift every imputed value of the variable in question by a fixed amount δ, inside the chained equations so that the shift propagates to the other imputations, and re-run the analysis over a grid of δ values. The shift is a direct statement of an MNAR assumption ("participants who skipped the examination have systolic pressure δ mmHg higher than similar participants who attended"), and the question becomes how large δ would have to be to change the conclusion, a so-called tipping point.

![Figure 5](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/multiple-imputation/figures/fig5_nhanes_delta.png)

*Figure 5. Delta-adjustment sensitivity analysis on NHANES 2009–2012 adults aged 20–79 (unweighted), m = 50 per point. Left: share with systolic pressure of 140 mmHg or more, when every imputed systolic pressure is shifted by δ mmHg. Right: coefficient on the income-to-poverty ratio in the blood-pressure regression, when every imputed poverty ratio is shifted by δ (clipped to the observed range 0–5). Bands are 95 per cent intervals; dashed lines are complete-case estimates.*

The two panels of Figure 5 show the two typical outcomes. The share with high systolic pressure is directly sensitive to assumptions about the missing readings: under MAR it is 14.3 per cent (95 per cent interval 13.6 to 15.0), and it rises to 15.6 per cent if participants without a reading have systolic pressures 10 mmHg higher than similar participants with one, and to 17.7 per cent at 20 mmHg. A shift of 7.5 mmHg is enough to move the estimate (15.1 per cent) outside the MAR interval. Seven and a half per cent of readings are missing, so each mmHg of shift moves the estimated share by a small but steady amount (the mean systolic pressure goes from 121.7 to 123.6 mmHg across the grid), and a δ large enough to matter for a public-health report is not implausible for people who did not attend an examination. Any published prevalence from these data should come with this curve, or at least with the δ at which its conclusion would change.

The regression coefficient on the poverty ratio is much less sensitive to assumptions about missing poverty ratios: shifting every imputed poverty ratio down by 1.5 units, a large change on a scale that runs from 0 to 5, moves the coefficient from −0.66 to −0.52 mmHg per unit, and shifting it up by 1.5 moves it to −0.62. The coefficient stays negative and its interval stays clear of zero across the whole grid. Shifting the missing blood-pressure readings moves it as well, to −0.72 at δ = 10 mmHg and −0.79 at δ = 20, but never toward zero. Two features of the curve deserve a comment. The response is not linear, because the poverty ratio is top-coded at 5 (15.7 per cent of observed values sit at the cap) and bounded at 0, so large shifts push imputed values against the bounds and lose their effect. And a shift that is the same for every imputed case moves a regression slope only to the extent that the imputed cases differ from the observed ones in their outcomes and other covariates; for the rest it moves the intercept, the same mechanism seen in the CPS experiment under MNAR on education. That is why regression coefficients are often more robust to MNAR assumptions than means and prevalences are. The general lesson is the same as in the CPS experiment: what needs a sensitivity analysis depends on the estimand.

Delta adjustment is the simplest member of a family. Pattern-mixture models specify the distribution of the missing values separately for each missingness pattern, of which delta adjustment is a special case; selection models specify the missingness mechanism directly, as in Heckman-type corrections (Heckman, 1979); and in clinical trials, reference-based methods such as "jump to reference" impute a participant's missing outcomes from the control arm. The National Research Council (2010) report on missing data in clinical trials made sensitivity analysis a standard expectation in that field. Outside clinical trials it remains rare, and it is usually the most informative paragraph in any analysis of incomplete data.

## Doing it in R and Python

R has the most mature tooling, and mice is the reference implementation. The NHANES analysis above, written in mice, is a few lines:

```r
library(mice)
ini  <- mice(nhanes_adults, maxit = 0)            # dry run: methods and predictor matrix
meth <- ini$method                                # "pmm" for numeric, "logreg" for binary
imp  <- mice(nhanes_adults, m = 20, maxit = 10, method = meth, seed = 2009)
fit  <- with(imp, lm(sbp ~ age + I((age - 50)^2) + female + bmi + poverty + chol +
                       black + hispanic + otherrace + smoker))
summary(pool(fit), conf.int = TRUE)               # Rubin's rules, Barnard-Rubin df
pool(fit)$pooled[, c("term", "fmi")]              # fraction of missing information
plot(imp)                                         # trace plots: check convergence
```

A delta adjustment uses the `post` argument, which runs after each imputation of a variable:

```r
post <- ini$post
post["sbp"] <- "imp[[j]][, i] <- imp[[j]][, i] + 10"   # MNAR: 10 mmHg higher
imp_mnar <- mice(nhanes_adults, m = 50, post = post, seed = 2009)
```

Around mice there is an ecosystem worth knowing. smcfcs (Bartlett, Seaman, White and Carpenter, 2015) imputes covariates compatibly with a substantive model that contains squares or interactions, the case where ordinary chained equations are biased (Seaman, Bartlett and White, 2012). Amelia (Honaker, King and Blackwell, 2011) implements joint multivariate-normal imputation with a bootstrap EM algorithm and handles time-series cross-section data. jomo does joint modelling for multilevel data. missForest (Stekhoven and Bühlmann, 2012) is a single random-forest imputation; the experiments above show why it belongs in prediction pipelines rather than in inference. Used inside mice (`method = "rf"`), forests do get the between-imputation variation that missForest lacks, but at the default settings they attenuated the return to schooling in every scenario tested here, so they deserve the same scrutiny as any other imputation model.

Python is thinner and needs more care. statsmodels has a chained-equations implementation in which every variable is imputed by predictive mean matching from a linear model on the others (with 20 donors by default; PMM is currently the only method), and which pools a fitted model:

```python
import statsmodels.api as sm
from statsmodels.imputation.mice import MICE, MICEData

imp = MICEData(nhanes_adults)                          # PMM for every incomplete column
model = MICE("sbp ~ age + age2 + female + bmi + poverty + chol + black"
             " + hispanic + otherrace + smoker", sm.OLS, imp)
res = model.fit(n_burnin=10, n_imputations=20)
print(res.summary())
```

scikit-learn's `IterativeImputer` is still marked experimental and must be enabled explicitly. With its defaults it is a single, deterministic regression imputation, which the CPS experiment showed to be the most biased method for inference when the outcome is among the predictors. Its documentation says as much: it returns a single imputation, and the setting intended for multiple imputation is `sample_posterior=True`, run m times with different seeds, with the results pooled by hand:

```python
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
import numpy as np, statsmodels.api as sm

est, var = [], []
for i in range(20):
    Z = IterativeImputer(sample_posterior=True, max_iter=10,
                         random_state=i).fit_transform(X_with_y)
    fit = sm.OLS(Z[:, 0], sm.add_constant(Z[:, 1:])).fit(cov_type="HC1")
    est.append(fit.params); var.append(fit.bse ** 2)
est, var = np.array(est), np.array(var)
qbar, ubar, b = est.mean(0), var.mean(0), est.var(0, ddof=1)
se = np.sqrt(ubar + (1 + 1 / 20) * b)                  # Rubin's rules
```

`sample_posterior=True` draws imputations from the predictive distribution of a `BayesianRidge` model, which is close to mice's `norm` method; it does not do predictive mean matching, and it is only as good as a linear-normal model for each variable. The miceforest package runs chained equations with LightGBM models and mean matching, the closest Python analogue of tree-based imputation in mice, and it deserves the same checks against a parametric model. The implementation written for this article is about 200 lines of NumPy and is in the repository linked below; it was checked against statsmodels on the NHANES regression, where the pooled coefficients of the two agree within Monte Carlo error.

For prediction pipelines, the scikit-learn tools are the right ones: `SimpleImputer(add_indicator=True)` for linear models, and nothing at all for `HistGradientBoostingRegressor` and `HistGradientBoostingClassifier`, which learn a direction for missing values at every split. The caveat from the deployment experiment applies to both: a model that has learned what missingness means has made the missingness process part of its input, and it will need to be monitored like any other input.

## A working order for incomplete data

The experiments suggest an order of work that avoids most of the failures documented above.

Start by describing the missingness before touching it. Count missing values per variable and, more importantly, the patterns in which they occur together; patterns usually map onto the data-collection process (an examination not attended, a questionnaire section skipped, a sensor offline), and the process is the best evidence available about the mechanism. Model the missingness indicators on observed variables to see what predicts them. This rules out MCAR when it is false and identifies the auxiliary variables that belong in the imputation model.

Write down the estimand before choosing the method. A regression slope, a mean, a prevalence and a prediction respond differently to the same missing data, and complete-case analysis can be the right answer for one and the wrong answer for another on the same data set.

If the goal is inference, impute multiply and properly, never singly. Include the outcome and all analysis variables in the imputation model, add auxiliary variables that predict the missing values or their missingness, and prefer predictive mean matching with an imputation model that includes the curvature and interactions the data may contain (at least the square of the outcome and its interactions with the other covariates) to a plain linear-normal model. Check any flexible engine, such as random forests inside chained equations, against that parametric model before relying on it. When the analysis model contains squares or interactions of an incomplete variable, use a method designed for it, such as substantive-model-compatible imputation (Bartlett, Seaman, White and Carpenter, 2015), rather than imputing the variable and its square independently (Seaman, Bartlett and White, 2012; von Hippel, 2009).

Choose m from the fraction of missing information, not from habit: run a pilot, estimate the FMI, and use von Hippel's rule. Report the FMI for the main parameters alongside the estimates, together with trace plots or another convergence check for the chained equations.

Run complete-case analysis as well, and treat a large disagreement between it and multiple imputation as information about the mechanism or about model misspecification, to be investigated rather than averaged away.

Finish with a sensitivity analysis for the assumptions that cannot be tested. Delta adjustment over a plausible range, with the tipping point reported, costs a loop around the imputation call.

If the goal is prediction, treat missingness as a feature: missing-value indicators or native handling in tree ensembles, with imputation models that never use the target. Then monitor the missingness rate and pattern of every feature in production, because the model's calibration depends on them.

For anyone who wants more than a working order, the book-length treatments are Little and Rubin (2019), van Buuren (2018) and Carpenter and Kenward (2013); Schafer and Graham (2002), Sterne, White, Carlin, Spratt, Royston, Kenward, Wood and Carpenter (2009) and Carpenter and Smuk (2021) are shorter guides to practice.

## What imputation can and cannot do

Multiple imputation addresses a specific data challenge: incomplete rows that contain information an analysis cannot afford to throw away. Done properly, it uses all of that information, it carries the uncertainty about the missing values into the standard errors, and under MAR with a good imputation model it gives valid inference where deletion is biased. In the CPS experiment with the analysis model correctly specified, the standard multiple imputations (Bayesian regression and PMM) had biases within a few per cent and close to nominal coverage under every mechanism, while complete cases failed when missingness depended on wages and every single imputation reported intervals that were too narrow. In the NHANES analysis it kept 2,331 participants that complete cases discarded, sharpened most standard errors and changed one substantive conclusion, about a group that was disproportionately among the incomplete.

It cannot create information that the data do not contain. Under MNAR it is as biased as the MAR assumption is wrong, as the mean of education showed, where every method missed the truth by about a year in every sample. It is only as good as its imputation model, as the real-wage experiment showed, where a linear imputation model turned an unbiased method into one with 23 per cent bias, a richer one brought most of that back, and a default random forest traded the overshoot for attenuation. Its standard errors are noisy when m is small, which makes results depend on a random seed. And it answers questions about parameters, not about predictions, which have their own rules and their own failure mode when the missingness process changes.

Three conjectures follow from the experiments and are stated as conjectures, since one population and one survey cannot establish them. The first is that published rankings of imputation methods by masked-cell RMSE systematically favour the methods that are worst for inference, because the conditional mean minimises that error and distorts every covariance; a benchmark that scored methods on the coverage of downstream estimates would likely reverse much of their order. The second is that leaving the outcome out of the imputation model is the most common serious error in applied work, because pipelines that impute as a preprocessing step, before the outcome is in view, make it the path of least resistance, and the 40 to 49 per cent attenuation seen here suggests the error is not small. The third is that, in production prediction systems, a meaningful share of unexplained calibration drift is caused by changes in missingness patterns rather than in the distribution of observed values, and goes undetected because monitoring tracks feature distributions and not their missingness. Each is testable with the code below.

## Code and data

All code is in the `multiple-imputation` folder of the Medium-Blog repository: https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/multiple-imputation. It contains the chained-equations implementation and Rubin's rules used throughout (`mi.py`), checks against hand-computed pooling and against statsmodels, one script per experiment, and the text output from which every number in this article was taken. The CPS 1988 data are read from the Rdatasets mirror of the AER package; the NHANES data from the repository of the NHANES R package.

## References

Barnard, J. and Rubin, D. B. (1999). Small-sample degrees of freedom with multiple imputation. *Biometrika*, 86(4), 948–955.

Bartlett, J. W., Seaman, S. R., White, I. R. and Carpenter, J. R. (2015). Multiple imputation of covariates by fully conditional specification: Accommodating the substantive model. *Statistical Methods in Medical Research*, 24(4), 462–487.

Bierens, H. J. and Ginther, D. K. (2001). Integrated conditional moment testing of quantile regression models. *Empirical Economics*, 26(1), 307–324.

Bodner, T. E. (2008). What improves with increased missing data imputations? *Structural Equation Modeling*, 15(4), 651–675.

Carpenter, J. R. and Kenward, M. G. (2013). *Multiple Imputation and its Application*. Wiley.

Carpenter, J. R. and Smuk, M. (2021). Missing data: A statistical framework for practice. *Biometrical Journal*, 63(5), 915–947.

Cro, S., Morris, T. P., Kenward, M. G. and Carpenter, J. R. (2020). Sensitivity analysis for clinical trials with missing continuous outcome data using controlled multiple imputation: A practical guide. *Statistics in Medicine*, 39(21), 2815–2842.

Doove, L. L., van Buuren, S. and Dusseldorp, E. (2014). Recursive partitioning for missing data imputation in the presence of interaction effects. *Computational Statistics & Data Analysis*, 72, 92–104.

Graham, J. W., Olchowski, A. E. and Gilreath, T. D. (2007). How many imputations are really needed? Some practical clarifications of multiple imputation theory. *Prevention Science*, 8(3), 206–213.

Groenwold, R. H. H., White, I. R., Donders, A. R. T., Carpenter, J. R., Altman, D. G. and Moons, K. G. M. (2012). Missing covariate data in clinical research: When and when not to use the missing-indicator method for analysis. *CMAJ*, 184(11), 1265–1269.

Heckman, J. J. (1979). Sample selection bias as a specification error. *Econometrica*, 47(1), 153–161.

Honaker, J., King, G. and Blackwell, M. (2011). Amelia II: A program for missing data. *Journal of Statistical Software*, 45(7), 1–47.

Hughes, R. A., Heron, J., Sterne, J. A. C. and Tilling, K. (2019). Accounting for missing data in statistical analyses: Multiple imputation is not always the answer. *International Journal of Epidemiology*, 48(4), 1294–1304.

Jones, M. P. (1996). Indicator and stratification methods for missing explanatory variables in multiple linear regression. *Journal of the American Statistical Association*, 91(433), 222–230.

Josse, J., Chen, J. M., Prost, N., Varoquaux, G. and Scornet, E. (2024). On the consistency of supervised learning with missing values. *Statistical Papers*, 65(9), 5447–5479.

Kleiber, C. and Zeileis, A. (2008). *Applied Econometrics with R*. Springer.

Le Morvan, M., Josse, J., Scornet, E. and Varoquaux, G. (2021). What's a good imputation to predict with missing values? *Advances in Neural Information Processing Systems*, 34, 11530–11540.

Lee, K. J. and Carlin, J. B. (2010). Multiple imputation for missing data: Fully conditional specification versus multivariate normal imputation. *American Journal of Epidemiology*, 171(5), 624–632.

Little, R. J. A. (1988a). Missing-data adjustments in large surveys. *Journal of Business & Economic Statistics*, 6(3), 287–296.

Little, R. J. A. (1988b). A test of missing completely at random for multivariate data with missing values. *Journal of the American Statistical Association*, 83(404), 1198–1202.

Little, R. J. A. (1992). Regression with missing X's: A review. *Journal of the American Statistical Association*, 87(420), 1227–1237.

Little, R. J. A. and Rubin, D. B. (2019). *Statistical Analysis with Missing Data* (3rd ed.). Wiley.

Meng, X.-L. (1994). Multiple-imputation inferences with uncongenial sources of input. *Statistical Science*, 9(4), 538–558.

Mincer, J. (1974). *Schooling, Experience, and Earnings*. National Bureau of Economic Research.

Moons, K. G. M., Donders, R. A. R. T., Stijnen, T. and Harrell, F. E. (2006). Using the outcome for imputation of missing predictor values was preferred. *Journal of Clinical Epidemiology*, 59(10), 1092–1101.

Morris, T. P., White, I. R. and Royston, P. (2014). Tuning multiple imputation by predictive mean matching and local residual draws. *BMC Medical Research Methodology*, 14, 75.

National Research Council (2010). *The Prevention and Treatment of Missing Data in Clinical Trials*. National Academies Press.

Perez-Lebel, A., Varoquaux, G., Le Morvan, M., Josse, J. and Poline, J.-B. (2022). Benchmarking missing-values approaches for predictive models on health databases. *GigaScience*, 11, giac013.

Pruim, R. (2026). NHANES: Data from the US National Health and Nutrition Examination Study. R package version 2.1.4. https://cran.r-project.org/package=NHANES

Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3), 581–592.

Rubin, D. B. (1986). Statistical matching using file concatenation with adjusted weights and multiple imputations. *Journal of Business & Economic Statistics*, 4(1), 87–94.

Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys*. Wiley.

Schafer, J. L. and Graham, J. W. (2002). Missing data: Our view of the state of the art. *Psychological Methods*, 7(2), 147–177.

Seaman, S. R., Bartlett, J. W. and White, I. R. (2012). Multiple imputation of missing covariates with non-linear effects and interactions: An evaluation of statistical methods. *BMC Medical Research Methodology*, 12, 46.

Shah, A. D., Bartlett, J. W., Carpenter, J., Nicholas, O. and Hemingway, H. (2014). Comparison of random forest and parametric imputation models for imputing missing data using MICE: A CALIBER study. *American Journal of Epidemiology*, 179(6), 764–774.

Stekhoven, D. J. and Bühlmann, P. (2012). MissForest—non-parametric missing value imputation for mixed-type data. *Bioinformatics*, 28(1), 112–118.

Sterne, J. A. C., White, I. R., Carlin, J. B., Spratt, M., Royston, P., Kenward, M. G., Wood, A. M. and Carpenter, J. R. (2009). Multiple imputation for missing data in epidemiological and clinical research: Potential and pitfalls. *BMJ*, 338, b2393.

Twala, B. E. T. H., Jones, M. C. and Hand, D. J. (2008). Good methods for coping with missing data in decision trees. *Pattern Recognition Letters*, 29(7), 950–956.

van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.). Chapman & Hall/CRC. Online edition: https://stefvanbuuren.name/fimd/

van Buuren, S. and Groothuis-Oudshoorn, K. (2011). mice: Multivariate imputation by chained equations in R. *Journal of Statistical Software*, 45(3), 1–67.

von Hippel, P. T. (2009). How to impute interactions, squares, and other transformed variables. *Sociological Methodology*, 39(1), 265–291.

von Hippel, P. T. (2020). How many imputations do you need? A two-stage calculation using a quadratic rule. *Sociological Methods & Research*, 49(3), 699–718.

White, I. R. and Carlin, J. B. (2010). Bias and efficiency of multiple imputation compared with complete-case analysis for missing covariate values. *Statistics in Medicine*, 29(28), 2920–2931.

White, I. R., Royston, P. and Wood, A. M. (2011). Multiple imputation using chained equations: Issues and guidance for practice. *Statistics in Medicine*, 30(4), 377–399.

White, I. R. and Thompson, S. G. (2005). Adjusting for partially missing baseline measurements in randomized trials. *Statistics in Medicine*, 24(7), 993–1007.
