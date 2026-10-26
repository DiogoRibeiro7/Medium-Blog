# Robust Regression Under Data-Entry Errors: Breakdown, Efficiency and the Outliers That Are the Signal

## What M, S, MM and least-trimmed-squares estimators protect against, what they cost on clean data, and when downweighting unusual observations gives the wrong answer

A diamond recorded at 20 carats instead of 2, because someone typed the decimal point one place to the right. A price entered as 33,260 dollars instead of 3,326, because a stray zero was typed at the end. A column of depths filled with the carat weights, because a spreadsheet was pasted one column off. None of these errors is exotic, and in a data set of tens of thousands of rows a handful of them is close to certain. Least squares treats each of them as information, and some of them carry enough weight to bend a regression line on their own.

The usual defence is a rule: compute the residuals, drop anything more than three standard deviations away, or drop anything with a large Cook's distance, and refit. The rule is popular because it is simple and because it seems to work on the textbook example with one bad point. It fails in a specific and predictable way when the bad points are numerous or come in groups, because the fit it relies on to find them has already been pulled toward them. The statistical answer to this data challenge is robust regression: estimators that limit how much any observation can influence the fit, and the best of which can tolerate up to half the data being arbitrarily wrong.

This guide covers what robust regression estimates, how the main estimators work (M, S, MM and least trimmed squares), what robustness costs on clean data, and when it is the wrong tool, because the unusual observations are not errors but the thing the analysis is about. Each claim is checked against data. The core experiment injects realistic data-entry errors into samples drawn from the diamonds in the ggplot2 data set and compares thirteen estimators against a known answer. Two classic data sets with known anomalies show how deletion rules fail. The full diamonds data turn up their own impossible rows. And 4,624 real motor-insurance claims show what happens when a robust estimator meets a heavy-tailed outcome whose large values are real money. The M, S, MM and LTS estimators were written for the article; the MM-estimator reproduces the coefficients of R's robustbase package to four decimals, and the code is linked at the end.

## Two old data sets that still make the point

Rousseeuw and Leroy's (1987) book on robust regression used a short series that has become the standard first example: the number of international telephone calls made from Belgium each year from 1950 to 1973, in millions. The series grows steadily, except that from 1964 to 1969 the source recorded a different quantity, the total number of minutes of calls, so those six values are several times too large. The left panel of Figure 1 shows the result. The least-squares slope is 5.04 million calls per year, more than four times the slope of the MM fit described below. Huber's M-estimator, the robust method most statistical software offers first, gives 2.04; median regression gives 1.56. The MM-estimator gives 1.10, and least trimmed squares (LTS) gives 1.16, close to the growth of the correctly recorded years.

![Figure 1](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/robust-regression/figures/fig1_classic_cases.png)

*Figure 1. Left: Belgian international telephone calls, 1950–1973 (MASS::phones); the orange points are the years 1964–1969, when the source recorded minutes instead of calls. Middle: the Hertzsprung–Russell diagram of the star cluster CYG OB1 (robustbase::starsCYG); the orange points are four red giants. Right: outlier map for the stars, the robust distance of each star's temperature from the bulk (minimum covariance determinant) against its standardised MM residual.*

The usual diagnostics do not find the problem. Externally studentised residuals from the least-squares fit flag one year at a threshold of 2.5, 1969. Cook's distance with the common cutoff of 4/n flags 1969, 1972 and 1973, so two of its three flags are correctly recorded years, pulled away from the distorted line. The six bad years have inflated the residual scale and tilted the line toward themselves, which makes the residuals that are supposed to expose them look ordinary. This is masking, and the false alarms on good years are swamping (Rousseeuw and van Zomeren, 1990). The MM residuals, measured against a fit and a scale that the bad years cannot move, exceed 2.5 robust standard deviations for the years 1964 to 1970 and for no others; Rousseeuw and Leroy note that 1963 and 1970 were partly affected too, because the change of recording did not coincide with the calendar year. A bisquare M-estimator started from least squares also gets this one right (slope 1.10), because the bad years are vertical outliers and the least-squares start, though distorted, is close enough for the iterations to escape.

The second classic is astronomical. The star cluster CYG OB1 has 47 stars measured on two variables, log surface temperature and log light intensity; 43 lie on the main sequence and four are red giants, cool and very bright, a different population (observations 11, 20, 30 and 34; Rousseeuw and Leroy, 1987). Plotted with temperature decreasing from left to right, as astronomers do, the main sequence runs from upper left to lower right and the giants sit in the top right corner. They are bad leverage points: unusual in the explanatory variable and far from the line the rest of the data follow. The middle panel shows what they do. The least-squares slope is −0.41, so that hotter stars appear dimmer, the opposite of the main-sequence relationship. Huber's M-estimator gives −0.43 and median regression −0.69, no better. M-estimators bound the influence of a large residual, but a leverage point does not have a large residual under a fit it has already captured, so they cannot resist it. A bisquare M-estimator started from least squares gives −0.42: this time the start is on the wrong side and the iterations never leave it. The MM-estimator gives a slope of 2.25 and LTS gives 4.22, both following the main sequence. Least-squares diagnostics again do poorly: no studentised residual exceeds 2.5, and Cook's distance flags three of the giants plus a main-sequence star (14) while missing the fourth giant (11). The MM residuals exceed 2.5 robust standard deviations for exactly the four giants.

The right panel shows the diagnostic that robust fits make possible, the outlier map of Rousseeuw and van Zomeren (1990). The horizontal axis measures how unusual each observation's explanatory variables are, by a robust distance computed from the minimum covariance determinant (Rousseeuw and Van Driessen, 1999); the vertical axis is the standardised residual from the robust fit. Points to the right and far from zero vertically are bad leverage points, the ones that destroy least squares. Points to the right but near zero are good leverage points, unusual in x but consistent with the model, and they improve precision. Points near the left with large residuals are vertical outliers. The four giants sit alone in the top right corner.

## Influence, breakdown and efficiency

Three ideas organise everything that follows.

The influence function, introduced by Hampel (1974) and developed in the book by Hampel, Ronchetti, Rousseeuw and Stahel (1986), measures how much an estimator changes when a small amount of contamination is placed at a point. For least squares it is unbounded in both the residual and the explanatory variables: an observation that is far enough away, in either direction, can move the estimate as much as it likes. Bounding it is the first goal of robust estimation.

The breakdown point is the smallest fraction of the data that, replaced by arbitrary values, can drive the estimate arbitrarily far (Hampel, 1971; Donoho and Huber, 1983). Least squares breaks down with a single bad observation, a breakdown point of 1/n, which tends to zero. So does median regression, and so do M-estimators in regression, because their protection covers large residuals but not leverage: the telephone data and the stars showed both failures. The highest breakdown point a regression-equivariant estimator can have is about 50 per cent; beyond that, the bad data are the majority and no method can tell which group is the model.

Efficiency is the price. On clean data with normal errors, least squares is the most efficient unbiased estimator, and every robust estimator has a larger sampling variance. The asymptotic efficiency relative to least squares is 64 per cent for median regression (2/π), 95 per cent for Huber's M-estimator with its usual tuning constant, and as low as 28.7 per cent for the high-breakdown S-estimator and about 7 per cent for LTS. The field began with Tukey's (1960) demonstration that a few per cent of contamination is enough to make the sample mean and standard deviation far less efficient than simple alternatives, and its history since, told by Stigler (2010), is largely a search for estimators that are both highly robust and highly efficient. The MM-estimator was its most successful answer for regression; Huber and Ronchetti (2009) give the theory.

### M-estimators

Huber (1964, 1973) proposed replacing the squared residual in least squares by a function that grows more slowly. The M-estimator minimises the sum of ρ(r_i / s) over observations, where r_i is the residual and s is a scale estimate, and solves

```
Σ ψ(r_i / s) x_i = 0,          ψ = ρ'
```

Least squares is ρ(u) = u²/2 and ψ(u) = u, so every residual contributes in proportion to its size. Huber's choice is quadratic near zero and linear beyond a constant c, so ψ is capped. Tukey's bisquare (Beaton and Tukey, 1974) goes further: ψ rises, then returns to zero, so that residuals beyond c receive no weight at all. Figure 2 shows the two functions and the weights they imply.

![Figure 2](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/robust-regression/figures/fig2_psi_weights.png)

*Figure 2. Left: the ψ functions of least squares, Huber (c = 1.345) and the bisquare (c = 4.685). Right: the corresponding weights ψ(u)/u used in iteratively reweighted least squares. Both constants give 95 per cent efficiency under normal errors.*

The estimating equation can be read as weighted least squares with weights w_i = ψ(u_i)/u_i, which suggests the standard algorithm: start from some fit, compute weights from the residuals, refit by weighted least squares, and repeat. This is iteratively reweighted least squares (IRLS), and it is what statsmodels' `RLM`, R's `MASS::rlm` and most implementations do. Two details matter in practice. The scale s has to be estimated robustly, usually by the median absolute deviation of the residuals, because the residual standard deviation is itself destroyed by outliers. And with the bisquare, the objective is not convex: it has several local minima, and IRLS finds the one nearest its starting point. Started from least squares, a bisquare M-estimator inherits whatever distortion least squares has, and can converge to the wrong solution with full confidence. That is the reason for the next two estimators.

### High-breakdown estimators: LTS and S

Rousseeuw (1984) proposed the first regression estimators with a 50 per cent breakdown point. Least median of squares minimises the median of the squared residuals; least trimmed squares minimises the sum of the h smallest squared residuals, with h about half the sample, so that the fit is determined by the best-fitting half of the data whichever half that is. The S-estimator of Rousseeuw and Yohai (1984) minimises a robust scale of the residuals, the M-scale s that solves

```
(1 / (n − p)) Σ ρ(r_i / s) = b
```

With the bisquare ρ, a tuning constant of 1.548 and b = 0.5, the S-estimator has a 50 per cent breakdown point.

None of these objectives can be minimised by a smooth algorithm, because they have many local minima. The practical solutions draw many random subsamples of p observations, fit each exactly, refine the best candidates by a few local steps, and keep the overall best: FAST-LTS for least trimmed squares (Rousseeuw and Van Driessen, 2006) and FAST-S for S-estimators (Salibian-Barrera and Yohai, 2006). With enough subsamples, at least one is likely to contain only good observations, and its fit then leads to the solution that ignores the bad ones. R's robustbase uses 500 subsamples by default, as does the code here. With many dummy variables, random subsamples are often singular, a problem Koller and Stahel (2017) addressed with a nonsingular subsampling scheme that robustbase now offers.

The cost of high breakdown is efficiency: the S-estimator's is 28.7 per cent under normal errors, by the formula (E ψ')² / E ψ², and LTS's is lower still.

### MM-estimators

Yohai's (1987) MM-estimator combines the two properties. It starts from a high-breakdown S-estimate, keeps the S-scale fixed, and then runs a bisquare M-estimation from that starting point with a larger tuning constant (4.685, for 95 per cent efficiency). Because the starting point already ignores the bad observations, IRLS converges to the local minimum near the correct fit, and because the final step uses the larger constant, the estimator recovers most of the efficiency of least squares. The result has the 50 per cent breakdown point of the S-estimator and, under normal errors, 95 per cent efficiency. It is the default of `lmrob` in robustbase, and the estimator most robust-statistics texts now recommend as a starting point (Maronna, Martin, Yohai and Salibián-Barrera, 2019).

The implementation written for this article follows lmrob's defaults: bisquare ρ throughout, a 50 per cent breakdown S-estimate computed by FAST-S with 500 subsamples, and a 95 per cent efficient MM step. On the two data sets that robustbase uses in its own regression tests it reproduces the published results: for the stars, coefficients −4.9694 and 2.2532 with a robust scale of 0.4715; for Brownlee's stack-loss data, coefficients −41.5246, 0.9388, 0.5796 and −0.1129 with a scale of 1.91. Standard errors use the sandwich formula for M-estimators evaluated at the final fit, s² A⁻¹BA⁻¹/(n − p), with A the mean of ψ′(u) x xᵀ and B the mean of ψ(u)² x xᵀ over observations, in which observations with zero weight drop out of both terms. For the stars this gives standard errors of 3.35 and 0.756, against robustbase's 3.41 and 0.769, which additionally accounts for the estimation of the scale (Croux, Dhaene and Hoorelbeke, 2003; Koller and Stahel, 2011).

### And the rest of the toolbox

Data scientists more often meet robust regression through scikit-learn, which offers three estimators with different logic. `HuberRegressor` is a Huber M-estimator that estimates the scale jointly with the coefficients (Owen, 2007), with a small ridge penalty by default. `TheilSenRegressor` generalises the Theil (1950) and Sen (1968) estimator, the median of pairwise slopes, to several variables through a spatial median of subsample fits (Dang, Peng, Wang and Zhang, 2009); its breakdown point falls as the number of variables grows. `RANSACRegressor` implements random sample consensus (Fischler and Bolles, 1981), an algorithm from computer vision: fit random minimal subsets, count the observations within a residual threshold of each fit, keep the fit with the largest consensus set, and refit by least squares on its inliers. And median regression (least absolute deviations), the 0.5 quantile regression of Koenker and Bassett (1978), is often described as robust, which is true for vertical outliers and false for leverage points.

## An experiment with known errors

The core experiment uses real covariates and a known answer. The population is the 53,837 diamonds in ggplot2's data set (Wickham, 2016) whose recorded dimensions imply a density between 0.0055 and 0.0068 carats per cubic millimetre, the narrow band real diamonds occupy, so that the population itself contains none of the recording errors discussed later. For each diamond the covariates are its log carat weight and its cut, colour and clarity grades coded as ordered scores. Log price is regenerated from the fitted regression of real log price on those covariates, with normal errors of the same standard deviation (0.146), so that the regression model is exactly right and its coefficients are known. The coefficient of interest is the one on log carat, the elasticity of price with respect to weight, which is 1.878 in the population: a diamond 10 per cent heavier is, other grades equal, about 20 per cent more expensive.

Each replication draws 500 diamonds without replacement and corrupts a share of them, from 1 to 40 per cent, in one of two realistic ways. In the first, the price is typed with an extra zero, which multiplies it by ten: a vertical outlier, a large error in the response at an ordinary value of the covariates. In the second, the carat weight is recorded with the decimal point one place to the right, while the price stays correct: a bad leverage point, an error in a covariate that moves the observation far along the x axis and off the line. A third set of scenarios corrupts nothing but draws the errors from a t distribution with 5, 3, 2 or 1 degrees of freedom, scaled to the same interquartile range as the normal errors, to measure the efficiency cost and benefit of robust estimators when the errors are merely heavy-tailed. Each scenario is repeated 300 times; the Monte Carlo standard error of a coverage estimate near 95 per cent is about 1.3 percentage points.

Thirteen estimators are compared. Least squares with heteroscedasticity-robust standard errors is the baseline. Three deletion rules refit least squares after removing observations: once, those with an externally studentised residual above 3 in absolute value (Belsley, Kuh and Welsch, 1980); repeatedly, until no such residual remains; and once, those with Cook's (1977) distance above 4/n. Then come median regression (statsmodels `QuantReg`), Huber's M-estimator (statsmodels `RLM`), scikit-learn's `HuberRegressor`, a bisquare M-estimator started from least squares (statsmodels `RLM` with Tukey's biweight), scikit-learn's Theil–Sen and RANSAC with their defaults, and the three high-breakdown estimators, LTS, S and MM.

### Errors in the price cost precision, not accuracy

The first surprise is that prices typed with an extra zero do not bias the slope of any estimator. Over 300 samples, every one of the thirteen methods has a mean log-carat coefficient within 1 per cent of the truth at every contamination level from 1 to 40 per cent. The reason is that the corrupted rows are chosen at random, so the shift of log(10) in their log price is unrelated to their carat weight and is absorbed mostly by the intercept. A vertical outlier at an ordinary value of the covariates pulls the fitted line up or down far more than it rotates it.

What the errors do destroy is precision, and the left panel of Figure 3 shows how much. The root mean squared error (RMSE) of the least-squares slope is 0.0124 on clean data, 0.0227 with 1 per cent of prices corrupted, 0.0446 at 5 per cent and 0.107 at 40 per cent: five bad prices in 500 nearly double the error, and at 40 per cent it is almost nine times larger. Least squares is at least honest about it. Its heteroscedasticity-robust standard errors grow with the damage, and its 95 per cent confidence intervals cover the true slope in 92 to 97 per cent of samples at every level. The errors are noise, and least squares reports them as noise.

The MM-estimator barely notices them. Its RMSE runs from 0.0123 on clean data to 0.0130 at 10 per cent and 0.0162 at 40 per cent, and its coverage stays between 93.3 and 95.3 per cent at every contamination level. At 40 per cent contamination it is almost seven times as precise as least squares.

![Figure 3](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/robust-regression/figures/fig3_bias_curves.png)

*Figure 3. Left: RMSE of the log-carat coefficient (log scale) when a share of prices is multiplied by ten. Right: relative bias of the same coefficient when a share of carat weights is multiplied by ten and prices are left correct. Each point is 300 samples of 500 diamonds; the share 0 is the clean-data scenario. S, scikit-learn's HuberRegressor and the two studentised-residual rules are omitted for legibility; S's bias is indistinguishable from MM's.*

The other methods share a pattern that is worth recognising, because it is how most of them fail in the carat scenario too: they work perfectly up to a threshold and then collapse to least squares or worse. Deleting observations with an absolute studentised residual above 3 works up to 5 per cent of corrupted prices (RMSE 0.0126) and fails at 10 per cent (0.0755, with coverage of 82 per cent). Applying the same rule repeatedly works up to 10 per cent and at 20 per cent gives exactly the least-squares answer, because the first pass finds nothing to delete. Deleting observations with Cook's distance above 4/n works reasonably up to 10 per cent and at 20 per cent is worse than doing nothing (RMSE 0.0881 against 0.0796 for least squares, coverage 85 per cent), because it removes a non-random 40 per cent of the bad observations, those its distorted fit happens to expose, and keeps the rest.

Huber's M-estimator and the bisquare M-estimator started from least squares, both from statsmodels, survive further, up to 20 per cent, and then both collapse at 30 per cent, with RMSEs of 0.116 and 0.109. Their failure has a specific mechanism, and it is not the ψ function but the scale. Both estimate the residual scale by the median absolute deviation (MAD) of the residuals from the current fit. When 30 or 40 per cent of the residuals are shifted by 2.3 on the log scale, the MAD itself becomes so large that few or no residuals exceed the tuning constant times the scale, the bad observations keep most or all of their weight, and the estimator becomes as imprecise as least squares, or more so. At 40 per cent the statsmodels Huber estimate is the least-squares estimate in 298 of the 300 samples. scikit-learn's `HuberRegressor`, which estimates the scale jointly with the coefficients, holds on a little longer (RMSE 0.0354 at 30 per cent) and fails at 40 per cent. Median regression degrades gradually, from 0.0146 on clean data to 0.0511 at 40 per cent, never failing completely and never matching MM. RANSAC works up to 20 per cent, loses precision at 30 and fails at 40, because its default inlier threshold is the MAD of the response, which the errors inflate. Theil–Sen degrades steadily from the start.

S and LTS behave in the opposite direction, and the reason illuminates what high breakdown means. Both are imprecise on clean data, with RMSEs of 0.0223 and 0.0370, and both become more precise than on clean data once the contamination passes 10 per cent: S has an RMSE of 0.0169 and LTS 0.0235 at 40 per cent. LTS fits the best half of the sample. With no errors, that half is a trimmed selection of the good observations that discards information; with 40 per cent errors, the best half is drawn from the 60 per cent of good observations and keeps more than four-fifths of them. The S-scale behaves in the same way, because the bad observations inflate it, which makes the effective tuning constant on the good observations larger and the estimator closer to least squares on them. High-breakdown estimators are built for the worst case, and they are at their best when it arrives.

### Errors in the carat weight cost the answer

Errors in a covariate are a different matter, as the right panel of Figure 3 shows. When the decimal point slips in a carat weight, the diamond moves 2.3 units to the right on the log-carat axis, far outside the range of the good data, while its price stays where a diamond of its true size would be. It is a bad leverage point, and a few of them are enough to rotate the least-squares line.

With 1 per cent of carats corrupted, five diamonds in a sample of 500, the least-squares elasticity falls from 1.88 to a median of 1.57, a bias of 16.6 per cent. At 2 per cent the bias is 28 per cent, at 5 per cent 49 per cent, and at 40 per cent 83 per cent, an estimated elasticity of 0.32 where the truth is 1.88. The confidence interval never contains the true value, in any sample, at any level: least squares is wrong and confident.

The estimators that protect against vertical outliers do not protect against this. Huber's M-estimator is biased by 1.2 per cent at 1 per cent contamination, 6.7 per cent at 5 per cent and 33 per cent at 10 per cent; median regression by 0.9, 4.7 and 12 per cent; scikit-learn's Huber by 1.0, 5.6 and 16.6 per cent. Those first numbers look small, but they are large relative to the precision of the estimates: with 2 per cent of carats corrupted, Huber's 95 per cent confidence interval covers the true elasticity in 4 per cent of samples, and median regression's in 38 per cent. At 20 per cent contamination all three are as wrong as least squares. Theil–Sen, often recommended as a robust default, is biased by 10 per cent at 10 per cent contamination and by 34 per cent at 20 per cent.

The deletion rules hold for a while and then fail completely. Deleting large studentised residuals once works up to 2 per cent and is biased by 33 per cent at 5 per cent; repeatedly, up to 5 per cent and biased by 47 per cent at 10 per cent; Cook's distance, up to 5 per cent and biased by 42 per cent at 10 per cent. From 20 per cent on, all three are about as wrong as least squares.

The bisquare M-estimator started from least squares is the most instructive failure. Up to 10 per cent contamination its point estimate is essentially unbiased (−0.1 per cent), because the least-squares start, though badly distorted, is still in the basin of the right solution, and the bisquare's redescending ψ then removes the bad points entirely. At 20 per cent the start is in the wrong basin, the iterations converge confidently to an answer close to least squares', and the bias is 75 per cent. Even where its point estimate is right, its standard errors are not: the coverage of its confidence interval is 90 per cent at 1 per cent contamination, 84 per cent at 5 per cent and 76 per cent at 10 per cent. statsmodels' default covariance for `RLM`, Huber's H1 formula, multiplies a scalar factor by the inverse of X′X computed from all observations, including the bad leverage points, which spread the covariate and make X′X larger. At 10 per cent contamination the average reported standard error is 0.0081 while the actual standard deviation of the estimates is 0.0134. The sandwich formula used for MM here, in which zero-weight observations drop out of both terms, does not have that problem.

RANSAC survives up to 30 per cent and is biased by 14 per cent at 40 per cent. LTS, S and MM are unbiased, to within about 0.2 per cent, at every level up to 40 per cent, and MM's confidence intervals cover the truth in between 93.3 and 96.7 per cent of samples at every level. The MM-estimator's RMSE with 40 per cent of carats corrupted, 0.0164, is a third larger than its RMSE on clean data and a hundredth of least squares'.

### Masking, in numbers

Figure 4 shows the reason the deletion rules fail. It plots the share of the corrupted rows that each rule flags, and the pattern is the same in both scenarios: perfect detection while the errors are few, then a sharp drop to almost nothing.

![Figure 4](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/robust-regression/figures/fig4_detection.png)

*Figure 4. Share of corrupted rows flagged by three least-squares deletion rules and by the MM-estimator (absolute residual above 3 robust scale units), by contamination level. 300 samples of 500 diamonds per point.*

The single |t| > 3 rule flags every corrupted price up to 5 per cent contamination, 41 per cent of them at 10 per cent, and none at 20 per cent or more. For corrupted carats the collapse comes sooner: all of them at 2 per cent, 54 per cent at 5 per cent, 6.6 per cent at 10 per cent and 0.3 per cent at 20 per cent. Repeating the rule extends its range by one step and no more. Cook's distance degrades more gently but has a cost even on clean data: with no errors at all, the 4/n cutoff flags 5.1 per cent of the observations, about 25 good diamonds out of every 500, and refitting without them increases the RMSE of the slope by 9 per cent and lowers the coverage of the least-squares interval to 91 per cent. That is swamping, good observations condemned by a rule that cannot tell them from bad ones.

The MM-estimator flags 100 per cent of the corrupted rows at every level in both scenarios. On clean data it flags 0.31 per cent of the observations, against 0.27 per cent expected from a normal distribution beyond three standard deviations, and with contamination its false-flag rate falls further, below 0.1 per cent at 10 per cent contamination. It does not suffer from masking for a simple reason: the fit and the scale from which its residuals are measured come from an S-estimate that the errors cannot move.

### What robustness costs, and what heavy tails return

The remaining scenarios have no errors at all. Figure 5 shows the RMSE of eight of the estimators relative to least squares when the errors are normal, and when they follow t distributions with 5, 3 and 2 degrees of freedom, matched to the normal on the interquartile range.

![Figure 5](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/robust-regression/figures/fig5_efficiency.png)

*Figure 5. RMSE of the log-carat coefficient relative to least squares (log scale), for normal errors and for t errors with 5, 3 and 2 degrees of freedom scaled to the same interquartile range. Below the dashed line, the estimator is more accurate than least squares. Cauchy errors (one degree of freedom) are left out because least squares' RMSE is about 1,250 times MM's.*

With normal errors the MM-estimator's RMSE is 0.99 times that of least squares (0.0123 against 0.0124). Its theoretical efficiency of 95 per cent implies a standard deviation 2.6 per cent larger; in these 300 samples the cost is within Monte Carlo error of nothing, and slightly below what asymptotic theory predicts. Huber's M-estimator and the bisquare are the same. The other estimators pay visibly: median regression 1.18 times the RMSE of least squares (its asymptotic value is 1.25), Theil–Sen 1.45, the S-estimator 1.80 (asymptotically 1.87) and LTS 2.99, which is better than its asymptotic efficiency of about 7 per cent implies but still three times worse than least squares. These are the numbers behind the advice never to report an S or LTS fit as the final answer: they are starting points with a high breakdown point, and MM exists to turn them into efficient estimators.

As the tails get heavier, the comparison turns. With 5 degrees of freedom MM's RMSE is 0.90 times that of least squares, with 3 degrees of freedom 0.72, and with 2, where the error variance is infinite, 0.43. With Cauchy errors, least squares has an RMSE of 18.9, produced by a few samples with enormous errors, against 0.015 for MM. Median regression, at 0.80 and 0.45 for 3 and 2 degrees of freedom, catches up with MM only in the heaviest cases, and with Cauchy errors is slightly better (0.0146 against 0.0151). With t errors, MM's confidence intervals cover the truth in 93 to 95 per cent of samples.

One detail of these scenarios matters for what follows. With t errors, the MM-estimator flags observations as outliers: 2.2 per cent of them with 5 degrees of freedom and 13 per cent with Cauchy errors. None of them is an error. They are real observations from a heavy-tailed distribution, flagged because they are far from the bulk. A robust fit identifies observations that do not look like the majority; whether they are mistakes is a separate question that the data alone cannot answer.

## Robust regression as a check on large real data

Simulations with injected errors answer the question of what each estimator can survive. On real data sets, the more common question is whether a fitted regression is being driven by a few rows nobody has looked at, and the useful answer is usually comparative: fit least squares and MM, and look at where and how much they disagree.

The full diamonds data set has its own errors. Twenty stones have a recorded dimension of zero, which any data scientist would drop. Of the remaining 53,920, eight have dimensions whose implied density (carat per cubic millimetre) is far outside the narrow range real diamonds occupy around 0.0061. Inspecting them shows three kinds of error. Three stones have a depth equal to their carat weight (a 1.07-carat stone recorded as 6.62 × 6.67 × 1.07 mm), which looks like a value pasted into the wrong column. Three have one dimension of 31.8 or 58.9 mm, a stone the size of a walnut; two of them are 0.51-carat stones whose three measurements are the same numbers in a different order, at slightly different prices and cut grades. And two duplicate rows describe a 0.25-carat stone with the dimensions of one twice its weight.

A regression of log price on the log of recorded volume (x × y × z) and the three grades is exposed to all of these through the volume. Least squares gives a volume elasticity of 1.8875; removing the eight impossible rows moves it to 1.8903, a change of two standard errors of the cleaned fit. The MM-estimator gives 1.9042 with or without them, identical to four decimals. Six of the eight have standardised MM residuals beyond 3, between 15 and 35 in absolute value, and receive weight zero. The other two, the duplicated 0.25-carat rows, sit within the bulk and are harmless, because their recorded size and price are mutually consistent and only their carat weight, which is not in this model, is wrong. In fairness to least squares, its own studentised residuals also exceed 3 for six of the eight rows here: eight errors among 54,000 observations are too few to mask themselves, and a deletion rule would have found them, along with 289 other rows. The problem the simulation exposed arises when errors are a few per cent of the data, not a few per ten thousand. Figure 6 shows the outlier map: the impossible rows are the points that stand alone, far above, far below or far to the right of the cloud.

![Figure 6](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/robust-regression/figures/fig6_diamonds_outlier_map.png)

*Figure 6. Outlier map of 53,920 diamonds (ggplot2::diamonds with all recorded dimensions positive) for the regression of log price on log volume and the cut, colour and clarity grades: robust distance of the covariates (minimum covariance determinant, log scale) against the standardised MM residual. Orange: the eight rows whose recorded dimensions are physically impossible. Dotted lines: ±3 on the residual axis and the 97.5 per cent χ² cutoff on the distance axis.*

The more interesting number is the difference between the two fits once the errors are gone. The MM elasticity is 1.9042 and the least-squares elasticity on the cleaned data is 1.8903, a difference of 0.014, or about ten least-squares standard errors. Nothing is contaminated any more; the difference reflects the fact that real residuals are not normal. About 1 per cent of diamonds (520 of 53,920, six of them the impossible rows) have MM residuals beyond three robust standard deviations, stones whose price is far from what their size and grades predict for reasons the model does not see. The MM-estimator downweights them and describes the bulk; least squares averages them in. With 54,000 observations, any such difference is statistically significant, and the relevant question is whether it matters: a volume elasticity of 1.890 against 1.904 is a 0.7 per cent difference, irrelevant for pricing a diamond and worth a sentence in a report.

The NOx data that robustbase distributes tell the same story without any obvious errors. Eight thousand and eighty-eight hourly measurements of nitrogen oxides next to a Swiss motorway are regressed on the log of the emissions of the passing traffic and the square root of wind speed. Least squares and MM agree on the emission elasticity (0.641 against 0.636, a difference of 0.8 standard errors) and differ on the wind effect by 2.6 standard errors (−1.018 against −0.985); 46 hours (0.57 per cent) have MM residuals beyond 3. A disagreement of that size points to a modest set of unusual hours, worth looking at, rather than to a broken model.

These two examples suggest the most common practical use of robust regression in data science, which is not to replace least squares but to audit it. When the two agree, least squares can be reported with more confidence than before. When they disagree, the rows that MM downweights are the first place to look, and the outlier map says whether they are errors in the response, errors in the covariates or a distinct subpopulation.

## When the outliers are the signal

Every estimator in this article that resists outliers does so by deciding that some observations are less informative than others. When the unusual observations are errors, that is the point. When they are real, it can be exactly the wrong decision, and insurance is the clearest example.

The data are 67,856 one-year motor policies from 2004–05 analysed by de Jong and Heller (2008), of which 4,624 had a claim. Claim amounts are heavily skewed: the mean is 2,014 Australian dollars, the median 762, the largest 55,922, and the top 10 per cent of claims account for 50.7 per cent of the total paid. An insurer needs the expected claim of each policy, the mean, because premiums collected across a portfolio must cover the total paid out. Eight models were fitted to 70 per cent of the claims and used to predict the remaining 30 per cent, over 100 random splits, with driver age band, sex, area, vehicle age and log vehicle value as covariates.

![Figure 7](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/robust-regression/figures/fig7_claims.png)

*Figure 7. Left: predicted total claim cost divided by actual total cost on the held-out claims, mean and 5–95 per cent range over 100 random 70/30 splits, for eight models of claim severity. Right: mean predicted and observed claim by driver age band, averaged over the splits, for the Gamma GLM and two versions of the MM fit on the log scale. Data: insuranceData::dataCar (de Jong and Heller, 2008).*

The standard actuarial model, a Gamma generalised linear model with a log link, predicts 101.0 per cent of the actual total on average, with a 5–95 per cent range across splits of 91.5 to 109.5 per cent. Least squares on the raw claim amounts does the same (101.1 per cent). The robust estimators fitted to the raw amounts do not: Huber's M-estimator predicts 61.3 per cent of the total and the MM-estimator 36.9 per cent. An insurer that priced on the MM fit would collect a little over a third of what it pays out. The MM fit is not wrong in its own terms; it is an excellent estimate of the conditional centre of the bulk of claims, which is close to the median claim. It is the wrong estimand.

The log scale tells a more subtle story. On log claims, the data have almost no outliers: only 0.1 per cent of MM residuals exceed three robust standard deviations, and the MM and least-squares scales are close (1.21 and 1.18). The heavy right tail in dollars is a nearly symmetric distribution in logs. Both least squares and MM on log claims predict the median well and the mean badly if their predictions are simply exponentiated: 45.9 and 43.1 per cent of the total. Applying Duan's (1983) smearing correction, multiplying the exponentiated prediction by the average of the exponentiated residuals, brings both back to 100.6 and 100.7 per cent. On a scale where the errors are close to normal, robust and classical fits agree, and the retransformation, not the estimator, decides whether the mean is right.

Even then the robust fit is not neutral. The right panel of Figure 7 shows the mean claim by driver age band. The Gamma GLM follows the observed means closely in every band (2,592 predicted against 2,605 observed for the youngest drivers, 1,849 against 1,870 for the oldest). The smeared MM fit compresses the age gradient: it predicts 2,422 for the youngest drivers, 7 per cent too low, and 1,981 for the oldest, 6 per cent too high. Downweighting the largest claims removes some of what makes young drivers expensive. In a pricing model, that is a cross-subsidy from old drivers to young ones produced by a modelling choice rather than a business decision.

The lesson is not that robust regression is wrong for skewed data. It is that the choice between robust and classical estimation is a choice of estimand, and should be made deliberately. If the large values are errors, the bulk is the target and robust regression estimates it. If the large values are real and the decision depends on totals or means (premiums, budgets, expected costs, average treatment effects on a monetary outcome), downweighting them biases the answer, and the right response is a model for the whole distribution, such as a Gamma or Tweedie GLM, a transformation with an honest retransformation, or a separate model for the tail.

## Doing it in R and Python

R has the most complete implementation, in robustbase (Maechler and colleagues, 2026) and in MASS (Venables and Ripley, 2002):

```r
library(robustbase)
fit <- lmrob(log.light ~ log.Te, data = starsCYG)    # MM: S start, bisquare, 95% efficiency
summary(fit)                                          # robust SEs, weights, flagged observations
plot(fit, which = 1)                                  # residuals vs robust distances (outlier map)
fit_lts <- ltsReg(log.light ~ log.Te, data = starsCYG)

library(MASS)
fit_m <- rlm(calls ~ year, data = phones)                     # Huber M (not high-breakdown)
fit_mm <- rlm(calls ~ year, data = phones, method = "MM")    # MM via rlm
```

For models with many factor variables, `lmrob(..., control = lmrob.control(setting = "KS2014"))` uses the nonsingular subsampling of Koller and Stahel (2017) and a different ψ function, and is the setting robustbase's authors recommend for small samples and categorical predictors. The rrcov package (Todorov and Filzmoser, 2009) provides robust covariance estimators, including the minimum covariance determinant used for robust distances.

Python's support is newer and thinner. statsmodels provides M-estimators through `RLM`, with Huber, bisquare and other norms, a MAD scale and Huber's H1 covariance by default:

```python
import statsmodels.api as sm
X = sm.add_constant(df[["log_te"]])
huber = sm.RLM(df.log_light, X, M=sm.robust.norms.HuberT()).fit()
bisq = sm.RLM(df.log_light, X, M=sm.robust.norms.TukeyBiweight()).fit()   # starts from OLS
```

As the experiments showed, `RLM` with the bisquare norm starts from least squares and therefore inherits its failure under leverage, and its default H1 covariance understates the standard errors when bad leverage points are present. The first problem has a direct fix: `RLM.fit` accepts a starting point and a fixed scale, so an MM-estimate is a single call once a high-breakdown S-estimate is available.

```python
from statsmodels.robust.norms import TukeyBiweight
mm = sm.RLM(y, X, M=TukeyBiweight(c=4.685)).fit(
    start_params=s_fit.coef, start_scale=s_fit.scale, update_scale=False)
```

Started from the S-estimates computed by the code for this article, this call reproduces the MM coefficients to machine precision on the stars and to within 10⁻⁸ on the stack-loss data. Its standard errors are still H1: for the stars, 0.96 and 0.22 against the sandwich values of 3.35 and 0.76, because the four giants remain in the X′X matrix that the H1 formula inverts. They need to be computed separately.

statsmodels 0.15 also added `RLMDetS` and `RLMDetSMM` in `statsmodels.robust.resistant_linear_model`, S- and MM-estimators whose S step starts from deterministic subsets rather than random subsamples, an approach developed by Hubert, Rousseeuw and Verdonck (2012) for robust covariance. On the stars, `RLMDetSMM` agrees closely with robustbase (coefficients −5.05 and 2.27 against −4.97 and 2.25), with the same H1 standard errors (0.22 for the slope). On Brownlee's 21-observation stack-loss data it lands on a different solution, an air-flow coefficient of 0.81 against robustbase's 0.94. The difference appears to come from the objective rather than from a failed search. In samples this small the S objective depends on the convention for the M-scale equation, and the two S solutions are each the better one under a different convention: averaged over n − p, as in robustbase, the FAST-S solution has the lower scale (1.91 against 2.03); averaged over n, the DetS solution does (1.09 against 1.38). With 21 observations and four coefficients, the choice moves the air-flow coefficient by about one standard error, which is worth knowing before comparing robust fits across software.

scikit-learn offers `HuberRegressor`, `TheilSenRegressor` and `RANSACRegressor`, and `MinCovDet` for robust distances:

```python
from sklearn.linear_model import HuberRegressor, RANSACRegressor, TheilSenRegressor
from sklearn.covariance import MinCovDet
rd = MinCovDet(random_state=0).fit(X_no_const).mahalanobis(X_no_const) ** 0.5
```

None of the three is high-breakdown in the sense of S or MM, and the experiment showed where each one fails. The S, MM and LTS estimators used here are about 250 lines of NumPy, with FAST-S and FAST-LTS, and are in the repository linked below with checks against robustbase's published results.

## A working order for regression with suspect data

The experiments suggest an order of work.

Fit least squares and a high-breakdown robust estimator, MM by default, side by side. The comparison is cheap and is the most informative single check of whether a regression is being driven by a minority of observations.

Draw the outlier map: robust residuals against robust distances of the covariates. It separates vertical outliers, bad leverage points, good leverage points and the bulk, and it does not suffer from masking, because neither axis is computed from a fit the outliers can move.

Investigate the flagged rows before deciding anything. Some will be errors that can be corrected at the source (the pasted column, the duplicated stone); some will be real observations from a different population (the giants), which may deserve their own model; and some will be the expensive tail that the analysis is about.

Decide the estimand explicitly. If the question is about the bulk of the data and the unusual observations are errors or a population that is not of interest, report the robust fit, with robust standard errors. If the question involves means or totals of an outcome whose large values are real, do not use a robust estimator to answer it; model the whole distribution instead.

Do not use deletion rules based on least-squares residuals or Cook's distance as the defence against contamination. They work for a single isolated error and fail, through masking and swamping, as soon as errors are numerous or clustered, which is how data-entry problems usually arrive.

Never start a bisquare M-estimator from least squares. Start it from a high-breakdown fit, which is what MM does.

Report the robustness choice and its consequences. A sentence giving the least-squares estimate, the robust estimate and the share of observations downweighted tells a reader more than either estimate alone.

## What robust regression can and cannot do

Robust regression addresses a specific data challenge: a minority of observations that do not follow the model a majority follows, because they are errors or because they come from somewhere else. In the experiment with injected errors, the MM-estimator kept its bias below 0.2 per cent and the coverage of its confidence intervals between 93 and 97 per cent with up to 40 per cent of prices or carat weights corrupted, at a cost on clean data that was within Monte Carlo error of nothing. Five bad carat weights in a sample of 500 cut the least-squares elasticity by 17 per cent, and every estimator without a high-breakdown starting point failed at some contamination level between 1 and 40 per cent, while the deletion rules data scientists use to protect least squares stopped seeing most of the errors once they reached 10 to 20 per cent of the data. On real data, MM found impossible rows in a data set of 54,000 diamonds that are hard to spot by eye, and it identified the anomalous years and stars of two classic examples that least-squares diagnostics missed or misidentified.

It cannot decide what the analysis is about. It answers questions about the bulk of the data, and when the tail is the point, as with insurance claims, it answers the wrong question with great precision: an MM fit to the raw claim amounts predicted 37 per cent of the total cost. It is less efficient than least squares when nothing is wrong, by about 5 per cent for MM and far more for S and LTS on their own. Its standard errors are approximations that work well in large samples and less well in small ones, where the refinements of Koller and Stahel (2011) matter. And high-breakdown estimation has a computational side that ordinary regression lacks: random subsampling, local minima, and a dependence on the number of subsamples that is easy to forget.

Three conjectures follow from the experiments and are offered as conjectures. The first is that, in production data pipelines, the common practice of removing observations beyond three standard deviations before fitting removes mostly genuine observations when errors are rare and the data heavy-tailed, as it did with t-distributed errors here, and misses most of the errors when they are common, because the threshold is computed from a fit and a scale that the errors distort; a robust fit with a flagging rule, followed by a look at what it flags, would do better on both counts. The second is that a meaningful share of published regressions on administrative or scraped data would change sign or magnitude under an MM fit, because data-entry errors in covariates, the bad-leverage case in which every non-high-breakdown method in the experiment failed, are common and rarely checked. The third is that in pricing and cost models, the habit of fitting robust or median-based models to skewed monetary outcomes compresses the estimated effects of risk factors, as it did for driver age here, and produces cross-subsidies that nobody chose. Each is testable with the code below.

## Code and data

All code is in the `robust-regression` folder of the Medium-Blog repository: https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/robust-regression. It contains the M, S, MM and LTS implementations (`rob.py`), checks against robustbase's published results and against statsmodels, one script per experiment, and the text output from which the experimental results in this article were taken. The data come from the Rdatasets mirror of the ggplot2, MASS, robustbase, datasets and insuranceData R packages.

## References

Beaton, A. E. and Tukey, J. W. (1974). The fitting of power series, meaning polynomials, illustrated on band-spectroscopic data. Technometrics 16(2), 147–185.

Belsley, D. A., Kuh, E. and Welsch, R. E. (1980). Regression Diagnostics: Identifying Influential Data and Sources of Collinearity. Wiley.

Cook, R. D. (1977). Detection of influential observation in linear regression. Technometrics 19(1), 15–18.

Croux, C., Dhaene, G. and Hoorelbeke, D. (2003). Robust standard errors for robust estimators. Discussion Paper DPS 03.16, Center for Economic Studies, K.U. Leuven.

Dang, X., Peng, H., Wang, X. and Zhang, H. (2009). Theil–Sen estimators in a multiple linear regression model. Unpublished manuscript, University of Mississippi.

de Jong, P. and Heller, G. Z. (2008). Generalized Linear Models for Insurance Data. Cambridge University Press.

Donoho, D. L. and Huber, P. J. (1983). The notion of breakdown point. In P. J. Bickel, K. A. Doksum and J. L. Hodges (eds.), A Festschrift for Erich L. Lehmann, 157–184. Wadsworth.

Duan, N. (1983). Smearing estimate: A nonparametric retransformation method. Journal of the American Statistical Association 78(383), 605–610.

Fischler, M. A. and Bolles, R. C. (1981). Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography. Communications of the ACM 24(6), 381–395.

Hampel, F. R. (1971). A general qualitative definition of robustness. Annals of Mathematical Statistics 42(6), 1887–1896.

Hampel, F. R. (1974). The influence curve and its role in robust estimation. Journal of the American Statistical Association 69(346), 383–393.

Hampel, F. R., Ronchetti, E. M., Rousseeuw, P. J. and Stahel, W. A. (1986). Robust Statistics: The Approach Based on Influence Functions. Wiley.

Huber, P. J. (1964). Robust estimation of a location parameter. Annals of Mathematical Statistics 35(1), 73–101.

Huber, P. J. (1973). Robust regression: Asymptotics, conjectures and Monte Carlo. Annals of Statistics 1(5), 799–821.

Huber, P. J. and Ronchetti, E. M. (2009). Robust Statistics, 2nd ed. Wiley.

Hubert, M., Rousseeuw, P. J. and Verdonck, T. (2012). A deterministic algorithm for robust location and scatter. Journal of Computational and Graphical Statistics 21(3), 618–637.

Koenker, R. and Bassett, G. (1978). Regression quantiles. Econometrica 46(1), 33–50.

Koller, M. and Stahel, W. A. (2011). Sharpening Wald-type inference in robust regression for small samples. Computational Statistics & Data Analysis 55(8), 2504–2515.

Koller, M. and Stahel, W. A. (2017). Nonsingular subsampling for regression S estimators with categorical predictors. Computational Statistics 32(2), 631–646.

Maechler, M., Rousseeuw, P., Croux, C., Todorov, V., Ruckstuhl, A., Salibian-Barrera, M., Verbeke, T., Koller, M., Conceicao, E. L. T. and di Palma, M. A. (2026). robustbase: Basic Robust Statistics. R package version 0.99-7. https://CRAN.R-project.org/package=robustbase

Maronna, R. A., Martin, R. D., Yohai, V. J. and Salibián-Barrera, M. (2019). Robust Statistics: Theory and Methods (with R), 2nd ed. Wiley.

Owen, A. B. (2007). A robust hybrid of lasso and ridge regression. Contemporary Mathematics 443, 59–71.

Rousseeuw, P. J. (1984). Least median of squares regression. Journal of the American Statistical Association 79(388), 871–880.

Rousseeuw, P. J. and Leroy, A. M. (1987). Robust Regression and Outlier Detection. Wiley.

Rousseeuw, P. J. and Van Driessen, K. (1999). A fast algorithm for the minimum covariance determinant estimator. Technometrics 41(3), 212–223.

Rousseeuw, P. J. and Van Driessen, K. (2006). Computing LTS regression for large data sets. Data Mining and Knowledge Discovery 12(1), 29–45.

Rousseeuw, P. J. and van Zomeren, B. C. (1990). Unmasking multivariate outliers and leverage points. Journal of the American Statistical Association 85(411), 633–639.

Rousseeuw, P. J. and Yohai, V. J. (1984). Robust regression by means of S-estimators. In J. Franke, W. Härdle and R. D. Martin (eds.), Robust and Nonlinear Time Series Analysis, Lecture Notes in Statistics 26, 256–272. Springer.

Salibian-Barrera, M. and Yohai, V. J. (2006). A fast algorithm for S-regression estimates. Journal of Computational and Graphical Statistics 15(2), 414–427.

Sen, P. K. (1968). Estimates of the regression coefficient based on Kendall's tau. Journal of the American Statistical Association 63(324), 1379–1389.

Stigler, S. M. (2010). The changing history of robustness. The American Statistician 64(4), 277–281.

Theil, H. (1950). A rank-invariant method of linear and polynomial regression analysis, I, II and III. Proceedings of the Koninklijke Nederlandse Akademie van Wetenschappen 53, 386–392, 521–525 and 1397–1412.

Todorov, V. and Filzmoser, P. (2009). An object-oriented framework for robust multivariate analysis. Journal of Statistical Software 32(3), 1–47.

Tukey, J. W. (1960). A survey of sampling from contaminated distributions. In I. Olkin et al. (eds.), Contributions to Probability and Statistics: Essays in Honor of Harold Hotelling, 448–485. Stanford University Press.

Venables, W. N. and Ripley, B. D. (2002). Modern Applied Statistics with S, 4th ed. Springer.

Wickham, H. (2016). ggplot2: Elegant Graphics for Data Analysis, 2nd ed. Springer.

Yohai, V. J. (1987). High breakdown-point and high efficiency robust estimates for regression. Annals of Statistics 15(2), 642–656.
