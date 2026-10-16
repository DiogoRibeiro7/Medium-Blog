# When Regression Gets the Causal Effect Wrong

*Part 1 of Causal Inference in Statistics — omitted-variable bias, randomization and an experiment with a known treatment effect. Python code provided.*

An observational regression can be precise, statistically significant and wrong about the effect of an intervention. Increasing the sample size does not resolve that problem. To see why, I generated data from a model in which the causal effect is known, and estimated that effect three ways: by regressing the outcome on treatment alone, by including the confounder, and by randomly assigning treatment. Each method was evaluated on 1,000 independently generated datasets at three sample sizes.

The results are not subtle. With 10,000 observations, the unadjusted regression estimated an effect of 2.0005, while the true effect was 1. Its average reported standard error was only 0.0123. The adjusted and randomized analyses both estimated approximately 1. The increasingly narrow confidence interval around the naive estimate was evidence of precision, not of causal validity.

## The experiment

Let U represent a characteristic that affects both treatment X and outcome Y. The data are generated as

\[
U,\varepsilon_X,\varepsilon_Y\stackrel{\mathrm{ind}}{\sim}N(0,1),\qquad
X=U+\varepsilon_X,\qquad
Y=X+2U+\varepsilon_Y.
\]

The structural effect of setting X to a different value, holding the other causes of Y fixed, is exactly 1. In potential-outcome notation, \(Y(x)=x+2U+\varepsilon_Y\), so \(Y(x+1)-Y(x)=1\) for every unit in this experiment.

The problem is that U raises X and Y. An ordinary regression of Y on X alone therefore combines the causal effect with the association due to U. Its population slope is

\[
\frac{\operatorname{Cov}(X,Y)}{\operatorname{Var}(X)}
=1+2\frac{\operatorname{Cov}(X,U)}{\operatorname{Var}(X)}
=1+2\frac{1}{2}=2.
\]

This is not a small-sample problem. Two is the population quantity the naive estimator approaches as the sample grows.

The adjusted regression includes U alongside X. Under the specified model, the coefficient of X estimates the structural effect. For the randomized comparison, X is generated independently of U instead, while Y retains the same structural dependence on X and U. The randomized design therefore removes the common cause of treatment and outcome from the assignment mechanism, although the unexplained variation in outcomes remains.

The three approaches use ordinary least squares and its conventional homoskedastic standard error. The simulations satisfy the error conditions needed for that calculation. The reported 95% intervals use the approximation \(\widehat\tau\pm1.96\widehat{\mathrm{SE}}\).

## What the estimates say

The following figures are averages across 1,000 independent datasets at each sample size, with random seed 20261016.

| Observations | Unadjusted OLS | Adjusted OLS | Randomized OLS |
| ---: | ---: | ---: | ---: |
| 100 | 1.9948 | 0.9966 | 1.0017 |
| 1,000 | 2.0015 | 1.0004 | 0.9984 |
| 10,000 | 2.0005 | 1.0001 | 0.9996 |

At all three sample sizes, the unadjusted estimator sits near 2 rather than the true effect of 1. The other estimators sit near 1. This is what the model predicts: adjusting for the common cause or breaking its link with treatment can identify the desired effect here, while adding observations to the unadjusted analysis cannot.

The confidence intervals expose the difference more clearly.

| Observations | Unadjusted coverage | Adjusted coverage | Randomized coverage |
| ---: | ---: | ---: | ---: |
| 100 | 0.000 | 0.943 | 0.946 |
| 1,000 | 0.000 | 0.942 | 0.960 |
| 10,000 | 0.000 | 0.951 | 0.953 |

Coverage is the fraction of simulated intervals containing the true effect. None of the unadjusted intervals covered 1, even at n=100. For the adjusted and randomized analyses, coverage is close to the nominal 95%. Small departures are expected from Monte Carlo variability and use of the normal approximation.

At n=10,000, the mean standard errors were 0.0123 for unadjusted regression, 0.0100 for adjusted regression and 0.0224 for randomized regression. The unadjusted estimate was not noisy. It was precisely centred on a different estimand.

## Why adjustment and randomization differ

Adjustment succeeds in this experiment because U was observed, its role in the outcome model was correctly specified, and treatment retained variation after conditioning on U. An observational study cannot assume that the relevant confounders are all measured, or that a linear adjustment is sufficient. A narrow interval after adjustment says nothing by itself about an unmeasured common cause.

Randomization operates differently. It makes treatment independent of U by design, rather than asking an outcome model to remove the association after treatment was assigned. This experiment gives the randomized estimator a larger sampling variance than the adjusted estimator, because the unadjusted randomized analysis leaves U-related outcome variation in the residual. Randomization protects the causal comparison from this confounding, but it does not guarantee that the most basic estimator is statistically most efficient.

The simulation deliberately makes both identification strategies work. It is a controlled comparison, not evidence that adjustment will usually match randomized evidence in real data.

## What the experiment cannot establish

The confounder is known here because I constructed the data. In an actual observational study, the absence of a measured U would be precisely the difficulty, and it cannot be diagnosed from the apparent precision of the regression coefficient. The effect is also constant, the equations are linear, and the errors are independent Gaussian variables with constant variance. Different treatment assignment mechanisms, effect heterogeneity, measurement error and model misspecification would change the numerical results.

The comparison does not test how to discover an appropriate adjustment set. It only shows what happens after the confounding structure has been specified correctly, incorrectly, or removed through random assignment.

## Conclusion

With 10,000 observations, naive regression estimated a treatment effect of 2.0005 with a mean standard error of 0.0123. The true effect was 1. More data reduced uncertainty around the wrong population coefficient; they did not turn that coefficient into a causal effect.

In this experiment, adjustment and randomization recovered the causal quantity because their identifying conditions were satisfied. The distinction is not between a simple model and a sophisticated model, but between a statistical parameter and the causal question one wants it to answer.

The next question is what happens when an observed adjustment variable is not a confounder at all. Controlling for more variables can sometimes make the causal estimate worse, not better.

## Code

The accompanying `experiment.py` reproduces all estimates in the tables using NumPy. It fixes the random seed, reports mean estimates, bias, root mean squared error, mean standard errors and interval coverage, and does not require an external dataset.

## References

- Holland, P. W. (1986). Statistics and Causal Inference. *Journal of the American Statistical Association*, 81(396), 945–960. https://doi.org/10.1080/01621459.1986.10478354
- Rubin, D. B. (1974). Estimating Causal Effects of Treatments in Randomized and Nonrandomized Studies. *Journal of Educational Psychology*, 66(5), 688–701. https://doi.org/10.1037/h0037350
- Hernán, M. A. and Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC. https://miguelhernan.org/whatifbook
