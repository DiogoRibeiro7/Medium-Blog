# The Baseline Is Not a Straw Man

**Status:** draft  
**Theme:** modelling practice, scientific method, data science

A baseline is often treated as the model we intend to beat.

That is too weak a role.

A good baseline is a **scientific control**. It tells us which part of the problem was already solvable before we added complexity.

If the baseline is deliberately poor, the comparison may make a sophisticated model look impressive without teaching us much.

## Start with the estimand, not the model family

Before choosing an algorithm, define what is being estimated or predicted and how success will be judged.

That gives a sequence such as

\[
\text{estimand}
\rightarrow
\text{baseline}
\rightarrow
\text{validation}
\rightarrow
\text{uncertainty}
\rightarrow
\text{complexity}.
\]

Starting with a model reverses the logic. We end up asking where a technique can be applied rather than what the problem requires.

## What makes a baseline strong?

A strong baseline should be simple enough to understand and credible enough that beating it means something.

Depending on the problem, that may be:

- the historical mean;
- a seasonal naive forecast;
- ordinary least squares;
- logistic regression;
- a transparent state-space model;
- a simple persistence rule;
- a hand-designed representation with a conventional classifier.

The point is not that simple methods are always better.

The point is that the baseline should capture the obvious structure before we attribute gains to sophisticated machinery.

## Complexity can improve the wrong part of the system

Suppose a forecasting model reduces RMSE by 2% but requires ten times the infrastructure, is poorly calibrated in the region that drives decisions, and degrades sharply after a distribution shift.

Was the model better?

The answer depends on the actual decision problem.

Model comparison should therefore include not only predictive averages but also the properties that matter operationally:

\[
\text{calibration},
\quad
\text{stability},
\quad
\text{latency},
\quad
\text{cost},
\quad
\text{failure modes},
\quad
\text{uncertainty}.
\]

A baseline makes these trade-offs visible because it gives complexity something concrete to justify.

## A failed complex model can be a useful result

Suppose a neural model and a simple statistical model perform essentially the same after careful validation.

That is not a disappointing experiment.

It suggests that the available signal may already be captured by the simpler representation, or that the sample does not contain enough information to support the extra flexibility.

Either conclusion is scientifically useful.

The wrong response is to keep tuning until a small test-set difference appears.

## Baselines expose leakage and weak validation

Simple models are also diagnostic tools.

If a trivial baseline performs implausibly well, investigate the data split, target construction, temporal leakage, duplicated observations, and preprocessing before celebrating.

Likewise, if every sophisticated model performs dramatically better than a sensible baseline, that gap deserves explanation.

Large gains are not impossible. They are claims that should survive scrutiny.

## Use complexity as a hypothesis

Rather than assuming a more flexible model will help, formulate what it is expected to capture.

For example:

\[
H_1:
\text{nonlinear interactions contain predictive information absent from the additive baseline.}
\]

or

\[
H_1:
\text{long-range temporal dependence matters beyond the short-memory baseline.}
\]

Now complexity has a purpose that can be falsified.

If the complex model wins, inspect whether it wins where the hypothesis predicted. If it does not, the extra machinery has not justified itself.

## The practical rule

I would not ask

> What is the strongest model we can build?

I would ask

> What is the simplest model that captures the structure we can currently defend, and what evidence says we need more?

That changes the role of the baseline completely.

\[
\boxed{
\text{The baseline is not the opponent. It is the experiment's control.}
}
\]
