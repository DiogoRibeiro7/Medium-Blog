# Before Deep Learning, Look at the Geometry

**Status:** draft  
**Theme:** longitudinal data, clustering, representation

Longitudinal clustering problems are often introduced as though the main decision were which algorithm to use.

It usually is not.

The harder question is what object should be clustered in the first place.

A subject observed through time is not naturally a row in a rectangular table. It is a trajectory. Depending on the problem, the meaningful variation may lie in level, trend, persistence, periodicity, volatility, change points, recovery time, or some combination of these.

If the representation is wrong, a sophisticated clustering algorithm can only organize the wrong geometry more efficiently.

## Start with the scientific distinction

Suppose two latent groups differ primarily in temporal persistence rather than mean level.

Then clustering raw observations can fail even when the groups are genuinely different. The problem is not necessarily insufficient model capacity. The discriminating information may live in a summary such as an autocorrelation, spectral statistic, transition rate, or another feature that reflects dependence through time.

The modelling sequence should therefore begin with

\[
\text{scientific distinction}
\rightarrow
\text{temporal representation}
\rightarrow
\text{clustering method}.
\]

Not the other way around.

## A useful thought experiment

Consider two stationary processes with the same marginal mean and variance but different autocorrelation structures.

If we ignore ordering and treat their observations as exchangeable samples, much of the class information disappears.

A persistence feature such as

\[
\widehat\rho_1
=
\frac{\sum_{t=2}^{T}(X_t-\bar X)(X_{t-1}-\bar X)}
{\sum_{t=1}^{T}(X_t-\bar X)^2}
\]

can recover part of the distinction because it represents the feature that actually differs between groups.

This does not imply that lag-one autocorrelation is universally sufficient. It illustrates the more general point: **representation should encode the mechanism by which trajectories differ**.

## The baseline should be hard to beat for the right reason

A good longitudinal clustering study should include simple baselines that make strong assumptions explicit.

For example:

- hand-designed temporal summaries plus Gaussian mixture modelling;
- distances between standardized trajectories plus hierarchical clustering;
- low-dimensional functional representations plus k-means;
- a simple hidden-state model when discrete regimes are scientifically plausible.

These are not straw men. They answer an important question:

\[
\boxed{
\text{How much of the problem is already solved by choosing the right representation?}
}
\]

If a simple model performs well after representation is fixed, that is useful scientific information.

## Finite sample behaviour matters

Even when a temporal feature is theoretically discriminative, estimating it from short trajectories introduces noise.

That noise propagates into the clustering problem.

So it is useful to separate at least three layers:

\[
\text{population feature separation}
\]

\[
\text{finite-sample feature estimation}
\]

\[
\text{mixture or clustering estimation}.
\]

Without this decomposition, performance losses are easily attributed to the wrong cause.

For example, poor clustering may come from weak population separation, noisy feature estimation, or instability in fitting the latent mixture. These mechanisms suggest different remedies.

## Deep learning is sometimes appropriate

There are longitudinal datasets for which learned representations are justified: large samples, complex multivariate structure, irregular events, long-range dependencies, or objectives that cannot be captured by a small number of interpretable features.

But "sequence data" is not itself an argument for a recurrent network or transformer.

The relevant comparison is not

\[
\text{old method versus new method}.
\]

It is

\[
\boxed{
\text{Does the more flexible representation capture information that the simpler one genuinely misses?}
}
\]

That question can be tested.

## The practical workflow

For a new longitudinal clustering problem, I would usually begin with:

1. define what scientifically distinguishes the groups;
2. construct a small set of interpretable temporal representations;
3. quantify their finite-sample variability;
4. fit simple clustering models first;
5. inspect failure modes rather than only average clustering scores;
6. escalate representation complexity only when the residual structure justifies it.

This approach is not anti-machine-learning. It simply puts modelling before machinery.

The main lesson is simple:

\[
\boxed{
\text{In longitudinal clustering, geometry is often chosen before it is learned.}
}
\]
