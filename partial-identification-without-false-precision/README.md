# When the Data Refuse a Point Estimate

**Status:** draft  
**Theme:** partial identification, econometrics, statistical inference

Statistical work often begins with an assumption that is so familiar we barely notice it: somewhere in the data there is a single parameter waiting to be estimated.

Sometimes there is not.

The data, together with assumptions we are willing to defend, may identify only a **set of plausible parameter values**. Forcing that problem into a point estimate does not create information. It creates precision that the model has not earned.

That is the basic motivation for partial identification.

## Identification comes before estimation

Suppose \(\theta\) is the quantity we care about and the model implies a collection of moment inequalities

\[
E[m_j(W,\theta)] \ge 0,
\qquad j=1,\ldots,J.
\]

Here \(W\) denotes the observed data and each \(m_j\) encodes one restriction implied by the model.

The population identified set is

\[
\Theta_I
=
\left\{
\theta\in\Theta:
E[m_j(W,\theta)]\ge 0
\text{ for all }j
\right\}.
\]

If \(\Theta_I\) contains exactly one point, we have point identification.

If it contains an interval, region, or several admissible values, then the model is partially identified.

That is not a failure of estimation. It is a statement about what can and cannot be learned from the information available.

## A simple example

Suppose a parameter is known only to satisfy two population restrictions:

\[
\theta \ge a
\]

and

\[
\theta \le b.
\]

Then the information content of the model is

\[
\theta\in[a,b].
\]

Choosing the midpoint

\[
\widehat\theta=\frac{a+b}{2}
\]

may be convenient, but it is not an identified fact. It adds a decision rule that was never implied by the data.

This is the central discipline of partial identification: **do not confuse a convenient representative of a set with a parameter that the model uniquely determines**.

## Why point estimates can become misleading

The attraction of a point estimate is obvious. It is easy to rank, optimize, graph, report, and feed into the next model.

But this convenience can hide three very different sources of uncertainty:

1. sampling uncertainty;
2. model uncertainty;
3. identification uncertainty.

The first asks how much an estimate would vary across samples.

The second asks how conclusions depend on assumptions.

The third asks whether the assumptions and population distribution determine a unique value at all.

More data can reduce sampling uncertainty. It cannot automatically eliminate identification uncertainty.

With an arbitrarily large sample, the answer may still be

\[
\boxed{\theta\in\Theta_I}
\]

rather than

\[
\boxed{\theta=\theta_0}.
\]

## Moment inequalities are natural in many problems

Inequalities arise whenever theory gives bounds rather than exact equations.

Examples include revealed-preference restrictions, incomplete models, missing counterfactual information, interval observations, strategic interactions, and settings where only monotonicity or sign restrictions are credible.

The important modelling question is not “how do I get a point estimate anyway?”

It is:

\[
\boxed{
\text{What restrictions can I actually defend?}
}
\]

Once those restrictions are explicit, the identified set becomes part of the scientific result.

## Sample moments do not define the population set exactly

In data we replace population expectations by sample analogues,

\[
\widehat g_j(\theta)
=
\frac{1}{n}\sum_{i=1}^{n}m_j(W_i,\theta).
\]

A naive approach would keep every \(\theta\) satisfying

\[
\widehat g_j(\theta)\ge0
\quad\text{for all }j.
\]

But finite samples introduce noise. A parameter value compatible with the population restrictions can produce a slightly negative sample moment, while an incompatible value can look acceptable by chance.

This is where inference becomes the real problem.

The task is not merely to compute a feasible set from observed sample moments. We need uncertainty procedures that respect the one-sided nature of the restrictions and produce confidence regions with a defensible coverage interpretation.

## Boundaries matter

Moment-inequality problems are often hardest near the boundary

\[
E[m_j(W,\theta)] = 0.
\]

A strongly positive moment is easy to classify as non-binding. A clearly negative moment gives evidence against the candidate \(\theta\).

The delicate case is a moment close to zero, where sampling noise can change which restrictions appear active.

This is one reason the geometry of the identified set matters. Inference is affected not only by the number of moments but by which constraints are binding or nearly binding at a candidate parameter value.

## Partial identification is often the more informative answer

A wide identified set is sometimes described as disappointing.

I think that is backwards.

If the available information supports only a wide set, reporting that width tells us exactly what is missing. It can reveal which assumptions are carrying identification and which additional measurements or design changes would be valuable.

A narrow point estimate produced by an unjustified assumption can hide that information completely.

In that sense, partial identification is not a weaker version of point estimation. It is a framework for being explicit about the boundary between **what the data say** and **what the analyst added**.

## The practical habit

Before choosing an estimator, ask:

\[
\boxed{
\text{What does the model identify in the population?}
}
\]

If the answer is a set, keep the set.

Then build inference around that object rather than collapsing it prematurely into a number.

Precision is valuable only after identification has earned it.
