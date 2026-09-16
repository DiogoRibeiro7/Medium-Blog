# Twenty Seeds Are Not a Phase Diagram

**Status:** draft  
**Theme:** simulation, statistical validation, reproducibility

A simulation study can be numerically correct and still tell the wrong scientific story.

The failure often appears when we turn noisy Monte Carlo summaries into categorical statements: stable versus unstable, positive versus negative, regime A versus regime B, phase transition here rather than there. Once a threshold is involved, the scientific object is no longer just a mean. It is the **topology of a classification rule under simulation noise**.

That needs more replication than many experiments receive.

## The hidden random variable in the conclusion

Suppose a simulation produces a binary event on each seed. Let

\[
X_i \in \{0,1\},
\qquad
P(X_i=1)=p.
\]

We run \(N\) seeds and classify the cell according to whether the observed proportion exceeds one half:

\[
\widehat p_N = \frac{1}{N}\sum_{i=1}^{N}X_i,
\qquad
\text{class}=\mathbf 1\{\widehat p_N>0.5\}.
\]

If the true probability is \(p=0.51\), then the underlying process is technically on the positive side of the boundary. But with only 21 independent runs, the probability that the observed majority lands on the wrong side is

\[
P\!\left(\operatorname{Binomial}(21,0.51)\le 10\right)
\approx 0.463.
\]

So a true 51/49 split can look like the opposite regime in almost half of small experiments.

Nothing is wrong with the random-number generator. Nothing is wrong with the code. The problem is the strength of the conclusion being extracted from weak replication.

## Means can look stable while regimes are unstable

This is easy to miss because ordinary summary statistics may already look reassuring.

Imagine that an experiment compares two methods through a per-seed difference \(\Delta_i\). The Monte Carlo mean

\[
\overline\Delta_N
=
\frac{1}{N}\sum_{i=1}^{N}\Delta_i
\]

may move very little when we go from 20 to 100 seeds. That can create the impression that the experiment has converged.

But now define a second quantity,

\[
q=P(\Delta_i<0),
\]

and use the rule \(q>0.5\) to identify a qualitative regime. If \(q\) is close to one half, the **mean effect can be stable while the regime label remains fragile**.

This distinction matters whenever the scientific claim depends on a sign, majority, ordering, crossing, or phase boundary.

## Replication should depend on distance to the decision boundary

There is no useful universal rule saying that 20, 50, or 100 seeds are enough.

The required replication depends on the estimand and on how close the result lies to the boundary used by the conclusion.

A practical workflow is:

1. use a modest number of seeds for broad screening;
2. identify cells close to the relevant decision boundary;
3. increase replication selectively for those cells;
4. require the qualitative conclusion to survive the escalation;
5. report when a candidate transition disappears under stronger replication.

This is more efficient than running every cell at the maximum seed count, and more defensible than treating all cells as equally certain.

## Failed transitions are useful results

Suppose a 20-seed screen suggests

\[
A\rightarrow C.
\]

At 100 seeds the transition becomes

\[
A\rightarrow B,
\]

and at 200 seeds it remains \(A\rightarrow B\).

It is tempting to describe the first result as wasted computation. It is not.

The disappearing \(A\rightarrow C\) transition teaches us something important about the experiment: **the topology itself was sensitive to Monte Carlo error**. That is methodological information. It tells us which parts of the phase diagram require stronger evidence and which summaries are too close to a classification boundary to support categorical claims.

A falsified transition can improve a study more than another stable average.

## Report the hierarchy, not only the final number

A strong simulation paper or technical note should preserve some record of the escalation:

\[
N=20
\rightarrow
N=100
\rightarrow
N=200.
\]

The reader should be able to see which conclusions were obvious early, which were boundary cases, and which changed when replication increased.

This makes the analysis more transparent. It also prevents a common form of accidental hindsight: presenting the final regime map as though every boundary had always been clear.

## A useful distinction

For simulation work I find it useful to separate three questions:

\[
\boxed{
\text{Is the mean stable?}
}
\]

\[
\boxed{
\text{Is the uncertainty small enough?}
}
\]

\[
\boxed{
\text{Is the qualitative conclusion stable?}
}
\]

Those are not the same question.

The last one is often the hardest, because a categorical claim can remain unstable long after a mean looks settled.

## The broader lesson

Monte Carlo replication is not a cosmetic parameter to be chosen once at the top of a notebook. It is part of the inferential design.

If the conclusion is continuous, study the uncertainty of the continuous estimand. If the conclusion is a threshold, ordering, or phase transition, study the uncertainty of that decision too.

The more interesting the topology, the less we should trust a phase diagram drawn from a handful of seeds.
