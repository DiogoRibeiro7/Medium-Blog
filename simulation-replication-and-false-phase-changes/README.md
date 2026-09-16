# Twenty Seeds Are Not a Phase Diagram

**Status:** review  
**Theme:** simulation, statistical validation, reproducibility

A simulation study can be numerically correct and still tell the wrong scientific story.

The failure often appears when we turn noisy Monte Carlo summaries into categorical statements: stable versus unstable, positive versus negative, regime A versus regime B, phase transition here rather than there. Once a threshold is involved, the scientific object is no longer just a mean. It is also the **classification induced by a noisy estimate**.

That distinction matters because a mean can look settled while the qualitative map built from it remains unstable.

The title is deliberately provocative. Twenty replications are not automatically wrong, and there are settings where even a small simulation is enough to distinguish effects that are far apart. The point is narrower:

> **A replication count is not evidence by itself. Its adequacy depends on the estimand, the Monte Carlo uncertainty, and the distance to the decision boundary used by the conclusion.**

## The hidden random variable in the conclusion

Suppose a simulation produces a binary event on each replication. Let

\[
X_i \in \{0,1\},
\qquad
P(X_i=1)=p.
\]

We run \(N\) independent replications and classify the cell according to whether the observed proportion exceeds one half:

\[
\widehat p_N
=
\frac{1}{N}\sum_{i=1}^{N}X_i,
\qquad
\text{class}
=
\mathbf 1\{\widehat p_N>0.5\}.
\]

There are now two stochastic objects in the analysis.

The first is the estimator \(\widehat p_N\).

The second is the **qualitative label obtained after thresholding it**.

Those are not equally stable.

If the true probability is

\[
p=0.51,
\]

then the population quantity is technically on the positive side of the boundary. But with only 21 independent replications, the probability that the observed majority lands on the other side is

\[
P\!\left(\operatorname{Binomial}(21,0.51)\le 10\right)
\approx 0.463.
\]

So a genuine 51/49 split can look like the opposite regime in almost half of such experiments.

Even increasing the number of replications does not make a tiny margin disappear immediately:

| Replications \(N\) | \(P(\widehat p_N\le0.5)\) when \(p=0.51\) |
| ---: | ---: |
| 21 | 0.463 |
| 101 | 0.420 |
| 201 | 0.388 |
| 1,001 | 0.263 |

Under this exact majority rule, roughly 6,765 odd-numbered independent replications are required before the wrong-side probability falls below 5% when the true probability is only \(0.51\).

That number is not a recommended universal seed count. It shows how expensive a **categorical claim near a hard boundary** can become.

Nothing is wrong with the random-number generator. Nothing is wrong with the code. The problem is the strength of the conclusion being extracted from weak separation.

## Monte Carlo error should be visible

For the Bernoulli example,

\[
\operatorname{MCSE}(\widehat p_N)
=
\sqrt{\frac{p(1-p)}{N}}.
\]

Because

\[
p(1-p)\le \frac14,
\]

the worst-case Monte Carlo standard error is

\[
\operatorname{MCSE}(\widehat p_N)
\le
\frac{1}{2\sqrt N}.
\]

That gives a useful scale before we know the true value of \(p\):

| \(N\) | Maximum MCSE | Approx. \(1.96\times\) MCSE |
| ---: | ---: | ---: |
| 20 | 0.112 | 0.219 |
| 50 | 0.071 | 0.139 |
| 100 | 0.050 | 0.098 |
| 200 | 0.035 | 0.069 |
| 500 | 0.022 | 0.044 |
| 1,000 | 0.016 | 0.031 |

The last column is not being offered as an exact confidence interval. It is a simple scale calculation showing how much Monte Carlo variability remains possible around a proportion near one half.

This is why choosing a replication count by habit is dangerous. The relevant question is not

\[
\text{“Is }N=100\text{ a respectable number?”}
\]

but

\[
\boxed{
\text{“Is the Monte Carlo uncertainty small relative to the claim I am making?”}
}
\]

## Means can look stable while regimes are unstable

This is easy to miss because ordinary summary statistics may already look reassuring.

Imagine that an experiment compares two methods through a per-replication difference \(\Delta_i\). The Monte Carlo mean

\[
\overline\Delta_N
=
\frac{1}{N}\sum_{i=1}^{N}\Delta_i
\]

may move very little when we go from 20 to 100 replications. That can create the impression that the experiment has converged.

Now define another population quantity,

\[
q=P(\Delta_i<0),
\]

and use the rule

\[
q>0.5
\]

to identify a qualitative regime.

The mean and the majority probability answer different questions. A small positive \(E[\Delta]\) can coexist with a value of \(q\) close to one half, particularly when the distribution is heterogeneous, skewed, or concentrated around zero.

So we may have

\[
\overline\Delta_N
\approx
\text{stable}
\]

while

\[
\mathbf 1\{\widehat q_N>0.5\}
\approx
\text{unstable}.
\]

This distinction matters whenever the scientific claim depends on a sign, majority, ranking, crossing, stability class, or phase boundary.

## A phase diagram is a collection of decisions

Suppose a study evaluates a grid of parameter settings

\[
\lambda_1,\lambda_2,\ldots,\lambda_K
\]

and assigns each cell to one of several regimes.

The resulting phase diagram can look deterministic on the page. Statistically, however, it is a collection of classification decisions made from finite Monte Carlo samples.

If several neighbouring cells lie close to a decision boundary, the apparent topology can depend on random simulation noise:

\[
A\rightarrow C
\]

in one run, but

\[
A\rightarrow B\rightarrow C
\]

in another.

The important question is therefore not only whether each cell has a reasonable point estimate. It is whether the **adjacency, ordering, and transition structure** are stable enough to support the scientific story being told about the diagram.

That is a stronger requirement.

## Replication should depend on distance to the decision boundary

There is no useful universal rule saying that 20, 50, 100, or 1,000 replications are enough.

The required replication depends on at least four things:

1. the performance measure being estimated;
2. its Monte Carlo variance;
3. the threshold or comparison used to create the scientific conclusion;
4. the distance between the true quantity and that boundary.

A cell with

\[
p\approx0.95
\]

is a very different inferential problem from a cell with

\[
p\approx0.51.
\]

Treating them as though they require the same replication budget wastes computation on the first and risks overclaiming on the second.

A practical design is therefore **adaptive in precision, not adaptive in storytelling**.

### Stage 1: broad screening

Use a moderate number of replications to explore the parameter space and locate potentially interesting regions.

### Stage 2: identify dangerous cells

Flag cells that are close to a decision boundary, have large Monte Carlo error, or create a qualitative topology change.

### Stage 3: escalate replication

Allocate additional independent replications to those cells until the uncertainty is small enough for the intended claim, or until the study concludes that the boundary cannot be located precisely with the available computational budget.

### Stage 4: challenge the topology

Ask whether the first transition, ordering, or phase label survives the escalation.

### Stage 5: report the instability

If a candidate transition disappears, report that fact. Do not rewrite the history of the experiment as though the final topology was obvious from the start.

## Screening is not confirmation

There is an important caveat to adaptive replication.

If we search a large parameter grid using noisy estimates, select the most interesting cells, and then treat those same noisy estimates as confirmatory evidence, selection itself can exaggerate the apparent signal.

A clean approach is to separate the roles of the replications.

For example:

\[
\text{screening seeds}
\quad\perp\quad
\text{confirmation seeds},
\]

or to pre-specify a cumulative escalation rule before looking at the final classification.

The exact design can vary, but the principle should be explicit:

\[
\boxed{
\text{Do not let a noisy search result silently become its own confirmation.}
}
\]

This matters especially when the goal is to find rare topology changes or adversarial parameter settings.

## Failed transitions are useful results

Suppose a 20-replication screen suggests

\[
A\rightarrow C.
\]

At 100 replications the transition becomes

\[
A\rightarrow B,
\]

and at 200 replications it remains \(A\rightarrow B\).

It is tempting to describe the first result as wasted computation. It is not.

The disappearing \(A\rightarrow C\) transition teaches us something important about the experiment: **the topology itself was sensitive to Monte Carlo error**.

That tells us which parts of the parameter space require stronger evidence and which summaries are too close to a classification boundary to support categorical claims.

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

For each important performance measure, I would report at least:

- the Monte Carlo estimate;
- a Monte Carlo standard error or another appropriate simulation-uncertainty measure;
- the number of successful replications actually contributing to the estimate;
- the rule used to transform the estimate into any qualitative label;
- any replication escalation used for boundary cases.

If a phase diagram is central to the conclusion, I would also preserve the underlying continuous statistics rather than publishing only the final colours.

A colour map is much easier to interpret when the reader can see how far each cell is from the boundary that created the colour.

## Three different notions of convergence

For simulation work it is useful to separate three questions.

First:

\[
\boxed{
\text{Is the estimated performance measure numerically stable?}
}
\]

Second:

\[
\boxed{
\text{Is its Monte Carlo uncertainty small enough for the desired precision?}
}
\]

Third:

\[
\boxed{
\text{Is the qualitative scientific conclusion stable?}
}
\]

Those are not the same question.

The last one can be the hardest because thresholding destroys information. Two estimates that differ only slightly may receive different categorical labels, while two cells with very different distances from the boundary may be displayed with the same colour.

## Reproducible companion calculation

The accompanying script [`replication_diagnostics.py`](replication_diagnostics.py) uses only the Python standard library. It reproduces the exact \(p=0.51\) wrong-majority probabilities, computes the maximum Bernoulli Monte Carlo standard error, and finds the minimum odd replication count needed to push the wrong-side probability below a chosen target.

The point of including the script is not computational sophistication. It is that claims about simulation precision should themselves be easy to reproduce.

## The broader lesson

Monte Carlo replication is not a cosmetic parameter to be chosen once at the top of a notebook. It is part of the inferential design.

If the conclusion is continuous, study the uncertainty of the continuous estimand. If the conclusion is a threshold, ordering, or phase transition, study the uncertainty of that decision too.

The right replication count is therefore not

\[
20,
\quad
100,
\quad
1{,}000,
\]

in the abstract.

It is the amount of simulation needed to make the uncertainty small enough **relative to the scientific claim**.

And when the claim is about topology, that can require substantially more evidence than a stable-looking mean.

## References

- Koehler E, Brown E, Haneuse S. *On the Assessment of Monte Carlo Error in Simulation-Based Statistical Analyses*. The American Statistician. 2009;63(2):155–162. DOI: [10.1198/tast.2009.0030](https://doi.org/10.1198/tast.2009.0030).
- Morris TP, White IR, Crowther MJ. *Using simulation studies to evaluate statistical methods*. Statistics in Medicine. 2019;38(11):2074–2102. DOI: [10.1002/sim.8086](https://doi.org/10.1002/sim.8086).
