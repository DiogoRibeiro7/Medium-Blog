# Part 20 code companion: using capacity versus filling capacity

The Part 20 trial simulation ranks heterogeneous treatment benefits for a
fixed 20% capacity. This independent analytical calculation examines a
different decision: is the available capacity an *upper limit*, or must it
be fully used even when some individuals would be harmed by treatment?

It leaves `experiment.py`, `validate_policy.py`, `results.csv`, and article
manuscripts unchanged.

## Population model

The original linear structural model gives the treatment benefit
`B(X) = 0.5 + 1.5 X`, with `X` standard normal. It is the reduction in
an adverse outcome from treatment. Under an oracle ranking, higher X has
higher benefit. At exact capacity `q`, treat the top fraction `q`.

For `0 < q < 1`, the cutoff is `z_q = Phi^{-1}(1-q)`. Using the standard
normal density `phi`, the population gain for the exact-capacity policy is

```text
G_exact(q) = E[B(X) * I(X > z_q)] = 0.5*q + 1.5*phi(z_q).
```

At `q=0`, the gain is zero; at `q=1` it equals the population mean benefit
0.5. These are exact expectations, not simulation or extrapolation.

The benefit changes sign at `X=-1/3`. Thus the population fraction with
positive benefit is `Phi(1/3)`, approximately 63.06%. If capacity is only an
upper limit, the welfare-maximizing policy treats at most this fraction:

```text
q_used = min(q, Phi(1/3))
G_at_most(q) = G_exact(q_used).
```

Beyond that threshold, an instruction to fill capacity allocates treatment
to additional people whose true benefit is negative. The at-most-capacity
policy instead leaves capacity unused, avoiding the associated welfare loss.

## Reproduce

Python 3.12+, no dependencies beyond the standard library:

```bash
python causal_inference_statistics/part20/part20_capacity_frontier.py
python -m unittest discover -s causal_inference_statistics/part20 \
  -p 'test_part20_capacity_frontier.py' -v
```

At 20% capacity, the Gaussian population calculation gives about 0.5199
benefit per population member, consistent with the archived simulation's
0.5198. The comparison uses the original *known* linear DGP and exact
normal integrals. It does not claim that trained CATE models can rank
perfectly, nor that the estimated positive-benefit threshold is certain.

If treatment has a positive per-person resource cost, the welfare cutoff
would be above zero and the optimal treated share smaller. If the policy
requires exact use of capacity, the at-most-capacity option is unavailable.
Other effects such as allocation fairness, risk, uncertainty, and harm
constraints are outside this one-dimensional population model.
