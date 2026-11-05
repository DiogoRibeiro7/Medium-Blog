# Part 17 companion: negative-control overlap and diagnostic power

This code-only extension continues the negative-control experiment without
altering `experiment.py`, its `results.csv`, or the article manuscript. It asks
how much the negative-control outcome must share the actual unmeasured
confounder before a nominal 5% association test is likely to reject.

## Analytical design

Draw independent standard normal `C`, `V`, `e_A`, `e_Y`, and `e_N`:

```text
A = C + e_A
Y = A + 1.5*C + V + e_Y
N = rho*C + sqrt(1-rho**2)*V + e_N,  -1 <= rho <= 1
```

The intervention effect on `Y` is one; the observed slope of `Y` on `A` is
always 1.75 and its bias is always +0.75. `N` is never causally affected by
`A`. At `rho=1`, it detects the confounder as in the shared-confounder case;
at `rho=0`, it shares no confounder with treatment, as in the
outcome-specific-confounder case. The coefficient `rho` is a structural
*alignment parameter*, not the correlation between `A` and `N`.

Since `Var(A)=Var(N)=2` and `Cov(A,N)=rho`, the negative-control regression
has population slope `rho/2` and conditional residual variance
`2-rho**2/2`. For sample size `n` the leading normal-approximation standard
error is `sqrt((2-rho**2/2)/(2*n))`. The reported detection probability is
the normal two-sided Wald rejection probability at the specified nominal
level; it is **not** a simulated finite-sample rejection fraction.

## Run

Requires Python 3.12 or later; no third-party dependencies:

```bash
python causal_inference_statistics/part17/part17_control_sensitivity.py
python -m unittest discover -s causal_inference_statistics/part17 \
  -p 'test_part17_control_sensitivity.py' -v
```

At `n=1200`, diagnostic power is the nominal 5% when `rho=0`, grows with
absolute alignment, and approaches one when `|rho|=1`. The causal bias in
the primary regression remains +0.75 throughout. Consequently, a failed
negative-control test cannot bound causal bias without an independently
justified assumption relating the control to the confounding process.

This benchmark assumes independent Gaussian observations, a valid control
exclusion restriction, a correctly specified intercept/slope regression,
one predeclared two-sided test, and an asymptotic normal approximation.
It makes no claim about empirical datasets, post-selection inference, or
control validity when the design assumptions fail.
