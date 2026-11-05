# Part 18 code companion: missing-outcome tipping points

The original Part 18 experiment compares complete-case regression, response
weighting, and outcome regression under MCAR, MAR, and MNAR. This separate
analysis makes the missing-outcome identification assumption explicit through
a **pattern-mixture sensitivity parameter** for each randomized treatment arm.

It does not change `experiment.py`, `results.csv`, or `article.md`.

## Exact decomposition

Within arm `a`, let `r_a` be the fraction with observed outcomes and `m_a`
the observed (respondent) outcome mean. Define an unknown shift
`delta_a = E[Y|A=a,R=0] - E[Y|A=a,R=1]`. The full-data arm mean is exactly

```text
mu_a = m_a + (1-r_a)*delta_a

mu_1 - mu_0 = (m_1 - m_0) + (1-r_1)*delta_1 - (1-r_0)*delta_0
```

The expression is an identity for realized arm means when the assumed shifts
equal the actual respondent/nonrespondent differences. Without those missing
outcomes, it is a *sensitivity calculation*, not an identified estimate.
Under random assignment the full-data arm-mean contrast estimates the ATE,
but even the full-data contrast is not exactly the finite-sample individual
causal effect.

Given separately assumed intervals for both shifts, the lower contrast uses
the smallest treated shift and largest control shift; the upper contrast uses
the opposite endpoints. These bounds are conditional on the stated intervals.
They are not empirical confidence intervals and do not account for sampling
uncertainty or correlations between estimated response fractions and means.

If the treated response fraction is less than 1, the treated shift that makes
the completed contrast exactly zero at a fixed control shift is

```text
delta_1^* = [-(m_1-m_0)+(1-r_0)*delta_0]/(1-r_1).
```

This is a **tipping point**, not an assertion that the missing participants
actually have that outcome mean.

## Reproduce

Run with Python 3.12 or newer, using only the standard library:

```bash
python causal_inference_statistics/part18/part18_pattern_mixture.py
python -m unittest discover -s causal_inference_statistics/part18 \
  -p 'test_part18_pattern_mixture.py' -v
```

The script uses *illustrative* respondent means 2.0 and 1.2, response
fractions 0.55 and 0.70, and assumed missing-minus-observed shifts between
−2 and 0 in each arm. The observed contrast is 0.8, but the completed
contrast ranges from −0.1 to 1.4. Holding the control shift at zero, a
treated shift of roughly −1.78 produces a zero completed contrast.

These numbers are not outputs from Part 18's Monte Carlo and must not be
presented as its empirical results. Without credible limits on missing-outcome
differences, the observed outcomes alone cannot yield finite numerical bounds
for the complete-data contrast. For observational treatment, randomized-arm
interpretation also fails unless confounding is addressed separately.
