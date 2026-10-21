# Break-even quantity elasticity

`R/quantity_response.R` restates the break-even customer loss as an
elasticity, so it can be compared with published demand estimates.

With the price unchanged, a portion cut `r` raises the price per unit of
food by `r / (1 - r)`. If meals sold follow a constant elasticity `eps`
with respect to portion size, retained meals are `(1 - r)^eps` of the
baseline and

```text
Delta = N * ((1 - r)^eps * (m + S) - m) - I
eps*  = log((m + I/N) / (m + S)) / log(1 - r)
```

For small cuts and `I = 0`, `eps*` tends to `c * s * a / m`.

For the illustrative restaurant (`example_assumptions()`):

- `eps*` at a 15% cut is 0.147; between 10% and 30% it ranges from 0.129 to
  a maximum of 0.154; the no-implementation limit is 0.2125.
- At the ice-cream package-size elasticity of 0.12 (Çakır & Balagtas, 2014)
  a 15% cut loses 1.93% of meals and gains €88.92 a month; at the same
  study's price elasticity of 0.5 it loses 7.80% of meals and about €1,123.

`pilot_weeks_needed()` gives the weeks per arm needed to detect the
break-even loss (2.36%) with a two-sided 5% test and 80% power, treating
weeks as independent: about 69 weeks at a 5% unexplained weekly CV and
about 273 at 10%.

None of the benchmarks measures restaurant diners, and the constant
elasticity is a local approximation: the Stanford spoon-size experiments
found a non-proportional response to larger cuts. These are break-even
conditions and calibrations, not estimates of restaurant demand.
