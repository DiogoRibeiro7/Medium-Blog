# Partial identification of restaurant contribution-margin effects

## Question

Can a restaurant reduce default portion size while holding the menu price
fixed, without losing contribution after allowing for possible lost sales,
uncertain food input savings and implementation costs?

## Identification status

The URBAN-WASTE source has 3,321 recorded observations, but fewer than 12%
have an eligible whole-day portion denominator. Its units and measurement
windows remain unverified. It has no ingredient purchasing prices or customer
demand response. The Finnish intervention concerns school food waste, not
restaurant portions sold at unchanged prices.

Consequently, the uncertainty intervals below are **analyst-assumed
possibilities**, not statistically identified sets calibrated by the food
waste datasets. Bounds are *deterministic and sharp conditional on the
assumed rectangular uncertainty set*, not confidence or credible intervals.

## Model

Let:

- N be the original number of meals sold per month.
- p be the original price (EUR/meal).
- c be ingredient expense (EUR/meal).
- v be other variable expense (EUR/meal).
- m = p-c-v > 0, the baseline contribution per meal.
- r be the proportion by which a default portion is reduced.
- s be the portion of ingredient cost affected by that change.
- a be the fraction of theoretically avoidable ingredient expense that
  actually translates into reduced purchases (0 to 1).
- l be the fraction of meal purchases lost (0 <= l < 1).
- I be incremental implementation expense (EUR/month).

Ingredient saving per retained meal is S = c*s*r*a. The change in monthly
contribution is:

    Δ = N * ((1-l)*(m+S) - m) - I.

All the parameters except p,c,v,N are uncertain within stated intervals.
With m > 0 and nonnegative costs, Δ is monotonic in each uncertain parameter:
it increases with s,r,a, and decreases with l,I. Its lower and upper bounds
therefore occur at opposite corners of the specified rectangle:

    Δ_min = N*((1-l_max)*(m+c*s_min*r_min*a_min)-m) - I_max
    Δ_max = N*((1-l_min)*(m+c*s_max*r_max*a_max)-m) - I_min.

A positive Δ_min means all considered assumptions imply increased
contribution. A negative Δ_max means they all imply a decline. Otherwise
the sign depends on assumptions; equality at zero also belongs to this
last category.

The break-even customer-loss threshold, given S and I, is:

    l* = (S - I/N) / (m+S).

A negative l* means *even unchanged customer demand is insufficient*.
It is not an estimated elasticity, and its interval endpoints are not
evidence-backed predictions.

## Illustrative assumptions (not measured restaurant values)

The initial example uses p=EUR15, c=EUR5, v=EUR2 and N=2,500 monthly meals.
The uncertainty rectangle assumes r in [10%,20%], s in [25%,55%],
a in [50%,95%], l in [0%,5%], and I in [EUR75,EUR250].
No empirical source is claimed for these parameter intervals.

The R scripts export \`profit_assumptions.csv\`, \`profit_bounds.csv\`
and \`profit_frontier.csv\` under the project \`outputs/\` directory.

Run from the project root:

    Rscript tests/test-profit-bounds.R
    Rscript scripts/analyse-profit-frontier.R

The supplied R tests compare all 32 uncertainty-set corners with the
existing \`evaluate_scenario()\` implementation.

## Decision relevance and limits

The model explicitly rules out these unsupported shortcuts:

1. Equating observed food-waste mass with avoidable procurement cost.
2. Treating missing commercial portions as zero waste or as random missingness.
3. Assuming the 21% change in institutional plate waste is a 21% saving
   for a restaurant.
4. Inferring calorie reductions without food composition, or asserting
   universal nutritional benefits from smaller plates.
5. Selecting an empirically optimal r without a demand response function.

When the sign is ambiguous, the frontier displays which hypothetical
customer losses and portion reductions could be profitable, not which
will occur. Future research can obtain independent published ingredient
cost and demand-response bounds and defend those values explicitly.
