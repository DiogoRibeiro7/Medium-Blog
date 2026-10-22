## What uncertainty permits us to conclude

One way to finish the analysis would be to pick a customer-loss figure and announce whether the change succeeds. That produces an answer, but it hides the fact that no reliable estimate of the demand response exists. The alternative is to map the region of assumptions in which the restaurant comes out ahead. That does not turn guesses into observations. It makes the role of each guess visible.

For the example restaurant, let the portion cut range from 10% to 20%, the share of ingredient cost affected from 25% to 55%, the share of the saving that reaches purchasing from 50% to 95%, customer loss from zero to 5%, and implementation cost from €75 to €250 a month. These are ranges chosen by the analyst. They are not confidence intervals estimated from the commercial waste file or from the Finnish school.

Within that box, and as long as the baseline margin is positive, Δ rises with the affected cost share, the size of the cut and the share of saving realised, and falls with customer loss and implementation cost. Because Δ is monotone in every input, its smallest and largest values over the box sit at two opposite corners, and they can be written down exactly. There is no need for a Monte Carlo simulation to find an extreme that algebra delivers directly.

Let S_min = c × s_min × r_min × a_min and S_max = c × s_max × r_max × a_max. The smallest change in contribution over the box is

Δ_min = N × [(1 − l_max) × (m + S_min) − m] − I_max,

and the largest is

Δ_max = N × [(1 − l_min) × (m + S_max) − m] − I_min.

For the example, the saving per retained meal ranges from €0.0625 to €0.5225, and the monthly change in contribution from about −€1,101.56 to +€1,231.25. The interval contains zero, so profitability is not robust over these assumptions: both a gain and a loss are compatible with them. This is not a 95% interval, a posterior distribution or a forecast for any operator.

Which assumption matters most? Varying one input at a time across its range, with the others held at the baseline and no customers lost, answers that directly. Moving customer loss from zero to 5% swings the result by more than €1,000, about twice as much as moving the affected cost share across its entire range, and several times more than implementation cost. The cost-side inputs are things an owner can measure from invoices and recipes. The input that dominates is the one nobody has measured.

![One-at-a-time ranges of the monthly contribution change for each input, against the joint range](figures/fig07_profit_tornado.png)
*Figure 7. Change in monthly contribution as each input moves across its assumed range with the others at the baseline, and the exact range when all five move together. Hypothetical parameters.*

The break-even loss threshold can be bounded the same way. At the unfavourable corner it is below zero, so the change fails even if nobody leaves. At the favourable corner it is positive, so there is some tolerance for lost sales. Repeating the calculation over a grid of portion cuts and customer losses produces a map with three regions: combinations that lose money under every cost assumption in the box, combinations that gain under every one, and a wide band in between where the sign depends on the cost assumptions. The region where the gain is guaranteed is a thin strip along the bottom: cuts deeper than about 15%, and losses of no more than about 1% of meals.

![Classification of portion cut and customer loss combinations by whether the gain or loss holds for every cost assumption](figures/fig08_feasibility_map.png)
*Figure 8. For each combination of portion cut and customer loss, whether the contribution change is positive under every cost assumption in the box, negative under every one, or depends on them. The line is the break-even loss at the baseline cost assumptions. Hypothetical parameters.*

The map does not encode how demand responds to portion size. Treating r and l as independent coordinates lets us inspect possible outcomes; it does not reveal the curve that connects them, and choosing r to maximise contribution without that curve would be a mathematical solution to an invented problem.

## The same threshold, written as an elasticity

There is one more thing the algebra can do without data, and it turns out to be the most useful. A threshold expressed as a share of lost meals is hard to compare with anything, because nobody publishes the share of meals lost after a 15% portion cut. A threshold expressed as an elasticity can be compared with what economists have measured elsewhere.

When the price stays the same and the portion shrinks by r, the price per unit of food rises by r / (1 − r), about 17.65% for a 15% cut. Suppose meals sold respond to that effective price with a constant elasticity ε, which is equivalent to saying that meals sold scale with portion size raised to the power ε. Then the meals kept after the cut are a fraction (1 − r)^ε of the meals sold before, the share lost is l = 1 − (1 − r)^ε, and the change in contribution becomes

Δ = N × [(1 − r)^ε × (m + S) − m] − I.

Setting Δ = 0 gives the largest elasticity at which the cut still breaks even:

ε* = ln[(m + I/N) / (m + S)] / ln(1 − r).

For the illustrative restaurant and a 15% cut, ε* is about 0.147. Each 1% removed from the plate may cost at most about 0.15% of meals sold. Without implementation cost, and for small cuts, the threshold tends to c × s × a / m, the avoidable ingredient cost per unit of portion cut divided by the margin per meal, which here is 1.7 / 8 = 0.2125. What the restaurant can afford is set by its cost structure, not by how deep it cuts: between cuts of 10% and 30% the threshold barely moves, from about 0.13 to a maximum of about 0.154, because the saving and the effective price increase grow together.

How does 0.15 compare with anything measured? The closest evidence I found is not about restaurants at all. Using supermarket scanner data from the Chicago ice-cream market, where several brands shrank their cartons, Çakır and Balagtas estimated that demand responds to package size with an elasticity of about 0.12, roughly a quarter of its response to price, which they put at about 0.5 ([Journal of Retailing, 2014](https://doi.org/10.1016/j.jretai.2013.06.002)). A systematic review of 160 studies by Andreyeva, Long and Brownell found food away from home among the most price-responsive categories, with elasticities of about 0.7 to 0.8 in absolute value ([American Journal of Public Health, 2010](https://doi.org/10.2105/AJPH.2008.151415)).

If restaurant diners reacted to a smaller plate the way Chicago shoppers reacted to a smaller carton, a 15% cut would cost about 1.93% of meals and the example restaurant would gain about €88.92 a month, inside the break-even line but not by much. If they reacted to the smaller plate as they would to an equivalent price rise, at the ice-cream price elasticity of 0.5 the cut would cost about 7.80% of meals and roughly €1,123 a month. At the restaurant price elasticities in the review, the loss would be larger still.

![Break-even elasticity of meals sold to portion size against published elasticity estimates](figures/fig09_elasticity_threshold.png)
*Figure 9. The largest elasticity of meals sold with respect to portion size at which the illustrative restaurant breaks even, with and without implementation cost, against published estimates from other markets. None of the benchmarks measures restaurant diners. Sources: Çakır & Balagtas (2014); Andreyeva, Long & Brownell (2010).*

None of these numbers measures what a restaurant needs. An ice-cream carton is bought once and eaten at home; a plate is looked at, shared, photographed and remembered, and the decision to return is made weeks later. The restaurant response could be weaker than the carton response, because diners rarely weigh their food, or stronger, because the plate sits in full view. What the comparison does establish is where the decision lives. The break-even elasticity sits just above the package-size elasticity from the ice-cream study and well below every price elasticity in the review. A restaurant cutting portions at an unchanged price is betting that its customers treat a smaller plate much more like a smaller carton than like a higher bill. That is a specific, testable claim, and it is a more useful thing to know than a projected annual saving.

Industry data can constrain the cost side in the same spirit. The US National Restaurant Association reported 2024 respondent medians for food and non-alcoholic beverage costs of 32.0% of sales in full-service restaurants and 32.4% in limited-service ones, and among full-service operators 33.7% below US$2 million in annual sales and 31.0% above. The project turns these into clearly labelled cross-market scenarios, but a median across US businesses is not the ingredient bill of a particular dish in Portugal, and the spread of four medians is not a measure of uncertainty about any one restaurant. Nor do the waste files measure purchasing savings or customer response, so those parameters stay as explicit inputs rather than numbers borrowed from data that cannot identify them.

A useful result does not have to be a confident point estimate. A tight threshold can on its own rule out careless business plans, and it tells an operator exactly which quantity a pilot would need to measure.
