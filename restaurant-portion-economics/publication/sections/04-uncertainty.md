## What uncertainty permits us to conclude

It is tempting to choose a customer-loss figure and announce whether the intervention succeeds. That produces one answer, but it disguises the fact that no reliable demand-response estimate is available. An alternative is to identify the region of assumed parameters in which the restaurant benefits. This does not transform guesses into observations; it makes the role of assumptions explicit.

For the original example, let portion reduction vary from 10% to 20%, the share of ingredient cost affected from 25% to 55%, procurement savings realisation from 50% to 95%, customer loss from zero to 5%, and monthly implementation cost from €75 to €250. These are illustrative uncertainty intervals selected by the analyst, not statistical confidence intervals estimated from the commercial waste file or Finnish school observations.

Within this rectangular set, and assuming a positive baseline contribution margin, the change Δ increases when the affected cost share, portion reduction, or procurement savings realisation increases. It decreases when customer loss or implementation expense increases. This monotonicity yields exact conditional lower and upper bounds from the corners of the assumed parameter rectangle. There is no need for a Monte Carlo simulation to calculate an extremum that follows algebraically.

Let S_min = c × s_min × r_min × a_min and S_max = c × s_max × r_max × a_max. The smallest contribution change over the rectangle is

Δ_min = N × [(1 − l_max) × (m + S_min) − m] − I_max,

and the largest is

Δ_max = N × [(1 − l_min) × (m + S_max) − m] − I_min.

For the example, the smallest saving per retained meal is €0.0625 and the largest is €0.5225. The resulting monthly contribution interval is approximately −€1,101.56 to +€1,231.25. Since it spans zero, profitability is not robust over this assumed parameter set. The classification means only that both profit and loss are compatible with our assumptions. It is not a 95% interval, a posterior distribution, or a forecast for any operator.

The break-even customer-loss threshold is likewise bounded over the scenario rectangle. At the unfavorable corner it is below zero, indicating that the proposed change fails even without losing customers. At the favorable corner it is positive, indicating some tolerance for lost sales. These contrasting outcomes are exactly why presenting a single projected annual saving as the value of the idea would be misleading.

A particularly important distinction is that the rectangles do not encode how demand responds to portion size. A larger portion reduction may increase procurement savings, but it may also cause a larger change in customer perception or repeat purchases. Treating r and l as independent scenario coordinates lets us inspect possible outcomes, not infer the curve connecting them. Optimizing r without such a curve would be a mathematical solution to an invented problem, not an empirical recommendation.

Industry data can sometimes help constrain the plausible magnitude of costs. The US National Restaurant Association reported 2024 respondent medians for food and non-alcoholic beverage expenses of 32.0% of sales in full-service restaurants and 32.4% in limited-service restaurants. For full-service establishments in different annual sales categories, its reported medians were 33.7% for businesses below US$2 million and 31.0% for businesses at or above that threshold. These are useful external descriptions of operating cost structures, but they are not universal bounds, and their denominator is business sales rather than meals.

The R project records the original links and creates transparent sensitivity calculations if such ratios are applied to a hypothetical meal price. Those calculations are labelled cross-market proxy scenarios. They do not estimate a Portuguese restaurant's ingredient bill. It would be unsound to treat a median across US businesses as the price of ingredients in a particular dish, or to infer its uncertainty by taking the range of those four published medians.

The same discipline applies to food-waste observations. The commercial dataset does not measure procurement savings caused by smaller portions. The Finnish dataset cannot identify restaurant customers' willingness to continue paying the same amount for smaller meals. Combining them numerically would not generate the missing causal link. The model therefore keeps those parameters as uncertain inputs even after extensive data collection and analysis.

A useful result need not be a confident point estimate. Identifying a small customer-loss threshold can itself rule out careless business plans. It can motivate better communication, gradual menu redesign, optional refills, or a more controlled operational change. A conclusion that depends on an unobserved response is still informative when that dependency is made visible.
