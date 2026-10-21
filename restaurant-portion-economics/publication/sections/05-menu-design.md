## Three designs, three different business risks

Reducing the size of a meal is not a complete product decision. The restaurant must also decide how to communicate the change, whether to adjust the price, and how to accommodate customers who need or prefer more food. Those design choices change the accounting, but they also change the unobserved behavioural response. The model should represent both effects without claiming that the response has been measured.

Consider first a smaller default portion at the original price. This is closest to the initial business idea. No menu revenue is surrendered on an individual retained sale, and the procurement saving may increase the contribution per meal. But the relative price per unit of default portion mass increases. If a €15 dish loses 15% of its mass while retaining its price, the nominal price per unit mass rises by about 17.65%. That arithmetic does not prove that customers perceive a 17.65% price increase, because restaurant value includes quality, service, location, atmosphere, and convenience. It nonetheless explains why an unannounced change can be interpreted as worse value even if the customer would have been satisfied by the smaller meal.

The second design offers a smaller portion with a discount. Here the restaurant gives up some revenue on every retained sale in exchange for a potentially more acceptable proposition. In the illustrative baseline, a 5% discount on a €15 meal removes €0.75 of price per transaction. That loss is greater than the assumed €0.255 ingredient saving from a 15% portion reduction. With no change in customers, the incremental contribution is already negative, before implementation costs are considered. Discounting might attract more transactions or change the customer mix, but those effects need separate evidence. One cannot rescue a negative accounting result by silently assuming that customers will arrive.

The third design offers a smaller default portion at the original price, with an optional free refill. This can reduce unnecessary food for people who do not want the full amount while preserving access for people who do. Operationally, it introduces a different kind of uncertainty. Some customers will request a refill. Each refill can restore part or all of the ingredient quantity saved by the smaller default, and it may require staff time, service equipment, or additional kitchen coordination. Refills could also improve perceived value or customer trust, but that potential behavioural benefit must not be counted twice as both an assumed increase in demand and a costless service.

The R code generalises the original contribution equation. Let d be the proportional discount, u the fraction of retained customers requesting one refill, f the fraction of the default ingredient saving restored by each refill, and h the handling cost per refill. The ingredient saving after refills, averaged over retained transactions, is

S_refill = c × s × r × a × (1 − u × f).

The revised contribution per retained meal is

m_new = p × (1 − d) − c − v + S_refill − u × h.

With monthly meal loss l and implementation expense I, the change in contribution becomes

Δ = N × (1 − l) × m_new − N × m − I.

This model treats the option of a refill explicitly instead of regarding it as automatically free to the operator. It also distinguishes a monetary discount from the quantity of food saved. Those differences matter because two interventions with the same plated mass can have different contributions, repeat-purchase implications, and service costs.

Under a hypothetical free-refill scenario with 15% smaller default portions, 25% refill uptake, complete restoration of the affected ingredient saving on refill, €0.20 handling expense for each refill, and €200 monthly implementation cost, the increase in contribution is approximately €153.13 if demand is unchanged. The same assumptions with substantially higher refill uptake could eliminate the advantage. In the project these parameters are scenario inputs; they are not extracted from the Finnish data or from the hotel and restaurant tracking records.

For a discounted portion with a 5% price reduction, the same baseline cost assumptions produce a negative incremental contribution even if not a single customer is lost. That example shows why the intuitive claim that smaller portions permit lower prices while increasing margin is not generally true. The discount may exceed the avoidable cost; the missing demand response then becomes even more consequential.

Evidence on portion size is relevant but does not settle the competition among these designs. Studies have examined smaller serving utensils, portion labels, plant-based choices, and smaller meal options at lower prices. A smaller option sold for less cannot identify what happens when the same customer receives less for the same price. A spoon-size intervention in a university dining hall does not identify how a paying restaurant customer values a familiar plated meal. A simulated menu choice need not represent actual repeat visits. The underlying intervention, population, and price policy must all be kept in view.

A rigorous experiment would randomise sufficiently similar customer opportunities to receive one of the designs, define the relevant time horizon, record total transactions and repeat visits, and quantify costs attributable to preparation and service. That is not something an individual researcher can plausibly conduct at scale without an operator. The present work instead isolates exactly which unknown quantities matter, so that a small commercial pilot—if one becomes feasible—can be designed to learn the right things rather than simply count empty plates.

There may be a fourth design that changes the restaurant's own production without changing the customer-facing portion at all. Better demand forecasts, staggered preparation, smaller batch production, and recipe adjustments can reduce kitchen waste while preserving the perceived promise of the menu. We cannot claim that those strategies are always superior. We can say that they place less immediate pressure on the customer-value problem and should therefore be compared with visible portion reductions rather than excluded from the decision set.

The choice among designs is a problem of product design under uncertain behaviour, not merely an exercise in dividing kilograms by meals. The restaurant should choose a strategy only after understanding where the waste originates, how much cost is avoidable, and which customer commitments the menu must preserve.
