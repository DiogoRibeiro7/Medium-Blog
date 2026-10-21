## Three designs, three different business risks

Making the meal smaller is not, by itself, a complete product decision. The restaurant also has to decide how to tell customers, whether to change the price, and what to offer people who need or want more food. Those choices change the accounting, and they also change the behavioural response that nobody has measured. The model should represent both without pretending the second is known.

The first design is a smaller default portion at the original price, which is the business idea as first stated. No revenue is given up on any retained sale and the purchasing saving raises the contribution per meal. But the price per unit of food goes up: if a €15 dish loses 15% of its mass and keeps its price, the price per gram rises by about 17.65%. That does not mean customers perceive a 17.65% price increase, because what a restaurant sells includes quality, service, location, atmosphere and convenience. It does explain why an unannounced change can be read as worse value even by a customer who would have been perfectly satisfied with the smaller meal.

The second design is a smaller portion with a discount. Now the restaurant gives up some revenue on every retained sale in exchange for a proposition that may be easier to accept. In the illustrative case, a 5% discount on a €15 meal gives away €0.75 per transaction, nearly three times the €0.255 that a 15% smaller portion saves in ingredients. Contribution falls by about €1,387.50 a month before a single customer reacts and after implementation costs. A discount might attract more customers or change the mix of customers, but those effects need their own evidence, and a negative accounting result cannot be rescued by quietly assuming that people will turn up.

The third design is a smaller default portion at the original price with a free refill on request. This trims food for people who do not want the full amount while preserving access for people who do. It brings its own uncertainty. Some customers will ask for the refill, and each refill gives back some or all of the ingredient saving and costs staff time, service equipment and coordination in the kitchen. Refills might also improve how customers judge value and how much they trust the restaurant, but that possible benefit cannot be counted twice, once as an assumed rise in demand and again as a costless service.

The R code generalises the contribution equation to cover all three. Let d be the proportional discount, u the fraction of retained customers who ask for one refill, f the fraction of the ingredient saving that a refill gives back, and h the cost of handling a refill. The ingredient saving after refills, averaged over retained transactions, is

S_refill = c × s × r × a × (1 − u × f).

The contribution per retained meal becomes

m_new = p × (1 − d) − c − v + S_refill − u × h,

and with a monthly loss of meals l and implementation cost I, the change in contribution is

Δ = N × (1 − l) × m_new − N × m − I.

The refill is treated as something with a cost to the operator rather than as free, and a cash discount is kept separate from the quantity of food saved. Two designs with the same plated mass can have different contributions, different implications for repeat visits and different service costs.

With 15% smaller default portions, a quarter of customers asking for a refill, each refill restoring the full saving on the affected ingredients, €0.20 to handle each refill and €200 a month to implement, contribution rises by about €153.13 a month if demand does not change. If half of the customers ask, the gain becomes a loss of about €131 a month before anyone leaves. The refill design is fragile in a different way from the same-price design: it depends less on how many customers leave and more on how many come back to the counter.

![Monthly contribution change against customer loss for three menu designs](figures/fig10_menu_designs.png)
*Figure 10. Change in monthly contribution for the same 15% portion cut under three menu designs, as the share of meals lost rises from zero to 5%. Hypothetical parameters.*

Published portion-size studies are relevant here, but none of them settles the choice between these designs. In controlled experiments in Stanford University dining halls, Voşki and colleagues found that a serving spoon 25% smaller reduced the meat taken by about 18% a day without lowering diners' satisfaction, while a spoon 50% smaller produced no reduction in meat served and a significant fall in satisfaction ([Stanford experiments](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12001533/)). In three Dutch restaurants, Reinders and colleagues cut the meat on main-dish plates by an average of 12.5% and doubled the vegetables; satisfaction with the visit as a whole did not change, satisfaction with the main dish fell slightly, and vegetable consumption rose ([International Journal of Behavioral Nutrition and Physical Activity, 2017](https://doi.org/10.1186/s12966-017-0496-9)). A randomised field experiment by Hua and colleagues in US cafés tested whether relabelling a smaller main course as the standard portion shifts what diners choose ([Appetite, 2026](https://www.sciencedirect.com/science/article/pii/S0195666326000565)).

Each of these answers a nearby question rather than the one here. A smaller option sold for less cannot tell us what happens when the same customer receives less for the same price. A spoon in a university dining hall is not a plated restaurant meal that a paying customer has come to expect. A measure of satisfaction on the day is not a measure of whether that customer returns next month. The Stanford result carries one lesson worth taking seriously all the same: the response to a smaller serving was not proportional. The moderate cut worked and the large one provoked compensation and dissatisfaction, which is exactly the kind of curve the constant-elasticity model in the previous section cannot represent, and a reason to treat any single elasticity as a local approximation.

There is also a fourth design that changes the restaurant's production without changing the customer-facing portion at all. Better demand forecasts, staggered preparation, smaller batches and adjustments to recipes can cut kitchen waste while keeping the menu's promise intact, and Figure 1 suggests that is where two thirds of the waste arises. I cannot claim those strategies always win, but they put far less pressure on the customer-value problem, and they belong in the comparison rather than outside it.

Choosing between these designs is product design under uncertain behaviour, not an exercise in dividing kilograms by meals.
