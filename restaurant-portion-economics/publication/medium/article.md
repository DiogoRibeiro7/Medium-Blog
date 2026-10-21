# The Economics of the Uneaten Meal

*Can restaurants improve their margins by serving less food without charging less? An investigation of food-waste data, customer value, and the mathematics of an uncomfortable trade-off.*

A customer orders a meal. The restaurant purchases the ingredients, stores them, prepares the food, serves it, and pays people to complete each stage. Some of the food may never leave the kitchen. Some may reach the table but never be eaten. The restaurant has paid for both. The customer, meanwhile, has purchased a meal rather than a specified number of grams. Those elementary facts suggest a business question that deserves more attention than it usually receives: how much of the quantity served is necessary to deliver the value for which the customer is paying?

There is a tempting answer. Reduce the size of the meal, keep the price unchanged, and retain the difference in ingredients and preparation costs. Add the reduction in discarded food and perhaps the kitchen earns more while producing less waste. It sounds like an easy improvement to operating margin. It is not. It is a hypothesis with a cost side, a demand side, and a missing empirical link between them.

Consider two customers paying the same price. One values the quality of the ingredients and would happily receive a slightly smaller serving. The other regards the portion as a measure of what a restaurant owes in return for the money. Their responses to a reduction need not be similar. Even the first may object if the restaurant changes the menu without explaining why. A saving of a few cents per dish can disappear when enough customers stop coming. A better model must therefore ask not only how much less food a restaurant could prepare, but how many transactions it can afford to lose.

There is also a nutritional complication. A meal's mass, its energy content, and its quality are different objects. Reducing a large portion of fried potatoes is not the same intervention as reducing the protein, vegetables, or energy availability of a meal that was already inadequate. Nor is it appropriate to assume that everyone benefits from eating fewer calories. The relevant target is unnecessary production relative to customer needs, not indiscriminate restriction. Good portion management should allow people who require more food to obtain it.

The scale of the issue is real. The United Nations Environment Programme estimated that 1.05 billion tonnes of food waste were generated in 2022 at household, food-service, and retail levels, including inedible parts. Food services accounted for approximately 290 million tonnes. The figures describe a global systems problem, not an estimate of what any particular restaurant could save. They nevertheless make the sector a reasonable place to investigate the economic role of serving decisions ([UNEP, Food Waste Index Report 2024](https://www.unep.org/en/news-and-stories/press-release/world-squanders-over-1-billion-meals-day-un-report)).

My starting point was deliberately practical. I do not own a restaurant, have privileged access to restaurant point-of-sale systems, or command a field-research team. I can, however, use public datasets, inspect the observation processes, construct a defensible accounting model, and identify the precise conditions under which a smaller portion could be commercially attractive. That approach is narrower than claiming to solve food waste. It is also more honest than reporting an imagined number for monthly savings.

The work accompanying this article is written in R, in the [Medium-Blog restaurant-portion-economics project](https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/restaurant-portion-economics). It uses publicly archived waste observations from European hotel and restaurant kitchens and from one Finnish school. It combines these with industry cost benchmarks and a set of explicit, adjustable business assumptions. The data and the assumptions are never treated as interchangeable. Where the data cannot identify an effect, the analysis says so.

Three questions organise the investigation. First, what does the public evidence actually measure about food waste, and which records are usable? Second, how does the cost of serving food enter a restaurant's contribution margin? Third, what customer-demand response would make a smaller portion profitable, unprofitable, or genuinely uncertain? We can answer the second question mathematically. The first depends on careful data auditing. The third remains the central commercial uncertainty, and that is where our conclusions must stop.

This is not a proposal for restaurants to disguise price increases by quietly serving less. That strategy can destroy the trust on which repeat business depends. It is a proposal to understand food quantity as a design variable, alongside recipe composition, price, service style, and purchasing. If an efficient default serving could deliver the same valued experience with less unnecessary material, the economic and environmental interests might align. The task is to determine when they do, and when they do not.

## What exactly are we trying to reduce?

The first mistake in food-waste analysis is to treat all discarded food as if it came from portions that were too large. Food can be lost during procurement, storage, preparation, serving, and consumption. Those pathways have different causes and different solutions. A tomato spoiled before preparation will not be saved by reducing the rice served with a main course. A large quantity of pasta left on a customer's plate might respond to a change in default size. Kitchen trimmings may respond to preparation technique, recipe design, or supplier specification. A model that merges these outcomes implicitly assumes a common mechanism that may not exist.

Imagine a restaurant that purchases 100 units of a particular ingredient. Some of that quantity may be discarded in storage; some becomes inedible preparation residue; some is converted into saleable food; some of the served amount is later discarded by customers. We can describe the material balance, symbolically, as follows:

Purchased mass = storage losses + preparation losses + served mass + other inventory changes.

Served mass = consumed mass + customer leftovers + other serving losses.

These are accounting identities, provided that the boundaries are defined and quantities are measured consistently. They are not estimates of a causal effect. They merely force us to distinguish where a change occurs. If the kitchen cuts its default plated quantity, it may reduce the amount prepared. But it may also replace customer leftovers with more food held at the end of service. Then the accounting location of waste changes without the quantity purchased declining.

For a restaurant's profit, the relevant monetary variable is avoidable purchasing or production cost, not discarded mass by itself. A kilogram of cheap starch, a kilogram of fish, and a kilogram of mixed scraps differ in replacement cost. If the same percentage of waste is removed from each category, the financial effect is different. The value of preventing spoilage can also include purchasing administration, cooling, storage, and disposal expenses, but those components should be introduced only when they are actually avoidable and measured. A disposal charge already fixed for the month is not saved because one container is marginally lighter.

The second mistake is to confuse the amount served with the amount eaten. A smaller default portion may reduce food served, but a customer can ask for additional food, order a side, take food home, or compensate later. The environmental interpretation depends on what happens to total production and consumption, not merely what remains on one plate. In institutional canteens, the response to a smaller ladle or spoon may differ from the response to changing a restaurant's advertised portion. The mechanism and the decision-maker are not identical.

The third mistake concerns calories. The total mass of a dish does not reveal its energy content. Water-rich vegetables can contribute substantial weight with relatively few calories; oils and sauces can contribute relatively little mass while containing considerable energy. Ingredient composition, edible yield, serving size, and the actual amount consumed are required before making nutritional claims. A cost model built from waste weights alone cannot estimate the energy saved. For this reason, the present study keeps nutrition as a constraint and an unresolved empirical module, rather than producing decorative calorie figures.

The appropriate unit of analysis may also change with the decision. A chef deciding how much fish to portion needs ingredient-level data. A restaurant owner choosing whether to offer a smaller lunch needs per-transaction contribution and customer behaviour. A municipal policymaker estimating waste diversion needs material flows aggregated over establishments. These are legitimate but different estimands. Treating them as one number produces attractive dashboards and poor decisions.

Our project focuses on contribution to the restaurant after a hypothetical portion intervention. To interpret public waste data alongside that question, we first audit whether records are measurements of waste, what kind of waste they describe, whether their denominators are observed, and whether records from different establishments are comparable. Even if all those conditions are satisfied, waste data alone do not answer the demand question. They supply information about a potential physical problem, not the full commercial response to its proposed solution.

A good business idea can fail because the intervention addresses the wrong part of the process. If the most expensive waste occurs during preparation, reducing what reaches the plate may have little effect. If customer leftovers are concentrated in a cheap side dish, reducing that component may save little money but alter the perception of generosity. If ingredients can be prepared in smaller batches without affecting the dining experience, the best improvement may be operational rather than visible. This is why the analysis begins with the flow of food rather than a universal target of 10%, 15%, or 20% smaller meals.

The economic question is therefore more specific than "can a restaurant waste less?" It is: can a change in the default quantity and composition served to a customer reduce avoidable variable costs sufficiently to offset its implementation expense and any change in demand? The public data can illuminate pieces of that question. They cannot make the unobserved pieces disappear.

## What the public data can tell us

The first source is the URBAN-WASTE Food Tracking Device dataset, published by Eriksson and Lindgren in 2019 and archived on Zenodo (record 3355300). It contains measurements collected in hotels and restaurants. The original file is downloaded by an R script, checked against the publisher's checksum, and analysed without placing a redistributed copy in the repository. This provenance matters: a result should be traceable to a particular source, not to an undocumented spreadsheet circulating between articles.

The ingestion itself required several iterations. An early parser misread the delimiter and retained only 1,586 rows from 29 kitchen labels. The corrected, checksum-verified parser recovered 3,321 rows from 34 labels. Had we published the first summary, the numerical story would have been wrong before any statistical model was fitted. An analysis pipeline that passes a unit test on a synthetic CSV can still misinterpret a real CSV. A reproducible result needs source-file validation as well as functional tests.

The more consequential difficulty is missing data. Of those 3,321 observations, 380 contain the information required for a candidate whole-day weight-per-portion ratio: a reported waste weight, a positive whole-day portion count, and usable kitchen and timestamp labels. That is 11.44% of the original records. The selected records belong to 25 of the 34 kitchens. The remaining kitchens are not necessarily waste-free, nor should their missing denominators be treated as zero. The reason a count is absent may depend on kitchen, service type, recording practice, or other characteristics also associated with waste. We have no evidence that the missingness mechanism is random.

Selection is uneven even within the usable sample. One kitchen accounts for 121 of the 380 eligible rows, approximately 31.84%. Five kitchens account for 292, or 76.84%. The eligible records contain only about 7.71% of the source's recorded waste weight. Each quantity describes the *observed file*, not an underlying population of restaurants. It would be indefensible to label the ratio computed from the selected rows an average European restaurant's waste per meal.

A pooled ratio of recorded waste to whole-day portions is approximately 0.24353 in the file's unverified weight units per reported portion. The median of individual row ratios is around 0.16125. Removing rows above the empirical 95th percentile of row-level ratios changes the pooled ratio to about 0.20644. Leaving out one kitchen at a time changes it to values approximately between 0.20418 and 0.33135. These different answers do not establish which answer is correct. They describe how the numerical summary depends on weighting, outliers, and the restaurants represented.

It would be tempting to attach a kilogram label to those numbers and multiply by food cost. We do neither. The available source documentation does not securely establish the field's weight unit, the waste stage, or whether its observation window aligns with the whole-day portion count. Until those definitions are checked, the ratio is a structural diagnostic rather than a credible physical measurement per meal. Its uncertainty is not repaired by adding extra decimal places, a confidence interval, or a more complicated regression.

The second source is a Finnish school-catering dataset published by Kuusisto, Latva and Wilén in 2026 (Zenodo record 20825233). It reports 155 school days of plate waste, kitchen/serving waste, and numbers of diners. The file distinguishes 135 baseline days and 20 intervention days, with a repeating 25-day menu. Unlike the commercial CSV, it provides mass and per-diner units. It nevertheless has a different setting and intervention from the business idea being tested.

Using the published per-diner figures and matching the intervention days to baseline observations for the same menu, plate waste is approximately 21.73% lower during the intervention period, and kitchen/serving waste approximately 52.57% lower. These are descriptive comparisons, not estimated effects of shrinking meal portions. The intervention relates to school food waste; the observations cannot tell us how restaurant customers would respond to paying the same price for less food.

The quality audit also revealed a consequential discrepancy on intervention day 136. The recorded plate waste is 11.633 kilograms and the reported number of diners is 340, implying about 34.21 grams per diner. Yet the workbook's reported rate is 8.25 grams per diner. For kitchen/serving waste, the recorded mass is 2.410 kilograms and the reported rate is 21.77 grams per diner, whereas the mass and diner count imply about 7.09 grams. We cannot decide which recorded value is authoritative merely by preferring the one that agrees with a desired conclusion.

The R analysis therefore reports three versions: published per-diner values, exclusion of discordant days, and rates reconstructed from the reported masses and diner counts. Reconstructed rates change the plate-waste comparison to approximately 16.34% below the matched baseline; omitting the flagged day gives approximately 16.39% below. The direction remains negative, but its magnitude changes materially. For kitchen/serving waste, the mass-derived change is approximately 55.81% below the matched baseline, against 52.57% from the published rates. Both sensitivity analyses are reported, not substituted silently for the published data.

A reader should notice the methodological lesson. Public data are accessible, but accessibility is not synonymous with comparability, and a measurement is not automatically an identified intervention effect. The responsible sequence is to verify files, understand the observation process, inspect missingness and measurement units, disclose inconsistencies, and only then consider estimation. Our commercial data confirm that food-waste observations exist, while making the limits of their interpretation unusually visible. The Finnish observations suggest that waste rates can vary substantially across periods and data treatments. Neither dataset tells us how many paying restaurant customers a portion reduction would lose.

## The contribution-margin equation

Suppose a restaurant sells N meals per month at price p. Let c be its ingredient cost per meal and v the remaining variable expense attributable to serving that meal. The original contribution per meal is m = p − c − v. Fixed operating costs are relevant to the survival of the business, but when they truly remain unchanged between alternatives, they cancel from the incremental comparison. We study contribution rather than pretending to estimate net profit from unavailable accounts.

Let r be the fraction by which the default portion is reduced. Not every ingredient is affected. Some costs belong to sauces, garnishes, minimum purchasing units, fixed recipe components, packaging, and items that the restaurant should not reduce. We therefore let s represent the share of ingredient cost to which portion reduction actually applies. Even for that share, smaller preparation does not always translate into a smaller purchasing bill. Let a be the fraction of the theoretical reduction realised in actual procurement.

The avoidable ingredient expense per retained customer in this accounting model is:

S = c × s × r × a.

The equation is deliberately transparent. It says neither that every restaurant can save this amount nor that observed discarded food directly supplies any of its parameters. Each factor has a physical or operational meaning that could eventually be measured in a particular restaurant.

Now suppose that the new portion reduces the number of meals purchased by a proportion l. Let I be monthly implementation expense, including, where appropriate, revised training, menu materials, changes to kitchen processes, or other incremental expenses. The original monthly contribution is N × m. Under the intervention it is N × (1 − l) × (m + S) − I. The incremental result is therefore

Δ = N × [(1 − l) × (m + S) − m] − I.

Expanding the brackets reveals the conflict:

Δ = N × [S − l × (m + S)] − I.

The first term is the procurement saving that would occur if demand remained constant. The second removes the contribution that disappears when a fraction of sales is lost, including the savings no longer available on those lost transactions. The third term is implementation cost. At this level of abstraction, the model is an accounting identity conditional on the assumptions; it is not a fitted behavioural relation.

A restaurant breaks even when Δ = 0. Solving for the maximum admissible proportional customer loss gives

l* = (S − I/N) / (m + S),

provided m + S is positive. If l* is negative, even retaining every customer would not make the change worthwhile at the assumed savings and implementation cost. If l* is positive but small, the intervention depends on customer acceptance to a degree that can outweigh the apparent food saving. An operator may find the value more useful than a claim that portions should universally be reduced by a particular percentage.

Consider an illustrative restaurant selling 2,500 meals per month at €15 each. Assume €5 in ingredients and €2 of other variable expense per meal. The contribution per meal is €8 before fixed costs. Now assume a 15% reduction in default portion size, affecting 40% of ingredient costs, with 85% procurement savings realisation and €150 monthly implementation expense.

The projected saving per retained meal is €5 × 0.40 × 0.15 × 0.85 = €0.255. With no customer loss, the monthly increase in contribution is 2,500 × €0.255 − €150 = €487.50. With a 1% sales loss, the increase is approximately €281.13. With a 2% loss it is approximately €74.75. With a 5% loss, the change is approximately −€544.38. These are arithmetic scenarios, not observed restaurant outcomes.

The break-even threshold in this example is about 2.36% of monthly meal transactions. This means that a relatively modest demand response could absorb all the projected savings. It does not mean that reducing portions by 15% *will* lose 2.36% of customers. Estimating that relationship requires evidence we do not possess.

Nor should those input assumptions be interpreted as Portuguese restaurant benchmarks. They are chosen to make the model inspectable. We later evaluate US industry cost-to-sales medians as external sensitivity anchors, but a business-level food-and-non-alcoholic-beverage ratio is not an ingredient cost for a particular €15 meal. Translating it directly into a restaurant's accounts would confound the unit of observation and the composition of cost.

This is the point at which the business proposition becomes mathematically useful. A restaurant owner need not first accept a sweeping sustainability argument. The question can be stated in operational terms: What is the actually avoidable ingredient cost per retained meal, and how much transaction volume may be lost before the intervention stops paying? Those two variables determine much of the answer, subject to implementation costs and other changes in variable expense.

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

## What can a single independent researcher offer a restaurant?

After several rounds of public-source validation, descriptive analysis, and economic modelling, we arrive at an awkward but useful conclusion. The biggest unknown for the original proposition is not the algebra. It is the behavioural response to a visible reduction in what a customer receives at an unchanged price. A restaurant may be able to save on ingredients, but the value of the saving depends on how many meals it continues to sell.

For an independent researcher without access to proprietary accounts, it would be misleading to advertise a precise percentage improvement in restaurant margin. Neither the 3,321 commercial food-waste records nor the school-catering intervention supplies the missing customer-demand and cost data. Even the commercial waste-per-portion statistics require caution because the denominator exists for a narrow, unevenly distributed subset and the physical unit has not been established conclusively. The right product at this stage is a transparent decision framework, not a promise of verified savings.

There is nonetheless a serviceable commercial proposition. A restaurant can be asked to compare its current menu with feasible alternatives using information it already possesses: number of meals sold, recipe quantities, invoice-level ingredient costs, portions prepared and discarded, customer complaints, and repeat sales where available. It need not buy a new sensor platform or install an elaborate machine-learning system before learning whether the economics are plausible. Its own existing accounting and menu data would usually be more relevant than a cross-country benchmark.

The first task would be to select a small number of high-volume dishes, establish ingredient-level avoidable costs, and distinguish kitchen waste from plate waste. The second would be to identify modifications unlikely to compromise nutritional adequacy or the dish's identity. The third would be to examine break-even demand thresholds, allowing implementation and serving costs to remain explicit. Only if the possible upside is material relative to the uncertainty would it make sense to test an intervention.

This sequence can be offered as a diagnostic rather than as a claim to optimise the entire restaurant. A hypothetical owner of the €15-meal restaurant in our example learns that a €0.255 saving per retained meal, after €150 monthly implementation expense, tolerates only about a 2.36% reduction in transactions. That is not a decision to shrink the meal. It is a threshold against which customer-research evidence, trial results, or informed operating judgement can be compared.

Several practical risks would need attention before a real-world trial. Customers may change their ordering habits, share dishes differently, or substitute more expensive sides. Staff may compensate for smaller servings by giving extra sauce or garnish. Food purchased in bulk may not decline in proportion to portion size. Waste removed from plates may reappear as end-of-service leftovers if preparation schedules do not change. The relationship between portion size and demand may differ by dish, daypart, restaurant positioning, and regular versus occasional customers. A single monthly average conceals those distinctions.

Customer communication is part of the intervention. If the restaurant silently reduces portions while continuing to advertise the old dish, it risks damaging perceived fairness, even if many customers would have chosen the smaller quantity themselves. Offering a clear smaller default, flexible refill, or genuine choice may be more acceptable, but these strategies change cost and service conditions. A sustainability narrative cannot substitute for an actual customer experience. Reducing waste should not become a euphemism for selling materially less under an unchanged promise.

Nor is nutritional quality optional. The optimal serving in a narrow profit equation is not necessarily appropriate for a person with higher energy needs, a child, an older person at nutritional risk, or someone whose circumstances make additional food essential. A responsible menu policy would evaluate energy and nutritional composition separately from mass and would allow genuine variation in needs. None of our empirical datasets is sufficient to produce those dietary judgements for individual customers.

This is also a reminder that a mathematical model is useful partly because it can tell us when it lacks the evidence for an answer. We should not add increasingly elaborate statistical methods simply because the repository can accommodate them. A regression of waste against kitchen labels will not reveal what a customer would have bought under a different portion size and price. A sophisticated prediction of food waste cannot identify the profit from changing the menu without modelling purchase behaviour. More computation does not repair a missing identification strategy.

## What the results establish—and what they do not

The public commercial tracking dataset contains genuine observations from food-service kitchens. Its initial ingestion was sufficiently fragile that the wrong delimiter produced a substantially truncated file, demonstrating why provenance and source-level checks matter. Once corrected, only 380 of 3,321 records were suitable for even a provisional whole-day waste-to-portion diagnostic, and those were concentrated in a subset of kitchens. The data therefore cannot be used to announce an average cost saving per restaurant meal.

The Finnish dataset provides a different kind of evidence, with clearly reported mass and diner counts. Its descriptive intervention-period differences are sizeable, but the treatment was not a commercial fixed-price portion reduction. The internal disagreement between some recorded masses and per-diner rates changes the estimated plate-waste difference from approximately 21.73% to approximately 16.34% under one alternative reconstruction. A credible article must show that discrepancy rather than choose whichever result sounds more compelling.

Our mathematical model establishes a conditional fact. If a reduction in default serving size creates actual avoidable ingredient savings, and if the resulting contribution on retained sales exceeds lost contribution and implementation expense, then the intervention improves monthly contribution. That sentence sounds obvious once stated, but its thresholds can be operationally decisive. In the illustrative restaurant, only a small loss in transactions is needed to offset apparently attractive ingredient savings.

The sensitivity and partial-identification calculations establish another conditional fact: over a reasonably broad *assumed* set of procurement, implementation and customer-response parameters, both a positive and a negative business result are possible. The model can identify break-even surfaces. It cannot provide the probability of falling on either side without additional evidence. It does not establish an optimal universal portion size or a guaranteed way to improve profitability.

The next empirical step should not be another arbitrary model feature. It should be improved source definition, credible food-cost detail, and defensible information about customer response to portions sold at unchanged prices. For an independent researcher, that may mean systematic review and transparent sensitivity rather than a large original field experiment. For an interested restaurant, it may mean a small operational pilot with a controlled comparison and explicit customer safeguards.

The underlying idea remains worth investigating. Restaurants pay for food that customers do not necessarily need or want, and waste reduction may create room for better margins. But the economically relevant quantity is not simply food discarded or calories removed. It is avoidable cost net of behaviour change, service effects, and implementation expense. Any honest business case must place the customer in the equation.

The question is not whether restaurants could serve less. They can. The question is whether they can serve *the right amount*, reduce unnecessary production, and maintain the value that makes customers return. That is a much more demanding proposition—and a more useful one to analyse.

## References and reproducibility

- United Nations Environment Programme. *Food Waste Index Report 2024*. https://www.unep.org/resources/global-food-waste-index-report-2024
- Eriksson and Lindgren. URBAN-WASTE Food Tracking Device dataset (2019). Zenodo record 3355300: https://zenodo.org/records/3355300
- Kuusisto, Latva and Wilén. Finnish school food-waste observations (2026). Zenodo record 20825233: https://zenodo.org/records/20825233
- National Restaurant Association. Food cost ratios across restaurant formats (2024 observations): https://www.restaurant.org/research-and-media/research/restaurant-economic-insights/analysis-commentary/restaurant-operators-kept-food-cost-ratios-in-check-in-2024/
- National Restaurant Association. Differences by restaurant sales volume (2024 observations): https://restaurant.org/research-and-media/research/restaurant-economic-insights/analysis-commentary/higher-volume-restaurants-reported-lower-food-cost-ratios-in-2024/
- The peer-reviewed portion-size studies, original source links, designs and pricing qualifications are recorded in data/reference/portion_response_evidence.csv in the project repository.
- Code, protocols and source audit: https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/restaurant-portion-economics

All money figures in the example are explicitly hypothetical. The full R reproduction and its regression tests are available in the project. The dataset analysis and the financial assumptions are separate components, and neither is presented as a causal estimate of restaurant profitability.
