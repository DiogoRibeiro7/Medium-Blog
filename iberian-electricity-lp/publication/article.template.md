# The Electricity Price Is a Shadow Price

*{{n_programmes}} linear programmes built from the Iberian market's published bids, and what they say about the price*

*Data: OMIE (OMI-Polo Español, S.A.), [omie.es](https://www.omie.es).*

Every day, by noon, buyers and sellers of electricity in Spain and Portugal send OMIE, the Iberian market operator, their bids for each quarter-hour of the next day, and an algorithm sets one price for each quarter-hour. A textbook says what that price should be. If every bid were divisible and every quarter-hour cleared on its own, clearing the market would be a linear programme, and the price would be the shadow price of its one constraint: that supply equals demand.

The real clearing is not that programme. Spain and Portugal are cleared together with the rest of Europe by an algorithm called EUPHEMIA, and some Iberian sell orders are indivisible or tie quarter-hours together. But OMIE publishes every bid, accepted or not, so the textbook can be put to a test. I took {{n_days}} days from the first year of quarter-hourly prices, {{first_day}} to {{last_day}}, and solved {{n_programmes}} linear programmes built from the published bids. The design was registered before any bid was compared with a price, and every number below is read by code from the result files in the [project repository](https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/iberian-electricity-lp).

In short:

- **The price is a shadow price.** Once the outcome of the indivisible orders and the flows across the borders are taken as given, the published price was a shadow price of the remaining linear programme in every one of the {{n_periods}} quarter-hours tested, in Spain and in Portugal alike.
- **The indivisible orders matter.** Treating them as ordinary divisible bids would have lowered the price by {{e2_drop}} €/MWh on average, against my expectation of no change. Almost all of that comes from one kind of order, whose owner can require it to cover a fixed cost such as a plant's start-up.
- **Zero-priced supply lowers the price, but not a price already at or below zero.** {{e3_mw}} MW more supply at zero price would have lowered the price by {{e3_drop}} €/MWh on average. The effect is smaller around midday because {{nonpos_midday_pct}} of midday prices were already at or below zero; among prices above zero, no midday difference was established.

## A market is a linear programme

Take one quarter-hour. A sell bid is a step: up to q MW at a price of at least p. A buy bid is a step: up to Q MW at a price of at most P. Accept s MW of each sell step and b MW of each buy step to

> maximise Σ P·b − Σ p·s, subject to Σ s − Σ b = R and 0 ≤ s ≤ q, 0 ≤ b ≤ Q for every step.

The objective is the buyers' bids minus the sellers' offers for what is traded: the gains from trade. R is the volume fixed in advance, such as exports minus imports. There is one constraint, so there is one dual variable λ, the shadow price: how much the best objective changes per extra MW the constraint requires.

Complementary slackness, the condition linking an optimal solution to its dual, makes λ concrete. A value λ is an optimal dual exactly when every sell step priced below λ is fully accepted, every sell step priced above λ is rejected, the same holds in reverse for buy steps, and only a step priced exactly at λ may be accepted in part. That is the rule of a uniform-price auction, so a shadow price is a clearing price, and a clearing price is a shadow price.

The shadow prices form an interval, not always a single point. If one step is partly accepted, its price is the only shadow price. If none is, any price between the dearest accepted step and the cheapest rejected one clears the market.

## The bids

OMIE publishes, for each day, every offered and every accepted step of each quarter-hour, with the prices and the flows across the borders. I used {{days_per_month}} days drawn at random from each month, {{n_days}} in all.

- **Size.** A median of {{steps_median}} bid steps per quarter-hour, or {{steps_summed}} after merging steps of the same kind at the same price, which changes no programme here.
- **Two zones, sometimes.** Spain and Portugal share one price unless the lines between them are full. Then the market splits into two zones with two prices: in {{n_split}} of the {{n_periods}} quarter-hours ({{split_pct}}), on {{split_days}} days. Each zone is then its own programme, with the flow between them fixed, which makes {{n_zone_periods}} zone-quarter-hours in all.
- **Indivisible orders.** Besides simple steps, sellers can submit three kinds of order that a linear programme cannot represent (EUPHEMIA's public description):
  - *block orders*, a volume over several quarter-hours at one price, accepted all or nothing, or down to a minimum share;
  - *exclusive blocks*, groups of blocks of which at most one is accepted;
  - *scalable complex orders*, a set of steps over the day with a minimum income condition. The money the order earns must cover a fixed term, such as a plant's start-up cost, as well as its steps' prices; if it does not, the whole order is rejected, even its steps priced below the price.

These three kinds made up {{nc_offered_pct}} of the sell volume offered. Of what was offered, {{acc_c01}} of the block volume was accepted, {{acc_c04}} of the exclusive-block volume and {{acc_c02}} of the scalable-complex volume.

## Two programmes

For every zone and quarter-hour I built two programmes from the bids:

- **The realised programme.** The simple steps are its variables. The accepted volume of the indivisible orders and every cross-border flow are fixed at what the market decided, as part of R.
- **The relaxed programme.** As the realised one, but every step of the indivisible orders becomes an ordinary divisible step at its offered price. The flows stay fixed.

Each programme's interval of shadow prices was computed exactly from the bids, in cents and tenths of a megawatt. Every programme was also solved with the HiGHS solver through SciPy {{scipy_version}}; in all {{n_programmes}} solves, HiGHS's shadow price lay in the exact interval and its optimum matched the exact one.

![One quarter-hour's bids, as the two programmes see them](figures/fig1_curves.png)

*One quarter-hour of the first sampled day: of that day's quarter-hours, the one whose change from the relaxation is nearest the average. Blue: the sell steps of the realised programme, starting from its fixed supply. Orange: the same with the indivisible orders made divisible. Black: demand. Each price is where supply meets demand.*

Intervals below are percentile bootstraps over the {{n_days}} days, resampling whole days. For the two questions that decide something, they are simultaneous ({{simultaneous_level}} each); the others are 95%.

## The price is a shadow price

The registered rule asked for the published price to lie within {{e1_tolerance}} €/MWh of the realised programme's shadow prices in at least {{e1_threshold}} of zone-quarter-hours. It lay inside them in every one of the {{n_zone_periods}}: in the joint market, in Spain and in Portugal. The programme's accepted volume matched the published one within {{volume_tolerance}} MW in every one too. In {{wide_pct}} of them no step was partly accepted, so the shadow prices formed an interval, with a median width of {{wide_median}} €/MWh, and the published price lay inside it.

So, once the indivisible orders and the border flows are settled, the Iberian price is exactly the textbook's shadow price. The hard part of the clearing is deciding which indivisible orders to accept and how much to send across borders. What is left after that is a linear programme, and the price is its dual.

## What the indivisible orders do to the price

For each quarter-hour, δ is the relaxed programme's price minus the published price, in the Spanish or joint zone. (When the relaxed programme has an interval of shadow prices, its price is the point of the interval nearest the published price, so that δ counts only the change the relaxation forces.)

I expected no direction, and δ = 0 in most quarter-hours. That expectation failed.

> mean δ = {{e2_mean}} €/MWh (simultaneous interval {{e2_mean_low}} to {{e2_mean_high}})

The interval excludes zero, so by the registered rule the indivisible orders move the average price. The price moved by more than a cent in {{e2_moved}} of quarter-hours: down in {{e2_down}} of the {{n_periods}}, up in {{e2_up}}.

Relaxing one kind of order at a time shows where it comes from:

- **Scalable complex orders only:** {{e2_sco}} €/MWh ({{e2_sco_low}} to {{e2_sco_high}}). Relaxing them alone never raised the price in any quarter-hour. The minimum income condition rejects a whole order, including steps priced below the price; as divisible bids those steps are accepted, and the extra cheap supply lowers the price.
- **Blocks only:** {{e2_blocks}} €/MWh ({{e2_blocks_low}} to {{e2_blocks_high}}), and in both directions: down in {{e2_blocks_down}} quarter-hours, up in {{e2_blocks_up}}. EUPHEMIA judges a block against all the quarter-hours it spans together, so an accepted block can be priced above the price in some of them. As a divisible bid it is not accepted there, and the price rises.

![The change in price from making the indivisible orders divisible](figures/fig2_relaxation.png)

*One dot a day: the mean change over that day's quarter-hours. Black: the mean over days with its simultaneous interval.*

This is not the price a market without these orders would have. Their owners submit them because the conditions protect them, against starting a plant that would not recover its start-up cost, for instance, and without them they would bid differently. δ measures how much the conditions bind in the clearing, with every bid held as it was.

## {{e3_mw}} MW more at zero price

Supply offered at zero price, as wind and solar output often is, pushes the dearest accepted offers out of the merit order and lowers the price: the merit-order effect (Sensfuß and coauthors, 2008). For each quarter-hour, m is the realised programme's price with one more sell step of {{e3_mw}} MW at zero price, minus the published price, with every bid and flow held fixed.

> mean m = {{e3_mean}} €/MWh (simultaneous interval {{e3_mean_low}} to {{e3_mean_high}})

The effect grows roughly in proportion to the size: {{e3_small_mw}} MW lowers the price by {{e3_small_drop}} €/MWh on average, and {{e3_large_mw}} MW by {{e3_large_drop}}.

I expected the effect to be smaller around midday, from {{midday_from}} to {{midday_to}}, when solar output keeps the price low and the supply curve flat. By the registered rule it is: the mean size of m is {{e3_outside_abs}} €/MWh outside midday and {{e3_midday_abs}} at midday, a gap of

> G = {{e3_g}} €/MWh (simultaneous interval {{e3_g_low}} to {{e3_g_high}}).

The interval lies above zero. But the reason is not the one I gave. A step at zero price cannot lower a price that is already at or below zero, and {{nonpos_midday_pct}} of midday quarter-hours were priced there, against {{nonpos_other_pct}} at other hours. In those quarter-hours m is zero. Among the quarter-hours priced above zero, the mean size of m was {{pos_midday_abs}} €/MWh at midday and {{pos_outside_abs}} at other hours, a difference of {{pos_g}} (95% interval {{pos_g_low}} to {{pos_g_high}}): no difference is established. This split was computed after the registered results and decides nothing.

![The effect of {{e3_mw}} MW more at zero price by hour, and how often the price was at or below zero](figures/fig3_hours.png)

*Top: the mean change in price by hour with its simultaneous interval. Bottom: the share of quarter-hours priced at or below zero. Shaded: the registered midday.*

## The French border

Holding every bid fixed, setting the flows with France to zero would have changed the price on average by:

- **{{e4_imports}} €/MWh** (95% interval {{e4_imports_low}} to {{e4_imports_high}}) in the {{e4_imports_n}} quarter-hours that imported from France: without the imports, the price rises;
- **{{e4_exports}} €/MWh** ({{e4_exports_low}} to {{e4_exports_high}}) in the {{e4_exports_n}} that exported to France: without the exports, it falls.

This question was registered as descriptive and decides nothing.

## What this does not say

- **Not a replication of EUPHEMIA.** The flows with France and the outcome of the indivisible orders are taken as given, or, in the relaxation, ignored. The bids from the rest of Europe are not modelled.
- **Not behaviour.** Every bid is held fixed. Real bidders would respond, and with the border flows fixed, extra Iberian supply cannot be exported, so the merit-order figures overstate the effect when the lines to France have room.
- **Not welfare, and not a forecast.** Prices are the object.
- **Not other years.** One year of the quarter-hourly market, from {{n_days}} sampled days.

## What to take from it

- **For anyone who teaches linear programming:** the shadow price of a balance constraint is not a metaphor here. Once the indivisible orders and the border flows are decided, the published price is the dual of a linear programme, to the cent, in every quarter-hour tested.
- **For anyone who models the market:** the non-convex part is where the price is made. A model that treats every Iberian bid as divisible, with the border flows as they were, would have put the price {{e2_drop}} €/MWh lower on average, almost all of it from the scalable complex orders.
- **For anyone who estimates merit-order effects from bid curves:** look at how often the price is already at or below zero. An average effect that is smaller at midday can come entirely from those quarter-hours.

## Data, code and acknowledgements

The data tables, the code, the research plan with every change made to it and the full results are in the [project repository](https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/iberian-electricity-lp). Data: OMIE (OMI-Polo Español, S.A.), [omie.es](https://www.omie.es). OMIE's public information may be used freely, keeping its original content intact and citing the source. The repository does not redistribute OMIE's files; it holds tables derived from them (the offered steps summed by price, and one row per market quarter-hour), with each file's name and SHA-256 hash.

The order types are described in the EUPHEMIA Public Description: Single Price Coupling Algorithm, 18 December 2025, published by the NEMO Committee.

Software: HiGHS, through SciPy {{scipy_version}}. Huangfu, Q., and J. A. J. Hall. 2018: Parallelizing the dual revised simplex method, Mathematical Programming Computation, 10(1), 119-142. doi:10.1007/s12532-017-0130-5. Virtanen, P., et al. 2020: SciPy 1.0: fundamental algorithms for scientific computing in Python, Nature Methods, 17(3), 261-272. doi:10.1038/s41592-019-0686-2.

Sensfuß, F., M. Ragwitz and M. Genoese. 2008: The merit-order effect: A detailed analysis of the price effect of renewable electricity generation on spot market prices in Germany, Energy Policy, 36(8), 3086-3094. doi:10.1016/j.enpol.2008.03.035.
