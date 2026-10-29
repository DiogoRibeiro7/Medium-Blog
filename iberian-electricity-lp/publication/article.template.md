# Taking the Iberian Electricity Price Apart with Linear Programming

*{{n_programmes}} linear programmes rebuilt from OMIE's published bids: what the non-convex orders and zero-priced supply do to the price*

*Data: OMIE (OMI-Polo Español, S.A.), [omie.es](https://www.omie.es).*

Every day, by noon, buyers and sellers of electricity in Spain and Portugal send OMIE, the Iberian market operator, their bids for each quarter-hour of the next day, and an algorithm sets the price of each quarter-hour. A textbook says what that price should be. If every bid were divisible and every quarter-hour cleared on its own, clearing the market would be a linear programme, and the price would be the shadow price of its one constraint: that supply equals demand.

The real clearing is not that programme. Spain and Portugal are cleared together with the rest of Europe by an algorithm called EUPHEMIA, and some sell orders are non-convex: blocks accepted all or nothing, and orders with conditions that tie quarter-hours together. But once EUPHEMIA has chosen which of those to accept and how much power crosses each border, what is left is a linear programme, and its rules make the price that programme's shadow price. OMIE publishes every bid step, accepted or not, so the programme can be rebuilt, and then changed. What if the non-convex orders were ordinary bids? What if more supply arrived at zero price?

I took {{n_days}} days from the first year of quarter-hourly prices, {{first_day}} to {{last_day}}, and solved {{n_programmes}} linear programmes rebuilt from the published bids. The design was registered before any bid was compared with a price, a critical review was made before publication, and every number below is read by code from the result files in the [project repository](https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/iberian-electricity-lp).

In short:

- **The rebuild reproduces the price.** With the non-convex orders' outcome and the border flows taken as given, the published price was a shadow price of the rebuilt programme in every one of the {{n_periods}} quarter-hours, in Spain and in Portugal alike. EUPHEMIA's rules require this, so it checks the data and the code rather than discovering anything; it is what makes the next two questions possible.
- **Relaxing the non-convex orders lowers the price.** Treating those orders as ordinary divisible bids, with the border flows as they were, would have lowered the price by up to {{e2_drop}} €/MWh on average, against my expectation of no change. Even in the quarter-hours when the line to France was full, the fall was {{line_full_drop}} €/MWh.
- **Zero-priced supply cannot push a price below zero.** {{e3_mw}} MW more supply at zero price would have lowered the price by {{e3_drop}} €/MWh on average. The effect is smaller around midday, as I expected, but only half of my reason holds up: in an analysis added after the results, the gap comes from the {{nonpos_midday_pct}} of midday prices that were already at or below zero.

## A market is a linear programme

Take one quarter-hour. A sell bid is a step: up to q MW at a price of at least p. A buy bid is a step: up to Q MW at a price of at most P. Accept s MW of each sell step and b MW of each buy step to

> maximise Σ P·b − Σ p·s, subject to Σ s − Σ b = R and 0 ≤ s ≤ q, 0 ≤ b ≤ Q for every step.

The objective is the buyers' bids minus the sellers' offers for what is traded: the gains from trade. R is the volume fixed in advance, such as exports minus imports. There is one constraint, so there is one dual variable λ, the shadow price: how much the best objective changes per extra MW the constraint requires.

Complementary slackness, the condition linking an optimal solution to its dual, makes λ concrete. A value λ is an optimal dual exactly when every sell step priced below λ is fully accepted, every sell step priced above λ is rejected, the same holds in reverse for buy steps, and only a step priced exactly at λ may be accepted in part. That is the rule of a uniform-price auction, so a shadow price is a clearing price, and a clearing price is a shadow price.

The shadow prices form an interval, not always a single point. If a step is partly accepted, its price is the only shadow price. If none is, any price between the neighbouring steps, accepted on one side and rejected on the other, clears the market.

Fixing a market's all-or-nothing choices and pricing what is left as a linear programme is a known way to price markets that are not convex (O'Neill and coauthors, 2005), and it is how EUPHEMIA's prices relate to its choices.

## The bids

OMIE publishes, for each day, every offered and every accepted bid step of each quarter-hour, with the prices and the flows across the borders. The steps are anonymous, and the orders' conditions are not published. I used {{days_per_month}} days drawn at random from each month, {{n_days}} in all.

- **Size.** A median of {{steps_median}} bid steps per quarter-hour in the joint or Spanish market, or {{steps_summed}} after merging steps of the same kind at the same price, which changes no programme here.
- **Two zones, sometimes.** Spain and Portugal share one price unless the lines between them are full. Then the market splits into two zones: in {{n_split}} of the {{n_periods}} quarter-hours ({{split_pct}}), on {{split_days}} days, and their prices differed in {{split_differ}} of them. Each zone is then its own programme, with the flow between them fixed, which makes {{n_zone_periods}} zone-quarter-hours in all.
- **Non-convex orders.** Besides simple steps, sellers can submit three kinds of order that a linear programme cannot represent (EUPHEMIA's public description):
  - *block orders*, a volume over several quarter-hours, possibly different in each, at one price, accepted all or nothing or down to a minimum share;
  - *exclusive blocks*, groups of blocks whose accepted shares add up to at most one, so that of blocks that must be accepted in full, at most one is;
  - *scalable complex orders*, a set of steps over the day that is activated as a whole or not at all. Once activated, its steps are divisible above a minimum power. It can carry a minimum income condition, under which the money the order earns must cover a fixed term, such as a plant's start-up cost, as well as its steps' prices, and limits on how fast its output changes. If its conditions are not met, the whole order is rejected, even its steps priced below the price.

These three kinds made up {{nc_offered_pct}} of the sell volume offered. Of what was offered, {{acc_c01}} of the block volume was accepted, {{acc_c04}} of the exclusive-block volume and {{acc_c02}} of the scalable-complex volume.

## Two programmes

For every zone and quarter-hour I built two programmes from the bids:

- **The realised programme.** The simple steps are its variables. The accepted volume of the non-convex orders and every cross-border flow are fixed at what the market decided, as part of R.
- **The relaxed programme.** As the realised one, but every step of the non-convex orders becomes an ordinary divisible step at its offered price, and their fixed terms, which the files do not show, are dropped. The flows stay fixed.

Each programme's interval of shadow prices was computed exactly from the bids, in cents and tenths of a megawatt. Every programme was also solved with the HiGHS solver through SciPy {{scipy_version}}; in all {{n_programmes}} solves, HiGHS's shadow price lay in the exact interval and its optimum matched the exact one.

![One quarter-hour's bids, as the two programmes see them](figures/fig1_curves.png)

*One quarter-hour of the first sampled day: of that day's joint quarter-hours, the one whose change from the relaxation is nearest the average over all quarter-hours ({{e2_mean}} €/MWh). Blue: the sell steps of the realised programme, starting from its fixed supply. Orange: the same with the non-convex orders made divisible. Black: demand. Each price is where supply meets demand.*

Intervals below are percentile bootstraps over the {{n_days}} days, resampling whole days. Those of the second and third questions are at {{simultaneous_level}} each, so that their two decisions hold together; the others are 95%.

## The rebuild reproduces the price

The registered rule asked for the published price to lie within {{e1_tolerance}} €/MWh of the realised programme's shadow prices in at least {{e1_threshold}} of zone-quarter-hours. It lay inside them in every one of the {{n_zone_periods}}: in the joint market, in Spain and in Portugal. The programme's accepted volume matched the published one within {{volume_tolerance}} MW in every one too. In {{wide_pct}} of them the shadow prices formed an interval rather than a point, with a median width of {{wide_median}} €/MWh, and the published price lay inside it.

This is not a discovery about the price. EUPHEMIA must accept every in-the-money order in full and reject every out-of-the-money one, and it checks its solutions for exactly this complementary slackness, given its choice of non-convex orders. So the result says that OMIE's public files and my rebuild agree to the cent, which is what the next two questions need.

## What the non-convex orders do to the price

For each quarter-hour, δ is the relaxed programme's price minus the published price, in the Spanish or joint zone. (When the relaxed programme has an interval of shadow prices, its price is the point of the interval nearest the published price, so that δ counts only the change the relaxation forces.)

I expected no direction, and δ = 0 in most quarter-hours. That expectation failed, as a trial day outside the sample had already suggested before the study's programmes were solved; the plan's change log records it.

> mean δ = {{e2_mean}} €/MWh (interval {{e2_mean_low}} to {{e2_mean_high}})

The interval excludes zero, so by the registered rule the non-convex orders move the average price. The price moved by more than a cent in {{e2_moved}} of quarter-hours: down in {{e2_down}} of the {{n_periods}}, up in {{e2_up}}.

The size is an upper bound for this kind of relaxation, for two reasons the review found (the figures here were computed after the results):

- **The fixed terms are dropped, not spread over the volume.** Spreading them would raise the offers, and with them the price.
- **The flows with France stay fixed.** In {{line_room_pct}} of quarter-hours the line had room, and there the extra cheap supply could have gone to France. Mean δ was {{line_full_delta}} €/MWh in the quarter-hours when the line was full (95% interval {{line_full_delta_low}} to {{line_full_delta_high}}), against {{line_room_delta}} when it had room.

It also mixes two kinds of quarter-hour: in joint ones the relaxation acts on the whole Iberian market (mean δ {{joint_delta}} €/MWh), in split ones on Spain alone ({{split_delta}}).

Relaxing one kind of order at a time shows where it comes from, although the parts do not add up:

- **Scalable complex orders only:** {{e2_sco}} €/MWh ({{e2_sco_low}} to {{e2_sco_high}}), as much as relaxing every order. It never raised the price in any quarter-hour, which follows from the rules: the accepted volume of these orders never exceeded their offered volume priced at or below the price, so relaxing them only adds supply below it. Which condition kept the rejected steps out, the minimum income condition, the limits on output changes or the minimum power, the anonymous files cannot say.
- **Blocks only:** {{e2_blocks}} €/MWh as registered, but that figure counts the blocks of an exclusive group as if all of them could be accepted. With the exclusive blocks left as they were, it is {{blocks_c01}}. Blocks move the price both ways, down in {{e2_blocks_down}} quarter-hours and up in {{e2_blocks_up}}: EUPHEMIA judges a block against all the quarter-hours it spans together, so an accepted block can be priced above the price in some of them, and there the relaxation removes it.

![The change in price from making the non-convex orders divisible](figures/fig2_relaxation.png)

*One dot a day: the mean change over that day's quarter-hours. Black: the mean over days with its {{simultaneous_level}} interval. The block row is the registered one, with exclusive blocks counted as additive.*

This is not the price a market without these orders would have. Their owners submit them because the conditions protect them, against starting a plant that would not recover its start-up cost, for instance, and without them they would bid differently. δ measures how much the conditions bind in the clearing, with every bid held as it was.

## {{e3_mw}} MW more at zero price

Supply offered at zero price, as wind and solar output often is, pushes the dearest accepted offers out of the merit order and lowers the price: the merit-order effect (Sensfuß and coauthors, 2008; on the Spanish market, Ciarreta and coauthors, 2014). For each quarter-hour, m is the realised programme's price with one more sell step of {{e3_mw}} MW at zero price, minus the published price, with every bid, the border flows and the non-convex orders' acceptance held fixed.

> mean m = {{e3_mean}} €/MWh (interval {{e3_mean_low}} to {{e3_mean_high}})

The effect grows roughly in proportion to the size: {{e3_small_mw}} MW lowers the price by {{e3_small_drop}} €/MWh on average, and {{e3_large_mw}} MW by {{e3_large_drop}}.

I expected the effect to be smaller around midday, from {{midday_from}} to {{midday_to}}, when solar output keeps the price low and the supply curve flat. By the registered rule it is: the mean size of m is {{e3_outside_abs}} €/MWh outside midday and {{e3_midday_abs}} at midday, a gap of

> G = {{e3_g}} €/MWh (interval {{e3_g_low}} to {{e3_g_high}}).

The interval lies above zero, but only half of my reason holds up. What follows was computed after the results and decides nothing. A step at zero price cannot push the price below zero. Where the price was already at or below zero, {{nonpos_midday_pct}} of midday quarter-hours against {{nonpos_other_pct}} at other hours, m is zero, and that accounts for the gap: low midday prices, the first half of my reason, are enough. The same floor holds at every price, since m can never be larger than the price itself, and midday prices above zero were lower too: a median of {{pos_midday_median}} €/MWh against {{pos_outside_median}}. Compared within the same band of prices, the effect was larger at midday in {{bands_larger}} of {{bands_n}} bands, so the data give no support to the second half, a flatter supply curve at midday.

![The effect of {{e3_mw}} MW more at zero price by hour, and how often the price was at or below zero](figures/fig3_hours.png)

*Top: the mean change in price by hour, each with its {{simultaneous_level}} interval (one per hour, not joint across the hours). Bottom: the share of quarter-hours priced at or below zero. Shaded: the registered midday.*

## The French border

Holding every bid and the non-convex orders' acceptance fixed, setting the flows with France to zero would have changed the price on average by:

- **{{e4_imports}} €/MWh** (95% interval {{e4_imports_low}} to {{e4_imports_high}}) in the {{e4_imports_n}} quarter-hours that imported from France: without the imports, the price rises;
- **{{e4_exports}} €/MWh** ({{e4_exports_low}} to {{e4_exports_high}}) in the {{e4_exports_n}} that exported to France: without the exports, it falls.

This question was registered as descriptive and decides nothing.

## What this does not say

- **Not a replication of EUPHEMIA.** The flows with France and the outcome of the non-convex orders are taken as given, or, in the relaxation, ignored. The bids from the rest of Europe are not modelled.
- **Not behaviour.** Every bid is held fixed, and real bidders would respond. With the border flows fixed, extra Iberian supply cannot be exported, so the relaxation and the merit-order figures overstate the effect when the line to France has room. With the non-convex orders' acceptance fixed in the last two questions, a moved price would in reality accept or reject some of them differently.
- **Not welfare, and not a forecast.** Prices are the object.
- **Not other years.** One year of the quarter-hourly market, from {{n_days}} sampled days.

## What to take from it

- **For anyone who teaches linear programming:** once a power exchange has made its all-or-nothing choices, its price is the shadow price of a linear programme. The published order books let you check that to the cent, and then use the programme to ask what-if questions.
- **For anyone who models the market:** the conditions on non-convex orders matter. Treating those orders as ordinary bids would have lowered the price by up to {{e2_drop}} €/MWh on average, about {{e2_share_of_price}} of the mean price of {{mean_price}} €/MWh.
- **For anyone who estimates merit-order effects from bid curves:** check how often the price is already at or below zero, and compare like prices. An effect that looks smaller at midday can come entirely from the zero floor.

## Data, code and acknowledgements

The data tables, the code, the research plan with every change made to it, the full results and the critical review, with the corrections it led to, are in the [project repository](https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/iberian-electricity-lp). Data: OMIE (OMI-Polo Español, S.A.), [omie.es](https://www.omie.es). OMIE's public information may be used freely, keeping its original content intact and citing the source. The repository does not redistribute OMIE's files; it holds tables derived from them (the offered steps summed by price, and one row per market quarter-hour), with each file's name and SHA-256 hash.

The order types and EUPHEMIA's acceptance rules (sections 5.1, 5.3, 5.4 and 8.2) are from the EUPHEMIA Public Description: Single Price Coupling Algorithm, 18 December 2025, published by the NEMO Committee.

O'Neill, R. P., P. M. Sotkiewicz, B. F. Hobbs, M. H. Rothkopf and W. R. Stewart. 2005: Efficient market-clearing prices in markets with nonconvexities, European Journal of Operational Research, 164(1), 269-285. doi:10.1016/j.ejor.2003.12.011.

Software: HiGHS, through SciPy {{scipy_version}}. Huangfu, Q., and J. A. J. Hall. 2018: Parallelizing the dual revised simplex method, Mathematical Programming Computation, 10(1), 119-142. doi:10.1007/s12532-017-0130-5. Virtanen, P., et al. 2020: SciPy 1.0: fundamental algorithms for scientific computing in Python, Nature Methods, 17(3), 261-272. doi:10.1038/s41592-019-0686-2.

Sensfuß, F., M. Ragwitz and M. Genoese. 2008: The merit-order effect: A detailed analysis of the price effect of renewable electricity generation on spot market prices in Germany, Energy Policy, 36(8), 3086-3094. doi:10.1016/j.enpol.2008.03.035.

Ciarreta, A., M. P. Espinosa and C. Pizarro-Irizar. 2014: Is green energy expensive? Empirical evidence from the Spanish electricity market, Energy Policy, 69, 205-215. doi:10.1016/j.enpol.2014.02.025.
