# Straight Lines, Real Roads

*What planning delivery routes on straight-line distances costs, over {{n_days}} simulated days of deliveries in Lisbon and Porto*

*Map data © OpenStreetMap contributors, available under the Open Database Licence ([openstreetmap.org/copyright](https://www.openstreetmap.org/copyright)).*

A distributor plans tomorrow's delivery routes tonight, and the plan needs the distance between every pair of stops. Road distances need a road network and a shortest-path computation for every pair. Straight-line distances need two coordinates and a formula. Planning on straight lines is the cheaper shortcut, and its cost is easy to define: the routes are planned on straight lines but driven on roads.

I measured that cost on the OpenStreetMap road networks of Lisbon and Porto, with simulated daily orders from the cities' pharmacies. The design was registered before any distance was computed or any route was solved, and every number below is read by code from the result files in the [project repository](https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/routing-on-real-roads).

In short: routes planned on straight lines drove {{e1_min}} to {{e1_max}} further on the roads than routes planned on road distances. An analysis added after a critical review of the study traces {{dir_min}} to {{dir_max}} of that to a simple cause: on a straight line a route is as long one way round as the other, so the plan leaves each route's direction to chance, and one-way streets make the direction matter. A river crossed by few bridges did not raise the gap by the registered rule, against what I expected: in Lisbon a rise is not established, and in Porto the gap fell, which the same analysis traces to route direction rather than to the water. Multiplying straight-line distances by the area's median detour factor, to estimate what the routes will cost, understated the road distance the straight-line plans drove by {{e4_own_min}} to {{e4_own_max}}.

## Two plans for the same day

Each day's orders define one routing problem: a depot, the pharmacies that ordered, and identical vans that carry {{capacity}} orders each, as many vans as needed. The plan minimises the total distance driven, with no time windows: the capacitated vehicle routing problem.

I solve each problem twice. The *straight-line plan* minimises total distance computed on straight-line distances; the *road plan* minimises it on road distances. Both are then driven on the roads, and the regret of planning on straight lines is

> regret = (road distance of the straight-line plan) ÷ (road distance of the road plan) − 1.

One fact shapes what follows. If every road distance were the straight-line distance times one constant α, every plan's road distance would be α times its straight-line distance, and a plan that minimises one would minimise the other. A single detour factor can correct a cost estimate, but it cannot change a plan. For exactly optimal plans, any regret must therefore come from how unevenly the roads detour: from pairs of stops where the road is much longer than the straight line beside pairs where it is barely longer, and from one-way streets, which make the road distance depend on the direction while the straight line does not. That last point leaves even an exactly optimal straight-line plan incomplete: a route and its reverse have the same straight-line length, so the plan does not say which way round to drive it.

The detours are uneven. Over every ordered pair of pharmacies in Lisboa, the road is a median {{alpha_lisbon_city}} times the straight line, but one pair in twenty is below {{ratio_p5_lisbon_city}} and one in twenty above {{ratio_p95_lisbon_city}}. The other three areas have similar medians and longer upper tails.

![The ratio of road to straight-line distance over pairs of pharmacies](figures/fig1_ratios.png)

*Each panel is one area; the black line is the median ratio.*

## The data and the runs

- **Roads.** OpenStreetMap, from Geofabrik's extract of Portugal dated {{extract_date}}. The drivable network is what OSMnx's `drive` filter keeps, with one-way streets respected, clipped to the area plus {{buffer_km}} km so that routes may leave a municipality briefly. Around Lisbon that is a directed graph of {{nodes_lisbon}} nodes (junctions and dead ends) and {{edges_lisbon}} edges; around Porto, {{nodes_porto}} and {{edges_porto}}.
- **Customers.** OpenStreetMap's pharmacies: {{n_lisboa}} in Lisboa, {{n_almada}} in Almada, {{n_porto}} in Porto and {{n_gaia}} in Vila Nova de Gaia. They stand in for a distributor's customers: real places, spread across each city. Their orders are not public, so each pharmacy orders one unit a day with probability {{pi_lisbon}} in Lisbon and {{pi_porto}} in Porto, independently, over {{n_days}} simulated days. That gives on average {{orders_lisbon_city}} orders a day in Lisboa, {{orders_lisbon_both}} with Almada, {{orders_porto_city}} in Porto and {{orders_porto_both}} with Gaia.
- **Depot.** The road node nearest the mean location of the city's own pharmacies.
- **Distances.** The road distance is the shortest directed path between the nodes nearest two pharmacies, so it can differ by direction. The straight-line distance is the great-circle distance between the same two nodes.
- **Solver.** PyVRP {{pyvrp_version}}, an iterated local search, stopped after {{iterations}} iterations and run with {{n_seeds}} random seeds for each plan. An iteration budget, unlike a time limit, gives the same routes on any machine; the tests re-solve two of the committed problems to check it.

## How the comparison works

There are four areas, two in each city: the city alone, and the city with the municipality across its river (Almada across the Tagus, Vila Nova de Gaia across the Douro). Within a city, both areas share one road network and one depot, and the larger area's orders on a day are the smaller one's plus the far bank's. The comparison between them is paired by day.

A solver like this one is a randomised search, and two runs with different seeds can return different plans. So the regret is compared with a placebo, fixed in advance: the road plan against itself under another seed. The *seed noise* is the average gap, in either direction, between the road distances of two road plans that differ only in their seed.

The decision rules were fixed in advance. Intervals are percentile bootstraps over the {{n_days}} days, so they describe how the results vary between days of simulated orders, not between cities or designs. For the two questions that decide something, each interval is at {{simultaneous_level}}, Bonferroni over the four areas: the regret exceeds the noise in an area if the interval of their difference lies above zero, and the river raises the regret if the interval of the change lies above zero. The river question makes only two comparisons, so for it this level is conservative.

## Planning on straight lines: {{e1_min}} to {{e1_max}} more road

Per area, with 95% intervals, the straight-line plans drove further by (the kilometres were computed after the results):

- **Lisboa:** {{e1_lisbon_city}} ({{e1_lisbon_city_low}} to {{e1_lisbon_city_high}}), about {{km_lisbon_city}} km a day.
- **Lisboa with Almada:** {{e1_lisbon_both}} ({{e1_lisbon_both_low}} to {{e1_lisbon_both_high}}), about {{km_lisbon_both}} km a day.
- **Porto:** {{e1_porto_city}} ({{e1_porto_city_low}} to {{e1_porto_city_high}}), about {{km_porto_city}} km a day.
- **Porto with Vila Nova de Gaia:** {{e1_porto_both}} ({{e1_porto_both_low}} to {{e1_porto_both_high}}), about {{km_porto_both}} km a day.

Changing only the seed moves the road plan's distance by {{noise_min}} to {{noise_max}} on average. By the registered rule, the regret exceeds the noise in all four areas: on no day in any area was the regret below {{r_day_min}}, and on no day was that noise above {{noise_day_max}}.

The straight-line plan is much less stable, which I checked only after a critical review of the study. Changing its seed moves its road distance by {{noise_s_min}} to {{noise_s_max}} on average, while its straight-line distance moves by at most {{noise_s_flat_max}}. Against that noise the regret is {{times_s_min}} to {{times_s_max}} times larger, so a single straight-line run says less than the averages do. The two plans use almost the same number of routes; they differ in which pharmacies share a route, in the order they are visited, and in the direction each route is driven.

![Per-day regret and seed noise in each area](figures/fig2_regret.png)

*One dot a day. Blue: the straight-line plan against the road plan, averaged over seeds. Orange: the straight-line plan against itself under another seed. Grey: the road plan against itself under another seed. Black: the mean over days with its 95% interval.*

## Where the regret comes from: which way round

The straight-line plan's instability has a simple cause. On straight lines a route and its reverse have the same length, so the solver has no reason to prefer either, and which one it returns depends on the seed. In {{same_min}} to {{same_max}} of the pairs of seeds, the two straight-line plans of a day have the same routes up to direction, yet drive different road distances, because one-way streets make the two directions differ on the road.

So I drove each straight-line route in its better direction on the road. That needs no new plan and no full table of road distances, only the road distances of the plan's own legs in both directions. The regret falls to:

- **Lisboa:** {{or_lisbon_city}} ({{or_lisbon_city_low}} to {{or_lisbon_city_high}}), from {{e1_lisbon_city}}.
- **Lisboa with Almada:** {{or_lisbon_both}} ({{or_lisbon_both_low}} to {{or_lisbon_both_high}}), from {{e1_lisbon_both}}.
- **Porto:** {{or_porto_city}} ({{or_porto_city_low}} to {{or_porto_city_high}}), from {{e1_porto_city}}.
- **Porto with Vila Nova de Gaia:** {{or_porto_both}} ({{or_porto_both_low}} to {{or_porto_both_high}}), from {{e1_porto_both}}.

Direction accounts for {{dir_min}} to {{dir_max}} of the regret in the main design, and for {{dir_var_min}} to {{dir_var_max}} in the variations described below. What remains, {{or_min}} to {{or_max}}, comes from which pharmacies share a route and the order in which they are visited. These figures were computed after the review, from the same solves, and decide nothing.

## The river: no established rise

I expected the regret to rise when the area spans water that few bridges cross, and to rise more in Lisbon than in Porto. In the Lisbon network, every shortest road path between the two banks crosses the 25 de Abril bridge; in Porto, {{porto_bridges}} bridges carry {{porto_bridge_share}} of them.

- **Lisbon:** adding Almada changes the regret by {{e3_lisbon}} percentage points (registered interval {{e3_lisbon_low}} to {{e3_lisbon_high}}). The interval includes zero, so by the registered rule a rise is not established. The estimate has the sign I expected, and its 95% interval, {{e3_lisbon_95_low}} to {{e3_lisbon_95_high}}, would lie above zero: the result is inconclusive rather than a failed expectation.
- **Porto:** adding Vila Nova de Gaia changes it by {{e3_porto}} points ({{e3_porto_low}} to {{e3_porto_high}}). The whole interval lies below zero: as a share of the road distance, the regret fell.

Porto's fall is not what it seems. In kilometres, adding Gaia raised the regret, by {{e3_km_porto}} km a day ({{e3_km_porto_low}} to {{e3_km_porto_high}} km): Gaia adds more road than regret, as the road plan's distance grows from {{road_km_porto_city}} to {{road_km_porto_both}} km a day. And the fall in the share comes from route direction. Porto's own streets are the most one-directional here: between two Porto pharmacies, the road distance differs between the two directions by a median {{asym_porto_city}}, against {{asym_porto_other}} between two in Gaia and {{asym_porto_across}} across the river. With every straight-line route driven in its better direction, Porto's fall disappears: adding Gaia changes the regret by {{e3o_porto}} points ({{e3o_porto_low}} to {{e3o_porto_high}}). Adding Almada then changes it by {{e3o_lisbon}} points ({{e3o_lisbon_low}} to {{e3o_lisbon_high}}); that interval lies above zero, but these analyses came after the review, and neither decides anything.

The comparison has a limit the design cannot remove. Adding the far bank adds a whole municipality, with {{orders_up_lisbon}} more orders a day in Lisbon and {{orders_up_porto}} more in Porto, over a larger area, so the water and the size of the area cannot be told apart.

![Per-day change in the regret when the far bank is added](figures/fig3_water.png)

*One dot a day: the regret with the far bank minus the regret in the city alone, on the same day. Black: the mean over days with its {{simultaneous_level}} interval.*

## The detour factor falls short

Straight-line distances are also used to estimate what the routes will cost. The estimate here multiplies the straight-line plan's straight-line distance by α, the area's median ratio of road to straight-line distance ({{alpha_min}} to {{alpha_max}} across the four areas), which comes from the distance tables alone, before any route is solved.

It falls short everywhere. Against the road distance the straight-line plan actually drives, it is short on average by {{e4_own_lisbon_city}} in Lisboa, {{e4_own_lisbon_both}} with Almada, {{e4_own_porto_city}} in Porto and {{e4_own_porto_both}} with Gaia. Against the road plan's distance, the shortest on offer, it is still short by {{e4_best_min}} to {{e4_best_max}}. In every area it falls short on every one of the {{n_days}} days, against either.

The reason lies in the legs that routes actually drive. Write κ for a plan's own ratio of road to straight-line distance on a day: its total road distance over its total straight-line distance. On each day, the estimate's error against that plan's road distance is then exactly

> error = α ÷ κ − 1,

so the estimate falls short whenever a plan's legs detour more than the median pair of pharmacies. They do, for two reasons:

- **Routes are chains of short hops, and short hops detour more.** The road plans' legs average {{leg_min}} to {{leg_max}} km in a straight line, depot legs included, while two pharmacies picked at random are a median {{pair_min}} to {{pair_max}} km apart. Between pharmacies half a kilometre to a kilometre apart, the road is a median {{short_min}} to {{short_max}} times the straight line; beyond {{long_km}} km, {{long_min}} to {{long_max}}.
- **α is a median, and distances add up.** The ratio is skewed to the right: averaged over pairs of pharmacies it is {{ratio_mean_min}} to {{ratio_mean_max}}, above the median.

Averaged over days, κ is {{kappa_road_min}} to {{kappa_road_max}} on the legs the road plans drive, already above α, and {{kappa_straight_min}} to {{kappa_straight_max}} on the straight-line plans' legs. That the straight-line plan's is the higher follows from how the plans are made: when both are optimal for their own distances, the straight-line plan drives at least as far on the road as the road plan, and no further on straight lines. Beyond that, in {{legs_typical}} of the four areas the straight-line plan's legs detour about as much as other legs of the same length. The exception is Lisboa with Almada, where they detour {{almada_legs}} times the straight line against {{almada_like}} for pairs of the same length: straight lines across the estuary hide the detour to the bridge. That fits the optimizer's curse (Smith and Winkler, 2006): choose the option whose estimate looks best, and its true cost will on average exceed the estimate.

The values of κ, the leg lengths and these comparisons were computed after the registered results, from the same solves, and decide nothing.

![The detour-factor estimate against the road distance driven, day by day](figures/fig4_estimate.png)

*One dot a day, seed means. Filled: against the road distance the straight-line plan drives. Hollow: against the road plan's road distance. Below the diagonal, the estimate falls short.*

## How robust is this?

I fixed {{n_variants}} variations in advance, each run on the first {{sens_days}} days with {{sens_seeds}} seeds and set beside the main design on the same days and seeds: a depot at the mean of all the pair's pharmacies, vans of {{cap_small}} and of {{cap_large}} orders instead of {{capacity}}, and {{iter_large}} iterations instead of {{iterations}}.

- **The regret persists.** It stays between {{sens_e1_min}} and {{sens_e1_max}} in every variation and area.
- **Van size matters, up to a point.** With the days paired, the regret is higher with vans of {{capacity}} than with vans of {{cap_small}} in every area, by {{cap_up_min}} to {{cap_up_max}} points on average. From {{capacity}} to {{cap_large}} there is no consistent change: it rises a little in {{cap_24_up}} areas and falls in Lisboa with Almada. Smaller vans make more trips to and from the depot, legs that both plans drive in much the same way; that fits the first step, but the variations do not test it. These paired comparisons were made after the review.
- **More search changes little.** {{effort_ratio}} times the search effort moves no estimate by more than {{iter_max_change}} percentage points, so the results do not hinge on the search budget.
- **The river.** Porto's change from the far bank is negative in every variation, but in {{porto_e3_with_zero}} of them its 95% interval includes zero, and one variation puts the depot in Gaia, across the river from Porto. Lisbon's is positive in {{lisbon_e3_positive}} of the {{n_variants}} variations and negative with the largest vans; {{lisbon_e3_above}} of the positive ones have 95% intervals above zero. None of these intervals is corrected for the many comparisons, and the variations decide nothing.

![Regret and the change from the far bank under each variation](figures/fig5_sensitivity.png)

*The main design on the first {{sens_days}} days with {{sens_seeds}} seeds, then each variation fixed in advance, on the same days and seeds. Lines are 95% intervals over days.*

## What this does not say

- **Not time.** Distance only: no speeds, traffic or time windows. The regret in driving time could be larger or smaller.
- **Not every road rule.** Turn restrictions are not modelled, and they lengthen some road paths. OSMnx's `drive` filter also keeps a few ways that a van may not use, such as bus-only lanes and roads marked no access; a check on Porto during the review found that leaving them out changes little.
- **Not real orders.** The pharmacies are real, but their orders are simulated, one unit each, and OpenStreetMap does not map every pharmacy.
- **Not optimal plans.** Each plan is the best that a fixed search found. The road plans' agreement between seeds and the longer search show that the search had settled; they do not show that the plans are optimal.
- **Not every city or design.** Four areas in two Portuguese cities, chosen for their rivers, under one design. The intervals cover only the variation between simulated days; the van size moves the estimates more than that. A city with a regular street grid could give a different answer.
- **Not other uses.** Choosing depot locations or sizing a fleet on straight-line distances are different questions.

## What to do with it

If road distances can be computed, plan on them. Everything here came from a free map extract and open-source tools. If a plan must be made on straight lines, at least choose each route's direction on the road: that needs only the road distances of the plan's own legs, and here it removed {{dir_min}} to {{dir_max}} of the regret. Without that fix, expect routes longer than they need to be, by {{sens_e1_min}} to {{sens_e1_max}} across every area and variation here. And do not budget with the median detour factor: the legs that routes actually drive are short, and short legs detour more than the typical pair of stops.

## Data, code and acknowledgements

The data, the code, the research plan with every change made to it, the critical review of the study and the full results are in the [project repository](https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/routing-on-real-roads). Map data © OpenStreetMap contributors, available under the Open Database Licence ([openstreetmap.org/copyright](https://www.openstreetmap.org/copyright)), from Geofabrik's extract `{{extract_file}}`. The derived tables in the repository (pharmacy locations, snapped nodes and distance matrices) and the solves computed from them are published under the same licence.

Software: OSMnx {{osmnx_version}}, from Boeing, G. 2025: Modeling and Analyzing Urban Networks and Amenities with OSMnx, Geographical Analysis, 57(4), 567-577. doi:10.1111/gean.70009. PyVRP {{pyvrp_version}}, from Wouda, N. A., L. Lan and W. Kool. 2024: PyVRP: A High-Performance VRP Solver Package, INFORMS Journal on Computing, 36(4), 943-955. doi:10.1287/ijoc.2023.0055.

Smith, J. E., and R. L. Winkler. 2006: The Optimizer's Curse: Skepticism and Postdecision Surprise in Decision Analysis, Management Science, 52(3), 311-322. doi:10.1287/mnsc.1050.0451
