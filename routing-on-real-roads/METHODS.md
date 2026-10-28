# Routing on real roads: research plan

Registered 2026-10-28, before any distance was computed or any route was solved. Changes
after this date go in the change log at the end, dated, with what had been seen when each
was made.

## 1. Question

A distributor plans each day's delivery routes before its vans leave. Straight-line
distances are the usual shortcut when road distances are unavailable or expensive to
compute. This study asks four questions about that shortcut, on real road networks in
Portugal.

- **Q1, planning regret.** Routes planned on straight-line distances are driven on roads.
  How much longer are they there than routes planned on road distances?
- **Q2, against solver noise.** Is that gap larger than the gap between two runs of the same
  solver that differ only in their random seed?
- **Q3, water.** Is the gap larger when the service area is split by water that only a few
  bridges cross?
- **Q4, the cost estimate.** A straight-line cost multiplied by one circuity factor is a
  common estimate of road cost. How far off is it, plan by plan?

**A fact, not a result.** With the objective of section 5, multiplying every distance by the
same factor α > 0 multiplies every solution's cost by α, so the set of optimal solutions does
not change. A circuity factor therefore corrects the cost estimate (Q4) but cannot change
the plan, and it leaves Q1's regret unchanged. Plans are made on unscaled distances.

## 2. What was seen before this plan was written

- **The extract.** Geofabrik's `portugal-261027.osm.pbf` (OpenStreetMap data, 426,796,277
  bytes). Its MD5, `bc612795c7b549382d86abfc169b4b20`, matches the one Geofabrik
  publishes. SHA-256:
  `73ae017f091747109dbfe0417896090d204f6bed73dd603170646df791e8961a`.
- **Coverage counts.** OSM pharmacies (`amenity=pharmacy`, nodes and ways) inside each
  candidate municipality's OSM boundary (`admin_level=7`) in that extract: Lisboa 293,
  Almada 51, Porto 104, Vila Nova de Gaia 60. Others counted while choosing the areas:
  Amadora 39, Braga 38, Coimbra 41, Gondomar 16, Loures 44, Matosinhos 32, Odivelas 28,
  Oeiras 41, Seixal 26. Portugal as a whole has 2,761.
- **Tools.** A toy run of PyVRP 0.14.1 on 12 random points, which are not study data,
  confirmed three things: it accepts an asymmetric distance matrix; with an iteration
  budget, the same seed gives the same routes; and a different seed can give different
  ones.
- **Not seen.** No road network was built, no distance was computed and no route was
  solved on study data.

## 3. Data

- **Road network.** The extract above, kept outside the repository, with its hashes in a
  manifest. Geofabrik keeps dated extracts for a few days only, so the derived tables of
  section 4 are committed and every later step runs offline from them.
- **The drivable network.** Ways that OSMnx 2.1.1's `drive` filter keeps. Each way's
  `oneway` tag is respected, so the network is directed. The network is clipped to the
  pair's area (section 4) buffered by 3 km, so that routes may use roads just outside a
  municipality's boundary. Only its largest strongly connected component is used.
- **Customers.** OSM pharmacies inside the area's municipalities (section 2). A way's
  location is the mean of its nodes. Pharmacies are real places that wholesalers serve
  every day. Their orders are not public, so orders are simulated (section 4).
- **Licence.** OpenStreetMap data © OpenStreetMap contributors, under the Open Database
  Licence (ODbL). The committed derived tables (pharmacy locations, snapped nodes and
  distance matrices) are a derived database: they are published under the ODbL, with
  that notice.

## 4. Areas, instances and distances

- **Two pairs of areas.** In each pair, one area is a city alone and the other adds the
  municipality across the water.
  - **Lisbon:** Lisboa, against Lisboa with Almada, across the Tagus estuary. The only
    road crossing expected within the 3 km buffer is the 25 de Abril bridge; the data step
    reports the crossings the network actually has.
  - **Porto:** Porto, against Porto with Vila Nova de Gaia, across the Douro, which several
    road bridges cross.
- **One network and one depot per pair.**
  - **Network:** both areas of a pair use the network of the larger one. The smaller
    area's distances are then a submatrix of the larger one's, and within a pair only the
    customers differ.
  - **Depot:** the network node nearest the mean location of the city's own pharmacies
    (Lisboa's or Porto's), for both areas of the pair.
- **Snapping.** Each pharmacy and the depot snap to the nearest node of the network's
  largest strongly connected component, by great-circle distance. Two pharmacies that
  snap to the same node are kept as two customers.
- **Distances, in whole metres:**
  - **Road distance `d_R(i, j)`:** the length of the shortest directed path from i's node
    to j's node. It is not symmetric in general.
  - **Straight-line distance `d_S(i, j)`:** the great-circle distance between the two
    nodes (haversine, Earth radius 6,371,008.8 m).
- **Days.** One seeded draw per day, for D = 40 days per pair (seed 20261028). On each day,
  every pharmacy of the pair's larger area orders independently with probability π. The
  smaller area's instance is the same day's orders restricted to its municipality, so
  the two areas are paired by day.
  - **π by pair:** 0.2 for Lisbon (about 59 orders in Lisboa and 69 with Almada) and 0.5
    for Porto (about 52 and 82).
- **Orders and vehicles.** Each order is one unit. Vehicles are identical, unlimited in
  number, with a capacity of Q = 12 units; there are no time windows.

## 5. Plans and evaluation

- **Objective.** Minimise the total distance driven, with no fixed vehicle cost: the
  capacitated vehicle routing problem.
- **Two plans per instance and seed.**
  - **`P_S`:** solved on `d_S`.
  - **`P_R`:** solved on `d_R`.
- **Solver.** PyVRP 0.14.1 (hybrid genetic search), stopped after 5,000 iterations, with
  seeds s = 1, 2, 3, 4. An iteration budget, unlike a time limit, gives the same routes on
  any machine.
- **Evaluation.** Every plan's routes are costed on both matrices: `C_R(P)` is the total
  road distance of plan P's routes (depot legs included), and `C_S(P)` is the total
  straight-line distance.

## 6. Estimands and decision rules

All intervals are percentile bootstraps over days (9,999 resamples, seed 20261029). The four
areas are reported together, so an interval that decides anything is a simultaneous 95%
interval across the four areas (Bonferroni: 98.75% each).

- **E1, planning regret (Q1).** For day d and seed s,
  `r(d, s) = C_R(P_S(d, s)) / C_R(P_R(d, s)) − 1`.
  - **The estimand:** `R = mean over d of r̄(d)`, where `r̄(d)` is the mean over seeds.
  - **Reported:** R with its interval, per area.
- **E2, against solver noise (Q2).** The seed placebo is
  `q(d, s, s') = C_R(P_R(d, s)) / C_R(P_R(d, s')) − 1` for s ≠ s'.
  - **The estimand:** `G = mean over d of [ r̄(d) − mean over s ≠ s' of |q(d, s, s')| ]`.
  - **Decision:** straight-line planning costs more than solver noise in an area if G's
    simultaneous interval lies above zero.
- **E3, water (Q3).** For each pair, `Δ = mean over d of [ r̄(d) in the larger area −
  r̄(d) in the city alone ]`, paired by day.
  - **Decision:** the water raises the regret in a pair if Δ's simultaneous interval lies
    above zero.
  - **Expectation, stated now:** Δ is positive in both pairs and larger for Lisbon (one
    bridge) than for Porto (several). The comparison between the two pairs is descriptive.
- **E4, the cost estimate (Q4).** The area's circuity factor α is the median of
  `d_R(i, j) / d_S(i, j)` over all ordered pairs of distinct pharmacies in the area. It
  comes from the matrices alone, before any routing.
  - **The plan's own cost:** `e(d) = α · C̄_S(P_S(d)) / C̄_R(P_S(d)) − 1`, with seed means.
  - **The best road cost:** `e*(d) = α · C̄_S(P_S(d)) / C̄_R(P_R(d)) − 1`.
  - **Reported:** the mean, the 5th and 95th percentiles over days, and the interval of the
    mean, per area. No decision rule; E4 is descriptive.
- **Descriptive.**
  - **The matrices:** the distribution of `d_R / d_S` over pharmacy pairs, and the
    asymmetry `|d_R(i, j) − d_R(j, i)| / ((d_R(i, j) + d_R(j, i)) / 2)`.
  - **Per plan:** the number of routes and, in the two larger areas, the share of routes
    that serve customers on both banks (each customer's bank is its municipality).

## 7. Sensitivity analyses (fixed in advance)

Each is run on the first 20 days with seeds 1 and 2, and reports E1 and E3. A variant
changes no decision; it is reported beside the main results.

- **Depot:** the network node nearest the mean location of all the larger area's
  pharmacies.
- **Capacity:** Q = 6 (more routes, more depot legs) and Q = 24.
- **Search effort:** 20,000 iterations instead of 5,000.

## 8. What this study does not claim

- **Not time.** Distance, not travel time: no speeds, traffic, turns or time windows.
- **Not real orders.** Pharmacies are real places, but their orders are simulated, and
  OSM does not map every pharmacy.
- **Not every city.** Two Portuguese metropolitan cores, chosen for their water barriers.
- **Not optimal solutions.** Plans are the best solutions a fixed search effort finds. E2
  measures the solver's own noise; the search-effort variant checks whether more effort
  changes the answer.
- **Not other uses of straight-line distance.** Questions about locating facilities or
  sizing fleets with straight-line distances are outside this plan.

## 9. Reproducibility and reporting

- **Versions:** pinned in `pyproject.toml` (PyVRP 0.14.1, OSMnx 2.1.1, osmium 4.3.2).
- **Seeds:** the extract's hashes and every seed are recorded in the manifests and results.
- **Steps, each in its own pull request:**
  - the data step (network, snapped pharmacies, matrices and day draws), with no routing;
  - the routing and estimates, reported in `docs/RESULTS.md` against this plan;
  - the article, generated from the results with no hand-typed numbers;
  - a critical review before publication.

## Change log

- **2026-10-28.** Plan registered.
- **2026-10-28, the data step, before any route: details the plan left open.**
  - **The drive filter.** It is copied from OSMnx 2.1.1 in Overpass semantics
    (unanchored, case-sensitive patterns). A test reads OSMnx's own filter string and
    compares the two.
  - **Building the network.** Every drivable way with a node inside the bounding box of
    the buffered area is read, with all its nodes. OSMnx builds the unsimplified directed
    graph, clips it to the buffered area, simplifies it and keeps the largest strongly
    connected component. The buffer is computed in EPSG:3763 (ETRS89 / Portugal TM06).
  - **Locations.** A pharmacy mapped as a way is located at the mean of its distinct
    nodes. The depot's target is the arithmetic mean of the city's pharmacies' longitudes
    and latitudes.
  - **The order of points.** The depot comes first, then the city's pharmacies, then the
    other municipality's, each block sorted by OSM type and id. The smaller area's
    matrix is the leading block of the larger one's.
  - **Days.** One generator, seeded 20261028, draws Lisbon's 40 × 344 uniforms first and
    then Porto's 40 × 164. A pharmacy orders when its draw is below the pair's
    probability.
  - **Earth radius.** OSMnx measures edge lengths with 6,371,009 m; the straight-line
    distance uses the registered 6,371,008.8 m. The relative difference, 3e-8, is far
    below the rounding to whole metres. The tests check that no road distance is shorter
    than the straight line by more than that 1 m of rounding.
  - **E4's α.** Two pharmacies that snap to the same node have no ratio, so α is the
    median over pairs with a positive straight-line distance. Such pairs number 24
    ordered pairs in each Lisbon area and 10 in each Porto area.
  - **Crossings.** For paths between the two banks, the data step reports the edge whose
    two ends lie in the pair's two municipalities. Those municipalities meet only in the
    river.
  - **What was seen.** The data report, `docs/DATA.md`. It gives the network sizes,
    snapping distances, orders per day, the distribution of road over straight-line
    distance (α from 1.28 to 1.32 across the four areas) and its asymmetry. It also shows
    where the paths between the banks cross: in Lisbon only the 25 de Abril bridge, in
    Porto three bridges. No route was solved.
- **2026-10-28, before any route: details sections 5-7 left open, fixed in the code before
  the run.**
  - **The depot variant's distances.** Section 7's depot was not in the data step's tables.
    It was added as the last point of each pair before any route was solved. The earlier
    tables are the leading block of the new ones, unchanged (checked), and the days are
    unchanged. Porto's variant depot lies in Vila Nova de Gaia, Lisbon's in Lisboa.
  - **A repeatable crossing report.** OSMnx merges the names of the ways an edge joins into
    a list whose order follows Python's per-process string hashing. The names are now
    sorted; builds under two hash seeds are byte-identical.
  - **The model given to PyVRP.** One depot, a delivery of 1 per order, and one vehicle
    type with as many vehicles as orders. Distances come from the planning matrix, and the
    solver uses its default parameters. Every location gets the EPSG:3763 coordinates of
    its snapped node: PyVRP's SWAP* operator compares routes by their clients' angles
    around the depot, and both plans get the same coordinates.
  - **Checks in every solve.** The plan must be feasible, and its cost on its own matrix
    must equal the cost PyVRP reports.
  - **The bootstrap.** One generator, seeded 20261029, draws the 40-day resamples first,
    then the 20-day ones. Every estimand and area reuses them.
  - **E2's placebo** averages over all 12 ordered pairs of distinct seeds.
  - **Levels.** E1 and E4 are reported with 95% intervals; E2 and E3 decide with
    simultaneous 98.75% intervals.
  - **Sensitivity reporting.** Each variant is shown beside the main design restricted to
    the same 20 days and seeds 1 and 2.
  - **Running.** Four worker processes. The final table is sorted, so it does not depend
    on the order in which solves finish.
  - **What was seen.** The run times of four solves on the largest instances (4-10 s);
    their costs were not looked at.
- **2026-10-28, after the run: the results.** All 2,560 registered solves ran
  (`results/solves.csv`, in 56 minutes on four processes, `results/run_log.txt`), and
  `docs/RESULTS.md` reports section 6's estimands under its rules. No rule was changed.
  - **The expectation for E3 did not hold.** The water does not raise the regret in either
    pair by the registered rule. In Porto, Δ's whole interval lies below zero. The report
    says so in one sentence; that sentence is a description, not a new decision rule.
- **2026-10-28, the article: descriptive figures added after the results.** The article
  (`publication/`) is generated from the committed results, and a test checks each claim it
  makes in words against them. To explain E4, it quotes quantities computed after every
  registered result had been seen, from the same solves. They are in
  `results/estimates.json` under `post_hoc` and in a labelled section of `docs/RESULTS.md`,
  and they decide nothing.
  - **A plan's circuity** κ(d): its road cost over its straight-line cost on day d (seed
    means), averaged over days. E4's error on the plan's own cost is exactly α / κ(d) − 1.
  - **Distances driven:** the road distance each plan drives a day, and the road plans'
    mean straight-line leg against the median straight-line distance between two of the
    area's pharmacies.
  - **E3.** The article also quotes the asymmetry of road distances from `docs/DATA.md` as
    one pattern that fits E3's result in both cities, and says that two cities cannot
    test it.
  - No registered number changed.
