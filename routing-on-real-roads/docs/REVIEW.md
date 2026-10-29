# Critical review of the routing study (2026-10-29)

This review was written after the registered results ([RESULTS.md](RESULTS.md)) and the
generated article ([../publication/article.md](../publication/article.md)), and before
publication.

**How it was done.**

- **Three independent reviewers.** Each read the plan, the code, the data and the outputs
  from one angle: the road networks, the matrices and the solver; the estimands and the
  inference; and the claims and their communication. All three were read-only and had to
  give evidence for every finding.
- **Independent verification.** Every finding marked **verified** below was recomputed
  with new scripts: from the committed tables and solves, from the per-pair OSM files in
  `data/raw/`, from the installed PyVRP 0.14.1, or by re-solving registered instances. The
  findings that could not be checked this way are marked **plausible**.
- **Nothing was changed.** No result, plan, code or text was changed in the course of
  this review. The changes it recommends are listed at the end, for a separate step.

## Bottom line

The code does what [METHODS.md](../METHODS.md) says, and the numbers are right:

- **The matrices.** Both straight-line matrices equal an independent haversine on the
  snapped nodes. Porto's road matrix was rebuilt from `porto_drive.osm` with a separate
  graph, without OSMnx, and all 27,390 off-diagonal entries match. No road distance is
  shorter than the straight line, and the data step's matrices are exactly the leading
  block of the current ones.
- **The plans.** All 2,560 committed plans were re-costed independently: every order is
  served once, no route exceeds the capacity, and both costs and the counts of routes
  serving both banks match. Registered solves re-run in this review reproduce their
  routes exactly: four by a reviewer, and two more with every coordinate set to zero
  (M2).
- **The estimands.** Every registered number was recomputed without the project's code,
  bootstrap intervals included, with no difference above 5e-11.
- **The article.** Every number in it matches the results.
- **The decisions.** The regret exceeds the solver noise in all four areas, and Porto's
  negative Δ survives t and BCa intervals and is not explained by the number of orders.

But four problems change what the study can claim:

1. **A third to a half of the regret is the direction in which each straight-line route
   is driven.** A route and its reverse cost the same on straight lines, so the solver
   picks the direction arbitrarily, and on roads with one-way streets the direction
   matters. All three reviewers found this independently. Driving each straight-line route
   in its better direction removes 22% to 50% of the regret. The article never mentions it.
2. **"More than 60 times the solver's own run-to-run noise" compares with the wrong
   noise.** The placebo measures the road plan, whose road distance is its own objective.
   The straight-line plan's road distance moves 1.4% to 2.8% between seeds, so the regret
   is 3 to 6 times that noise.
3. **The river results say less about water than the article implies.** Porto's fall in
   the regret comes entirely from route direction, it is a fall in a ratio while the
   regret in kilometres rises, and it compares the city with the city plus a whole
   municipality. In Lisbon, "the expectation failed" is too strong for an inconclusive
   result whose estimate has the expected sign.
4. **The optimizer's-curse explanation of E4 is not supported.** The straight-line plan's
   legs detour about as much as any pair of pharmacies of the same length, except with
   Almada. That they detour more than the road plan's follows by construction.

## What happens to each conclusion

| Claim | Registered verdict | After review |
| --- | --- | --- |
| E1, the regret: 7.1% to 8.9% | reported | **Holds as registered:** the extra road distance of the straight-line plan as the solver returns it, averaged over seeds. 22% to 50% of it is the arbitrary direction of each route (M1). With every route driven in its better direction, the regret is 4.5% to 6.6% (post hoc). |
| E2, the regret exceeds solver noise | yes, in all four areas | **Holds, and is nearly mechanical** (I8). "More than 60 times the noise" uses the road plan's noise. Against the straight-line plan's own seed noise the regret is 3 to 6 times larger (I1). |
| E3, Lisbon: Δ = +1.3 [−0.2, +3.0] | water does not raise the regret | **Holds as registered, but is inconclusive rather than a failed expectation.** The estimate has the expected sign, its 95% interval lies above zero, and the study could reliably detect only rises above about 2.2 points (I2). |
| E3, Porto: Δ = −1.5 [−2.5, −0.5] | water does not raise the regret | **The decision holds; the fall is not about water.** It comes entirely from route direction (M1). In kilometres a day the regret rises (I3), and the comparison adds a whole municipality. |
| E4: the detour factor falls short by 13.7% to 19.4% | descriptive | **The numbers hold; the explanation does not** (I4). Short legs detour more than the median pair, and α is a median of a right-skewed ratio. The optimizer's curse is not shown beyond leg length, except with Almada. |
| Sensitivity analyses | no decision changes | **Hold.** The capacity pattern fails in Lisboa with Almada (I6). Porto's depot variant puts the depot across the river (M4). |

## Findings

Severity: **high**, could change a stated conclusion or mislead a reader about one;
**medium**, weakens a conclusion or its reading; **low**, a detail.

### The networks, the matrices and the solver

**M1. A third to a half of the regret is route direction.** High; verified.

- **Why it happens.** The straight-line matrix is symmetric, so a route and its reverse
  have the same straight-line cost, and the solver breaks the tie by seed. The road matrix
  is not symmetric, because of one-way streets.
- **An example.** On Lisboa's first day, the straight-line plans of seeds 2, 3 and 4 are
  the same five routes; they differ only in the direction of two of them. All three cost
  67,645 m on straight lines. On the road they cost 100,776 m, 101,654 m and 103,056 m, a
  spread of 2.3% from direction alone.
- **How often.** In 29% to 51% of ordered seed pairs per area, the two straight-line plans
  have the same routes up to direction, and their road costs differ in 770 of those 796
  seed pairs.
- **What direction is worth.** Each straight-line route is driven in its better direction
  on the road, with no new solve. That needs the road distances of the plan's own legs in
  both directions, not a full matrix. The regret becomes:

  | Area | Regret as registered | Routes in their better direction | Share due to direction |
  | --- | ---: | ---: | ---: |
  | Lisboa | 7.13% | 4.78% | 33% |
  | Lisboa with Almada | 8.45% | 6.58% | 22% |
  | Porto | 8.92% | 4.48% | 50% |
  | Porto with Vila Nova de Gaia | 7.43% | 4.49% | 40% |

  Across all four sensitivity variants, the share ranges from 19% to 61% (Porto with vans
  of 6).
- **Not a search failure.** The road plans' routes are already in their better direction:
  reversing improves 1 of 3,658 routes.
- **What it changes.**
  - *E1.* The registered regret is a fair measure of using the solver's straight-line plan
    as it comes. But a third to a half of it is avoidable with a few road distances.
  - *The explanation.* The article says the plans "differ in which pharmacies share a
    route and in the order they are visited". It should say that the direction alone is
    a large share. For exactly optimal plans, too, the straight-line plan is not unique,
    and its regret depends on how the ties are broken.
  - *E3.* See I3.
  - *The advice.* "Expect routes about 7.1% to 8.9% longer than they need to be" should
    add that orienting each route on road distances removes a third to a half of that.

**M2. PyVRP 0.14.1 is an iterated local search, and the reason given for the coordinates
is wrong.** Medium; verified.

- **The algorithm.** METHODS §5 and the article call PyVRP 0.14.1 "a hybrid genetic
  search". The installed package runs `IteratedLocalSearch` (`pyvrp/solve.py`, "Solver
  parameters for PyVRP's iterated local search algorithm") and has no genetic algorithm.
  The cited paper (Wouda, Lan and Kool, 2024) describes the earlier, hybrid genetic search
  version.
- **The coordinates.** The change log and `routing.py`'s docstring give the EPSG:3763
  coordinates to PyVRP because "PyVRP's SWAP* operator compares routes by their clients'
  angles". Version 0.14.1 has no SWAP* operator (`pyvrp/search/__init__.py`, `OPERATORS`).
  Two registered instances re-solved with every coordinate set to zero give the same
  routes: Lisboa with Almada, day 1, straight-line plan, seed 1; and Porto with Gaia,
  day 7, road plan, seed 4.
- **What it changes.** Nothing in the results. The coordinates are harmless, but the
  description of the solver and the stated reason are wrong.

**M3. The drivable network keeps some ways a van may not use.** Low; verified (tags),
plausible (size of the effect).

- **The tags.** OSMnx's `drive` filter, as registered, keeps these among the ways read:
  - *Porto:* `highway=busway` (25 ways), `highway=bus_stop` (16), `highway=crossing` (4)
    and `access=no` (25);
  - *Lisbon:* `access=no` (61, six of them motorway links) and `highway=crossing` (1);
  - *both:* `junction=circular` with no `oneway` tag, which OSMnx treats as two-way
    (20 in Lisbon, 10 in Porto); and in Porto 8 motorway ways with no `oneway` tag.
- **A pharmacy on a crossing.** Porto's pharmacy 1 (node 473863503) snaps to a node that
  lies only on `highway=crossing` way 1337759781.
- **The effect.** The network reviewer rebuilt Porto's network without these ways, with
  roundabouts and untagged motorways one-way. In that rebuild 7.4% of Porto's pairs
  change, α moves from 1.291 to 1.288, and the regret of the committed plans for Porto
  alone rises by 0.27 points, on the 19 days when pharmacy 1 did not order. This was not
  recomputed here, and Lisbon was not rebuilt.
- **What it changes.** Probably nothing material. METHODS should state it as a
  limitation of the registered filter.

**M4. Porto's depot variant puts the depot across the river.** Low to medium; verified.
The variant depot is in Vila Nova de Gaia (`points.csv`, Porto index 165), which the
change log records. In that variant, every route of "Porto alone" crosses the Douro, so
its E3 does not compare a city without water against one split by water. The article's
"Porto's change from the far bank is negative in every variation" counts it without
saying so.

**M5. A pharmacy snapped onto a motorway.** Low; verified. Porto's pharmacy 164 (Gaia,
way 1530620002) snaps 114 m to node 1145512768, which lies only on motorway and motorway
link ways. Its median road-to-straight ratio from the other pharmacies is 2.30, the
highest of the 164 (the median over pharmacies is 1.31). It inflates distances to one
customer in Porto with Gaia; the effect on the regret is small.

**M6. Smaller points.**

- **The Lisbon boundaries.** "Those municipalities meet only in the river" (the change
  log, the crossing report): the network reviewer measured a gap of about 1.6 km of
  estuary between the OSM boundaries of Lisboa and Almada. Plausible; not rechecked. The
  crossing report still finds the bridge on every path, because the bridge's simplified
  edge joins a node in Lisboa to one in Almada.
- **Turn restrictions.** They change distances, not only times, but METHODS §8 and the
  article list them under "Not time". Verified: only ways are read, so turn-restriction
  relations play no part.
- **Pharmacies mapped twice.** Two pairs are probably the same pharmacy mapped twice and
  are kept as two customers: in Porto node/1270263273 and node/12158362559, 6.1 m apart;
  in Lisbon node/1212671093 and node/12307074452, 14.8 m apart. Verified (distances).
- **The order draw.** Porto's pharmacies ordered on 47.6% of pharmacy-days against the
  registered 50% (3,124 of 6,560; z = −3.85). The code is right: the draws regenerate
  exactly from the registered seed, and the article reports the realised averages. It is
  an unusual draw, not an error. Verified.
- **Reproducibility.** "The tests re-solve problems from the committed results" is true
  of two solves, both of Porto alone (`tests/test_results.py`). Verified.

### The estimands and the inference

**I1. The noise behind "more than 60 times" is the road plan's.** High; verified.

- **The placebo.** E2's placebo compares the road plan with itself, on the road matrix
  it optimises. That measures how well the search converges there.
- **The straight-line plan's noise.** The mean gap between two seeds' road distances is:

  | Area | Road plan, on the road | Straight-line plan, on the road | Straight-line plan, on straight lines | Regret ÷ straight-line plan's noise |
  | --- | ---: | ---: | ---: | ---: |
  | Lisboa | 0.08% | 1.49% | 0.17% | 4.8 |
  | Lisboa with Almada | 0.08% | 1.40% | 0.08% | 6.0 |
  | Porto | 0.03% | 2.81% | 0.03% | 3.2 |
  | Porto with Vila Nova de Gaia | 0.11% | 1.82% | 0.05% | 4.1 |

  Within a day, the regret of single seeds has a standard deviation of 1.2 to 2.6 points.
  Most of this is route direction (M1): with each route costed as the average of its two
  directions, the straight-line plan's seed noise falls to 0.05% to 0.44%.
- **What it changes.** E2's decision stands, because the regret is a systematic shift.
  But "more than 60 times the solver's own run-to-run noise" overstates how stable a
  single straight-line run is. Figure 2 shows only the road plan's noise. METHODS §8,
  "E2 measures the solver's own noise", needs the same qualification.

**I2. Lisbon's E3 is inconclusive, not a failed expectation.** Medium (wording); verified.

- **The numbers.** Δ = +1.32 points. The registered 98.75% interval is −0.17 to +3.05.
  At 97.5% it is −0.03 to +2.86, and at 95% it is +0.12 to +2.63.
- **The level.** The 98.75% level is Bonferroni over the four areas (METHODS §6), but E3
  makes two comparisons. The registered level is conservative for it, but it is the
  registered rule.
- **Power.** The day-level standard deviation of Δ is 4.07 points, and pairing by day
  gains nothing (the correlation between the two areas' daily regrets is −0.05). At the
  registered level, the study had 80% power only for rises above about 2.2 points in
  Lisbon and 1.5 in Porto.
- **What it changes.** "The expectation failed in both cities" and "did not widen that
  gap" should read: in Lisbon, the estimate has the expected sign but is not established.
  The article flags the uncorrected 95% intervals of two sensitivity variants but not the
  main design's.

**I3. Porto's fall is relative, comes from route direction, and is fragile.** High (for
the reading); verified.

- **Relative, not absolute.** Adding the far bank raises the regret in kilometres:

  | Pair | Δ km a day [98.75%] | Δ metres per order [98.75%] |
  | --- | ---: | ---: |
  | Lisbon | +5.4 [+3.4, +7.6] | +64 [+34, +101] |
  | Porto | +6.5 [+5.2, +7.8] | +44 [+28, +62] |

  The ratio falls in Porto because Gaia adds much more road: the road plan drives 59.3 km
  a day in Porto and 158.9 km with Gaia. The article quotes 5.3 against 11.8 km a day
  beside "the far bank lowered the regret".
- **Direction.** With every straight-line route driven in its better direction (M1),
  Δ is +0.01 [−0.83, +0.89] in Porto and +1.80 [+0.22, +3.50] in Lisbon (98.75%, post
  hoc). The direction part of Δ is −1.50 [−2.47, −0.52] in Porto and −0.48 [−0.93, −0.02]
  in Lisbon. So Porto's whole fall is direction, and the rest of the regret rises with the
  far bank in Lisbon. Porto's own pairs are much more asymmetric (median 7.7%) than Gaia's
  (2.5%) or those across the river (3.3%).
- **Fragile.** Two of Porto's four sensitivity variants have 95% intervals that include
  zero: vans of 24 (−2.7 to +0.2) and 20,000 iterations (−2.6 to +0.1). One variant is
  the Gaia depot (M4).
- **Confounded.** E3 compares the city with the city plus a whole municipality: 17% more
  orders in Lisbon and 59% more in Porto, over a larger area. No same-bank control was
  registered, so water cannot be separated from size. The number of orders does not
  explain the fall: within Porto alone, the regret rises with the day's orders.
- **What it changes.** Present Porto's fall as a description outside the registered
  rule, of the relative regret, and say that the absolute regret rose. Replace "these
  data do not say" with the direction analysis, labelled post hoc.

**I4. The optimizer's-curse explanation of E4 is not supported.** Medium to high;
verified.

- **The claim.** The straight-line plan "selects its own errors": among the possible
  hops, it picks those where the straight line understates the road most.
- **The check.** Each plan's legs, depot legs included, against all ordered pairs of the
  area's points with the same straight-line length (100 m bins, ratio of sums within each
  bin):

  | Area | Straight-line plan's legs | Same-length pairs | Road plan's legs | Same-length pairs |
  | --- | ---: | ---: | ---: | ---: |
  | Lisboa | 1.542 | 1.518 | 1.374 | 1.505 |
  | Lisboa with Almada | 1.632 | 1.505 | 1.382 | 1.494 |
  | Porto | 1.604 | 1.620 | 1.408 | 1.601 |
  | Porto with Vila Nova de Gaia | 1.507 | 1.508 | 1.355 | 1.499 |

  Only Lisboa with Almada shows an excess, where straight lines across the estuary hide
  the detour to the bridge. Elsewhere, the straight-line plan's legs detour like any legs
  of their length; it is the road plan that picks legs with less detour.
- **By construction.** For plans optimal on their own matrices, C_R(P_S) ≥ C_R(P_R) and
  C_S(P_S) ≤ C_S(P_R), so κ for the straight-line plan is at least κ for the road plan.
  "The straight-line plan's most of all" is therefore not evidence of a curse. It held on
  40 of 40 days in every area.
- **What the data do show.**
  - *Short legs detour more.* The median ratio over pharmacy pairs is 1.48 to 1.51
    between 0.5 and 1 km apart, against 1.25 to 1.31 beyond 2.5 km; under 0.5 km it is
    1.85 to 1.89 in the Porto areas.
  - *α is a median.* The ratio is skewed to the right, and route costs add up. Over
    pharmacy pairs, the mean ratio is 1.345 to 1.432 and the ratio of sums 1.288 to
    1.404, against α of 1.284 to 1.317.
- **What it changes.** Rewrite the bullet: keep "routes are chains of short hops", add
  the median-against-mean point, and say that κ_S ≥ κ_R follows from the construction.
  The optimizer's curse (Smith and Winkler, 2006) fits only the case with Almada.

**I5. The intervals describe simulated days, not cities.** Medium; verified.

- **What they cover.** The day-to-day variation of simulated orders over fixed
  pharmacies, network, depot, capacity and order probability. Their width is set by the
  study's own 40 days.
- **What they miss.** Design choices move the estimates more than the intervals do:
  capacity alone moves Lisboa's regret from 5.8% to 8.0%, wider than its interval
  (6.4% to 7.9%), and changes the sign of Lisbon's E3.
- **What it changes.** "Expect routes about 7.1% to 8.9% longer ... in cities like these"
  should use the range over the variants (5.6% to 9.1%) and say what the intervals cover.

**I6. The capacity pattern is selective.** Medium to low; verified. "It grows with the
vans' capacity in both cities alone." Paired by day over the first 20 days (95%):

| Area | Vans of 6 → 12 | 12 → 24 |
| --- | ---: | ---: |
| Lisboa | +1.41 [+0.35, +2.41] | +0.78 [+0.02, +1.56] |
| Lisboa with Almada | +1.40 [+0.23, +2.52] | −1.37 [−2.80, −0.13] |
| Porto | +1.08 [−0.02, +2.21] | +0.44 [−0.84, +1.71] |
| Porto with Vila Nova de Gaia | +1.77 [+1.04, +2.57] | +0.44 [−0.73, +1.60] |

From 6 to 12 the regret rises in every area. From 12 to 24 the rise is not established
in Porto, and in Lisboa with Almada the regret falls.

**I7. The E4 identity is exact per day, but the quoted κ is a mean.** Low; verified. The
article prints "error = α ÷ κ − 1" beside κ averaged over days. α divided by the mean κ,
minus 1, gives −16.7% and −19.3% for Lisboa and Lisboa with Almada, against reported mean
errors of −16.6% and −19.2%.

**I8. E2 is nearly mechanical, and "bound" overstates.** Low; verified.

- **Mechanical.** The straight-line plan is never shorter on the road than the road plan,
  in any of the 640 seed-days, and the placebo is at most 0.11%. E2 amounts to "the
  regret exceeds about 0.1%".
- **"Bound".** "The seed noise and the longer search bound how much that matters"
  overstates. Agreement between seeds and stability under four times the search show
  convergence; they do not bound the distance to an optimal plan.

### The claims and their communication

**C1. The account of where the regret comes from leaves out direction.** High; verified.
See M1. "For exactly optimal plans, any regret must therefore come from how unevenly the
roads detour ... and from one-way streets" is right, but a reader cannot tell that the
straight-line plan drives each route in a direction the solver picks arbitrarily, or that
this is a large share of the regret.

**C2. The river section.** Medium; verified.

- **The crossing metric.** "They send more routes across it than the road plans, 19% of
  routes against 15% in Lisbon and 22% against 18% in Porto." Those figures count routes
  that serve customers on both banks. With the depot on the city bank, every route with a
  far-bank customer crosses. The shares of routes that cross are 19.7% against 18.9% in
  Lisbon and 42.3% against 40.6% in Porto. The road plans send more routes that serve only
  the far bank (4.2% against 1.0% in Lisbon).
- **The one-way-streets pattern.** "In Porto alone, the road distance ... differs between
  the two directions by a median 7.7%, against 4.3% once Gaia is added." This is mostly
  a change in which pairs are counted. Porto's own pairs stay at 7.7%; Gaia's are 2.5% and
  those across the river 3.3%. In Lisbon, Almada's own pairs are 2.7% against Lisboa's
  3.9%. M1 and I3 test the pattern directly.
- **"Three bridges."** The three named bridges carry 99.5% of Porto's paths between the
  banks (the manifest's shares sum to 0.9947); the rest use crossings under 1% each.

**C3. The subtitle inflates the sample.** Medium; verified. "In 2,560 routing problems
solved": by the article's own definition ("Each day's orders define one routing
problem"), the main design has 160 problems, each solved 8 times, and 1,280 of the 2,560
solves are sensitivity runs. Inference rests on 40 days per area.

**C4. The advice overgeneralises.** Medium; verified. See I5 and M1. "Expect routes about
7.1% to 8.9% longer" is the range of point estimates for one setup. The variants give
5.6% to 9.1%, and choosing each route's direction on the road gives 4.5% to 6.6%.

**C5. "One detour factor" in the summary.** Medium to low. The shortfall is for α, the
median ratio over all pairs of pharmacies. "What to do with it" says "the median detour
factor"; the summary should too.

**C6. Smaller points.** Low; verified, except as noted.

- **Post-hoc labels.** The kilometres a day are post hoc (`post_hoc` in
  `results/estimates.json`), and so is the use of the asymmetry to explain E3 (the change
  log). Only κ and the leg lengths are labelled in the article.
- **"Simultaneous across the four areas."** The river intervals are two comparisons at
  the four-area level (I2); say so.
- **"Two Portuguese city centres."** These are whole municipalities, and Gaia is largely
  suburban. METHODS §8 says "metropolitan cores".
- **The licence of the derived tables.** The repository's root licence is Apache-2.0,
  and `data/derived/` has no licence or notice file; the ODbL is declared only in the
  README, METHODS and DATA.md. `results/solves.csv` is derived from the matrices too. A
  notice in `data/derived/` would make "published under the same licence" unambiguous.
- **The figures.**
  - Figure 2 shows only the road plan's noise (I1).
  - Figure 5's "Main design" row is the main design on the first 20 days and seeds 1 and
    2; its label should say so.
  - Figure 5 uses blue for Lisbon, while blue means the straight-line plan in figures 2
    and 4.
  - The footer shared by every figure says "Routes: PyVRP", including figure 1, which
    shows no routes.

## What holds up

- **The extract and the network.** The extract's hashes match. Porto's road matrix
  reproduces exactly from an independent build. Clipping to 3 km changes two of Porto's
  entries, by at most 234 m. Every pharmacy's snapped node lies in its own municipality,
  and every snapped node is in the strongly connected component.
- **The matrices.** Straight-line distances match haversine exactly; no road distance is
  below the straight line; no triangle-inequality excess exceeds 1 m; diagonals are zero.
  The point order and the leading-block property hold, and α, the asymmetry percentiles
  and the same-node counts in DATA.md recompute as defined.
- **The routing model.** One depot, unit deliveries, capacity Q, as many vehicles as
  orders, no fixed cost, every edge given. Every plan is feasible and costed with its
  depot legs. The road plans are not under-searched: their routes are already in their
  better direction, and the best of four seeds is within 0.02% to 0.09% of the seed mean.
- **The estimands.** They are computed as registered: the ratios, the seed means, the 12
  ordered seed pairs, α over pairs with a positive straight-line distance, e(d) and e*(d),
  and the bootstrap with the registered seed and generator order. The identity
  e(d) = α / κ(d) − 1 holds exactly day by day.
- **The decisions.** E2 holds in every area. Porto's negative Δ survives t and BCa
  intervals. More search changes no estimate by more than 0.12 points, and Porto's plans
  with 20,000 iterations are the same as with 5,000.
- **The article's numbers and citations.** Every number matches, including the extremes
  ("on no day ... below 2.7%", "on no day ... above 0.8%"). The three references exist
  as cited (checked on Crossref): Boeing (2025), Wouda, Lan and Kool (2024), and Smith and
  Winkler (2006), whose statement of the curse is paraphrased correctly. The OSM and ODbL
  attribution is correct.
- **Registration.** The registered sections of METHODS.md are unchanged since the plan's
  commit, which precedes the data; every later change is a dated change-log entry.

## What should change before publication

None of this was done in this review. Each change belongs in a corrections step, with
its own entry in the change log.

- **Route direction (M1).**
  - Add a post-hoc analysis: the straight-line plans with each route driven in its
    better direction, and the share of the regret that removes, by area and variant.
  - Say in the article where the regret comes from, direction included, and that exactly
    optimal straight-line plans are not unique.
  - Add to the advice that orienting each route on road distances needs only the road
    distances of the plan's own legs, and removes a third to a half of the regret.
- **The noise (I1).** Drop "more than 60 times", or say it is the road plan's noise and
  give the straight-line plan's 1.4% to 2.8%. Show or mention it in figure 2. Qualify
  METHODS §8 in the change log.
- **The river (I2, I3, C2).**
  - Lisbon: an inconclusive result with the expected sign, not a failed expectation.
  - Porto: a fall in the relative regret only, with the absolute regret rising, entirely
    from route direction (post hoc), fragile across the variants, and confounded with the
    extra orders and area.
  - Describe E3 as adding the far-bank municipality, which cannot separate water from
    size.
  - Correct the crossing shares and the asymmetry argument; say "three bridges carry
    99.5% of the paths".
- **E4 (I4, I7).** Replace the optimizer's-curse bullet with short legs and the median
  against the mean; say that κ_S ≥ κ_R follows from the construction; keep the curse, if
  at all, for the case with Almada. Fix the α ÷ κ example.
- **The solver (M2).** "An iterated local search" in METHODS (by change-log entry), the
  article and `routing.py`; correct the reason for the coordinates.
- **Scope (I5, I6, C3, C4, C5, I8).** Fix the subtitle, the advice range, the capacity
  bullet, "one detour factor" and "bound"; say what the intervals cover.
- **The network (M3 to M6).** State the drive filter's bus-only and `access=no` ways, the
  motorway-snapped pharmacy, the Gaia depot of Porto's variant and turn restrictions as
  limitations; fix "meet only in the river".
- **Wording and figures (C6).** Label the post-hoc material, fix the figure labels and
  colours, and add a licence notice to `data/derived/`.
